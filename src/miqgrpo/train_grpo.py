"""GRPO training on the frozen MolecularIQ dataset.

    python -m miqgrpo.train_grpo preflight --config configs/experiments/count.yaml
    python -m miqgrpo.train_grpo train     --config configs/experiments/count.yaml

preflight costs a couple of minutes and catches the failures that otherwise
only show up an hour into a paid GPU run: invalid batch arithmetic, a dataset
whose hash moved, prompts that render wrong, rewards that do not line up with
their prompts, or a task the policy already solves perfectly (no reward
variance, no gradient).

This module loads a frozen artifact and nothing else. It never imports the
question generator.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, dump_config, load_experiment_config
from .paths import dataset_dir, ensure_dirs, refuse_to_overwrite, run_dir
from .prompts import SYSTEM_PROMPT
from .provenance import capture, content_hash, hash_config, sha256_tree, write_json
from .rewards import RewardConfig, build_reward_functions, score_completion

__all__ = ["main", "batch_arithmetic", "preflight", "train"]


def load_frozen_split(config: ExperimentConfig, split: str):
    """Load one split of the frozen artifact, checking its identity first."""
    from datasets import load_from_disk

    artifact = dataset_dir(config.data.artifact_id)
    dataset_path = artifact / "dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset artifact '{config.data.artifact_id}' not found at {dataset_path}. "
            f"Build it first: bash scripts/02_build_dataset.sh (configs/dataset.yaml)"
        )

    manifest = json.loads((artifact / "manifest.json").read_text())
    dataset = load_from_disk(str(dataset_path))

    # The content hash is machine-independent, so anyone can check the pinned value.
    expected = config.data.expected_hash or manifest.get("content_hash")
    actual = content_hash(dataset)
    if expected and expected != actual:
        raise ValueError(
            f"dataset artifact '{config.data.artifact_id}' is not the one this experiment "
            f"was configured against.\n  expected {expected}\n  actual   {actual}\n"
            f"If you rebuilt it, your environment differs from the reference one "
            f"(compare against requirements-lock.txt); if you changed the recipe, "
            f"that is a new dataset_artifact_id and a new experiment."
        )

    if split not in dataset:
        raise KeyError(
            f"split '{split}' not in artifact (have: {sorted(dataset.keys())})"
        )
    return dataset[split], manifest, actual


def batch_arithmetic(config: ExperimentConfig, world_size: int) -> dict[str, Any]:
    """Resolve and validate TRL's generation/optimisation batch sizes.

    TRL derives generation_batch_size from the per-device batch, the world
    size and gradient accumulation, then requires it to be divisible by
    num_generations so every group is complete. Getting this wrong throws
    only once the trainer is constructed, after the model has loaded.
    """
    grpo = config.grpo
    generation_batch_size = (
        grpo.per_device_train_batch_size * world_size * grpo.gradient_accumulation_steps
    )
    numbers = {
        "world_size": world_size,
        "per_device_train_batch_size": grpo.per_device_train_batch_size,
        "gradient_accumulation_steps": grpo.gradient_accumulation_steps,
        "steps_per_generation": grpo.gradient_accumulation_steps,
        "generation_batch_size": generation_batch_size,
        "num_generations": grpo.num_generations,
        "prompts_per_generation_batch": generation_batch_size // grpo.num_generations
        if grpo.num_generations
        else 0,
        "completions_per_optimizer_step": generation_batch_size,
    }

    problems = []
    if grpo.num_generations < 2:
        problems.append("num_generations must be >= 2 for GRPO to form advantages")
    if generation_batch_size % grpo.num_generations != 0:
        problems.append(
            f"generation_batch_size ({generation_batch_size}) is not divisible by "
            f"num_generations ({grpo.num_generations})"
        )
    numbers["problems"] = problems
    return numbers


def _torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }[name]


def load_model_and_tokenizer(config: ExperimentConfig):
    """The policy and its tokenizer.

    Loads model.id or, when model.init_from is set, the final
    checkpoint of that run (e.g. a supervised warm-up).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {"dtype": _torch_dtype(config.model.dtype)}
    if config.model.revision:
        kwargs["revision"] = config.model.revision
    if config.model.attn_implementation:
        kwargs["attn_implementation"] = config.model.attn_implementation

    source: str = config.model.id
    if config.model.init_from:
        start = run_dir(config.model.init_from) / "final"
        if not (start / "config.json").exists():
            raise FileNotFoundError(
                f"model.init_from '{config.model.init_from}': no checkpoint at {start} (train it first)"
            )
        source = str(start)
        kwargs.pop("revision", None)
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=None if config.model.init_from else config.model.revision
    )
    if tokenizer.pad_token is None:
        # Qwen2.5-Instruct has a pad token; warn if that ever changes.
        print("  ! tokenizer has no pad_token; falling back to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_peft_config(config: ExperimentConfig):
    """A LoRA config when model.peft.enabled, else None (full fine-tuning)."""
    if not config.model.peft.enabled:
        return None
    from peft import LoraConfig

    peft = config.model.peft
    return LoraConfig(
        r=peft.r,
        lora_alpha=peft.lora_alpha,
        lora_dropout=peft.lora_dropout,
        target_modules=peft.target_modules,
        bias=peft.bias,
        task_type="CAUSAL_LM",
    )


def _reward_config(config: Any, split: Any) -> RewardConfig:
    """Reward settings, plus what the reward derives from the training split.

    The per-property typical values are read from the frozen training split
    itself (never from dev or benchmark data), and logged either way so the
    typical-answer share can be watched during training.
    """
    from .dev_eval import typical_values

    typical = typical_values(split) if config.task_family in ("count", "index") else {}
    return RewardConfig(**vars(config.rewards), typical_values=typical)


def sampling_kwargs(grpo: Any) -> dict[str, Any]:
    """The rollout sampling settings, passed explicitly for both backends.

    Every value is passed as configured. A missing (None) value lets
    transformers fall back to the model's generation_config, which for
    Qwen2.5-0.5B-Instruct is top_k=20, top_p=0.8, temperature=0.7 and
    repetition_penalty=1.1 -- silently different from the config. top_k=0 and
    top_p=1.0 mean "off" in both transformers and vLLM.
    """
    return {
        "temperature": grpo.temperature,
        "top_p": grpo.top_p,
        "top_k": int(grpo.top_k or 0),
        "min_p": grpo.min_p,
        "repetition_penalty": grpo.repetition_penalty,
    }


def preflight(config_path: Path, n_prompts: int = 4, n_generations: int | None = None) -> None:
    """Inspect one real rollout before spending GPU hours on the full run."""
    import torch

    config = load_experiment_config(config_path)
    world_size = max(1, torch.cuda.device_count() or 1)

    print(f"experiment : {config.id}")
    print(f"task family: {config.task_family}")
    print(f"model      : {config.model.id} ({config.model.dtype})")
    print()

    print("== batch arithmetic ==")
    numbers = batch_arithmetic(config, world_size)
    for key, value in numbers.items():
        if key != "problems":
            print(f"  {key:34s} {value}")
    if numbers["problems"]:
        for problem in numbers["problems"]:
            print(f"  INVALID: {problem}")
        raise SystemExit(1)
    print("  arithmetic OK")
    print()

    print("== dataset ==")
    split, manifest, actual_hash = load_frozen_split(config, config.data.split)
    print(f"  artifact       {config.data.artifact_id}")
    print(f"  hash           {actual_hash}")
    print(f"  split          {config.data.split}  ({len(split)} examples)")
    families = Counter(split["task_family"])
    print(f"  task families  {dict(families)}")
    if set(families) != {config.task_family}:
        raise SystemExit(
            f"split contains families {sorted(families)}, expected only "
            f"'{config.task_family}' -- this would not be a single-task run"
        )
    steps_available = len(split) // numbers["prompts_per_generation_batch"]
    print(
        f"  one epoch      {steps_available} optimizer steps "
        f"({numbers['prompts_per_generation_batch']} prompts per step)"
    )
    if config.grpo.max_steps > 0 and config.grpo.max_steps > steps_available:
        print(
            f"  ! max_steps ({config.grpo.max_steps}) exceeds one epoch "
            f"({steps_available}); prompts will repeat"
        )
    print()

    print("== prompt rendering ==")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.id, revision=config.model.revision
    )
    example = split[0]
    rendered = tokenizer.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True
    )
    print(f"  system prompt matches official: {example['prompt'][0]['content'] == SYSTEM_PROMPT}")
    print(f"  rendered prompt tokens: {len(tokenizer(rendered)['input_ids'])}")
    print("  ---8<--- rendered prompt ---8<---")
    print("  " + rendered.replace("\n", "\n  ")[:1500])
    print("  ---8<--------------------------8<---")
    print()

    print("== rollout ==")
    model, tokenizer = load_model_and_tokenizer(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    group = n_generations or config.grpo.num_generations
    reward_config = _reward_config(config, split)
    reward_functions = build_reward_functions(reward_config)
    all_rewards: list[float] = []
    zero_variance_groups = 0
    duplicate_fraction: list[float] = []
    lengths: list[int] = []

    for index in range(min(n_prompts, len(split))):
        row = split[index]
        rendered = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([rendered] * group, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                **sampling_kwargs(config.grpo),
                max_new_tokens=config.grpo.max_completion_length,
                pad_token_id=tokenizer.pad_token_id,
            )
        completions = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        lengths.extend(
            len(tokenizer(c)["input_ids"]) for c in completions
        )
        duplicate_fraction.append(1.0 - len(set(completions)) / len(completions))

        scored = [
            score_completion(
                text, row["task_type"], row["target_json"], row["constraints_json"]
            )
            for text in completions
        ]
        # The same callables training uses, so preflight cannot drift from it.
        columns = {
            name: [row[name]] * len(completions)
            for name in ("task_type", "target_json", "constraints_json")
        }
        per_function = [fn(completions=completions, **columns) for fn in reward_functions]
        rewards = [
            sum(value for value in values if value is not None)
            for values in zip(*per_function)
        ]
        all_rewards.extend(rewards)
        if len(set(round(r, 6) for r in rewards)) == 1:
            zero_variance_groups += 1

        print(f"  [{row['example_id']}] {row['feature']}")
        print(f"    Q: {row['question'][:160]}")
        print(f"    target: {(row['target_json'] or row['constraints_json'])[:160]}")
        for text, s, reward in list(zip(completions, scored, rewards))[:3]:
            snippet = text.strip().replace("\n", " ")[:120]
            print(
                f"    -> reward={reward:.3f} status={s.status:<28} "
                f"extracted={s.extracted[:60]!r}"
            )
            print(f"       {snippet!r}")
        print(
            f"    group: mean={statistics.mean(rewards):.3f} "
            f"std={statistics.pstdev(rewards):.3f}"
        )

    print()
    print("== rollout summary ==")
    print(f"  reward mean            {statistics.mean(all_rewards):.4f}")
    print(f"  reward std             {statistics.pstdev(all_rewards):.4f}")
    print(f"  zero-variance groups   {zero_variance_groups}/{min(n_prompts, len(split))}")
    print(f"  duplicate completions  {statistics.mean(duplicate_fraction):.3f}")
    print(f"  completion length mean {statistics.mean(lengths):.1f}")
    print(f"  completion length max  {max(lengths)} (cap {config.grpo.max_completion_length})")
    if zero_variance_groups == min(n_prompts, len(split)):
        print(
            "  ! every group had identical rewards. GRPO gets no gradient from such "
            "groups -- check parsing, difficulty and sampling temperature before scaling."
        )


class MetricsJsonlCallback:
    """Append every logged metrics dict to logs/metrics.jsonl.

    trainer_state.json holds the same history, but only from the last save.
    A plain append-only JSONL survives preemption and makes the plotting script
    trivial.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        """Append one JSON line per logging step (main process only)."""
        if logs is None or not state.is_world_process_zero:
            return
        record = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


def train(config_path: Path, resume: bool = False) -> Path:
    """Train one experiment with TRL's GRPOTrainer.

    Loads and checks the config and the frozen dataset (content hash), writes
    frozen_config.yaml and provenance.json into $MIQ_RUNS/<id>,
    trains, and saves the final checkpoint to $MIQ_RUNS/<id>/final with a
    training_summary.json. resume continues from the newest checkpoint
    of an interrupted run instead of refusing an existing run directory.
    Returns the final checkpoint directory.
    """
    import torch
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    config = load_experiment_config(config_path)
    out_dir = run_dir(config.id)
    checkpoints = out_dir / "checkpoints"
    if not resume:
        refuse_to_overwrite(out_dir, f"run directory for experiment '{config.id}'")
    ensure_dirs(out_dir, out_dir / "logs", checkpoints)

    world_size = max(1, torch.cuda.device_count() or 1)
    numbers = batch_arithmetic(config, world_size)
    if numbers["problems"]:
        raise SystemExit("invalid batch arithmetic: " + "; ".join(numbers["problems"]))

    split, manifest, dataset_hash = load_frozen_split(config, config.data.split)
    eval_split = None
    if config.runtime.eval_steps and config.data.dev_split:
        eval_split, _, _ = load_frozen_split(config, config.data.dev_split)
        if config.runtime.eval_max_examples:
            eval_split = eval_split.select(
                range(min(config.runtime.eval_max_examples, len(eval_split)))
            )

    dump_config(config, out_dir / "frozen_config.yaml")
    write_json(
        out_dir / "provenance.json",
        capture(
            "train_grpo",
            {
                "experiment_id": config.id,
                "task_family": config.task_family,
                "dataset_artifact_id": config.data.artifact_id,
                "dataset_hash": dataset_hash,
                "dataset_manifest_hash": hash_config(manifest),
                "config_hash": hash_config(config.as_dict()),
                "batch_arithmetic": numbers,
                "benchmark_integrity": config.benchmark_integrity.__dict__,
            },
        ),
    )

    model, tokenizer = load_model_and_tokenizer(config)
    reward_config = _reward_config(config, split)
    reward_funcs = build_reward_functions(reward_config)

    grpo = config.grpo
    runtime = config.runtime
    args = GRPOConfig(
        output_dir=str(checkpoints),
        run_name=config.id,
        seed=runtime.seed,
        learning_rate=grpo.learning_rate,
        lr_scheduler_type=grpo.lr_scheduler_type,
        warmup_steps=grpo.warmup_steps,
        weight_decay=grpo.weight_decay,
        max_grad_norm=grpo.max_grad_norm,
        per_device_train_batch_size=grpo.per_device_train_batch_size,
        gradient_accumulation_steps=grpo.gradient_accumulation_steps,
        num_generations=grpo.num_generations,
        max_completion_length=grpo.max_completion_length,
        **sampling_kwargs(grpo),
        beta=grpo.beta,
        epsilon=grpo.epsilon,
        epsilon_high=grpo.epsilon_high,
        loss_type=grpo.loss_type,
        scale_rewards=grpo.scale_rewards,
        num_iterations=grpo.num_iterations,
        mask_truncated_completions=grpo.mask_truncated_completions,
        max_steps=grpo.max_steps,
        num_train_epochs=grpo.num_train_epochs,
        gradient_checkpointing=grpo.gradient_checkpointing,
        bf16=grpo.bf16,
        eval_strategy="steps" if eval_split is not None else "no",
        eval_steps=runtime.eval_steps,
        eval_on_start=bool(runtime.eval_on_start and eval_split is not None),
        per_device_eval_batch_size=runtime.per_device_eval_batch_size,
        num_generations_eval=runtime.num_generations_eval,
        logging_steps=runtime.logging_steps,
        save_steps=runtime.save_steps,
        save_total_limit=runtime.save_total_limit,
        save_strategy="steps",
        log_completions=runtime.log_completions,
        num_completions_to_print=runtime.num_completions_to_print,
        use_vllm=runtime.use_vllm,
        vllm_mode=runtime.vllm_mode,
        vllm_gpu_memory_utilization=runtime.vllm_gpu_memory_utilization,
        report_to=runtime.report_to,
        dataloader_num_workers=runtime.dataloader_num_workers,
    )

    jsonl = MetricsJsonlCallback(out_dir / "logs" / "metrics.jsonl")
    callback = type("MetricsCallback", (TrainerCallback,), {"on_log": jsonl.on_log})()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=split,
        eval_dataset=eval_split,
        processing_class=tokenizer,
        peft_config=build_peft_config(config),
        callbacks=[callback],
    )

    print(f"training '{config.id}' on {len(split)} {config.task_family} examples")
    trainer.train(resume_from_checkpoint=resume or None)

    final = out_dir / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    write_json(
        out_dir / "training_summary.json",
        {
            "experiment_id": config.id,
            "task_family": config.task_family,
            "final_checkpoint": str(final),
            "final_checkpoint_hash": sha256_tree(final),
            "global_step": trainer.state.global_step,
            "dataset_artifact_id": config.data.artifact_id,
            "dataset_hash": dataset_hash,
            "batch_arithmetic": numbers,
        },
    )
    print(f"done. final checkpoint: {final}")
    return final


def main(argv: list[str] | None = None) -> None:
    """Command line: preflight --config <yaml> or train --config <yaml> [--resume]."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preflight", help="validate config, data and one rollout")
    pre.add_argument("--config", type=Path, required=True)
    pre.add_argument("--n-prompts", type=int, default=4)
    pre.add_argument("--n-generations", type=int, default=None)

    run = sub.add_parser("train", help="run GRPO training")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--resume", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "preflight":
        preflight(args.config, args.n_prompts, args.n_generations)
    else:
        train(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
