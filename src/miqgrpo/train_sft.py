"""Supervised warm-up on worked solutions, before GRPO (the extension study).

    python -m miqgrpo.train_sft train --config extension/configs/sft/count-0.5b.yaml

Trains on one task family's split of a frozen SFT dataset built by
miqgrpo.sft_data (prompt = the benchmark's system + user turns, completion
= the worked solution). The loss covers the completion only. The final
checkpoint is written to $MIQ_RUNS/<id>/final; a GRPO config starts from it
with model.init_from: <id>.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import dataset_dir, ensure_dirs, refuse_to_overwrite, run_dir
from .provenance import capture, content_hash, hash_config, sha256_tree, write_json

__all__ = ["SFTExperiment", "load_sft_config", "train", "main"]


@dataclass
class SFTSettings:
    """Supervised fine-tuning settings (sft: section); names follow TRL's SFTConfig."""

    learning_rate: float = 1.0e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_length: int = 2048
    gradient_checkpointing: bool = True
    bf16: bool = True
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 500


@dataclass
class SFTExperiment:
    """One supervised warm-up run, fully resolved from its YAML file."""

    id: str
    task_family: str
    notes: str = ""
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"
    artifact_id: str = ""
    expected_hash: str = ""
    sft: SFTSettings = field(default_factory=SFTSettings)

    def validate(self) -> None:
        """Reject an unknown task family or an unpinned dataset."""
        if self.task_family not in ("count", "index", "constraint_generation"):
            raise ValueError(f"unknown task_family: {self.task_family}")
        if not self.artifact_id or not self.expected_hash:
            raise ValueError("an SFT run must pin its dataset artifact and content hash")


def load_sft_config(path: Path) -> SFTExperiment:
    """Read and validate an SFT config (extension/configs/sft/*.yaml).

    Unknown keys in the sft section are an error, and the run must train
    on its own task family's split.
    """
    raw = yaml.safe_load(path.read_text()) or {}
    experiment, model, data = raw["experiment"], raw.get("model") or {}, raw["data"]
    if data.get("split", experiment["task_family"]) != experiment["task_family"]:
        raise ValueError("an SFT run trains on its own task family's split")
    known = set(SFTSettings.__dataclass_fields__)
    unknown = set(raw.get("sft") or {}) - known
    if unknown:
        raise ValueError(f"unknown sft keys: {sorted(unknown)}")
    config = SFTExperiment(
        id=experiment["id"],
        task_family=experiment["task_family"],
        notes=experiment.get("notes", ""),
        model_id=model.get("id", "Qwen/Qwen2.5-0.5B-Instruct"),
        dtype=model.get("dtype", "bfloat16"),
        artifact_id=data["artifact_id"],
        expected_hash=data["expected_hash"],
        sft=SFTSettings(**(raw.get("sft") or {})),
    )
    config.validate()
    return config


def train(config_path: Path) -> Path:
    """Run supervised fine-tuning for one config and return the final checkpoint.

    Checks the dataset's content hash, records the frozen config and
    provenance, trains with the loss on completions only, and writes
    $MIQ_RUNS/<id>/final plus training_summary.json.
    """
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from .train_grpo import MetricsJsonlCallback

    config = load_sft_config(config_path)
    out_dir = run_dir(config.id)
    refuse_to_overwrite(out_dir, f"run directory for SFT run '{config.id}'")
    ensure_dirs(out_dir, out_dir / "logs")

    dataset = load_from_disk(str(dataset_dir(config.artifact_id) / "dataset"))
    actual = content_hash(dataset)
    if actual != config.expected_hash:
        raise ValueError(f"SFT dataset '{config.artifact_id}' hash {actual} != pinned {config.expected_hash}")
    split = dataset[config.task_family].select_columns(["prompt", "completion"])

    with open(out_dir / "frozen_config.yaml", "w") as fh:
        yaml.safe_dump(asdict(config), fh, sort_keys=False)
    write_json(out_dir / "provenance.json", capture("train_sft", {
        "experiment_id": config.id,
        "task_family": config.task_family,
        "dataset_artifact_id": config.artifact_id,
        "dataset_hash": actual,
        "config_hash": hash_config(asdict(config)),
    }))

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[config.dtype]
    model = AutoModelForCausalLM.from_pretrained(config.model_id, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    s = config.sft
    args = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        run_name=config.id,
        seed=s.seed,
        learning_rate=s.learning_rate,
        lr_scheduler_type=s.lr_scheduler_type,
        # transformers 5: a float below 1 is a fraction of the total steps.
        warmup_steps=s.warmup_ratio,
        weight_decay=s.weight_decay,
        num_train_epochs=s.num_train_epochs,
        per_device_train_batch_size=s.per_device_train_batch_size,
        gradient_accumulation_steps=s.gradient_accumulation_steps,
        max_length=s.max_length,
        gradient_checkpointing=s.gradient_checkpointing,
        bf16=s.bf16,
        logging_steps=s.logging_steps,
        save_steps=s.save_steps,
        save_total_limit=1,
        # Loss on the worked solution only, never on the prompt.
        completion_only_loss=True,
        packing=False,
        report_to="none",
    )
    jsonl = MetricsJsonlCallback(out_dir / "logs" / "metrics.jsonl")
    callback = type("MetricsCallback", (TrainerCallback,), {"on_log": jsonl.on_log})()
    trainer = SFTTrainer(model=model, args=args, train_dataset=split,
                         processing_class=tokenizer, callbacks=[callback])
    print(f"SFT '{config.id}' on {len(split)} {config.task_family} examples")
    trainer.train()

    final = out_dir / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    write_json(out_dir / "training_summary.json", {
        "experiment_id": config.id,
        "task_family": config.task_family,
        "final_checkpoint": str(final),
        "final_checkpoint_hash": sha256_tree(final),
        "global_step": trainer.state.global_step,
        "dataset_artifact_id": config.artifact_id,
        "dataset_hash": actual,
    })
    print(f"done. final checkpoint: {final}")
    return final


def main(argv: list[str] | None = None) -> None:
    """Command line: train --config <yaml>."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("train")
    run.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    train(args.config)


if __name__ == "__main__":
    main()
