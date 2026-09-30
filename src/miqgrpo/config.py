"""Experiment configuration.

One YAML file fully determines a training run. It is resolved once at launch and
copied into the run directory as frozen_config.yaml; the copy, not the
original, is what a result refers to. Editing a config after launch therefore
cannot retroactively change what a run claims to have done.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ExperimentConfig",
    "load_experiment_config",
    "dump_config",
]


def _subset(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys the dataclass declares, and say so when one is unknown."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown keys for {cls.__name__}: {sorted(unknown)}; "
            f"known keys are {sorted(known)}"
        )
    return {k: v for k, v in data.items() if k in known}


@dataclass
class PeftSettings:
    """Optional LoRA adapter (model.peft). Every reported run trains all weights."""

    enabled: bool = False
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    bias: str = "none"


@dataclass
class ModelSettings:
    """The policy to train (model: section of an experiment config)."""

    # Hugging Face model id of the base model
    id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Hub revision of id; None uses the cached default
    revision: str | None = None
    # start from $MIQ_RUNS/<init_from>/final (e.g. an SFT warm-up) instead of id
    init_from: str | None = None
    # torch dtype of the weights: bfloat16, float16 or float32
    dtype: str = "bfloat16"
    # passed to from_pretrained; None lets transformers choose
    attn_implementation: str | None = None
    peft: PeftSettings = field(default_factory=PeftSettings)


@dataclass
class DataSettings:
    """Which frozen dataset a run trains on (data: section)."""

    # dataset artifact under $MIQ_DATA/processed/ (built by build_dataset)
    artifact_id: str = "miq-train"
    # training split; must equal the experiment's task family
    split: str = "count"
    # held-out split of the same family scored during training (<family>_dev)
    dev_split: str | None = None
    # content hash of the dataset; training refuses any other data
    expected_hash: str | None = None


@dataclass
class GRPOSettings:
    """Optimisation and sampling settings (grpo: section).

    Field names and meanings are those of TRL's GRPOConfig, to which
    train_grpo.train passes them unchanged (the sampling fields through
    train_grpo.sampling_kwargs). The global batch is
    per_device_train_batch_size * gradient_accumulation_steps completions,
    i.e. that number divided by num_generations questions per step.
    """

    learning_rate: float = 1e-6
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_steps: int = 10
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    num_generations: int = 8
    max_completion_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float | None = None
    repetition_penalty: float = 1.0
    beta: float = 0.0
    epsilon: float = 0.2
    epsilon_high: float | None = None
    loss_type: str = "dapo"
    scale_rewards: str = "group"
    num_iterations: int = 1
    mask_truncated_completions: bool = False
    max_steps: int = 500
    num_train_epochs: float = 1.0
    gradient_checkpointing: bool = True
    bf16: bool = True


@dataclass
class RewardSettings:
    """Weights of the reward terms (rewards: section; see rewards.py)."""

    # the official verifier's score for the whole question
    correctness_weight: float = 1.0
    # weight of the per-property credit (rewards.PartialCreditReward); 0 = off
    partial_weight: float = 0.0
    # weight of properties whose true value is 0 or []
    trivial_weight: float = 0.0
    # a wrong "none" answer scores -none_penalty in the per-property credit
    none_penalty: float = 0.0


@dataclass
class RuntimeSettings:
    """Logging, checkpointing, dev-split evaluation and rollout backend (runtime:)."""

    seed: int = 42
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int | None = 3
    # evaluate on the dev split every N steps (None = never)
    eval_steps: int | None = None
    # number of dev questions per evaluation
    eval_max_examples: int | None = None
    # also evaluate before the first optimizer step, as the reference point
    eval_on_start: bool = False
    per_device_eval_batch_size: int = 16
    num_generations_eval: int = 4
    log_completions: bool = True
    num_completions_to_print: int = 4
    # generate rollouts with vLLM instead of transformers (much faster)
    use_vllm: bool = False
    # "colocate" runs vLLM inside the training process on the same GPU
    vllm_mode: str = "colocate"
    # share of GPU memory reserved for vLLM; affects speed, not results
    vllm_gpu_memory_utilization: float = 0.3
    report_to: str = "none"
    dataloader_num_workers: int = 0


@dataclass
class BenchmarkIntegrity:
    """Declarations that the official benchmark stayed test-only.

    Copied into every run's provenance; tests/test_integrity.py checks
    these declarations for every shipped experiment config.
    """

    official_benchmark_used_for_training: bool = False
    official_benchmark_used_for_hparam_selection: bool = False
    official_benchmark_used_for_prompt_tuning: bool = False
    official_benchmark_used_for_extraction_tuning: bool = False
    official_benchmark_used_for_checkpoint_selection: bool = False
    official_benchmark_run_ids: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """One training run, fully resolved from its YAML file.

    id names the run directory $MIQ_RUNS/<id>; task_family is one of
    count, index or constraint_generation.
    """

    id: str
    task_family: str
    notes: str = ""
    model: ModelSettings = field(default_factory=ModelSettings)
    data: DataSettings = field(default_factory=DataSettings)
    grpo: GRPOSettings = field(default_factory=GRPOSettings)
    rewards: RewardSettings = field(default_factory=RewardSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    benchmark_integrity: BenchmarkIntegrity = field(default_factory=BenchmarkIntegrity)

    def as_dict(self) -> dict[str, Any]:
        """The config as nested plain dicts (what frozen_config.yaml stores)."""
        return asdict(self)

    def validate(self) -> None:
        """Reject configs that would not be the single-task run they claim to be."""
        if self.task_family not in ("count", "index", "constraint_generation"):
            raise ValueError(f"unknown task_family: {self.task_family}")
        if self.data.split != self.task_family:
            raise ValueError(
                f"single-task run '{self.id}' trains on family '{self.task_family}' but "
                f"reads split '{self.data.split}'. These must match, or the run is not "
                f"the single-task experiment it claims to be."
            )
        if self.rewards.correctness_weight <= 0:
            raise ValueError("correctness_weight must be positive: it is the official score")
        if self.rewards.partial_weight < 0:
            raise ValueError("partial_weight must not be negative")
        if not 0 <= self.rewards.trivial_weight <= 1:
            raise ValueError("trivial_weight must be in [0, 1]")
        if self.rewards.none_penalty < 0:
            raise ValueError("none_penalty must not be negative")
        if self.runtime.eval_steps:
            if not self.data.dev_split:
                raise ValueError("eval_steps is set but data.dev_split is empty")
            # TRL requires the global eval batch to hold whole generation groups.
            if self.runtime.per_device_eval_batch_size % self.runtime.num_generations_eval:
                raise ValueError(
                    f"per_device_eval_batch_size ({self.runtime.per_device_eval_batch_size}) "
                    f"must be divisible by num_generations_eval "
                    f"({self.runtime.num_generations_eval})"
                )


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Read and validate an experiment YAML file.

    Unknown keys in any section are an error rather than silently ignored, so a
    typo cannot leave a setting at its default.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    experiment = dict(raw.get("experiment") or {})
    model_raw = dict(raw.get("model") or {})
    peft_raw = dict(model_raw.pop("peft", {}) or {})

    config = ExperimentConfig(
        id=experiment["id"],
        task_family=experiment["task_family"],
        notes=experiment.get("notes", ""),
        model=ModelSettings(
            **_subset(ModelSettings, {**model_raw, "peft": PeftSettings(**_subset(PeftSettings, peft_raw))})
        ),
        data=DataSettings(**_subset(DataSettings, raw.get("data") or {})),
        grpo=GRPOSettings(**_subset(GRPOSettings, raw.get("grpo") or {})),
        rewards=RewardSettings(**_subset(RewardSettings, raw.get("rewards") or {})),
        runtime=RuntimeSettings(**_subset(RuntimeSettings, raw.get("runtime") or {})),
        benchmark_integrity=BenchmarkIntegrity(
            **_subset(BenchmarkIntegrity, raw.get("benchmark_integrity") or {})
        ),
    )
    config.validate()
    return config


def dump_config(config: ExperimentConfig, path: Path) -> None:
    """Write config as YAML, keeping the field order of the dataclasses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(config.as_dict(), fh, sort_keys=False)
