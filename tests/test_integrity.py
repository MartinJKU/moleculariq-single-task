"""The benchmark stays held out and artifacts stay immutable."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from miqgrpo.build_dataset import TRAIN_POOL_ID, _check_sources
from miqgrpo.config import ExperimentConfig, load_experiment_config
from miqgrpo.paths import REPO_ROOT, refuse_to_overwrite
from miqgrpo.train_grpo import batch_arithmetic

EXPERIMENTS = sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml"))
DATASET_CONFIG = REPO_ROOT / "configs" / "dataset.yaml"


@pytest.mark.parametrize(
    "dataset_id",
    ["ml-jku/moleculariq-v0.0", "moleculariq-v0.0", "someone/moleculariq_benchmark"],
)
def test_preprocessing_refuses_the_official_benchmark(dataset_id):
    with pytest.raises(ValueError, match="benchmark|training pool"):
        _check_sources({"source": {"dataset_id": dataset_id, "split": "test"}})


@pytest.mark.parametrize("split", ["val_hard", "val_easy", "test"])
def test_preprocessing_refuses_hidden_pools(split):
    with pytest.raises(ValueError):
        _check_sources({"source": {"dataset_id": TRAIN_POOL_ID, "split": split}})


def test_preprocessing_accepts_the_training_pool():
    _check_sources({"source": {"dataset_id": TRAIN_POOL_ID, "split": "train"}})


def test_there_is_one_experiment_per_task_family():
    configs = [load_experiment_config(p) for p in EXPERIMENTS]
    assert sorted(c.task_family for c in configs) == ["constraint_generation", "count", "index"]


def test_each_experiment_is_named_after_its_task_family():
    """One config, one run directory, one name: configs/experiments/<family>.yaml."""
    for path in EXPERIMENTS:
        config = load_experiment_config(path)
        assert path.stem == config.id == config.task_family


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_shipped_experiment_configs_are_valid(path: Path):
    config = load_experiment_config(path)
    config.validate()
    assert config.data.split == config.task_family
    assert config.model.id == "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_shipped_experiment_configs_have_valid_batch_arithmetic(path: Path):
    numbers = batch_arithmetic(load_experiment_config(path), world_size=1)
    assert numbers["problems"] == []
    assert numbers["prompts_per_generation_batch"] >= 2


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_shipped_experiment_configs_declare_benchmark_integrity(path: Path):
    integrity = load_experiment_config(path).benchmark_integrity
    assert integrity.official_benchmark_used_for_training is False
    assert integrity.official_benchmark_used_for_hparam_selection is False
    assert integrity.official_benchmark_used_for_prompt_tuning is False
    assert integrity.official_benchmark_used_for_checkpoint_selection is False


def test_the_dataset_config_only_reads_the_training_pool():
    _check_sources(yaml.safe_load(DATASET_CONFIG.read_text()))


def test_experiment_ids_are_unique():
    ids = [load_experiment_config(p).id for p in EXPERIMENTS]
    assert len(ids) == len(set(ids))


def _config(**overrides) -> ExperimentConfig:
    config = ExperimentConfig(id="x", task_family="count")
    config.data.split = "count"
    for key, value in overrides.items():
        target, _, field = key.partition("__")
        setattr(getattr(config, target) if field else config, field or target, value)
    return config


def test_split_must_match_the_declared_task_family():
    """Otherwise a 'count-only' run could quietly train on something else."""
    config = _config()
    config.data.split = "index"
    with pytest.raises(ValueError, match="single-task"):
        config.validate()


def test_the_official_score_must_stay_in_the_reward():
    config = _config()
    config.rewards.correctness_weight = 0.0
    with pytest.raises(ValueError, match="correctness_weight"):
        config.validate()


def test_retired_reward_terms_are_rejected():
    """The answer-shape and valid-SMILES bonuses are gone; configs naming them fail."""
    from miqgrpo.config import RewardSettings, _subset

    with pytest.raises(ValueError, match="unknown keys"):
        _subset(RewardSettings, {"format_weight": 0.1})


def test_unknown_task_family_is_rejected():
    config = _config()
    config.task_family = "spectroscopy"
    with pytest.raises(ValueError, match="task_family"):
        config.validate()


def test_eval_settings_must_form_whole_generation_groups():
    config = _config()
    config.data.dev_split = "count_dev"
    config.runtime.eval_steps = 50
    config.runtime.per_device_eval_batch_size = 6
    config.runtime.num_generations_eval = 4
    with pytest.raises(ValueError, match="num_generations_eval"):
        config.validate()


def test_eval_without_a_dev_split_is_rejected():
    config = _config()
    config.data.dev_split = None
    config.runtime.eval_steps = 50
    with pytest.raises(ValueError, match="dev_split"):
        config.validate()


def test_unknown_config_keys_are_rejected(tmp_path: Path):
    """A typo that silently does nothing is worse than a crash."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"id": "x", "task_family": "count"},
                "data": {"split": "count"},
                "grpo": {"learnign_rate": 1e-5},
            }
        )
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_experiment_config(path)


def test_indivisible_generation_batch_is_flagged():
    config = _config()
    config.grpo.per_device_train_batch_size = 5
    config.grpo.gradient_accumulation_steps = 1
    config.grpo.num_generations = 4
    problems = batch_arithmetic(config, world_size=1)["problems"]
    assert any("divisible" in p for p in problems)


def test_single_generation_is_flagged():
    config = _config()
    config.grpo.num_generations = 1
    problems = batch_arithmetic(config, world_size=1)["problems"]
    assert any("num_generations" in p for p in problems)


def test_world_size_scales_the_generation_batch():
    config = _config()
    one = batch_arithmetic(config, world_size=1)
    four = batch_arithmetic(config, world_size=4)
    assert four["generation_batch_size"] == 4 * one["generation_batch_size"]


def test_refuse_to_overwrite_a_populated_directory(tmp_path: Path):
    (tmp_path / "artifact").mkdir()
    (tmp_path / "artifact" / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError, match="already exists"):
        refuse_to_overwrite(tmp_path / "artifact", "dataset artifact")


def test_refuse_to_overwrite_allows_a_fresh_or_empty_directory(tmp_path: Path):
    refuse_to_overwrite(tmp_path / "missing", "run directory")
    (tmp_path / "empty").mkdir()
    refuse_to_overwrite(tmp_path / "empty", "run directory")


def test_limit_without_smoke_is_refused():
    from miqgrpo.evaluate import run_benchmark

    with pytest.raises(SystemExit, match="not be a MolecularIQ score"):
        run_benchmark(run_id="x", model_path="m", label="baseline", limit=10)


def test_smoke_runs_are_marked_and_kept_apart(tmp_path, monkeypatch):
    import miqgrpo.evaluate as evaluate

    monkeypatch.setattr(evaluate, "evaluation_dir", lambda run_id: tmp_path / run_id)
    out_dir = evaluate.run_benchmark(
        run_id="probe",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        label="baseline",
        smoke=True,
        dry_run=True,
    )
    import json

    manifest = json.loads((out_dir / "eval_manifest.json").read_text())
    assert out_dir.name.endswith("-smoke")
    assert manifest["full_benchmark"] is False
    assert manifest["integrity"]["is_infrastructure_smoke_test"] is True


def test_full_run_command_carries_no_limit_and_the_official_task(tmp_path):
    from miqgrpo.evaluate import OFFICIAL_TASK, build_command
    from miqgrpo.prompts import SYSTEM_PROMPT

    command = build_command(
        "model", tmp_path, "vllm", "auto", "bfloat16", 0.85, None, dry_run=True
    )
    assert "--limit" not in command
    assert OFFICIAL_TASK in command
    assert "--apply_chat_template" in command
    assert SYSTEM_PROMPT in command
    assert "--gen_kwargs" not in command


def test_generation_override_is_recorded_when_used(tmp_path, monkeypatch):
    """A deviation from the official task config must appear in the manifest."""
    import json

    import miqgrpo.evaluate as evaluate

    monkeypatch.setattr(evaluate, "evaluation_dir", lambda run_id: tmp_path / run_id)
    out_dir = evaluate.run_benchmark(
        run_id="probe",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        label="baseline",
        backend="hf",
        gen_kwargs={"max_gen_toks": 28672, "temperature": 1.0},
        dry_run=True,
    )
    manifest = json.loads((out_dir / "eval_manifest.json").read_text())
    assert manifest["generation_overrides"] == {
        "max_gen_toks": 28672,
        "temperature": 1.0,
    }
    assert "--gen_kwargs" in manifest["command"]
    assert "max_gen_toks=28672,temperature=1.0" in manifest["command"]
    assert manifest["full_benchmark"] is True


def test_no_generation_override_by_default(tmp_path, monkeypatch):
    import json

    import miqgrpo.evaluate as evaluate

    monkeypatch.setattr(evaluate, "evaluation_dir", lambda run_id: tmp_path / run_id)
    out_dir = evaluate.run_benchmark(
        run_id="probe2",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        label="baseline",
        backend="hf",
        dry_run=True,
    )
    manifest = json.loads((out_dir / "eval_manifest.json").read_text())
    assert manifest["generation_overrides"] is None
    assert "--gen_kwargs" not in manifest["command"]


def test_gen_kwargs_parsing():
    from miqgrpo.evaluate import parse_gen_kwargs

    assert parse_gen_kwargs(None) is None
    assert parse_gen_kwargs("") is None
    assert parse_gen_kwargs("a=1,b=2.5") == {"a": "1", "b": "2.5"}
    assert parse_gen_kwargs(" a = 1 , b = x ") == {"a": "1", "b": "x"}
    with pytest.raises(SystemExit, match="malformed"):
        parse_gen_kwargs("a=1,oops")


def test_shipped_evaluation_config_reproduces_the_official_sampling():
    """The HF backend must be told to sample the way vLLM did by default.

    Left unset, HF falls back to Qwen's generation_config (0.7/0.8/20/1.1),
    which is not what the official runs used -- and that difference would be
    silent.
    """
    spec = yaml.safe_load((REPO_ROOT / "configs" / "evaluation.yaml").read_text())
    assert spec["task"] == "moleculariq_pass_at_k"
    assert spec["backend"] == "vllm"
    assert spec["gen_kwargs"] == {
        "max_gen_toks": 28672,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
    }


def test_missing_checkpoint_path_fails_fast():
    """transformers turns an unresolvable path into a confusing Hub error."""
    from miqgrpo.evaluate import run_benchmark

    with pytest.raises(SystemExit, match="checkpoint not found"):
        run_benchmark(
            run_id="x",
            model_path="runs/count/final",
            label="count",
            dry_run=True,
        )


def test_hub_ids_are_not_mistaken_for_paths(tmp_path, monkeypatch):
    import miqgrpo.evaluate as evaluate

    monkeypatch.setattr(evaluate, "evaluation_dir", lambda run_id: tmp_path / run_id)
    evaluate.run_benchmark(
        run_id="probe3",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        label="baseline",
        dry_run=True,
    )


def test_evaluate_all_resolves_checkpoints_under_miq_runs(tmp_path, monkeypatch):
    """A relative runs/ path breaks whenever MIQ_RUNS points off-repo."""
    import miqgrpo.evaluate as evaluate

    runs = tmp_path / "runs"
    (runs / "count" / "final").mkdir(parents=True)
    (runs / "count" / "final" / "config.json").write_text("{}")
    monkeypatch.setattr(evaluate, "run_dir", lambda experiment: runs / experiment)
    monkeypatch.setattr(evaluate, "evaluation_dir", lambda run_id: tmp_path / "results" / run_id)
    seen = []
    monkeypatch.setattr(evaluate, "run_benchmark", lambda **kwargs: seen.append(kwargs))

    evaluate.run_all(REPO_ROOT / "configs" / "evaluation.yaml", only=["count"])
    assert [call["model_path"] for call in seen] == [str(runs / "count" / "final")]
    assert seen[0]["gen_kwargs"]["max_gen_toks"] == 28672


def test_evaluate_all_refuses_a_missing_checkpoint(tmp_path, monkeypatch):
    import miqgrpo.evaluate as evaluate

    monkeypatch.setattr(evaluate, "run_dir", lambda experiment: tmp_path / experiment)
    with pytest.raises(SystemExit, match="missing checkpoints"):
        evaluate.run_all(REPO_ROOT / "configs" / "evaluation.yaml", only=["count"])


def test_evaluate_all_skips_finished_and_refuses_interrupted_runs(tmp_path, monkeypatch):
    import miqgrpo.evaluate as evaluate

    results = tmp_path / "results"
    (results / "miq-eval-base").mkdir(parents=True)
    (results / "miq-eval-base" / "summary.json").write_text("{}")
    monkeypatch.setattr(evaluate, "evaluation_dir", lambda run_id: results / run_id)
    seen = []
    monkeypatch.setattr(evaluate, "run_benchmark", lambda **kwargs: seen.append(kwargs))
    evaluate.run_all(REPO_ROOT / "configs" / "evaluation.yaml", only=["base"])
    assert seen == []

    (results / "miq-eval-base" / "summary.json").unlink()
    (results / "miq-eval-base" / "output.log").write_text("killed")
    with pytest.raises(SystemExit, match="interrupted"):
        evaluate.run_all(REPO_ROOT / "configs" / "evaluation.yaml", only=["base"])


def test_rollout_sampling_settings_are_passed_explicitly():
    """top_k=0 must reach the sampler as 0 ("off"), never as None.

    transformers replaces a None with the model's generation_config value --
    top_k=20 for Qwen2.5-0.5B-Instruct -- so None silently changes sampling.
    """
    from miqgrpo.config import GRPOSettings
    from miqgrpo.train_grpo import sampling_kwargs

    settings = sampling_kwargs(GRPOSettings(top_k=0, top_p=1.0, temperature=1.0, repetition_penalty=1.0))
    assert settings["top_k"] == 0
    assert all(settings[key] is not None for key in ("temperature", "top_p", "top_k", "repetition_penalty"))


def test_vllm_evaluations_go_through_the_compat_wrapper(tmp_path):
    """Only the launcher differs; task, prompt and sampling arguments do not."""
    from miqgrpo.evaluate import build_command

    common = dict(model_path="m", output_path=tmp_path, batch_size="auto", dtype="bfloat16",
                  gpu_memory_utilization=0.3, limit=None,
                  gen_kwargs={"max_gen_toks": 28672, "temperature": 1.0}, dry_run=True)
    vllm = build_command(backend="vllm", **common)
    hf = build_command(backend="hf", **common)
    assert vllm[1:3] == ["-m", "miqgrpo.lm_eval_compat"]
    assert "miqgrpo.lm_eval_compat" not in hf
    tail = lambda cmd: cmd[cmd.index("--tasks"):]
    assert tail(vllm) == tail(hf)
