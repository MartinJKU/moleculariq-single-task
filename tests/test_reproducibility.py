"""Reported numbers are reproducible and traceable to their configs and data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from miqgrpo.build_dataset import family_seed, reference_molecules
from miqgrpo.config import load_experiment_config
from miqgrpo.paths import REPO_ROOT
from miqgrpo.provenance import content_hash

CONFIGS = REPO_ROOT / "configs"
DATASET = yaml.safe_load((CONFIGS / "dataset.yaml").read_text())
EVALUATION = yaml.safe_load((CONFIGS / "evaluation.yaml").read_text())
REPORT = yaml.safe_load((CONFIGS / "report.yaml").read_text())
EXPERIMENTS = {p.stem: p for p in sorted((CONFIGS / "experiments").glob("*.yaml"))}


def test_content_hash_ignores_last_digit_float_noise():
    """RDKit's Bertz complexity differs in the last bit between arm64 and x86_64."""
    a = {"train": [{"molecular_complexity": 237.3841741815625, "smiles": "CCO"}]}
    b = {"train": [{"molecular_complexity": 237.38417418156246, "smiles": "CCO"}]}
    assert content_hash(a) == content_hash(b)


def test_content_hash_sees_real_changes():
    base = {"train": [{"value": 237.38417, "smiles": "CCO"}]}
    assert content_hash(base) != content_hash({"train": [{"value": 237.38418, "smiles": "CCO"}]})
    assert content_hash(base) != content_hash({"train": [{"value": 237.38417, "smiles": "CCN"}]})
    assert content_hash(base) != content_hash({"dev": base["train"]})


def test_content_hash_depends_on_row_order_but_not_key_order():
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert content_hash({"train": rows}) == content_hash({"train": [{"b": 2, "a": 1}, {"b": 4, "a": 3}]})
    assert content_hash({"train": rows}) != content_hash({"train": rows[::-1]})


def test_content_hash_is_stable():
    assert (
        content_hash({"train": [{"a": 1.5, "b": "x"}]})
        == "5169e2d2e159e61c6845d75c8816fccf132d84f73d87c98ceed345bad5e2d921"
    )


def test_family_seeds_do_not_depend_on_the_process():
    """Python's hash() of a str is randomised per process; CRC32 is not."""
    config = {"generation": {"seed": 42}}
    assert family_seed(config, "count") == 8588
    assert family_seed(config, "index") == 6699
    assert family_seed(config, "constraint_generation") == 3085


def test_pinned_family_seeds_take_precedence():
    config = {"generation": {"seed": 42, "family_seeds": {"count": 7}}}
    assert family_seed(config, "count") == 7
    assert family_seed(config, "index") == 6699


def test_shipped_dataset_config_pins_its_source_and_content():
    assert DATASET["source"]["revision"], "the training pool revision must be pinned"
    assert DATASET["generation"]["seed"] == 42
    assert len(DATASET["expected_content_hash"]) == 64


def test_reference_molecules_are_a_deterministic_sample():
    molecules = [f"C{'C' * i}O" for i in range(100)]
    config = {"generation": {"seed": 42, "n_reference_molecules": 10}}
    first = reference_molecules(config, molecules)
    assert first == reference_molecules(config, molecules)
    assert len(first) == 10 and set(first) <= set(molecules)


@pytest.mark.parametrize("path", EXPERIMENTS.values(), ids=lambda p: p.stem)
def test_every_experiment_is_pinned_to_the_dataset(path):
    config = load_experiment_config(path)
    assert config.data.artifact_id == DATASET["dataset_artifact_id"]
    assert config.data.expected_hash == DATASET["expected_content_hash"]


def test_evaluation_runs_are_unique_and_point_at_real_experiments():
    run_ids = [run["run_id"] for run in EVALUATION["runs"]]
    labels = [run["label"] for run in EVALUATION["runs"]]
    assert len(set(run_ids)) == len(run_ids)
    assert len(set(labels)) == len(labels)
    for run in EVALUATION["runs"]:
        assert ("model" in run) != ("experiment" in run), run["run_id"]
        if "experiment" in run:
            assert run["experiment"] in EXPERIMENTS, run["experiment"]


def test_every_trained_model_is_evaluated_or_says_why_not():
    evaluated = {run.get("experiment") for run in EVALUATION["runs"]}
    skipped = set(EVALUATION.get("not_evaluated") or {})
    assert set(EXPERIMENTS) <= evaluated | skipped
    assert not evaluated & skipped


def test_the_baseline_is_the_untrained_base_model():
    baseline = next(run for run in EVALUATION["runs"] if run["label"] == "base")
    assert baseline["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert "experiment" not in baseline


def _evaluation(run_id: str) -> dict:
    return next(run for run in EVALUATION["runs"] if run["run_id"] == run_id)


def test_report_covers_the_base_model_and_every_trained_model():
    assert _evaluation(REPORT["baseline"])["label"] == "base"
    assert {model["training"] for model in REPORT["models"]} == set(EXPERIMENTS)
    assert REPORT["dataset"] == DATASET["dataset_artifact_id"]


def test_report_evaluations_belong_to_the_stated_training_runs():
    for model in REPORT["models"]:
        assert _evaluation(model["evaluation"])["experiment"] == model["training"]
        config = load_experiment_config(EXPERIMENTS[model["training"]])
        assert model["dev"]["split"] == config.data.dev_split


def test_report_builds_from_minimal_results(tmp_path, monkeypatch):
    """End to end on synthetic results: every file, nothing read from real results."""
    import miqgrpo.analysis as analysis
    import miqgrpo.report as report

    results, runs = tmp_path / "results", tmp_path / "runs"
    for index, run in enumerate(EVALUATION["runs"]):
        directory = results / run["run_id"]
        (directory / "raw").mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps(
            {"metrics": {"pass_at_1": 0.02, "pass_at_3": 0.04, "avg_accuracy": 0.02}}
        ))
        (directory / "eval_manifest.json").write_text("{}")
        with open(directory / "raw" / "samples_x.jsonl", "w") as fh:
            for item in range(60):
                task = ("count", "index", "generation")[item % 3]
                fh.write(json.dumps({
                    "doc_id": item,
                    "doc": {"task_type": task, "complexity_bin": "0-250", "multi_task_load": 1},
                    "avg_accuracy": ((item + index) % 5) / 4,
                    "resps": [["<answer>{}</answer>"] * 3],
                }) + "\n")
    for experiment in EXPERIMENTS:
        (runs / experiment / "logs").mkdir(parents=True)
        with open(runs / experiment / "logs" / "metrics.jsonl", "w") as fh:
            for step in range(5, 55, 5):
                fh.write(json.dumps({"step": step, "reward": 0.1, "entropy": 0.5,
                                     "partial/nontrivial_credit": 0.05}) + "\n")
            fh.write(json.dumps({"step": 50, "eval_reward": 0.1,
                                 "eval_correctness/other_questions": 0.05}) + "\n")
    monkeypatch.setattr(analysis, "evaluation_dir", lambda run_id: results / run_id)
    monkeypatch.setattr(analysis, "run_dir", lambda experiment: runs / experiment)
    monkeypatch.setattr(report, "RESULTS_ROOT", results)
    monkeypatch.setattr(report, "dataset_dir", lambda artifact: tmp_path / "no-dataset")

    out = tmp_path / "report"
    report.main(["--spec", str(CONFIGS / "report.yaml"), "--out", str(out)])

    files = {p.name for p in out.iterdir()}
    assert {"analysis.txt", "summary.csv", "benchmark_overall.png", "benchmark_by_task_type.png",
            "transfer_matrix.png", "where_accuracy_comes_from.png", "dev_learning_curves.png",
            "training_curves.png", "training_diagnostics.png"} <= files
    text = (out / "analysis.txt").read_text()
    assert "headline (all items)" in text and "25_dev_evaluate" in text
