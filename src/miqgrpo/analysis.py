"""Loading benchmark results and training logs, and the statistics the report uses.

The headline metrics of each run come from the official harness's own results
file and are reported unchanged. Everything else here -- per-task breakdowns and
confidence intervals -- is computed from the per-item records the harness writes
with --log_samples.

Every model answers the same 5,111 items, so comparisons between two runs are
paired: the bootstrap resamples *items*, the unit of independent variation.
Evaluation itself is deterministic (seeded sampling), so these intervals capture
item-sampling uncertainty but not training-seed variance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .paths import evaluation_dir, run_dir

__all__ = [
    "METRIC",
    "TASK_LABELS",
    "EvalRun",
    "accuracy_by",
    "bootstrap_ci",
    "load_eval_run",
    "load_training_metrics",
    "paired_differences",
    "response_lengths",
    "series",
]

# Per-item metric for breakdowns: mean correctness over the three attempts.
METRIC = "avg_accuracy"

# Benchmark task types, in reporting order.
TASK_LABELS = {
    "count": "Counting",
    "index": "Index attribution",
    "generation": "Constrained generation",
}


@dataclass
class EvalRun:
    """One official benchmark run: its manifest, headline and per-item records."""

    run_id: str
    directory: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def headline(self) -> dict[str, float]:
        """The harness's own headline metrics: pass@1, pass@3, avg_accuracy."""
        return self.summary.get("metrics") or {}


def _newest(paths: Iterable[Path]) -> Path | None:
    # Newest file only (ISO timestamps sort by time): a repeated run must not double-count items.
    ordered = sorted(paths)
    return ordered[-1] if ordered else None


def load_eval_run(run_id: str) -> EvalRun:
    """Load a finished benchmark run from $MIQ_RESULTS/moleculariq/<run_id>/.

    Reads the harness summary, the evaluation manifest and the newest per-item
    sample log. Raises FileNotFoundError for a run without summary.json
    (still running or interrupted).
    """
    directory = evaluation_dir(run_id)
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"no completed evaluation '{run_id}' at {directory}")
    manifest_path = directory / "eval_manifest.json"
    samples_path = _newest(directory.rglob("samples_*.jsonl"))
    samples: list[dict[str, Any]] = []
    if samples_path is not None:
        with open(samples_path) as fh:
            samples = [json.loads(line) for line in fh if line.strip()]
    return EvalRun(
        run_id=run_id,
        directory=directory,
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {},
        summary=json.loads(summary_path.read_text()),
        samples=samples,
    )


def _key(record: dict[str, Any]) -> Any:
    doc = record.get("doc") or {}
    return record.get("doc_id", doc.get("uid"))


def _field(record: dict[str, Any], name: str) -> Any:
    doc = record.get("doc") or {}
    return doc.get(name, record.get(name))


def accuracy_by(
    samples: Iterable[dict[str, Any]], by: str = "task_type", metric: str = METRIC
) -> dict[str, float]:
    """Mean of a per-item metric, grouped by a benchmark field."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in samples:
        value = record.get(metric)
        group = _field(record, by)
        if isinstance(value, (int, float)) and group is not None:
            buckets[str(group)].append(float(value))
    return {k: sum(v) / len(v) for k, v in sorted(buckets.items()) if v}


def paired_differences(
    first: Sequence[dict[str, Any]],
    second: Sequence[dict[str, Any]],
    metric: str = METRIC,
) -> tuple[list[float], dict[str, list[float]]]:
    """Per-item first - second, overall and by task type."""
    other = {_key(record): record for record in second}
    overall: list[float] = []
    by_task: dict[str, list[float]] = defaultdict(list)
    for record in first:
        match = other.get(_key(record))
        if match is None:
            continue
        a, b = record.get(metric), match.get(metric)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        diff = float(a) - float(b)
        overall.append(diff)
        by_task[str(_field(record, "task_type"))].append(diff)
    return overall, dict(by_task)


def bootstrap_ci(
    diffs: Sequence[float], iterations: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap interval, resampling items."""
    import numpy as np

    if not diffs:
        return 0.0, 0.0, 0.0
    values = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations)
    chunk = 1_000  # bounds memory at chunk x n indices
    for start in range(0, iterations, chunk):
        stop = min(start + chunk, iterations)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def _responses(record: dict[str, Any]) -> list[str]:
    # resps is [[r1, r2, r3]]: flatten before measuring single responses.
    out: list[str] = []
    stack = list(record.get("filtered_resps") or record.get("resps") or [])
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
        elif item is not None:
            out.append(item if isinstance(item, str) else str(item))
    return out


def response_lengths(samples: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Median, 99th percentile and maximum length (characters) of all sampled answers."""
    lengths = sorted(len(r) for record in samples for r in _responses(record))
    if not lengths:
        return {}
    n = len(lengths)
    return {
        "n": n,
        "median": lengths[n // 2],
        "p99": lengths[int(n * 0.99)],
        "max": lengths[-1],
    }


def load_training_metrics(experiment_id: str) -> list[dict[str, Any]]:
    """Every logged row of $MIQ_RUNS/<experiment_id>/logs/metrics.jsonl; [] if absent."""
    path = run_dir(experiment_id) / "logs" / "metrics.jsonl"
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def series(records: Sequence[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    """(steps, values) of one logged metric, skipping rows that do not have it."""
    steps, values = [], []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            steps.append(int(record.get("step", len(steps))))
            values.append(float(value))
    return steps, values
