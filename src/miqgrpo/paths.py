"""Filesystem layout.

Every heavy artifact lives under a root that can be redirected with an
environment variable, because the same code runs in three places:

    laptop      -> repo-local ./data, ./runs, ./results
    GPU machine -> a data volume, e.g. /data/miq/...
    Leonardo    -> $WORK / $SCRATCH (login-node home is too small)

Nothing here creates directories as a side effect of import; call
ensure_dirs explicitly from a CLI entry point.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (this file is <root>/src/miqgrpo/paths.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _root(env_var: str, default: Path) -> Path:
    raw = os.environ.get(env_var)
    return Path(raw).expanduser().resolve() if raw else default


DATA_ROOT = _root("MIQ_DATA", REPO_ROOT / "data")
RUNS_ROOT = _root("MIQ_RUNS", REPO_ROOT / "runs")
RESULTS_ROOT = _root("MIQ_RESULTS", REPO_ROOT / "results")

# Frozen, versioned training datasets: data/processed/<dataset_artifact_id>/
PROCESSED_ROOT = DATA_ROOT / "processed"

# Official benchmark results: results/moleculariq/<evaluation_run_id>/
BENCHMARK_RESULTS_ROOT = RESULTS_ROOT / "moleculariq"

# The report: results/report/.
REPORT_ROOT = RESULTS_ROOT / "report"


def dataset_dir(artifact_id: str) -> Path:
    """Directory holding one frozen dataset artifact."""
    return PROCESSED_ROOT / artifact_id


def run_dir(experiment_id: str) -> Path:
    """Directory holding one training experiment."""
    return RUNS_ROOT / experiment_id


def evaluation_dir(evaluation_run_id: str) -> Path:
    """Directory holding one official benchmark run."""
    return BENCHMARK_RESULTS_ROOT / evaluation_run_id


def ensure_dirs(*paths: Path) -> None:
    """Create each directory (and its parents) if it does not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def refuse_to_overwrite(path: Path, what: str) -> None:
    """Fail closed instead of silently replacing an immutable artifact.

    Dataset artifacts, run directories and benchmark results are append-only by
    project rule; re-running a stage with an ID that already exists is a bug in
    the caller, not something to paper over.
    """
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"{what} already exists and is not empty: {path}\n"
            f"Pick a new ID instead of overwriting it "
            f"(or delete the directory by hand if you are certain)."
        )
