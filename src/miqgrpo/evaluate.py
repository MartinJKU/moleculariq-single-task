"""Official MolecularIQ benchmark evaluation.

    python -m miqgrpo.evaluate all --config configs/evaluation.yaml
    python -m miqgrpo.evaluate run --run-id <id> --model-path <path> --label <label> ...

This shells out to the official moleculariq-eval harness (a fork of
lm-evaluation-harness with the MolecularIQ task built in) and runs the **whole**
benchmark: all 5,111 test items, the official moleculariq_pass_at_k task
config, the official system instruction, the official extraction and the
official verifier. Nothing here re-implements scoring.

The benchmark is test-only. --limit is refused unless --smoke is passed,
and a smoke run is written to a separate directory with full_benchmark:
false stamped in its manifest so it can never be quoted as a result.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .paths import ensure_dirs, evaluation_dir, refuse_to_overwrite, run_dir
from .prompts import SYSTEM_PROMPT
from .provenance import capture, environment_report, sha256_tree, write_json

__all__ = ["run_benchmark", "main"]

# Official task; the system-prompt variant, since Qwen2.5-Instruct is a chat model.
OFFICIAL_TASK = "moleculariq_pass_at_k"

# Metrics the official task emits and that we report verbatim.
OFFICIAL_METRICS = ("pass_at_1", "pass_at_3", "avg_accuracy")


def _require_lm_eval(optional: bool = False) -> str:
    """Locate the official harness. optional keeps --dry-run usable."""
    executable = shutil.which("lm_eval")
    if executable is not None:
        return executable
    if optional:
        return "lm_eval"
    raise SystemExit(
        "lm_eval not found on PATH.\n"
        "Install the official harness:\n"
        "  git clone https://github.com/ml-jku/moleculariq-eval.git\n"
        "  pip install -e 'moleculariq-eval[vllm]'\n"
        "  pip install moleculariq-core rdkit"
    )


def _harness_version() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import lm_eval

        info["lm_eval_version"] = getattr(lm_eval, "__version__", None)
        info["lm_eval_path"] = str(Path(lm_eval.__file__).parent)
        repo = Path(lm_eval.__file__).parent.parent
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        info["moleculariq_eval_commit"] = (
            commit.stdout.strip() if commit.returncode == 0 else None
        )
    except Exception as exc:  # noqa: BLE001
        info["lm_eval_import_error"] = str(exc)
    return info


def _training_provenance(experiment_id: str | None) -> dict[str, Any]:
    """Pull the training side of the chain so the result is self-describing."""
    if not experiment_id:
        return {}
    directory = run_dir(experiment_id)
    provenance: dict[str, Any] = {"experiment_id": experiment_id}
    for name, key in (
        ("provenance.json", "training_provenance"),
        ("training_summary.json", "training_summary"),
    ):
        path = directory / name
        if path.exists():
            provenance[key] = json.loads(path.read_text())
    config_path = directory / "frozen_config.yaml"
    if config_path.exists():
        provenance["frozen_config"] = yaml.safe_load(config_path.read_text())
    return provenance


def build_command(
    model_path: str,
    output_path: Path,
    backend: str,
    batch_size: str,
    dtype: str,
    gpu_memory_utilization: float,
    limit: int | None,
    gen_kwargs: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> list[str]:
    """The official harness command line for one benchmark run.

    The vLLM backend is started through miqgrpo.lm_eval_compat (two
    compatibility shims for vLLM 0.30); everything else is the harness's own
    CLI with the official task. gen_kwargs are passed through unchanged.
    """
    if backend == "vllm":
        model_args = (
            f"pretrained={model_path},dtype={dtype},"
            f"gpu_memory_utilization={gpu_memory_utilization}"
        )
    else:
        model_args = f"pretrained={model_path},dtype={dtype}"

    # vLLM backend via miqgrpo.lm_eval_compat (two vLLM 0.30 shims); the CLI itself is unmodified.
    launcher = (
        [sys.executable, "-m", "miqgrpo.lm_eval_compat"]
        if backend == "vllm"
        else [_require_lm_eval(optional=dry_run)]
    )
    command = [
        *launcher,
        "--model",
        backend,
        "--model_args",
        model_args,
        "--tasks",
        OFFICIAL_TASK,
        "--apply_chat_template",
        "--system_instruction",
        SYSTEM_PROMPT,
        "--batch_size",
        batch_size,
        "--log_samples",
        "--output_path",
        str(output_path),
    ]
    if gen_kwargs:
        # max_gen_toks: the task's 32768 equals the context window and makes the harness assert.
        # Sampling parameters: the vLLM defaults the official runs used (docs/official-semantics.md).
        rendered = ",".join(f"{key}={value}" for key, value in gen_kwargs.items())
        command += ["--gen_kwargs", rendered]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def _collect_results(raw_dir: Path) -> dict[str, Any]:
    """Find the harness's own result JSON; do not recompute anything from it."""
    candidates = sorted(raw_dir.rglob("results_*.json"))
    if not candidates:
        return {}
    payload = json.loads(candidates[-1].read_text())
    task_results = (payload.get("results") or {}).get(OFFICIAL_TASK, {})
    headline = {
        metric: task_results.get(f"{metric},all", task_results.get(metric))
        for metric in OFFICIAL_METRICS
    }
    return {
        "results_file": str(candidates[-1]),
        "task_results": task_results,
        "headline": headline,
        "n_samples": payload.get("n-samples"),
        "config": payload.get("config"),
        "versions": payload.get("versions"),
    }


def run_benchmark(
    run_id: str,
    model_path: str,
    label: str,
    experiment_id: str | None = None,
    backend: str = "hf",
    batch_size: str = "auto",
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.85,
    limit: int | None = None,
    gen_kwargs: dict[str, Any] | None = None,
    smoke: bool = False,
    dry_run: bool = False,
) -> Path:
    """Run the whole official benchmark on one model and record everything about it.

    Writes $MIQ_RESULTS/moleculariq/<run_id>/: eval_manifest.json and
    .yaml (command, checkpoint hash, generation settings, training
    provenance), environment.txt, output.log, the harness output under
    raw/ and summary.json with the headline metrics. Refuses to
    overwrite an existing run. limit is only allowed together with
    smoke, which writes to <run_id>-smoke and marks the run as an
    infrastructure test.
    """
    if limit is not None and not smoke:
        raise SystemExit(
            "--limit truncates the benchmark, so the result would not be a "
            "MolecularIQ score. Re-run with --smoke to mark it as an "
            "infrastructure test instead."
        )
    if smoke and limit is None:
        limit = 10

    _check_model_path(model_path)

    out_dir = evaluation_dir(run_id if not smoke else f"{run_id}-smoke")
    refuse_to_overwrite(out_dir, f"benchmark result '{run_id}'")
    raw_dir = out_dir / "raw"
    ensure_dirs(out_dir, raw_dir)

    command = build_command(
        model_path,
        raw_dir,
        backend,
        batch_size,
        dtype,
        gpu_memory_utilization,
        limit,
        gen_kwargs=gen_kwargs,
        dry_run=dry_run,
    )

    checkpoint_hash = None
    local_checkpoint = Path(model_path)
    if local_checkpoint.exists() and local_checkpoint.is_dir():
        checkpoint_hash = sha256_tree(local_checkpoint)

    manifest: dict[str, Any] = {
        "evaluation_run_id": run_id,
        "label": label,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": model_path,
        "checkpoint_hash": checkpoint_hash,
        "task": OFFICIAL_TASK,
        "backend": backend,
        "batch_size": batch_size,
        "dtype": dtype,
        "limit": limit,
        "full_benchmark": limit is None,
        "system_instruction": SYSTEM_PROMPT,
        "system_instruction_source": "moleculariq-eval task_processor.SYSTEM_PROMPT",
        "apply_chat_template": True,
        # Everything else comes from the official task YAML.
        "generation_overrides": dict(gen_kwargs) if gen_kwargs else None,
        "command": command,
        "harness": _harness_version(),
        "training": _training_provenance(experiment_id),
        "integrity": {
            "benchmark_used_for_training": False,
            "benchmark_used_for_hparam_selection": False,
            "benchmark_used_for_prompt_tuning": False,
            "benchmark_used_for_extraction_tuning": False,
            "benchmark_used_for_checkpoint_selection": False,
            "is_infrastructure_smoke_test": bool(smoke),
        },
    }

    print(f"evaluation run : {run_id}")
    print(f"model          : {model_path}")
    print(f"task           : {OFFICIAL_TASK} (whole benchmark: {limit is None})")
    print(f"output         : {out_dir}")
    print()
    print("command:")
    print("  " + " ".join(_quote(part) for part in command))
    print()

    if dry_run:
        write_json(out_dir / "eval_manifest.json", manifest)
        print("dry run; nothing executed")
        return out_dir

    (out_dir / "environment.txt").write_text(
        json.dumps(environment_report(), indent=2, default=str) + "\n"
    )

    returncode = _run_streaming(command, out_dir / "output.log")

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["returncode"] = returncode
    manifest.update(_collect_results(raw_dir))
    write_json(out_dir / "eval_manifest.json", manifest)
    with open(out_dir / "eval_manifest.yaml", "w") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    write_json(out_dir / "provenance.json", capture("evaluate", {"run_id": run_id}))

    if returncode != 0:
        print(f"lm_eval exited {returncode}; full output in {out_dir / 'output.log'}")
        sys.exit(returncode)

    headline = manifest.get("headline") or {}
    print("official metrics:")
    for metric, value in headline.items():
        print(f"  {metric:14s} {value}")
    summary = {
        "evaluation_run_id": run_id,
        "label": label,
        "checkpoint": model_path,
        "full_benchmark": manifest["full_benchmark"],
        "metrics": headline,
        "task_results": manifest.get("task_results", {}),
    }
    write_json(out_dir / "summary.json", summary)
    return out_dir


def _run_streaming(command: list[str], log_path: Path) -> int:
    """Run lm_eval, showing its output live *and* keeping a copy on disk.

    Swallowing the harness's output into a file meant a multi-hour run showed
    nothing at all after the launch banner -- indistinguishable from a hang.
    Read raw chunks rather than lines so tqdm's carriage-return progress bars
    come through as they are written instead of buffering until a newline.

    stderr is merged into stdout: the progress bar lives on stderr, and one
    interleaved log is easier to read after the fact than two half-stories.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    assert process.stdout is not None
    with open(log_path, "wb") as fh:
        while True:
            chunk = process.stdout.read1(65536)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            fh.write(chunk)
            fh.flush()
    return process.wait()


def _check_model_path(model_path: str) -> None:
    """Catch a local checkpoint path that does not exist, before loading.

    transformers treats an unresolvable path as a Hub repo id, so a wrong
    checkpoint path surfaces minutes later as a cryptic "Repo id must be in the
    form 'repo_name' or 'namespace/repo_name'" instead of "that directory is not
    there". A Hub id has exactly one slash and no leading separator; anything
    else that does not exist on disk is a broken path.
    """
    if Path(model_path).exists():
        return
    looks_like_hub_id = (
        model_path.count("/") == 1
        and not model_path.startswith(("/", ".", "~"))
    )
    if looks_like_hub_id:
        return
    raise SystemExit(
        f"checkpoint not found: {model_path}\n"
        f"This looks like a filesystem path, but nothing is there. transformers "
        f"would fall back to treating it as a Hugging Face repo id and fail with "
        f"a confusing error several minutes from now.\n"
        f"If your checkpoints live on a volume, check $MIQ_RUNS."
    )


def parse_gen_kwargs(raw: str | None) -> dict[str, Any] | None:
    """Parse k=v,k=v into an ordered mapping, preserving written form."""
    if not raw:
        return None
    parsed: dict[str, Any] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"malformed --gen-kwargs entry {item!r}; expected key=value")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed or None


def run_all(config_path: Path, only: list[str] | None = None, dry_run: bool = False) -> None:
    """Evaluate every model listed in an evaluation config, skipping finished ones.

    A run is finished when its summary.json exists -- that file is only
    written after lm_eval exits cleanly. A directory without it is an
    interrupted run and must be removed by hand: result directories are
    append-only, so nothing here deletes one.
    """
    spec = yaml.safe_load(config_path.read_text())
    gen_kwargs = dict(spec.get("gen_kwargs") or {}) or None
    runs = [r for r in spec["runs"] if not only or r["label"] in only]

    # Check every checkpoint first, not after hours of evaluation.
    missing = [
        str(run_dir(r["experiment"]) / "final")
        for r in runs
        if r.get("experiment") and not (run_dir(r["experiment"]) / "final" / "config.json").exists()
    ]
    if missing and not dry_run:
        raise SystemExit("missing checkpoints (train them first, or check $MIQ_RUNS):\n  " + "\n  ".join(missing))

    for run in runs:
        out_dir = evaluation_dir(run["run_id"])
        model_path = run.get("model") or str(run_dir(run["experiment"]) / "final")
        if model_path in missing:
            print(f"  no checkpoint, skipped in dry run: {run['run_id']}")
            continue
        if (out_dir / "summary.json").exists():
            print(f"  done       {run['run_id']}")
            continue
        if out_dir.exists() and any(out_dir.iterdir()):
            raise SystemExit(
                f"{out_dir} exists but has no summary.json: an interrupted run. "
                f"Delete it and re-run this command."
            )
        print(f"  evaluating {run['run_id']}")
        run_benchmark(
            run_id=run["run_id"],
            model_path=model_path,
            label=run["label"],
            experiment_id=run.get("experiment"),
            backend=run.get("backend", spec.get("backend", "hf")),
            batch_size=str(run.get("batch_size", "auto")),
            dtype=spec.get("dtype", "bfloat16"),
            gpu_memory_utilization=float(
                run.get("gpu_memory_utilization", spec.get("gpu_memory_utilization", 0.85))
            ),
            gen_kwargs=gen_kwargs,
            dry_run=dry_run,
        )


def _quote(part: str) -> str:
    return f"'{part}'" if any(c in part for c in " \n\"'") else part


def main(argv: list[str] | None = None) -> None:
    """Command line: run evaluates one model, all every run in an evaluation config."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the official benchmark on one model")
    run.add_argument("--run-id", required=True, help="unique evaluation_run_id")
    run.add_argument("--model-path", required=True, help="HF model id or checkpoint dir")
    run.add_argument("--label", required=True, help="short name used in plots")
    run.add_argument("--experiment", default=None, help="training experiment_id, if any")
    # The reported results use vllm (configs/evaluation.yaml); hf gives the same scores, slower.
    run.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    run.add_argument("--batch-size", default="auto")
    run.add_argument("--dtype", default="bfloat16")
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run.add_argument(
        "--gen-kwargs",
        default=None,
        help=(
            "comma-separated key=value generation overrides passed through to "
            "lm_eval, e.g. 'max_gen_toks=28672,temperature=1.0'. Recorded in the "
            "run manifest under generation_overrides"
        ),
    )
    run.add_argument("--limit", type=int, default=None)
    run.add_argument(
        "--smoke",
        action="store_true",
        help="infrastructure test on a handful of items; never a reportable score",
    )
    run.add_argument("--dry-run", action="store_true")

    every = sub.add_parser("all", help="evaluate every model in an evaluation config")
    every.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    every.add_argument("--only", default=None, help="comma-separated labels to restrict to")
    every.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "all":
        run_all(
            args.config,
            only=args.only.split(",") if args.only else None,
            dry_run=args.dry_run,
        )
        return
    run_benchmark(
        run_id=args.run_id,
        model_path=args.model_path,
        label=args.label,
        experiment_id=args.experiment,
        backend=args.backend,
        batch_size=args.batch_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit=args.limit,
        gen_kwargs=parse_gen_kwargs(args.gen_kwargs),
        smoke=args.smoke,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
