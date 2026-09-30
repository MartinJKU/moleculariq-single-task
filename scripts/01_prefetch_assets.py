#!/usr/bin/env python3
"""Download everything the GPU phases need."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_POOL_ID = "ml-jku/moleculariq-trainPool"
BENCHMARK_ID = "ml-jku/moleculariq-v0.0"


def prefetch_model(model_id: str) -> dict[str, str]:
    """Download a model snapshot into the HF cache; returns its id, Hub revision and local path."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id)
    from huggingface_hub import HfApi

    revision = HfApi().model_info(model_id).sha
    print(f"  model    {model_id} -> {path}")
    return {"id": model_id, "revision": revision, "local_path": path}


def prefetch_dataset(dataset_id: str, split: str = "train") -> dict[str, object]:
    """Download one split of a Hub dataset at its current revision; returns id, split, revision and row count."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(dataset_id).sha
    dataset = load_dataset(dataset_id, split=split, revision=revision)
    print(f"  dataset  {dataset_id} [{split}] -> {len(dataset)} rows @ {revision[:8]}")
    return {
        "id": dataset_id,
        "split": split,
        "revision": revision,
        "rows": len(dataset),
    }


def verify_offline() -> list[str]:
    """Re-resolve everything with the Hub switched off."""
    problems: list[str] = []
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(MODEL_ID)
        print("  offline: tokenizer OK")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"tokenizer: {exc}")

    for dataset_id, split in ((TRAIN_POOL_ID, "train"), (BENCHMARK_ID, "test")):
        try:
            from datasets import load_dataset

            load_dataset(dataset_id, split=split)
            print(f"  offline: {dataset_id} [{split}] OK")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{dataset_id}: {exc}")
    return problems


def main() -> None:
    """Prefetch the policy, the training pool and the benchmark, check they load offline, and write assets_manifest.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="skip the official test set (only useful if you evaluate elsewhere)",
    )
    parser.add_argument("--out", type=Path, default=Path("assets_manifest.json"))
    args = parser.parse_args()

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        sys.exit("HF_HUB_OFFLINE=1 is set; this script needs the network")

    print(f"HF_HOME = {os.environ.get('HF_HOME', '(default)')}")
    print("downloading:")
    manifest: dict[str, object] = {
        "model": prefetch_model(args.model),
        "train_pool": prefetch_dataset(TRAIN_POOL_ID, "train"),
    }
    if not args.skip_benchmark:
        manifest["benchmark"] = prefetch_dataset(BENCHMARK_ID, "test")
        manifest["benchmark_note"] = (
            "test-only; never read by preprocessing or training"
        )

    print("verifying the offline path:")
    problems = verify_offline()

    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {args.out}")

    if problems:
        print("\noffline resolution FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("all assets resolve with HF_HUB_OFFLINE=1")


if __name__ == "__main__":
    main()
