"""Worked solutions for a supervised warm-up before GRPO (the extension study).

    python -m miqgrpo.sft_data build --config extension/configs/sft_data.yaml

Builds a frozen dataset of (prompt, completion) pairs from the *training*
splits of a frozen GRPO dataset. The prompt is the question exactly as GRPO and
the benchmark render it. The completion writes out what a careful solver would:

    Atoms: C(0), C(1), O(2), c(3), ...                  (every question)
    halogen_atom_count: Cl(7), Br(12) -> 2              (atoms behind each property)
    hydrogen_atom_count: C(0) 3, C(1) 2, O(2) 1 -> 6
    <answer>{"halogen_atom_count": 2, "hydrogen_atom_count": 6}</answer>

and, for constrained generation, the molecule the question was built from with
its value for each constraint. The atom notation is the one the official system
prompt itself uses ("CCO": C(0), C(1), O(2)); lowercase marks an aromatic atom.

Everything in a completion comes from the stored targets and the official
property code (moleculariq_core via PropertyEngine) applied to the
molecule the question shows. Every completion is scored by the official
extractor and verifier before it is kept, and must be fully correct. The dev
splits and the benchmark are never read.
"""

from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import yaml

from .paths import dataset_dir, ensure_dirs, refuse_to_overwrite
from .provenance import capture, content_hash, hash_config, sha256_tree, write_json

__all__ = ["atom_labels", "completion_for", "build", "verify", "main"]

TASK_FAMILIES = ("count", "index", "constraint_generation")


def _parse(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")
    return mol


def atom_labels(smiles: str) -> list[str]:
    """C(0)-style labels in the benchmark's indexing (left to right, heavy atoms).

    RDKit keeps atoms in the order they are written and drops plain [H]
    while keeping isotopic hydrogens, which is exactly the official convention
    ("skip [H], include [2H]/[3H]"); tests/test_sft_data.py checks it against
    the system prompt's own examples.
    """
    labels = []
    for atom in _parse(smiles).GetAtoms():
        symbol = atom.GetSymbol()
        if atom.GetIsotope():
            symbol = f"[{atom.GetIsotope()}{symbol}]"
        elif atom.GetIsAromatic():
            symbol = symbol.lower()
        labels.append(f"{symbol}({atom.GetIdx()})")
    return labels


def _atoms(labels: list[str], indices: Any) -> str:
    if not isinstance(indices, list):
        raise ValueError("not an index list")
    if not indices:
        return "none"
    return ", ".join(labels[i] for i in indices)


def _hydrogen_line(key: str, smiles: str, labels: list[str], value: Any) -> str:
    mol = _parse(smiles)
    parts = [f"{labels[a.GetIdx()]} {a.GetTotalNumHs()}" for a in mol.GetAtoms() if a.GetTotalNumHs()]
    return f"{key}: {', '.join(parts) or 'none'} -> {value}"


def _formula_line(key: str, smiles: str, value: Any) -> str:
    from rdkit.Chem import rdMolDescriptors

    formula = rdMolDescriptors.CalcMolFormula(_parse(smiles))
    if formula != value:
        raise ValueError(f"RDKit formula {formula} differs from the stored {value}")
    mol = _parse(smiles)
    tally: dict[str, int] = {}
    for atom in mol.GetAtoms():
        tally[atom.GetSymbol()] = tally.get(atom.GetSymbol(), 0) + 1
    hydrogens = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    if hydrogens:
        tally["H"] = tally.get("H", 0) + hydrogens
    parts = ", ".join(f"{element} {n}" for element, n in sorted(tally.items()))
    return f"{key}: {parts} -> {value}"


def _property_line(key: str, value: Any, smiles: str, labels: list[str], engine: Any) -> str:
    """One line per requested property: the atoms behind it, then its value."""
    if key == "hydrogen_atom_count":
        return _hydrogen_line(key, smiles, labels, value)
    if key == "molecular_formula_count":
        return _formula_line(key, smiles, value)
    if isinstance(value, list):  # an index property: the atoms are the answer
        return f"{key}: {_atoms(labels, value)} -> {json.dumps(value)}"
    if key.endswith("_count"):
        twin = key[: -len("_count")] + "_index"
        try:
            indices = engine.compute(smiles, twin)
            return f"{key}: {_atoms(labels, indices)} -> {value}"
        except Exception:  # noqa: BLE001 - no atom list for this property
            pass
    return f"{key} -> {value}"


def _constraint_text(constraint: dict[str, Any]) -> str:
    name = constraint.get("type") or constraint.get("property")
    operator = constraint.get("operator", "=")
    if "value" in constraint:
        return f"{name} {operator} {constraint['value']}"
    bounds = {k: v for k, v in constraint.items() if k not in ("type", "property", "operator")}
    return f"{name} {operator} {json.dumps(bounds, sort_keys=True)}"


def completion_for(row: dict[str, Any], engine: Any) -> str:
    """The worked solution for one frozen training row (see the module docstring)."""
    family = row["task_family"]
    if family == "constraint_generation":
        constraints = json.loads(row["constraints_json"])
        smiles = row["witness_smiles"]
        lines = ["Constraints: " + "; ".join(_constraint_text(c) for c in constraints) + "."]
        lines.append(f"Candidate: {smiles}")
        checks = []
        for constraint in constraints:
            name = constraint.get("type") or constraint.get("property")
            checks.append(f"{name} = {json.dumps(engine.compute(smiles, name), default=str)}")
        lines.append("Check: " + "; ".join(checks))
        answer = {"smiles": smiles}
    else:
        smiles = row["question_smiles"]
        target = json.loads(row["target_json"])
        labels = atom_labels(smiles)
        lines = ["Atoms: " + ", ".join(labels)]
        lines += [_property_line(key, value, smiles, labels, engine) for key, value in target.items()]
        answer = target
    return "\n".join(lines) + "\n<answer>" + json.dumps(answer) + "</answer>"


_ENGINE = None


def _work(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Worker: trace one row and keep it only if the official scorer marks it correct."""
    global _ENGINE
    from .generation import PropertyEngine
    from .rewards import score_completion

    if _ENGINE is None:
        _ENGINE = PropertyEngine()
    try:
        text = completion_for(row, _ENGINE)
    except Exception as error:  # noqa: BLE001 - recorded as a drop reason
        return None, f"trace failed: {type(error).__name__}"
    scored = score_completion(text, row["task_type"], row["target_json"], row["constraints_json"])
    if scored.correctness != 1.0:
        return None, f"not verified: {scored.status}"
    return {
        "example_id": row["example_id"].replace("miq-train", "miq-sft", 1),
        "source_example_id": row["example_id"],
        "task_family": row["task_family"],
        "task_type": row["task_type"],
        "prompt": row["prompt"],
        "completion": [{"role": "assistant", "content": text}],
        "target_json": row["target_json"],
        "constraints_json": row["constraints_json"],
    }, None


def build(config_path: Path, overwrite: bool = False) -> Path:
    """Build the supervised warm-up dataset from a frozen training set.

    Reads the source artifact (refusing a different content hash). Per family,
    it writes worked solutions for training questions in a seeded order and
    keeps the first examples_per_family whose completion the official
    scorer accepts; dev splits are never read. The result is frozen with its
    own content hash. Returns the artifact directory.
    """
    from datasets import Dataset, DatasetDict, load_from_disk
    from transformers import AutoTokenizer

    config = yaml.safe_load(config_path.read_text())
    artifact_id = config["dataset_artifact_id"]
    source = config["source"]
    out_dir = dataset_dir(artifact_id)
    if not overwrite:
        refuse_to_overwrite(out_dir, f"SFT dataset '{artifact_id}'")
    ensure_dirs(out_dir)

    source_dir = dataset_dir(source["artifact_id"])
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    if source_manifest.get("content_hash") != source["expected_content_hash"]:
        raise ValueError(f"source dataset '{source['artifact_id']}' is not the pinned one")
    frozen = load_from_disk(str(source_dir / "dataset"))
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer", "Qwen/Qwen2.5-0.5B-Instruct"))

    splits, stats = {}, {}
    for family in TASK_FAMILIES:
        n = int(config["examples_per_family"][family])
        rows = frozen[family]  # the training split only; *_dev is never read
        order = list(range(len(rows)))
        random.Random(f"{config['seed']}:{family}").shuffle(order)
        # Trace twice the target so dropped rows can be replaced in seeded order.
        candidates = [rows[i] for i in order[: 2 * n]]
        with Pool(int(config.get("n_workers", 8))) as pool:
            results = pool.map(_work, candidates, chunksize=64)
        kept, drops = [], {}
        for record, reason in results:
            if record is None:
                drops[reason] = drops.get(reason, 0) + 1
            elif len(kept) < n:
                kept.append(record)
        if len(kept) < n:
            raise ValueError(f"{family}: only {len(kept)} of {n} traced rows verified")
        lengths = sorted(len(tokenizer(r["completion"][0]["content"])["input_ids"]) for r in kept)
        stats[family] = {
            "examples": len(kept),
            "dropped": drops,
            "completion_tokens": {q: lengths[int(q / 100 * (len(lengths) - 1))] for q in (50, 90, 99, 100)},
        }
        splits[family] = kept
        print(f"  {family}: {len(kept)} examples, dropped {drops}, tokens {stats[family]['completion_tokens']}")

    dataset = DatasetDict({name: Dataset.from_list(rows) for name, rows in splits.items()})
    dataset_path = out_dir / "dataset"
    dataset.save_to_disk(str(dataset_path))
    manifest = {
        "dataset_artifact_id": artifact_id,
        "source_artifact_id": source["artifact_id"],
        "source_content_hash": source_manifest["content_hash"],
        "source_splits_read": list(TASK_FAMILIES),
        "config": config,
        "config_hash": hash_config(config),
        "counts_by_split": {name: len(split) for name, split in dataset.items()},
        "num_examples": sum(len(split) for split in dataset.values()),
        "stats": stats,
        "integrity": {
            "official_benchmark_used_as_input": False,
            "dev_splits_read": False,
            "every_completion_verified_by_official_scorer": True,
        },
        "content_hash": content_hash(dataset),
        "artifact_file_hash": sha256_tree(dataset_path),
    }
    write_json(out_dir / "manifest.json", manifest)
    with open(out_dir / "preprocessing_config.yaml", "w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    write_json(out_dir / "provenance.json", capture("sft_data", {"artifact_id": artifact_id}))
    print(f"  content hash: {manifest['content_hash']}")
    expected = config.get("expected_content_hash")
    if expected and expected != manifest["content_hash"]:
        print(f"  ! differs from the pinned hash {expected}")
    return out_dir


def verify(artifact_id: str) -> None:
    """Reload the artifact and re-score every completion with the official scorer."""
    from datasets import load_from_disk

    from .rewards import score_completion

    out_dir = dataset_dir(artifact_id)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    dataset = load_from_disk(str(out_dir / "dataset"))
    failures = []
    if content_hash(dataset) != manifest["content_hash"]:
        failures.append("content hash changed")
    for name, split in dataset.items():
        for row in split:
            text = row["completion"][0]["content"]
            if score_completion(text, row["task_type"], row["target_json"], row["constraints_json"]).correctness != 1.0:
                failures.append(f"{row['example_id']}: completion no longer verifies")
    if failures:
        raise SystemExit("verification failed:\n  " + "\n  ".join(failures[:20]))
    print(f"verified {manifest['num_examples']} SFT examples; content hash {manifest['content_hash']}")


def main(argv: list[str] | None = None) -> None:
    """Command line: build --config <yaml> or verify --artifact <id>."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--config", type=Path, required=True)
    build_parser.add_argument("--overwrite", action="store_true")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--artifact", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.config, overwrite=args.overwrite)
    else:
        verify(args.artifact)


if __name__ == "__main__":
    main()
