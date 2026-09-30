"""Worked solutions use the benchmark's atom indexing and verify with the official scorer."""

from __future__ import annotations

import json
import re

import pytest

from miqgrpo.generation import PropertyEngine
from miqgrpo.prompts import SYSTEM_PROMPT
from miqgrpo.rewards import score_completion
from miqgrpo.sft_data import atom_labels, completion_for


@pytest.fixture(scope="module")
def engine():
    return PropertyEngine()


def test_atom_labels_reproduce_the_system_prompts_examples():
    examples = re.findall(r'- "([^"]+)": ((?:\w+\(\d+\)(?:, )?)+)', SYSTEM_PROMPT)
    assert len(examples) == 3
    for smiles, listing in examples:
        assert ", ".join(atom_labels(smiles)) == listing


def test_atom_labels_skip_plain_hydrogens_and_keep_isotopes():
    assert atom_labels("[H]OC([2H])C") == ["O(0)", "C(1)", "[2H](2)", "C(3)"]


def test_aromatic_atoms_are_lowercase_even_when_the_smiles_is_kekulised():
    assert atom_labels("C1=CC=CC=C1Cl") == atom_labels("c1ccccc1Cl")
    assert atom_labels("c1ccccc1Cl")[0] == "c(0)" and atom_labels("c1ccccc1Cl")[6] == "Cl(6)"


def _row(family, task_type, target=None, constraints=None, smiles=None, witness=None):
    return {
        "task_family": family,
        "task_type": task_type,
        "target_json": json.dumps(target) if target is not None else None,
        "constraints_json": json.dumps(constraints) if constraints is not None else None,
        "question_smiles": smiles,
        "witness_smiles": witness or smiles,
    }


@pytest.mark.parametrize(
    "row",
    [
        _row("count", "multi_count", {"halogen_atom_count": 1, "hydrogen_atom_count": 5},
             smiles="c1ccccc1Cl"),
        _row("count", "single_count", {"molecular_formula_count": "C2H6O"}, smiles="CCO"),
        _row("count", "single_count", {"ring_count": 1}, smiles="C1CCCCC1O"),
        _row("index", "multi_index", {"halogen_atom_index": [6], "heavy_atom_index": [0, 1, 2, 3, 4, 5, 6]},
             smiles="c1ccccc1Cl"),
        _row("index", "single_index", {"halogen_atom_index": []}, smiles="CCO"),
        _row("constraint_generation", "constraint_generation",
             constraints=[{"type": "carbon_atom_count", "operator": "=", "value": 6},
                          {"type": "aromatic_ring_count", "operator": "=", "value": 1}],
             witness="c1ccccc1Cl"),
    ],
    ids=["count-multi", "formula", "ring", "index-multi", "index-empty", "generation"],
)
def test_every_worked_solution_is_scored_correct_by_the_official_scorer(row, engine):
    text = completion_for(row, engine)
    scored = score_completion(text, row["task_type"], row["target_json"], row["constraints_json"])
    assert scored.correctness == 1.0, text
    assert text.count("<answer>") == 1 and text.rstrip().endswith("</answer>")


def test_a_count_solution_names_the_atoms_behind_the_count(engine):
    text = completion_for(_row("count", "single_count", {"halogen_atom_count": 1}, smiles="c1ccccc1Cl"), engine)
    assert text.splitlines()[0] == "Atoms: c(0), c(1), c(2), c(3), c(4), c(5), Cl(6)"
    assert "halogen_atom_count: Cl(6) -> 1" in text


def test_a_generation_solution_checks_each_constraint(engine):
    row = _row("constraint_generation", "constraint_generation",
               constraints=[{"type": "carbon_atom_count", "operator": "=", "value": 6}], witness="c1ccccc1Cl")
    text = completion_for(row, engine)
    assert "Candidate: c1ccccc1Cl" in text and "carbon_atom_count = 6" in text


def test_extension_configs_load_and_pin_the_sft_dataset():
    from pathlib import Path

    import yaml

    from miqgrpo.config import load_experiment_config
    from miqgrpo.paths import REPO_ROOT
    from miqgrpo.train_sft import load_sft_config

    root = REPO_ROOT / "extension" / "configs"
    pinned = yaml.safe_load((root / "sft_data.yaml").read_text())["expected_content_hash"]
    for path in sorted((root / "sft").glob("*.yaml")):
        config = load_sft_config(path)
        assert config.expected_hash == pinned and config.artifact_id == "miq-sft"
    for path in sorted((root / "grpo").glob("*.yaml")):
        config = load_experiment_config(path)
        if config.model.init_from:
            assert (root / "sft" / f"{config.model.init_from.removeprefix('sft-')}.yaml").exists()
        assert Path(path).stem == config.id
