"""Generated questions are answerable, verified, and not solvable from the prompt."""

from __future__ import annotations

import json
import random

import pytest

from miqgrpo.build_dataset import _leaks_target, validate_example
from miqgrpo.generation import (
    GenerationSpec,
    PropertyEngine,
    QuestionGenerator,
    ReferenceTable,
    build_property_catalog,
    complexity_bin_of,
    oracle_score,
    transform_smiles,
)

MOLECULES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "C[C@H](N)C(=O)O",
    "OCC1OC(O)C(O)C(O)C1O",
    "FC(F)(F)c1ccc(Br)cc1",
    "CCOC(=O)c1ccccc1N",
    "C1CC2CCC1CC2",
    "CCCCCCCCCC",
    "Clc1ccc(cc1)C(=O)NCCN1CCOCC1",
    "O=C(O)c1ccccc1O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Nc1ccc(cc1)S(=O)(=O)N",
]


@pytest.fixture(scope="module")
def engine():
    return PropertyEngine(seed=0)


@pytest.fixture(scope="module")
def catalog(engine):
    return build_property_catalog(engine)


@pytest.fixture(scope="module")
def table(catalog, engine):
    return ReferenceTable.build(MOLECULES, catalog, engine)


@pytest.fixture(scope="module")
def generator(catalog, table):
    return QuestionGenerator(seed=123, catalog=catalog, table=table)


def test_catalog_keeps_a_usable_property_space(catalog):
    summary = catalog.summary()
    assert summary["n_count_properties"] > 20
    assert summary["n_index_properties"] > 20
    assert summary["n_constraint_properties"] > 20


def test_catalog_records_why_properties_were_rejected(catalog):
    for key, entry in catalog.rejected.items():
        assert entry["reason"], key


def test_catalog_spans_many_categories(catalog):
    """A catalog collapsed onto one category would make a monotonous dataset."""
    assert len(catalog.count) >= 10
    assert len(catalog.constraint) >= 10


def test_transform_smiles_always_returns_a_readable_molecule():
    from rdkit import Chem

    rng = random.Random(0)
    for smiles in MOLECULES:
        for _ in range(10):
            out, _, _ = transform_smiles(smiles, rng)
            assert Chem.MolFromSmiles(out) is not None


def test_transform_smiles_preserves_the_molecule():
    from rdkit import Chem

    rng = random.Random(1)
    for smiles in MOLECULES:
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        for _ in range(5):
            out, _, _ = transform_smiles(smiles, rng)
            assert Chem.MolToSmiles(Chem.MolFromSmiles(out)) == canonical


def test_index_targets_follow_the_displayed_smiles(generator):
    """Randomised SMILES renumber atoms; targets must follow the shown string.

    The benchmark defines indices "reading the SMILES string left to right", so
    a target computed on the canonical form would be wrong for half the items.
    """
    spec = GenerationSpec(family="index", n_examples=20, randomize_prob=1.0)
    produced = 0
    for example in generator.generate(spec):
        recomputed = {
            prop: sorted(set(generator.engine.compute(example.question_smiles, prop)))
            for prop in example.properties
        }
        assert recomputed == example.target
        produced += 1
    assert produced == 20


@pytest.mark.parametrize("family", ["count", "index", "constraint_generation"])
def test_every_generated_example_passes_the_official_oracle(generator, family):
    spec = GenerationSpec(family=family, n_examples=25, max_constraint_prevalence=1.0)
    produced = 0
    for example in generator.generate(spec):
        assert oracle_score(example) == 1.0
        assert validate_example(example) is None
        produced += 1
    assert produced == 25


@pytest.mark.parametrize("family", ["count", "index"])
def test_multitask_load_controls_property_count(generator, family):
    for load in (1, 2, 3):
        spec = GenerationSpec(
            family=family, n_examples=5, multitask_weights={load: 1.0}
        )
        for example in generator.generate(spec):
            assert example.multitask_load == load
            assert len(example.properties) == load
            assert len(example.target) == load


def test_constraint_examples_are_satisfiable_by_their_witness(generator):
    spec = GenerationSpec(
        family="constraint_generation", n_examples=15, max_constraint_prevalence=1.0
    )
    for example in generator.generate(spec):
        assert example.witness_smiles
        assert oracle_score(example) == 1.0


def test_prevalence_filter_rejects_trivial_constraints(catalog, table):
    """"Generate a molecule with no boc group" is satisfied by anything."""
    generator = QuestionGenerator(seed=7, catalog=catalog, table=table)
    strict = GenerationSpec(
        family="constraint_generation",
        n_examples=10,
        max_constraint_prevalence=0.2,
        max_attempts_per_example=400,
    )
    for example in generator.generate(strict):
        assert example.constraint_prevalence is not None
        assert example.constraint_prevalence <= 0.2


def test_complexity_bins_are_labelled_consistently():
    assert complexity_bin_of(0.0) == "0-250"
    assert complexity_bin_of(249.9) == "0-250"
    assert complexity_bin_of(250.0) == "250-1000"
    assert complexity_bin_of(10_000.0) == "1000-inf"


def test_leak_detector_catches_a_planted_target():
    target = {"ring_count": 2}
    assert _leaks_target("How many rings? ring_count: 2", target) is not None
    assert _leaks_target('answer {"ring_count": 2}', target) is not None


def test_leak_detector_accepts_a_question_that_only_names_the_key():
    target = {"ring_count": 2}
    question = "How many rings are in CCO? Return JSON with key `ring_count`."
    assert _leaks_target(question, target) is None


def test_generated_questions_do_not_leak_targets(generator):
    for family in ("count", "index"):
        spec = GenerationSpec(family=family, n_examples=30)
        for example in generator.generate(spec):
            assert _leaks_target(example.question, example.target) is None


def test_question_shows_the_molecule_it_asks_about(generator):
    for family in ("count", "index"):
        spec = GenerationSpec(family=family, n_examples=15)
        for example in generator.generate(spec):
            assert example.question_smiles in example.question


def test_validate_example_rejects_a_target_that_stopped_matching_its_molecule(
    generator, engine
):
    """A corrupted target is self-consistent, so only recomputation catches it."""
    spec = GenerationSpec(family="count", n_examples=1)
    example = next(iter(generator.generate(spec)))
    key = next(iter(example.target))
    example.target[key] = example.target[key] + 1

    assert oracle_score(example) == 1.0
    assert validate_example(example) is None
    assert validate_example(example, engine) == "target_does_not_match_molecule"


def test_generator_fails_loudly_when_it_cannot_meet_the_quota(catalog, table):
    """Silently returning a short dataset would skew the task mix unnoticed."""
    generator = QuestionGenerator(seed=99, catalog=catalog, table=table)
    spec = GenerationSpec(
        family="constraint_generation",
        n_examples=5,
        max_constraint_prevalence=0.0,
        min_constraint_prevalence=0.0,
        max_attempts_per_example=20,
    )
    with pytest.raises(RuntimeError, match="only generated"):
        list(generator.generate(spec))
    assert generator.drops["constraint_too_common"] > 0


def test_generation_is_reproducible_from_the_seed(catalog, table):
    def run(seed: int) -> list[str]:
        generator = QuestionGenerator(seed=seed, catalog=catalog, table=table)
        spec = GenerationSpec(family="count", n_examples=10)
        return [example.question for example in generator.generate(spec)]

    assert run(2024) == run(2024)
    assert run(2024) != run(2025)


def test_targets_are_json_serialisable(generator):
    for family in ("count", "index"):
        spec = GenerationSpec(family=family, n_examples=10)
        for example in generator.generate(spec):
            json.dumps(example.target)


def test_parallel_reference_table_matches_the_serial_one(catalog):
    """Workers must produce the same table, in the same order.

    Row order feeds seeded sampling, so a reshuffle here would silently make
    every dataset build unreproducible.
    """
    molecules = MOLECULES * 3
    serial = ReferenceTable.build(molecules, catalog, PropertyEngine())
    parallel = ReferenceTable.build(molecules, catalog, PropertyEngine(), n_workers=3)

    assert parallel.smiles == serial.smiles
    assert parallel.complexity_bin == serial.complexity_bin
    assert parallel.values == serial.values
    assert [round(c, 6) for c in parallel.complexity] == [
        round(c, 6) for c in serial.complexity
    ]


def _is_all_zero(example) -> bool:
    return all(v == 0 or v == [] for v in example.target.values())


def test_trivial_questions_can_be_capped(catalog, table):
    generator = QuestionGenerator(seed=5, catalog=catalog, table=table)
    spec = GenerationSpec(family="count", n_examples=40, zero_weight=1.0, max_trivial_fraction=0.05)
    examples = list(generator.generate(spec))
    assert sum(_is_all_zero(e) for e in examples) <= int(0.05 * 40)


def test_property_weight_zero_removes_a_property(catalog, table):
    removed = "ring_count"
    generator = QuestionGenerator(seed=6, catalog=catalog, table=table)
    spec = GenerationSpec(family="count", n_examples=40, property_weights={removed: 0.0})
    examples = list(generator.generate(spec))
    assert all(removed not in e.target for e in examples)


def test_constraints_met_by_a_default_molecule_are_rejected(catalog, table, engine):
    defaults = ["CC(O)C", "CCO"]
    generator = QuestionGenerator(seed=7, catalog=catalog, table=table)
    spec = GenerationSpec(
        family="constraint_generation",
        n_examples=10,
        max_constraint_prevalence=1.0,
        default_molecules=defaults,
    )
    examples = list(generator.generate(spec))
    assert examples
    for example in examples:
        for constraint in example.constraints:
            for smiles in defaults:
                assert engine.compute(smiles, constraint["type"]) != constraint["value"]


def test_default_knobs_leave_generation_unchanged(catalog, table):
    """Every knob is inert at its default, so older dataset configs still rebuild bit for bit."""
    def run(spec: GenerationSpec) -> list[str]:
        generator = QuestionGenerator(seed=77, catalog=catalog, table=table)
        return [e.question for e in generator.generate(spec)]

    plain = GenerationSpec(family="index", n_examples=12)
    explicit = GenerationSpec(
        family="index", n_examples=12, zero_weight=0.5, max_trivial_fraction=None,
        property_weights={}, default_molecules=[],
    )
    assert run(plain) == run(explicit)


def test_cached_sampling_draws_exactly_like_uncached(table):
    """Naming the row subset only caches its weights; the draws must not change."""
    prop = next(iter(table.values))
    rows = list(range(len(table.smiles)))
    plain, cached = random.Random(3), random.Random(3)
    for _ in range(200):
        a = table.sample_by_value_frequency(prop, plain, zero_weight=0.3, rows=rows)
        b = table.sample_by_value_frequency(prop, cached, zero_weight=0.3, rows=rows, rows_key="all")
        assert a == b


def test_unseedable_properties_are_skipped_only_when_asked(catalog, table):
    defaults = ["CC(O)C", "CCO"]
    spec = GenerationSpec(family="constraint_generation", n_examples=5, max_constraint_prevalence=1.0,
                          default_molecules=defaults, skip_unseedable_properties=True)
    generator = QuestionGenerator(seed=8, catalog=catalog, table=table)
    examples = list(generator.generate(spec))
    assert examples and generator.drops["unseedable_properties_skipped"] > 0
    assert all(c["type"] not in generator._unseedable for e in examples for c in e.constraints)
    plain = QuestionGenerator(seed=8, catalog=catalog, table=table)
    plain.spec = GenerationSpec(family="constraint_generation", n_examples=1, default_molecules=defaults)
    assert plain._unseedable == set()
