"""Offline MolecularIQ question generation.

This module owns *all* question generation. It runs once, before any GPU is
allocated, and its output is frozen to disk. The GRPO training loop never
imports it.

Everything chemistry-facing is delegated to moleculariq_core:

    question text      NaturalLanguageFormatter + TASKS templates
    ground truth       SymbolicSolver (via MolecularIQD.compute_property)
    verification       evaluate_answer  (the official reward dispatcher)

Two conventions are worth stating explicitly.

**Targets are computed on the SMILES the question displays**, not on the
canonical form of the source molecule. Atom indices depend on SMILES atom
ordering, and the benchmark defines them "reading the SMILES string left to
right", so a randomised representation must be scored against its own ordering
(verified against official items -- see docs/official-semantics.md).

**Questions must be non-trivial.** A dataset of "how many sulfonic acid groups"
questions whose answer is always 0, or "generate a molecule with no boc group"
constraints that every molecule satisfies, teaches a policy to emit a constant
and gives GRPO no within-group reward variance to learn from. Sampling therefore
mirrors the official generator: inverse-frequency weighting over property values
with zero-valued molecules down-weighted, plus a prevalence ceiling on
generation constraints.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import GraphDescriptors

from moleculariq_core import (
    CONSTRAINT_MAP,
    COUNT_MAP,
    INDEX_MAP,
    TASKS,
    MolecularIQD,
    NaturalLanguageFormatter,
    evaluate_answer,
)

RDLogger.DisableLog("rdApp.*")

__all__ = [
    "TASK_FAMILIES",
    "COMPLEXITY_BINS",
    "GeneratedExample",
    "GenerationSpec",
    "PropertyCatalog",
    "QuestionGenerator",
    "ReferenceTable",
    "build_property_catalog",
    "complexity_bin_of",
    "complexity_of",
    "oracle_score",
    "transform_smiles",
]

TASK_FAMILIES = ("count", "index", "constraint_generation")

# Bertz-complexity bins, matching the axis the benchmark reports on.
COMPLEXITY_BINS: tuple[tuple[str, float, float], ...] = (
    ("0-250", 0.0, 250.0),
    ("250-1000", 250.0, 1000.0),
    ("1000-inf", 1000.0, float("inf")),
)

# Categories in the core maps that are not plain per-molecule features.
_SKIP_CATEGORIES = frozenset({"functional_groups"})

_FG_PREFIX = "functional_group_"


def complexity_of(smiles: str) -> float | None:
    """Bertz CT complexity, or None when RDKit cannot parse the molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return float(GraphDescriptors.BertzCT(mol))
    except Exception:
        return None


def complexity_bin_of(complexity: float) -> str:
    """Name of the complexity bin that contains a molecule's complexity value."""
    for name, low, high in COMPLEXITY_BINS:
        if low <= complexity < high:
            return name
    return COMPLEXITY_BINS[-1][0]


def canonical_smiles(smiles: str) -> str | None:
    """RDKit canonical SMILES, or None if the string does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def transform_smiles(
    smiles: str,
    rng: random.Random,
    randomize_prob: float = 0.5,
    kekulize_prob: float = 0.5,
) -> tuple[str, bool, bool]:
    """Re-render a molecule in one of the benchmark's representation variants.

    Mirrors moleculariq-benchmark's transform_smiles: independently
    decide whether to randomise atom order and whether to kekulise. Returns
    (smiles, was_randomized, was_kekulized); on any RDKit failure the input
    is returned unchanged with both flags false.

    Randomisation renumbers atoms with the caller's rng rather than using
    RDKit's doRandom=True, which draws from RDKit's *global* generator: with
    doRandom the same preprocessing seed produces a different dataset on
    every run, which would quietly break artifact reproducibility.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, False, False

    randomized = rng.random() < randomize_prob
    kekulized = rng.random() < kekulize_prob
    try:
        working = mol
        if randomized:
            order = list(range(working.GetNumAtoms()))
            rng.shuffle(order)
            working = Chem.RenumberAtoms(working, order)
        if kekulized:
            working = Chem.Mol(working)
            Chem.Kekulize(working, clearAromaticFlags=True)
        out = Chem.MolToSmiles(
            working, canonical=not randomized, kekuleSmiles=kekulized
        )
    except Exception:
        return smiles, False, False

    # A re-rendered string RDKit can no longer read is useless as a prompt.
    if not out or Chem.MolFromSmiles(out) is None:
        return smiles, False, False
    return out, randomized, kekulized


def _json_safe(value: Any) -> Any:
    """Convert solver output into something json.dumps accepts."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float, str)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    for caster in (int, float, str):  # numpy scalars and friends
        try:
            return caster(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return str(value)


def _is_zeroish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, (list, tuple, set, dict, str)):
        return len(value) == 0
    return False


class PropertyEngine:
    """Thin wrapper over MolecularIQD with a functional-group fast path.

    MolecularIQD.compute_property recomputes the whole 400-key functional
    group table for *every* functional_group_* property. Since most of the
    property space is functional groups, that turns a 0.4 ms molecule into a
    50 ms one. Computing the table once per molecule keeps generation cheap
    while still going through the official solver.
    """

    def __init__(self, seed: int = 0) -> None:
        self.mqd = MolecularIQD(seed=seed, cache_properties=False)
        self._fg_smiles: str | None = None
        self._fg_data: dict[str, Any] = {}

    def _fg(self, smiles: str) -> dict[str, Any]:
        if smiles != self._fg_smiles:
            try:
                self._fg_data = self.mqd.solver.functional_group_solver.get_counts_and_indices(
                    smiles
                )
            except Exception:
                self._fg_data = {}
            self._fg_smiles = smiles
        return self._fg_data

    def compute(self, smiles: str, prop: str) -> Any:
        """Value of one official property for one molecule."""
        if prop.startswith(_FG_PREFIX):
            data = self._fg(smiles)
            if prop in data:
                return data[prop]
            return 0 if ("_count" in prop or "_nbrInstances" in prop) else []
        return self.mqd.compute_property(smiles, prop)


@dataclass
class PropertyCatalog:
    """Which properties are usable for which task family, decided empirically.

    Hand-maintaining a list of "properties that work" rots the moment
    moleculariq-core changes -- and it already has gaps (several reaction
    templates reference a solver method that does not exist). Instead each
    candidate property is probed on a handful of molecules and kept only if the
    *official* verifier scores its own ground truth as correct. A property that
    cannot verify its own answer would hand GRPO an unlearnable reward.
    """

    count: dict[str, list[str]] = field(default_factory=dict)
    index: dict[str, list[str]] = field(default_factory=dict)
    constraint: dict[str, list[str]] = field(default_factory=dict)
    rejected: dict[str, dict[str, str]] = field(default_factory=dict)

    def properties(self, family: str) -> dict[str, list[str]]:
        """{category: [property, ...]} usable for one task family."""
        if family == "count":
            return self.count
        if family == "index":
            return self.index
        if family == "constraint_generation":
            return self.constraint
        raise ValueError(f"unknown task family: {family}")

    def flat(self, family: str) -> list[tuple[str, str]]:
        """[(category, property), ...] for one family."""
        return [
            (category, prop)
            for category, props in sorted(self.properties(family).items())
            for prop in props
        ]

    def all_properties(self) -> list[str]:
        """Every property usable in any family, in first-seen order."""
        seen: list[str] = []
        for family in TASK_FAMILIES:
            for _, prop in self.flat(family):
                if prop not in seen:
                    seen.append(prop)
        return seen

    def summary(self) -> dict[str, Any]:
        """Counts and lists of usable and rejected properties (stored in the manifest)."""
        return {
            "n_count_properties": sum(len(v) for v in self.count.values()),
            "n_index_properties": sum(len(v) for v in self.index.values()),
            "n_constraint_properties": sum(len(v) for v in self.constraint.values()),
            "count": {k: sorted(v) for k, v in sorted(self.count.items())},
            "index": {k: sorted(v) for k, v in sorted(self.index.items())},
            "constraint": {k: sorted(v) for k, v in sorted(self.constraint.items())},
            "n_rejected": len(self.rejected),
            "rejected": self.rejected,
        }


# Small, structurally diverse molecules used to probe property support.
_PROBE_SMILES: tuple[str, ...] = (
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin: ester, acid, aromatic ring
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine: fused heterocycles
    "C[C@H](N)C(=O)O",  # alanine: R/S stereocentre
    "C/C=C/C(=O)OCC",  # E-alkene ester: double-bond stereochemistry
    "OCC1OC(O)C(O)C(O)C1O",  # sugar-like: many OH, unspecified stereo
    "FC(F)(F)c1ccc(Br)cc1",  # halogens
    "C1CC2CCC1CC2",  # bridged bicycle: bridgeheads
    "CCCCCCCCCC",  # plain chain
)


def _candidate_properties(
    engine: PropertyEngine, probe_smiles: str, include_functional_groups: bool
) -> dict[str, dict[str, list[str]]]:
    """Every property we *might* use, grouped by family and category."""
    candidates: dict[str, dict[str, list[str]]] = {
        "count": {},
        "index": {},
        "constraint_generation": {},
    }
    sources = (
        ("count", COUNT_MAP),
        ("index", INDEX_MAP),
        ("constraint_generation", CONSTRAINT_MAP),
    )
    for family, mapping in sources:
        for category, props in mapping.items():
            if category in _SKIP_CATEGORIES or not props:
                continue
            candidates[family][category] = list(props)

    if include_functional_groups:
        fg_keys = sorted(engine._fg(probe_smiles))
        buckets = {
            "count": [k for k in fg_keys if k.endswith("_count")],
            "index": [k for k in fg_keys if k.endswith("_index")],
            "constraint_generation": [k for k in fg_keys if k.endswith("_nbrInstances")],
        }
        for family, keys in buckets.items():
            if keys:
                candidates[family]["functional_group"] = keys

    return candidates


def build_property_catalog(
    engine: PropertyEngine | None = None,
    probe_smiles: Sequence[str] = _PROBE_SMILES,
    include_functional_groups: bool = True,
    min_successes: int = 3,
) -> PropertyCatalog:
    """Probe every candidate property against the official verifier.

    A property is kept when, on at least min_successes probe molecules, the
    verifier scores the property's own computed ground truth as 1.0 -- and
    never scores it wrong.
    """
    engine = engine or PropertyEngine()
    catalog = PropertyCatalog()
    candidates = _candidate_properties(engine, probe_smiles[0], include_functional_groups)

    for family, by_category in candidates.items():
        keep: dict[str, list[str]] = {}
        for category, props in by_category.items():
            for prop in props:
                ok, reason = _probe_property(
                    engine, family, prop, probe_smiles, min_successes
                )
                if ok:
                    keep.setdefault(category, []).append(prop)
                else:
                    catalog.rejected[f"{family}:{prop}"] = {
                        "category": category,
                        "reason": reason,
                    }
        setattr(
            catalog,
            {"count": "count", "index": "index", "constraint_generation": "constraint"}[
                family
            ],
            keep,
        )
    return catalog


def _probe_property(
    engine: PropertyEngine,
    family: str,
    prop: str,
    probe_smiles: Sequence[str],
    min_successes: int,
) -> tuple[bool, str]:
    successes = 0
    for smiles in probe_smiles:
        try:
            value = engine.compute(smiles, prop)
        except Exception as exc:  # noqa: BLE001 - probing is best-effort by design
            return False, f"compute failed: {type(exc).__name__}: {exc}"[:200]

        if value is None:
            continue

        try:
            if family == "count":
                if isinstance(value, (list, tuple, set, dict)):
                    return False, "count property returned a container, not a scalar"
                target = {prop: _json_safe(value)}
                score = evaluate_answer(
                    task_type="single_count",
                    predicted=json.dumps(target),
                    target=target,
                )
            elif family == "index":
                if not isinstance(value, (list, tuple)):
                    return False, "index property did not return a list"
                target = {prop: _json_safe(list(value))}
                score = evaluate_answer(
                    task_type="single_index",
                    predicted=json.dumps(target),
                    target=target,
                )
            else:
                if isinstance(value, (list, tuple, set, dict)):
                    return False, "constraint property returned a container"
                constraints = [{"type": prop, "operator": "=", "value": _json_safe(value)}]
                score = evaluate_answer(
                    task_type="constraint_generation",
                    predicted=json.dumps({"smiles": smiles}),
                    constraints=constraints,
                )
        except Exception as exc:  # noqa: BLE001
            return False, f"verifier failed: {type(exc).__name__}: {exc}"[:200]

        if float(score) != 1.0:
            return False, "verifier rejected its own ground truth"
        successes += 1

    if successes < min_successes:
        return False, f"only {successes} probe molecules produced a value"
    return True, ""


def _row_for(
    raw: str,
    props: Sequence[str],
    index_props: set[str],
    engine: PropertyEngine,
) -> tuple[str, float, dict[str, Any]] | None:
    """One reference-table row, or None when the molecule is unusable."""
    canonical = canonical_smiles(raw)
    if canonical is None:
        return None
    bertz = complexity_of(canonical)
    if bertz is None:
        return None

    row: dict[str, Any] = {}
    for prop in props:
        try:
            value = engine.compute(canonical, prop)
        except Exception:
            return None
        if prop in index_props:
            # Only the list length matters for sampling; indices are recomputed on the displayed SMILES.
            row[prop] = len(value) if isinstance(value, (list, tuple)) else 0
        else:
            row[prop] = _json_safe(value)
    return canonical, bertz, row


def _serial_rows(
    molecules: Sequence[str],
    props: Sequence[str],
    index_props: set[str],
    engine: PropertyEngine,
    progress: bool,
) -> list[tuple[str, float, dict[str, Any]]]:
    rows = []
    for position, raw in enumerate(molecules):
        if progress and position and position % 5000 == 0:
            print(f"    reference table: {position}/{len(molecules)}", flush=True)
        row = _row_for(raw, props, index_props, engine)
        if row is not None:
            rows.append(row)
    return rows


def _worker_chunk(
    args: tuple[Sequence[str], Sequence[str], set[str]]
) -> list[tuple[str, float, dict[str, Any]]]:
    """Process-pool entry point: each worker builds its own solver."""
    chunk, props, index_props = args
    engine = PropertyEngine()
    rows = []
    for raw in chunk:
        row = _row_for(raw, props, index_props, engine)
        if row is not None:
            rows.append(row)
    return rows


def _parallel_rows(
    molecules: Sequence[str],
    props: Sequence[str],
    index_props: set[str],
    n_workers: int,
    progress: bool,
) -> list[tuple[str, float, dict[str, Any]]]:
    import multiprocessing as mp

    # In-order chunks keep the table's row order, which seeded sampling depends on.
    chunk_size = max(1, len(molecules) // (n_workers * 4) or 1)
    chunks = [
        (list(molecules[i : i + chunk_size]), list(props), set(index_props))
        for i in range(0, len(molecules), chunk_size)
    ]
    rows: list[tuple[str, float, dict[str, Any]]] = []
    with mp.get_context("spawn").Pool(n_workers) as pool:
        for done, chunk_rows in enumerate(pool.imap(_worker_chunk, chunks), start=1):
            rows.extend(chunk_rows)
            if progress:
                print(
                    f"    reference table: chunk {done}/{len(chunks)} "
                    f"({len(rows)} molecules)",
                    flush=True,
                )
    return rows


@dataclass
class ReferenceTable:
    """Property statistics over a sample of the training pool.

    Serves two purposes:

    * pick molecules that actually *exhibit* the property being asked about,
      via inverse-frequency weighting over values (the official strategy);
    * estimate how many real molecules satisfy a candidate generation
      constraint, so trivially-satisfiable constraints can be rejected.

    Index properties are stored as the *length* of the index list. The actual
    indices are recomputed later on the displayed SMILES; only the magnitude
    matters for sampling.
    """

    smiles: list[str]
    complexity: list[float]
    complexity_bin: list[str]
    values: dict[str, list[Any]]
    value_counts: dict[str, Counter]
    rows_by_bin: dict[str, list[int]] = field(default_factory=dict)
    # (property, value) -> matching row indices, filled lazily by _rows_matching
    _row_index: dict[tuple[str, Any], frozenset[int]] = field(default_factory=dict)
    # sampling key -> (candidate rows, cumulative weights), filled lazily
    _draw_cache: dict[tuple[Any, ...], tuple[list[int], list[float]]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        molecules: Sequence[str],
        catalog: PropertyCatalog,
        engine: PropertyEngine | None = None,
        progress: bool = False,
        n_workers: int = 0,
    ) -> "ReferenceTable":
        """Compute every catalog property for every reference molecule.

        Roughly 50 ms per molecule, so 20k molecules is ~15 minutes serially --
        the slowest part of preprocessing by a wide margin. n_workers > 1
        splits it across processes; the result is identical either way, which
        tests/test_generation.py asserts.
        """
        engine = engine or PropertyEngine()
        props = catalog.all_properties()
        index_props = {prop for _, prop in catalog.flat("index")}

        if n_workers and n_workers > 1:
            rows = _parallel_rows(molecules, props, index_props, n_workers, progress)
        else:
            rows = _serial_rows(molecules, props, index_props, engine, progress)

        smiles: list[str] = []
        complexity: list[float] = []
        bins: list[str] = []
        values: dict[str, list[Any]] = {prop: [] for prop in props}
        for canonical, bertz, row in rows:
            smiles.append(canonical)
            complexity.append(bertz)
            bins.append(complexity_bin_of(bertz))
            for prop in props:
                values[prop].append(row[prop])

        counts = {
            prop: Counter(v for v in column if isinstance(v, (int, float, str, bool)))
            for prop, column in values.items()
        }
        rows_by_bin: dict[str, list[int]] = {}
        for row, name in enumerate(bins):
            rows_by_bin.setdefault(name, []).append(row)
        return cls(
            smiles=smiles,
            complexity=complexity,
            complexity_bin=bins,
            values=values,
            value_counts=counts,
            rows_by_bin=rows_by_bin,
        )

    def __len__(self) -> int:
        return len(self.smiles)

    def prevalence(self, prop: str, value: Any) -> float:
        """Fraction of reference molecules whose prop equals value."""
        if not self.smiles:
            return 0.0
        return self.value_counts.get(prop, Counter()).get(value, 0) / len(self.smiles)

    def _rows_matching(self, prop: str, value: Any) -> frozenset[int]:
        """Row indices whose prop equals value, memoised.

        Scanning the whole table per candidate constraint made generation the
        slowest part of preprocessing (tens of thousands of candidates x tens of
        thousands of molecules). Building each (property, value) row set once
        turns the joint check into a set intersection.
        """
        key = (prop, value if isinstance(value, (int, float, str, bool)) else str(value))
        cached = self._row_index.get(key)
        if cached is None:
            column = self.values.get(prop)
            cached = (
                frozenset(i for i, v in enumerate(column) if v == value)
                if column
                else frozenset()
            )
            self._row_index[key] = cached
        return cached

    def joint_prevalence(self, constraints: Sequence[dict[str, Any]]) -> float:
        """Exact fraction of reference molecules satisfying *all* constraints.

        Measured directly rather than multiplied out: molecular properties are
        strongly correlated, so the product of marginals would badly understate
        how easy a constraint set actually is.
        """
        if not self.smiles:
            return 0.0
        row_sets = []
        for constraint in constraints:
            prop = constraint["type"]
            if prop not in self.values:
                return 1.0  # unknown -> treat as trivially satisfiable
            row_sets.append(self._rows_matching(prop, constraint["value"]))

        row_sets.sort(key=len)  # intersect the rarest first
        hits: frozenset[int] | set[int] = row_sets[0]
        for row_set in row_sets[1:]:
            if not hits:
                break
            hits = hits & row_set
        return len(hits) / len(self.smiles)

    def sample_by_value_frequency(
        self,
        prop: str,
        rng: random.Random,
        zero_weight: float = 0.5,
        rows: Sequence[int] | None = None,
        power: float = 1.0,
        rows_key: Any = None,
    ) -> int | None:
        """Sample a molecule index, favouring rare values of prop.

        Mirrors the official sample_with_inverse_frequency: weight each
        molecule by 1/frequency(its value) and halve the weight of zero-valued
        molecules, so questions are mostly about features the molecule has.
        power < 1 softens the preference for rare values (0 samples
        molecules uniformly, so values are as common as they are in the pool).

        rows restricts the draw to a subset, which the generator uses to
        hold a target complexity mix. rows_key names that subset: the
        weights of a named subset are computed once and reused, which keeps
        large reference tables fast. Draws are identical either way --
        random.choices turns weights into exactly these cumulative
        weights itself.
        """
        column = self.values.get(prop)
        if not column:
            return None
        counts = self.value_counts.get(prop)
        if not counts:
            return None
        key = None if rows_key is None else (prop, rows_key, zero_weight, power)
        cached = self._draw_cache.get(key) if key is not None else None
        if cached is None:
            candidates = list(rows) if rows is not None else range(len(column))
            weights = []
            kept: list[int] = []
            for row in candidates:
                value = column[row]
                frequency = counts.get(value, 0)
                if frequency <= 0:
                    continue
                weight = 1.0 / frequency if power == 1.0 else 1.0 / frequency**power
                if _is_zeroish(value):
                    weight *= zero_weight
                kept.append(row)
                weights.append(weight)
            cached = (kept, list(itertools.accumulate(weights)))
            if key is not None:
                self._draw_cache[key] = cached
        kept, cumulative = cached
        if not kept or cumulative[-1] <= 0:
            return None
        return rng.choices(kept, cum_weights=cumulative, k=1)[0]


@dataclass
class GeneratedExample:
    """One offline-generated training item, before dataset serialisation."""

    task_family: str
    task_type: str
    question: str
    feature: str
    supercategory: str
    properties: list[str]
    multitask_load: int
    question_smiles: str | None
    molecule_id: str
    witness_smiles: str
    molecular_complexity: float
    complexity_bin: str
    is_randomized: bool
    is_kekulized: bool
    question_seed: int
    target: dict[str, Any] | None = None
    constraints: list[dict[str, Any]] | None = None
    constraint_prevalence: float | None = None


@dataclass
class GenerationSpec:
    """Everything that shapes one task family's slice of the dataset."""

    family: str
    n_examples: int
    # relative weight of each multitask load (1 = single task, 2/3/5 = multi)
    multitask_weights: dict[int, float] = field(
        default_factory=lambda: {1: 0.55, 2: 0.2, 3: 0.15, 5: 0.1}
    )
    # relative weight of each Bertz-complexity bin
    complexity_weights: dict[str, float] = field(
        default_factory=lambda: {"0-250": 0.5, "250-1000": 0.35, "1000-inf": 0.15}
    )
    randomize_prob: float = 0.5
    kekulize_prob: float = 0.5
    # reject constraint sets met by more than this fraction of reference molecules
    max_constraint_prevalence: float = 0.03
    # ...or by fewer than this fraction
    min_constraint_prevalence: float = 0.0
    max_attempts_per_example: int = 40
    # weight of an absent feature (0 / []) when choosing what to ask
    zero_weight: float = 0.5
    # cap on questions answered entirely by 0 / [] (fraction); None = no cap
    max_trivial_fraction: float | None = None
    # per-property multiplier on how often it is asked (0 removes it)
    property_weights: dict[str, float] = field(default_factory=dict)
    # reject a generation question if one of these molecules meets any constraint
    default_molecules: list[str] = field(default_factory=list)
    # exponent of the inverse-frequency molecule weighting (1 = official, 0 = uniform)
    value_frequency_power: float = 1.0
    # with default_molecules: skip properties no reference molecule can seed
    skip_unseedable_properties: bool = False


class QuestionGenerator:
    """Builds MolecularIQ questions for molecules from the training pool.

    Question wording comes from TASKS[...]["question_templates"] and
    NaturalLanguageFormatter, i.e. the same machinery the official benchmark
    generator uses, so training prompts sit in the same distribution as test
    prompts without ever reading a test item.
    """

    def __init__(
        self,
        seed: int,
        catalog: PropertyCatalog,
        table: ReferenceTable,
        engine: PropertyEngine | None = None,
        enable_random_phrasing: bool = True,
    ) -> None:
        self.seed = seed
        self.catalog = catalog
        self.table = table
        self.engine = engine or PropertyEngine(seed=seed)
        self.rng = random.Random(seed)
        self.formatter = NaturalLanguageFormatter(
            rng=self.rng, enable_random_phrasing=enable_random_phrasing
        )
        self._templates = {
            ("count", True): TASKS["single_count"]["question_templates"],
            ("count", False): TASKS["multi_count"]["question_templates"],
            ("index", True): TASKS["single_index_identification"]["question_templates"],
            ("index", False): TASKS["multi_index_identification"]["question_templates"],
        }
        self._constraint_templates = TASKS["constraint_generation"]["question_templates"]
        self.property_usage: Counter = Counter()
        self.category_usage: Counter = Counter()
        self.drops: Counter = Counter()
        self.spec: GenerationSpec | None = None
        self._unseedable: set[str] = set()
        self._default_values: dict[str, set[Any]] = {}


    def _pick_category(self, family: str) -> str | None:
        """Category-first sampling, balanced by usage.

        Functional groups make up ~75% of the property space. Sampling a
        property uniformly would make three quarters of the dataset functional
        group questions; the official generator gives each *category* an equal
        budget instead, and so do we.
        """
        categories = sorted(
            category
            for category, props in self.catalog.properties(family).items()
            if any(self._property_weight(p) > 0 for p in props)
        )
        if not categories:
            return None
        weights = [1.0 / (1 + self.category_usage[f"{family}:{c}"]) for c in categories]
        return self.rng.choices(categories, weights=weights, k=1)[0]

    def _property_weight(self, prop: str) -> float:
        if self.spec is None:
            return 1.0
        if prop in self._unseedable:
            return 0.0
        return float(self.spec.property_weights.get(prop, 1.0))

    def _find_unseedable(self, family: str) -> set[str]:
        """Properties whose every value in the table is met by a default molecule."""
        dead = set()
        for _, prop in self.catalog.flat(family):
            values = {v for v in (self.table.values.get(prop) or []) if v is not None}
            if not any(
                not self._met_by_default({"type": prop, "value": value})
                for value in values
                if self.table.value_counts.get(prop, {}).get(value, 0) > 0
            ):
                dead.add(prop)
        return dead

    def _zero_weight(self) -> float:
        return self.spec.zero_weight if self.spec is not None else 0.5

    def _pick_property(
        self,
        family: str,
        category: str,
        exclude: Sequence[str] = (),
        molecule_values: dict[str, Any] | None = None,
    ) -> str | None:
        props = [
            p
            for p in self.catalog.properties(family)[category]
            if p not in exclude and self._property_weight(p) > 0
        ]
        if not props:
            return None
        weights = []
        for prop in props:
            weight = self._property_weight(prop) / (1 + self.property_usage[f"{family}:{prop}"])
            if molecule_values is not None and _is_zeroish(molecule_values.get(prop)):
                weight *= self._zero_weight()
            weights.append(weight)
        return self.rng.choices(props, weights=weights, k=1)[0]

    def _note_usage(self, family: str, properties: Sequence[str]) -> None:
        for prop in properties:
            self.property_usage[f"{family}:{prop}"] += 1
            self.category_usage[f"{family}:{_category_of(prop)}"] += 1

    def _rows_for_target_complexity(self, spec: GenerationSpec) -> list[int] | None:
        """Draw a complexity bin, then hand back the molecules in it."""
        return self._bin_for_target_complexity(spec)[1]

    def _bin_for_target_complexity(
        self, spec: GenerationSpec
    ) -> tuple[str | None, list[int] | None]:
        """Draw a complexity bin; return its name and its molecules."""
        bins = [b for b in spec.complexity_weights if self.table.rows_by_bin.get(b)]
        if not bins:
            return None, None
        weights = [spec.complexity_weights[b] for b in bins]
        chosen = self.rng.choices(bins, weights=weights, k=1)[0]
        return chosen, self.table.rows_by_bin[chosen]

    def _molecule_values(self, family: str, row: int) -> dict[str, Any]:
        return {
            prop: self.table.values[prop][row]
            for _, prop in self.catalog.flat(family)
            if prop in self.table.values
        }


    def generate(self, spec: GenerationSpec) -> Iterator[GeneratedExample]:
        """Yield spec.n_examples validated examples for one task family."""
        loads = sorted(spec.multitask_weights)
        weights = [spec.multitask_weights[load] for load in loads]

        self.spec = spec
        if spec.skip_unseedable_properties and spec.default_molecules:
            self._unseedable = self._find_unseedable(spec.family)
            self.drops["unseedable_properties_skipped"] = len(self._unseedable)
        trivial_budget = (
            None
            if spec.max_trivial_fraction is None
            else int(spec.max_trivial_fraction * spec.n_examples)
        )
        trivial = 0
        produced = 0
        attempts = 0
        budget = spec.n_examples * spec.max_attempts_per_example
        while produced < spec.n_examples and attempts < budget:
            attempts += 1
            load = self.rng.choices(loads, weights=weights, k=1)[0]
            if spec.family == "constraint_generation":
                example = self._constraint_example(load, spec)
            else:
                example = self._count_or_index_example(spec.family, load, spec)
            if example is None:
                continue
            is_trivial = example.target is not None and all(
                _is_zeroish(value) for value in example.target.values()
            )
            if is_trivial and trivial_budget is not None and trivial >= trivial_budget:
                self.drops["trivial_budget_exhausted"] += 1
                continue
            if oracle_score(example) != 1.0:
                self.drops["oracle_rejected"] += 1
                continue
            trivial += is_trivial
            produced += 1
            yield example

        if produced < spec.n_examples:
            raise RuntimeError(
                f"only generated {produced}/{spec.n_examples} examples for "
                f"'{spec.family}' within {budget} attempts; drop reasons: "
                f"{dict(self.drops)}"
            )

    def _met_by_default(self, constraint: dict[str, Any]) -> bool:
        """Does any of the spec's default molecules satisfy this constraint?"""
        value = constraint["value"]
        if value is None:
            return False
        return _value_key(value) in self._default_value_set(constraint["type"])

    def _default_value_set(self, prop: str) -> set[Any]:
        """The values the default molecules take for prop (cached)."""
        cached = self._default_values.get(prop)
        if cached is None:
            cached = set()
            for smiles in self.spec.default_molecules if self.spec else ():
                try:
                    value = _json_safe(self.engine.compute(smiles, prop))
                except Exception:
                    continue
                if value is not None:
                    cached.add(_value_key(value))
            self._default_values[prop] = cached
        return cached

    def _add_unblocked_properties(
        self,
        family: str,
        load: int,
        properties: list[str],
        molecule_values: dict[str, Any],
    ) -> None:
        """Add constraints whose value on this molecule no default molecule has.

        Filtering while choosing, rather than rejecting finished questions,
        keeps multi-constraint questions possible: with five constraints, one
        of them almost always happens to match some small default molecule.
        """
        tries = 0
        while len(properties) < load and tries < 12 * (load - 1):
            tries += 1
            category = self._pick_category(family)
            if category is None:
                return
            blocked = [
                prop
                for prop in self.catalog.properties(family)[category]
                if molecule_values.get(prop) is not None
                and self._met_by_default(
                    {"type": prop, "value": _json_safe(molecule_values[prop])}
                )
            ]
            extra = self._pick_property(
                family,
                category,
                exclude=[*properties, *blocked],
                molecule_values=molecule_values,
            )
            if extra is not None:
                properties.append(extra)


    def _count_or_index_example(
        self, family: str, load: int, spec: GenerationSpec
    ) -> GeneratedExample | None:
        category = self._pick_category(family)
        if category is None:
            self.drops["no_category"] += 1
            return None
        seed_property = self._pick_property(family, category)
        if seed_property is None:
            self.drops["no_property"] += 1
            return None

        chosen_bin, rows = self._bin_for_target_complexity(spec)
        row = self.table.sample_by_value_frequency(
            seed_property,
            self.rng,
            zero_weight=spec.zero_weight,
            rows=rows,
            power=spec.value_frequency_power,
            rows_key=chosen_bin,
        )
        if row is None:
            self.drops["no_molecule"] += 1
            return None
        source_smiles = self.table.smiles[row]

        properties = [seed_property]
        if load > 1:
            molecule_values = self._molecule_values(family, row)
            for _ in range(load - 1):
                extra_category = self._pick_category(family)
                if extra_category is None:
                    break
                extra = self._pick_property(
                    family,
                    extra_category,
                    exclude=properties,
                    molecule_values=molecule_values,
                )
                if extra is None:
                    continue
                properties.append(extra)
            if len(properties) < load:
                self.drops["not_enough_properties"] += 1
                return None

        question_smiles, randomized, kekulized = transform_smiles(
            source_smiles, self.rng, spec.randomize_prob, spec.kekulize_prob
        )
        complexity = complexity_of(question_smiles)
        if complexity is None:
            self.drops["unparseable_question_smiles"] += 1
            return None

        # Ground truth on the displayed string, whose atom order defines the indices.
        target: dict[str, Any] = {}
        for prop in properties:
            try:
                value = self.engine.compute(question_smiles, prop)
            except Exception:
                self.drops["property_computation_failed"] += 1
                return None
            if value is None:
                self.drops["property_is_none"] += 1
                return None
            if family == "index":
                if not isinstance(value, (list, tuple)):
                    self.drops["index_not_a_list"] += 1
                    return None
                target[prop] = sorted({int(i) for i in value})
            else:
                if isinstance(value, (list, tuple, set, dict)):
                    self.drops["count_not_scalar"] += 1
                    return None
                target[prop] = _json_safe(value)

        single = load == 1
        natural = [self.formatter.technical_to_natural(p, family) for p in properties]
        placeholder = "{count_type}" if family == "count" else "{index_type}"
        replacement = "{count_types}" if family == "count" else "{index_types}"
        template = self.rng.choice(self._templates[(family, single)]).replace(
            placeholder, replacement
        )
        builder = (
            self.formatter.format_count_query
            if family == "count"
            else self.formatter.format_index_query
        )
        question = builder(
            question_smiles,
            natural,
            template=template,
            include_key_hint=True,
            key_names=properties,
        )

        self._note_usage(family, properties)
        category_name = _category_of(properties[0])
        feature = (
            f"single_{family}_{category_name}"
            if single
            else f"multi_{family}_nbr_{load}"
        )
        return GeneratedExample(
            task_family=family,
            task_type=("single_" if single else "multi_")
            + ("count" if family == "count" else "index"),
            question=question,
            feature=feature,
            supercategory=category_name,
            properties=properties,
            multitask_load=load,
            question_smiles=question_smiles,
            molecule_id=source_smiles,
            witness_smiles=question_smiles,
            molecular_complexity=complexity,
            complexity_bin=complexity_bin_of(complexity),
            is_randomized=randomized,
            is_kekulized=kekulized,
            question_seed=self.rng.randrange(2**31),
            target=target,
        )

    def _constraint_example(
        self, load: int, spec: GenerationSpec
    ) -> GeneratedExample | None:
        """Build a constraint-generation item seeded from a real molecule.

        Constraints are read off one pool molecule, which guarantees the item is
        satisfiable: at least that molecule meets them. The molecule is stored
        only as a witness -- correctness is never string equality to it, since
        any molecule meeting the constraints is equally correct.
        """
        family = "constraint_generation"
        category = self._pick_category(family)
        if category is None:
            self.drops["no_category"] += 1
            return None
        seed_property = self._pick_property(family, category)
        if seed_property is None:
            self.drops["no_property"] += 1
            return None

        chosen_bin, rows = self._bin_for_target_complexity(spec)
        rows_key: Any = chosen_bin
        if spec.default_molecules and rows is not None:
            # Only molecules whose value no default molecule shares.
            rows_key = (chosen_bin, "not-met-by-default")
            if (seed_property, rows_key, spec.zero_weight, spec.value_frequency_power) not in self.table._draw_cache:
                column = self.table.values.get(seed_property) or []
                rows = [
                    row
                    for row in rows
                    if row < len(column)
                    and not self._met_by_default({"type": seed_property, "value": column[row]})
                ]
        row = self.table.sample_by_value_frequency(
            seed_property,
            self.rng,
            zero_weight=spec.zero_weight,
            rows=rows,
            power=spec.value_frequency_power,
            rows_key=rows_key,
        )
        if row is None:
            self.drops["no_molecule"] += 1
            return None
        source_smiles = self.table.smiles[row]

        properties = [seed_property]
        if load > 1:
            molecule_values = self._molecule_values(family, row)
            if spec.default_molecules:
                self._add_unblocked_properties(family, load, properties, molecule_values)
            else:
                for _ in range(load - 1):
                    extra_category = self._pick_category(family)
                    if extra_category is None:
                        break
                    extra = self._pick_property(
                        family,
                        extra_category,
                        exclude=properties,
                        molecule_values=molecule_values,
                    )
                    if extra is None:
                        continue
                    properties.append(extra)
            if len(properties) < load:
                self.drops["not_enough_properties"] += 1
                return None

        constraints: list[dict[str, Any]] = []
        for prop in properties:
            try:
                value = self.engine.compute(source_smiles, prop)
            except Exception:
                self.drops["property_computation_failed"] += 1
                return None
            if value is None or isinstance(value, (list, tuple, set, dict)):
                self.drops["constraint_not_scalar"] += 1
                return None
            constraints.append({"type": prop, "operator": "=", "value": _json_safe(value)})

        # A constraint a default answer already meets pays that answer for free.
        if spec.default_molecules and any(
            self._met_by_default(constraint) for constraint in constraints
        ):
            self.drops["met_by_default_molecule"] += 1
            return None

        # Reject constraints nearly every molecule meets.
        prevalence = self.table.joint_prevalence(constraints)
        if prevalence > spec.max_constraint_prevalence:
            self.drops["constraint_too_common"] += 1
            return None
        if prevalence < spec.min_constraint_prevalence:
            self.drops["constraint_too_rare"] += 1
            return None

        constraint_text = self.formatter.format_constraints_list(constraints)
        template = self.rng.choice(self._constraint_templates)
        question = template.format(constraint=constraint_text).rstrip()
        question += self.formatter.format_constraint_hint()

        self._note_usage(family, properties)
        category_name = _category_of(properties[0])
        feature = (
            f"single_constraint_gen_exact_{category_name}"
            if load == 1
            else f"multi_constraint_generation_{load}_constraints"
        )
        complexity = self.table.complexity[row]
        return GeneratedExample(
            task_family=family,
            task_type="constraint_generation",
            question=question,
            feature=feature,
            supercategory=category_name,
            properties=properties,
            multitask_load=load,
            question_smiles=None,
            molecule_id=source_smiles,
            witness_smiles=source_smiles,
            molecular_complexity=complexity,
            complexity_bin=self.table.complexity_bin[row],
            is_randomized=False,
            is_kekulized=False,
            question_seed=self.rng.randrange(2**31),
            constraints=constraints,
            constraint_prevalence=prevalence,
        )


def _value_key(value: Any) -> Any:
    """A hashable form under which 2 and 2.0 compare equal."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return tuple(_value_key(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _value_key(v)) for k, v in value.items()))
    return value


def _category_of(prop: str) -> str:
    """Mirror the benchmark's get_property_category."""
    for mapping in (COUNT_MAP, INDEX_MAP, CONSTRAINT_MAP):
        for category, props in mapping.items():
            if prop in props:
                return category
    if prop.startswith(_FG_PREFIX):
        return "functional_group"
    if prop.startswith("template_based_reaction_prediction_"):
        return "reaction_templates"
    return prop


def recompute_target(
    question_smiles: str,
    properties: Sequence[str],
    family: str,
    engine: PropertyEngine | None = None,
) -> dict[str, Any] | None:
    """Recompute a count/index target from scratch off the displayed SMILES.

    oracle_score only proves the verifier accepts a stored answer as its own
    ground truth; it compares the target against itself and would happily accept
    a target that had been corrupted after generation. This recomputes from the
    molecule, so a stored answer that no longer matches the question is caught.
    """
    engine = engine or PropertyEngine()
    target: dict[str, Any] = {}
    for prop in properties:
        try:
            value = engine.compute(question_smiles, prop)
        except Exception:
            return None
        if value is None:
            return None
        if family == "index":
            if not isinstance(value, (list, tuple)):
                return None
            target[prop] = sorted({int(i) for i in value})
        else:
            if isinstance(value, (list, tuple, set, dict)):
                return None
            target[prop] = _json_safe(value)
    return target


def oracle_score(example: GeneratedExample) -> float:
    """Score the example's own ground truth with the official verifier.

    Every generated row must pass this. A row whose stored answer the verifier
    calls wrong is a row GRPO could never earn reward on, and would quietly add
    noise to the gradient.
    """
    if example.task_family == "constraint_generation":
        return float(
            evaluate_answer(
                task_type="constraint_generation",
                predicted=json.dumps({"smiles": example.witness_smiles}),
                constraints=example.constraints,
            )
        )
    return float(
        evaluate_answer(
            task_type=example.task_type,
            predicted=json.dumps(example.target),
            target=example.target,
        )
    )
