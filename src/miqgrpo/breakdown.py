"""Where a model learns: per-property outcomes and the constant-answer reference.

Item-level accuracy says *whether* a score moved, not what moved it. Two things
separate learned chemistry from a learned default answer.

Per-property outcomes. For every attempt the official scorer reports which
requested properties matched (count and index questions) and which constraints
were satisfied (constrained generation). Each (question, property) or
(question, constraint) pair is one *unit* here.

The constant answer. A fixed answer that ignores the molecule is already
correct on some units: 0 for a count, [] for an index list, and for
generation the example molecule the official system prompt shows. A unit is
*trivial* when that constant answer is correct on it, and a question is trivial
when the constant answer solves all of its units. A gain on trivial units can
come from drifting towards the constant answer; only a gain on non-trivial units
shows the model working something out about the molecule.

Everything is scored by moleculariq_core.evaluate_answer, the function the
training reward and the official benchmark use.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

from .prompts import SYSTEM_PROMPT

__all__ = [
    "CONSTANT_SMILES",
    "Question",
    "Scored",
    "Unit",
    "constant_answer",
    "constant_share",
    "is_trivial",
    "paired_split",
    "property_changes",
    "question_from_columns",
    "question_from_doc",
    "question_from_row",
    "unit_credit",
    "score_samples",
    "score_units",
    "split_accuracy",
    "unit_accuracy",
    "units",
]

# Example molecule of the official system prompt (a policy can learn to repeat it).
CONSTANT_SMILES = re.search(r'"smiles":\s*"([^"]+)"', SYSTEM_PROMPT).group(1)

_SCORER_TASK = {"count": "count", "index": "index", "generation": "constraint_generation"}


@dataclass(frozen=True)
class Question:
    """One question as the scorer needs it, from the benchmark or the training set."""

    task: str  # count | index | generation
    target: str | None  # canonical JSON of the target dict (count, index)
    constraints: str | None  # canonical JSON of the constraint list (generation)

    @property
    def target_dict(self) -> dict[str, Any]:
        """Stored count/index answer as a dict (empty for generation questions)."""
        return json.loads(self.target) if self.target else {}

    @property
    def constraint_list(self) -> list[dict[str, Any]]:
        """Stored generation constraints (empty for count/index questions)."""
        return json.loads(self.constraints) if self.constraints else []


@dataclass(frozen=True)
class Unit:
    """One separately scorable part of a question.

    A requested property of a count or index question, or one constraint of a
    generation question. trivial marks units a constant answer gets right.
    """

    key: str  # property key, or the canonical JSON of one constraint
    label: str  # property name used for grouping
    trivial: bool


def _canonical(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    return json.dumps(value, sort_keys=True)


def question_from_doc(doc: dict[str, Any]) -> Question:
    """A benchmark item (doc of an lm_eval sample record)."""
    task = doc["task_type"]
    if task == "generation":
        return Question(task, None, _canonical(doc.get("constraints")))
    return Question(task, _canonical(doc.get("target")), None)


def question_from_row(row: dict[str, Any]) -> Question:
    """A row of the frozen training dataset."""
    family = row["task_family"]
    if family == "constraint_generation":
        return Question("generation", None, _canonical(row.get("constraints_json")))
    return Question(family, _canonical(row.get("target_json")), None)


def question_from_columns(
    task: str, target_json: str | None, constraints_json: str | None
) -> Question | None:
    """From the columns a reward function receives; task is count|index|generation."""
    try:
        if task == "generation":
            constraints = _canonical(constraints_json) if constraints_json else None
            return Question(task, None, constraints) if constraints else None
        target = _canonical(target_json) if target_json else None
        return Question(task, target, None) if target else None
    except (json.JSONDecodeError, TypeError):
        return None


def constant_answer(question: Question) -> str:
    """The answer that ignores the molecule entirely."""
    if question.task == "generation":
        return json.dumps({"smiles": CONSTANT_SMILES})
    empty: Any = 0 if question.task == "count" else []
    return json.dumps({key: empty for key in question.target_dict})


def _as_text(answer: Any) -> str | None:
    if answer is None:
        return None
    return answer if isinstance(answer, str) else json.dumps(answer)


@lru_cache(maxsize=500_000)
def score_units(question: Question, answer: str | None) -> tuple[tuple[str, bool], ...]:
    """Per-unit outcome of one answer: ((unit key, correct), ...).

    An answer the scorer cannot parse gets every unit wrong. Cached, because
    collapsed policies repeat the same answer thousands of times.
    """
    keys = _unit_keys(question)
    if not answer:
        return tuple((key, False) for key in keys)
    try:
        if question.task == "generation":
            report = _evaluate(
                task_type="constraint_generation",
                predicted=answer,
                constraints=question.constraint_list,
                return_details=True,
            )
            status = {
                _canonical(item["constraint"]): bool(item.get("satisfied"))
                for item in (report.get("details") or [])
                if isinstance(item, dict) and "constraint" in item
            }
        else:
            report = _evaluate(
                task_type=_SCORER_TASK[question.task],
                predicted=answer,
                target=question.target_dict,
                return_details=True,
            )
            status = {
                key: bool(item.get("match"))
                for key, item in (report.get("details") or {}).items()
                if isinstance(item, dict)
            }
    except Exception:  # noqa: BLE001 - an unscorable answer is a wrong answer
        status = {}
    return tuple((key, status.get(key, False)) for key in keys)


def _evaluate(**kwargs: Any) -> dict[str, Any]:
    from moleculariq_core import evaluate_answer
    from rdkit import RDLogger

    # Model answers are full of malformed SMILES; RDKit would log every one.
    RDLogger.DisableLog("rdApp.*")

    report = evaluate_answer(**kwargs)
    return report if isinstance(report, dict) else {}


def _unit_keys(question: Question) -> tuple[str, ...]:
    if question.task == "generation":
        return tuple(_canonical(c) for c in question.constraint_list)
    return tuple(question.target_dict)


@lru_cache(maxsize=100_000)
def units(question: Question) -> tuple[Unit, ...]:
    """The question's units, each marked trivial if the constant answer gets it."""
    constant = dict(score_units(question, constant_answer(question)))
    labels = (
        # The benchmark calls a constraint's property "property", the training set "type".
        [c.get("property") or c.get("type") for c in question.constraint_list]
        if question.task == "generation"
        else list(question.target_dict)
    )
    return tuple(
        Unit(key, label, constant.get(key, False))
        for key, label in zip(_unit_keys(question), labels)
    )


def is_trivial(question: Question) -> bool:
    """True when the constant answer solves the whole question."""
    found = units(question)
    return bool(found) and all(unit.trivial for unit in found)


def solved_by_any(question: Question, molecules: Iterable[str]) -> bool:
    """True when one of molecules meets every constraint of a generation question.

    With the default molecules a dataset was filtered against, this is the
    wider version of is_trivial: a question some answer written without
    reading the question already solves.
    """
    for smiles in molecules:
        found = score_units(question, json.dumps({"smiles": smiles}))
        if found and all(correct for _, correct in found):
            return True
    return False


def generated_smiles(answers: Sequence[Any]) -> list[str | None]:
    """The SMILES of each extracted generation answer (None when it has none)."""
    out: list[str | None] = []
    for answer in answers:
        try:
            parsed = json.loads(answer) if isinstance(answer, str) else answer
        except (json.JSONDecodeError, TypeError):
            parsed = None
        smiles = parsed.get("smiles") if isinstance(parsed, dict) else None
        out.append(smiles if isinstance(smiles, str) else None)
    return out


def unit_accuracy(
    question: Question, answers: Sequence[Any]
) -> dict[str, float]:
    """Fraction of attempts that got each unit right."""
    keys = [unit.key for unit in units(question)]
    hits: dict[str, float] = defaultdict(float)
    for answer in answers:
        for key, correct in score_units(question, _as_text(answer)):
            hits[key] += correct
    n = max(1, len(answers))
    return {key: hits[key] / n for key in keys}


def constant_share(question: Question, answers: Iterable[Any]) -> float:
    """Fraction of attempts that are the constant answer, up to equivalence.

    For count and index an attempt counts when every requested value is 0 or
    []; for generation when it is the constant molecule.
    """
    answers = [_as_text(a) for a in answers]
    if not answers:
        return 0.0
    hits = 0
    for answer in answers:
        try:
            parsed = json.loads(answer) if answer else None
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if not isinstance(parsed, dict):
            continue
        if question.task == "generation":
            hits += parsed.get("smiles") == CONSTANT_SMILES
        else:
            values = [parsed.get(key) for key in question.target_dict]
            hits += bool(values) and all(v == 0 or v == [] for v in values)
    return hits / len(answers)


@dataclass
class Scored:
    """One benchmark question answered by one model (all its attempts)."""

    key: Any  # benchmark doc_id
    question: Question
    accuracy: float  # official item-level score, averaged over attempts
    unit_accuracy: dict[str, float]
    constant_share: float


def score_samples(samples: Iterable[dict[str, Any]]) -> list[Scored]:
    """Per-unit outcomes of every benchmark item, from lm_eval sample records."""
    out = []
    for record in samples:
        question = question_from_doc(record["doc"])
        answers = record.get("extracted_answers") or []
        out.append(
            Scored(
                key=record.get("doc_id"),
                question=question,
                accuracy=float(record.get("avg_accuracy", 0.0)),
                unit_accuracy=unit_accuracy(question, answers),
                constant_share=constant_share(question, answers),
            )
        )
    return out


def split_accuracy(scored: Sequence[Scored], task: str) -> dict[str, tuple[float, int]]:
    """Accuracy on questions the constant answer solves vs all other questions.

    Also reports unit-level accuracy on non-trivial units, the finest measure of
    whether anything about the molecule was worked out.
    """
    trivial = [s.accuracy for s in scored if s.question.task == task and is_trivial(s.question)]
    other = [s.accuracy for s in scored if s.question.task == task and not is_trivial(s.question)]
    other_units = [
        s.unit_accuracy[u.key]
        for s in scored
        if s.question.task == task
        for u in units(s.question)
        if not u.trivial
    ]
    shares = [s.constant_share for s in scored if s.question.task == task]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    return {
        "trivial": (mean(trivial), len(trivial)),
        "other": (mean(other), len(other)),
        "other_units": (mean(other_units), len(other_units)),
        "constant_share": (mean(shares), len(shares)),
    }


def paired_split(
    first: Sequence[Scored], second: Sequence[Scored], task: str, trivial: bool
) -> list[float]:
    """Per-item first - second on one side of the trivial/other split."""
    other = {s.key: s for s in second}
    return [
        s.accuracy - other[s.key].accuracy
        for s in first
        if s.question.task == task and is_trivial(s.question) == trivial and s.key in other
    ]


def property_changes(
    first: Sequence[Scored], second: Sequence[Scored], task: str, min_units: int = 40
) -> dict[str, tuple[float, list[float]]]:
    """Per property: base accuracy of second and per-unit first - second.

    Non-trivial units only. A property occurs at most once per question, so the
    differences are independent across questions and can be bootstrapped as-is.
    """
    other = {s.key: s for s in second}
    base: dict[str, list[float]] = defaultdict(list)
    diffs: dict[str, list[float]] = defaultdict(list)
    for s in first:
        match = other.get(s.key)
        if s.question.task != task or match is None:
            continue
        for unit in units(s.question):
            if unit.trivial:
                continue
            base[unit.label].append(match.unit_accuracy[unit.key])
            diffs[unit.label].append(s.unit_accuracy[unit.key] - match.unit_accuracy[unit.key])
    return {
        label: (sum(base[label]) / len(base[label]), diffs[label])
        for label in diffs
        if len(diffs[label]) >= min_units
    }


def _index_list(value: Any) -> set[int] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in value):
        return None
    return set(value)


@lru_cache(maxsize=200_000)
def unit_credit(question: Question, answer: str | None) -> tuple[tuple[str, float], ...]:
    """Credit in [0, 1] for each unit of one answer: the partial-credit reward.

    Count properties and generation constraints get the official verdict (1 or
    0). An index property gets the overlap between the predicted and the true
    atom set (intersection over union), so a list that is half right earns
    something; an exactly right list, which the official scorer accepts, earns 1.
    """
    official = dict(score_units(question, answer))
    if question.task != "index":
        return tuple((key, float(correct)) for key, correct in official.items())
    try:
        parsed = json.loads(answer) if answer else None
    except (json.JSONDecodeError, TypeError):
        parsed = None
    parsed = parsed if isinstance(parsed, dict) else {}
    target = question.target_dict
    credit = []
    for key, correct in official.items():
        if correct:
            credit.append((key, 1.0))
            continue
        predicted, truth = _index_list(parsed.get(key)), set(target[key])
        if predicted is None or not (predicted | truth):
            credit.append((key, 0.0))
        else:
            credit.append((key, len(predicted & truth) / len(predicted | truth)))
    return tuple(credit)
