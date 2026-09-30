"""GRPO runtime rewards.

Scores *new* model completions against targets and constraints that were
computed offline and frozen into the dataset. This module never generates a
question, never resamples a property and never touches the benchmark.

The scoring path is deliberately identical to the official benchmark's:

    raw completion
      -> extract_moleculariq_answer()   (vendored from moleculariq-eval)
      -> evaluate_answer()              (moleculariq_core reward dispatcher)
      -> 0.0 / 1.0

so that "reward goes up" and "benchmark score goes up" mean the same thing.

The second term is partial credit per requested property (see
PartialCreditReward), which gives GRPO a learning signal on questions
the policy cannot yet answer completely.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Callable, Sequence

from moleculariq_core import evaluate_answer, valid_smiles

from .breakdown import (
    constant_share,
    is_trivial,
    question_from_columns,
    score_units,
    unit_credit,
    units,
)
from .prompts import completion_to_text
from .vendor import extract_moleculariq_answer

__all__ = [
    "FormatStatus",
    "ParseStatus",
    "ScoredCompletion",
    "RewardConfig",
    "CorrectnessReward",
    "PartialCreditReward",
    "build_reward_functions",
    "score_completion",
]


class ParseStatus:
    """Whether the *official* extractor + verifier produced a usable answer.

    Deliberately separate from FormatStatus. The official extractor
    happily recovers a bare json block, so a completion with no <answer>
    tags is a formatting miss, not a parse failure -- conflating the two made
    every early-training rollout look broken when the answers were in fact
    being read correctly.
    """

    OK = "OK"
    EMPTY = "EMPTY"
    MALFORMED = "MALFORMED"
    INVALID_SMILES = "INVALID_SMILES"
    VERIFIER_ERROR = "VERIFIER_ERROR"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"


class FormatStatus:
    """Whether the completion used the answer envelope the prompt asked for."""

    OK = "OK"
    NO_ANSWER_TAGS = "NO_ANSWER_TAGS"
    MULTIPLE_ANSWER_TAGS = "MULTIPLE_ANSWER_TAGS"
    CONFLICTING_ANSWER_TAGS = "CONFLICTING_ANSWER_TAGS"
    INVALID_JSON = "INVALID_JSON"
    EMPTY = "EMPTY"


_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

_COUNT_TASKS = frozenset({"single_count", "multi_count", "count"})
_INDEX_TASKS = frozenset({"single_index", "multi_index", "index"})
_GENERATION_TASKS = frozenset(
    {"constraint_generation", "single_constraint_generation", "multi_constraint_generation", "generation"}
)


@dataclass
class ScoredCompletion:
    """One completion's reward plus everything needed to debug it."""

    correctness: float = 0.0
    status: str = ParseStatus.EMPTY
    format_status: str = FormatStatus.EMPTY
    extracted: str = ""
    # the full extracted answer (extracted is truncated for logging)
    answer: str | None = None
    answer_blocks: int = 0
    conflicting_answers: bool = False
    valid_smiles: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        """The score as a plain dict (for logging)."""
        return asdict(self)


@lru_cache(maxsize=8192)
def _loads(payload: str) -> Any:
    """Parse a stored JSON column.

    Cached on the JSON *text*, so a cache hit can only ever return the value
    that text encodes -- there is no key under which two different targets
    could collide.
    """
    return json.loads(payload)


def _normalise_task_type(task_type: str) -> str | None:
    task = (task_type or "").lower().replace("-", "_")
    if task in _COUNT_TASKS or task in _INDEX_TASKS:
        return task
    if task in _GENERATION_TASKS:
        return "constraint_generation"
    return None


def _answer_blocks(text: str) -> list[str]:
    return [block.strip() for block in _ANSWER_BLOCK.findall(text)]


def _as_json_object(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _format_status(text: str, blocks: Sequence[str], extracted: Any) -> str:
    """Did the completion use the <answer>{...}</answer> envelope it was asked for?

    A diagnostic only: the reward is the official score, which the official
    extractor computes with or without the envelope.
    """
    if not text.strip():
        return FormatStatus.EMPTY
    if len(blocks) == 1:
        status = FormatStatus.OK
    elif len(blocks) > 1:
        status = (
            FormatStatus.CONFLICTING_ANSWER_TAGS
            if len(set(blocks)) > 1
            else FormatStatus.MULTIPLE_ANSWER_TAGS
        )
    else:
        status = FormatStatus.NO_ANSWER_TAGS
    payload = _as_json_object(blocks[-1] if blocks else extracted)
    if payload is None:
        payload = _as_json_object(extracted)
    if payload is None and status == FormatStatus.OK:
        status = FormatStatus.INVALID_JSON
    return status


def _extract_smiles(extracted: Any) -> str | None:
    """Pull the SMILES out of whatever the official extractor returned."""
    value: Any = extracted
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                value = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return text
        else:
            return text
    if isinstance(value, dict):
        for key in value:
            if str(key).lower() == "smiles":
                return str(value[key]).strip()
        if len(value) == 1:
            return str(next(iter(value.values()))).strip()
        return None
    if isinstance(value, list) and len(value) == 1:
        return str(value[0]).strip()
    return None if value is None else str(value).strip()


def score_completion(
    text: str,
    task_type: str,
    target_json: str | None,
    constraints_json: str | None,
) -> ScoredCompletion:
    """Turn one raw completion into exactly one finite, diagnosed reward.

    Malformed model output is expected during RL and must never take the run
    down, so every failure mode resolves to a status plus reward 0.
    """
    result = ScoredCompletion()

    if not isinstance(text, str) or not text.strip():
        result.status = ParseStatus.EMPTY
        result.format_status = FormatStatus.EMPTY
        return result

    blocks = _answer_blocks(text)
    result.answer_blocks = len(blocks)
    result.conflicting_answers = len({b for b in blocks}) > 1

    normalised = _normalise_task_type(task_type)
    if normalised is None:
        result.status = ParseStatus.UNSUPPORTED_TASK
        result.format_status = FormatStatus.NO_ANSWER_TAGS
        return result

    extracted = extract_moleculariq_answer(text)
    result.format_status = _format_status(text, blocks, extracted)

    if extracted is None:
        result.status = ParseStatus.MALFORMED
        return result
    result.answer = extracted if isinstance(extracted, str) else json.dumps(extracted)
    result.extracted = result.answer[:500]

    if normalised == "constraint_generation":
        smiles = _extract_smiles(extracted)
        result.valid_smiles = bool(smiles) and valid_smiles(smiles)
        if not result.valid_smiles:
            # Still run the verifier so the reward stays the official one.
            result.status = ParseStatus.INVALID_SMILES

    try:
        if normalised == "constraint_generation":
            constraints = _loads(constraints_json) if constraints_json else None
            if not constraints:
                result.status = ParseStatus.UNSUPPORTED_TASK
                return result
            score = evaluate_answer(
                task_type="constraint_generation",
                predicted=extracted,
                constraints=constraints,
            )
        else:
            target = _loads(target_json) if target_json else None
            if target is None:
                result.status = ParseStatus.UNSUPPORTED_TASK
                return result
            score = evaluate_answer(
                task_type=normalised, predicted=extracted, target=target
            )
    except Exception:  # noqa: BLE001 - a verifier crash must not kill training
        result.status = ParseStatus.VERIFIER_ERROR
        return result

    if isinstance(score, dict):
        score = score.get("reward", 0.0)
    try:
        correctness = float(score)
    except (TypeError, ValueError):
        result.status = ParseStatus.VERIFIER_ERROR
        return result
    if correctness != correctness or correctness in (float("inf"), float("-inf")):
        result.status = ParseStatus.VERIFIER_ERROR
        return result

    result.correctness = correctness
    if result.status == ParseStatus.EMPTY:
        result.status = ParseStatus.OK
    return result


@dataclass
class RewardConfig:
    """Reward weights, recorded in the experiment config.

    reward = correctness_weight x official score of the whole question
           + partial_weight     x per-property credit (PartialCreditReward)
    """

    correctness_weight: float = 1.0
    partial_weight: float = 0.0
    # weight of properties a constant answer gets right (true value 0 or []); 0 = off
    trivial_weight: float = 0.0
    # a wrong "none" answer scores -none_penalty instead of 0
    none_penalty: float = 0.0
    # property -> most common value in the training split (diagnostics only)
    typical_values: dict[str, Any] = field(default_factory=dict)
    log_diagnostics: bool = True

    def as_dict(self) -> dict[str, Any]:
        """The reward settings as a plain dict (recorded in the run's provenance)."""
        return asdict(self)


class _BaseReward:
    """Shared scoring + diagnostics for the TRL reward callables.

    Both callables need the same per-completion score, so it is computed once
    per batch and cached on a shared holder.
    """

    def __init__(self, cache: "_ScoreCache", config: RewardConfig) -> None:
        self._cache = cache
        self._config = config


@dataclass
class _ScoreCache:
    """Per-batch scores, shared between the reward callables of one trainer."""

    key: tuple[int, ...] = field(default_factory=tuple)
    scores: list[ScoredCompletion] = field(default_factory=list)

    def get(
        self,
        completions: Sequence[Any],
        task_type: Sequence[str],
        target_json: Sequence[str | None],
        constraints_json: Sequence[str | None],
    ) -> list[ScoredCompletion]:
        """Scores of this batch, computed once and reused by every reward callable."""
        texts = [completion_to_text(c) for c in completions]
        key = tuple(hash((t, tt)) for t, tt in zip(texts, task_type))
        if key == self.key and len(self.scores) == len(texts):
            return self.scores
        self.scores = [
            score_completion(text, task, target, constraints)
            for text, task, target, constraints in zip(
                texts, task_type, target_json, constraints_json
            )
        ]
        self.key = key
        return self.scores


def _columns(
    n: int, kwargs: dict[str, Any], name: str
) -> list[Any]:
    """Fetch a dataset column TRL repeated across the generation group."""
    column = kwargs.get(name)
    if column is None:
        return [None] * n
    return list(column)


class CorrectnessReward(_BaseReward):
    """Binary official-verifier reward. This is the signal that matters."""

    def __call__(
        self,
        completions: Sequence[Any] | None = None,
        log_metric: Callable[[str, float], None] | None = None,
        log_extra: Callable[[str, list], None] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        completions = completions or []
        n = len(completions)
        task_type = _columns(n, kwargs, "task_type")
        scores = self._cache.get(
            completions,
            task_type,
            _columns(n, kwargs, "target_json"),
            _columns(n, kwargs, "constraints_json"),
        )

        if self._config.log_diagnostics and n:
            questions = _questions(
                task_type,
                _columns(n, kwargs, "target_json"),
                _columns(n, kwargs, "constraints_json"),
            )
            self._log(scores, questions, log_metric, log_extra)

        weight = self._config.correctness_weight
        return [weight * s.correctness for s in scores]

    def _log(
        self,
        scores: Sequence[ScoredCompletion],
        questions: Sequence[Any],
        log_metric: Callable[[str, float], None] | None,
        log_extra: Callable[[str, list], None] | None,
    ) -> None:
        n = len(scores)
        if log_metric is not None:
            _log_shortcuts(scores, questions, log_metric)
            if self._config.typical_values:
                _typical_diagnostics(scores, questions, self._config.typical_values, log_metric)
            ok = sum(1 for s in scores if s.status == ParseStatus.OK)
            log_metric("parse/failure_fraction", 1.0 - ok / n)
            log_metric(
                "parse/answer_tag_fraction",
                sum(1 for s in scores if s.answer_blocks) / n,
            )
            log_metric(
                "parse/well_formed_fraction",
                sum(1 for s in scores if s.format_status == FormatStatus.OK) / n,
            )
            log_metric(
                "parse/conflicting_answer_fraction",
                sum(1 for s in scores if s.conflicting_answers) / n,
            )
            log_metric(
                "verifier/error_fraction",
                sum(1 for s in scores if s.status == ParseStatus.VERIFIER_ERROR) / n,
            )
            log_metric(
                "reward/correctness_mean", sum(s.correctness for s in scores) / n
            )
            checked = [s for s in scores if s.valid_smiles is not None]
            if checked:
                log_metric(
                    "chem/invalid_smiles_fraction",
                    sum(1 for s in checked if not s.valid_smiles) / len(checked),
                )
        if log_extra is not None:
            log_extra("parse_status", [s.status for s in scores])
            log_extra("format_status", [s.format_status for s in scores])
            log_extra("extracted_answer", [s.extracted for s in scores])


_BREAKDOWN_TASK = {**{t: "count" for t in _COUNT_TASKS}, **{t: "index" for t in _INDEX_TASKS}}


def _questions(
    task_type: Sequence[str | None],
    target_json: Sequence[str | None],
    constraints_json: Sequence[str | None],
) -> list[Any]:
    """The scorer's view of each sample's question, or None when unusable."""
    out = []
    for task, target, constraints in zip(task_type, target_json, constraints_json):
        normalised = _normalise_task_type(task or "")
        if normalised is None:
            out.append(None)
            continue
        kind = _BREAKDOWN_TASK.get(normalised, "generation")
        out.append(question_from_columns(kind, target, constraints))
    return out


def _nontrivial_credit(question: Any, answer: str | None) -> float | None:
    """Mean partial credit over the units a constant answer gets wrong."""
    return _weighted_credit(question, answer, trivial_weight=0.0)


def _is_none_value(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0) or value == []


def _answered_none(question: Any, answer: str | None) -> set[str]:
    """The count/index properties this answer claims are absent (0 or [])."""
    if question.task not in ("count", "index") or not answer:
        return set()
    try:
        parsed = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return set()
    if not isinstance(parsed, dict):
        return set()
    return {key for key in question.target_dict if key in parsed and _is_none_value(parsed[key])}


def _weighted_credit(
    question: Any,
    answer: str | None,
    trivial_weight: float,
    none_penalty: float = 0.0,
) -> float | None:
    """Weighted mean partial credit over a question's units.

    Units a constant answer gets right (true value 0 or [], or a constraint the
    prompt's example molecule meets) weigh trivial_weight, all others 1.
    None when no unit carries weight. A "none" answer on a unit whose true
    value is not none scores -none_penalty instead of 0.
    """
    credit = dict(unit_credit(question, answer))
    if none_penalty:
        for key in _answered_none(question, answer):
            if not _is_none_value(question.target_dict[key]):
                credit[key] = -none_penalty
    weights, keys = [], []
    for unit in units(question):
        weight = trivial_weight if unit.trivial else 1.0
        if weight > 0:
            weights.append(weight)
            keys.append(unit.key)
    if not keys:
        return None
    return sum(w * credit[k] for w, k in zip(weights, keys)) / sum(weights)


def _typical_diagnostics(
    scores: Sequence[ScoredCompletion],
    questions: Sequence[Any],
    typical: dict[str, Any],
    log_metric: Callable[[str, float], None],
) -> None:
    """How often answers are the per-property default, and accuracy beyond it."""
    said, atypical = [], []
    for score, question in zip(scores, questions):
        if question is None or question.task not in ("count", "index"):
            continue
        try:
            parsed = json.loads(score.answer) if score.answer else {}
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        parsed = parsed if isinstance(parsed, dict) else {}
        official = dict(score_units(question, score.answer))
        target = question.target_dict
        for unit in units(question):
            if unit.trivial or unit.key not in typical:
                continue
            said.append(parsed.get(unit.key) == typical[unit.key])
            if target[unit.key] != typical[unit.key]:
                atypical.append(official[unit.key])
    if said:
        log_metric("shortcut/typical_answer_share", sum(said) / len(said))
    if atypical:
        log_metric("correctness/atypical_properties", sum(atypical) / len(atypical))


def _log_shortcuts(
    scores: Sequence[ScoredCompletion],
    questions: Sequence[Any],
    log_metric: Callable[[str, float], None],
) -> None:
    """Is the reward being earned by working the question out, or by a default?

    Logged every step so a policy drifting towards a constant answer shows up
    during training rather than in an evaluation afterwards (see
    miqgrpo.breakdown for the definitions).
    """
    pairs = [(s, q) for s, q in zip(scores, questions) if q is not None]
    if not pairs:
        return
    log_metric(
        "shortcut/constant_answer_share",
        sum(constant_share(q, [s.answer]) for s, q in pairs) / len(pairs),
    )
    trivial = [s.correctness for s, q in pairs if is_trivial(q)]
    other = [s.correctness for s, q in pairs if not is_trivial(q)]
    if trivial:
        log_metric("correctness/trivial_questions", sum(trivial) / len(trivial))
    if other:
        log_metric("correctness/other_questions", sum(other) / len(other))
    credit = [c for c in (_nontrivial_credit(q, s.answer) for s, q in pairs) if c is not None]
    if credit:
        log_metric("partial/nontrivial_credit", sum(credit) / len(credit))
    # "none" answers (0 / []) per property: how often given, how often right.
    said = right = asked = 0
    for s, q in pairs:
        if q.task not in ("count", "index"):
            continue
        claims = _answered_none(q, s.answer)
        asked += len(q.target_dict)
        said += len(claims)
        right += sum(_is_none_value(q.target_dict[key]) for key in claims)
    if asked:
        log_metric("none/answer_rate", said / asked)
    if said:
        log_metric("none/precision", right / said)


class PartialCreditReward(_BaseReward):
    """Credit for each requested property the answer gets right.

    The official score is all-or-nothing per question, so a group in which no
    rollout answers every property gets identical rewards and no gradient --
    which, for a 0.5B policy, is most groups. This term scores each property
    (count, index) or constraint (generation) separately: the official verdict
    for counts and constraints, atom-set overlap for index lists.

    By default only properties a constant answer gets wrong count (true value
    not 0 or [], constraint not met by the system prompt's example
    molecule), so defaulting earns nothing here; trivial_weight > 0 gives
    a correct "none" credit too, and none_penalty makes a wrong "none"
    cost, so that "none" pays only where the feature really is absent.
    Questions with no weighted property get None, which TRL leaves out of
    the sum rather than scoring as zero.
    """

    def __call__(
        self,
        completions: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> list[float | None]:
        completions = completions or []
        n = len(completions)
        task_type = _columns(n, kwargs, "task_type")
        target_json = _columns(n, kwargs, "target_json")
        constraints_json = _columns(n, kwargs, "constraints_json")
        scores = self._cache.get(completions, task_type, target_json, constraints_json)
        weight = self._config.partial_weight
        out: list[float | None] = []
        config = self._config
        for score, question in zip(scores, _questions(task_type, target_json, constraints_json)):
            credit = (
                None
                if question is None
                else _weighted_credit(question, score.answer, config.trivial_weight, config.none_penalty)
            )
            out.append(None if credit is None else weight * credit)
        return out


def build_reward_functions(config: RewardConfig) -> list[Callable[..., Any]]:
    """Build the reward callables for one experiment.

    A partial-credit term with zero weight is left out entirely so it does not
    clutter the logged metrics with a constant zero column.
    """
    cache = _ScoreCache()
    functions: list[Callable[..., Any]] = [CorrectnessReward(cache, config)]
    if config.partial_weight:
        functions.append(PartialCreditReward(cache, config))
    return functions
