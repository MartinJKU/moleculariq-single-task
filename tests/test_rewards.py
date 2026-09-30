"""The GRPO reward pays for correct answers and not for exploits of the scorer."""

from __future__ import annotations

import json

import pytest

from miqgrpo.rewards import (
    FormatStatus,
    ParseStatus,
    RewardConfig,
    build_reward_functions,
    score_completion,
)

COUNT_TARGET = json.dumps({"ring_count": 2})
INDEX_TARGET = json.dumps({"ring_index": [0, 1, 2, 3, 4, 5]})
CONSTRAINTS = json.dumps(
    [{"type": "aromatic_ring_count", "operator": "=", "value": 1}]
)


def correctness(text: str, task_type: str, target=None, constraints=None) -> float:
    return score_completion(text, task_type, target, constraints).correctness


def test_correct_count_scores_one():
    assert correctness('<answer>{"ring_count": 2}</answer>', "single_count", COUNT_TARGET) == 1.0


def test_wrong_count_scores_zero():
    assert correctness('<answer>{"ring_count": 3}</answer>', "single_count", COUNT_TARGET) == 0.0


def test_correct_indices_score_one():
    text = '<answer>{"ring_index": [0, 1, 2, 3, 4, 5]}</answer>'
    assert correctness(text, "single_index", INDEX_TARGET) == 1.0


def test_off_by_one_indices_score_zero():
    """The classic index bug: a shifted list must not be accepted."""
    text = '<answer>{"ring_index": [1, 2, 3, 4, 5, 6]}</answer>'
    assert correctness(text, "single_index", INDEX_TARGET) == 0.0


def test_index_order_does_not_matter():
    """Official semantics compare sets, so a permutation is still correct."""
    text = '<answer>{"ring_index": [5, 4, 3, 2, 1, 0]}</answer>'
    assert correctness(text, "single_index", INDEX_TARGET) == 1.0


def test_empty_completion():
    result = score_completion("", "single_count", COUNT_TARGET, None)
    assert result.correctness == 0.0
    assert result.status == ParseStatus.EMPTY
    assert result.format_status == FormatStatus.EMPTY


def test_whitespace_only_completion():
    result = score_completion("   \n\t ", "single_count", COUNT_TARGET, None)
    assert result.correctness == 0.0
    assert result.status == ParseStatus.EMPTY


def test_malformed_structured_output_does_not_crash():
    text = '<answer>{"ring_count": </answer>'
    result = score_completion(text, "single_count", COUNT_TARGET, None)
    assert result.correctness == 0.0
    assert result.format_status == FormatStatus.INVALID_JSON


def test_conflicting_answers_use_official_last_block_rule():
    """Two blocks: the official extractor reads the last one, so we do too."""
    text = '<answer>{"ring_count": 2}</answer> ... <answer>{"ring_count": 7}</answer>'
    result = score_completion(text, "single_count", COUNT_TARGET, None)
    assert result.conflicting_answers is True
    assert result.correctness == 0.0
    assert result.format_status == FormatStatus.CONFLICTING_ANSWER_TAGS


def test_duplicate_identical_answers_are_not_conflicting():
    text = '<answer>{"ring_count": 2}</answer> <answer>{"ring_count": 2}</answer>'
    result = score_completion(text, "single_count", COUNT_TARGET, None)
    assert result.conflicting_answers is False
    assert result.correctness == 1.0


def test_invalid_smiles_scores_zero():
    text = '<answer>{"smiles": "C1CC"}</answer>'
    result = score_completion(text, "constraint_generation", None, CONSTRAINTS)
    assert result.correctness == 0.0
    assert result.valid_smiles is False
    assert result.status == ParseStatus.INVALID_SMILES


def test_valid_smiles_violating_the_constraint_scores_zero():
    """Chemically fine, but it does not satisfy what was asked."""
    text = '<answer>{"smiles": "CCO"}</answer>'
    result = score_completion(text, "constraint_generation", None, CONSTRAINTS)
    assert result.valid_smiles is True
    assert result.correctness == 0.0


def test_any_molecule_satisfying_the_constraints_is_correct():
    """Correctness is constraint satisfaction, never equality to one reference."""
    for smiles in ("c1ccccc1", "Cc1ccccc1", "OCc1ccccc1O", "c1ccc(Cl)cc1"):
        text = f'<answer>{{"smiles": "{smiles}"}}</answer>'
        assert correctness(text, "constraint_generation", None, CONSTRAINTS) == 1.0, smiles


def test_multi_constraint_requires_all_constraints():
    constraints = json.dumps(
        [
            {"type": "aromatic_ring_count", "operator": "=", "value": 1},
            {"type": "halogen_atom_count", "operator": "=", "value": 1},
        ]
    )
    assert correctness('<answer>{"smiles": "c1ccccc1"}</answer>', "constraint_generation", None, constraints) == 0.0
    assert correctness('<answer>{"smiles": "Clc1ccccc1"}</answer>', "constraint_generation", None, constraints) == 1.0


def test_answer_format_is_a_diagnostic_not_a_reward():
    """The official extractor reads a bare JSON answer, so the reward must too."""
    tagged = score_completion('<answer>{"ring_count": 2}</answer>', "single_count", COUNT_TARGET, None)
    bare = score_completion('the answer is {"ring_count": 2}', "single_count", COUNT_TARGET, None)
    assert tagged.format_status == FormatStatus.OK
    assert bare.format_status == FormatStatus.NO_ANSWER_TAGS
    assert tagged.correctness == bare.correctness == 1.0


def test_verifier_exception_is_contained(monkeypatch):
    import miqgrpo.rewards as rewards

    def boom(*args, **kwargs):
        raise RuntimeError("verifier exploded")

    monkeypatch.setattr(rewards, "evaluate_answer", boom)
    result = rewards.score_completion(
        '<answer>{"ring_count": 2}</answer>', "single_count", COUNT_TARGET, None
    )
    assert result.correctness == 0.0
    assert result.status == ParseStatus.VERIFIER_ERROR


def test_unknown_task_type_is_reported_not_raised():
    result = score_completion('<answer>{"x": 1}</answer>', "elephant", COUNT_TARGET, None)
    assert result.status == ParseStatus.UNSUPPORTED_TASK
    assert result.correctness == 0.0


def test_missing_target_is_reported_not_raised():
    result = score_completion('<answer>{"ring_count": 2}</answer>', "single_count", None, None)
    assert result.correctness == 0.0
    assert result.status == ParseStatus.UNSUPPORTED_TASK


@pytest.mark.parametrize(
    "text",
    [
        '  <answer>{"ring_count": 2}</answer>  ',
        '<answer>\n{"ring_count": 2}\n</answer>',
        '```json\n<answer>{"ring_count": 2}</answer>\n```',
        'Let me think...\n<think>hmm</think>\n<answer>{"ring_count": 2}</answer>',
        'blah blah <answer>{"ring_count": 2}</answer> trailing prose',
        '<answer>{"ring_count" : 2}</answer>',
        '<answer>{"ring_count": 2.0}</answer>',
    ],
)
def test_correct_answer_survives_surrounding_noise(text):
    assert correctness(text, "single_count", COUNT_TARGET) == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "\x00\x01\x02",
        "<answer>" * 500,
        '<answer>{"ring_count": ' + "9" * 5000 + "}</answer>",
        "𝕬𝖓𝖘𝖜𝖊𝖗: two",
        '<answer>{"ring_count": 2}',
        "</answer>{\"ring_count\": 2}<answer>",
    ],
)
def test_hostile_input_produces_one_finite_reward(text):
    result = score_completion(text, "single_count", COUNT_TARGET, None)
    assert result.correctness in (0.0, 1.0)


def _kwargs(n, task_type, target=None, constraints=None):
    return {
        "task_type": [task_type] * n,
        "target_json": [target] * n,
        "constraints_json": [constraints] * n,
    }


def test_reward_functions_return_one_value_per_completion():
    funcs = build_reward_functions(RewardConfig(partial_weight=1.0))
    completions = [
        [{"role": "assistant", "content": '<answer>{"ring_count": 2}</answer>'}],
        [{"role": "assistant", "content": "nonsense"}],
        [{"role": "assistant", "content": '<answer>{"ring_count": 3}</answer>'}],
    ]
    for func in funcs:
        values = func(completions=completions, **_kwargs(3, "single_count", COUNT_TARGET))
        assert len(values) == 3
        assert all(v is None or isinstance(v, float) for v in values)


def test_correctness_reward_matches_string_and_conversational_completions():
    funcs = build_reward_functions(RewardConfig())
    text = '<answer>{"ring_count": 2}</answer>'
    as_string = funcs[0](completions=[text], **_kwargs(1, "single_count", COUNT_TARGET))
    as_messages = funcs[0](
        completions=[[{"role": "assistant", "content": text}]],
        **_kwargs(1, "single_count", COUNT_TARGET),
    )
    assert as_string == as_messages == [1.0]


def test_score_cache_does_not_leak_across_different_completions():
    """A cache keyed loosely could hand one completion another's reward."""
    funcs = build_reward_functions(RewardConfig())
    right = '<answer>{"ring_count": 2}</answer>'
    wrong = '<answer>{"ring_count": 8}</answer>'
    assert funcs[0](completions=[right], **_kwargs(1, "single_count", COUNT_TARGET)) == [1.0]
    assert funcs[0](completions=[wrong], **_kwargs(1, "single_count", COUNT_TARGET)) == [0.0]
    assert funcs[0](completions=[right], **_kwargs(1, "single_count", COUNT_TARGET)) == [1.0]


def test_log_metric_hook_is_called():
    funcs = build_reward_functions(RewardConfig())
    seen: dict[str, float] = {}
    funcs[0](
        completions=[[{"role": "assistant", "content": '<answer>{"ring_count": 2}</answer>'}]],
        log_metric=lambda name, value: seen.__setitem__(name, value),
        **_kwargs(1, "single_count", COUNT_TARGET),
    )
    assert "reward/correctness_mean" in seen
    assert "parse/failure_fraction" in seen


def _partial(config: RewardConfig | None = None):
    funcs = build_reward_functions(config or RewardConfig(partial_weight=1.0))
    return funcs[-1]


def test_partial_credit_is_off_unless_weighted():
    names = [type(f).__name__ for f in build_reward_functions(RewardConfig())]
    assert "PartialCreditReward" not in names


def test_partial_credit_scores_each_count_property():
    target = json.dumps({"ring_count": 2, "carbon_atom_count": 6, "halogen_atom_count": 1})
    partial = _partial()
    values = partial(
        completions=[
            '<answer>{"ring_count": 2, "carbon_atom_count": 6, "halogen_atom_count": 1}</answer>',
            '<answer>{"ring_count": 2, "carbon_atom_count": 5, "halogen_atom_count": 0}</answer>',
            "nonsense",
        ],
        **_kwargs(3, "multi_count", target),
    )
    assert values == [pytest.approx(1.0), pytest.approx(1 / 3), 0.0]


def test_a_default_answer_earns_no_partial_credit():
    """Properties whose true value is 0 are left out, so answering 0 pays nothing."""
    target = json.dumps({"ring_count": 2, "halogen_atom_count": 0})
    partial = _partial()
    values = partial(
        completions=['<answer>{"ring_count": 0, "halogen_atom_count": 0}</answer>'],
        **_kwargs(1, "multi_count", target),
    )
    assert values == [0.0]


def test_a_question_only_a_default_answers_gets_no_partial_term():
    target = json.dumps({"ring_count": 0})
    values = _partial()(
        completions=['<answer>{"ring_count": 0}</answer>'], **_kwargs(1, "single_count", target)
    )
    assert values == [None]


def test_index_partial_credit_is_atom_overlap():
    target = json.dumps({"ring_index": [0, 1, 2, 3]})
    values = _partial()(
        completions=[
            '<answer>{"ring_index": [3, 2, 1, 0]}</answer>',
            '<answer>{"ring_index": [0, 1]}</answer>',
            '<answer>{"ring_index": [0, 1, 2, 3, 4, 5, 6, 7]}</answer>',
            '<answer>{"ring_index": [9]}</answer>',
            '<answer>{"ring_index": "0, 1"}</answer>',
        ],
        **_kwargs(5, "single_index", target),
    )
    assert values == [pytest.approx(1.0), pytest.approx(0.5), pytest.approx(0.5), 0.0, 0.0]


def test_listing_every_atom_is_not_a_winning_strategy():
    """Over-inclusion is penalised by the union, so a guess-everything list stays low."""
    target = json.dumps({"halogen_atom_index": [7]})
    everything = json.dumps({"halogen_atom_index": list(range(20))})
    value = _partial()(
        completions=[f"<answer>{everything}</answer>"], **_kwargs(1, "single_index", target)
    )[0]
    assert value == pytest.approx(1 / 20)


def test_generation_partial_credit_counts_satisfied_constraints():
    constraints = json.dumps(
        [
            {"type": "aromatic_ring_count", "operator": "=", "value": 1},
            {"type": "halogen_atom_count", "operator": "=", "value": 1},
        ]
    )
    values = _partial()(
        completions=[
            '<answer>{"smiles": "Clc1ccccc1"}</answer>',
            '<answer>{"smiles": "c1ccccc1"}</answer>',
            '<answer>{"smiles": "CC(O)C"}</answer>',
        ],
        **_kwargs(3, "constraint_generation", None, constraints),
    )
    assert values == [pytest.approx(1.0), pytest.approx(0.5), 0.0]


def test_partial_credit_never_outranks_a_correct_answer():
    """Total reward of a fully correct answer beats any partially correct one."""
    target = json.dumps({"ring_count": 2, "carbon_atom_count": 6})
    config = RewardConfig(partial_weight=1.0)
    funcs = build_reward_functions(config)
    texts = [
        '<answer>{"ring_count": 2, "carbon_atom_count": 6}</answer>',
        '<answer>{"ring_count": 2, "carbon_atom_count": 5}</answer>',
    ]
    per_function = [f(completions=texts, **_kwargs(2, "multi_count", target)) for f in funcs]
    totals = [sum(v for v in values if v is not None) for values in zip(*per_function)]
    assert totals[0] > totals[1] > 0


def test_shortcut_diagnostics_are_logged():
    funcs = build_reward_functions(RewardConfig(partial_weight=1.0))
    seen: dict[str, float] = {}
    funcs[0](
        completions=['<answer>{"ring_count": 0}</answer>', '<answer>{"ring_count": 2}</answer>'],
        log_metric=lambda name, value: seen.__setitem__(name, value),
        **_kwargs(2, "single_count", COUNT_TARGET),
    )
    assert seen["shortcut/constant_answer_share"] == pytest.approx(0.5)
    assert seen["correctness/other_questions"] == pytest.approx(0.5)
    assert seen["partial/nontrivial_credit"] == pytest.approx(0.5)
    assert "correctness/trivial_questions" not in seen


def test_typical_diagnostics_are_logged_when_typical_values_are_known():
    config = RewardConfig(partial_weight=1.0, typical_values={"ring_count": 1})
    seen: dict[str, float] = {}
    build_reward_functions(config)[0](
        completions=['<answer>{"ring_count": 1}</answer>', '<answer>{"ring_count": 3}</answer>'],
        log_metric=lambda name, value: seen.__setitem__(name, value),
        **_kwargs(2, "single_count", json.dumps({"ring_count": 3})),
    )
    assert seen["shortcut/typical_answer_share"] == pytest.approx(0.5)
    assert seen["correctness/atypical_properties"] == pytest.approx(0.5)


def test_trivial_weight_gives_a_correct_none_some_credit():
    target = json.dumps({"ring_count": 2, "halogen_atom_count": 0})
    text = ['<answer>{"ring_count": 0, "halogen_atom_count": 0}</answer>']
    off = build_reward_functions(RewardConfig(partial_weight=1.0))[-1]
    on = build_reward_functions(RewardConfig(partial_weight=1.0, trivial_weight=0.25))[-1]
    assert off(completions=text, **_kwargs(1, "multi_count", target)) == [0.0]
    assert on(completions=text, **_kwargs(1, "multi_count", target)) == [pytest.approx(0.25 / 1.25)]


def test_trivial_weight_scores_an_all_none_question():
    target = json.dumps({"halogen_atom_index": []})
    on = build_reward_functions(RewardConfig(partial_weight=1.0, trivial_weight=0.25))[-1]
    values = on(
        completions=['<answer>{"halogen_atom_index": []}</answer>', '<answer>{"halogen_atom_index": [3]}</answer>'],
        **_kwargs(2, "single_index", target),
    )
    assert values == [pytest.approx(1.0), 0.0]


def _none_config(penalty=0.5):
    return RewardConfig(partial_weight=1.0, trivial_weight=1.0, none_penalty=penalty)


def test_a_wrong_none_costs_and_a_right_none_pays():
    target = json.dumps({"ring_count": 2, "halogen_atom_count": 0})
    partial = build_reward_functions(_none_config())[-1]
    values = partial(
        completions=[
            '<answer>{"ring_count": 2, "halogen_atom_count": 0}</answer>',
            '<answer>{"ring_count": 0, "halogen_atom_count": 0}</answer>',
            '<answer>{"ring_count": 3, "halogen_atom_count": 1}</answer>',
        ],
        **_kwargs(3, "multi_count", target),
    )
    assert values == [pytest.approx(1.0), pytest.approx((-0.5 + 1.0) / 2), pytest.approx(0.0)]


def test_guessing_none_everywhere_loses_to_trying():
    target = json.dumps({"ring_index": [1, 2], "halogen_atom_index": [5]})
    partial = build_reward_functions(_none_config())[-1]
    none, attempt = partial(
        completions=[
            '<answer>{"ring_index": [], "halogen_atom_index": []}</answer>',
            '<answer>{"ring_index": [1, 7], "halogen_atom_index": [3]}</answer>',
        ],
        **_kwargs(2, "multi_index", target),
    )
    assert none == pytest.approx(-0.5) and attempt > none


def test_none_penalty_zero_changes_nothing():
    target = json.dumps({"ring_count": 2})
    text = ['<answer>{"ring_count": 0}</answer>']
    plain = build_reward_functions(RewardConfig(partial_weight=1.0))[-1]
    zero = build_reward_functions(RewardConfig(partial_weight=1.0, none_penalty=0.0))[-1]
    assert plain(completions=text, **_kwargs(1, "single_count", target)) == \
        zero(completions=text, **_kwargs(1, "single_count", target)) == [0.0]


def test_none_diagnostics_are_logged():
    seen: dict[str, float] = {}
    build_reward_functions(_none_config())[0](
        completions=['<answer>{"ring_count": 0}</answer>', '<answer>{"ring_count": 2}</answer>'],
        log_metric=lambda name, value: seen.__setitem__(name, value),
        **_kwargs(2, "single_count", json.dumps({"ring_count": 2})),
    )
    assert seen["none/answer_rate"] == pytest.approx(0.5)
    assert seen["none/precision"] == pytest.approx(0.0)
