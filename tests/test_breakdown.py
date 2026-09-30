"""Per-property scoring and the constant-answer reference agree with the official scorer."""

from __future__ import annotations

import json

import pytest
from moleculariq_core import evaluate_answer

from miqgrpo.breakdown import (
    CONSTANT_SMILES,
    Question,
    Scored,
    constant_answer,
    constant_share,
    generated_smiles,
    is_trivial,
    paired_split,
    property_changes,
    question_from_doc,
    question_from_row,
    score_units,
    solved_by_any,
    split_accuracy,
    unit_accuracy,
    units,
)
from miqgrpo.prompts import SYSTEM_PROMPT


def count_question(target: dict) -> Question:
    return question_from_doc({"task_type": "count", "target": json.dumps(target)})


def generation_question(constraints: list, key: str = "property") -> Question:
    renamed = [{key if k == "property" else k: v for k, v in c.items()} for c in constraints]
    return question_from_doc({"task_type": "generation", "constraints": renamed})


def test_constant_molecule_is_the_system_prompts_example():
    assert CONSTANT_SMILES == "CC(O)C"
    assert f'"smiles": "{CONSTANT_SMILES}"' in SYSTEM_PROMPT


def test_constant_answers_ignore_the_molecule():
    assert json.loads(constant_answer(count_question({"a_count": 3, "b_count": 0}))) == {
        "a_count": 0,
        "b_count": 0,
    }
    index = question_from_doc({"task_type": "index", "target": '{"ring_index": [1, 2]}'})
    assert json.loads(constant_answer(index)) == {"ring_index": []}
    generation = generation_question([{"property": "ring_count", "operator": "=", "value": 1}])
    assert json.loads(constant_answer(generation)) == {"smiles": CONSTANT_SMILES}


def test_a_unit_is_trivial_exactly_when_the_constant_answer_gets_it():
    question = count_question({"ring_count": 0, "carbon_atom_count": 5})
    trivial = {unit.label: unit.trivial for unit in units(question)}
    assert trivial == {"ring_count": True, "carbon_atom_count": False}
    assert not is_trivial(question)
    assert is_trivial(count_question({"ring_count": 0, "halogen_atom_count": 0}))


def test_generation_triviality_uses_the_official_verifier():
    satisfied = {"property": "ring_count", "operator": "=", "value": 0}
    unsatisfied = {"property": "ring_count", "operator": "=", "value": 2}
    assert is_trivial(generation_question([satisfied]))
    assert not is_trivial(generation_question([unsatisfied]))
    mixed = units(generation_question([satisfied, unsatisfied]))
    assert [unit.trivial for unit in mixed] == [True, False]


def test_default_molecules_widen_the_constant_answer():
    two_carbons = generation_question([{"property": "carbon_atom_count", "operator": "=", "value": 2}])
    assert not is_trivial(two_carbons)
    assert solved_by_any(two_carbons, ["CC(O)C", "CCO"])
    assert not solved_by_any(two_carbons, ["CC(O)C", "CCC"])
    one_ring = generation_question([{"property": "ring_count", "operator": "=", "value": 1}])
    assert not solved_by_any(one_ring, ["CC(O)C", "CCO", "CCC"])


def test_generated_smiles_reads_extracted_answers():
    answers = ['{"smiles": "CCO"}', {"smiles": "CCC"}, "not json", '{"other": 1}', None]
    assert generated_smiles(answers) == ["CCO", "CCC", None, None, None]


def test_training_constraints_named_type_are_handled():
    row = {
        "task_family": "constraint_generation",
        "constraints_json": json.dumps([{"type": "ring_count", "operator": "=", "value": 0}]),
        "target_json": None,
    }
    question = question_from_row(row)
    assert [unit.label for unit in units(question)] == ["ring_count"]
    assert is_trivial(question)


@pytest.mark.parametrize(
    "answer",
    [
        '{"ring_count": 2, "carbon_atom_count": 5}',
        '{"ring_count": 2, "carbon_atom_count": 4}',
        '{"ring_count": 1, "carbon_atom_count": 4}',
        '{"ring_count": 2}',
        "not json",
    ],
)
def test_units_agree_with_the_official_item_score(answer):
    target = {"ring_count": 2, "carbon_atom_count": 5}
    question = count_question(target)
    item = all(correct for _, correct in score_units(question, answer))
    assert float(item) == float(evaluate_answer(task_type="count", predicted=answer, target=target))


def test_index_units_are_order_insensitive_like_the_official_scorer():
    question = question_from_doc({"task_type": "index", "target": '{"ring_index": [1, 2, 3]}'})
    assert dict(score_units(question, '{"ring_index": [3, 1, 2]}')) == {"ring_index": True}
    assert dict(score_units(question, '{"ring_index": [1, 2]}')) == {"ring_index": False}


def test_an_empty_answer_gets_every_unit_wrong():
    question = count_question({"ring_count": 0})
    assert dict(score_units(question, None)) == {"ring_count": False}
    assert unit_accuracy(question, []) == {"ring_count": 0.0}


def test_unit_accuracy_averages_attempts():
    question = count_question({"ring_count": 2, "carbon_atom_count": 5})
    accuracy = unit_accuracy(
        question,
        ['{"ring_count": 2, "carbon_atom_count": 5}', '{"ring_count": 2, "carbon_atom_count": 1}'],
    )
    assert accuracy == {"ring_count": 1.0, "carbon_atom_count": 0.5}


def test_constant_share_recognises_the_constant_up_to_equivalence():
    count = count_question({"ring_count": 2, "carbon_atom_count": 5})
    assert constant_share(count, ['{"carbon_atom_count": 0, "ring_count": 0}', '{"ring_count": 0}']) == 0.5
    generation = generation_question([{"property": "ring_count", "operator": "=", "value": 1}])
    assert constant_share(generation, ['{"smiles": "CC(O)C"}', '{"smiles": "c1ccccc1"}']) == 0.5


def _scored(key, question, accuracy, units_correct: dict[str, float]) -> Scored:
    return Scored(key=key, question=question, accuracy=accuracy,
                  unit_accuracy=units_correct, constant_share=0.0)


def test_split_and_paired_differences_separate_trivial_questions():
    trivial = count_question({"ring_count": 0})
    real = count_question({"ring_count": 3})
    base = [_scored(0, trivial, 0.0, {"ring_count": 0.0}), _scored(1, real, 0.0, {"ring_count": 0.0})]
    model = [_scored(0, trivial, 1.0, {"ring_count": 1.0}), _scored(1, real, 0.0, {"ring_count": 0.0})]
    split = split_accuracy(model, "count")
    assert split["trivial"] == (1.0, 1)
    assert split["other"] == (0.0, 1)
    assert split["other_units"] == (0.0, 1)
    assert paired_split(model, base, "count", trivial=True) == [1.0]
    assert paired_split(model, base, "count", trivial=False) == [0.0]


def test_property_changes_use_non_trivial_units_only():
    base, model = [], []
    for key in range(50):
        question = count_question({"ring_count": key % 2 + 1, "halogen_atom_count": 0})
        base.append(_scored(key, question, 0.0, {"ring_count": 0.0, "halogen_atom_count": 0.0}))
        model.append(_scored(key, question, 0.0, {"ring_count": 1.0, "halogen_atom_count": 1.0}))
    changes = property_changes(model, base, "count", min_units=40)
    assert set(changes) == {"ring_count"}
    base_accuracy, diffs = changes["ring_count"]
    assert base_accuracy == 0.0 and len(diffs) == 50 and set(diffs) == {1.0}


def test_dev_eval_summary_separates_typical_and_atypical_values():
    from miqgrpo.dev_eval import summarise, typical_values

    train = [{"target_json": json.dumps({"ring_count": 1})}] * 3 + [{"target_json": json.dumps({"ring_count": 2})}]
    typical = typical_values(train)
    assert typical == {"ring_count": 1}
    rows = [
        {"task_family": "count", "task_type": "single_count", "target_json": json.dumps({"ring_count": 1}), "constraints_json": None},
        {"task_family": "count", "task_type": "single_count", "target_json": json.dumps({"ring_count": 3}), "constraints_json": None},
    ]
    completions = [
        ['<answer>{"ring_count": 1}</answer>', '<answer>{"ring_count": 1}</answer>'],
        ['<answer>{"ring_count": 1}</answer>', '<answer>{"ring_count": 3}</answer>'],
    ]
    summary = summarise(rows, completions, typical)
    assert summary["properties"]["typical"]["mean"] == 1.0
    assert summary["properties"]["atypical"]["mean"] == 0.5
    assert summary["typical_answer_share"]["mean"] == 0.75
    assert summary["questions"]["other"]["mean"] == 0.75
