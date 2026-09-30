"""Training renders exactly the prompt the benchmark evaluates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from miqgrpo.prompts import SYSTEM_PROMPT, build_prompt_messages, completion_to_text
from miqgrpo.vendor import extract_moleculariq_answer

VENDOR_DIR = Path(__file__).resolve().parents[1] / "src" / "miqgrpo" / "vendor"


def _strip_trailing_spaces(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


def test_system_prompt_matches_the_core_library_modulo_trailing_space():
    """The vendored prompt vs the installed core library's 'concise' style.

    The two official sources differ by a single trailing space (core's
    ``SYSTEM_PROMPTS["concise"]`` writes ``"Examples: "``, the eval repo's
    ``task_processor.SYSTEM_PROMPT`` writes ``"Examples:"``). The eval repo's
    copy is the authoritative one because that is what the benchmark actually
    passes to the model, and it is what we vendored -- but the wording must not
    drift beyond that whitespace, or training and evaluation stop describing the
    same task.
    """
    from moleculariq_core import SYSTEM_PROMPTS

    assert _strip_trailing_spaces(SYSTEM_PROMPT) == _strip_trailing_spaces(
        SYSTEM_PROMPTS["concise"]
    )


def test_system_prompt_is_vendored_verbatim_from_the_eval_repo():
    """Byte-exact against the copy recorded at vendoring time."""
    import hashlib

    digest = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    manifest = json.loads((VENDOR_DIR / "VENDOR.json").read_text())
    expected = manifest["moleculariq_system_prompt.py"]["sha256"]
    assert digest == expected, (
        "the vendored system prompt changed; re-vendor it from "
        "moleculariq-eval and update VENDOR.json deliberately"
    )


def test_vendor_manifest_records_provenance():
    manifest = json.loads((VENDOR_DIR / "VENDOR.json").read_text())
    for name in ("moleculariq_extractors.py", "moleculariq_system_prompt.py"):
        entry = manifest[name]
        assert entry["repo"].startswith("https://github.com/ml-jku/")
        assert len(entry["commit"]) == 40


def test_prompt_is_exactly_two_turns():
    messages = build_prompt_messages("How many rings are in CCO?")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == "How many rings are in CCO?"


def test_prompt_rejects_empty_question():
    with pytest.raises(ValueError):
        build_prompt_messages("   ")


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("plain string", "plain string"),
        ([{"role": "assistant", "content": "abc"}], "abc"),
        ({"role": "assistant", "content": "abc"}, "abc"),
        (None, ""),
        ([], ""),
    ],
)
def test_completion_normalisation(completion, expected):
    assert completion_to_text(completion) == expected


def test_vendored_extractor_handles_the_official_shapes():
    assert extract_moleculariq_answer('<answer>{"ring_count": 2}</answer>') == '{"ring_count": 2}'
    assert extract_moleculariq_answer('<think>x</think><answer>ok</answer>') == "ok"
    assert extract_moleculariq_answer('```json\n{"ring_count": 2}\n```') == '{"ring_count": 2}'
    assert extract_moleculariq_answer("") is None


def test_rendered_chat_prompt_is_stable():
    """Regression-pin the exact rendered string for one fixed example."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct"
        )
    except Exception as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")

    messages = build_prompt_messages("How many rings are in CCO?")
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert rendered.startswith("<|im_start|>system\n")
    assert SYSTEM_PROMPT in rendered
    assert rendered.rstrip().endswith("<|im_start|>assistant")
    assert "<|im_start|>user\nHow many rings are in CCO?<|im_end|>" in rendered
