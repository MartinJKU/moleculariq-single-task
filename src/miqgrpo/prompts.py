"""Model-visible prompt construction.

The official benchmark renders a MolecularIQ item as

    system : SYSTEM_PROMPT            (passed to lm_eval as --system_instruction)
    user   : doc["question"]          (verbatim, no wrapper text)

and applies the model's own chat template (--apply_chat_template). Training
renders exactly the same two turns, so the policy is optimised under the prompt
it is evaluated under. Any divergence here silently costs benchmark points, so
the equality is asserted in tests/test_prompts.py.
"""

from __future__ import annotations

from typing import Any, Iterable

from .vendor import SYSTEM_PROMPT

__all__ = [
    "SYSTEM_PROMPT",
    "build_prompt_messages",
    "render_question",
    "completion_to_text",
]


def build_prompt_messages(question: str) -> list[dict[str, str]]:
    """Return the conversational prompt column for one training example.

    TRL detects this list-of-messages shape as a conversational dataset and
    applies the tokenizer's chat template itself, which is what lm-eval does
    under --apply_chat_template.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def render_question(messages: Iterable[dict[str, str]]) -> str:
    """Inverse of build_prompt_messages -- the user turn only."""
    for message in messages:
        if message.get("role") == "user":
            return message["content"]
    raise ValueError("no user turn in prompt")


def completion_to_text(completion: Any) -> str:
    """Normalise one TRL completion into the raw string the verifier sees.

    With a conversational dataset TRL hands back [{"role": "assistant",
    "content": "..."}]; with a standard dataset it hands back a plain string.
    Both shapes occur depending on config, so reward code must not assume one.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        parts = []
        for message in completion:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(message))
        return "".join(parts)
    return "" if completion is None else str(completion)
