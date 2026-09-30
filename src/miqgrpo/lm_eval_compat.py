"""The official lm_eval command line, with two compatibility shims for vLLM 0.30.

    python -m miqgrpo.lm_eval_compat <lm_eval arguments>

moleculariq-eval's vLLM backend was written against an older vLLM. Two
things changed in vLLM 0.30, and this module bridges exactly those before
running the unmodified official CLI:

* vllm.entrypoints.chat_utils.resolve_hf_chat_template moved to
  vllm.renderers.hf.resolve_chat_template with the same arguments. The
  backend uses it only to look up the chat template -- for Qwen2.5 the
  tokenizer's own template, the one the transformers backend applies -- so the
  prompt text is unchanged.
* LLM(swap_space=...) was removed. It sized the CPU swap area used when the
  KV cache runs out of GPU memory; it never affected what is generated, so the
  argument is dropped.

Nothing in task construction, sampling, extraction or scoring is touched.
"""

from __future__ import annotations

import sys

__all__ = ["install_aliases", "main"]


def install_aliases() -> list[str]:
    """Apply the shims this vLLM needs; returns the names of those applied."""
    try:
        import vllm
        import vllm.entrypoints.chat_utils as chat_utils
    except ImportError:
        return []
    applied = []
    if not hasattr(chat_utils, "resolve_hf_chat_template"):
        from vllm.renderers.hf import resolve_chat_template

        chat_utils.resolve_hf_chat_template = resolve_chat_template
        applied.append("resolve_hf_chat_template")

    import inspect

    from vllm.engine.arg_utils import EngineArgs

    if "swap_space" not in inspect.signature(EngineArgs).parameters:
        base = vllm.LLM

        class LLM(base):  # type: ignore[misc, valid-type]
            def __init__(self, *args, swap_space=None, **kwargs):  # noqa: ARG002
                super().__init__(*args, **kwargs)

        vllm.LLM = LLM
        applied.append("LLM(swap_space)")
    return applied


def main() -> None:
    """Start the official lm_eval CLI, applying the vLLM shims first when the vLLM backend is used."""
    if "vllm" in sys.argv:
        applied = install_aliases()
        print(f"miqgrpo.lm_eval_compat: applied {applied or 'nothing'}", file=sys.stderr)
    from lm_eval.__main__ import cli_evaluate

    sys.argv[0] = "lm_eval"
    sys.exit(cli_evaluate())


if __name__ == "__main__":
    main()
