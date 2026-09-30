"""Score a model on the held-out dev split of a frozen training dataset.

    python -m miqgrpo.dev_eval --model runs/<experiment>/final \\
        --artifact miq-train --split count_dev --samples 8 --out results/dev/miq-train/<name>.json

The dev split is the part of the *training* distribution held out from
training, so this is the place to compare checkpoints and settings -- the
official benchmark stays untouched. Generation uses vLLM with the benchmark's
sampling (temperature 1.0, no top-k/top-p, no repetition penalty) and the same
chat prompt; answers go through the official extractor and scorer.

Besides accuracy it reports where accuracy comes from (see miqgrpo.breakdown):
questions a constant answer solves vs the rest, and -- for count and index --
properties whose true value is the most common value of that property in the
*training* split ("typical") vs the rest. A policy can raise accuracy on
typical values by learning a per-property default; only accuracy on atypical
values shows it reading the molecule.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .breakdown import (
    constant_share,
    is_trivial,
    question_from_row,
    score_units,
    unit_credit,
    units,
)
from .paths import dataset_dir, ensure_dirs
from .prompts import build_prompt_messages
from .rewards import score_completion

__all__ = ["main", "typical_values", "summarise"]


def typical_values(rows: Any) -> dict[str, Any]:
    """Most common true value of each count/index property in a split."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if not row.get("target_json") or row["target_json"] == "None":
            continue
        for key, value in json.loads(row["target_json"]).items():
            counts[key][json.dumps(value, sort_keys=True)] += 1
    return {key: json.loads(c.most_common(1)[0][0]) for key, c in counts.items()}


def summarise(
    rows: list[dict[str, Any]], completions: list[list[str]], typical: dict[str, Any]
) -> dict[str, Any]:
    """Accuracy and its breakdown for one model on one split.

    completions[i] are the raw sampled completions for rows[i]; each goes
    through the official extractor and scorer exactly as a training reward does.
    """
    item: dict[str, list[float]] = defaultdict(list)
    unit: dict[str, list[float]] = defaultdict(list)
    by_property: dict[str, list[float]] = defaultdict(list)
    credit_by_property: dict[str, list[float]] = defaultdict(list)
    answer_size: dict[str, list[float]] = defaultdict(list)
    shares: list[float] = []
    typical_said: list[float] = []
    for row, texts in zip(rows, completions):
        question = question_from_row(row)
        scored = [
            score_completion(text, row["task_type"], row["target_json"], row["constraints_json"])
            for text in texts
        ]
        attempts = [s.answer for s in scored]
        accuracy = sum(s.correctness for s in scored) / len(scored)
        item["all"].append(accuracy)
        item["trivial" if is_trivial(question) else "other"].append(accuracy)
        shares.append(constant_share(question, attempts))
        per_attempt = [dict(score_units(question, a)) for a in attempts]
        per_credit = [dict(unit_credit(question, a)) for a in attempts]
        parsed = []
        for a in attempts:
            try:
                value = json.loads(a) if a else None
            except (json.JSONDecodeError, TypeError):
                value = None
            parsed.append(value if isinstance(value, dict) else {})
        for u in units(question):
            if u.trivial:
                continue
            score = sum(p[u.key] for p in per_attempt) / len(per_attempt)
            credit = sum(c[u.key] for c in per_credit) / len(per_credit)
            unit["nontrivial"].append(score)
            unit["nontrivial_credit"].append(credit)
            by_property[u.label].append(score)
            credit_by_property[u.label].append(credit)
            if question.task == "index":
                # Predicted vs true list length: overlap credit can be inflated by long lists.
                truth = max(1, len(question.target_dict[u.key]))
                sizes = [len(p.get(u.key)) / truth for p in parsed if isinstance(p.get(u.key), list)]
                if sizes:
                    answer_size[u.label].append(sum(sizes) / len(sizes))
            if question.task in ("count", "index") and u.key in typical:
                is_typical = question.target_dict[u.key] == typical[u.key]
                unit["typical" if is_typical else "atypical"].append(score)
                typical_said.append(
                    sum(p.get(u.key) == typical[u.key] for p in parsed) / len(parsed)
                )

    def mean(values: list[float]) -> dict[str, float]:
        return {"mean": sum(values) / len(values) if values else float("nan"), "n": len(values)}

    return {
        "questions": {k: mean(v) for k, v in item.items()},
        "properties": {k: mean(v) for k, v in unit.items()},
        "constant_answer_share": mean(shares),
        "typical_answer_share": mean(typical_said),
        "by_property": {k: mean(v) for k, v in sorted(by_property.items())},
        "credit_by_property": {k: mean(v) for k, v in sorted(credit_by_property.items())},
        "answer_size_ratio_by_property": {k: mean(v) for k, v in sorted(answer_size.items())},
    }


def main(argv: list[str] | None = None) -> None:
    """Command line: sample answers for every question of one dev split and write the breakdown.

    The output JSON holds the summarise result plus the model, split and
    sampling settings it was produced with.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF model id or checkpoint directory")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--split", required=True, help="e.g. count_dev")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from datasets import load_from_disk
    from vllm import LLM, SamplingParams

    dataset = load_from_disk(str(dataset_dir(args.artifact) / "dataset"))
    rows = [dict(r) for r in dataset[args.split]]
    train_split = args.split.removesuffix("_dev")
    typical = typical_values(dataset[train_split]) if train_split in ("count", "index") else {}

    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True)
    params = SamplingParams(n=args.samples, temperature=1.0, top_p=1.0, top_k=0,
                            repetition_penalty=1.0, max_tokens=args.max_tokens, seed=args.seed)
    messages = [build_prompt_messages(r["question"]) for r in rows]
    outputs = llm.chat(messages, params, use_tqdm=True)
    completions = [[o.text for o in out.outputs] for out in outputs]

    result = {
        "model": args.model,
        "artifact": args.artifact,
        "split": args.split,
        "samples": args.samples,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        **summarise(rows, completions, typical),
    }
    ensure_dirs(args.out.parent)
    args.out.write_text(json.dumps(result, indent=2))
    q, p = result["questions"], result["properties"]
    print(f"{args.model} on {args.split}: all {q['all']['mean']:.1%}  "
          f"other {q.get('other', {}).get('mean', float('nan')):.1%}  "
          f"trivial {q.get('trivial', {}).get('mean', float('nan')):.1%}  "
          f"non-trivial properties {p['nontrivial']['mean']:.1%} (credit {p['nontrivial_credit']['mean']:.1%})  "
          f"typical {p.get('typical', {}).get('mean', float('nan')):.1%}  "
          f"atypical {p.get('atypical', {}).get('mean', float('nan')):.1%}  "
          f"constant answers {result['constant_answer_share']['mean']:.1%}  "
          f"typical answers {result['typical_answer_share']['mean']:.1%}")


if __name__ == "__main__":
    main()
