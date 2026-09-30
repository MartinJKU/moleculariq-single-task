#!/usr/bin/env bash
# Follow a training run's key numbers (reward, correctness, constant-answer share)
#   bash scripts/watch_run.sh <experiment-id>

set -euo pipefail
RUNS="${MIQ_RUNS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/runs}"
LOG="$RUNS/${1:?usage: watch_run.sh <experiment-id>}/logs/metrics.jsonl"
until [ -f "$LOG" ]; do echo "waiting for $LOG"; sleep 10; done

read -r -d '' FORMAT <<'PY' || true
import json, sys

def fmt(d, key, pct=True):
    v = d.get(key)
    if not isinstance(v, (int, float)):
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"

print(f"{'':5s} {'step':>5s} {'reward':>7s} {'other':>7s} {'partial':>7s} {'const':>7s} {'trivial':>7s}")
for line in sys.stdin:
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    for prefix, tag in (("", "TRAIN"), ("eval_", "EVAL")):
        if prefix + "reward" not in d:
            continue
        cells = [
            fmt(d, prefix + "reward", pct=False),
            fmt(d, prefix + "correctness/other_questions"),
            fmt(d, prefix + "partial/nontrivial_credit"),
            fmt(d, prefix + "shortcut/constant_answer_share"),
            fmt(d, prefix + "correctness/trivial_questions"),
        ]
        print(f"{tag:5s} {d.get('step', 0):5d} " + " ".join(f"{c:>7s}" for c in cells))
PY

tail -n +1 -F "$LOG" | python3 -u -c "$FORMAT"
