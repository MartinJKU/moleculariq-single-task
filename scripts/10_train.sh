#!/usr/bin/env bash
# Train the three models one after another
#   bash scripts/10_train.sh                  # count, index, constraint_generation
#   bash scripts/10_train.sh count            # one model
# Resume an interrupted run: python -m miqgrpo.train_grpo train --config configs/experiments/<name>.yaml --resume

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
RUNS_ROOT="${MIQ_RUNS:-$PROJECT_DIR/runs}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(count index constraint_generation)

echo "checkpoints : $RUNS_ROOT"
for name in "${MODELS[@]}"; do
  config="configs/experiments/${name}.yaml"
  [ -f "$config" ] || { echo "no config $config" >&2; exit 1; }
  out="$RUNS_ROOT/$name"
  echo
  echo "=============================================================="
  echo " $name"
  echo "=============================================================="
  if [ -f "$out/training_summary.json" ]; then
    echo "  already trained: $out/final"
    continue
  fi
  if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
    echo "  $out exists but training did not finish." >&2
    echo "  Resume it:  python -m miqgrpo.train_grpo train --config $config --resume" >&2
    exit 1
  fi
  python -m miqgrpo.train_grpo preflight --config "$config"
  python -m miqgrpo.train_grpo train --config "$config"
done

echo
echo "checkpoints:"
for name in "${MODELS[@]}"; do
  echo "  $RUNS_ROOT/${name}/final"
done
echo "next: bash scripts/20_evaluate.sh"
