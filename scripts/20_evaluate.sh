#!/usr/bin/env bash
# The whole official benchmark for the base model and the three models
#   bash scripts/20_evaluate.sh                    # everything in configs/evaluation.yaml
#   bash scripts/20_evaluate.sh --only base,count
#   bash scripts/20_evaluate.sh --dry-run          # print the lm_eval commands

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# $MIQ_RUNS must be exported, or checkpoints are looked for inside the repository.
echo "checkpoints : ${MIQ_RUNS:-$PROJECT_DIR/runs}"
echo "results     : ${MIQ_RESULTS:-$PROJECT_DIR/results}/moleculariq"
echo

python -m miqgrpo.evaluate all --config configs/evaluation.yaml "$@"

echo
echo "next: bash scripts/25_dev_evaluate.sh, then bash scripts/30_report.sh"
