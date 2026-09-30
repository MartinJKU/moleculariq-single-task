#!/usr/bin/env bash
# Dev-split breakdowns for the report
#   bash scripts/25_dev_evaluate.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
RUNS_ROOT="${MIQ_RUNS:-$PROJECT_DIR/runs}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

JOBS="$(mktemp)"
trap 'rm -f "$JOBS"' EXIT
python - > "$JOBS" <<'PY'
import yaml
spec = yaml.safe_load(open("configs/report.yaml"))
for model in spec["models"]:
    dev = model["dev"]
    print(dev["base"], "Qwen/Qwen2.5-0.5B-Instruct", dev["split"], spec["dataset"])
    print(dev["model"], "RUN:" + model["training"], dev["split"], spec["dataset"])
PY

while read -r name source split artifact; do
  out="${MIQ_RESULTS:-$PROJECT_DIR/results}/dev/$artifact/$name.json"
  if [ -f "$out" ]; then echo "exists: $out"; continue; fi
  model="$source"
  [[ "$source" == RUN:* ]] && model="$RUNS_ROOT/${source#RUN:}/final"
  python -m miqgrpo.dev_eval --model "$model" --artifact "$artifact" --split "$split" \
    --samples 8 --out "$out"
done < "$JOBS"
