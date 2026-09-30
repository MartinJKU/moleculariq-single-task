#!/usr/bin/env bash
# The whole pipeline, dataset to report; Run scripts 00 and 01 first.
#   bash scripts/run_all.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

ARTIFACT="${MIQ_DATA:-$PROJECT_DIR/data}/processed/$(python3 -c "import yaml;print(yaml.safe_load(open('configs/dataset.yaml'))['dataset_artifact_id'])")"
if [ -f "$ARTIFACT/manifest.json" ]; then
  echo "dataset already built: $ARTIFACT"
else
  bash scripts/02_build_dataset.sh configs/dataset.yaml
fi
bash scripts/10_train.sh
bash scripts/20_evaluate.sh
bash scripts/25_dev_evaluate.sh
bash scripts/30_report.sh
