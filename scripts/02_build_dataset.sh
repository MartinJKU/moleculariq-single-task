#!/usr/bin/env bash
# Build the frozen training set

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${1:-$PROJECT_DIR/configs/dataset.yaml}"
ARTIFACT_ID="$(python3 -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['dataset_artifact_id'])" "$CONFIG")"

cd "$PROJECT_DIR"
python -m miqgrpo.build_dataset build --config "$CONFIG"
python -m miqgrpo.build_dataset verify --artifact "$ARTIFACT_ID"

echo
echo "dataset '$ARTIFACT_ID' is frozen. Training checks its content hash against"
echo "the one pinned in each experiment config before it starts."
