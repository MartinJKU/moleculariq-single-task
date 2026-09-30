#!/usr/bin/env bash
# Build the report -> needs scripts 20 and 25.
#   bash scripts/30_report.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

python -m miqgrpo.report --spec configs/report.yaml "$@"
