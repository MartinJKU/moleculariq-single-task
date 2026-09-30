#!/usr/bin/env bash
# Pack the report, metrics and manifests into one archive and print how to download it.
#   bash scripts/40_bundle_results.sh              # report, metrics, manifests
#   bash scripts/40_bundle_results.sh --with-raw   # + per-sample logs (large)
# This was easier for me as i had to get the results via scp all the time
# and this script packaged the results for me and gave me the copy command.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${MIQ_DATA:-$PROJECT_DIR/data}"
RUNS="${MIQ_RUNS:-$PROJECT_DIR/runs}"
RESULTS="${MIQ_RESULTS:-$PROJECT_DIR/results}"
OUT_DIR="${OUT_DIR:-$(dirname "$RESULTS")}"

WITH_RAW=0
[ "${1:-}" = "--with-raw" ] && WITH_RAW=1

echo "reading from:"
echo "  data     $DATA"
echo "  runs     $RUNS"
echo "  results  $RESULTS"
if [ -z "${MIQ_RESULTS:-}" ]; then
  echo
  echo "  ! MIQ_RESULTS is not set in this environment; the paths above are"
  echo "    repo-relative defaults and probably not where your results are."
  echo "    Run 'source ~/.bashrc' or pass them inline."
fi
echo

STAGE="$OUT_DIR/miq-bundle"
rm -rf "$STAGE"
mkdir -p "$STAGE/evaluations" "$STAGE/training" "$STAGE/dataset"

copied=0
missing=0
take () {
  if [ -e "$1" ]; then
    cp -r "$1" "$2" && copied=$((copied + 1))
  else
    echo "  missing: $1"
    missing=$((missing + 1))
  fi
}

cd "$PROJECT_DIR"
EVALUATIONS=$(python3 -c "import yaml;print(' '.join(r['run_id'] for r in yaml.safe_load(open('configs/evaluation.yaml'))['runs']))")
DATASET=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/dataset.yaml'))['dataset_artifact_id'])")

echo "collecting:"
take "$RESULTS/report" "$STAGE/"
[ -d "$RESULTS/dev/$DATASET" ] && mkdir -p "$STAGE/dev" && take "$RESULTS/dev/$DATASET" "$STAGE/dev/"

for name in $EVALUATIONS; do
  d="$RESULTS/moleculariq/$name/"
  [ -d "$d" ] || { echo "  missing: $d"; missing=$((missing + 1)); continue; }
  mkdir -p "$STAGE/evaluations/$name"
  for f in summary.json eval_manifest.json eval_manifest.yaml provenance.json environment.txt; do
    [ -e "$d/$f" ] && cp "$d/$f" "$STAGE/evaluations/$name/" && copied=$((copied + 1))
  done
  if [ "$WITH_RAW" = "1" ]; then
    newest=$(ls -1 "$d"raw/**/samples_*.jsonl "$d"raw/samples_*.jsonl 2>/dev/null | sort | tail -1 || true)
    if [ -n "$newest" ]; then
      cp "$newest" "$STAGE/evaluations/$name/" && copied=$((copied + 1))
    fi
    newest_results=$(ls -1 "$d"raw/**/results_*.json "$d"raw/results_*.json 2>/dev/null | sort | tail -1 || true)
    [ -n "$newest_results" ] && cp "$newest_results" "$STAGE/evaluations/$name/" && copied=$((copied + 1))
  fi
done

for config in configs/experiments/*.yaml; do
  name=$(basename "$config" .yaml)
  d="$RUNS/$name/"
  [ -d "$d" ] || { echo "  missing: $d"; missing=$((missing + 1)); continue; }
  mkdir -p "$STAGE/training/$name"
  for f in frozen_config.yaml provenance.json training_summary.json; do
    [ -e "$d/$f" ] && cp "$d/$f" "$STAGE/training/$name/" && copied=$((copied + 1))
  done
  [ -e "$d/logs/metrics.jsonl" ] && cp "$d/logs/metrics.jsonl" "$STAGE/training/$name/" && copied=$((copied + 1))
  if [ -d "$d/checkpoints/completions" ]; then
    mkdir -p "$STAGE/training/$name/completions"
    cp "$d"/checkpoints/completions/*.parquet "$STAGE/training/$name/completions/" && copied=$((copied + 1))
  fi
done

artifact="$DATA/processed/$DATASET"
mkdir -p "$STAGE/dataset/$DATASET"
for f in manifest.json preprocessing_config.yaml provenance.json; do
  [ -e "$artifact/$f" ] && cp "$artifact/$f" "$STAGE/dataset/$DATASET/" && copied=$((copied + 1))
done

[ -e "$PROJECT_DIR/assets_manifest.json" ] && cp "$PROJECT_DIR/assets_manifest.json" "$STAGE/" && copied=$((copied + 1))

{
  echo "commit: $(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch: $(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "dirty:  $(git -C "$PROJECT_DIR" status --porcelain 2>/dev/null | wc -l) modified files"
  echo
  git -C "$PROJECT_DIR" status --porcelain 2>/dev/null || true
  echo
  git -C "$PROJECT_DIR" diff HEAD 2>/dev/null || true
} > "$STAGE/code_revision.txt"

ARCHIVE="$OUT_DIR/miq-report-bundle.tgz"
rm -f "$ARCHIVE"
tar czf "$ARCHIVE" -C "$OUT_DIR" miq-bundle

echo
echo "=============================================================="
echo "  files collected : $copied"
[ "$missing" -gt 0 ] && echo "  missing         : $missing (listed above)"
echo "  archive         : $ARCHIVE"
echo "  size            : $(du -h "$ARCHIVE" | cut -f1)"

if command -v sha256sum >/dev/null 2>&1; then
  echo "  sha256          : $(sha256sum "$ARCHIVE" | cut -d' ' -f1)"
else
  echo "  sha256          : $(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)"
fi
echo "=============================================================="
echo
echo
echo "  ssh -T <your-ssh-target> \"cat $ARCHIVE\" > ~/Downloads/$(basename "$ARCHIVE")"
echo
