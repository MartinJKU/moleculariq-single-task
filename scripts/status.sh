#!/usr/bin/env bash
# Show what is running, finished or interrupted. Changes nothing.
#   bash scripts/status.sh

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${MIQ_DATA:-$REPO/data}"
RUNS="${MIQ_RUNS:-$REPO/runs}"
RESULTS="${MIQ_RESULTS:-$REPO/results}"

echo "=============================================================="
echo " paths"
echo "=============================================================="
printf "  data     %s\n" "$DATA"
printf "  runs     %s\n" "$RUNS"
printf "  results  %s\n" "$RESULTS"
unset_vars=""
[ -z "${MIQ_DATA:-}" ]    && unset_vars="$unset_vars MIQ_DATA"
[ -z "${MIQ_RUNS:-}" ]    && unset_vars="$unset_vars MIQ_RUNS"
[ -z "${MIQ_RESULTS:-}" ] && unset_vars="$unset_vars MIQ_RESULTS"
if [ -n "$unset_vars" ]; then
  echo
  echo "  ! unset in this shell:$unset_vars"
  echo "  ! falling back to repo-relative paths, which is probably NOT where"
  echo "    your artifacts are. Run 'source ~/.bashrc', or set them inline:"
  echo "      MIQ_DATA=/data/miq/data MIQ_RUNS=/data/miq/runs \\"
  echo "      MIQ_RESULTS=/data/miq/results bash scripts/status.sh"
fi
echo

echo "=============================================================="
echo " live processes"
echo "=============================================================="
live=$(pgrep -af "train_grpo|lm_eval|build_dataset|dev_eval|10_train|20_evaluate" 2>/dev/null \
  | grep -v "status.sh" | grep -v "^$$ " || true)
if [ -n "$live" ]; then
  echo "$live" | sed 's/^/  /'
else
  echo "  nothing running"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "  --- gpu ---"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
    | sed 's/^/  /' || true
fi
echo
echo "  --- tmux sessions ---"
tmux ls 2>/dev/null | sed 's/^/  /' || echo "  none"

echo
echo "=============================================================="
echo " dataset"
echo "=============================================================="
for d in "$DATA"/processed/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  if [ -f "$d/manifest.json" ]; then
    python3 - "$d/manifest.json" "$name" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"  READY     {sys.argv[2]}  {m['num_examples']} examples  content hash {(m.get('content_hash') or m.get('dataset_hash', '?'))[:12]}")
PY
  else
    echo "  PARTIAL   $name (no manifest - delete before rebuilding)"
  fi
done
[ -d "$DATA/processed" ] || echo "  none built yet"

echo
echo "=============================================================="
echo " training runs (configs/experiments)"
echo "=============================================================="
for config in "$REPO"/configs/experiments/*.yaml; do
  name=$(basename "$config" .yaml)
  d="$RUNS/$name"
  if [ -f "$d/training_summary.json" ]; then
    step=$(python3 -c "import json;print(json.load(open('$d/training_summary.json'))['global_step'])" 2>/dev/null)
    echo "  COMPLETE  $name (step $step, checkpoint at $d/final)"
  elif [ -f "$d/logs/metrics.jsonl" ]; then
    last=$(tail -1 "$d/logs/metrics.jsonl" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(f\"step {d.get('step')} reward {d.get('reward','?')}\")" 2>/dev/null)
    echo "  UNFINISHED $name ($last) - resume with --resume"
  else
    echo "  NOT RUN   $name"
  fi
done

echo
echo "=============================================================="
echo " benchmark results"
echo "=============================================================="
complete=0
for name in $(python3 -c "import yaml;print(' '.join(r['run_id'] for r in yaml.safe_load(open('$REPO/configs/evaluation.yaml'))['runs']))"); do
  d="$RESULTS/moleculariq/$name"
  if [ ! -d "$d" ]; then
    echo "  NOT RUN   $name"
  elif [ -f "$d/summary.json" ]; then
    complete=$((complete + 1))
    python3 - "$d/summary.json" "$name" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
metrics = s.get("metrics") or {}
bits = "  ".join(
    f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
    for k, v in metrics.items()
)
full = "full" if s.get("full_benchmark") else "NOT FULL"
print(f"  COMPLETE  {sys.argv[2]}  [{full}]  {bits}")
PY
  elif [ -f "$d/eval_manifest.json" ]; then
    rc=$(python3 -c "import json;print(json.load(open('$d/eval_manifest.json')).get('returncode'))" 2>/dev/null)
    log="$d/output.log"; [ -f "$log" ] || log="$d/stderr.log"
    echo "  FAILED    $name (lm_eval exited $rc; see $log)"
  elif pgrep -af "lm_eval" 2>/dev/null | grep -q "$name"; then
    started=$(stat -c %y "$d" 2>/dev/null | cut -d. -f1 || echo "?")
    echo "  RUNNING   $name (started $started) - leave it alone"
  else
    echo "  PARTIAL   $name (killed mid-run; rm -rf it before retrying)"
  fi
done
expected=$(grep -c "^  - run_id:" "$REPO/configs/evaluation.yaml")
echo
echo "  $complete/$expected models evaluated (configs/evaluation.yaml)"
if [ "$complete" -ge "$expected" ]; then
  echo "  -> ready for: bash scripts/30_report.sh"
fi
