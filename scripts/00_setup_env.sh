#!/usr/bin/env bash
# Create the environment (once per machine, needs network): this package and

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv}"
THIRD_PARTY="${THIRD_PARTY:-$PROJECT_DIR/third_party}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
LOCK="$PROJECT_DIR/requirements-lock.txt"

MOLECULARIQ_CORE_COMMIT="a1b89635371c3cd942e44ebeec63ec3665e7743d"
MOLECULARIQ_EVAL_COMMIT="425ecaaa8faf65aa43aa60ec0f584b7b7f060063"

echo "project : $PROJECT_DIR"
echo "venv    : $VENV_PATH"

if [ ! -d "$VENV_PATH" ]; then
  python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip wheel setuptools

mkdir -p "$THIRD_PARTY"

clone_at() {
  local url="$1" dest="$2" commit="$3"
  if [ ! -d "$dest/.git" ]; then
    git clone "$url" "$dest"
  fi
  git -C "$dest" fetch --all --quiet
  git -C "$dest" checkout --quiet "$commit"
  echo "  $(basename "$dest") @ $(git -C "$dest" rev-parse --short HEAD)"
}

echo "== official MolecularIQ packages =="
clone_at https://github.com/ml-jku/moleculariq-core.git \
         "$THIRD_PARTY/moleculariq-core" "$MOLECULARIQ_CORE_COMMIT"
clone_at https://github.com/ml-jku/moleculariq-eval.git \
         "$THIRD_PARTY/moleculariq-eval" "$MOLECULARIQ_EVAL_COMMIT"

python -m pip install -c "$LOCK" -e "$THIRD_PARTY/moleculariq-core"
if [ "$INSTALL_VLLM" = "1" ]; then
  python -m pip install -c "$LOCK" -e "$THIRD_PARTY/moleculariq-eval[vllm]"
  python -m pip install -c "$LOCK" "ray>=2.9"
else
  python -m pip install -c "$LOCK" -e "$THIRD_PARTY/moleculariq-eval"
fi

echo "== this package =="
python -m pip install -c "$LOCK" -e "$PROJECT_DIR[dev]"

echo
python - <<'PY'
import importlib
for name in ("torch", "transformers", "trl", "accelerate", "datasets", "rdkit",
             "moleculariq_core", "lm_eval"):
    try:
        module = importlib.import_module(name)
        print(f"  {name:20s} {getattr(module, '__version__', 'ok')}")
    except Exception as exc:
        print(f"  {name:20s} MISSING ({exc})")
PY

echo
echo "next: python scripts/01_prefetch_assets.py"
