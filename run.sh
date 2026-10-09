#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cmd="${1:-}"
if [[ "$cmd" == setup ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    python3.11 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
  fi
  exit 0
fi
if [[ ! -x .venv/bin/python ]]; then
  echo 'Run: bash run.sh setup' >&2
  exit 2
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
case "$cmd" in
  validate)
    .venv/bin/python scripts/validate_manifest.py --config configs/base_clean.yaml
    .venv/bin/python scripts/validate_manifest.py --config configs/base_realistic_mix_l2.yaml ;;
  smoke) .venv/bin/python scripts/smoke_test.py ;;
  base_clean) .venv/bin/python scripts/run_inference.py --config configs/base_clean.yaml --offline ;;
  base_degraded) .venv/bin/python scripts/run_inference.py --config configs/base_realistic_mix_l2.yaml --offline ;;
  score_baseline)
    .venv/bin/python scripts/evaluate_textvqa.py --clean-run outputs/base_clean --degraded-run outputs/base_realistic_mix_l2 --output-dir outputs/comparison_clean_vs_realistic_mix_l2 ;;
  *) echo 'Usage: bash run.sh {setup|validate|smoke|base_clean|base_degraded|score_baseline}' >&2; exit 2 ;;
esac
