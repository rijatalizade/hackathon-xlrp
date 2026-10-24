#!/usr/bin/env bash
# One command: raw data -> outputs/predictions.csv. Extra args go to src.pipeline
# (e.g. ./run.sh --sample, ./run.sh --skip-graph).
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null || true; source .venv/bin/activate
pip install -q -r requirements.txt
python -m src.pipeline --mode auto "$@"
if [[ " $* " == *" --sample "* ]]; then
  python scripts/validate_submission.py --sample
else
  python scripts/validate_submission.py
fi
