#!/usr/bin/env bash
# One command: organiser split folders -> outputs/predictions.csv (validated).
# Needs data/raw/{train,validation,test}/ (see README). Extra args go to src.pipeline,
# e.g. ./run.sh --sample (synthetic mini data) or ./run.sh --target validation.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null || true; source .venv/bin/activate
python -m pip install -q -r requirements.txt
python -m src.pipeline "$@"
python scripts/validate_submission.py
