"""Check a predictions CSV against the test tx_ids. Exits non-zero on any failure.

    python scripts/validate_submission.py [--pred outputs/predictions.csv] [--sample]

Data source defaults to whatever the pipeline run that wrote --pred used (run_meta.json
next to it).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PREDICTIONS_PATH, load_params  # noqa: E402
from src.data.load import load_raw, split  # noqa: E402


def validate(pred_path: Path, sample: bool, data_dir: Path | None) -> list[str]:
    sub = load_params()["submission"]
    id_col, score_col = sub["id_col"], sub["score_col"]
    _, test = split(load_raw(sample=sample, data_dir=data_dir)[0])
    pred = pd.read_csv(pred_path, dtype={id_col: str})
    errors = []
    if list(pred.columns) != [id_col, score_col]:
        errors.append(f"columns {list(pred.columns)} != {[id_col, score_col]}")
        return errors
    if len(pred) != len(test):
        errors.append(f"row count {len(pred)} != test size {len(test)}")
    if not pred[id_col].is_unique:
        errors.append(f"{pred[id_col].duplicated().sum()} duplicate ids")
    missing = set(test["tx_id"]) - set(pred[id_col])
    extra = set(pred[id_col]) - set(test["tx_id"])
    if missing:
        errors.append(f"{len(missing)} test ids missing, e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:3]}")
    if pred[score_col].isna().any():
        errors.append(f"{pred[score_col].isna().sum()} NaN scores")
    if not pred[score_col].between(0, 1).all():
        errors.append("scores outside [0, 1]")
    return errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path, default=PREDICTIONS_PATH)
    p.add_argument("--sample", action="store_true", default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    a = p.parse_args()
    meta_path = a.pred.parent / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    sample = a.sample if a.sample is not None else meta.get("sample", False)
    data_dir = a.data_dir or (Path(meta["data_dir"]) if meta.get("data_dir") else None)

    errors = validate(a.pred, sample, data_dir)
    if errors:
        print("SUBMISSION INVALID:\n  - " + "\n  - ".join(errors))
        sys.exit(1)
    print(f"Submission OK: {a.pred}")


if __name__ == "__main__":
    main()
