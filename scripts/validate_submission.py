"""Check a predictions CSV against the target split's TRANSACTION_HASHes (site rules).
Exits non-zero on any failure.

    python scripts/validate_submission.py [--pred outputs/predictions.csv] [--target test] [--sample]

Target split defaults to whatever the pipeline run that wrote --pred used (run_meta.json
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
from src.data.load import load_split, split_dir  # noqa: E402

MAX_BYTES, MAX_ROWS = 12 * 1024 * 1024, 100_000


def validate(pred_path: Path, target: str, sample: bool) -> list[str]:
    cfg = load_params()
    id_col, score_col = cfg["submission"]["id_col"], cfg["submission"]["score_col"]
    ids = set(load_split(split_dir(cfg, target, sample))[0]["tx_id"])
    pred = pd.read_csv(pred_path, dtype={id_col: str})
    errors = []
    if list(pred.columns[:2]) != [id_col, score_col] or set(pred.columns[2:]) - {"reason"}:
        return [f"columns {list(pred.columns)} != {[id_col, score_col]} (+ optional reason)"]
    if pred_path.stat().st_size > MAX_BYTES or len(pred) > MAX_ROWS:
        errors.append("file over 12 MiB or 100,000 rows")
    if len(pred) != len(ids):
        errors.append(f"row count {len(pred)} != {target} size {len(ids)}")
    if not pred[id_col].is_unique:
        errors.append(f"{pred[id_col].duplicated().sum()} duplicate ids")
    missing, extra = ids - set(pred[id_col]), set(pred[id_col]) - ids
    if missing:
        errors.append(f"{len(missing)} {target} ids missing, e.g. {sorted(missing)[:2]}")
    if extra:
        errors.append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:2]}")
    if pred[score_col].isna().any():
        errors.append(f"{pred[score_col].isna().sum()} NaN scores")
    if not pred[score_col].between(0, 1).all():
        errors.append("scores outside [0, 1]")
    return errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path, default=PREDICTIONS_PATH)
    p.add_argument("--target", default=None, choices=["test", "validation"])
    p.add_argument("--sample", action="store_true", default=None)
    a = p.parse_args()
    meta_path = a.pred.parent / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    target = a.target or meta.get("target", "test")
    sample = a.sample if a.sample is not None else meta.get("sample", False)

    errors = validate(a.pred, target, sample)
    if errors:
        print("SUBMISSION INVALID:\n  - " + "\n  - ".join(errors))
        sys.exit(1)
    print(f"Submission OK: {a.pred} ({target}{', sample' if sample else ''})")


if __name__ == "__main__":
    main()
