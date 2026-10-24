"""Label-free overfitting guard: does any feature's distribution shift between splits?

Per-feature Kolmogorov–Smirnov statistic (0 = identical distributions, 1 = disjoint)
between train and another split. A feature with a large KS is likely to behave
differently on test: drop it, or normalise it within the split.

(A joint train-vs-test classifier is NOT used: whole-week per-wallet totals fingerprint
wallets, and wallets are disjoint between splits, so it always reaches AUC ~1 without
indicating a harmful shift.)

    python scripts/drift_check.py [--other validation|test] [--top 15]
Run the pipeline for that target first: it writes features_*.parquet next to its --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUTS_DIR  # noqa: E402

WARN_KS = 0.1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--other", default="validation", choices=["validation", "test"])
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--run-dir", type=Path, default=OUTPUTS_DIR, help="folder of a pipeline run")
    a = p.parse_args()
    A = pd.read_parquet(a.run_dir / "features_train.parquet")
    B = pd.read_parquet(a.run_dir / f"features_{a.other}.parquet")
    rows = []
    for c in A.columns:
        x, z = A[c].dropna(), B[c].dropna() if c in B else pd.Series(dtype=float)
        if len(x) and len(z):
            rows.append((c, ks_2samp(x, z).statistic, x.median(), z.median()))
    df = pd.DataFrame(rows, columns=["feature", "ks", "median_train", f"median_{a.other}"])
    df = df.sort_values("ks", ascending=False).set_index("feature")
    print(f"Feature drift train vs {a.other} (KS; > {WARN_KS} worth a look)\n")
    print(df.head(a.top).round(4).to_string())
    flagged = df.index[df["ks"] > WARN_KS].tolist()
    print(f"\n{len(flagged)} feature(s) above {WARN_KS}: {flagged}")


if __name__ == "__main__":
    main()
