"""Paired comparison of two pipeline runs' grouped-CV fold scores (same folds & seeds).

    python scripts/compare_runs.py outputs/runs/base/run_meta.json outputs/runs/new/run_meta.json

A change is worth keeping only if the mean paired gain is clearly above its noise
(rule of thumb: mean gain > 2 x standard error, and most folds improve).
"""
from __future__ import annotations

import json
import sys

import numpy as np


def main() -> None:
    a, b = (json.load(open(p))["cv"]["lgbm"] for p in sys.argv[1:3])
    fa, fb = np.array(a["fold_aps"]), np.array(b["fold_aps"])
    d = fb - fa
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"AP {a['ap']:.2f} -> {b['ap']:.2f}  | paired gain {d.mean():+.2f} ± {se:.2f} (SE), "
          f"{(d > 0).sum()}/{len(d)} folds improve  -> {'KEEP' if d.mean() > 2 * se else 'noise'}")
    for k in ["ap_easy", "ap_medium", "ap_hard", "top_precision", "top_recall"]:
        if k in a and k in b:
            print(f"  {k:14s} {a[k]:7.2f} -> {b[k]:7.2f}")


if __name__ == "__main__":
    main()
