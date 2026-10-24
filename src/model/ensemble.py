"""Rank-average ensembling. Owner: P4."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def rank_average(scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted mean of per-model normalised ranks, min-max scaled to [0, 1].
    Models missing from `scores` are skipped and remaining weights renormalised."""
    used = {k: w for k, w in weights.items() if k in scores and w > 0}
    if not used:
        raise ValueError(f"No weighted models among {list(scores)}")
    n = len(next(iter(scores.values())))
    total = sum(used.values())
    out = np.zeros(n)
    for k, w in used.items():
        out += w * rankdata(scores[k], method="average") / n
    out /= total
    lo, hi = out.min(), out.max()
    return np.full(n, 0.5) if hi == lo else (out - lo) / (hi - lo)
