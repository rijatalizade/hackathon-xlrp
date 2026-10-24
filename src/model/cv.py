"""Grouped, repeated cross-validation and site-style scoring. Owner: P4.

Anti-overfitting rules baked in:
- folds are grouped by scenario (cfg.cv.group_col): one scenario never sits in both
  the fit and the check part;
- CV is repeated with several seeds and fold spread is reported, so a "gain" smaller
  than the noise is visible as such;
- AP is also broken down by difficulty, so easy cases can't hide weak medium/hard ones.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

log = logging.getLogger(__name__)


def ap100(y, s) -> float:
    """Site score: 100 x average precision."""
    y = np.asarray(y).astype(int)
    return float(100 * average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def metrics(y, s) -> dict:
    """{'roc_auc', 'average_precision'} (AP as a fraction, like sklearn)."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, s)),
            "average_precision": float(average_precision_score(y, s))}


def top_frac_pr(y, s, ids, frac: float) -> tuple[float, float]:
    """Precision and recall in the top `frac` of scores; ties broken by id ascending (site rule)."""
    y = np.asarray(y).astype(int)
    k = max(1, int(np.ceil(frac * len(y))))
    order = np.lexsort((np.asarray(ids), -np.asarray(s)))[:k]
    hits = y[order].sum()
    return float(hits / k), float(hits / max(1, y.sum()))


def groups_for(tx: pd.DataFrame, cfg: dict) -> np.ndarray:
    """CV group per tx: its scenario id, or the tx itself if it has none."""
    col = cfg["cv"]["group_col"]
    g = tx[col] if col in tx else pd.Series(np.nan, index=tx.index)
    return g.fillna(tx["tx_id"]).astype(str).to_numpy()


def oof_scores(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: dict,
               fit_score: Callable, seed: int) -> tuple[np.ndarray, list[float]]:
    """One grouped k-fold pass. fit_score(X_fit, y_fit, X_check) -> scores.
    Returns (out-of-fold scores, per-fold AP x100)."""
    skf = StratifiedGroupKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
    oof, fold_aps = np.zeros(len(X)), []
    for fit_idx, chk_idx in skf.split(X, y, groups):
        oof[chk_idx] = fit_score(X.iloc[fit_idx], y[fit_idx], X.iloc[chk_idx])
        fold_aps.append(ap100(y[chk_idx], oof[chk_idx]))
    return oof, fold_aps


def evaluate(y, s, tx: pd.DataFrame, cfg: dict) -> dict:
    """AP x100 overall and per difficulty (that difficulty's positives vs all negatives),
    plus site-style precision/recall in the top 1%."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s)
    p, r = top_frac_pr(y, s, tx["tx_id"].to_numpy(), cfg["cv"]["top_frac"])
    out = {"ap": ap100(y, s), "top_precision": p, "top_recall": r}
    if "meta_difficulty" in tx:
        d = tx["meta_difficulty"].to_numpy()
        for level in ["easy", "medium", "hard"]:
            keep = (y == 0) | (d == level)
            if (y[keep] == 1).any():
                out[f"ap_{level}"] = ap100(y[keep], s[keep])
    return out


def repeated_cv(X: pd.DataFrame, y: np.ndarray, tx: pd.DataFrame, cfg: dict,
                fit_score: Callable, repeats: int | None = None) -> dict:
    """Repeated grouped CV. Returns {'oof': first repeat's OOF scores, 'summary': {...}}.
    summary.ap = mean over repeats of overall OOF AP; ap_fold_std = std of all fold APs."""
    groups = groups_for(tx, cfg)
    repeats = repeats or cfg["cv"].get("repeats", 1)
    runs, folds, first_oof = [], [], None
    for r in range(repeats):
        oof, fold_aps = oof_scores(X, y, groups, cfg, fit_score, cfg["seed"] + r)
        first_oof = oof if first_oof is None else first_oof
        runs.append(evaluate(y, oof, tx, cfg))
        folds += fold_aps
    summary = {k: float(np.nanmean([run[k] for run in runs])) for k in runs[0]}
    summary["ap_fold_std"] = float(np.std(folds))
    summary["ap_fold_min"], summary["ap_fold_max"] = float(np.min(folds)), float(np.max(folds))
    summary["repeats"] = repeats
    summary["fold_aps"] = [round(a, 4) for a in folds]
    return {"oof": first_oof, "summary": summary}


def format_summary(name: str, s: dict) -> str:
    diff = "  ".join(f"{k[3:]}={s[k]:.1f}" for k in ["ap_easy", "ap_medium", "ap_hard"] if k in s)
    spread = (f" ±{s['ap_fold_std']:.1f} (folds {s['ap_fold_min']:.1f}–{s['ap_fold_max']:.1f})"
              if "ap_fold_std" in s else "")
    return (f"{name:9s} AP={s['ap']:.1f}{spread}  | {diff}  | top1% P={s['top_precision']:.2f} "
            f"R={s['top_recall']:.2f}")
