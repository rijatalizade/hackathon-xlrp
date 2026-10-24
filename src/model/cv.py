"""Cross-validation harness. Owner: P4."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.model import train

log = logging.getLogger(__name__)


def metrics(y, s) -> dict:
    """{'roc_auc', 'average_precision'} for labels y and scores s."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, s)),
            "average_precision": float(average_precision_score(y, s))}


def oof_lgbm(X: pd.DataFrame, y: pd.Series, cfg: dict) -> tuple[np.ndarray, dict]:
    """Stratified k-fold out-of-fold scores for the supervised model.
    Returns (oof_scores aligned to X, metrics dict incl. the primary `metric`)."""
    n_splits = min(cfg["cv"]["n_splits"], int(y.sum()), int((1 - y).sum()))
    if n_splits < 2:
        raise ValueError("Too few positives/negatives for CV")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    oof = np.zeros(len(X))
    for k, (tr, va) in enumerate(skf.split(X, y)):
        bundle = train.train_lgbm(X.iloc[tr], y.iloc[tr], cfg["models"]["lgbm"], cfg["seed"],
                                  cfg.get("n_jobs", 1))
        oof[va] = train.score(bundle, X.iloc[va])
    m = metrics(y, oof)
    m["metric"] = m[cfg["cv"]["metric"]]
    return oof, m
