"""Model training. Owner: P4.

Each trained model is saved as a bundle {"model", "columns", "fill", "kind"} so
predict.py can score new rows with exactly the same feature layout.
"""
from __future__ import annotations

import functools
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import MODELS_DIR

log = logging.getLogger(__name__)


@functools.cache
def _lightgbm():
    """The lightgbm module, or None if it can't load (warns once)."""
    try:
        import lightgbm
        return lightgbm
    except (ImportError, OSError) as e:
        log.warning("LightGBM unavailable (%s); falling back to HistGradientBoosting. "
                    "On macOS: `brew install libomp`.", str(e).splitlines()[0][:120])
        return None


def _lgbm_classifier(params: dict, seed: int, n_jobs: int):
    """LightGBM if importable, else sklearn HistGradientBoosting."""
    lgb = _lightgbm()
    if lgb is not None:
        return lgb.LGBMClassifier(**params, random_state=seed, n_jobs=n_jobs)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_leaf_nodes=params.get("num_leaves", 31),
            min_samples_leaf=params.get("min_child_samples", 20),
            l2_regularization=params.get("reg_lambda", 0.0),
            random_state=seed)


def train_lgbm(X: pd.DataFrame, y: pd.Series, params: dict, seed: int, n_jobs: int = 1) -> dict:
    """Fit a gradient-boosted classifier; returns a model bundle (not saved)."""
    model = _lgbm_classifier(params, seed, n_jobs)
    model.fit(X, y.astype(int))
    return {"kind": "lgbm", "model": model, "columns": list(X.columns), "fill": None}


def train_iforest(X: pd.DataFrame, seed: int, params: dict | None = None, n_jobs: int = 1) -> dict:
    """Fit an IsolationForest on median-filled features; returns a model bundle."""
    fill = X.median(numeric_only=True).fillna(0)
    model = IsolationForest(**(params or {}), random_state=seed, n_jobs=n_jobs)
    model.fit(X.fillna(fill))
    return {"kind": "iforest", "model": model, "columns": list(X.columns), "fill": fill}


def score(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Higher = more suspicious, for any bundle kind."""
    X = X.reindex(columns=bundle["columns"])
    if bundle["kind"] == "iforest":
        return -bundle["model"].score_samples(X.fillna(bundle["fill"]))
    return bundle["model"].predict_proba(X)[:, 1]


def save(bundle: dict, name: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / f"{name}.joblib")


def load(name: str) -> dict:
    return joblib.load(MODELS_DIR / f"{name}.joblib")
