"""Model training. Owner: P4.

Each trained model is saved as a bundle {"model", "columns", "fill", "kind"} in
<run dir>/models/ so predict.py can score new rows with exactly the same feature layout.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import OUTPUTS_DIR

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
    """Fit a gradient-boosted classifier; returns a model bundle (not saved).
    params["bag_seeds"] > 1 fits that many models with seeds seed, seed+1, ... and the
    bundle scores by averaging their probabilities (lower variance, same features)."""
    params = dict(params)
    bags = int(params.pop("bag_seeds", 1))
    models = []
    for i in range(bags):
        model = _lgbm_classifier(params, seed + i, n_jobs)
        model.fit(X, y.astype(int))
        models.append(model)
    return {"kind": "lgbm", "model": models[0], "models": models, "columns": list(X.columns),
            "fill": None}


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
    models = bundle.get("models", [bundle["model"]])
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def save(bundle: dict, name: str, run_dir: Path = OUTPUTS_DIR) -> None:
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, run_dir / "models" / f"{name}.joblib")


def load(name: str, run_dir: Path = OUTPUTS_DIR) -> dict:
    return joblib.load(run_dir / "models" / f"{name}.joblib")
