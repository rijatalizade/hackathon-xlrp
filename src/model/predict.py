"""Score target-split txs with saved models and write the submission CSV. Owner: P1."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PREDICTIONS_PATH
from src.model import ensemble, train


def component_scores(X: pd.DataFrame, cfg: dict, mode: str, run_dir: Path) -> dict[str, np.ndarray]:
    """Scores of every ensemble component with a non-zero weight, from saved models."""
    scores = {}
    for name, w in cfg["ensemble"][mode].items():
        if w <= 0:
            continue
        if name == "rules":
            if "rule_score" in X:
                scores["rules"] = X["rule_score"].fillna(0).to_numpy()
        elif name != "lgbm" or mode == "supervised":
            scores[name] = train.score(train.load(name, run_dir), X)
    return scores


def predict(X_target: pd.DataFrame, cfg: dict, mode: str, out_path: Path = PREDICTIONS_PATH
            ) -> pd.DataFrame:
    """Rank-average the components, write and return the submission CSV.
    X_target is indexed by tx_id (one row per target tx); models are read from
    out_path's folder (the run dir)."""
    final = ensemble.rank_average(component_scores(X_target, cfg, mode, out_path.parent),
                                   cfg["ensemble"][mode])
    sub = cfg["submission"]
    df = pd.DataFrame({sub["id_col"]: X_target.index.to_numpy(), sub["score_col"]: final})
    assert df[sub["id_col"]].is_unique, "duplicate ids"
    assert not df[sub["score_col"]].isna().any(), "NaN scores"
    assert np.all((final >= 0) & (final <= 1)), "scores outside [0, 1]"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format=sub.get("float_format", "%.8f"))
    return df
