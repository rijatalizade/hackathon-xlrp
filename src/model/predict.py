"""Score test txs with saved models and write the submission CSV. Owner: P1."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PREDICTIONS_PATH
from src.model import ensemble, train


def predict(X_test: pd.DataFrame, cfg: dict, mode: str, out_path: Path = PREDICTIONS_PATH
            ) -> pd.DataFrame:
    """Load saved models, rank-average with the rule score, write and return the CSV.
    X_test is indexed by tx_id (one row per test tx)."""
    weights = cfg["ensemble"][mode]
    scores = {}
    for name in weights:
        if name == "rules":
            if "rule_score" in X_test:
                scores["rules"] = X_test["rule_score"].fillna(0).to_numpy()
        elif name != "lgbm" or mode == "supervised":
            scores[name] = train.score(train.load(name), X_test)
    final = ensemble.rank_average(scores, weights)

    sub = cfg["submission"]
    df = pd.DataFrame({sub["id_col"]: X_test.index.to_numpy(), sub["score_col"]: final})
    assert df[sub["id_col"]].is_unique, "duplicate ids"
    assert not df[sub["score_col"]].isna().any(), "NaN scores"
    assert np.all((final >= 0) & (final <= 1)), "scores outside [0, 1]"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format=sub.get("float_format", "%.8f"))
    return df
