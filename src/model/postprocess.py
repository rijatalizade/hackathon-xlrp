"""Score post-processing: lift each tx toward the strongest score in its burst. Owner: P4.

Error analysis (24 Oct): 151 of 800 easy positives ranked below the top 3%, almost all of
them the *edges* of bursts the model had already caught (fan-out leaves that keep the money,
source payments that were not just funded). Ranking the burst instead of the single payment
lifts them. Label-free: components are the same 10-minute continuation/burst links as
`src/features/motif.py`, scores are rank-normalised within the split first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.features.motif import components


def component_lift(scores: np.ndarray, tx: pd.DataFrame, cfg: dict) -> np.ndarray:
    """(1 - w) * rank(score) + w * max rank in the tx's component; `tx` rows align with
    `scores`. Config: cfg["postprocess"]["component_lift"] = {enabled, window_secs, weight}."""
    pcfg = (cfg.get("postprocess") or {}).get("component_lift") or {}
    if not pcfg.get("enabled", False) or len(scores) == 0:
        return np.asarray(scores, dtype=float)
    w, window = float(pcfg["weight"]), int(pcfg["window_secs"])
    order = np.lexsort((tx["tx_id"].to_numpy(), tx["ts"].to_numpy()))  # components need time order
    t = tx.iloc[order]
    comp = components(t["ts"].to_numpy(), t["value_usd"].to_numpy(), t["sender"].to_numpy(),
                      t["receiver"].to_numpy(), window, window, None)
    r = rankdata(scores) / len(scores)
    cmax = pd.Series(r[order]).groupby(comp).transform("max").to_numpy()
    lifted = np.empty(len(scores))
    lifted[order] = (1 - w) * r[order] + w * cmax
    return lifted
