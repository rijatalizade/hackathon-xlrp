"""Per-transaction features (prefix `tx_`). Owner: P2.

No asset identities (issuers differ between splits) and no calendar date (test is a
different week). No conversion / legs / currency / spread features either: the
organisers state asset and conversion choices "do not inspect suspicious labels"; EDA
confirmed equal rates for suspicious and legitimate txs, and these are the features
that drift most between train and test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import validate

PREFIX = "tx_"

def build(tx: pd.DataFrame, legs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Value, round-ness, fee, hour, sender z-score, gaps."""
    fcfg = cfg.get("features", {})
    v = tx["value_usd"]
    f = pd.DataFrame(index=pd.Index(tx["tx_id"], name="tx_id"))
    f["tx_value_usd"] = v.to_numpy()
    f["tx_log_value_usd"] = np.log1p(v).to_numpy()
    for level in fcfg.get("round_levels", [10, 100, 1000]):
        f[f"tx_is_round_{level}"] = ((tx["amount_out"] > 0)
                                     & np.isclose(tx["amount_out"] % level, 0)).to_numpy()
    f["tx_fee"] = tx["fee"].to_numpy()
    f["tx_hour"] = tx["timestamp"].dt.hour.to_numpy()

    by_sender = v.groupby(tx["sender"])
    std = by_sender.transform("std").replace(0, np.nan)
    f["tx_value_z_sender"] = ((v - by_sender.transform("mean")) / std).to_numpy()
    f["tx_secs_since_sender_prev"] = tx["ts"].groupby(tx["sender"]).diff().to_numpy()
    f["tx_secs_since_receiver_prev"] = tx["ts"].groupby(tx["receiver"]).diff().to_numpy()
    return validate(f, PREFIX)
