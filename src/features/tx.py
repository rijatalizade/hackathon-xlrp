"""Per-transaction features (prefix `tx_`). Owner: P2."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import validate

PREFIX = "tx_"


def build(tx: pd.DataFrame, accounts: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    """Amount, round-ness, fee, time-of-day, conversion flag, sender z-score, gaps."""
    fcfg = cfg.get("features", {})
    amt = tx["amount"]
    f = pd.DataFrame(index=pd.Index(tx["tx_id"], name="tx_id"))
    f["tx_amount"] = amt.to_numpy()
    f["tx_log_amount"] = np.log1p(amt.clip(lower=0)).to_numpy()
    for level in fcfg.get("round_levels", [100, 1000, 10000]):
        f[f"tx_is_round_{level}"] = ((amt > 0) & (amt % level == 0)).to_numpy()
    f["tx_fee"] = tx["fee"].to_numpy() if "fee" in tx else np.nan
    f["tx_hour"] = tx["timestamp"].dt.hour.to_numpy()
    f["tx_dayofweek"] = tx["timestamp"].dt.day_of_week.to_numpy()

    conv = pd.Series(False, index=tx.index)
    if "currency" in tx:
        conv |= tx["currency"].notna() & (tx["currency"] != fcfg.get("native_currency", "XRP"))
    if "tx_type" in tx:
        conv |= tx["tx_type"].isin(fcfg.get("conversion_tx_types", []))
    f["tx_is_conversion"] = conv.to_numpy()

    by_sender = amt.groupby(tx["sender"])
    std = by_sender.transform("std").replace(0, np.nan)
    f["tx_amount_z_sender"] = ((amt - by_sender.transform("mean")) / std).to_numpy()

    secs = tx["timestamp"].dt.as_unit("s").astype("int64")
    f["tx_secs_since_sender_prev"] = secs.groupby(tx["sender"]).diff().to_numpy()
    f["tx_secs_since_receiver_prev"] = secs.groupby(tx["receiver"]).diff().to_numpy()
    return validate(f, PREFIX)
