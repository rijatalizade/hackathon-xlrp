"""Sender / receiver aggregate features (prefix `acct_s_*`, `acct_r_*`). Owner: P2.

All aggregates are point-in-time: computed over the account's activity (as sender OR
receiver) up to and including the current tx, never using later txs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import rolling_count, to_seconds, validate

PREFIX = "acct_"


def _activity(tx: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (tx, side) with acct, counterparty, direction."""
    secs = to_seconds(tx["timestamp"])
    pos = np.arange(len(tx))
    common = {"tx_id": tx["tx_id"].to_numpy(), "ts": secs, "amt": tx["amount"].to_numpy(),
              "pos": pos}
    out = pd.DataFrame({**common, "acct": tx["sender"].to_numpy(),
                        "cp": tx["receiver"].to_numpy(), "is_in": 0})
    inc = pd.DataFrame({**common, "acct": tx["receiver"].to_numpy(),
                        "cp": tx["sender"].to_numpy(), "is_in": 1})
    long = pd.concat([out, inc], ignore_index=True)
    return long.sort_values(["acct", "ts", "pos", "is_in"], kind="mergesort").reset_index(
        drop=True)


def build(tx: pd.DataFrame, accounts: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    """n_tx_total/1h/24h, vol_in/out, inout_ratio, distinct counterparties, age,
    mean gap and burstiness, for both sender (acct_s_) and receiver (acct_r_)."""
    windows = cfg.get("features", {}).get("account_windows_secs", {"1h": 3600, "24h": 86400})
    long = _activity(tx)
    g = long.groupby("acct", sort=False)
    a = pd.DataFrame(index=long.index)

    n = g.cumcount().to_numpy() + 1
    a["n_tx_total"] = n
    for name, w in windows.items():
        a[f"n_tx_{name}"] = rolling_count(long["acct"], long["ts"], int(w))

    amt_in = long["amt"].where(long["is_in"] == 1, 0.0)
    amt_out = long["amt"].where(long["is_in"] == 0, 0.0)
    a["vol_in"] = amt_in.groupby(long["acct"]).cumsum()
    a["vol_out"] = amt_out.groupby(long["acct"]).cumsum()
    total = (a["vol_in"] + a["vol_out"]).replace(0, np.nan)
    a["inout_ratio"] = a["vol_in"] / total  # 0.5 = balanced in/out (mule-like)

    first_pair = (~long.duplicated(["acct", "cp"])).astype(int)
    a["n_distinct_counterparties"] = first_pair.groupby(long["acct"]).cumsum()
    a["age_secs"] = long["ts"] - g["ts"].transform("min")

    gap = g["ts"].diff()
    n_gaps = pd.Series(n - 1, index=long.index).replace(0, np.nan)
    gsum = gap.fillna(0).groupby(long["acct"]).cumsum()
    gsq = (gap.fillna(0) ** 2).groupby(long["acct"]).cumsum()
    mean_gap = gsum / n_gaps
    std_gap = np.sqrt((gsq / n_gaps - mean_gap ** 2).clip(lower=0))
    a["mean_gap_secs"] = mean_gap
    a["burstiness"] = std_gap / mean_gap.replace(0, np.nan)

    a["tx_id"] = long["tx_id"]
    sides = []
    for flag, tag in [(0, "s"), (1, "r")]:
        side = a[long["is_in"].to_numpy() == flag].drop_duplicates("tx_id").set_index("tx_id")
        sides.append(side.add_prefix(f"acct_{tag}_"))
    f = pd.concat(sides, axis=1).reindex(tx["tx_id"])
    return validate(f, PREFIX)
