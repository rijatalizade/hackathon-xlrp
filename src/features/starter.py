"""Port of the organisers' starter_notebook.ipynb features (prefix `starter_`). Owner: P2.

Kept close to the original so our v0 reproduces its ~30 AP baseline; values are the
same USD reference values (`value_usd`). Whole-week aggregates are allowed (offline task).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import validate

PREFIX = "starter_"
HOUR, TEN_MINUTES = 3600, 600


def _by_account(col: pd.Series, t: np.ndarray, v: np.ndarray):
    """{account: (sorted times, values in the same order)}; input is time-sorted."""
    out = {}
    for a, idx in col.groupby(col, sort=False).indices.items():
        out[a] = (t[idx], v[idx])
    return out


def _count(index, account, lo, hi):
    """Number of the account's payments with time in [lo, hi]."""
    times = index.get(account, (None,))[0]
    if times is None:
        return 0
    return int(np.searchsorted(times, hi, "right") - np.searchsorted(times, lo, "left"))


def build(tx: pd.DataFrame, legs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The 11 starter-notebook rules, one row per tx."""
    tx = tx.sort_values("ts", kind="mergesort").reset_index(drop=True)
    t, v = tx["ts"].to_numpy(), tx["value_usd"].to_numpy()
    sends = _by_account(tx["sender"], t, v)
    gets = _by_account(tx["receiver"], t, v)
    S, R = tx["sender"].to_numpy(), tx["receiver"].to_numpy()

    f = pd.DataFrame(index=pd.Index(tx["tx_id"], name="tx_id"))
    f["starter_log_value"] = np.log1p(v)
    f["starter_sender_payments"] = tx["sender"].map(tx["sender"].value_counts()).to_numpy()
    f["starter_receiver_payments"] = tx["receiver"].map(tx["receiver"].value_counts()).to_numpy()
    f["starter_receiver_forwards_1h"] = [_count(sends, a, ti, ti + HOUR) > 0
                                         for a, ti in zip(R, t)]
    f["starter_sender_received_1h_before"] = [_count(gets, a, ti - HOUR, ti) > 0
                                              for a, ti in zip(S, t)]
    f["starter_sender_burst_10m"] = [_count(sends, a, ti - TEN_MINUTES, ti + TEN_MINUTES)
                                     for a, ti in zip(S, t)]
    f["starter_receiver_burst_10m"] = [_count(gets, a, ti - TEN_MINUTES, ti + TEN_MINUTES)
                                       for a, ti in zip(R, t)]

    def forwards_similar(a, ti, amount):
        if a not in sends:
            return False
        times, vals = sends[a]
        later = vals[np.searchsorted(times, ti, "right"):np.searchsorted(times, ti + HOUR, "right")]
        return bool(np.any(np.abs(later - amount) <= 0.1 * amount))

    def received_similar(a, ti, amount):
        if a not in gets:
            return False
        times, vals = gets[a]
        earlier = vals[np.searchsorted(times, ti - HOUR, "left"):np.searchsorted(times, ti, "left")]
        return bool(np.any(np.abs(earlier - amount) <= 0.1 * amount))

    def minutes_to_next_send(a, ti):
        if a not in sends:
            return 60.0
        times = sends[a][0]
        i = np.searchsorted(times, ti, "right")
        return min(60.0, (times[i] - ti) / 60) if i < len(times) else 60.0

    f["starter_receiver_forwards_similar_1h"] = [forwards_similar(a, ti, x)
                                                 for a, ti, x in zip(R, t, v)]
    f["starter_sender_received_similar_1h"] = [received_similar(a, ti, x)
                                               for a, ti, x in zip(S, t, v)]
    f["starter_receiver_minutes_to_forward"] = [minutes_to_next_send(a, ti) for a, ti in zip(R, t)]

    sent_total = tx.groupby("sender")["value_usd"].sum()
    got_total = tx.groupby("receiver")["value_usd"].sum()
    pass_through = (sent_total / got_total.reindex(sent_total.index)).clip(0, 3)
    f["starter_sender_pass_through"] = tx["sender"].map(pass_through).fillna(0).to_numpy()
    f["starter_receiver_pass_through"] = tx["receiver"].map(pass_through).fillna(0).to_numpy()
    return validate(f.astype(float), PREFIX)
