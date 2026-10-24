"""Multi-hour money-flow timing (prefix `flow_`). Owner: P2.

EDA (24 Oct): suspicious activity moves money faster than its legitimate look-alikes —
relay hubs forward after ~3h vs ~30h, motifs complete ~8x faster — at time scales the
1h starter windows can't see. Per tx we measure, for the receiver, how soon and how much
of the money moves on, and for the sender, how recently it was funded. Whole-week view
(offline task), within one split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import validate

PREFIX = "flow_"
NEVER = 7 * 86400.0  # "no such event in the week"


def _side(tx: pd.DataFrame, col: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{wallet: (times, cumulative USD)} of txs where wallet is `col`, time-sorted."""
    t, v = tx["ts"].to_numpy(), tx["value_usd"].to_numpy()
    return {a: (t[i], np.cumsum(v[i])) for a, i in tx.groupby(col, sort=False).indices.items()}


def build(tx: pd.DataFrame, legs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Receiver dwell / forwarding and sender funding recency at several horizons."""
    horizons = cfg["features"].get("flow_horizons_secs", {"1h": 3600, "6h": 21600, "24h": 86400})
    tx = tx.sort_values(["ts", "tx_id"], kind="mergesort").reset_index(drop=True)
    out_of, in_of = _side(tx, "sender"), _side(tx, "receiver")
    t, v = tx["ts"].to_numpy(), tx["value_usd"].to_numpy()
    n = len(tx)

    r_next, r_half, s_last, s_funded = (np.full(n, NEVER) for _ in range(4))
    r_fwd = {h: np.zeros(n) for h in horizons}
    s_in = {h: np.zeros(n) for h in horizons}
    for k, (snd, rcv) in enumerate(zip(tx["sender"].to_numpy(), tx["receiver"].to_numpy())):
        ti, vi = t[k], v[k]
        if rcv in out_of:  # what the receiver sends after getting this payment
            times, cum = out_of[rcv]
            j = np.searchsorted(times, ti, "right")
            if j < len(times):
                base = cum[j - 1] if j else 0.0
                r_next[k] = times[j] - ti
                h = np.searchsorted(cum, base + 0.5 * vi, "left")
                if h < len(times):
                    r_half[k] = times[h] - ti
                for name, w in horizons.items():
                    e = np.searchsorted(times, ti + w, "right")
                    r_fwd[name][k] = ((cum[e - 1] if e else 0.0) - base) / vi
        if snd in in_of:  # what the sender received before making this payment
            times, cum = in_of[snd]
            j = np.searchsorted(times, ti, "left")  # strictly before
            if j > 0:
                s_last[k] = ti - times[j - 1]
                f = np.searchsorted(cum, cum[j - 1] - vi, "right")  # latest start covering vi
                if cum[j - 1] >= vi:
                    s_funded[k] = ti - times[f]
                for name, w in horizons.items():
                    b = np.searchsorted(times, ti - w, "left")
                    s_in[name][k] = (cum[j - 1] - (cum[b - 1] if b else 0.0)) / vi

    f = pd.DataFrame(index=pd.Index(tx["tx_id"], name="tx_id"))
    f["flow_r_log_secs_to_next_send"] = np.log1p(r_next)
    f["flow_r_log_secs_to_forward_half"] = np.log1p(r_half)
    f["flow_s_log_secs_since_last_recv"] = np.log1p(s_last)
    f["flow_s_log_secs_since_funded"] = np.log1p(s_funded)
    for name in horizons:
        f[f"flow_r_fwd_frac_{name}"] = np.clip(r_fwd[name], 0, 10)
        f[f"flow_s_in_frac_{name}"] = np.clip(s_in[name], 0, 10)
    return validate(f, PREFIX)
