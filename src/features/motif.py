"""Temporal motif reconstruction (prefix `motif_`). Owner: P3.

EDA (24 Oct): planted motifs (chain, cycle, fan-out/in, bipartite, stack, ...) have
legitimate look-alikes of the same shape; the suspicious ones complete much faster
(median 0.13h vs 1.0h). So we rebuild motifs from the payments themselves and describe
them at several time scales:

- links: money continues (A->B then B's next send within W; similar value for long W),
  or a wallet sends / receives again within 10 min (fan-out / fan-in bursts);
- components of those links at each W: size, span, rate, distinct wallets;
- chains: payment funded by the sender's latest similar-valued receipt within W
  (depth, value retention); cycles: such a chain returns to its origin.
Labels and scenario ids are never used; built within one split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import validate

PREFIX = "motif_"


def _find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def components(t: np.ndarray, v: np.ndarray, snd: np.ndarray, rcv: np.ndarray, window: int,
               burst_window: int, tol: float | None) -> np.ndarray:
    """Component id per tx (time-sorted input). Links: a wallet's consecutive sends (or
    receives) <= burst_window apart, and continuation to the receiver's first send within
    (t, t + window] — of similar value (within tol) if tol is set. Long windows need tol,
    otherwise busy wallets percolate everything into one component."""
    n = len(t)
    parent = np.arange(n)

    def union(a, b):
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    out_idx = pd.Series(np.arange(n)).groupby(snd).indices
    in_idx = pd.Series(np.arange(n)).groupby(rcv).indices
    for idx in list(out_idx.values()) + list(in_idx.values()):  # bursts: consecutive, gap <= W
        gaps = np.diff(t[idx])
        for a, b, g in zip(idx[:-1], idx[1:], gaps):
            if g <= burst_window:
                union(a, b)
    for k in range(n):  # continuation: receiver's first (similar-valued) send in (t, t + W]
        nxt = out_idx.get(rcv[k])
        if nxt is None:
            continue
        lo = np.searchsorted(t[nxt], t[k], "right")
        hi = np.searchsorted(t[nxt], t[k] + window, "right")
        for j in nxt[lo:hi]:
            if tol is None or abs(v[j] - v[k]) <= tol * v[k]:
                union(k, j)
                break
    return np.array([_find(parent, i) for i in range(n)])


def chains(t: np.ndarray, v: np.ndarray, snd: np.ndarray, rcv: np.ndarray, window: int,
           tol: float, k_max: int):
    """Per tx: chain depth (hops of similar-valued money ending here), value retention from
    the chain start, and whether the chain (<= k_max hops) returns to the receiver."""
    n = len(t)
    in_idx = pd.Series(np.arange(n)).groupby(rcv).indices
    pred = np.full(n, -1)
    for k in range(n):
        prev = in_idx.get(snd[k])
        if prev is None:
            continue
        lo = np.searchsorted(t[prev], t[k] - window, "left")
        hi = np.searchsorted(t[prev], t[k], "left")
        for j in prev[lo:hi][::-1]:  # latest similar-valued receipt first
            if abs(v[j] - v[k]) <= tol * v[j]:
                pred[k] = j
                break
    depth, start_v = np.ones(n), v.copy()
    for k in range(n):  # time-sorted: predecessors come first
        if pred[k] >= 0:
            depth[k] = depth[pred[k]] + 1
            start_v[k] = start_v[pred[k]]
    cycle = np.zeros(n, dtype=bool)
    for k in range(n):
        j, hops, path = pred[k], 1, [k]
        while j >= 0 and hops < k_max:
            path.append(j)
            if snd[j] == rcv[k]:
                cycle[path] = True
                break
            j, hops = pred[j], hops + 1
    return depth, v / start_v, cycle


def build(tx: pd.DataFrame, legs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Component size/span/rate/wallets at each window; chain depth, retention, cycles."""
    mcfg = cfg["features"]["motif"]
    tx = tx.sort_values(["ts", "tx_id"], kind="mergesort").reset_index(drop=True)
    t, v = tx["ts"].to_numpy(), tx["value_usd"].to_numpy()
    snd, rcv = tx["sender"].to_numpy(), tx["receiver"].to_numpy()
    f = pd.DataFrame(index=pd.Index(tx["tx_id"], name="tx_id"))

    for name, w in mcfg["windows_secs"].items():
        comp = components(t, v, snd, rcv, int(w), min(int(w), int(mcfg["burst_window_secs"])),
                          None if w <= mcfg["burst_window_secs"] else mcfg["chain_tol"])
        g = pd.DataFrame({"c": comp, "t": t, "s": snd, "r": rcv, "v": v}).groupby("c")
        size = g["t"].transform("size").to_numpy()
        span = (g["t"].transform("max") - g["t"].transform("min")).to_numpy()
        wallets = g.apply(lambda d: len(set(d["s"]) | set(d["r"])), include_groups=False)
        f[f"motif_size_{name}"] = size
        f[f"motif_log_span_{name}"] = np.log1p(span)
        f[f"motif_rate_{name}"] = size / (span / 60 + 1)  # txs per minute
        f[f"motif_wallets_{name}"] = pd.Series(comp).map(wallets).to_numpy()

    depth, retention, cycle = chains(t, v, snd, rcv, int(mcfg["chain_window_secs"]),
                                     mcfg["chain_tol"], mcfg["cycle_max_hops"])
    f["motif_chain_depth"] = depth
    f["motif_chain_retention"] = retention
    f["motif_cycle_member"] = cycle
    return validate(f, PREFIX)
