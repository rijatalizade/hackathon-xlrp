"""Feature-module contract and shared helpers.

Every module in src/features exposes:

    def build(tx: pd.DataFrame, accounts: pd.DataFrame | None, cfg: dict) -> pd.DataFrame

returning a DataFrame indexed by tx_id (one row per tx in `tx`), all columns prefixed
with the module prefix, and no all-NaN columns. `tx` never contains `label`.
Finish every build() with `return validate(df, PREFIX)`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build(tx: pd.DataFrame, accounts: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    """The contract every feature module implements (see module docstring)."""
    raise NotImplementedError


def validate(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Enforce the contract: index == tx_id (unique), every column starts with `prefix`,
    no duplicate columns, bool -> int, all-NaN columns dropped."""
    df = df.copy()
    df.index.name = "tx_id"
    if df.index.has_duplicates:
        raise ValueError(f"{prefix}: duplicate tx_id in index")
    bad = [c for c in df.columns if not str(c).startswith(prefix)]
    if bad:
        raise ValueError(f"{prefix}: columns without prefix {bad}")
    if df.columns.has_duplicates:
        raise ValueError(f"{prefix}: duplicate columns")
    for c in df.columns[df.dtypes == bool]:
        df[c] = df[c].astype(np.int8)
    empty = df.columns[df.isna().all()].tolist()
    if empty:
        log.warning("%s: dropping all-NaN columns %s", prefix, empty)
        df = df.drop(columns=empty)
    return df


def to_seconds(ts: pd.Series) -> np.ndarray:
    """UTC datetime Series -> int64 unix seconds."""
    return ts.dt.as_unit("s").astype("int64").to_numpy()


def rolling_count(groups, secs, window_secs: int) -> np.ndarray:
    """Point-in-time count of rows in the same group within (t - window, t], counting
    rows up to and including the current one in input order (ties: earlier rows only).

    `groups` and `secs` are aligned 1-D arrays; result is aligned to the input order.
    Pass `-secs` to count forward in time instead.
    """
    secs = np.asarray(secs, dtype=np.int64)
    if len(secs) == 0:
        return np.zeros(0, dtype=np.int64)
    codes = pd.factorize(np.asarray(groups), sort=True)[0].astype(np.int64)
    order = np.lexsort((secs, codes))  # stable: ties keep input order
    rel = secs - secs.min()
    span = int(rel.max()) + window_secs + 1
    key = codes[order] * span + rel[order]
    left = np.searchsorted(key, key - window_secs, side="right")
    counts = np.arange(1, len(key) + 1) - left
    out = np.empty_like(counts)
    out[order] = counts
    return out
