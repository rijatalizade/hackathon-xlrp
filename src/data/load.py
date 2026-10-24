"""Load organiser files into canonical DataFrames. Owner: P1.

Canonical tx columns: tx_id, timestamp (UTC), sender, receiver, amount, and optionally
currency, tx_type, fee, dest_tag, memo, label, split. Unmapped columns become `extra_*`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DIR, SAMPLE_DIR, load_schema

log = logging.getLogger(__name__)

REQUIRED = ["tx_id", "timestamp", "sender", "receiver", "amount"]
RIPPLE_EPOCH_OFFSET = 946684800  # 2000-01-01T00:00:00Z in unix seconds


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _list_files(data_dir: Path) -> list[Path]:
    files = sorted(p for p in data_dir.iterdir() if p.suffix in {".csv", ".parquet"})
    if not files:
        raise FileNotFoundError(f"No .csv/.parquet files in {data_dir}")
    return files


def _rename(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Organiser names -> canonical; anything unmapped -> extra_<name>."""
    mapping = {v: k for k, v in (schema.get("columns") or {}).items() if v}
    renamed = {c: mapping.get(c, f"extra_{c}") for c in df.columns}
    unmapped = [c for c, new in renamed.items() if new.startswith("extra_")]
    if unmapped:
        log.info("Unmapped columns kept as extra_*: %s", unmapped)
    return df.rename(columns=renamed)


def _parse_timestamp(s: pd.Series, unit: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        if unit == "ripple":
            return pd.to_datetime(s + RIPPLE_EPOCH_OFFSET, unit="s", utc=True)
        return pd.to_datetime(s, unit=unit, utc=True)
    return pd.to_datetime(s, utc=True, format="mixed")


def load_raw(cfg: dict | None = None, sample: bool = False, data_dir: Path | None = None):
    """Read every CSV/parquet in data/raw (or data/sample) into canonical form.

    Returns (tx, accounts). `tx` is sorted by (timestamp, tx_id), has unique tx_id,
    UTC `timestamp` and a `split` column in {train, test}. `accounts` is None unless
    an accounts file is present.
    """
    schema = load_schema()
    data_dir = Path(data_dir) if data_dir else (SAMPLE_DIR if sample else RAW_DIR)
    names = schema.get("file_split") or {}
    acct_key = schema.get("accounts_file_contains", "account")
    label_key = schema.get("labels_file_contains", "label")

    tx_parts, accounts, labels = [], None, None
    for path in _list_files(data_dir):
        stem = path.stem.lower()
        df = _read_file(path)
        if acct_key and acct_key in stem:
            accounts = _rename(df, schema)
            log.info("Accounts file %s: %d rows", path.name, len(df))
            continue
        df = _rename(df, schema)
        if label_key and label_key in stem and set(df.columns) <= {"tx_id", "label"}:
            labels = df
            log.info("Labels file %s: %d rows", path.name, len(df))
            continue
        if "split" not in df.columns:
            for split_name, key in names.items():
                if key in stem:
                    df["split"] = split_name
        log.info("Tx file %s: %d rows, columns=%s", path.name, len(df), list(df.columns))
        tx_parts.append(df)

    tx = pd.concat(tx_parts, ignore_index=True)
    if labels is not None:
        tx = tx.drop(columns=["label"], errors="ignore").merge(labels, on="tx_id", how="left")

    missing = [c for c in REQUIRED if c not in tx.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}; edit config/schema.yaml")
    if "split" not in tx.columns:
        raise ValueError("No `split` column and no train/test file names; edit schema.yaml")

    tx["tx_id"] = tx["tx_id"].astype(str)
    if tx["tx_id"].duplicated().any():
        raise ValueError(f"{tx['tx_id'].duplicated().sum()} duplicate tx_id values")
    tx["timestamp"] = _parse_timestamp(tx["timestamp"], schema.get("timestamp_unit", "s"))
    tx["sender"] = tx["sender"].astype(str)
    tx["receiver"] = tx["receiver"].astype(str)
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
    tx["split"] = tx["split"].astype(str).str.strip().str.lower()
    if "label" in tx.columns:
        tx["label"] = pd.to_numeric(tx["label"], errors="coerce")

    tx = tx.sort_values(["timestamp", "tx_id"], kind="mergesort").reset_index(drop=True)
    log.info("Loaded %d txs from %s; schema found: %s", len(tx), data_dir, list(tx.columns))
    return tx, accounts


def split(tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) according to the `split` column."""
    train = tx[tx["split"] == "train"]
    test = tx[tx["split"] == "test"]
    if test.empty:
        raise ValueError("No test rows found (split == 'test')")
    return train, test


def detect_mode(tx: pd.DataFrame) -> str:
    """'supervised' if the train split has non-null labels with both classes, else 'unsupervised'."""
    if "label" not in tx.columns:
        return "unsupervised"
    y = tx.loc[tx["split"] == "train", "label"].dropna()
    return "supervised" if len(np.unique(y)) >= 2 else "unsupervised"
