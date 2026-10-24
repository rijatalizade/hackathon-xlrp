"""Load one organiser split folder into the canonical (tx, legs) tables. Owner: P1.

The column contract is in ARCHITECTURE.md §2. Each split folder holds
transactions.parquet, transfers.parquet, market_rates.csv and (train only) labels.csv.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT

log = logging.getLogger(__name__)

XRP_TOKEN = "0x1111222211112222111122221111222211112222"
LABEL_META = {"family": "meta_family", "difficulty": "meta_difficulty",
              "scenario_id": "meta_scenario_id", "cohort_id": "meta_cohort_id"}


def split_dir(cfg: dict, name: str, sample: bool = False) -> Path:
    """Folder of split `name` (train | validation | test) from params.yaml `data`."""
    key = "sample_dir" if sample else "raw_dir"
    return ROOT / cfg["data"][key] / name


def _int(s: pd.Series) -> pd.Series:
    """Decimal/object integer column -> int64."""
    return pd.to_numeric(s.astype(str)).astype("int64")


def _asset(token: pd.Series, symbol: pd.Series) -> pd.Series:
    """'XRP' or 'issuer:CUR' (the key used in market_rates.csv)."""
    return pd.Series(np.where(token == XRP_TOKEN, "XRP", token + ":" + symbol), index=token.index)


def load_rates(d: Path) -> pd.DataFrame:
    """Hourly reference prices: columns hour (unix s), asset, price_usd."""
    r = pd.read_csv(d / "market_rates.csv")
    return pd.DataFrame({"hour": r["TIMESTAMP"].astype("int64"), "asset": r["ASSET"],
                         "price_usd": r["PRICE_MICRO_USD"] / 1e6})


def _value_usd(ts: pd.Series, asset: pd.Series, amount: pd.Series, rates: pd.DataFrame
               ) -> np.ndarray:
    """amount * price of `asset` in the hour containing `ts`."""
    key = pd.DataFrame({"hour": (ts // 3600 * 3600).to_numpy(), "asset": asset.to_numpy()})
    price = key.merge(rates, on=["hour", "asset"], how="left")["price_usd"]
    if price.isna().any():
        raise ValueError(f"{price.isna().sum()} legs without a market rate")
    return amount.to_numpy() * price.to_numpy()


def load_split(d: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one split folder. Returns (tx, legs) per the ARCHITECTURE.md §2 contract.
    `tx` has `label` and `meta_*` columns only if labels.csv exists."""
    d = Path(d)
    raw = pd.read_parquet(d / "transactions.parquet",
                          columns=["TRANSACTION_HASH", "TIMESTAMP", "ACCOUNT", "FEE",
                                   "EXTRA_TRANSACTION_FIELDS"])
    tr = pd.read_parquet(d / "transfers.parquet",
                         columns=["TRANSACTION_HASH", "TRANSFER_INDEX", "TRANSFER_TYPE",
                                  "FROM_ADDRESS", "TO_ADDRESS", "TOKEN_ADDRESS", "SYMBOL",
                                  "VALUE", "TIMESTAMP"])
    rates = load_rates(d)

    legs = pd.DataFrame({
        "tx_id": tr["TRANSACTION_HASH"].astype(str),
        "leg": _int(tr["TRANSFER_INDEX"]),
        "from_addr": tr["FROM_ADDRESS"].astype(str),
        "to_addr": tr["TO_ADDRESS"].astype(str),
        "asset": _asset(tr["TOKEN_ADDRESS"].astype(str), tr["SYMBOL"].astype(str)),
        "amount": tr["VALUE"].astype(float),
        "transfer_type": tr["TRANSFER_TYPE"].astype(str),
    })
    legs["value_usd"] = _value_usd(_int(tr["TIMESTAMP"]), legs["asset"], legs["amount"], rates)
    legs = legs.sort_values(["tx_id", "leg"], kind="mergesort").reset_index(drop=True)

    g = legs.groupby("tx_id", sort=False)
    first, last = g.first(), g.last()
    mid = legs[legs["leg"] > 0].groupby("tx_id", sort=False)["from_addr"].agg(";".join)

    tx = pd.DataFrame({
        "tx_id": raw["TRANSACTION_HASH"].astype(str),
        "ts": _int(raw["TIMESTAMP"]),
        "sender": raw["ACCOUNT"].astype(str),
        "receiver": raw["EXTRA_TRANSACTION_FIELDS"].map(lambda s: json.loads(s)["Destination"]),
        "fee": raw["FEE"].astype(float),
    })
    if tx["tx_id"].duplicated().any():
        raise ValueError("duplicate TRANSACTION_HASH")
    tx["timestamp"] = pd.to_datetime(tx["ts"], unit="s", utc=True)
    idx = tx["tx_id"]
    tx["asset_out"] = last["asset"].reindex(idx).to_numpy()
    tx["amount_out"] = last["amount"].reindex(idx).to_numpy()
    tx["value_usd"] = last["value_usd"].reindex(idx).to_numpy()
    tx["asset_in"] = first["asset"].reindex(idx).to_numpy()
    tx["amount_in"] = first["amount"].reindex(idx).to_numpy()
    tx["value_in_usd"] = first["value_usd"].reindex(idx).to_numpy()
    tx["n_legs"] = g.size().reindex(idx).fillna(0).astype(int).to_numpy()
    tx["is_conversion"] = (tx["asset_in"] != tx["asset_out"]).astype(int)
    tx["providers"] = mid.reindex(idx).fillna("").to_numpy()
    if tx["value_usd"].isna().any():
        raise ValueError(f"{tx['value_usd'].isna().sum()} transactions without transfer legs")

    labels_path = d / "labels.csv"
    if labels_path.exists():
        lab = pd.read_csv(labels_path).rename(columns={"TRANSACTION_HASH": "tx_id",
                                                       "is_suspicious": "label", **LABEL_META})
        keep = ["tx_id", "label"] + [c for c in LABEL_META.values() if c in lab]
        tx = tx.merge(lab[keep], on="tx_id", how="left")
        if tx["label"].isna().any():
            raise ValueError(f"{tx['label'].isna().sum()} transactions without a label")

    tx = tx.sort_values(["timestamp", "tx_id"], kind="mergesort").reset_index(drop=True)
    log.info("Loaded %s: %d txs, %d legs%s", d, len(tx), len(legs),
             f", {int(tx['label'].sum())} suspicious" if "label" in tx else "")
    return tx, legs


def feature_view(tx: pd.DataFrame) -> pd.DataFrame:
    """`tx` as feature modules may see it: no label, no meta_* columns."""
    return tx.drop(columns=[c for c in tx.columns if c == "label" or c.startswith("meta_")])


def detect_mode(tx: pd.DataFrame) -> str:
    """'supervised' if `tx` (the train split) has labels of both classes."""
    if "label" not in tx:
        return "unsupervised"
    return "supervised" if tx["label"].nunique() >= 2 else "unsupervised"
