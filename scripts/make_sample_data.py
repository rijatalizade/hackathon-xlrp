"""Generate tiny SYNTHETIC split folders in the organiser format, for tests only.

THIS IS NOT THE ORGANISER DATA. Writes data/sample/{train,test}/ with
transactions.parquet, transfers.parquet, market_rates.csv (+ labels.csv for train),
using only the columns src/data/load.py reads. Planted suspicious scenarios (each with
its own scenario_id): structuring, pass-through mules, 4-hop chains, 3-cycles.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import SAMPLE_DIR, SEED  # noqa: E402
from src.data.load import XRP_TOKEN  # noqa: E402

WEEK = 7 * 86400
PRICES = {"XRP": 0.65, "USD": 1.0, "EUR": 1.1}


def _addr(rng, n):
    alphabet = list("rpshnaf39wBUDNEGHJKLM4PQRST7VWXYZ2bcdeCg65jkm8oFqi1tuvAxyz")
    return ["r" + "".join(rng.choice(alphabet, 24)) for _ in range(n)]


def make_split(seed: int, t0: int, n_accounts: int = 300, n_normal: int = 2500):
    rng = np.random.default_rng(seed)
    accts = _addr(rng, n_accounts)
    issuer, provider = _addr(rng, 2)
    rows = []  # (t, sender, receiver, usd, currency, label, scenario)

    def add(t, s, r, usd, label=0, scen=None, cur=None):
        cur = cur or rng.choice(["XRP", "USD", "EUR"], p=[0.6, 0.25, 0.15])
        rows.append((int(t0 + t), s, r, float(usd), cur, label, scen))

    for _ in range(n_normal):
        s, r = rng.choice(n_accounts, 2, replace=False)
        add(rng.uniform(0, WEEK), accts[s], accts[r], rng.lognormal(4, 1.2))

    pool = iter(rng.permutation(n_accounts))
    pick = lambda: accts[next(pool)]  # noqa: E731
    sid = iter(range(10_000))
    for _ in range(3):  # structuring
        s, t, k = pick(), rng.uniform(0, WEEK - 8 * 3600), f"S{next(sid)}"
        for _ in range(8):
            add(t + rng.uniform(0, 8 * 3600), s, accts[rng.integers(n_accounts)],
                rng.uniform(860, 995), 1, k)
    for _ in range(4):  # mules
        mule, dest, k = pick(), pick(), f"S{next(sid)}"
        for _ in range(4):
            t, v = rng.uniform(0, WEEK - 3600), rng.uniform(200, 2000)
            add(t, accts[rng.integers(n_accounts)], mule, v, 1, k)
            add(t + rng.uniform(60, 900), mule, dest, v * rng.uniform(0.9, 0.95), 1, k)
    for _ in range(5):  # 4-hop chains
        chain, k = [pick() for _ in range(5)], f"S{next(sid)}"
        t, v = rng.uniform(0, WEEK - 3600), rng.uniform(500, 3000)
        for a, b in zip(chain, chain[1:]):
            t += rng.uniform(60, 600)
            v *= rng.uniform(0.98, 1.0)
            add(t, a, b, v, 1, k)
    for _ in range(5):  # 3-cycles
        cyc, k = [pick() for _ in range(3)], f"S{next(sid)}"
        t, v = rng.uniform(0, WEEK - 3600), rng.uniform(300, 3000)
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            t += rng.uniform(120, 900)
            add(t, a, b, v, 1, k)

    df = pd.DataFrame(rows, columns=["ts", "sender", "receiver", "usd", "cur", "label", "scen"])
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    df["hash"] = [hashlib.sha256(f"{seed}-{i}".encode()).hexdigest().upper() for i in range(len(df))]

    def asset(cur):
        return (XRP_TOKEN, "XRP", "XRP") if cur == "XRP" else (issuer, cur, f"{issuer}:{cur}")

    txs, legs = [], []
    for r in df.itertuples(index=False):
        tok, sym, _ = asset(r.cur)
        units = r.usd / PRICES[r.cur]
        delivered = (json.dumps(str(int(units * 1e6))) if r.cur == "XRP" else
                     json.dumps({"currency": r.cur, "issuer": issuer, "value": f"{units:.6f}"}))
        txs.append({"TRANSACTION_HASH": r.hash, "TIMESTAMP": r.ts, "ACCOUNT": r.sender,
                    "FEE": 0.00001, "DELIVERED_AMOUNT": delivered,
                    "EXTRA_TRANSACTION_FIELDS": json.dumps({"Destination": r.receiver})})
        if rng.random() < 0.3:  # conversion via the provider: pay XRP, deliver currency
            src = "USD" if r.cur == "XRP" else "XRP"
            stok, ssym, _ = asset(src)
            legs.append((r.hash, 0, "path_payment", r.sender, provider, stok, ssym,
                         r.usd * 1.01 / PRICES[src], r.ts))
            legs.append((r.hash, 1, "path_payment", provider, r.receiver, tok, sym, units, r.ts))
        else:
            legs.append((r.hash, 0, "direct_payment", r.sender, r.receiver, tok, sym, units, r.ts))

    transfers = pd.DataFrame(legs, columns=["TRANSACTION_HASH", "TRANSFER_INDEX", "TRANSFER_TYPE",
                                            "FROM_ADDRESS", "TO_ADDRESS", "TOKEN_ADDRESS",
                                            "SYMBOL", "VALUE", "TIMESTAMP"])
    hours = range(t0 // 3600 * 3600, t0 + WEEK + 3600, 3600)
    rates = pd.DataFrame([(h, "XRP" if c == "XRP" else f"{issuer}:{c}", int(p * 1e6))
                          for h in hours for c, p in PRICES.items()],
                         columns=["TIMESTAMP", "ASSET", "PRICE_MICRO_USD"])
    labels = pd.DataFrame({"TRANSACTION_HASH": df["hash"], "is_suspicious": df["label"],
                           "family": np.where(df["label"] == 1, "planted", "background"),
                           "difficulty": np.where(df["label"] == 1, "easy", "normal"),
                           "scenario_id": df["scen"], "cohort_id": np.nan})
    return pd.DataFrame(txs), transfers, rates, labels


def main() -> None:
    for name, seed, t0, n_acc, n_norm in [("train", SEED, 1_788_220_800, 300, 2500),
                                          ("test", SEED + 1, 1_789_862_400, 250, 2000)]:
        d = SAMPLE_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        txs, transfers, rates, labels = make_split(seed, t0, n_acc, n_norm)
        txs.to_parquet(d / "transactions.parquet", index=False)
        transfers.to_parquet(d / "transfers.parquet", index=False)
        rates.to_csv(d / "market_rates.csv", index=False)
        if name == "train":
            labels.to_csv(d / "labels.csv", index=False)
        print(f"Wrote SYNTHETIC sample {d}: {len(txs)} txs, {int(labels.is_suspicious.sum())} "
              f"planted suspicious")


if __name__ == "__main__":
    main()
