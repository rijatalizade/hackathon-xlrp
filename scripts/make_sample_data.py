"""Generate a tiny SYNTHETIC dataset so the pipeline runs without organiser data.

THIS IS NOT THE ORGANISER DATA. It only exists for smoke tests and development.
~5,000 txs, ~400 accounts, 48h window, with planted typologies (label=1):
structuring sender, two pass-through mules, a 4-hop layering chain, a 3-cycle.
Writes data/sample/transactions.csv with `label` (train rows only) and `split`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import SAMPLE_DIR, SEED  # noqa: E402

N_ACCOUNTS = 400
N_NORMAL = 4800
WINDOW_SECS = 48 * 3600
T0 = pd.Timestamp("2026-10-01T00:00:00Z")
CURRENCIES = ["XRP", "USD", "EUR", "BTC"]
TX_TYPES = ["Payment", "OfferCreate", "TrustSet"]


def _accounts(rng: np.random.Generator, n: int) -> list[str]:
    alphabet = list("rpshnaf39wBUDNEGHJKLM4PQRST7VWXYZ2bcdeCg65jkm8oFqi1tuvAxyz")
    return ["r" + "".join(rng.choice(alphabet, 24)) for _ in range(n)]


def make(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    accts = _accounts(rng, N_ACCOUNTS)
    rows = []

    def add(t, s, r, amt, label, cur="XRP", typ="Payment"):
        rows.append((float(t), s, r, round(float(amt), 2), cur, typ, label))

    # Normal background traffic.
    for _ in range(N_NORMAL):
        s, r = rng.choice(N_ACCOUNTS, 2, replace=False)
        cur = rng.choice(CURRENCIES, p=[0.8, 0.1, 0.07, 0.03])
        typ = rng.choice(TX_TYPES, p=[0.85, 0.12, 0.03])
        amt = rng.lognormal(5.5, 1.5)
        if rng.random() < 0.05:
            amt = round(amt, -2)
        add(rng.uniform(0, WINDOW_SECS), accts[s], accts[r], amt, 0, cur, typ)

    pool = iter(rng.permutation(N_ACCOUNTS))
    pick = lambda: accts[next(pool)]  # noqa: E731 — unique accounts per typology

    # Structuring: one sender, many just-below-10k payments within a few hours.
    smurf, t = pick(), rng.uniform(0, WINDOW_SECS - 6 * 3600)
    for _ in range(20):
        add(t + rng.uniform(0, 6 * 3600), smurf, accts[rng.integers(N_ACCOUNTS)],
            rng.uniform(8600, 9990), 1)

    # Pass-through mules: receive, then forward ~90-95% within minutes.
    for _ in range(2):
        mule, dest = pick(), pick()
        for _ in range(8):
            t, amt = rng.uniform(0, WINDOW_SECS - 3600), rng.uniform(2000, 20000)
            add(t, accts[rng.integers(N_ACCOUNTS)], mule, amt, 1)
            add(t + rng.uniform(60, 900), mule, dest, amt * rng.uniform(0.9, 0.95), 1)

    # Layering chain: 4 hops of the same-ish amount within minutes; repeated.
    chain = [pick() for _ in range(5)]
    for _ in range(4):
        t, amt = rng.uniform(0, WINDOW_SECS - 3600), rng.uniform(5000, 50000)
        for a, b in zip(chain, chain[1:]):
            t += rng.uniform(60, 600)
            amt *= rng.uniform(0.98, 1.0)
            add(t, a, b, amt, 1)

    # Round-trip 3-cycle: A -> B -> C -> A, with a currency hop; repeated.
    cyc = [pick() for _ in range(3)]
    for _ in range(4):
        t, amt = rng.uniform(0, WINDOW_SECS - 3600), rng.uniform(3000, 30000)
        for (a, b), cur in zip(zip(cyc, cyc[1:] + cyc[:1]), ["XRP", "USD", "XRP"]):
            t += rng.uniform(120, 900)
            add(t, a, b, amt, 1, cur, "Payment")

    df = pd.DataFrame(rows, columns=["t", "sender", "receiver", "amount", "currency",
                                     "tx_type", "label"])
    df = df.sort_values("t", kind="mergesort").reset_index(drop=True)
    df["timestamp"] = (T0 + pd.to_timedelta(df["t"], unit="s")).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    df["tx_id"] = [f"tx{i:06d}" for i in rng.permutation(len(df))]
    df["fee"] = np.round(rng.uniform(0.00001, 0.0002, len(df)), 6)
    df["split"] = np.where(rng.random(len(df)) < 0.7, "train", "test")
    df["label"] = df["label"].where(df["split"] == "train")  # test labels hidden
    cols = ["tx_id", "timestamp", "sender", "receiver", "amount", "currency", "tx_type",
            "fee", "label", "split"]
    return df[cols]


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    df = make()
    out = SAMPLE_DIR / "transactions.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC sample txs ({int(df['label'].sum())} positive in train) "
          f"to {out}")


if __name__ == "__main__":
    main()
