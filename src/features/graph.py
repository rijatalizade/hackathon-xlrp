"""Graph features from the sender -> receiver network (prefix `graph_`). Owner: P3.

The graph is built from all txs of the split (labels never used), so these features
use the whole week, not point-in-time. Always built within ONE split (wallets are
disjoint between splits).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.features.base import validate

PREFIX = "graph_"
# Time-respecting chains, cycles and fan-out/in bursts live in src/features/motif.py.


def build_graph(tx: pd.DataFrame) -> nx.DiGraph:
    """DiGraph with one edge per (sender, receiver) pair; attrs `weight` (count), `amount`."""
    edges = (tx.groupby(["sender", "receiver"], sort=True)["value_usd"]
               .agg(weight="size", amount="sum").reset_index())
    G = nx.DiGraph()
    G.add_nodes_from(sorted(set(tx["sender"]) | set(tx["receiver"])))
    G.add_edges_from((r.sender, r.receiver, {"weight": int(r.weight), "amount": float(r.amount)})
                     for r in edges.itertuples(index=False))
    return G


def node_features(G: nx.DiGraph) -> pd.DataFrame:
    """Per-account in/out degree (distinct counterparties), pagerank, weakly-CC size."""
    pr = nx.pagerank(G, alpha=0.85, weight="weight")
    cc_size = {}
    for comp in nx.weakly_connected_components(G):
        for node in comp:
            cc_size[node] = len(comp)
    nodes = list(G.nodes)
    return pd.DataFrame({
        "in_degree": [G.in_degree(n) for n in nodes],
        "out_degree": [G.out_degree(n) for n in nodes],
        "pagerank": [pr[n] for n in nodes],
        "weakly_cc_size": [cc_size[n] for n in nodes],
    }, index=pd.Index(nodes, name="acct"))


def neighbour_mean_score(G: nx.DiGraph, tx: pd.DataFrame, prior: pd.Series) -> pd.Series:
    """TODO(P3): second-pass feature. Given a prior score per tx_id (e.g. first-pass
    model OOF/test scores), return the mean prior score of txs touching the sender's and
    receiver's neighbours. Index = tx_id. Must not use labels directly."""
    raise NotImplementedError


def build(tx: pd.DataFrame, legs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Sender/receiver node features joined onto each tx."""
    G = build_graph(tx)
    nf = node_features(G)
    parts = []
    for col, tag in [("sender", "s"), ("receiver", "r")]:
        side = nf.reindex(tx[col].to_numpy())
        side.index = pd.Index(tx["tx_id"].to_numpy(), name="tx_id")
        parts.append(side.add_prefix(f"graph_{tag}_"))
    f = pd.concat(parts, axis=1)
    f["graph_s_out_r_in_ratio"] = (f["graph_s_out_degree"]
                                   / f["graph_r_in_degree"].replace(0, np.nan))
    return validate(f, PREFIX)
