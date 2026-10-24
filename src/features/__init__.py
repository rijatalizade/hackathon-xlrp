"""Feature modules. Each exposes build(tx, legs, cfg) -> DataFrame indexed by tx_id."""
from src.features import account, flow, graph, motif, rules, starter, tx

MODULES = {"starter": starter, "tx": tx, "account": account, "flow": flow, "motif": motif, "graph": graph, "rules": rules}
