"""Feature modules. Each exposes build(tx, legs, cfg) -> DataFrame indexed by tx_id."""
from src.features import account, flow, graph, motif, rules, shapes, starter, tx

MODULES = {"starter": starter, "tx": tx, "account": account, "flow": flow, "motif": motif, "shapes": shapes, "graph": graph, "rules": rules}
