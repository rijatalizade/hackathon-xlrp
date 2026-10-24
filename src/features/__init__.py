"""Feature modules. Each exposes build(tx, legs, cfg) -> DataFrame indexed by tx_id."""
from src.features import account, graph, rules, starter, tx

MODULES = {"starter": starter, "tx": tx, "account": account, "graph": graph, "rules": rules}
