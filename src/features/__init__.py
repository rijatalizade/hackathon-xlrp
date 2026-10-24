"""Feature modules. Each exposes build(tx, accounts, cfg) -> DataFrame indexed by tx_id."""
from src.features import account, graph, rules, tx

MODULES = {"tx": tx, "account": account, "graph": graph, "rules": rules}
