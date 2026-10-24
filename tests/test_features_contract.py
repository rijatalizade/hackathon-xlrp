import pandas as pd
import pytest

from src.config import load_params
from src.data.load import load_raw
from src.features import MODULES


@pytest.fixture(scope="module")
def data():
    tx, accounts = load_raw(sample=True)
    return tx.drop(columns=["label"]), accounts, load_params()


@pytest.mark.parametrize("name", list(MODULES))
def test_contract(name, data):
    tx, accounts, cfg = data
    f = MODULES[name].build(tx, accounts, cfg)
    prefix = MODULES[name].PREFIX
    assert isinstance(f, pd.DataFrame)
    assert f.index.name == "tx_id"
    assert f.index.is_unique
    assert set(f.index) == set(tx["tx_id"])
    assert len(f) == len(tx)
    assert all(c.startswith(prefix) for c in f.columns), f"columns must start with {prefix}"
    assert not f.columns.has_duplicates
    assert not f.isna().all().any(), "no all-NaN columns"


def test_no_cross_module_column_clash(data):
    tx, accounts, cfg = data
    cols = [c for m in MODULES.values() for c in m.build(tx, accounts, cfg).columns]
    assert len(cols) == len(set(cols))
