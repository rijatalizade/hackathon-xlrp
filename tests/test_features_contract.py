import pandas as pd
import pytest

from src.config import load_params
from src.data.load import feature_view, load_split, split_dir
from src.features import MODULES


@pytest.fixture(scope="module")
def data():
    cfg = load_params()
    tx, legs = load_split(split_dir(cfg, "train", sample=True))
    return feature_view(tx), legs, cfg


@pytest.mark.parametrize("name", list(MODULES))
def test_contract(name, data):
    tx, legs, cfg = data
    f = MODULES[name].build(tx, legs, cfg)
    prefix = MODULES[name].PREFIX
    assert isinstance(f, pd.DataFrame)
    assert f.index.name == "tx_id"
    assert f.index.is_unique
    assert set(f.index) == set(tx["tx_id"])
    assert all(c.startswith(prefix) for c in f.columns), f"columns must start with {prefix}"
    assert not f.columns.has_duplicates
    assert not f.isna().all().any(), "no all-NaN columns"
    assert all(pd.api.types.is_numeric_dtype(t) for t in f.dtypes), "features must be numeric"


def test_no_cross_module_column_clash(data):
    tx, legs, cfg = data
    cols = [c for m in MODULES.values() for c in m.build(tx, legs, cfg).columns]
    assert len(cols) == len(set(cols))
