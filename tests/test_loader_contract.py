from src.config import load_params
from src.data.load import detect_mode, feature_view, load_split, split_dir

CONTRACT = ["tx_id", "timestamp", "ts", "sender", "receiver", "asset_out", "amount_out",
            "asset_in", "amount_in", "value_usd", "value_in_usd", "n_legs", "is_conversion",
            "providers", "fee"]


def test_split_matches_contract():
    tx, legs = load_split(split_dir(load_params(), "train", sample=True))
    assert all(c in tx for c in CONTRACT)
    assert tx["tx_id"].is_unique
    assert tx["timestamp"].is_monotonic_increasing
    assert tx["value_usd"].gt(0).all()
    assert set(legs["tx_id"]) == set(tx["tx_id"])
    assert detect_mode(tx) == "supervised"


def test_test_split_has_no_labels():
    tx, _ = load_split(split_dir(load_params(), "test", sample=True))
    assert "label" not in tx and not any(c.startswith("meta_") for c in tx)


def test_feature_view_hides_labels_and_meta():
    tx, _ = load_split(split_dir(load_params(), "train", sample=True))
    fv = feature_view(tx)
    assert "label" not in fv and not any(c.startswith("meta_") for c in fv)
