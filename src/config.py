"""Global constants and config loaders. SEED is the single source of randomness."""
from __future__ import annotations

from pathlib import Path

import yaml

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"
LEADERBOARD_LOG = OUTPUTS_DIR / "leaderboard_log.csv"


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_params() -> dict:
    """Return config/params.yaml as a dict, with `seed` injected."""
    params = _read_yaml(CONFIG_DIR / "params.yaml")
    params["seed"] = SEED
    return params


def load_schema() -> dict:
    """Return config/schema.yaml as a dict."""
    return _read_yaml(CONFIG_DIR / "schema.yaml")
