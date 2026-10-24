import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def sample_data():
    """Ensure data/sample exists (it is generated, not organiser data)."""
    from src.config import SAMPLE_DIR
    if not (SAMPLE_DIR / "transactions.csv").exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        import make_sample_data
        make_sample_data.main()
