PY := .venv/bin/python

.PHONY: setup sample features run test clean submit-check

setup:
	python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

sample:
	$(PY) scripts/make_sample_data.py

features:
	$(PY) -c "import logging; logging.basicConfig(level=logging.INFO); \
	from src.config import load_params, FEATURES_PATH; from src.data.load import load_raw; \
	from src.pipeline import build_features; cfg = load_params(); tx, a = load_raw(cfg); \
	FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True); \
	build_features(tx, a, cfg).to_parquet(FEATURES_PATH); print('wrote', FEATURES_PATH)"

run:
	./run.sh

test:
	$(PY) -m pytest -q

clean:
	rm -rf outputs/models outputs/predictions.csv outputs/run_meta.json \
	       data/processed/features.parquet .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

submit-check:
	$(PY) scripts/validate_submission.py
