PY := .venv/bin/python

.PHONY: setup sample cv run test drift zip clean submit-check

setup:
	python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

sample:
	$(PY) scripts/make_sample_data.py

cv:            ## grouped CV report on train + predictions for validation/ (no test needed)
	$(PY) -m src.pipeline --target validation --out outputs/validation/predictions.csv

run:           ## final: predictions for test/
	./run.sh

test:
	$(PY) -m pytest -q

drift:         ## per-feature train-vs-test distribution shift
	$(PY) -m src.pipeline --target test --no-cv
	$(PY) scripts/drift_check.py --other test

zip:           ## code ZIP for upload (tracked files only: no data, no venv)
	git archive --format=zip -o outputs/code.zip HEAD
	@ls -lh outputs/code.zip

clean:
	rm -rf outputs/models outputs/predictions.csv outputs/run_meta.json outputs/validation \
	       data/processed/*.parquet .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

submit-check:
	$(PY) scripts/validate_submission.py
