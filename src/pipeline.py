"""End-to-end: load -> features -> train -> predict -> predictions.csv. Owner: P1.

    python -m src.pipeline --mode auto|supervised|unsupervised [--sample] [--skip-graph]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURES_PATH, PREDICTIONS_PATH, load_params
from src.data.load import detect_mode, load_raw
from src.features import MODULES
from src.model import cv, ensemble, predict, train

log = logging.getLogger("pipeline")


def build_features(tx: pd.DataFrame, accounts, cfg: dict, skip: set[str] = frozenset()
                   ) -> pd.DataFrame:
    """Run every feature module; a module that raises is logged and skipped so one
    broken module never blocks a submission. Result is aligned to tx['tx_id']."""
    tx = tx.drop(columns=["label"], errors="ignore")  # labels never reach features
    parts = []
    for name, module in MODULES.items():
        if name in skip:
            log.info("features/%s: skipped", name)
            continue
        t = time.time()
        try:
            f = module.build(tx, accounts, cfg)
            parts.append(f.reindex(tx["tx_id"]))
            log.info("features/%s: %d cols in %.1fs", name, f.shape[1], time.time() - t)
        except Exception:
            log.exception("features/%s FAILED — continuing without it", name)
    if not parts:
        raise RuntimeError("Every feature module failed")
    return pd.concat(parts, axis=1).astype(float)


def run(mode: str = "auto", sample: bool = False, skip_graph: bool = False,
        data_dir: Path | None = None, out: Path = PREDICTIONS_PATH) -> dict:
    cfg = load_params()
    seed, n_jobs = cfg["seed"], cfg.get("n_jobs", 1)
    tx, accounts = load_raw(cfg, sample=sample, data_dir=data_dir)
    mode = detect_mode(tx) if mode == "auto" else mode
    log.info("Mode: %s", mode)

    X = build_features(tx, accounts, cfg, skip={"graph"} if skip_graph else set())
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(FEATURES_PATH)

    is_train = (tx["split"] == "train").to_numpy()
    is_test = (tx["split"] == "test").to_numpy()
    X_train, X_test = X[is_train], X[is_test]

    train.save(train.train_iforest(X, seed, cfg["models"]["iforest"], n_jobs), "iforest")

    report = {}
    has_labels = "label" in tx and tx.loc[is_train, "label"].notna().any()
    if has_labels:
        lab = tx.loc[is_train, "label"].notna().to_numpy()
        y = tx.loc[is_train, "label"][lab].astype(int).reset_index(drop=True)
        Xl = X_train[lab]
        comp = {"iforest": train.score(train.load("iforest"), Xl)}
        if "rule_score" in Xl:
            comp["rules"] = Xl["rule_score"].fillna(0).to_numpy()
        if mode == "supervised":
            comp["lgbm"], _ = cv.oof_lgbm(Xl, y, cfg)
            lab_idx = np.flatnonzero(is_train)[lab]
            train.save(train.train_lgbm(X.iloc[lab_idx], y, cfg["models"]["lgbm"], seed,
                                        n_jobs), "lgbm")
        comp["ensemble"] = ensemble.rank_average(comp, cfg["ensemble"][mode])
        report = {k: cv.metrics(y, s) for k, s in comp.items()}

    sub = predict.predict(X_test, cfg, mode, out_path=out)
    meta = {"mode": mode, "sample": sample, "data_dir": str(data_dir or ""),
            "n_test": len(sub), "n_features": X.shape[1], "cv": report,
            "primary_metric": cfg["cv"]["metric"]}
    (out.parent / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    log.info("Wrote %d predictions to %s", len(sub), out)
    for name, m in report.items():
        log.info("CV %-9s roc_auc=%.4f  average_precision=%.4f", name, m["roc_auc"],
                 m["average_precision"])
    if "ensemble" in report:
        print(f"CV {cfg['cv']['metric']} (ensemble, train): "
              f"{report['ensemble'][cfg['cv']['metric']]:.4f}")
    return meta


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="auto", choices=["auto", "supervised", "unsupervised"])
    p.add_argument("--sample", action="store_true", help="use data/sample instead of data/raw")
    p.add_argument("--skip-graph", action="store_true")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=PREDICTIONS_PATH)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(a.mode, a.sample, a.skip_graph, a.data_dir, a.out)


if __name__ == "__main__":
    main()
