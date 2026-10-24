"""End-to-end: load splits -> features (per split) -> grouped CV -> train -> predict.

Every artifact of a run (features_*.parquet, models/, oof_train.parquet, run_meta.json)
is written next to --out, so runs (incl. test runs on the sample) never overwrite each other.

    python -m src.pipeline [--target test|validation] [--sample] [--only starter,tx]
                           [--skip graph] [--repeats N] [--no-cv]
Owner: P1.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from src.config import PREDICTIONS_PATH, load_params
from src.data.load import detect_mode, feature_view, load_split, split_dir
from src.features import MODULES
from src.model import cv, ensemble, links, predict, train

log = logging.getLogger("pipeline")
ALL_MODULES = list(MODULES) + ["links"]  # "links" = learned scenario links (src/model/links.py)


def build_features(tx: pd.DataFrame, legs: pd.DataFrame, cfg: dict,
                   modules: list[str] | None = None) -> pd.DataFrame:
    """Run feature modules on ONE split; a module that raises is logged and skipped so
    one broken module never blocks a submission. Result is aligned to tx['tx_id']."""
    tx = feature_view(tx)  # labels / meta never reach features
    parts = []
    for name in [m for m in (modules or list(MODULES)) if m in MODULES]:
        t = time.time()
        try:
            f = MODULES[name].build(tx, legs, cfg)
            parts.append(f.reindex(tx["tx_id"]))
            log.info("features/%s: %d cols in %.1fs", name, f.shape[1], time.time() - t)
        except Exception:
            log.exception("features/%s FAILED — continuing without it", name)
    if not parts:
        raise RuntimeError("Every feature module failed")
    return pd.concat(parts, axis=1).astype(float)


def _lgbm_fit_score(cfg):
    def fit_score(X_fit, y_fit, X_chk):
        b = train.train_lgbm(X_fit, pd.Series(y_fit), cfg["models"]["lgbm"], cfg["seed"],
                             cfg.get("n_jobs", 1))
        return train.score(b, X_chk)
    return fit_score


def run(target: str = "test", sample: bool = False, modules: list[str] | None = None,
        repeats: int | None = None, do_cv: bool = True, out: Path = PREDICTIONS_PATH) -> dict:
    cfg = load_params()
    seed, n_jobs = cfg["seed"], cfg.get("n_jobs", 1)
    tr_tx, tr_legs = load_split(split_dir(cfg, "train", sample))
    tg_tx, tg_legs = load_split(split_dir(cfg, target, sample))
    mode = detect_mode(tr_tx)
    log.info("Mode: %s, target split: %s", mode, target)

    modules = modules or ALL_MODULES
    X_tr = build_features(tr_tx, tr_legs, cfg, modules)
    X_tg = build_features(tg_tx, tg_legs, cfg, modules)
    if "links" in modules and "meta_scenario_id" in tr_tx:
        t = time.time()
        link_model, L_tr = links.fit(tr_tx, cfg)  # train rows: out-of-fold link probabilities
        L_tg = links.transform(link_model, feature_view(tg_tx), cfg)
        X_tr = pd.concat([X_tr, L_tr.reindex(X_tr.index)], axis=1)
        X_tg = pd.concat([X_tg, L_tg.reindex(X_tg.index)], axis=1)
        log.info("features/links: %d cols in %.1fs", L_tr.shape[1], time.time() - t)
    missing = [c for c in X_tr.columns if c not in X_tg.columns]
    if missing:
        log.warning("Target split lacks %d train features (filled NaN): %s", len(missing), missing)
    X_tg = X_tg.reindex(columns=X_tr.columns)
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    X_tr.to_parquet(run_dir / "features_train.parquet")
    X_tg.to_parquet(run_dir / f"features_{target}.parquet")

    train.save(train.train_iforest(X_tr, seed, cfg["models"]["iforest"], n_jobs), "iforest",
               run_dir)

    report = {}
    if "label" in tr_tx and do_cv:
        y = tr_tx["label"].astype(int).to_numpy()
        comp = {"iforest": train.score(train.load("iforest", run_dir), X_tr)}
        if "rule_score" in X_tr:
            comp["rules"] = X_tr["rule_score"].fillna(0).to_numpy()
        if mode == "supervised":
            res = cv.repeated_cv(X_tr, y, tr_tx, cfg, _lgbm_fit_score(cfg), repeats)
            comp["lgbm"] = res["oof"]
            report["lgbm"] = res["summary"]
            oof = tr_tx[["tx_id", "label"] + [c for c in tr_tx if c.startswith("meta_")]].copy()
            oof["oof"] = res["oof"]
            oof.to_parquet(run_dir / "oof_train.parquet")  # for error analysis (P4)
        for name in ["iforest", "rules"]:
            if name in comp:
                report[name] = cv.evaluate(y, comp[name], tr_tx, cfg)
        report["ensemble"] = cv.evaluate(y, ensemble.rank_average(comp, cfg["ensemble"][mode]),
                                         tr_tx, cfg)
    if mode == "supervised":
        train.save(train.train_lgbm(X_tr, tr_tx["label"].astype(int), cfg["models"]["lgbm"],
                                    seed, n_jobs), "lgbm", run_dir)

    sub = predict.predict(X_tg, cfg, mode, out_path=out)
    meta = {"mode": mode, "sample": sample, "target": target, "n_rows": len(sub),
            "n_features": X_tr.shape[1], "modules": modules, "cv": report}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    log.info("Wrote %d predictions for %s to %s", len(sub), target, out)
    if report:
        print("\nGrouped CV on train (AP x100; per-difficulty = those positives vs all negatives)")
        for name, s in report.items():
            print("  " + cv.format_summary(name, s))
    return meta


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="test", choices=["test", "validation"])
    p.add_argument("--sample", action="store_true", help="use data/sample instead of data/raw")
    p.add_argument("--only", default=None, help="comma-separated feature modules to use")
    p.add_argument("--skip", default="", help="comma-separated feature modules to skip")
    p.add_argument("--repeats", type=int, default=None, help="CV repeats (default params.yaml)")
    p.add_argument("--no-cv", action="store_true")
    p.add_argument("--out", type=Path, default=PREDICTIONS_PATH)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    modules = a.only.split(",") if a.only else list(ALL_MODULES)
    modules = [m for m in modules if m not in set(filter(None, a.skip.split(",")))]
    unknown = set(modules) - set(ALL_MODULES)
    if unknown:
        p.error(f"unknown feature modules {sorted(unknown)}; choose from {ALL_MODULES}")
    run(a.target, a.sample, modules, a.repeats, not a.no_cv, a.out)


if __name__ == "__main__":
    main()
