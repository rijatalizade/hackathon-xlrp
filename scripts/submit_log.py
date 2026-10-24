"""Append a leaderboard submission to outputs/leaderboard_log.csv. Owner: P1.

    python scripts/submit_log.py --name v1 --cv 0.91 --lb 0.89 --notes "iforest + rules"
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import LEADERBOARD_LOG, ROOT  # noqa: E402

HEADER = ["timestamp", "commit", "submission_name", "cv_score", "lb_score", "notes"]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--cv", type=float, default=None)
    p.add_argument("--lb", type=float, default=None)
    p.add_argument("--notes", default="")
    a = p.parse_args()
    new = not LEADERBOARD_LOG.exists() or LEADERBOARD_LOG.stat().st_size == 0
    row = [datetime.now(timezone.utc).isoformat(timespec="seconds"), git_commit(), a.name,
           "" if a.cv is None else a.cv, "" if a.lb is None else a.lb, a.notes]
    with open(LEADERBOARD_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerow(row)
    print("Logged:", dict(zip(HEADER, row)))


if __name__ == "__main__":
    main()
