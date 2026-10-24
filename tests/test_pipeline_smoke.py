import subprocess
import sys
import time

from tests.conftest import ROOT


def test_sample_run_end_to_end(tmp_path):
    out = tmp_path / "predictions.csv"
    t = time.time()
    subprocess.run([sys.executable, "-m", "src.pipeline", "--sample", "--mode", "auto",
                    "--out", str(out)], cwd=ROOT, check=True)
    assert time.time() - t < 60
    res = subprocess.run([sys.executable, "scripts/validate_submission.py", "--sample",
                          "--pred", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
