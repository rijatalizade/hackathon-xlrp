import subprocess
import sys

from tests.conftest import ROOT


def test_two_runs_byte_identical(tmp_path):
    outs = []
    for i in range(2):
        out = tmp_path / f"pred_{i}.csv"
        subprocess.run([sys.executable, "-m", "src.pipeline", "--sample", "--no-cv",
                        "--out", str(out)], cwd=ROOT, check=True, capture_output=True)
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
