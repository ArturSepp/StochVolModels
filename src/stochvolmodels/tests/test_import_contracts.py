import json
import subprocess
import sys


def test_import_does_not_mutate_numpy_random_state() -> None:
    script = """
import json
import numpy as np

np.random.seed(12345)
expected = np.random.random()
np.random.seed(12345)
import stochvolmodels
actual = np.random.random()
print(json.dumps({"expected": expected, "actual": actual}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["actual"] == result["expected"]
