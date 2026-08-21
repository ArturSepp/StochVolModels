import json
import subprocess
import sys

import pytest


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


@pytest.mark.parametrize("module_name", ["bsm", "bachelier"])
def test_removed_analytic_module_paths_are_not_importable(module_name: str) -> None:
    """The 2.0 release intentionally provides no old-path compatibility facade."""
    with pytest.raises(ModuleNotFoundError):
        __import__(f"stochvolmodels.pricers.analytic.{module_name}")
