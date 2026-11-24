"""Run provisional PDE module examples in empty namespaces and standalone guide scripts.

Use the installed wheel's interpreter with -I to check a clean core environment.
The optional script paths run through their normal __main__ entry points.
"""
from __future__ import annotations

import argparse
import doctest
import importlib
from pathlib import Path
import runpy
import sys

MODULES = (
    "pde_solvers",
    "models.lsv.local_vol",
    "models.logsv.coefficients",
    "models.lsv",
    "models.lsv.simulation",
    "data.density",
    "valuation",
    "pricers.density_pricer",
    "pricers.local_vol_pde_pricer",
    "pricers.logsv_pde_pricer",
    "models.lsv.propagation",
    "fitters.lsv_projection",
    "pricers.lsv_mc_pricer",
    "fitters.dupire",
    "fitters.lsv_fixed_point",
    "fitters.lsv_parametric",
    "fitters.lsv_market",
    "fitters.lsv_market_pipeline",
)


def main() -> None:
    """Import the reviewed modules and execute each docstring without module globals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("scripts", nargs="*", type=Path)
    args = parser.parse_args()
    package = importlib.import_module("stochvolmodels")
    if args.installed and not Path(package.__file__).resolve().is_relative_to(Path(sys.prefix)):
        raise SystemExit(f"Expected an installed wheel under {sys.prefix}, got {package.__file__}")
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    count = 0
    for name in MODULES:
        module = importlib.import_module("stochvolmodels." + name)
        for test in finder.find(module, globs={}):
            if test.examples:
                count += 1
                runner.run(test)
    result = runner.summarize()
    if result.failed or result.attempted == 0 or count < 9:
        raise SystemExit("PDE module examples failed or their coverage decreased")
    print(f"PDE examples passed: {len(MODULES)} module imports, "
          f"{count} independent docstrings, {result.attempted} examples")
    for script in args.scripts:
        runpy.run_path(str(script.resolve()), run_name="__main__")
        print(f"Standalone guide passed: {script.name}")


if __name__ == "__main__":
    main()
