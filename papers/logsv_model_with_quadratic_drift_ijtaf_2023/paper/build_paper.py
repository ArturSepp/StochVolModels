"""Build the corrected IJTAF author manuscript without replacing the published PDF.

Use the repository interpreter and an installed pdfLaTeX distribution. Compilation
and auxiliary files stay in an operating-system temporary directory. The source
contains its bibliography inline and the historical author figures are bundled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


SOURCE_NAME = "lognormal_stoch_vol_final.tex"
OUTPUT_NAME = "lognormal_stoch_vol_final_corrected_20261029.pdf"
PUBLISHED_NAME = "log-normal-stochastic-volatility-model-with-quadratic-drift.pdf"


def build(output: Path | None = None) -> Path:
    """Compile the saved manuscript, check references and export a separate PDF."""
    source = Path(__file__).resolve().parent
    target = (output or source / OUTPUT_NAME).expanduser().resolve()
    if target == (source / PUBLISHED_NAME).resolve():
        raise ValueError("The published PDF is an immutable reference; choose another output.")
    compiler = shutil.which("pdflatex")
    if compiler is None:
        raise RuntimeError("pdfLaTeX is required; use an existing MiKTeX or TeX Live installation.")
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                             errors="replace", check=True).stdout
    options = ["-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape"]
    if "MiKTeX" in version:
        options.append("--disable-installer")
    manuscript = (source / SOURCE_NAME).read_text(encoding="utf-8")
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}", manuscript)
    for figure in figures:
        asset = (source / figure).resolve()
        if not asset.is_relative_to(source) or not asset.is_file():
            raise FileNotFoundError(f"Missing local manuscript figure: {figure}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="logsv-corrected-paper-") as directory:
        work = Path(directory)
        shutil.copy2(source / SOURCE_NAME, work / SOURCE_NAME)
        for figure in figures:
            destination = work / figure
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / figure, destination)
        for pass_number in range(1, 5):
            result = subprocess.run([compiler, *options, SOURCE_NAME], cwd=work,
                                    capture_output=True, text=True, errors="replace", check=False)
            if result.returncode:
                raise RuntimeError(f"pdfLaTeX pass {pass_number} failed:\n"
                                   f"{result.stdout[-18000:]}\n{result.stderr[-3000:]}")
        log = (work / Path(SOURCE_NAME).with_suffix(".log")).read_text(errors="replace")
        for problem in ("There were undefined references", "There were undefined citations",
                        "multiply defined", "Rerun to get cross-references right"):
            if problem in log:
                raise RuntimeError(f"Unresolved LaTeX diagnostic: {problem}")
        warnings = [line for line in log.splitlines() if "Overfull " in line]
        if warnings:
            raise RuntimeError("Layout overflow must be fixed before export:\n" + "\n".join(warnings))
        shutil.copy2(work / Path(SOURCE_NAME).with_suffix(".pdf"), target)
        record = {
            "source": SOURCE_NAME,
            "source_sha256": hashlib.sha256((source / SOURCE_NAME).read_bytes()).hexdigest(),
            "output": target.name,
            "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "compiler": version.splitlines()[0],
            "passes": 4,
            "figures": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                        for name in figures},
            "figure_status": "Historical author figures; not recalibrated by this build",
        }
        target.with_suffix(".build.json").write_text(json.dumps(record, indent=2) + "\n",
                                                   encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="PDF destination; defaults beside the source.")
    print(build(parser.parse_args().output))
