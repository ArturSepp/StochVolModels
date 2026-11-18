"""Compile the barrier manuscript using pdfLaTeX in a temporary directory.

The bibliography is inline. An installed MiKTeX or TeX Live distribution is required.
No Python packages or virtual environment are created by this script.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def build(output: Path) -> Path:
    """Run three LaTeX passes, check references, and export only the compiled PDF."""
    source = Path(__file__).resolve().parent
    compiler = shutil.which("pdflatex")
    if compiler is None:
        raise RuntimeError("pdfLaTeX is required; install MiKTeX or TeX Live.")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="barrier-paper-") as directory:
        work = Path(directory)
        for name in ("HestonPaper.tex", "tcilatex.tex"):
            shutil.copy2(source / name, work / name)
        shutil.copytree(source / "figures", work / "figures")
        for pass_number in range(1, 4):
            result = subprocess.run(
                [compiler, "-interaction=nonstopmode", "-halt-on-error",
                 "-no-shell-escape", "HestonPaper.tex"],
                cwd=work, capture_output=True, text=True, errors="replace", check=False,
            )
            if result.returncode:
                raise RuntimeError(f"pdfLaTeX pass {pass_number} failed:\n{result.stdout}\n"
                                   f"{result.stderr}")
        log = (work / "HestonPaper.log").read_text(errors="replace")
        for problem in ("There were undefined references", "There were undefined citations",
                        "multiply defined", "Rerun to get cross-references right", "Overfull "):
            if problem in log:
                raise RuntimeError(f"Unresolved LaTeX diagnostic: {problem}\n{log}")
        for line in log.splitlines():
            if "Warning" in line or "Overfull" in line or "Output written" in line:
                print(line)
        target = output / "HestonPaper.pdf"
        shutil.copy2(work / target.name, target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent,
                        help="PDF destination directory; auxiliary files stay temporary.")
    print(build(parser.parse_args().output))
