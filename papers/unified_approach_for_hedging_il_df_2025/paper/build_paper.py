"""Compile the corrected manuscript in a temporary directory with pdfLaTeX.

No Python environment is created. The source includes its bibliography inline.
Run with an installed TeX distribution containing the packages in the preamble.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def build(output: Path) -> Path:
    """Build three passes, reject unresolved references, and export the PDF."""
    source = Path(__file__).resolve().parent
    name = "AMM IL paper v4"
    compiler = shutil.which("pdflatex")
    if compiler is None:
        raise RuntimeError("pdfLaTeX is required; install MiKTeX or TeX Live.")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="il-paper-") as directory:
        work = Path(directory)
        shutil.copy2(source / f"{name}.tex", work)
        shutil.copytree(source / "new_figures", work / "new_figures")
        for pass_number in range(1, 4):
            result = subprocess.run(
                [compiler, "-interaction=nonstopmode", "-halt-on-error",
                 "-no-shell-escape", f"{name}.tex"],
                cwd=work, capture_output=True, text=True, errors="replace", check=False,
            )
            if result.returncode:
                raise RuntimeError(f"pdfLaTeX pass {pass_number} failed:\n{result.stdout}\n"
                                   f"{result.stderr}")
        log = (work / f"{name}.log").read_text(errors="replace")
        for problem in ("There were undefined references", "There were undefined citations",
                        "multiply defined", "Overfull "):
            if problem in log:
                raise RuntimeError(f"Unresolved LaTeX diagnostic: {problem}\n{log}")
        target = output / f"{name}.pdf"
        shutil.copy2(work / target.name, target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination for the compiled PDF; build files stay temporary.")
    print(build(parser.parse_args().output))
