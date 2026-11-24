"""Run the pricing, estimation, PDE and small historical examples offline and headless."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN_EXAMPLE = """
import runpy
import sys

attempts = []
def offline(event, args):
    if event in {"socket.connect", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo",
                 "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo"}:
        attempts.append(event)
        raise RuntimeError(f"Offline example attempted {event}")

sys.addaudithook(offline)
import matplotlib
assert matplotlib.get_backend().lower() == "agg"
module = sys.argv.pop(1)
runpy.run_module(module, run_name="__main__", alter_sys=True)
if attempts:
    raise RuntimeError(f"Network operations during example: {attempts}")
"""


def example_modules():
    """Discover executable examples, including plotting and propagation stress scripts."""
    for folder in ("pricers", "estimation", "pde_solvers"):
        for path in sorted((ROOT / "examples" / folder).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if any(
                isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
                for node in tree.body
            ):
                yield ".".join(path.relative_to(ROOT).with_suffix("").parts)
    yield "validation.pde_reference.run_cases"


def check_example(module: str) -> int:
    """Run one module and retain its stdout and traceback for reproducibility."""
    env = dict(os.environ, MPLBACKEND="Agg")
    process = subprocess.run(
        [sys.executable, "-c", RUN_EXAMPLE, module],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    print(
        f"[{'PASS' if process.returncode == 0 else 'FAIL'}] {module}\n{process.stdout}", flush=True
    )
    return process.returncode


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(check_example, example_modules()))
    if not codes:
        raise RuntimeError("No examples discovered")
    print(f"{codes.count(0)}/{len(codes)} examples passed")
    sys.exit(int(any(codes)))
