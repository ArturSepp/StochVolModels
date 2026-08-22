"""Validate the output-free Colab quickstart and its released-script handoff."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    """Reject notebook output, Heston drift, or a missing released LogSV quickstart handoff."""
    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "getting_started" / "quickstart_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert code, "the Colab notebook has no executable cells"
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code)
    assert "LogSvParams" in source and "LogSVPricer" in source
    assert "Heston" not in source
    assert 'f"v{release}/examples/getting_started/quickstart.py"' in source
    assert "%run quickstart.py" in source
    print("quickstart-notebook-check: PASS")


if __name__ == "__main__":
    main()
