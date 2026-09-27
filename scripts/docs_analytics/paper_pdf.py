"""Producer of paper reproductions: crops of figures of the open-access IJTAF article.

Sepp and Rakhmonov (2023) is published open access under the Creative Commons Attribution 4.0
licence, which permits reproduction with attribution and an indication of changes. Only its
figures whose inputs cannot be distributed are reproduced; figures that can be recomputed offline
are regenerated instead. Crops are rendered from the article PDF in ``papers/`` with ``pdftoppm``
(poppler) at a recorded resolution. Crop boxes are in PDF points from the top-left corner of the
rendered page and stay inside the text block, so running heads and the download stamp in the page
margin are excluded. Several boxes are stacked vertically, which allows sub-captions to be left
out. The factor HJM article is not open access and is never cropped.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.docs_analytics.registry import ROOT

# the text block of the IJTAF page in points; a crop must stay inside it
TEXT_BLOCK = (100.0, 500.0)


def _pdftoppm() -> str:
    """Return the pdftoppm executable or explain how to obtain it."""
    tool = shutil.which("pdftoppm")
    if tool is None:
        raise RuntimeError("pdftoppm (poppler) is required for paper reproductions.")
    return tool


def produce_pdf_crop(name: str, spec: dict, output_dir: Path) -> dict:
    """Render one or more boxes of a PDF page and stack them into one image."""
    parameters = spec["parameters"]
    pdf = ROOT / spec["paper_module"]
    dpi = parameters["dpi"]
    scale = dpi / 72.0
    tool = _pdftoppm()
    pieces = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, (x, y, width, height) in enumerate(parameters["boxes_pt"]):
            stem = Path(tmp) / f"box{index}"
            subprocess.run(
                [tool, "-f", str(parameters["page"]), "-l", str(parameters["page"]),
                 "-r", str(dpi), "-x", str(round(x * scale)), "-y", str(round(y * scale)),
                 "-W", str(round(width * scale)), "-H", str(round(height * scale)),
                 "-png", "-singlefile", str(pdf), str(stem)],
                check=True, capture_output=True)
            with Image.open(f"{stem}.png") as piece:
                pieces.append(piece.convert("RGB"))
    gap = round(parameters.get("gap_pt", 0.0) * scale)
    width = max(piece.width for piece in pieces)
    height = sum(piece.height for piece in pieces) + gap * (len(pieces) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    top = 0
    for piece in pieces:
        canvas.paste(piece, ((width - piece.width) // 2, top))
        top += piece.height + gap
    canvas.save(output_dir / "images" / name)

    version = subprocess.run([tool, "-v"], capture_output=True, text=True)
    boxes = pd.DataFrame(parameters["boxes_pt"], columns=["x_pt", "y_pt", "width_pt", "height_pt"])
    boxes["renderer"] = (version.stderr or version.stdout).splitlines()[0]
    inside = all(TEXT_BLOCK[0] <= x and x + w <= TEXT_BLOCK[1]
                 for x, _, w, _ in parameters["boxes_pt"])
    return {"tables": {"crop_boxes": boxes}, "checks": {"crop_inside_text_block": inside}}
