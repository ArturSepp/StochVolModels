"""Shared helpers of the documentation producers: canonical scripts, figure style and saving."""

from __future__ import annotations

import runpy
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from scripts.docs_analytics.registry import ROOT  # noqa: E402

# Previews are drawn about 9 inches wide at 150 dpi and displayed at the article width of about
# 740 pixels; a 12-point base font keeps labels legible at that scale.
DPI = 150
STYLE = {"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "legend.fontsize": 11,
         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5}


def load_script(relative: str) -> dict:
    """Execute a canonical documentation script as a module and return its namespace."""
    return runpy.run_path(str(ROOT / relative), run_name="documentation_producer")


@contextmanager
def figure_style():
    """Apply the documentation font sizes to figures drawn inside the block."""
    with plt.rc_context(STYLE):
        yield


def thin_ticks(fig: plt.Figure, x_bins: int = 5, y_bins: int = 6) -> None:
    """Limit tick counts so labels do not collide at the article width.

    The y locator uses steps of 1, 2 and 5, so percentage labels printed without decimals stay
    distinct. Only the tick positions change; the data and the formatters are untouched.
    """
    for ax in fig.axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=x_bins))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_bins, steps=[1, 2, 5, 10]))


def save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    """Save a figure as the exhibit ``name`` of the bundle and close it."""
    path = output_dir / "images" / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
