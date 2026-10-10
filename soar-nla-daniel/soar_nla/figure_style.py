"""Shared look of the paper figures: rcParams, palette and a PNG + PDF saver.

The section modules that draw a paper figure (`<section>/paper_figure.py`) read
their numbers from the shipped result files and only take the styling from here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#0F4D92"
BLUE = "#3775BA"
RED = "#B64342"
PINK = "#E9A6A1"
GREEN = "#8BCF8B"
TEAL = "#42949E"
GOLD = "#FFD700"
GREY = "#CFCECE"
INK = "#272727"
MUTED = "#767676"

ERROR_KW = {"elinewidth": 1.8, "capthick": 1.8, "ecolor": "black"}


def apply(font_size: float = 16) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.linewidth": 2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, out_dir: Path, stem: str, extensions: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in extensions:
        path = out_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    for path in paths:
        print(f"wrote {path}")
    return paths
