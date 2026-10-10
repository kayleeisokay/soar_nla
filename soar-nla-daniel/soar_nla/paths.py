"""Default output locations. Override the root with SOAR_RESULTS."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(os.environ.get("SOAR_RESULTS", REPO_ROOT / "results"))


def results_dir(section: str) -> Path:
    return RESULTS_ROOT / section
