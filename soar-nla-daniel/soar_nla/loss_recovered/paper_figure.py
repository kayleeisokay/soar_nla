"""Paper figure of "Loss Recovered": NLA loss recovered per token span, and the SAE on tokens 51-100.

Reads `paper_numbers.json` (point estimates and bootstrap 95% intervals).
Output: `<results>/figures/loss_recovered.{png,pdf}`.

    python -m soar_nla.loss_recovered.paper_figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soar_nla import figure_style as fs

from .common import default_results_dir

# (x label, span, method, color)
BARS = (
    ("1–50", "prefix_50", "nla", fs.BLUE),
    ("1–100", "prefix_100", "nla", fs.NAVY),
    ("51–100", "suffix_50_100", "nla", fs.GREEN),
    ("51–100, SAE", "suffix_50_100", "sae_norm", fs.TEAL),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--numbers", type=Path, default=default_results_dir() / "paper_numbers.json")
    parser.add_argument("--out-dir", type=Path, default=default_results_dir() / "figures")
    args = parser.parse_args()
    settings = json.loads(args.numbers.read_text())["settings"]

    entries = [settings[span]["loss_recovered"][method] for _, span, method, _ in BARS]
    means = np.array([100 * entry["value"] for entry in entries])
    low, high = (np.array([100 * entry["ci95"][i] for entry in entries]) for i in (0, 1))

    fs.apply(16)
    x = np.arange(len(BARS))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        x, means, width=0.64, yerr=np.vstack((means - low, high - means)), capsize=6,
        color=[color for *_, color in BARS], edgecolor="black", linewidth=1.5, error_kw=fs.ERROR_KW,
    )
    ax.set_xticks(x, [label for label, *_ in BARS])
    ax.set_xlabel("Token span")
    ax.set_ylabel("Loss recovered (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="both", width=1.5, length=5)
    for bar, mean, upper in zip(bars, means, high):
        ax.annotate(
            f"{mean:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, upper), xytext=(0, 10),
            textcoords="offset points", ha="center", va="bottom", fontsize=14, fontweight="bold",
        )
    fig.tight_layout(pad=1.2)
    fs.save(fig, args.out_dir, "loss_recovered")


if __name__ == "__main__":
    main()
