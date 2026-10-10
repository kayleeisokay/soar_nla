"""Paper figure of "Benchmarks using chat template": clean vs NLA-patched accuracy.

Reads the clean and NLA rows of the figure CSV written by `figure.py score`
(ARC-Easy and GSM8K: exact accuracy with Wilson 95% intervals; IFEval: mean of
prompt- and instruction-level strict accuracy with prompt-bootstrap 95% intervals).
The paraphrase and random-vector controls are in the full figure (`figure.py plot`).

Output: `<results>/figures/clean_vs_nla.{png,pdf}`.

    python -m soar_nla.benchmarks.paper_figure
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soar_nla import figure_style as fs
from soar_nla.benchmarks.figure import STEM_NAME, TASKS
from soar_nla.paths import results_dir

TASK_LABELS = {"arc_easy": "ARC Easy", "gsm8k": "GSM8K", "ifeval": "IFEval"}
SERIES = (("clean", "Clean", fs.NAVY), ("nla", "NLA patched", fs.RED))


def read_rows(path: Path) -> dict[tuple[str, str], tuple[float, float, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["task"], row["condition"]): (
                100 * float(row["accuracy"]), 100 * float(row["ci_low"]), 100 * float(row["ci_high"])
            )
            for row in csv.DictReader(handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    figures = results_dir("benchmarks") / "figures"
    parser.add_argument("--csv", type=Path, default=figures / f"{STEM_NAME}.csv")
    parser.add_argument("--out-dir", type=Path, default=figures)
    args = parser.parse_args()
    rows = read_rows(args.csv)

    fs.apply(14)
    x = np.arange(len(TASKS))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for (condition, label, color), offset in zip(SERIES, (-width / 2, width / 2)):
        values = np.array([rows[task, condition] for task in TASKS])
        mean, low, high = values.T
        bars = ax.bar(
            x + offset, mean, width, label=label, color=color, edgecolor="black", linewidth=1.3,
            yerr=np.vstack((mean - low, high - mean)), capsize=5,
            error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": "black"},
        )
        ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11)

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, [TASK_LABELS[task] for task in TASKS])
    ax.set_ylim(0, 108)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="both", width=1.5, length=5)
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout(pad=1.2)
    fs.save(fig, args.out_dir, "clean_vs_nla")


if __name__ == "__main__":
    main()
