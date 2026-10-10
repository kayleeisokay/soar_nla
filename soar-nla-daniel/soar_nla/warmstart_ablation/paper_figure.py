"""Paper figures of the warm-start ablation (Appendix "Skipping Warm-Start").

`training_fve`: training-batch FVE of the three runs up to step 600, with each
curve's start and best value called out (`fve_curve_*.csv`, unsmoothed).
`selfdescribe_judge`: Gemma-3-27B-IT forced-choice accuracy per SelfDescribe
attribute with Wilson 95% intervals (`judge_summary.json`), next to the chance
rate of each attribute.

Legend labels: B is the cold-start run *without* KL and C the cold-start run
with KL 0.01 (see `runs.py`).

Outputs in `<results>/figures/`: `training_fve.{png,pdf}`, `selfdescribe_judge.{png,pdf}`.

    python -m soar_nla.warmstart_ablation.paper_figure
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from soar_nla import figure_style as fs
from soar_nla.paths import results_dir
from soar_nla.warmstart_ablation.runs import COMPARE_STEP, FVE_CSV

LABELS = {"A": "Original", "B": "No warm-start, no KL", "C": "No warm-start"}
# Plot order and colors of each figure, as in the paper.
TRAINING_RUNS = (("C", fs.NAVY), ("B", fs.TEAL), ("A", fs.RED))
JUDGE_RUNS = (("A", fs.NAVY), ("B", fs.RED), ("C", fs.GREEN))
ATTRIBUTES = ("Gender", "Occupation", "Country", "Religion", "POOLED")

# Callout text positions (step, FVE %) chosen so the boxes do not overlap.
START_CALLOUTS = {"C": (35, -105, "left"), "B": (35, -116, "left"), "A": (35, 38, "left")}
BEST_CALLOUTS = {"C": (465, 57, "left"), "B": (320, 91, "left"), "A": (595, 91, "right")}


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps, fve = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["step"] and row["fve_train"]:
                steps.append(int(row["step"]))
                fve.append(float(row["fve_train"]))
    if not steps:
        raise ValueError(f"no training FVE values in {path}")
    return np.asarray(steps), 100 * np.asarray(fve)


def callout(
    ax: plt.Axes, text: str, xy: tuple[float, float], xytext: tuple[float, float], ha: str, color: str
) -> None:
    ax.annotate(
        text, xy=xy, xytext=xytext, fontsize=9.5, color=color, fontweight="bold", ha=ha, va="center",
        bbox={"facecolor": "white", "edgecolor": color, "linewidth": 1.0, "alpha": 0.90, "pad": 3.0},
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.0}, zorder=6,
    )


def training_fve(csv_dir: Path, out_dir: Path, max_step: int = COMPARE_STEP) -> None:
    fs.apply(16)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for run, color in TRAINING_RUNS:
        steps, fve = load_curve(csv_dir / FVE_CSV[run])
        visible = steps <= max_step
        steps, fve = steps[visible], fve[visible]
        best = int(np.argmax(fve))
        ax.plot(steps, fve, color=color, linewidth=2.2, alpha=0.92, label=LABELS[run])
        ax.plot(steps[best], fve[best], marker="o", markersize=7, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=2, zorder=5)
        x, y, ha = START_CALLOUTS[run]
        callout(ax, f"Start {fve[0]:.1f}%", (steps[0], fve[0]), (x, y), ha, color)
        x, y, ha = BEST_CALLOUTS[run]
        callout(ax, f"Best {fve[best]:.1f}% @ {steps[best]}", (steps[best], fve[best]), (x, y), ha, color)

    ax.axhline(0, color=fs.MUTED, linewidth=1.2, zorder=0)
    ax.set_xlabel("Step")
    ax.set_ylabel("FVE (%)")
    ax.set_xlim(0, max_step)
    ax.set_ylim(-125, 100)
    ax.set_xticks(np.arange(0, max_step + 1, 100))
    ax.set_yticks(np.arange(-120, 81, 20))
    ax.tick_params(axis="both", width=1.5, length=5)
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout(pad=1.2)
    fs.save(fig, out_dir, "training_fve")


def selfdescribe_judge(summary: dict, out_dir: Path) -> None:
    pooled_label = {"POOLED": "Pooled"}
    ticks = [f"{pooled_label.get(a, a)}\n(n={summary['A'][a]['n']:,})" for a in ATTRIBUTES]
    for run, _ in JUDGE_RUNS:
        if [summary[run][a]["n"] for a in ATTRIBUTES] != [summary["A"][a]["n"] for a in ATTRIBUTES]:
            raise ValueError(f"run {run} was judged on a different item set than run A")
    series = [
        (LABELS[run], color, 100 * np.array([summary[run][a]["accuracy"] for a in ATTRIBUTES]),
         100 * np.array([summary[run][a]["wilson95"] for a in ATTRIBUTES]))
        for run, color in JUDGE_RUNS
    ]
    # Chance: one over the number of answer options of each attribute (pooled: item-weighted).
    chance = 100 * np.array([summary["A"][a]["chance"] for a in ATTRIBUTES])
    series.append(("Random (Control)", fs.GOLD, chance, None))

    fs.apply(15)
    x = np.arange(len(ATTRIBUTES))
    width = 0.15
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for (label, color, values, ci), offset in zip(series, offsets):
        yerr = None if ci is None else np.vstack((values - ci[:, 0], ci[:, 1] - values))
        bars = ax.bar(
            x + offset, values, width=width, label=label, color=color, edgecolor="black", linewidth=1.3,
            yerr=yerr, capsize=4 if yerr is not None else 0,
            error_kw={"elinewidth": 1.4, "capthick": 1.4, "ecolor": "black"},
        )
        tops = values if ci is None else ci[:, 1]
        for bar, value, top in zip(bars, values, tops):
            ax.text(bar.get_x() + bar.get_width() / 2, top + 1.5, f"{value:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, rotation=90, color=fs.INK)

    ax.set_xticks(x, ticks)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="both", width=1.5, length=5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(series), fontsize=12)
    fig.tight_layout(pad=1.2)
    fs.save(fig, out_dir, "selfdescribe_judge")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = results_dir("warmstart_ablation")
    parser.add_argument("--csv-dir", type=Path, default=root, help="directory with fve_curve_*.csv")
    parser.add_argument("--judge-summary", type=Path, default=root / "judge_summary.json")
    parser.add_argument("--out-dir", type=Path, default=root / "figures")
    args = parser.parse_args()

    training_fve(args.csv_dir, args.out_dir)
    selfdescribe_judge(json.loads(args.judge_summary.read_text()), args.out_dir)


if __name__ == "__main__":
    main()
