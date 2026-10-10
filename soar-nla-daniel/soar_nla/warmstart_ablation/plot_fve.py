#!/usr/bin/env python3
"""Plot in-distribution (training-batch) FVE during RL for the three Qwen2.5-7B NLA runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from soar_nla.paths import results_dir  # noqa: E402
from soar_nla.warmstart_ablation.runs import FVE_CSV, RUN_LABELS, RUNS  # noqa: E402

COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}


def load(path: Path):
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return ([int(r["step"]) for r in rows],
            [float(r["fve_train"]) for r in rows],
            [float(r["fve_train_ema"]) for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", type=Path, default=results_dir("warmstart_ablation"))
    ap.add_argument("--out", type=Path, default=results_dir("warmstart_ablation") / "fve_train.png")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    for run in RUNS:
        step, raw, ema = load(a.csv_dir / FVE_CSV[run])
        color = COLORS[run]
        ax.plot(step, raw, color=color, lw=0.9, alpha=0.22, zorder=2)
        ax.plot(step, ema, color=color, lw=2.0, label=f"{run}: {RUN_LABELS[run]}", zorder=3)
        # direct label at the end of each smoothed curve
        ax.annotate(f"{ema[-1]*100:.1f}%", xy=(step[-1], ema[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color="#3d3d38", fontsize=9, va="center")

    ax.set_xlabel("RL step")
    ax.set_ylabel("In-distribution FVE (training batch)")
    ax.set_title("Qwen2.5-7B NLA: training-set FVE during RL", fontsize=12, pad=10)
    ax.set_ylim(-0.2, 0.85)
    ax.set_xlim(0, 820)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    ax.grid(True, axis="y", color="#dededa", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b5ad")
    ax.tick_params(colors="#6b6b64", length=3)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
