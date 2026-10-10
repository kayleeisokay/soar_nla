"""gcg_tradeoff.png: the reconstruction / AV-plausibility frontier traced by GCG.

The figure parses the per-example GCG rows rather than summary.json, so the
numbers drawn on the axes and the numbers in the write-up cannot drift apart.
Either the full `results.jsonl` from `soar_nla.gcg.gcg merge` or the shipped
`gcg_losses.jsonl.gz` projection works as input. By default the figure is drawn on
the paper's 1,573 late-position rollouts (`--subset all` draws all 2,000).

    python -m soar_nla.gcg.plots
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from soar_nla.alpaca.subsets import default_rollouts_path, late_positions, load_jsonl, load_rollouts
from soar_nla.paths import results_dir

# Reference data-viz palette (light surface), used unchanged.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
POS = "#2a78d6"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, color=AXIS)


def load_gcg(path: Path, keep: set[str] | None = None) -> dict[float, list[dict]]:
    rows = load_jsonl(path)
    by_beta: dict[float, list[dict]] = {}
    for r in rows:
        if keep is None or r["sample_id"] in keep:
            by_beta.setdefault(float(r["beta"]), []).append(r)
    return by_beta


def figure_gcg(by_beta: dict[float, list[dict]], model: str, out: Path) -> None:
    def mean(rows: list[dict], key: str) -> float:
        return statistics.mean(r[key] for r in rows)

    any_rows = next(iter(by_beta.values()))
    base_fve = 1.0 - mean(any_rows, "original_ar_loss")
    base_nll = mean(any_rows, "original_av_nll")

    points = []
    for beta in sorted(by_beta, reverse=True):
        rows = by_beta[beta]
        points.append(
            {
                "beta": beta,
                "nll": mean(rows, "optimized_av_nll"),
                "fve": 1.0 - mean(rows, "optimized_ar_loss"),
                "improved": sum(
                    1 for r in rows if r["optimized_ar_loss"] < r["original_ar_loss"]
                )
                / len(rows),
                "n": len(rows),
            }
        )
    points.sort(key=lambda p: p["nll"])

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)

    ax.axhline(base_fve, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.text(
        4.68,
        base_fve + 0.003,
        "AV explanation as written",
        fontsize=9,
        color=MUTED,
        va="bottom",
        ha="right",
    )

    xs = [base_nll] + [p["nll"] for p in points]
    ys = [base_fve] + [p["fve"] for p in points]
    ax.plot(xs, ys, color=POS, linewidth=2.0, zorder=3, solid_capstyle="round")
    ax.scatter(
        [p["nll"] for p in points],
        [p["fve"] for p in points],
        s=110,
        color=POS,
        zorder=5,
        linewidths=2.0,
        edgecolors=SURFACE,  # 2px surface ring
    )
    ax.scatter(
        [base_nll], [base_fve], s=110, color=INK, zorder=5,
        linewidths=2.0, edgecolors=SURFACE,
    )

    ax.annotate(
        "original AV explanation",
        xy=(base_nll, base_fve),
        xytext=(base_nll + 0.42, base_fve + 0.004),
        fontsize=9.5,
        color=INK,
        fontweight=600,
        va="bottom",
        ha="left",
        arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.8},
    )
    # All three sit below their marker: above the curve is the subtitle band.
    offsets = {0.1: (0.22, -0.006), 0.01: (0.0, -0.007), 0.0: (-0.14, -0.007)}
    for p in points:
        dx, dy = offsets.get(p["beta"], (0.2, -0.007))
        ha = "right" if dx < 0 else ("center" if dx == 0 else "left")
        ax.text(
            p["nll"] + dx,
            p["fve"] + dy,
            f"β = {p['beta']:g}\n{p['improved']:.1%} improved",
            fontsize=9.5,
            color=INK,
            fontweight=600,
            va="top",
            ha=ha,
            linespacing=1.4,
        )

    ax.set_xlim(-0.15, 4.75)
    ax.set_ylim(0.706, 0.822)
    ax.set_xlabel(
        "Mean AV negative log-likelihood of the string  "
        "(cost in AV plausibility →)",
        fontsize=10,
        color=INK_2,
        labelpad=7,
    )
    ax.set_ylabel(
        "Mean fraction of variance explained\nby the AR reconstruction",
        fontsize=10,
        color=INK_2,
        labelpad=8,
    )

    fig.text(
        0.055,
        0.965,
        "The AV explanation is far from the string the AR reconstructs best",
        fontsize=12,
        fontweight=600,
        color=INK,
        va="top",
        ha="left",
    )
    fig.text(
        0.055,
        0.905,
        f"{model} · n = {points[0]['n']:,} per KL weight · 20 GCG steps · "
        "up-and-left is better",
        fontsize=9,
        color=MUTED,
        va="top",
        ha="left",
    )

    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.115, right=0.97)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gcg-results", type=Path, default=results_dir("gcg") / "gcg_losses.jsonl.gz",
                        help="GCG results.jsonl or the gcg_losses.jsonl.gz projection")
    parser.add_argument("--rollouts", type=Path, default=None,
                        help="soar_nla.alpaca rollouts.jsonl[.gz] (default: results/alpaca)")
    parser.add_argument("--subset", choices=("late_positions", "all"), default="late_positions",
                        help="late_positions: the paper's 1,573 rollouts with sampled position > 50")
    parser.add_argument("--model", default="gemma12b", help="label for the subtitle")
    parser.add_argument("--out", type=Path, default=results_dir("gcg") / "gcg_tradeoff.png")
    args = parser.parse_args()

    keep = None
    if args.subset == "late_positions":
        keep = {row["sample_id"] for row in late_positions(load_rollouts(args.rollouts or default_rollouts_path()))}
    gcg = load_gcg(args.gcg_results, keep)
    if not gcg:
        raise SystemExit(f"no GCG rows in {args.gcg_results}")
    figure_gcg(gcg, args.model, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
