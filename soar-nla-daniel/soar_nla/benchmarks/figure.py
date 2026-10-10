"""Three-task bar graph for the all-prompt-token benchmark interventions.

ARC-Easy and GSM8K show exact accuracy with Wilson 95% intervals; IFEval
shows the equal-weight mean of strict prompt- and instruction-level accuracy
with 10,000-draw prompt-bootstrap 95% intervals. All IFEval conditions,
including clean, use the same Google checker.

Two stages: ``score`` rescans the per-example rollouts and writes the figure
CSV (needs the IFEval checker dependencies, not matplotlib); ``plot`` and
``plot-legend`` draw the CSV (need only matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from soar_nla.paths import results_dir

TASKS = ("arc_easy", "gsm8k", "ifeval")
CONDITIONS = ("clean", "nla", "para_medium", "para_heavy", "random_vec")
TITLES = {
    "arc_easy": "ARC-Easy · n=2,376",
    "gsm8k": "GSM8K · n=1,319",
    "ifeval": "IFEval · n=541",
}
DRAWS = 10_000
STEM_NAME = "allpos_accuracy_strict_average"
LEGEND_STEM_NAME = "allpos_accuracy_bars_legend"

CONDITION_LABELS = {
    "clean": "Baseline",
    "nla": "NLA round-trip",
    "para_medium": "Paraphrase (medium)",
    "para_heavy": "Paraphrase (heavy)",
    "random_vec": "Random vector",
}
CONDITION_COLORS = {
    "clean": "#CFCECE",
    "nla": "#0F4D92",
    "para_medium": "#AADCA9",
    "para_heavy": "#8BCF8B",
    "random_vec": "#9A4D8E",
}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half


def exact_accuracy(source, task: str) -> dict[str, tuple[float, tuple[float, float]]]:
    from soar_nla.benchmarks.allpos import correctness, scorer_for

    scorer = scorer_for(task) if task == "arc_easy" else None
    results = {}
    for condition in CONDITIONS:
        flags = correctness(task, source.load(task, condition), scorer)["accuracy"]
        results[condition] = (sum(flags) / len(flags), wilson(sum(flags), len(flags)))
    return results


def ifeval_strict_average(source) -> dict[str, tuple[float, tuple[float, float]]]:
    from soar_nla.benchmarks import ifeval as module

    dataset = module.load_rows()
    rows = [dict(dataset[i]) for i in range(len(dataset))]
    draws = np.random.default_rng(0).integers(0, len(rows), size=(DRAWS, len(rows)))
    results = {}
    for condition in CONDITIONS:
        generated = source.load("ifeval", condition)
        responses = {int(generated[i]["key"]): generated[i]["output"] for i in range(len(generated))}
        scored = {int(row["key"]): row for row in module.score_rows(rows, responses)["rows"]}
        ordered = [scored[int(row["key"])] for row in rows]
        prompt = np.asarray([int(row["strict_follow_all_instructions"]) for row in ordered])
        correct = np.asarray([sum(row["strict_follow_instruction_list"]) for row in ordered])
        total = np.asarray([len(row["strict_follow_instruction_list"]) for row in ordered])
        value = 0.5 * (prompt.mean() + correct.sum() / total.sum())
        sampled = 0.5 * (prompt[draws].mean(axis=1) + correct[draws].sum(axis=1) / total[draws].sum(axis=1))
        low, high = np.percentile(sampled, [2.5, 97.5])
        results[condition] = (float(value), (float(low), float(high)))
    return results


def score(args) -> None:
    from soar_nla.benchmarks.rollouts import source_from_args

    source = source_from_args(args)
    data = {task: exact_accuracy(source, task) for task in ("arc_easy", "gsm8k")}
    data["ifeval"] = ifeval_strict_average(source)
    path = args.figures_dir / f"{STEM_NAME}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "condition", "metric", "accuracy", "ci_low", "ci_high"])
        for task in TASKS:
            metric = "strict_average" if task == "ifeval" else "exact_accuracy"
            for condition in CONDITIONS:
                value, (low, high) = data[task][condition]
                writer.writerow([task, condition, metric, value, low, high])
    print(f"WROTE {path}")


def read_csv(figures_dir: Path) -> dict[tuple[str, str], dict]:
    with (figures_dir / f"{STEM_NAME}.csv").open() as handle:
        return {(row["task"], row["condition"]): row for row in csv.DictReader(handle)}


def style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def finalize(plt, fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.4)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.06,
        )
    plt.close(fig)


def plot(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_csv(args.figures_dir)
    style(plt)
    labels = [CONDITION_LABELS[condition] for condition in CONDITIONS]
    fig, axes = plt.subplots(1, 3, figsize=(24 * len(CONDITIONS) / 7, 7.2), sharey=True)
    x = np.arange(len(CONDITIONS))
    for ax, task in zip(axes, TASKS, strict=True):
        rows = [data[(task, c)] for c in CONDITIONS]
        values = np.asarray([100 * float(r["accuracy"]) for r in rows])
        lower = values - np.asarray([100 * float(r["ci_low"]) for r in rows])
        upper = np.asarray([100 * float(r["ci_high"]) for r in rows]) - values
        bars = ax.bar(
            x, values, width=0.77,
            color=[CONDITION_COLORS[c] for c in CONDITIONS],
            edgecolor="#262626", linewidth=0.8,
            yerr=np.asarray([lower, upper]), capsize=2.3,
            error_kw={"elinewidth": 1.0, "ecolor": "#262626"},
        )
        for bar, value, error in zip(bars, values, upper, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + error + 1.4,
                f"{value:.1f}", ha="center", va="bottom", fontsize=8,
            )
        ax.set_title(TITLES[task], fontweight="bold", fontsize=12)
        ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=9)
        ax.set_ylim(0, 107)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy (%)")
    stem = args.figures_dir / STEM_NAME
    finalize(plt, fig, stem)
    print(f"WROTE {stem}.{{pdf,svg,png}}")


def plot_legend(args) -> None:
    """One panel per benchmark, five bars each, no tick labels, shared legend panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_csv(args.figures_dir)
    plt.rcParams.update({
        "font.family": ["Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 24,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 3,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    titles = {"arc_easy": "ARC-Easy", "gsm8k": "GSM8K", "ifeval": "IFEval (strict avg.)"}
    fig = plt.figure(figsize=(28, 6))
    x = np.arange(len(CONDITIONS))
    for index, task in enumerate(TASKS):
        ax = fig.add_subplot(1, 4, index + 1)
        rows = [data[(task, c)] for c in CONDITIONS]
        values = np.asarray([100 * float(r["accuracy"]) for r in rows])
        lower = values - np.asarray([100 * float(r["ci_low"]) for r in rows])
        upper = np.asarray([100 * float(r["ci_high"]) for r in rows]) - values
        ax.bar(
            x, values, yerr=np.asarray([lower, upper]), capsize=5,
            color=[CONDITION_COLORS[c] for c in CONDITIONS],
            label=[CONDITION_LABELS[c] for c in CONDITIONS],
        )
        for position, value, error in zip(x, values, upper, strict=True):
            ax.text(position, value + error + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=16)
        if index == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.set_ylabel("Accuracy (%)", fontsize=32)
        ax.set_xticks([])
        ax.set_ylim(0, 110)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_title(titles[task], fontsize=28)
    ax = fig.add_subplot(1, 4, 4)
    ax.legend(handles, labels)
    ax.set_axis_off()
    fig.tight_layout(pad=2)
    stem = args.figures_dir / LEGEND_STEM_NAME
    for extension in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=300)
    plt.close(fig)
    print(f"WROTE {stem}.{{pdf,svg,png}}")


def main() -> None:
    from soar_nla.benchmarks.rollouts import add_source_args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("score", "plot", "plot-legend"))
    parser.add_argument("--figures-dir", type=Path, default=results_dir("benchmarks") / "figures")
    add_source_args(parser)
    args = parser.parse_args()
    {"score": score, "plot": plot, "plot-legend": plot_legend}[args.stage](args)


if __name__ == "__main__":
    main()
