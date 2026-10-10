"""Paired raw-vs-chat figures for the 2-shot English->French replication.

Writes, under ``--figures-dir``:

* ``fv_english_french_paired.{pdf,svg,png}`` and ``figure_data.csv``: accuracy
  per arm for the reference's reported raw-prompt run, our raw-prompt run,
  and our chat-template run on the same 50 queries (with paired bootstrap
  intervals for chat minus raw in the CSV);
* ``fv_english_french_shot_baselines.{pdf,svg,png}`` and
  ``baseline_shot_sweep_data.csv``: unpatched accuracy at 1, 2 and 4 shots.

The reference values come from the results CSV that the reference executor
wrote for its English->French shot sweep
(``reference_shot_sweep_english_french.csv``).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from soar_nla.paths import results_dir

RESULTS = results_dir("fv_chat_template")
ARMS = (
    ("baseline", "Baseline", "acc_baseline"),
    ("identity", "Identity", None),
    ("nla", "NLA round-trip", "acc_nla"),
    ("para_medium", "Medium paraphrase", "acc_para_medium"),
    ("para_heavy", "Heavy paraphrase", "acc_para_heavy"),
    ("random_vec", "Random vector", "acc_random_vec"),
    ("mismatch_cross_task", "Cross-task native swap", "acc_mismatch_cross_task"),
    ("mismatch_same_task", "Same-task native swap", "acc_mismatch_same_task"),
)
PLOT_ARMS = (
    ("baseline", "Baseline"),
    ("nla", "NLA round-trip"),
    ("para_medium", "Paraphrase (medium)"),
    ("para_heavy", "Paraphrase (heavy)"),
    ("mismatch_same_task", "Mismatched FV (same-task)"),
    ("mismatch_cross_task", "Mismatched FV (cross-task)"),
    ("random_vec", "Random vector"),
)
SHOT_COUNTS = (1, 2, 4)
REFERENCE_LABEL = "Reference (reported)"


def paired_interval(raw: np.ndarray, chat: np.ndarray, seed: int) -> tuple[float, float]:
    differences = chat.astype(float) - raw.astype(float)
    generator = np.random.default_rng(seed)
    samples = generator.choice(differences, size=(20000, len(differences)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.percentile(samples, [2.5, 97.5]))


def baseline_shot_sweep(plt, reference_csv: Path, shot_dirs: dict[int, Path], destination: Path) -> None:
    with reference_csv.open(newline="") as handle:
        published_rows = {int(row["n_shots"]): row for row in csv.DictReader(handle)}
    reference = [float(published_rows[shot]["acc_baseline"]) for shot in SHOT_COUNTS]
    raw = []
    chat = []
    data = []
    for shot in SHOT_COUNTS:
        summary = json.loads((shot_dirs[shot] / "summary.json").read_text())
        raw_entry = summary["conditions"]["raw_baseline"]
        chat_entry = summary["conditions"]["chat_baseline"]
        if raw_entry["completed"] != 50 or chat_entry["completed"] != 50:
            raise RuntimeError(f"incomplete {shot}-shot baseline")
        raw.append(raw_entry["accuracy"])
        chat.append(chat_entry["accuracy"])
        data.append({
            "shots": shot, "n": 50, "reference_raw_accuracy": reference[len(data)],
            "reproduced_raw_accuracy": raw[-1], "chat_accuracy": chat[-1],
        })
    with (destination / "baseline_shot_sweep_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    x = np.arange(len(SHOT_COUNTS))
    fig, axis = plt.subplots(figsize=(6.7, 4.7))
    axis.plot(x - 0.035, reference, "D--", color="#4D4D4D", markersize=7,
              linewidth=1.8, label=REFERENCE_LABEL)
    axis.plot(x, raw, "o-", color="#0F4D92", markersize=8,
              linewidth=2.2, label="Reproduced raw")
    axis.plot(x + 0.035, chat, "s-", color="#42949E", markersize=8,
              linewidth=2.2, label="Paired chat")
    axis.set_xticks(x, [f"{shot} shot{'s' if shot > 1 else ''}" for shot in SHOT_COUNTS])
    axis.set_xlim(-0.22, 2.22)
    axis.set_ylim(0, 0.67)
    axis.set_ylabel("Exact first-line accuracy")
    axis.set_xlabel("English→French in-context examples")
    axis.set_title("English→French baselines · same 50 queries per shot count", fontweight="bold")
    axis.legend(loc="upper left", frameon=False)
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.7)
    axis.set_axisbelow(True)
    fig.tight_layout(pad=1.2)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(destination / f"fv_english_french_shot_baselines.{suffix}",
                    dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-csv", type=Path, default=RESULTS / "reference_shot_sweep_english_french.csv")
    parser.add_argument("--shot1-dir", type=Path, default=RESULTS / "shot1")
    parser.add_argument("--shot2-dir", type=Path, default=RESULTS / "shot2")
    parser.add_argument("--shot4-dir", type=Path, default=RESULTS / "shot4")
    parser.add_argument("--figures-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = args.figures_dir
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.2, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    shot_dirs = {1: args.shot1_dir, 2: args.shot2_dir, 4: args.shot4_dir}
    baseline_shot_sweep(plt, args.reference_csv, shot_dirs, destination)
    rows = json.loads((args.shot2_dir / "rows.json").read_text())
    with args.reference_csv.open(newline="") as handle:
        reference = next(row for row in csv.DictReader(handle) if int(row["n_shots"]) == 2)
    if len(rows) != 50 or any(
        any(f"{style}_{arm}" not in row["arms"] for style in ("raw", "chat") for arm, _, _ in ARMS)
        for row in rows
    ):
        print("Paired intervention figure awaits all 50 raw/chat rows")
        return
    raw = []
    chat = []
    published = []
    data_rows = []
    for index, (arm, label, source_key) in enumerate(ARMS):
        raw_values = np.array([row["arms"][f"raw_{arm}"]["correct"] for row in rows], dtype=bool)
        chat_values = np.array([row["arms"][f"chat_{arm}"]["correct"] for row in rows], dtype=bool)
        raw_score = float(raw_values.mean())
        chat_score = float(chat_values.mean())
        ref_score = float(reference[source_key]) if source_key else np.nan
        low, high = paired_interval(raw_values, chat_values, seed=index)
        raw.append(raw_score)
        chat.append(chat_score)
        published.append(ref_score)
        data_rows.append({
            "arm": arm, "label": label, "n": len(rows),
            "reference_accuracy": ref_score, "raw_accuracy": raw_score,
            "chat_accuracy": chat_score, "chat_minus_raw": chat_score - raw_score,
            "paired_bootstrap_95_low": low, "paired_bootstrap_95_high": high,
        })

    with (destination / "figure_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data_rows[0]))
        writer.writeheader()
        writer.writerows(data_rows)

    order = [[arm for arm, _, _ in ARMS].index(arm) for arm, _ in PLOT_ARMS]
    x = np.arange(len(PLOT_ARMS))
    width = 0.26
    fig, axis = plt.subplots(figsize=(12.6, 5.0))
    series = (
        ([published[index] for index in order], REFERENCE_LABEL, "#4D4D4D"),
        ([raw[index] for index in order], "Reproduced raw", "#0F4D92"),
        ([chat[index] for index in order], "Paired chat", "#42949E"),
    )
    for offset, (values, label, color) in zip((-width, 0.0, width), series, strict=True):
        axis.bar(x + offset, values, width=width, color=color, edgecolor="black",
                 linewidth=1.0, label=label, zorder=3)
    axis.set_xticks(x, [label for _, label in PLOT_ARMS], rotation=30, ha="right")
    axis.set_ylim(0, 0.5)
    axis.set_yticks(np.arange(0, 0.51, 0.1))
    axis.set_ylabel("Exact first-line accuracy")
    axis.legend(loc="upper right", frameon=False)
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.7)
    axis.set_axisbelow(True)
    fig.tight_layout(pad=1.0)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(destination / f"fv_english_french_paired.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"WROTE {destination}")


if __name__ == "__main__":
    main()
