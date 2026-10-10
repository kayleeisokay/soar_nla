"""Faithfulness paper figures: FVE by explanation source, and one AV-vs-Sonnet example.

`fve_by_explanation`: FVE of AV and Sonnet explanations through the AR of the
custom L32 NLA, after RL and at the warm-start SFT checkpoint, on the strict clean
subset (1,997 rows), with the bootstrap 95% CIs from `fve_table.json`.

`av_vs_sonnet_example`: the stepped-spillway passage of `qualitative_example.txt`
with the AV explanation (RL checkpoint) and the Sonnet explanation side by side.
The grounded / confabulated highlighting is a manual reading of the passage
(phrase lists below); the two FVEs are the record's per-row values.

Outputs in `<results>/figures/`: `fve_by_explanation.{png,pdf}`,
`av_vs_sonnet_example.{png,pdf}`.

    python -m soar_nla.faithfulness.paper_figure
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from soar_nla import figure_style as fs
from soar_nla import text_panels as tp
from soar_nla.paths import results_dir

# (stage, arm, label, color), in the paper's order.
FVE_BARS = (
    ("rl", "av", "AV\nexplanation", fs.NAVY),
    ("rl", "gold", "Sonnet\nexplanation", fs.RED),
    ("sft", "av", "AV explanation\nWarm-start", fs.BLUE),
    ("sft", "gold", "Sonnet explanation\nWarm-start", fs.PINK),
)
SUBSET = "strict_clean"

# Manual reading of the example against its passage.
AV_GROUNDED = (
    "Academic/technical paper on hydraulic structures",
    "stepped spillways",
    "water flow condition",
)
AV_CONFABULATED = (
    "Japanese engineering context for chute spillway design",
    "in mountainous regions",
    "The flow characteristics of a stepped spillway are studied for a supercritical flow.",
    "likely a Froude number or flow depth value",
    'Final token "."',
    "for a supercritical flow of height ..",
    "a measurement or range value",
    "the first drop of water over a chute spillway",
    'Example: "flow rate" or "hydraulic jump height" like "1.0" or "over a low weir"',
)
SONNET_GROUNDED = (
    "CFD study of stepped spillways using Flow-3D software",
    "fluid flow characteristics such",
    "specific hydraulic parameters",
    'The final word "such" begins a "such as" construction',
    "velocity profiles, pressure distribution, or turbulent kinetic energy",
)
SONNET_CONFABULATED: tuple[str, ...] = ()


def fve_by_explanation(table: dict, out_dir: Path) -> None:
    arms = [table["stages"][stage][SUBSET]["arms"][arm] for stage, arm, _, _ in FVE_BARS]
    means = np.array([100 * a["fve"] for a in arms])
    ci = np.array([[100 * v for v in a["fve_ci95"]] for a in arms])

    fs.apply(15)
    x = np.arange(len(FVE_BARS))
    fig, ax = plt.subplots(figsize=(9, 6.2))
    bars = ax.bar(
        x, means, width=0.64, yerr=np.vstack((means - ci[:, 0], ci[:, 1] - means)), capsize=6,
        color=[color for *_, color in FVE_BARS], edgecolor="black", linewidth=1.5, error_kw=fs.ERROR_KW,
    )
    ax.set_xticks(x, [label for _, _, label, _ in FVE_BARS])
    ax.set_ylabel("FVE (%)")
    ax.set_ylim(0, 80)
    ax.set_yticks(np.arange(0, 81, 10))
    ax.tick_params(axis="both", width=1.5, length=5)
    ax.tick_params(axis="x", length=0, pad=10, labelsize=13)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean - 4, f"{mean:.2f}%",
                ha="center", va="top", fontsize=13, fontweight="bold", color="white")
    fig.tight_layout(pad=1.2)
    fs.save(fig, out_dir, "fve_by_explanation")


def numbered_section(text: str, number: int, title: str, next_number: int) -> str:
    start = re.search(rf"^\s*{number}\. {re.escape(title)}\s*$\n=+\n", text, re.MULTILINE)
    if not start:
        raise ValueError(f"missing section {number}: {title}")
    tail = text[start.end():]
    end = re.search(rf"^=+\n{next_number}\. ", tail, re.MULTILINE)
    return tail[:end.start()].strip() if end else tail.strip()


def tagged_explanation(text: str, marker: str) -> str:
    start = text.index(marker) + len(marker)
    match = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text[start:], re.DOTALL)
    if not match:
        raise ValueError(f"no explanation after {marker!r}")
    return match.group(1).strip()


def styled_tokens(text: str, grounded: tuple[str, ...], confabulated: tuple[str, ...]):
    """Tokenize `text`; tokens overlapping a listed phrase take that phrase's style."""
    spans = []
    lower = text.lower()
    for phrases, style in ((grounded, "grounded"), (confabulated, "confabulated")):
        for phrase in phrases:
            start = lower.find(phrase.lower())
            if start < 0:
                raise ValueError(f"highlight phrase not found in explanation: {phrase!r}")
            spans.append((start, start + len(phrase), style))
    tokens = []
    for match in re.finditer(r"</?[^>\s]+>|\w+(?:['’]\w+)?|[^\w\s]", text):
        style = "plain"
        for start, end, candidate in spans:
            if match.start() < end and match.end() > start:
                style = candidate
        tokens.append((match.group(), style))
    return tokens


def draw_styled_lines(ax: plt.Axes, lines, x: float, y: float) -> None:
    styles = {"plain": (fs.INK, None), "grounded": (tp.GREEN, tp.GREEN_WASH), "confabulated": (tp.RED, tp.RED_WASH)}
    for row, line in enumerate(lines):
        baseline = y + (row + 0.5) * tp.LINE_H
        # Merge neighbouring tokens of the same style into one wash band.
        runs = []
        run_style = run_start = run_end = None
        for col, token, style in line:
            token_end = col + len(token)
            if style != "plain" and style == run_style and col <= run_end + 1:
                run_end = token_end
                continue
            if run_style is not None:
                runs.append((run_start, run_end, run_style))
            run_style, run_start, run_end = (style, col, token_end) if style != "plain" else (None, None, None)
        if run_style is not None:
            runs.append((run_start, run_end, run_style))
        for start, end, style in runs:
            ax.add_patch(Rectangle(
                (x + start * tp.CHAR_W, baseline - 0.40 * tp.LINE_H), (end - start) * tp.CHAR_W, 0.80 * tp.LINE_H,
                facecolor=styles[style][1], edgecolor="none", zorder=2,
            ))
        for col, token, style in line:
            tp.mono_text(ax, x + col * tp.CHAR_W, baseline, token, styles[style][0])


def av_vs_sonnet_example(record: str, out_dir: Path) -> None:
    target = re.search(r"token text '([^']*)'", record).group(1)
    marked = target.strip()
    passage = numbered_section(record, 2, "PASSAGE (exactly what the target model read)", 3)
    passage = re.sub(r"\s+", " ", passage)
    passage = re.sub(rf"\b{re.escape(marked)}$", f"[{marked}]", passage)
    passage = textwrap.fill(passage, width=118)

    av = tagged_explanation(record, "--- 5b. STORED generation, inner text - THIS IS WHAT WAS SCORED ---")
    gold = numbered_section(record, 6, "GOLD EXPLANATION (Sonnet, written with the passage in view)", 7)
    gold = gold.split("\n\nFact-check:", 1)[0].strip()
    av_fve = float(re.search(r"av \(round-trip\)\s+\S+\s+(-?\d+\.\d+)", record).group(1))
    gold_fve = float(re.search(r"gold \(Sonnet\)\s+\S+\s+(-?\d+\.\d+)", record).group(1))

    panel_width = 3.18
    columns = int((panel_width - 0.22) / tp.CHAR_W)
    av_lines = tp.wrap_tokens(styled_tokens(av, AV_GROUNDED, AV_CONFABULATED), columns)
    gold_lines = tp.wrap_tokens(styled_tokens(gold, SONNET_GROUNDED, SONNET_CONFABULATED), columns)
    explanation_height = max(len(av_lines), len(gold_lines)) * tp.LINE_H + 0.20
    passage_height = max(0.76, len(passage.splitlines()) * tp.LINE_H + 0.26)
    header_y = 1.13 + passage_height
    box_y = header_y + 0.46
    fig_height = box_y + explanation_height + 0.34

    fig = plt.figure(figsize=(7, fig_height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7)
    ax.set_ylim(fig_height, 0)
    ax.axis("off")

    left_x, right_x = 0.30, 3.52
    tp.heading(ax, left_x, 0.28, "(a)  Input")
    tp.caption(ax, left_x, 0.56, "PASSAGE")
    tp.add_box(ax, left_x, 0.64, 6.40, passage_height, fs.MUTED)
    tp.draw_marked_text(ax, passage, left_x + 0.11, 0.64 + passage_height / 2, marked)
    ax.text(6.70, 0.64 + passage_height + 0.17, f"selected token: “{target}”",
            fontsize=tp.BODY_PT, color=tp.BLUE, ha="right", style="italic")

    tp.heading(ax, left_x, header_y, "(b)  AV explanation")
    tp.heading(ax, right_x, header_y, "(c)  Sonnet explanation")
    ax.text(left_x, header_y + 0.30, f"FVE  {av_fve:.3f}", fontsize=7.5, fontweight="bold", color=tp.BLUE)
    ax.text(right_x, header_y + 0.30, f"FVE  {gold_fve:.3f}", fontsize=7.5, fontweight="bold", color=tp.RED)
    legend = ((5.35, "grounded", tp.GREEN, tp.GREEN_WASH), (6.25, "confabulated", tp.RED, tp.RED_WASH))
    for x, label, color, wash in legend:
        ax.text(x, header_y + 0.30, label, fontsize=5.8, color=color, va="center", ha="center",
                bbox={"facecolor": wash, "edgecolor": "none", "pad": 2.2})

    tp.add_box(ax, left_x, box_y, panel_width, explanation_height, tp.BLUE)
    tp.add_box(ax, right_x, box_y, panel_width, explanation_height, tp.RED)
    draw_styled_lines(ax, av_lines, left_x + 0.11, box_y + 0.10)
    draw_styled_lines(ax, gold_lines, right_x + 0.11, box_y + 0.10)

    out_dir.mkdir(parents=True, exist_ok=True)
    for extension, dpi in (("png", 600), ("pdf", 300)):
        path = out_dir / f"av_vs_sonnet_example.{extension}"
        fig.savefig(path, dpi=dpi, facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = results_dir("faithfulness")
    parser.add_argument("--fve-table", type=Path, default=root / "fve_table.json")
    parser.add_argument("--example", type=Path, default=root / "qualitative_example.txt")
    parser.add_argument("--out-dir", type=Path, default=root / "figures")
    args = parser.parse_args()

    fve_by_explanation(json.loads(args.fve_table.read_text()), args.out_dir)
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    av_vs_sonnet_example(args.example.read_text(encoding="utf-8"), args.out_dir)


if __name__ == "__main__":
    main()
