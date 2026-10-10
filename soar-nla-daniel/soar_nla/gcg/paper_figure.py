"""Fig. 3 of the paper: GCG FVE across β (a) next to one qualitative example (b–d).

(a) reads the per-β GCG and random-control FVE means and bootstrap 95% CIs from
`random_control/summary.json` (n = 1,573 late-position rollouts); the dotted line
is the original AV explanations' FVE. (b–d) draw the vacation-packing rollout
from `qualitative_example.txt` (prompt, response with the sampled token, the
original and the β = 0.01 GCG explanation, with changed words highlighted); its
two FVEs come from `gcg_losses.jsonl.gz`.

Outputs in `<results>/figures/`: `gcg_beta_fve.{png,pdf}`,
`gcg_example_beta0.01.{png,pdf}` and the side-by-side composite used in the paper,
`gcg_fve_and_example.png`. Needs the `figures` extra (Pillow for the composite,
uharfbuzz + fontTools for the Devanagari token in the GCG text).

    python -m soar_nla.gcg.paper_figure
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from soar_nla import figure_style as fs
from soar_nla import text_panels as tp
from soar_nla.alpaca.subsets import load_jsonl
from soar_nla.paths import results_dir

BETAS = (0.1, 0.01, 0.0)
# Tick labels as printed in the paper.
BETA_LABELS = {0.1: "0.100", 0.01: "0.0100", 0.0: "0.00"}
EXAMPLE_BETA = 0.01
SCORER = "kitft"


def format_3sf(value: float) -> str:
    return f"{value:.3g}"


def beta_fve_panel(summary: dict, out_dir: Path) -> Path:
    scores = summary["scorers"][SCORER]
    gcg = np.array([100 * scores[str(b)]["gcg_fve"]["mean"] for b in BETAS])
    gcg_ci = np.array([[100 * v for v in scores[str(b)]["gcg_fve"]["ci95"]] for b in BETAS])
    rnd = np.array([100 * scores[str(b)]["random_fve"]["mean"] for b in BETAS])
    rnd_ci = np.array([[100 * v for v in scores[str(b)]["random_fve"]["ci95"]] for b in BETAS])
    # The original explanations are the same text for every β; their FVE is the shared baseline.
    baseline = 100 * scores[str(BETAS[0])]["original_fve"]["mean"]

    fs.apply(16)
    x = np.arange(len(BETAS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.5))
    gcg_bars = ax.bar(
        x - width / 2, gcg, width=width, yerr=np.vstack((gcg - gcg_ci[:, 0], gcg_ci[:, 1] - gcg)),
        capsize=6, color=fs.NAVY, edgecolor="black", linewidth=1.5, error_kw=fs.ERROR_KW, label="GCG",
    )
    rnd_bars = ax.bar(
        x + width / 2, rnd, width=width, yerr=np.vstack((rnd - rnd_ci[:, 0], rnd_ci[:, 1] - rnd)),
        capsize=6, color=fs.GREY, edgecolor="black", linewidth=1.5, error_kw=fs.ERROR_KW,
        label="Random (control)",
    )
    ax.axhline(baseline, color=fs.RED, linestyle=":", linewidth=2.2, label="No GCG baseline", zorder=0)
    ax.text(0.985, baseline, f"{format_3sf(baseline)}%", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=12, fontweight="bold", color=fs.RED)

    ax.set_xticks(x, [BETA_LABELS[b] for b in BETAS])
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("FVE (%)")
    ax.set_ylim(60, 84)
    ax.set_yticks(np.arange(60, 84.1, 4))
    ax.tick_params(axis="both", width=1.5, length=5)
    ax.legend(fontsize=12, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    for bars, means, ci in ((gcg_bars, gcg, gcg_ci), (rnd_bars, rnd, rnd_ci)):
        for bar, mean, upper in zip(bars, means, ci[:, 1]):
            ax.annotate(f"{format_3sf(mean)}%", xy=(bar.get_x() + bar.get_width() / 2, upper),
                        xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
                        fontsize=12, fontweight="bold")
    fig.tight_layout(pad=1.2)
    return fs.save(fig, out_dir, "gcg_beta_fve")[0]


def section(text: str, heading: str, next_heading: str | None = None) -> str:
    start = re.search(rf"^{re.escape(heading)}\n=+\n", text, re.MULTILINE)
    if not start:
        raise ValueError(f"missing section: {heading}")
    tail = text[start.end():]
    if next_heading:
        end = re.search(rf"\n\n{re.escape(next_heading)}\n=+", tail)
        if end:
            tail = tail[:end.start()]
    return tail.strip()


def field(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(.+)$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing field: {name}")
    return match.group(1).strip()


def diff_words(original: str, edited: str):
    """Word/punctuation pieces of both texts, marked when they fall in a non-equal diff opcode."""
    # Keep a whole Devanagari sequence (with vowel signs) as one piece.
    pattern = re.compile(r"[ऀ-ॿ]+|\w+(?:['’]\w+)?|[^\w\s]", re.UNICODE)
    left, right = pattern.findall(original), pattern.findall(edited)
    left_marked = [(token, False) for token in left]
    right_marked = [(token, False) for token in right]
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
        if tag != "equal":
            for i in range(i1, i2):
                left_marked[i] = (left[i], True)
            for j in range(j1, j2):
                right_marked[j] = (right[j], True)
    return left_marked, right_marked


def example_fves(losses_path: Path, sample_id: str, beta: float) -> tuple[float, float]:
    for row in load_jsonl(losses_path):
        if row["sample_id"] == sample_id and float(row["beta"]) == beta:
            return 1.0 - row["original_ar_loss"], 1.0 - row["optimized_ar_loss"]
    raise KeyError(f"no GCG row for sample {sample_id} at beta {beta} in {losses_path}")


def example_panel(record: str, original_fve: float, gcg_fve: float, out_dir: Path) -> Path:
    """Panels (b) input, (c) original AV explanation, (d) GCG explanation."""
    beta = float(field(record, "GCG beta"))
    target = field(record, "Selected token").strip('"')
    prompt = section(record, "FULL PROMPT", "FULL MODEL GENERATION (Gemma 12B rollout)")
    prompt = re.sub(r"^user\n|\nmodel$", "", prompt).strip()
    generation = section(record, "FULL MODEL GENERATION (Gemma 12B rollout)", "ORIGINAL AV EXPLANATION")
    original = section(record, "ORIGINAL AV EXPLANATION", f"GCG-EDITED AV EXPLANATION (beta = {beta:g})")
    edited = section(record, f"GCG-EDITED AV EXPLANATION (beta = {beta:g})")

    # Mark the sampled token (" of", assistant token 140) in the flattened response.
    marked = target.strip()
    context = re.sub(r"\s+", " ", generation.replace("**", ""))
    context = context.replace(" regardless of ", f" regardless [{marked}] ", 1)
    if f"[{marked}]" not in context:
        raise ValueError("could not locate the sampled token in the response")
    context = textwrap.fill(context, width=118)
    context_height = max(0.88, len(context.splitlines()) * tp.LINE_H + 0.28)
    context_shift = context_height - 0.88

    original_tokens, edited_tokens = diff_words(original, edited)
    panel_width = 3.18
    columns = int((panel_width - 0.22) / tp.CHAR_W)
    left_lines = tp.wrap_tokens(original_tokens, columns)
    right_lines = tp.wrap_tokens(edited_tokens, columns)
    explanation_height = max(len(left_lines), len(right_lines)) * tp.LINE_H + 0.20

    fig_height = 3.65 + context_shift + explanation_height
    fig = plt.figure(figsize=(7, fig_height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7)
    ax.set_ylim(fig_height, 0)
    ax.axis("off")

    left_x, right_x, full_width = 0.30, 3.52, 6.40
    tp.heading(ax, left_x, 0.28, "(b)  Input")
    tp.caption(ax, left_x, 0.56, "INSTRUCTION")
    tp.add_box(ax, left_x, 0.64, full_width, 0.42, fs.MUTED)
    ax.text(left_x + 0.11, 0.85, prompt, fontsize=tp.BODY_PT, family=tp.MONO, color=fs.INK, va="center")
    tp.caption(ax, left_x, 1.29, "FULL MODEL RESPONSE")
    tp.add_box(ax, left_x, 1.37, full_width, context_height, fs.MUTED)
    tp.draw_marked_text(ax, context, left_x + 0.11, 1.37 + context_height / 2, marked)
    ax.text(6.70, 2.42 + context_shift, f"selected token: “{target}”", fontsize=tp.BODY_PT,
            color=tp.BLUE, ha="right", style="italic")

    header_y = 2.85 + context_shift
    tp.heading(ax, left_x, header_y, "(c)  Original AV explanation")
    tp.heading(ax, right_x, header_y, rf"(d)  After GCG ($\beta={beta:.4f}$)")
    ax.text(left_x, header_y + 0.30, f"FVE  {original_fve:.3f}", fontsize=7.5, fontweight="bold", color=tp.BLUE)
    ax.text(right_x, header_y + 0.30, f"FVE  {gcg_fve:.3f}   ({gcg_fve - original_fve:+.3f})",
            fontsize=7.5, fontweight="bold", color=tp.GREEN)
    box_y = header_y + 0.46
    tp.add_box(ax, left_x, box_y, panel_width, explanation_height, tp.BLUE)
    tp.add_box(ax, right_x, box_y, panel_width, explanation_height, tp.RED)
    tp.draw_token_lines(ax, left_lines, left_x + 0.11, box_y + 0.10, tp.BLUE, tp.BLUE_WASH)
    tp.draw_token_lines(ax, right_lines, right_x + 0.11, box_y + 0.10, tp.RED, tp.RED_WASH)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"gcg_example_beta{beta:g}"
    paths = []
    for extension, dpi in (("png", 600), ("pdf", 300)):
        path = out_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, facecolor="white")
        paths.append(path)
        print(f"wrote {path}")
    plt.close(fig)
    return paths[0]


def side_by_side(beta_png: Path, example_png: Path, out: Path) -> None:
    """Scale (a) to the example's width, title it, and place it left of (b–d)."""
    from PIL import Image, ImageDraw, ImageFont

    example = Image.open(example_png).convert("RGB")
    beta = Image.open(beta_png).convert("RGB")
    width = example.width
    beta = beta.resize((width, round(beta.height * width / beta.width)), Image.Resampling.LANCZOS)

    title_band, gutter = 150, 100
    canvas = Image.new("RGB", (2 * width + gutter, max(title_band + beta.height, example.height)), "white")
    canvas.paste(beta, (0, title_band))
    canvas.paste(example, (width + gutter, 0))
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"), 82)
    ImageDraw.Draw(canvas).text((180, 92), "(a)  Aggregate FVE across β", fill="#242424", font=font, anchor="ls")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, dpi=(300, 300), optimize=True)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = results_dir("gcg")
    parser.add_argument("--control-summary", type=Path, default=root / "random_control" / "summary.json")
    parser.add_argument("--gcg-results", type=Path, default=root / "gcg_losses.jsonl.gz",
                        help="GCG results.jsonl or the gcg_losses.jsonl.gz projection")
    parser.add_argument("--example", type=Path, default=root / "qualitative_example.txt")
    parser.add_argument("--out-dir", type=Path, default=root / "figures")
    args = parser.parse_args()

    beta_png = beta_fve_panel(json.loads(args.control_summary.read_text()), args.out_dir)
    record = args.example.read_text(encoding="utf-8")
    sample_id, beta = field(record, "Sample ID"), float(field(record, "GCG beta"))
    original_fve, gcg_fve = example_fves(args.gcg_results, sample_id, beta)
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    example_png = example_panel(record, original_fve, gcg_fve, args.out_dir)
    side_by_side(beta_png, example_png, args.out_dir / "gcg_fve_and_example.png")


if __name__ == "__main__":
    main()
