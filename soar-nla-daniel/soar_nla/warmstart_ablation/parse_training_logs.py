#!/usr/bin/env python3
"""Rebuild the per-step FVE curves of the warm-start ablation from trainer logs.

`nla.train_rl_vllm` runs with `--no-wandb`, so the text log is the only record
of the training curve. Every data-parallel rank prints its own step line,

    step 0537 | r -0.177 | FVE 75.6% | ar_mse 0.1766 | kl 0.9512 | ent 1.938 | ext 100% | t 61s

and rank 0 prints the held-out evaluation every `--eval-every` steps,

    [eval@540] reward -0.188 | FVE 74.1% | ext 100%

Ranks write concurrently, so lines are sometimes glued together; the parser
therefore scans the whole text with a regex instead of splitting on newlines.
The training-batch FVE of a step is the mean over ranks (each rank scores 1/8
of the global batch), the held-out FVE is the rank-0 eval value.

A run that was resumed is passed as several logs in chronological order. A
later log takes over from its first step: all steps >= that step from earlier
logs are discarded (this is how a branch from an earlier checkpoint, e.g. run
B's restart at step 200, is stitched).

Output columns (one row per step):
    step, fve_train (rank mean), fve_train_ema (EMA, alpha 0.1),
    fve_train_std / _min / _max (population std / min / max over ranks),
    n_ranks, reward, ar_mse, kl, entropy, ext_rate (rank means),
    fve_heldout (empty on non-eval steps), source_log (basename of the log)
All fractions, not percentages; values rounded to 6 decimals.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

_NUM = r"(-?(?:[\d.]+|nan|inf))"
STEP_RE = re.compile(
    rf"step (\d{{4,}}) \| r {_NUM} \| FVE {_NUM}% \| ar_mse {_NUM} \| kl {_NUM} "
    rf"\| ent {_NUM} \| ext {_NUM}% \| t (\d+)s"
)
EVAL_RE = re.compile(rf"\[eval@(\d+)\] reward {_NUM} \| FVE {_NUM}% \| ext {_NUM}%")

COLUMNS = [
    "step", "fve_train", "fve_train_ema", "fve_train_std", "fve_train_min",
    "fve_train_max", "n_ranks", "reward", "ar_mse", "kl", "entropy", "ext_rate",
    "fve_heldout", "source_log",
]
EMA_ALPHA = 0.1


def parse_log(path: Path) -> tuple[dict[int, list[list[float]]], dict[int, float]]:
    """Return ({step: [[r, fve%, ar_mse, kl, ent, ext%], ...per rank]}, {step: eval fve%})."""
    text = path.read_text(errors="replace")
    steps: dict[int, list[list[float]]] = defaultdict(list)
    for m in STEP_RE.finditer(text):
        steps[int(m.group(1))].append([float(x) for x in m.groups()[1:7]])
    evals = {int(m.group(1)): float(m.group(3)) for m in EVAL_RE.finditer(text)}
    return dict(steps), evals


def _mean(values: list[float]) -> float:
    # fsum: exact summation, so the 6-decimal rounding below is reproducible.
    return math.fsum(values) / len(values)


def _round(x: float) -> float:
    return round(x, 6)


def stitch(logs: list[Path]) -> list[dict]:
    per_step: dict[int, tuple[list[list[float]], str]] = {}
    held: dict[int, float] = {}
    for path in logs:
        steps, evals = parse_log(path)
        if not steps:
            raise ValueError(f"no step lines found in {path}")
        first = min(steps)
        per_step = {s: v for s, v in per_step.items() if s < first}
        held = {s: v for s, v in held.items() if s < first}
        for s, ranks in steps.items():
            per_step[s] = (ranks, path.name)
        for s, v in evals.items():
            if s >= first:
                held[s] = v

    rows, ema = [], None
    for s in sorted(per_step):
        ranks, src = per_step[s]
        fve = [r[1] / 100 for r in ranks]
        m = _mean(fve)
        ema = m if ema is None else (1 - EMA_ALPHA) * ema + EMA_ALPHA * m
        rows.append({
            "step": s,
            "fve_train": _round(m),
            "fve_train_ema": _round(ema),
            "fve_train_std": _round(statistics.pstdev(fve)),
            "fve_train_min": _round(min(fve)),
            "fve_train_max": _round(max(fve)),
            "n_ranks": len(ranks),
            "reward": _round(_mean([r[0] for r in ranks])),
            "ar_mse": _round(_mean([r[2] for r in ranks])),
            "kl": _round(_mean([r[3] for r in ranks])),
            "entropy": _round(_mean([r[4] for r in ranks])),
            "ext_rate": _round(_mean([r[5] for r in ranks]) / 100),
            "fve_heldout": _round(held[s] / 100) if s in held else "",
            "source_log": src,
        })
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def compare_csv(new: Path, ref: Path) -> int:
    """Count rows whose text differs from a reference CSV (0 = identical)."""
    a = new.read_text().splitlines()
    b = ref.read_text().splitlines()
    n_diff = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
    print(f"[compare] {new.name} vs {ref}: {len(a)} vs {len(b)} lines, {n_diff} differ")
    return n_diff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", nargs="+", type=Path, required=True,
                    help="trainer logs of one run, in chronological (leg) order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None,
                    help="optional reference CSV; exits non-zero if any row differs")
    a = ap.parse_args()

    rows = stitch(a.logs)
    write_csv(rows, a.out)
    print(f"[done] {len(rows)} steps ({rows[0]['step']}..{rows[-1]['step']}) -> {a.out}")
    if a.compare is not None and compare_csv(a.out, a.compare):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
