#!/usr/bin/env python3
"""Reconstruction numbers of the warm-start ablation (paper appendix, "Skipping Warm-Start").

From the per-step curves written by parse_training_logs.py:

  peak FVE      best single training-batch FVE (rank mean over the 8 DP ranks,
                256 prompts x 8 samples) at any step <= --max-step
  held-out FVE  the last periodic held-out evaluation (128 prompts from
                held-out documents, every 10 steps) at a step <= --max-step

--max-step defaults to 600, the largest budget available to all three runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from soar_nla.paths import results_dir
from soar_nla.warmstart_ablation.runs import COMPARE_STEP, FVE_CSV, RUN_LABELS, RUNS


def pct(value: str) -> str:
    """Fraction (as written in the CSV) -> percent with one decimal, half-up."""
    return str((Decimal(value) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def summarize(path: Path, max_step: int) -> dict:
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["step"]) <= max_step]
    peak = max(rows, key=lambda r: float(r["fve_train"]))
    held = [r for r in rows if r["fve_heldout"] != ""]
    last = held[-1]
    return {
        "csv": path.name,
        "last_step": int(rows[-1]["step"]),
        "peak_fve_train": float(peak["fve_train"]),
        "peak_fve_train_pct": pct(peak["fve_train"]),
        "peak_step": int(peak["step"]),
        "heldout_fve": float(last["fve_heldout"]),
        "heldout_fve_pct": pct(last["fve_heldout"]),
        "heldout_step": int(last["step"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv-dir", type=Path, default=results_dir("warmstart_ablation"),
                    help="directory holding the three fve_curve_*.csv files")
    ap.add_argument("--max-step", type=int, default=COMPARE_STEP)
    ap.add_argument("--out-dir", type=Path, default=results_dir("warmstart_ablation"))
    a = ap.parse_args()

    table = {run: summarize(a.csv_dir / FVE_CSV[run], a.max_step) for run in RUNS}
    lines = [
        f"| Run | Setting | Peak training-batch FVE (step <= {a.max_step}) | Held-out FVE (last eval <= {a.max_step}) |",
        "|---|---|---|---|",
    ]
    for run in RUNS:
        t = table[run]
        lines.append(
            f"| {run} | {RUN_LABELS[run]} | {t['peak_fve_train_pct']}% "
            f"({t['peak_fve_train']:.4f} @ step {t['peak_step']}) | "
            f"{t['heldout_fve_pct']}% ({t['heldout_fve']:.3f} @ step {t['heldout_step']}) |"
        )
    md = "\n".join(lines)
    print(md)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "fve_table.md").write_text(md + "\n")
    (a.out_dir / "fve_table.json").write_text(
        json.dumps({"max_step": a.max_step, "runs": table}, indent=2) + "\n")
    print(f"[done] -> {a.out_dir / 'fve_table.md'}, {a.out_dir / 'fve_table.json'}")


if __name__ == "__main__":
    main()
