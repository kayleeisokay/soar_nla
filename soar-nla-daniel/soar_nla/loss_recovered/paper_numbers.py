"""Loss Recovered numbers from the per-window losses (CPU only).

    LR = (H_0 - H_method) / (H_0 - H_orig)

with H the mean next-token CE over all scored tokens of all windows (every
window contributes the same number of tokens per span, so this is the mean of
the window means). 95% intervals: 10,000 bootstrap resamples of windows, the
same resampled windows for every method of a span.

Without arguments this reads the shipped `results/loss_recovered/window_losses.jsonl.gz`
and `evaluation.json`, writes `paper_numbers.json` and checks the paper's values.

    python -m soar_nla.loss_recovered.paper_numbers
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from soar_nla.io_utils import write_json

from .common import default_results_dir
from .config import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, SETTINGS

# (claim, span, method, value quoted in the paper in %)
PAPER_CLAIMS = (
    ("NLA, tokens 51-100 (1-50 clean)", "suffix_50_100", "nla", 90.88),
    ("NLA, tokens 1-50", "prefix_50", "nla", 64.37),
    ("NLA, tokens 1-100", "prefix_100", "nla", 69.95),
    ("SAE (16k, L0 = 20.82), tokens 51-100, norm-rescaled", "suffix_50_100", "sae_norm", 97.58),
)
PAPER_SAE_L0 = 20.82
PAPER_WINDOWS = 1543


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def loss_recovered(clean: np.ndarray, method: np.ndarray, zero: np.ndarray) -> float:
    return float((zero.mean() - method.mean()) / (zero.mean() - clean.mean()))


def summarize(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for setting, (start, stop) in SETTINGS.items():
        ce = {name: np.array([row["settings"][setting]["ce"][name] for row in rows]) for name in rows[0]["settings"][setting]["ce"]}
        methods = [name for name in ce if name not in ("clean", "zero")]
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = {method: np.empty(BOOTSTRAP_RESAMPLES) for method in methods}
        for i in range(BOOTSTRAP_RESAMPLES):
            idx = rng.integers(0, len(rows), len(rows))
            for method in methods:
                draws[method][i] = loss_recovered(ce["clean"][idx], ce[method][idx], ce["zero"][idx])
        out[setting] = {
            "span": [start, stop],
            "scored_tokens_per_window": stop - start - 1,
            "mean_ce": {name: float(values.mean()) for name, values in ce.items()},
            "loss_recovered": {
                method: {
                    "value": loss_recovered(ce["clean"], ce[method], ce["zero"]),
                    "ci95": [float(q) for q in np.quantile(draws[method], [0.025, 0.975])],
                }
                for method in methods
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--out", type=Path, default=None, help="default: <results-dir>/paper_numbers.json")
    args = parser.parse_args()

    rows = load_rows(args.results_dir / "window_losses.jsonl.gz")
    evaluation = json.loads((args.results_dir / "evaluation.json").read_text())
    settings = summarize(rows)
    claims, failures = {}, 0
    for claim, setting, method, expected in PAPER_CLAIMS:
        value = settings[setting]["loss_recovered"][method]["value"]
        ok = round(100 * value, 2) == expected
        failures += not ok
        claims[claim] = {"span": setting, "method": method, "value": value, "paper_percent": expected, "ok": ok}
        print(f"{'ok' if ok else 'MISMATCH':8s} {claim}: {100 * value:.2f}% (paper {expected:.2f}%)")
    mean_l0 = evaluation["sae"]["mean_l0"]
    for claim, ok, text in (
        ("SAE mean L0", round(mean_l0, 2) == PAPER_SAE_L0, f"{mean_l0:.2f} (paper {PAPER_SAE_L0})"),
        ("Windows", len(rows) == PAPER_WINDOWS, f"{len(rows)} (paper {PAPER_WINDOWS:,})"),
    ):
        failures += not ok
        print(f"{'ok' if ok else 'MISMATCH':8s} {claim}: {text}")

    write_json(
        args.out or args.results_dir / "paper_numbers.json",
        {
            "formula": "(mean_zero_ce - mean_method_ce) / (mean_zero_ce - mean_clean_ce)",
            "windows": len(rows),
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "window"},
            "sae": evaluation.get("sae"),
            "settings": settings,
            "paper_claims": claims,
        },
    )
    if failures:
        raise SystemExit(f"{failures} paper number(s) do not match")


if __name__ == "__main__":
    main()
