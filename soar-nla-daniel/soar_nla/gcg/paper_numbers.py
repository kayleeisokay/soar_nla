"""Recompute the GCG numbers quoted in the paper from the per-example results.

FVE is per-example `1 - ar_loss`, where `ar_loss = MSE(normalize(AR(x)),
normalize(a)) / Var(v)` as recorded by `soar_nla.gcg.gcg` (see
`soar_nla.gcg.models`). The GCG gain is the mean of
`original_ar_loss - optimized_ar_loss`; "±" is the standard error of the mean.

The paper's GCG results use `late_positions`: the 1,573 of the 2,000 Alpaca
rollouts whose sampled position is after sequence token 50 (the NLA was
trained on positions 50 and later). The random control was run on the same
set. Per example and beta the paper reports:

- FVE of the original and the GCG explanation;
- the edit count: positions whose final GCG token differs from the original
  (the number of edits the random control makes);
- mean per-token NLL of the explanation text under Gemma-3-12B-IT itself, with
  no activation or prompt (`soar_nla.gcg.perplexity`; betas 0.01 and 0 only),
  next to the AV-policy NLL that the objective penalizes.

The sentence-level faithfulness of the beta=0.01 explanations (17.5% vs 14.3%
supported) is checked by `soar_nla.faithfulness.paper_numbers`.

Other subsets, for reference: `all` 2,000 rollouts; `long_responses` 1,543
(`assistant_token_count >= 100`); `sol_word_tokens`, the
143 of the 200 surgically edited examples whose sampled token contains a letter
or digit ("GCG on the same examples", 8.4% at beta=0.01).

    python -m soar_nla.gcg.paper_numbers
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from soar_nla.alpaca.subsets import (
    default_rollouts_path,
    is_word_token,
    late_positions,
    load_jsonl,
    load_rollouts,
    long_responses,
)
from soar_nla.io_utils import write_json
from soar_nla.paths import results_dir

from . import BETAS

SLIM_FIELDS = (
    "sample_id",
    "selection_rank",
    "beta",
    "token_count",
    "steps_accepted",
    "edit_count",
    "roundtrip_token_stable",
    "original_ar_loss",
    "optimized_ar_loss",
    "original_av_nll",
    "optimized_av_nll",
)

PAPER_SUBSET = "late_positions"

# (description, value in the paper, recomputed value, decimals compared)
PAPER_CLAIMS = (
    ("original AV explanation FVE (%)", 72.1, lambda p: 100 * p["0.0"]["original_fve"], 1),
    ("original AV explanation FVE SE (%)", 0.5, lambda p: 100 * p["0.0"]["original_fve_se"], 1),
    ("GCG FVE, beta=0 (%)", 80.7, lambda p: 100 * p["0.0"]["gcg_fve"], 1),
    ("GCG FVE, beta=0.01 (%)", 81.4, lambda p: 100 * p["0.01"]["gcg_fve"], 1),
    ("Gemma NLL, original (nats/token)", 3.84, lambda p: p["0.01"]["original_gemma_nll"], 2),
    ("Gemma NLL, beta=0.01 (nats/token)", 4.63, lambda p: p["0.01"]["gcg_gemma_nll"], 2),
    ("Gemma NLL, beta=0 (nats/token)", 5.58, lambda p: p["0.0"]["gcg_gemma_nll"], 2),
    ("mean edit count, beta=0.1", 4.36, lambda p: p["0.1"]["mean_edit_count"], 2),
    ("mean edit count, beta=0.01", 14.36, lambda p: p["0.01"]["mean_edit_count"], 2),
    ("mean edit count, beta=0", 16.41, lambda p: p["0.0"]["mean_edit_count"], 2),
)


def edit_count(row: dict[str, Any]) -> int:
    """Positions whose final GCG token differs from the original token."""
    if "edit_count" in row:
        return int(row["edit_count"])
    original: dict[int, int] = {}
    final: dict[int, int] = {}
    for step in row["trajectory"]:
        if step["accepted"]:
            original.setdefault(step["position"], step["old_token_id"])
            final[step["position"]] = step["new_token_id"]
    return sum(1 for position, token in final.items() if token != original[position])


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def gcg_block(rows: list[dict[str, Any]], ids: set[str], gemma_nll: dict | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for beta in BETAS:
        subset = [row for row in rows if float(row["beta"]) == beta and row["sample_id"] in ids]
        original = np.array([row["original_ar_loss"] for row in subset], dtype=np.float64)
        optimized = np.array([row["optimized_ar_loss"] for row in subset], dtype=np.float64)
        original_fve, original_se = _mean_se(1.0 - original)
        gcg_fve, gcg_se = _mean_se(1.0 - optimized)
        entry = {
            "n": len(subset),
            "original_fve": original_fve,
            "original_fve_se": original_se,
            "gcg_fve": gcg_fve,
            "gcg_fve_se": gcg_se,
            "gcg_fve_gain": float((original - optimized).mean()),
            "mean_original_av_nll": float(np.mean([row["original_av_nll"] for row in subset])),
            "mean_optimized_av_nll": float(np.mean([row["optimized_av_nll"] for row in subset])),
            "mean_edit_count": float(np.mean([edit_count(row) for row in subset])),
        }
        if gemma_nll is not None:
            scored_original = gemma_nll.get(("original", None), {})
            scored_gcg = gemma_nll.get(("optimized", beta), {})
            originals = [scored_original[row["sample_id"]] for row in subset if row["sample_id"] in scored_original]
            optimized_nll = [scored_gcg[row["sample_id"]] for row in subset if row["sample_id"] in scored_gcg]
            entry["original_gemma_nll"] = float(np.mean(originals)) if originals else None
            entry["gcg_gemma_nll"] = float(np.mean(optimized_nll)) if optimized_nll else None
            entry["n_gemma_nll"] = len(optimized_nll)
        block[str(beta)] = entry
    return block


def control_block(rows: list[dict[str, Any]], ids: set[str]) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for beta in BETAS:
        subset = [row for row in rows if float(row["beta"]) == beta and row["sample_id"] in ids]
        block[str(beta)] = {
            "n": len(subset),
            **{
                field: float(np.mean([row[field] for row in subset]))
                for field in ("original_fve", "gcg_fve", "random_fve", "gcg_fve_gain", "random_fve_gain")
            },
        }
    return block


def load_gemma_nll(path: Path) -> dict | None:
    """{(kind, beta): {sample_id: mean_nll}} from soar_nla.gcg.perplexity output."""
    if not path.exists():
        return None
    table: dict = {}
    for row in load_jsonl(path):
        if row["mean_nll"] is None:
            continue
        beta = None if row["beta"] is None else float(row["beta"])
        table.setdefault((row["kind"], beta), {})[row["sample_id"]] = row["mean_nll"]
    return table


def export_slim(gcg_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        for row in gcg_rows:
            slim = {field: row[field] for field in SLIM_FIELDS if field != "edit_count"}
            slim["edit_count"] = edit_count(row)
            handle.write((json.dumps(slim, sort_keys=True) + "\n").encode("utf-8"))
    print(f"wrote {len(gcg_rows)} rows to {path}")


def _default_perplexity() -> Path:
    directory = results_dir("gcg") / "perplexity"
    plain = directory / "results.jsonl"
    return plain if plain.exists() else directory / "results.jsonl.gz"


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gcg_dir = results_dir("gcg")
    edits_dir = results_dir("surgical_edits")
    parser.add_argument("--gcg-results", type=Path, default=gcg_dir / "gcg_losses.jsonl.gz",
                        help="GCG results.jsonl (or the gcg_losses.jsonl.gz projection)")
    parser.add_argument("--rollouts", type=Path, default=None,
                        help="soar_nla.alpaca rollouts.jsonl[.gz] (default: results/alpaca)")
    parser.add_argument("--control-scores", type=Path,
                        default=gcg_dir / "random_control" / "ar_scores" / "kitft.jsonl.gz",
                        help="controls score-ar output for the kitft AR")
    parser.add_argument("--perplexity", type=Path, default=None,
                        help="soar_nla.gcg.perplexity output (default: results/gcg/perplexity)")
    parser.add_argument("--sol-edits", type=Path, nargs="+",
                        default=[edits_dir / "sol_equal_quote_length_100.jsonl",
                                 edits_dir / "sol_equal_quote_length_101_200.jsonl"],
                        help="surgical-edit JSONLs; only their sample_ids are used")
    parser.add_argument("--output", type=Path, default=gcg_dir / "paper_numbers.json")
    parser.add_argument("--export-gcg-losses", type=Path, default=None,
                        help="also write the per-example loss projection of --gcg-results here (.jsonl.gz)")
    args = parser.parse_args()

    gcg_rows = load_jsonl(args.gcg_results)
    if args.export_gcg_losses:
        export_slim(gcg_rows, args.export_gcg_losses)
    rollouts = load_rollouts(args.rollouts or default_rollouts_path())
    by_id = {row["sample_id"]: row for row in rollouts}
    sol_ids = [row["sample_id"] for path in args.sol_edits for row in load_jsonl(path)]
    if len(set(sol_ids)) != len(sol_ids):
        raise RuntimeError("duplicate sample_ids across --sol-edits files")
    gemma_nll = load_gemma_nll(args.perplexity or _default_perplexity())

    subsets = {
        "late_positions": {row["sample_id"] for row in late_positions(rollouts)},
        "all": {row["sample_id"] for row in rollouts},
        "long_responses": {row["sample_id"] for row in long_responses(rollouts)},
        "sol_all": set(sol_ids),
        "sol_word_tokens": {sid for sid in sol_ids if is_word_token(by_id[sid]["sampled_token_text"])},
    }

    report: dict[str, Any] = {
        "fve_definition": "FVE = 1 - ar_loss; ar_loss = MSE(normalize(AR(x)), normalize(a)) / Var(v)",
        "paper_subset": PAPER_SUBSET,
        "subset_sizes": {name: len(ids) for name, ids in subsets.items()},
        "gcg": {
            name: gcg_block(gcg_rows, ids, gemma_nll if name == PAPER_SUBSET else None)
            for name, ids in subsets.items()
        },
    }
    control_rows = load_jsonl(args.control_scores) if args.control_scores.exists() else []
    if control_rows:
        report["random_control"] = {PAPER_SUBSET: control_block(control_rows, subsets[PAPER_SUBSET])}

    paper = report["gcg"][PAPER_SUBSET]
    checks = []
    for name, value, compute, decimals in PAPER_CLAIMS:
        try:
            got = compute(paper)
        except (KeyError, TypeError):
            got = None
        ok = got is not None and f"{got:.{decimals}f}" == f"{value:.{decimals}f}"
        checks.append({"claim": name, "paper": value, "recomputed": got, "match": ok})
    report["paper_claims"] = {
        "checks": checks,
        "gcg_beta0.01_gain_on_sol_word_tokens": report["gcg"]["sol_word_tokens"]["0.01"]["gcg_fve_gain"],
    }
    write_json(args.output, report)

    control = report.get("random_control", {}).get(PAPER_SUBSET, {})
    print(f"Paper subset: {PAPER_SUBSET} (n = {len(subsets[PAPER_SUBSET])})\n")
    print(f"{'beta':>5}  {'orig FVE ± SE':>15}  {'GCG FVE ± SE':>15}  {'gain':>8}  {'random':>8}  "
          f"{'edits':>6}  {'AV NLL orig->GCG':>16}  {'Gemma NLL orig->GCG':>19}")
    for beta in BETAS:
        row = paper[str(beta)]
        random_gain = control.get(str(beta), {}).get("random_fve_gain")
        gemma = (f"{row['original_gemma_nll']:.2f} -> {row['gcg_gemma_nll']:.2f}"
                 if row.get("gcg_gemma_nll") is not None else "-")
        print(f"{beta:>5g}  {row['original_fve']:.4f}±{row['original_fve_se']:.4f}  "
              f"{row['gcg_fve']:.4f}±{row['gcg_fve_se']:.4f}  {row['gcg_fve_gain']:+.4f}  "
              f"{_fmt(random_gain, '+.4f'):>8}  {row['mean_edit_count']:6.2f}  "
              f"{row['mean_original_av_nll']:.3f} -> {row['mean_optimized_av_nll']:.3f}  {gemma:>19}")
    print()
    for check in checks:
        print(f"{'OK ' if check['match'] else 'BAD'} {check['claim']:<38} paper {check['paper']:>6}  "
              f"recomputed {_fmt(check['recomputed'], '.4f')}")
    sol = report["paper_claims"]["gcg_beta0.01_gain_on_sol_word_tokens"]
    print(f"\nGCG beta=0.01 gain on the 143 word-token surgical-edit examples: {sol:+.4f} (paper: 8.4%)")
    print("\nOther subsets (FVE gain, beta = 0.1 / 0.01 / 0):")
    for name, block in report["gcg"].items():
        if name != PAPER_SUBSET:
            gains = " / ".join(f"{block[str(b)]['gcg_fve_gain']:+.4f}" for b in BETAS)
            print(f"  {name:16s} n={block['0.1']['n']:5d}  {gains}")
    print(f"wrote {args.output}")
    if not all(check["match"] for check in checks):
        raise SystemExit("some paper numbers did not reproduce")


if __name__ == "__main__":
    main()
