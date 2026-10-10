"""Break down the FVE change from surgical edits by what happened to the final-token claim.

AV explanations end with a clause `Final token "<claim>" ...`. Using that claim
and the true sampled token, each edited example is assigned to:

- `final_token_kept`: the editor's claim is the identical string to the AV's.
- `final_token_changed`: the editor rewrote the claim string. Split by whether
  the AV's claim was wrong:
  - `final_token_changed_av_wrong`: the passage does not end with the AV's
    claim, ignoring case, whitespace, backticks and quote marks (a claim that is
    empty after that normalization, e.g. only whitespace, counts as correct when
    the passage ends in whitespace). Every wrong AV claim was changed.
  - `final_token_changed_av_right`: the change only altered formatting.
- `word_tokens` / `punctuation_whitespace_tokens`: the true sampled token does /
  does not contain a letter or digit (`soar_nla.alpaca.subsets.is_word_token`).

For every group: mean per-example FVE delta (edited - original) with the same
10,000-resample bootstrap (seed 0) as `score_corrected_av.py`.

    python -m soar_nla.faithfulness.surgical_edits.analysis
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np

from soar_nla.alpaca.subsets import default_rollouts_path, is_word_token, load_jsonl, load_rollouts
from soar_nla.io_utils import write_json
from soar_nla.paths import results_dir

FINAL_TOKEN = re.compile(r'(Final token\s+(?:`)?")([^"]*)', re.IGNORECASE)
_IGNORED = re.compile(r"[\s`'\"]")


def claimed_token(text: str) -> str | None:
    match = FINAL_TOKEN.search(text)
    return match.group(2) if match else None


def _normalise(text: str) -> str:
    return _IGNORED.sub("", text).lower()


def claim_matches_passage(claim: str, passage: str) -> bool:
    """Does the passage end with the claimed final token (lenient match)?"""
    if claim and passage.endswith(claim):
        return True
    if _normalise(claim):
        return _normalise(passage).endswith(_normalise(claim))
    return passage[-1:].isspace()


def _summary(deltas: np.ndarray, original: np.ndarray, corrected: np.ndarray) -> dict[str, Any]:
    if len(deltas) == 0:
        return {"n": 0}
    rng = np.random.default_rng(0)
    boot = rng.choice(deltas, size=(10_000, len(deltas)), replace=True).mean(axis=1)
    return {
        "n": int(len(deltas)),
        "original_mean_fve": float(original.mean()),
        "corrected_mean_fve": float(corrected.mean()),
        "mean_fve_delta": float(deltas.mean()),
        "mean_fve_delta_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "fraction_reconstruction_improved": float((deltas > 0).mean()),
    }


def reconstruction_path(edits_path: Path) -> Path:
    stem = edits_path.with_suffix("")
    return stem.with_name(stem.name + "_reconstruction.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    edits_dir = results_dir("surgical_edits")
    parser.add_argument("--edits", type=Path, nargs="+",
                        default=[edits_dir / "sol_equal_quote_length_100.jsonl",
                                 edits_dir / "sol_equal_quote_length_101_200.jsonl"],
                        help="correct_av_explanations.py outputs; each needs its <stem>_reconstruction.jsonl")
    parser.add_argument("--rollouts", type=Path, default=None,
                        help="soar_nla.alpaca rollouts.jsonl[.gz] (default: results/alpaca)")
    parser.add_argument("--output", type=Path, default=edits_dir / "analysis.json")
    args = parser.parse_args()

    rollouts = {row["sample_id"]: row for row in load_rollouts(args.rollouts or default_rollouts_path())}
    edits: list[dict[str, Any]] = []
    scores: dict[str, dict[str, Any]] = {}
    for path in args.edits:
        edits.extend(load_jsonl(path))
        scores.update({row["sample_id"]: row for row in load_jsonl(reconstruction_path(path))})
    if len({row["sample_id"] for row in edits}) != len(edits):
        raise RuntimeError("duplicate sample_ids across --edits files")

    av_claims = [claimed_token(row["original_explanation"]) for row in edits]
    edit_claims = [claimed_token(row["corrected_explanation"]) for row in edits]
    if any(claim is None for claim in av_claims + edit_claims):
        raise RuntimeError("an explanation has no 'Final token \"...\"' clause")
    kept = np.array([a == b for a, b in zip(av_claims, edit_claims)])
    av_right = np.array([claim_matches_passage(a, row["passage"]) for a, row in zip(av_claims, edits)])
    word = np.array([is_word_token(rollouts[row["sample_id"]]["sampled_token_text"]) for row in edits])
    deltas = np.array([scores[row["sample_id"]]["fve_delta"] for row in edits])
    original = np.array([scores[row["sample_id"]]["original_fve"] for row in edits])
    corrected = np.array([scores[row["sample_id"]]["corrected_fve"] for row in edits])

    groups = {
        "all": np.ones(len(edits), dtype=bool),
        "final_token_kept": kept,
        "final_token_changed": ~kept,
        "final_token_changed_av_wrong": ~kept & ~av_right,
        "final_token_changed_av_right": ~kept & av_right,
        "word_tokens": word,
        "punctuation_whitespace_tokens": ~word,
    }
    report: dict[str, Any] = {
        "groups": {name: _summary(deltas[mask], original[mask], corrected[mask]) for name, mask in groups.items()},
        "av_final_token_wrong": {
            "all": int((~av_right).sum()),
            "word_tokens": int((~av_right & word).sum()),
            "punctuation_whitespace_tokens": int((~av_right & ~word).sum()),
            "wrong_but_kept": int((~av_right & kept).sum()),
        },
    }
    report["paper_claims"] = {
        "overall_delta_fve": report["groups"]["all"]["mean_fve_delta"],
        "kept_final_token_delta_fve": report["groups"]["final_token_kept"]["mean_fve_delta"],
        "changed_final_token_av_wrong_delta_fve": report["groups"]["final_token_changed_av_wrong"]["mean_fve_delta"],
        "word_tokens_delta_fve": report["groups"]["word_tokens"]["mean_fve_delta"],
    }
    write_json(args.output, report)

    print(f"{'group':32s} {'n':>4s} {'orig FVE':>9s} {'edit FVE':>9s} {'delta':>8s}  95% CI")
    for name, value in report["groups"].items():
        if not value["n"]:
            continue
        low, high = value["mean_fve_delta_ci95"]
        print(
            f"{name:32s} {value['n']:4d} {value['original_mean_fve']:9.4f} {value['corrected_mean_fve']:9.4f} "
            f"{value['mean_fve_delta']:+8.4f}  [{low:+.4f}, {high:+.4f}]"
        )
    print(f"AV final-token claim wrong: {report['av_final_token_wrong']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
