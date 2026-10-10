"""Random-substitution control for the GCG experiment.

For each example and beta, the control makes the same number of token edits as
the final GCG explanation (its internal-token Hamming distance from the AV
explanation), but at uniformly random positions with uniformly random
non-special vocabulary tokens and no model-based selection. GCG, control and
original explanations are then scored by the same AR.

Scope is frozen to examples whose sampled full-sequence position is strictly
after token 50 (`soar_nla.alpaca.subsets.late_positions`, 1,573 rollouts).

    python -m soar_nla.gcg.controls unconditional-random
    python -m soar_nla.gcg.controls score-ar --scorer kitft      # one GPU
    python -m soar_nla.gcg.controls analyze
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from soar_nla.alpaca.subsets import MIN_SEQUENCE_TOKEN_INDEX
from soar_nla.hub import MODEL_SPECS, model_spec, resolve_snapshot
from soar_nla.io_utils import read_jsonl, stable_seed, write_json, write_jsonl
from soar_nla.nla_inference import NLACritic
from soar_nla.paths import results_dir

from . import BETAS
from .models import empirical_directional_variance, normalise_rows

# Feeds stable_seed for the random draws and bootstrap intervals; keep verbatim.
PROTOCOL = "av-faithfulness-gcg-controls-v1"
SCORERS = ("kitft",)


def _load_source(gcg_results: Path, rollouts_path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = read_jsonl(gcg_results)
    rollouts = {row["sample_id"]: row for row in read_jsonl(rollouts_path)}
    rows = [
        row
        for row in rows
        if int(rollouts[row["sample_id"]]["sequence_token_index"]) > MIN_SEQUENCE_TOKEN_INDEX
    ]
    rows.sort(key=lambda row: (int(row["selection_rank"]), -float(row["beta"])))
    expected_betas = set(BETAS)
    by_sample: dict[str, set[float]] = defaultdict(set)
    for row in rows:
        by_sample[row["sample_id"]].add(float(row["beta"]))
    bad = {sample_id: values for sample_id, values in by_sample.items() if values != expected_betas}
    if bad:
        raise RuntimeError(f"eligible samples lack complete beta rows: {list(bad.items())[:3]}")
    return rows, rollouts


def _internal_ids(row: dict, tokenizer) -> tuple[list[int], list[int], list[int]]:
    original = tokenizer(row["original_explanation"], add_special_tokens=False)["input_ids"]
    final = list(original)
    for step in row["trajectory"]:
        if not step["accepted"]:
            continue
        position = int(step["position"])
        old = int(step["old_token_id"])
        new = int(step["new_token_id"])
        if final[position] != old:
            raise RuntimeError(
                f"trajectory mismatch for {row['sample_id']} beta={row['beta']} "
                f"step={step['step']} pos={position}: have {final[position]}, expected {old}"
            )
        final[position] = new
    decoded = tokenizer.decode(final, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != row["optimized_explanation"]:
        raise RuntimeError(f"stored optimized text does not match trajectory for {row['sample_id']}")
    changed = [position for position, (left, right) in enumerate(zip(original, final, strict=True)) if left != right]
    return list(original), final, changed


def _unconditional_random(row: dict, tokenizer) -> tuple[str, int, list[int], list[int]]:
    """One fully random edit-count control, with no model-based selection."""
    original, final, changed = _internal_ids(row, tokenizer)
    if not changed:
        return row["original_explanation"], 0, [], []
    rng = random.Random(stable_seed(PROTOCOL, row["sample_id"], row["beta"], "fully-random-edit-count"))
    forbidden = set(tokenizer.all_special_ids)
    vocab_size = len(tokenizer)
    for _attempt in range(100_000):
        positions = rng.sample(range(len(original)), len(changed))
        assigned = []
        for position in positions:
            while True:
                token = rng.randrange(vocab_size)
                if token not in forbidden and token != original[position]:
                    assigned.append(token)
                    break
        candidate = list(original)
        for position, token in zip(positions, assigned, strict=True):
            candidate[position] = token
        if sum(a != b for a, b in zip(original, candidate, strict=True)) != len(changed):
            continue
        return (
            tokenizer.decode(candidate, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            len(changed),
            positions,
            assigned,
        )
    raise RuntimeError(f"failed unconditional random draw for {row['sample_id']} beta={row['beta']}")


def unconditional_random(args: argparse.Namespace) -> None:
    rows, rollouts = _load_source(args.gcg_results, args.rollouts)
    spec = model_spec(args.model)
    checkpoint = resolve_snapshot(spec.target_id, spec.target_revision)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    output = []
    for index, row in enumerate(rows, start=1):
        text, distance, positions, replacements = _unconditional_random(row, tokenizer)
        output.append({
            "protocol": "av-faithfulness-gcg-random-edit-count-v1",
            "model": args.model,
            "sample_id": row["sample_id"],
            "selection_rank": row["selection_rank"],
            "assistant_token_index": rollouts[row["sample_id"]]["assistant_token_index"],
            "sequence_token_index": rollouts[row["sample_id"]]["sequence_token_index"],
            "beta": row["beta"],
            "original_explanation": row["original_explanation"],
            "gcg_explanation": row["optimized_explanation"],
            "random_explanation": text,
            "accepted_steps": row["steps_accepted"],
            "final_token_hamming_distance": distance,
            "gcg_changed_positions": _internal_ids(row, tokenizer)[2],
            "random_changed_positions": positions,
            "replacement_token_ids": replacements,
            "replacement_token_multiset_matched": False,
            "selection": "same GCG final Hamming distance; uniformly random positions and non-special vocabulary tokens; no model scoring",
        })
        if index % 500 == 0:
            print(f"[unconditional random] {index}/{len(rows)}", flush=True)
    out = Path(args.output_dir) / "random" / "results.jsonl"
    write_jsonl(out, output)
    print(f"wrote {len(output)} unconditional controls to {out}", flush=True)


def _score_text_jobs(
    jobs: list[dict],
    activations: torch.Tensor,
    variance: float,
    reconstruct: Callable[[list[str]], torch.Tensor],
    batch_size: int,
) -> dict[tuple[str, str, float | None], float]:
    losses: dict[tuple[str, str, float | None], float] = {}
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        predictions = reconstruct([job["text"] for job in chunk]).float().cpu()
        gold = torch.stack([activations[int(job["selection_rank"])] for job in chunk]).float()
        pred_n = normalise_rows(predictions, math.sqrt(activations.shape[1]))
        gold_n = normalise_rows(gold, math.sqrt(activations.shape[1]))
        values = ((pred_n - gold_n) ** 2).mean(dim=-1) / variance
        for job, value in zip(chunk, values.tolist(), strict=True):
            losses[(job["sample_id"], job["kind"], job["beta"])] = float(value)
        print(f"[AR score] {min(start + batch_size, len(jobs))}/{len(jobs)} texts", flush=True)
    return losses


def _kitft_reconstructor(args: argparse.Namespace):
    spec = model_spec(args.model)
    checkpoint = resolve_snapshot(spec.ar_id, spec.ar_revision)
    critic = NLACritic(checkpoint, device=args.device, dtype=torch.bfloat16)
    tokenizer = critic.tokenizer
    tokenizer.padding_side = "right"

    @torch.inference_mode()
    def reconstruct(texts: list[str]) -> torch.Tensor:
        prompts = [critic.template.format(explanation=text) for text in texts]
        batch = tokenizer(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
        ids = batch["input_ids"].to(args.device)
        mask = batch["attention_mask"].to(args.device)
        hidden = critic.backbone.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        last = mask.sum(dim=1) - 1
        return critic.value_head(hidden[torch.arange(len(ids), device=args.device), last]).float().cpu()

    return reconstruct, f"{spec.ar_id}@{spec.ar_revision}", "kitft exact critic template; add_special_tokens=True"


def score_ar(args: argparse.Namespace) -> None:
    controls = read_jsonl(Path(args.output_dir) / "random" / "results.jsonl")
    if not controls:
        raise RuntimeError("generate the random controls before AR scoring")
    activations = load_file(
        str(Path(args.alpaca_dir) / args.model / "rollouts" / "activations.safetensors")
    )["activations"].float()
    variance = empirical_directional_variance(activations, math.sqrt(activations.shape[1]))

    reconstruct, checkpoint, contract = _kitft_reconstructor(args)

    jobs: list[dict] = []
    seen_original: set[str] = set()
    for row in controls:
        sample_id = row["sample_id"]
        if sample_id not in seen_original:
            jobs.append(
                {
                    "sample_id": sample_id,
                    "selection_rank": row["selection_rank"],
                    "kind": "original",
                    "beta": None,
                    "text": row["original_explanation"],
                }
            )
            seen_original.add(sample_id)
        for kind, field in (("gcg", "gcg_explanation"), ("random", "random_explanation")):
            jobs.append(
                {
                    "sample_id": sample_id,
                    "selection_rank": row["selection_rank"],
                    "kind": kind,
                    "beta": float(row["beta"]),
                    "text": row[field],
                }
            )
    losses = _score_text_jobs(jobs, activations, variance, reconstruct, args.batch_size)
    output = []
    for row in controls:
        key = row["sample_id"]
        beta = float(row["beta"])
        original = losses[(key, "original", None)]
        gcg = losses[(key, "gcg", beta)]
        random_loss = losses[(key, "random", beta)]
        output.append(
            {
                "protocol": PROTOCOL,
                "scorer": args.scorer,
                "scorer_checkpoint": checkpoint,
                "sample_id": key,
                "selection_rank": row["selection_rank"],
                "assistant_token_index": row["assistant_token_index"],
                "sequence_token_index": row["sequence_token_index"],
                "beta": beta,
                "activation_variance": variance,
                "original_ar_loss": original,
                "gcg_ar_loss": gcg,
                "random_ar_loss": random_loss,
                "original_fve": 1.0 - original,
                "gcg_fve": 1.0 - gcg,
                "random_fve": 1.0 - random_loss,
                "gcg_fve_gain": original - gcg,
                "random_fve_gain": original - random_loss,
                "gcg_minus_random_fve": random_loss - gcg,
            }
        )
    out_dir = Path(args.output_dir) / "ar_scores"
    write_jsonl(out_dir / f"{args.scorer}.jsonl", output)
    write_json(
        out_dir / f"{args.scorer}.manifest.json",
        {
            "protocol": PROTOCOL,
            "scorer": args.scorer,
            "checkpoint": checkpoint,
            "contract": contract,
            "n_rows": len(output),
            "n_samples": len(seen_original),
            "token_filter": f"sequence_token_index > {MIN_SEQUENCE_TOKEN_INDEX}",
            "activation_variance": variance,
        },
    )
    print(f"wrote {len(output)} {args.scorer} AR score rows", flush=True)


def _bootstrap(values: np.ndarray, seed: int, repetitions: int) -> tuple[float, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    chunk = 500
    for start in range(0, repetitions, chunk):
        size = min(chunk, repetitions - start)
        picks = rng.integers(0, len(values), size=(size, len(values)))
        draws[start : start + size] = values[picks].mean(axis=1)
    return float(values.mean()), [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _stat(values: list[float], label: str, repetitions: int) -> dict:
    mean, ci = _bootstrap(
        np.asarray(values), stable_seed(PROTOCOL, label) % (2**32), repetitions
    )
    return {"mean": mean, "ci95": ci}


def analyze(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    controls = read_jsonl(output_dir / "random" / "results.jsonl")
    scores = {
        scorer: read_jsonl(output_dir / "ar_scores" / f"{scorer}.jsonl")
        for scorer in SCORERS
    }
    expected = len(controls)
    if any(len(rows) != expected for rows in scores.values()):
        raise RuntimeError(f"score lengths do not match {expected} controls")

    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "token_filter": f"sequence_token_index > {MIN_SEQUENCE_TOKEN_INDEX}",
        "n_samples": len({row["sample_id"] for row in controls}),
        "n_beta_rows": len(controls),
        "betas": {},
        "scorers": {},
    }
    for beta in BETAS:
        subset = [row for row in controls if float(row["beta"]) == beta]
        distances = np.asarray([row["final_token_hamming_distance"] for row in subset], dtype=np.float64)
        report["betas"][str(beta)] = {
            "n": len(subset),
            "edit_distance": {
                "mean": float(distances.mean()),
                "median": float(np.median(distances)),
                "min": int(distances.min()),
                "max": int(distances.max()),
                "exact_match_rate": 1.0,
                "replacement_token_multiset_match_rate": float(
                    np.mean([row["replacement_token_multiset_matched"] for row in subset])
                ),
            },
        }

    for scorer, rows in scores.items():
        block = {}
        for beta in BETAS:
            subset = [row for row in rows if float(row["beta"]) == beta]
            metrics = {}
            for field in (
                "original_fve",
                "gcg_fve",
                "random_fve",
                "gcg_fve_gain",
                "random_fve_gain",
                "gcg_minus_random_fve",
            ):
                metrics[field] = _stat(
                    [float(row[field]) for row in subset], f"{scorer}:{beta}:{field}", args.resamples
                )
            metrics["gcg_gain_positive_rate"] = float(
                np.mean([row["gcg_fve_gain"] > 0 for row in subset])
            )
            metrics["gcg_beats_random_rate"] = float(
                np.mean([row["gcg_minus_random_fve"] > 0 for row in subset])
            )
            block[str(beta)] = {"n": len(subset), **metrics}
        report["scorers"][scorer] = block

    write_json(output_dir / "summary.json", report)
    lines = [
        "# GCG unconditional random-substitution control",
        "",
        f"Scope: {report['n_samples']} Gemma-3-12B Alpaca rollouts with sequence_token_index > 50.",
        "",
        "Random controls match only the final internal-token Hamming distance;",
        "positions and non-special vocabulary tokens use one unconditional uniform draw, with no model-score selection.",
        "All intervals are paired 95% percentile bootstrap intervals over examples.",
        "",
    ]
    for scorer in SCORERS:
        lines.extend(
            [
                f"## {scorer}",
                "",
                "| beta | original FVE | GCG FVE | random FVE | GCG gain | random gain | GCG − random |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for beta in BETAS:
            value = report["scorers"][scorer][str(beta)]
            def cell(field: str) -> str:
                stat = value[field]
                return f"{stat['mean']:.4f} [{stat['ci95'][0]:.4f}, {stat['ci95'][1]:.4f}]"
            lines.append(
                f"| {beta:g} | {cell('original_fve')} | {cell('gcg_fve')} | {cell('random_fve')} | "
                f"{cell('gcg_fve_gain')} | {cell('random_fve_gain')} | {cell('gcg_minus_random_fve')} |"
            )
        lines.append("")
    lines.extend(["## Edit-distance diagnostics", "", "| beta | edit distance mean (range) |", "|---:|---:|"])
    for beta in BETAS:
        distance = report["betas"][str(beta)]["edit_distance"]
        lines.append(f"| {beta:g} | {distance['mean']:.2f} ({distance['min']}–{distance['max']}) |")
    lines.append("")
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output_dir / 'summary.json'} and REPORT.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    default_output = results_dir("gcg") / "random_control"

    unconditional_p = sub.add_parser("unconditional-random", help="draw the random-edit controls (CPU only)")
    unconditional_p.add_argument("--model", default="gemma12b", choices=sorted(MODEL_SPECS))
    unconditional_p.add_argument("--gcg-results", type=Path, default=results_dir("gcg") / "results.jsonl")
    unconditional_p.add_argument(
        "--rollouts", type=Path, default=results_dir("alpaca") / "gemma12b" / "rollouts" / "rollouts.jsonl"
    )
    unconditional_p.add_argument("--output-dir", type=Path, default=default_output)

    score_p = sub.add_parser("score-ar", help="score original, GCG and random explanations with the AR (one GPU)")
    score_p.add_argument("--model", default="gemma12b", choices=sorted(MODEL_SPECS))
    score_p.add_argument("--alpaca-dir", type=Path, default=results_dir("alpaca"))
    score_p.add_argument("--output-dir", type=Path, default=default_output)
    score_p.add_argument("--scorer", choices=SCORERS, default="kitft")
    score_p.add_argument("--device", default="cuda:0")
    score_p.add_argument("--batch-size", type=int, default=8)

    analyze_p = sub.add_parser("analyze", help="bootstrap summary.json and REPORT.md")
    analyze_p.add_argument("--output-dir", type=Path, default=default_output)
    analyze_p.add_argument("--resamples", type=int, default=10_000)

    args = parser.parse_args()
    if args.command == "unconditional-random":
        unconditional_random(args)
    elif args.command == "score-ar":
        score_ar(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
