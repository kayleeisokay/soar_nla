"""Compare AR reconstruction of original and surgically edited AV explanations.

Per-example FVE is `1 - MSE / Var(v)` with both vectors rescaled to the AR's
`mse_scale` and `Var(v)` the empirical variance of the 2,000 Alpaca activations
(identical to the GCG convention). `fve_delta = corrected_fve - original_fve`.

    python -m soar_nla.faithfulness.surgical_edits.score_corrected_av \\
        --corrections results/surgical_edits/sol_equal_quote_length_100.jsonl

writes `<stem>_reconstruction.jsonl` and `<stem>_reconstruction_summary.json`
next to the input unless --output/--summary are given. One GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from soar_nla.gcg.models import empirical_directional_variance, normalise_rows
from soar_nla.hub import MODEL_SPECS, model_spec, resolve_snapshot
from soar_nla.io_utils import read_jsonl
from soar_nla.nla_inference import NLACritic
from soar_nla.paths import results_dir


@torch.inference_mode()
def reconstruct(critic: NLACritic, texts: list[str]) -> torch.Tensor:
    prompts = [critic.template.format(explanation=text) for text in texts]
    critic.tokenizer.padding_side = "right"
    batch = critic.tokenizer(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
    ids = batch["input_ids"].to(critic.device)
    mask = batch["attention_mask"].to(critic.device)
    hidden = critic.backbone.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    last = mask.sum(dim=1) - 1
    return critic.value_head(hidden[torch.arange(len(ids), device=ids.device), last]).float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corrections", type=Path, required=True,
                        help="output of correct_av_explanations.py")
    parser.add_argument("--alpaca-dir", type=Path, default=results_dir("alpaca") / "gemma12b",
                        help="soar_nla.alpaca results for one model (rollouts/)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--model", default="gemma12b", choices=sorted(MODEL_SPECS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    stem = args.corrections.with_suffix("")
    args.output = args.output or stem.with_name(stem.name + "_reconstruction.jsonl")
    args.summary = args.summary or stem.with_name(stem.name + "_reconstruction_summary.json")

    spec = model_spec(args.model)
    corrections = read_jsonl(args.corrections)
    rollouts = read_jsonl(args.alpaca_dir / "rollouts" / "rollouts.jsonl")
    rollout_by_id = {row["sample_id"]: row for row in rollouts}
    activations = load_file(str(args.alpaca_dir / "rollouts" / "activations.safetensors"))["activations"].float()

    critic = NLACritic(
        resolve_snapshot(spec.ar_id, spec.ar_revision),
        device=args.device,
        dtype=torch.bfloat16,
    )
    variance = empirical_directional_variance(activations, critic.mse_scale)
    output = []
    for start in range(0, len(corrections), args.batch_size):
        chunk = corrections[start : start + args.batch_size]
        texts = [r["original_explanation"] for r in chunk] + [r["corrected_explanation"] for r in chunk]
        predictions = reconstruct(critic, texts)
        original_predictions, corrected_predictions = predictions.chunk(2)
        ranks = torch.tensor([int(rollout_by_id[r["sample_id"]]["selection_rank"]) for r in chunk])
        gold = activations[ranks]
        gold_n = normalise_rows(gold, critic.mse_scale)
        original_n = normalise_rows(original_predictions, critic.mse_scale)
        corrected_n = normalise_rows(corrected_predictions, critic.mse_scale)
        original_mse = ((original_n - gold_n) ** 2).mean(dim=-1)
        corrected_mse = ((corrected_n - gold_n) ** 2).mean(dim=-1)
        for row, before, after in zip(chunk, original_mse.tolist(), corrected_mse.tolist(), strict=True):
            output.append({
                "sample_id": row["sample_id"],
                "original_mse": before,
                "corrected_mse": after,
                "original_fve": 1.0 - before / variance,
                "corrected_fve": 1.0 - after / variance,
                "fve_delta": (before - after) / variance,
                "reconstruction_improved": after < before,
            })
        print(f"[score] {min(start + args.batch_size, len(corrections))}/{len(corrections)}", flush=True)

    deltas = np.asarray([r["fve_delta"] for r in output])
    rng = np.random.default_rng(0)
    boot = rng.choice(deltas, size=(10_000, len(deltas)), replace=True).mean(axis=1)
    summary = {
        "n": len(output),
        "activation_variance": variance,
        "original_mean_fve": float(np.mean([r["original_fve"] for r in output])),
        "corrected_mean_fve": float(np.mean([r["corrected_fve"] for r in output])),
        "mean_fve_delta": float(deltas.mean()),
        "mean_fve_delta_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "fraction_reconstruction_improved": float(np.mean([r["reconstruction_improved"] for r in output])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in output:
            handle.write(json.dumps(row) + "\n")
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
