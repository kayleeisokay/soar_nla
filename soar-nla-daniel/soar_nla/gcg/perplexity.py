"""Unconditioned fluency scoring of AV/GCG explanations under the target model.

The AV NLL recorded by the GCG stage is conditioned on the activation and is
the term the optimizer penalizes, so it cannot serve as a held-out readability
measure. This module scores the explanation text on its own -- no activation,
no AV prompt wrapper, no chat template -- under the pinned target model, which
makes it an independent fluency signal comparable to the perplexity filters
used to detect adversarial suffixes.

Both a whole-text perplexity and a sliding-window maximum are reported. GCG
damage is local, so a mean over a 200-token explanation dilutes a short run of
garbage tokens that a windowed maximum still catches.

Explanations begin mid-thought, so the leading tokens are unpredictable for
reasons that have nothing to do with GCG. Scoring them puts 97% of windowed
maxima in the first eight positions and measures the abrupt opening instead of
the attack. A warmup prefix is therefore excluded from both statistics.

Absolute perplexities here are not comparable to running text: these are
unusual meta-commentary strings scored bare, with no chat template and no
preceding context. The paired original-vs-optimized contrast is the result;
the absolute level is not.

    python -m soar_nla.gcg.perplexity            # 1 GPU, writes results/gcg/perplexity/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from soar_nla.alpaca.subsets import MIN_SEQUENCE_TOKEN_INDEX, default_rollouts_path, load_jsonl
from soar_nla.hub import model_spec, resolve_snapshot
from soar_nla.paths import results_dir

WINDOW = 16
WARMUP = 8


def load_rows(gcg_path: Path, rollouts_path: Path, min_sequence_index: int) -> list[dict]:
    rollouts = {record["sample_id"]: record for record in load_jsonl(rollouts_path)}
    return [
        record
        for record in load_jsonl(gcg_path)
        if rollouts[record["sample_id"]]["sequence_token_index"] > min_sequence_index
    ]


@torch.no_grad()
def score_texts(
    model, tokenizer, texts: list[str], batch_size: int, device: str, warmup: int = WARMUP
) -> list[dict]:
    """Return per-text mean NLL and windowed maximum NLL, both in nats."""
    out: list[dict] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        # Predict token t+1 from position t; the first token has no context.
        shift_logits = logits[:, :-1]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].bool()

        token_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view(shift_labels.shape)

        for row in range(len(batch)):
            nll = token_nll[row][shift_mask[row]].double().cpu()
            # Drop the warmup prefix; those tokens have no usable left context.
            nll = nll[warmup:]
            n = int(nll.numel())
            if n == 0:
                out.append({"n_tokens": 0, "mean_nll": None, "window_max_nll": None})
                continue
            mean_nll = float(nll.mean())
            if n >= WINDOW:
                sums = torch.cumsum(torch.cat([torch.zeros(1, dtype=nll.dtype), nll]), 0)
                window_means = (sums[WINDOW:] - sums[:-WINDOW]) / WINDOW
                window_max = float(window_means.max())
            else:
                window_max = mean_nll
            out.append({"n_tokens": n, "mean_nll": mean_nll, "window_max_nll": window_max})
    return out


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gemma12b")
    parser.add_argument("--gcg-results", type=Path, default=results_dir("gcg") / "results.jsonl",
                        help="full GCG results.jsonl (needs the explanation texts)")
    parser.add_argument("--rollouts", type=Path, default=None,
                        help="soar_nla.alpaca rollouts.jsonl[.gz] (default: results/alpaca)")
    parser.add_argument("--min-sequence-index", type=int, default=MIN_SEQUENCE_TOKEN_INDEX)
    parser.add_argument("--betas", default="0.01,0.0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=results_dir("gcg") / "perplexity" / "results.jsonl")
    args = parser.parse_args()

    betas = [float(value) for value in args.betas.split(",")]
    spec = model_spec(args.model)

    rows = load_rows(args.gcg_results, args.rollouts or default_rollouts_path(args.model), args.min_sequence_index)
    print(f"loaded {len(rows)} gcg rows (sequence_token_index > {args.min_sequence_index})", flush=True)

    path = resolve_snapshot(spec.target_id, spec.target_revision)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(args.device)
    model.eval()
    print(f"loaded {spec.target_id} @ {spec.target_revision[:8]} on {args.device}", flush=True)

    # The originals are identical across betas; score the deduplicated set once.
    originals: dict[str, str] = {}
    for row in rows:
        originals.setdefault(row["sample_id"], row["original_explanation"])

    jobs: list[tuple[str, float | None, list[str], list[str]]] = [
        ("original", None, list(originals.keys()), list(originals.values()))
    ]
    for beta in betas:
        subset = [row for row in rows if row["beta"] == beta]
        jobs.append(
            ("optimized", beta, [r["sample_id"] for r in subset], [r["optimized_explanation"] for r in subset])
        )

    records = []
    for kind, beta, sample_ids, texts in jobs:
        print(f"scoring {kind} beta={beta} n={len(texts)}", flush=True)
        scored = score_texts(model, tokenizer, texts, args.batch_size, args.device, args.warmup)
        for sample_id, text, result in zip(sample_ids, texts, scored, strict=True):
            records.append(
                {
                    "protocol": "av-faithfulness-v1",
                    "model": args.model,
                    "scorer_id": spec.target_id,
                    "scorer_revision": spec.target_revision,
                    "kind": kind,
                    "beta": beta,
                    "sample_id": sample_id,
                    "window": WINDOW,
                    "warmup": args.warmup,
                    **result,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records to {args.out}", flush=True)

    for kind, beta, _, _ in jobs:
        subset = [r for r in records if r["kind"] == kind and r["beta"] == beta and r["mean_nll"] is not None]
        mean_nll = sum(r["mean_nll"] for r in subset) / len(subset)
        window = sorted(r["window_max_nll"] for r in subset)
        median_window = window[len(window) // 2]
        label = kind if beta is None else f"beta={beta}"
        print(
            f"{label:>12}  n={len(subset):<5} "
            f"mean NLL={mean_nll:.3f} nats  PPL(exp mean NLL)={math.exp(mean_nll):>8.2f}  "
            f"median windowed-max PPL={math.exp(median_window):>9.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
