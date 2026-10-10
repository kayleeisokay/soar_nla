"""Greedy Coordinate Gradient (GCG) over every token of an AV explanation.

Starting from the AV explanation, GCG substitutes payload tokens while the AR
and AV prompt wrappers stay fixed. Each beta is optimized separately on

    L(x) = MSE(normalize(AR(x)), normalize(a)) / Var(v) + beta * mean_token_NLL_AV(x | a)

The regularizer is KL to the AV under GCG's deterministic candidate policy,
`KL(delta_x || p_AV(. | a)) = -log p_AV(x | a)`, length-normalized.

At each step the optimizer (1) differentiates the reconstruction term through
the AR, (2) differentiates the AV NLL, including both the token's role as
context and its relaxed one-hot role as the prediction target, (3) computes the
exact combined vocabulary gradient in chunks, and (4) samples one-token
candidates from the per-position top-k gradient choices, evaluates the exact
joint loss and greedily accepts the best improving substitution.

Gemma's tokenizer can merge final explanation punctuation with the AR's closing
tag. The gradient uses the usual token-splice approximation, but candidate
evaluation decodes each candidate and re-tokenizes the complete AR prompt, so
every greedy decision and every reported AR loss is from the exact string path.
`ar_gradient_splice_stable` records whether the approximation happened to be
token-identical for each starting explanation. Token length is fixed and
special tokens cannot be inserted.
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from soar_nla.alpaca.config import N_EXAMPLES, PROTOCOL_VERSION
from soar_nla.hub import MODEL_SPECS, model_spec
from soar_nla.io_utils import ResumableJSONL, read_jsonl, stable_seed, write_json, write_jsonl
from soar_nla.paths import results_dir

from . import BETAS
from .models import (
    DifferentiableAR,
    DifferentiableAV,
    empirical_directional_variance,
    load_ar,
    load_av,
)


@dataclass
class StepRecord:
    step: int
    accepted: bool
    position: int | None
    old_token_id: int | None
    new_token_id: int | None
    old_token_text: str | None
    new_token_text: str | None
    ar_loss: float
    av_nll: float
    objective: float


class GCGOptimizer:
    """Greedy Coordinate Gradient over every token of an AV explanation.

    AR and AV may share a GPU or live on separate GPUs. The vocabulary-wide
    combined gradient is accumulated in chunks, so this computes the true top-k
    under the joint first-order objective.
    """

    def __init__(
        self,
        ar: DifferentiableAR,
        av: DifferentiableAV,
        variance: float,
        *,
        topk: int,
        candidate_batch_size: int,
        exact_microbatch_size: int,
        vocab_chunk_size: int,
    ):
        self.ar = ar
        self.av = av
        self.variance = variance
        self.topk = topk
        self.candidate_batch_size = candidate_batch_size
        self.exact_microbatch_size = exact_microbatch_size
        self.vocab_chunk_size = vocab_chunk_size
        if len(ar.tokenizer) != len(av.tokenizer):
            raise RuntimeError(f"AV/AR vocab mismatch: {len(av.tokenizer)} vs {len(ar.tokenizer)}")
        self.vocab_size = len(av.tokenizer)
        # Gemma pads its embedding table to a multiple of 64: 262,208 rows for
        # 262,145 tokenizer IDs. Padded rows are parameters, not valid tokens.
        if ar.embed_weight.shape[0] < self.vocab_size or av.embed_weight.shape[0] < self.vocab_size:
            raise RuntimeError("embedding table is smaller than the tokenizer vocabulary")
        self.forbidden_ids = sorted(set(ar.tokenizer.all_special_ids) | set(av.tokenizer.all_special_ids))

    def _combined_topk(
        self,
        current_ids: torch.Tensor,
        ar_gradient: torch.Tensor,
        av_gradient: torch.Tensor | None,
        av_log_probs: torch.Tensor | None,
        beta: float,
    ) -> torch.Tensor:
        """Lowest-gradient token IDs per position under the joint objective."""
        length = len(current_ids)
        keep = min(self.topk, self.vocab_size - len(self.forbidden_ids) - 1)
        best_scores = torch.full((length, keep), float("inf"), device=self.ar.device)
        best_ids = torch.full((length, keep), -1, dtype=torch.long, device=self.ar.device)
        ar_grad = ar_gradient.to(self.ar.device, self.ar.embed_weight.dtype)
        av_grad = None if av_gradient is None else av_gradient.to(self.av.device, self.av.embed_weight.dtype)
        for start in range(0, self.vocab_size, self.vocab_chunk_size):
            end = min(start + self.vocab_chunk_size, self.vocab_size)
            ar_embeddings = self.ar.embed_weight[start:end] * self.ar.embed_scale
            score = (ar_grad @ ar_embeddings.T).float()
            if beta:
                assert av_grad is not None and av_log_probs is not None
                av_embeddings = self.av.embed_weight[start:end] * self.av.embed_scale
                # Context contribution plus the relaxed target-label contribution.
                av_score = (av_grad @ av_embeddings.T).float()
                av_score = av_score + (-av_log_probs[:, start:end] / length)
                score.add_(av_score.to(self.ar.device), alpha=beta)
            local_ids = torch.arange(start, end, device=self.ar.device).view(1, -1).expand(length, -1)
            if self.forbidden_ids:
                forbidden = [token - start for token in self.forbidden_ids if start <= token < end]
                if forbidden:
                    score[:, forbidden] = float("inf")
            current = current_ids.to(self.ar.device)
            in_chunk = (current >= start) & (current < end)
            if in_chunk.any():
                score[in_chunk, current[in_chunk] - start] = float("inf")
            merged_scores = torch.cat((best_scores, score), dim=1)
            merged_ids = torch.cat((best_ids, local_ids), dim=1)
            best_scores, indices = torch.topk(merged_scores, k=keep, dim=1, largest=False, sorted=True)
            best_ids = merged_ids.gather(1, indices)
            del score, merged_scores, merged_ids, indices
        if (best_ids < 0).any():
            raise RuntimeError("failed to obtain GCG candidates")
        return best_ids.cpu()

    def _sample_candidates(self, x_ids: torch.Tensor, top_ids: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        candidates = []
        seen: set[tuple[int, int]] = set()
        attempts = 0
        limit = self.candidate_batch_size * 20
        while len(candidates) < self.candidate_batch_size and attempts < limit:
            attempts += 1
            position = int(torch.randint(len(x_ids), (1,), generator=generator).item())
            column = int(torch.randint(top_ids.shape[1], (1,), generator=generator).item())
            token = int(top_ids[position, column].item())
            key = (position, token)
            if key in seen or token == int(x_ids[position]):
                continue
            seen.add(key)
            candidate = x_ids.clone()
            candidate[position] = token
            candidates.append(candidate)
        if not candidates:
            raise RuntimeError("GCG candidate sampler produced no substitutions")
        return torch.stack(candidates)

    def _exact_losses(
        self, candidates: torch.Tensor, activation: torch.Tensor, beta: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.ar.device == self.av.device:
            # Avoid simultaneous temporary activations/logits when both 12B
            # checkpoints share one 80 GiB device.
            ar_losses = self.ar.losses(candidates, activation, self.variance, self.exact_microbatch_size)
            av_nlls = self.av.nlls(candidates, activation, self.exact_microbatch_size)
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                ar_future = pool.submit(
                    self.ar.losses, candidates, activation, self.variance, self.exact_microbatch_size
                )
                # Measure NLL even for beta=0 so fluency degradation remains
                # observable in the unregularized control.
                av_future = pool.submit(self.av.nlls, candidates, activation, self.exact_microbatch_size)
                ar_losses = ar_future.result()
                av_nlls = av_future.result()
        return ar_losses, av_nlls, ar_losses + beta * av_nlls

    def optimise(
        self,
        explanation: str,
        activation: torch.Tensor,
        beta: float,
        steps: int,
        seed: int,
        max_explanation_tokens: int,
    ) -> dict[str, Any]:
        x_ids = torch.tensor(self.av.tokenizer(explanation, add_special_tokens=False)["input_ids"], dtype=torch.long)
        if not 0 < len(x_ids) <= max_explanation_tokens:
            raise RuntimeError(f"explanation has {len(x_ids)} tokens; allowed range is 1..{max_explanation_tokens}")
        # Gemma can merge final punctuation with the AR's closing </text> tag.
        # Gradients use the conventional differentiable splice approximation;
        # every candidate decision below uses DifferentiableAR.losses(), which
        # decodes and re-tokenizes the complete prompt exactly.
        full = self.ar.tokenizer(self.ar.critic.template.format(explanation=explanation), add_special_tokens=True)[
            "input_ids"
        ]
        spliced = self.ar.prefix_ids + x_ids.tolist() + self.ar.suffix_ids
        ar_gradient_splice_stable = full == spliced

        trajectory: list[StepRecord] = []
        original_ids = x_ids.clone()
        initial_ar, initial_av, initial_objective = self._exact_losses(x_ids.unsqueeze(0), activation, beta)
        current_ar = float(initial_ar[0].item())
        current_av = float(initial_av[0].item())
        current_objective = float(initial_objective[0].item())
        for step in range(steps):
            _spliced_ar_value, ar_gradient = self.ar.gradient(x_ids, activation, self.variance)
            if beta:
                _gradient_av_value, av_gradient, av_log_probs = self.av.gradient(x_ids, activation)
            else:
                av_gradient, av_log_probs = None, None
            top_ids = self._combined_topk(x_ids, ar_gradient, av_gradient, av_log_probs, beta)
            candidates = self._sample_candidates(x_ids, top_ids, stable_seed(seed, step, "candidates"))
            ar_losses, av_nlls, objectives = self._exact_losses(candidates, activation, beta)
            best_index = int(objectives.argmin().item())
            best_objective = float(objectives[best_index].item())
            accepted = best_objective < current_objective
            if accepted:
                changed = (candidates[best_index] != x_ids).nonzero().flatten()
                if len(changed) != 1:
                    raise RuntimeError(f"candidate changed {len(changed)} positions instead of one")
                position = int(changed[0].item())
                old_token = int(x_ids[position].item())
                new_token = int(candidates[best_index, position].item())
                x_ids = candidates[best_index].clone()
                current_ar = float(ar_losses[best_index].item())
                current_av = float(av_nlls[best_index].item())
                current_objective = best_objective
            else:
                position = old_token = new_token = None
            trajectory.append(
                StepRecord(
                    step=step,
                    accepted=accepted,
                    position=position,
                    old_token_id=old_token,
                    new_token_id=new_token,
                    old_token_text=(
                        None if old_token is None else self.av.tokenizer.decode([old_token], skip_special_tokens=False)
                    ),
                    new_token_text=(
                        None if new_token is None else self.av.tokenizer.decode([new_token], skip_special_tokens=False)
                    ),
                    ar_loss=current_ar,
                    av_nll=current_av,
                    objective=current_objective,
                )
            )
            print(
                f"  beta={beta:g} step={step + 1}/{steps} obj={current_objective:.6f} "
                f"ar={current_ar:.6f} nll={current_av:.4f} accepted={accepted}",
                flush=True,
            )

        final_ar_values, final_av_values, _final_objectives = self._exact_losses(
            torch.stack((original_ids, x_ids)), activation, beta
        )
        final_text = self.av.tokenizer.decode(x_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        roundtrip = self.av.tokenizer(final_text, add_special_tokens=False)["input_ids"] == x_ids.tolist()
        return {
            "beta": beta,
            "steps_requested": steps,
            "steps_accepted": sum(record.accepted for record in trajectory),
            "token_count": len(x_ids),
            "ar_gradient_splice_stable": ar_gradient_splice_stable,
            "original_explanation": explanation,
            "canonical_original_explanation": self.av.tokenizer.decode(
                original_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
            "optimized_explanation": final_text,
            "roundtrip_token_stable": roundtrip,
            "original_ar_loss": float(final_ar_values[0].item()),
            "optimized_ar_loss": float(final_ar_values[1].item()),
            "original_av_nll": float(final_av_values[0].item()),
            "optimized_av_nll": float(final_av_values[1].item()),
            "original_objective": float((final_ar_values[0] + beta * final_av_values[0]).item()),
            "optimized_objective": float((final_ar_values[1] + beta * final_av_values[1]).item()),
            "trajectory": [asdict(record) for record in trajectory],
        }


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() not in (1, 2):
        raise RuntimeError("GCG worker expects one shared GPU or two visible GPUs (AR first, AV second)")
    spec = model_spec(args.model)
    rollouts_dir = args.alpaca_dir / spec.key / "rollouts"
    rows = read_jsonl(rollouts_dir / "rollouts.jsonl")
    decoded = read_jsonl(args.alpaca_dir / spec.key / "decode" / "explanations.jsonl")
    activations = load_file(str(rollouts_dir / "activations.safetensors"))["activations"].float()
    if not (len(rows) == len(decoded) == len(activations) == args.n):
        raise RuntimeError("merged rollouts and decodes must both be complete")
    decode_by_id = {row["sample_id"]: row for row in decoded}
    ar, ar_path = load_ar(spec, "cuda:0")
    av_device = "cuda:1" if torch.cuda.device_count() == 2 else "cuda:0"
    av, av_path = load_av(spec, av_device)
    variance = empirical_directional_variance(activations, ar.scale)
    optimizer = GCGOptimizer(
        ar,
        av,
        variance,
        topk=args.topk,
        candidate_batch_size=args.candidate_batch_size,
        exact_microbatch_size=args.exact_microbatch_size,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    output_dir = args.output_dir
    output_path = output_dir / f"shard-{args.shard:02d}-of-{args.num_shards:02d}.jsonl"
    work = [row for row in rows if row["selection_rank"] % args.num_shards == args.shard]
    betas = tuple(float(value) for value in args.betas)
    if any(not math.isfinite(beta) or beta < 0 for beta in betas):
        raise ValueError(f"betas must be finite and nonnegative: {betas}")

    with ResumableJSONL(output_path, ("sample_id", "beta")) as sink:
        for row_index, row in enumerate(work, start=1):
            rank = int(row["selection_rank"])
            explanation = decode_by_id[row["sample_id"]]["explanation"]
            for beta in betas:
                if sink.contains(sample_id=row["sample_id"], beta=beta):
                    continue
                print(
                    f"[{spec.key} GCG shard {args.shard}] sample {row_index}/{len(work)} "
                    f"id={row['sample_id']} beta={beta:g}",
                    flush=True,
                )
                result = optimizer.optimise(
                    explanation,
                    activations[rank],
                    beta,
                    args.steps,
                    stable_seed(PROTOCOL_VERSION, spec.key, row["sample_id"], beta),
                    args.max_explanation_tokens,
                )
                sink.append(
                    {
                        "protocol": PROTOCOL_VERSION,
                        "model": spec.key,
                        "sample_id": row["sample_id"],
                        "selection_rank": rank,
                        "activation_variance": variance,
                        **result,
                    }
                )
    write_json(
        output_dir / f"shard-{args.shard:02d}-of-{args.num_shards:02d}.manifest.json",
        {
            "protocol": PROTOCOL_VERSION,
            "model": spec.__dict__,
            "ar_checkpoint": str(ar_path),
            "av_checkpoint": str(av_path),
            "activation_variance": variance,
            "shard": args.shard,
            "num_shards": args.num_shards,
            "betas": betas,
            "steps": args.steps,
            "topk": args.topk,
            "candidate_batch_size": args.candidate_batch_size,
            "exact_microbatch_size": args.exact_microbatch_size,
            "vocab_chunk_size": args.vocab_chunk_size,
            "devices_per_worker": torch.cuda.device_count(),
            "regularizer": "mean_token_KL(delta_x || p_AV(. | activation, x_prefix)) = mean_token_AV_NLL",
        },
    )


def merge(args: argparse.Namespace) -> None:
    spec = model_spec(args.model)
    directory = args.output_dir
    rows = []
    for shard in range(args.num_shards):
        rows.extend(read_jsonl(directory / f"shard-{shard:02d}-of-{args.num_shards:02d}.jsonl"))
    rows.sort(key=lambda row: (row["selection_rank"], -row["beta"]))
    expected = args.n * len(args.betas)
    keys = {(row["sample_id"], row["beta"]) for row in rows}
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError(f"expected {expected} unique GCG rows, got {len(rows)}")
    write_jsonl(directory / "results.jsonl", rows)
    summary: dict[str, Any] = {"protocol": PROTOCOL_VERSION, "model": spec.key, "rows": len(rows), "betas": {}}
    for beta in (float(value) for value in args.betas):
        subset = [row for row in rows if row["beta"] == beta]
        summary["betas"][str(beta)] = {
            "n": len(subset),
            "mean_original_ar_loss": sum(row["original_ar_loss"] for row in subset) / len(subset),
            "mean_optimized_ar_loss": sum(row["optimized_ar_loss"] for row in subset) / len(subset),
            "mean_original_av_nll": sum(row["original_av_nll"] for row in subset) / len(subset),
            "mean_optimized_av_nll": sum(row["optimized_av_nll"] for row in subset) / len(subset),
            "mean_original_objective": sum(row["original_objective"] for row in subset) / len(subset),
            "mean_optimized_objective": sum(row["optimized_objective"] for row in subset) / len(subset),
            "mean_accepted_steps": sum(row["steps_accepted"] for row in subset) / len(subset),
            "roundtrip_token_stable_rate": sum(row["roundtrip_token_stable"] for row in subset) / len(subset),
        }
    write_json(directory / "summary.json", summary)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="gemma12b", choices=sorted(MODEL_SPECS))
    parser.add_argument("--output-dir", type=Path, default=results_dir("gcg"))
    parser.add_argument("--n", type=int, default=N_EXAMPLES)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--betas", type=float, nargs="+", default=list(BETAS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker", help="optimize one shard (one GPU, or two: AR first, AV second)")
    add_common(worker)
    worker.add_argument("--alpaca-dir", type=Path, default=results_dir("alpaca"),
                        help="soar_nla.alpaca results (rollouts, activations, AV explanations)")
    worker.add_argument("--shard", type=int, required=True)
    worker.add_argument("--steps", type=int, default=20)
    worker.add_argument("--topk", type=int, default=64)
    worker.add_argument("--candidate-batch-size", type=int, default=64)
    worker.add_argument("--exact-microbatch-size", type=int, default=4)
    worker.add_argument("--vocab-chunk-size", type=int, default=8192)
    worker.add_argument("--max-explanation-tokens", type=int, default=256)
    merger = sub.add_parser("merge", help="merge shards into results.jsonl and summary.json")
    add_common(merger)
    args = parser.parse_args()
    run_worker(args) if args.command == "worker" else merge(args)


if __name__ == "__main__":
    main()
