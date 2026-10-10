"""AR reconstruction of every AV explanation (raw AR output; norm matching happens in evaluate).

Shard `s` of `k` handles activation indices s, s+k, s+2k, ... in right-padded
batches of 16; the paper run used 4 shards. `merge` interleaves the shards back
into `reconstructions.npy` ([windows * 128, d_model], float32).

    CUDA_VISIBLE_DEVICES=0 python -m soar_nla.loss_recovered.reconstruct worker --shard 0 --num-shards 4
    python -m soar_nla.loss_recovered.reconstruct merge --num-shards 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from soar_nla.hub import model_spec, resolve_snapshot
from soar_nla.io_utils import read_jsonl, write_json
from soar_nla.nla_inference import NLACritic

from .common import default_results_dir, shard_suffix
from .config import AR_BATCH_SIZE, MODEL


def run_worker(args: argparse.Namespace) -> None:
    spec = model_spec(MODEL)
    explanations = {int(row["index"]): row["explanation"] for row in read_jsonl(args.output_dir / "explanations.jsonl")}
    n = len(explanations)
    if set(explanations) != set(range(n)):
        raise RuntimeError("explanations.jsonl is incomplete; run verbalize first")
    indices = list(range(args.shard, n, args.num_shards))

    critic = NLACritic(resolve_snapshot(spec.ar_id, spec.ar_revision), device="cuda:0", dtype=torch.bfloat16)
    tokenizer = critic.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    suffix = shard_suffix(args.shard, args.num_shards)
    out_path = args.output_dir / f"reconstructions.{suffix}.npy"
    progress_path = args.output_dir / f"reconstructions.{suffix}.progress.json"
    if out_path.exists() and progress_path.exists():
        out = np.load(out_path, mmap_mode="r+")
        completed = int(json.loads(progress_path.read_text())["completed"])
    else:
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(len(indices), spec.d_model))
        completed = 0
    if out.shape != (len(indices), spec.d_model):
        raise RuntimeError(f"{out_path} has shape {out.shape}, expected {(len(indices), spec.d_model)}")

    with torch.inference_mode():
        for start in range(completed, len(indices), args.batch_size):
            batch = indices[start : start + args.batch_size]
            prompts = [critic.template.format(explanation=explanations[i]) for i in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda:0")
            hidden = critic.backbone.model(**encoded, use_cache=False).last_hidden_state
            last = encoded.attention_mask.sum(dim=1) - 1
            pred = critic.value_head(hidden[torch.arange(len(batch), device="cuda:0"), last])
            out[start : start + len(batch)] = pred.float().cpu().numpy()
            out.flush()
            completed = start + len(batch)
            write_json(progress_path, {"completed": completed, "count": len(indices)})
            if completed % (args.batch_size * 200) == 0 or completed == len(indices):
                print(f"[reconstruct {suffix}] {completed}/{len(indices)}", flush=True)


def run_merge(args: argparse.Namespace) -> None:
    spec = model_spec(MODEL)
    n = int(np.load(args.output_dir / "activations.npy", mmap_mode="r").shape[0])
    merged = np.lib.format.open_memmap(
        args.output_dir / "reconstructions.npy", mode="w+", dtype=np.float32, shape=(n, spec.d_model)
    )
    for shard in range(args.num_shards):
        suffix = shard_suffix(shard, args.num_shards)
        progress = json.loads((args.output_dir / f"reconstructions.{suffix}.progress.json").read_text())
        indices = list(range(shard, n, args.num_shards))
        if progress["completed"] != len(indices):
            raise RuntimeError(f"reconstruction shard {suffix} is incomplete")
        merged[indices] = np.load(args.output_dir / f"reconstructions.{suffix}.npy", mmap_mode="r")
    merged.flush()
    write_json(
        args.output_dir / "reconstruction.json",
        {"ar": f"{spec.ar_id}@{spec.ar_revision}", "count": n, "batch_size": args.batch_size, "num_shards": args.num_shards},
    )
    print(f"merged {args.num_shards} shards into {merged.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("worker", "merge"))
    parser.add_argument("--output-dir", type=Path, default=default_results_dir())
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=AR_BATCH_SIZE)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"invalid shard {args.shard}/{args.num_shards}")
    if args.stage == "worker":
        run_worker(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
