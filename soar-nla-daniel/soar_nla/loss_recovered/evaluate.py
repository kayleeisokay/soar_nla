"""Next-token CE with the layer-32 activations of a token span replaced, one window at a time.

For every window and every span in `config.SETTINGS` the target model runs on
the window's first `stop` tokens: once clean, and once per replacement with the
layer-32 output at positions [start, stop) overwritten jointly (positions before
`start` stay clean). Replacements: `zero` (zero ablation), `nla`, `nla_raw`,
`sae`, `sae_norm` (see `config.METHODS`). Windows run at batch size one, so the
per-window numbers do not depend on sharding.

`merge` writes `window_losses.jsonl.gz` (one row per window) and `evaluation.json`.

    CUDA_VISIBLE_DEVICES=0 python -m soar_nla.loss_recovered.evaluate worker --shard 0 --num-shards 4
    python -m soar_nla.loss_recovered.evaluate merge --num-shards 4
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from soar_nla.hub import GEMMA_SCOPE_2_12B, load_target, model_spec, resolve_decoder_layers
from soar_nla.io_utils import ResumableJSONL, read_jsonl, write_json

from .common import default_results_dir, load_windows, replace_span, shard_suffix, span_ce
from .config import METHODS, MODEL, SAE_ID, SAE_PARAMS_DIR, SETTINGS


def load_sae(device: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Gemma Scope 2 JumpReLU SAE on the layer-32 residual stream, in bfloat16."""
    repo, revision = GEMMA_SCOPE_2_12B
    config = json.loads(Path(hf_hub_download(repo, f"{SAE_PARAMS_DIR}/config.json", revision=revision)).read_text())
    if config["hf_hook_point_in"] != "model.layers.32.output" or config["architecture"] != "jump_relu":
        raise ValueError(f"unexpected SAE config: {config}")
    if config.get("affine_connection"):
        raise ValueError("SAEs with an affine skip connection are not supported")
    params = load_file(hf_hub_download(repo, f"{SAE_PARAMS_DIR}/params.safetensors", revision=revision))
    state = {name: params[name].to(device=device, dtype=torch.bfloat16) for name in ("w_enc", "w_dec", "b_enc", "b_dec", "threshold")}
    return state, config


def sae_forward(x: torch.Tensor, state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    pre = x @ state["w_enc"] + state["b_enc"]
    features = torch.relu(pre) * (pre > state["threshold"])
    return features @ state["w_dec"] + state["b_dec"], (features != 0).sum(dim=-1)


def run_worker(args: argparse.Namespace) -> None:
    spec = model_spec(MODEL)
    windows = load_windows(args.output_dir)
    n_windows, context_size = windows.shape
    if max(stop for _, stop in SETTINGS.values()) > context_size:
        raise ValueError(f"windows hold {context_size} tokens, too short for {SETTINGS}")
    if args.limit is not None:
        n_windows = min(n_windows, args.limit)
    activations = np.load(args.output_dir / "activations.npy", mmap_mode="r")
    use_nla = bool({"nla", "nla_raw"} & set(args.methods))
    use_sae = bool({"sae", "sae_norm"} & set(args.methods))
    reconstructions = np.load(args.output_dir / "reconstructions.npy", mmap_mode="r") if use_nla else None

    model, _, _ = load_target(spec, "cuda:0", checkpoint=args.target)
    layer = resolve_decoder_layers(model)[spec.layer]
    sae_state = load_sae("cuda:0")[0] if use_sae else None

    output = args.output_dir / f"window_losses.{shard_suffix(args.shard, args.num_shards)}.jsonl"
    with ResumableJSONL(output, ("window_index",)) as sink, torch.inference_mode():
        todo = [w for w in range(args.shard, n_windows, args.num_shards) if not sink.contains(window_index=w)]
        for done, window in enumerate(todo, start=1):
            rows = slice(window * context_size, (window + 1) * context_size)
            original = torch.from_numpy(np.array(activations[rows], dtype=np.float32))
            replacements = {"zero": torch.zeros_like(original)}
            record: dict[str, object] = {"window_index": window}
            if use_nla:
                raw = torch.from_numpy(np.array(reconstructions[rows], dtype=np.float32))
                replacements["nla_raw"] = raw
                replacements["nla"] = raw * (original.norm(dim=-1, keepdim=True) / raw.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            if use_sae:
                x = original.to("cuda:0", torch.bfloat16)
                decoded, l0 = sae_forward(x, sae_state)
                record["sae"] = {
                    "l0_mean": float(l0.double().mean().item()),
                    "cosine_mean": float(F.cosine_similarity(x.float(), decoded.float(), dim=-1).double().mean().item()),
                }
                replacements["sae"] = decoded
                scale = x.float().norm(dim=-1, keepdim=True) / decoded.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
                replacements["sae_norm"] = decoded * scale.to(decoded.dtype)

            settings = {}
            for name, (start, stop) in SETTINGS.items():
                ids = torch.as_tensor(np.array(windows[window, :stop]), dtype=torch.long, device="cuda:0").unsqueeze(0)
                ce = {"clean": span_ce(model(input_ids=ids, use_cache=False).logits, ids, start, stop)}
                for method in ("zero", *args.methods):
                    with replace_span(layer, start, stop, replacements[method][start:stop]):
                        ce[method] = span_ce(model(input_ids=ids, use_cache=False).logits, ids, start, stop)
                settings[name] = {"start": start, "stop": stop, "scored_tokens": stop - start - 1, "ce": ce}
            record["settings"] = settings
            sink.append(record)
            if done % 25 == 0 or done == len(todo):
                print(f"[evaluate {shard_suffix(args.shard, args.num_shards)}] {done}/{len(todo)} windows", flush=True)


def run_merge(args: argparse.Namespace) -> None:
    spec = model_spec(MODEL)
    n_windows = int(load_windows(args.output_dir).shape[0])
    rows = []
    for shard in range(args.num_shards):
        rows.extend(read_jsonl(args.output_dir / f"window_losses.{shard_suffix(shard, args.num_shards)}.jsonl"))
    rows.sort(key=lambda row: row["window_index"])
    if [row["window_index"] for row in rows] != list(range(n_windows)):
        raise RuntimeError("window losses are incomplete or duplicated")
    methods = sorted(set(rows[0]["settings"]["prefix_50"]["ce"]) - {"clean", "zero"})
    with gzip.open(args.output_dir / "window_losses.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    meta = {
        "model": f"{spec.target_id}@{spec.target_revision}",
        "av": f"{spec.av_id}@{spec.av_revision}",
        "ar": f"{spec.ar_id}@{spec.ar_revision}",
        "layer": spec.layer,
        "windows": n_windows,
        "settings": SETTINGS,
        "methods": methods,
    }
    if "sae" in rows[0]:
        meta["sae"] = {
            "repo": f"{GEMMA_SCOPE_2_12B[0]}@{GEMMA_SCOPE_2_12B[1]}",
            "sae_id": SAE_ID,
            # every window has the same number of positions, so window means average exactly
            "mean_l0": float(np.mean([row["sae"]["l0_mean"] for row in rows])),
            "mean_cosine": float(np.mean([row["sae"]["cosine_mean"] for row in rows])),
        }
    write_json(args.output_dir / "evaluation.json", meta)
    print(f"merged {n_windows} windows -> {args.output_dir / 'window_losses.jsonl.gz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("worker", "merge"))
    parser.add_argument("--output-dir", type=Path, default=default_results_dir())
    parser.add_argument("--target", type=Path, default=None, help="local target checkpoint (default: pinned snapshot)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--limit", type=int, default=None, help="only the first N windows (smoke test)")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"invalid shard {args.shard}/{args.num_shards}")
    if args.stage == "worker":
        run_worker(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
