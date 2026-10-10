"""AV and AR stages of the benchmark round trips (resumable, one shard per GPU).

* ``decode``: greedy AV verbalization through an SGLang server hosting the
  AV (the GSM8K run; one server per shard, ``--workers`` concurrent requests).
* ``decode-vllm``: greedy AV verbalization with an in-process vLLM engine fed
  ``prompt_embeds`` (the IFEval and ARC-Easy runs).
* ``reconstruct``: AR reconstruction of each explanation, rescaled to the L2
  norm of the native activation at the same flat index (the NLA reconstruction
  is direction-only). Also used for the paraphrase arms, whose
  ``explanations/`` hold paraphrased text.

All stages read ``<activations-dir>/shard_NN.safetensors`` and write
``<out-dir>/explanations/shard_NN.jsonl`` or
``<out-dir>/reconstructions/shard_NN.safetensors``.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from soar_nla.benchmarks import common


def read_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(json.loads(line)["flat_index"]) for line in path.read_text().splitlines() if line.strip()}


def load_activations(args):
    from safetensors.torch import load_file

    return load_file(str(common.activations_dir(args) / f"shard_{args.shard:02d}.safetensors"))["activations"]


def explanation_record(index: int, raw: str) -> dict:
    from soar_nla.nla_inference import EXPLANATION_RE

    match = EXPLANATION_RE.search(raw)
    return {
        "flat_index": index,
        "parse_ok": match is not None,
        "explanation": match.group(1).strip() if match else raw.strip(),
    }


def decode(args) -> None:
    """Threaded greedy AV decoding against one SGLang server."""
    from soar_nla.nla_inference import NLAClient

    av = args.av or common.av_path()
    acts = load_activations(args)
    out_dir = Path(args.out_dir) / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"shard_{args.shard:02d}.jsonl"
    done = read_done(out)
    local = threading.local()

    def one(index: int) -> dict:
        if not hasattr(local, "client"):
            local.client = NLAClient(av, sglang_url=args.sglang_url, device="cpu")
        raw = local.client.generate(
            acts[index].float(), temperature=0.0,
            max_new_tokens=args.max_new_tokens, extract_explanation=False,
        )
        return explanation_record(index, raw)

    pending = [i for i in range(acts.shape[0]) if i not in done]
    print(f"[decode {args.shard}] {len(pending)} pending / {acts.shape[0]}", flush=True)
    with out.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, i) for i in pending]
        for count, future in enumerate(as_completed(futures), 1):
            handle.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            if count % 25 == 0:
                handle.flush()
            if count % 100 == 0 or count == len(pending):
                print(f"[decode {args.shard}] {count}/{len(pending)}", flush=True)


def decode_vllm(args) -> None:
    """Batched greedy AV decoding using vLLM prompt embeddings."""
    import torch
    from vllm import LLM, SamplingParams

    from soar_nla.nla_inference import NLAClient

    av = args.av or common.av_path()
    acts = load_activations(args)
    out_dir = Path(args.out_dir) / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"shard_{args.shard:02d}.jsonl"
    done = read_done(out)
    # Reuse the canonical, sidecar-validated embedding construction. No HTTP
    # request is made; the resulting prompt embeddings go straight to vLLM.
    builder = NLAClient(av, sglang_url="http://127.0.0.1:1", device="cpu")
    llm = LLM(
        model=av, tokenizer=av, dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=512, gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True, enable_prompt_embeds=True,
        limit_mm_per_prompt={"image": 0, "video": 0}, seed=0,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    pending = [i for i in range(acts.shape[0]) if i not in done]
    print(f"[decode {args.shard}] {len(pending)} pending / {acts.shape[0]}", flush=True)
    with out.open("a") as handle:
        for start in range(0, len(pending), args.request_batch_size):
            batch_indices = pending[start:start + args.request_batch_size]
            prompts = []
            for index in batch_indices:
                embeds, _ = builder._build_embeds(acts[index].float(), None)
                prompts.append({"prompt_embeds": torch.from_numpy(embeds)})
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            for index, result in zip(batch_indices, outputs, strict=True):
                handle.write(json.dumps(explanation_record(index, result.outputs[0].text), ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[decode {args.shard}] {min(start + len(batch_indices), len(pending))}/{len(pending)}", flush=True)


def reconstruct(args) -> None:
    import torch
    from safetensors.torch import save_file

    from soar_nla.nla_inference import NLACritic

    acts = load_activations(args).float()
    base = Path(args.out_dir)
    rows = [json.loads(line) for line in (base / "explanations" / f"shard_{args.shard:02d}.jsonl").read_text().splitlines() if line]
    by_index = {int(row["flat_index"]): row for row in rows}
    if set(by_index) != set(range(acts.shape[0])):
        raise RuntimeError(f"incomplete explanation shard {args.shard}")
    critic = NLACritic(args.ar or common.ar_path(), device="cuda:0", dtype=torch.bfloat16)
    tokenizer = critic.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    recon = torch.empty_like(acts)
    for start in range(0, acts.shape[0], args.batch_size):
        stop = min(start + args.batch_size, acts.shape[0])
        prompts = [critic.template.format(explanation=by_index[i]["explanation"]) for i in range(start, stop)]
        batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda:0")
        with torch.inference_mode():
            hidden = critic.backbone.model(**batch, use_cache=False).last_hidden_state
            last = batch.attention_mask.sum(dim=1) - 1
            pred = critic.value_head(hidden[torch.arange(stop - start, device="cuda:0"), last]).float().cpu()
        # NLA reconstruction is direction-only; preserve each original norm.
        pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        recon[start:stop] = pred * acts[start:stop].norm(dim=-1, keepdim=True)
        if stop % 512 < args.batch_size or stop == acts.shape[0]:
            print(f"[reconstruct {args.shard}] {stop}/{acts.shape[0]}", flush=True)
    out_dir = base / "reconstructions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"shard_{args.shard:02d}.safetensors"
    temp = out.with_suffix(".tmp.safetensors")
    # Some early Gemma residuals have norms > fp16's finite range; bfloat16
    # preserves the model's native exponent range for norm-matched vectors.
    save_file({"reconstructions": recon.bfloat16().contiguous()}, str(temp))
    temp.replace(out)
    print(f"WROTE {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("decode", "decode-vllm", "reconstruct"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--activations-dir", type=Path, default=None,
                        help="native activations (default: <out-dir>/activations)")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--av", default=None, help="AV checkpoint directory (default: pinned kitft L32 AV)")
    parser.add_argument("--ar", default=None, help="AR checkpoint directory (default: pinned kitft L32 AR)")
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--request-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    args = parser.parse_args()
    {"decode": decode, "decode-vllm": decode_vllm, "reconstruct": reconstruct}[args.stage](args)


if __name__ == "__main__":
    main()
