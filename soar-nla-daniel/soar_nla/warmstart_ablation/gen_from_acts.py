#!/usr/bin/env python3
"""Generate one NLA explanation per SelfDescribe activation for one RL checkpoint.

Reuses train_rl_vllm.py's own rollout/injection path so the generations match
what the training-time eval produces: canonical actor prompt (row 0 of the RL
parquet), injection via the patched vllm-lens SteeringVector, temperature 1.0,
max_new_tokens 256, one sample per activation. Writes one JSONL row per
activation, schema-compatible with the trainer's `eval_raw_completions.jsonl`
plus the SelfDescribe labels.

Actor construction mirrors train_rl_vllm.py's two modes:
  run A (warm start)  --av-adapter given: base + RL LoRA (the RL LoRA continues
                      the SFT adapter); vLLM serves the merged SFT AV.
  runs B, C (cold)    base + RL LoRA; vLLM serves the base model.
Run B must be decoded with --raw-fallback (it was trained and evaluated with
--raw-completion-fallback because it stops closing its <explanation> tag).
"""
import argparse, json, time
from pathlib import Path

from soar_nla.warmstart_ablation.runs import QWEN25_7B


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)                 # A | B | C
    ap.add_argument("--av-ckpt", required=True)             # model vLLM serves
    ap.add_argument("--base-ckpt", default=QWEN25_7B[0])
    ap.add_argument("--base-revision", default=QWEN25_7B[1],
                    help=f"HF revision used when --base-ckpt/--av-ckpt is {QWEN25_7B[0]} "
                         "('' = resolve by name)")
    ap.add_argument("--av-adapter", default=None)           # SFT adapter (run A only)
    ap.add_argument("--lora", required=True)                # RL LoRA @ iter_000600
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--raw-fallback", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.35)
    ap.add_argument("--vllm-max-len", type=int, default=1024)
    ap.add_argument("--step", type=int, default=600, help="checkpoint step (metadata only)")
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM as VLLM

from nla.config import load_nla_config
from nla.utils import build_prompt_text
from nla.train_rl_vllm import (
    rollout_batch_vllm, sync_actor_to_vllm, resolve_actor_message,
)


def _resolve(name, revision):
    """Pin the base model to its recorded revision; local paths pass through."""
    if revision and name == QWEN25_7B[0]:
        from soar_nla.hub import resolve_snapshot
        return str(resolve_snapshot(name, revision))
    return name


def main():
    args = _parse()
    args.base_ckpt = _resolve(args.base_ckpt, args.base_revision)
    args.av_ckpt = _resolve(args.av_ckpt, args.base_revision)
    t0 = time.time()
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.av_ckpt)
    cfg = load_nla_config(args.meta, tokenizer)
    inject_char = cfg.injection_char
    inj_id = cfg.injection_token_id
    left_id = getattr(cfg, "injection_left_neighbor_id", None)
    right_id = getattr(cfg, "injection_right_neighbor_id", None)

    # ---- rows: activations extracted from SelfDescribe prompts ----
    import pyarrow.parquet as _pq
    _pf = _pq.ParquetFile(args.parquet)
    CANON = _pf.read_row_group(0, columns=["prompt"]).column("prompt").to_pylist()[0]
    rows = []
    for line in open(args.acts):
        d = json.loads(line)
        vec = torch.tensor(d["activation"], dtype=torch.float32)
        rows.append({"prompt": CANON, "activation": vec.numpy(),
                     "actor_context": vec.unsqueeze(0).numpy(),   # K=1
                     "source": d["user_prompt"], "doc_id": d["tag"],
                     "attr": d["attr"], "attr_class": d["attr_class"],
                     "pred_prob": d["pred_prob"]})
    print(f"[gen] {len(rows)} activation rows", flush=True)

    # ---- actor: mirrors train_rl_vllm.py's two construction modes ----
    if args.av_adapter is not None:
        from peft import PeftModel
        print(f"[actor] CONTINUED: base {args.base_ckpt} + RL LoRA {args.lora}", flush=True)
        actor = AutoModelForCausalLM.from_pretrained(
            args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        actor = PeftModel.from_pretrained(actor, args.lora, adapter_name="default")
        actor.set_adapter("default")
    else:
        from peft import PeftModel
        print(f"[actor] loading {args.av_ckpt} + RL LoRA {args.lora}", flush=True)
        actor = AutoModelForCausalLM.from_pretrained(
            args.av_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
        actor = PeftModel.from_pretrained(actor, args.lora)
    actor.eval()
    lora_norm = sum(p.detach().float().pow(2).sum().item()
                    for n, p in actor.named_parameters() if "lora_" in n)
    print(f"[actor] sum(lora_param^2) = {lora_norm:.3e} (must be > 0)", flush=True)
    assert lora_norm > 0, "LoRA loaded but all-zero — wrong adapter path?"

    llm = VLLM(model=args.av_ckpt, tokenizer=args.av_ckpt, dtype="bfloat16",
               gpu_memory_utilization=args.vllm_gpu_mem, max_model_len=args.vllm_max_len,
               tensor_parallel_size=1, enforce_eager=True, enable_prefix_caching=False,
               seed=args.seed)
    print("[vllm] ready; syncing actor weights", flush=True)
    secs = sync_actor_to_vllm(actor, llm, ipc=False)
    print(f"[vllm] sync done in {secs:.1f}s", flush=True)

    prompts_with_acts = [
        (build_prompt_text(r["prompt"], inject_char, tokenizer),
         torch.tensor(r["actor_context"], dtype=torch.float32)) for r in rows
    ]
    resps = rollout_batch_vllm(llm, tokenizer, prompts_with_acts, inj_id, 1,
                               args.max_new_tokens, args.temperature,
                               left_id=left_id, right_id=right_id)
    by_idx = {r["prompt_idx"]: r for r in resps}

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n_ext = 0
    with out.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            r = by_idx.get(i, {})
            text = r.get("text", "")
            expl, tag_extracted, used_raw = resolve_actor_message(text, args.raw_fallback)
            n_ext += expl is not None
            fh.write(json.dumps({
                "run": args.run, "step": args.step, "idx": i,
            "attr": row["attr"], "attr_class": row["attr_class"],
            "pred_prob": row["pred_prob"],
                "extracted": expl is not None,
                "tag_extracted": tag_extracted,
                "used_raw_fallback": used_raw,
                "truncated": bool(r.get("truncated", False)),
                "raw_response": text,
                "source": row["source"],
                "doc_id": row["doc_id"],
                "explanation": expl if expl is not None else "<extraction failed>",
            }, ensure_ascii=False) + "\n")
    print(f"[done] {args.run}: {len(rows)} rows, extraction {n_ext}/{len(rows)} "
          f"-> {out}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
