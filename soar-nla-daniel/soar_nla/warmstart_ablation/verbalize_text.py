#!/usr/bin/env python3
"""Verbalize one arbitrary text with a Qwen2.5-7B-Instruct L20 NLA checkpoint.

Extracts the raw output of decoder block `layer_index` (from the sidecar) at the
final token of the text, injects it at the marker of the canonical actor
prompt, and decodes with HF generate: one greedy completion plus
`--n-samples` samples at T=1.0 (torch seeds 0..n-1).

Used for the warm-start (run A) example verbalization in the paper appendix:
--lora <A>/rl_vllm/iter_000800 on the source text in
results/warmstart_ablation/example_source_text.txt. For run A the RL LoRA
continues the SFT adapter, so base + RL LoRA is the complete policy.
"""
import argparse

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.schema import EXPLANATION_RE
from nla.utils import build_prompt_text, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers

from soar_nla.warmstart_ablation.runs import QWEN25_7B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--lora", required=True, help="RL LoRA directory (iter_NNNNNN)")
    ap.add_argument("--parquet", required=True,
                    help="RL parquet (sidecar + canonical actor prompt from row 0)")
    ap.add_argument("--base", default=QWEN25_7B[0])
    ap.add_argument("--base-revision", default=QWEN25_7B[1],
                    help=f"HF revision used when --base is {QWEN25_7B[0]} ('' = by name)")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()

    base = a.base
    if a.base_revision and base == QWEN25_7B[0]:
        from soar_nla.hub import resolve_snapshot
        base = str(resolve_snapshot(base, a.base_revision))

    text = open(a.text_file).read().rstrip("\n")
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(base)
    cfg = load_nla_config(a.parquet, tok)
    print(f"[cfg] d_model={cfg.d_model} layer={cfg.extraction_layer_index} K={cfg.actor_context_size}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    # ---- 1. extract: raw output of decoder block `layer_index` at the FINAL token ----
    cap = [None]
    h = resolve_decoder_layers(model)[cfg.extraction_layer_index].register_forward_hook(
        lambda m, i, out: cap.__setitem__(0, (out[0] if isinstance(out, tuple) else out).detach()))
    ids = tok.encode(text, add_special_tokens=False)
    with torch.no_grad():
        model(input_ids=torch.tensor([ids], device=device))
    vec = cap[0][0, len(ids) - 1].float()
    h.remove()
    print(f"[act] {len(ids)} tokens, last={tok.decode(ids[-1:])!r}, ||v||={vec.norm():.1f}", flush=True)

    # ---- 2. verbalize: base + RL LoRA, activation injected at the marker token ----
    actor = PeftModel.from_pretrained(model, a.lora, adapter_name="default")
    actor.set_adapter("default"); actor.eval()
    lnorm = sum(p.detach().float().pow(2).sum().item() for n, p in actor.named_parameters() if "lora_" in n)
    assert lnorm > 0, "LoRA all-zero"
    print(f"[actor] {a.lora}  sum(lora^2)={lnorm:.3e}", flush=True)

    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id,
                           layer_idx=1)

    canon = pq.ParquetFile(a.parquet).read_row_group(0, columns=["prompt"]).column("prompt").to_pylist()[0]
    ptxt = build_prompt_text(canon, cfg.injection_char, tok)
    pt = torch.tensor([tok.encode(ptxt, add_special_tokens=False)], device=device)
    vref[0] = vec.unsqueeze(0).unsqueeze(0).to(device)   # (batch=1, K=1, d)

    def gen(**kw):
        with torch.no_grad():
            out = actor.generate(input_ids=pt, attention_mask=torch.ones_like(pt),
                                 max_new_tokens=a.max_new_tokens, pad_token_id=tok.eos_token_id,
                                 return_dict_in_generate=True, **kw)
        return tok.decode(out.sequences[0, pt.shape[1]:], skip_special_tokens=True)

    results = [("greedy", gen(do_sample=False))]
    for i in range(a.n_samples):
        torch.manual_seed(i)
        results.append((f"sample T=1.0 #{i}", gen(do_sample=True, temperature=1.0, top_p=1.0)))
    vref[0] = None

    for tag, resp in results:
        m = EXPLANATION_RE.search(resp)
        ntok = len(tok.encode(resp, add_special_tokens=False))
        print(f"\n{'='*80}\n[{tag}]  raw_tokens={ntok}/{a.max_new_tokens}  closed_tag={m is not None}\n{'='*80}")
        print("--- RAW RESPONSE ---")
        print(resp)


if __name__ == "__main__":
    main()
