#!/usr/bin/env python3
"""Extract L20 residual activations from SelfDescribe prompts.

Matches the NLA dataset contract (training/scripts/regenerate_activations.py):
the activation is the OUTPUT of decoder block `--layer` at the FINAL token of
the text, raw / unnormalized.

Two probe locations (`--mode`):
  for       truncate each prompt so its final token is the last ' for' -- the
            word in "...infobox for me." (inside the constant task suffix)
  append_i  drop the task suffix, append ' I' to the self-description and
            extract there (probes who the model thinks is speaking). Used for
            all reported results.

Paper run: --attr-class all --n-per-attr 0 --mode append_i (2,585 rows).
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.utils.arch_adapters import resolve_decoder_layers

from soar_nla.hub import resolve_snapshot
from soar_nla.warmstart_ablation.runs import QWEN25_7B, SELFDESCRIBE, SELFDESCRIBE_FILE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=None,
                    help="local prompts.jsonl; default: download the pinned "
                         f"{SELFDESCRIBE[0]} revision")
    ap.add_argument("--base", default=QWEN25_7B[0])
    ap.add_argument("--base-revision", default=QWEN25_7B[1],
                    help="HF revision of --base when it is a hub id ('' = use --base as given)")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--attr-class", default="Gender", help="attr_class to keep, or 'all'")
    ap.add_argument("--n-per-attr", type=int, default=40, help="cap per attr value; 0 = no cap")
    ap.add_argument("--mode", choices=["for","append_i"], default="for",
                    help="'for': truncate at final ' for' of the task suffix. "
                         "'append_i': drop the task suffix, append ' I' and "
                         "extract there (probes who the model thinks is speaking).")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prompts = a.prompts or hf_hub_download(
        SELFDESCRIBE[0], SELFDESCRIBE_FILE, repo_type="dataset", revision=SELFDESCRIBE[1])
    base = str(resolve_snapshot(a.base, a.base_revision)) if a.base_revision else a.base

    rows = [json.loads(l) for l in open(prompts)]
    if a.attr_class != "all":
        rows = [r for r in rows if r["attr_class"] == a.attr_class]
    byattr = {}
    for r in rows:
        byattr.setdefault((r["attr_class"], r["attr"]), []).append(r)
    sel = []
    for attr in sorted(byattr):
        sel += byattr[attr] if a.n_per_attr == 0 else byattr[attr][: a.n_per_attr]
    import collections as _c
    _byc = _c.Counter(r["attr_class"] for r in sel)
    print(f"[data] {len(sel)} prompts | {len(byattr)} (class,value) cells | by class: {dict(_byc)}", flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    cap = [None]
    def hook(_m, _i, out):
        cap[0] = (out[0] if isinstance(out, tuple) else out).detach()
    h = resolve_decoder_layers(model)[a.layer].register_forward_hook(hook)

    out_rows, skipped = [], 0
    for r in sel:
        text = r["user_prompt"]
        if a.mode == "for":
            # truncate so the LAST token is the final ' for'
            i = text.rfind(" for ")
            if i < 0:
                skipped += 1; continue
            trunc = text[: i + len(" for")]      # "...infobox for"
            want = "for"
        else:
            # drop the fixed task suffix, keep the self-description, append " I"
            desc = text.rsplit("Write a hypothetical", 1)[0].strip()
            if not desc:
                skipped += 1; continue
            trunc = desc + " I"
            want = "I"
        ids = tok.encode(trunc, add_special_tokens=False)
        if tok.decode(ids[-1:]).strip() != want:
            skipped += 1; continue
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device="cuda"))
        vec = cap[0][0, len(ids) - 1].float().cpu().numpy()
        out_rows.append({
            "attr_class": r["attr_class"], "attr": r["attr"],
            "pred_prob": r["pred_prob"], "tag": r["tag"],
            "user_prompt": text, "truncated_text": trunc,
            "n_raw_tokens": len(ids),
            "activation": vec.astype(np.float32).tolist(),
        })
    h.remove()
    print(f"[done] {len(out_rows)} activations, {skipped} skipped "
          f"(d={len(out_rows[0]['activation'])})", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
