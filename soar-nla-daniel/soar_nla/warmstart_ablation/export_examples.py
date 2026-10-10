#!/usr/bin/env python3
"""Export complete held-out AV/AR protocol examples from an RL checkpoint.

For each of five fixed held-out rows of the RL parquet: greedy AV explanation
of the gold activation (HF generate + Karvonen injection hook, 256 new
tokens), AR reconstruction MSE/cosine, and a round trip (AV applied to the AR
reconstruction). Produces the cold-start example verbalizations of the paper
appendix (row ":train:71467"):

  run B (no SFT, KL 0)     --adapter <B>/iter_000600 --critic <B>/critic_latest
                           --raw-completion-fallback
  run C (no SFT, KL 0.01)  --adapter <C>/iter_000400 --critic <C critic @ step 400>

The AV message does not depend on --critic; only the reconstruction lines do.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.schema import extract_explanation, normalize_activation
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook
from nla.val_split import is_val_doc

from soar_nla.warmstart_ablation.runs import QWEN25_7B


# The five held-out rows used for the qualitative examples, keyed by the
# `<split>:<doc_index>` tail of their doc_id (every doc_id in the corpus shares
# the same corpus prefix). The suffix disambiguates the desired activation row
# when a document contributes multiple truncated positions. Example 3
# (":train:71467", "...The annual") is the one shown in the paper.
EXAMPLES = (
    (":train:4600", "Price list for"),
    (":train:90595", "citation rates of"),
    (":train:71467", "The annual"),
    (":train:49709", 'Uses numbers, directional "E" and "XO" patterns.'),
    (":train:60998", "most of today"),
)


def _example_key(doc_id: str) -> str | None:
    for key, _ in EXAMPLES:
        if doc_id.endswith(key):
            return key
    return None


def load_examples(parquet_path: str) -> list[dict]:
    wanted = dict(EXAMPLES)
    found: dict[str, dict] = {}
    pf = pq.ParquetFile(parquet_path)
    columns = [
        "doc_id",
        "detokenized_text_truncated",
        "prompt",
        "activation_vector",
    ]
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx, columns=columns)
        doc_ids = rg.column("doc_id").to_pylist()
        sources = rg.column("detokenized_text_truncated").to_pylist()
        prompts = rg.column("prompt").to_pylist()
        acts_col = rg.column("activation_vector").combine_chunks()
        acts = np.asarray(acts_col.flatten(), dtype=np.float32).reshape(len(doc_ids), -1)
        for i, doc_id in enumerate(doc_ids):
            key = _example_key(doc_id)
            suffix = wanted.get(key) if key is not None else None
            source = sources[i] or ""
            if suffix is None or not source.rstrip().endswith(suffix):
                continue
            assert is_val_doc(doc_id, 100), f"{doc_id} is not in the held-out bucket"
            assert key not in found, f"ambiguous row for {doc_id} / {suffix!r}"
            found[key] = {
                "doc_id": doc_id,
                "source": source,
                "prompt": prompts[i],
                "activation": acts[i].copy(),
            }
        if len(found) == len(EXAMPLES):
            break
    missing = [key for key, _ in EXAMPLES if key not in found]
    if missing:
        raise RuntimeError(f"could not find exact held-out rows: {missing}")
    return [found[key] for key, _ in EXAMPLES]


def generate_message(
    actor,
    tokenizer,
    vectors_ref: list[torch.Tensor | None],
    prompt_text: str,
    activation: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, str | None]:
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    vectors_ref[0] = activation.to(device=device, dtype=torch.float32).unsqueeze(0)
    try:
        with torch.inference_mode():
            output_ids = actor.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        vectors_ref[0] = None
    raw = tokenizer.decode(
        output_ids[0, input_ids.shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return raw, extract_explanation(raw)


def outside_tag_text(raw: str, explanation: str | None) -> tuple[str, str]:
    if explanation is None:
        return raw, ""
    open_tag = "<explanation>"
    close_tag = "</explanation>"
    start = raw.find(open_tag)
    end = raw.find(close_tag, start + len(open_tag))
    if start < 0 or end < 0:
        return raw, ""
    return raw[:start], raw[end + len(close_tag) :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", default=QWEN25_7B[0])
    parser.add_argument("--base-revision", default=QWEN25_7B[1],
                        help=f"HF revision used when --base-ckpt is {QWEN25_7B[0]} ('' = by name)")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--critic", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--raw-completion-fallback",
        action="store_true",
        help="Use a non-empty raw completion as the AR message when tags do not close.",
    )
    parser.add_argument(
        "--source-tail-chars",
        type=int,
        default=0,
        help="Keep only this many trailing source characters; 0 saves the full text.",
    )
    args = parser.parse_args()
    if args.base_revision and args.base_ckpt == QWEN25_7B[0]:
        from soar_nla.hub import resolve_snapshot
        args.base_ckpt = str(resolve_snapshot(args.base_ckpt, args.base_revision))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")

    rows = load_examples(args.parquet)
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.parquet, tokenizer)
    assert cfg.critic_prompt_template is not None

    actor_base = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    actor = PeftModel.from_pretrained(
        actor_base, args.adapter, is_trainable=False
    ).to(device)
    actor.eval()
    vectors_ref: list[torch.Tensor | None] = [None]
    register_karvonen_hook(
        actor,
        vectors_ref,
        cfg.injection_token_id,
        cfg.injection_left_neighbor_id,
        cfg.injection_right_neighbor_id,
        layer_idx=1,
    )

    critic = NLACriticModel.from_pretrained(
        args.critic,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    critic.eval()

    mse_scale = cfg.mse_scale
    chunks = [
        f"Selected {len(rows)} rows from the exact held-out 10.0% doc bucket.",
        f"Actor adapter: {args.adapter}",
        f"Critic: {args.critic}",
        "Generation: greedy, max_new_tokens="
        f"{args.max_new_tokens}; complete raw completions and extracted messages retained; "
        f"raw_completion_fallback={args.raw_completion_fallback}.",
    ]

    for idx, row in enumerate(rows, start=1):
        gold = torch.as_tensor(row["activation"], dtype=torch.float32)
        prompt_text = build_prompt_text(row["prompt"], cfg.injection_char, tokenizer)
        raw, tagged_message = generate_message(
            actor,
            tokenizer,
            vectors_ref,
            prompt_text,
            gold,
            device,
            args.max_new_tokens,
        )
        used_raw_fallback = (
            tagged_message is None
            and args.raw_completion_fallback
            and bool(raw.strip())
        )
        message = (
            tagged_message
            if tagged_message is not None
            else raw.strip() if used_raw_fallback else None
        )
        prefix, suffix = outside_tag_text(raw, tagged_message)
        round_used_raw_fallback = False

        if message is None:
            pred = None
            mse = float("nan")
            cosine = float("nan")
            round_raw = ""
            round_message = None
            similarity = float("nan")
        else:
            critic_text = cfg.critic_prompt_template.format(explanation=message)
            critic_ids = tokenizer.encode(critic_text, add_special_tokens=False)
            x = torch.tensor([critic_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                pred = critic_predict(critic, x, None, mse_scale)[0]
            pred_norm = normalize_activation(pred.unsqueeze(0), mse_scale)[0]
            gold_norm = normalize_activation(gold.to(device).unsqueeze(0), mse_scale)[0]
            mse = F.mse_loss(pred_norm, gold_norm).item()
            cosine = F.cosine_similarity(
                pred.float().unsqueeze(0), gold.to(device).float().unsqueeze(0)
            ).item()
            round_raw, round_tagged_message = generate_message(
                actor,
                tokenizer,
                vectors_ref,
                prompt_text,
                pred.detach().cpu(),
                device,
                args.max_new_tokens,
            )
            round_used_raw_fallback = (
                round_tagged_message is None
                and args.raw_completion_fallback
                and bool(round_raw.strip())
            )
            round_message = (
                round_tagged_message
                if round_tagged_message is not None
                else round_raw.strip() if round_used_raw_fallback else None
            )
            similarity = (
                SequenceMatcher(None, message, round_message).ratio()
                if round_message is not None
                else float("nan")
            )

        sep = "=" * 100
        source_text = (
            row["source"][-args.source_tail_chars :]
            if args.source_tail_chars > 0
            else row["source"]
        )
        source_label = (
            f"SOURCE TAIL (last {args.source_tail_chars} characters):"
            if args.source_tail_chars > 0
            else "FULL SOURCE TEXT:"
        )
        chunks.extend(
            [
                "",
                sep,
                f"EXAMPLE {idx} | held-out doc_id={row['doc_id']}",
                source_label,
                source_text,
                "",
                "RAW AV COMPLETION FROM GOLD ACTIVATION:",
                raw,
                "",
                "AV MESSAGE FROM GOLD ACTIVATION "
                f"({'RAW FALLBACK' if used_raw_fallback else 'TAG-EXTRACTED'}):",
                message if message is not None else "<EXTRACTION FAILED>",
                "",
                "TEXT OUTSIDE <explanation> TAGS:",
                f"PREFIX: {prefix!r}",
                f"SUFFIX: {suffix!r}",
                "",
                f"AR RECONSTRUCTION: mse={mse:.4f} reward={-mse:.4f} "
                f"cosine={cosine:.4f}",
                f"ROUND-TRIP RAW COMPLETION "
                f"({'RAW FALLBACK' if round_used_raw_fallback else 'TAG-EXTRACTED'}; "
                f"char similarity={similarity:.3f}):",
                round_raw,
                "",
                "ROUND-TRIP MESSAGE:",
                round_message if round_message is not None else "<EXTRACTION FAILED>",
            ]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print(f"Saved {output.resolve()} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
