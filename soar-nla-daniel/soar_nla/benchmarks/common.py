"""Shared runtime for the all-prompt-token NLA benchmark round trips.

Every benchmark (GSM8K, IFEval, ARC-Easy) uses the same intervention on
Gemma-3-12B-IT: format the example with the chat template, record the
residual after decoder block 32 at every prompt token (``extract``), verbalize
each one with the AV and reconstruct it with the AR (``roundtrip``), rescale
the reconstruction to the original L2 norm, then decode greedily while a
prefill hook replaces every prompt-token residual after block 32 with its
reconstruction. Generated-token activations are not patched.

Run directories are sharded by example (``range(shard, n, num_shards)``):
``activations/shard_NN.{safetensors,json}`` holds the flat bf16 prompt states
and their per-example offsets, ``explanations/`` and ``reconstructions/`` the
AV text and AR vectors at the same flat indices, and
``<condition>_shard_NN.jsonl`` the generations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from soar_nla.hub import GEMMA3_12B, KITFT_L32_AR, KITFT_L32_AV, resolve_snapshot
from soar_nla.paths import results_dir

LAYER = 32
NUM_SHARDS = 8
TARGET_MODEL = GEMMA3_12B[0]

# (repo, config, split, revision). The revisions are the dataset snapshots the
# paper runs loaded.
DATASETS = {
    "gsm8k": ("openai/gsm8k", "main", "test", "740312add88f781978c0658806c59bc2815b9866"),
    "ifeval": ("google/IFEval", None, "train", "966cd89545d6b6acfd7638bc708b98261ca58e84"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", "210d026faf9955653af8916fad021475a3f00453"),
}


def default_run_dir(task: str) -> Path:
    return results_dir("benchmarks") / task


def target_path() -> str:
    return str(resolve_snapshot(*GEMMA3_12B))


def av_path() -> str:
    return str(resolve_snapshot(*KITFT_L32_AV))


def ar_path() -> str:
    return str(resolve_snapshot(*KITFT_L32_AR))


def load_benchmark(task: str):
    from datasets import load_dataset

    repo, config, split, revision = DATASETS[task]
    return load_dataset(repo, config, split=split, revision=revision)


def shard_rows(all_rows, shard: int, num_shards: int) -> list[dict]:
    return [{**all_rows[i], "_index": i} for i in range(shard, len(all_rows), num_shards)]


def chat_prompts(tokenizer, contents: list[str]) -> list[str]:
    """One user turn per example, followed by the model-turn header."""
    return [tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    ) for content in contents]


def load_tokenizer(model_path: str, padding_side: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(model_path: str, padding_side: str = "left"):
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_path, padding_side)
    wrapper = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        trust_remote_code=True, attn_implementation="sdpa",
    ).eval()
    # Gemma-3 checkpoints may load as a multimodal wrapper whose
    # ``language_model`` is the bare Gemma3TextModel (layers directly on it),
    # or as a text causal-LM wrapper (layers below ``model``).
    language_model = getattr(wrapper, "language_model", wrapper)
    backbone = getattr(language_model, "model", language_model)
    return wrapper, backbone, tokenizer


def generation_metadata(token_ids, eos_token_ids) -> dict:
    """Record exact generated length (including EOS) and EOS-versus-length termination."""
    ids = [int(token_id) for token_id in token_ids]
    if isinstance(eos_token_ids, int):
        eos = {eos_token_ids}
    else:
        eos = {int(token_id) for token_id in (eos_token_ids or [])}
    for position, token_id in enumerate(ids):
        if token_id in eos:
            return {"generated_tokens": position + 1, "finish_reason": "eos"}
    return {"generated_tokens": len(ids), "finish_reason": "length"}


def run_extract(
    out_dir: Path, shard: int, texts: list[str], meta: dict, *,
    model_path: str, batch_size: int, padding_side: str,
) -> None:
    """Capture every prompt-position residual after Gemma block 32.

    ``padding_side`` selects how valid tokens are sliced out of each padded
    batch: with right padding they occupy the first ``length`` positions, with
    left padding the last ``length``. GSM8K was extracted with right padding,
    IFEval and ARC-Easy with left padding.
    """
    import torch
    from safetensors.torch import save_file

    activation_dir = out_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)
    final = activation_dir / f"shard_{shard:02d}.safetensors"
    meta_final = activation_dir / f"shard_{shard:02d}.json"
    if final.exists() and meta_final.exists():
        print(f"EXISTS {final}")
        return
    wrapper, backbone, tokenizer = load_model(model_path, padding_side)
    chunks, lengths = [], []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], return_tensors="pt",
                          padding=True, add_special_tokens=False).to("cuda:0")
        storage = {}

        def hook(_module, _inputs, output):
            storage["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()

        handle = backbone.layers[LAYER].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                wrapper(**batch, use_cache=False)
        finally:
            handle.remove()
        for row_index, length in enumerate(batch.attention_mask.sum(dim=1).tolist()):
            if padding_side == "left":
                # The valid prompt occupies the final ``length`` positions;
                # slicing from zero would replay pad states and silently
                # misalign every activation.
                chunks.append(storage["hidden"][row_index, -length:].bfloat16().cpu())
            else:
                chunks.append(storage["hidden"][row_index, :length].bfloat16().cpu())
            lengths.append(int(length))
        print(f"[extract {shard}] {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    flat = torch.cat(chunks, dim=0).contiguous()
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    temp = final.with_suffix(".tmp.safetensors")
    save_file({"activations": flat}, str(temp))
    temp.replace(final)
    meta_temp = meta_final.with_suffix(".tmp.json")
    meta_temp.write_text(json.dumps({**meta, "lengths": lengths, "offsets": offsets}, indent=2) + "\n")
    meta_temp.replace(meta_final)
    print(f"WROTE {final} {tuple(flat.shape)}", flush=True)


def load_replay(out_dir: Path, activations_dir: Path, shard: int, condition: str):
    """Return (lengths, offsets, vectors) for an ``identity`` or ``patched`` rollout.

    ``identity`` replays the captured activations themselves; ``patched``
    replays ``<out_dir>/reconstructions`` (the NLA round trip, or a
    paraphrase/random arm that stores its replacement vectors there).
    """
    from safetensors.torch import load_file

    meta = json.loads((activations_dir / f"shard_{shard:02d}.json").read_text())
    if condition == "identity":
        vectors = load_file(str(activations_dir / f"shard_{shard:02d}.safetensors"))["activations"]
    else:
        vectors = load_file(str(out_dir / "reconstructions" / f"shard_{shard:02d}.safetensors"))["reconstructions"]
    if vectors.shape[0] != meta["offsets"][-1]:
        raise RuntimeError("replacement count does not match activation offsets")
    return meta["lengths"], meta["offsets"], vectors


def generate(
    wrapper, backbone, tokenizer, texts: list[str], *, batch_size: int, max_new_tokens: int,
    label: str, replay=None,
) -> list[tuple[str, dict]]:
    """Greedy generation; with ``replay=(lengths, offsets, vectors)`` replace
    every layer-32 prompt state on prefill. Returns (text, metadata) pairs."""
    import torch

    results = []
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        batch = tokenizer(texts[start:stop], return_tensors="pt", padding=True,
                          add_special_tokens=False).to("cuda:0")
        max_len = batch.input_ids.shape[1]
        handle = None
        fired = False
        if replay is not None:
            lengths, offsets, source = replay
            replacement = torch.zeros((stop - start, max_len, source.shape[1]), dtype=torch.bfloat16)
            masks = torch.zeros((stop - start, max_len), dtype=torch.bool)
            for row_index, item in enumerate(range(start, stop)):
                length = lengths[item]
                replacement[row_index, -length:] = source[offsets[item]:offsets[item + 1]]
                masks[row_index, -length:] = True

            def hook(_module, _inputs, output):
                nonlocal fired
                hidden = output[0] if isinstance(output, tuple) else output
                if fired or hidden.shape[1] != max_len:
                    return output
                patched = torch.where(
                    masks.to(hidden.device)[..., None],
                    replacement.to(hidden.device, hidden.dtype), hidden,
                )
                fired = True
                return (patched, *output[1:]) if isinstance(output, tuple) else patched

            handle = backbone.layers[LAYER].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                generated = wrapper.generate(
                    **batch, do_sample=False, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, use_cache=True,
                    disable_compile=True,
                )
        finally:
            if handle is not None:
                handle.remove()
        if replay is not None and not fired:
            raise RuntimeError(f"{label} layer-32 prefill hook did not fire")
        new = generated[:, max_len:].cpu()
        for row_index in range(stop - start):
            text = tokenizer.decode(new[row_index], skip_special_tokens=True).strip()
            if text.startswith("model\n"):
                text = text[6:].strip()
            results.append((text, generation_metadata(new[row_index], wrapper.generation_config.eos_token_id)))
        print(f"[{label}] {stop}/{len(texts)}", flush=True)
    return results


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(".tmp.jsonl")
    with temp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)
    print(f"WROTE {path} ({len(rows)} examples)", flush=True)


def run_hf(args, condition: str, rows: list[dict], texts: list[str]) -> None:
    """Generate ``clean``, ``identity`` or ``patched`` responses for one shard
    (IFEval and ARC-Easy output schema)."""
    out_dir = Path(args.out_dir)
    replay = None
    if condition in ("identity", "patched"):
        replay = load_replay(out_dir, activations_dir(args), args.shard, condition)
    wrapper, backbone, tokenizer = load_model(args.model or target_path(), "left")
    results = generate(
        wrapper, backbone, tokenizer, texts, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens, label=f"{condition} {args.shard}", replay=replay,
    )
    write_jsonl(out_dir / f"{condition}_shard_{args.shard:02d}.jsonl", [{
        "key": row["key"], "index": row["_index"], "prompt": row["prompt"], "output": text, **meta,
    } for row, (text, meta) in zip(rows, results, strict=True)])


def activations_dir(args) -> Path:
    if getattr(args, "activations_dir", None):
        return Path(args.activations_dir)
    return Path(args.out_dir) / "activations"


def exact_mcnemar(first_only: int, second_only: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    discordant = first_only + second_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1))
    return min(1.0, 2 * tail / 2**discordant)


def add_common_args(parser, *, task: str, max_new_tokens: int, batch_size: int) -> None:
    parser.add_argument("--model", default=None,
                        help="target checkpoint directory (default: pinned google/gemma-3-12b-it snapshot)")
    parser.add_argument("--out-dir", type=Path, default=default_run_dir(task))
    parser.add_argument("--activations-dir", type=Path, default=None,
                        help="native activations to replay against (default: <out-dir>/activations); "
                             "set for the paraphrase/random arms, whose out-dir holds only the replacements")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    parser.add_argument("--batch-size", type=int, default=batch_size)
