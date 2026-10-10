"""Greedy Gemma rollouts and one sampled layer-32 activation per response.

Each selected Alpaca prompt is answered greedily (at most 192 new tokens). One
assistant-token position per response is chosen by a deterministic hash of
`(PROTOCOL_VERSION, model, sample_id)`, uniformly over the generated span, and
the output of decoder block `spec.layer` at that position is saved.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from soar_nla.hub import MODEL_SPECS, load_target, model_spec, resolve_decoder_layers
from soar_nla.io_utils import read_jsonl, stable_seed, write_json, write_jsonl
from soar_nla.paths import results_dir

from .config import N_EXAMPLES, PROTOCOL_VERSION


def _truncate_user(tokenizer: Any, text: str, max_tokens: int) -> tuple[str, bool]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text, False
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True), True


def _strip_generation(ids: list[int], tokenizer: Any) -> list[int]:
    stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    output = []
    for token in ids:
        if token in stop_ids:
            break
        if token not in tokenizer.all_special_ids:
            output.append(token)
    return output


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("rollout worker expects exactly one visible GPU")
    spec = model_spec(args.model)
    selected = read_jsonl(args.data_dir / "alpaca_selected.jsonl")
    if len(selected) != args.n:
        raise RuntimeError(f"expected {args.n} selected samples, found {len(selected)}")
    work = [row for row in selected if row["selection_rank"] % args.num_shards == args.shard]
    model, tokenizer, checkpoint = load_target(spec)
    layers = resolve_decoder_layers(model)
    if spec.layer >= len(layers):
        raise RuntimeError(f"layer {spec.layer} outside target with {len(layers)} layers")

    records: list[dict[str, Any]] = []
    activation_rows: list[torch.Tensor] = []
    for batch_start in range(0, len(work), args.batch_size):
        batch_rows = work[batch_start : batch_start + args.batch_size]
        rendered_prompts = []
        prompt_ids = []
        truncated_flags = []
        for row in batch_rows:
            user_text, truncated = _truncate_user(tokenizer, row["user_prompt"], args.max_user_tokens)
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}], tokenize=True, add_generation_prompt=True
            )
            rendered_prompts.append(user_text)
            prompt_ids.append(ids)
            truncated_flags.append(truncated)
        tokenizer.padding_side = "left"
        batch = tokenizer.pad(
            {"input_ids": prompt_ids}, padding=True, return_tensors="pt", return_attention_mask=True
        ).to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        suffix = generated[:, batch["input_ids"].shape[1] :].cpu().tolist()
        assistant_ids = [_strip_generation(ids, tokenizer) for ids in suffix]
        if any(not ids for ids in assistant_ids):
            bad = [row["sample_id"] for row, ids in zip(batch_rows, assistant_ids, strict=True) if not ids]
            raise RuntimeError(f"empty assistant generation for {bad}")

        full_ids = [prompt + response for prompt, response in zip(prompt_ids, assistant_ids, strict=True)]
        tokenizer.padding_side = "right"
        forward_batch = tokenizer.pad(
            {"input_ids": full_ids}, padding=True, return_tensors="pt", return_attention_mask=True
        ).to("cuda:0")
        captured: dict[str, torch.Tensor] = {}

        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            storage: dict[str, torch.Tensor] = captured,
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            storage["hidden"] = hidden.detach().float().cpu()

        handle = layers[spec.layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                model(**forward_batch, use_cache=False)
        finally:
            handle.remove()
        hidden = captured["hidden"]
        for local_index, (row, prompt, response) in enumerate(zip(batch_rows, prompt_ids, assistant_ids, strict=True)):
            response_index = stable_seed(PROTOCOL_VERSION, spec.key, row["sample_id"], "assistant-position") % len(
                response
            )
            sequence_index = len(prompt) + response_index
            activation = hidden[local_index, sequence_index].contiguous()
            if activation.numel() != spec.d_model:
                raise RuntimeError(f"activation shape {tuple(activation.shape)} != ({spec.d_model},)")
            through_ids = (prompt + response)[: sequence_index + 1]
            before_ids = (prompt + response)[:sequence_index]
            record = {
                "protocol": PROTOCOL_VERSION,
                "model": spec.key,
                "sample_id": row["sample_id"],
                "selection_rank": row["selection_rank"],
                "dataset_index": row["dataset_index"],
                "user_prompt": rendered_prompts[local_index],
                "user_prompt_truncated": truncated_flags[local_index],
                "assistant_text": tokenizer.decode(response, skip_special_tokens=True),
                "prompt_token_count": len(prompt),
                "assistant_token_count": len(response),
                "assistant_token_index": int(response_index),
                "sequence_token_index": int(sequence_index),
                "sampled_token_id": int(response[response_index]),
                "sampled_token_text": tokenizer.decode([response[response_index]], skip_special_tokens=False),
                "preceding_text_exclusive": tokenizer.decode(before_ids, skip_special_tokens=True),
                "preceding_text_inclusive": tokenizer.decode(through_ids, skip_special_tokens=True),
                "activation_norm": float(activation.norm().item()),
                "activation_sha256": hashlib.sha256(activation.numpy().tobytes()).hexdigest(),
            }
            records.append(record)
            activation_rows.append(activation)
        print(f"[{spec.key} shard {args.shard}] {len(records)}/{len(work)}", flush=True)

    order = sorted(range(len(records)), key=lambda index: records[index]["selection_rank"])
    records = [records[index] for index in order]
    activations = torch.stack([activation_rows[index] for index in order]).float().contiguous()
    stem = f"shard-{args.shard:02d}-of-{args.num_shards:02d}"
    output_dir = args.output_dir / spec.key / "rollouts"
    write_jsonl(output_dir / f"{stem}.jsonl", records)
    save_file({"activations": activations}, str(output_dir / f"{stem}.safetensors"))
    write_json(
        output_dir / f"{stem}.manifest.json",
        {
            "protocol": PROTOCOL_VERSION,
            "model": spec.__dict__,
            "target_checkpoint": str(checkpoint),
            "shard": args.shard,
            "num_shards": args.num_shards,
            "rows": len(records),
            "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def merge(args: argparse.Namespace) -> None:
    spec = model_spec(args.model)
    source = args.output_dir / spec.key / "rollouts"
    rows = []
    vectors = []
    for shard in range(args.num_shards):
        stem = f"shard-{shard:02d}-of-{args.num_shards:02d}"
        shard_rows = read_jsonl(source / f"{stem}.jsonl")
        shard_vectors = load_file(str(source / f"{stem}.safetensors"))["activations"].float()
        if len(shard_rows) != len(shard_vectors):
            raise RuntimeError(f"metadata/vector mismatch in {stem}")
        rows.extend(shard_rows)
        vectors.extend(shard_vectors.unbind())
    order = sorted(range(len(rows)), key=lambda index: rows[index]["selection_rank"])
    rows = [rows[index] for index in order]
    activations = torch.stack([vectors[index] for index in order]).contiguous()
    if len(rows) != args.n or len({row["sample_id"] for row in rows}) != args.n:
        raise RuntimeError(f"expected {args.n} distinct merged rows, got {len(rows)}")
    if len({row["activation_sha256"] for row in rows}) != args.n:
        raise RuntimeError("sampled assistant-span activations are not all distinct")
    if activations.shape != (args.n, spec.d_model):
        raise RuntimeError(f"unexpected activation shape: {tuple(activations.shape)}")
    write_jsonl(source / "rollouts.jsonl", rows)
    save_file({"activations": activations}, str(source / "activations.safetensors"))
    write_json(
        source / "manifest.json",
        {
            "protocol": PROTOCOL_VERSION,
            "model": spec.__dict__,
            "rows": len(rows),
            "activation_shape": list(activations.shape),
            "distinct_sample_ids": len({row["sample_id"] for row in rows}),
            "distinct_activation_hashes": len({row["activation_sha256"] for row in rows}),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker", help="generate and capture one shard (one GPU)")
    worker.add_argument("--model", default="gemma12b", choices=sorted(MODEL_SPECS))
    worker.add_argument("--data-dir", type=Path, default=results_dir("alpaca") / "data")
    worker.add_argument("--output-dir", type=Path, default=results_dir("alpaca"))
    worker.add_argument("--n", type=int, default=N_EXAMPLES)
    worker.add_argument("--shard", type=int, required=True)
    worker.add_argument("--num-shards", type=int, default=4)
    worker.add_argument("--batch-size", type=int, default=4)
    worker.add_argument("--max-user-tokens", type=int, default=768)
    worker.add_argument("--max-new-tokens", type=int, default=192)
    merger = sub.add_parser("merge", help="merge shards into rollouts.jsonl and activations.safetensors")
    merger.add_argument("--model", default="gemma12b", choices=sorted(MODEL_SPECS))
    merger.add_argument("--output-dir", type=Path, default=results_dir("alpaca"))
    merger.add_argument("--n", type=int, default=N_EXAMPLES)
    merger.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()
    if args.command == "worker":
        run_worker(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
