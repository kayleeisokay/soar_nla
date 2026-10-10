"""Sharded GSM8K evaluation for Gemma-3-12B-IT with the all-prompt-token NLA round trip.

Modes, in pipeline order (8 shards, one GPU each):

* ``extract``: layer-32 residual at every chat-prompt token (right padding).
* ``clean``: matched HF greedy generation, no hook.
* ``identity``: replay the captured activations themselves (hook-fidelity control).
* ``patched``: replay ``<out-dir>/reconstructions`` (norm-matched AR(AV(h)), or a
  paraphrase/random arm).
* ``summarize``: ``matched_long_generation_summary.json``.

The AV/AR stages between ``extract`` and ``patched`` live in
``soar_nla.benchmarks.roundtrip`` (``decode`` via SGLang, then ``reconstruct``).
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from soar_nla.benchmarks import common

N_EXAMPLES = 1319
# Shard-file prefixes. The clean and identity rollouts are "matched" HF
# controls for the patched rollout (same runtime, batching and decoding).
PREFIXES = {"clean": "matched_clean", "identity": "matched_identity", "patched": "patched"}


def canonical_number(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    return format(value.normalize(), "f")


def reference_answer(text: str) -> str:
    match = re.search(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)", text)
    if not match:
        raise ValueError(f"No GSM8K reference answer in {text!r}")
    return canonical_number(match.group(1))  # type: ignore[return-value]


def predicted_answer(text: str) -> str | None:
    # Standard GSM8K evaluation: compare the final numeric expression in the
    # response.  In particular, do not preferentially select a marker match:
    # that fails on common forms such as ``Final Answer: ... \boxed{18}``.
    candidates = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", text)
    return canonical_number(candidates[-1]) if candidates else None


def shard_prompts(tokenizer, dataset, shard: int, num_shards: int) -> tuple[list[int], list[str]]:
    indices = list(range(shard, len(dataset), num_shards))
    return indices, common.chat_prompts(tokenizer, [dataset[i]["question"] for i in indices])


def run_extract(args) -> None:
    dataset = common.load_benchmark("gsm8k")
    model_path = args.model or common.target_path()
    tokenizer = common.load_tokenizer(model_path, "right")
    indices, prompts = shard_prompts(tokenizer, dataset, args.shard, args.num_shards)
    common.run_extract(
        Path(args.out_dir), args.shard, prompts, {"indices": indices},
        model_path=model_path, batch_size=args.batch_size, padding_side="right",
    )


def run_generation(args, condition: str) -> None:
    """Matched HF clean, identity-replay, or patched generation for one shard."""
    out_dir = Path(args.out_dir)
    dataset = common.load_benchmark("gsm8k")
    meta = json.loads((common.activations_dir(args) / f"shard_{args.shard:02d}.json").read_text())
    indices = meta["indices"]
    replay = None
    if condition in ("identity", "patched"):
        replay = common.load_replay(out_dir, common.activations_dir(args), args.shard, condition)
    wrapper, backbone, tokenizer = common.load_model(args.model or common.target_path(), "left")
    prompts = common.chat_prompts(tokenizer, [dataset[i]["question"] for i in indices])
    results = common.generate(
        wrapper, backbone, tokenizer, prompts, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens, label=f"{condition} {args.shard}", replay=replay,
    )
    rows = []
    for index, (output, generation) in zip(indices, results, strict=True):
        gold = reference_answer(dataset[index]["answer"])
        pred = predicted_answer(output)
        row = {"index": index}
        if condition == "patched":
            row["question"] = dataset[index]["question"]
        row.update({"reference": gold, "prediction": pred, "correct": pred == gold,
                    "output": output, **generation})
        rows.append(row)
    common.write_jsonl(out_dir / f"{PREFIXES[condition]}_shard_{args.shard:02d}.jsonl", rows)


def load_condition(base: Path, prefix: str, num_shards: int) -> list[dict]:
    rows = []
    for shard in range(num_shards):
        path = base / f"{prefix}_shard_{shard:02d}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    rows.sort(key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(N_EXAMPLES)):
        raise RuntimeError(f"{prefix} does not cover GSM8K exactly once")
    return rows


def summarize(args) -> None:
    """Summarize the matched HF runs, including exact generation stop reasons."""
    base = Path(args.out_dir)
    conditions = {"clean": load_condition(base, PREFIXES["clean"], args.num_shards)}
    if all((base / f"{PREFIXES['identity']}_shard_{shard:02d}.jsonl").exists()
           for shard in range(args.num_shards)):
        conditions["identity_replay"] = load_condition(base, PREFIXES["identity"], args.num_shards)
    conditions["nla_roundtrip"] = load_condition(base, PREFIXES["patched"], args.num_shards)
    report = {
        "model": common.TARGET_MODEL,
        "dataset": "openai/gsm8k main test",
        "n": N_EXAMPLES,
        "decoding": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "conditions": {},
    }
    for name, rows in conditions.items():
        correct = sum(bool(row["correct"]) for row in rows)
        report["conditions"][name] = {
            "correct": correct,
            "accuracy": correct / len(rows),
            "generation_cap_hits": sum(row.get("finish_reason") == "length" for row in rows),
            "generation_cap_hit_indices": [
                row["index"] for row in rows if row.get("finish_reason") == "length"
            ],
            "max_generated_tokens_including_eos": max(row["generated_tokens"] for row in rows),
        }
    clean = conditions["clean"]
    patched = conditions["nla_roundtrip"]
    clean_only = sum(c["correct"] and not p["correct"] for c, p in zip(clean, patched))
    nla_only = sum(not c["correct"] and p["correct"] for c, p in zip(clean, patched))
    report["paired_clean_vs_nla"] = {
        "both_correct": sum(c["correct"] and p["correct"] for c, p in zip(clean, patched)),
        "clean_only_correct": clean_only,
        "nla_only_correct": nla_only,
        "both_incorrect": sum(not c["correct"] and not p["correct"] for c, p in zip(clean, patched)),
        "accuracy_difference_points": 100.0 * (nla_only - clean_only) / N_EXAMPLES,
        "two_sided_exact_mcnemar_p": common.exact_mcnemar(clean_only, nla_only),
    }
    output = base / "matched_long_generation_summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("extract", "clean", "identity", "patched", "summarize"))
    common.add_common_args(parser, task="gsm8k", max_new_tokens=4096, batch_size=8)
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.mode == "extract":
        run_extract(args)
    elif args.mode == "summarize":
        summarize(args)
    else:
        run_generation(args, args.mode)


if __name__ == "__main__":
    main()
