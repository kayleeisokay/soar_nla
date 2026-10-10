"""Sharded ARC-Easy evaluation for Gemma-3-12B-IT with the all-prompt-token NLA round trip.

Same runtime and intervention as ``ifeval`` (left-padded extraction, HF greedy
generation). Each question is shown with all labeled answer choices and an
instruction to answer with the letter only; scoring is exact answer-letter
accuracy. Modes: ``extract``, ``clean``, ``patched``, ``summarize``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from soar_nla.benchmarks import common


def load_rows():
    dataset = common.load_benchmark("arc_easy")
    rows = []
    for index, row in enumerate(dataset):
        item = {
            "key": index,
            "id": row["id"],
            "question": row["question"],
            "choices": row["choices"],
            "answerKey": row["answerKey"],
        }
        item["prompt"] = render_prompt(item)
        rows.append(item)
    return rows


def render_prompt(row: dict) -> str:
    choices = "\n".join(
        f"{label}. {text}"
        for label, text in zip(row["choices"]["label"], row["choices"]["text"], strict=True)
    )
    return (
        f"{row['question']}\n\n{choices}\n\n"
        "Respond with only the letter of the correct answer."
    )


def prompt_texts(tokenizer, rows):
    return common.chat_prompts(tokenizer, [render_prompt(row) for row in rows])


def canonical_choice(label: str, source_labels: list[str]) -> str | None:
    cleaned = label.strip().strip("*()[] .,: ").upper()
    if cleaned in source_labels:
        return chr(ord("A") + source_labels.index(cleaned))
    if len(cleaned) == 1 and "A" <= cleaned <= chr(ord("A") + len(source_labels) - 1):
        return cleaned
    return None


def predicted_label(text: str, source_labels: list[str]) -> str | None:
    direct = canonical_choice(text, source_labels)
    if direct is not None:
        return direct
    match = re.match(
        r"^\s*(?:the\s+(?:correct\s+)?answer\s+is\s*)?\**[\(\[]?([A-E1-5])[\)\].,:]?\**\b",
        text,
        re.I,
    )
    if match:
        return canonical_choice(match.group(1), source_labels)
    return None


def is_correct(row: dict, output: str) -> bool:
    labels = [str(x).upper() for x in row["choices"]["label"]]
    return predicted_label(output, labels) == canonical_choice(str(row["answerKey"]), labels)


def summarize(args) -> None:
    import torch
    from safetensors.torch import load_file

    base = Path(args.out_dir)
    dataset = load_rows()
    expected = set(range(len(dataset)))
    conditions = {}
    for condition in ("clean", "patched"):
        by_index = {}
        for shard in range(args.num_shards):
            path = base / f"{condition}_shard_{shard:02d}.jsonl"
            for line in path.read_text().splitlines():
                item = json.loads(line)
                index = int(item["index"])
                source_labels = [str(x).upper() for x in dataset[index]["choices"]["label"]]
                prediction = predicted_label(item["output"], source_labels)
                reference = canonical_choice(str(dataset[index]["answerKey"]), source_labels)
                item.update({
                    "id": dataset[index]["id"],
                    "reference": reference,
                    "prediction": prediction,
                    "correct": prediction == reference,
                })
                by_index[index] = item
        if set(by_index) != expected:
            raise RuntimeError(f"{condition} does not cover ARC-Easy exactly")
        conditions[condition] = [by_index[i] for i in range(len(dataset))]

    summary_conditions = {}
    for condition, rows in conditions.items():
        correct = sum(row["correct"] for row in rows)
        capped = [row["index"] for row in rows if row.get("finish_reason") == "length"]
        summary_conditions[condition] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows),
            "unparsed_predictions": sum(row["prediction"] is None for row in rows),
            "generation_cap_hits": len(capped),
            "generation_cap_hit_indices": capped,
        }

    clean, patched = conditions["clean"], conditions["patched"]
    both = sum(c["correct"] and p["correct"] for c, p in zip(clean, patched, strict=True))
    clean_only = sum(c["correct"] and not p["correct"] for c, p in zip(clean, patched, strict=True))
    nla_only = sum(not c["correct"] and p["correct"] for c, p in zip(clean, patched, strict=True))
    neither = len(dataset) - both - clean_only - nla_only

    cosines = []
    parse_failures = 0
    for shard in range(args.num_shards):
        acts = load_file(str(base / "activations" / f"shard_{shard:02d}.safetensors"))["activations"].float()
        recon = load_file(str(base / "reconstructions" / f"shard_{shard:02d}.safetensors"))["reconstructions"].float()
        cosines.append(torch.nn.functional.cosine_similarity(acts, recon, dim=-1))
        explanation_path = base / "explanations" / f"shard_{shard:02d}.jsonl"
        parse_failures += sum(not json.loads(line)["parse_ok"] for line in explanation_path.read_text().splitlines())

    report = {
        "model": common.TARGET_MODEL,
        "dataset": "allenai/ai2_arc ARC-Easy test",
        "n": len(dataset),
        "protocol": {
            "prompt": "Gemma chat template; one user message with question, labeled choices, and an instruction to return only the answer letter",
            "decoding": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
            "scoring": "exact answer-letter accuracy",
            "intervention": "replace every layer-32 prefill-token residual after block 32 with greedy AV->text->AR reconstruction; preserve each original L2 norm; generated-token activations unpatched",
        },
        "conditions": summary_conditions,
        "paired_clean_vs_nla": {
            "both_correct": both,
            "clean_only_correct": clean_only,
            "nla_only_correct": nla_only,
            "both_incorrect": neither,
            "accuracy_difference_points": 100 * (nla_only - clean_only) / len(dataset),
            "two_sided_exact_mcnemar_p": common.exact_mcnemar(clean_only, nla_only),
        },
        "reconstruction": {
            "mean_cosine": float(torch.cat(cosines).mean()),
            "av_explanation_parse_failures": parse_failures,
        },
    }
    (base / "arc_easy_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for condition, rows in conditions.items():
        (base / f"{condition}_scored.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("extract", "clean", "patched", "summarize"))
    # The paper run used --batch-size 8 for extract and 16 for clean/patched.
    common.add_common_args(parser, task="arc_easy", max_new_tokens=32, batch_size=8)
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.mode == "summarize":
        summarize(args)
        return
    rows = common.shard_rows(load_rows(), args.shard, args.num_shards)
    tokenizer = common.load_tokenizer(args.model or common.target_path(), "left")
    texts = prompt_texts(tokenizer, rows)
    if args.mode == "extract":
        common.run_extract(
            Path(args.out_dir), args.shard, texts,
            {"indices": [row["_index"] for row in rows], "keys": [row["key"] for row in rows]},
            model_path=args.model or common.target_path(), batch_size=args.batch_size, padding_side="left",
        )
    else:
        common.run_hf(args, args.mode, rows, texts)


if __name__ == "__main__":
    main()
