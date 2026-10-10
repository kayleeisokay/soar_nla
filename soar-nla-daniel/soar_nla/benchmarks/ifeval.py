"""Sharded IFEval evaluation for Gemma-3-12B-IT with the all-prompt-token NLA round trip.

The model runtime and activation intervention mirror ``gsm8k``. Modes:
``extract`` (left padding), ``clean``, ``patched``, ``summarize``. IFEval
scoring uses Google's released strict/loose instruction checkers (vendored in
``ifeval_lib``); ``summarize`` and anything that scores responses need its
dependencies (absl-py, langdetect, nltk with punkt/punkt_tab, immutabledict).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from functools import lru_cache
from pathlib import Path

from soar_nla.benchmarks import common


@lru_cache(maxsize=1)
def _registry():
    # IFEval's response-language checker calls langdetect.  Pin its detector
    # seed so repeated summaries of the same generations are byte-for-byte
    # reproducible.
    try:  # pragma: no cover - only relevant for language-constrained examples.
        import langdetect
        langdetect.DetectorFactory.seed = 0
        # langdetect loads language profiles through os.listdir(); profile
        # order can vary across processes and change ties on short responses.
        # Sorting the profile filenames makes the official checker
        # reproducible here.
        from langdetect import detector_factory
        _listdir = detector_factory.os.listdir
        detector_factory.os.listdir = lambda path: sorted(_listdir(path))
    except ImportError:
        pass
    from soar_nla.benchmarks.ifeval_lib import instructions_registry

    return instructions_registry


def load_rows():
    return common.load_benchmark("ifeval")


def prompt_texts(tokenizer, rows):
    return common.chat_prompts(tokenizer, [row["prompt"] for row in rows])


def check_one(inp: dict, response: str, loose: bool) -> list[bool]:
    """Google IFEval's strict/loose per-instruction checks."""
    instructions_registry = _registry()
    if not loose:
        candidates = [response]
    else:
        lines = response.split("\n")
        response_remove_first = "\n".join(lines[1:]).strip()
        response_remove_last = "\n".join(lines[:-1]).strip()
        response_remove_both = "\n".join(lines[1:-1]).strip()
        revised_response = response.replace("*", "")
        revised_response_remove_first = response_remove_first.replace("*", "")
        revised_response_remove_last = response_remove_last.replace("*", "")
        revised_response_remove_both = response_remove_both.replace("*", "")
        candidates = [
            response,
            revised_response,
            response_remove_first,
            response_remove_last,
            response_remove_both,
            revised_response_remove_first,
            revised_response_remove_last,
            revised_response_remove_both,
        ]

    following = []
    for index, instruction_id in enumerate(inp["instruction_id_list"]):
        cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = cls(instruction_id)
        # The HF parquet conversion stores a schema-stabilized kwargs object
        # containing every possible field (mostly null).  Google's evaluator
        # receives only the fields relevant to the current instruction.
        all_kwargs = inp["kwargs"][index]
        allowed = instruction.get_instruction_args_keys()
        instruction.build_description(**{
            key: all_kwargs[key] for key in allowed if key in all_kwargs
        })
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp["prompt"])
        following.append(any(
            candidate.strip() and instruction.check_following(candidate)
            for candidate in candidates
        ))
    return following


def score_rows(rows: list[dict], responses: dict[int, str]) -> dict:
    """Return strict/loose prompt and instruction-level scores."""
    # Google runs the strict pass over the complete input list, followed by
    # the loose pass.  A few released examples contain an invalid letter
    # constraint (e.g. ``#``), for which the reference checker draws a random
    # fallback letter; seed that stream so the two passes are reproducible.
    random.seed(0)
    out = [{
        "key": row["key"],
        "prompt": row["prompt"],
        "instruction_id_list": row["instruction_id_list"],
        "response": responses[row["key"]],
    } for row in rows]

    # Keep strict and loose evaluation ordering identical to Google's
    # evaluation_main.py (all strict examples, then all loose examples).
    for item, row in zip(out, rows, strict=True):
        strict = check_one(row, item["response"], loose=False)
        item["strict_follow_instruction_list"] = strict
        item["strict_follow_all_instructions"] = all(strict)

    for item, row in zip(out, rows, strict=True):
        loose = check_one(row, item["response"], loose=True)
        item["loose_follow_instruction_list"] = loose
        item["loose_follow_all_instructions"] = all(loose)

    # Aggregate the two checker outputs.
    report = {}
    for name, all_field, list_field in (
        ("strict", "strict_follow_all_instructions", "strict_follow_instruction_list"),
        ("loose", "loose_follow_all_instructions", "loose_follow_instruction_list"),
    ):
        prompt_correct = sum(bool(item[all_field]) for item in out)
        instruction_total = sum(len(item[list_field]) for item in out)
        instruction_correct = sum(sum(item[list_field]) for item in out)
        by_id_total = collections.Counter()
        by_id_correct = collections.Counter()
        for item in out:
            for instruction_id, followed in zip(item["instruction_id_list"], item[list_field], strict=True):
                by_id_total[instruction_id] += 1
                by_id_correct[instruction_id] += int(followed)
        report[name] = {
            "prompt_correct": prompt_correct,
            "prompt_total": len(out),
            "prompt_accuracy": prompt_correct / len(out),
            "instruction_correct": instruction_correct,
            "instruction_total": instruction_total,
            "instruction_accuracy": instruction_correct / instruction_total,
            "by_instruction": {
                key: {"correct": by_id_correct[key], "total": by_id_total[key],
                      "accuracy": by_id_correct[key] / by_id_total[key]}
                for key in sorted(by_id_total)
            },
        }
    return {"rows": out, "summary": report}


def summarize(args) -> None:
    base = Path(args.out_dir)
    all_rows = load_rows()
    rows = [dict(all_rows[i]) for i in range(len(all_rows))]
    generation_conditions = ("clean", "patched")
    by_condition = {}
    generation_stats = {}
    for condition in generation_conditions:
        responses = {}
        generation_rows = []
        for shard in range(args.num_shards):
            path = base / f"{condition}_shard_{shard:02d}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            for line in path.read_text().splitlines():
                item = json.loads(line)
                responses[int(item["key"])] = item["output"]
                generation_rows.append(item)
        if set(responses) != {int(row["key"]) for row in rows}:
            raise RuntimeError(f"{condition} outputs do not cover IFEval exactly")
        by_condition[condition] = score_rows(rows, responses)
        cap_rows = [item for item in generation_rows if item.get("finish_reason") == "length"]
        generation_stats[condition] = {
            "generation_cap_hits": len(cap_rows),
            "generation_cap_hit_indices": sorted(int(item["index"]) for item in cap_rows),
            "max_generated_tokens_including_eos": max(
                int(item["generated_tokens"]) for item in generation_rows
            ),
        }
        for name, report in by_condition[condition]["summary"].items():
            print(f"{condition} {name}: prompt={report['prompt_correct']}/{report['prompt_total']} "
                  f"({report['prompt_accuracy']:.6f}), instruction="
                  f"{report['instruction_correct']}/{report['instruction_total']} "
                  f"({report['instruction_accuracy']:.6f})")

    paired = {}
    clean = by_condition["clean"]["rows"]
    for condition in ("patched",):
        other = by_condition[condition]["rows"]
        for mode in ("strict", "loose"):
            c_all = {x["key"]: x[f"{mode}_follow_all_instructions"] for x in clean}
            o_all = {x["key"]: x[f"{mode}_follow_all_instructions"] for x in other}
            paired[f"{condition}_{mode}"] = {
                "both_correct": sum(c_all[k] and o_all[k] for k in c_all),
                "clean_only_correct": sum(c_all[k] and not o_all[k] for k in c_all),
                "condition_only_correct": sum(not c_all[k] and o_all[k] for k in c_all),
                "both_incorrect": sum(not c_all[k] and not o_all[k] for k in c_all),
            }

    summary = {
        "model": common.TARGET_MODEL,
        "dataset": "google/IFEval train (official 541-prompt set)",
        "n": len(rows),
        "protocol": {
            "prompt": "Gemma chat template; one user message containing the IFEval prompt",
            "decoding": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
            "intervention": "replace every layer-32 prompt-token residual after block 32 with greedy AV->text->AR reconstruction; preserve each original L2 norm; generated-token activations unpatched",
            "scoring": "Google IFEval strict and loose checkers; prompt-level and instruction-level accuracy",
        },
        "conditions": {
            condition: {**by_condition[condition]["summary"], "generation": generation_stats[condition]}
            for condition in generation_conditions
        },
        "paired_outcomes": paired,
    }
    path = base / "ifeval_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    for condition in generation_conditions:
        (base / f"{condition}_scored.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in by_condition[condition]["rows"]) + "\n"
        )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("extract", "clean", "patched", "summarize"))
    common.add_common_args(parser, task="ifeval", max_new_tokens=4096, batch_size=4)
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
