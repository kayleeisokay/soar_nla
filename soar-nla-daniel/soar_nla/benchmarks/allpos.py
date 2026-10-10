"""All-prompt-token paraphrase and random arms for GSM8K, IFEval, and ARC-Easy.

Extends the all-token NLA round trips (every layer-32 prompt-token residual
replaced by a norm-matched AR reconstruction) with the paraphrase and
random-vector arms of the English->French function-vector protocol
(``soar_nla.fv_chat_template``), applied at every prompt position:

* ``para_medium`` / ``para_heavy``: each position's cached AV explanation is
  paraphrased with the protocol's tagged prompts (raw Gemma-3-12B-IT prompt,
  greedy, <= 512 new tokens), reconstructed by AR, and norm-matched to the
  native residual at that position.
* ``random_vec``: seeded Gaussian vector per position, norm-matched to the
  native residual at that position.

Each ``<out-dir>/<task>__<arm>`` directory holds ``explanations/`` (the
paraphrased text) and ``reconstructions/`` (the replacement vectors), so the
task modules' ``patched`` mode runs on it unchanged when given
``--activations-dir <task run>/activations``. Clean and unchanged-NLA rollouts
come from the task runs themselves (4096-token cap for GSM8K and IFEval, 32
for ARC-Easy).

Stages: ``setup``, ``random``, ``paraphrase`` (vLLM, one shard per GPU),
``summarize``. The AR reconstruction of the paraphrases and the rollouts are
run with ``roundtrip reconstruct`` and ``<task> patched`` (see run_allpos.sh).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from soar_nla.benchmarks import common
from soar_nla.benchmarks.rollouts import (
    ARMS,
    TASKS,
    RolloutSource,
    add_source_args,
    arm_dir,
    read_jsonl,
    task_dirs_from_args,
)
from soar_nla.fv_chat_template.protocol import PARAPHRASE_PROMPTS
from soar_nla.hub import GEMMA3_12B

PARA_ARMS = ("para_medium", "para_heavy")
PARA_MAX_NEW = 512
TAG_RE = re.compile(r"<transformed_text>(.*?)</transformed_text>", re.DOTALL)
CLOSE_TAG = "</transformed_text>"


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def word_overlap(original: str, transformed: str) -> float:
    # The protocol's metric: fraction of the original's word set kept.
    original_words = set(original.lower().split())
    transformed_words = set(transformed.lower().split())
    return len(original_words & transformed_words) / len(original_words) if original_words else 0.0


def setup(args) -> None:
    for task in TASKS:
        for arm in ARMS:
            (arm_dir(args.arms_dir, task, arm) / "logs").mkdir(parents=True, exist_ok=True)
    atomic_json(args.arms_dir / "manifest.json", {
        "protocol": "all-prompt-token layer-32 replacement; function-vector protocol paraphrase and random arms",
        "target_model": GEMMA3_12B[0],
        "target_revision": GEMMA3_12B[1],
        "layer": common.LAYER,
        "tasks": list(TASKS),
        "arms": {
            "para_medium": "AR(medium paraphrase of each position's AV text), norm-matched per position",
            "para_heavy": "AR(heavy paraphrase of each position's AV text), norm-matched per position",
            "random_vec": "seeded Gaussian per position, norm-matched to the native residual",
        },
        "paraphraser": {
            "model": GEMMA3_12B[0], "prompt_style": "raw (no chat template), BOS added",
            "decoding": "greedy", "max_new_tokens": PARA_MAX_NEW,
            "stop": CLOSE_TAG,
            "extraction": "first <transformed_text>...</transformed_text>; full output if no tags",
        },
    })
    print(f"set up {args.arms_dir}", flush=True)


def paraphrase(args) -> None:
    from vllm import LLM, SamplingParams

    task_dirs = task_dirs_from_args(args)
    llm = None
    sampling = SamplingParams(
        temperature=0.0, max_tokens=PARA_MAX_NEW, stop=[CLOSE_TAG],
        include_stop_str_in_output=True, seed=0,
    )
    shard = args.shard
    for task in TASKS:
        source = read_jsonl(task_dirs[task] / "explanations" / f"shard_{shard:02d}.jsonl")
        source.sort(key=lambda row: int(row["flat_index"]))
        if args.limit:
            source = source[: args.limit]
        for arm in PARA_ARMS:
            directory = arm_dir(args.arms_dir, task, arm) / "explanations"
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / f"shard_{shard:02d}.jsonl"
            done = {int(row["flat_index"]) for row in read_jsonl(output)}
            pending = [row for row in source if int(row["flat_index"]) not in done]
            print(f"[paraphrase {shard}] {task}/{arm}: {len(pending)} pending / {len(source)}", flush=True)
            if not pending:
                continue
            if llm is None:
                target = common.target_path()
                llm = LLM(
                    model=target, tokenizer=target,
                    dtype="bfloat16", tensor_parallel_size=1, max_model_len=2048,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    trust_remote_code=True, seed=0,
                    limit_mm_per_prompt={"image": 0, "video": 0},
                )
            with output.open("a") as handle:
                for start in range(0, len(pending), args.request_batch_size):
                    chunk = pending[start : start + args.request_batch_size]
                    prompts = [PARAPHRASE_PROMPTS[arm].format(text=row["explanation"]) for row in chunk]
                    results = llm.generate(prompts, sampling, use_tqdm=False)
                    for row, result in zip(chunk, results, strict=True):
                        raw = result.outputs[0].text
                        match = TAG_RE.search(raw)
                        transformed = match.group(1).strip() if match else raw.strip()
                        handle.write(json.dumps({
                            "flat_index": int(row["flat_index"]),
                            "parse_ok": match is not None,
                            "explanation": transformed,
                            "source_explanation": row["explanation"],
                            "word_overlap": word_overlap(row["explanation"], transformed),
                            "generated_tokens": len(result.outputs[0].token_ids),
                            "finish_reason": result.outputs[0].finish_reason,
                        # ASCII escapes keep rows intact for the reconstruct
                        # stage, which reads with str.splitlines.
                        }, ensure_ascii=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    print(f"[paraphrase {shard}] {task}/{arm} "
                          f"{min(start + len(chunk), len(pending))}/{len(pending)}", flush=True)


def random_vectors(args) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    task_dirs = task_dirs_from_args(args)
    for task_number, task in enumerate(TASKS):
        directory = arm_dir(args.arms_dir, task, "random_vec") / "reconstructions"
        directory.mkdir(parents=True, exist_ok=True)
        for shard in range(common.NUM_SHARDS):
            destination = directory / f"shard_{shard:02d}.safetensors"
            if destination.exists():
                continue
            native = load_file(str(task_dirs[task] / "activations" / f"shard_{shard:02d}.safetensors"))["activations"].float()
            generator = torch.Generator().manual_seed(args.seed + 1009 * (task_number + 1) + shard)
            noise = torch.randn(native.shape, generator=generator)
            noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            vectors = noise * native.norm(dim=-1, keepdim=True)
            temporary = destination.with_suffix(".tmp.safetensors")
            save_file({"reconstructions": vectors.bfloat16().contiguous()}, str(temporary))
            os.replace(temporary, destination)
            print(f"WROTE {destination} {tuple(vectors.shape)}", flush=True)


def correctness(task: str, rows: dict[int, dict], scorer) -> dict[str, list[bool]]:
    """Return per-example correctness lists (dataset order) for each metric."""
    if task == "gsm8k":
        return {"accuracy": [bool(rows[i]["correct"]) for i in range(len(rows))]}
    if task == "arc_easy":
        dataset = scorer["dataset"]
        arc = scorer["module"]
        flags = []
        for i in range(len(dataset)):
            labels = [str(x).upper() for x in dataset[i]["choices"]["label"]]
            flags.append(arc.predicted_label(rows[i]["output"], labels)
                         == arc.canonical_choice(str(dataset[i]["answerKey"]), labels))
        return {"accuracy": flags}
    dataset = scorer["dataset"]
    responses = {int(rows[i]["key"]): rows[i]["output"] for i in range(len(rows))}
    scored = scorer["module"].score_rows([dict(dataset[i]) for i in range(len(dataset))], responses)
    by_key = {int(row["key"]): row for row in scored["rows"]}
    ordered = [by_key[int(dataset[i]["key"])] for i in range(len(dataset))]
    instruction_flags = [flag for row in ordered for flag in row["strict_follow_instruction_list"]]
    return {
        "prompt_strict": [bool(row["strict_follow_all_instructions"]) for row in ordered],
        "instruction_strict": [bool(flag) for flag in instruction_flags],
    }


def scorer_for(task: str):
    """Dataset and scoring module for ``correctness`` (None for GSM8K)."""
    if task == "ifeval":
        from soar_nla.benchmarks import ifeval as module
    elif task == "arc_easy":
        from soar_nla.benchmarks import arc_easy as module
    else:
        return None
    return {"module": module, "dataset": module.load_rows()}


def compare(clean: list[bool], other: list[bool]) -> dict:
    n = len(clean)
    correct = sum(other)
    accuracy = correct / n
    clean_only = sum(c and not o for c, o in zip(clean, other, strict=True))
    other_only = sum(o and not c for c, o in zip(clean, other, strict=True))
    delta = (other_only - clean_only) / n
    delta_var = ((clean_only + other_only) / n - delta * delta) / n
    return {
        "correct": correct, "n": n, "accuracy": accuracy,
        "se": math.sqrt(accuracy * (1 - accuracy) / n),
        "delta_vs_clean": delta, "delta_se": math.sqrt(max(delta_var, 0.0)),
        "clean_only_correct": clean_only, "condition_only_correct": other_only,
        "two_sided_exact_mcnemar_p": common.exact_mcnemar(clean_only, other_only),
    }


def summarize(args) -> None:
    import torch
    from safetensors.torch import load_file

    task_dirs = task_dirs_from_args(args)
    source = RolloutSource(task_dirs=task_dirs, arms_dir=args.arms_dir)
    summary = {"tasks": {}}
    csv_rows = ["task,metric,condition,correct,n,accuracy,se,delta_vs_clean,delta_se,mcnemar_p,mean_cosine_to_native,tag_rate,mean_word_overlap"]
    for task in TASKS:
        scorer = scorer_for(task)
        conditions = {"clean": source.load(task, "clean"), "nla": source.load(task, "nla")}
        vector_dirs = {"nla": task_dirs[task]}
        for arm in ARMS:
            if source.available(task, arm):
                conditions[arm] = source.load(task, arm)
                vector_dirs[arm] = arm_dir(args.arms_dir, task, arm)
        flags = {name: correctness(task, rows, scorer) for name, rows in conditions.items()}

        task_summary = {}
        for name in conditions:
            cosines = []
            if name in vector_dirs:
                for shard in range(common.NUM_SHARDS):
                    native = load_file(str(task_dirs[task] / "activations" / f"shard_{shard:02d}.safetensors"))["activations"].float()
                    vectors = load_file(str(vector_dirs[name] / "reconstructions" / f"shard_{shard:02d}.safetensors"))["reconstructions"].float()
                    cosines.append(torch.nn.functional.cosine_similarity(native, vectors, dim=-1))
            paraphrase_rows = []
            if name in PARA_ARMS:
                for shard in range(common.NUM_SHARDS):
                    paraphrase_rows += read_jsonl(vector_dirs[name] / "explanations" / f"shard_{shard:02d}.jsonl")
            capped = sum(row.get("finish_reason") == "length" for row in conditions[name].values())
            entry = {
                "metrics": {metric: compare(flags["clean"][metric], values) for metric, values in flags[name].items()},
                "generation_cap_hits": capped,
                "mean_cosine_to_native": float(torch.cat(cosines).mean()) if cosines else (1.0 if name == "clean" else None),
            }
            if paraphrase_rows:
                entry["paraphrase"] = {
                    "rows": len(paraphrase_rows),
                    "tag_rate": sum(row["parse_ok"] for row in paraphrase_rows) / len(paraphrase_rows),
                    "mean_word_overlap": sum(row["word_overlap"] for row in paraphrase_rows) / len(paraphrase_rows),
                    "paraphrase_cap_hits": sum(row["finish_reason"] == "length" for row in paraphrase_rows),
                }
            task_summary[name] = entry
            for metric, result in entry["metrics"].items():
                para = entry.get("paraphrase", {})
                csv_rows.append(",".join(str(value) for value in (
                    task, metric, name, result["correct"], result["n"], result["accuracy"], result["se"],
                    result["delta_vs_clean"], result["delta_se"], result["two_sided_exact_mcnemar_p"],
                    entry["mean_cosine_to_native"], para.get("tag_rate", ""), para.get("mean_word_overlap", ""),
                )))
        summary["tasks"][task] = task_summary
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.summary_dir / "summary.json", summary)
    (args.summary_dir / "summary.csv").write_text("\n".join(csv_rows) + "\n")
    print("\n".join(csv_rows), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("setup", "paraphrase", "random", "summarize"))
    add_source_args(parser)
    parser.add_argument("--summary-dir", type=Path, default=None,
                        help="where summarize writes summary.{json,csv} (default: --arms-dir)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="paraphrase only the first N rows per shard (smoke test)")
    parser.add_argument("--request-batch-size", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.rollouts is not None:
        parser.error("--rollouts is not supported here; the stages need the run directories")
    args.summary_dir = args.summary_dir or args.arms_dir
    args.arms_dir.mkdir(parents=True, exist_ok=True)
    {"setup": setup, "paraphrase": paraphrase, "random": random_vectors, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
