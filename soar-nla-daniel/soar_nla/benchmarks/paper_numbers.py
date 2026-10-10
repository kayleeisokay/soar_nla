"""Recompute the benchmark accuracies of the paper figure from per-example rollouts.

For every task and condition this rescores the generated text (GSM8K: last
number against the dataset reference, cross-checked against the stored
``correct`` flag; ARC-Easy: answer letter; IFEval: Google's strict checkers,
mean of prompt- and instruction-level accuracy) and compares the result with
the value shown in the paper figure, in percent to two decimals.

``--export DIR`` additionally writes each rollout as
``DIR/<task>__<condition>.jsonl.gz`` (the files committed under
``results/benchmarks/rollouts``); ``--rollouts DIR`` reads them back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from soar_nla.benchmarks.rollouts import CONDITIONS, add_source_args, source_from_args

# Accuracy (%) in the paper figure (allpos_accuracy_bars_legend). ARC-Easy and
# GSM8K: exact accuracy; IFEval: mean of strict prompt- and instruction-level
# accuracy.
PAPER = {
    "arc_easy": {"clean": 96.93, "nla": 96.55, "para_medium": 92.80, "para_heavy": 91.62, "random_vec": 0.00},
    "gsm8k": {"clean": 94.16, "nla": 86.58, "para_medium": 81.73, "para_heavy": 76.50, "random_vec": 9.25},
    "ifeval": {"clean": 85.29, "nla": 62.31, "para_medium": 53.54, "para_heavy": 50.88, "random_vec": 21.74},
}


def gsm8k_accuracy(rows: dict[int, dict], dataset) -> float:
    from soar_nla.benchmarks.gsm8k import predicted_answer, reference_answer

    if sorted(rows) != list(range(len(dataset))):
        raise RuntimeError("GSM8K rollout does not cover the test set exactly once")
    correct = 0
    for index in range(len(dataset)):
        flag = predicted_answer(rows[index]["output"]) == reference_answer(dataset[index]["answer"])
        if flag != bool(rows[index]["correct"]):
            raise RuntimeError(f"GSM8K index {index}: rescored correctness disagrees with the stored flag")
        correct += flag
    return correct / len(dataset)


def arc_accuracy(rows: dict[int, dict], dataset: list[dict]) -> float:
    from soar_nla.benchmarks.arc_easy import is_correct

    if sorted(rows) != list(range(len(dataset))):
        raise RuntimeError("ARC-Easy rollout does not cover the test set exactly once")
    return sum(is_correct(dataset[i], rows[i]["output"]) for i in range(len(dataset))) / len(dataset)


def ifeval_strict_average(rows: dict[int, dict], dataset) -> float:
    from soar_nla.benchmarks.ifeval import score_rows

    inputs = [dict(dataset[i]) for i in range(len(dataset))]
    responses = {int(row["key"]): row["output"] for row in rows.values()}
    strict = score_rows(inputs, responses)["summary"]["strict"]
    return 0.5 * (strict["prompt_accuracy"] + strict["instruction_accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(parser)
    parser.add_argument("--export", type=Path, default=None,
                        help="also write <task>__<condition>.jsonl.gz rollouts here")
    parser.add_argument("--tasks", nargs="+", default=list(PAPER), choices=list(PAPER))
    args = parser.parse_args()
    if args.export is not None and args.rollouts is not None:
        parser.error("--export reads run directories; drop --rollouts")
    source = source_from_args(args)

    from soar_nla.benchmarks import arc_easy, common

    failures = []
    print(f"{'task':<9} {'condition':<12} {'recomputed':>10} {'paper':>7}")
    for task in args.tasks:
        if task == "gsm8k":
            dataset = common.load_benchmark("gsm8k")
            metric = gsm8k_accuracy
        elif task == "arc_easy":
            dataset = arc_easy.load_rows()
            metric = arc_accuracy
        else:
            dataset = common.load_benchmark("ifeval")
            metric = ifeval_strict_average
        for condition in CONDITIONS:
            rows = source.load(task, condition)
            if args.export is not None:
                source.export(task, condition, args.export)
            value = round(100 * metric(rows, dataset), 2)
            expected = PAPER[task][condition]
            status = "ok" if abs(value - expected) < 1e-9 else "MISMATCH"
            if status != "ok":
                failures.append((task, condition, value, expected))
            print(f"{task:<9} {condition:<12} {value:>10.2f} {expected:>7.2f}  {status}", flush=True)
    if failures:
        print(f"{len(failures)} value(s) differ from the paper", file=sys.stderr)
        sys.exit(1)
    print("all values match the paper figure")


if __name__ == "__main__":
    main()
