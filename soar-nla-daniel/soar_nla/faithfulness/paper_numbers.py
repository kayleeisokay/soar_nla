"""Recompute the Faithfulness section's numbers and check them against the paper.

FVE: the four-arm table at both checkpoint pairs, from the fve.py run directories
(<fve-dir>/{sft,rl}/{dataset,extract,reconstruct}). The paper reports the strict-clean rows:
the 2,000-row eval set minus the 3 documents whose text also appears in the custom NLA's
training data (contamination.py strict), i.e. 1,997 rows / 1,850 documents. On all 2,000
rows the SFT `av` FVE is 0.4825 (48.3%, vs 48.2% strict); the other paper values round the same.

Sentence judge: the Supported / Unsupported / Illegible table over the first two paragraphs
of each explanation (sentence_judge.py aggregate), and the GCG section's comparison of the
official AV explanation with its beta=0.01 GCG rewrite on the same scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from soar_nla.io_utils import write_json
from soar_nla.paths import results_dir

from .config import ARMS, PROTOCOL_VERSION
from .fve import load_cosines
from .metrics import evaluate, unit

STAGES = ("sft", "rl")

# Paper values, in percent.
PAPER_FVE = {
    ("sft", "gold"): 60.8,
    ("sft", "av"): 48.2,
    ("rl", "av"): 70.1,
    ("rl", "gold"): 53.8,
    ("rl", "av_shuffled"): -74.8,
    ("rl", "gold_shuffled"): -65.6,
}
PAPER_JUDGE = {
    ("custom_sonnet", "Supported"): 73.1,
    ("custom_sonnet", "Unsupported"): 26.6,
    ("custom_sonnet", "Illegible"): 0.3,
    ("custom_av", "Supported"): 6.3,
    ("custom_av", "Unsupported"): 93.3,
    ("custom_av", "Illegible"): 0.4,
    # GCG section: "14.3% supported compared to 17.5%".
    ("alpaca_av", "Supported"): 17.5,
    ("alpaca_gcg_beta_0.01", "Supported"): 14.3,
}
ARM_NAMES = {"av": "AV", "gold": "Sonnet", "gold_shuffled": "random Sonnet", "av_shuffled": "random AV"}


def fve_tables(fve_dir: Path, excluded: set[str], resamples: int) -> dict:
    tables: dict = {}
    doc_lists = []
    for stage in STAGES:
        rows, gold, cosine = load_cosines(fve_dir / stage)
        doc_key = [row["doc_id"] for row in rows]
        doc_lists.append(doc_key)
        keep = torch.tensor([doc not in excluded for doc in doc_key])
        tables[stage] = {
            # Seeds as in the original analysis, so the bootstrap intervals reproduce too.
            "strict_clean": evaluate(keep, doc_key, unit(gold), cosine, resamples, f"strict:{stage}", arms=ARMS),
            "all_rows": evaluate(torch.ones(len(rows), dtype=torch.bool), doc_key, unit(gold), cosine,
                                 resamples, f"{PROTOCOL_VERSION}:easynla:{stage}", arms=ARMS),
        }
    if doc_lists[0] != doc_lists[1]:
        raise RuntimeError("the sft and rl runs were built on different eval rows")
    return tables


def _fve_markdown(tables: dict) -> str:
    strict = {stage: tables[stage]["strict_clean"] for stage in STAGES}
    head = strict["rl"]
    lines = [f"Strict clean subset: {head['n_rows']} rows / {head['n_docs']} docs.", "",
             "| arm | SFT FVE | 95% CI | RL FVE | 95% CI |", "|---|---|---|---|---|"]
    for arm in ARMS:
        cells = []
        for stage in STAGES:
            value = strict[stage]["arms"][arm]
            low, high = value["fve_ci95"]
            cells += [f"{value['fve']:.4f}", f"[{low:.4f}, {high:.4f}]"]
        lines.append(f"| `{arm}` ({ARM_NAMES[arm]}) | " + " | ".join(cells) + " |")
    cells = []
    for stage in STAGES:
        delta = strict[stage]["gold_minus_av"]
        low, high = delta["delta_fve_ci95"]
        cells += [f"{delta['delta_fve']:+.4f}", f"[{low:+.4f}, {high:+.4f}]"]
    lines.append("| **gold − av** | " + " | ".join(cells) + " |")
    wins = [strict[stage]["gold_minus_av"] for stage in STAGES]
    lines.append("| gold better on | " + " | ".join(f"{w['rows_gold_better']}/{w['rows_total']} | " for w in wins) + "|")
    return "\n".join(lines) + "\n"


def _check(name: str, paper: float, value: float) -> dict:
    ok = f"{value:.1f}" == f"{paper:.1f}"
    print(f"{'OK ' if ok else 'BAD'} {name:<44} paper {paper:>6.1f}   recomputed {value:>8.3f}")
    return {"claim": name, "paper": paper, "recomputed": value, "match": ok}


def main() -> None:
    root = results_dir("faithfulness")
    parser = argparse.ArgumentParser(prog="soar_nla.faithfulness.paper_numbers", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fve-dir", type=Path, default=root / "fve",
                        help="fve.py --results-dir (holds sft/ and rl/)")
    parser.add_argument("--strict-exclude", type=Path, default=root / "strict_clean_excluded_docs.json")
    parser.add_argument("--judge-results", type=Path, default=root / "sentence_labels.jsonl.gz",
                        help="judge output (results.jsonl) or the slim sentence_labels.jsonl.gz")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=root)
    args = parser.parse_args()

    excluded = set(json.loads(args.strict_exclude.read_text())["excluded_doc_ids"])
    # The FVE check needs the pipeline's activations and reconstructions, which are not
    # shipped; without them only the sentence-judge numbers are recomputed.
    have_fve = all((args.fve_dir / stage / "reconstruct" / "reconstructions.safetensors").exists()
                   for stage in STAGES)
    if have_fve:
        tables = fve_tables(args.fve_dir, excluded, args.resamples)
        write_json(args.output_dir / "fve_table.json", {
            "protocol": PROTOCOL_VERSION, "strict_exclude": args.strict_exclude.name,
            "excluded_doc_ids": sorted(excluded), "stages": tables,
        })
        (args.output_dir / "fve_table.md").write_text(_fve_markdown(tables), encoding="utf-8")
        print(_fve_markdown(tables))
    else:
        print(f"no reconstructions under {args.fve_dir}; skipping the FVE checks (run run_fve.sh first)\n")

    from .sentence_judge import PAPER_SCOPE, _read_rows, label_paragraphs, tabulate

    judge = tabulate(label_paragraphs(_read_rows(args.judge_results)))[PAPER_SCOPE]

    checks = []
    for (stage, arm), paper in PAPER_FVE.items() if have_fve else ():
        value = 100 * tables[stage]["strict_clean"]["arms"][arm]["fve"]
        checks.append(_check(f"FVE {stage.upper()} {ARM_NAMES[arm]}", paper, value))
    for (arm, label), paper in PAPER_JUDGE.items():
        value = judge[arm]["percent"][label]
        checks.append(_check(f"judge {arm} {label} (n={judge[arm]['n_sentences']})", paper, value))
    if have_fve:
        write_json(args.output_dir / "paper_numbers.json", {"checks": checks})
    if not all(check["match"] for check in checks):
        raise SystemExit("some paper numbers did not reproduce")


if __name__ == "__main__":
    main()
