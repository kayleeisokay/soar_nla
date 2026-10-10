"""The three explanation edits for one saved ICL item (the paper's "Example explanation edits" figure).

  python -m soar_nla.icl_edits.example_edits [--results DIR] [--task country-capital --idx 0] [--seeds 0 1 2]

CPU only. Reads the saved final-token AV explanation of the item from the raw-text layer-32 run and
re-applies the edits from edits.py: query-and-answer substitution (rewrite=False), final-paragraph
rewrite (substitute=False) and the combined edit, which must equal the `edited` text saved by the
run. Also lists the answers the model gave under each arm. Default item: country-capital idx 0,
China -> Beijing redirected to Chile -> Santiago.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from soar_nla.paths import results_dir

from . import tasks as D
from .edits import icl_edit
from .jsonio import read_json

ARMS = [("substitute_only", "Query-and-answer substitution"), ("rewrite_only", "Final-paragraph rewrite"),
        ("edited", "Combined edit"), ("donor", "Donor patching")]


def example(rows: list[dict], task: str, idx: int, seeds: list[int]) -> str:
    item = D.redirect_items(task)[idx]
    lines = [f"# Example explanation edits: {task} item {idx}", "",
             f"Base query {item['query']!r} (answer {item['gold']!r}) -> target query "
             f"{item['target_query']!r} (answer {item['target']!r}). Raw-text prompt, layer-32 NLA; the "
             "explanation is the AV's account of the final prompt token (`:`).", ""]
    by_seed = {r["seed"]: r for r in rows if r["task"] == task and r["idx"] == idx}
    for sd in seeds:
        r = by_seed[sd]
        assert (r["query"], r["gold"], r["target"]) == (item["query"], item["gold"], item["target"])
        texts = {"original": r["explanation"],
                 "substitute_only": icl_edit(r["explanation"], item, rewrite=False),
                 "rewrite_only": icl_edit(r["explanation"], item, substitute=False),
                 "edited": icl_edit(r["explanation"], item)}
        assert texts["edited"] == r["edited"], f"seed {sd}: combined edit differs from the saved text"
        lines += [f"## Seed {sd}", "", f"Unpatched answer: {r['baseline']!r}", "",
                  "| arm | answer, final token | answer, last four tokens |", "|---|---|---|"]
        lines += [f"| {name} | {r[f'last1|{arm}']!r} | {r[f'last4|{arm}']!r} |" for arm, name in ARMS]
        lines.append("")
        for key, name in [("original", "Original explanation"), *ARMS[:3]]:
            lines += [f"### {name}", "", "```", texts[key], "```", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="soar_nla.icl_edits.example_edits")
    ap.add_argument("--results", type=Path, default=results_dir("icl_edits") / "gemma3-12b",
                    help="results directory of the layer-32 config")
    ap.add_argument("--file", default="icl_redirect_raw-a.json", help="per-row file holding the task")
    ap.add_argument("--task", default="country-capital", choices=D.TASKS)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", type=Path, default=None,
                    help="markdown output (default: <results>/../figures/example_edits_<task>.md)")
    a = ap.parse_args(argv)
    text = example(read_json(a.results / a.file)["rows"], a.task, a.idx, a.seeds)
    out = a.out or a.results.parent / "figures" / f"example_edits_{a.task}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
