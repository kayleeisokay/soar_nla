"""Summaries (with item-bootstrap intervals) for one config's results directory.

  python -m soar_nla.icl_edits.analysis --config gemma3-12b [--results DIR]

Reads icl_{redirect,switch}[_raw][-tag].json (plain or .gz) and writes summary.json next to them:
per format/experiment, the target rate of every arm at every span with a 95% item-bootstrap
interval, per-task rates, and the donor layer sweep.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .config import load_config
from .jsonio import read_json, result_files
from .stats import cluster_ci

REDIRECT_ARMS = ["edited", "donor", "roundtrip", "noop", "substitute_only", "rewrite_only", "template"]
SWITCH_ARMS = ["edited", "donor", "roundtrip", "noop", "answer_only", "task_only", "answer+task"]


def load_icl(res: Path, kind: str, fmt: str = "chat"):
    rows, sweep = [], []
    stem = f"icl_{kind}" + ("" if fmt == "chat" else f"_{fmt}")
    for f in result_files(res, stem):
        d = read_json(f); rows += d["rows"]; sweep += d["sweep"]
    return rows, sweep


def _item(r):
    return (r.get("task") or r.get("case"), r.get("idx"))


def summarize(res: Path, name: str, layer: int) -> dict:
    S = {"model": name, "layer": layer}
    for fmt, (kind, arms) in [(f, k) for f in ("chat", "raw") for k in (
            ("redirect", REDIRECT_ARMS), ("switch", SWITCH_ARMS))]:
        rows, sweep = load_icl(res, kind, fmt)
        if not rows:
            continue
        spans = [sp for sp in ("last1", "last4", "query+last4", "query") if f"{sp}|edited|hit" in rows[0]]
        out = {"n": len(rows), "format": fmt, "native_gold": cluster_ci(rows, _item, lambda r: r["baseline|gold"])}
        for sp in spans:
            out[sp] = {a: cluster_ci(rows, _item, lambda r, k=f"{sp}|{a}|hit": r[k]) for a in arms}
            out[sp]["roundtrip_keeps_gold"] = cluster_ci(rows, _item, lambda r, k=f"{sp}|roundtrip|gold": r[k])
            per_item = defaultdict(int)
            for r in rows:
                per_item[_item(r)] += r[f"{sp}|edited|hit"]
            out[sp]["items_ge5"] = [sum(v >= 5 for v in per_item.values()), len(per_item)]
        out["per_task"] = {t: {sp: {a: sum(r[f"{sp}|{a}|hit"] for r in rows if r["task"] == t) /
                                     max(1, sum(r["task"] == t for r in rows)) for a in arms}
                               for sp in spans}
                           for t in dict.fromkeys(r["task"] for r in rows)}
        acc = defaultdict(list)
        for r in sweep:
            acc[(r["span"], r["layer"])].append(r["hit"])
        out["sweep"] = {f"{sp}|{L}": sum(v) / len(v) for (sp, L), v in sorted(acc.items())}
        S[f"icl_{kind}" + ("" if fmt == "chat" else f"_{fmt}")] = out
    with open(res / "summary.json", "w") as f:
        json.dump(S, f, indent=1)
    return S


def main(argv=None):
    ap = argparse.ArgumentParser(prog="soar_nla.icl_edits.analysis")
    ap.add_argument("--config", required=True, help="config name or YAML path")
    ap.add_argument("--results", type=Path, default=None,
                    help="results directory of this config (default: results_dir('icl_edits')/<name>)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    S = summarize(a.results or cfg.results, cfg.name, cfg.layer)
    for k, v in S.items():
        if isinstance(v, dict):
            print(k, {sp: {arm: round(ci[0], 4) for arm, ci in v[sp].items() if arm != "items_ge5"}
                      for sp in ("last1", "last4", "query") if sp in v})


if __name__ == "__main__":
    main()
