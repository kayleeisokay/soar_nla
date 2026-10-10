"""Recompute every number the paper quotes for ICL edits and task switching from the per-row results.

  python -m soar_nla.icl_edits.paper_numbers [--results ROOT] [--out FILE]

ROOT holds one directory per config (gemma3-12b, gemma3-12b-custom-L24, gemma3-12b-custom-L40)
with the icl_*.json (or .json.gz) files written by `run exp`. Each claim is checked against the value
as printed in the paper, within half a unit of its last printed digit; claims the paper states only
qualitatively ("nearly ineffective") are listed with their value and no check.
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from soar_nla.paths import results_dir

from .analysis import load_icl

L32, L24, L40 = "gemma3-12b", "gemma3-12b-custom-L24", "gemma3-12b-custom-L40"
EARLY = (0, 4, 8, 12, 16, 20, 24)

# (paper section, claim, config, experiment, format, metric, value as printed in the paper)
# metric: ("rate", span, arm[, task]) | ("sweep", span, layer) | ("sweep_range", span, layers)
#         | ("max_rate", spans, arms)
CLAIMS = [
    ("ICL edits", "raw L32, final token: query-and-answer substitution", L32, "redirect", "raw", ("rate", "last1", "substitute_only"), "64%"),
    ("ICL edits", "raw L32, final token: final-paragraph rewrite", L32, "redirect", "raw", ("rate", "last1", "rewrite_only"), "46%"),
    ("ICL edits", "raw L32, final token: combined edit", L32, "redirect", "raw", ("rate", "last1", "edited"), "79%"),
    ("ICL edits", "raw L32, final token: donor", L32, "redirect", "raw", ("rate", "last1", "donor"), "81%"),
    ("ICL edits", "raw L32, final token: Capitalize combined edit (\"remains weak\")", L32, "redirect", "raw", ("rate", "last1", "edited", "capitalize"), None),
    ("ICL edits", "raw L32, final token: Capitalize donor (\"remains weak\")", L32, "redirect", "raw", ("rate", "last1", "donor", "capitalize"), None),
    ("ICL edits", "raw L32, four tokens: query-and-answer substitution", L32, "redirect", "raw", ("rate", "last4", "substitute_only"), "98%"),
    ("ICL edits", "raw L32, four tokens: combined edit", L32, "redirect", "raw", ("rate", "last4", "edited"), "98%"),
    ("ICL edits", "raw L32, four tokens: final-unit rewrite", L32, "redirect", "raw", ("rate", "last4", "rewrite_only"), "59%"),
    ("ICL edits", "raw L32, four tokens: donor", L32, "redirect", "raw", ("rate", "last4", "donor"), "97%"),

    ("Appendix: donor patching", "query token, layers 0-24", L32, "redirect", "raw", ("sweep_range", "query", EARLY), "94-98%"),
    ("Appendix: donor patching", "query token, layer 28", L32, "redirect", "raw", ("sweep", "query", 28), "47%"),
    ("Appendix: donor patching", "query token, layer 32", L32, "redirect", "raw", ("sweep", "query", 32), "7%"),
    ("Appendix: donor patching", "final ':' token, layers 0-24 (\"no effect\")", L32, "redirect", "raw", ("sweep_range", "last1", EARLY), "0-0%"),
    ("Appendix: donor patching", "final ':' token, layer 28", L32, "redirect", "raw", ("sweep", "last1", 28), "38%"),
    ("Appendix: donor patching", "final ':' token, layer 32", L32, "redirect", "raw", ("sweep", "last1", 32), "78%"),
    ("Appendix: donor patching", "final ':' token, layer 40", L32, "redirect", "raw", ("sweep", "last1", 40), "98%"),
    ("Appendix: donor patching", "last four tokens, layers 0-40", L32, "redirect", "raw", ("sweep_range", "last4", EARLY + (28, 32, 36, 40)), "94-98%"),

    ("Appendix: other layers", "L24, query token: substitution", L24, "redirect", "raw", ("rate", "query", "substitute_only"), "96%"),
    ("Appendix: other layers", "L24, query token: donor", L24, "redirect", "raw", ("rate", "query", "donor"), "97%"),
    ("Appendix: other layers", "L24, query token: combined edit", L24, "redirect", "raw", ("rate", "query", "edited"), "88%"),
    ("Appendix: other layers", "L24, query token: rewrite only", L24, "redirect", "raw", ("rate", "query", "rewrite_only"), "19%"),
    ("Appendix: other layers", "L24, four tokens: substitution", L24, "redirect", "raw", ("rate", "last4", "substitute_only"), "92%"),
    ("Appendix: other layers", "L24, four tokens: combined edit", L24, "redirect", "raw", ("rate", "last4", "edited"), "81%"),
    ("Appendix: other layers", "L24, four tokens: rewrite only", L24, "redirect", "raw", ("rate", "last4", "rewrite_only"), "17%"),
    ("Appendix: other layers", "L24, four tokens: donor", L24, "redirect", "raw", ("rate", "last4", "donor"), None),
    ("Appendix: other layers", "L24, final token: combined edit (\"nearly ineffective\")", L24, "redirect", "raw", ("rate", "last1", "edited"), None),
    ("Appendix: other layers", "L40, query token: any edit (\"no target answers\")", L40, "redirect", "raw", ("max_rate", ("query",), ("substitute_only", "rewrite_only", "edited")), "0%"),
    ("Appendix: other layers", "L40, final token: substitution", L40, "redirect", "raw", ("rate", "last1", "substitute_only"), "70%"),
    ("Appendix: other layers", "L40, final token: rewrite only", L40, "redirect", "raw", ("rate", "last1", "rewrite_only"), "58%"),
    ("Appendix: other layers", "L40, final token: combined edit", L40, "redirect", "raw", ("rate", "last1", "edited"), "70%"),
    ("Appendix: other layers", "L40, final token: donor", L40, "redirect", "raw", ("rate", "last1", "donor"), "97%"),
    ("Appendix: other layers", "L40, four tokens: substitution", L40, "redirect", "raw", ("rate", "last4", "substitute_only"), "87%"),
    ("Appendix: other layers", "L40, four tokens: rewrite only", L40, "redirect", "raw", ("rate", "last4", "rewrite_only"), "87%"),
    ("Appendix: other layers", "L40, four tokens: combined edit", L40, "redirect", "raw", ("rate", "last4", "edited"), "92%"),
    ("Appendix: other layers", "L40, four tokens: donor", L40, "redirect", "raw", ("rate", "last4", "donor"), "98%"),
    ("Appendix: other layers", "L40, final token: Capitalize combined edit", L40, "redirect", "raw", ("rate", "last1", "edited", "capitalize"), "40%"),
    ("Appendix: other layers", "L40, final token: Capitalize donor", L40, "redirect", "raw", ("rate", "last1", "donor", "capitalize"), "100%"),
    ("Appendix: other layers", "L40, four tokens: Capitalize combined edit", L40, "redirect", "raw", ("rate", "last4", "edited", "capitalize"), "92%"),

    ("Appendix: chat template", "chat L32, final token: combined edit", L32, "redirect", "chat", ("rate", "last1", "edited"), "72%"),
    ("Appendix: chat template", "chat L32, final token: donor", L32, "redirect", "chat", ("rate", "last1", "donor"), "8.7%"),
    ("Appendix: chat template", "chat L32, final token: substitution", L32, "redirect", "chat", ("rate", "last1", "substitute_only"), "48%"),
    ("Appendix: chat template", "chat L32, four tokens: combined edit", L32, "redirect", "chat", ("rate", "last4", "edited"), "92%"),
    ("Appendix: chat template", "chat L32, four tokens: donor", L32, "redirect", "chat", ("rate", "last4", "donor"), "84%"),
    ("Appendix: chat template", "chat L32, four tokens: substitution", L32, "redirect", "chat", ("rate", "last4", "substitute_only"), "79%"),

    ("Task switching", "raw L32, final token: task-and-answer substitution", L32, "switch", "raw", ("rate", "last1", "answer+task"), "0.440"),
    ("Task switching", "raw L32, final token: combined edit", L32, "switch", "raw", ("rate", "last1", "edited"), "0.847"),
    ("Task switching", "raw L32, final token: donor", L32, "switch", "raw", ("rate", "last1", "donor"), "0.840"),
    ("Task switching", "raw L32, four tokens: task-and-answer substitution", L32, "switch", "raw", ("rate", "last4", "answer+task"), "0.470"),
    ("Task switching", "raw L32, four tokens: combined edit", L32, "switch", "raw", ("rate", "last4", "edited"), "0.877"),
    ("Task switching", "raw L32, four tokens: donor", L32, "switch", "raw", ("rate", "last4", "donor"), "0.897"),
    ("Task switching", "raw L32, final token: answer-only substitution", L32, "switch", "raw", ("rate", "last1", "answer_only"), "0.057"),
    ("Task switching", "raw L32, four tokens: answer-only substitution", L32, "switch", "raw", ("rate", "last4", "answer_only"), "0.057"),
    ("Task switching", "raw L32, final token: task-only substitution", L32, "switch", "raw", ("rate", "last1", "task_only"), "0.003"),
    ("Task switching", "raw L32, four tokens: task-only substitution", L32, "switch", "raw", ("rate", "last4", "task_only"), "0.003"),
    ("Task switching", "raw L32: round-trip and no-op controls, both spans", L32, "switch", "raw", ("max_rate", ("last1", "last4"), ("roundtrip", "noop")), "0.000"),
    ("Task switching appendix: chat", "chat L32, final token: task-and-answer substitution", L32, "switch", "chat", ("rate", "last1", "answer+task"), "0.463"),
    ("Task switching appendix: chat", "chat L32, final token: combined edit", L32, "switch", "chat", ("rate", "last1", "edited"), "0.950"),
    ("Task switching appendix: chat", "chat L32, final token: donor", L32, "switch", "chat", ("rate", "last1", "donor"), "0.517"),
    ("Task switching appendix: chat", "chat L32, four tokens: task-and-answer substitution", L32, "switch", "chat", ("rate", "last4", "answer+task"), "0.543"),
    ("Task switching appendix: chat", "chat L32, four tokens: combined edit", L32, "switch", "chat", ("rate", "last4", "edited"), "0.983"),
    ("Task switching appendix: chat", "chat L32, four tokens: donor", L32, "switch", "chat", ("rate", "last4", "donor"), "0.783"),
    ("Task switching appendix: chat", "chat L32: round-trip and no-op controls, both spans", L32, "switch", "chat", ("max_rate", ("last1", "last4"), ("roundtrip", "noop")), "0.000"),
]


def _mean(xs):
    xs = list(xs)
    assert xs, "no rows"
    return sum(map(float, xs)) / len(xs)


def compute(metric, rows, sweep):
    kind = metric[0]
    if kind == "rate":
        _, span, arm, *task = metric
        rr = [r for r in rows if not task or r["task"] == task[0]]
        return _mean(r[f"{span}|{arm}|hit"] for r in rr)
    if kind == "max_rate":
        _, spans, arms = metric
        return max(_mean(r[f"{sp}|{a}|hit"] for r in rows) for sp in spans for a in arms)
    if kind == "sweep":
        _, span, layer = metric
        return _mean(r["hit"] for r in sweep if r["span"] == span and r["layer"] == layer)
    if kind == "sweep_range":
        _, span, layers = metric
        vals = [_mean(r["hit"] for r in sweep if r["span"] == span and r["layer"] == L) for L in layers]
        return (min(vals), max(vals))
    raise ValueError(kind)


def _parse(printed: str) -> tuple[list[float], float]:
    """Printed value(s) as fractions, plus half a unit of the last printed digit."""
    pct = printed.endswith("%")
    parts = printed.rstrip("%").split("-")
    scale = Decimal(100) if pct else Decimal(1)
    unit = max(Decimal(1).scaleb(Decimal(p).as_tuple().exponent) for p in parts) / scale
    return [float(Decimal(p) / scale) for p in parts], float(unit) / 2


def check(value, printed):
    if printed is None:
        return None
    want, tol = _parse(printed)
    got = list(value) if isinstance(value, tuple) else [value]
    return len(got) == len(want) and all(abs(g - w) <= tol + 1e-9 for g, w in zip(got, want))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="soar_nla.icl_edits.paper_numbers")
    ap.add_argument("--results", type=Path, default=results_dir("icl_edits"),
                    help="root holding one directory per config")
    ap.add_argument("--out", type=Path, default=None, help="JSON output (default: <results>/paper_numbers.json)")
    a = ap.parse_args(argv)
    cache, out, bad = {}, [], 0
    for section, claim, cfg, kind, fmt, metric, printed in CLAIMS:
        key = (cfg, kind, fmt)
        if key not in cache:
            cache[key] = load_icl(a.results / cfg, kind, fmt)
        value = compute(metric, *cache[key])
        ok = check(value, printed)
        bad += ok is False
        shown = (f"{value[0]:.4f}-{value[1]:.4f}" if isinstance(value, tuple) else f"{value:.4f}")
        print(f"{'OK  ' if ok else ('    ' if ok is None else 'FAIL')} {section:32s} {claim:68s} {shown:>13s}  "
              f"paper {printed or '-'}")
        out.append(dict(section=section, claim=claim, config=cfg, experiment=f"icl_{kind}", format=fmt,
                        metric=list(metric), value=value, paper=printed, matches=ok))
    n = sum(r["matches"] is not None for r in out)
    print(f"\n{n - bad}/{n} quoted numbers reproduced ({len(out) - n} qualitative claims listed without a check)")
    path = a.out or a.results / "paper_numbers.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"saved {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
