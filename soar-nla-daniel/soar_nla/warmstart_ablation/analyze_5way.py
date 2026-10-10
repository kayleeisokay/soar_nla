#!/usr/bin/env python3
"""Summarise judge_5way output: accuracy vs 20% chance, real vs control."""
import argparse, gzip, json, math, collections
from pathlib import Path


def load(p):
    """Read a judge_5way JSONL (optionally .gz); missing/None path -> []."""
    if not p or not Path(p).exists():
        return []
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def agg(rows):
    # [n_correct, n, sum p_gold, n p_gold finite, sum margin, sum 1/n_options]
    d = collections.defaultdict(lambda: [0, 0, 0.0, 0, 0.0, 0.0])
    for r in rows:
        if r.get("pick_letter") is None:
            continue
        key = (r["run"], r["attr_class"])
        d[key][0] += bool(r["correct"])
        d[key][1] += 1
        pg = r["p_gold"]
        if pg == pg:                       # gold letter present in top-k logprobs
            d[key][2] += pg; d[key][3] += 1
        d[key][4] += r["margin"]
        d[key][5] += 1.0 / len(r["options"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--control")
    ap.add_argument("--out-json", default=None,
                    help="optionally also write per-run/class accuracies as JSON")
    a = ap.parse_args()

    real, ctrl = load(a.real), load(a.control)
    R, C = agg(real), agg(ctrl)
    runs = sorted({k[0] for k in R})
    classes = ["Gender", "Occupation", "Country", "Religion"]

    print("\n=== forced choice: real accuracy [95% CI], control, chance ===\n")
    hdr = f"{'class':<12}{'n':>6}" + "".join(f"{r:>32}" for r in runs)
    print(hdr)
    print("-" * len(hdr))
    for cls in classes + ["POOLED"]:
        cells, n = [], 0
        for run in runs:
            if cls == "POOLED":
                k = sum(R[(run, c)][0] for c in classes)
                m = sum(R[(run, c)][1] for c in classes)
                ch = sum(R[(run, c)][5] for c in classes)
                kc = sum(C[(run, c)][0] for c in classes) if C else 0
                mc = sum(C[(run, c)][1] for c in classes) if C else 0
            else:
                k, m = R[(run, cls)][0], R[(run, cls)][1]
                kc, mc = (C[(run, cls)][0], C[(run, cls)][1]) if C else (0, 0)
            if cls != "POOLED":
                ch = R[(run, cls)][5]
            n = max(n, m)
            s = f"{k/m:6.1%} [{wilson(k,m)[0]:.0%},{wilson(k,m)[1]:.0%}]" if m else "     -"
            if mc:
                s += f" ctl{kc/mc:5.1%}"
            if m:
                s += f" ch{ch/m:5.1%}"
            cells.append(s)
        print(f"{cls:<12}{n:>6}" + "".join(f"{c:>32}" for c in cells))

    print("\n=== mean P(gold) / mean margin, real arm ===\n")
    print(f"{'class':<12}" + "".join(f"{r:>20}" for r in runs))
    print("(P(gold) averaged over items where gold was inside the top-20 logprobs)")
    for cls in classes:
        cells = []
        for run in runs:
            k, m, pg, npg, mg, _ = R[(run, cls)]
            cells.append(f"{pg/npg:.3f} / {mg/m:.3f}" if npg else "-")
        print(f"{cls:<12}" + "".join(f"{c:>20}" for c in cells))

    if a.out_json:
        summary = {}
        for run in runs:
            summary[run] = {}
            for cls in classes + ["POOLED"]:
                keys = classes if cls == "POOLED" else [cls]
                k = sum(R[(run, c)][0] for c in keys); m = sum(R[(run, c)][1] for c in keys)
                ch = sum(R[(run, c)][5] for c in keys)
                kc = sum(C[(run, c)][0] for c in keys) if C else 0
                mc = sum(C[(run, c)][1] for c in keys) if C else 0
                summary[run][cls] = {
                    "n": m, "correct": k, "accuracy": k / m if m else None,
                    "wilson95": list(wilson(k, m)) if m else None,
                    "control_n": mc, "control_correct": kc,
                    "control_accuracy": kc / mc if mc else None,
                    "chance": ch / m if m else None,
                }
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[json] -> {a.out_json}")

    pos = collections.Counter(r["gold_letter"] for r in real)
    pick = collections.Counter(r["pick_letter"] for r in real if r["pick_letter"])
    print("\ngold letter dist :", dict(sorted(pos.items())))
    print("pick letter dist :", dict(sorted(pick.items())))

    print("\n=== Gender: recall by true value (real arm) ===\n")
    g = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in real:
        if r["attr_class"] == "Gender" and r["pick_letter"]:
            c = g[r["run"]][r["attr"]]
            c[0] += bool(r["correct"]); c[1] += 1
    vals = sorted({v for run in g for v in g[run]})
    print(f"{'value':<14}" + "".join(f"{r:>16}" for r in runs))
    for v in vals:
        print(f"{v:<14}" + "".join(
            f"{g[r][v][0]/g[r][v][1]:>10.1%} ({g[r][v][1]:>3})" if g[r][v][1] else f"{'-':>16}"
            for r in runs))


if __name__ == "__main__":
    main()
