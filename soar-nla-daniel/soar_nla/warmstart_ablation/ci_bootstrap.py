#!/usr/bin/env python3
"""Confidence intervals for judge accuracy and paired run differences.

Three intervals, from least to most conservative:
  wilson   - per-cell Wilson score (assumes items independent)
  item     - paired bootstrap over items (same resample for A, B, C)
  cluster  - paired, class-stratified cluster bootstrap. Items sharing an
             attribute value (e.g. all 'nurse' prompts) are resampled together,
             because they are written from the same persona and are not
             independent. Gender has only 3 values, which are fixed by design,
             so Gender items are resampled within each value instead.
All bootstrap intervals are percentile intervals, B = 10,000.
"""
import argparse, gzip, json, math, collections
import numpy as np

CLASSES = ["Gender", "Occupation", "Country", "Religion"]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    for path, name in zip(a.inputs, a.names):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh]
        by = collections.defaultdict(dict)
        meta = {}
        for r in rows:
            k = (r["attr_class"], r["idx"])
            by[k][r["run"]] = bool(r["correct"])
            meta[k] = r["attr"]
        keys = sorted(by)
        X = np.array([[by[k][r] for r in "ABC"] for k in keys], dtype=float)  # items x runs
        cls = np.array([k[0] for k in keys])
        grp = np.array([f"{k[0]}|{meta[k]}" for k in keys])

        def boot(mask, cluster):
            """B x 3 matrix of resampled accuracies over the items in mask."""
            idx = np.where(mask)[0]
            out = np.zeros((a.B, 3))
            tot = np.zeros(a.B)
            strata = CLASSES if mask.sum() > 0 else []
            for c in strata:
                sidx = idx[cls[idx] == c]
                if len(sidx) == 0:
                    continue
                if cluster and c != "Gender":
                    units = [sidx[grp[sidx] == g] for g in np.unique(grp[sidx])]
                    sums = np.array([X[u].sum(0) for u in units])      # clusters x 3
                    cnts = np.array([len(u) for u in units])
                    draw = rng.integers(0, len(units), (a.B, len(units)))
                    out += sums[draw].sum(1)
                    tot += cnts[draw].sum(1)
                else:  # items, stratified by value (fixed design factor)
                    for g in np.unique(grp[sidx]):
                        u = sidx[grp[sidx] == g]
                        draw = rng.integers(0, len(u), (a.B, len(u)))
                        out += X[u][draw].sum(1)
                        tot += len(u)
            return out / tot[:, None]

        def pct(v):
            return np.percentile(v, [2.5, 97.5])

        print(f"\n################ judge = {name} ################")
        print("accuracy: point  | Wilson | item bootstrap | cluster bootstrap")
        for c in CLASSES + ["POOLED"]:
            mask = np.ones(len(keys), bool) if c == "POOLED" else (cls == c)
            bi, bc = boot(mask, False), boot(mask, True)
            for j, run in enumerate("ABC"):
                k, n = int(X[mask, j].sum()), int(mask.sum())
                w = wilson(k, n); pi = pct(bi[:, j]); pc = pct(bc[:, j])
                print(f"  {c:<11}{run}  {k/n:6.2%} | [{w[0]:.2%}, {w[1]:.2%}] | "
                      f"[{pi[0]:.2%}, {pi[1]:.2%}] | [{pc[0]:.2%}, {pc[1]:.2%}]")
            if c != "POOLED" or True:
                for x, y in [(0, 1), (0, 2), (1, 2)]:
                    d = X[mask, x].mean() - X[mask, y].mean()
                    di, dc = pct(bi[:, x] - bi[:, y]), pct(bc[:, x] - bc[:, y])
                    lab = f"{'ABC'[x]}-{'ABC'[y]}"
                    print(f"  {c:<11}{lab} {d*100:+6.2f} | item [{di[0]*100:+.2f}, {di[1]*100:+.2f}] | "
                          f"cluster [{dc[0]*100:+.2f}, {dc[1]*100:+.2f}]")


if __name__ == "__main__":
    main()
