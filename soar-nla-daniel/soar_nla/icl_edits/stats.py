"""Item-level cluster bootstrap. Seeds of one item share the item, so they are not independent
trials; resampling items (with all their seeds) gives honest intervals."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def cluster_ci(rows, item_key, hit, n_boot: int = 10000, seed: int = 0):
    """Returns (mean, lo, hi, n) with a 95% percentile interval."""
    by = defaultdict(list)
    for r in rows:
        by[item_key(r)].append(1.0 if hit(r) else 0.0)
    if not by:
        return (float("nan"), float("nan"), float("nan"), 0)
    sums = np.array([sum(v) for v in by.values()]); ns = np.array([len(v) for v in by.values()])
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(1) / ns[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(sums.sum() / ns.sum()), float(lo), float(hi), int(ns.sum())


def fmt(ci) -> str:
    m, lo, hi, n = ci
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
