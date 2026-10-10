"""FVE per arm, with a document-clustered bootstrap and a paired gold-vs-AV delta."""

from __future__ import annotations

import torch

from soar_nla.io_utils import stable_seed

from .config import ARMS


def unit(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _fve_from_sums(
    counts: torch.Tensor, doc_rows: torch.Tensor, doc_cos: torch.Tensor, doc_gold: torch.Tensor
) -> torch.Tensor:
    """FVE per resample from per-document sufficient statistics.

    Repo convention: both vectors normalized to mse_scale, so MSE = 2(1-cos)
    and the denominator is the directional variance of the gold set. With unit
    golds that variance is exactly 1 - ||mean||^2, which turns a 10k-resample
    document bootstrap into three matmuls instead of a loop.
    """
    n = counts @ doc_rows
    mean_cos = (counts @ doc_cos) / n
    mean = (counts @ doc_gold) / n.unsqueeze(-1)
    return 1.0 - (2.0 * (1.0 - mean_cos)) / (1.0 - (mean**2).sum(dim=-1))


def _interval(values: torch.Tensor) -> list[float]:
    low, high = torch.quantile(values, torch.tensor([0.025, 0.975]))
    return [float(low), float(high)]


def evaluate(
    keep: torch.Tensor,
    doc_key: list[str],
    unit_gold: torch.Tensor,
    cosine: dict[str, torch.Tensor],
    resamples: int,
    seed: str,
    arms: tuple[str, ...] = ARMS,
) -> dict:
    """Arms + paired delta over the rows selected by the boolean mask `keep`."""
    rows = torch.nonzero(keep).flatten()
    docs = sorted({doc_key[int(i)] for i in rows})
    position = {doc: rank for rank, doc in enumerate(docs)}
    index = torch.tensor([position[doc_key[int(i)]] for i in rows])

    doc_rows = torch.zeros(len(docs)).index_add_(0, index, torch.ones(len(rows)))
    doc_gold = torch.zeros(len(docs), unit_gold.shape[1]).index_add_(0, index, unit_gold[rows])
    doc_cos = {arm: torch.zeros(len(docs)).index_add_(0, index, cosine[arm][rows]) for arm in arms}

    ones = torch.ones(1, len(docs))
    point = {arm: float(_fve_from_sums(ones, doc_rows, doc_cos[arm], doc_gold)[0]) for arm in arms}

    generator = torch.Generator().manual_seed(stable_seed(seed) % (2**31))
    pick = torch.randint(len(docs), (resamples, len(docs)), generator=generator)
    counts = torch.zeros(resamples, len(docs)).scatter_add_(1, pick, torch.ones(resamples, len(docs)))
    draws = {arm: _fve_from_sums(counts, doc_rows, doc_cos[arm], doc_gold) for arm in arms}

    # Paired on the same document resample, so the delta interval absorbs the
    # document-level variance both arms share.
    delta = draws["gold"] - draws["av"]
    per_row = cosine["gold"][rows] - cosine["av"][rows]
    return {
        "n_rows": len(rows),
        "n_docs": len(docs),
        "arms": {
            arm: {
                "fve": point[arm],
                "fve_ci95": _interval(draws[arm]),
                "mean_cosine": float(cosine[arm][rows].mean()),
                "mean_mse_2_1_minus_cos": float((2.0 * (1.0 - cosine[arm][rows])).mean()),
            }
            for arm in arms
        },
        "gold_minus_av": {
            "delta_fve": point["gold"] - point["av"],
            "delta_fve_ci95": _interval(delta),
            "p_two_sided_bootstrap": float(2.0 * min((delta <= 0).float().mean(), (delta >= 0).float().mean())),
            "mean_delta_cosine": float(per_row.mean()),
            "rows_gold_better": int((per_row > 0).sum()),
            "rows_total": len(rows),
        },
    }


def print_table(title: str, block: dict) -> None:
    print(f"\n{title}  ({block['n_rows']} rows / {block['n_docs']} docs)")
    print(f"{'arm':<16}{'FVE':>9}{'95% CI':>22}{'cos':>9}{'MSE':>9}")
    for arm, value in block["arms"].items():
        low, high = value["fve_ci95"]
        print(
            f"{arm:<16}{value['fve']:>9.4f}{f'[{low:.4f}, {high:.4f}]':>22}"
            f"{value['mean_cosine']:>9.4f}{value['mean_mse_2_1_minus_cos']:>9.4f}"
        )
    delta = block["gold_minus_av"]
    low, high = delta["delta_fve_ci95"]
    print(
        f"{'gold - av':<16}{delta['delta_fve']:>+9.4f}{f'[{low:+.4f}, {high:+.4f}]':>22}"
        f"{delta['mean_delta_cosine']:>+9.4f}    p={delta['p_two_sided_bootstrap']:.4f}"
    )
    print(f"{'':<16}gold better on {delta['rows_gold_better']}/{delta['rows_total']} rows")
