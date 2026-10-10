"""Pure injection-hook logic — extracted for testability.

The most correctness-critical path in NLA: if injection fails or hits the wrong
position, the model sees the literal ㊗ character and outputs Chinese. This
function is the one place that must be right, so it's pure and unit-testable.
"""

import math

import torch


def _valid_marker_positions(row_ids, inj_id: int, left_id: int, right_id: int) -> list[int]:
    n = len(row_ids)
    return [
        p for p, token_id in enumerate(row_ids)
        if token_id == inj_id and 0 < p < n - 1
        and row_ids[p - 1] == left_id and row_ids[p + 1] == right_id
    ]


def _marker_vector_sites(
    input_ids: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
) -> list[tuple[int, int, torch.Tensor]]:
    """Pair validated marker sites with vectors in deterministic actor order.

    [B,K,d] is checked per row so one row's missing site cannot be offset by
    another row's extra site. [N,d] retains the historical packed/global path.
    """
    if vectors.ndim not in (2, 3):
        raise AssertionError(
            f"vectors must be [N,d_model] or [B,K,d_model], got {tuple(vectors.shape)}"
        )
    if vectors.ndim == 3:
        batch, context_size, _ = vectors.shape
        if batch != input_ids.shape[0]:
            raise AssertionError(
                f"context batch {batch} != input_ids batch {input_ids.shape[0]}"
            )
        sites = []
        for b in range(batch):
            positions = _valid_marker_positions(
                input_ids[b].tolist(), inj_id, left_id, right_id,
            )
            if len(positions) != context_size:
                raise RuntimeError(
                    f"row {b}: found {len(positions)} valid injection sites, "
                    f"expected context_size={context_size}. Check prompt template "
                    "drift, tokenizer mismatch, or a marker echoed in the response."
                )
            sites.extend((b, p, vectors[b, j]) for j, p in enumerate(positions))
        return sites

    positions = []
    for b in range(input_ids.shape[0]):
        positions.extend(
            (b, p) for p in _valid_marker_positions(
                input_ids[b].tolist(), inj_id, left_id, right_id,
            )
        )
    if len(positions) != vectors.shape[0]:
        raise RuntimeError(
            f"found {len(positions)} valid injection sites, expected "
            f"{vectors.shape[0]}. Check prompt template drift, tokenizer mismatch, "
            "or a marker echoed in the response."
        )
    return [(b, p, vectors[j]) for j, (b, p) in enumerate(positions)]


def inject_at_marked_positions(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
    seq_slice: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Overwrite embedding rows at injection-marker positions with activation vectors.

    input_ids: [B, S] — or [1, T_packed] for thd layout. The FULL token stream
        (broadcast across TP ranks — identical everywhere).
    embeddings: [B, S, d] (unsharded) or [B, S_local, d] (seq_slice set). The
        embedding layer output. Cloned; original unchanged.
    vectors: [N, d] for the historical packed/global path, or [B, K, d] for
        actor context. The latter requires exactly K valid sites in every row.
    inj_id, left_id, right_id: the injection token + its canonical neighbors.
    seq_slice: (start, end) if embeddings holds only positions [start:end) of
        the sequence dim. For Megatron with --sequence-parallel: each TP rank's
        LanguageModelEmbedding output covers [tp_rank * S/TP : (tp_rank+1) * S/TP).
        The scan still runs over FULL input_ids (count + vec_idx are global),
        writes skip positions outside the slice.

    A match is valid iff input_ids[b, p] == inj_id AND input_ids[b, p-1] == left_id
    AND input_ids[b, p+1] == right_id. The neighbor check rejects false positives
    from ㊗ appearing in response text (user pasted it, multi-turn context).

    Raises:
        AssertionError if GLOBAL count of valid matches != vectors.shape[0] —
        means prompt template drift, tokenizer version mismatch, or data corruption.
        Fires identically on every TP rank (scan is over full input_ids).
    """
    seq_len = input_ids.shape[-1]
    if seq_slice is None:
        start, end = 0, seq_len
        assert input_ids.shape == embeddings.shape[:-1], (
            f"input_ids {tuple(input_ids.shape)} and embeddings "
            f"{tuple(embeddings.shape[:-1])} batch dims must match"
        )
    else:
        start, end = seq_slice
        assert input_ids.shape[0] == embeddings.shape[0], (
            f"batch dim mismatch: input_ids {input_ids.shape[0]}, "
            f"embeddings {embeddings.shape[0]}"
        )
        assert embeddings.shape[1] == end - start, (
            f"seq_slice={seq_slice} spans {end - start} positions but "
            f"embeddings seq dim is {embeddings.shape[1]}. SP shard layout "
            f"mismatch — check tp_rank/tp_size computation."
        )
    assert vectors.ndim in (2, 3) and vectors.shape[-1] == embeddings.shape[-1], (
        f"vectors must be [N,d_model] or [B,K,d_model], got {tuple(vectors.shape)}, "
        f"d_model={embeddings.shape[-1]}"
    )
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    try:
        sites = _marker_vector_sites(input_ids, vectors, inj_id, left_id, right_id)
    except RuntimeError as exc:
        # Under PP, this hook only runs on stage 0. Bare assert leaves stage 1
        # hanging on P2P recv → 10min NCCL timeout with no error. Abort the
        # whole world so the real error surfaces.
        if torch.distributed.is_initialized():
            print(f"[inject_at_marked_positions] FATAL: {exc}", flush=True)
            torch.distributed.destroy_process_group()
        raise
    for b, p, vector in sites:
        if start <= p < end:
            out[b, p - start] = vector
    return out


def marker_well_formed(
    prompt_ids, inj_id: int, left_id: int, right_id: int,
    expected_count: int = 1,
) -> bool:
    """Per-rollout precondition for a correct Karvonen injection (HF path).

    True iff `prompt_ids` contains exactly ``expected_count`` marker tokens and
    every marker has its canonical left/right neighbors — i.e.
    the same validity test `inject_at_marked_positions` / `karvonen_inject_in_residual`
    apply per position. If this holds, the hook overwrites the marker's residual at
    the right place; if it doesn't, injection lands wrong (or the hook's count assert
    fires).

    This is a MECHANISM check, not an output-text proxy like CJK fraction: it can't
    be eroded by RL shifting the model's output distribution (the failure mode where
    a failed injection stops producing CJK once the policy has learned to avoid it).
    Cheap (pure token scan, no forward), so it's run per rollout to mask out any
    rollout whose marker drifted (tokenizer/template mismatch, marker echoed into the
    response and re-tokenized, etc.) before it can corrupt the AV/AR updates.
    """
    if expected_count < 1:
        return False
    all_positions = [i for i, t in enumerate(prompt_ids) if t == inj_id]
    valid = _valid_marker_positions(prompt_ids, inj_id, left_id, right_id)
    return len(all_positions) == expected_count and len(valid) == expected_count


def karvonen_inject_in_residual(
    input_ids: torch.Tensor,
    resid: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
) -> torch.Tensor:
    """ADD-norm-matched injection per Karvonen et al. 2025 (Activation Oracles, eq. 1).

    For each marker position p: h'_p = h_p + ||h_p|| * v / ||v||.

    Caller responsibility: register this hook on the OUTPUT of the second
    transformer layer (i.e. `model.model.layers[1].register_forward_hook(...)`),
    so the residual entering layer 2 is the modified one. Vectors should be
    RAW (no injection_scale normalization) — this function does its own norm
    match against the current residual.
    """
    assert input_ids.shape == resid.shape[:-1], (
        f"input_ids {tuple(input_ids.shape)} and resid {tuple(resid.shape[:-1])} batch dims must match"
    )
    assert vectors.ndim in (2, 3) and vectors.shape[-1] == resid.shape[-1], (
        f"vectors must be [N,d_model] or [B,K,d_model], got {tuple(vectors.shape)}, "
        f"d_model={resid.shape[-1]}"
    )
    out = resid.clone()
    vectors = vectors.to(out.device, out.dtype)
    try:
        sites = _marker_vector_sites(input_ids, vectors, inj_id, left_id, right_id)
    except RuntimeError as exc:
        if torch.distributed.is_initialized():
            print(f"[karvonen_inject_in_residual] FATAL: {exc}", flush=True)
            torch.distributed.destroy_process_group()
        raise
    for b, p, vector in sites:
        # Clone the slice before reading — otherwise out[b, p] is a VIEW into
        # `out`'s storage and the in-place write below modifies the same memory
        # the autograd graph references → "modified by inplace op" RuntimeError
        # at backward time.
        h_p = out[b, p].clone()
        v_unit = vector / (vector.norm() + 1e-9)
        out[b, p] = h_p + h_p.norm() * v_unit
    return out


def fixed_norm_replace_in_residual(
    input_ids: torch.Tensor,
    resid: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
    target_norm: float,
) -> torch.Tensor:
    """Replace marker residuals with activation directions at a fixed L2 norm.

    For each valid marker position p:

        h'_p = target_norm * v / ||v||

    Unlike :func:`karvonen_inject_in_residual`, this removes the marker's
    original residual completely and does not use its norm.  The marker scan
    and exact-count invariant intentionally match the additive path.
    """
    if not math.isfinite(target_norm) or target_norm <= 0:
        raise ValueError(f"target_norm must be finite and > 0, got {target_norm}")
    assert input_ids.shape == resid.shape[:-1], (
        f"input_ids {tuple(input_ids.shape)} and resid "
        f"{tuple(resid.shape[:-1])} batch dims must match"
    )
    assert vectors.ndim in (2, 3) and vectors.shape[-1] == resid.shape[-1], (
        f"vectors must be [N,d_model] or [B,K,d_model], got {tuple(vectors.shape)}, "
        f"d_model={resid.shape[-1]}"
    )
    out = resid.clone()
    vectors = vectors.to(out.device, out.dtype)
    try:
        sites = _marker_vector_sites(input_ids, vectors, inj_id, left_id, right_id)
    except RuntimeError as exc:
        if torch.distributed.is_initialized():
            print(f"[fixed_norm_replace_in_residual] FATAL: {exc}", flush=True)
            torch.distributed.destroy_process_group()
        raise
    for b, p, v in sites:
        # Compute the normalization factor in fp32, then write in the residual
        # dtype. This is fixed-scale replacement, not norm matching.
        v_scaled = v.float() * (target_norm / (v.float().norm() + 1e-9))
        out[b, p] = v_scaled.to(out.dtype)
    return out
