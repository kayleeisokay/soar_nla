"""Karvonen activation injection for the vLLM (vllm-lens) path — shared by the
RL rollout (train_rl_vllm.rollout_batch_vllm) and the offline AV decoder used by
the counterfactual_evidence eval, so both inject IDENTICALLY.

The injection is a `vllm_lens.SteeringVector` applied at `injection_layer` (=1)
with `norm_match=True`: the activation is renormalized to the running residual
norm and ADDed at the marker position — the exact Karvonen formula
`h'_p = h_p + ||h_p|| · v/||v||` the AV was trained with. Requires the
vllm-lens norm-match patch (utils/patch_vllm_lens.py).
"""

from __future__ import annotations

import math


class _ReplacementSteeringVectorMixin:
    replace: bool = True


def find_marker_positions(
    prompt_ids, inj_id, left_id=None, right_id=None, expected_count=1,
):
    """The ordered injection-marker positions in an AV prompt.

    When left_id/right_id are given, a position only counts if its neighbors
    match the canonical template neighbors — the SAME validity test the HF
    injection path applies (nla.injection). Without it the two paths are
    asymmetric: a stray marker glyph with wrong neighbors (e.g. leaked into the
    source text) would be REJECTED by the HF hook but silently STEERED by vLLM.
    Asserts exactly ``expected_count`` valid markers either way."""
    n = len(prompt_ids)
    if left_id is None and right_id is None:
        positions = [i for i, t in enumerate(prompt_ids) if t == inj_id]
    else:
        positions = [
            i for i, t in enumerate(prompt_ids)
            if t == inj_id and 0 < i < n - 1
            and (left_id is None or prompt_ids[i - 1] == left_id)
            and (right_id is None or prompt_ids[i + 1] == right_id)
        ]
    assert len(positions) == expected_count, (
        f"expected exactly {expected_count} valid marker token(s) (inj_id={inj_id}, "
        f"left={left_id}, right={right_id}) in prompt, got {len(positions)}"
    )
    return positions


def find_marker_pos(prompt_ids, inj_id, left_id=None, right_id=None):
    """Backward-compatible single-marker wrapper."""
    return find_marker_positions(prompt_ids, inj_id, left_id, right_id, 1)[0]


def read_reset_steer_count(model):
    """vLLM `apply_model` helper (runs IN each worker): read+reset the steering-apply
    counter the vllm-lens patch maintains, so the trainer can verify injection
    actually happened during a rollout (a distribution-invariant check vs. inferring
    it from CJK output). `model` is unused. Returns -1 if the counter isn't present
    (patch not applied) so the trainer degrades gracefully to the CJK/marker checks.

    MUST live in an importable module (not the `python -m ...`-run trainer, which is
    `__main__`): vLLM's collective_rpc pickles this by qualified name to ship it to
    the workers, and a `__main__` function can't be pickled by reference.
    """
    try:
        import vllm_lens._worker_ext as _wx
        fn = getattr(_wx, "get_and_reset_steer_count", None)
        return int(fn()) if fn is not None else -1
    except Exception:
        return -1


def read_reset_steer_log(model):
    """vLLM `apply_model` helper (runs IN each worker): fetch+reset the
    PER-REQUEST steering coverage log the patched vllm-lens maintains
    (patch_vllm_lens fix (4)): {log_key: {covered, applied, positions,
    orphaned}}, keyed by the offline path's _steering_id ("_steer_{idx}").
    Returns {} when the patch (or the log hunk) isn't applied, so callers
    degrade gracefully. Must live in an importable module (collective_rpc
    pickles it by qualified name)."""
    try:
        import vllm_lens._worker_ext as _wx
        fn = getattr(_wx, "get_and_reset_steer_log", None)
        return dict(fn()) if fn is not None else {}
    except Exception:
        return {}


def _context_and_positions(activation, marker_positions):
    context = activation.detach().cpu().float()
    if context.ndim == 1:
        context = context.unsqueeze(0)
    if context.ndim != 2:
        raise ValueError(
            f"activation must be [d] or [K,d], got {tuple(context.shape)}"
        )
    positions = (
        [int(marker_positions)]
        if isinstance(marker_positions, int)
        else [int(p) for p in marker_positions]
    )
    if len(positions) != context.shape[0]:
        raise ValueError(
            f"got {len(positions)} marker positions for context shape {tuple(context.shape)}"
        )
    return context, positions


def build_steering_vector(activation, marker_positions, injection_layer=1):
    """One ordered activation context -> one positional SteeringVector.

    SHAPE MATTERS: activations must be 3-D [n_layers, n_positions, d]. vllm-lens
    only honors position_indices for 3-D tensors; a 2-D [n_layers, d] tensor
    takes the BROADCAST branch in _worker_ext.py (_apply_steering) and gets ADDed
    at EVERY token, silently ignoring position_indices.
    """
    from vllm_lens import SteeringVector

    context, positions = _context_and_positions(activation, marker_positions)

    return SteeringVector(
        activations=context.unsqueeze(0),  # [1, K, d]
        layer_indices=[injection_layer],
        scale=1.0,
        norm_match=True,
        position_indices=positions,
    )


def build_injection_vector(
    activation,
    marker_positions,
    injection_layer=1,
    injection_mode="add_norm_match",
    replacement_norm=150.0,
):
    """Build the vLLM payload for additive or fixed-norm replacement injection."""
    if injection_mode == "add_norm_match":
        return build_steering_vector(activation, marker_positions, injection_layer)
    if injection_mode != "replace_fixed":
        raise ValueError(f"unknown injection_mode={injection_mode!r}")
    if not math.isfinite(replacement_norm) or replacement_norm <= 0:
        raise ValueError(
            f"replacement_norm must be finite and > 0, got {replacement_norm}"
        )

    from vllm_lens import SteeringVector

    class ReplacementSteeringVector(SteeringVector):
        """Offline vLLM payload understood by the EasyNLA worker patch."""

        replace: bool = True

    v, positions = _context_and_positions(activation, marker_positions)
    v = v * (replacement_norm / (v.norm(dim=-1, keepdim=True) + 1e-9))
    return ReplacementSteeringVector(
        activations=v.unsqueeze(0),
        layer_indices=[injection_layer],
        scale=1.0,
        norm_match=False,
        position_indices=positions,
    )
