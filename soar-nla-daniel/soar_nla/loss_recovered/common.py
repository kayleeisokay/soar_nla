"""Shared paths and model helpers for the Loss Recovered stages."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from soar_nla.paths import results_dir


def default_results_dir() -> Path:
    return results_dir("loss_recovered")


def load_windows(out_dir: Path) -> np.ndarray:
    windows = np.load(out_dir / "windows.npy", mmap_mode="r")
    if windows.ndim != 2:
        raise ValueError(f"expected [windows, context] token ids, got {windows.shape}")
    return windows


def flat_index(window_index: int, position: int, context_size: int) -> int:
    """Row of a (window, position) activation in the flat [windows * context, d_model] arrays."""
    return window_index * context_size + position


def shard_suffix(shard: int, num_shards: int) -> str:
    return f"shard-{shard:02d}-of-{num_shards:02d}"


def layer_output(output: object) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def with_layer_output(output: object, hidden: torch.Tensor) -> object:
    return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


@contextlib.contextmanager
def replace_span(layer: torch.nn.Module, start: int, stop: int, replacement: torch.Tensor) -> Iterator[None]:
    """Overwrite the layer's output at positions [start, stop) with `replacement` ([stop - start, d_model])."""

    def hook(_module, _inputs, output):
        hidden = layer_output(output)
        updated = hidden.clone()
        updated[:, start:stop] = replacement.to(hidden.device, hidden.dtype).unsqueeze(0)
        return with_layer_output(output, updated)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def span_ce(logits: torch.Tensor, input_ids: torch.Tensor, start: int, stop: int) -> float:
    """Mean next-token cross-entropy for the targets at positions start+1 .. stop-1."""
    losses = F.cross_entropy(
        logits[:, start : stop - 1].float().reshape(-1, logits.shape[-1]),
        input_ids[:, start + 1 : stop].reshape(-1),
        reduction="none",
    )
    return float(losses.double().mean().item())
