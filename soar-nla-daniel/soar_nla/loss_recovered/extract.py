"""Capture the layer-32 residual stream (output of decoder block 32) at every window position.

Writes `activations.npy`, a float32 [windows * 128, d_model] array whose row
`window * 128 + position` is the clean activation at that window position.

    CUDA_VISIBLE_DEVICES=0 python -m soar_nla.loss_recovered.extract
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from soar_nla.hub import load_target, model_spec, resolve_decoder_layers
from soar_nla.io_utils import write_json

from .common import default_results_dir, layer_output, load_windows
from .config import MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=default_results_dir())
    parser.add_argument("--target", type=Path, default=None, help="local target checkpoint (default: pinned snapshot)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="only the first N windows (smoke test)")
    args = parser.parse_args()

    spec = model_spec(MODEL)
    windows = load_windows(args.output_dir)
    n_windows, context_size = windows.shape
    if args.limit is not None:
        n_windows = min(n_windows, args.limit)
    model, _, checkpoint = load_target(spec, "cuda:0", checkpoint=args.target)
    layer = resolve_decoder_layers(model)[spec.layer]

    activations = np.lib.format.open_memmap(
        args.output_dir / "activations.npy", mode="w+", dtype=np.float32, shape=(n_windows * context_size, spec.d_model)
    )
    captured: list[torch.Tensor] = []
    handle = layer.register_forward_hook(lambda _m, _i, output: captured.append(layer_output(output).float().cpu()))
    try:
        with torch.inference_mode():
            for start in range(0, n_windows, args.batch_size):
                stop = min(start + args.batch_size, n_windows)
                ids = torch.as_tensor(np.array(windows[start:stop]), dtype=torch.long, device="cuda:0")
                captured.clear()
                model(input_ids=ids, use_cache=False, logits_to_keep=1)
                if len(captured) != 1 or tuple(captured[0].shape) != (stop - start, context_size, spec.d_model):
                    raise RuntimeError(f"unexpected capture for windows {start}-{stop}")
                activations[start * context_size : stop * context_size] = captured[0].reshape(-1, spec.d_model).numpy()
                if stop % 100 < args.batch_size or stop == n_windows:
                    print(f"extract windows {stop}/{n_windows}", flush=True)
    finally:
        handle.remove()
    activations.flush()
    write_json(
        args.output_dir / "extraction.json",
        {
            "model": f"{spec.target_id}@{spec.target_revision}",
            "checkpoint": checkpoint.name,
            "layer": spec.layer,
            "activation": "output of decoder block 32 (residual stream), every window position",
            "windows": n_windows,
            "shape": [n_windows * context_size, spec.d_model],
            "dtype": "float32 (model runs in bfloat16)",
            "batch_size": args.batch_size,
        },
    )


if __name__ == "__main__":
    main()
