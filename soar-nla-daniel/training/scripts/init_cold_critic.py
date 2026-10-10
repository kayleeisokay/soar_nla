#!/usr/bin/env python3
"""Create an AR critic from raw pretrained weights, without task SFT.

The critic keeps blocks 0..K for activations extracted after block K and uses
the project's standard identity initialization for the value head.  No AV/AR
dataset examples are consumed by this initialization.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nla.config import load_nla_config
from nla.schema import sidecar_path_for
from nla.train_sft import init_critic_from_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if (out_dir / "config.json").exists() and (
        out_dir / "value_head.safetensors"
    ).exists():
        print(f"[cold-ar] checkpoint already complete; reusing {out_dir}")
        return
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"{out_dir} exists but is incomplete/non-empty; refusing to overwrite it"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.sidecar, tokenizer)
    if cfg.extraction_layer_index is None:
        raise SystemExit("sidecar does not define extraction.layer_index")
    num_layers = cfg.extraction_layer_index + 1

    print(
        f"[cold-ar] raw base={args.base_ckpt} extraction block="
        f"{cfg.extraction_layer_index} critic layers={num_layers}"
    )
    critic = init_critic_from_base(
        args.base_ckpt,
        num_layers,
        torch.bfloat16,
        quant_config=None,
        device_map=None,
        max_memory=None,
        strip_final_norm=True,
    )

    out_dir.mkdir(parents=True, exist_ok=False)
    critic.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    shutil.copy2(sidecar_path_for(args.sidecar), out_dir / "nla_meta.yaml")
    (out_dir / "cold_init.json").write_text(
        json.dumps(
            {
                "base_checkpoint": args.base_ckpt,
                "task_sft_steps": 0,
                "extraction_layer_index": cfg.extraction_layer_index,
                "critic_num_layers": num_layers,
                "value_head_init": "identity",
                "final_norm_stripped": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[cold-ar] saved task-untrained critic to {out_dir}")


if __name__ == "__main__":
    main()
