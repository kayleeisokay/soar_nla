"""Verify that merging an AR LoRA preserves held-out reconstruction FVE."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file
from transformers import AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.schema import compute_predict_mean_baselines
from nla.train_sft import heldout_fve_mse, init_critic_from_base, load_heldout_explanation_pairs


def load_lora_critic(base: str, adapter: Path):
    meta = json.loads((adapter / "ar_meta.json").read_text())
    critic = init_critic_from_base(
        base,
        meta["ar_num_layers"],
        torch.bfloat16,
        None,
        device_map=None,
        max_memory=None,
        strip_final_norm=meta.get("final_norm_stripped", False),
    ).to("cuda")
    inject_adapter_in_model(
        LoraConfig(
            r=meta["lora_r"],
            lora_alpha=meta["lora_alpha"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True,
            target_modules=meta["target_modules"],
        ),
        critic.backbone,
    )
    state = load_file(str(adapter / "ar_lora_value_head.safetensors"))
    _, unexpected = critic.load_state_dict(state, strict=False)
    assert not unexpected, unexpected[:5]
    return critic


def score(critic, tokenizer, pairs, template, mse_scale, baseline):
    critic.eval()
    mse, count = heldout_fve_mse(
        critic,
        tokenizer,
        pairs,
        template,
        mse_scale,
        "cuda",
        micro_batch=16,
    )
    return mse, (1.0 - mse / baseline) * 100.0, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--ar-dir", type=Path, required=True)
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--heldout-parquet", required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--max-fve-delta", type=float, default=0.1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.heldout_parquet, tokenizer)
    assert cfg.critic_prompt_template is not None
    pairs = load_heldout_explanation_pairs(args.heldout_parquet, args.rows)
    activations = torch.tensor(np.stack([activation for _, activation in pairs]))
    _, baseline = compute_predict_mean_baselines(activations, cfg.mse_scale)

    lora = load_lora_critic(args.base_ckpt, args.ar_dir)
    lora_mse, lora_fve, count = score(
        lora, tokenizer, pairs, cfg.critic_prompt_template, cfg.mse_scale, baseline
    )
    del lora
    gc.collect()
    torch.cuda.empty_cache()

    merged = NLACriticModel.from_pretrained(
        args.merged_dir,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    merged_mse, merged_fve, merged_count = score(
        merged, tokenizer, pairs, cfg.critic_prompt_template, cfg.mse_scale, baseline
    )

    delta = abs(lora_fve - merged_fve)
    print(
        f"AR merge validation: LoRA mse={lora_mse:.6f} FVE={lora_fve:.3f}% | "
        f"merged mse={merged_mse:.6f} FVE={merged_fve:.3f}% | "
        f"delta={delta:.4f}pp n={count}"
    )
    assert count == merged_count
    assert delta <= args.max_fve_delta, (
        f"AR merge changed held-out FVE by {delta:.4f}pp "
        f"(limit {args.max_fve_delta:.4f}pp)"
    )


if __name__ == "__main__":
    main()
