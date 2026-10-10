#!/usr/bin/env bash
# Merge the Gemma-3-12B-IT block-$LAYER AV + AR SFT adapters into bf16 HF
# checkpoints for the vLLM RL path (reference; run inside chand-ab/easy_nla @
# 479da2f — its merge resolves Gemma-3's text model, the vendored one does not).
#
# Checks worth running afterwards: merged/av_hf/config.json must list
# Gemma3ForCausalLM (not Gemma3ForConditionalGeneration, which routes vLLM to
# its multimodal implementation), merged/ar_hf/ar_meta.json must record
# ar_num_layers = LAYER + 1, and the merged value head must equal the trained one.
#
#   LAYER=32 EASY_NLA=... WORK=... PYTHON=... bash merge.sh
set -uo pipefail
source "$(dirname "$0")/_common.sh"
ITER=${ITER:-iter_0003834}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

"$PYTHON" scripts/merge_lora_to_hf.py \
  --base-ckpt "$BASE" \
  --mode both \
  --av-dir "$CKPT/av_sft/$ITER" \
  --ar-dir "$CKPT/ar_sft/$ITER" \
  --av-out "$CKPT/merged/av_hf" \
  --ar-out "$CKPT/merged/ar_hf"
