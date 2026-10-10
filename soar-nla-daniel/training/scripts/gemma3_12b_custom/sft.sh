#!/usr/bin/env bash
# Gemma-3-12B-IT block-$LAYER AV + AR warm-start SFT, 8-GPU data parallel
# (reference; run inside chand-ab/easy_nla @ 479da2f — --init-seed/--data-seed
# do not exist in the vendored trainer).
#
# Identical recipe for all three layers apart from DATA/CKPT and
# --ar-num-layers (= LAYER + 1). One epoch = 3,834 steps at global batch 64.
#
#   LAYER=32 EASY_NLA=... WORK=... PYTHON=... bash sft.sh
set -uo pipefail
source "$(dirname "$0")/_common.sh"
# eff_batch = BATCH * ACCUM must stay 64 (the tuned LRs assume it);
# BATCH=32 ACCUM=2 is an exactly equivalent fallback if 64 does not fit.
BATCH=${BATCH:-64}
ACCUM=${ACCUM:-1}
NPROC=${NPROC:-8}
# Safe for SFT: only the RL trainer's CUDA-IPC weight sync needs the legacy allocator.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$CKPT"
TORCHRUN=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC")

{
  echo "=== SFT $(date -Is) | base=$BASE data=$DATA ckpt=$CKPT batch=$BATCH accum=$ACCUM ==="
  # AV: LoRA on the full bf16 base (not 4-bit: merging a 4-bit-trained adapter
  # into bf16 for the vLLM RL path is lossy).
  "${TORCHRUN[@]}" -m nla.train_sft \
    --mode av --base-ckpt "$BASE" \
    --parquet "$DATA/av_sft_shuf.train.parquet" \
    --sidecar "$DATA/av_sft_shuf.train.parquet" \
    --heldout-parquet "$DATA/av_sft_shuf.val.parquet" \
    --save-dir "$CKPT/av_sft" \
    --use-lora --lr 1e-4 --batch-size "$BATCH" \
    --gradient-accumulation-steps "$ACCUM" \
    --init-seed 0 --data-seed 0 \
    --heldout-every 500 --save-every 1000 --sample-every 1000 \
    --no-wandb || { echo "AV FAILED"; exit 1; }
  # AR: --heldout-parquet must be an AV-split file (FVE is scored on
  # (explanation, activation) pairs; only the AV schema carries `response`).
  "${TORCHRUN[@]}" -m nla.train_sft \
    --mode ar --base-ckpt "$BASE" \
    --parquet "$DATA/ar_sft_shuf.train.parquet" \
    --sidecar "$DATA/ar_sft_shuf.train.parquet" \
    --heldout-parquet "$DATA/av_sft_shuf.val.parquet" \
    --save-dir "$CKPT/ar_sft" \
    --ar-num-layers "$AR_NUM_LAYERS" \
    --use-lora --lr 2e-5 --batch-size "$BATCH" \
    --gradient-accumulation-steps "$ACCUM" \
    --init-seed 0 --data-seed 0 \
    --heldout-every 500 --save-every 1000 \
    --no-wandb || { echo "AR FAILED"; exit 1; }
  echo "=== SFT ALL DONE $(date -Is) ==="
} 2>&1 | tee -a "$LOG_DIR/sft_L$LAYER.log"
