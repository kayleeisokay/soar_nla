#!/usr/bin/env bash
# Gemma-3-12B-IT block-$LAYER GRPO RL, 400 steps (reference; run inside
# chand-ab/easy_nla @ 479da2f — --critic-device, --nccl-weight-sync,
# --verify-sync, --sync-profile and --no-ipc-weight-sync do not exist in the
# vendored trainer).
#
# Same recipe for all three layers; only --vllm-gpu-mem differs (0.26 at L24 and
# L32, 0.22 at L40), which affects KV capacity and throughput, not training math.
#
# Topology on 8 x 80 GB: 4 data-parallel ranks, each with a tensor-parallel-2
# vLLM engine. GPUS=0,4,1,5,2,6,3,7 makes the TP pairs (0,4) (1,5) (2,6) (3,7):
# rank heads on GPUs 0-3, the critic offloaded to the TP partner (--critic-device 1).
#   --nccl-weight-sync    trainer -> TP rank 1 NCCL push, relayed over the TP group
#                         (requires --vllm-tp >= 2)
#   --verify-sync         checksum every delivered tensor
#   --no-ipc-weight-sync  IPC sync needs ~2x the pushed weights at 12B and OOMs
#                         on the second sync; configs/rl_vllm.yaml turns it on
# Eval sampling uses temperature 1.0 (eval_temperature unset), so held-out FVE of
# identical weights varies by a few points between evals.
#
#   LAYER=32 EASY_NLA=... WORK=... RL_PYTHON=... bash rl.sh
set -uo pipefail
source "$(dirname "$0")/_common.sh"
case $LAYER in 40) DEFAULT_VMEM=0.22 ;; *) DEFAULT_VMEM=0.26 ;; esac
VMEM=${VMEM:-$DEFAULT_VMEM}
SAVE=${SAVE:-$WORK/runs/L${LAYER}_rl}
STEPS=${STEPS:-400}
RESUME=${RESUME:-}
ITER=${ITER:-iter_0003834}
NPROC=${NPROC:-4}
TP=${TP:-2}
unset PYTORCH_CUDA_ALLOC_CONF          # CUDA-IPC weight sync needs the legacy allocator
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export CUDA_VISIBLE_DEVICES=${GPUS:-0,4,1,5,2,6,3,7}
RESUME_ARG=(); [ -n "$RESUME" ] && RESUME_ARG=(--resume-from-lora "$RESUME")

{
  echo "=== RL L$LAYER started $(date -Is) | steps=$STEPS av_iter=$ITER resume=${RESUME:-<none>} ==="
  "$RL_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    -m nla.train_rl_vllm --config configs/rl_vllm.yaml \
    --base-ckpt "$BASE" \
    --av-ckpt "$CKPT/merged/av_hf" \
    --av-adapter "$CKPT/av_sft/$ITER" \
    --ar-ckpt "$CKPT/merged/ar_hf" \
    --rl-parquet "$DATA/rl_shuf.parquet" \
    --sidecar "$DATA/rl_shuf.parquet" \
    --save-dir "$SAVE" "${RESUME_ARG[@]}" \
    --vllm-tp "$TP" --critic-device 1 --no-ipc-weight-sync \
    --nccl-weight-sync --verify-sync --sync-profile --gradient-checkpointing \
    --evals base_fve --num-steps "$STEPS" --batch-prompts 256 --group-size 8 \
    --ar-lora --logp-micro-batch 4 --critic-micro-batch 4 --vllm-gpu-mem "$VMEM" \
    --eval-every 10 --eval-n-prompts 128 --save-every 25 --no-wandb
  rc=$?
  echo "=== RL L$LAYER exited rc=$rc $(date -Is) ==="
  exit "$rc"
} 2>&1 | grep -vE "Processed prompts|Loading checkpoint shards" | tee -a "$LOG_DIR/rl_L${LAYER}_400.log"
