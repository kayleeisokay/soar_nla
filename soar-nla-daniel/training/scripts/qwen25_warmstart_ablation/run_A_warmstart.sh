#!/usr/bin/env bash
# Run A — warm start: AV + AR SFT (one epoch each), merge, then GRPO RL, KL 0.01.
#
# Reconstructed from the launch commands and logged configs of the paper run;
# the legs below are the ones that were actually executed:
#
#   sft    AV then AR SFT, 8-GPU data parallel, one epoch each (3,834 steps at
#          global batch 64). Expect AV held-out ppl ~4.07, AR held-out FVE ~62%.
#   merge  bf16 merges for vLLM (merged/av_hf) and the RL critic (merged/ar_hf);
#          the merged AR's held-out FVE must match the LoRA checkpoint's.
#   rl1    RL steps 0 -> 400, continuing the SFT adapter (frozen copy = KL
#          reference). Log: rl25.log. Expect eval FVE ~56% at step 0, ~73% at 400.
#   rl2    RL steps 400 -> 800, resumed from iter_000400 + critic_latest +
#          optim_latest (start step taken from the optimizer state).
#          Log: qwen25_l20_resume400_to800.log.
#
# The paper compares A, B and C at step 600 (rl_vllm/iter_000600).
#
#   DATA=... WORK=... PYTHON=... RL_PYTHON=... bash run_A_warmstart.sh [sft merge rl1 rl2]
set -uo pipefail
source "$(dirname "$0")/_common.sh"

CKPT=${CKPT:-$WORK/A_warmstart}
SAVE=$CKPT/rl_vllm
SFT_ITER=iter_0003834     # last SFT checkpoint = one epoch
STAGES=${*:-sft merge rl1 rl2}

run_sft() {
  # expandable_segments is safe for SFT (only the RL trainer's CUDA-IPC weight
  # sync needs the legacy allocator) and is what lets batch 64 fit on 80 GB.
  {
    echo "=== SFT started $(date -Is) | base=$MODEL data=$DATA ckpt=$CKPT ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun_main -m nla.train_sft \
      --mode av --base-ckpt "$MODEL" \
      --parquet "$DATA/av_sft_shuf.full.train.parquet" \
      --sidecar "$DATA/av_sft_shuf.full.train.parquet" \
      --heldout-parquet "$DATA/av_sft_shuf.full.val.parquet" \
      --save-dir "$CKPT/av_sft" \
      --use-lora --lr 1e-4 --batch-size 64 \
      --heldout-every 500 --save-every 1000 --sample-every 1000 \
      --no-wandb || { echo "########## AV FAILED ##########"; exit 1; }
    # --heldout-parquet is the AV val split: held-out FVE is scored on (gold
    # explanation, activation) pairs, and only the AV schema carries `response`.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun_main -m nla.train_sft \
      --mode ar --base-ckpt "$MODEL" \
      --parquet "$DATA/ar_sft_shuf.full.train.parquet" \
      --sidecar "$DATA/ar_sft_shuf.full.train.parquet" \
      --heldout-parquet "$DATA/av_sft_shuf.full.val.parquet" \
      --save-dir "$CKPT/ar_sft" \
      --use-lora --lr 2e-5 --batch-size 64 \
      --heldout-every 500 --save-every 1000 \
      --no-wandb || { echo "########## AR FAILED ##########"; exit 1; }
    echo "=== SFT ALL DONE $(date -Is) ==="
  } 2>&1 | tee -a "$LOG_DIR/sft25.log"
}

run_merge() {
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} "$PYTHON" scripts/merge_lora_to_hf.py \
    --base-ckpt "$MODEL" \
    --av-dir "$CKPT/av_sft/$SFT_ITER" --ar-dir "$CKPT/ar_sft/$SFT_ITER" \
    --av-out "$CKPT/merged/av_hf" --ar-out "$CKPT/merged/ar_hf" --mode both
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} "$PYTHON" scripts/validate_ar_merge.py \
    --base-ckpt "$MODEL" \
    --ar-dir "$CKPT/ar_sft/$SFT_ITER" --merged-dir "$CKPT/merged/ar_hf" \
    --heldout-parquet "$DATA/av_sft_shuf.full.val.parquet" \
    --rows 256 --max-fve-delta 0.1
}

# Tuned on 8 x A100-80GB for Qwen2.5-7B-Instruct (~62 s/step): its KV cache is
# small enough for --vllm-gpu-mem 0.24, and the memory that frees pays for
# running without gradient checkpointing.
run_rl() {  # steps log_name [resume_dir]
  local steps=$1 log=$2 resume=${3:-}
  local resume_arg=(); [ -n "$resume" ] && resume_arg=(--resume-from-lora "$resume")
  {
    echo "=== RL started $(date -Is) | model=$MODEL data=$DATA save=$SAVE resume=${resume:-<none>} ==="
    torchrun_rl -m nla.train_rl_vllm --config configs/rl_vllm.yaml \
      --base-ckpt "$MODEL" \
      --av-ckpt "$CKPT/merged/av_hf" \
      --av-adapter "$CKPT/av_sft/$SFT_ITER" \
      --ar-ckpt "$CKPT/merged/ar_hf" \
      --rl-parquet "$RL_DATA" \
      --sidecar "$RL_DATA" \
      --save-dir "$SAVE" "${resume_arg[@]}" \
      --evals base_fve --num-steps "$steps" --batch-prompts 256 --group-size 8 \
      --ar-lora --logp-micro-batch 4 --critic-micro-batch 4 --vllm-gpu-mem 0.24 \
      --eval-every 10 --eval-n-prompts 128 --save-every 25 --no-wandb
    rc=$?
    echo "=== RL exited rc=$rc $(date -Is) ==="
    exit "$rc"
  } 2>&1 | tee -a "$LOG_DIR/$log"
}

for stage in $STAGES; do
  case $stage in
    sft)   run_sft || exit $? ;;
    merge) run_merge || exit $? ;;
    rl1)   run_rl 400 rl25.log || exit $? ;;
    rl2)   run_rl 800 qwen25_l20_resume400_to800.log "$SAVE/iter_000400" || exit $? ;;
    *)     echo "unknown stage $stage (sft merge rl1 rl2)" >&2; exit 2 ;;
  esac
done
