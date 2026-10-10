#!/usr/bin/env bash
# Run C — cold start, KL 0.01: GRPO RL straight from the base model (fresh
# LoRA on Qwen2.5-7B-Instruct, untrained identity-head AR), no SFT data,
# KL 0.01 (k3, the trainer's default beta) against the base model.
#
# Reconstructed from the launch commands and logged configs of the paper run;
# the legs below are the ones that were actually executed:
#
#   leg1  RL steps 0 -> 400. Log: rl25_cold_nosft.log.
#   leg2  RL steps 400 -> 600, resumed from iter_000400 (+ critic_latest,
#         optim_latest) into the same save dir; appended to rl25_cold_nosft.log.
#   leg3  RL steps 600 -> 1200 (stopped by hand after step 823), resumed from
#         iter_000600 with all flags explicit. Its step-600 held-out eval (run
#         before the first update, i.e. of iter_000600) is the held-out number
#         reported for C. Log: rl25_cold_nosft_resume600_1200.log.
#
# The paper compares A, B and C at step 600 (iter_000600).
#
#   DATA=... WORK=... PYTHON=... RL_PYTHON=... bash run_C_cold_kl001.sh [leg1 leg2 leg3]
set -uo pipefail
source "$(dirname "$0")/_common.sh"

ROOT=${ROOT:-$WORK/C_cold_kl001}
SAVE=$ROOT/rl_vllm
LEG3_STEPS=${LEG3_STEPS:-1200}
STAGES=${*:-leg1 leg2 leg3}

cold_rl() {  # extra args...
  torchrun_rl -m nla.train_rl_vllm --config configs/rl_vllm.yaml \
    --base-ckpt "$MODEL" \
    --av-ckpt "$MODEL" \
    --ar-ckpt "$COLD_AR" \
    --rl-parquet "$RL_DATA" \
    --sidecar "$RL_DATA" \
    --save-dir "$SAVE" \
    "$@" \
    --evals base_fve \
    --batch-prompts 256 --group-size 8 \
    --max-new-tokens 256 --max-rows 0 \
    --ar-lora --logp-micro-batch 4 --critic-micro-batch 4 \
    --vllm-gpu-mem 0.24 \
    --eval-every 10 --eval-n-prompts 128 \
    --save-every 25 --no-wandb
}

run_leg1() {
  init_cold_critic
  {
    echo "=== COLD/COLD RL started $(date -Is) | model=$MODEL data=$DATA save=$SAVE ==="
    echo "=== task SFT: actor=0 steps, critic=0 steps | critic head=identity ==="
    cold_rl --num-steps 400
    RC=$?
    echo "=== COLD/COLD RL exited rc=$RC $(date -Is) ==="
    exit "$RC"
  } 2>&1 | tee -a "$LOG_DIR/rl25_cold_nosft.log"
}

run_leg2() {
  {
    echo "=== COLD/COLD RL RESUME 400->600 started $(date -Is) ==="
    cold_rl --resume-from-lora "$SAVE/iter_000400" --num-steps 600
    rc=$?
    echo "=== COLD/COLD RL RESUME exited rc=$rc $(date -Is) ==="
    exit "$rc"
  } 2>&1 | tee -a "$LOG_DIR/rl25_cold_nosft.log"
}

run_leg3() {
  {
    echo "Resuming cold no-SFT KL=0.01 run: step 600 -> $LEG3_STEPS"
    echo "Started (UTC): $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    torchrun_rl -m nla.train_rl_vllm \
      --config configs/rl_vllm.yaml \
      --base-ckpt "$MODEL" \
      --av-ckpt "$MODEL" \
      --ar-ckpt "$COLD_AR" \
      --rl-parquet "$RL_DATA" \
      --sidecar "$RL_DATA" \
      --save-dir "$SAVE" \
      --resume-from-lora "$SAVE/iter_000600" \
      --start-step 600 \
      --num-steps "$LEG3_STEPS" \
      --evals base_fve \
      --batch-prompts 256 \
      --group-size 8 \
      --max-new-tokens 256 \
      --max-rows 0 \
      --eval-skip-rows 0 \
      --kl-beta 0.01 \
      --kl-estimator k3 \
      --ar-lora \
      --logp-micro-batch 4 \
      --critic-micro-batch 4 \
      --vllm-gpu-mem 0.24 \
      --eval-every 10 \
      --eval-n-prompts 128 \
      --save-every 25 \
      --sampler-mismatch-thresh 0.1 \
      --no-wandb
  } 2>&1 | tee -a "$LOG_DIR/rl25_cold_nosft_resume600_1200.log"
}

for stage in $STAGES; do
  case $stage in
    leg1) run_leg1 || exit $? ;;
    leg2) run_leg2 || exit $? ;;
    leg3) run_leg3 || exit $? ;;
    *)    echo "unknown stage $stage (leg1 leg2 leg3)" >&2; exit 2 ;;
  esac
done
