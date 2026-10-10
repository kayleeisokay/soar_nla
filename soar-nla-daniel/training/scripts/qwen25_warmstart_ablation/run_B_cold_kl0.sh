#!/usr/bin/env bash
# Run B — cold start, KL 0: GRPO RL straight from the base model (fresh LoRA
# on Qwen2.5-7B-Instruct, untrained identity-head AR), no SFT data, beta = 0.
#
# Reconstructed from the launch commands and logged configs of the paper run;
# the legs below are the ones that were actually executed:
#
#   leg1  RL steps 0 -> 400, KL 0, strict tag parsing. Launched for 400 steps
#         and ran to completion, but diverged from ~step 220 (the actor stops
#         closing its <explanation> tag, extraction falls to 0%, reward NaN).
#         Only steps 0-199 of this leg are reported. Log: rl25_cold_origprompt_kl0.log.
#   leg2  RL steps 200 -> 600 branched from leg1's iter_000200 actor with a
#         FRESH optimizer (--no-resume-optimizer; leg1's optim_latest is from
#         step 400) and leg1's critic_latest (saved at step 400) as the AR, now
#         with --raw-completion-fallback (an untagged non-empty completion is
#         passed to the AR as the message) and --score-truncated. All flags are
#         explicit (no --config), as launched. Log: rl25_cold_origprompt_kl0_rawfallback.log.
#
# The paper compares A, B and C at step 600 (leg2's iter_000600).
#
#   DATA=... WORK=... PYTHON=... RL_PYTHON=... bash run_B_cold_kl0.sh [leg1 leg2]
set -uo pipefail
source "$(dirname "$0")/_common.sh"

ROOT=${ROOT:-$WORK/B_cold_kl0}
LEG1=$ROOT/leg1_strict/rl_vllm
LEG2=$ROOT/rl_vllm
STAGES=${*:-leg1 leg2}

run_leg1() {
  init_cold_critic
  mkdir -p "$LEG1"
  {
    echo "=== ORIGINAL-PROMPT COLD/COLD KL0 started $(date -Is) ==="
    torchrun_rl -m nla.train_rl_vllm --config configs/rl_vllm.yaml \
      --base-ckpt "$MODEL" --av-ckpt "$MODEL" --ar-ckpt "$COLD_AR" \
      --rl-parquet "$RL_DATA" --sidecar "$RL_DATA" \
      --save-dir "$LEG1" \
      --evals base_fve --num-steps 400 --batch-prompts 256 --group-size 8 \
      --max-new-tokens 256 --max-rows 0 \
      --ar-lora --logp-micro-batch 4 --critic-micro-batch 4 --vllm-gpu-mem 0.24 \
      --eval-every 10 --eval-n-prompts 128 --save-every 25 \
      --kl-beta 0 --no-wandb
    run_rc=$?
    echo "=== ORIGINAL-PROMPT COLD/COLD KL0 exited rc=$run_rc $(date -Is) ==="
    exit "$run_rc"
  } 2>&1 | tee -a "$LOG_DIR/rl25_cold_origprompt_kl0.log"
}

run_leg2() {
  mkdir -p "$LEG2"
  {
    echo "=== ORIGINAL-PROMPT KL0 RAW-FALLBACK 200->600 started $(date -Iseconds) ==="
    torchrun_rl --module nla.train_rl_vllm \
      --base-ckpt "$MODEL" \
      --av-ckpt "$MODEL" \
      --ar-ckpt "$LEG1/critic_latest" \
      --resume-from-lora "$LEG1/iter_000200" \
      --no-resume-optimizer \
      --rl-parquet "$RL_DATA" \
      --sidecar "$RL_DATA" \
      --save-dir "$LEG2" \
      --start-step 200 \
      --num-steps 600 \
      --batch-prompts 256 \
      --group-size 8 \
      --max-new-tokens 256 \
      --temperature 1.0 \
      --lr 1e-4 \
      --lora-r 128 \
      --lora-alpha 16 \
      --use-rslora \
      --train-critic \
      --critic-lr 8e-5 \
      --ar-lora \
      --ar-lora-r 64 \
      --ar-lora-alpha 16 \
      --length-penalty 0.01 \
      --length-threshold 192 \
      --critic-micro-batch 4 \
      --logp-micro-batch 4 \
      --vllm-gpu-mem 0.24 \
      --vllm-max-len 1024 \
      --vllm-tp 1 \
      --ipc-weight-sync \
      --save-every 25 \
      --eval-every 10 \
      --eval-n-prompts 128 \
      --eval-skip-rows 0 \
      --max-rows 0 \
      --val-rows 50000 \
      --kl-beta 0 \
      --sampler-mismatch-thresh 0.1 \
      --raw-completion-fallback \
      --score-truncated \
      --save-raw-evals \
      --no-wandb \
      --seed 0
    rc=$?
    echo "=== finished $(date -Iseconds) rc=$rc ==="
    exit "$rc"
  } 2>&1 | tee -a "$LOG_DIR/rl25_cold_origprompt_kl0_rawfallback.log"
}

for stage in $STAGES; do
  case $stage in
    leg1) run_leg1 || exit $? ;;
    leg2) run_leg2 || exit $? ;;
    *)    echo "unknown stage $stage (leg1 leg2)" >&2; exit 2 ;;
  esac
done
