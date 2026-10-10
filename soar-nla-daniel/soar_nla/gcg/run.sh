#!/usr/bin/env bash
# GCG on the shared Alpaca AV explanations, its random-substitution control,
# fluency scoring under the target model, the trade-off figure and the paper numbers.
#
#   bash soar_nla/gcg/run.sh {gcg|control|perplexity|figure|numbers|all}
#
# Requires `bash soar_nla/alpaca/run.sh all` first (rollouts, activations, AV explanations).
#
# Environment (defaults are what the paper run used):
#   PYTHON          interpreter with soar_nla importable              (python)
#   SOAR_RESULTS    results root; outputs go to $SOAR_RESULTS/gcg
#   GPUS            one GCG worker per listed GPU (AR and AV share it)  ("0 1 2 3 4 5 6 7")
#   CONTROL_GPU     GPU for scoring the control with the AR             (0)
#   PPL_GPU         GPU for the Gemma-3-12B-IT fluency scoring           (0)
#   GCG_STEPS / GCG_TOPK / GCG_CANDIDATES                               (20 / 64 / 64)
set -euo pipefail

PYTHON="${PYTHON:-python}"
STAGE="${1:-all}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
CONTROL_GPU="${CONTROL_GPU:-0}"
PPL_GPU="${PPL_GPU:-0}"
GCG_STEPS="${GCG_STEPS:-20}"
GCG_TOPK="${GCG_TOPK:-64}"
GCG_CANDIDATES="${GCG_CANDIDATES:-64}"
RESULTS_DIR="$("$PYTHON" -c 'from soar_nla.paths import results_dir; print(results_dir("gcg"))')"
ALPACA_DIR="$("$PYTHON" -c 'from soar_nla.paths import results_dir; print(results_dir("alpaca"))')"
CONTROL_DIR="$RESULTS_DIR/random_control"
LOG_DIR="$RESULTS_DIR/logs"
NUM_SHARDS="${#GPUS[@]}"
mkdir -p "$LOG_DIR"

wait_all() {
  local status=0 pid
  for pid in "$@"; do
    wait "$pid" || status=1
  done
  return "$status"
}

gcg_stage() {
  "$PYTHON" -c "from soar_nla.hub import resolve_snapshot, KITFT_L32_AV, KITFT_L32_AR; resolve_snapshot(*KITFT_L32_AV); resolve_snapshot(*KITFT_L32_AR)"
  local pids=() shard gpu
  for shard in "${!GPUS[@]}"; do
    gpu="${GPUS[$shard]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m soar_nla.gcg.gcg worker \
      --alpaca-dir "$ALPACA_DIR" --output-dir "$RESULTS_DIR" \
      --shard "$shard" --num-shards "$NUM_SHARDS" --steps "$GCG_STEPS" \
      --topk "$GCG_TOPK" --candidate-batch-size "$GCG_CANDIDATES" \
      >"$LOG_DIR/gcg_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
  "$PYTHON" -m soar_nla.gcg.gcg merge --output-dir "$RESULTS_DIR" --num-shards "$NUM_SHARDS"
}

control_stage() {
  "$PYTHON" -m soar_nla.gcg.controls unconditional-random \
    --gcg-results "$RESULTS_DIR/results.jsonl" \
    --rollouts "$ALPACA_DIR/gemma12b/rollouts/rollouts.jsonl" --output-dir "$CONTROL_DIR"
  CUDA_VISIBLE_DEVICES="$CONTROL_GPU" "$PYTHON" -m soar_nla.gcg.controls score-ar --scorer kitft \
    --alpaca-dir "$ALPACA_DIR" --output-dir "$CONTROL_DIR" \
    >"$LOG_DIR/control_ar_kitft.log" 2>&1
  "$PYTHON" -m soar_nla.gcg.controls analyze --output-dir "$CONTROL_DIR"
}

perplexity_stage() {
  CUDA_VISIBLE_DEVICES="$PPL_GPU" "$PYTHON" -m soar_nla.gcg.perplexity \
    --gcg-results "$RESULTS_DIR/results.jsonl" \
    --rollouts "$ALPACA_DIR/gemma12b/rollouts/rollouts.jsonl" \
    --out "$RESULTS_DIR/perplexity/results.jsonl" \
    >"$LOG_DIR/perplexity.log" 2>&1
}

figure_stage() {
  "$PYTHON" -m soar_nla.gcg.plots --gcg-results "$RESULTS_DIR/results.jsonl" \
    --rollouts "$ALPACA_DIR/gemma12b/rollouts/rollouts.jsonl" \
    --out "$RESULTS_DIR/gcg_tradeoff.png"
}

numbers_stage() {
  "$PYTHON" -m soar_nla.gcg.paper_numbers --gcg-results "$RESULTS_DIR/results.jsonl" \
    --rollouts "$ALPACA_DIR/gemma12b/rollouts/rollouts.jsonl" \
    --control-scores "$CONTROL_DIR/ar_scores/kitft.jsonl" \
    --perplexity "$RESULTS_DIR/perplexity/results.jsonl" \
    --export-gcg-losses "$RESULTS_DIR/gcg_losses.jsonl.gz"
}

case "$STAGE" in
  gcg) gcg_stage ;;
  control) control_stage ;;
  perplexity) perplexity_stage ;;
  figure) figure_stage ;;
  numbers) numbers_stage ;;
  all)
    gcg_stage
    control_stage
    perplexity_stage
    figure_stage
    numbers_stage
    ;;
  *)
    echo "usage: $0 {gcg|control|perplexity|figure|numbers|all}" >&2
    exit 2
    ;;
esac
