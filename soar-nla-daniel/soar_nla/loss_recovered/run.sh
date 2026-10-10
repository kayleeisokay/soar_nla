#!/usr/bin/env bash
# Loss Recovered: NLA round trip (and the SAE baseline) patched into Gemma-3-12B-IT layer 32.
#
#   bash soar_nla/loss_recovered/run.sh {prepare|extract|verbalize|reconstruct|evaluate|numbers|all}
#
# Environment (defaults are what the paper run used):
#   PYTHON              interpreter with soar_nla importable                 (python)
#   SOAR_SGLANG_PYTHON  interpreter with the patched SGLang                   ($PYTHON)
#   SOAR_RESULTS        results root; outputs go to $SOAR_RESULTS/loss_recovered
#   ALPACA              Lie Detection Rollouts google-gemma-3-12b-it/alpaca.parquet (prepare only)
#   GPUS                one AV server / AR shard / evaluation shard per GPU    ("0 1 2 3")
#   AV_PORT_BASE        SGLang port = base + shard index                      (30000)
#
# Keep four GPUs to reproduce the AR reconstructions bit for bit: the shard count fixes
# which explanations share a padded AR batch.
set -euo pipefail

PYTHON="${PYTHON:-python}"
SGLANG_PYTHON="${SOAR_SGLANG_PYTHON:-$PYTHON}"
STAGE="${1:-all}"
GPUS=(${GPUS:-0 1 2 3})
AV_PORT_BASE="${AV_PORT_BASE:-30000}"
OUT="$("$PYTHON" -c 'from soar_nla.paths import results_dir; print(results_dir("loss_recovered"))')"
LOG_DIR="$OUT/logs"
NUM_SHARDS="${#GPUS[@]}"
mkdir -p "$LOG_DIR"

wait_all() {
  local status=0 pid
  for pid in "$@"; do
    wait "$pid" || status=1
  done
  return "$status"
}

prepare_stage() {
  : "${ALPACA:?set ALPACA to the google-gemma-3-12b-it/alpaca.parquet of the Lie Detection Rollouts dataset}"
  "$PYTHON" -m soar_nla.loss_recovered.prepare --alpaca "$ALPACA" --output-dir "$OUT"
}

extract_stage() {
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" -m soar_nla.loss_recovered.extract --output-dir "$OUT" \
    >"$LOG_DIR/extract.log" 2>&1
}

verbalize_stage() {
  local checkpoint pids=() urls=() shard port
  checkpoint="$("$PYTHON" -c 'from soar_nla.hub import resolve_snapshot, KITFT_L32_AV; print(resolve_snapshot(*KITFT_L32_AV))')"
  for shard in "${!GPUS[@]}"; do
    port="$((AV_PORT_BASE + shard))"
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$SGLANG_PYTHON" -m sglang.launch_server \
      --model-path "$checkpoint" --port "$port" --host 127.0.0.1 --trust-remote-code \
      --mem-fraction-static 0.55 --context-length 512 --disable-cuda-graph --disable-radix-cache \
      >"$LOG_DIR/av_server_$shard.log" 2>&1 &
    pids+=("$!")
    urls+=("http://127.0.0.1:$port")
  done
  trap 'kill "${pids[@]}" 2>/dev/null || true; wait "${pids[@]}" 2>/dev/null || true' EXIT
  for url in "${urls[@]}"; do
    "$PYTHON" -m soar_nla.sglang_server "$url" --timeout 1800
  done
  "$PYTHON" -m soar_nla.loss_recovered.verbalize --output-dir "$OUT" \
    --urls "$(IFS=,; echo "${urls[*]}")" >"$LOG_DIR/verbalize.log" 2>&1
  kill "${pids[@]}" 2>/dev/null || true
  wait "${pids[@]}" 2>/dev/null || true
  trap - EXIT
}

reconstruct_stage() {
  local pids=() shard
  for shard in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PYTHON" -m soar_nla.loss_recovered.reconstruct worker \
      --output-dir "$OUT" --shard "$shard" --num-shards "$NUM_SHARDS" \
      >"$LOG_DIR/reconstruct_$shard.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
  "$PYTHON" -m soar_nla.loss_recovered.reconstruct merge --output-dir "$OUT" --num-shards "$NUM_SHARDS"
}

evaluate_stage() {
  local pids=() shard
  for shard in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PYTHON" -m soar_nla.loss_recovered.evaluate worker \
      --output-dir "$OUT" --shard "$shard" --num-shards "$NUM_SHARDS" \
      >"$LOG_DIR/evaluate_$shard.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
  "$PYTHON" -m soar_nla.loss_recovered.evaluate merge --output-dir "$OUT" --num-shards "$NUM_SHARDS"
}

numbers_stage() {
  "$PYTHON" -m soar_nla.loss_recovered.paper_numbers --results-dir "$OUT"
  "$PYTHON" -m soar_nla.loss_recovered.paper_figure --numbers "$OUT/paper_numbers.json" --out-dir "$OUT/figures"
}

case "$STAGE" in
  prepare) prepare_stage ;;
  extract) extract_stage ;;
  verbalize) verbalize_stage ;;
  reconstruct) reconstruct_stage ;;
  evaluate) evaluate_stage ;;
  numbers) numbers_stage ;;
  all)
    prepare_stage
    extract_stage
    verbalize_stage
    reconstruct_stage
    evaluate_stage
    numbers_stage
    ;;
  *)
    echo "usage: $0 {prepare|extract|verbalize|reconstruct|evaluate|numbers|all}" >&2
    exit 2
    ;;
esac
