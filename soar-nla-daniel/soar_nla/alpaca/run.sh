#!/usr/bin/env bash
# Build the shared Alpaca rollouts: prompt selection, greedy rollouts with one
# layer-32 activation each, and greedy AV decoding of those activations.
#
#   bash soar_nla/alpaca/run.sh {data|rollouts|decode|all}
#
# Environment (defaults are what the paper run used):
#   PYTHON              interpreter with soar_nla importable        (python)
#   SOAR_SGLANG_PYTHON  interpreter with the patched SGLang          ($PYTHON)
#   SOAR_RESULTS        results root; outputs go to $SOAR_RESULTS/alpaca
#   GPUS                one shard per listed GPU                     ("0 1 2 3")
#   N                   number of Alpaca prompts                     (2000)
#   ROLLOUT_BATCH       rollout batch size per GPU                   (4)
#   DECODE_PORT_BASE    SGLang port = base + GPU index               (31000)
set -euo pipefail

PYTHON="${PYTHON:-python}"
SGLANG_PYTHON="${SOAR_SGLANG_PYTHON:-$PYTHON}"
STAGE="${1:-all}"
N="${N:-2000}"
GPUS=(${GPUS:-0 1 2 3})
ROLLOUT_BATCH="${ROLLOUT_BATCH:-4}"
ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-192}"
DECODE_PORT_BASE="${DECODE_PORT_BASE:-31000}"
MODEL=gemma12b
RESULTS_DIR="$("$PYTHON" -c 'from soar_nla.paths import results_dir; print(results_dir("alpaca"))')"
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

snapshot_path() {
  "$PYTHON" -c "from soar_nla.hub import resolve_snapshot, $1; print(resolve_snapshot(*$1))"
}

data_stage() {
  "$PYTHON" -m soar_nla.alpaca.data --output-dir "$RESULTS_DIR/data" --n "$N"
}

rollout_stage() {
  snapshot_path GEMMA3_12B >/dev/null
  local pids=() shard gpu
  for shard in "${!GPUS[@]}"; do
    gpu="${GPUS[$shard]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m soar_nla.alpaca.rollouts worker \
      --model "$MODEL" --data-dir "$RESULTS_DIR/data" --output-dir "$RESULTS_DIR" \
      --n "$N" --shard "$shard" --num-shards "$NUM_SHARDS" --batch-size "$ROLLOUT_BATCH" \
      --max-new-tokens "$ROLLOUT_MAX_NEW_TOKENS" \
      >"$LOG_DIR/rollout_${MODEL}_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
  "$PYTHON" -m soar_nla.alpaca.rollouts merge --model "$MODEL" --output-dir "$RESULTS_DIR" \
    --n "$N" --num-shards "$NUM_SHARDS"
}

decode_one() (
  set -euo pipefail
  local gpu="$1" shard="$2" port="$3" checkpoint="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$checkpoint" --port "$port" --disable-radix-cache \
    --mem-fraction-static 0.82 --context-length 512 --trust-remote-code \
    >"$LOG_DIR/sglang_${MODEL}_${shard}.log" 2>&1 &
  local server_pid="$!"
  trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
  "$PYTHON" -m soar_nla.sglang_server "http://127.0.0.1:$port" --timeout 1200
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m soar_nla.alpaca.decode worker \
    --model "$MODEL" --results-dir "$RESULTS_DIR" --n "$N" \
    --shard "$shard" --num-shards "$NUM_SHARDS" --sglang-url "http://127.0.0.1:$port" \
    >"$LOG_DIR/decode_${MODEL}_${shard}.log" 2>&1
)

decode_stage() {
  local checkpoint pids=() shard gpu
  checkpoint="$(snapshot_path KITFT_L32_AV)"
  for shard in "${!GPUS[@]}"; do
    gpu="${GPUS[$shard]}"
    decode_one "$gpu" "$shard" "$((DECODE_PORT_BASE + gpu))" "$checkpoint" &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
  "$PYTHON" -m soar_nla.alpaca.decode merge --model "$MODEL" --results-dir "$RESULTS_DIR" \
    --n "$N" --num-shards "$NUM_SHARDS"
}

case "$STAGE" in
  data) data_stage ;;
  rollouts) rollout_stage ;;
  decode) decode_stage ;;
  all)
    data_stage
    rollout_stage
    decode_stage
    ;;
  *)
    echo "usage: $0 {data|rollouts|decode|all}" >&2
    exit 2
    ;;
esac
