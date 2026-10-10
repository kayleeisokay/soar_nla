#!/usr/bin/env bash
# GSM8K all-prompt-token NLA round trip with the paper settings.
# extract (layer-32 states, right padding) -> AV decode on one SGLang server
# per GPU -> AR reconstruct -> matched clean / identity / patched rollouts
# (greedy, 4096 new tokens) -> summary.
#
# Environment (see shards.sh): PYTHON, VLLM_PYTHON, GPUS, plus
#   SOAR_SGLANG_PYTHON  interpreter with the patched SGLang (default: $PYTHON)
#   BASE_PORT           AV server for shard i listens on BASE_PORT+i (default: 31000)
#   OUT                 run directory (default: results/benchmarks/gsm8k)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/shards.sh"
export SOAR_SGLANG_PYTHON=${SOAR_SGLANG_PYTHON:-$PYTHON}
BASE_PORT=${BASE_PORT:-31000}
OUT=${OUT:-$(results_root)/gsm8k}
LOGS="$OUT/logs"
mkdir -p "$LOGS"

# The paper run extracted in the vLLM environment and ran everything else in
# the main environment.
run_shards extract "$LOGS" "$VLLM_PYTHON" -m soar_nla.benchmarks.gsm8k extract \
    --out-dir "$OUT" --num-shards "$NUM_SHARDS" --batch-size 8

AV_DIR=$("$PYTHON" -c 'from soar_nla.benchmarks.common import av_path; print(av_path())')
server_pids=()
stop_servers() {
    local pid
    for pid in "${server_pids[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
}
trap stop_servers EXIT
for i in "${!GPU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=1 \
        setsid "$SOAR_SGLANG_PYTHON" -m sglang.launch_server --model-path "$AV_DIR" \
        --port $((BASE_PORT + i)) --host 127.0.0.1 \
        --disable-radix-cache --mem-fraction-static 0.80 --context-length 512 \
        --attention-backend triton --disable-cuda-graph --trust-remote-code --dtype bfloat16 \
        --log-level warning >"$LOGS/sglang_${i}.log" 2>&1 &
    server_pids+=("$!")
done
for i in "${!GPU_LIST[@]}"; do
    "$PYTHON" -m soar_nla.sglang_server "http://127.0.0.1:$((BASE_PORT + i))" --timeout 1800
done
pids=()
for i in "${!GPU_LIST[@]}"; do
    PYTHONUNBUFFERED=1 "$PYTHON" -m soar_nla.benchmarks.roundtrip decode --out-dir "$OUT" --shard "$i" \
        --sglang-url "http://127.0.0.1:$((BASE_PORT + i))" --workers 16 --max-new-tokens 192 \
        >"$LOGS/decode_${i}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
stop_servers
server_pids=()

run_shards reconstruct "$LOGS" "$PYTHON" -m soar_nla.benchmarks.roundtrip reconstruct \
    --out-dir "$OUT" --batch-size 64
for condition in clean identity patched; do
    run_shards "$condition" "$LOGS" "$PYTHON" -m soar_nla.benchmarks.gsm8k "$condition" \
        --out-dir "$OUT" --num-shards "$NUM_SHARDS" --max-new-tokens 4096 --batch-size 8
done
"$PYTHON" -m soar_nla.benchmarks.gsm8k summarize --out-dir "$OUT" \
    --num-shards "$NUM_SHARDS" --max-new-tokens 4096 >"$LOGS/summary.log" 2>&1
echo "Finished: $OUT"
