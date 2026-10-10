# Shared helpers for the benchmark launchers (sourced, not executed).
#
#   PYTHON       main interpreter: SGLang client, matplotlib (default: python)
#   VLLM_PYTHON  interpreter with vLLM and the IFEval checker dependencies
#                (default: $PYTHON)
#   GPUS         space-separated GPU ids, one shard per GPU (default: 0..7).
#                The paper runs used 8 shards; the shard count changes batch
#                composition, so keep 8 to reproduce them.

PYTHON=${PYTHON:-python}
VLLM_PYTHON=${VLLM_PYTHON:-$PYTHON}
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NUM_SHARDS=${#GPU_LIST[@]}

results_root() {
    "$PYTHON" -c 'from soar_nla.paths import results_dir; print(results_dir("benchmarks"))'
}

# run_shards LABEL LOG_DIR PYTHON_BIN ARGS... : one process per GPU with
# "--shard i" appended; logs to LOG_DIR/LABEL_i.log; fails if any shard fails.
run_shards() {
    local label="$1" log_dir="$2" python_bin="$3"
    shift 3
    mkdir -p "$log_dir"
    local pids=() i
    for i in "${!GPU_LIST[@]}"; do
        CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" PYTHONUNBUFFERED=1 \
            "$python_bin" "$@" --shard "$i" >"$log_dir/${label}_${i}.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0 pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "$label failed; inspect $log_dir/${label}_*.log" >&2
        return 1
    fi
    echo "$(date '+%F %T') $label done"
}
