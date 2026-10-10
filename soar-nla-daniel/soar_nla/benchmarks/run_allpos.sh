#!/usr/bin/env bash
# All-prompt-token paraphrase and random-vector arms (see allpos.py), run
# after run_gsm8k.sh, run_ifeval.sh and run_arc_easy.sh. Resumable: completed
# paraphrase rows, reconstruction shards and rollout shards are skipped.
#
# Environment (see shards.sh): PYTHON, VLLM_PYTHON, GPUS, plus
#   RUNS  directory holding the gsm8k/, ifeval/, arc_easy/ runs; the arms are
#         written to RUNS/allpos (default: results/benchmarks)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/shards.sh"
RUNS=${RUNS:-$(results_root)}
ARMS_DIR="$RUNS/allpos"
tasks=(ifeval gsm8k arc_easy)
para_arms=(para_medium para_heavy)
dirs=(--gsm8k-dir "$RUNS/gsm8k" --ifeval-dir "$RUNS/ifeval" --arc-easy-dir "$RUNS/arc_easy" --arms-dir "$ARMS_DIR")

mkdir -p "$ARMS_DIR/logs"

wait_all() {
    local label="$1"
    shift
    local failed=0 pid
    for pid in "$@"; do
        if ! wait "$pid"; then failed=1; fi
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "$label failed; inspect logs under $ARMS_DIR" >&2
        exit 1
    fi
}

rollout() {
    local task="$1" arm="$2"
    local dir="$ARMS_DIR/${task}__${arm}"
    local python_bin extra
    case "$task" in
        gsm8k) python_bin="$PYTHON"; extra=(--max-new-tokens 4096 --batch-size 8) ;;
        ifeval) python_bin="$VLLM_PYTHON"; extra=(--max-new-tokens 4096 --batch-size 4) ;;
        arc_easy) python_bin="$VLLM_PYTHON"; extra=(--max-new-tokens 32 --batch-size 16) ;;
    esac
    mkdir -p "$dir/logs"
    local pids=() i
    for i in "${!GPU_LIST[@]}"; do
        [[ -e "$dir/patched_shard_$(printf %02d "$i").jsonl" ]] && continue
        CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" PYTHONUNBUFFERED=1 \
            "$python_bin" -m "soar_nla.benchmarks.$task" patched --out-dir "$dir" \
            --activations-dir "$RUNS/$task/activations" \
            --shard "$i" --num-shards "$NUM_SHARDS" "${extra[@]}" \
            >"$dir/logs/patched_${i}.log" 2>&1 &
        pids+=("$!")
    done
    wait_all "rollout $task/$arm" "${pids[@]}"
    echo "$(date '+%F %T') rollout done: $task/$arm"
}

echo "$(date '+%F %T') setup"
"$VLLM_PYTHON" -m soar_nla.benchmarks.allpos setup "${dirs[@]}"
"$VLLM_PYTHON" -m soar_nla.benchmarks.allpos random "${dirs[@]}" >"$ARMS_DIR/logs/random.log" 2>&1

# Random arm first: it needs no paraphrases and validates the rollout path.
for task in "${tasks[@]}"; do rollout "$task" random_vec; done

echo "$(date '+%F %T') paraphrase"
pids=()
for i in "${!GPU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" PYTHONUNBUFFERED=1 \
        "$VLLM_PYTHON" -m soar_nla.benchmarks.allpos paraphrase "${dirs[@]}" --shard "$i" \
        >>"$ARMS_DIR/logs/paraphrase_${i}.log" 2>&1 &
    pids+=("$!")
done
wait_all paraphrase "${pids[@]}"

for task in "${tasks[@]}"; do
    for arm in "${para_arms[@]}"; do
        dir="$ARMS_DIR/${task}__${arm}"
        echo "$(date '+%F %T') reconstruct $task/$arm"
        pids=()
        for i in "${!GPU_LIST[@]}"; do
            [[ -e "$dir/reconstructions/shard_$(printf %02d "$i").safetensors" ]] && continue
            CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" PYTHONUNBUFFERED=1 \
                "$VLLM_PYTHON" -m soar_nla.benchmarks.roundtrip reconstruct --out-dir "$dir" \
                --activations-dir "$RUNS/$task/activations" --shard "$i" --batch-size 32 \
                >"$dir/logs/reconstruct_${i}.log" 2>&1 &
            pids+=("$!")
        done
        wait_all "reconstruct $task/$arm" "${pids[@]}"
    done
done

for task in "${tasks[@]}"; do
    for arm in "${para_arms[@]}"; do rollout "$task" "$arm"; done
done

echo "$(date '+%F %T') summarize"
"$VLLM_PYTHON" -m soar_nla.benchmarks.allpos summarize "${dirs[@]}" >"$ARMS_DIR/logs/summarize.log" 2>&1
echo "$(date '+%F %T') finished: $ARMS_DIR"
