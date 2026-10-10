#!/usr/bin/env bash
# IFEval all-prompt-token NLA round trip with the paper settings, entirely in
# the vLLM environment: extract (left padding) -> AV decode with vLLM
# prompt_embeds -> AR reconstruct -> clean / patched rollouts (greedy, 4096
# new tokens) -> summary with Google's checkers.
#
# Environment (see shards.sh): VLLM_PYTHON, GPUS, plus
#   OUT  run directory (default: results/benchmarks/ifeval)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/shards.sh"
OUT=${OUT:-$(results_root)/ifeval}
LOGS="$OUT/logs"

run_shards extract "$LOGS" "$VLLM_PYTHON" -m soar_nla.benchmarks.ifeval extract \
    --out-dir "$OUT" --num-shards "$NUM_SHARDS" --batch-size 8
run_shards decode "$LOGS" "$VLLM_PYTHON" -m soar_nla.benchmarks.roundtrip decode-vllm \
    --out-dir "$OUT" --request-batch-size 32 --max-new-tokens 192
run_shards reconstruct "$LOGS" "$VLLM_PYTHON" -m soar_nla.benchmarks.roundtrip reconstruct \
    --out-dir "$OUT" --batch-size 32
for condition in clean patched; do
    run_shards "$condition" "$LOGS" "$VLLM_PYTHON" -m soar_nla.benchmarks.ifeval "$condition" \
        --out-dir "$OUT" --num-shards "$NUM_SHARDS" --max-new-tokens 4096 --batch-size 4
done
"$VLLM_PYTHON" -m soar_nla.benchmarks.ifeval summarize --out-dir "$OUT" \
    --num-shards "$NUM_SHARDS" --max-new-tokens 4096 >"$LOGS/summary.log" 2>&1
echo "Finished: $OUT"
