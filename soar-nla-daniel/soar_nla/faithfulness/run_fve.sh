#!/usr/bin/env bash
# Warm-start FVE pipeline, as run for the paper: one shared eval set and activation file,
# then AV decode (sharded one GPU per shard) and AR reconstruction for each checkpoint pair.
set -euo pipefail

PYTHON=${PYTHON:-python}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
RESULTS=${RESULTS:-$($PYTHON -c 'from soar_nla.paths import results_dir; print(results_dir("faithfulness") / "fve")')}
SHARDS=${#GPUS[@]}
FVE="$PYTHON -m soar_nla.faithfulness.fve"

mkdir -p "$RESULTS/logs"

# The eval set and the target activations do not depend on the checkpoint pair:
# build them once and share them between rl/ and sft/.
$FVE dataset --spec rl --results-dir "$RESULTS"
CUDA_VISIBLE_DEVICES=${GPUS[0]} $FVE extract --spec rl --results-dir "$RESULTS" --device cuda:0 --batch-size 8
mkdir -p "$RESULTS/sft"
cp -r "$RESULTS/rl/dataset" "$RESULTS/rl/extract" "$RESULTS/sft/"

for SPEC in rl sft; do
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $FVE decode --spec "$SPEC" --results-dir "$RESULTS" \
      --device cuda:0 --batch-size 16 --temperature 0.0 --max-new-tokens 256 \
      --shard "$i" --num-shards "$SHARDS" > "$RESULTS/logs/decode_${SPEC}_$i.log" 2>&1 &
  done
  wait
  $FVE merge-decode --spec "$SPEC" --results-dir "$RESULTS" --num-shards "$SHARDS"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $FVE reconstruct --spec "$SPEC" --results-dir "$RESULTS" \
    --device cuda:0 --batch-size 32
  $FVE analyze --spec "$SPEC" --results-dir "$RESULTS"
done
