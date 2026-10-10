#!/usr/bin/env bash
# Raw-text ICL answer redirection at layers 24 and 40 with the custom EasyNLA pairs (paper appendix
# "Other layers"). Four independent workers, one per (layer, task shard); each worker uses a GPU pair:
# the target model and the AR on the first GPU, the AV (in-process, no SGLang) on the second.
# Results land in results/icl_edits/gemma3-12b-custom-L{24,40}/ (root overridable with $SOAR_RESULTS).
set -euo pipefail
PYTHON=${PYTHON:-python}
L24_GPUS=${L24_GPUS:-"0,1 2,3"}     # shard a, shard b
L40_GPUS=${L40_GPUS:-"4,5 6,7"}
LOGS=${LOGS:-$($PYTHON -c 'from soar_nla.paths import results_dir; print(results_dir("icl_edits") / "logs")')}
mkdir -p "$LOGS"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
SHARD_A="antonym capitalize country-capital"
SHARD_B="english-french present-past singular-plural"
pids=()
for layer in 24 40; do
  gpus=$L24_GPUS; [ $layer = 40 ] && gpus=$L40_GPUS
  read -r gpu_a gpu_b <<< "$gpus"
  for tag in a b; do
    gpu=$gpu_a; tasks=$SHARD_A
    [ $tag = b ] && gpu=$gpu_b && tasks=$SHARD_B
    $PYTHON -u -m soar_nla.icl_edits.run exp --config gemma3-12b-custom-L$layer --gpu $gpu \
        --exp icl_redirect --fmt raw --only $tasks --tag $tag > "$LOGS/l${layer}_raw_icl_redirect_$tag.log" 2>&1 &
    pids+=($!)
  done
done
status=0
for pid in "${pids[@]}"; do wait $pid || status=1; done
if [ $status != 0 ]; then echo "one or more workers failed; see $LOGS" >&2; exit 1; fi
for layer in 24 40; do $PYTHON -m soar_nla.icl_edits.analysis --config gemma3-12b-custom-L$layer; done
