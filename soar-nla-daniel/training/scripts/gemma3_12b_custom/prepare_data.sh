#!/usr/bin/env bash
# Gemma-3-12B-IT block-$LAYER training data (reference; run inside chand-ab/easy_nla @ 479da2f).
#
# Reuses the text, prompts and gold explanations of the Qwen3-8B FineFineWeb-100k
# NLA dataset and recomputes only the activations with Gemma (cross-tokenizer
# retarget: n_raw_tokens recomputed under the Gemma tokenizer, anchored at the
# final token of the same stored text). Then retargets the tokenizer-dependent
# sidecar fields (marker id 246566, neighbours 236813/954) and cuts the
# doc-disjoint SFT val splits.
#
# The original pipeline also gated on a position check before training: for 5
# rows, re-run Gemma with output_hidden_states and require the stored vector to
# match hidden_states[LAYER+1] at the final token (and not the adjacent layer or
# position).
#
#   LAYER=32 EASY_NLA=... WORK=... PYTHON=... bash prepare_data.sh
set -euo pipefail
source "$(dirname "$0")/_common.sh"
SRC_REV=${SRC_REV:-bcb2f759fb9abf7572ab5d156eb9736d0940fcb9}
GPUS=${GPUS:-8}

"$PYTHON" -c 'import sys; from huggingface_hub import snapshot_download as s; s("ceselder/qwen3-8b-nla-L24-finefineweb-100k", repo_type="dataset", revision=sys.argv[1], local_dir=sys.argv[2])' "$SRC_REV" "$SRC"

mkdir -p "$DATA"
for SPLIT in av_sft_shuf ar_sft_shuf rl_shuf; do
  if [ -f "$DATA/$SPLIT.parquet" ]; then echo "$SPLIT already done, skipping"; continue; fi
  rm -rf "$DATA/$SPLIT.parquet.shards"
  "$PYTHON" scripts/regenerate_activations.py \
    --in "$SRC/$SPLIT.parquet" --out "$DATA/$SPLIT.parquet" \
    --base-model "$BASE" --layer "$LAYER" --cross-model --gpus "$GPUS"
done

"$PYTHON" scripts/retarget_sidecar_tokens.py --base-model "$BASE" \
  "$DATA"/av_sft_shuf.parquet.nla_meta.yaml \
  "$DATA"/ar_sft_shuf.parquet.nla_meta.yaml \
  "$DATA"/rl_shuf.parquet.nla_meta.yaml
"$PYTHON" scripts/retarget_sidecar_tokens.py --check --base-model "$BASE" "$DATA"/*.nla_meta.yaml

for SPLIT in av_sft_shuf ar_sft_shuf; do
  [ -f "$DATA/$SPLIT.train.parquet" ] || \
    "$PYTHON" scripts/make_val_split.py --in "$DATA/$SPLIT.parquet" --val-rows 2000
done
ls -la "$DATA"
