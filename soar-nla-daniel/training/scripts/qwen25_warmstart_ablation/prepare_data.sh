#!/usr/bin/env bash
# Training data for the Qwen2.5-7B-Instruct block-20 NLAs (runs A, B, C).
#
# Text, prompts and gold explanations are reused unchanged from the Qwen3-8B
# FineFineWeb-100k NLA dataset (100k docs, 10 positions/doc at >= 50 tokens,
# explanations by Claude Sonnet 4.6; built with nla.datagen.run_pipeline and
# configs/datagen/qwen3_8b_finefineweb_100k.yaml). That dataset ships text only,
# and Qwen2.5-7B-Instruct tokenizes the corpus identically to Qwen3-8B (same
# marker token id), so only the activations are recomputed: output of decoder
# block 20 (= HF hidden_states[21]) at the final token, raw / unnormalized.
# ~0.003% of rows are dropped (text ends mid-multibyte-character).
#
# Two ways to get the same files:
#   MODE=download (default)  fetch the regenerated dataset as published
#   MODE=regenerate          rebuild it from the text-only source (~40 min on 8 GPUs)
#
#   DATA=/path/to/data25 bash scripts/qwen25_warmstart_ablation/prepare_data.sh
#
# Output in $DATA: {av_sft,ar_sft,rl}_shuf.full.parquet, the doc-disjoint
# {av,ar}_sft_shuf.full.{train,val}.parquet splits (val_split.is_val_doc, crc32
# on doc_id) and a .nla_meta.yaml sidecar next to every parquet.
set -euo pipefail
WORK=${WORK:-${DATA:?set DATA to the output dataset directory}} SKIP_WORK_DIR=1
source "$(dirname "$0")/_common.sh"

MODE=${MODE:-download}
LAYER=${LAYER:-20}
# Published regenerated dataset and its text-only source, at the revisions used.
DATA_REPO=${DATA_REPO:-Yooniel/qwen2.5-7b-instruct-nla-L20-finefineweb-100k}
DATA_REV=${DATA_REV:-811f4afffe7fc32e7bd217f0174462f7c07ff03a}
SRC_REPO=${SRC_REPO:-ceselder/qwen3-8b-nla-L24-finefineweb-100k}
SRC_REV=${SRC_REV:-bcb2f759fb9abf7572ab5d156eb9736d0940fcb9}
SRC=${SRC:-$DATA/src_text}
mkdir -p "$DATA"

hf_get() {  # repo revision local_dir
  "$PYTHON" -c 'import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], repo_type="dataset", revision=sys.argv[2], local_dir=sys.argv[3]))' "$@"
}

if [ "$MODE" = download ]; then
  hf_get "$DATA_REPO" "$DATA_REV" "$DATA"
  exit 0
fi

[ "$MODE" = regenerate ] || { echo "MODE must be download or regenerate" >&2; exit 2; }
hf_get "$SRC_REPO" "$SRC_REV" "$SRC"
for f in av_sft_shuf ar_sft_shuf rl_shuf; do
  out="$DATA/$f.full.parquet"
  # Resume-safe: a completed split is left alone.
  if [ -s "$out" ]; then
    echo "########## $f — already done, skipping ##########"
    continue
  fi
  rm -rf "$out.shards"
  "$PYTHON" -u scripts/regenerate_activations.py \
    --in "$SRC/$f.parquet" --out "$out" \
    --base-model "$MODEL" --layer "$LAYER" --gpus "$NPROC"
done
"$PYTHON" scripts/make_val_split.py --in "$DATA/av_sft_shuf.full.parquet"
"$PYTHON" scripts/make_val_split.py --in "$DATA/ar_sft_shuf.full.parquet"
ls -la "$DATA"/*.parquet
