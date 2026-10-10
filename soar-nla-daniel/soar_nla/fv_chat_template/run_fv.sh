#!/usr/bin/env bash
# English->French function-vector replication, raw prompt vs chat template.
# Fetch data -> sampling plan (CPU) -> 2-shot baseline -> 2-shot intervention
# arms -> 1- and 4-shot baselines -> figures and paper-number check.
#
# `run` resumes from an existing rows.json, so the default output root is a
# fresh directory next to the committed results (results/fv_chat_template/
# shot{1,2,4}); set OUT_ROOT to write elsewhere.
#
# Environment:
#   PYTHON              interpreter with torch/transformers/matplotlib (default: python)
#   SOAR_SGLANG_PYTHON  interpreter with the patched SGLang (default: $PYTHON)
#   OUT_ROOT            output root (default: results/fv_chat_template/rerun)
#   BASELINE_GPU        GPU for the baseline phases (default: 4)
#   TARGET_GPUS         target model, then float32 AR, for the full phase (default: 4,5)
#   AV_GPUS             tensor-parallel AV server (default: 6,7)
set -euo pipefail

PYTHON=${PYTHON:-python}
export SOAR_SGLANG_PYTHON=${SOAR_SGLANG_PYTHON:-$PYTHON}
RESULTS=$("$PYTHON" -c 'from soar_nla.paths import results_dir; print(results_dir("fv_chat_template"))')
OUT_ROOT=${OUT_ROOT:-$RESULTS/rerun}
BASELINE_GPU=${BASELINE_GPU:-4}
TARGET_GPUS=${TARGET_GPUS:-4,5}
AV_GPUS=${AV_GPUS:-6,7}

"$PYTHON" -m soar_nla.fv_chat_template.fetch_data
"$PYTHON" -m soar_nla.fv_chat_template.prepare
"$PYTHON" -m soar_nla.fv_chat_template.run --phase baseline --shots 2 \
    --out-dir "$OUT_ROOT/shot2" --baseline-gpu "$BASELINE_GPU"
"$PYTHON" -m soar_nla.fv_chat_template.run --phase full --shots 2 \
    --out-dir "$OUT_ROOT/shot2" --target-gpus "$TARGET_GPUS" --av-gpus "$AV_GPUS"
for shots in 1 4; do
    "$PYTHON" -m soar_nla.fv_chat_template.run --phase baseline --shots "$shots" \
        --out-dir "$OUT_ROOT/shot$shots" --baseline-gpu "$BASELINE_GPU"
done
"$PYTHON" -m soar_nla.fv_chat_template.figures --shot1-dir "$OUT_ROOT/shot1" \
    --shot2-dir "$OUT_ROOT/shot2" --shot4-dir "$OUT_ROOT/shot4" --figures-dir "$OUT_ROOT/figures"
"$PYTHON" -m soar_nla.fv_chat_template.paper_numbers --rows "$OUT_ROOT/shot2/rows.json"
