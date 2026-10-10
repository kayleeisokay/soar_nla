# Shared settings for the custom Gemma-3-12B-IT NLA launchers (reference only).
#
# These launchers need the trainer fork they were run with,
# github.com/chand-ab/easy_nla at commit 479da2f; the vendored trainer in
# training/ does not accept several of their flags (see README.md).
#
#   LAYER      24, 32 or 40
#   EASY_NLA   checkout of chand-ab/easy_nla @ 479da2f (its own .venv + env.sh)
#   WORK       root for data, checkpoints and logs
#   PYTHON     main-venv interpreter (data, SFT, merge)
#   RL_PYTHON  vllm-lens-venv interpreter (RL)

LAYER=${LAYER:?set LAYER to 24, 32 or 40}
case $LAYER in 24|32|40) ;; *) echo "LAYER must be 24, 32 or 40" >&2; exit 2 ;; esac
EASY_NLA=${EASY_NLA:?set EASY_NLA to a checkout of github.com/chand-ab/easy_nla at 479da2f}
WORK=${WORK:?set WORK to the directory for data, checkpoints and logs}
PYTHON=${PYTHON:-python}
RL_PYTHON=${RL_PYTHON:-python}

BASE=google/gemma-3-12b-it
SRC=${SRC:-$WORK/data/src_qwen3_L24}          # text-only Qwen3-8B source dataset
DATA=${DATA:-$WORK/data/gemma12b_L$LAYER}
CKPT=${CKPT:-$WORK/ckpts/gemma12b_L$LAYER}
LOG_DIR=${LOG_DIR:-$WORK/logs}
AR_NUM_LAYERS=$((LAYER + 1))                    # critic = blocks 0..LAYER
mkdir -p "$WORK" "$LOG_DIR"

cd "$EASY_NLA"
source ./env.sh
