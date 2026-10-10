# Shared settings for the Qwen2.5-7B-Instruct block-20 warm-start ablation.
# Sourced by prepare_data.sh and run_{A,B,C}_*.sh; every value below can be
# overridden from the environment.
#
#   DATA       dataset dir from prepare_data.sh (rl_shuf.full.parquet, *_sft_shuf.full.{train,val}.parquet + sidecars)
#   WORK       root for checkpoints; logs go to $LOG_DIR (default $WORK/logs)
#   PYTHON     interpreter of the main venv (SFT, merges, cold-critic init, regeneration)
#   RL_PYTHON  interpreter of the vllm-lens venv (nla.train_rl_vllm); see docs/vllm-lens-setup.md
#   GPUS       GPU list; the paper runs used one node with 8 x A100-80GB, all eight GPUs
#
# The RL batch (256 prompts x 8 samples) is split across the data-parallel
# ranks, so a different GPU count changes the per-rank batch and the sampling
# streams; use 8 GPUs to reproduce the reported curves.

TRAINING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$TRAINING_DIR"
source "$TRAINING_DIR/env.sh"

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
DATA=${DATA:?set DATA to the dataset directory written by prepare_data.sh}
WORK=${WORK:?set WORK to the directory that receives checkpoints and logs}
PYTHON=${PYTHON:-python}
RL_PYTHON=${RL_PYTHON:-python}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=$GPUS
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
LOG_DIR=${LOG_DIR:-$WORK/logs}
[ "${SKIP_WORK_DIR:-0}" = 1 ] || mkdir -p "$WORK" "$LOG_DIR"

RL_DATA="$DATA/rl_shuf.full.parquet"
# Cold-start AR: raw truncated base (blocks 0..20) + identity value head, no
# task SFT. Runs B and C share one copy (B's run used C's).
COLD_AR=${COLD_AR:-$WORK/cold_init/ar_hf}

if [ "$NPROC" != 8 ]; then
  echo "WARNING: $NPROC GPUs; the reported runs used 8 data-parallel ranks." >&2
fi

torchrun_main() { "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$@"; }
torchrun_rl() { "$RL_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$@"; }

init_cold_critic() {
  if [ ! -f "$COLD_AR/config.json" ] || [ ! -f "$COLD_AR/value_head.safetensors" ]; then
    "$PYTHON" scripts/init_cold_critic.py \
      --base-ckpt "$MODEL" \
      --sidecar "$RL_DATA" \
      --out-dir "$COLD_AR"
  fi
}
