# EasyNLA environment — source this before training.
#
# CUDA forward compatibility (optional)
# -------------------------------------
# The paper runs used cu128 wheels on a node whose kernel driver is on the CUDA
# 12.2 branch. vLLM exercises the VMM / IPC driver APIs for its weight sync, so
# an exact-matching CUDA 12.8 forward-compat userspace (libcuda 570.211.01, from
# the cuda-compat-12-8 package, installable without root) was put first on
# LD_LIBRARY_PATH. Point CUDA_COMPAT_DIR at that directory if your driver is
# older than the wheels; leave it unset otherwise.
if [ -n "${CUDA_COMPAT_DIR:-}" ]; then
  export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# The IPC weight sync in train_rl_vllm needs the legacy CUDA allocator —
# expandable_segments breaks it (see README).
unset PYTORCH_CUDA_ALLOC_CONF

# vLLM v1 runs EngineCore in a subprocess; the IPC weight sync ships a
# functools.partial through apply_model, which msgspec refuses without this.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
