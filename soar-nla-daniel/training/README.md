# training/ — vendored EasyNLA trainer

This sub-project is the NLA trainer used for the paper's **"Skipping Warm-Start"**
ablation (appendix). It is vendored from EasyNLA at commit `ef66a76`, plus the
local modifications the runs used (see `NOTICE`; MIT, see `LICENSE`). The
package name stays `nla`. Analysis of the trained runs lives in
`soar_nla/warmstart_ablation/`.

The ablation trains three NLAs on **`Qwen/Qwen2.5-7B-Instruct`**, reading the
residual stream after decoder block 20 (`d_model` 3584). The paper text does
not name the model. All three runs share data, sidecar, optimiser and GRPO
settings and differ only in initialisation and KL coefficient:

| Run | AV / AR initialisation | KL β | Launcher |
|---|---|---|---|
| A | warm start: AV + AR SFT on Sonnet-written summaries (one epoch each) | 0.01 | `scripts/qwen25_warmstart_ablation/run_A_warmstart.sh` |
| B | cold start: fresh LoRA on the base model; AR = truncated base + identity head | 0 | `scripts/qwen25_warmstart_ablation/run_B_cold_kl0.sh` |
| C | cold start, as B | 0.01 | `scripts/qwen25_warmstart_ablation/run_C_cold_kl001.sh` |

The paper compares the runs at **step 600**.

## Setup

The runs used one node with 8 × A100-80GB, CUDA 12.8 wheels and two
virtualenvs:

```bash
# main venv: data, SFT, merges, cold-critic init
python -m venv .venv && .venv/bin/pip install -e training/ bitsandbytes
# rollout venv: vllm==0.19.0 + vllm-lens==1.1.0, patched for norm-matched injection
bash training/scripts/install_vllm_lens.sh /path/to/vllm-lens-venv   # needs `uv`
```

The launchers `cd` into `training/`, so `python -m nla....` resolves the
package from the working directory in both venvs.

Read `docs/vllm-lens-setup.md` before changing the version pins. The patch
(`utils/patch_vllm_lens.py`) is required: an unpatched vllm-lens injects far
too weakly. If your driver is older than the CUDA 12.8 wheels, point
`CUDA_COMPAT_DIR` at a cuda-compat-12-8 userspace (see `env.sh`).

The launchers take their settings from the environment:

| Variable | Meaning |
|---|---|
| `DATA` | Dataset directory |
| `WORK` | Checkpoints; logs go to `$WORK/logs` |
| `PYTHON` | Main-venv interpreter |
| `RL_PYTHON` | vllm-lens-venv interpreter |
| `GPUS` | GPU list, default `0,…,7`. The RL batch is split across the 8 data-parallel ranks, so use 8 GPUs to reproduce the curves. |

## Commands

```bash
export DATA=/path/to/data25 WORK=/path/to/work PYTHON=.venv/bin/python RL_PYTHON=/path/to/vllm-lens-venv/bin/python
S=training/scripts/qwen25_warmstart_ablation
bash $S/prepare_data.sh            # MODE=download (default) or MODE=regenerate
bash $S/run_A_warmstart.sh         # stages: sft merge rl1 rl2
bash $S/run_B_cold_kl0.sh          # stages: leg1 leg2
bash $S/run_C_cold_kl001.sh        # stages: leg1 leg2 leg3
```

A stage list can be passed to run part of a script, e.g.
`run_A_warmstart.sh rl2`. Each launcher's header documents its legs. Here is
what was actually run:

| Run | Leg | Steps | Notes | Log |
|---|---|---|---|---|
| A | SFT | AV 3,834 + AR 3,834 | 8-GPU DP, global batch 64, AV lr 1e-4, AR lr 2e-5 | `sft25.log` |
| A | rl1 | 0 → 400 | `--av-adapter` continues the SFT LoRA, whose frozen copy is the KL reference | `rl25.log` |
| A | rl2 | 400 → 800 | resumed from `iter_000400` + `critic_latest` + `optim_latest` | `qwen25_l20_resume400_to800.log` |
| B | leg1 | 0 → 400 | KL 0, strict tag parsing; diverges from ~step 220 (tag never closed, extraction 0%); only 0–199 reported | `rl25_cold_origprompt_kl0.log` |
| B | leg2 | 200 → 600 | actor from leg1 `iter_000200`, AR from leg1 `critic_latest` (step 400), fresh optimizer, `--raw-completion-fallback --score-truncated` | `rl25_cold_origprompt_kl0_rawfallback.log` |
| C | leg1 | 0 → 400 | default β = 0.01 (k3) | `rl25_cold_nosft.log` |
| C | leg2 | 400 → 600 | resumed, same save dir | appended to `rl25_cold_nosft.log` |
| C | leg3 | 600 → 1200, stopped after step 823 | its step-600 eval is C's reported held-out FVE | `rl25_cold_nosft_resume600_1200.log` |

Shared RL settings:

- 256 prompts × 8 samples per step at T 1.0, at most 256 new tokens.
- AV: LoRA r128 α16 with rsLoRA, lr 1e-4. AR: LoRA r64 α16 plus a
  3584 × 3584 head, lr 8e-5, co-trained every step.
- Hinged length penalty of 0.01 per token past 192. AdamW8bit.
- Held-out evaluation every 10 steps on 128 prompts from doc-hash held-out
  documents (10%).
- Checkpoints every 25 steps.
- `--vllm-gpu-mem 0.24`, no gradient checkpointing (tuned for 80 GB).

Approximate wall clock: A RL ~62 s/step; C ~60 s/step; B leg 1 ~40 s/step.

All three runs read the same RL parquet and sidecar (`rl_shuf.full.parquet`).
B and C were launched against a sibling data directory whose SFT files had
reordered summary items. Its RL parquet and sidecar were symlinks to this
file, and cold-start runs never read the SFT files.

The per-step curves and the paper numbers are rebuilt from these logs by
`python -m soar_nla.warmstart_ablation.parse_training_logs` and `fve_table`
(see `soar_nla/warmstart_ablation/README.md`).

## Published data and checkpoints

| What | Hugging Face repo @ revision | Path inside the repo |
|---|---|---|
| Training data (Qwen2.5 block-20 activations + splits) | `Yooniel/qwen2.5-7b-instruct-nla-L20-finefineweb-100k` @ `811f4afffe7fc32e7bd217f0174462f7c07ff03a` | root |
| Text-only source dataset | `ceselder/qwen3-8b-nla-L24-finefineweb-100k` @ `bcb2f759fb9abf7572ab5d156eb9736d0940fcb9` | root |
| A: SFT adapters, merged AV/AR, RL iters 25…800 | `Yooniel/qwen2.5-7b-instruct-nla-L20` @ `8b35afe5186a27dabc7bd41f2cded6f133e4b5e6` | `av_sft/`, `ar_sft/`, `merged/`, `rl_vllm/` |
| B: leg 2 step 600 + critic; leg 1 step 400 | same | `other/ckpts25_cold_origprompt_kl0_rawfallback/rl_vllm/`, `other/ckpts25_cold_origprompt_kl0/rl_vllm/` |
| C: cold AR init, RL iters 25…800 | same | `cold_nosft/` |

The checkpoint repo also contains unrelated experiments. A dedicated release
repository with only the three step-600 checkpoints is still to be created:
`<HF_REPO_PLACEHOLDER>`.

## Code-version note

Run A ran on the trainer code committed as `ef66a76`. Its 400 → 800 leg ran from a clean
checkout of that commit with only the vllm-lens patcher's idempotency check
changed. Runs B and C ran on intermediate states of the local modifications.
The vendored tree is the final state. Every flag used by the three launchers
exists in it, and for single-position data its default code paths match those
states, with one known difference: without `--score-truncated`, a held-out
completion cut off at the token cap now scores as an extraction failure.
Before, it was scored if its tag parsed. This affects only the logged held-out
FVE of A and C, and only for completions that close the tag yet still hit the
cap.

## Layout

```
nla/                          trainer package (sidecar contract, injection, SFT, vLLM GRPO RL, datagen)
configs/rl_vllm.yaml          RL defaults the launchers start from
configs/datagen/              datagen config of the source text dataset (Qwen3-8B FineFineWeb-100k)
utils/patch_vllm_lens.py      required vllm-lens patch (idempotent)
scripts/                      data regeneration, val split, cold AR init, LoRA merge (+ validation), vllm-lens installer
scripts/qwen25_warmstart_ablation/   the three runs (+ prepare_data.sh)
scripts/gemma3_12b_custom/    reference launchers of the custom Gemma-3-12B NLAs (different trainer fork; see its README)
tests/                        CPU tests: python -m pytest tests (with nla installed); test_patcher_baseline needs vllm-lens
```
