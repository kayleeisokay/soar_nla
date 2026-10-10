# Custom Gemma-3-12B-IT NLAs (blocks 24 / 32 / 40) — reference launchers

The custom Gemma-3-12B-IT NLAs used in other sections of the paper
(`soar_nla.hub.CUSTOM_L24`, `CUSTOM_L32`, `CUSTOM_L40`, i.e.
`achand45/gemma-3-12b-it-nla-L{24,32,40}`) were trained with a different fork of
the same trainer: **[github.com/chand-ab/easy_nla](https://github.com/chand-ab/easy_nla)
at commit `479da2f`**. That fork branches from the same base commit as the
vendored trainer in `training/` (`ef66a76`) and adds Gemma-3 and memory/sync
support. The launchers here are cleaned, layer-parameterised copies of the
ones that produced the released checkpoints. They are kept **for reference**:
they do **not** run on the vendored trainer. Run them from a checkout of the
fork (`EASY_NLA=/path/to/easy_nla`).

## Trainer-version mismatch

Flags the launchers pass that the vendored trainer rejects:

| Launcher | Module | Missing in `training/` |
|---|---|---|
| `rl.sh` | `nla.train_rl_vllm` | `--critic-device`, `--nccl-weight-sync`, `--verify-sync`, `--sync-profile`, `--no-ipc-weight-sync` (vendored `--ipc-weight-sync` is `store_true`, with no negated form) |
| `sft.sh` | `nla.train_sft` | `--init-seed`, `--data-seed` (vendored trainer has a single `--seed`) |
| `prepare_data.sh` | `scripts/regenerate_activations.py` | `--cross-model` |
| `prepare_data.sh` | `scripts/retarget_sidecar_tokens.py` | the whole script |

There are also behavioural differences that raise no error:

- **Gemma-3 text-model resolution.** The vendored trainer does not resolve
  Gemma-3's text model on the RL actor, the AV SFT path or the AV merge. On
  Gemma-3 an `--av-adapter` warm start then loads onto zero keys, and the merge
  merges nothing.
- **`nla/datagen/injection_token_cache.yaml`** has no `google/gemma-3-12b-it`
  entry (marker `㈜`, id 246566).
- **Weight sync.** The fork adds the NCCL weight sync, per-bucket push, and
  critic offload with gradients staged through `cuda:0`.

Fork commits after `479da2f` change only notebooks and the README.

## Recipe

The recipe is identical for all three layers. Data are the text, prompts and
Sonnet gold explanations of `ceselder/qwen3-8b-nla-L24-finefineweb-100k`, with
activations regenerated with Gemma at block `LAYER`.

| | L24 | L32 | L40 |
|---|---|---|---|
| AV SFT | LoRA r128 α16, lr 1e-4, batch 64, 3,834 steps (1 epoch), warmup 50, min lr 2e-6, grad-ckpt, seeds 0/0 | same | same |
| AR SFT | as AV, lr 2e-5, no grad-ckpt, `--ar-num-layers` 25 | 33 | 41 |
| RL | 400 steps, 256 prompts × 8, lr 1e-4, critic lr 8e-5, KL 0.01 (k3), rsLoRA r128 α16, AR LoRA r64 α16, 256 new tokens, length penalty 0.01 past 192, T 1.0, warm start from `av_sft/iter_0003834` | same | same |
| RL topology | 4 DP ranks × vLLM TP 2, critic on the TP partner | same | same |
| `--vllm-gpu-mem` | 0.26 | 0.26 | 0.22 |
| Held-out FVE at step 390 (model card: mean of the last ten evals) | 62.1% (~60.9%) | 69.3% (~68.6%) | 59.9% (~59%) |

Order: `prepare_data.sh` → `sft.sh` → `merge.sh` → `rl.sh`, each with
`LAYER=24|32|40 EASY_NLA=... WORK=... PYTHON=... RL_PYTHON=...`. The released
repositories hold the SFT pair under `av_sft/`, `ar_sft/` and `merged/`, and
the RL run under `rl_vllm/`. The resolved configs are in each repo's
`*/run_config.yaml`.
