# Loss Recovered

Reproduces the paper's **Loss Recovered** section (Methods and Results) and its figure: how much
of Gemma-3-12B-IT's next-token loss survives when the layer-32 residual stream of a token span is
replaced by the NLA round trip, compared with a Gemma Scope 2 SAE.

## Protocol

- **Target**: `google/gemma-3-12b-it` @ `96b6f1e`, residual stream after decoder block 32.
  **NLA**: the official `kitft/nla-gemma3-12b-L32-av` @ `7aec225` / `-ar` @ `3d6901d`
  (pinned in `soar_nla.hub`).
- **Data**: Gemma-3-12B-IT responses to Alpaca prompts from the *Lie Detection Rollouts*
  dataset: subset `google-gemma-3-12b-it`, split `alpaca` (dataset revision `177a816`, file
  SHA-256 `20bafc4a…6b967`, checked by `prepare.py`). Each row is a system prompt
  ("You are an AI assistant."), an Alpaca instruction and the model's full response.
  These are **not** the shared `soar_nla.alpaca` rollouts.
- **Windows** (`prepare.py`): the first 2,000 conversations are rendered with the Gemma chat
  template (the system prompt is merged into the user turn) and tokenized. The 1,543
  conversations with at least 128 tokens each contribute their first 128 tokens; the other 457
  are dropped. Windows never cross a conversation boundary and are never padded.
- **Token spans**: positions count from the start of the chat-rendered sequence, so position 0
  is `<bos>` and the spans cover template, system/user and response tokens alike (no role
  masking). Paper "tokens 1–50" is positions [0, 50), "tokens 1–100" is [0, 100), and "tokens
  51–100 while keeping tokens 1–50 unchanged" is [50, 100). The activations of the whole span
  are replaced jointly. The loss is the next-token cross-entropy of the targets inside the span
  (49, 99 and 49 per window); the model only sees the window's first `stop` tokens.
- **NLA reconstruction**: greedy AV explanation of every window position (≤200 new tokens),
  AR reconstruction of each explanation, rescaled to the norm of the original activation
  (the AR is trained on direction only). `nla_raw` keeps the AR's own norm.
- **SAE baseline**: Gemma Scope 2 `resid_post_all/layer_32_width_16k_l0_small`
  (`google/gemma-scope-2-12b-it` @ `4c419f1`), JumpReLU in bfloat16, mean L0 20.82 on these
  windows. `sae_norm` (the paper's baseline) rescales each decoded vector to the original norm.
- **Metric** (SAEBench): `LR = (H_0 − H_method) / (H_0 − H_orig)`, with `H_orig` the clean
  CE and `H_0` the CE with the span zero-ablated, each a mean over all scored tokens of all
  windows. 95% intervals: 10,000 bootstrap resamples of windows (seed 0, the same resamples
  for every method of a span).

## Setup and hardware

- The paper run used 4 GPUs: one SGLang AV server per GPU for decoding (patched SGLang, see the
  top-level README), then one AR shard and one evaluation shard per GPU. Keep 4 GPUs to
  reproduce the AR reconstructions bit for bit (the shard count fixes which explanations share
  a padded AR batch).
- Decoding the 197,504 window positions is by far the most expensive stage.
- `activations.npy` and `reconstructions.npy` are 3.0 GB each (float32).
- `paper_numbers.py` and `paper_figure.py` are CPU-only.

## Commands

```bash
export ALPACA=/path/to/google-gemma-3-12b-it/alpaca.parquet   # Lie Detection Rollouts
bash soar_nla/loss_recovered/run.sh all     # or one stage at a time:
bash soar_nla/loss_recovered/run.sh prepare     # CPU: windows.npy, windows.jsonl
bash soar_nla/loss_recovered/run.sh extract     # 1 GPU: activations.npy
bash soar_nla/loss_recovered/run.sh verbalize   # 4 SGLang AV servers: explanations.jsonl
bash soar_nla/loss_recovered/run.sh reconstruct # 4 GPUs: reconstructions.npy
bash soar_nla/loss_recovered/run.sh evaluate    # 4 GPUs: window_losses.jsonl.gz, evaluation.json
bash soar_nla/loss_recovered/run.sh numbers     # CPU: paper_numbers.json, figures/loss_recovered.*
```

Individual entry points: `python -m soar_nla.loss_recovered.{prepare,extract,verbalize}`,
`python -m soar_nla.loss_recovered.{reconstruct,evaluate} {worker,merge}`,
`python -m soar_nla.loss_recovered.{paper_numbers,paper_figure}`. `extract` and
`evaluate worker` take `--limit N` for a smoke test (use a separate `--output-dir`).

To check the paper numbers and redraw the figure from the shipped files:

```bash
python -m soar_nla.loss_recovered.paper_numbers
python -m soar_nla.loss_recovered.paper_figure
```

## Outputs (`$SOAR_RESULTS/loss_recovered`, default `results/loss_recovered`)

| File | Content | Shipped |
|---|---|---|
| `windows.npy` | `[1543, 128]` token ids | no (`prepare` rebuilds it) |
| `windows.jsonl`, `preparation.json` | window → parquet row and conversation length; data hash and counts | yes |
| `activations.npy` | `[197504, 3840]` clean layer-32 activations, row = `window * 128 + position` | no (3.0 GB) |
| `explanations.jsonl` | AV explanation per row | no (160 MB) |
| `reconstructions.npy` | raw AR output per row | no (3.0 GB) |
| `window_losses.jsonl.gz` | per window and span: mean CE for `clean`, `zero`, `nla`, `nla_raw`, `sae`, `sae_norm`; per-window SAE L0 and cosine | yes |
| `evaluation.json` | checkpoints, spans, methods, mean SAE L0 / cosine | yes |
| `paper_numbers.json` | mean CEs, loss recovered and 95% CIs per span and method; paper checks | yes |
| `figures/loss_recovered.{png,pdf}` | the paper figure | yes |

The shipped `window_losses.jsonl.gz` was written by `evaluate.py` from the paper run's
activations and AR reconstructions; every per-window CE agrees with the paper run's own
evaluation output to within 4e-15.

## Paper claims

From `paper_numbers.json` (`settings.<span>.loss_recovered.<method>`); n = 1,543 windows.

| Claim | Span / method | Value [95% CI] |
|---|---|---|
| NLA, tokens 51–100 (1–50 clean): 90.88% | `suffix_50_100` / `nla` | 90.88% [90.56, 91.19] |
| NLA, tokens 1–50: 64.37% | `prefix_50` / `nla` | 64.37% [63.40, 65.34] |
| NLA, tokens 1–100: 69.95% | `prefix_100` / `nla` | 69.95% [69.45, 70.44] |
| SAE, 16k width, mean L0 = 20.82, tokens 51–100, norm-rescaled: 97.58% | `suffix_50_100` / `sae_norm`; `sae.mean_l0` | 97.58% [97.46, 97.70]; L0 20.82 |
| 1,543 Alpaca generations | `windows` | 1,543 |

Positions [0, 128) give 69.50% for the NLA. Without norm rescaling
the NLA gives 47.36% / 64.61% / 91.20% and the SAE 128.29% / 106.61% / 97.57% (tokens 1–50 /
1–100 / 51–100). On the spans that start at position 0 the SAE-patched CE is below the clean
CE (4.52 with and 4.31 without norm rescaling vs 6.11 nats on [0, 50)), so its loss recovered
exceeds 100% there.

## Notes

- The NLA was trained on positions ≥ 50, so spans starting at position 0 are partly out of its
  training distribution.
- Every evaluation forward pass runs at batch size one. On the paper run's software (torch 2.9.1,
  transformers 4.57.1) and inputs, `prepare` rebuilds the windows exactly, `extract` (batch 4) and
  `reconstruct` (batch 16, 4 shards) reproduce the saved arrays bit for bit, and `evaluate`
  reproduces every per-window CE to within 4e-15. AV decoding through SGLang is greedy but not
  batch-invariant, so a rerun of `verbalize` can change some explanations.
