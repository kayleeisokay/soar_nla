# GCG on AV explanations

Reproduces the paper's **GCG** section (Methods) and the GCG reference number in
**Faithfulness** ("still lower than GCG on the same examples (8.4% when β=0.01)").

Starting from the official Gemma-3-12B L32 AV explanation of each shared Alpaca
activation (`soar_nla.alpaca`), Greedy Coordinate Gradient substitutes explanation
tokens to minimize

```
L(x) = MSE(normalize(AR(x)), normalize(a)) / Var(v) + β · mean_token_NLL_AV(x | a),   β ∈ {0.1, 0.01, 0}
```

with 20 steps, 64 candidates per step and top-k = 64, over all explanation tokens; the
AR/AV prompt wrappers stay fixed. Every greedy decision uses the exact string path (decode,
re-tokenize the full AR prompt); see the `gcg.py` docstring for details.

**FVE.** `ar_loss = MSE / Var(v)`, where both vectors are rescaled to the AR's `mse_scale`
(`MSE = 2(1 − cos)`) and `Var(v) = 0.029737` is the empirical scalar variance of the 2,000
rescaled Alpaca activations. Per-example `FVE = 1 − ar_loss`; GCG gain = mean of
`original_ar_loss − optimized_ar_loss`.

**Control** (`controls.py`). For each example and β the control makes the same number of
token edits as the final GCG explanation, at uniformly random positions with uniformly random
non-special vocabulary tokens, and the AR scores original, GCG and control texts. The control
was run on the 1,573 rollouts whose sampled position is after sequence token 50.

## Which examples

The paper reports GCG on the **1,573** Alpaca rollouts whose sampled position is after
sequence token 50 (`soar_nla.alpaca.subsets.late_positions`; the NLA was trained on
positions 50 and later). The random control was run on the same set. The saved GCG run
optimized all 2,000 rollouts; per-example results do not depend on which other examples are in
the run (candidate seeds are per `(sample_id, β)` and `Var(v)` is always computed over all 2,000
activations), so the 1,573-row numbers are exactly what a 1,573-only run would give.
`paper_numbers.py` also reports all 2,000 and the 1,543 `long_responses` rollouts for reference.

**Fluency** (`perplexity.py`). Besides the AV-policy NLL the objective penalizes, each original
and GCG explanation (β = 0.01 and 0) is scored as bare text under Gemma-3-12B-IT itself: mean
per-token NLL after an 8-token warmup, no activation, prompt or chat template.

## Setup and hardware

- Needs `soar_nla.alpaca` outputs: `rollouts.jsonl`, `activations.safetensors`, `explanations.jsonl`.
- GCG: one worker per 80 GB GPU, AR and AV co-located in bf16. The paper run used 8 A100-80GB
  workers and took about 17 hours for 2,000 examples × 3 betas. A worker can also use two GPUs
  (`CUDA_VISIBLE_DEVICES=0,1`: AR on the first, AV on the second).
- The control draw is CPU-only (tokenizer); its AR scoring needs one GPU for a few minutes.
- Fluency scoring loads Gemma-3-12B-IT on one GPU.
- `paper_numbers.py` and `plots.py` are CPU-only.

## Commands

```bash
bash soar_nla/gcg/run.sh gcg       # 8 GPU workers, then merge -> results/gcg/results.jsonl, summary.json
bash soar_nla/gcg/run.sh control   # unconditional-random -> score-ar (1 GPU) -> analyze
bash soar_nla/gcg/run.sh perplexity  # Gemma-3-12B-IT NLL of original and GCG texts (1 GPU)
bash soar_nla/gcg/run.sh figure    # gcg_tradeoff.png
bash soar_nla/gcg/run.sh numbers   # paper_numbers.json (+ gcg_losses.jsonl.gz projection)
```

Individual entry points: `python -m soar_nla.gcg.gcg {worker,merge}`,
`python -m soar_nla.gcg.controls {unconditional-random,score-ar,analyze}`,
`python -m soar_nla.gcg.perplexity`, `python -m soar_nla.gcg.plots`, `python -m soar_nla.gcg.paper_numbers`.
Without arguments, `plots` and `paper_numbers` run on the shipped files below.

## Outputs (`$SOAR_RESULTS/gcg`, default `results/gcg`)

| File | Content | Shipped |
|---|---|---|
| `results.jsonl` | per `(sample_id, β)`: original/optimized text, losses, full trajectory | no (45 MB) |
| `gcg_losses.jsonl.gz` | projection of `results.jsonl` to the per-example losses, AV NLLs and edit counts | yes |
| `perplexity/results.jsonl[.gz]` | Gemma-3-12B-IT NLL of original and GCG texts (1,573 × {original, β=0.01, β=0}) | yes (`.gz`) |
| `summary.json` | per-β means over all 2,000 | yes |
| `gcg_tradeoff.png` | reconstruction vs AV-NLL frontier (n = 1,573 per β) | yes |
| `random_control/random/results.jsonl` | control texts | no (13 MB) |
| `random_control/ar_scores/kitft.jsonl[.gz]` | AR losses/FVE for original, GCG and control | yes (`.gz`) |
| `random_control/summary.json`, `REPORT.md` | bootstrap means and 95% CIs | yes |
| `paper_numbers.json` | everything in the table below | yes |

The shipped `kitft.jsonl.gz` is the paper run's file with the local checkpoint path in
`scorer_checkpoint` replaced by `repo@revision`.

## Paper figures

```bash
pip install -e ".[figures]"          # Pillow, uharfbuzz, fontTools (Devanagari token in the example)
python -m soar_nla.gcg.paper_figure  # CPU only, reads the shipped files
```

Fig. 3 of the paper is `figures/gcg_fve_and_example.png`: (a) `figures/gcg_beta_fve.{png,pdf}`,
GCG and random-control FVE per β with bootstrap 95% CIs from `random_control/summary.json`
(n = 1,573; dotted line = original AV explanations); (b–d) `figures/gcg_example_beta0.01.{png,pdf}`,
the record in `qualitative_example.txt` (sample `0d689d100cf3550e2a8a`, β = 0.01) with the changed
words highlighted and its two FVEs from `gcg_losses.jsonl.gz` (0.745 → 0.908).

## Paper claims

From `paper_numbers.json` (`gcg.late_positions.<β>` unless noted), computed from the per-example
files; n = 1,573 per β. "±" is the standard error of the mean.

| Claim | Output field | Value |
|---|---|---|
| Original AV explanations: FVE 72.1% ± 0.5% | `original_fve`, `original_fve_se` | 0.7211 ± 0.0054 |
| GCG β = 0: FVE 80.7% | `0.0.gcg_fve`, `gcg_fve_se` | 0.8065 ± 0.0042 |
| GCG β = 0.01: FVE 81.4% (highest) | `0.01.gcg_fve`, `gcg_fve_se` | 0.8141 ± 0.0040 |
| GCG β = 0.1: FVE 77.2% (Fig. 3) | `0.1.gcg_fve` | 0.7721 ± 0.0042 |
| Mean NLL under Gemma-3-12B-IT: original 3.84, β = 0.01 4.63, β = 0 5.58 nats/token | `original_gemma_nll`, `gcg_gemma_nll` | 3.838 / 4.629 / 5.577 |
| AV-policy NLL (the penalized term): original → GCG, β = 0.1 / 0.01 / 0 | `mean_original_av_nll`, `mean_optimized_av_nll` | 0.353 → 0.608 / 2.352 / 4.296 |
| Mean edit counts 4.36 / 14.36 / 16.41 (β = 0.1 / 0.01 / 0) | `mean_edit_count` | 4.36 / 14.36 / 16.41 |
| FVE gain: GCG vs same number of random edits, β = 0.1 / 0.01 / 0 | `gcg_fve_gain`; `random_control.late_positions.<β>.random_fve_gain` | +0.0510 / +0.0930 / +0.0854 vs −0.0180 / −0.0721 / −0.0871 |
| GCG explanations are not more faithful: 14.3% vs 17.5% supported (β = 0.01) | `soar_nla.faithfulness.paper_numbers` | 14.3% vs 17.5% (first two paragraphs, 3,298 sentences each) |
| GCG on the same examples as the surgical edits: 8.4% at β = 0.01 | `paper_claims["gcg_beta0.01_gain_on_sol_word_tokens"]` | 0.0844 (n = 143) |

The "same examples" for the 8.4% are the 143 of the 200 surgically edited rollouts whose sampled
token contains a letter or digit (see `soar_nla/faithfulness/surgical_edits/README.md`).
For reference, the FVE gains on all 2,000 rollouts are +0.0506 / +0.0918 / +0.0846 and on the
1,543 `long_responses` rollouts +0.0480 / +0.0893 / +0.0822.
