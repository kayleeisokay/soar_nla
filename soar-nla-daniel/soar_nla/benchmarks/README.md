# Benchmarks using the chat template

Reproduces the paper section **"Benchmarks using chat template"**: how much
Gemma-3-12B-IT's task performance survives when its layer-32 residual stream is
replaced by the NLA round trip at *every prompt token*, on ARC-Easy, GSM8K and
IFEval, compared with paraphrased-explanation and random-vector controls.

## Protocol

* **Target**: `google/gemma-3-12b-it` @ `96b6f1e`; **NLA**: `kitft/nla-gemma3-12b-L32-av` @ `7aec225`
  and `kitft/nla-gemma3-12b-L32-ar` @ `3d6901d` (all pinned in `soar_nla.hub`).
* **Prompt**: Gemma chat template, one user turn, generation prompt appended.
  GSM8K: the question only. IFEval: the IFEval prompt. ARC-Easy: question, all
  labeled choices (`A. ...`), and "Respond with only the letter of the correct answer."
* **Intervention** (`common.py`): during prefill a forward hook on decoder layer 32
  replaces the residual after block 32 at every prompt token with
  `AR(AV(h))`, rescaled to `‖h‖` at that position. `AV(h)` is the greedy AV
  explanation (≤192 new tokens); the AR reads the `<explanation>` text (or the full
  output when the tags are missing). Generated-token activations are not patched.
* **Decoding**: greedy; at most 4096 new tokens (GSM8K, IFEval) or 32 (ARC-Easy).
* **Scoring**: GSM8K exact match of the last number in the response; ARC-Easy exact
  answer letter; IFEval Google's strict checkers (vendored in `ifeval_lib/`), reported as
  the mean of prompt-level and instruction-level strict accuracy.
* **Controls** (`allpos.py`), applied at every prompt position:
  `para_medium` / `para_heavy` replace each position's AV explanation with a
  paraphrase by Gemma-3-12B-IT (the paraphrase prompts of the function-vector protocol,
  raw prompt, greedy, ≤512 tokens, stop at `</transformed_text>`) before the AR and
  the norm matching; `random_vec` uses a seeded Gaussian per position, norm-matched to
  the native residual.
* **Clean baseline**: the same HF runtime without a hook (for GSM8K also an identity
  replay of the captured states, reported in the GSM8K summary only).

Datasets (pinned in `common.DATASETS`): `openai/gsm8k` main/test (1,319) @ `740312a`,
`google/IFEval` train (541) @ `966cd89`, `allenai/ai2_arc` ARC-Easy/test (2,376) @ `210d026`.

## Setup

Hardware: 8× A100-80GB, one shard per GPU (the shard count fixes the batch
composition, so keep 8 shards to reproduce the paper runs).

Two Python environments were used, and the launchers keep the same split:

| Variable | Environment | Used for |
|---|---|---|
| `PYTHON` | torch 2.9.1, transformers 4.57.1, SGLang 0.5.9 patched with `patches/sglang/`, matplotlib | GSM8K AV servers/decode, GSM8K reconstruct and rollouts, plotting |
| `VLLM_PYTHON` | vLLM 0.19.0, torch 2.10.0, transformers 4.57.1, datasets 5.0.1, plus the IFEval checker dependencies `absl-py` 2.5.0, `langdetect` 1.0.9, `nltk` 3.10.3 (with `punkt` and `punkt_tab` data), `immutabledict` 4.3.1 | every extraction; IFEval and ARC-Easy decode, reconstruct and rollouts; all paraphrasing; scoring |

`SOAR_SGLANG_PYTHON` (default `$PYTHON`) is the interpreter that launches SGLang.
Plotting (`figure plot`, `figure plot-legend`) only reads the figure CSV and runs in
any environment with matplotlib.

## Commands

Run directories default to `results/benchmarks/<task>/` (override with `OUT`/`RUNS`);
bulk artifacts there are git-ignored.

```bash
export PYTHON=/path/to/main/python VLLM_PYTHON=/path/to/vllm/python
soar_nla/benchmarks/run_gsm8k.sh      # extract -> 8 SGLang AV servers + decode -> reconstruct -> clean/identity/patched -> summary
soar_nla/benchmarks/run_ifeval.sh     # extract -> decode-vllm -> reconstruct -> clean/patched -> summary
soar_nla/benchmarks/run_arc_easy.sh   # extract -> decode-vllm -> reconstruct -> clean/patched -> summary
soar_nla/benchmarks/run_allpos.sh     # random_vec + para_medium/para_heavy arms on the three runs above
soar_nla/benchmarks/make_figures.sh   # figure CSV (score), plot, plot-legend, paper-number check
```

Each launcher is a sequence of per-shard calls of the modules below, e.g.
`python -m soar_nla.benchmarks.gsm8k patched --shard 3 --num-shards 8`.

| Module | Role |
|---|---|
| `common.py` | prompt formatting, layer-32 capture (`run_extract`), prefill replay hook and greedy generation (`generate`, `run_hf`), pinned paths |
| `gsm8k.py`, `ifeval.py`, `arc_easy.py` | per-task modes `extract`, `clean`, (`identity`,) `patched`, `summarize` |
| `roundtrip.py` | AV `decode` (SGLang) / `decode-vllm` (vLLM `prompt_embeds`), AR `reconstruct` with norm matching |
| `allpos.py` | paraphrase / random arms: `setup`, `random`, `paraphrase`, `summarize` |
| `rollouts.py` | locate per-example rollouts in run directories or the exported `.jsonl.gz` files |
| `figure.py` | `score` (figure CSV), `plot`, `plot-legend` |
| `paper_numbers.py` | rescore every rollout and compare with the paper figure; `--export` writes `results/benchmarks/rollouts/` |

Settings of the paper runs, as passed by the launchers:

| Task | extract batch | AV decode | reconstruct batch | rollout batch | max new tokens |
|---|---|---|---|---|---|
| GSM8K | 8 (right padding) | SGLang, 1 server/GPU, 16 concurrent requests | 64 | 8 | 4096 |
| IFEval | 8 (left padding) | vLLM, request batch 32 | 32 | 4 | 4096 |
| ARC-Easy | 8 (left padding) | vLLM, request batch 64 | 32 | 16 | 32 |
| para arms | – | – | 32 (vLLM env) | as the task | as the task |

SGLang AV servers: `--disable-radix-cache --mem-fraction-static 0.80 --context-length 512
--attention-backend triton --disable-cuda-graph --trust-remote-code --dtype bfloat16`,
`SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=1`.

To recompute the figure and paper numbers without GPUs from the committed rollouts:

```bash
ROLLOUTS=results/benchmarks/rollouts soar_nla/benchmarks/make_figures.sh
```

## Outputs (`results/benchmarks/`)

* `gsm8k/matched_long_generation_summary.json`, `ifeval/ifeval_summary.json`,
  `arc_easy/arc_easy_summary.json`: clean vs NLA per task (accuracies, paired outcomes,
  exact McNemar p, generation-cap hits).
* `allpos/summary.{json,csv}`: all five conditions per task and metric, with paired
  deltas vs clean, mean cosine of the replacement vectors to the native residuals,
  paraphrase tag rate and word overlap.
* `figures/allpos_accuracy_bars_legend.{pdf,svg,png}` (the paper figure) and
  `figures/allpos_accuracy_strict_average.{pdf,svg,png,csv}` (same data, labeled axes).
  ARC-Easy and GSM8K bars: exact accuracy with Wilson 95% intervals; IFEval: strict
  average with 10,000-draw prompt-bootstrap 95% intervals.
* `rollouts/<task>__<condition>.jsonl.gz`: every per-example generation behind the
  figure (shard files concatenated verbatim; 0.2–2.4 MB each, 8.0 MB total).

## Paper figures

```bash
python -m soar_nla.benchmarks.paper_figure   # CPU only, reads figures/allpos_accuracy_strict_average.csv
```

`figures/clean_vs_nla.{png,pdf}`: clean vs NLA-patched accuracy on the three tasks, from the clean
and `nla` rows of the figure CSV (same intervals as `figure.py`).

## Paper claims

| Claim | Script | Output file / field | Value |
|---|---|---|---|
| ARC-Easy clean → NLA | `figure.py score`, `paper_numbers.py` | `figures/allpos_accuracy_strict_average.csv` `arc_easy,{clean,nla}` | 96.93% → 96.55% (2303 → 2294 / 2376) |
| GSM8K clean → NLA | same | `... gsm8k,{clean,nla}`; `gsm8k/matched_long_generation_summary.json` | 94.16% → 86.58% (1242 → 1142 / 1319) |
| IFEval strict average clean → NLA | same | `... ifeval,{clean,nla}` | 85.29% → 62.31% |
| IFEval prompt / instruction strict | `ifeval.py summarize` | `ifeval/ifeval_summary.json` `conditions.*.strict` | clean 82.81 / 87.77, NLA 56.75 / 67.87 |
| Paraphrase (medium / heavy) | `figure.py score` | `... <task>,para_medium/para_heavy` | ARC 92.80 / 91.62, GSM8K 81.73 / 76.50, IFEval 53.54 / 50.88 |
| Random vector | same | `... <task>,random_vec` | ARC 0.00, GSM8K 9.25, IFEval 21.74 |
| "almost no difference on ARC-Easy", larger drops on GSM8K and IFEval | `arc_easy.py summarize` | `arc_easy/arc_easy_summary.json` `paired_clean_vs_nla` | −0.38 points (McNemar p = 0.035); GSM8K −7.58 points (p = 6.7e-20) |

`paper_numbers.py` recomputes all fifteen bars from the generated text and checks them
against the values above; on the original per-row outputs every value matches.

## Notes

* **AV runtime differs by task**, as in the paper runs: GSM8K explanations came from
  SGLang (HTTP `input_embeds`), IFEval and ARC-Easy from vLLM `prompt_embeds`. Both are
  greedy but neither is bitwise batch-invariant, so a rerun can change a small number of
  explanations. The original IFEval decode was interrupted and resumed: roughly the
  first 570 positions of each shard were decoded with request batch 8, the rest with 32;
  the launcher uses 32 throughout.
* **Padding at extraction**: GSM8K states were captured from right-padded batches,
  IFEval and ARC-Easy from left-padded ones (the extraction slices the valid tokens
  accordingly). The plain forward pass does not pass `position_ids`, so left-padded rows
  see RoPE positions offset by their pad count; RoPE is relative, so this only changes
  bf16 rounding. Both behaviors are kept as run.
* GSM8K extraction ran in the vLLM environment and everything else for GSM8K in the main
  environment (different torch versions); IFEval and ARC-Easy ran entirely in the vLLM
  environment.
* IFEval is scored with Google's checker for every condition (clean 85.29). The figure does
  not use any lm-evaluation-harness rescoring.
* No all-token identity control was run for IFEval or ARC-Easy; for GSM8K the identity
  replay reaches 94.31% vs 94.16% clean.
* Only layer 32 is replaced; layers 0–31 at prompt positions stay intact and generated
  tokens attend to them, which is why some GSM8K random-vector outputs still solve the
  problem (122 / 1319).
