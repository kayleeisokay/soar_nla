# soar-nla

Code and result summaries for our paper on the fidelity, faithfulness and
editability of **natural language autoencoders (NLAs)**. An NLA pairs an
*activation verbalizer* (AV, vector → text) with an *activation reconstructor*
(AR, text → vector); see [Natural Language Autoencoders Produce Unsupervised
Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html).

Most experiments use Gemma-3-12B-IT at layer 32, with either the official
release (`kitft/nla-gemma3-12b-L32-{av,ar}`) or our custom NLAs trained with
the vendored trainer. All checkpoints are pinned to exact revisions in
`soar_nla/hub.py`.

## Paper sections → code

| Paper section | Code | Results | Status |
|---|---|---|---|
| Loss Recovered (+ SAE baseline) | [`soar_nla/loss_recovered`](soar_nla/loss_recovered/README.md) | `results/loss_recovered` | Reproduced |
| GCG (+ random-edit control, fluency, sentence judge) | [`soar_nla/gcg`](soar_nla/gcg/README.md), rollouts from [`soar_nla/alpaca`](soar_nla/alpaca/README.md), judge in [`soar_nla/faithfulness`](soar_nla/faithfulness/README.md) | `results/gcg`, `results/alpaca` | Reproduced |
| Faithfulness: AV vs Sonnet FVE, sentence judge | [`soar_nla/faithfulness`](soar_nla/faithfulness/README.md) | `results/faithfulness` | Reproduced |
| Faithfulness: surgical edits | [`soar_nla/faithfulness/surgical_edits`](soar_nla/faithfulness/surgical_edits/README.md) | `results/surgical_edits` | Reproduced |
| ICL edits: text reconstruction interventions (+ donor sweep, L24/L40, chat template) | [`soar_nla/icl_edits`](soar_nla/icl_edits/README.md) | `results/icl_edits` | Reproduced |
| Task-switching interventions (+ chat template) | [`soar_nla/icl_edits`](soar_nla/icl_edits/README.md) | `results/icl_edits` | Reproduced |
| Benchmarks using chat template (ARC-Easy, GSM8K, IFEval) | [`soar_nla/benchmarks`](soar_nla/benchmarks/README.md) | `results/benchmarks` | Reproduced |
| Function-vector replication with the chat template | [`soar_nla/fv_chat_template`](soar_nla/fv_chat_template/README.md) | `results/fv_chat_template` | Reproduced |
| Appendix: Skipping warm-start (cold-start RL, SelfDescribe judge) | [`soar_nla/warmstart_ablation`](soar_nla/warmstart_ablation/README.md), trainer in [`training/`](training/README.md) | `results/warmstart_ablation` | Reproduced |

"Reproduced" means the section's `paper_numbers.py` (or equivalent analysis
script) recomputes every number the paper quotes from saved per-example
outputs. Each section README has a *paper claim → script → output → value*
table and the commands to rerun the GPU stages.

## Repository layout

```
soar_nla/
  nla_inference.py     NLAClient (AV via SGLang input_embeds) and NLACritic (AR), from the NLA release
  hub.py               pinned model revisions, snapshot resolution, target-model loading
  sglang_server.py     launch / health-check SGLang AV servers
  io_utils.py          stable seeds, resumable JSONL
  apikeys.py, paths.py API keys from the environment; results root ($SOAR_RESULTS, default ./results)
  figure_style.py      shared look of the paper figures; text_panels.py: monospace text panels
  assets/fonts/        Noto Sans Devanagari (OFL) for one token in the GCG example figure
  <section>/           one subpackage per paper section (table above)
training/              vendored EasyNLA trainer (package `nla`, MIT) for the warm-start ablation
patches/sglang/        SGLang patches needed to drive Gemma-3 AVs with input_embeds
results/               small summaries, per-example outputs needed to recompute paper numbers, figures
```

## Setup

The experiments ran on one node with 8 × A100-80GB.

```bash
git clone <this repo> soar-nla && cd soar-nla
python -m venv .venv && . .venv/bin/activate
pip install -e ".[judge,ifeval,figures]"
pip install -e training/          # needed by faithfulness/ and warmstart_ablation/
huggingface-cli login             # Gemma-3 is gated
```

Additional environments:

- **SGLang** (AV decoding for most sections): install SGLang 0.5.9 from source
  and patch it, then point `SOAR_SGLANG_PYTHON` at that interpreter.
  ```bash
  bash patches/sglang/apply_sglang_patches.sh /path/to/sglang
  export SOAR_SGLANG_PYTHON=/path/to/sglang-env/bin/python
  ```
  Servers must run with `--disable-radix-cache` (the launchers do this).
- **vLLM** (vllm 0.19): IFEval/ARC-Easy AV decoding with `prompt_embeds`, the
  Gemma-3-27B-IT SelfDescribe judge, and the RL rollouts of the trainer (with
  the patched vllm-lens, see `training/docs/vllm-lens-setup.md`). Launchers
  read `VLLM_PYTHON` / `RL_PYTHON`.
- **Loss Recovered data**: the Gemma-3-12B-IT Alpaca split of the Lie Detection
  Rollouts dataset (`google-gemma-3-12b-it/alpaca.parquet`); see
  `soar_nla/loss_recovered/README.md`.
- **API keys**: `OPENAI_API_KEY` for the sentence-level judge and the surgical
  editor; `ANTHROPIC_API_KEY` only if you regenerate warm-start data. CLIs that
  call an API accept `--env-file` to load a dotenv file.

## Checking the paper numbers without GPUs

Each of these reads only files shipped under `results/`:

```bash
python -m soar_nla.loss_recovered.paper_numbers
python -m soar_nla.gcg.paper_numbers
python -m soar_nla.faithfulness.paper_numbers                 # judge table; FVE needs run_fve.sh outputs
python -m soar_nla.faithfulness.surgical_edits.analysis
python -m soar_nla.icl_edits.paper_numbers
python -m soar_nla.benchmarks.paper_numbers --rollouts results/benchmarks/rollouts
python -m soar_nla.fv_chat_template.paper_numbers
python -m soar_nla.warmstart_ablation.fve_table
python -m soar_nla.warmstart_ablation.analyze_5way \
    --real results/warmstart_ablation/judge5_real.jsonl.gz \
    --control results/warmstart_ablation/judge5_control.jsonl.gz
```

Activations, AR reconstructions and full GCG traces are not shipped; the
section pipelines regenerate them.

## Paper figures

All figures are drawn from the shipped files, on CPU:

| Paper figure | Command | Output (under `results/`) |
|---|---|---|
| Loss Recovered by token span | `python -m soar_nla.loss_recovered.paper_figure` | `loss_recovered/figures/loss_recovered.png` |
| Fig. 3: GCG FVE across β and example | `python -m soar_nla.gcg.paper_figure` | `gcg/figures/gcg_fve_and_example.png` |
| Skipping warm-start: training FVE, SelfDescribe judge | `python -m soar_nla.warmstart_ablation.paper_figure` | `warmstart_ablation/figures/{training_fve,selfdescribe_judge}.png` |
| Benchmarks: clean vs NLA patched | `python -m soar_nla.benchmarks.paper_figure` | `benchmarks/figures/clean_vs_nla.png` |
| Benchmarks with paraphrase and random-vector controls | `soar_nla/benchmarks/make_figures.sh` | `benchmarks/figures/allpos_accuracy_bars_legend.png` |
| Function vectors with the chat template | `python -m soar_nla.fv_chat_template.figures` (see its README) | `fv_chat_template/figures/fv_english_french_paired.png` |
| Faithfulness: FVE by explanation source, AV vs Sonnet example | `python -m soar_nla.faithfulness.paper_figure` | `faithfulness/figures/{fve_by_explanation,av_vs_sonnet_example}.png` |
| ICL edits, donor sweep, task switching | `python -m soar_nla.icl_edits.figures` | `icl_edits/figures/` |

## Licenses

Apache-2.0 (see `LICENSE`), except vendored code that keeps its own license:
`training/` (EasyNLA, MIT) and `soar_nla/benchmarks/ifeval_lib/` (Google
IFEval, Apache-2.0). `soar_nla/nla_inference.py` and `patches/sglang/` come
from the NLA release (Apache-2.0). `soar_nla/assets/fonts/NotoSansDevanagari.ttf`
is under the SIL Open Font License (`soar_nla/assets/fonts/OFL.txt`).
