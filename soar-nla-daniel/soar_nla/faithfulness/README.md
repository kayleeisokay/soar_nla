# Faithfulness

Reproduces the paper's **Faithfulness** section: does a more faithful explanation lead to a
better reconstruction?

1. **FVE of AV vs Sonnet explanations** (`fve.py`, `paper_numbers.py`). 2,000 held-out
   FineFineWeb activations of Gemma-3-12B-IT (layer 32, final token) are explained two ways — by
   the AV (round trip) and by Sonnet (the warm-start data recipe, written with the passage in
   view) — and both explanations go through the AR. Done for the custom L32 NLA's warm-start
   SFT pair and its final RL pair, with a random-pairing control (each explanation dealt to the
   wrong activation).
2. **Sentence-level faithfulness judge** (`sentence_judge.py`). Every sentence of the Sonnet and
   AV explanations of the same examples is labelled Supported / Unsupported / Illegible against
   the source passage by GPT-5.6 Luna (the prompt in the paper's appendix is `PROMPT`).
3. **Surgical edits** to official-NLA AV explanations: see
   [`surgical_edits/README.md`](surgical_edits/README.md).

## Model and data

- NLA: the custom EasyNLA-format pair `achand45/gemma-3-12b-it-nla-L32` @ `6bad086e…`
  (`soar_nla.hub.CUSTOM_L32`). We use it rather than the official release because its training
  data is public, so "held out" can be checked. Only the needed subfolders are downloaded.
  - `--spec sft`: AV `merged/av_hf`, AR `merged/ar_hf` (warm-start SFT).
  - `--spec rl`: AV = base `google/gemma-3-12b-it` + LoRA `rl_vllm/iter_000400` (the adapter
    already contains SFT+RL and goes on the raw base), AR `rl_vllm/critic_latest`.
  - Both read tokens, prompt templates and `mse_scale` (= √3840) from `rl_vllm/nla_meta.yaml`.
  - EasyNLA conventions (module docstring of `fve.py`): Karvonen norm-matched residual add on
    the output of `layers[1]` with raw vectors; `critic_predict` normalizes the AR's last hidden
    state to `mse_scale` before the value head; the AR prompt is tokenized without BOS.
- Sonnet explanations and passages: `asher577/easynla-warmstart-data` @ `0d710500…`,
  `av_sft_train.parquet` (`config.py`).
- Eval set (`fve.py dataset`): 2,000 rows / 1,853 documents drawn in a seeded order, skipping
  documents listed in the exclusion manifest `exclude_easynla.json.gz` (shipped here), rows
  without explanation tags, and texts shorter than 50 tokens; texts longer than 2,048 tokens are
  left-truncated. The activation is taken at the last token of the passage.

### Contamination exclusions (`contamination.py`)

- `exclusion` rebuilds the manifest: the doc_ids of every row in the custom NLA's five L32
  training parquets (`achand45/gemma-3-12b-it-nla-data` @ `c85065cf…`: AV-SFT, AR-SFT and RL
  pools, including their val splits; 99,956 doc_ids), plus 47,486 text fingerprints (SHA-1 of the
  first 150 whitespace-normalized, lower-cased characters) of a second Gemma-3-12B L32 RL dataset
  that was built locally from the same corpus file and is not public. With that file passed as
  `--local-rl-parquet`, the rebuild is byte-identical to the shipped manifest; without it you get
  the doc_id half only, which is why the manifest itself ships.
- `strict` flags eval documents whose doc_id or text fingerprint appears in the custom NLA's
  AV-SFT, AR-SFT or RL parquet (or in the first 120k rows of the local RL dataset, if given).
  It finds 3 documents / 3 rows, all by text match into the public SFT parquets, so it is
  reproducible from public data. The paper's FVE numbers are on the remaining **1,997 rows /
  1,850 documents** (`strict_clean_excluded_docs.json`).

Why both checks: doc_ids are scoped to a corpus file, and the warm-start data mixes two
independent 100k FineFineWeb draws while the custom NLA trained on only one. Excluding the
NLA's doc_ids leaves the eval set entirely in the other draw; fingerprints catch the texts the
two draws share.

### FVE

`MSE = 2(1 − cos)` between the reconstruction and the gold activation (both scaled to
`mse_scale`), divided by the directional variance of the gold set (`1 − ‖mean of unit golds‖²`).
95% intervals come from a 10,000-resample bootstrap clustered on documents; the gold − AV delta
is paired on the same resample. AV decoding is greedy, 256 new tokens.

### Sentence judge

- Sentences: each explanation is split on sentence-final punctuation followed by whitespace and
  on every newline (`SEPARATOR`); one judge call per sentence, with the passage and the full
  explanation as context. OpenAI Responses API, `gpt-5.6-luna`, `reasoning={"effort": "low"}`.
- Arms: `custom_sonnet` and `custom_av`, the **RL** AV's explanations of the same 2,000 rows
  (`prepare --custom-results <fve-dir>/rl`; all 2,000 rows, no strict filter). `prepare` also
  builds the GCG section's arms, `alpaca_av` and `alpaca_gcg_beta_0.01`: the official L32 AV
  explanation of each of the 1,573 late-position Alpaca rollouts and its β = 0.01 GCG rewrite
  (`--alpaca-results`, `--gcg-results`). The GCG text is split at the original explanation's
  sentence boundaries (token spans), so the two arms judge aligned sentences.
- The paper's tables keep the **first two paragraphs** of each explanation (paragraphs are
  separated by a blank line; every custom AV explanation has three, 1,996/2,000 Sonnet
  explanations have three, and 1,527/1,573 Alpaca AV explanations have three). The third
  paragraph is the final-token unit — which token the passage ends on and what should come
  next, and for the AV also `Example:` continuation lines — so it is mostly a prediction about
  unseen text. The Alpaca arms take paragraph indices from the original AV explanation.
  `aggregate` reports three scopes (Supported / Unsupported / Illegible, %):

  | scope | Sonnet (n) | custom AV (n) | Alpaca AV (n) | Alpaca GCG β=0.01 (n) |
  |---|---|---|---|---|
  | first two paragraphs (paper) | 73.1 / 26.6 / 0.3 (4,267) | 6.3 / 93.3 / 0.4 (4,331) | 17.5 / 82.4 / 0.1 (3,298) | 14.3 / 84.6 / 1.1 (3,298) |
  | all sentences | 63.0 / 36.8 / 0.3 (6,305) | 7.1 / 91.4 / 1.5 (10,786) | 12.9 / 84.5 / 2.6 (5,642) | 10.6 / 83.3 / 6.1 (5,642) |
  | after the second paragraph | 41.8 / 57.9 / 0.2 (2,038) | 7.7 / 90.1 / 2.3 (6,455) | 6.4 / 87.5 / 6.2 (2,344) | 5.5 / 81.4 / 13.1 (2,344) |

  Excluding only each explanation's last paragraph gives the same rounded Sonnet and custom-AV
  rates (Sonnet n = 4,262).

## Setup

- `pip install -e .` and, for `decode`/`reconstruct`, the vendored EasyNLA package
  (`pip install -e training/`, imported as `nla`). Keep `NLA_FREEZE_VALUE_HEAD` unset.
- Judge: `pip install -e ".[judge]"` and `OPENAI_API_KEY` (environment, or `--env-file`).
- Hardware: `extract`, `decode` and `reconstruct` each load one 12B model in bf16 per GPU
  (80 GB GPUs were used). The paper run decoded on 8 GPUs, one shard per GPU.
  `dataset`, `contamination`, `analyze`, `aggregate` and `paper_numbers` are CPU-only;
  `dataset` and `contamination` stream parquet columns from the Hugging Face Hub.

## Commands

```bash
# FVE pipeline (both checkpoint pairs); GPUS, PYTHON and RESULTS are overridable.
bash soar_nla/faithfulness/run_fve.sh
#   = fve dataset/extract once (--spec rl), copied to sft/, then per spec:
#     decode (sharded) -> merge-decode -> reconstruct -> analyze

# Sentence judge on the RL run's explanations and, for the GCG section, the Alpaca AV
# explanations and their beta=0.01 GCG rewrites (needs soar_nla.alpaca and soar_nla.gcg outputs)
python -m soar_nla.faithfulness.sentence_judge prepare
python -m soar_nla.faithfulness.sentence_judge judge
python -m soar_nla.faithfulness.sentence_judge aggregate \
    --labels-out results/faithfulness/sentence_labels.jsonl.gz

# Paper numbers (FVE strict-clean table + judge table, checked against the paper)
python -m soar_nla.faithfulness.paper_numbers

# Optional: rebuild the exclusion inputs
python -m soar_nla.faithfulness.contamination exclusion --local-rl-parquet <rl_shuf.full.parquet>
python -m soar_nla.faithfulness.contamination strict
```

`paper_numbers` reads `--fve-dir` (default `results/faithfulness/fve`, holding `sft/` and `rl/`),
`--strict-exclude` and `--judge-results`; by default it uses the shipped
`results/faithfulness/sentence_labels.jsonl.gz`, so the judge table recomputes without API
calls. Without the FVE pipeline outputs it checks only the judge numbers.

## Outputs (`results/faithfulness/`)

| file | content |
|---|---|
| `fve_table.json`, `fve_table.md` | four-arm FVE, SFT and RL, `strict_clean` (paper) and `all_rows` |
| `fve_report_{sft,rl}.json` | `fve analyze` reports on all 2,000 rows |
| `strict_clean_excluded_docs.json` | the 3 documents dropped for the paper's table |
| `sentence_judge_table.json`, `.md` | judge rates for the three scopes |
| `sentence_labels.jsonl.gz` | one record per judged sentence: arm, example, paragraph, label |
| `paper_numbers.json` | paper value vs recomputed value for every number below |

Pipeline intermediates (`fve/{sft,rl}/...`: rows, activations, explanations, reconstructions;
`sentence_judge/items.jsonl`, `results.jsonl`) are written next to them and are not tracked.

## Paper figures

```bash
python -m soar_nla.faithfulness.paper_figure   # CPU only, reads the shipped files
```

- `figures/fve_by_explanation.{png,pdf}`: AV and Sonnet FVE after RL and at the warm-start
  checkpoint, strict clean subset, bootstrap 95% CIs (`fve_table.json`).
- `figures/av_vs_sonnet_example.{png,pdf}`: the stepped-spillway passage of
  `qualitative_example.txt` (FineFineWeb row 1920, RL checkpoint), AV vs Sonnet explanation.
  The grounded / confabulated highlighting is a manual reading of the passage (phrase lists in
  `paper_figure.py`); the FVEs 0.801 / −0.162 are that row's values as recorded in the file.

## Paper claims

| paper claim | script | output → field | value |
|---|---|---|---|
| 2,000 FineFineWeb examples not used to train the NLA | `fve.py dataset`, `contamination.py` | `fve/rl/dataset/manifest.json` | 2,000 rows / 1,853 docs; FVE on 1,997 / 1,850 |
| SFT: Sonnet FVE 60.8% | `paper_numbers.py` | `fve_table.json` → `stages.sft.strict_clean.arms.gold.fve` | 0.6079 |
| SFT: AV FVE 48.2% | `paper_numbers.py` | `stages.sft.strict_clean.arms.av.fve` | 0.4822 |
| RL: AV FVE 70.1% | `paper_numbers.py` | `stages.rl.strict_clean.arms.av.fve` | 0.7006 |
| RL: Sonnet FVE 53.8% | `paper_numbers.py` | `stages.rl.strict_clean.arms.gold.fve` | 0.5383 |
| random AV explanation −74.8% | `paper_numbers.py` | `stages.rl.strict_clean.arms.av_shuffled.fve` | −0.7477 |
| random Sonnet explanation −65.6% | `paper_numbers.py` | `stages.rl.strict_clean.arms.gold_shuffled.fve` | −0.6558 |
| Sonnet sentences 73.1% / 26.6% / 0.3% | `sentence_judge.py aggregate` | `sentence_judge_table.json` → `table.first_two_paragraphs.custom_sonnet.percent` | 73.1 / 26.6 / 0.3 (n = 4,267) |
| AV sentences 6.3% / 93.3% / 0.4% | `sentence_judge.py aggregate` | `table.first_two_paragraphs.custom_av.percent` | 6.3 / 93.3 / 0.4 (n = 4,331) |
| GCG section: 17.5% (AV) vs 14.3% (GCG β = 0.01) supported | `sentence_judge.py aggregate` | `table.first_two_paragraphs.{alpaca_av,alpaca_gcg_beta_0.01}.percent.Supported` | 17.5 / 14.3 (n = 3,298 each) |

On all 2,000 rows the SFT AV FVE is 0.4825 (it would round to 48.3%); every other FVE above
rounds the same on either row set. The SFT pair's random-pairing values are −0.6309 (AV) and
−0.6123 (Sonnet).
