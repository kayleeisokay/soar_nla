# Function-vector replication with the chat template

Reproduces the paper's English→French function-vector result on the chat template
(the paragraph after "Benchmarks using chat template"): with the Gemma chat template,
patching the NLA reconstruction into the final prefill position recovers slightly more
of the in-context task than with the raw prompt.

The experiment follows the protocol of
<https://github.com/kayleeisokay/natural_language_autoencoders>
(commit `02d62272d90528742ef476c316dc39098f5ae798`, `notebooks/executor.py`): the same
task files, RNG plan, prompts, interventions, paraphrase prompts, token caps and scoring,
run on the 2-shot English→French configuration (50 queries), once with the raw prompt as
in the reference and once with the same prompt wrapped in the Gemma chat template.

## Protocol

* **Target**: `google/gemma-3-12b-it` @ `96b6f1e` (bf16, eager attention, vision tower
  removed). **NLA**: `kitft/nla-gemma3-12b-L32-av` @ `7aec225` served by SGLang, and
  `kitft/nla-gemma3-12b-L32-ar` @ `3d6901d` in float32.
* **Prompt** (`protocol.build_prompt`): `Answer with a single word only. No punctuation,
  no explanation.` followed by two `Q: <english>\nA: <french>` demonstrations and
  `Q: <query>\nA:`. Raw style: the tokenizer adds `<bos>`. Chat style: one user turn plus
  the generation prompt (the template supplies `<bos>`).
* **Patched position**: a forward hook on decoder layer 32 captures and, in the
  intervention arms, overwrites the residual at the **final prefill position only**.
  * Raw: the prompt ends `… A:`, so the patched token is `:` (id 236787), and the
    response starts right after it.
  * Chat: the prompt ends `… A:<end_of_turn>\n<start_of_turn>model\n` — tokens `:`
    (236787), `<end_of_turn>` (106), `\n` (107), `<start_of_turn>` (105), `model` (4368),
    `\n` (107). The patched token is the final `\n` after `<start_of_turn>model`, whose
    next-token prediction is the first response token. The four template tokens
    `<end_of_turn>`, `\n`, `<start_of_turn>`, `model` lie between the last prompt token
    `:` and the patched position and keep their native layer-32 states, as do all earlier
    positions. This is the "four chat template tokens between the last prompt token and
    the first response token" in the paper.
* **Arms** (all greedy, ≤256 new tokens):
  `baseline` (no patch); `identity` (the captured native vector, hook check);
  `nla` = AR(AV(h)) with the AV greedy (≤384 tokens) and the **raw AR output, not
  norm-matched** (as in the reference); `random_vec` = seeded Gaussian rescaled to ‖h‖;
  `mismatch_cross_task` / `mismatch_same_task` = the native final-position vector of a
  donor prompt from another task / the same task with a different query (donors from the
  sampling plan); `para_medium` / `para_heavy` = AR of a Gemma-3-12B-IT paraphrase
  (raw prompt, greedy, ≤512 tokens) of the AV explanation.
* **Scoring**: first line of the response, surrounding punctuation stripped,
  case-insensitive exact match with the gold French word.

Differences from the reference executor: the AR runs in float32 on a GPU rather than on
the CPU (small numerical differences), and every query is also run in chat style.

## Data

`fetch_data.py` downloads the 15 task files the sampling plan reads from the reference
repository at the pinned commit (`notebooks/datasets/`) and checks each SHA-256 against
`protocol.DATA_FILES`. All six tasks are needed even though only English→French is
evaluated, because the plan replays every RNG draw of the original sweep (all tasks and
1/2/4 shots in order). The files are fetched, not vendored, because the reference
repository states no license for them; they appear to derive from third-party
datasets with their own terms — the `en-fr.*.txt` files follow the MUSE bilingual
dictionary naming (MUSE dictionaries are CC BY-NC 4.0), the `*-pairs.{train,val,test}`
antonym files follow the AntSynNET format (Nguyen et al., 2017), and the small JSON
tasks are in the style of the function-vector task suite (Todd et al., 2024).

## Setup

* Hardware: 4× A100-80GB for the full phase — target on one GPU, float32 AR on a second
  (`--target-gpus`, default `4,5`), AV SGLang server with tensor parallelism 2 on two
  more (`--av-gpus`, default `6,7`, port 32367). Baseline phases use one GPU.
* Environment: torch, transformers 4.57.1, matplotlib, and an SGLang patched with
  `patches/sglang/` (`SOAR_SGLANG_PYTHON`, default the current interpreter). SGLang's
  triton backend may need `libnuma` on `LD_LIBRARY_PATH` and the Python headers on
  `CPATH`; set those in your environment if the server fails to start.

## Commands

```bash
soar_nla/fv_chat_template/run_fv.sh     # everything below, writing to results/fv_chat_template/rerun/

python -m soar_nla.fv_chat_template.fetch_data        # -> results/fv_chat_template/datasets/
python -m soar_nla.fv_chat_template.prepare           # -> results/fv_chat_template/sampling_manifest.json (CPU)
python -m soar_nla.fv_chat_template.run --phase baseline --shots 2 --out-dir OUT/shot2
python -m soar_nla.fv_chat_template.run --phase full --shots 2 --out-dir OUT/shot2
python -m soar_nla.fv_chat_template.run --phase baseline --shots 1 --out-dir OUT/shot1
python -m soar_nla.fv_chat_template.run --phase baseline --shots 4 --out-dir OUT/shot4
python -m soar_nla.fv_chat_template.figures --shot1-dir OUT/shot1 --shot2-dir OUT/shot2 --shot4-dir OUT/shot4 --figures-dir OUT/figures
python -m soar_nla.fv_chat_template.paper_numbers --rows OUT/shot2/rows.json
```

`run` resumes from an existing `rows.json`, so point `--out-dir` at a fresh directory to
regenerate (the defaults are the committed `results/fv_chat_template/shot<N>/`).
`prepare` reproduces the original sampling manifest byte for byte.

## Outputs (`results/fv_chat_template/`)

* `shot2/rows.json`: per query, every arm's output, prediction and correctness in both
  styles, plus the AV explanations, paraphrases and NLA reconstruction cosines.
  `shot2/summary.json`: per-arm accuracy and paired chat-vs-raw McNemar tests.
  `shot1/`, `shot4/`: baseline-only runs on the same plan.
  (`native.safetensors`, the captured vectors, is not committed.)
* `reference_shot_sweep_english_french.csv`: the results CSV the reference executor wrote
  for its English→French sweep (reference bars in the figures).
* `figures/fv_english_french_paired.{pdf,svg,png}` + `figure_data.csv` (with paired
  bootstrap 95% intervals for chat − raw), `figures/fv_english_french_shot_baselines.*` +
  `baseline_shot_sweep_data.csv`.

## Paper claims

All values are accuracy over the same 50 queries (raw prompt / chat template).

| Claim | Script | Output file / field | Value |
|---|---|---|---|
| Baseline | `run.py`, `paper_numbers.py` | `shot2/summary.json` `conditions.{raw,chat}_baseline` | 38% / 40% |
| NLA round trip | same | `conditions.{raw,chat}_nla` | 34% / 38% |
| Medium paraphrase | same | `conditions.{raw,chat}_para_medium` | 34% / 38% |
| Heavy paraphrase | same | `conditions.{raw,chat}_para_heavy` | 26% / 34% |
| Random vector | same | `conditions.{raw,chat}_random_vec` | 2% / 0% |
| Cross-task native swap | same | `conditions.{raw,chat}_mismatch_cross_task` | 4% / 10% |
| Same-task native swap | same | `conditions.{raw,chat}_mismatch_same_task` | 0% / 16% |
| Reference-reported raw values | `figures.py` | `figures/figure_data.csv` `reference_accuracy` | 38 / 34 / 30 / 28 / 2 / 2 / 0% |

`paper_numbers.py` rescores every arm from the stored outputs and matches all seven
pairs above. The identity arm reproduces the baseline exactly (38% / 40%). The chat-vs-raw
NLA gain is 2/50 queries (paired McNemar p = 0.5; `paired_chat_vs_raw.nla`), so at n = 50
it is suggestive rather than conclusive. 1- and 4-shot baselines: 14% / 14% and 50% / 50%
(reference raw 14% and 52%).
