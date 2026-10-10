# ICL edits and task switching

Reproduces the paper sections **"ICL edits / Text reconstruction interventions"** (with its
appendices: donor patching across layers, NLA edits at layers 24 and 40, chat template) and
**"Task-switching interventions"** (with its chat-template appendix), on Gemma-3-12B-it.

For a prompt position, the AV explains the residual activation in English; the explanation is
edited with plain string substitutions; the AR reconstructs an activation from the edited text; the
reconstruction replaces the original residual at the same layer and position:

```
h  = residual at the NLA layer, chosen positions
z  = AV(h)            English explanation
z* = edit(z)          substitution / final-paragraph rewrite / both
h <- AR(z*)           literal replacement during prefill, rescaled to the native norm of h
```

## Protocol details not spelled out in the paper

- **Tasks.** Answer redirection uses six Function Vectors tasks (antonym, capitalize,
  country-capital, English-French, present-past, singular-plural; `tasks.py`), 10 evaluation items
  per task x 10 seeds = 600 trials per condition. The seed picks the 10 demonstrations
  (`zlib.crc32` of task, query, seed). The redirect **target of item i is the gold answer of item
  (i + 3) mod 10** of the same task, and its query is that item's query.
  Task switching uses three switches (French->Spanish, capital->currency, antonym->comparative),
  10 queries x 10 seeds = 300 trials per condition; the target is the same query answered under the
  target task.
- **Formats.** `raw` (the paper's main results): the bare 10-shot `Q:x\nA:y` text with `<bos>`, as
  in Function Vectors; the final token is the `:` after `A`, and the four-token span is
  `{query} \n A :`, so it includes the query token. `chat` (the appendices): the same text in one
  chat user turn followed by `Answer immediately and only in one word.`; the final token is the
  last assistant-header token and the four-token span is the header `\n <start_of_turn> model \n`.
  The two formats differ in instruction and patched positions, not only in templating.
- **AV sampling and selection.** Each patched position gets **3 AV draws** (temperature 1, at most
  256 new tokens, seeds `7919*seed + 131*position + 17*draw`; the easynla backend seeds each batch of
  64 draws with its first draw's seed). One draw per position is kept by a
  **text-only score** (`edits.icl_select_score` / `edits.switch_select_score`), never by the
  model's output. All arms of a position use the same chosen draw.
- **Edits.** Query-and-answer substitution replaces whole words, case-preserving (gold -> target,
  query -> target query). The final-paragraph rewrite (also called the "final-unit rewrite" in the
  paper) replaces the last `\n\n`-separated paragraph with a fixed template naming the target four
  times (`edits.rewrite_final_unit`). Task switching substitutes the answer word and task-description
  phrases from a **hand-written task lexicon** (`tasks.TASK_LEXICON`, read off real AV output,
  longest phrase first).
- **Patching.** AR outputs are **rescaled to the native norm** of the residual they replace and
  written into the output of the NLA-layer decoder block **during the prefill pass only**. The
  model then decodes **greedily for 8 new tokens**; a trial succeeds if the first output line (minus
  an echoed `A:`, bold markers and punctuation) equals the target (accent- and case-insensitive,
  except case-sensitive for capitalize).
- **Controls**, in the same generation batch: unpatched model, unedited round trip `AR(AV(h))`,
  a no-op edit through identical machinery, an AV-free template explanation, and **donor patching**
  (the same positions patched with real activations from a prompt that natively gives the target:
  the target query with the same demonstrations, or the same query with target-task
  demonstrations for switching). Seeds 0-2 also get a donor sweep over layers
  0, 4, ..., 44, 47 (**3 seeds**, 180 trials per layer and span for redirection).
  The `query+last4` span and the `template` arm are computed but not reported; they share
  generation batches with the reported arms, so they are kept to keep runs bit-identical.
- **Layers 24 and 40 use different NLAs.** Layer 32 uses the official released pair
  (`kitft/nla-gemma3-12b-L32-{av,ar}`). Layers 24 and 40 use custom EasyNLA pairs
  (`soar_nla.hub.CUSTOM_L24/L40`, RL iteration 400) trained with one shared recipe, which differs
  from the released pair (LoRA AV with additive injection after decoder layer 1; see `easynla.py`).
  Comparisons across layers therefore confound layer with NLA training.

## Setup

- One 8 x A100-80GB node was used. Layer 32 (`kitft` backend): 4 SGLang AV servers (GPUs 4-7) plus
  one worker per GPU holding the target model and the AR. Layers 24/40 (`easynla` backend): four
  workers, each on a GPU pair (target + AR on the first GPU, AV in-process on the second).
  About 8.5 minutes per task per worker.
- Python deps: `torch`, `transformers` (4.57), `huggingface_hub`, `safetensors`, `pyyaml`, `numpy`,
  `matplotlib`; `peft` for the easynla backend; SGLang (0.5.9) patched with
  `patches/sglang/apply_sglang_patches.sh` for the AV servers (interpreter: `$SOAR_SGLANG_PYTHON`).
- Every checkpoint is loaded at the revision pinned in `soar_nla/hub.py`; a config naming an unpinned
  repo is rejected. Results go to `results/icl_edits/<config>/` (root overridable with `$SOAR_RESULTS`).

## Commands

```bash
# 1. Experiments (GPU). GPU lists are env vars, e.g. AV_GPUS=4,5,6,7 REDIRECT_A_GPU=0 ...
soar_nla/icl_edits/launch_l32_raw.sh        # layer 32, raw text: redirect (2 shards) + switch
soar_nla/icl_edits/launch_l32_chat.sh       # layer 32, chat template: redirect (2 shards) + switch
soar_nla/icl_edits/launch_l24_l40_raw.sh    # layers 24 and 40, raw text, redirect only
#    single run:  python -m soar_nla.icl_edits.run serve-av --config gemma3-12b --gpus 4,5 --ports 31000,31001
#                 python -m soar_nla.icl_edits.run exp --config gemma3-12b --gpu 0 --av-ports 31000,31001 \
#                     --exp icl_redirect --fmt raw --only antonym --tag a
# 2. Summaries with item-bootstrap intervals (CPU; the launchers also run this)
python -m soar_nla.icl_edits.analysis --config gemma3-12b
python -m soar_nla.icl_edits.analysis --config gemma3-12b-custom-L24
python -m soar_nla.icl_edits.analysis --config gemma3-12b-custom-L40
# 3. Figures, rate tables and results paragraphs (CPU)
python -m soar_nla.icl_edits.figures
# 4. Example explanation edits figure text (CPU)
python -m soar_nla.icl_edits.example_edits
# 5. Every number quoted in the paper, checked against its printed value (CPU)
python -m soar_nla.icl_edits.paper_numbers
```

Steps 2-5 run on the shipped results as they are (the per-row files are stored gzipped; the readers
accept `.json` or `.json.gz` and prefer `.json`). An experiment resumes from its plain `.json` file:
a rerun skips finished tasks and redoes a partially finished one; it does not read the shipped `.gz`
copies, so a fresh run writes new `.json` files next to them.

## Outputs

`results/icl_edits/<config>/`: per-row results `icl_redirect[_raw]-{a,b}.json.gz`,
`icl_switch[_raw].json.gz` (rows: every arm x span answer and hit; `sweep`: donor layer sweep),
`summary.json` (`analysis`), and for layers 24/40 the run manifest. Sizes: 75-125 KB per gzipped file,
about 1 MB in total.

`results/icl_edits/figures/` (PNG + PDF):

| file | paper figure | earlier name (duplicates removed) |
|---|---|---|
| `icl_redirect_raw_L32` | ICL edits: target rates per task, final token vs four tokens | `icl_tasks_raw_shared_axes` (= `icl_nla_layer32_raw_matched`) |
| `example_edits_country-capital.md` | "Example explanation edits" (China -> Chile, Beijing -> Santiago) | none (new) |
| `donor_sweep_raw_L32` (+ `_caption.txt`) | Appendix: donor patching across layers | `donor_patching_raw` |
| `donor_sweep_raw_L32_by_task` | supporting: final-token donor sweep per task | `donor_patching_raw_by_task` |
| `icl_redirect_raw_L24_L40` | Appendix: layers 24 and 40 (stacked) | `icl_nla_layer24_40_raw_stacked_dots` (= `_merged`, `_stacked`, `_stacked_with_query`) |
| `icl_redirect_raw_L24`, `icl_redirect_raw_L40` | Appendix: one layer per figure | `icl_nla_layer{24,40}_raw` (= `_matched`) |
| `icl_redirect_chat_L32` | Appendix: chat template | `icl_tasks_chat_header_only` (= `icl_tasks_chat_shared_axes`, `icl_nla_layer32_chat_matched`) |
| `task_switch_raw_L32` | Task switching: three switches, query held fixed | `icl_task_switching_raw_shared_axes` |
| `task_switch_chat_L32` | Task switching appendix: chat template | `icl_task_switching_chat_shared_axes` (= `_chat_header_only`) |
| `task_switch_raw_chat_L32` | supporting: both formats stacked | `icl_task_switching_raw_chat_stacked` |

Rate tables with provenance: `icl_redirect_raw_L24_L40_rates.json`, `icl_redirect_chat_L32_rates.json`,
`task_switch_L32_rates.json`; results paragraphs with computed numbers: `icl_redirect_raw_L24_L40.md`
(including the controls paragraph), `icl_redirect_chat_L32.md`, `task_switch_L32.md`.
`results/icl_edits/paper_numbers.json`: every quoted number with its recomputed value.

## Paper claims

All values below are recomputed by `paper_numbers.py` from the per-row files (fields
`<span>|<arm>|hit`, averaged over rows; sweep values from `sweep` rows); the same rates appear in
`<config>/summary.json` under `icl_<kind>[_raw].<span>.<arm>[0]` and `...sweep["<span>|<layer>"]`.
Spans: `last1` final token, `last4` last four tokens, `query` query token.

| paper claim | config / file | span, arm | value |
|---|---|---|---|
| ICL, final token: substitution 64%, rewrite 46%, combined 79%, donor 81% | gemma3-12b / `icl_redirect_raw` | last1: substitute_only, rewrite_only, edited, donor | .635, .460, .787, .805 |
| ICL, Capitalize weak under combined edit and donor (final token) | gemma3-12b / `icl_redirect_raw`, per_task.capitalize | last1: edited, donor | .31, .28 |
| ICL, four tokens: substitution 98%, combined 98%, rewrite 59%, donor 97% | gemma3-12b / `icl_redirect_raw` | last4: substitute_only, edited, rewrite_only, donor | .977, .977, .585, .970 |
| Donor sweep: query token 94-98% through L24, 47% L28, 7% L32 | gemma3-12b / `icl_redirect_raw` sweep | query | .944-.983, .472, .067 |
| Donor sweep: final `:` 0 through L24, 38% L28, 78% L32, 98% L40 | gemma3-12b / `icl_redirect_raw` sweep | last1 | 0, .383, .783, .978 |
| Donor sweep: last four tokens 94-98% through L40 | gemma3-12b / `icl_redirect_raw` sweep | last4 | .944-.983 |
| L24, query token: substitution 96%, donor 97%, combined 88%, rewrite 19% | gemma3-12b-custom-L24 / `icl_redirect_raw` | query | .960, .970, .882, .193 |
| L24, four tokens: substitution 92%, combined 81%, rewrite 17% | gemma3-12b-custom-L24 / `icl_redirect_raw` | last4 | .922, .810, .165 (donor .977) |
| L24, final-token edits nearly ineffective | gemma3-12b-custom-L24 / `icl_redirect_raw` | last1: edited | .008 |
| L40, query-token edits produce no target answers | gemma3-12b-custom-L40 / `icl_redirect_raw` | query: all edits | 0 |
| L40, final token: substitution 70%, rewrite 58%, combined 70%, donor 97% | gemma3-12b-custom-L40 / `icl_redirect_raw` | last1 | .695, .583, .697, .973 |
| L40, four tokens: 87%, 87%, 92%, donor 98% | gemma3-12b-custom-L40 / `icl_redirect_raw` | last4 | .870, .873, .920, .975 |
| L40, Capitalize: combined 40% vs donor 100% (final token), 92% (four tokens) | gemma3-12b-custom-L40 / per_task.capitalize | last1, last4: edited, donor | .40, 1.00, .92 |
| Chat: combined 72% vs donor 8.7% (final token); 92% vs 84% (four tokens) | gemma3-12b / `icl_redirect` | last1, last4: edited, donor | .717, .087; .923, .835 |
| Chat: substitution 48% (final token), 79% (four tokens) | gemma3-12b / `icl_redirect` | last1, last4: substitute_only | .475, .793 |
| Switch, final token: task+answer 0.440, combined 0.847, donor 0.840 | gemma3-12b / `icl_switch_raw` | last1: answer+task, edited, donor | .440, .847, .840 |
| Switch, four tokens: task+answer 0.470, combined 0.877, donor 0.897 | gemma3-12b / `icl_switch_raw` | last4 | .470, .877, .897 |
| Switch: answer-only 0.057, task-only 0.003 | gemma3-12b / `icl_switch_raw` | last1 and last4: answer_only, task_only | .057, .003 (both spans) |
| Switch, chat, final token: 0.463 / 0.950 / donor 0.517 | gemma3-12b / `icl_switch` | last1 | .463, .950, .517 |
| Switch, chat, four tokens: 0.543 / 0.983 / donor 0.783 | gemma3-12b / `icl_switch` | last4 | .543, .983, .783 |
| Switch: round-trip and no-op controls give no target answers (both formats) | gemma3-12b / `icl_switch[_raw]` | last1, last4: roundtrip, noop | 0 |

## Layout

```
configs/             gemma3-12b (official L32 pair), gemma3-12b-custom-L24 / -L40 (custom pairs)
config.py            config loading; every repo must be pinned in soar_nla.hub
runtime.py           target + AR on one GPU, AV over SGLang, prefill-only residual replacement
easynla.py           EasyNLA backend (LoRA AV with additive injection, truncated AR)
tasks.py             FV tasks, task switches, task lexicon, demonstration sampling
edits.py             substitution, final-paragraph rewrite, no-op, text-only draw selection
experiment.py        one experiment: prompts, AV draws, edits, AR, patching, donor sweep
scoring.py, stats.py answer parsing; item-level cluster bootstrap
run.py               CLI: serve-av, exp
analysis.py          summary.json
figures.py, figure_template.py   paper figures and rate tables
example_edits.py     the three edit variants of one saved explanation
paper_numbers.py     recompute and check every quoted number
launch_*.sh          one launcher per setting
```
