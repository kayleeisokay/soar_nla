# Surgical edits to AV explanations

Reproduces the surgical-edit paragraphs of the paper's **Faithfulness** section: GPT-5.6 Sol
corrects factual errors in AV explanations of the official Gemma-3-12B L32 NLA, and the AR
reconstructs the original and edited explanations.

- Examples: the shared Alpaca rollouts (`soar_nla.alpaca`) whose sampled position is after
  sequence token 50 (1,573), in selection-rank order; the first 200 were edited, in two runs of 100.
- Editor: `correct_av_explanations.py`, OpenAI Responses API, model `gpt-5.6-sol`,
  `reasoning={"effort": "low"}`, `max_output_tokens=2048`. The prompt (`PROMPT`, with its three
  worked examples) is the exact string used for the paper. It asks the editor to copy the
  explanation and change only words or phrases that conflict with the passage, to make claims
  about quotations, exact wording and the final token match the passage, to replace a wrong
  quotation with a verbatim span of the same word count, and to leave predictions unchanged.
- Scoring: `score_corrected_av.py`, per-example FVE under the GCG convention
  (`soar_nla/gcg/README.md`), `fve_delta = edited FVE − original FVE`.
- Breakdown: `analysis.py`, by what happened to the explanation's `Final token "..."` claim and
  by the type of the true sampled token (definitions in its docstring).

## Setup

- `pip install openai` (the `judge` extra) and `OPENAI_API_KEY` in the environment or in a
  dotenv file passed with `--env-file`.
- Scoring needs one GPU and the `soar_nla.alpaca` activations; `analysis.py` is CPU-only.

## Commands

```bash
python -m soar_nla.faithfulness.surgical_edits.correct_av_explanations \
    --offset 0 --limit 100 --output results/surgical_edits/sol_equal_quote_length_100.jsonl
python -m soar_nla.faithfulness.surgical_edits.correct_av_explanations \
    --offset 100 --limit 100 --output results/surgical_edits/sol_equal_quote_length_101_200.jsonl

python -m soar_nla.faithfulness.surgical_edits.score_corrected_av \
    --corrections results/surgical_edits/sol_equal_quote_length_100.jsonl
python -m soar_nla.faithfulness.surgical_edits.score_corrected_av \
    --corrections results/surgical_edits/sol_equal_quote_length_101_200.jsonl

python -m soar_nla.faithfulness.surgical_edits.analysis
```

Editor outputs are not deterministic, so a rerun gives different edits; the paper's edits are
shipped. `analysis.py` with no arguments runs on the shipped files.

## Outputs (`results/surgical_edits`, all shipped)

| File | Content |
|---|---|
| `sol_equal_quote_length_100.jsonl`, `sol_equal_quote_length_101_200.jsonl` | the paper's 200 Sol edits: passage, original and edited explanation (216 KB, 204 KB) |
| `*_reconstruction.jsonl` | per-example original/edited MSE and FVE |
| `*_reconstruction_summary.json` | per-run mean ΔFVE, bootstrap CI |
| `analysis.json` | the breakdown below |

## Paper claims

| Claim | Script | Output field (`analysis.json`) | Value |
|---|---|---|---|
| Across all 200 examples, ΔFVE −0.2% | `analysis.py` | `groups.all.mean_fve_delta` | −0.0017 (n = 200) |
| Editor kept the AV's final token: +3.1% | `analysis.py` | `groups.final_token_kept.mean_fve_delta` | +0.0309 (n = 137) |
| Editor changed it, even when the AV's token was wrong: −5.4% | `analysis.py` | `groups.final_token_changed_av_wrong.mean_fve_delta` | −0.0536 (n = 32) |
| Excluding whitespace and punctuation tokens: +2.4% | `analysis.py` | `groups.word_tokens.mean_fve_delta` | +0.0238 (n = 143) |
| GCG on the same examples: 8.4% at β = 0.01 | `soar_nla.gcg.paper_numbers` | `results/gcg/paper_numbers.json` | 0.0844 (n = 143) |

Notes on the definitions:

- "Kept" means the `Final token "..."` string in the edited explanation is identical to the AV's.
  All changed claims together give −0.0727 (n = 63): −0.0536 where the AV's claim was wrong and
  −0.0925 (n = 31) where the editor changed only formatting.
- "Wrong" means the passage does not end with the AV's claim after ignoring case, whitespace,
  backticks and quotes (a whitespace-only claim counts as right if the passage ends in
  whitespace). 32 of 200 AV claims are wrong: 5 of 143 word tokens and 27 of 57
  punctuation/whitespace tokens. Every wrong claim was changed by the editor.
- Whitespace/punctuation exclusion keeps the 143 examples whose true sampled token contains a
  letter or digit; this is also the subset for the GCG 8.4%. On it the edits give
  +0.0238 [+0.0093, +0.0362].
