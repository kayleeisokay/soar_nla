# Shared Alpaca rollouts

The data behind two parts of the paper: **GCG** (`soar_nla.gcg`) and the
**surgical edits** in Faithfulness (`soar_nla.faithfulness.surgical_edits`).
**Loss Recovered** uses different Alpaca rollouts (`soar_nla.loss_recovered`).

## Protocol

- Dataset: `tatsu-lab/alpaca` at revision `dce01c9b08f87459cf36a430d809084718273017`, train split.
  Prompts are rendered as `instruction` or `instruction\n\nInput:\n{input}`, de-duplicated by
  content hash, ordered by `stable_hash("av-faithfulness-v1-alpaca-2000", index, hash)`, and the
  first 2,000 are kept (`data.py`).
- Rollouts: greedy Gemma-3-12B-it chat completions, at most 192 new tokens, user prompt truncated
  to 768 tokens (`rollouts.py`).
- Activation: one assistant-token position per response, chosen as
  `stable_seed("av-faithfulness-v1", "gemma12b", sample_id, "assistant-position") % len(response)`,
  i.e. uniform over the generated span. The saved vector is the output of decoder block 32.
- AV decoding: the official `kitft/nla-gemma3-12b-L32-av` (pinned in `soar_nla.hub`), greedy,
  at most 220 new tokens, the checkpoint's own prompt and injection scale (`decode.py`).

`config.py` holds the frozen constants. `PROTOCOL_VERSION` and `SELECTION_SEED` feed the hashes
above, so changing either string changes which prompts and positions are used.

## Subsets (`subsets.py`)

| Helper | Filter | Rollouts | Used by |
|---|---|---:|---|
| `long_responses` | `assistant_token_count >= 100` | 1,543 | reference only (not the Loss Recovered windows) |
| `late_positions` | `sequence_token_index > 50` | 1,573 | GCG and its random control, surgical edits |
| both | | 1,348 | |

`is_word_token(sampled_token_text)` separates sampled tokens containing a letter or digit
(1,056 of the 1,573) from punctuation, markdown and whitespace tokens.

## Setup and hardware

- 4 GPUs with 80 GB for rollouts (one shard per GPU, batch size 4) and decoding.
- Decoding drives the AV through SGLang `input_embeds`. SGLang must be patched with
  `patches/sglang/apply_sglang_patches.sh`; point `SOAR_SGLANG_PYTHON` at that interpreter.
- Greedy bf16 generation is batched with left padding, so outputs can depend on which prompts
  share a batch. The paper used 4 shards (`selection_rank % 4`) with batch size 4; keep
  `GPUS` at four entries and `ROLLOUT_BATCH=4` to reproduce it exactly.

## Commands

```bash
bash soar_nla/alpaca/run.sh data       # CPU; downloads Alpaca, writes data/alpaca_selected.jsonl
bash soar_nla/alpaca/run.sh rollouts   # 4 GPUs
bash soar_nla/alpaca/run.sh decode     # 4 GPUs, one SGLang server each
```

or the stages individually:

```bash
python -m soar_nla.alpaca.data
CUDA_VISIBLE_DEVICES=0 python -m soar_nla.alpaca.rollouts worker --shard 0 --num-shards 4   # ... shards 1-3
python -m soar_nla.alpaca.rollouts merge
python -m soar_nla.alpaca.decode worker --shard 0 --num-shards 4 --sglang-url http://127.0.0.1:31000  # ...
python -m soar_nla.alpaca.decode merge
```

## Outputs (`$SOAR_RESULTS/alpaca`, default `results/alpaca`)

| File | Content |
|---|---|
| `data/alpaca_selected.jsonl`, `data/dataset_manifest.json` | the 2,000 selected prompts |
| `gemma12b/rollouts/rollouts.jsonl` | one row per prompt: response, sampled position, sampled token, passage through that token |
| `gemma12b/rollouts/activations.safetensors` | `[2000, 3840]` fp32 layer-32 activations, row = `selection_rank` |
| `gemma12b/decode/explanations.jsonl` | AV explanation per rollout |

Shipped in this repository: `data/dataset_manifest.json`, `gemma12b/rollouts/manifest.json` and
`gemma12b/rollouts/rollouts.jsonl.gz` (the paper's rollouts, without activations). The analysis
scripts read the `.gz` file when no regenerated `rollouts.jsonl` exists. Activations and AV
explanations are not shipped; regenerate them with the commands above.

## Paper claims

| Claim | Script | Output | Value |
|---|---|---|---|
| 1,573 Alpaca prompts (surgical edits) | `subsets.late_positions` | `rollouts.jsonl` | 1,573 |
