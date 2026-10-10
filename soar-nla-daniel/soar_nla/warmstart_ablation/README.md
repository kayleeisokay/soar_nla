# Warm-start ablation (appendix "Skipping Warm-Start")

This directory reproduces the appendix section **Skipping Warm-Start**. The
question is whether an NLA can be trained with RL directly from the base model,
skipping the SFT warm start on LLM-written summaries, and what that costs in
reconstruction and in information a third party can recover from the
explanations.

All three NLAs are trained on **`Qwen/Qwen2.5-7B-Instruct`**, reading the
residual stream after decoder block 20. The paper text does not name the
model. Training lives in `training/` (vendored EasyNLA); the launchers are in
`training/scripts/qwen25_warmstart_ablation/` and documented in
`training/README.md`.

| Run | Initialisation | KL β |
|---|---|---|
| A | warm start: AV/AR SFT, then RL | 0.01 |
| B | cold start: RL from the base model (fresh LoRA, identity-head AR) | 0 |
| C | cold start | 0.01 |

All comparisons are at RL step 600, the largest budget common to all three runs.

## Setup

- **Python packages.** The `nla` package must be importable: either
  `pip install -e training/` or put `training/` on `PYTHONPATH`.
- **CPU-only modules.** The analysis modules (`parse_training_logs`,
  `fve_table`, `plot_fve`, `analyze_5way`, `ci_bootstrap`) need only the
  standard scientific stack and matplotlib.
- **vLLM modules.** `gen_from_acts` and `judge_5way` run in the vllm-lens
  venv (vllm 0.19.0, patched vllm-lens; see `training/docs/vllm-lens-setup.md`)
  with `VLLM_ALLOW_INSECURE_SERIALIZATION=1`. Source `training/env.sh`.
- **GPUs.** Generation needs 1 GPU (80 GB) per run: about 3 min per run for
  2,585 activations. The judge needs 2 GPUs (TP=2): about 4 min per arm plus
  engine start-up.
- **Pinned models and data.** All are pinned in `runs.py`:
  - Qwen2.5-7B-Instruct `a09a354…`
  - SelfDescribe `52aff66…`
  - Gemma-3-27B-IT via `soar_nla.hub.GEMMA3_27B`
  - the published data and checkpoint revisions

## Commands

### 1. FVE curves and the reconstruction numbers (CPU, from the trainer logs)

```bash
W=soar_nla.warmstart_ablation
python -m $W.parse_training_logs --logs $LOGS/rl25.log $LOGS/qwen25_l20_resume400_to800.log \
    --out results/warmstart_ablation/fve_curve_A_warmstart_kl001.csv
python -m $W.parse_training_logs --logs $LOGS/rl25_cold_origprompt_kl0.log $LOGS/rl25_cold_origprompt_kl0_rawfallback.log \
    --out results/warmstart_ablation/fve_curve_B_cold_kl0.csv
python -m $W.parse_training_logs --logs $LOGS/rl25_cold_nosft.log $LOGS/rl25_cold_nosft_resume600_1200.log \
    --out results/warmstart_ablation/fve_curve_C_cold_kl001.csv
python -m $W.fve_table          # -> fve_table.{md,json}
python -m $W.plot_fve           # -> fve_train.png
```

`$LOGS` is the training launchers' `$WORK/logs`. The trainer logs are not
shipped (10–15 MB each). The three CSVs they produce are in `results/`.
`parse_training_logs` regenerates the CSVs used for the paper byte for byte
(`--compare` checks this).

### 2. SelfDescribe attribute recovery (GPU)

```bash
W=soar_nla.warmstart_ablation; O=$WORK_DIR/selfdescribe
# activations: drop the constant task suffix, append " I", take block-20 output at that token
python -m $W.extract_selfdescribe_acts --attr-class all --n-per-attr 0 --mode append_i --out $O/all_acts_I.jsonl
# step-600 checkpoints + RL sidecar (prints the gen_from_acts arguments per run)
python -m $W.fetch_checkpoints --out $O/hf
# one explanation per activation and run; output names must be all_{A,B,C}.jsonl (the judge reads the run from them)
python -m $W.gen_from_acts --run A --av-ckpt $O/hf/ckpts/merged/av_hf --av-adapter $O/hf/ckpts/av_sft/iter_0003834 \
    --lora $O/hf/ckpts/rl_vllm/iter_000600 --parquet $O/hf/data/rl_shuf.full.parquet --meta $O/hf/data/rl_shuf.full.parquet \
    --acts $O/all_acts_I.jsonl --out $O/all_A.jsonl
python -m $W.gen_from_acts --run B --av-ckpt Qwen/Qwen2.5-7B-Instruct --raw-fallback \
    --lora $O/hf/ckpts/other/ckpts25_cold_origprompt_kl0_rawfallback/rl_vllm/iter_000600 \
    --parquet $O/hf/data/rl_shuf.full.parquet --meta $O/hf/data/rl_shuf.full.parquet --acts $O/all_acts_I.jsonl --out $O/all_B.jsonl
python -m $W.gen_from_acts --run C --av-ckpt Qwen/Qwen2.5-7B-Instruct \
    --lora $O/hf/ckpts/cold_nosft/rl_vllm/iter_000600 \
    --parquet $O/hf/data/rl_shuf.full.parquet --meta $O/hf/data/rl_shuf.full.parquet --acts $O/all_acts_I.jsonl --out $O/all_C.jsonl
# judge: real arm and control arm (each item shown a donor description from the same run and class)
CUDA_VISIBLE_DEVICES=0,1 python -m $W.judge_5way --inputs $O/all_A.jsonl $O/all_B.jsonl $O/all_C.jsonl --tp 2 --out $O/judge5_real.jsonl
CUDA_VISIBLE_DEVICES=0,1 python -m $W.judge_5way --inputs $O/all_A.jsonl $O/all_B.jsonl $O/all_C.jsonl --tp 2 --control --out $O/judge5_control.jsonl
# summaries (CPU; also accept the shipped .jsonl.gz files)
python -m $W.analyze_5way --real $O/judge5_real.jsonl --control $O/judge5_control.jsonl \
    --out-json results/warmstart_ablation/judge_summary.json | tee results/warmstart_ablation/judge_summary.txt
python -m $W.ci_bootstrap --inputs $O/judge5_real.jsonl --names gemma-3-27b-it > results/warmstart_ablation/judge_ci_bootstrap.txt
```

#### Judge protocol

- **Options.** Five options: the gold value plus 4 distractors drawn from the
  same class. Gender has only 3 values, so its items are 3-way; pooled chance
  is 23.7%.
- **Seeding.** Options are shuffled with a per-item seed, `"{seed}|{class}|{idx}"`.
- **Decoding.** Greedy, one output token. The letter posterior comes from the
  top-20 logprobs.
- **Control arm.** The donor description comes from the same run and attribute
  class, never the item's own. Donors are drawn with `random.Random(seed + 999)`.

The prompt template and question wording are those of the paper.

Generation uses the trainer's own rollout path (`rollout_batch_vllm`): T = 1.0,
256 new tokens, one sample per activation. Run B is decoded with the raw
completion fallback it was trained with. It stops closing its
`<explanation>` tag, so the whole completion is the message. Extraction
succeeded for 2,585/2,585 (A), 2,585/2,585 (B, all via the fallback) and
2,568/2,585 (C).

### 3. Example verbalizations (GPU, qualitative)

The appendix shows one explanation per run of the held-out document ending in
"… The annual" (doc `…:train:71467`). These are single generations, not numbers:

| Run | Checkpoint | How |
|---|---|---|
| A | `rl_vllm/iter_000800` | `verbalize_text.py`: activation re-extracted from `results/warmstart_ablation/example_source_text.txt`, the document without its byline. Prints a greedy completion plus 4 samples; the run behind the paper used `--max-new-tokens 512`. Which completion the paper shows is not recorded. |
| B | leg-2 `iter_000600` | `export_examples.py --raw-completion-fallback` (greedy), example 3 |
| C | `cold_nosft/rl_vllm/iter_000400` | `export_examples.py` (greedy), example 3 |

```bash
python -m $W.verbalize_text --text-file results/warmstart_ablation/example_source_text.txt \
    --lora $O/hf/ckpts/rl_vllm/iter_000800 --parquet $O/hf/data/rl_shuf.full.parquet --max-new-tokens 512
python -m $W.export_examples --adapter <B iter_000600> --critic <B critic_latest> --raw-completion-fallback \
    --parquet $O/hf/data/rl_shuf.full.parquet --output $O/examples_B.txt
python -m $W.export_examples --adapter <C iter_000400> --critic <C critic_latest> \
    --parquet $O/hf/data/rl_shuf.full.parquet --output $O/examples_C.txt
```

`fetch_checkpoints` does not download these folders; get them from the same
repo with `huggingface-cli download --include`. The AV message does not depend
on `--critic`; only the printed reconstruction does. The original C export
used the step-400 critic, which is not published.

## Outputs (`results/warmstart_ablation/`)

| File | Content |
|---|---|
| `fve_curve_{A_warmstart_kl001,B_cold_kl0,C_cold_kl001}.csv` | Per-step training-batch FVE (rank mean, EMA, std/min/max over the 8 ranks), reward, AR MSE, KL, entropy, extraction rate, held-out FVE every 10 steps, source log |
| `fve_table.{md,json}` | Peak and held-out FVE at steps ≤ 600 |
| `fve_train.png` | Training-batch FVE curves |
| `judge5_{real,control}.jsonl.gz` | Per-item judge outputs (7,755 rows each: options, gold/pick letter, posterior) |
| `judge_summary.{txt,json}` | Accuracy per run and class, Wilson CI, control accuracy, chance |
| `judge_ci_bootstrap.txt` | Wilson / item-bootstrap / cluster-bootstrap CIs and paired run differences |
| `example_source_text.txt` | Source text of the run-A example verbalization |

## Paper figures

```bash
python -m soar_nla.warmstart_ablation.paper_figure   # CPU only, reads the shipped files
```

- `figures/training_fve.{png,pdf}`: training-batch FVE of A, B and C up to step 600 with start and
  best values called out (`fve_curve_*.csv`, unsmoothed; `fve_train.png` is the EMA view).
- `figures/selfdescribe_judge.{png,pdf}`: judge accuracy per attribute with Wilson 95% intervals
  and each attribute's chance rate (`judge_summary.json`).

Legends: "Original" = A, "No warm-start, no KL" = B, "No warm-start" = C.

## Paper claims

| Paper claim | Script | Output → field | Value |
|---|---|---|---|
| Peak FVE, A (warm start) 75.9% | `fve_table` | `fve_table.json` → `runs.A.peak_fve_train` | 0.758875 (step 537) |
| Peak FVE, B (cold, KL 0) 72.3% | `fve_table` | `runs.B.peak_fve_train` | 0.7225 (step 537) |
| Peak FVE, C (cold, KL 0.01) 68.9% | `fve_table` | `runs.C.peak_fve_train` | 0.689375 (step 597) |
| Held-out FVE 74.4 / 70.9 / 66.5% | `fve_table` | `runs.{A,B,C}.heldout_fve` | 0.744 (step 600) / 0.709 (step 590) / 0.665 (step 600) |
| 2,585 SelfDescribe examples | `analyze_5way` | `judge_summary.json` → `{A,B,C}.POOLED.n` | 2585 |
| Judge accuracy 69.9 / 68.2 / 60.9% | `analyze_5way` | `judge_summary.json` → `{A,B,C}.POOLED.accuracy` | 0.6994 / 0.6824 / 0.6085 |
| Control accuracy close to chance | `analyze_5way` | `{A,B,C}.POOLED.control_accuracy` vs `chance` | 0.2453 / 0.2387 / 0.2545 vs 0.237 |

How the numbers are defined:

- **Peak FVE** is the best single training-batch step at step ≤ 600: the mean
  over the 8 data-parallel ranks, each scoring 1/8 of the 256 × 8 batch.
- **Held-out FVE** is the last periodic evaluation at step ≤ 600: 128 prompts
  from held-out documents. B's run ends at step 599, so its last evaluation is
  at step 590.
- **The step-600 evaluation** runs before the update of step 600, so it scores
  checkpoint `iter_000600`. For C it comes from the first evaluation of its
  third leg.
