"""The three runs of the warm-start ablation and where their artifacts live.

All three train an NLA on Qwen/Qwen2.5-7B-Instruct, residual stream after
decoder block 20, with the same data, sidecar, optimiser and GRPO settings.
They differ only in actor initialisation and KL coefficient:

    A  warm start: AV/AR SFT on Sonnet-written summaries, then RL, KL 0.01
    B  cold start: RL from the base model (fresh LoRA, identity-head AR), KL 0
    C  cold start: RL from the base model, KL 0.01

Training launchers: training/scripts/qwen25_warmstart_ablation/run_{A,B,C}_*.sh.
"""

from __future__ import annotations

RUNS = ("A", "B", "C")

RUN_LABELS = {
    "A": "Warm start (SFT), KL 0.01",
    "B": "Cold start, KL 0",
    "C": "Cold start, KL 0.01",
}

# Per-step training curves rebuilt from the trainer logs by parse_training_logs.py.
FVE_CSV = {
    "A": "fve_curve_A_warmstart_kl001.csv",
    "B": "fve_curve_B_cold_kl0.csv",
    "C": "fve_curve_C_cold_kl001.csv",
}

# Trainer logs of each run, in leg order (see training/scripts/qwen25_warmstart_ablation).
TRAINING_LOGS = {
    "A": ("rl25.log", "qwen25_l20_resume400_to800.log"),
    "B": ("rl25_cold_origprompt_kl0.log", "rl25_cold_origprompt_kl0_rawfallback.log"),
    "C": ("rl25_cold_nosft.log", "rl25_cold_nosft_resume600_1200.log"),
}

# All three runs are compared at step 600, the largest budget common to all of them.
COMPARE_STEP = 600

# Pinned Hugging Face revisions. The runs resolved these repos by name; the
# revisions below are the ones current at run time (each repo's last commit
# predates the runs).
QWEN25_7B = ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")
SELFDESCRIBE = (
    "Transluce/SelfDescribe-Llama-3.1-70B-Instruct",
    "52aff6679579efb92dcf30e4bf2b8ddc20f132ad",
)
SELFDESCRIBE_FILE = "prompts.jsonl"

# Training data (activations regenerated for Qwen2.5-7B-Instruct block 20) and
# the checkpoints of the three runs, as published during the project. The model
# repo also holds unrelated experiments; only the subfolders below are used here.
NLA_DATA = (
    "Yooniel/qwen2.5-7b-instruct-nla-L20-finefineweb-100k",
    "811f4afffe7fc32e7bd217f0174462f7c07ff03a",
)
NLA_CKPTS = (
    "Yooniel/qwen2.5-7b-instruct-nla-L20",
    "8b35afe5186a27dabc7bd41f2cded6f133e4b5e6",
)
CKPT_SUBDIRS = {
    # gen_from_acts.py inputs at the compared step.
    "A": {
        "av_ckpt": "merged/av_hf",
        "av_adapter": "av_sft/iter_0003834",
        "lora": "rl_vllm/iter_000600",
    },
    "B": {"lora": "other/ckpts25_cold_origprompt_kl0_rawfallback/rl_vllm/iter_000600"},
    "C": {"lora": "cold_nosft/rl_vllm/iter_000600"},
}
