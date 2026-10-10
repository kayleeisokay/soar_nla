# %%
from __future__ import annotations

# -----------------------------------------------------------------------------
# EDIT THESE VALUES FOR YOUR CLUSTER ALLOCATION
# -----------------------------------------------------------------------------
# USER_NAME is your Linux username / /home folder name.
# VISIBLE_GPUS are the physical GPU IDs assigned to you, comma-separated.
# After CUDA masking, the first listed physical GPU becomes cuda:0 in this notebook.
# SGLANG_PHYSICAL_GPU is the physical GPU reserved for the SGLang AV server.
USER_NAME = "kaylee"
VISIBLE_GPUS = "0, 1, 2, 3"
SGLANG_PHYSICAL_GPU = "1,2"
LOCAL_DEVICE = "cuda:0"
SGLANG_TP_SIZE = 4
# -----------------------------------------------------------------------------

# Run this notebook from a fresh kernel. CUDA_VISIBLE_DEVICES must be set before
# importing torch, so this setup cell intentionally defines paths/GPUs first.
import os
from pathlib import Path

# %% [markdown]
# NLA round-trip accuracy degradation experiment.
# 
# Idea: patch a single token's activation at a single layer with its
# NLA round-trip reconstruction 
# $$
# (h -> AV(h) -> text -> AR(text) -> h_hat)
# $$
# then let generation continue normally. Compare downstream task accuracy
# against (a) an unpatched baseline and (b) a no-op patch control that
# re-injects the *original* activation unchanged.
# 
# This script assumes you have:
#   - a HF-loadable Gemma model (`MODEL_NAME`)
#   - your own trained AV and AR modules from natural_language_autoencoders,
#     exposed as callables: av(h_vec) -> str, ar(text) -> torch.Tensor
#   - GSM8K test split via `datasets`
# 
# Fill in the two TODOs (load_av_ar, extract_final_answer) for your repo's
# actual API / answer-parsing format before running.

# %% [markdown]
# ## Experiment Overview
# 
# All experiments share one AV server and use the same target tokenizer/model state where possible. The notebook deliberately unloads the 27B target before loading the AR, then reloads the target for introspection, so the full target and AR do not coexist on the notebook GPU.
# 
# Results are printed and displayed near the cells that produce them. Round-trip and paraphrasing summaries are also written beneath the configurable experiment output root; introspection and zero-vector results remain notebook-only.

# %%
HOME = Path("/home") / USER_NAME
NLA_REPO = Path(os.environ.get("NLA_REPO", HOME / "natural_language_autoencoders"))
SGLANG_REPO = Path(os.environ.get("SGLANG_REPO", HOME / "sglang"))
VENV_PYTHON = Path(os.environ.get("NLA_VENV_PYTHON", NLA_REPO / ".venv/bin/python"))
HF_CACHE = Path(os.environ.get("HF_HOME", HOME / ".cache/huggingface"))
ROOT = Path(os.environ.get("NLA_EXPERIMENT_ROOT", HOME / "experiments/nla_experiments"))
LIBNUMA_DIR = Path(
    os.environ.get(
        "NLA_LIBNUMA_DIR", NLA_REPO / "vendor/libnuma/usr/lib/x86_64-linux-gnu"
    )
)
PYTHON_INCLUDE_DIRS = [
    Path(
        os.environ.get(
            "NLA_PYTHON_INCLUDE",
            NLA_REPO / "vendor/python3.10-dev/usr/include/python3.10",
        )
    ),
    Path(
        os.environ.get(
            "NLA_PYTHON_MULTIARCH_INCLUDE",
            NLA_REPO / "vendor/python3.10-dev/usr/include",
        )
    ),
]
ROOT.mkdir(parents=True, exist_ok=True)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPUS
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
if LIBNUMA_DIR.exists():
    os.environ["LD_LIBRARY_PATH"] = (
        f"{LIBNUMA_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
_existing_cpath = os.environ.get("CPATH", "")
_include_paths = [str(path) for path in PYTHON_INCLUDE_DIRS if path.exists()]
if _include_paths:
    os.environ["CPATH"] = ":".join([*_include_paths, _existing_cpath])

import gc
import math
import random
import re
import subprocess
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Local NLA inference helpers from the released repo.
import sys

sys.path.insert(0, str(NLA_REPO))
from nla_inference import NLAClient, NLACritic

TARGET_MODEL = "google/gemma-3-12b-it"
ACTOR_REPO_ID = os.environ.get("NLA_ACTOR_DIR", "kitft/nla-gemma3-12b-L32-av")
AR_REPO_ID = os.environ.get("NLA_AR_DIR", "kitft/nla-gemma3-12b-L32-ar")
SGLANG_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30000")
SGLANG_LOG = Path.home() / "logs" / "sglang_av_server.log"
SGLANG_LOG.parent.mkdir(parents=True, exist_ok=True)
LAYER_INDEX = 32
DEVICE = LOCAL_DEVICE
SEED = 1234
N_EXAMPLES = 200


def resolve_checkpoint(path_or_repo_id: str) -> str:
    """Return a local checkpoint path with nla_meta.yaml available."""
    candidate = Path(path_or_repo_id).expanduser()
    if candidate.exists():
        assert (
            candidate / "nla_meta.yaml"
        ).exists(), f"Missing nla_meta.yaml in {candidate}"
        return str(candidate)
    local_path = Path(snapshot_download(repo_id=path_or_repo_id))
    assert (
        local_path / "nla_meta.yaml"
    ).exists(), f"Missing nla_meta.yaml in downloaded snapshot {local_path}"
    return str(local_path)


ACTOR_DIR = resolve_checkpoint(ACTOR_REPO_ID)
AR_DIR = resolve_checkpoint(AR_REPO_ID)

assert NLA_REPO.exists(), f"Missing NLA repo: {NLA_REPO}"
assert SGLANG_REPO.exists(), f"Missing patched SGLang checkout: {SGLANG_REPO}"
assert VENV_PYTHON.exists(), f"Missing NLA venv Python: {VENV_PYTHON}"

print("home:", HOME)
print("NLA repo:", NLA_REPO)
print("SGLang checkout:", SGLANG_REPO)
print("HF cache:", HF_CACHE)
print("experiment output:", ROOT)
print("vendored libnuma:", LIBNUMA_DIR if LIBNUMA_DIR.exists() else "not found")
print("vendored Python includes:", [p for p in PYTHON_INCLUDE_DIRS if p.exists()])
print("visible physical GPUs:", VISIBLE_GPUS)
print("SGLang physical GPU:", SGLANG_PHYSICAL_GPU)
print("notebook device:", DEVICE)
print("actor checkpoint:", ACTOR_DIR)
print("AR checkpoint:", AR_DIR)
print(torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available():
    print("visible cuda devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i}", torch.cuda.get_device_name(i))

UV_PYTHON_INCLUDE = "/home/kaylee/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/include/python3.10"
assert os.path.exists(os.path.join(UV_PYTHON_INCLUDE, "Python.h")), "Python.h not found at that path"

os.environ["CPATH"] = f"{UV_PYTHON_INCLUDE}:{os.environ.get('CPATH', '')}"
print("CPATH now:", os.environ["CPATH"])

REAL_LIBNUMA_DIR = "/home/kaylee/.local/lib"
assert os.path.exists(os.path.join(REAL_LIBNUMA_DIR, "libnuma.so.1")), "libnuma.so.1 not found there"

os.environ["LD_LIBRARY_PATH"] = f"{REAL_LIBNUMA_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"
print("LD_LIBRARY_PATH now:", os.environ["LD_LIBRARY_PATH"])

# !rm -rf ~/.triton/cache

print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("CPATH:", os.environ.get("CPATH"))

# %% [markdown]
# ## Start The AV Server
# 
# The AV uses SGLang because the checkpoint expects `input_embeds` injection. This notebook launches the patched editable SGLang checkout through the NLA `.venv`, pinned to the physical GPU chosen in the setup cell.
# 
# 

# %%
def launch_sglang_actor() -> subprocess.Popen:
    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = SGLANG_PHYSICAL_GPU
    env["SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR"] = "1"
    env["PYTHONPATH"] = f"{NLA_REPO}:{env.get('PYTHONPATH', '')}"
    if LIBNUMA_DIR.exists():
        env["LD_LIBRARY_PATH"] = f"{LIBNUMA_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    include_paths = [str(path) for path in PYTHON_INCLUDE_DIRS if path.exists()]
    if include_paths:
        env["CPATH"] = ":".join([*include_paths, env.get("CPATH", "")])

    cmd = [
        str(VENV_PYTHON),
        "-m",
        "sglang.launch_server",
        "--model-path",
        ACTOR_DIR,
        "--port",
        "30000",
        "--host",
        "127.0.0.1",
        "--tp-size",
        "2",
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.80",
        "--context-length",
        "512",
        "--attention-backend",
        "triton",
        "--disable-cuda-graph",
        "--trust-remote-code",
        "--log-level",
        "warning",
    ]
    print("Launching SGLang on physical GPU", SGLANG_PHYSICAL_GPU)
    print(" ".join(cmd))
    print("SGLang server logs ->", SGLANG_LOG)
    log_f = open(SGLANG_LOG, "ab", buffering=0)
    return subprocess.Popen(
        cmd, env=env, cwd=str(NLA_REPO), stdout=log_f, stderr=subprocess.STDOUT
    )



def sglang_is_healthy() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen(SGLANG_URL + "/health", timeout=2).read()
        return True
    except Exception:
        return False


def wait_for_sglang(timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if sglang_is_healthy():
            print("SGLang is healthy")
            return
        try:
            import urllib.request

            urllib.request.urlopen(SGLANG_URL + "/health", timeout=2).read()
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"SGLang did not become healthy: {last_error!r}")


sglang_proc = None
if sglang_is_healthy():
    print("SGLang is already healthy at", SGLANG_URL)
else:
    sglang_proc = launch_sglang_actor()
    wait_for_sglang()

# %%
print(sglang_is_healthy())

# %% [markdown]
# # Experiment 1 - GSM8k (Deprecated)

# %%
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="balanced",
    max_memory={0: "40GiB", 3: "40GiB"},
    trust_remote_code=True, attn_implementation="eager",
).eval()
del model.model.vision_tower
gc.collect()
torch.cuda.empty_cache()
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
print("model" in globals())
if "model" in globals():
    print(model.device if hasattr(model, "device") else next(model.parameters()).device)

# %%
client = NLAClient(ACTOR_DIR, sglang_url=SGLANG_URL)
critic = NLACritic(AR_DIR, device="cpu", dtype=torch.float32)
print("Target and AV client ready")


# %% [markdown]
# The reconstruction error is
# $$
# \mathcal{L} = \mathbb{E}_{h_l \sim \mathcal{H}} \mathbb{E}_{z \sim AV(\cdot | h_l)} [||h_l - AR(z)||_2^2]
# $$
# where $\mathcal{H}$ is the distribution produced by extracting layer $l$ activations from $M$ on a corpus of text. 
# 
# The FVE is
# $$
# 1 - \frac{L}{ \mathbb{E}_{h_l \sim \mathcal{H}} ||h_l - \bar{h_l}||^2_2}
# $$

# %%
def av(h: torch.Tensor) -> str:
    v = h.detach().float().cpu().numpy()
    return client.generate(v, temperature=0.0, max_new_tokens=384, extract_explanation=True)

PARAPHRASE_PROMPT = (
    "Please paraphrase the text provided below, preserving its exact meaning "
    "but expressing it in substantially different words and sentence structure "
    "than the original. Do not simply swap individual words for synonyms -- "
    "restructure the sentence(s) genuinely.\n\n"
    "<text>\n{text}\n</text>\n\n"
    "Include your final paraphrased text in <transformed_text> tags."
)

def ar(text: str) -> torch.Tensor:
    return critic.reconstruct(text)  # raw, unnormalized -- matches h's scale


# --- Patch hook: replace the residual stream at the last prompt token ---
class LastTokenPatcher:
    def __init__(self, patch_vec=None):
        self.patch_vec = patch_vec
        self.captured = None
        self._done = False  # only patch once

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Only act on the initial multi-token prefill pass, not each
        # single-token decode step that follows.
        if hidden.shape[1] > 1 and not self._done:
            self.captured = hidden[0, -1, :].detach().clone()
            if self.patch_vec is not None:
                hidden = hidden.clone()
                hidden[0, -1, :] = self.patch_vec.to(hidden.dtype).to(hidden.device)
            self._done = True
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        return output


def mse_cos_from_reconstruction(h_hat: torch.Tensor, h_orig: torch.Tensor, mse_scale: float):
    """Per-example, norm-matched metrics (for reporting alongside FVE, not used *in* FVE)."""
    pred = h_hat.float()
    gold = h_orig.float().cpu()
    pred_n = pred / pred.norm().clamp_min(1e-12) * mse_scale
    gold_n = gold / gold.norm().clamp_min(1e-12) * mse_scale
    mse = ((pred_n - gold_n) ** 2).mean().item()
    cos = (pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item()
    return mse, cos


def raw_sq_error(h_hat: torch.Tensor, h_orig: torch.Tensor) -> float:
    """||h_orig - h_hat||^2 on raw, unnormalized activations -- this is what
    the paper's L and FVE are defined over, so it must NOT be norm-rescaled
    the way mse_cos_from_reconstruction's inputs are."""
    assert h_hat.dim() == 1 and h_orig.dim() == 1, f"{h_hat.shape=} {h_orig.shape=}"
    return ((h_orig.float().cpu() - h_hat.float().cpu()) ** 2).sum().item()


def fve(sq_errors: list, h_orig_list: list) -> float:
    """FVE = 1 - E[||h - AR(z)||^2] / E[||h - h_bar||^2], both expectations
    taken over the same sample. h_bar is estimated as the sample mean of
    h_orig_list -- this must be computed over the full run, not per-example."""
    h_stack = torch.stack([h.float().cpu() for h in h_orig_list])  # [N, d]
    h_bar = h_stack.mean(dim=0)
    variance = ((h_stack - h_bar) ** 2).sum(dim=1).mean().item()
    L = sum(sq_errors) / len(sq_errors)
    return 1 - L / variance


def get_layer_module(m, layer_idx):
    return m.model.language_model.layers[layer_idx]

# https://github.com/kitft/natural_language_autoencoders/blob/main/examples/gemma27b_layer41_step6000.txt
VAR_NRM = 0.0579  # Gemma-3-27B / layer-41 predict-the-mean MSE from training set

def fve_nrm(mse_nrm: float) -> float:
    """Fraction of training-set activation variance explained (normalized space)."""
    return 1.0 - float(mse_nrm) / VAR_NRM


def run_with_patch(prompt: str, patch_vec, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    patcher = LastTokenPatcher(patch_vec)
    handle = get_layer_module(model, LAYER_INDEX).register_forward_hook(patcher)
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        handle.remove()
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, patcher.captured



# %% [markdown]
# Standard reconstruction metrics disagree sharply depending on whether they're scale-normalized: cosine similarity (0.992) indicates the AR recovers the activation's direction well, while raw-scale FVE is strongly negative because 81% of total squared error is concentrated in just 10 of 5,376 dimensions — a small set of "massive activation" dimensions with near-constant, very large magnitude across examples. This suggests the AR is not well-calibrated on these specific outlier dimensions, even though it captures the semantically relevant structure of the activation.

# %% [markdown]
# # Experiment 2 - Tasks from Paper

# %% [markdown]
# ## New Tasks
# 
# We construct a diverse array of over 40 relatively simple tasks to test whether function
# vectors can be extracted in diverse settings. To simplify the presentation of our analysis, we focus on
# a representative sample of 6 tasks:
# - Antonym. Given an input word, generate the word with opposite meaning.
#     - Nguyen et al (2017)
# - Capitalize. Given an input word, generate the same word with a capital first letter.
#     - ChatGPT
# - Country-Capital. Given a country name, generate the capital city.
#     - ChatGPT
# - English-French. Given an English word, generate the French translation of the word.
#     - Conneau et al (2017)
# - Present-Past. Given a verb in present tense, generate the verb’s simple past inflection.
#     - ChatGPT
# - Singular-Plural. Given a singular noun, generate its plural inflection
#     - ChatGPT
# 
# 
# For each task have a plot for number of shots (X axis) and accuracy (y-axis). 

# %%
import math
from nla.schema import normalize_activation

D_MODEL = model.config.text_config.hidden_size
MSE_SCALE = math.sqrt(D_MODEL)

# %%
"""
Shot-count sweep version of the NLA round-trip eval on the 6 ICL
function-vector tasks (Todd et al. 2023, arXiv:2310.15213).

For each task, sweeps the number of ICL demonstrations (x-axis) and
plots accuracy (y-axis) for four conditions:
  - baseline:                    no patching at all -- Gemma out of the
                                 box, plain few-shot ICL, no function
                                 vector involved
  - nla_roundtrip:               h patched with AR(verbalize(h))
  - paraphrase_roundtrip_medium: h patched with AR(paraphrase_medium(verbalize(h)))
  - paraphrase_roundtrip_heavy:  h patched with AR(paraphrase_heavy(verbalize(h)))

Produces one PNG per task plus a combined 2x3 grid, and caches raw
results to JSON so you can replot without rerunning the model.

Notebook cells -- paste/import your existing definitions (av, ar,
run_with_patch, LastTokenPatcher, client, critic, model, tokenizer,
DEVICE) before running this. `paraphrase()` is now defined below
(parameterized on prompt_template) rather than imported -- just make
sure `model`/`tokenizer`/`DEVICE` are already in scope when it runs.

Dataset sources, all expected flat under datasets/:
  - antonym:           Nguyen et al. 2017 (AntSynNET) -- github.com/nguyenkh/AntSynNET
  - capitalize:        ChatGPT-generated
  - country-capital:   ChatGPT-generated
  - english-french:    Conneau et al. 2017 (MUSE bilingual dictionaries) --
                        both en-fr.0-5000.txt (train) and en-fr.5000-6500.txt
                        (test) splits, concatenated into one pool
  - present-past:      ChatGPT-generated
  - singular-plural:   ChatGPT-generated
"""

# %%
import json
import random
import re
import string
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

TASKS = [
    "antonym",
    "capitalize",
    "country-capital",
    "english-french",
    "present-past",
    "singular-plural",
]

# Shot counts to sweep. (Zero-shot removed -- sweep starts at 2.)
# SHOT_COUNTS = [2, 4, 6, 8, 10]
SHOT_COUNTS = [1, 2, 4]
N_PER_CONFIG = 50
MAX_NEW_TOKENS = 256
SEED = 0

DATA_DIR = Path("datasets")
OUT_DIR = Path("/mnt/ssd-1/soar-nla/kaylee")
RESULTS_CACHE = OUT_DIR / "fv_shot_sweep_results.json"
CASE_SENSITIVE_TASKS = {"capitalize"}

# Two paraphrase intensities. PARAPHRASE_PROMPT_MEDIUM should be set to
# whatever your original PARAPHRASE_PROMPT already was (the one
# paraphrase() used to read as a hardcoded global). Fill in
# PARAPHRASE_PROMPT_HEAVY with your stronger-rewrite prompt. Both must
# contain a `{text}` format placeholder, same as the original.
PARAPHRASE_PROMPT_MEDIUM = (
    "Please paraphrase the text provided below, preserving its exact meaning "
    "but expressing it in substantially different words and sentence structure "
    "than the original. Do not simply swap individual words for synonyms -- "
    "restructure the sentence(s) genuinely.\n\n"
    "<text>\n{text}\n</text>\n\n"
    "Include your final paraphrased text in <transformed_text> tags."
)
PARAPHRASE_PROMPT_HEAVY = (
    "Please paraphrase aggressively. Preserve meaning and uncertainty, but "
    "substantially change wording, syntax, and organization. Do not simply swap individual words for synonyms -- "
    "restructure the sentence(s) genuinely.\n\n"
    "Text:\n{text}\n\n"
    "Respond with only the paraphrased text, wrapped exactly like this:\n"
    "<transformed_text>your paraphrase here</transformed_text>"
)


# %%
# --------------------------------------------------------------------------
# Paraphrasing (Gemma-based rewrite of the AV's verbalization)
# --------------------------------------------------------------------------

def paraphrase(text: str, prompt_template: str = PARAPHRASE_PROMPT_MEDIUM, max_new_tokens: int = 512) -> str:
    """Rewrites `text` using `prompt_template` (PARAPHRASE_PROMPT_MEDIUM or
    PARAPHRASE_PROMPT_HEAVY). Was hardcoded to a single global
    PARAPHRASE_PROMPT -- now takes the template explicitly so medium and
    heavy paraphrase conditions can share one function.
    """
    prompt = prompt_template.format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    match = re.search(r"<transformed_text>(.*?)</transformed_text>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        print(f"[paraphrase] WARNING: no <transformed_text> tags. Raw[:200]={raw[:200]!r}")
        return raw


# %%
# --------------------------------------------------------------------------
# Per-task dataset loaders -- three different source formats, so three loaders
# --------------------------------------------------------------------------

def load_antonym(data_dir: Path, pos_tags=("verb", "noun", "adjective"),
                  splits=("train", "val", "test"),
                  tokenizer=None) -> list[dict]:
    """AntSynNET (Nguyen et al. 2017), antonym pairs only.

    Files are named like 'verb-pairs.train', 'noun-pairs.val',
    'adjective-pairs.test' -- tab-separated: word1, word2, label,
    where label is 1 for antonym pairs and 0 for synonym pairs.

    Combines all POS categories across all splits into one pool,
    keeps only label==1 rows, de-dupes exact (input, output) pairs,
    and -- if a tokenizer is passed -- further filters to pairs where
    BOTH words tokenize as a single token, matching Todd et al. 2023
    (2,398 antonym pairs kept after that step).
    """
    seen = set()
    pairs = []
    for pos in pos_tags:
        for split in splits:
            path = data_dir / f"{pos}-pairs.{split}"
            if not path.exists():
                print(f"[warn] missing file: {path}")
                continue
            with open(path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    w1, w2, label = parts[0], parts[1], parts[2]
                    if label != "1":          # keep antonyms only, drop synonyms (0)
                        continue
                    key = (w1, w2)
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append({"input": w1, "output": w2})

    if tokenizer is not None:
        def is_single_token(word: str) -> bool:
            return len(tokenizer.encode(" " + word, add_special_tokens=False)) == 1
        pairs = [p for p in pairs
                 if is_single_token(p["input"]) and is_single_token(p["output"])]

    return pairs


def load_english_french(paths) -> list[dict]:
    """Conneau et al. 2017 (MUSE) bilingual dictionary format: one
    'english_word french_word' pair per line, whitespace-separated.
    Accepts a single path or a list of paths -- MUSE ships en-fr as
    separate train (0-5000) / test (5000-6500) splits, but nothing in
    this pipeline trains/evaluates a mapping, so both are concatenated
    into one pool rather than kept separate.

    MUSE dictionaries list multiple valid translations for some English
    words (e.g. 'chat' -> 'discuter'/'discussion'/'chat'/...). Only the
    FIRST translation seen for a given English word is kept -- later
    duplicates are dropped -- so every English word maps to exactly one
    canonical French output. This avoids two problems: (a) the same
    input appearing as both a demonstration and the query with
    conflicting gold answers, and (b) is_correct() marking a model's
    valid-but-different translation as wrong non-deterministically.
    It does NOT fix the underlying scoring looseness -- a model that
    produces a *dropped* valid translation still scores as incorrect.
    If that turns out to matter, switch to scoring against the full
    set of valid translations per word instead of exact-match-one-string.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    seen = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                en, fr = parts[0], parts[1]
                seen.setdefault(en, fr)
    return [{"input": en, "output": fr} for en, fr in seen.items()]


def load_json_pairs(path: Path) -> list[dict]:
    """ChatGPT-generated tasks (capitalize, country-capital, present-past,
    singular-plural). Expects a JSON list of {"input": ..., "output": ...}.
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "examples" in raw:
        raw = raw["examples"]
    return [{"input": ex["input"], "output": ex["output"]} for ex in raw]


# task -> (loader function, path(s) relative to DATA_DIR)
# adjust filenames once you know what your antonym download is actually named
TASK_LOADERS = {
    "antonym": (load_antonym, None),   # loader takes the data_dir directly
    "capitalize": (load_json_pairs, "capitalize.json"),
    "country-capital": (load_json_pairs, "country-capital.json"),
    "english-french": (load_english_french, ["en-fr.0-5000.txt", "en-fr.5000-6500.txt"]),
    "present-past": (load_json_pairs, "present-past.json"),
    "singular-plural": (load_json_pairs, "singular-plural.json"),
}

def load_fv_task(task_name: str, data_dir: Path = DATA_DIR, tokenizer=None) -> list[dict]:
    loader, rel_path = TASK_LOADERS[task_name]
    if task_name == "antonym":
        return loader(data_dir, tokenizer=tokenizer)
    if isinstance(rel_path, list):
        full_paths = [data_dir / p for p in rel_path]
    else:
        full_paths = data_dir / rel_path
    return loader(full_paths)


# %%
# --------------------------------------------------------------------------
# Prompting / scoring
# --------------------------------------------------------------------------

def build_icl_prompt(pairs: list[dict], query_idx: int, n_shots: int, rng: random.Random,
                      extra_exclude_words: set[str] | None = None) -> tuple[str, str]:
    """Excludes candidate demonstrations by matching INPUT WORD, not just
    index -- some datasets (e.g. english-french before dedup, or any task
    with accidental duplicate inputs) can otherwise let the same word
    appear as both a demonstration and the query, or as two demonstrations
    with different gold outputs, which confuses the in-context signal.

    extra_exclude_words: additional input words to keep out of the demo
    pool, beyond the query's own word. Used when building a prompt for a
    DIFFERENT query but you still want it to avoid leaking some other
    example (e.g. a same-task mismatch prompt should never reveal the
    original query's word as a demonstration).
    """
    query_word = pairs[query_idx]["input"]
    excluded_words = {query_word} | (extra_exclude_words or set())
    pool = [i for i in range(len(pairs)) if i != query_idx and pairs[i]["input"] not in excluded_words]
    shot_idxs = rng.sample(pool, min(n_shots, len(pool)))
    lines = [f"Q: {pairs[i]['input']}\nA: {pairs[i]['output']}" for i in shot_idxs]
    lines.append(f"Q: {query_word}\nA:")
    instruction = "Answer with a single word only. No punctuation, no explanation."
    prompt = instruction + "\n\n" + "\n\n".join(lines)
    gold = pairs[query_idx]["output"]
    return prompt, gold


def extract_word_answer(text: str) -> str:
    first_line = text.strip().split("\n")[0]
    return first_line.strip(string.punctuation + " ")


def is_correct(pred: str, gold: str, case_sensitive: bool) -> bool:
    if not case_sensitive:
        pred, gold = pred.lower(), gold.lower()
    return pred == gold


def is_meaningfully_paraphrased(original: str, paraphrased: str, max_overlap: float = 0.7) -> bool:
    orig_words = set(original.lower().split())
    para_words = set(paraphrased.lower().split())
    if not orig_words:
        return True
    overlap = len(orig_words & para_words) / len(orig_words)
    return overlap < max_overlap


# %%
# --------------------------------------------------------------------------
# One (task, n_shots) configuration
# --------------------------------------------------------------------------

def run_config(task_name: str, n_shots: int, pairs: list[dict], rng: random.Random,
                all_pairs: dict[str, list[dict]], task_examples: dict, verbose=False) -> dict:
    n = min(N_PER_CONFIG, len(pairs))
    query_idxs = rng.sample(range(len(pairs)), n)
    case_sensitive = task_name in CASE_SENSITIVE_TASKS

    baseline_correct = 0
    nla_correct = 0
    random_vec_correct = 0
    mismatch_cross_task_correct = 0
    mismatch_same_task_correct = 0
    para_medium_correct = 0
    para_heavy_correct = 0
    para_medium_overlaps = []
    para_heavy_overlaps = []
    para_medium_low_overlap_count = 0
    para_heavy_low_overlap_count = 0

    # NEW: task_examples is shared across all run_config() calls for this
    # task (i.e. across the n_shots sweep) -- caller initializes one entry
    # per task_name before the sweep starts, and we fill in the first
    # correct/incorrect NLA-condition example we find, whichever n_shots
    # config that happens to be. Once both slots are filled, we stop
    # touching this task's entry.
    task_examples.setdefault(task_name, {"correct": None, "incorrect": None})

    # Tasks to draw a mismatched (real, wrong-task) activation vector from.
    other_tasks = [t for t in all_pairs if t != task_name and len(all_pairs[t]) > 0]

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"CONFIG: task={task_name!r}  n_shots={n_shots}  n={n}")
        print(f"{'=' * 70}")

    for i, qi in enumerate(query_idxs):
        prompt, gold = build_icl_prompt(pairs, qi, n_shots, rng)

        text_baseline, h_orig = run_with_patch(prompt, patch_vec=None, max_new_tokens=MAX_NEW_TOKENS)
        pred_baseline = extract_word_answer(text_baseline)
        correct_baseline = is_correct(pred_baseline, gold, case_sensitive)
        if verbose:
            print(f"\n--- sample {i + 1}/{n}  |  gold={gold!r} ---")
            print(f"  [baseline]  pred={pred_baseline!r:<15} correct={correct_baseline}")
        baseline_correct += correct_baseline

        explanation = av(h_orig)
        if verbose:
            print(f"  explanation : {explanation[:300]}")

        # NLA round-trip
        h_hat = ar(explanation)
        text_nla, _ = run_with_patch(prompt, patch_vec=h_hat, max_new_tokens=MAX_NEW_TOKENS)
        pred = extract_word_answer(text_nla)
        correct = is_correct(pred, gold, case_sensitive)
        if verbose:
            print(f"  [nla]       pred={pred!r:<15} correct={correct}")
        nla_correct += correct

        # NEW: fill this task's correct/incorrect example slot the first
        # time we see one, across the whole n_shots sweep for this task.
        slot = "correct" if correct else "incorrect"
        if task_examples[task_name][slot] is None:
            task_examples[task_name][slot] = {
                "task": task_name,
                "n_shots": n_shots,
                "query_input": pairs[qi]["input"],
                "gold": gold,
                "explanation": explanation,
                "pred": pred,
            }

        # Random activation vector -- null baseline, same norm as h_orig
        # so scale isn't confounded with "content of the vector was garbage".
        seed_i = rng.getrandbits(32)
        gen = torch.Generator(device=h_orig.device).manual_seed(seed_i)
        noise = torch.randn(h_orig.shape, generator=gen, device=h_orig.device, dtype=h_orig.dtype)
        h_random = noise * (h_orig.norm() / (noise.norm() + 1e-8))
        text_random, _ = run_with_patch(prompt, patch_vec=h_random, max_new_tokens=MAX_NEW_TOKENS)
        pred_random = extract_word_answer(text_random)
        correct_random = is_correct(pred_random, gold, case_sensitive)
        if verbose:
            print(f"  [random]    pred={pred_random!r:<15} correct={correct_random}  "
                  f"(||h_orig||={h_orig.norm():.3f}, ||h_random||={h_random.norm():.3f})")
        random_vec_correct += correct_random

        # NEW: mismatched activation vector -- a real (not NLA-reconstructed,
        # not random) activation, captured from a DIFFERENT task's ICL prompt,
        # patched directly into the current prompt. Tests content-specificity
        # rather than off-manifold sensitivity (that's what acc_random_vec is for).
        foreign_task = rng.choice(other_tasks)
        foreign_pairs = all_pairs[foreign_task]
        foreign_qi = rng.randrange(len(foreign_pairs))
        foreign_prompt, _ = build_icl_prompt(foreign_pairs, foreign_qi, n_shots, rng)
        _, h_mismatch_cross_task = run_with_patch(foreign_prompt, patch_vec=None, max_new_tokens=MAX_NEW_TOKENS)
        # discard the generated text above -- only the captured activation matters

        text_mismatch_cross_task, _ = run_with_patch(prompt, patch_vec=h_mismatch_cross_task, max_new_tokens=MAX_NEW_TOKENS)
        pred_mismatch_cross_task = extract_word_answer(text_mismatch_cross_task)
        correct_mismatch_cross_task = is_correct(pred_mismatch_cross_task, gold, case_sensitive)
        if verbose:
            print(f"  [mismatch-cross-task]  pred={pred_mismatch_cross_task!r:<15} correct={correct_mismatch_cross_task}  "
                  f"(from task={foreign_task!r})")
        mismatch_cross_task_correct += correct_mismatch_cross_task

        # NEW: mismatched activation vector -- same task, different example.
        # A real (not NLA-reconstructed, not random) activation captured from
        # a DIFFERENT example within the SAME task, patched into the current
        # prompt. Distinguishes "wrong content, same task format" (this) from
        # "wrong content, different task entirely" (acc_mismatch_cross_task).
        qi_word = pairs[qi]["input"]
        same_task_pool = [i for i in range(len(pairs)) if i != qi and pairs[i]["input"] != qi_word]
        same_task_qi = rng.choice(same_task_pool)
        same_task_prompt, _ = build_icl_prompt(pairs, same_task_qi, n_shots, rng,
                                                extra_exclude_words={qi_word})
        _, h_mismatch_same_task = run_with_patch(same_task_prompt, patch_vec=None, max_new_tokens=MAX_NEW_TOKENS)
        # discard the generated text above -- only the captured activation matters

        text_mismatch_same_task, _ = run_with_patch(prompt, patch_vec=h_mismatch_same_task, max_new_tokens=MAX_NEW_TOKENS)
        pred_mismatch_same_task = extract_word_answer(text_mismatch_same_task)
        correct_mismatch_same_task = is_correct(pred_mismatch_same_task, gold, case_sensitive)
        if verbose:
            print(f"  [mismatch-same-task]  pred={pred_mismatch_same_task!r:<15} correct={correct_mismatch_same_task}  "
                  f"(from example idx={same_task_qi} within task={task_name!r})")
        mismatch_same_task_correct += correct_mismatch_same_task

        # Paraphrase round-trip (medium)
        paraphrased_medium = paraphrase(explanation, prompt_template=PARAPHRASE_PROMPT_MEDIUM)
        orig_words = set(explanation.lower().split())
        para_medium_words = set(paraphrased_medium.lower().split())
        overlap_medium = len(orig_words & para_medium_words) / len(orig_words) if orig_words else 0.0
        para_medium_overlaps.append(overlap_medium)
        if not is_meaningfully_paraphrased(explanation, paraphrased_medium):
            para_medium_low_overlap_count += 1
            if verbose:
                print(f"  [para-medium] WARNING: low reword rate (overlap={overlap_medium:.2f})")

        h_hat_para_medium = ar(paraphrased_medium)
        text_para_medium, _ = run_with_patch(prompt, patch_vec=h_hat_para_medium, max_new_tokens=MAX_NEW_TOKENS)
        pred_para_medium = extract_word_answer(text_para_medium)
        correct_para_medium = is_correct(pred_para_medium, gold, case_sensitive)
        if verbose:
            print(f"  [para-medium] text : {paraphrased_medium[:300]}")
            print(f"  [para-medium]      pred={pred_para_medium!r:<15} correct={correct_para_medium}")
        para_medium_correct += correct_para_medium

        # Paraphrase round-trip (heavy)
        paraphrased_heavy = paraphrase(explanation, prompt_template=PARAPHRASE_PROMPT_HEAVY)
        para_heavy_words = set(paraphrased_heavy.lower().split())
        overlap_heavy = len(orig_words & para_heavy_words) / len(orig_words) if orig_words else 0.0
        para_heavy_overlaps.append(overlap_heavy)
        if not is_meaningfully_paraphrased(explanation, paraphrased_heavy):
            para_heavy_low_overlap_count += 1
            if verbose:
                print(f"  [para-heavy] WARNING: low reword rate (overlap={overlap_heavy:.2f})")

        h_hat_para_heavy = ar(paraphrased_heavy)
        text_para_heavy, _ = run_with_patch(prompt, patch_vec=h_hat_para_heavy, max_new_tokens=MAX_NEW_TOKENS)
        pred_para_heavy = extract_word_answer(text_para_heavy)
        correct_para_heavy = is_correct(pred_para_heavy, gold, case_sensitive)
        if verbose:
            print(f"  [para-heavy] text : {paraphrased_heavy[:300]}")
            print(f"  [para-heavy]      pred={pred_para_heavy!r:<15} correct={correct_para_heavy}")
        para_heavy_correct += correct_para_heavy

    if verbose:
        print(f"\n{'-' * 70}")
        print(f"SUMMARY  task={task_name!r}  n_shots={n_shots}")
        print(f"  acc_baseline (no NLA)       = {baseline_correct / n:.3f}")
        print(f"  acc_nla (round-trip)        = {nla_correct / n:.3f}")
        print(f"  acc_random_vec              = {random_vec_correct / n:.3f}")
        print(f"  acc_mismatch (foreign task) = {mismatch_cross_task_correct / n:.3f}")
        print(f"  acc_mismatch_same_task      = {mismatch_same_task_correct / n:.3f}")
        print(f"  acc_para_medium (paraphrase)= {para_medium_correct / n:.3f}")
        print(f"  acc_para_heavy (paraphrase) = {para_heavy_correct / n:.3f}")
        print(f"  avg_para_medium_overlap     = {sum(para_medium_overlaps) / n:.3f}")
        print(f"  avg_para_heavy_overlap      = {sum(para_heavy_overlaps) / n:.3f}")
        print(f"  para_medium_low_overlap_count = {para_medium_low_overlap_count}")
        print(f"  para_heavy_low_overlap_count  = {para_heavy_low_overlap_count}")
        print(f"{'-' * 70}\n")

    def se(k):
        p = k / n
        return (p * (1 - p) / n) ** 0.5

    return {
        "n_shots": n_shots,
        "n": n,
        "acc_baseline": baseline_correct / n,
        "se_baseline": se(baseline_correct),
        "acc_nla": nla_correct / n,
        "se_nla": se(nla_correct),
        "acc_random_vec": random_vec_correct / n,
        "se_random_vec": se(random_vec_correct),
        "acc_mismatch_cross_task": mismatch_cross_task_correct / n,
        "se_mismatch_cross_task": se(mismatch_cross_task_correct),
        "acc_mismatch_same_task": mismatch_same_task_correct / n,
        "se_mismatch_same_task": se(mismatch_same_task_correct),
        "acc_para_medium": para_medium_correct / n,
        "se_para_medium": se(para_medium_correct),
        "acc_para_heavy": para_heavy_correct / n,
        "se_para_heavy": se(para_heavy_correct),
        "avg_para_medium_overlap": sum(para_medium_overlaps) / n,
        "avg_para_heavy_overlap": sum(para_heavy_overlaps) / n,
        "para_medium_low_overlap_count": para_medium_low_overlap_count,
        "para_heavy_low_overlap_count": para_heavy_low_overlap_count,
    }

# %% [markdown]
# - might have to loosen to check if word is there or not instead of expecting exact output
# - The av/ar round-trip through natural language seems to preserve the subject matter far more reliably than it preserves what specifically to do with that subject — the fine-grained target word, or the task's input→output mapping. That's a meaningful and fairly interpretable finding for whatever you're writing up, distinct from just "accuracy is low."

# %%
# %%
# --------------------------------------------------------------------------
# Sweep + plotting
# --------------------------------------------------------------------------
#
# One DataFrame per task, saved to its own CSV the moment that task
# finishes -- not batched up into one big dict/JSON at the end. If task 4
# of 6 throws (OOM, a bad dataset file, whatever), tasks 1-3 are already
# on disk as fv_shot_sweep_<task>.csv and load_sweep_results() below can
# pull them back in for plotting without rerunning anything. The sweep
# also catches per-task exceptions and moves on to the next task rather
# than aborting the whole run.

from datetime import datetime

# %%
# --------------------------------------------------------------------------
# Sweep + plotting
# --------------------------------------------------------------------------

import pandas as pd

# Condition -> (column name, marker, label) used by both plotting loops
# below. Single source of truth so per-task and combined-grid plots can't
# drift out of sync with each other.
CONDITIONS = [
    ("acc_baseline", "o", "Baseline (no NLA)"),
    ("acc_nla", "^", "NLA round-trip"),
    ("acc_random_vec", "x", "Random activation vector"),
    ("acc_para_medium", "s", "Paraphrase round-trip (medium)"),
    ("acc_para_heavy", "d", "Paraphrase round-trip (heavy)"),
    ("acc_mismatch_cross_task", "p", "Mismatched vector (foreign task)"),
    ("acc_mismatch_same_task", "*", "Mismatched vector (same task)"),
]


def run_sweep(run_id: str | None = None) -> dict[str, pd.DataFrame]:
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    rng = random.Random(SEED)
    results: dict[str, pd.DataFrame] = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # NEW: load every task's pairs upfront so run_config can sample a
    # foreign-task query for the mismatched-vector condition.
    all_pairs: dict[str, list[dict]] = {}
    for task_name in TASKS:
        try:
            all_pairs[task_name] = load_fv_task(task_name)
        except Exception as e:
            print(f"[warn] could not preload {task_name!r} for cross-task sampling: {e!r}")

    # NEW: accumulates one correct + one incorrect verbalization example
    # per task, across the whole n_shots sweep. Populated in-place by
    # run_config().
    task_examples: dict[str, dict] = {}

    for task_name in TASKS:
        try:
            pairs = all_pairs[task_name]
            rows = [
                run_config(task_name, k, pairs, rng, all_pairs, task_examples, verbose=True)
                for k in tqdm(SHOT_COUNTS, desc=task_name, unit="shot-count")
            ]
        except Exception as e:
            print(f"[error] task {task_name!r} failed, skipping it and continuing: {e!r}")
            continue

        df = pd.DataFrame(rows)
        results[task_name] = df

        csv_path = OUT_DIR / f"fv_shot_sweep_{task_name}_{run_id}.csv"
        try:
            df.to_csv(csv_path, index=False)
            print(f"[ok] {task_name} -> {csv_path}")
        except OSError as e:
            print(f"[warn] ran {task_name} but could not save its CSV: {e}")

    # NEW: dump per-task correct/incorrect verbalization examples alongside
    # the main results CSVs, for qualitative inspection (e.g. for the PI).
    examples_path = OUT_DIR / f"fv_shot_sweep_examples_{run_id}.json"
    try:
        with open(examples_path, "w") as f:
            json.dump(task_examples, f, indent=2)
        print(f"[ok] examples -> {examples_path}")
    except OSError as e:
        print(f"[warn] ran sweep but could not save examples JSON: {e}")

    return results, run_id


def load_sweep_results(run_id: str, tasks: list[str] = TASKS) -> dict[str, pd.DataFrame]:
    """Reload whatever per-task CSVs already exist on disk for a given
    run_id -- e.g. after a kernel crash mid-sweep, or if you just want to
    replot without rerunning the model. Silently skips tasks that never
    finished/saved.
    """
    results: dict[str, pd.DataFrame] = {}
    for task_name in tasks:
        csv_path = OUT_DIR / f"fv_shot_sweep_{task_name}_{run_id}.csv"
        if csv_path.exists():
            results[task_name] = pd.read_csv(csv_path)
        else:
            print(f"[warn] no cached CSV for {task_name} at {csv_path}")
    return results


def plot_sweep(results: dict[str, pd.DataFrame], run_id: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_name, df in results.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        for col, marker, label in CONDITIONS:
            if col not in df.columns:
                print(f"[warn] {task_name}: missing column {col!r}, skipping that line")
                continue
            se_col = col.replace("acc_", "se_", 1)
            yerr = df[se_col] if se_col in df.columns else None
            ax.errorbar(df["n_shots"], df[col], yerr=yerr, marker=marker, label=label,
                        capsize=3, elinewidth=1, markeredgewidth=1)
        ax.set_xlabel("Number of shots")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.2, 1.2)
        ax.set_title(task_name)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"fv_shot_sweep_{task_name}_{run_id}.png", dpi=150)
        plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, task_name in zip(axes.flat, TASKS):
        if task_name not in results:
            ax.set_title(f"{task_name} (no data)")
            ax.axis("off")
            continue
        df = results[task_name]
        for col, marker, label in CONDITIONS:
            if col not in df.columns:
                continue
            se_col = col.replace("acc_", "se_", 1)
            yerr = df[se_col] if se_col in df.columns else None
            ax.errorbar(df["n_shots"], df[col], yerr=yerr, marker=marker, label=label,
                        capsize=3, elinewidth=1, markeredgewidth=1)
        ax.set_title(task_name)
        ax.set_xlabel("Number of shots")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.2, 1.2)
        ax.grid(alpha=0.3)
    handled_axes = [ax for ax in axes.flat if ax.lines]
    if handled_axes:
        handled_axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fv_shot_sweep_all_tasks_{run_id}.png", dpi=150)
    plt.show()


results, run_id = run_sweep()
plot_sweep(results, run_id)
print(f"This run's ID: {run_id}")

# %% [markdown]
# ## Cleanup
# 
# Close the AV client's HTTP session and stop SGLang only if this notebook launched it. An AV server that was already running before the notebook is left untouched.
# 

# %%
if "client" in globals() and hasattr(client, "_http"):
    client._http.close()
    print("Closed NLA client HTTP session")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if "sglang_proc" in globals() and sglang_proc is not None:
    sglang_proc.terminate()
    sglang_proc.wait(timeout=30)
    print("Stopped SGLang launched by this notebook")
else:
    print("No SGLang subprocess launched by this notebook")

# Optional final memory check:
# !nvidia-smi

# %%



