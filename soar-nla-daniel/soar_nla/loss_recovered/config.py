"""Frozen protocol constants for the Loss Recovered evaluation."""

from __future__ import annotations

MODEL = "gemma12b"  # soar_nla.hub.MODEL_SPECS key: Gemma-3-12B-IT, official L32 AV/AR

# Gemma-3-12B-IT Alpaca rollouts from the "Lie Detection Rollouts" dataset
# (subset google-gemma-3-12b-it, split alpaca). Not the soar_nla.alpaca rollouts.
ALPACA_FILE = "google-gemma-3-12b-it/alpaca.parquet"
ALPACA_SHA256 = "20bafc4a3a3945a1acf0630e14cff06daceab70687bc46f01ecc26e86106b967"
N_CONVERSATIONS = 2_000  # first rows of the parquet
CONTEXT_SIZE = 128  # one prefix window per conversation; shorter conversations are dropped

# Token spans [start, stop) of the window whose layer-32 activations are replaced
# jointly; positions before `start` stay clean. Loss is the next-token CE for
# targets inside the span: stop - start - 1 tokens per window.
SETTINGS = {
    "prefix_50": (0, 50),  # paper: tokens 1-50
    "prefix_100": (0, 100),  # paper: tokens 1-100
    "prefix_128": (0, 128),  # the full window
    "suffix_50_100": (50, 100),  # paper: tokens 51-100 with 1-50 clean
}

# Replacements of the layer-32 activation, scored against clean and zero-ablated runs.
#   nla       AR(AV(h)) rescaled to ||h|| (the paper's NLA)
#   nla_raw   AR(AV(h)) as output by the AR (diagnostic)
#   sae       SAE(h)
#   sae_norm  SAE(h) rescaled to ||h|| (the paper's SAE baseline)
METHODS = ("nla", "nla_raw", "sae", "sae_norm")

SAE_PARAMS_DIR = "resid_post_all/layer_32_width_16k_l0_small"
SAE_ID = "layer_32_width_16k_l0_small"

AV_MAX_NEW_TOKENS = 200
AR_BATCH_SIZE = 16
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
