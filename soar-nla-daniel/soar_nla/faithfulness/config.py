"""Frozen protocol constants for the warm-start FVE comparison.

`PROTOCOL_VERSION`, `SELECTION_SEED` and `SHUFFLE_SEED` feed `stable_seed`/`seeded_rng`:
the selected rows, the shuffled-arm derangement and the bootstrap draws depend on these
exact strings.
"""

from __future__ import annotations

from soar_nla.hub import CUSTOM_L32, GEMMA3_12B

PROTOCOL_VERSION = "warmstart-fve-v1"

# Sonnet-written explanations over FineFineWeb documents, produced with the NLA warm-start
# recipe. The stored `activation_vector` column is Qwen3-8B layer 24 (d=4096) and is unusable
# here — we keep only the source text and the gold explanation, and re-extract the activation
# with the Gemma target. The TRAIN split is used on purpose: it covers ~37k documents against the
# custom NLA's ~25k av_sft pool, so most of it was never in any custom-NLA stage (see
# contamination.py for the exclusions).
DATASET_ID = "asher577/easynla-warmstart-data"
DATASET_REVISION = "0d7105007fa4e907f118c9e73edf7edbf44308e7"
DATASET_FILE = "av_sft_train.parquet"

SELECTION_SEED = "nla-warmstart-fve-easynla-v1"
SHUFFLE_SEED = "nla-warmstart-fve-v1-shuffle"

# Mirrors stage-0's _MIN_POSITION: the NLA never saw activations with less
# left-context than this, so shorter rows would measure OOD behaviour.
MIN_POSITION = 50

# Left-truncation cap. The extraction position is the LAST token of the text,
# so anything we drop has to come off the front (BOS is re-attached).
MAX_CONTEXT = 2048

# The two shuffled arms are floors: the same explanations dealt to the WRONG
# activations, on a shared derangement so they are matched. gold_shuffled says what
# the AR recovers from generic oracle text; av_shuffled says the same for AV prose,
# which is what rules out "the AR simply finds AV register easier to read".
ARMS = ("av", "gold", "gold_shuffled", "av_shuffled")

# The custom NLA (EasyNLA format) and the data it was trained on. Its training parquets are
# public, which is why this experiment uses it rather than the official release.
NLA_REPO, NLA_REVISION = CUSTOM_L32
NLA_DATA_ID = "achand45/gemma-3-12b-it-nla-data"
NLA_DATA_REVISION = "c85065cf31c931f34794aff3a8187d9fa74b60db"
NLA_DATA_LAYER_DIR = "L32"

TARGET_ID, TARGET_REVISION = GEMMA3_12B
