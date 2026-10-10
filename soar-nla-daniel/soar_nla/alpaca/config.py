"""Frozen protocol constants for the shared Alpaca rollouts.

`PROTOCOL_VERSION` and `SELECTION_SEED` feed `stable_hash`/`stable_seed`: the
sampled prompts, sampled token positions and GCG candidate draws all depend on
these exact strings.
"""

from __future__ import annotations

PROTOCOL_VERSION = "av-faithfulness-v1"

DATASET_ID = "tatsu-lab/alpaca"
DATASET_REVISION = "dce01c9b08f87459cf36a430d809084718273017"
N_EXAMPLES = 2_000
SELECTION_SEED = "av-faithfulness-v1-alpaca-2000"
