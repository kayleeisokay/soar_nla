"""Loading and filtering the shared Alpaca rollouts.

Two filters on the same 2,000 rollouts:

- `long_responses`: `assistant_token_count >= 100`, 1,543 rollouts, reported for
  reference only. Loss Recovered uses other rollouts (`soar_nla.loss_recovered`);
  its 1,543 windows are a different set.
- `late_positions`: `sequence_token_index > 50`, i.e. the sampled position lies
  after token 50 of the full chat-rendered sequence, 1,573 rollouts. GCG, its
  random-substitution control and the surgical-edit experiment use this set
  (the NLA was trained on positions 50 and later).

The two sets overlap on 1,348 rollouts.

`is_word_token` splits sampled tokens into those containing a letter or digit
and whitespace/punctuation/markdown tokens (`.`, `**`, `\\n`, runs of spaces).
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

from soar_nla.paths import results_dir

MIN_ASSISTANT_TOKENS = 100
MIN_SEQUENCE_TOKEN_INDEX = 50
_WORD_CHAR = re.compile(r"[A-Za-z0-9]")


def is_word_token(token_text: str) -> bool:
    """True if the decoded token contains an ASCII letter or digit."""
    return bool(_WORD_CHAR.search(token_text))


def default_rollouts_path(model: str = "gemma12b") -> Path:
    """`rollouts.jsonl` written by `soar_nla.alpaca.rollouts`, else the shipped gzip copy."""
    directory = results_dir("alpaca") / model / "rollouts"
    plain = directory / "rollouts.jsonl"
    return plain if plain.exists() else directory / "rollouts.jsonl.gz"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a `.jsonl` or `.jsonl.gz` file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rollouts(path: Path | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl(path or default_rollouts_path())
    return sorted(rows, key=lambda row: int(row["selection_rank"]))


def long_responses(rows: list[dict[str, Any]], min_tokens: int = MIN_ASSISTANT_TOKENS) -> list[dict[str, Any]]:
    """Rollouts whose greedy response has at least `min_tokens` tokens (1,543 of 2,000)."""
    return [row for row in rows if int(row["assistant_token_count"]) >= min_tokens]


def late_positions(rows: list[dict[str, Any]], min_index: int = MIN_SEQUENCE_TOKEN_INDEX) -> list[dict[str, Any]]:
    """Rollouts whose sampled position is strictly after sequence token `min_index` (1,573 of 2,000)."""
    return [row for row in rows if int(row["sequence_token_index"]) > min_index]
