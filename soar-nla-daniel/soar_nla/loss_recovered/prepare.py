"""Build the evaluation windows: one 128-token prefix of each chat-rendered Alpaca conversation.

Each of the first 2,000 conversations is rendered with the Gemma chat template
(BOS, system prompt merged into the user turn, user turn, model turn) and
tokenized. Conversations shorter than 128 tokens are dropped; the rest
contribute their first 128 tokens. Windows never cross a conversation boundary
and are never padded, so every position of every window is a real token.

    python -m soar_nla.loss_recovered.prepare --alpaca /path/to/google-gemma-3-12b-it/alpaca.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from soar_nla.hub import model_spec, resolve_snapshot
from soar_nla.io_utils import write_json, write_jsonl

from .common import default_results_dir
from .config import ALPACA_FILE, ALPACA_SHA256, CONTEXT_SIZE, MODEL, N_CONVERSATIONS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpaca", type=Path, required=True, help=f"the rollouts parquet ({ALPACA_FILE})")
    parser.add_argument("--output-dir", type=Path, default=default_results_dir())
    parser.add_argument("--tokenizer", type=Path, default=None, help="local tokenizer (default: pinned Gemma-3-12B-IT)")
    parser.add_argument("--n", type=int, default=N_CONVERSATIONS)
    parser.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    parser.add_argument("--allow-other-data", action="store_true", help="accept a parquet with a different SHA-256")
    args = parser.parse_args()

    digest = sha256_file(args.alpaca)
    if digest != ALPACA_SHA256 and not args.allow_other_data:
        raise SystemExit(f"{args.alpaca} has SHA-256 {digest}, expected {ALPACA_SHA256} (pass --allow-other-data)")
    table = pq.read_table(args.alpaca, columns=["messages"])
    if len(table) < args.n:
        raise ValueError(f"{args.alpaca} has {len(table)} rows, need {args.n}")

    spec = model_spec(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or resolve_snapshot(spec.target_id, spec.target_revision), trust_remote_code=True
    )

    windows: list[list[int]] = []
    manifest: list[dict[str, int]] = []
    for source_index in range(args.n):
        messages = json.loads(table["messages"][source_index].as_py())
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if len(ids) < args.context_size:
            continue
        manifest.append(
            {"window_index": len(windows), "source_index": source_index, "conversation_token_count": len(ids)}
        )
        windows.append([int(token) for token in ids[: args.context_size]])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "windows.npy", np.asarray(windows, dtype=np.int64))
    write_jsonl(args.output_dir / "windows.jsonl", manifest)
    write_json(
        args.output_dir / "preparation.json",
        {
            "model": f"{spec.target_id}@{spec.target_revision}",
            "alpaca_file": ALPACA_FILE,
            "alpaca_sha256": digest,
            "conversations": args.n,
            "context_size": args.context_size,
            "windows": len(windows),
            "excluded_short_conversations": args.n - len(windows),
            "activation_positions": len(windows) * args.context_size,
        },
    )
    print(f"wrote {len(windows)} windows x {args.context_size} tokens; dropped {args.n - len(windows)} short conversations")


if __name__ == "__main__":
    main()
