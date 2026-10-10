"""Recompute the 2-shot English->French accuracies (raw prompt / chat template) from rows.json.

Each arm's accuracy is re-derived from the stored generations (first line,
punctuation stripped, case-insensitive match to the gold French word), checked
against the stored per-row ``correct`` flag, and compared with the paper (in
percent, n=50 queries).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from soar_nla.fv_chat_template.protocol import correct
from soar_nla.paths import results_dir

# Accuracy (%) per arm: (raw prompt, chat template).
PAPER = {
    "baseline": (38, 40),
    "nla": (34, 38),
    "para_medium": (34, 38),
    "para_heavy": (26, 34),
    "random_vec": (2, 0),
    "mismatch_cross_task": (4, 10),
    "mismatch_same_task": (0, 16),
}


def accuracy(rows: list[dict], key: str) -> float:
    flags = []
    for row in rows:
        entry = row["arms"][key]
        flag = correct(entry["output"], row["gold"])
        if flag != entry["correct"]:
            raise RuntimeError(f"position {row['position']} {key}: rescored correctness disagrees with the stored flag")
        flags.append(flag)
    return 100 * sum(flags) / len(flags)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=results_dir("fv_chat_template") / "shot2" / "rows.json")
    args = parser.parse_args()
    rows = json.loads(args.rows.read_text())
    if len(rows) != 50:
        raise RuntimeError(f"expected 50 rows, found {len(rows)}")
    failures = 0
    print(f"{'arm':<20} {'raw':>6} {'chat':>6}   paper")
    for arm in ("identity", *PAPER):
        raw = accuracy(rows, f"raw_{arm}")
        chat = accuracy(rows, f"chat_{arm}")
        expected = PAPER.get(arm)
        status = ""
        if expected is not None:
            status = "ok" if (raw, chat) == expected else "MISMATCH"
            failures += status != "ok"
        paper = f"{expected[0]}/{expected[1]}" if expected else "-"
        print(f"{arm:<20} {raw:>6.0f} {chat:>6.0f}   {paper:<7} {status}")
    if failures:
        print(f"{failures} arm(s) differ from the paper", file=sys.stderr)
        sys.exit(1)
    print("all arms match the paper")


if __name__ == "__main__":
    main()
