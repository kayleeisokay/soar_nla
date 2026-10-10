from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def stable_hash(*parts: object) -> str:
    data = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_seed(*parts: object) -> int:
    return int(stable_hash(*parts)[:16], 16)


def seeded_rng(*parts: object) -> random.Random:
    return random.Random(stable_seed(*parts))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


class ResumableJSONL:
    """Append-only JSONL with a caller-defined stable record key."""

    def __init__(self, path: Path, key_fields: tuple[str, ...]):
        self.path = path
        self.key_fields = key_fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = read_jsonl(path)
        self.done = {self.key(row) for row in self.rows}
        self.handle = self.path.open("a", encoding="utf-8")

    def key(self, row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(row[field] for field in self.key_fields)

    def contains(self, **fields: Any) -> bool:
        return tuple(fields[field] for field in self.key_fields) in self.done

    def append(self, row: dict[str, Any]) -> None:
        key = self.key(row)
        if key in self.done:
            raise ValueError(f"duplicate result key: {key}")
        self.handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.done.add(key)

    def close(self) -> None:
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
