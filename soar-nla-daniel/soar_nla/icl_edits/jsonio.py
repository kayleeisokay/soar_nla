from __future__ import annotations

import gzip
import json
import os
from pathlib import Path


def save_json(obj, path: Path) -> None:
    """Atomic write, so a killed run never leaves a truncated results file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


def load_json(path: Path, default):
    return json.load(open(path)) if path.exists() else default


def read_json(path: Path):
    """Read `path`, or `path.gz` if only the gzipped copy exists (the shipped per-row results)."""
    path = Path(path)
    if path.exists():
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    gz = path.with_name(path.name + ".gz")
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        return json.load(f)


def result_files(res: Path, stem: str) -> list[Path]:
    """`{stem}.json` then the sorted shards `{stem}-*.json`, each plain or gzipped (plain preferred).

    Returned paths name the plain file; read them with read_json."""
    names = set()
    for pat in (f"{stem}.json", f"{stem}-*.json", f"{stem}.json.gz", f"{stem}-*.json.gz"):
        names.update(p.name.removesuffix(".gz") for p in Path(res).glob(pat))
    head = [n for n in names if n == f"{stem}.json"]
    return [Path(res) / n for n in head + sorted(n for n in names if n != f"{stem}.json")]
