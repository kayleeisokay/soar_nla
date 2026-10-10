"""API-key lookup for the LLM judges and editors.

Keys are read from the environment. `--env-file` options on the CLIs load a
dotenv-style file first, without overriding variables that are already set.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


def api_key(name: str, env_file: Path | None = None) -> str:
    load_env_file(env_file)
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; export it or pass --env-file")
    return value
