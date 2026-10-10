"""Download the function-vector task files used by the protocol and verify their SHA-256.

The files are fetched from the reference repository at the pinned commit
(``notebooks/datasets/``) rather than vendored here, because their upstream
licenses are not stated there (see the package README). Existing files with the
expected hash are kept; anything else is an error.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from soar_nla.fv_chat_template.protocol import (
    DATA_FILES,
    REFERENCE_COMMIT,
    REFERENCE_DATA_DIR,
    REFERENCE_REPO,
)
from soar_nla.paths import results_dir

DEFAULT_DATA_DIR = results_dir("fv_chat_template") / "datasets"


def raw_url(filename: str) -> str:
    owner_repo = REFERENCE_REPO.removeprefix("https://github.com/")
    return f"https://raw.githubusercontent.com/{owner_repo}/{REFERENCE_COMMIT}/{REFERENCE_DATA_DIR}/{filename}"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(data_dir: Path, timeout: float = 60.0) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected in DATA_FILES.items():
        destination = data_dir / filename
        if destination.exists():
            actual = sha256(destination.read_bytes())
            if actual != expected:
                raise RuntimeError(f"{destination} has SHA-256 {actual}, expected {expected}")
            print(f"ok      {filename}")
            continue
        with urllib.request.urlopen(raw_url(filename), timeout=timeout) as response:
            data = response.read()
        actual = sha256(data)
        if actual != expected:
            raise RuntimeError(f"downloaded {filename} has SHA-256 {actual}, expected {expected}")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
        print(f"fetched {filename} ({len(data)} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    fetch(args.data_dir)


if __name__ == "__main__":
    main()
