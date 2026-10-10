"""Launch and health-check SGLang servers that host the activation verbalizer (AV).

The AV is driven through `input_embeds`, which carry no token IDs, so
`--disable-radix-cache` is required: the radix cache would otherwise alias
different requests to the same entry. SGLang must be patched with
`patches/sglang/apply_sglang_patches.sh` before Gemma-3 AVs will accept
`input_embeds`.

Environment:
  SOAR_SGLANG_PYTHON  interpreter that has the patched SGLang installed
                      (default: the current interpreter).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SGLANG_PYTHON = os.environ.get("SOAR_SGLANG_PYTHON", sys.executable)


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def wait_healthy(urls: list[str], timeout: float = 1800) -> None:
    deadline = time.monotonic() + timeout
    while not all(healthy(url) for url in urls):
        if time.monotonic() > deadline:
            raise TimeoutError(f"SGLang servers not healthy after {timeout:.0f}s: {urls}")
        time.sleep(5)


def launch(
    model_path: str | Path,
    gpus: list[str],
    ports: list[int],
    log_dir: Path,
    *,
    mem_fraction: float = 0.80,
    context_length: int = 512,
    tp_size: int = 1,
    extra_args: tuple[str, ...] = ("--attention-backend", "triton", "--disable-cuda-graph"),
    wait_s: float = 1800,
) -> list[str]:
    """Start one server per (gpu group, port) unless one is already healthy there.

    `gpus` holds one CUDA_VISIBLE_DEVICES string per server (e.g. "4" or "6,7").
    Blocks until every server answers /health and returns their base URLs.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    urls = [f"http://127.0.0.1:{port}" for port in ports]
    for gpu, port, url in zip(gpus, ports, urls):
        if healthy(url):
            continue
        env = os.environ.copy()
        env.update(
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_VISIBLE_DEVICES=gpu,
            SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR="1",
        )
        cmd = [
            SGLANG_PYTHON, "-m", "sglang.launch_server",
            "--model-path", str(model_path),
            "--port", str(port),
            "--host", "127.0.0.1",
            "--disable-radix-cache",
            "--mem-fraction-static", str(mem_fraction),
            "--context-length", str(context_length),
            "--tp-size", str(tp_size),
            "--trust-remote-code",
            "--log-level", "warning",
            *extra_args,
        ]
        with open(log_dir / f"sglang_{port}.log", "ab") as log:
            subprocess.Popen(cmd, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        print(f"launched AV on GPU {gpu}, port {port}", flush=True)
    wait_healthy(urls, wait_s)
    print(f"AV servers healthy: {urls}", flush=True)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Block until an SGLang server answers /health.")
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    wait_healthy([args.url], args.timeout)
    print(f"ready: {args.url}", flush=True)


if __name__ == "__main__":
    main()
