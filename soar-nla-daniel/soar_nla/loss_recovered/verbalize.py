"""Greedy AV explanation of every window activation, served by one or more SGLang AV servers.

Activation `index` goes to server `index % len(urls)`. The output is resumable:
rerunning skips indices already in `explanations.jsonl`.

    python -m soar_nla.loss_recovered.verbalize --urls http://127.0.0.1:30000,http://127.0.0.1:30001
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from soar_nla.hub import model_spec, resolve_snapshot
from soar_nla.io_utils import ResumableJSONL, read_jsonl, write_json
from soar_nla.nla_inference import NLAClient

from .common import default_results_dir
from .config import AV_MAX_NEW_TOKENS, MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=default_results_dir())
    parser.add_argument("--urls", required=True, help="comma-separated SGLang AV server URLs")
    parser.add_argument("--workers-per-server", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=AV_MAX_NEW_TOKENS)
    args = parser.parse_args()

    spec = model_spec(MODEL)
    vectors = np.load(args.output_dir / "activations.npy", mmap_mode="r")
    context_size = int(np.load(args.output_dir / "windows.npy", mmap_mode="r").shape[1])
    av_dir = resolve_snapshot(spec.av_id, spec.av_revision)
    urls = [url.strip().rstrip("/") for url in args.urls.split(",") if url.strip()]
    clients = [NLAClient(av_dir, sglang_url=url, device="cpu") for url in urls]

    def explain(index: int) -> dict[str, object]:
        client = index % len(clients)
        for attempt in range(4):
            try:
                text = clients[client].generate(vectors[index], temperature=0.0, max_new_tokens=args.max_new_tokens)
                break
            except Exception as exc:  # transient server errors: back off and retry
                if attempt == 3:
                    raise RuntimeError(f"AV failed for index {index}") from exc
                time.sleep(2**attempt)
        return {
            "index": index,
            "window_index": index // context_size,
            "position": index % context_size,
            "server": client,
            "explanation": text,
        }

    output = args.output_dir / "explanations.jsonl"
    with ResumableJSONL(output, ("index",)) as sink:
        pending = [index for index in range(len(vectors)) if not sink.contains(index=index)]
        with ThreadPoolExecutor(max_workers=args.workers_per_server * len(clients)) as pool:
            futures = [pool.submit(explain, index) for index in pending]
            for done, future in enumerate(as_completed(futures), start=1):
                sink.append(future.result())
                if done % 1000 == 0 or done == len(pending):
                    print(f"verbalize {len(vectors) - len(pending) + done}/{len(vectors)}", flush=True)

    if {int(row["index"]) for row in read_jsonl(output)} != set(range(len(vectors))):
        raise RuntimeError(f"{output} is incomplete")
    write_json(
        args.output_dir / "verbalization.json",
        {
            "av": f"{spec.av_id}@{spec.av_revision}",
            "count": len(vectors),
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "servers": len(urls),
            "workers_per_server": args.workers_per_server,
        },
    )


if __name__ == "__main__":
    main()
