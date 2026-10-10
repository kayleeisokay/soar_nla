"""Greedy AV decoding of the sampled rollout activations through an SGLang server."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from safetensors.torch import load_file

from soar_nla.hub import MODEL_SPECS, model_spec, resolve_snapshot
from soar_nla.io_utils import ResumableJSONL, read_jsonl, write_json, write_jsonl
from soar_nla.nla_inference import EXPLANATION_RE, NLAClient
from soar_nla.paths import results_dir

from .config import N_EXAMPLES, PROTOCOL_VERSION


def run_worker(args: argparse.Namespace) -> None:
    spec = model_spec(args.model)
    source = args.results_dir / spec.key / "rollouts"
    rows = read_jsonl(source / "rollouts.jsonl")
    activations = load_file(str(source / "activations.safetensors"))["activations"].float()
    if len(rows) != args.n or len(activations) != args.n:
        raise RuntimeError("rollouts must be merged before AV decoding")
    av_dir = resolve_snapshot(spec.av_id, spec.av_revision)
    client = NLAClient(av_dir, sglang_url=args.sglang_url, device="cpu")
    work = [row for row in rows if row["selection_rank"] % args.num_shards == args.shard]
    output_dir = args.results_dir / spec.key / "decode"
    output_path = output_dir / f"shard-{args.shard:02d}-of-{args.num_shards:02d}.jsonl"

    def decode_one(row: dict[str, object]) -> dict[str, object]:
        rank = int(row["selection_rank"])
        raw = client.generate(
            activations[rank],
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
            extract_explanation=False,
        )
        match = EXPLANATION_RE.search(raw)
        return {
            "protocol": PROTOCOL_VERSION,
            "model": spec.key,
            "sample_id": row["sample_id"],
            "selection_rank": rank,
            "parse_ok": match is not None,
            "raw_generation": raw,
            "explanation": match.group(1).strip() if match else raw.strip(),
        }

    with ResumableJSONL(output_path, ("sample_id",)) as sink:
        pending = [row for row in work if not sink.contains(sample_id=row["sample_id"])]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(decode_one, row): row for row in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                sink.append(result)
                if completed % 20 == 0 or completed == len(pending):
                    print(f"[{spec.key} decode shard {args.shard}] {completed}/{len(pending)} new", flush=True)
    client._http.close()
    write_json(
        output_dir / f"shard-{args.shard:02d}-of-{args.num_shards:02d}.manifest.json",
        {
            "protocol": PROTOCOL_VERSION,
            "model": spec.__dict__,
            "av_checkpoint": str(av_dir),
            "sglang_url": args.sglang_url,
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "shard": args.shard,
            "num_shards": args.num_shards,
        },
    )


def merge(args: argparse.Namespace) -> None:
    spec = model_spec(args.model)
    directory = args.results_dir / spec.key / "decode"
    rows = []
    for shard in range(args.num_shards):
        rows.extend(read_jsonl(directory / f"shard-{shard:02d}-of-{args.num_shards:02d}.jsonl"))
    rows.sort(key=lambda row: row["selection_rank"])
    if len(rows) != args.n or len({row["sample_id"] for row in rows}) != args.n:
        raise RuntimeError(f"expected {args.n} unique decoded rows, got {len(rows)}")
    write_jsonl(directory / "explanations.jsonl", rows)
    write_json(
        directory / "manifest.json",
        {
            "protocol": PROTOCOL_VERSION,
            "model": spec.__dict__,
            "rows": len(rows),
            "parse_failures": sum(not row["parse_ok"] for row in rows),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker", help="decode one shard against a running AV server")
    worker.add_argument("--model", default="gemma12b", choices=sorted(MODEL_SPECS))
    worker.add_argument("--results-dir", type=Path, default=results_dir("alpaca"))
    worker.add_argument("--n", type=int, default=N_EXAMPLES)
    worker.add_argument("--shard", type=int, required=True)
    worker.add_argument("--num-shards", type=int, default=4)
    worker.add_argument("--sglang-url", required=True)
    worker.add_argument("--workers", type=int, default=8)
    worker.add_argument("--max-new-tokens", type=int, default=220)
    merger = sub.add_parser("merge", help="merge decode shards into explanations.jsonl")
    merger.add_argument("--model", default="gemma12b", choices=sorted(MODEL_SPECS))
    merger.add_argument("--results-dir", type=Path, default=results_dir("alpaca"))
    merger.add_argument("--n", type=int, default=N_EXAMPLES)
    merger.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()
    run_worker(args) if args.command == "worker" else merge(args)


if __name__ == "__main__":
    main()
