"""Download the pinned Alpaca snapshot and freeze the 2,000-prompt sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from soar_nla.hub import MODEL_SPECS
from soar_nla.io_utils import stable_hash, write_json, write_jsonl
from soar_nla.paths import results_dir

from .config import DATASET_ID, DATASET_REVISION, N_EXAMPLES, SELECTION_SEED


def render_user_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if not input_text:
        return instruction
    return f"{instruction}\n\nInput:\n{input_text}"


def download_and_select(output_dir: Path, n: int = N_EXAMPLES, include_models: bool = False) -> None:
    """Download the pinned Alpaca snapshot and freeze a content-distinct sample."""
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    candidates = []
    seen_prompts: set[str] = set()
    for dataset_index, item in enumerate(dataset):
        user_prompt = render_user_prompt(str(item["instruction"]), str(item["input"]))
        content_hash = stable_hash(user_prompt)
        if content_hash in seen_prompts:
            continue
        seen_prompts.add(content_hash)
        candidates.append(
            {
                "dataset_index": dataset_index,
                "sample_id": content_hash[:20],
                "order_hash": stable_hash(SELECTION_SEED, dataset_index, content_hash),
                "instruction": str(item["instruction"]),
                "input": str(item["input"]),
                "reference_output": str(item["output"]),
                "user_prompt": user_prompt,
            }
        )
    candidates.sort(key=lambda row: row["order_hash"])
    if len(candidates) < n:
        raise RuntimeError(f"only {len(candidates)} distinct Alpaca prompts, requested {n}")
    selected = candidates[:n]
    for rank, row in enumerate(selected):
        row["selection_rank"] = rank
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "alpaca_selected.jsonl", selected)
    write_json(
        output_dir / "dataset_manifest.json",
        {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "split": "train",
            "selection_seed": SELECTION_SEED,
            "n": n,
            "distinct_user_prompt_hashes": len({row["sample_id"] for row in selected}),
        },
    )
    if include_models:
        for spec in MODEL_SPECS.values():
            for repo_id, revision in (
                (spec.target_id, spec.target_revision),
                (spec.av_id, spec.av_revision),
                (spec.ar_id, spec.ar_revision),
            ):
                snapshot_download(repo_id=repo_id, revision=revision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=results_dir("alpaca") / "data")
    parser.add_argument("--n", type=int, default=N_EXAMPLES)
    parser.add_argument("--include-models", action="store_true", help="also download the pinned target/AV/AR")
    args = parser.parse_args()
    download_and_select(args.output_dir, args.n, args.include_models)


if __name__ == "__main__":
    main()
