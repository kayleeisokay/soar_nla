"""Recreate the sampling/prompt plan of the reference shot-sweep executor.

CPU-only. Replays every RNG draw preceding (and following) the 2-shot
English->French configuration, including the other tasks and the 1- and
4-shot configs, and writes ``sampling_manifest.json`` with the per-row query,
demonstrations, random-vector seed and the cross-task and same-task donor
prompts, plus the SHA-256 of every data file read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from soar_nla.fv_chat_template.fetch_data import DEFAULT_DATA_DIR
from soar_nla.fv_chat_template.protocol import (
    DATA_FILES,
    REFERENCE_COMMIT,
    REFERENCE_FILE,
    REFERENCE_REPO,
    SHOT_COUNTS,
    TASKS,
    build_prompt,
    load_pairs,
)
from soar_nla.paths import results_dir

DEFAULT_MANIFEST = results_dir("fv_chat_template") / "sampling_manifest.json"


def prepare(data_dir: Path, n_per_config: int = 50, seed: int = 0):
    pairs = {task: load_pairs(data_dir, task) for task in TASKS}
    rng = random.Random(seed)
    configs = {}
    for task in TASKS:
        for n_shots in SHOT_COUNTS:
            task_pairs = pairs[task]
            query_indices = rng.sample(range(len(task_pairs)), min(n_per_config, len(task_pairs)))
            other_tasks = [name for name in pairs if name != task and len(pairs[name]) > 0]
            rows = []
            for position, query_index in enumerate(query_indices):
                prompt, gold, demos = build_prompt(task_pairs, query_index, n_shots, rng)
                random_seed = rng.getrandbits(32)
                foreign_task = rng.choice(other_tasks)
                foreign_pairs = pairs[foreign_task]
                foreign_index = rng.randrange(len(foreign_pairs))
                foreign_prompt, _, foreign_demos = build_prompt(
                    foreign_pairs, foreign_index, n_shots, rng
                )
                query_word = task_pairs[query_index]["input"]
                same_pool = [
                    index for index in range(len(task_pairs))
                    if index != query_index and task_pairs[index]["input"] != query_word
                ]
                same_index = rng.choice(same_pool)
                same_prompt, _, same_demos = build_prompt(
                    task_pairs, same_index, n_shots, rng,
                    extra_exclude_words={query_word},
                )
                rows.append({
                    "position": position,
                    "task": task,
                    "n_shots": n_shots,
                    "query_index": query_index,
                    "query_input": query_word,
                    "gold": gold,
                    "demo_indices": demos,
                    "prompt_raw": prompt,
                    "random_seed": random_seed,
                    "foreign_task": foreign_task,
                    "foreign_query_index": foreign_index,
                    "foreign_demo_indices": foreign_demos,
                    "foreign_prompt_raw": foreign_prompt,
                    "same_task_query_index": same_index,
                    "same_task_demo_indices": same_demos,
                    "same_task_prompt_raw": same_prompt,
                })
            configs[f"{task}:{n_shots}"] = rows
    hashes = {
        filename: hashlib.sha256((data_dir / filename).read_bytes()).hexdigest()
        for filename in DATA_FILES
    }
    mismatched = [name for name, digest in hashes.items() if digest != DATA_FILES[name]]
    if mismatched:
        raise RuntimeError(f"data files differ from the reference commit: {mismatched}")
    return {
        "reference_repo": REFERENCE_REPO,
        "reference_commit": REFERENCE_COMMIT,
        "reference_file": REFERENCE_FILE,
        "dataset_files_sha256": hashes,
        "seed": seed,
        "n_per_config": n_per_config,
        "shot_counts": list(SHOT_COUNTS),
        "task_order": list(TASKS),
        "dataset_sizes": {task: len(task_pairs) for task, task_pairs in pairs.items()},
        "configs": configs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = prepare(args.data_dir, args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.out)
    for name in ("english-french:1", "english-french:2", "english-french:4"):
        print(name, "first queries:", [row["query_input"] for row in result["configs"][name][:6]])


if __name__ == "__main__":
    main()
