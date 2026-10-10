"""Per-example rollouts for the benchmark summaries, figure and paper numbers.

A rollout is the per-example generation file of one (task, condition):

* ``clean`` and ``nla`` come from the task run directory
  (``matched_clean`` / ``patched`` shards for GSM8K, ``clean`` / ``patched``
  for IFEval and ARC-Easy);
* ``para_medium``, ``para_heavy`` and ``random_vec`` come from
  ``<arms-dir>/<task>__<arm>/patched_shard_NN.jsonl``.

They can also be read from an exported directory of gzipped JSONL files,
``<task>__<condition>.jsonl.gz`` (shards concatenated in shard order, lines
unchanged), as committed under ``results/benchmarks/rollouts``.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from soar_nla.benchmarks import common

TASKS = ("ifeval", "gsm8k", "arc_easy")
ARMS = ("para_medium", "para_heavy", "random_vec")
CONDITIONS = ("clean", "nla", *ARMS)
CLEAN_PREFIX = {"gsm8k": "matched_clean", "ifeval": "clean", "arc_easy": "clean"}


def read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    import json

    if not path.exists():
        return []
    # Split on "\n" only: str.splitlines also breaks on U+0085/U+2028, which
    # degenerate generations contain inside JSON strings.
    return [json.loads(line) for line in read_text(path).split("\n") if line.strip()]


def arm_dir(arms_dir: Path, task: str, arm: str) -> Path:
    return arms_dir / f"{task}__{arm}"


class RolloutSource:
    """Locate rollouts in run directories or in an exported directory."""

    def __init__(self, task_dirs: dict[str, Path] | None = None, arms_dir: Path | None = None,
                 exported: Path | None = None, num_shards: int = common.NUM_SHARDS):
        self.task_dirs = task_dirs or {}
        self.arms_dir = arms_dir
        self.exported = exported
        self.num_shards = num_shards

    def shard_files(self, task: str, condition: str) -> list[Path]:
        if condition in ("clean", "nla"):
            directory = self.task_dirs[task]
            prefix = CLEAN_PREFIX[task] if condition == "clean" else "patched"
        else:
            directory = arm_dir(self.arms_dir, task, condition)
            prefix = "patched"
        return [directory / f"{prefix}_shard_{shard:02d}.jsonl" for shard in range(self.num_shards)]

    def available(self, task: str, condition: str) -> bool:
        if self.exported is not None:
            return (self.exported / f"{task}__{condition}.jsonl.gz").exists()
        return all(path.exists() for path in self.shard_files(task, condition))

    def load(self, task: str, condition: str) -> dict[int, dict]:
        if self.exported is not None:
            paths = [self.exported / f"{task}__{condition}.jsonl.gz"]
        else:
            paths = self.shard_files(task, condition)
        rows = {}
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
            for row in read_jsonl(path):
                rows[int(row["index"])] = row
        return rows

    def export(self, task: str, condition: str, destination: Path) -> Path:
        """Concatenate the shard files verbatim into one gzipped JSONL file."""
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{task}__{condition}.jsonl.gz"
        with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as handle:
            for shard_path in self.shard_files(task, condition):
                data = shard_path.read_bytes()
                if data and not data.endswith(b"\n"):
                    data += b"\n"
                handle.write(data)
        return path


def add_source_args(parser) -> None:
    group = parser.add_argument_group("rollout locations")
    group.add_argument("--rollouts", type=Path, default=None,
                       help="exported <task>__<condition>.jsonl.gz directory; overrides the run directories")
    group.add_argument("--gsm8k-dir", type=Path, default=common.default_run_dir("gsm8k"))
    group.add_argument("--ifeval-dir", type=Path, default=common.default_run_dir("ifeval"))
    group.add_argument("--arc-easy-dir", type=Path, default=common.default_run_dir("arc_easy"))
    group.add_argument("--arms-dir", type=Path, default=common.default_run_dir("allpos"),
                       help="directory holding <task>__<arm>/ paraphrase and random-vector runs")


def task_dirs_from_args(args) -> dict[str, Path]:
    return {"gsm8k": args.gsm8k_dir, "ifeval": args.ifeval_dir, "arc_easy": args.arc_easy_dir}


def source_from_args(args) -> RolloutSource:
    if args.rollouts is not None:
        return RolloutSource(exported=args.rollouts)
    return RolloutSource(task_dirs=task_dirs_from_args(args), arms_dir=args.arms_dir)
