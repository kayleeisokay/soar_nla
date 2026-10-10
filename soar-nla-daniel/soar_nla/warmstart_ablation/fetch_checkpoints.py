#!/usr/bin/env python3
"""Download the pinned step-600 checkpoints and RL-data sidecar used by gen_from_acts.py.

Writes `<out>/ckpts/...` (subfolders of the checkpoint repo listed in
runs.CKPT_SUBDIRS) and `<out>/data/rl_shuf.full.parquet` (+ sidecar), then
prints the gen_from_acts.py arguments for each run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from soar_nla.warmstart_ablation.runs import CKPT_SUBDIRS, NLA_CKPTS, NLA_DATA, QWEN25_7B, RUNS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS))
    ap.add_argument("--skip-data", action="store_true",
                    help="do not download the 3 GB RL parquet (only its first row group is read)")
    a = ap.parse_args()

    # fnmatch "*" also matches "/", so this includes nested reference/ adapters.
    patterns = [f"{sub}/*" for run in a.runs for sub in CKPT_SUBDIRS[run].values()]
    ckpt_root = Path(snapshot_download(NLA_CKPTS[0], revision=NLA_CKPTS[1],
                                       local_dir=a.out / "ckpts", allow_patterns=patterns))
    data_root = a.out / "data"
    if not a.skip_data:
        snapshot_download(NLA_DATA[0], repo_type="dataset", revision=NLA_DATA[1], local_dir=data_root,
                          allow_patterns=["rl_shuf.full.parquet", "rl_shuf.full.parquet.nla_meta.yaml"])
    parquet = data_root / "rl_shuf.full.parquet"
    for run in a.runs:
        sub = CKPT_SUBDIRS[run]
        av_ckpt = ckpt_root / sub["av_ckpt"] if "av_ckpt" in sub else QWEN25_7B[0]
        cmd = [f"--run {run}", f"--av-ckpt {av_ckpt}", f"--lora {ckpt_root / sub['lora']}",
               f"--parquet {parquet}", f"--meta {parquet}"]
        if "av_adapter" in sub:
            cmd.append(f"--av-adapter {ckpt_root / sub['av_adapter']}")
        if run == "B":
            cmd.append("--raw-fallback")
        print(f"[{run}] " + " ".join(cmd))


if __name__ == "__main__":
    main()
