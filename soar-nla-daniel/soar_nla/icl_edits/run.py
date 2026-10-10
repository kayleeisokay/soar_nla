"""Command line entry point.

  python -m soar_nla.icl_edits.run serve-av --config gemma3-12b --gpus 4,5,6,7 --ports 31000,31001,31002,31003
  python -m soar_nla.icl_edits.run exp --config gemma3-12b --gpu 0 --av-ports 31000,31001,31002,31003 \
      --exp icl_redirect --fmt raw [--only antonym capitalize] [--tag a]
  python -m soar_nla.icl_edits.run exp --config gemma3-12b-custom-L24 --gpu 0,1 --exp icl_redirect --fmt raw

Experiments: icl_redirect (six Function Vectors tasks), icl_switch (three task switches).
`--config` takes a config name from configs/ or a path to a YAML file.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from soar_nla.paths import results_dir


def main(argv=None):
    ap = argparse.ArgumentParser(prog="soar_nla.icl_edits.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve-av", help="launch SGLang AV servers (kitft backend) and wait until healthy")
    s.add_argument("--config", required=True)
    s.add_argument("--gpus", required=True, help="one CUDA_VISIBLE_DEVICES entry per server, comma separated")
    s.add_argument("--ports", required=True)
    s.add_argument("--log-dir", type=Path, default=results_dir("icl_edits") / "logs")
    e = sub.add_parser("exp", help="run one experiment")
    e.add_argument("--config", required=True)
    e.add_argument("--gpu", required=True, help="physical GPU for the target model and the AR; with "
                   "backend easynla, a second GPU (`--gpu 1,2`) holds the AV")
    e.add_argument("--device", default="cuda:0", help="torch device of the target model and the AR "
                   "(relative to --gpu)")
    e.add_argument("--av-device", default="cuda:1", help="easynla backend: torch device of the AV")
    e.add_argument("--av-ports", default="", help="SGLang AV ports (kitft backend)")
    e.add_argument("--av-host", default="127.0.0.1", help="SGLang AV host (kitft backend)")
    e.add_argument("--exp", required=True, choices=["icl_redirect", "icl_switch"])
    e.add_argument("--fmt", default="chat", help="prompt format(s), comma separated: chat|raw")
    e.add_argument("--only", nargs="*", help="restrict to these task names")
    e.add_argument("--tag", default="", help="suffix for the results file (parallel shards)")
    a = ap.parse_args(argv)

    from .config import load_config
    cfg = load_config(a.config)
    if a.cmd == "serve-av":
        assert cfg.backend == "kitft", "the easynla backend runs its AV in-process; no server needed"
        from soar_nla.sglang_server import launch
        from soar_nla.hub import resolve_snapshot
        from .config import pinned_revision
        launch(resolve_snapshot(cfg.av, pinned_revision(cfg.av)), a.gpus.split(","),
               [int(p) for p in a.ports.split(",")], a.log_dir)
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    from .runtime import Runtime
    rt = Runtime(cfg, av_ports=[int(p) for p in a.av_ports.split(",") if p],
                 av_device=a.av_device if cfg.backend == "easynla" else None,
                 device=a.device, av_host=a.av_host)
    from . import experiment
    for fmt in a.fmt.split(","):
        experiment.run(rt, cfg, a.exp.split("_")[1], only=a.only, tag=a.tag, fmt=fmt)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
