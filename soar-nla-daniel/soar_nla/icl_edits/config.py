"""Model configs (configs/*.yaml): target, NLA pair, NLA layer, layer count and donor-sweep layers.

Every Hugging Face repo a config names must have a pinned revision in soar_nla.hub; that revision is
what gets loaded. Results go to soar_nla.paths.results_dir("icl_edits") / <config name>.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from soar_nla import hub
from soar_nla.paths import results_dir

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
CONFIGS = ("gemma3-12b", "gemma3-12b-custom-L24", "gemma3-12b-custom-L40")
PINNED = dict([hub.GEMMA3_12B, hub.KITFT_L32_AV, hub.KITFT_L32_AR, hub.CUSTOM_L24, hub.CUSTOM_L40])


def pinned_revision(repo: str) -> str:
    """Pinned commit for `repo` (or for the repo part of an easynla `repo:subdir` spec)."""
    repo = repo.split(":", 1)[0]
    try:
        return PINNED[repo]
    except KeyError as exc:
        raise ValueError(f"{repo} has no pinned revision in soar_nla.hub") from exc


@dataclass(frozen=True)
class ModelConfig:
    name: str
    target: str
    av: str
    ar: str
    layer: int
    n_layers: int
    sweep_layers: tuple[int, ...]
    backend: str = "kitft"      # kitft (SGLang AV, embedding injection) | easynla (see easynla.py)
    sidecar: str = ""           # easynla only: `repo:path/to/nla_meta.yaml`

    @property
    def results(self) -> Path:
        p = results_dir("icl_edits") / self.name
        p.mkdir(parents=True, exist_ok=True)
        return p


def config_path(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    return p if p.suffix in (".yaml", ".yml") else CONFIG_DIR / f"{name_or_path}.yaml"


def load_config(name_or_path: str | Path) -> ModelConfig:
    """Load a config by name (`gemma3-12b`) or by path to a YAML file."""
    d = yaml.safe_load(config_path(name_or_path).read_text())
    d["sweep_layers"] = tuple(d["sweep_layers"])
    cfg = ModelConfig(**d)
    for repo in (cfg.target, cfg.av, cfg.ar, *([cfg.sidecar] if cfg.sidecar else [])):
        pinned_revision(repo)
    return cfg
