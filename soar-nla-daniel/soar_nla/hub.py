"""Pinned checkpoints and model loading shared by every experiment.

Every model the paper uses is listed here with the exact Hugging Face revision
the reported numbers were produced with. Experiments look models up by key
instead of hardcoding repo IDs or cache paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

GEMMA3_12B = ("google/gemma-3-12b-it", "96b6f1eccf38110c56df3a15bffe176da04bfd80")
GEMMA3_27B = ("google/gemma-3-27b-it", "005ad3404e59d6023443cb575daa05336842228a")

# Official released Gemma-3-12B NLA (layer 32).
KITFT_L32_AV = ("kitft/nla-gemma3-12b-L32-av", "7aec22599e8a9cd533564868999b443fcc963cf4")
KITFT_L32_AR = ("kitft/nla-gemma3-12b-L32-ar", "3d6901d8243182d642af9dea452ca91549a94615")

# Custom Gemma-3-12B NLAs trained with the vendored EasyNLA trainer (training/).
# Each repo holds the warm-start SFT pair under merged/ and the RL pair under rl_vllm/.
CUSTOM_L24 = ("achand45/gemma-3-12b-it-nla-L24", "83e015a0e9af61cb703695d20fd63c0a77c7bf16")
CUSTOM_L32 = ("achand45/gemma-3-12b-it-nla-L32", "6bad086e3653a8e4466b430bf6713b9de547821f")
CUSTOM_L40 = ("achand45/gemma-3-12b-it-nla-L40", "edb2ef921706d837927a3138d9c22a5f13add89a")

# Gemma Scope 2 residual-stream SAEs for Gemma-3-12B-IT (Loss Recovered SAE baseline).
GEMMA_SCOPE_2_12B = ("google/gemma-scope-2-12b-it", "4c419f1ba0be8b7754d4151d4f26c23b92a9029e")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    target_id: str
    target_revision: str
    av_id: str
    av_revision: str
    ar_id: str
    ar_revision: str
    layer: int
    d_model: int


MODEL_SPECS = {
    "gemma12b": ModelSpec(
        key="gemma12b",
        target_id=GEMMA3_12B[0],
        target_revision=GEMMA3_12B[1],
        av_id=KITFT_L32_AV[0],
        av_revision=KITFT_L32_AV[1],
        ar_id=KITFT_L32_AR[0],
        ar_revision=KITFT_L32_AR[1],
        layer=32,
        d_model=3840,
    ),
}


def model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"unknown model {key!r}; choose from {sorted(MODEL_SPECS)}") from exc


def _snapshot_complete(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    index_path = path / "model.safetensors.index.json"
    single_path = path / "model.safetensors"
    if single_path.exists():
        return True
    if not index_path.exists():
        return False
    try:
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    except (KeyError, OSError, ValueError):
        return False
    return all((path / filename).exists() for filename in set(weight_map.values()))


def resolve_snapshot(repo_id: str, revision: str, local_files_only: bool = True) -> Path:
    """Return the local path of a pinned snapshot, downloading it if it is not cached."""
    try:
        path = Path(snapshot_download(repo_id=repo_id, revision=revision, local_files_only=local_files_only))
        if not _snapshot_complete(path):
            raise LocalEntryNotFoundError(f"incomplete cached snapshot at {path}")
    except LocalEntryNotFoundError:
        if not local_files_only:
            raise
        path = Path(snapshot_download(repo_id=repo_id, revision=revision, local_files_only=False))
        if not _snapshot_complete(path):
            raise RuntimeError(f"downloaded snapshot is incomplete: {path}")
    if path.name != revision:
        raise RuntimeError(f"resolved {repo_id} to {path.name}, expected pinned revision {revision}")
    return path


def resolve_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = [
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ]
    found = []
    seen: set[int] = set()
    for candidate in candidates:
        if hasattr(candidate, "layers") and id(candidate) not in seen:
            found.append(candidate.layers)
            seen.add(id(candidate))
    if len(found) != 1:
        raise TypeError(f"could not uniquely resolve decoder layers on {type(model).__name__}: {len(found)} matches")
    return found[0]


def load_target(spec: ModelSpec, device: str = "cuda:0", checkpoint: Path | None = None) -> tuple[Any, Any, Path]:
    """Load the target model; `checkpoint` overrides the pinned snapshot with a local copy."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = checkpoint or resolve_snapshot(spec.target_id, spec.target_revision)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()
    model.requires_grad_(False)
    return model, tokenizer, checkpoint
