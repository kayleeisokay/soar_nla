"""Target model + AR on one GPU, AV over SGLang. Every intervention is a literal replacement of the
residual stream at chosen (layer, position) pairs, applied during the prefill pass only.

Import this module only after CUDA_VISIBLE_DEVICES is set (run.py does that).
"""
from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from soar_nla.hub import resolve_snapshot
from soar_nla.nla_inference import NLAClient, NLACritic

from .config import ModelConfig, pinned_revision

INSTRUCTION = "Answer immediately and only in one word."


def snapshot(repo: str) -> Path:
    """Local path of `repo` at its pinned revision (downloaded if not cached)."""
    return resolve_snapshot(repo, pinned_revision(repo))


class _Patcher:
    """Forward hook: on the first (prefill) call, overwrite hidden[b, pos] for every row b."""

    def __init__(self, layer: int, patches: list[dict]):
        self.layer, self.patches, self.fired = layer, patches, False

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.fired or hidden.shape[1] == 1:
            return output
        self.fired = True
        n = hidden.shape[1]
        hidden = hidden.clone()
        for b, pl in enumerate(self.patches):
            for p, v in (pl or {}).get(self.layer, {}).items():
                hidden[b, p % n, :] = v.to(hidden.device, hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


class Runtime:
    def __init__(self, cfg: ModelConfig, av_ports: list[int] | None = None, load_ar: bool = True,
                 av_device: str | None = None, device: str = "cuda:0", av_host: str = "127.0.0.1"):
        self.cfg, self.device = cfg, device
        target = snapshot(cfg.target)
        self.tok = AutoTokenizer.from_pretrained(target)
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            target, dtype=torch.bfloat16, device_map={"": device},
            attn_implementation="eager").eval()
        if hasattr(self.model.model, "vision_tower"):
            del self.model.model.vision_tower
        gc.collect(); torch.cuda.empty_cache()
        inner = self.model.model
        self.layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
        assert len(self.layers) == cfg.n_layers, (len(self.layers), cfg.n_layers)

        self.ar, self.av = None, []
        if cfg.backend == "easynla":
            from .easynla import EasyAR, EasyAV, load_sidecar
            meta = load_sidecar(cfg.sidecar)
            assert meta["extraction"]["layer_index"] == cfg.layer, "sidecar layer != config layer"
            if load_ar:
                self.ar = EasyAR(target, cfg.ar, meta, device)
            if av_device:
                self.av = [EasyAV(target, cfg.av, meta, av_device)]
            return
        if load_ar:
            ar_dir = snapshot(cfg.ar)
            meta = yaml.safe_load((ar_dir / "nla_meta.yaml").read_text())
            assert meta["critic"]["extraction_layer_index"] == cfg.layer, "sidecar layer != config layer"
            self.ar = NLACritic(ar_dir, device=device, dtype=torch.bfloat16)
        if av_ports:
            av_dir = snapshot(cfg.av)
            meta = yaml.safe_load((av_dir / "nla_meta.yaml").read_text())
            assert meta["extraction_layer_index"] == cfg.layer, "sidecar layer != config layer"
            self.av = [NLAClient(av_dir, sglang_url=f"http://{av_host}:{p}") for p in av_ports]

    # ------------------------------------------------------------------ prompts
    def chat(self, user: str) -> list[int]:
        ids = self.tok.apply_chat_template([{"role": "user", "content": user}],
                                           tokenize=True, add_generation_prompt=True)
        return list(ids["input_ids"] if isinstance(ids, dict) else ids)

    def raw(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=True)["input_ids"]

    # ------------------------------------------------------------- activations
    @torch.inference_mode()
    def acts(self, ids: list[int]) -> torch.Tensor:
        """Residual-stream output of every decoder layer: [n_layers, T, d] float32 on CPU."""
        x = torch.tensor([ids], device=self.device)
        hs = self.model(input_ids=x, output_hidden_states=True, use_cache=False).hidden_states
        return torch.stack([h[0] for h in hs[1:]]).float().cpu()   # hs[L+1] is the output of layer L

    @torch.inference_mode()
    def generate(self, ids: list[int], patches: list[dict], max_new_tokens: int = 8,
                 sample: bool = False, temperature: float = 1.0, seed: int | None = None,
                 batch: int = 24) -> list[str]:
        """One row per entry of `patches` ({layer: {pos: vector}}); all rows share `ids`."""
        out = []
        for i in range(0, len(patches), batch):
            chunk = patches[i:i + batch]
            x = torch.tensor([ids] * len(chunk), device=self.device)
            layers = sorted({L for pl in chunk if pl for L in pl})
            hooks = [self.layers[L].register_forward_hook(_Patcher(L, chunk)) for L in layers]
            try:
                if seed is not None:
                    torch.manual_seed(seed + i); torch.cuda.manual_seed_all(seed + i)
                kw = dict(do_sample=sample, max_new_tokens=max_new_tokens,
                          pad_token_id=self.tok.pad_token_id, disable_compile=True)
                if sample:
                    kw.update(temperature=temperature, top_p=1.0, top_k=0)
                g = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), **kw)
            finally:
                for h in hooks:
                    h.remove()
            out += [self.tok.decode(g[b, len(ids):], skip_special_tokens=True) for b in range(len(chunk))]
        return out

    # ---------------------------------------------------------------- NLA I/O
    @torch.inference_mode()
    def reconstruct(self, texts: list[str], batch: int = 16) -> list[torch.Tensor]:
        """AR(text) for each text, left-padded batches; returns raw (unnormalized) vectors."""
        if self.cfg.backend == "easynla":
            return self.ar.reconstruct(texts, batch)
        tk, out = self.ar.tokenizer, []
        pad = tk.pad_token_id if tk.pad_token_id is not None else 0
        for i in range(0, len(texts), batch):
            ids = [tk(self.ar.template.format(explanation=t), add_special_tokens=True)["input_ids"]
                   for t in texts[i:i + batch]]
            L = max(len(x) for x in ids)
            inp = torch.tensor([[pad] * (L - len(x)) + x for x in ids], device=self.ar.device)
            att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device=self.ar.device)
            h = self.ar.backbone.model(inp, attention_mask=att, use_cache=False).last_hidden_state[:, -1]
            out += [v.float().cpu() for v in self.ar.value_head(h)]
        return out

    def verbalize(self, jobs: list[tuple[torch.Tensor, int]], workers: int = 64) -> list[str]:
        """AV explanation for each (vector, sampling_seed), temperature 1, spread over all servers."""
        if self.cfg.backend == "easynla":
            return self.av[0].verbalize(jobs)
        def one(a):
            i, (v, sd) = a
            for attempt in range(5):
                try:
                    return self.av[i % len(self.av)].generate(
                        np.asarray(v, dtype=np.float32), temperature=1.0, max_new_tokens=256,
                        sampling_seed=int(sd))
                except Exception:
                    if attempt == 4:
                        raise
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, list(enumerate(jobs))))


def rescale(v: torch.Tensor, norm: float) -> torch.Tensor:
    """Give an AR reconstruction the native norm of the residual it replaces."""
    return v / v.norm().clamp_min(1e-12) * float(norm)
