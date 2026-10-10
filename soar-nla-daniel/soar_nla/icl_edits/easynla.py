"""Backend for EasyNLA checkpoints, e.g. the custom layer-24/40 pairs (soar_nla.hub.CUSTOM_L24/L40).

These differ from the kitft releases in both halves:
  AV  the target's text stack plus a LoRA adapter. The activation is added at the marker token on
      the output of decoder layer 1, norm-matched: h <- h + |h| * v/|v| (Karvonen et al., 2025),
      instead of replacing the marker's input embedding.
  AR  truncated backbone with the final norm stripped; the last token's hidden state is scaled to
      sqrt(d_model) before an fp32 linear value head. Critic text is tokenized without <bos>.
Prompts and marker token IDs come from the checkpoint's sidecar (nla_meta.yaml).

A checkpoint is named `repo:subdir`, e.g. `achand45/gemma-3-12b-it-nla-L24:rl_vllm/iter_000400`; the
repo is loaded at its revision pinned in soar_nla.hub. `base` is the local path of the pinned target.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import pinned_revision

EXPLANATION_RE = re.compile(r"<explanation>(.*?)(?:</explanation>|$)", re.DOTALL)


def resolve(spec: str) -> Path:
    """`repo:subdir` -> local path of subdir (a file or a directory) at the pinned revision,
    downloading only what it needs."""
    repo, sub = spec.split(":", 1)
    pats = [sub] if sub.endswith((".yaml", ".json")) else [f"{sub}/*"]
    kw = dict(repo_id=repo, revision=pinned_revision(repo), allow_patterns=pats,
              ignore_patterns=[f"{sub}/reference/*"])
    try:
        path = Path(snapshot_download(**kw, local_files_only=True)) / sub
        if not path.exists() or (path.is_dir() and not any(path.iterdir())):
            raise LocalEntryNotFoundError(f"{spec} is not cached")
    except LocalEntryNotFoundError:
        path = Path(snapshot_download(**kw)) / sub
    return path


def load_sidecar(spec: str) -> dict:
    return yaml.safe_load(resolve(spec).read_text())


def _text_causal_lm(base: str | Path, device: str) -> nn.Module:
    """Gemma-3 checkpoints load as a multimodal wrapper; return a text-only CausalLM on `device`."""
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": device})
    inner = getattr(m.model, "language_model", None)
    if inner is None:
        return m.eval()
    with torch.device("meta"):
        text = AutoModelForCausalLM.from_config(inner.config)
    text.model = inner
    text.lm_head = nn.Linear(inner.config.hidden_size, inner.config.vocab_size, bias=False, device=device,
                             dtype=torch.bfloat16)
    text.lm_head.weight = inner.embed_tokens.weight
    del m
    return text.eval()


class EasyAV:
    """Batched sampling from the AV; all rows share one prompt and differ only in the injected vector."""

    def __init__(self, base: str | Path, adapter: str, meta: dict, device: str = "cuda:1"):
        from peft import PeftModel
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(base)
        model = PeftModel.from_pretrained(_text_causal_lm(base, device), str(resolve(adapter)))
        self.model = model.merge_and_unload().eval()
        t = meta["tokens"]
        user = meta["prompt_templates"]["actor"].format(injection_char=t["injection_char"])
        text = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                            add_generation_prompt=True)
        self.ids = self.tok.encode(text, add_special_tokens=False)
        hits = [i for i in range(1, len(self.ids) - 1) if self.ids[i] == t["injection_token_id"]
                and self.ids[i - 1] == t["injection_left_neighbor_id"] and self.ids[i + 1] == t["injection_right_neighbor_id"]]
        assert len(hits) == 1, f"marker trigram found {len(hits)} times in the AV prompt"
        self.pos, self.vecs = hits[0], None
        self.model.model.layers[1].register_forward_hook(self._inject)

    def _inject(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.vecs is None or h.shape[1] == 1:
            return output
        h = h.clone()
        cur = h[:, self.pos].float()
        v = self.vecs.to(h.device)
        h[:, self.pos] = (cur + cur.norm(dim=-1, keepdim=True) * v / v.norm(dim=-1, keepdim=True)).to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    @torch.inference_mode()
    def verbalize(self, jobs: list[tuple[torch.Tensor, int]], batch: int = 64, max_new_tokens: int = 256) -> list[str]:
        """Temperature-1 explanation per (vector, seed). Sampling is seeded per batch by its first job."""
        out = []
        for i in range(0, len(jobs), batch):
            chunk = jobs[i:i + batch]
            self.vecs = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v, _ in chunk])
            x = torch.tensor([self.ids] * len(chunk), device=self.device)
            torch.manual_seed(int(chunk[0][1])); torch.cuda.manual_seed_all(int(chunk[0][1]))
            try:
                g = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True,
                                        temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=max_new_tokens,
                                        pad_token_id=self.tok.pad_token_id, disable_compile=True)
            finally:
                self.vecs = None
            for row in g[:, len(self.ids):]:
                m = EXPLANATION_RE.search(self.tok.decode(row, skip_special_tokens=True))
                out.append(m.group(1).strip() if m else "")
        return out


class EasyAR:
    def __init__(self, base: str | Path, ckpt: str, meta: dict, device: str = "cuda:0"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(base)
        d = resolve(ckpt)
        m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map={"": device})
        assert m.config.num_hidden_layers == meta["extraction"]["layer_index"] + 1
        m.model.norm = nn.Identity()
        self.backbone = m.model.eval()
        self.value_head = nn.Linear(m.config.hidden_size, m.config.hidden_size, bias=False).to(device)
        self.value_head.load_state_dict(load_file(str(d / "value_head.safetensors")))
        self.template = meta["prompt_templates"]["critic"]
        self.scale = math.sqrt(m.config.hidden_size)

    @torch.inference_mode()
    def reconstruct(self, texts: list[str], batch: int = 16) -> list[torch.Tensor]:
        pad, out = self.tok.pad_token_id or 0, []
        for i in range(0, len(texts), batch):
            ids = [self.tok.encode(self.template.format(explanation=t), add_special_tokens=False)
                   for t in texts[i:i + batch]]
            L = max(len(x) for x in ids)
            inp = torch.tensor([x + [pad] * (L - len(x)) for x in ids], device=self.device)
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids], device=self.device)
            h = self.backbone(input_ids=inp, attention_mask=att, use_cache=False).last_hidden_state
            last = h[torch.arange(len(ids), device=self.device), att.sum(1) - 1].float()
            last = last / last.norm(dim=-1, keepdim=True) * self.scale
            out += [v.cpu() for v in self.value_head(last)]
        return out
