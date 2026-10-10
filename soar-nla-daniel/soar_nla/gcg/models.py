"""Differentiable AR/AV wrappers for GCG and the FVE normalisation they share.

Per-example FVE is `1 - MSE(normalize(pred), normalize(gold)) / Var(v)`, where
both vectors are rescaled to the AR's `mse_scale` (so `MSE = 2(1 - cos)`) and
`Var(v)` is the empirical scalar variance of the 2,000 gold Alpaca activations
after the same rescaling. GCG's `ar_loss` is exactly `MSE / Var(v)`, so
`FVE = 1 - ar_loss`.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from soar_nla.hub import ModelSpec, resolve_snapshot
from soar_nla.nla_inference import NLACritic, load_nla_config, normalize_activation


def normalise_rows(value: torch.Tensor, scale: float) -> torch.Tensor:
    return value.float() * (scale / value.float().norm(dim=-1, keepdim=True).clamp_min(1e-12))


def empirical_directional_variance(activations: torch.Tensor, scale: float) -> float:
    normalised = normalise_rows(activations, scale)
    mean = normalised.mean(dim=0)
    return float(((normalised - mean) ** 2).mean().item())


class DifferentiableAR:
    """Differentiable token-splice view of :class:`soar_nla.nla_inference.NLACritic`."""

    def __init__(self, ar_dir: Path, device: str):
        self.critic = NLACritic(ar_dir, device=device, dtype=torch.bfloat16)
        self.device = torch.device(device)
        self.tokenizer = self.critic.tokenizer
        self.scale = self.critic.mse_scale
        prefix_text, suffix_text = self.critic.template.split("{explanation}")
        self.prefix_ids = self.tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
        self.suffix_ids = self.tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
        self.embed_tokens = self.critic.backbone.get_input_embeddings()
        self.embed_weight = self.embed_tokens.weight
        embed_scale = getattr(self.embed_tokens, "embed_scale", 1.0)
        self.embed_scale = float(embed_scale.item() if torch.is_tensor(embed_scale) else embed_scale)

    def _fixed_embeds(self, ids: list[int]) -> torch.Tensor:
        token_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        return self.embed_tokens(token_ids).detach()

    def gradient(self, x_ids: torch.Tensor, gold: torch.Tensor, variance: float) -> tuple[float, torch.Tensor]:
        x_ids = x_ids.to(self.device)
        x_embeds = self.embed_tokens(x_ids.unsqueeze(0)).detach().requires_grad_(True)
        embeds = torch.cat((self._fixed_embeds(self.prefix_ids), x_embeds, self._fixed_embeds(self.suffix_ids)), dim=1)
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device)
        hidden = self.critic.backbone.model(
            inputs_embeds=embeds, attention_mask=mask, use_cache=False
        ).last_hidden_state
        pred = self.critic.value_head(hidden[:, -1]).float()
        pred_n = normalise_rows(pred, self.scale)
        gold_n = normalise_rows(gold.to(self.device).view(1, -1), self.scale)
        loss = ((pred_n - gold_n) ** 2).mean() / variance
        loss.backward()
        gradient = x_embeds.grad.detach()[0]
        value = float(loss.detach().item())
        self.critic.backbone.zero_grad(set_to_none=True)
        self.critic.value_head.zero_grad(set_to_none=True)
        return value, gradient

    @torch.inference_mode()
    def losses(self, candidates: torch.Tensor, gold: torch.Tensor, variance: float, batch_size: int) -> torch.Tensor:
        """Exact string-path losses, including tokenizer merges at template boundaries."""
        output = []
        gold_n = normalise_rows(gold.to(self.device).view(1, -1), self.scale)
        for start in range(0, len(candidates), batch_size):
            x = candidates[start : start + batch_size]
            texts = self.tokenizer.batch_decode(
                x.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts = [self.critic.template.format(explanation=text) for text in texts]
            self.tokenizer.padding_side = "right"
            encoded = self.tokenizer(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
            ids = encoded["input_ids"].to(self.device)
            mask = encoded["attention_mask"].to(self.device)
            hidden = self.critic.backbone.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            last = mask.sum(dim=1) - 1
            pred = self.critic.value_head(hidden[torch.arange(len(ids), device=self.device), last]).float()
            output.append((((normalise_rows(pred, self.scale) - gold_n) ** 2).mean(dim=-1) / variance).cpu())
        return torch.cat(output)


class DifferentiableAV:
    """Teacher-forced AV NLL with the activation injected exactly as in NLAClient."""

    def __init__(self, av_dir: Path, device: str):
        self.dir = av_dir
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(av_dir, local_files_only=True, trust_remote_code=True)
        self.cfg = load_nla_config(av_dir, self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            av_dir,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).eval()
        self.model.requires_grad_(False)
        self.embed_tokens = self.model.get_input_embeddings()
        self.embed_weight = self.embed_tokens.weight
        embed_scale = getattr(self.embed_tokens, "embed_scale", 1.0)
        self.embed_scale = float(embed_scale.item() if torch.is_tensor(embed_scale) else embed_scale)
        # Wrapper is fixed; GCG changes every token in the explanation payload.
        self.wrapper_prefix_ids = self.tokenizer("<explanation>\n", add_special_tokens=False)["input_ids"]

    def injected_prompt_embeds(self, activation: torch.Tensor) -> torch.Tensor:
        content = self.cfg.actor_prompt_template.format(injection_char=self.cfg.injection_char)
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True
        )
        ids_t = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        embeds = self.embed_tokens(ids_t).detach().clone()
        matches = [
            pos
            for pos, token in enumerate(ids)
            if token == self.cfg.injection_token_id
            and 0 < pos < len(ids) - 1
            and ids[pos - 1] == self.cfg.injection_left_neighbor_id
            and ids[pos + 1] == self.cfg.injection_right_neighbor_id
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one AV injection site, got {matches}")
        vector = normalize_activation(activation.float().view(1, -1).to(self.device), self.cfg.injection_scale).to(
            embeds.dtype
        )
        embeds[0, matches[0]] = vector[0]
        return embeds

    def gradient(self, x_ids: torch.Tensor, activation: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
        x_ids = x_ids.to(self.device)
        prompt = self.injected_prompt_embeds(activation)
        wrapper_ids = torch.tensor(self.wrapper_prefix_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        wrapper = self.embed_tokens(wrapper_ids).detach()
        x_embeds = self.embed_tokens(x_ids.unsqueeze(0)).detach().requires_grad_(True)
        embeds = torch.cat((prompt, wrapper, x_embeds), dim=1)
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device)
        logits = self.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).logits.float()
        start = prompt.shape[1] + wrapper.shape[1]
        token_logits = logits[0, start - 1 : start + len(x_ids) - 1]
        log_probs = F.log_softmax(token_logits, dim=-1)
        nll = -log_probs.gather(1, x_ids[:, None]).mean()
        nll.backward()
        gradient = x_embeds.grad.detach()[0]
        value = float(nll.detach().item())
        self.model.zero_grad(set_to_none=True)
        return value, gradient, log_probs.detach()

    @torch.inference_mode()
    def nlls(self, candidates: torch.Tensor, activation: torch.Tensor, batch_size: int) -> torch.Tensor:
        output = []
        prompt_one = self.injected_prompt_embeds(activation)
        wrapper_ids = torch.tensor(self.wrapper_prefix_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        wrapper_one = self.embed_tokens(wrapper_ids).detach()
        for start_index in range(0, len(candidates), batch_size):
            x = candidates[start_index : start_index + batch_size].to(self.device)
            prompt = prompt_one.expand(len(x), -1, -1)
            wrapper = wrapper_one.expand(len(x), -1, -1)
            x_embeds = self.embed_tokens(x)
            embeds = torch.cat((prompt, wrapper, x_embeds), dim=1)
            mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device)
            logits = self.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).logits.float()
            first = prompt.shape[1] + wrapper.shape[1] - 1
            token_logits = logits[:, first : first + x.shape[1]]
            log_probs = F.log_softmax(token_logits, dim=-1)
            output.append((-log_probs.gather(2, x.unsqueeze(-1)).squeeze(-1).mean(dim=-1)).cpu())
        return torch.cat(output)


def load_ar(spec: ModelSpec, device: str) -> tuple[DifferentiableAR, Path]:
    path = resolve_snapshot(spec.ar_id, spec.ar_revision)
    return DifferentiableAR(path, device), path


def load_av(spec: ModelSpec, device: str) -> tuple[DifferentiableAV, Path]:
    path = resolve_snapshot(spec.av_id, spec.av_revision)
    return DifferentiableAV(path, device), path
