"""NLA round-trip eval on the capitalize task only: baseline, nla_roundtrip,
paraphrase (medium/heavy), mismatch (same-task/cross-task), random_vec.
Produces a single bar chart with error bars, matching the reference figure.

Usage:
    python capitalize_L40/nla_roundtrip.py --nla-dir ./nla-L40/rl_vllm \
        --data-dir notebooks/datasets --n 50
"""

import argparse
import json
import random
import re
import string
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.injection import marker_well_formed
from nla.models import NLACriticModel
from nla.schema import EXPLANATION_RE, normalize_activation, resolve_target_scale
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers, resolve_text_model

PARAPHRASE_PROMPT_MEDIUM = (
    "Please paraphrase the text provided below, preserving its exact meaning "
    "but expressing it in substantially different words and sentence structure "
    "than the original. Do not simply swap individual words for synonyms -- "
    "restructure the sentence(s) genuinely.\n\n"
    "<text>\n{text}\n</text>\n\n"
    "Include your final paraphrased text in <transformed_text> tags."
)
PARAPHRASE_PROMPT_HEAVY = (
    "Please paraphrase aggressively. Preserve meaning and uncertainty, but "
    "substantially change wording, syntax, and organization. Do not simply "
    "swap individual words for synonyms -- restructure the sentence(s) "
    "genuinely.\n\n"
    "Text:\n{text}\n\n"
    "Respond with only the paraphrased text, wrapped exactly like this:\n"
    "<transformed_text>your paraphrase here</transformed_text>"
)

PRIMARY_TASK = "capitalize"
# Other tasks in the six-task battery, used as the cross-task mismatch pool.
OTHER_TASKS = ["antonym", "country-capital", "english-french",
               "present-past", "singular-plural"]


class LastTokenPatcher:
    def __init__(self, patch_vec=None):
        self.patch_vec = patch_vec
        self.captured = None
        self._done = False

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > 1 and not self._done:
            self.captured = hidden[0, -1, :].detach().clone()
            if self.patch_vec is not None:
                hidden = hidden.clone()
                hidden[0, -1, :] = self.patch_vec.to(hidden.dtype).to(hidden.device)
            self._done = True
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        return output


def load_json_pairs(path):
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "examples" in raw:
        raw = raw["examples"]
    return [{"input": ex["input"], "output": ex["output"]} for ex in raw]


def load_antonym(data_dir, pos_tags=("verb", "noun", "adjective"),
                  splits=("train", "val", "test")):
    seen = set()
    pairs = []
    for pos in pos_tags:
        for split in splits:
            path = data_dir / f"{pos}-pairs.{split}"
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3 or parts[2] != "1":
                        continue
                    key = (parts[0], parts[1])
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append({"input": parts[0], "output": parts[1]})
    return pairs


def load_english_french(paths):
    seen = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                seen.setdefault(parts[0], parts[1])
    return [{"input": en, "output": fr} for en, fr in seen.items()]


def load_task(task_name, data_dir):
    if task_name == "antonym":
        return load_antonym(data_dir)
    if task_name == "english-french":
        return load_english_french(
            [data_dir / "en-fr.0-5000.txt", data_dir / "en-fr.5000-6500.txt"])
    return load_json_pairs(data_dir / f"{task_name}.json")


def build_icl_prompt(pairs, query_idx, n_shots, rng, extra_exclude_words=None):
    query_word = pairs[query_idx]["input"]
    excluded = {query_word} | (extra_exclude_words or set())
    pool = [i for i in range(len(pairs)) if i != query_idx and pairs[i]["input"] not in excluded]
    shot_idxs = rng.sample(pool, min(n_shots, len(pool)))
    lines = [f"Q: {pairs[i]['input']}\nA: {pairs[i]['output']}" for i in shot_idxs]
    lines.append(f"Q: {query_word}\nA:")
    instruction = "Answer with a single word only. No punctuation, no explanation."
    prompt = instruction + "\n\n" + "\n\n".join(lines)
    return prompt, pairs[query_idx]["output"]


def build_foreign_prompt(pairs, n_shots, rng):
    idxs = rng.sample(range(len(pairs)), min(n_shots + 1, len(pairs)))
    query_idx, shot_idxs = idxs[0], idxs[1:]
    lines = [f"Q: {pairs[i]['input']}\nA: {pairs[i]['output']}" for i in shot_idxs]
    lines.append(f"Q: {pairs[query_idx]['input']}\nA:")
    instruction = "Answer with a single word only. No punctuation, no explanation."
    return instruction + "\n\n" + "\n\n".join(lines)


def extract_answer(text):
    first_line = text.strip().split("\n")[0]
    return first_line.strip(string.punctuation + " ")


def is_correct(pred, gold, case_sensitive):
    if not case_sensitive:
        pred, gold = pred.lower(), gold.lower()
    return pred == gold


def load_ar_critic(ar_dir, device):
    ar_dir = Path(ar_dir)
    assert (ar_dir / "config.json").exists(), (
        f"{ar_dir} has no config.json — point at rl_vllm/critic_latest or merged/ar_hf."
    )
    critic = NLACriticModel.from_pretrained(
        str(ar_dir), torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    return critic.eval()


def se(k, n):
    p = k / n
    return (p * (1 - p) / n) ** 0.5


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--nla-dir", required=True)
    p.add_argument("--data-dir", required=True, type=Path,
                    help="Dir containing capitalize.json + the other 5 task files.")
    p.add_argument("--base-ckpt", default="google/gemma-3-12b-it")
    p.add_argument("--av-lora", default=None)
    p.add_argument("--ar-ckpt", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ar-device", default=None)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--n-shots", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out", default="capitalize_roundtrip.png")
    p.add_argument("--results-json", default=None,
                    help="If set, dump raw per-arm correct-counts + n here for merging shards.")
    args = p.parse_args()

    nla_dir = Path(args.nla_dir)
    DEVICE, ar_dev = args.device, args.ar_device or args.device
    rng = random.Random(args.seed)

    pairs = load_task(PRIMARY_TASK, args.data_dir)
    other_pairs = {t: load_task(t, args.data_dir) for t in OTHER_TASKS}
    case_sensitive = True  # capitalize needs exact-case matching

    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(str(nla_dir), tokenizer)
    assert cfg.injection_scale is None, "these checkpoints inject RAW activations"
    MSE_SCALE = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    LAYER_INDEX = cfg.extraction_layer_index

    base = resolve_text_model(AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"))
    model = PeftModel.from_pretrained(base.to(DEVICE).eval(),
                                      str(args.av_lora or nla_dir / "iter_000400")).eval()
    layers = resolve_decoder_layers(model.get_base_model())
    vref = [None]
    register_karvonen_hook(model, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id,
                           layer_idx=1)
    critic = load_ar_critic(args.ar_ckpt or nla_dir / "critic_latest", ar_dev)

    def av(h):
        ptxt = build_prompt_text(
            [{"role": "user", "content": cfg.actor_prompt_template.format(injection_char="<INJECT>")}],
            cfg.injection_char, tokenizer)
        ids = tokenizer.encode(ptxt, add_special_tokens=False)
        assert marker_well_formed(ids, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                                  cfg.injection_right_neighbor_id), "marker/template drift"
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        vref[0] = h[None].float().to(DEVICE)
        try:
            with torch.no_grad():
                out = model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                                     max_new_tokens=384, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
        finally:
            vref[0] = None
        m = EXPLANATION_RE.search(tokenizer.decode(out[0, x.shape[1]:], skip_special_tokens=True))
        return m.group(1).strip() if m else None

    def ar(text, norm):
        ids = tokenizer.encode(cfg.critic_prompt_template.format(explanation=text),
                               add_special_tokens=False)
        with torch.no_grad():
            v = critic_predict(critic, torch.tensor([ids], dtype=torch.long, device=ar_dev),
                               None, MSE_SCALE)[0].float().cpu()
        return v if MSE_SCALE is None else normalize_activation(v[None], norm)[0]

    def run_with_patch(prompt, patch_vec, max_new_tokens):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        patcher = LastTokenPatcher(patch_vec)
        handle = layers[LAYER_INDEX].register_forward_hook(patcher)
        try:
            with torch.no_grad(), model.disable_adapter():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
        finally:
            handle.remove()
        assert patcher._done, f"hook on block {LAYER_INDEX} never fired"
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text, patcher.captured

    def paraphrase(text, prompt_template, max_new_tokens=512):
        prompt = prompt_template.format(text=text)
        ptxt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(ptxt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        with torch.no_grad(), model.disable_adapter():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        match = re.search(r"<transformed_text>(.*?)</transformed_text>", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        print(f"[paraphrase] WARNING: no <transformed_text> tags. Raw[:200]={raw[:200]!r}")
        return raw

    ARMS = ["baseline", "nla_roundtrip", "paraphrase_medium", "paraphrase_heavy",
            "mismatch_same_task", "mismatch_cross_task", "random_vec"]
    correct = {a: 0 for a in ARMS}

    n = min(args.n, len(pairs))
    query_idxs = rng.sample(range(len(pairs)), n)

    for i, qi in enumerate(query_idxs):
        prompt, gold = build_icl_prompt(pairs, qi, args.n_shots, rng)
        text_baseline, h_orig = run_with_patch(prompt, None, args.max_new_tokens)
        pred_baseline = extract_answer(text_baseline)
        correct["baseline"] += is_correct(pred_baseline, gold, case_sensitive)

        explanation = av(h_orig)
        assert explanation, "no <explanation> extracted — check the marker injection"
        norm = float(h_orig.norm())

        h_hat = ar(explanation, norm)
        text_nla, _ = run_with_patch(prompt, h_hat, args.max_new_tokens)
        correct["nla_roundtrip"] += is_correct(extract_answer(text_nla), gold, case_sensitive)

        paraphrased_medium = paraphrase(explanation, PARAPHRASE_PROMPT_MEDIUM)
        h_hat_med = ar(paraphrased_medium, norm)
        text_med, _ = run_with_patch(prompt, h_hat_med, args.max_new_tokens)
        correct["paraphrase_medium"] += is_correct(extract_answer(text_med), gold, case_sensitive)

        paraphrased_heavy = paraphrase(explanation, PARAPHRASE_PROMPT_HEAVY)
        h_hat_heavy = ar(paraphrased_heavy, norm)
        text_heavy, _ = run_with_patch(prompt, h_hat_heavy, args.max_new_tokens)
        correct["paraphrase_heavy"] += is_correct(extract_answer(text_heavy), gold, case_sensitive)

        qi_word = pairs[qi]["input"]
        same_pool = [j for j in range(len(pairs)) if j != qi and pairs[j]["input"] != qi_word]
        same_qi = rng.choice(same_pool)
        same_prompt, _ = build_icl_prompt(pairs, same_qi, args.n_shots, rng,
                                          extra_exclude_words={qi_word})
        _, h_mismatch_same = run_with_patch(same_prompt, None, args.max_new_tokens)
        text_same, _ = run_with_patch(prompt, h_mismatch_same.float().cpu(), args.max_new_tokens)
        correct["mismatch_same_task"] += is_correct(extract_answer(text_same), gold, case_sensitive)

        foreign_task = rng.choice(OTHER_TASKS)
        foreign_prompt = build_foreign_prompt(other_pairs[foreign_task], args.n_shots, rng)
        _, h_mismatch_cross = run_with_patch(foreign_prompt, None, args.max_new_tokens)
        text_cross, _ = run_with_patch(prompt, h_mismatch_cross.float().cpu(), args.max_new_tokens)
        correct["mismatch_cross_task"] += is_correct(extract_answer(text_cross), gold, case_sensitive)

        noise = torch.randn(h_orig.shape, generator=torch.Generator().manual_seed(i),
                            dtype=torch.float32)
        h_random = normalize_activation(noise[None], norm)[0]
        text_random, _ = run_with_patch(prompt, h_random, args.max_new_tokens)
        correct["random_vec"] += is_correct(extract_answer(text_random), gold, case_sensitive)

        print(f"[{i + 1}/{n}] gold={gold!r} explanation={explanation[:80]!r}")

    print(f"\n{'arm':<20}{'accuracy':>10}{'se':>8}")
    for arm in ARMS:
        acc = correct[arm] / n
        print(f"{arm:<20}{acc:>10.2f}{se(correct[arm], n):>8.3f}")

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump({"n": n, "correct": correct}, f, indent=2)
        print(f"Saved raw counts -> {args.results_json}")

    # --- Plot, matching the reference figure ---
    labels = ["Baseline (no intervention)", "NLA round-trip",
              "Paraphrase round-trip (medium)", "Paraphrase round-trip (heavy)",
              "Mismatched FV (same-task)", "Mismatched FV (cross-task)",
              "Random vector"]
    colors = ["gray", "C0", "C1", "C1", "C4", "C4", "C4"]
    accs = [correct[a] / n for a in ARMS]
    errs = [se(correct[a], n) for a in ARMS]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, accs, yerr=errs, color=colors, edgecolor="black",
          capsize=4, error_kw={"elinewidth": 1})
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(PRIMARY_TASK)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved plot -> {args.out}")


if __name__ == "__main__":
    main()