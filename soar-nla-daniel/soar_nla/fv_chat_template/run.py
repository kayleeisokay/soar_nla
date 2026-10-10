"""2-shot English->French function-vector sweep, paired raw-prompt vs chat-template.

Uses the exact prompt/dataset/RNG plan from ``prepare`` and the interventions
of the reference executor (see ``protocol``). Each of the 50 queries is run
twice: once as the raw prompt (BOS + prompt, as in the reference), once
wrapped in the Gemma chat template. In both styles a hook on decoder layer 32
captures the residual at the final prefill position (raw: the ``:`` of the
trailing ``A:``; chat: the ``\\n`` after ``<start_of_turn>model``) and, in the
intervention arms, overwrites it there:

* ``identity``: the captured native vector (hook-fidelity check);
* ``nla``: AR(AV(h)), the raw AR output (not norm-matched, as in the reference);
* ``random_vec``: seeded Gaussian rescaled to the native norm;
* ``mismatch_cross_task`` / ``mismatch_same_task``: the native vector of a
  donor prompt from another task / the same task with a different query;
* ``para_medium`` / ``para_heavy``: AR of a paraphrase of the AV explanation.

``--phase baseline`` runs the unpatched generations and saves the native
vectors (one GPU); ``--phase full`` runs every intervention arm (target and
float32 AR on two GPUs, the AV served by SGLang with tensor parallelism on two
more). Results are checkpointed per row in ``rows.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from soar_nla.fv_chat_template.prepare import DEFAULT_MANIFEST
from soar_nla.fv_chat_template.protocol import (
    PARAPHRASE_PROMPTS,
    REFERENCE_COMMIT,
    correct,
    extract_word_answer,
)
from soar_nla.paths import results_dir

# soar_nla.hub imports torch, so it is imported inside functions: main() must
# set CUDA_VISIBLE_DEVICES before torch is first imported.

LAYER = 32
MAX_NEW = 256
AV_MAX_NEW = 384
PARA_MAX_NEW = 512
ARMS = (
    "baseline", "identity", "nla", "random_vec", "mismatch_cross_task",
    "mismatch_same_task", "para_medium", "para_heavy",
)


def default_out_dir(shots: int) -> Path:
    return results_dir("fv_chat_template") / f"shot{shots}"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def backbone(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    language_model = getattr(model, "language_model", model)
    return language_model.model if hasattr(language_model, "model") else language_model


class LastTokenPatcher:
    """Capture (and optionally overwrite) the final prefill position at one layer."""

    def __init__(self, patch=None):
        self.patch = patch
        self.capture = None
        self.done = False

    def __call__(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > 1 and not self.done:
            self.capture = hidden[0, -1, :].detach().float().cpu()
            self.done = True
            if self.patch is not None:
                hidden = hidden.clone()
                hidden[0, -1, :] = self.patch.to(hidden.device, hidden.dtype)
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return output


def serialized_prompt(tokenizer, raw_prompt, style):
    if style == "raw":
        return raw_prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_generation(model, tokenizer, layer, raw_prompt, style, patch=None, max_new=MAX_NEW):
    import torch

    prompt = serialized_prompt(tokenizer, raw_prompt, style)
    # The chat template already starts with <bos>; the raw prompt gets it here.
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=(style == "raw")
    ).to("cuda:0")
    hook = LastTokenPatcher(patch)
    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                disable_compile=True,
            )
    finally:
        handle.remove()
    if hook.capture is None:
        raise RuntimeError("last-token capture hook did not fire")
    prefix_length = inputs.input_ids.shape[1]
    return tokenizer.decode(generated[0][prefix_length:], skip_special_tokens=True), hook.capture


def healthy(url):
    try:
        urllib.request.urlopen(url + "/health", timeout=2).read()
        return True
    except Exception:
        return False


def launch_av(actor_dir, args):
    """Start the AV SGLang server (tensor parallel over ``--av-gpus``)."""
    from soar_nla.sglang_server import SGLANG_PYTHON

    url = f"http://127.0.0.1:{args.av_port}"
    if healthy(url):
        raise RuntimeError(f"refusing to reuse unknown server at {url}")
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.av_gpus
    env["SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR"] = "1"
    tp_size = len(args.av_gpus.split(","))
    command = [
        SGLANG_PYTHON, "-m", "sglang.launch_server",
        "--model-path", str(actor_dir), "--port", str(args.av_port),
        "--host", "127.0.0.1", "--tp-size", str(tp_size),
        "--disable-radix-cache", "--mem-fraction-static", "0.80",
        "--context-length", "512", "--attention-backend", "triton",
        "--disable-cuda-graph", "--trust-remote-code", "--dtype", "bfloat16",
        "--log-level", "warning",
    ]
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "av_sglang.log"
    log = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 1200
    while time.time() < deadline:
        if healthy(url):
            print(f"AV server healthy on GPUs {args.av_gpus}", flush=True)
            return process, url
        if process.poll() is not None:
            raise RuntimeError(f"AV server exited {process.returncode}; see {log_path}")
        time.sleep(5)
    raise TimeoutError("AV server did not become healthy")


def stop_av(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=90)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soar_nla.hub import GEMMA3_12B, resolve_snapshot

    model_dir = resolve_snapshot(*GEMMA3_12B)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        trust_remote_code=True, attn_implementation="eager",
        local_files_only=True,
    ).eval()
    if hasattr(model.model, "vision_tower"):
        del model.model.vision_tower
    return model, tokenizer, backbone(model).layers[LAYER]


def report(rows, shots):
    from soar_nla.hub import GEMMA3_12B

    result = {
        "source_commit": REFERENCE_COMMIT,
        "task": "english-french", "n_shots": shots, "n": len(rows),
        "target_model": GEMMA3_12B[0], "target_revision": GEMMA3_12B[1], "layer": LAYER,
        "max_new_tokens": MAX_NEW,
        "style_difference": "raw tokenizer prompt vs Gemma chat template around same prompt",
        "conditions": {},
        "paired_chat_vs_raw": {},
    }
    for style in ("raw", "chat"):
        for arm in ARMS:
            key = f"{style}_{arm}"
            done = [row["arms"][key]["correct"] for row in rows if key in row["arms"]]
            result["conditions"][key] = {
                "completed": len(done), "correct": sum(done),
                "accuracy": sum(done) / len(done) if done else None,
            }
    for arm in ARMS:
        pairs = [
            (row["arms"][f"raw_{arm}"]["correct"], row["arms"][f"chat_{arm}"]["correct"])
            for row in rows
            if f"raw_{arm}" in row["arms"] and f"chat_{arm}" in row["arms"]
        ]
        raw_only = sum(raw and not chat for raw, chat in pairs)
        chat_only = sum(chat and not raw for raw, chat in pairs)
        discordant = raw_only + chat_only
        p_value = min(
            1.0,
            2 * sum(math.comb(discordant, k) for k in range(min(raw_only, chat_only) + 1))
            / 2**discordant,
        ) if discordant else 1.0
        result["paired_chat_vs_raw"][arm] = {
            "completed": len(pairs),
            "both_correct": sum(raw and chat for raw, chat in pairs),
            "raw_only_correct": raw_only,
            "chat_only_correct": chat_only,
            "both_incorrect": sum(not raw and not chat for raw, chat in pairs),
            "delta_accuracy_points": 100 * (chat_only - raw_only) / len(pairs) if pairs else None,
            "two_sided_exact_mcnemar_p": p_value if pairs else None,
        }
    return result


def store_progress(out_dir, rows, shots):
    atomic_json(out_dir / "rows.json", rows)
    atomic_json(out_dir / "summary.json", report(rows, shots))


def score_entry(output, gold):
    return {"output": output, "prediction": extract_word_answer(output), "correct": correct(output, gold)}


def baseline(args, plan_rows, rows):
    import torch
    from safetensors.torch import save_file

    model, tokenizer, layer = load_model()
    vectors = {style: [] for style in ("raw", "chat")}
    for style in ("raw", "chat"):
        for position, plan in enumerate(plan_rows):
            output, captured = run_generation(model, tokenizer, layer, plan["prompt_raw"], style)
            vectors[style].append(captured)
            rows[position]["arms"][f"{style}_baseline"] = score_entry(output, plan["gold"])
            if (position + 1) % 10 == 0:
                print(f"baseline {style}: {position + 1}/{len(plan_rows)}", flush=True)
        store_progress(args.out_dir, rows, args.shots)
    destination = args.out_dir / "native.safetensors"
    temporary = destination.with_suffix(".tmp.safetensors")
    save_file({style: torch.stack(values).contiguous() for style, values in vectors.items()}, str(temporary))
    os.replace(temporary, destination)
    store_progress(args.out_dir, rows, args.shots)
    print(json.dumps(report(rows, args.shots), indent=2), flush=True)


def paraphrase(model, tokenizer, layer, explanation, arm):
    prompt = PARAPHRASE_PROMPTS[arm].format(text=explanation)
    output, _ = run_generation(model, tokenizer, layer, prompt, "raw", max_new=PARA_MAX_NEW)
    match = re.search(r"<transformed_text>(.*?)</transformed_text>", output, re.DOTALL)
    transformed = match.group(1).strip() if match else output.strip()
    original_words = set(explanation.lower().split())
    transformed_words = set(transformed.lower().split())
    overlap = len(original_words & transformed_words) / len(original_words) if original_words else 0.0
    return transformed, overlap, bool(match)


def full(args, plan_rows, rows):
    import torch
    from safetensors.torch import load_file

    from soar_nla.hub import KITFT_L32_AR, KITFT_L32_AV, resolve_snapshot
    from soar_nla.nla_inference import NLAClient, NLACritic

    native = load_file(str(args.out_dir / "native.safetensors"))
    model, tokenizer, layer = load_model()
    ar_dir = resolve_snapshot(*KITFT_L32_AR)
    print("loading float32 AR on the second target-side GPU", flush=True)
    critic = NLACritic(ar_dir, device="cuda:1", dtype=torch.float32)
    actor_dir = resolve_snapshot(*KITFT_L32_AV)
    av_process = None
    client = None
    try:
        av_process, av_url = launch_av(actor_dir, args)
        client = NLAClient(actor_dir, sglang_url=av_url)
        for style in ("raw", "chat"):
            for position, plan in enumerate(plan_rows):
                row = rows[position]
                arms = row["arms"]
                gold = plan["gold"]
                prompt = plan["prompt_raw"]
                original = native[style][position]
                prefix = f"{style}_"
                if prefix + "identity" not in arms:
                    output, _ = run_generation(model, tokenizer, layer, prompt, style, patch=original)
                    arms[prefix + "identity"] = score_entry(output, gold)
                if prefix + "nla" not in arms:
                    explanation = client.generate(
                        original.numpy(), temperature=0.0,
                        max_new_tokens=AV_MAX_NEW, extract_explanation=True,
                    )
                    row[f"{style}_explanation"] = explanation
                    reconstructed = critic.reconstruct(explanation)
                    output, _ = run_generation(model, tokenizer, layer, prompt, style, patch=reconstructed)
                    arms[prefix + "nla"] = score_entry(output, gold)
                    arms[prefix + "nla"]["reconstruction_cosine"] = float(
                        torch.nn.functional.cosine_similarity(original, reconstructed, dim=0)
                    )
                if prefix + "random_vec" not in arms:
                    reference = original.to("cuda:0", dtype=torch.bfloat16)
                    generator = torch.Generator(device="cuda:0").manual_seed(plan["random_seed"])
                    noise = torch.randn(
                        reference.shape, generator=generator,
                        device="cuda:0", dtype=reference.dtype,
                    )
                    random_vector = noise * (reference.norm() / (noise.norm() + 1e-8))
                    output, _ = run_generation(model, tokenizer, layer, prompt, style, patch=random_vector)
                    arms[prefix + "random_vec"] = score_entry(output, gold)
                for arm, donor_field in (
                    ("mismatch_cross_task", "foreign_prompt_raw"),
                    ("mismatch_same_task", "same_task_prompt_raw"),
                ):
                    if prefix + arm not in arms:
                        # The donor's prefill residual is independent of the
                        # rest of its generation; one token suffices to capture it.
                        _, donor = run_generation(
                            model, tokenizer, layer, plan[donor_field], style, max_new=1
                        )
                        output, _ = run_generation(model, tokenizer, layer, prompt, style, patch=donor)
                        arms[prefix + arm] = score_entry(output, gold)
                for arm in ("para_medium", "para_heavy"):
                    if prefix + arm not in arms:
                        transformed, overlap, tagged = paraphrase(
                            model, tokenizer, layer, row[f"{style}_explanation"], arm
                        )
                        reconstructed = critic.reconstruct(transformed)
                        output, _ = run_generation(model, tokenizer, layer, prompt, style, patch=reconstructed)
                        arms[prefix + arm] = {
                            **score_entry(output, gold),
                            "paraphrase": transformed, "word_overlap": overlap,
                            "transformed_text_tags_found": tagged,
                        }
                store_progress(args.out_dir, rows, args.shots)
                print(f"full {style}: {position + 1}/{len(plan_rows)}", flush=True)
    finally:
        if client is not None and hasattr(client, "_http"):
            client._http.close()
        stop_av(av_process)
    print(json.dumps(report(rows, args.shots), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("baseline", "full"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: results/fv_chat_template/shot<N>")
    parser.add_argument("--baseline-gpu", default="4",
                        help="physical GPU for the baseline phase")
    parser.add_argument("--target-gpus", default="4,5",
                        help="physical GPUs for the full phase: target model, then float32 AR")
    parser.add_argument("--av-gpus", default="6,7",
                        help="physical GPUs for the AV SGLang server (tensor parallel over all of them)")
    parser.add_argument("--av-port", type=int, default=32367)
    args = parser.parse_args()
    args.out_dir = args.out_dir or default_out_dir(args.shots)
    # Set before torch is first imported (all torch imports are deferred).
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.baseline_gpu if args.phase == "baseline" else args.target_gpus
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    plan_rows = manifest["configs"][f"english-french:{args.shots}"]
    path = args.out_dir / "rows.json"
    if path.exists():
        rows = json.loads(path.read_text())
        if len(rows) != len(plan_rows) or any(
            row["query_index"] != plan["query_index"]
            for row, plan in zip(rows, plan_rows)
        ):
            raise RuntimeError("cached rows do not match sampling manifest")
    else:
        rows = [
            {"position": plan["position"], "query_index": plan["query_index"],
             "query_input": plan["query_input"], "gold": plan["gold"], "arms": {}}
            for plan in plan_rows
        ]
    if args.phase == "baseline":
        baseline(args, plan_rows, rows)
    else:
        full(args, plan_rows, rows)


if __name__ == "__main__":
    main()
