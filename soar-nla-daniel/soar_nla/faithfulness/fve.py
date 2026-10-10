"""Warm-start FVE: AV round-trip vs Sonnet explanations through the custom L32 NLA.

Compares two ways of producing the explanation the AR reads back:
  av    — round-trip: gold activation -> AV -> text -> AR
  gold  — Sonnet's explanation of the same context (warm-start recipe) -> AR

Sonnet wrote its explanation with the source text in front of it, so it is the more faithful
description by construction. Whether that buys FVE is the question: the AR was trained to read
AV-shaped text, not oracle text. Run once per checkpoint pair (`--spec sft`, `--spec rl`).

The custom NLA (soar_nla.hub.CUSTOM_L32) is EasyNLA-format, which differs from the official
release in three ways that silently produce wrong numbers if mixed up:

  injection  Karvonen norm-matched ADD in the residual — h'_p = h_p + ||h_p||·v/||v||
             — hooked on the OUTPUT of layers[1], with RAW vectors. NOT an
             embedding overwrite, and there is no `injection_scale`.
  AR forward `critic_predict` normalizes the backbone's last hidden state to
             mse_scale BEFORE the value head.
  AR tokens  the critic prompt is encoded with add_special_tokens=False (no BOS).

mse_scale is absent from the sidecar, so it resolves to sqrt(d_model) (= 61.9677) via
EasyNLA's SCALE_SQRT_D sentinel. Stages `decode` and `reconstruct` need the EasyNLA package
(`nla`, vendored under training/).

Stages, in order: dataset, extract, decode (sharded) + merge-decode, reconstruct, analyze.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from soar_nla.hub import resolve_snapshot
from soar_nla.io_utils import read_jsonl, seeded_rng, stable_seed, write_json, write_jsonl
from soar_nla.paths import results_dir

from .config import (
    ARMS,
    DATASET_FILE,
    DATASET_ID,
    DATASET_REVISION,
    MAX_CONTEXT,
    MIN_POSITION,
    NLA_REPO,
    NLA_REVISION,
    PROTOCOL_VERSION,
    SELECTION_SEED,
    SHUFFLE_SEED,
    TARGET_ID,
    TARGET_REVISION,
)
from .contamination import SHIPPED_EXCLUSION, fingerprint, load_exclusion

EXPLANATION_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)

# Both checkpoint pairs read the RL run's sidecar (tokens, prompt templates, mse_scale).
SIDECAR_SUBDIR = "rl_vllm"


def extract_explanation(response: str) -> str | None:
    match = EXPLANATION_RE.search(response or "")
    return match.group(1).strip() if match else None


def _weights_present(path: Path) -> bool:
    if (path / "adapter_config.json").exists():
        return any(path.glob("adapter_model.*"))
    index = path / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        return all((path / shard).exists() for shard in shards)
    return (path / "model.safetensors").exists()


def nla_subdir(subdir: str, sidecar_only: bool = False) -> Path:
    """Local path of one subfolder of the pinned custom NLA repo, downloading only that folder
    (or only its nla_meta.yaml with `sidecar_only`)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    kwargs = dict(
        repo_id=NLA_REPO, revision=NLA_REVISION,
        allow_patterns=[f"{subdir}/nla_meta.yaml" if sidecar_only else f"{subdir}/*"],
        # iter_*/reference is the frozen SFT adapter for the RL KL term; inference never reads it.
        ignore_patterns=[f"{subdir}/reference/*"],
    )
    present = (lambda path: (path / "nla_meta.yaml").exists()) if sidecar_only else _weights_present
    try:
        root = Path(snapshot_download(**kwargs, local_files_only=True))
        if not present(root / subdir):
            raise LocalEntryNotFoundError(f"{NLA_REPO}/{subdir} is not fully cached")
    except LocalEntryNotFoundError:
        root = Path(snapshot_download(**kwargs))
    if root.name != NLA_REVISION:
        raise RuntimeError(f"resolved {NLA_REPO} to {root.name}, expected pinned revision {NLA_REVISION}")
    if not present(root / subdir):
        raise RuntimeError(f"{NLA_REPO}/{subdir} is incomplete at {root}")
    return root / subdir


@dataclass(frozen=True)
class EasyNLASpec:
    key: str
    av_subdir: str | None  # None: the raw target checkpoint (the RL adapter goes on the base)
    av_lora_subdir: str | None
    ar_subdir: str
    target_id: str = TARGET_ID
    target_revision: str = TARGET_REVISION
    layer: int = 32
    d_model: int = 3840

    @property
    def av_dir(self) -> Path:
        if self.av_subdir is None:
            return resolve_snapshot(self.target_id, self.target_revision)
        return nla_subdir(self.av_subdir)

    @property
    def av_lora(self) -> Path | None:
        return None if self.av_lora_subdir is None else nla_subdir(self.av_lora_subdir)

    @property
    def ar_dir(self) -> Path:
        return nla_subdir(self.ar_subdir)


SPECS = {
    # CONTINUED-ADAPTER mode: rl_vllm/iter_000400 is the trainable adapter and
    # already contains SFT+RL, so it goes on the RAW base — the saved
    # iter_000400/reference adapter (frozen SFT, for the KL term) is what marks
    # this mode. Stacking it on merged/av_hf double-counts the SFT and the AV
    # stops closing its tags (measured: 1/4 well-formed vs 4/4 on the raw base).
    "rl": EasyNLASpec(key="rl", av_subdir=None, av_lora_subdir="rl_vllm/iter_000400",
                      ar_subdir="rl_vllm/critic_latest"),
    # Pre-RL (warm-start SFT) pair. RL is the stage that co-adapts AV to AR, so this
    # is the control for the co-adaptation reading of the RL result.
    "sft": EasyNLASpec(key="sft", av_subdir="merged/av_hf", av_lora_subdir=None,
                       ar_subdir="merged/ar_hf"),
}


def spec_for(key: str) -> EasyNLASpec:
    return SPECS[key]


# ─── stage 1: dataset (clean-by-construction selection) ─────────────────────


def dataset(args: argparse.Namespace) -> None:
    """Select N rows whose documents the custom NLA provably never trained on.

    Source is the warm-start data's av_sft TRAIN split, which covers ~37k documents against
    the custom NLA's ~25k av_sft pool, so most of it was never in any custom-NLA stage.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer

    exclude = load_exclusion(Path(args.exclude))
    excluded_docs = set(exclude["excluded_doc_ids"])
    excluded_fps = set(exclude["excluded_fingerprints"])

    records = []
    path = f"datasets/{DATASET_ID}@{DATASET_REVISION}/{DATASET_FILE}"
    with HfFileSystem().open(path, "rb") as handle:
        table = pq.ParquetFile(handle)
        for batch in table.iter_batches(
            batch_size=16384,
            columns=["doc_id", "response", "detokenized_text_truncated", "n_raw_tokens"],
        ):
            records.extend(
                zip(
                    batch.column("doc_id").to_pylist(),
                    batch.column("response").to_pylist(),
                    batch.column("detokenized_text_truncated").to_pylist(),
                    batch.column("n_raw_tokens").to_pylist(),
                )
            )
    print(f"[dataset] {len(records)} candidate rows from {DATASET_FILE}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_snapshot(TARGET_ID, TARGET_REVISION), local_files_only=True, trust_remote_code=True,
    )

    order = sorted(range(len(records)), key=lambda i: stable_seed(SELECTION_SEED, records[i][0], i))
    rows: list[dict] = []
    dropped = {"doc_excluded": 0, "text_excluded": 0, "no_tags": 0, "short": 0}
    truncated = 0
    for index in order:
        if len(rows) == args.n:
            break
        doc_id, response, text, n_raw = records[index]
        if doc_id in excluded_docs:
            dropped["doc_excluded"] += 1
            continue
        if fingerprint(text) in excluded_fps:
            dropped["text_excluded"] += 1
            continue
        gold = extract_explanation(response)
        if gold is None:
            dropped["no_tags"] += 1
            continue
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if len(ids) < MIN_POSITION:
            dropped["short"] += 1
            continue
        kept = [int(t) for t in ids]
        if len(kept) > MAX_CONTEXT:
            kept = kept[:1] + kept[-(MAX_CONTEXT - 1):]
            truncated += 1
        rows.append({
            "protocol": PROTOCOL_VERSION, "row_rank": len(rows), "doc_id": doc_id,
            "gold_explanation": gold, "n_target_tokens": len(ids),
            "n_source_tokens": int(n_raw), "truncated": len(ids) > MAX_CONTEXT,
            "input_ids": kept,
        })
    if len(rows) != args.n:
        raise RuntimeError(f"only {len(rows)} usable rows; lower --n")

    out = args.results_dir / args.spec / "dataset"
    write_jsonl(out / "rows.jsonl", rows)
    write_json(out / "manifest.json", {
        "protocol": PROTOCOL_VERSION, "source": f"{DATASET_ID}/{DATASET_FILE}",
        "dataset_revision": DATASET_REVISION, "selection_seed": SELECTION_SEED,
        "n": len(rows), "n_docs": len({r["doc_id"] for r in rows}),
        "dropped": dropped, "left_truncated": truncated,
        "exclusion_manifest": str(args.exclude),
    })
    print(f"[dataset] {len(rows)} rows / {len({r['doc_id'] for r in rows})} docs -> {out}")
    print(f"[dataset] dropped {dropped}; left-truncated {truncated}")


# ─── stage 2: extract ───────────────────────────────────────────────────────


def extract(args: argparse.Namespace) -> None:
    """Gemma-3-12B-IT layer-32 residual at the final token."""
    from soar_nla.hub import load_target, resolve_decoder_layers

    spec = spec_for(args.spec)
    rows = read_jsonl(args.results_dir / spec.key / "dataset" / "rows.jsonl")
    model, tokenizer, checkpoint = load_target(spec, device=args.device)
    layers = resolve_decoder_layers(model)
    pad_id = tokenizer.pad_token_id
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().float().cpu()

    acts = torch.zeros(len(rows), spec.d_model, dtype=torch.float32)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
    for start in range(0, len(order), args.batch_size):
        chunk = [rows[i] for i in order[start : start + args.batch_size]]
        width = max(len(row["input_ids"]) for row in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, row in enumerate(chunk):
            length = len(row["input_ids"])
            ids[j, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
            mask[j, :length] = 1
        handle = layers[spec.layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                model(input_ids=ids.to(args.device), attention_mask=mask.to(args.device), use_cache=False)
        finally:
            handle.remove()
        last = mask.sum(dim=1) - 1
        for j, row in enumerate(chunk):
            acts[row["row_rank"]] = captured["hidden"][j, last[j]]
        print(f"[extract] {min(start + args.batch_size, len(order))}/{len(order)}", flush=True)

    out = args.results_dir / spec.key / "extract"
    out.mkdir(parents=True, exist_ok=True)
    save_file({"activations": acts.contiguous()}, str(out / "activations.safetensors"))
    norms = acts.norm(dim=-1)
    write_json(out / "manifest.json", {
        "protocol": PROTOCOL_VERSION, "target_checkpoint": str(checkpoint), "layer": spec.layer,
        "hook": f"decoder_layers[{spec.layer}] output", "n_rows": len(rows),
        "norm": {"mean": float(norms.mean()), "median": float(norms.median())},
    })
    print(f"[extract] wrote {tuple(acts.shape)}; ||v|| mean={norms.mean():.0f}")


# ─── stage 3: decode (AV, Karvonen injection) ───────────────────────────────


def decode(args: argparse.Namespace) -> None:
    """AV-verbalize each activation with EasyNLA's Karvonen residual injection."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nla.config import load_nla_config
    from nla.utils import build_prompt_text, register_karvonen_hook
    from nla.utils.arch_adapters import resolve_text_model

    spec = spec_for(args.spec)
    base = args.results_dir / spec.key
    rows = read_jsonl(base / "dataset" / "rows.jsonl")
    acts = load_file(str(base / "extract" / "activations.safetensors"))["activations"]

    av_dir, av_lora = spec.av_dir, spec.av_lora
    tokenizer = AutoTokenizer.from_pretrained(str(av_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(av_dir), dtype=torch.bfloat16, device_map={"": args.device},
        trust_remote_code=True, attn_implementation="eager",
    ).eval()
    # google/gemma-3-12b-it loads as the multimodal wrapper; the adapter and the
    # Karvonen hook both target the text model. No-op for merged text-only AVs.
    model = resolve_text_model(model)
    if av_lora is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(av_lora), is_trainable=False).eval()
        print(f"[decode] applied RL LoRA {av_lora.name}")

    cfg = load_nla_config(str(nla_subdir(SIDECAR_SUBDIR, sidecar_only=True)), tokenizer=tokenizer)
    inj_id, left_id, right_id = cfg.injection_token_id, cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id

    prompt_msgs = [{"role": "user", "content": cfg.actor_prompt_template.replace("{injection_char}", "<INJECT>")}]
    prompt_text = build_prompt_text(prompt_msgs, cfg.injection_char, tokenizer)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    marker = [i for i, t in enumerate(prompt_ids) if t == inj_id]
    assert len(marker) == 1, f"expected exactly one marker, found {len(marker)}"
    position = marker[0]
    assert prompt_ids[position - 1] == left_id and prompt_ids[position + 1] == right_id, (
        "marker neighbors do not match the sidecar — template drift"
    )
    print(f"[decode] prompt {len(prompt_ids)} tokens, marker at {position}")

    # The hook reads vectors_ref[0] on every forward; RAW vectors, norm-matched
    # inside karvonen_inject_in_residual against the live residual.
    vectors_ref = [None]
    register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1)

    work = [r["row_rank"] for r in rows if r["row_rank"] % args.num_shards == args.shard]
    results: list[dict] = []
    ids_base = torch.tensor(prompt_ids, dtype=torch.long, device=args.device)
    for start in range(0, len(work), args.batch_size):
        ranks = work[start : start + args.batch_size]
        size = len(ranks)
        batch_ids = ids_base.unsqueeze(0).repeat(size, 1)
        vectors_ref[0] = acts[torch.tensor(ranks)].float().to(args.device)
        try:
            with torch.inference_mode():
                out = model.generate(
                    input_ids=batch_ids, attention_mask=torch.ones_like(batch_ids),
                    do_sample=args.temperature > 0, temperature=args.temperature or None,
                    max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                    disable_compile=True,
                )
        finally:
            vectors_ref[0] = None
        for j, rank in enumerate(ranks):
            raw = tokenizer.decode(out[j, batch_ids.shape[1]:], skip_special_tokens=True)
            match = EXPLANATION_RE.search(raw)
            results.append({
                "protocol": PROTOCOL_VERSION, "row_rank": rank,
                "parse_ok": match is not None,
                "explanation": match.group(1).strip() if match else raw.strip(),
            })
        print(f"[decode shard {args.shard}] {start + size}/{len(work)}", flush=True)

    out_dir = base / "decode"
    name = ("explanations.jsonl" if args.num_shards == 1
            else f"shard-{args.shard:02d}-of-{args.num_shards:02d}.jsonl")
    write_jsonl(out_dir / name, results)
    write_json(out_dir / "manifest.json", {
        "protocol": PROTOCOL_VERSION, "av_dir": str(av_dir), "av_lora": str(av_lora),
        "injection": "karvonen_inject_in_residual @ layers[1], raw vectors, norm-matched",
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
        "parse_failures": sum(1 for r in results if not r["parse_ok"]),
    })
    print(f"[decode] {len(results)} explanations, "
          f"{sum(1 for r in results if not r['parse_ok'])} parse failures")


def merge_decode(args: argparse.Namespace) -> None:
    spec = spec_for(args.spec)
    directory = args.results_dir / spec.key / "decode"
    rows: list[dict] = []
    for shard in range(args.num_shards):
        rows.extend(read_jsonl(directory / f"shard-{shard:02d}-of-{args.num_shards:02d}.jsonl"))
    rows.sort(key=lambda row: row["row_rank"])
    if [r["row_rank"] for r in rows] != list(range(args.n)):
        raise RuntimeError(f"merged {len(rows)} rows, expected ranks 0..{args.n - 1}")
    write_jsonl(directory / "explanations.jsonl", rows)
    bad = sum(1 for r in rows if not r["parse_ok"])
    print(f"[decode merge] {len(rows)} explanations, {bad} without <explanation> tags")


# ─── stage 4: reconstruct (AR) ──────────────────────────────────────────────


def shuffle_mapping(n: int) -> dict[int, int]:
    """Row -> donor row for the shuffled arms: one cycle through a seeded permutation, so it is
    a derangement and both shuffled arms use the same pairing."""
    cycle = list(range(n))
    seeded_rng(SHUFFLE_SEED).shuffle(cycle)
    return {cycle[k]: cycle[(k + 1) % len(cycle)] for k in range(len(cycle))}


def reconstruct(args: argparse.Namespace) -> None:
    """AR every arm back to a vector, via EasyNLA's critic_predict contract."""
    from transformers import AutoTokenizer

    from nla.config import load_nla_config
    from nla.models import NLACriticModel
    from nla.utils import critic_predict

    spec = spec_for(args.spec)
    base = args.results_dir / spec.key
    rows = read_jsonl(base / "dataset" / "rows.jsonl")
    decoded = read_jsonl(base / "decode" / "explanations.jsonl")
    assert [r["row_rank"] for r in decoded] == list(range(len(rows))), "decode/dataset misaligned"

    ar_dir = spec.ar_dir
    tokenizer = AutoTokenizer.from_pretrained(str(spec.av_dir), trust_remote_code=True)
    cfg = load_nla_config(str(nla_subdir(SIDECAR_SUBDIR, sidecar_only=True)), tokenizer=tokenizer)
    critic = NLACriticModel.from_pretrained(
        str(ar_dir), torch_dtype=torch.bfloat16,
        device_map={"": args.device}, trust_remote_code=True,
    ).eval()
    print(f"[reconstruct] critic {ar_dir.name}, mse_scale={cfg.mse_scale}")

    gold = [row["gold_explanation"] for row in rows]
    mapping = shuffle_mapping(len(rows))
    av = [row["explanation"] for row in decoded]
    texts = {
        "av": av,
        "gold": gold,
        "gold_shuffled": [gold[mapping[i]] for i in range(len(rows))],
        "av_shuffled": [av[mapping[i]] for i in range(len(rows))],
    }

    pad_id = tokenizer.eos_token_id
    out_tensors: dict[str, torch.Tensor] = {}
    for arm in ARMS:
        preds = torch.empty(len(rows), spec.d_model, dtype=torch.float32)
        # add_special_tokens=False: AR-SFT and RL-time critic scoring both encode
        # the critic prompt without BOS (EasyNLA train_sft.py). The official
        # release's AR is the opposite convention — do not copy that one here.
        encoded = [
            tokenizer.encode(cfg.critic_prompt_template.format(explanation=text),
                             add_special_tokens=False)[: args.max_len]
            for text in texts[arm]
        ]
        for start in range(0, len(rows), args.batch_size):
            chunk = encoded[start : start + args.batch_size]
            width = max(len(ids) for ids in chunk)
            batch = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=args.device)
            attn = torch.zeros((len(chunk), width), dtype=torch.long, device=args.device)
            for j, ids in enumerate(chunk):
                batch[j, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=args.device)
                attn[j, : len(ids)] = 1
            with torch.inference_mode():
                preds[start : start + len(chunk)] = critic_predict(
                    critic, batch, attn, cfg.mse_scale
                ).float().cpu()
            print(f"[reconstruct {arm}] {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
        out_tensors[arm] = preds.contiguous()

    out = base / "reconstruct"
    out.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, str(out / "reconstructions.safetensors"))
    write_json(out / "manifest.json", {
        "protocol": PROTOCOL_VERSION, "ar_dir": str(ar_dir), "mse_scale": float(cfg.mse_scale),
        "arms": list(ARMS), "n_rows": len(rows), "add_special_tokens": False,
        "mean_explanation_chars": {arm: sum(map(len, texts[arm])) / len(rows) for arm in ARMS},
    })
    print(f"[reconstruct] wrote {len(ARMS)} arms x {len(rows)}")


# ─── stage 5: analyze ───────────────────────────────────────────────────────


def load_cosines(run_dir: Path) -> tuple[list[dict], torch.Tensor, dict[str, torch.Tensor]]:
    """(dataset rows, gold activations, per-arm cosine(reconstruction, gold)) for one stage."""
    rows = read_jsonl(run_dir / "dataset" / "rows.jsonl")
    gold = load_file(str(run_dir / "extract" / "activations.safetensors"))["activations"].float()
    preds = load_file(str(run_dir / "reconstruct" / "reconstructions.safetensors"))
    cosine = {arm: torch.nn.functional.cosine_similarity(preds[arm].float(), gold, dim=-1) for arm in ARMS}
    return rows, gold, cosine


def analyze(args: argparse.Namespace) -> None:
    """All-rows report for one stage. The paper's table is paper_numbers.py (strict-clean rows)."""
    from .metrics import evaluate, print_table, unit

    spec = spec_for(args.spec)
    base = args.results_dir / spec.key
    rows, gold, cosine = load_cosines(base)
    decoded = read_jsonl(base / "decode" / "explanations.jsonl")
    doc_key = [row["doc_id"] for row in rows]
    report = {
        "protocol": PROTOCOL_VERSION,
        "spec": {"key": spec.key, "nla": f"{NLA_REPO}@{NLA_REVISION}", "av": spec.av_subdir or TARGET_ID,
                 "av_lora": spec.av_lora_subdir, "ar": spec.ar_subdir, "layer": spec.layer},
        "contamination": "documents excluded from every custom-NLA training stage by construction",
        "av_parse_failures": sum(1 for r in decoded if not r["parse_ok"]),
        "full": evaluate(torch.ones(len(rows), dtype=torch.bool), doc_key, unit(gold), cosine,
                         args.resamples, f"{PROTOCOL_VERSION}:easynla:{spec.key}", arms=ARMS),
    }
    write_json(args.report or base / "report.json", report)
    print(f"\n{'=' * 80}\nWarm-start FVE — custom gemma-3-12b-it L32 NLA [{spec.key}]")
    print(f"AV parse failures: {report['av_parse_failures']}\n{'=' * 80}")
    print_table("HELD-OUT (clean by construction)", report["full"])
    print()


# ─── CLI ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soar_nla.faithfulness.fve")
    subs = parser.add_subparsers(dest="stage", required=True)

    def add(name, handler):
        sub = subs.add_parser(name)
        sub.set_defaults(handler=handler)
        sub.add_argument("--spec", choices=sorted(SPECS), default="rl")
        sub.add_argument("--results-dir", type=Path, default=results_dir("faithfulness") / "fve")
        sub.add_argument("--n", type=int, default=2000)
        return sub

    sub = add("dataset", dataset)
    sub.add_argument("--exclude", type=Path, default=SHIPPED_EXCLUSION)

    sub = add("extract", extract)
    sub.add_argument("--device", default="cuda:0")
    sub.add_argument("--batch-size", type=int, default=8)

    sub = add("decode", decode)
    sub.add_argument("--device", default="cuda:0")
    sub.add_argument("--batch-size", type=int, default=16)
    sub.add_argument("--max-new-tokens", type=int, default=256)
    sub.add_argument("--temperature", type=float, default=0.0)
    sub.add_argument("--shard", type=int, default=0)
    sub.add_argument("--num-shards", type=int, default=1)

    sub = add("merge-decode", merge_decode)
    sub.add_argument("--num-shards", type=int, default=1)

    sub = add("reconstruct", reconstruct)
    sub.add_argument("--device", default="cuda:0")
    sub.add_argument("--batch-size", type=int, default=32)
    sub.add_argument("--max-len", type=int, default=1024)

    sub = add("analyze", analyze)
    sub.add_argument("--resamples", type=int, default=10000)
    sub.add_argument("--report", type=Path, default=None, help="default: <results-dir>/<spec>/report.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
