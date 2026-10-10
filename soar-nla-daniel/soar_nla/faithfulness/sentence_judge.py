"""Prepare, judge and aggregate sentence-level faithfulness of explanations.

Two comparisons, each judged sentence by sentence against the source passage with the full
explanation as context:

- Faithfulness section: `custom_sonnet` and `custom_av` explain the same 2,000 FineFineWeb
  activations of the warm-start FVE eval set (fve.py) — the Sonnet explanation from the
  warm-start data and the custom RL checkpoint's AV explanation.
- GCG section: `alpaca_av` and `alpaca_gcg_beta_0.01` are the official L32 AV explanation of
  the 1,573 late-position Alpaca rollouts (`soar_nla.alpaca`) and its beta=0.01 GCG rewrite
  (`soar_nla.gcg`). The GCG text is split at the original explanation's sentence boundaries
  (token spans), so the two arms judge aligned sentences.

The paper's tables (aggregate) keep the first two paragraphs of each explanation. Both
explanation styles end with a third, final-token unit — what the final token is and what
should come next, for the AV followed by "Example:" continuation lines — which predicts
unseen text rather than describing the passage.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from soar_nla.apikeys import api_key
from soar_nla.hub import model_spec, resolve_snapshot
from soar_nla.io_utils import read_jsonl, write_json
from soar_nla.paths import results_dir

LABELS = ("Supported", "Unsupported", "Illegible")
ARMS = ("custom_sonnet", "custom_av", "alpaca_av", "alpaca_gcg_beta_0.01")
ALPACA_ARMS = ("alpaca_av", "alpaca_gcg_beta_0.01")
MIN_SEQUENCE_TOKEN_INDEX = 50
SEPARATOR = re.compile(r"(?<=[.!?])\s+|\n+")
PARAGRAPH = "\n\n"
PROMPT = """You are checking whether a statement from an explanation is supported by a source passage. The passage may end mid-sentence or mid-word.

SOURCE PASSAGE:
{source}

FULL EXPLANATION:
{explanation}

STATEMENT:
{statement}

Classify the statement as:

- Supported: Every substantive claim about the observed passage is supported.
- Unsupported: At least one substantive claim about the observed passage is contradicted by, or absent from, the passage.
- Illegible: The statement is incoherent and cannot be judged reliably.

Answer with only one label (Supported, Unsupported, or Illegible) and nothing else."""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sentence_char_spans(text: str) -> list[tuple[int, int]]:
    """Nonempty spans matching re.split(SEPARATOR, text), without separators."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SEPARATOR.finditer(text):
        left, right = start, match.start()
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = match.end()
    left, right = start, len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        spans.append((left, right))
    return spans


def _plain_sentences(text: str) -> list[str]:
    return [text[start:end] for start, end in sentence_char_spans(text)]


def _token_spans(text: str, tokenizer) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    spans: list[tuple[int, int]] = []
    for char_start, char_end in sentence_char_spans(text):
        members = [i for i, (start, end) in enumerate(offsets) if end > char_start and start < char_end]
        if not members:
            raise ValueError(f"sentence has no tokens: {text[char_start:char_end]!r}")
        spans.append((members[0], members[-1] + 1))
    return ids, spans


def _apply_gcg(original_ids: list[int], record: dict) -> list[int]:
    optimized = list(original_ids)
    for step in record["trajectory"]:
        if step["accepted"]:
            position = step["position"]
            if optimized[position] != step["old_token_id"]:
                raise ValueError(f"GCG trajectory mismatch at position {position}")
            optimized[position] = step["new_token_id"]
    if len(optimized) != record["token_count"]:
        raise ValueError("GCG token-count mismatch")
    return optimized


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    spec = model_spec("gemma12b")
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_snapshot(spec.target_id, spec.target_revision),
        local_files_only=True,
        trust_remote_code=True,
    )
    items: list[dict] = []

    def add_plain(arm: str, example_id: str, source: str, explanation: str) -> None:
        for index, statement in enumerate(_plain_sentences(explanation)):
            items.append({
                "key": f"{arm}|{example_id}|{index}",
                "arm": arm,
                "example_id": example_id,
                "statement_index": index,
                "source": source,
                "explanation": explanation,
                "statement": statement,
            })

    # The same 2,000 activations have Sonnet and RL-AV text.
    custom = args.custom_results
    custom_rows = read_jsonl(custom / "dataset" / "rows.jsonl")
    custom_av = read_jsonl(custom / "decode" / "explanations.jsonl")
    if len(custom_rows) != len(custom_av):
        raise ValueError("custom dataset/decode lengths differ")
    for row, decoded in zip(custom_rows, custom_av, strict=True):
        source = tokenizer.decode(row["input_ids"], skip_special_tokens=True)
        example_id = str(row["row_rank"])
        add_plain("custom_sonnet", example_id, source, row["gold_explanation"])
        add_plain("custom_av", example_id, source, decoded["explanation"])

    # Alpaca: require sequence position > 50 and a beta=.01 GCG result.
    alpaca = args.alpaca_results
    rollouts = {row["sample_id"]: row for row in read_jsonl(alpaca / "rollouts" / "rollouts.jsonl")}
    av_rows = {row["sample_id"]: row for row in read_jsonl(alpaca / "decode" / "explanations.jsonl")}
    gcg_rows = {
        row["sample_id"]: row
        for row in read_jsonl(args.gcg_results)
        if float(row["beta"]) == 0.01
    }
    eligible = sorted(
        sample_id for sample_id in rollouts.keys() & av_rows.keys() & gcg_rows.keys()
        if rollouts[sample_id]["sequence_token_index"] > MIN_SEQUENCE_TOKEN_INDEX
    )
    for sample_id in eligible:
        rollout, av, gcg = rollouts[sample_id], av_rows[sample_id], gcg_rows[sample_id]
        original_ids, spans = _token_spans(av["explanation"], tokenizer)
        optimized_ids = _apply_gcg(original_ids, gcg)
        canonical_original = tokenizer.decode(
            original_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if canonical_original != gcg["canonical_original_explanation"]:
            raise ValueError(f"canonical original mismatch for {sample_id}")
        source = rollout["preceding_text_inclusive"]
        optimized_explanation = tokenizer.decode(
            optimized_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        for index, (start, end) in enumerate(spans):
            av_statement = tokenizer.decode(
                original_ids[start:end], skip_special_tokens=False, clean_up_tokenization_spaces=False
            ).strip()
            gcg_statement = tokenizer.decode(
                optimized_ids[start:end], skip_special_tokens=False, clean_up_tokenization_spaces=False
            ).strip()
            for arm, explanation, statement in (
                ("alpaca_av", canonical_original, av_statement),
                ("alpaca_gcg_beta_0.01", optimized_explanation, gcg_statement),
            ):
                items.append({
                    "key": f"{arm}|{sample_id}|{index}",
                    "arm": arm,
                    "example_id": sample_id,
                    "statement_index": index,
                    "source": source,
                    "explanation": explanation,
                    "statement": statement,
                })

    if len({row["key"] for row in items}) != len(items):
        raise ValueError("duplicate item keys")
    _write_jsonl(args.output, items)
    counts: dict[str, int] = {}
    for row in items:
        counts[row["arm"]] = counts.get(row["arm"], 0) + 1
    print(json.dumps({"output": str(args.output), "total": len(items), "arms": counts}, indent=2))


def _parse_label(text: str) -> str | None:
    match = re.fullmatch(r"\s*(Supported|Unsupported|Prediction|Illegible)\s*[.!]?\s*", text, re.I)
    if not match:
        return None
    return next(label for label in LABELS if label.lower() == match.group(1).lower())


def _judge_one(client, model: str, item: dict, retries: int, reasoning_effort: str) -> dict:
    prompt = PROMPT.format(
        source=item["source"], explanation=item["explanation"], statement=item["statement"]
    )
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=512,
                reasoning={"effort": reasoning_effort},
            )
            raw = response.output_text.strip()
            label = _parse_label(raw)
            if label is None:
                raise ValueError(f"unparseable response: {raw[:120]!r}")
            return {
                "label": label,
                "raw": raw,
                "response_id": response.id,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except Exception as exc:
            if attempt + 1 == retries:
                return {"error": str(exc)[:500]}
            time.sleep(min(60, 2**attempt + random.random()))
    raise AssertionError("unreachable")


def judge(args: argparse.Namespace) -> None:
    from openai import OpenAI

    items = read_jsonl(args.input)
    partial = Path(str(args.output) + ".partial")
    done: dict[str, dict] = {}
    if partial.exists():
        for row in read_jsonl(partial):
            if "error" not in row:
                done[row["key"]] = row
    todo = [row for row in items if row["key"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[judge] {len(items)} items | {len(done)} cached | {len(todo)} calls", flush=True)
    client = OpenAI(api_key=api_key("OPENAI_API_KEY", args.env_file))
    lock = threading.Lock()
    errors = 0
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_judge_one, client, args.model, row, args.retries, args.reasoning_effort): row
            for row in todo
        }
        for index, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            result = {"key": item["key"], **future.result()}
            with lock:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                if "error" in result:
                    errors += 1
                else:
                    done[item["key"]] = result
            if index % 250 == 0 or index == len(todo):
                print(f"  {index}/{len(todo)} | errors={errors}", flush=True)
    merged = [{**item, **done[item["key"]]} for item in items if item["key"] in done]
    _write_jsonl(args.output, merged)
    print(f"[judge] wrote {len(merged)}/{len(items)} rows to {args.output}")


# ─── aggregate ──────────────────────────────────────────────────────────────


def _read_rows(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return read_jsonl(path)


def _paragraph_of(explanation: str, start: int) -> tuple[int, int]:
    """(index of the paragraph containing character `start`, number of paragraphs)."""
    bounds, offset = [], 0
    for paragraph in explanation.split(PARAGRAPH):
        bounds.append(offset + len(paragraph))
        offset += len(paragraph) + len(PARAGRAPH)
    return next(i for i, bound in enumerate(bounds) if start < bound), len(bounds)


def label_paragraphs(rows: list[dict]) -> list[dict]:
    """One slim record per judged sentence, with the paragraph it sits in.

    Paragraphs are the explanation split on a blank line. SEPARATOR also splits on every
    newline, so no sentence spans two paragraphs. The Alpaca arms share the original AV
    explanation's sentence spans, so both take their paragraph from the `alpaca_av` row.
    Rows that already carry `paragraph` (the slim file written by `aggregate --labels-out`)
    pass through unchanged.
    """
    alpaca_paragraphs: dict[tuple[str, int], tuple[int, int]] = {}
    for row in rows:
        if row["arm"] == "alpaca_av" and "paragraph" not in row:
            spans = sentence_char_spans(row["explanation"])
            alpaca_paragraphs[(row["example_id"], row["statement_index"])] = _paragraph_of(
                row["explanation"], spans[row["statement_index"]][0]
            )
    slim: list[dict] = []
    for row in rows:
        if row["arm"] not in ARMS:
            continue
        if "paragraph" in row:
            slim.append(row)
            continue
        if row["arm"] in ALPACA_ARMS:
            paragraph, n_paragraphs = alpaca_paragraphs[(row["example_id"], row["statement_index"])]
        else:
            explanation = row["explanation"]
            start, end = sentence_char_spans(explanation)[row["statement_index"]]
            if explanation[start:end] != row["statement"]:
                raise ValueError(f"statement does not match its span: {row['key']}")
            paragraph, n_paragraphs = _paragraph_of(explanation, start)
        slim.append({
            "key": row["key"], "arm": row["arm"], "example_id": row["example_id"],
            "statement_index": row["statement_index"], "paragraph": paragraph,
            "n_paragraphs": n_paragraphs, "label": row["label"],
        })
    return slim


PAPER_SCOPE = "first_two_paragraphs"
SCOPES = {
    # The paper's tables: sentences in the first two paragraphs (description of the passage),
    # leaving out the final-token unit and anything after it.
    PAPER_SCOPE: lambda row: row["paragraph"] < 2,
    "all_sentences": lambda row: True,
    "after_second_paragraph": lambda row: row["paragraph"] >= 2,
}


def tabulate(slim: list[dict]) -> dict:
    table: dict = {}
    for scope, keep in SCOPES.items():
        table[scope] = {}
        for arm in ARMS:
            chosen = [row for row in slim if row["arm"] == arm and keep(row)]
            if not chosen:
                continue
            counts = Counter(row["label"] for row in chosen)
            table[scope][arm] = {
                "n_sentences": len(chosen),
                "n_explanations": len({row["example_id"] for row in chosen}),
                "counts": {label: counts[label] for label in LABELS},
                "percent": {label: 100.0 * counts[label] / len(chosen) for label in LABELS},
            }
    return table


def _markdown(table: dict) -> str:
    names = {"custom_sonnet": "Sonnet", "custom_av": "AV (custom RL)",
             "alpaca_av": "Alpaca AV (official L32)", "alpaca_gcg_beta_0.01": "Alpaca GCG β=0.01"}
    lines = []
    for scope, block in table.items():
        lines += [f"### {scope}", "", "| explanation | n sentences | Supported | Unsupported | Illegible |",
                  "|---|---|---|---|---|"]
        for arm, value in block.items():
            pct = value["percent"]
            lines.append(f"| {names[arm]} | {value['n_sentences']:,} | {pct['Supported']:.1f}% | "
                         f"{pct['Unsupported']:.1f}% | {pct['Illegible']:.1f}% |")
        lines.append("")
    return "\n".join(lines)


def aggregate(args: argparse.Namespace) -> dict:
    slim = label_paragraphs(_read_rows(args.results))
    table = tabulate(slim)
    out = args.output_dir
    write_json(out / "sentence_judge_table.json", {
        "results": args.results.name, "judge": args.model, "reasoning_effort": args.reasoning_effort,
        "paper_scope": PAPER_SCOPE, "table": table,
    })
    (out / "sentence_judge_table.md").write_text(_markdown(table), encoding="utf-8")
    if args.labels_out is not None:
        args.labels_out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(args.labels_out, "wb", mtime=0) as handle:
            for row in sorted(slim, key=lambda r: (r["arm"], r["example_id"].zfill(20), r["statement_index"])):
                handle.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
    print(_markdown(table))
    return table


def build_parser() -> argparse.ArgumentParser:
    root = results_dir("faithfulness")
    parser = argparse.ArgumentParser(prog="soar_nla.faithfulness.sentence_judge")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--custom-results", type=Path, default=root / "fve" / "rl")
    prep.add_argument("--alpaca-results", type=Path, default=results_dir("alpaca") / "gemma12b",
                      help="soar_nla.alpaca output for the model (rollouts/ and decode/)")
    prep.add_argument("--gcg-results", type=Path, default=results_dir("gcg") / "results.jsonl",
                      help="full soar_nla.gcg results.jsonl (needs the trajectories)")
    prep.add_argument("--output", type=Path, default=root / "sentence_judge" / "items.jsonl")
    prep.set_defaults(func=prepare)

    run = sub.add_parser("judge")
    run.add_argument("--input", type=Path, default=root / "sentence_judge" / "items.jsonl")
    run.add_argument("--output", type=Path, default=root / "sentence_judge" / "results.jsonl")
    run.add_argument("--model", default="gpt-5.6-luna")
    run.add_argument("--env-file", type=Path, default=None, help="dotenv file with OPENAI_API_KEY")
    run.add_argument("--workers", type=int, default=32)
    run.add_argument("--retries", type=int, default=8)
    run.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"), default="low")
    run.add_argument("--limit", type=int, default=0)
    run.set_defaults(func=judge)

    agg = sub.add_parser("aggregate", help="Supported/Unsupported/Illegible rates per arm")
    agg.add_argument("--results", type=Path, default=root / "sentence_judge" / "results.jsonl",
                     help="judge output, or the slim sentence_labels.jsonl.gz")
    agg.add_argument("--output-dir", type=Path, default=root)
    agg.add_argument("--labels-out", type=Path, default=None,
                     help="also write slim per-sentence labels (gzipped JSONL)")
    agg.add_argument("--model", default="gpt-5.6-luna", help="recorded in the table metadata")
    agg.add_argument("--reasoning-effort", default="low", help="recorded in the table metadata")
    agg.set_defaults(func=aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
