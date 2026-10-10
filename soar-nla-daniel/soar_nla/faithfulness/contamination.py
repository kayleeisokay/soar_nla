"""Contamination exclusions for the warm-start FVE eval set.

The eval rows come from the warm-start data's TRAIN split minus every document the custom NLA
could have seen. Two artifacts, one per subcommand:

  exclusion  The manifest the `dataset` stage of fve.py selects against: the doc_ids of every
             row in the five custom-NLA L32 parquets (train AND val splits of AV-SFT, AR-SFT and
             RL), plus text fingerprints of a second Gemma-3-12B L32 RL dataset built locally
             from the same corpus file. That local dataset is not public, so the manifest the
             paper's eval set was selected with ships with this package
             (exclude_easynla.json.gz); rebuilding without `--local-rl-parquet` gives the
             doc_id half only.
  strict     After selection, flag eval documents whose doc_id OR text fingerprint appears in
             the custom NLA's AV-SFT, AR-SFT or RL parquet, or in the first 120k rows of the
             local RL dataset. This removes 3 docs / 3 rows; the paper's FVE table is computed
             on the remaining 1,997 rows (paper_numbers.py).

Why fingerprints: doc_ids are corpus-file-scoped ("<corpus parquet>:train:<index>"). The
warm-start data mixes two independent 100k FineFineWeb draws (`finefineweb_100k.parquet` and
`finefineweb_100k_v2.parquet`); the custom NLA trained only on the first, so a doc_id join only
compares that half. Excluding the custom NLA's doc_ids leaves the eval set entirely inside the
`_v2` draw, and the fingerprint checks catch the few texts the two draws share.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import zlib
from pathlib import Path

from soar_nla.io_utils import read_jsonl, write_json
from soar_nla.paths import results_dir

from .config import (
    DATASET_FILE,
    DATASET_ID,
    DATASET_REVISION,
    NLA_DATA_ID,
    NLA_DATA_LAYER_DIR,
    NLA_DATA_REVISION,
)

NLA_PARQUETS = {
    "av_sft": "av_sft_shuf.parquet",
    "ar_sft": "ar_sft_shuf.parquet",
    "rl": "rl_shuf.parquet",
    "av_sft_val": "av_sft_shuf.val.parquet",
    "ar_sft_val": "ar_sft_shuf.val.parquet",
}

# The RL trainer holds out documents by hash at load time (nla.val_split):
# val_doc_permille(50,000 val rows, 499,846 rows) = 100.
RL_VAL_PERMILLE = 100

LOCAL_RL_STRICT_ROWS = 120_000

SHIPPED_EXCLUSION = Path(__file__).with_name("exclude_easynla.json.gz")


def fingerprint(text: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", text[:150]).strip().lower().encode()).hexdigest()


def is_val_doc(doc_id, permille: int) -> bool:
    """Same bucket as nla.val_split.is_val_doc."""
    return (zlib.crc32(str(doc_id).encode("utf-8")) % 1000) < permille


def load_exclusion(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _open_hf(repo_id: str, revision: str, filename: str):
    from huggingface_hub import HfFileSystem

    return HfFileSystem().open(f"datasets/{repo_id}@{revision}/{filename}", "rb")


def scan(source, with_text: bool, docs: set[str] | None = None, max_rows: int | None = None) -> dict:
    """doc_id -> its shortest `detokenized_text_truncated` (None without text).

    `source` is a path or an open binary handle. `docs` restricts the result; `max_rows`
    stops after that many rows in file order.
    """
    import pyarrow.parquet as pq

    columns = ["doc_id", "detokenized_text_truncated"] if with_text else ["doc_id"]
    best: dict = {}
    seen = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=16384, columns=columns):
        ids = batch.column("doc_id").to_pylist()
        texts = batch.column("detokenized_text_truncated").to_pylist() if with_text else [None] * len(ids)
        for doc, text in zip(ids, texts):
            if max_rows is not None and seen >= max_rows:
                return best
            seen += 1
            if docs is not None and doc not in docs:
                continue
            if not with_text:
                best.setdefault(doc, None)
            elif doc not in best or len(text) < len(best[doc]):
                best[doc] = text
    return best


def _scan_nla_parquet(key: str, with_text: bool) -> dict:
    name = f"{NLA_DATA_LAYER_DIR}/{NLA_PARQUETS[key]}"
    with _open_hf(NLA_DATA_ID, NLA_DATA_REVISION, name) as handle:
        found = scan(handle, with_text)
    print(f"[contamination] {name}: {len(found)} docs", flush=True)
    return found


def _fingerprints(texts) -> set[str]:
    return {fingerprint(text) for text in texts if text}


def exclusion(args: argparse.Namespace) -> None:
    docs = {key: set(_scan_nla_parquet(key, with_text=False)) for key in NLA_PARQUETS}
    av_train = docs["av_sft"] - docs["av_sft_val"]
    ar_train = docs["ar_sft"] - docs["ar_sft_val"]
    rl_train = {doc for doc in docs["rl"] if not is_val_doc(doc, RL_VAL_PERMILLE)}
    # Conservative: also drop the val-split docs shipped in the pools, so nothing
    # that ever sat in a custom-NLA parquet can enter the eval.
    pool = set().union(*docs.values())
    excluded = av_train | ar_train | rl_train | pool

    fingerprints: set[str] = set()
    if args.local_rl_parquet is not None:
        fingerprints = _fingerprints(scan(args.local_rl_parquet, with_text=True).values())
    else:
        print("[contamination] no --local-rl-parquet: fingerprints left empty, so the result "
              "will not match the shipped manifest", flush=True)

    manifest = {
        "excluded_doc_ids": sorted(excluded),
        "excluded_fingerprints": sorted(fingerprints),
        "sources": {
            "achand45_av_train": len(av_train),
            "achand45_ar_train": len(ar_train),
            "achand45_rl_train": len(rl_train),
            "achand45_any_pool": len(pool),
            "local_gemma3_rl_docs_by_text": len(fingerprints),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "wt", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    print(f"[contamination] {len(excluded)} doc_ids, {len(fingerprints)} fingerprints -> {args.output}")


def find_leaks(eval_text: dict[str, str], sources: dict[str, tuple[set[str], set[str]]]) -> dict[str, list[str]]:
    """Eval doc -> names of the sources that contain its doc_id or its text fingerprint."""
    leaks = {}
    for doc in sorted(eval_text):
        text_print = fingerprint(eval_text[doc])
        reasons = [name for name, (ids, prints) in sources.items() if doc in ids or text_print in prints]
        if reasons:
            leaks[doc] = reasons
    return leaks


def strict(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.rows)
    eval_docs = {row["doc_id"] for row in rows}
    with _open_hf(DATASET_ID, DATASET_REVISION, DATASET_FILE) as handle:
        eval_text = scan(handle, with_text=True, docs=eval_docs)
    missing = eval_docs - eval_text.keys()
    if missing:
        raise RuntimeError(f"{len(missing)} eval docs not found in {DATASET_FILE}")

    sft = {key: _scan_nla_parquet(key, with_text=True) for key in ("av_sft", "ar_sft")}
    rl = _scan_nla_parquet("rl", with_text=True)
    sources = {
        "custom_nla_sft": (set(sft["av_sft"]) | set(sft["ar_sft"]),
                         _fingerprints(sft["av_sft"].values()) | _fingerprints(sft["ar_sft"].values())),
        "custom_nla_rl": (set(rl), _fingerprints(rl.values())),
    }
    if args.local_rl_parquet is not None:
        local = scan(args.local_rl_parquet, with_text=True, max_rows=args.local_rl_rows)
        sources[f"local_gemma3_rl_first_{args.local_rl_rows}"] = (set(local), _fingerprints(local.values()))
    else:
        print("[contamination] no --local-rl-parquet: checking the public parquets only", flush=True)

    leaks = find_leaks(eval_text, sources)
    dropped_rows = sum(1 for row in rows if row["doc_id"] in leaks)
    write_json(args.output, {
        "excluded_doc_ids": sorted(leaks),
        "reasons": leaks,
        "n_eval_rows": len(rows),
        "n_eval_docs": len(eval_docs),
        "n_rows_dropped": dropped_rows,
        "sources": sorted(sources),
    })
    print(f"[contamination] strict filter drops {len(leaks)} docs / {dropped_rows} rows -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soar_nla.faithfulness.contamination", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    root = results_dir("faithfulness")

    sub = subs.add_parser("exclusion", help="rebuild the eval-set exclusion manifest")
    sub.add_argument("--local-rl-parquet", type=Path, default=None,
                     help="local Gemma-3-12B L32 rl_shuf.full.parquet (text fingerprints)")
    sub.add_argument("--output", type=Path, default=root / "exclude_easynla.json.gz")
    sub.set_defaults(func=exclusion)

    sub = subs.add_parser("strict", help="flag eval docs that leak by doc_id or text")
    sub.add_argument("--rows", type=Path, default=root / "fve" / "rl" / "dataset" / "rows.jsonl")
    sub.add_argument("--local-rl-parquet", type=Path, default=None)
    sub.add_argument("--local-rl-rows", type=int, default=LOCAL_RL_STRICT_ROWS)
    sub.add_argument("--output", type=Path, default=root / "strict_clean_excluded_docs.json")
    sub.set_defaults(func=strict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
