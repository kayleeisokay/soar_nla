"""The function-vector shot-sweep protocol that this replication follows.

Constants, dataset loaders, prompt construction, paraphrase prompts and
answer scoring are taken from ``notebooks/executor.py`` of the reference
repository below at ``REFERENCE_COMMIT``. The paper uses only the 2-shot
English->French configuration (50 queries), but the sampling plan replays
every RNG draw of the original sweep, which visits all six tasks and shot
counts in order; every task's data file is therefore needed to reproduce the
English->French queries, demonstrations and donor prompts.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

REFERENCE_REPO = "https://github.com/kayleeisokay/natural_language_autoencoders"
REFERENCE_COMMIT = "02d62272d90528742ef476c316dc39098f5ae798"
REFERENCE_FILE = "notebooks/executor.py"
REFERENCE_DATA_DIR = "notebooks/datasets"

TASKS = (
    "antonym", "capitalize", "country-capital", "english-french",
    "present-past", "singular-plural",
)
SHOT_COUNTS = (1, 2, 4)
INSTRUCTION = "Answer with a single word only. No punctuation, no explanation."

# SHA-256 of every reference data file the sampling plan reads.
DATA_FILES = {
    "verb-pairs.train": "7a3593059df866a867031db8a05b9f41941ba75f45ec4c12dc89c81d5bdb9423",
    "verb-pairs.val": "43712a1098cbcc959aa98d925f34006246a50debb4b415fbae2d0f6ccd16f09b",
    "verb-pairs.test": "e963c3053c8ee01e828737773f6fc81b55072d534e162455b2445d2444eb538f",
    "noun-pairs.train": "e15f0a281e6676f5c1fac6bc3bd8c894fba2e3109b216d882ee4d467fe4f968e",
    "noun-pairs.val": "f06d3bf0f7b57f366f04f45610510e70f81af5f71fe5e3c5824f778e3dfa089f",
    "noun-pairs.test": "8bba47ac01560175a070f97e208781b330f859a5263f8010565250f10dcf725f",
    "adjective-pairs.train": "e13a19c1cc35ab27635eda51d3cbaf2cd6788bb80e7accd9d2333bb2dd09f2ac",
    "adjective-pairs.val": "315e5206144ec566239ab75b803755f0df9c10ad8faef3b2bffb344f60ecd3fc",
    "adjective-pairs.test": "8fb05da7a4854c1088aea011e33ad1a9adad8219905014087de081e9e82ca42c",
    "capitalize.json": "ad3e4cb8e4f33ea16fb11de9ecd226d8a6fd945fa8a8bb8e441d4e30cdd9ac72",
    "country-capital.json": "7228490c71177db37f7392614470e610f562374601dc6a51d8c883f07d6d4220",
    "en-fr.0-5000.txt": "96cfce2c345fdbf254310d63fc7230bc23529d4ecc669bb4a5e6bbede5db3182",
    "en-fr.5000-6500.txt": "ff4855a4cecf168652d8dff2870a9edb43b94dac0fc0eacaef27457af91f5d01",
    "present-past.json": "662f86f9e6dbba01c4f88d4c28162803a79deeca894fb1d0e788dd2cd9c0f2a2",
    "singular-plural.json": "ba09b3573872d34c0d6a4b4120d0594470782ae5a825be68581e826140f53397",
}

# Paraphrase prompts of the reference executor, verbatim.
PARAPHRASE_PROMPTS = {
    "para_medium": (
        "Please paraphrase the text provided below, preserving its exact meaning "
        "but expressing it in substantially different words and sentence structure "
        "than the original. Do not simply swap individual words for synonyms -- "
        "restructure the sentence(s) genuinely.\n\n"
        "<text>\n{text}\n</text>\n\n"
        "Include your final paraphrased text in <transformed_text> tags."
    ),
    "para_heavy": (
        "Please paraphrase aggressively. Preserve meaning and uncertainty, but "
        "substantially change wording, syntax, and organization. Do not simply swap individual words for synonyms -- "
        "restructure the sentence(s) genuinely.\n\n"
        "Text:\n{text}\n\n"
        "Respond with only the paraphrased text, wrapped exactly like this:\n"
        "<transformed_text>your paraphrase here</transformed_text>"
    ),
}


def load_antonym(data_dir: Path) -> list[dict[str, str]]:
    seen = set()
    pairs = []
    for pos in ("verb", "noun", "adjective"):
        for split in ("train", "val", "test"):
            with (data_dir / f"{pos}-pairs.{split}").open() as handle:
                for line in handle:
                    fields = line.strip().split("\t")
                    if len(fields) < 3 or fields[2] != "1":
                        continue
                    key = (fields[0], fields[1])
                    if key not in seen:
                        seen.add(key)
                        pairs.append({"input": fields[0], "output": fields[1]})
    return pairs


def load_english_french(data_dir: Path) -> list[dict[str, str]]:
    seen = {}
    for filename in ("en-fr.0-5000.txt", "en-fr.5000-6500.txt"):
        with (data_dir / filename).open() as handle:
            for line in handle:
                fields = line.strip().split()
                if len(fields) >= 2:
                    seen.setdefault(fields[0], fields[1])
    return [{"input": english, "output": french} for english, french in seen.items()]


def load_pairs(data_dir: Path, task: str) -> list[dict[str, str]]:
    if task == "antonym":
        return load_antonym(data_dir)
    if task == "english-french":
        return load_english_french(data_dir)
    with (data_dir / f"{task}.json").open() as handle:
        raw = json.load(handle)
    if isinstance(raw, dict) and "examples" in raw:
        raw = raw["examples"]
    return [{"input": row["input"], "output": row["output"]} for row in raw]


def build_prompt(pairs, query_index, n_shots, rng, extra_exclude_words=None):
    query_word = pairs[query_index]["input"]
    excluded = {query_word} | (extra_exclude_words or set())
    pool = [
        index for index in range(len(pairs))
        if index != query_index and pairs[index]["input"] not in excluded
    ]
    demo_indices = rng.sample(pool, min(n_shots, len(pool)))
    lines = [f"Q: {pairs[index]['input']}\nA: {pairs[index]['output']}" for index in demo_indices]
    lines.append(f"Q: {query_word}\nA:")
    prompt = INSTRUCTION + "\n\n" + "\n\n".join(lines)
    return prompt, pairs[query_index]["output"], demo_indices


def extract_word_answer(text):
    return text.strip().split("\n")[0].strip(string.punctuation + " ")


def correct(text, gold):
    return extract_word_answer(text).lower() == gold.lower()
