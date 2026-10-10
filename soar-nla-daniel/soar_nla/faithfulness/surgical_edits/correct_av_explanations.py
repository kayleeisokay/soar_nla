"""Surgically correct factual errors in AV explanations of the shared Alpaca rollouts.

An LLM editor copies each AV explanation, changing only words or phrases that
conflict with the passage: non-verbatim quotations are replaced by verbatim
passage spans of the same word count and the claimed final token is fixed,
while predictions about text after the passage are left unchanged.

Eligible examples are the rollouts whose sampled position is after sequence
token 50 (1,573), in selection-rank order. The paper edited the first 200 with
GPT-5.6 Sol in two runs:

    python -m soar_nla.faithfulness.surgical_edits.correct_av_explanations \\
        --offset 0 --limit 100 --output results/surgical_edits/sol_equal_quote_length_100.jsonl
    python -m soar_nla.faithfulness.surgical_edits.correct_av_explanations \\
        --offset 100 --limit 100 --output results/surgical_edits/sol_equal_quote_length_101_200.jsonl

Requires the `openai` package and OPENAI_API_KEY (or --env-file).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from soar_nla.alpaca.subsets import late_positions
from soar_nla.apikeys import api_key
from soar_nla.io_utils import read_jsonl
from soar_nla.paths import results_dir


PROMPT = """You are surgically correcting an explanation that describes a passage. The passage may end mid-sentence or mid-word.

Copy the explanation exactly, changing only specific words or phrases that conflict with the passage. Claims about quotations, exact wording, and the observed final token must accurately match the passage. When correcting a quotation, replace it with a verbatim passage span containing the same number of whitespace-delimited words as the original quotation. Leave predictions about text after the passage unchanged.

EXAMPLE 1

PASSAGE:
user
Describe a situation in which you would use the word “enigma”
model
Okay, here's a situation where the word "enigma" would be perfectly fitting, followed by an explanation of why:

Dr. Eleanor Vance, a brilliant but reclusive astrophysicist, had dedicated her life to searching for extraterrestrial life. For decades, she'd published groundbreaking papers, but she rarely appeared in public and refused to discuss her personal life. Then, suddenly,

ORIGINAL EXPLANATION:
Narrative structure: established setup of a mystery/conflict, now delivering a plot twist about a mathematician's sudden breakthrough.

The phrase "Then, suddenly, just weeks before her retirement," signals a dramatic turn — the scientist has received a cryptic message or discovery, implying a sudden announcement or release of findings.

Final token "retirement," ends a temporal clause ("Suddenly,") requiring a specific event — likely "she sent a paper" or "she stopped sending messages" or "after decades of silence, she released a single video." or "she transmitted a solution" or "she received a mysterious result." or "the team posted a final finding to the forum."

EDITED EXPLANATION:
Narrative structure: established setup of a mystery/conflict, now delivering a plot twist about an astrophysicist's sudden breakthrough.

The phrase "to discuss her personal life. Then, suddenly," signals a dramatic turn — the scientist has received a cryptic message or discovery, implying a sudden announcement or release of findings.

Final token "suddenly," ends a temporal clause ("Then, suddenly,") requiring a specific event — likely "she sent a paper" or "she stopped sending messages" or "after decades of silence, she released a single video." or "she transmitted a solution" or "she received a mysterious result." or "the team posted a final finding to the forum."


EXAMPLE 2

PASSAGE:
user
Describe how weather forecasting models are built.
model
Okay, let's break down how weather forecasting models are built. It's a hugely complex process, involving physics, mathematics, computer science, and a lot of data. Here's a layered explanation, from the fundamental principles to the practical implementation.  I'll organize it into sections: 1. The Foundation (Physics & Math), 2. Data Collection & Assimilation, 3. The Model It

ORIGINAL EXPLANATION:
Structured explainer article format: technical weather modeling breakdown, with "The Model It" heading signaling a breakdown of the forecast system.

The phrase "First, the model itself. The Model It" sets up a list of components — the weather model, the forecast engine, or the model's internal layers.

Final token "It" opens a topic phrase ("The Model It...The Model It"), immediately requiring a noun phrase like "self" or "self: the simulation" — likely "self, including..." or "Itself" or " itself: the actual code/layers" or "A few things" — naming the model's output or categories.

EDITED EXPLANATION:
Structured explainer article format: technical weather modeling breakdown, with "The Model It" heading signaling a breakdown of the forecast system.

The phrase "sections: 1. The Foundation (Physics & Math)," sets up a list of components — the weather model, the forecast engine, or the model's internal layers.

Final token "It" opens a topic phrase ("Assimilation, 3. The Model It"), immediately requiring a noun phrase like "self" or "self: the simulation" — likely "self, including..." or "Itself" or " itself: the actual code/layers" or "A few things" — naming the model's output or categories.


EXAMPLE 3

PASSAGE:
user
Describe the concept of market capitalization.
model
Okay, let's break down the concept of market capitalization (often shortened to "market cap"). It's a really important term in finance and investing. Here's a comprehensive explanation, covering what it is, how it's calculated, what it means, and how it's used:

**1. What is Market Capitalization?**

Market capitalization is essentially a measure of a company's **total value in

ORIGINAL EXPLANATION:
Structured financial/definition article format signals a formal definition of a company, with "market capitalization" as the topic.

The phrase "calculated value of a company's value in" sets up a standard financial definition of market capitalization, establishing the formula or concept.

Final token "in" opens a noun phrase ("the total value of the company's stock value in..."), requiring a noun phrase like "the stock market" or "the financial markets" — likely "the stock market" or "the public eye" or "the stock market, typically expressed as..." or "the market, assuming the company is publicly traded."

EDITED EXPLANATION:
Structured financial/definition article format signals a formal definition of a company, with "market capitalization" as the topic.

The phrase "measure of a company's **total value in" sets up a standard financial definition of market capitalization, establishing the formula or concept.

Final token "in" opens a noun phrase ("essentially a measure of a company's **total value in"), requiring a noun phrase like "the stock market" or "the financial markets" — likely "the stock market" or "the public eye" or "the stock market, typically expressed as..." or "the market, assuming the company is publicly traded."


Now correct the following explanation.

PASSAGE:
{passage}

ORIGINAL EXPLANATION:
{explanation}

Return only the edited explanation."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpaca-dir", type=Path, default=results_dir("alpaca") / "gemma12b",
                        help="soar_nla.alpaca results for one model (rollouts/ and decode/)")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: results/surgical_edits/edits_<offset+1>_<offset+limit>.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--env-file", type=Path, default=None, help="dotenv file providing OPENAI_API_KEY")
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.output is None:
        args.output = results_dir("surgical_edits") / f"edits_{args.offset + 1}_{args.offset + args.limit}.jsonl"

    from openai import OpenAI

    rollouts = read_jsonl(args.alpaca_dir / "rollouts" / "rollouts.jsonl")
    explanations = {
        row["sample_id"]: row["explanation"]
        for row in read_jsonl(args.alpaca_dir / "decode" / "explanations.jsonl")
    }
    eligible = [row for row in late_positions(rollouts) if row["sample_id"] in explanations]
    selected = eligible[args.offset : args.offset + args.limit]
    client = OpenAI(api_key=api_key("OPENAI_API_KEY", args.env_file))

    def correct(row: dict) -> dict:
        explanation = explanations[row["sample_id"]]
        prompt = PROMPT.format(passage=row["preceding_text_inclusive"], explanation=explanation)
        for attempt in range(args.retries):
            try:
                response = client.responses.create(
                    model=args.model,
                    input=prompt,
                    max_output_tokens=2048,
                    reasoning={"effort": "low"},
                )
                corrected = response.output_text.strip()
                if not corrected:
                    raise ValueError("empty correction")
                return {
                    "sample_id": row["sample_id"],
                    "sequence_token_index": row["sequence_token_index"],
                    "passage": row["preceding_text_inclusive"],
                    "original_explanation": explanation,
                    "corrected_explanation": corrected,
                    "response_id": response.id,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            except Exception as exc:
                if "credit_balance_exhausted" in str(exc) or "insufficient_quota" in str(exc):
                    raise SystemExit(str(exc)) from exc
                if attempt + 1 == args.retries:
                    raise
                time.sleep(min(60, 2**attempt + random.random()))

    output = [None] * len(selected)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(correct, row): index for index, row in enumerate(selected)}
        completed = 0
        for future in as_completed(futures):
            output[futures[future]] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(selected):
                print(f"[correct] {completed}/{len(selected)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[correct] wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
