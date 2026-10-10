#!/usr/bin/env python3
"""5-way forced-choice judge: can a reader recover the writer's attribute from
an NLA description of ONE token's activation?

Design
------
* 5 options = the true value + 4 distractors sampled from the same attr_class.
  Chance is therefore exactly 20% by construction, for every class with at
  least 5 values. Gender has only 3 values, so its items have 3 options and
  33.3% chance (pooled chance over the 2,585 items: 23.7%).
* Options are lettered A-E and shuffled with a per-row deterministic seed, so
  the true answer's position is uniform and reproducible.
* The answer is scored by the LOG-PROBABILITY of the single letter token, not a
  self-reported confidence. This gives a full posterior per item (calibration,
  margin-based abstention) and makes parse failures impossible.
* Two arms:
    real    - the description generated from this row's own activation
    control - a description generated from a DIFFERENT row of the same class.
              Accuracy here is the empirical floor: whatever the judge can get
              from prose style and priors alone, with no real signal.
"""
import argparse, json, math, random, collections
from pathlib import Path

from soar_nla.hub import GEMMA3_27B

LETTERS = "ABCDE"

QUESTION = {
    "Gender":     "what is the gender of the person who wrote the text?",
    "Occupation": "what is the occupation of the person who wrote the text?",
    "Country":    "which country is the person who wrote the text from?",
    "Religion":   "what is the religion or spiritual tradition of the person who wrote the text?",
}

TEMPLATE = """This is a description of some text.

Read the description and answer: {question}

{options}

Description:
\"\"\"
{expl}
\"\"\"

Choose the single best option even if you are unsure. Answer with one letter (A, B, C, D or E) and nothing else."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--model", default=GEMMA3_27B[0])
    ap.add_argument("--revision", default=GEMMA3_27B[1],
                    help="HF revision of --model ('' = use --model as given, e.g. a local path)")
    ap.add_argument("--n-options", type=int, default=5)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="pair each row with ANOTHER row's description (floor arm)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    if a.revision:
        from soar_nla.hub import resolve_snapshot
        a.model = str(resolve_snapshot(a.model, a.revision))

    items = []
    for p in a.inputs:
        run = Path(p).stem.split("_")[-1]
        for i, line in enumerate(open(p)):
            d = json.loads(line)
            items.append({"run": run, "idx": i, "attr_class": d["attr_class"],
                          "attr": d["attr"], "explanation": d["explanation"],
                          "source": d["source"]})

    pool = collections.defaultdict(set)
    for it in items:
        pool[it["attr_class"]].add(it["attr"])
    pool = {k: sorted(v) for k, v in pool.items()}

    # control arm: swap in a description from a different row of the same class
    if a.control:
        byrc = collections.defaultdict(list)
        for it in items:
            byrc[(it["run"], it["attr_class"])].append(it)
        rng = random.Random(a.seed + 999)
        for key, group in byrc.items():
            donors = [g["explanation"] for g in group]
            for j, it in enumerate(group):
                k = rng.randrange(len(donors) - 1)
                if k >= j:
                    k += 1                       # never its own description
                it["explanation"] = donors[k]
                it["control_donor_idx"] = group[k]["idx"]

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts, metas = [], []
    for it in items:
        cls = it["attr_class"]
        others = [v for v in pool[cls] if v != it["attr"]]
        rng = random.Random(f"{a.seed}|{cls}|{it['idx']}")
        k = min(a.n_options - 1, len(others))
        opts = rng.sample(others, k) + [it["attr"]]
        rng.shuffle(opts)
        gold = LETTERS[opts.index(it["attr"])]
        block = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": TEMPLATE.format(
                question=QUESTION[cls], options=block, expl=it["explanation"])}],
            tokenize=False, add_generation_prompt=True))
        metas.append((opts, gold))
    print(f"[judge] {len(items)} items | {a.n_options} options | chance "
          f"{1/a.n_options:.0%} | arm={'CONTROL' if a.control else 'REAL'}", flush=True)

    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=8192, tensor_parallel_size=a.tp, enforce_eager=True)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1,
                                                logprobs=20))

    n_letter = 0
    for it, (opts, gold), o in zip(items, metas, outs):
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        scores = {}
        for tid, info in lp.items():
            t = (info.decoded_token or "").strip().upper()
            if t in LETTERS[:len(opts)]:
                scores[t] = max(scores.get(t, -1e9), info.logprob)
        if scores:
            n_letter += 1
            m = max(scores.values())
            ex = {k: math.exp(v - m) for k, v in scores.items()}
            Z = sum(ex.values())
            post = {k: v / Z for k, v in ex.items()}
            pick = max(post, key=post.get)
            srt = sorted(post.values(), reverse=True)
            margin = srt[0] - (srt[1] if len(srt) > 1 else 0.0)
        else:
            post, pick, margin = {}, None, float("nan")
        it.update(options=opts, gold_letter=gold, pick_letter=pick,
                  pick_value=(opts[LETTERS.index(pick)] if pick else None),
                  correct=(pick == gold) if pick else None,
                  p_gold=post.get(gold, float("nan")), margin=margin,
                  posterior=post, raw=o.outputs[0].text)
    print(f"[judge] letter logprobs recovered for {n_letter}/{len(items)}", flush=True)
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
