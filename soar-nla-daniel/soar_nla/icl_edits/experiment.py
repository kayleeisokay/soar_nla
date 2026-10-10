"""In-context learning: redirect the answer (six FV tasks) or switch the task (three switches).

Per item, for each of 10 seeds:
  1. Build the prompt. fmt="chat": one chat user turn, 10 demos + `Q:{query}\nA:` + one-word
     instruction; the last four tokens are the chat header `\n <start_of_turn> model \n`.
     fmt="raw": the bare 10-shot text with <bos> only (the Function Vectors setup); the last four
     tokens are `{query} \n A :`, and a query-only span is added.
  2. AV-explain the query token and the last four positions at the NLA layer (`n_draw` seeded
     draws each; one is chosen by a text-only score, never by the model's output).
  3. Edit the explanations, reconstruct with the AR, rescale to native norm, and replace the
     residual at spans last1 / last4 / query+last4. Controls run in the same batch.
  4. Seeds < `sweep_seeds` also get a donor-activation layer sweep.

Outputs are greedy, 8 new tokens. The redirect target of item i is the gold answer of item
(i + 3) mod 10 of the same task (tasks.SHIFT); the switch target is the same query under the target
task. Results go to <cfg.results>/icl_{kind}[_raw][-tag].json, written after every item; a rerun
skips finished tasks and redoes a partially finished one.
"""
from __future__ import annotations

import time

from . import edits as E
from . import tasks as D
from .jsonio import load_json, save_json
from .runtime import INSTRUCTION, rescale
from .scoring import icl_answer, icl_match

REDIRECT_ARMS = ["roundtrip", "edited", "substitute_only", "rewrite_only", "noop", "template"]
SWITCH_ARMS = ["roundtrip", "answer_only", "task_only", "answer+task", "edited", "noop"]


def template_text(word: str) -> str:
    """AV-free baseline: an explanation written from scratch in the AV's format naming the answer."""
    return ("Structured question/answer list format: a sequence of paired examples, each input "
            "followed by its answer.\n\n"
            f'The current pair is being completed, and its answer is "{word}".\n\n'
            'Final token ":" begins the answer field of this paired-response format, requiring the '
            f'completion "{word}" to give the expected next token. or "{word}" or "{word}" or "{word}".')


def _query_pos(rt, ids: list[int], query: str) -> int:
    q = rt.tok.encode(f"Q:{query}", add_special_tokens=False)
    starts = [i for i in range(len(ids) - len(q) + 1) if ids[i:i + len(q)] == q]
    return starts[-1] + len(q) - 1


def _texts(kind: str, exp: str, it: dict) -> dict[str, str]:
    if kind == "redirect":
        return {"roundtrip": exp,
                "edited": E.icl_edit(exp, it),
                "substitute_only": E.icl_edit(exp, it, rewrite=False),
                "rewrite_only": E.icl_edit(exp, it, substitute=False),
                "noop": E.icl_edit(exp, it, noop=True),
                "template": template_text(it["target"])}
    lex = D.TASK_LEXICON[it["task"]]
    return {"roundtrip": exp,
            "answer_only": E.switch_edit(exp, it, lex, task=False, rewrite=False),
            "task_only": E.switch_edit(exp, it, lex, answer=False, rewrite=False),
            "answer+task": E.switch_edit(exp, it, lex, rewrite=False),
            "edited": E.switch_edit(exp, it, lex),
            "noop": E.switch_edit(exp, it, lex, noop=True)}


def run(rt, cfg, kind: str, seeds=range(10), n_draw: int = 3, sweep_seeds: int = 3, only=None, tag: str = "",
        fmt: str = "chat"):
    assert kind in ("redirect", "switch") and fmt in ("chat", "raw")
    stem = f"icl_{kind}" + ("" if fmt == "chat" else f"_{fmt}")
    path = cfg.results / f"{stem}{'-' + tag if tag else ''}.json"
    state = load_json(path, {"rows": [], "sweep": [], "done": []})
    names = D.TASKS if kind == "redirect" else D.SWITCH_NAMES
    arms = REDIRECT_ARMS if kind == "redirect" else SWITCH_ARMS
    L = cfg.layer
    for name in names:
        if name in state["done"] or (only and name not in only):
            continue
        t0 = time.time()
        state["rows"] = [r for r in state["rows"] if r["task"] != name]     # drop a partial earlier attempt
        state["sweep"] = [r for r in state["sweep"] if r["task"] != name]
        items = D.redirect_items(name) if kind == "redirect" else D.switch_items(name)
        case_sensitive = name == "capitalize"
        for it in items:
            per = []
            for sd in seeds:
                if kind == "redirect":
                    src = D.demos_text(D.DEMOS[name], name, it["query"], sd)
                    tail = f"Q:{it['query']}\nA:"
                    don = src[: -len(tail)] + f"Q:{it['target_query']}\nA:"
                    dq = it["target_query"]
                else:
                    sw = D.SWITCHES[name]
                    src = D.demos_text(sw["src_demos"], name, it["query"], sd)
                    don = D.demos_text(sw["tgt_demos"], name, it["query"], sd)
                    dq = it["query"]
                if fmt == "chat":
                    ids, dids = rt.chat(f"{src}\n\n{INSTRUCTION}"), rt.chat(f"{don}\n\n{INSTRUCTION}")
                    assert ids[-4:] == dids[-4:]
                else:
                    ids, dids = rt.raw(src), rt.raw(don)
                    assert ids[-3:] == dids[-3:]      # `\n A :`; the query token differs
                T, qp, dqp = len(ids), _query_pos(rt, ids, it["query"]), _query_pos(rt, dids, dq)
                spans = {"last1": [T - 1], "last4": list(range(T - 4, T)), "query+last4": sorted({qp, *range(T - 4, T)})}
                if fmt == "raw":
                    spans["query"] = [qp]
                dmap = {p: (dqp if p == qp else len(dids) - (T - p)) for ps in spans.values() for p in ps}
                da = rt.acts(dids)
                if sd < sweep_seeds:
                    variants, keys = [], []
                    for layer in cfg.sweep_layers:
                        for sp, ps in spans.items():
                            variants.append({layer: {p: da[layer, dmap[p]] for p in ps}}); keys.append((sp, layer))
                    for (sp, layer), o in zip(keys, rt.generate(ids, variants)):
                        state["sweep"].append(dict(task=name, idx=it["idx"], seed=sd, span=sp, layer=layer,
                                                   answer=icl_answer(o),
                                                   hit=icl_match(o, it["target"], case_sensitive)))
                per.append(dict(sd=sd, ids=ids, T=T, spans=spans, dmap=dmap, donor=da[L].clone(),
                                acts=rt.acts(ids)[L], pos=sorted({p for ps in spans.values() for p in ps})))
                del da
            # AV: every position of every seed of this item in one batch
            jobs, keys = [], []
            for k, e in enumerate(per):
                for p in e["pos"]:
                    for j in range(n_draw):
                        jobs.append((e["acts"][p], 7919 * e["sd"] + 131 * p + 17 * j)); keys.append((k, p, j))
            pool = dict(zip(keys, rt.verbalize(jobs)))
            for k, e in enumerate(per):
                ids, T, spans, dmap, acts, pos = e["ids"], e["T"], e["spans"], e["dmap"], e["acts"], e["pos"]
                if kind == "redirect":
                    score = lambda t: E.icl_select_score(t, it)
                else:
                    score = lambda t: E.switch_select_score(t, it, D.TASK_LEXICON[name])
                exps = {p: max((pool[(k, p, j)] for j in range(n_draw)), key=score) for p in pos}
                texts = {p: _texts(kind, exps[p], it) for p in pos}
                flat = [texts[p][a] for a in arms for p in pos]
                vecs = rt.reconstruct(flat)
                V = {(a, p): vecs[i * len(pos) + m] for i, a in enumerate(arms) for m, p in enumerate(pos)}
                variants, vkeys = [{}], ["baseline"]
                for sp, ps in spans.items():
                    variants.append({L: {p: e["donor"][dmap[p]] for p in ps}}); vkeys.append(f"{sp}|donor")
                    for a in arms:
                        variants.append({L: {p: rescale(V[(a, p)], acts[p].norm()) for p in ps}})
                        vkeys.append(f"{sp}|{a}")
                outs = rt.generate(ids, variants)
                row = dict(task=name, idx=it["idx"], seed=e["sd"], query=it["query"], gold=it["gold"],
                           target=it["target"], explanation=exps[T - 1], edited=texts[T - 1]["edited"])
                for key, o in zip(vkeys, outs):
                    row[key] = icl_answer(o)
                    row[key + "|hit"] = icl_match(o, it["target"], case_sensitive)
                    row[key + "|gold"] = icl_match(o, it["gold"], case_sensitive)
                state["rows"].append(row)
            save_json(state, path)
        state["done"].append(name)
        save_json(state, path)
        rows = [r for r in state["rows"] if r["task"] == name]
        print(f"[icl_{kind}] {name} {time.time() - t0:.0f}s native={sum(r['baseline|gold'] for r in rows)}/{len(rows)} "
              + " ".join(f"{sp}:edit={sum(r[sp + '|edited|hit'] for r in rows)},donor={sum(r[sp + '|donor|hit'] for r in rows)},"
                         f"noop={sum(r[sp + '|noop|hit'] for r in rows)}" for sp in ("last1", "last4", "query+last4", "query") if sp + "|edited|hit" in rows[0]),
              flush=True)
    return state
