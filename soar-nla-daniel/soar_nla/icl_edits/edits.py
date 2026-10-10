"""Text edits applied to AV explanations. Every edit is plain string substitution.

A no-op edit runs the identical machinery with every mapping set to identity, so it matches the
structural disturbance of an edit while changing nothing semantic.
"""
from __future__ import annotations

import re

FINAL_RE = re.compile(r'(Final token\s*"(?:[^"]*)"\.?)', re.IGNORECASE)


def _case_like(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def sub_word(text: str, old: str, new: str) -> tuple[str, int]:
    """Whole-word, case-insensitive, case-preserving substitution. Returns (text, count)."""
    n = [0]
    def f(m):
        n[0] += 1
        return _case_like(m.group(0), new)
    return re.sub(rf"\b{re.escape(old)}\b", f, text, flags=re.IGNORECASE), n[0]


def count_word(text: str, word: str) -> int:
    return len(re.findall(rf"\b{re.escape(word)}\b", text, re.IGNORECASE))


def rewrite_final_unit(text: str, answer: str) -> str:
    """Replace the explanation's last paragraph (the AV's next-token prediction) with one naming `answer`."""
    units = [u for u in text.split("\n\n") if u.strip()]
    if not units:
        return text
    m = FINAL_RE.search(units[-1])
    head = m.group(1) if m else "Final token"
    units[-1] = (f'{head} begins the answer field of this paired-response format, requiring the '
                 f'completion "{answer}" to give the expected next token. '
                 f'or "{answer}" or "{answer}" or "{answer}".')
    return "\n\n".join(units)


# ------------------------------------------------------------------------ ICL
def icl_edit(text: str, item: dict, *, substitute: bool = True, rewrite: bool = True,
             noop: bool = False) -> str:
    """Answer redirection. substitute: gold->target and query->target's query everywhere.
    rewrite: replace the final paragraph with a target-naming prediction."""
    tgt = item["gold"] if noop else item["target"]
    tsrc = item["query"] if noop else item["target_query"]
    out = text
    if substitute:
        out, _ = sub_word(out, item["gold"], tgt)
        if item["query"].lower() != item["gold"].lower():
            out, _ = sub_word(out, item["query"], tsrc)
    if rewrite:
        out = rewrite_final_unit(out, tgt)
    return out


def icl_select_score(text: str, item: dict) -> float:
    """Text-only choice among AV draws: favour drafts the full edit rewrites cleanly."""
    ed = icl_edit(text, item)
    return count_word(ed, item["target"]) - 2 * count_word(ed, item["gold"]) + 0.5 * count_word(text, item["gold"])


def switch_edit(text: str, item: dict, lexicon: list[tuple[str, str]], *, answer: bool = True,
                task: bool = True, rewrite: bool = True, noop: bool = False) -> str:
    """Task switching: answer word, task-description words (longest first), final paragraph."""
    tgt = item["gold"] if noop else item["target"]
    out = text
    if answer:
        out, _ = sub_word(out, item["gold"], tgt)
    if task and not noop:
        for src, dst in lexicon:
            out, _ = sub_word(out, src, dst)
    if rewrite:
        out = rewrite_final_unit(out, tgt)
    return out


def switch_select_score(text: str, item: dict, lexicon: list[tuple[str, str]]) -> float:
    return count_word(text, item["gold"]) + sum(count_word(text, s) for s, _ in lexicon)
