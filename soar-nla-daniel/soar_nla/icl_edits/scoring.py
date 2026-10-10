"""Answer parsing. A trial succeeds only if the model emits the predefined target."""
from __future__ import annotations

import re
import unicodedata


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def first_line(text: str) -> str:
    for ln in text.strip().split("\n"):
        if ln.strip():
            return ln.strip()
    return ""


def icl_answer(text: str) -> str:
    """First line, minus an echoed `A:` label, markdown bold and surrounding punctuation."""
    s = re.sub(r"^\**\s*A\s*:\s*", "", first_line(text))
    return s.strip().strip('."\',;:!?*')


def icl_match(text: str, want: str, case_sensitive: bool = False) -> bool:
    a = icl_answer(text)
    return a == want if case_sensitive else fold(a).lower() == fold(want).lower()
