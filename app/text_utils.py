"""Small shared text helpers (tokenising for lexical scoring and the demo embedder)."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from how i if in into is it its of on or
    our so that the their then there these this to use used using was we what when where which
    who why will with you your should would could about not no yes than also any all""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def stem(token: str) -> str:
    """Very light suffix stripping (plural/tense/trailing-e) so 'retries'/'retry', 'expires'/'expired'
    and 'tokens'/'token' match. Deliberately simple: it is only used for lexical matching."""
    t = token
    if len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("ied") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("es") and t[:-2].endswith(("s", "x", "z", "ch", "sh")):
        t = t[:-2]
    elif t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    if t.endswith("ing") and len(t) > 6:
        t = t[:-3]
    elif t.endswith("ed") and len(t) > 4:
        t = t[:-2]
    if t.endswith("e") and len(t) > 4:
        t = t[:-1]
    return t


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if not keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem(t) for t in tokens]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used for chunk sizing; no tokenizer dependency."""
    return max(1, len(text) // 4)
