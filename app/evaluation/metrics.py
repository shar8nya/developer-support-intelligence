"""Retrieval + answer metrics. Pure functions, no I/O."""
from __future__ import annotations

from typing import Iterable, Sequence

from app.text_utils import tokenize


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def recall_at_k(ranked_docs: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top-k (document-level, deduped)."""
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel & set(ranked_docs[:k])) / len(rel)


def hit_at_k(ranked_docs: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if at least one relevant document is in the top-k."""
    return 1.0 if set(relevant) & set(ranked_docs[:k]) else 0.0


def reciprocal_rank(ranked_docs: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for i, d in enumerate(ranked_docs, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ----------------------------------------------------------- answer-level checks
def keyword_coverage(answer: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 1.0
    a = answer.lower()
    return sum(1 for k in keywords if k.lower() in a) / len(keywords)


def citation_precision(cited_docs: Sequence[str], relevant: Iterable[str]) -> float:
    """Fraction of cited documents that are truly relevant (1.0 when nothing is cited)."""
    if not cited_docs:
        return 1.0
    rel = set(relevant)
    return sum(1 for d in cited_docs if d in rel) / len(cited_docs)


def sentence_support(sentence: str, evidence: str, min_overlap: float = 0.6) -> bool:
    """Lexical groundedness proxy: are >= min_overlap of the sentence's content words in the evidence?"""
    words = set(tokenize(sentence))
    if not words:
        return True
    return len(words & set(tokenize(evidence))) / len(words) >= min_overlap
