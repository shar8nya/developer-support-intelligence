"""Lightweight, dependency-free reranker.

Blends the dense similarity with lexical evidence (query-term coverage in the chunk text and in
the title / heading path). It is cheap and deterministic. For higher quality, swap in a
cross-encoder (e.g. Cohere Rerank, bge-reranker) by implementing `rerank()` with the same signature.
"""
from __future__ import annotations

from app.schemas import RetrievedChunk
from app.text_utils import tokenize


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    q_terms = set(tokenize(query))
    if not q_terms or not chunks:
        return chunks[:top_k]
    rescored: list[RetrievedChunk] = []
    for c in chunks:
        body = set(tokenize(c.content))
        head = set(tokenize(f"{c.title} {c.heading_path}"))
        coverage = len(q_terms & body) / len(q_terms)
        head_cov = len(q_terms & head) / len(q_terms)
        score = 0.5 * max(c.similarity, 0.0) + 0.4 * coverage + 0.1 * head_cov
        rescored.append(c.model_copy(update={"score": score}))
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored[:top_k]
