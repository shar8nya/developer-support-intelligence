from __future__ import annotations

from typing import Optional

from app.config import Settings
from app.providers.base import EmbeddingProvider
from app.repositories.base import Repository
from app.schemas import RetrievedChunk, SourceType
from app.services.reranker import rerank


class RetrievalService:
    def __init__(self, repo: Repository, embedder: EmbeddingProvider, settings: Settings):
        self.repo, self.embedder, self.settings = repo, embedder, settings

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_hybrid: Optional[bool] = None,
        use_rerank: Optional[bool] = None,
        source_types: Optional[list[SourceType]] = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.top_k
        hybrid = self.settings.use_hybrid if use_hybrid is None else use_hybrid
        do_rerank = self.settings.use_rerank if use_rerank is None else use_rerank
        pool = min(top_k * 3, 60) if do_rerank else top_k       # over-fetch candidates for the reranker

        embedding = self.embedder.embed_query(query)
        if hybrid:
            hits = self.repo.hybrid_search(query, embedding, pool, source_types)
        else:
            hits = self.repo.vector_search(embedding, pool, source_types)
        return rerank(query, hits, top_k) if do_rerank else hits[:top_k]
