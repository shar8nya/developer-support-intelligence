"""Dependency container: wires settings -> repository/providers -> services."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import PROJECT_ROOT, Settings
from app.ingestion.pipeline import IngestionPipeline
from app.providers.base import EmbeddingProvider, LLMProvider
from app.providers.factory import build_embedder, build_llm
from app.repositories.base import Repository
from app.repositories.factory import build_repository
from app.schemas import IngestRequest
from app.services.rag import RAGService
from app.services.retrieval import RetrievalService

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    repo: Repository
    embedder: EmbeddingProvider
    llm: LLMProvider
    retrieval: RetrievalService
    rag: RAGService
    pipeline: IngestionPipeline


def build_container(settings: Settings, auto_ingest_demo: bool = True, *, repo: Repository | None = None,
                    embedder: EmbeddingProvider | None = None, llm: LLMProvider | None = None) -> Container:
    repo = repo or build_repository(settings)
    embedder = embedder or build_embedder(settings)
    llm = llm or build_llm(settings)
    retrieval = RetrievalService(repo, embedder, settings)
    container = Container(settings, repo, embedder, llm, retrieval,
                          RAGService(repo, retrieval, llm, settings),
                          IngestionPipeline(repo, embedder, settings))
    if settings.is_demo and settings.demo_auto_ingest and auto_ingest_demo:
        for sub in ("sample_docs", "sample_issues"):
            path = PROJECT_ROOT / "data" / sub
            if path.exists():
                stats = container.pipeline.run(IngestRequest(source="local", path=str(path)))
                log.info("demo corpus %s: %s new docs, %s chunks", sub, stats.documents_new, stats.chunks_written)
    return container
