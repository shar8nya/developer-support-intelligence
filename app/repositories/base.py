from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional
from uuid import UUID

from app.schemas import (
    Chunk, IngestJob, IngestStats, JobStatus, RawDocument, RetrievedChunk, SourceType, StoredDocument,
)


class Repository(ABC):
    """Storage abstraction. Two implementations: InMemoryRepository and SupabaseRepository."""

    backend: str

    @abstractmethod
    def ping(self) -> None:
        """Raise RepositoryError if the store is unreachable."""

    @abstractmethod
    def stats(self) -> dict[str, int]: ...

    # ---- documents ----------------------------------------------------------
    @abstractmethod
    def get_document_by_uri(self, source_uri: str) -> Optional[StoredDocument]: ...

    @abstractmethod
    def upsert_document(
        self, raw: RawDocument, content_hash: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> tuple[StoredDocument, Literal["created", "updated"]]:
        """Atomically create the document or replace its chunks."""

    @abstractmethod
    def delete_document(self, source_uri: str) -> bool: ...

    # ---- retrieval ----------------------------------------------------------
    @abstractmethod
    def vector_search(
        self, embedding: list[float], top_k: int, source_types: Optional[list[SourceType]] = None
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    def hybrid_search(
        self, query: str, embedding: list[float], top_k: int,
        source_types: Optional[list[SourceType]] = None,
    ) -> list[RetrievedChunk]: ...

    # ---- jobs ---------------------------------------------------------------
    @abstractmethod
    def create_job(self, source: str, params: dict[str, Any]) -> IngestJob: ...

    @abstractmethod
    def update_job(
        self, job_id: UUID, status: Optional[JobStatus] = None,
        stats: Optional[IngestStats] = None, error: Optional[str] = None,
    ) -> IngestJob: ...

    @abstractmethod
    def get_job(self, job_id: UUID) -> Optional[IngestJob]: ...

    # ---- chat log & feedback --------------------------------------------------
    @abstractmethod
    def save_interaction(
        self, question: str, answer: str, abstained: bool, citations: list[dict], latency_ms: int
    ) -> UUID: ...

    @abstractmethod
    def save_feedback(self, interaction_id: UUID, rating: int, comment: Optional[str]) -> UUID:
        """Raises NotFoundError if the interaction does not exist."""

    def close(self) -> None:  # pragma: no cover - optional
        pass
