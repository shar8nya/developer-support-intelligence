"""Pydantic models shared by the API, services and repositories."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class SourceType(str, Enum):
    docs = "docs"
    github_issue = "github_issue"
    web = "web"
    local = "local"


# ---------------------------------------------------------------- ingestion
class RawDocument(BaseModel):
    """A document as produced by a connector, before chunking."""
    source_type: SourceType
    source_uri: str                    # stable unique id (URL or path)
    title: str
    content: str
    url: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_index: int
    content: str
    heading_path: str = ""
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredDocument(BaseModel):
    id: UUID
    source_type: SourceType
    source_uri: str
    title: str
    url: Optional[str] = None
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- retrieval
class RetrievedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    heading_path: str = ""
    title: str
    url: Optional[str] = None
    source_uri: str
    source_type: SourceType
    similarity: float = 0.0            # dense cosine similarity (0..1, higher = closer)
    score: float = 0.0                 # final ranking score (RRF / rerank)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- API: chat
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    use_hybrid: Optional[bool] = None
    use_rerank: Optional[bool] = None
    source_types: Optional[list[SourceType]] = None


class Citation(BaseModel):
    index: int                         # the [n] marker used in the answer
    chunk_id: UUID
    document_id: UUID
    title: str
    url: Optional[str] = None
    source_type: SourceType
    heading_path: str = ""
    snippet: str
    similarity: float
    score: float


class SearchResult(BaseModel):
    chunk_id: UUID
    document_id: UUID
    title: str
    url: Optional[str] = None
    source_uri: str
    source_type: SourceType
    heading_path: str = ""
    content: str
    similarity: float
    score: float


class ChatResponse(BaseModel):
    interaction_id: UUID
    answer: str
    abstained: bool
    abstain_reason: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[SearchResult] = Field(default_factory=list)
    latency_ms: int
    mode: Literal["demo", "live"]


# ---------------------------------------------------------------- API: search
class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    use_hybrid: Optional[bool] = None
    use_rerank: Optional[bool] = None
    source_types: Optional[list[SourceType]] = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    latency_ms: int


# ---------------------------------------------------------------- API: ingest
class IngestRequest(BaseModel):
    source: Literal["local", "github_docs", "github_issues", "url"]
    force: bool = False                # re-embed even if content is unchanged

    # local
    path: Optional[str] = Field(default=None, description="Directory of .md/.mdx/.txt/.json files")
    # github_docs / github_issues
    repo: Optional[str] = Field(default=None, description="owner/name, e.g. supabase/supabase")
    branch: str = "master"
    path_prefix: str = ""
    max_files: int = Field(default=50, ge=1, le=2000)
    state: Literal["open", "closed", "all"] = "all"
    labels: Optional[str] = None
    max_issues: int = Field(default=50, ge=1, le=1000)
    include_comments: bool = True
    # url
    urls: Optional[list[str]] = None

    @model_validator(mode="after")
    def _check_required(self) -> "IngestRequest":
        if self.source == "local" and not self.path:
            raise ValueError("'path' is required for source=local")
        if self.source in ("github_docs", "github_issues") and not (self.repo and "/" in self.repo):
            raise ValueError("'repo' (owner/name) is required for GitHub sources")
        if self.source == "url" and not self.urls:
            raise ValueError("'urls' is required for source=url")
        return self


class IngestStats(BaseModel):
    documents_seen: int = 0
    documents_new: int = 0
    documents_updated: int = 0
    documents_unchanged: int = 0
    documents_failed: int = 0
    chunks_written: int = 0
    errors: list[str] = Field(default_factory=list)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class IngestJob(BaseModel):
    id: UUID
    status: JobStatus
    source: str
    params: dict[str, Any] = Field(default_factory=dict)
    stats: IngestStats = Field(default_factory=IngestStats)
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------- API: feedback / health
class FeedbackRequest(BaseModel):
    interaction_id: UUID
    rating: Literal[-1, 1]
    comment: Optional[str] = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    id: UUID
    status: Literal["recorded"] = "recorded"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    mode: Literal["demo", "live"]
    version: str
    database: str
    embedding_provider: str
    llm_provider: str
    documents: int = 0
    chunks: int = 0
    problems: list[str] = Field(default_factory=list)
