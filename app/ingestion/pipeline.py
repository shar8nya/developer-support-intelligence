"""Ingestion pipeline: normalise -> dedupe (content hash) -> chunk -> embed -> store."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable
from uuid import UUID

from app.config import Settings
from app.exceptions import AppError, IngestionError
from app.ingestion.chunker import chunk_markdown
from app.ingestion.connectors import (
    Connector, GitHubDocsConnector, GitHubIssuesConnector, LocalConnector, UrlConnector,
)
from app.ingestion.normalize import content_hash, normalize_markdown
from app.providers.base import EmbeddingProvider
from app.repositories.base import Repository
from app.schemas import IngestRequest, IngestStats, JobStatus, RawDocument

log = logging.getLogger(__name__)


def resolve_local_path(path: str, settings: Settings) -> Path:
    p = Path(path)
    if not p.is_absolute():
        # relative paths are resolved against the project root (so `data/sample_docs` just works)
        p = settings.local_ingest_root.parent / p
    p = p.resolve()
    if not settings.allow_any_local_path:
        root = settings.local_ingest_root.resolve()
        if root != p and root not in p.parents:
            raise IngestionError(f"Local ingestion is restricted to {root} (set ALLOW_ANY_LOCAL_PATH=true to lift this).")
    return p


def build_connector(req: IngestRequest, settings: Settings) -> Connector:
    if req.source == "local":
        return LocalConnector(resolve_local_path(req.path or "", settings))
    if req.source == "github_docs":
        return GitHubDocsConnector(req.repo or "", req.branch, req.path_prefix, req.max_files, settings.github_token)
    if req.source == "github_issues":
        return GitHubIssuesConnector(req.repo or "", req.state, req.labels, req.max_issues,
                                     req.include_comments, settings.github_token)
    if req.source == "url":
        return UrlConnector(req.urls or [])
    raise IngestionError(f"Unknown source '{req.source}'")


class IngestionPipeline:
    def __init__(self, repo: Repository, embedder: EmbeddingProvider, settings: Settings):
        self.repo, self.embedder, self.settings = repo, embedder, settings

    @property
    def _fingerprint(self) -> str:
        """Anything that changes the stored chunks must change the hash, so config changes re-index."""
        s = self.settings
        return f"{s.chunk_max_tokens}:{s.chunk_overlap_tokens}:{self.embedder.name}:{getattr(self.embedder, 'model', '')}:{self.embedder.dimensions}"

    def ingest_documents(self, docs: Iterable[RawDocument], force: bool = False) -> IngestStats:
        stats = IngestStats()
        for raw in docs:
            stats.documents_seen += 1
            try:
                self._ingest_one(raw, force, stats)
            except AppError as exc:
                stats.documents_failed += 1
                stats.errors.append(f"{raw.source_uri}: {exc.message}")
            except Exception as exc:  # noqa: BLE001 - one bad document must not abort the run
                log.exception("failed to ingest %s", raw.source_uri)
                stats.documents_failed += 1
                stats.errors.append(f"{raw.source_uri}: {exc}")
            stats.errors = stats.errors[:20]
        return stats

    def _ingest_one(self, raw: RawDocument, force: bool, stats: IngestStats) -> None:
        content = normalize_markdown(raw.content)
        if len(content) < 10:
            raise IngestionError("empty content")
        raw = raw.model_copy(update={"content": content})
        digest = content_hash(raw.title, content, self._fingerprint)
        existing = self.repo.get_document_by_uri(raw.source_uri)
        if existing and existing.content_hash == digest and not force:
            stats.documents_unchanged += 1
            return
        chunks = chunk_markdown(content, title=raw.title, max_tokens=self.settings.chunk_max_tokens,
                                overlap_tokens=self.settings.chunk_overlap_tokens)
        if not chunks:
            raise IngestionError("no chunks produced")
        embeddings = self.embedder.embed_documents([c.content for c in chunks])
        _, action = self.repo.upsert_document(raw, digest, chunks, embeddings)
        stats.chunks_written += len(chunks)
        if action == "created":
            stats.documents_new += 1
        else:
            stats.documents_updated += 1

    def run(self, req: IngestRequest) -> IngestStats:
        return self.ingest_documents(build_connector(req, self.settings).fetch(), force=req.force)

    def run_job(self, job_id: UUID, req: IngestRequest) -> None:
        """Execute an ingestion job, recording status in the repository. Never raises."""
        try:
            self.repo.update_job(job_id, status=JobStatus.running)
            stats = self.run(req)
            if stats.documents_seen > 0 and stats.documents_failed == stats.documents_seen:
                self.repo.update_job(job_id, JobStatus.failed, stats, "all documents failed; see stats.errors")
            else:
                self.repo.update_job(job_id, JobStatus.succeeded, stats)
        except AppError as exc:
            self._fail(job_id, exc.message)
        except Exception as exc:  # noqa: BLE001
            log.exception("ingestion job %s crashed", job_id)
            self._fail(job_id, f"{type(exc).__name__}: {exc}")

    def _fail(self, job_id: UUID, message: str) -> None:
        try:
            self.repo.update_job(job_id, JobStatus.failed, None, message)
        except Exception:  # noqa: BLE001
            log.exception("could not record failure for job %s", job_id)
