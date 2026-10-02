"""In-memory repository: used by demo mode, unit tests and the offline evaluation.

Its retrieval semantics deliberately mirror the SQL functions (cosine similarity,
BM25-style keyword ranking, Reciprocal Rank Fusion with k=60).
"""
from __future__ import annotations

import math
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

import numpy as np

from app.exceptions import NotFoundError
from app.repositories.base import Repository
from app.schemas import (
    Chunk, IngestJob, IngestStats, JobStatus, RawDocument, RetrievedChunk, SourceType, StoredDocument,
)
from app.text_utils import tokenize


class _StoredChunk:
    __slots__ = ("id", "document_id", "chunk", "embedding", "terms")

    def __init__(self, document_id: UUID, chunk: Chunk, embedding: list[float]):
        self.id = uuid4()
        self.document_id = document_id
        self.chunk = chunk
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.terms = Counter(tokenize(chunk.content))


class InMemoryRepository(Repository):
    backend = "memory"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._docs: dict[str, StoredDocument] = {}          # by source_uri
        self._chunks: dict[UUID, list[_StoredChunk]] = {}   # by document id
        self._jobs: dict[UUID, IngestJob] = {}
        self._interactions: dict[UUID, dict] = {}
        self._feedback: list[dict] = []

    def ping(self) -> None:
        return None

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"documents": len(self._docs), "chunks": sum(len(v) for v in self._chunks.values())}

    # ---- documents ----
    def get_document_by_uri(self, source_uri: str) -> Optional[StoredDocument]:
        with self._lock:
            return self._docs.get(source_uri)

    def upsert_document(self, raw: RawDocument, content_hash: str, chunks: list[Chunk],
                        embeddings: list[list[float]]) -> tuple[StoredDocument, Literal["created", "updated"]]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks/embeddings length mismatch")
        with self._lock:
            existing = self._docs.get(raw.source_uri)
            doc_id = existing.id if existing else uuid4()
            doc = StoredDocument(id=doc_id, source_type=raw.source_type, source_uri=raw.source_uri,
                                 title=raw.title, url=raw.url, content_hash=content_hash,
                                 metadata=raw.metadata)
            self._docs[raw.source_uri] = doc
            self._chunks[doc_id] = [_StoredChunk(doc_id, c, e) for c, e in zip(chunks, embeddings)]
            return doc, ("updated" if existing else "created")

    def delete_document(self, source_uri: str) -> bool:
        with self._lock:
            doc = self._docs.pop(source_uri, None)
            if doc:
                self._chunks.pop(doc.id, None)
            return doc is not None

    # ---- retrieval ----
    def _candidates(self, source_types: Optional[list[SourceType]]) -> list[tuple[StoredDocument, _StoredChunk]]:
        allowed = set(source_types) if source_types else None
        out = []
        for doc in self._docs.values():
            if allowed and doc.source_type not in allowed:
                continue
            out.extend((doc, ch) for ch in self._chunks.get(doc.id, []))
        return out

    @staticmethod
    def _to_result(doc: StoredDocument, ch: _StoredChunk, sim: float, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=ch.id, document_id=doc.id, content=ch.chunk.content,
            heading_path=ch.chunk.heading_path, title=doc.title, url=doc.url,
            source_uri=doc.source_uri, source_type=doc.source_type,
            similarity=float(sim), score=float(score), metadata=ch.chunk.metadata,
        )

    def _dense_ranked(self, embedding, cands):
        q = np.asarray(embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        rows = []
        for doc, ch in cands:
            n = float(np.linalg.norm(ch.embedding)) or 1.0
            rows.append((float(np.dot(q, ch.embedding)) / (qn * n), doc, ch))
        rows.sort(key=lambda r: r[0], reverse=True)
        return rows

    def vector_search(self, embedding, top_k, source_types=None):
        with self._lock:
            ranked = self._dense_ranked(embedding, self._candidates(source_types))[:top_k]
            return [self._to_result(d, c, s, s) for s, d, c in ranked]

    def hybrid_search(self, query, embedding, top_k, source_types=None):
        with self._lock:
            cands = self._candidates(source_types)
            if not cands:
                return []
            dense = self._dense_ranked(embedding, cands)
            sim_by_chunk = {c.id: s for s, _, c in dense}
            pool = top_k * 2

            # BM25 keyword ranking over the candidate set
            q_terms = set(tokenize(query))
            n_docs = len(cands)
            avg_len = sum(sum(c.terms.values()) for _, c in cands) / n_docs or 1.0
            df = {t: sum(1 for _, c in cands if t in c.terms) for t in q_terms}
            bm25 = []
            for doc, ch in cands:
                length = sum(ch.terms.values()) or 1
                score = 0.0
                for t in q_terms:
                    tf = ch.terms.get(t, 0)
                    if not tf:
                        continue
                    idf = math.log(1 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
                    score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / avg_len))
                if score > 0:
                    bm25.append((score, doc, ch))
            bm25.sort(key=lambda r: r[0], reverse=True)

            rrf: dict[UUID, float] = {}
            objs: dict[UUID, tuple[StoredDocument, _StoredChunk]] = {}
            for rank, (_, d, c) in enumerate(bm25[:pool], start=1):
                rrf[c.id] = rrf.get(c.id, 0.0) + 1.0 / (60 + rank)
                objs[c.id] = (d, c)
            for rank, (_, d, c) in enumerate(dense[:pool], start=1):
                rrf[c.id] = rrf.get(c.id, 0.0) + 1.0 / (60 + rank)
                objs[c.id] = (d, c)
            top = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            return [self._to_result(*objs[cid], sim_by_chunk[cid], score) for cid, score in top]

    # ---- jobs ----
    def create_job(self, source: str, params: dict[str, Any]) -> IngestJob:
        now = datetime.now(timezone.utc)
        job = IngestJob(id=uuid4(), status=JobStatus.queued, source=source, params=params,
                        created_at=now, updated_at=now)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def update_job(self, job_id, status=None, stats=None, error=None) -> IngestJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFoundError(f"Ingestion job {job_id} not found")
            updated = job.model_copy(update={
                "status": status or job.status,
                "stats": stats or job.stats,
                "error": error if error is not None else job.error,
                "updated_at": datetime.now(timezone.utc),
            })
            self._jobs[job_id] = updated
            return updated

    def get_job(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    # ---- chat log / feedback ----
    def save_interaction(self, question, answer, abstained, citations, latency_ms) -> UUID:
        iid = uuid4()
        with self._lock:
            self._interactions[iid] = dict(question=question, answer=answer, abstained=abstained,
                                           citations=citations, latency_ms=latency_ms)
        return iid

    def save_feedback(self, interaction_id, rating, comment) -> UUID:
        with self._lock:
            if interaction_id not in self._interactions:
                raise NotFoundError("Unknown interaction_id")
            fid = uuid4()
            self._feedback.append(dict(id=fid, interaction_id=interaction_id, rating=rating, comment=comment))
            return fid
