"""Supabase / PostgreSQL + pgvector repository (psycopg 3 + connection pool).

Connects with a normal Postgres connection string, so it works with Supabase (direct,
session pooler or transaction pooler) and with any local PostgreSQL that has pgvector.
"""
from __future__ import annotations

import json
from typing import Any, Literal, Optional
from uuid import UUID

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.exceptions import NotFoundError, RepositoryError
from app.repositories.base import Repository
from app.schemas import (
    Chunk, IngestJob, IngestStats, JobStatus, RawDocument, RetrievedChunk, SourceType, StoredDocument,
)


def _vec(v: list[float]) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


def _types(source_types: Optional[list[SourceType]]) -> Optional[list[str]]:
    return [SourceType(s).value for s in source_types] if source_types else None


class SupabaseRepository(Repository):
    backend = "supabase"

    def __init__(self, database_url: str, min_size: int = 1, max_size: int = 5):
        if not database_url:
            raise RepositoryError("DATABASE_URL is not set")
        self._pool = ConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            open=False,
            timeout=15,
            configure=self._configure,
            # prepare_threshold=None -> no server-side prepared statements, required by
            # Supabase's transaction pooler (port 6543); harmless elsewhere.
            kwargs={"row_factory": dict_row, "prepare_threshold": None, "autocommit": False},
        )
        try:
            self._pool.open(wait=True, timeout=15)
        except Exception as exc:  # noqa: BLE001
            raise RepositoryError(f"Could not connect to the database: {exc}") from exc

    @staticmethod
    def _configure(conn: psycopg.Connection) -> None:
        # register_vector needs the extension to exist -> tolerate a not-yet-migrated database
        try:
            register_vector(conn)
        except psycopg.ProgrammingError:
            pass
        conn.commit()

    def _wrap(self, exc: Exception) -> RepositoryError:
        return RepositoryError(f"Database error: {exc}")

    def close(self) -> None:
        self._pool.close()

    def ping(self) -> None:
        try:
            with self._pool.connection() as conn:
                conn.execute("select 1")
        except Exception as exc:  # noqa: BLE001
            raise self._wrap(exc) from exc

    def stats(self) -> dict[str, int]:
        try:
            with self._pool.connection() as conn:
                d = conn.execute("select count(*) as n from public.documents").fetchone()["n"]
                c = conn.execute("select count(*) as n from public.document_chunks").fetchone()["n"]
            return {"documents": d, "chunks": c}
        except psycopg.Error as exc:
            raise self._wrap(exc) from exc

    # ---- documents ----
    @staticmethod
    def _doc(row: dict) -> StoredDocument:
        return StoredDocument(id=row["id"], source_type=row["source_type"], source_uri=row["source_uri"],
                              title=row["title"], url=row["url"], content_hash=row["content_hash"],
                              metadata=row["metadata"] or {})

    def get_document_by_uri(self, source_uri: str) -> Optional[StoredDocument]:
        try:
            with self._pool.connection() as conn:
                row = conn.execute("select * from public.documents where source_uri = %s",
                                   (source_uri,)).fetchone()
            return self._doc(row) if row else None
        except psycopg.Error as exc:
            raise self._wrap(exc) from exc

    def upsert_document(self, raw: RawDocument, content_hash: str, chunks: list[Chunk],
                        embeddings: list[list[float]]) -> tuple[StoredDocument, Literal["created", "updated"]]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks/embeddings length mismatch")
        try:
            with self._pool.connection() as conn:  # one transaction: commit on success, rollback on error
                row = conn.execute(
                    """
                    insert into public.documents (source_type, source_uri, title, url, content_hash, metadata)
                    values (%s, %s, %s, %s, %s, %s)
                    on conflict (source_uri) do update set
                        source_type = excluded.source_type, title = excluded.title, url = excluded.url,
                        content_hash = excluded.content_hash, metadata = excluded.metadata,
                        updated_at = now()
                    returning *, (xmax = 0) as inserted
                    """,
                    (raw.source_type.value, raw.source_uri, raw.title, raw.url, content_hash,
                     Jsonb(raw.metadata)),
                ).fetchone()
                conn.execute("delete from public.document_chunks where document_id = %s", (row["id"],))
                if chunks:
                    with conn.cursor() as cur:
                        cur.executemany(
                            """insert into public.document_chunks
                               (document_id, chunk_index, content, heading_path, token_count, embedding, metadata)
                               values (%s, %s, %s, %s, %s, %s, %s)""",
                            [(row["id"], c.chunk_index, c.content, c.heading_path, c.token_count,
                              _vec(e), Jsonb(c.metadata)) for c, e in zip(chunks, embeddings)],
                        )
            return self._doc(row), ("created" if row["inserted"] else "updated")
        except psycopg.Error as exc:
            raise self._wrap(exc) from exc

    def delete_document(self, source_uri: str) -> bool:
        try:
            with self._pool.connection() as conn:
                cur = conn.execute("delete from public.documents where source_uri = %s", (source_uri,))
                return cur.rowcount > 0
        except psycopg.Error as exc:
            raise self._wrap(exc) from exc

    # ---- retrieval ----
    @staticmethod
    def _result(r: dict) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=r["chunk_id"], document_id=r["document_id"], content=r["content"],
            heading_path=r["heading_path"] or "", title=r["title"], url=r["url"],
            source_uri=r["source_uri"], source_type=r["source_type"],
            similarity=float(r["similarity"]), score=float(r["score"]), metadata=r["metadata"] or {},
        )

    def vector_search(self, embedding, top_k, source_types=None):
        try:
            with self._pool.connection() as conn:
                rows = conn.execute(
                    "select * from public.match_chunks(%s::vector, %s, %s::text[])",
                    (_vec(embedding), top_k, _types(source_types)),
                ).fetchall()
            return [self._result(r) for r in rows]
        except psycopg.Error as exc:
            raise self._wrap(exc) from exc

    def hybrid_search(self, query, embedding, top_k, source_types=None):
        try:
            with self._pool.connection() as conn:
                rows = conn.execute(
                    "select * from public.hybrid_search_chunks(%s, %s::vector, %s, %s::text[])",
                    (query, _vec(embedding), top_k, _types(source_types)),
                ).fetchall()
            return [self._result(r) for r in rows]
        except psycopg.Error as exc:
            raise self._wrap(exc) from exc

    # ---- jobs ----
    @staticmethod
    def _job(r: dict) -> IngestJob:
        return IngestJob(id=r["id"], status=r["status"], source=r["source"], params=r["params"] or {},
                         stats=IngestStats(**(r["stats"] or {})), error=r["error"],
                         created_at=r["created_at"], updated_at=r["updated_at"])

    def create_job(self, source: str, params: dict[str, Any]) -> IngestJob:
        try:
            with self._pool.connection() as conn:
                row = conn.execute(
                    "insert into public.ingestion_jobs (source, params, stats) values (%s, %s, %s) returning *",
                    (source, Jsonb(params), Jsonb(IngestStats().model_dump())),
                ).fetchone()
            return self._job(row)
        except psycopg.Error as exc:
            raise self._wrap(exc) from exc

    def update_job(self, job_id, status=None, stats=None, error=None) -> IngestJob:
        try:
            with self._pool.connection() as conn:
                row = conn.execute(
                    """update public.ingestion_jobs set
                         status = coalesce(%s, status),
                         stats  = coalesce(%s, stats),
                         error  = coalesce(%s, error),
                         updated_at = now()
                       where id = %s returning *""",
                    (status.value if status else None,
                     Jsonb(stats.model_dump()) if stats else None, error, job_id),
                ).fetchone()
            if row is None:
                raise NotFoundError(f"Ingestion job {job_id} not found")
            return self._job(row)
        except psycopg.Error as exc:
            raise self._wrap(exc) from exc

    def get_job(self, job_id):
        try:
            with self._pool.connection() as conn:
                row = conn.execute("select * from public.ingestion_jobs where id = %s", (job_id,)).fetchone()
            return self._job(row) if row else None
        except psycopg.Error as exc:
            raise self._wrap(exc) from exc

    # ---- chat log / feedback ----
    def save_interaction(self, question, answer, abstained, citations, latency_ms) -> UUID:
        try:
            with self._pool.connection() as conn:
                row = conn.execute(
                    """insert into public.chat_interactions (question, answer, abstained, citations, latency_ms)
                       values (%s, %s, %s, %s, %s) returning id""",
                    (question, answer, abstained, Jsonb(json.loads(json.dumps(citations, default=str))), latency_ms),
                ).fetchone()
            return row["id"]
        except psycopg.Error as exc:
            raise self._wrap(exc) from exc

    def save_feedback(self, interaction_id, rating, comment) -> UUID:
        try:
            with self._pool.connection() as conn:
                exists = conn.execute("select 1 from public.chat_interactions where id = %s",
                                      (interaction_id,)).fetchone()
                if not exists:
                    raise NotFoundError("Unknown interaction_id")
                row = conn.execute(
                    "insert into public.feedback (interaction_id, rating, comment) values (%s, %s, %s) returning id",
                    (interaction_id, rating, comment),
                ).fetchone()
            return row["id"]
        except psycopg.Error as exc:
            raise self._wrap(exc) from exc
