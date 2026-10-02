"""Integration tests against a REAL PostgreSQL + pgvector (skipped unless TEST_DATABASE_URL is set).

    set TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/dsi_test     (Windows cmd)
    python scripts/apply_migrations.py --database-url %TEST_DATABASE_URL%
    pytest -m integration

The database must be disposable: the tests TRUNCATE all tables.
"""
import os
from pathlib import Path

import pytest

DB = os.getenv("TEST_DATABASE_URL")
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not DB, reason="TEST_DATABASE_URL not set")]


@pytest.fixture(scope="module")
def repo():
    from app.repositories.supabase import SupabaseRepository
    r = SupabaseRepository(DB)
    with r._pool.connection() as conn:
        conn.execute("truncate public.feedback, public.chat_interactions, public.ingestion_jobs, "
                     "public.document_chunks, public.documents cascade")
    yield r
    r.close()


@pytest.fixture
def container(repo):
    from app.config import Settings
    from app.container import build_container
    from app.schemas import IngestRequest
    with repo._pool.connection() as conn:
        conn.execute("truncate public.document_chunks, public.documents cascade")
    c = build_container(Settings(app_mode="demo", _env_file=None), auto_ingest_demo=False, repo=repo)
    root = Path(__file__).resolve().parent.parent / "data"
    for sub in ("sample_docs", "sample_issues"):
        assert c.pipeline.run(IngestRequest(source="local", path=str(root / sub))).documents_failed == 0
    return c


def test_ping_and_stats(repo):
    repo.ping()
    assert set(repo.stats()) == {"documents", "chunks"}


def test_ingest_dedupes_in_real_database(container):
    from app.schemas import IngestRequest
    root = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
    s = container.pipeline.run(IngestRequest(source="local", path=str(root)))
    assert s.documents_unchanged == 11 and s.chunks_written == 0
    assert container.repo.stats()["documents"] == 17


def test_vector_and_hybrid_search_via_sql_functions(container):
    q = "How long do OAuth access tokens last before they expire?"
    emb = container.embedder.embed_query(q)
    v = container.repo.vector_search(emb, 3)
    h = container.repo.hybrid_search(q, emb, 3)
    assert v[0].title == "Authentication and tokens" and h[0].title == "Authentication and tokens"
    assert v[0].similarity > 0.2 and h[0].score > 0
    issues = container.repo.hybrid_search(q, emb, 5, ["github_issue"])
    assert issues and all(r.source_type.value == "github_issue" for r in issues)


def test_full_rag_flow_on_real_database(container):
    from app.schemas import ChatRequest
    resp = container.rag.answer(ChatRequest(question="How are webhook signatures computed?"))
    assert not resp.abstained and "HMAC" in resp.answer and resp.citations
    assert container.repo.save_feedback(resp.interaction_id, 1, "ok")


def test_job_lifecycle_and_cascade_delete(container):
    from app.schemas import IngestStats, JobStatus
    job = container.repo.create_job("local", {"path": "x"})
    done = container.repo.update_job(job.id, JobStatus.succeeded, IngestStats(documents_new=3))
    assert done.status == JobStatus.succeeded and container.repo.get_job(job.id).stats.documents_new == 3
    before = container.repo.stats()["chunks"]
    assert container.repo.delete_document("local:pagination.md")
    assert container.repo.stats()["chunks"] < before      # chunks removed by ON DELETE CASCADE
