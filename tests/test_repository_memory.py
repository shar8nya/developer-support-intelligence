import pytest

from app.exceptions import NotFoundError
from app.ingestion.chunker import chunk_markdown
from app.providers.embeddings import HashingEmbedder
from app.repositories.memory import InMemoryRepository
from app.schemas import IngestStats, JobStatus, RawDocument, SourceType

EMB = HashingEmbedder(128)


def _put(repo, uri, text, source_type=SourceType.docs, digest="h"):
    raw = RawDocument(source_type=source_type, source_uri=uri, title=uri, content=text)
    ch = chunk_markdown(text, title=uri)
    return repo.upsert_document(raw, digest, ch, EMB.embed_documents([c.content for c in ch]))


def test_upsert_create_update_and_delete():
    repo = InMemoryRepository()
    _, a1 = _put(repo, "u1", "# A\n\nAccess tokens expire after 60 minutes.")
    _, a2 = _put(repo, "u1", "# A\n\nRefresh tokens rotate on every use.", digest="h2")
    assert (a1, a2) == ("created", "updated")
    assert repo.stats() == {"documents": 1, "chunks": 1}
    assert repo.get_document_by_uri("u1").content_hash == "h2"
    assert repo.delete_document("u1") and not repo.delete_document("u1")
    assert repo.stats() == {"documents": 0, "chunks": 0}


def test_vector_and_hybrid_search_rank_relevant_first_and_filter_by_type():
    repo = InMemoryRepository()
    _put(repo, "auth", "# Auth\n\nAccess tokens expire after 60 minutes.")
    _put(repo, "page", "# Pagination\n\nUse the cursor parameter to fetch the next page.")
    _put(repo, "issue", "# Bug\n\nRefresh token fails with 401.", SourceType.github_issue)
    q = "when do access tokens expire"
    qe = EMB.embed_query(q)
    assert repo.vector_search(qe, 3)[0].source_uri == "auth"
    assert repo.hybrid_search(q, qe, 3)[0].source_uri == "auth"
    only_issues = repo.hybrid_search(q, qe, 5, [SourceType.github_issue])
    assert [r.source_uri for r in only_issues] == ["issue"]
    assert all(0.0 <= abs(r.similarity) <= 1.0 + 1e-6 for r in repo.vector_search(qe, 3))


def test_search_on_empty_store():
    repo = InMemoryRepository()
    assert repo.vector_search(EMB.embed_query("x"), 3) == []
    assert repo.hybrid_search("x", EMB.embed_query("x"), 3) == []


def test_jobs_and_feedback():
    repo = InMemoryRepository()
    job = repo.create_job("local", {"path": "x"})
    assert job.status == JobStatus.queued
    upd = repo.update_job(job.id, JobStatus.succeeded, IngestStats(documents_new=2))
    assert upd.status == JobStatus.succeeded and repo.get_job(job.id).stats.documents_new == 2
    with pytest.raises(NotFoundError):
        repo.update_job(__import__("uuid").uuid4(), JobStatus.failed)
    iid = repo.save_interaction("q", "a", False, [], 5)
    assert repo.save_feedback(iid, 1, "good")
    with pytest.raises(NotFoundError):
        repo.save_feedback(__import__("uuid").uuid4(), -1, None)
