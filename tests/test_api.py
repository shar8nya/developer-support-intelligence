"""API integration tests through FastAPI's TestClient (demo mode, in-memory store)."""
import time
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import Settings
from app.container import build_container
from app.main import create_app
from app.repositories.memory import InMemoryRepository


def test_health(client):
    r = client.get("/health")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok" and body["mode"] == "demo"
    assert body["documents"] == 17 and body["chunks"] > 17


def test_openapi_lists_all_required_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/health", "/chat", "/search", "/ingest", "/ingest/{job_id}", "/feedback"):
        assert p in paths


def test_chat_returns_cited_answer(client):
    r = client.post("/chat", json={"question": "What is the rate limit for the Free plan?"})
    body = r.json()
    assert r.status_code == 200 and not body["abstained"]
    assert "100" in body["answer"] and body["citations"][0]["title"] == "Rate limits"
    assert body["retrieved"] and body["latency_ms"] >= 0 and body["interaction_id"]


def test_chat_abstains(client):
    body = client.post("/chat", json={"question": "How do I migrate my existing data from Trello?"}).json()
    assert body["abstained"] and body["citations"] == []


def test_chat_validation_errors(client):
    assert client.post("/chat", json={"question": "x"}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422
    assert client.post("/chat", json={"question": "valid question", "top_k": 500}).status_code == 422
    assert client.post("/chat", json={"question": "valid question", "source_types": ["nope"]}).status_code == 422


def test_search(client):
    r = client.post("/search", json={"query": "webhook retries", "top_k": 3})
    body = r.json()
    assert r.status_code == 200 and len(body["results"]) == 3
    assert body["results"][0]["title"] == "Webhooks" and body["results"][0]["url"]
    only_docs = client.post("/search", json={"query": "webhook", "source_types": ["docs"]}).json()["results"]
    assert all(x["source_type"] == "docs" for x in only_docs)


def test_search_modes_are_accepted(client):
    for hybrid in (True, False):
        for rr in (True, False):
            r = client.post("/search", json={"query": "pagination cursor", "use_hybrid": hybrid, "use_rerank": rr})
            assert r.status_code == 200 and r.json()["results"][0]["title"] == "Pagination"


def test_feedback_roundtrip_and_errors(client):
    iid = client.post("/chat", json={"question": "How do I revoke a token?"}).json()["interaction_id"]
    ok = client.post("/feedback", json={"interaction_id": iid, "rating": 1, "comment": "helpful"})
    assert ok.status_code == 201 and ok.json()["status"] == "recorded"
    assert client.post("/feedback", json={"interaction_id": str(uuid4()), "rating": 1}).status_code == 404
    assert client.post("/feedback", json={"interaction_id": iid, "rating": 5}).status_code == 422
    assert client.post("/feedback", json={"interaction_id": "not-a-uuid", "rating": 1}).status_code == 422


def test_ingest_job_lifecycle(settings):
    empty = build_container(settings, auto_ingest_demo=False, repo=InMemoryRepository())
    with TestClient(create_app(empty)) as c:
        r = c.post("/ingest", json={"source": "local", "path": "data/sample_docs"})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        for _ in range(50):  # background task finishes right after the response in TestClient
            job = c.get(f"/ingest/{job_id}").json()
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert job["status"] == "succeeded" and job["stats"]["documents_new"] == 11
        # second run: everything is unchanged (dedupe)
        job2 = c.get(f"/ingest/{c.post('/ingest', json={'source': 'local', 'path': 'data/sample_docs'}).json()['job_id']}").json()
        assert job2["stats"]["documents_unchanged"] == 11 and job2["stats"]["chunks_written"] == 0
        assert c.get("/health").json()["documents"] == 11


def test_ingest_errors(client):
    assert client.get(f"/ingest/{uuid4()}").status_code == 404
    assert client.post("/ingest", json={"source": "local"}).status_code == 422
    r = client.post("/ingest", json={"source": "local", "path": "/etc"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "ingestion_error"
    assert client.post("/ingest", json={"source": "local", "path": "data/missing_dir"}).status_code == 400


def test_api_key_protection(container):
    protected = container.__class__(**{**container.__dict__, "settings": Settings(app_mode="demo", api_key="s3cret", _env_file=None)})
    with TestClient(create_app(protected)) as c:
        assert c.get("/health").status_code == 200                       # health stays open
        assert c.post("/chat", json={"question": "How do I revoke a token?"}).status_code == 401
        assert c.post("/chat", json={"question": "How do I revoke a token?"}, headers={"X-API-Key": "bad"}).status_code == 401
        ok = c.post("/chat", json={"question": "How do I revoke a token?"}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200


def test_unexpected_errors_are_sanitised(container):
    def boom(*a, **k):
        raise RuntimeError("secret internals")
    container.rag.answer = boom
    with TestClient(create_app(container), raise_server_exceptions=False) as c:
        r = c.post("/chat", json={"question": "How do I revoke a token?"})
        assert r.status_code == 500 and "secret internals" not in r.text


def test_provider_error_maps_to_502(container):
    from app.exceptions import ProviderError

    def boom(*a, **k):
        raise ProviderError("upstream down")
    container.rag.answer = boom
    with TestClient(create_app(container)) as c:
        r = c.post("/chat", json={"question": "How do I revoke a token?"})
        assert r.status_code == 502 and r.json()["error"]["code"] == "provider_error"
