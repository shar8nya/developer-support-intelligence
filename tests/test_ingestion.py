import json

import httpx
import pytest

from app.config import Settings
from app.exceptions import IngestionError
from app.ingestion.connectors import (
    GitHubDocsConnector, GitHubIssuesConnector, LocalConnector, UrlConnector, assert_public_http_url,
    issue_to_document,
)
from app.ingestion.pipeline import IngestionPipeline, resolve_local_path
from app.schemas import IngestRequest, RawDocument, SourceType
from tests.conftest import SAMPLE_DOCS, SAMPLE_ISSUES


def _doc(uri="u", text="# T\n\nSome sufficiently long document body text."):
    return RawDocument(source_type=SourceType.docs, source_uri=uri, title="T", content=text)


def test_unchanged_content_is_not_reingested(empty_container):
    p = empty_container.pipeline
    s1 = p.ingest_documents([_doc()])
    s2 = p.ingest_documents([_doc()])
    assert (s1.documents_new, s1.chunks_written > 0) == (1, True)
    assert (s2.documents_unchanged, s2.documents_new, s2.chunks_written) == (1, 0, 0)


def test_changed_content_replaces_chunks(empty_container):
    p, repo = empty_container.pipeline, empty_container.repo
    p.ingest_documents([_doc(text="# T\n\nOld body text that is long enough to keep.")])
    s = p.ingest_documents([_doc(text="# T\n\nBrand new body text that is long enough too.")])
    assert s.documents_updated == 1 and repo.stats()["documents"] == 1
    hits = repo.vector_search(empty_container.embedder.embed_query("brand new body"), 5)
    assert all("Old body" not in h.content for h in hits)


def test_force_reingests_and_empty_documents_fail_gracefully(empty_container):
    p = empty_container.pipeline
    p.ingest_documents([_doc()])
    assert p.ingest_documents([_doc()], force=True).documents_updated == 1
    s = p.ingest_documents([_doc("e", text="   "), _doc("ok")])
    assert s.documents_failed == 1 and s.documents_new == 1 and "empty content" in s.errors[0]


def test_config_change_triggers_reindex(empty_container, settings):
    empty_container.pipeline.ingest_documents([_doc()])
    other = IngestionPipeline(empty_container.repo, empty_container.embedder,
                              Settings(app_mode="demo", chunk_max_tokens=120, _env_file=None))
    assert other.ingest_documents([_doc()]).documents_updated == 1


def test_local_connector_reads_docs_and_issue_json():
    docs = list(LocalConnector(SAMPLE_DOCS).fetch())
    assert len(docs) == 11 and all(d.source_type == SourceType.docs for d in docs)
    auth = next(d for d in docs if d.source_uri == "local:authentication.md")
    assert auth.title == "Authentication and tokens" and auth.url.startswith("https://docs.acme")
    issues = list(LocalConnector(SAMPLE_ISSUES).fetch())
    assert len(issues) == 6 and issues[0].source_type == SourceType.github_issue
    assert "Discussion" in issues[0].content


def test_local_path_sandbox(settings):
    assert resolve_local_path("data/sample_docs", settings).name == "sample_docs"
    with pytest.raises(IngestionError):
        resolve_local_path("/etc", settings)
    with pytest.raises(IngestionError):
        resolve_local_path("data/../app", settings)


def test_issue_to_document_shape():
    d = issue_to_document({"number": 7, "title": "Boom", "state": "open", "body": "It broke", "labels": [{"name": "bug"}],
                           "html_url": "https://github.com/o/r/issues/7"},
                          [{"user": {"login": "bob"}, "body": "same here"}], "o/r")
    assert d.source_uri == "github:o/r#issue-7" and d.title == "#7: Boom"
    assert "labels: bug" in d.content and "@bob" in d.content and d.source_type == SourceType.github_issue


def _gh_transport(routes):
    def handler(req: httpx.Request) -> httpx.Response:
        for prefix, resp in routes.items():
            if str(req.url).startswith(prefix):
                return resp(req) if callable(resp) else httpx.Response(200, json=resp)
        return httpx.Response(404, json={"message": "nope"})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_github_docs_connector_filters_by_prefix_and_extension():
    tree = {"tree": [{"type": "blob", "path": "docs/a.md"}, {"type": "blob", "path": "docs/b.png"},
                     {"type": "blob", "path": "other/c.md"}, {"type": "blob", "path": "docs/sub/d.mdx"}]}
    client = _gh_transport({
        "https://api.github.com/repos/o/r/git/trees/main": tree,
        "https://raw.githubusercontent.com/o/r/main/docs/a.md": lambda r: httpx.Response(200, text="---\ntitle: A\n---\n# A\n\nbody a is long enough"),
        "https://raw.githubusercontent.com/o/r/main/docs/sub/d.mdx": lambda r: httpx.Response(200, text="# D\n\n<Note/>body d is long enough"),
    })
    docs = list(GitHubDocsConnector("o/r", "main", "docs", 10, client=client).fetch())
    assert [d.source_uri for d in docs] == ["github:o/r/docs/a.md", "github:o/r/docs/sub/d.mdx"]
    assert docs[0].title == "A" and docs[0].url == "https://github.com/o/r/blob/main/docs/a.md"
    assert "<Note" not in docs[1].content


def test_github_docs_connector_falls_back_to_default_branch():
    client = _gh_transport({
        "https://api.github.com/repos/o/r/git/trees/master": lambda r: httpx.Response(404, json={}),
        "https://api.github.com/repos/o/r/git/trees/main": {"tree": [{"type": "blob", "path": "a.md"}]},
        "https://api.github.com/repos/o/r": {"default_branch": "main"},
        "https://raw.githubusercontent.com/o/r/main/a.md": lambda r: httpx.Response(200, text="# A\n\nbody body body"),
    })
    docs = list(GitHubDocsConnector("o/r", "master", "", 5, client=client).fetch())
    assert len(docs) == 1 and docs[0].metadata["branch"] == "main"


def test_github_issues_connector_skips_pull_requests_and_fetches_comments():
    issues = [{"number": 1, "title": "Real", "state": "open", "body": "b", "labels": [], "comments": 1,
               "comments_url": "https://api.github.com/repos/o/r/issues/1/comments", "html_url": "h1"},
              {"number": 2, "title": "A PR", "state": "open", "body": "b", "labels": [], "pull_request": {}, "comments": 0}]
    pages = {"n": 0}

    def issues_handler(req):
        pages["n"] += 1
        return httpx.Response(200, json=issues if pages["n"] == 1 else [])

    client = _gh_transport({
        "https://api.github.com/repos/o/r/issues/1/comments": [{"user": {"login": "z"}, "body": "a comment"}],
        "https://api.github.com/repos/o/r/issues": issues_handler,
    })
    docs = list(GitHubIssuesConnector("o/r", max_issues=5, client=client).fetch())
    assert [d.source_uri for d in docs] == ["github:o/r#issue-1"] and "a comment" in docs[0].content


def test_github_rate_limit_message():
    client = _gh_transport({"https://api.github.com/repos/o/r/issues":
                            lambda r: httpx.Response(403, headers={"x-ratelimit-remaining": "0"}, json={})})
    with pytest.raises(IngestionError, match="GITHUB_TOKEN"):
        list(GitHubIssuesConnector("o/r", client=client).fetch())


def test_url_connector_and_ssrf_guard():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, headers={"content-type": "text/html"}, text="<title>Hi</title><h1>Head</h1><p>Paragraph text</p>")))
    docs = list(UrlConnector(["https://example.com/x"], client=client, check_ssrf=False).fetch())
    assert docs[0].title == "Hi" and "Paragraph text" in docs[0].content and docs[0].source_type == SourceType.web
    for bad in ("http://127.0.0.1/admin", "http://localhost:8000", "http://169.254.169.254/latest", "file:///etc/passwd"):
        with pytest.raises(IngestionError):
            assert_public_http_url(bad)


def test_ingest_request_validation():
    with pytest.raises(ValueError):
        IngestRequest(source="local")
    with pytest.raises(ValueError):
        IngestRequest(source="github_issues", repo="noslash")
    with pytest.raises(ValueError):
        IngestRequest(source="url")
    assert IngestRequest(source="github_docs", repo="a/b").branch == "master"


def test_run_job_records_failure(empty_container):
    repo = empty_container.repo
    job = repo.create_job("github_docs", {})
    bad = IngestRequest(source="local", path="data/does_not_exist")
    empty_container.pipeline.run_job(job.id, bad)
    j = repo.get_job(job.id)
    assert j.status.value == "failed" and "does not exist" in j.error
