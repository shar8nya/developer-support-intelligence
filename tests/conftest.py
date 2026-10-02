from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Tests must never pick up a developer's real .env
os.environ["APP_MODE"] = "demo"

from app.config import PROJECT_ROOT, Settings  # noqa: E402
from app.container import Container, build_container  # noqa: E402
from app.main import create_app  # noqa: E402
from app.repositories.memory import InMemoryRepository  # noqa: E402

SAMPLE_DOCS = PROJECT_ROOT / "data" / "sample_docs"
SAMPLE_ISSUES = PROJECT_ROOT / "data" / "sample_issues"


@pytest.fixture
def settings() -> Settings:
    return Settings(app_mode="demo", _env_file=None)


@pytest.fixture
def container(settings) -> Container:
    """Demo container with the sample corpus already ingested (fresh in-memory store per test)."""
    c = build_container(settings, auto_ingest_demo=False, repo=InMemoryRepository())
    from app.schemas import IngestRequest
    for p in (SAMPLE_DOCS, SAMPLE_ISSUES):
        c.pipeline.run(IngestRequest(source="local", path=str(p)))
    return c


@pytest.fixture
def empty_container(settings) -> Container:
    return build_container(settings, auto_ingest_demo=False, repo=InMemoryRepository())


@pytest.fixture
def client(container) -> TestClient:
    with TestClient(create_app(container)) as c:
        yield c
