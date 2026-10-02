import pytest

from app.config import PROJECT_ROOT, Settings


def test_env_example_parses_to_safe_defaults(monkeypatch):
    """A user who just copies .env.example must get a working demo, with no accidental values."""
    for k in ("APP_MODE", "API_KEY", "GITHUB_TOKEN", "DATABASE_URL"):
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=str(PROJECT_ROOT / ".env.example"))
    assert s.app_mode == "demo" and s.is_demo and s.validate_live() == []
    assert not s.api_key and not s.github_token and not s.database_url and not s.openai_api_key
    assert s.embedding_dimensions == 1536 and s.use_hybrid and s.require_citations
    assert not s.allow_any_local_path


def test_env_example_has_no_inline_comments_after_values():
    for line in (PROJECT_ROOT / ".env.example").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            assert " #" not in line, f"inline comment would be parsed into the value: {line}"


def test_live_mode_reports_every_missing_setting():
    s = Settings(app_mode="live", _env_file=None)
    problems = " ".join(s.validate_live())
    assert "DATABASE_URL" in problems and "OPENAI_API_KEY" in problems
    ok = Settings(app_mode="live", database_url="postgresql://x", openai_api_key="k", _env_file=None)
    assert ok.validate_live() == []
    anth = Settings(app_mode="live", database_url="x", openai_api_key="k", llm_provider="anthropic", _env_file=None)
    assert any("ANTHROPIC_API_KEY" in p for p in anth.validate_live())


def test_demo_mode_forces_offline_providers():
    s = Settings(app_mode="demo", embedding_provider="openai", llm_provider="openai",
                 database_url="postgresql://x", _env_file=None)
    assert s.embedding_provider == "demo" and s.llm_provider == "demo" and s.database_url is None


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        Settings(app_mode="prod", _env_file=None)
    with pytest.raises(ValueError):
        Settings(min_similarity=2, _env_file=None)
