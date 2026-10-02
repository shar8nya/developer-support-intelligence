"""Central configuration (12-factor style: env vars / .env)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    app_mode: Literal["demo", "live"] = "demo"

    database_url: Optional[str] = None

    embedding_provider: Literal["openai", "demo"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"

    llm_provider: Literal["openai", "anthropic", "demo"] = "openai"
    llm_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None

    top_k: int = Field(5, ge=1, le=20)
    use_hybrid: bool = True
    use_rerank: bool = True
    min_similarity: float = Field(0.20, ge=0.0, le=1.0)
    require_citations: bool = True
    max_context_chars: int = 12000

    chunk_max_tokens: int = Field(350, ge=50)
    chunk_overlap_tokens: int = Field(50, ge=0)

    github_token: Optional[str] = None
    demo_auto_ingest: bool = True
    http_timeout: float = 30.0

    api_base_url: str = "http://127.0.0.1:8000"
    api_key: Optional[str] = None                   # if set, clients must send X-API-Key
    # Security: the /ingest endpoint may only read local files below this directory
    local_ingest_root: Path = PROJECT_ROOT / "data"
    allow_any_local_path: bool = False

    @model_validator(mode="after")
    def _apply_mode(self) -> "Settings":
        if self.app_mode == "demo":
            # Demo mode never needs credentials or a database.
            self.embedding_provider = "demo"
            self.llm_provider = "demo"
            self.database_url = None
        return self

    @property
    def is_demo(self) -> bool:
        return self.app_mode == "demo"

    def validate_live(self) -> list[str]:
        """Return a list of human-readable configuration problems for live mode."""
        problems: list[str] = []
        if self.is_demo:
            return problems
        if not self.database_url:
            problems.append("DATABASE_URL is required in live mode (Supabase connection string).")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            problems.append("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai.")
        if self.llm_provider == "openai" and not self.openai_api_key:
            problems.append("OPENAI_API_KEY is required for LLM_PROVIDER=openai.")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            problems.append("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic.")
        return problems


@lru_cache
def get_settings() -> Settings:
    return Settings()
