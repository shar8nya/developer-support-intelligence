from __future__ import annotations

from app.config import Settings
from app.exceptions import ConfigurationError
from app.providers.base import EmbeddingProvider, LLMProvider
from app.providers.embeddings import HashingEmbedder, OpenAIEmbedder
from app.providers.llm import AnthropicLLM, ExtractiveDemoLLM, OpenAILLM


def build_embedder(s: Settings) -> EmbeddingProvider:
    if s.embedding_provider == "demo":
        return HashingEmbedder(s.embedding_dimensions)
    if s.embedding_provider == "openai":
        return OpenAIEmbedder(s.openai_api_key, s.embedding_model, s.embedding_dimensions,
                              s.openai_base_url, s.http_timeout)
    raise ConfigurationError(f"Unknown EMBEDDING_PROVIDER '{s.embedding_provider}'")


def build_llm(s: Settings) -> LLMProvider:
    if s.llm_provider == "demo":
        return ExtractiveDemoLLM()
    if s.llm_provider == "openai":
        return OpenAILLM(s.openai_api_key, s.llm_model, s.openai_base_url)
    if s.llm_provider == "anthropic":
        return AnthropicLLM(s.anthropic_api_key, s.llm_model)
    raise ConfigurationError(f"Unknown LLM_PROVIDER '{s.llm_provider}'")
