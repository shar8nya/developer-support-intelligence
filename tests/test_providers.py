import httpx
import pytest

from app.exceptions import ConfigurationError, ProviderError
from app.providers.base import ContextBlock, GenerationInput
from app.providers.embeddings import HashingEmbedder, OpenAIEmbedder
from app.providers.llm import AnthropicLLM, ExtractiveDemoLLM, OpenAILLM


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_hashing_embedder_is_deterministic_normalised_and_similarity_ordered():
    e = HashingEmbedder(256)
    a, b = e.embed_documents(["access tokens expire after one hour", "access tokens expire after one hour"])
    assert a == b and len(a) == 256
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    q = e.embed_query("when do access tokens expire")
    near = e.embed_query("access tokens expire after one hour")
    far = e.embed_query("pagination cursor parameters")
    assert dot(q, near) > dot(q, far)


def test_openai_embedder_batches_and_sends_dimensions():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = req.read()
        import json
        payload = json.loads(body)
        calls.append(payload)
        assert req.headers["authorization"] == "Bearer k"
        data = [{"index": i, "embedding": [0.1] * 8} for i in range(len(payload["input"]))][::-1]  # out of order
        return httpx.Response(200, json={"data": data})

    emb = OpenAIEmbedder("k", "text-embedding-3-small", 8, batch_size=2, client=_client(handler))
    out = emb.embed_documents(["a", "b", "c"])
    assert len(out) == 3 and len(calls) == 2
    assert calls[0]["dimensions"] == 8 and calls[0]["input"] == ["a", "b"]


def test_openai_embedder_dimension_mismatch_is_a_clear_error():
    handler = lambda req: httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 3}]})
    emb = OpenAIEmbedder("k", "custom-model", 8, client=_client(handler))
    with pytest.raises(ProviderError, match="EMBEDDING_DIMENSIONS"):
        emb.embed_documents(["a"])


def test_openai_embedder_retries_then_fails(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        return httpx.Response(429, text="slow down")

    emb = OpenAIEmbedder("k", dimensions=8, max_retries=3, client=_client(handler))
    with pytest.raises(ProviderError, match="429"):
        emb.embed_documents(["a"])
    assert n["c"] == 3


def test_missing_keys_raise_configuration_error():
    with pytest.raises(ConfigurationError):
        OpenAIEmbedder(None)
    with pytest.raises(ConfigurationError):
        OpenAILLM(None)
    with pytest.raises(ConfigurationError):
        AnthropicLLM(None, "m")


def test_openai_llm_parses_response():
    def handler(req):
        import json
        p = json.loads(req.read())
        assert p["messages"][0]["role"] == "system"
        return httpx.Response(200, json={"choices": [{"message": {"content": " Hello [1] "}}]})

    llm = OpenAILLM("k", client=_client(handler))
    assert llm.generate(GenerationInput(system="s", messages=[{"role": "user", "content": "q"}])) == "Hello [1]"


def test_anthropic_llm_parses_response_and_sends_headers():
    def handler(req):
        import json
        assert req.headers["x-api-key"] == "k" and "anthropic-version" in req.headers
        p = json.loads(req.read())
        assert p["system"] == "s" and p["messages"][0]["role"] == "user"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Answer [1]"}]})

    llm = AnthropicLLM("k", "some-model", client=_client(handler))
    assert llm.generate(GenerationInput(system="s", messages=[{"role": "user", "content": "q"}])) == "Answer [1]"


def test_extractive_llm_quotes_relevant_sentence_with_marker_and_declines_otherwise():
    ctx = [ContextBlock(1, "A", "Access tokens expire after 60 minutes. Unrelated sentence about cats and dogs."),
           ContextBlock(2, "B", "Pagination uses cursors that expire after 24 hours.")]
    llm = ExtractiveDemoLLM()
    out = llm.generate(GenerationInput("s", [], "How long until access tokens expire?", ctx))
    assert "60 minutes" in out and out.rstrip().endswith("[1]") and "cats" not in out
    assert llm.generate(GenerationInput("s", [], "kubernetes helm chart values", ctx)) == "INSUFFICIENT_EVIDENCE"
