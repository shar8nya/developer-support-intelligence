"""Embedding providers.

* HashingEmbedder - deterministic, offline, no API key. Bag-of-words feature hashing.
  It is NOT semantic (no synonyms) - it exists so demo mode, tests and CI run anywhere.
* OpenAIEmbedder  - OpenAI (or any OpenAI-compatible) /embeddings endpoint via httpx.
"""
from __future__ import annotations

import hashlib
import math
import time
from typing import Optional

import httpx

from app.exceptions import ConfigurationError, ProviderError
from app.providers.base import EmbeddingProvider
from app.text_utils import tokenize


class HashingEmbedder(EmbeddingProvider):
    name = "demo-hashing"

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = tokenize(text)
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]  # unigrams + bigrams
        for i, feat in enumerate(feats):
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            idx = h % self.dimensions
            sign = 1.0 if (h >> 63) & 1 else -1.0
            weight = 1.0 if "_" not in feat else 0.5
            vec[idx] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class OpenAIEmbedder(EmbeddingProvider):
    name = "openai"

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        batch_size: int = 64,
        max_retries: int = 4,
        client: Optional[httpx.Client] = None,
    ):
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for the OpenAI embedding provider.")
        self.api_key, self.model, self.dimensions = api_key, model, dimensions
        self.base_url = base_url.rstrip("/")
        self.batch_size, self.max_retries = batch_size, max_retries
        self._client = client or httpx.Client(timeout=timeout)

    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last = f"network error: {exc}"
            else:
                if r.status_code == 200:
                    return r.json()
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    break
            time.sleep(min(2 ** attempt, 8) * 0.5)
        raise ProviderError(f"Embedding request failed ({last})")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ")[:24000] for t in texts[i : i + self.batch_size]]
            payload: dict = {"model": self.model, "input": batch}
            if self.model.startswith("text-embedding-3"):
                payload["dimensions"] = self.dimensions
            data = self._post(payload)
            items = sorted(data.get("data", []), key=lambda d: d["index"])
            if len(items) != len(batch):
                raise ProviderError("Embedding response size mismatch")
            for item in items:
                emb = item["embedding"]
                if len(emb) != self.dimensions:
                    raise ProviderError(
                        f"Embedding has {len(emb)} dims but EMBEDDING_DIMENSIONS={self.dimensions}. "
                        "They must match the vector(N) column in the database."
                    )
                out.append(emb)
        return out
