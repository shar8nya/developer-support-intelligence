"""LLM providers: OpenAI-compatible chat, Anthropic messages, and an offline extractive demo."""
from __future__ import annotations

import re
import time
from typing import Optional

import httpx

from app.exceptions import ConfigurationError, ProviderError
from app.providers.base import GenerationInput, LLMProvider
from app.text_utils import tokenize

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def _retrying_post(client: httpx.Client, url: str, headers: dict, payload: dict, retries: int = 3) -> dict:
    last: Optional[str] = None
    for attempt in range(retries):
        try:
            r = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            last = f"network error: {exc}"
        else:
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in (429, 500, 502, 503, 504, 529):
                break
        time.sleep(min(2 ** attempt, 8) * 0.5)
    raise ProviderError(f"LLM request failed ({last})")


class OpenAILLM(LLMProvider):
    name = "openai"

    def __init__(self, api_key: Optional[str], model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1", timeout: float = 60.0,
                 client: Optional[httpx.Client] = None):
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for the OpenAI LLM provider.")
        self.api_key, self.model = api_key, model
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    def generate(self, inp: GenerationInput) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [{"role": "system", "content": inp.system}, *inp.messages],
        }
        data = _retrying_post(
            self._client, f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, payload,
        )
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected OpenAI response shape: {exc}") from exc


class AnthropicLLM(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: Optional[str], model: str, timeout: float = 60.0,
                 base_url: str = "https://api.anthropic.com", client: Optional[httpx.Client] = None):
        if not api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY is required for the Anthropic LLM provider.")
        self.api_key, self.model = api_key, model
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    def generate(self, inp: GenerationInput) -> str:
        payload = {"model": self.model, "max_tokens": 1024, "temperature": 0.1,
                   "system": inp.system, "messages": inp.messages}
        data = _retrying_post(
            self._client, f"{self.base_url}/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
             "content-type": "application/json"}, payload,
        )
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"Unexpected Anthropic response shape: {exc}") from exc


_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class ExtractiveDemoLLM(LLMProvider):
    """Offline 'LLM': answers by quoting the most question-relevant sentences from the context.

    It cannot reason or paraphrase - it only demonstrates the grounded/cited/abstaining
    pipeline without credentials. Every sentence it emits carries the [n] of its source.
    """
    name = "demo-extractive"
    model = "extractive-v1"

    def __init__(self, max_sentences: int = 3, min_overlap: float = 0.34):
        self.max_sentences, self.min_overlap = max_sentences, min_overlap

    def generate(self, inp: GenerationInput) -> str:
        q_terms = set(tokenize(inp.question))
        if not q_terms:
            return INSUFFICIENT
        scored: list[tuple[float, int, int, str]] = []
        for ctx in inp.contexts:
            head, sep, rest = ctx.text.partition("\n\n")
            body = rest if sep and len(head) < 200 and "." not in head else ctx.text  # drop "Title > Heading" prefix
            for pos, sent in enumerate(_SENT_RE.split(body)):
                sent = sent.strip()
                if len(sent) < 25 or sent.startswith(("```", "#")):
                    continue
                s_terms = set(tokenize(sent))
                overlap = len(q_terms & s_terms) / len(q_terms)
                if overlap >= self.min_overlap:
                    scored.append((overlap, -ctx.index, -pos, sent))
        if not scored:
            return INSUFFICIENT
        scored.sort(reverse=True)
        chosen, seen = [], set()
        for overlap, neg_idx, neg_pos, sent in scored:
            if sent in seen:
                continue
            seen.add(sent)
            chosen.append((-neg_idx, -neg_pos, sent))
            if len(chosen) >= self.max_sentences:
                break
        chosen.sort()
        return " ".join(f"{re.sub(r'^[-*] ', '', s).strip()} [{idx}]" for idx, _, s in chosen)
