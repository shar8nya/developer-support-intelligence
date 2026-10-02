from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class EmbeddingProvider(ABC):
    name: str
    dimensions: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@dataclass
class ContextBlock:
    index: int          # the [n] citation marker
    title: str
    text: str


@dataclass
class GenerationInput:
    """Everything an LLM provider may need. Real LLMs use `system` + `messages`;
    the offline demo LLM uses the structured `question` + `contexts`."""
    system: str
    messages: list[dict[str, str]]          # [{"role": "user"|"assistant", "content": ...}]
    question: str = ""
    contexts: list[ContextBlock] = field(default_factory=list)


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def generate(self, inp: GenerationInput) -> str: ...
