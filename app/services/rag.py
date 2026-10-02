"""RAG orchestration: retrieve -> gate on evidence -> generate -> validate citations."""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from app.config import Settings
from app.exceptions import RepositoryError
from app.providers.base import LLMProvider
from app.repositories.base import Repository
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    RetrievedChunk,
    SearchResult,
)
from app.services.generation import (
    build_contexts,
    build_generation_input,
    is_refusal,
    parse_citations,
    repair_missing_citations,
)
from app.services.retrieval import RetrievalService

log = logging.getLogger(__name__)

ABSTAIN_MESSAGE = (
    "I couldn't find enough information in the indexed documentation and GitHub issues "
    "to answer that reliably, so I won't guess. Try rephrasing the question, or ingest "
    "more sources that cover this topic."
)


def to_search_result(c: RetrievedChunk) -> SearchResult:
    return SearchResult(
        chunk_id=c.chunk_id,
        document_id=c.document_id,
        title=c.title,
        url=c.url,
        source_uri=c.source_uri,
        source_type=c.source_type,
        heading_path=c.heading_path,
        content=c.content,
        similarity=round(c.similarity, 4),
        score=round(c.score, 4),
    )


def _snippet(c: RetrievedChunk, limit: int = 320) -> str:
    text = c.content

    if c.heading_path and "\n\n" in text:
        # Strip the heading-path prefix added by the chunker.
        text = text.split("\n\n", 1)[1]

    text = " ".join(text.split())

    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class RAGService:
    def __init__(
        self,
        repo: Repository,
        retrieval: RetrievalService,
        llm: LLMProvider,
        settings: Settings,
    ):
        self.repo = repo
        self.retrieval = retrieval
        self.llm = llm
        self.settings = settings

    @staticmethod
    def _retrieval_query(req: ChatRequest) -> str:
        """Follow-ups like 'and how do I revoke it?' are unsearchable alone:
        prepend the last user turn.
        """
        if req.history and len(req.question.split()) <= 5:
            prev = [t.content for t in req.history if t.role == "user"]

            if prev:
                return f"{prev[-1]} {req.question}"

        return req.question

    def answer(self, req: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()

        chunks = self.retrieval.search(
            self._retrieval_query(req),
            req.top_k,
            req.use_hybrid,
            req.use_rerank,
            req.source_types,
        )

        answer = ABSTAIN_MESSAGE
        abstained = True
        reason = None
        citations: list[Citation] = []

        best = max(
            (c.similarity for c in chunks),
            default=0.0,
        )

        # ------------------------------------------------------------
        # Evidence gate
        # ------------------------------------------------------------
        if not chunks or best < self.settings.min_similarity:
            reason = "no_relevant_evidence"

        else:
            contexts = build_contexts(
                chunks,
                self.settings.max_context_chars,
            )

            used = chunks[: len(contexts)]

            # --------------------------------------------------------
            # First generation
            # --------------------------------------------------------
            raw = self.llm.generate(
                build_generation_input(
                    req.question,
                    contexts,
                    used,
                    req.history,
                )
            )

            if is_refusal(raw):
                reason = "model_declined"

            else:
                parsed = parse_citations(
                    raw,
                    len(contexts),
                )

                # ----------------------------------------------------
                # Citation requirement
                # ----------------------------------------------------
                if (
                    not parsed.cited
                    and self.settings.require_citations
                ):
                    reason = "no_valid_citations"

                    log.warning(
                        "LLM answer had no valid citations; abstaining"
                    )

                else:
                    # ------------------------------------------------
                    # Deterministic citation repair
                    # ------------------------------------------------
                    repaired_answer = repair_missing_citations(
                        parsed.text,
                        contexts,
                    )

                    final_parsed = parse_citations(
                        repaired_answer,
                        len(contexts),
                    )

                    # --------------------------------------------
                    # Final citation validation
                    # --------------------------------------------
                    if (
                        self.settings.require_citations
                        and not final_parsed.cited
                    ):
                        answer = ABSTAIN_MESSAGE
                        abstained = True
                        reason = "citation_validation_failed"

                        log.warning(
                            "Citation repair produced no valid citations; abstaining"
                        )

                    else:
                        answer = final_parsed.text
                        abstained = False
                        reason = None

                        citations = [
                            Citation(
                                index=n,
                                chunk_id=used[n - 1].chunk_id,
                                document_id=used[n - 1].document_id,
                                title=used[n - 1].title,
                                url=used[n - 1].url,
                                source_type=used[n - 1].source_type,
                                heading_path=used[n - 1].heading_path,
                                snippet=_snippet(used[n - 1]),
                                similarity=round(
                                    used[n - 1].similarity,
                                    4,
                                ),
                                score=round(
                                    used[n - 1].score,
                                    4,
                                ),
                            )
                            for n in final_parsed.cited
                        ]

        # ------------------------------------------------------------
        # Save interaction
        # ------------------------------------------------------------
        latency = int(
            (time.perf_counter() - t0) * 1000
        )

        try:
            interaction_id = self.repo.save_interaction(
                req.question,
                answer,
                abstained,
                [
                    c.model_dump(mode="json")
                    for c in citations
                ],
                latency,
            )

        except RepositoryError as exc:
            # Logging must never break answering.
            log.warning(
                "could not persist interaction: %s",
                exc,
            )

            interaction_id = uuid4()

        # ------------------------------------------------------------
        # Final response
        # ------------------------------------------------------------
        return ChatResponse(
            interaction_id=interaction_id,
            answer=answer,
            abstained=abstained,
            abstain_reason=reason,
            citations=citations,
            retrieved=[
                to_search_result(c)
                for c in chunks
            ],
            latency_ms=latency,
            mode="demo" if self.settings.is_demo else "live",
        )