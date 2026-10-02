"""RAG behaviour: grounded answers, citation correctness, abstention."""
import pytest

from app.evaluation.grounding import check_response
from app.providers.base import GenerationInput, LLMProvider
from app.schemas import ChatRequest, ChatTurn, SourceType
from app.services.generation import is_refusal, parse_citations
from app.services.rag import ABSTAIN_MESSAGE, RAGService


class ScriptedLLM(LLMProvider):
    name, model = "scripted", "s"

    def __init__(self, reply: str):
        self.reply, self.seen = reply, []

    def generate(self, inp: GenerationInput) -> str:
        self.seen.append(inp)
        return self.reply


def _rag(container, reply):
    llm = ScriptedLLM(reply)
    return RAGService(container.repo, container.retrieval, llm, container.settings), llm


def test_answer_is_grounded_and_cited(container):
    resp = container.rag.answer(ChatRequest(question="How long do OAuth access tokens last before they expire?"))
    assert not resp.abstained and "60 minutes" in resp.answer
    assert resp.citations and all(f"[{c.index}]" in resp.answer for c in resp.citations)
    assert any(c.title == "Authentication and tokens" for c in resp.citations)
    report = check_response(resp)
    assert report.ok, report.problems


def test_citations_point_to_retrieved_chunks_with_urls(container):
    resp = container.rag.answer(ChatRequest(question="How are webhook signatures computed?"))
    retrieved_ids = {r.chunk_id for r in resp.retrieved}
    assert resp.citations and all(c.chunk_id in retrieved_ids for c in resp.citations)
    assert resp.citations[0].url and resp.citations[0].snippet


def test_abstains_when_no_relevant_evidence(container):
    resp = container.rag.answer(ChatRequest(question="What is the capital of France?"))
    assert resp.abstained and resp.answer == ABSTAIN_MESSAGE and resp.citations == []
    assert resp.abstain_reason in {"no_relevant_evidence", "model_declined"}
    assert check_response(resp).ok


def test_abstains_on_empty_index(empty_container):
    resp = empty_container.rag.answer(ChatRequest(question="How do I revoke a token?"))
    assert resp.abstained and resp.abstain_reason == "no_relevant_evidence" and resp.retrieved == []


def test_low_similarity_never_reaches_the_llm(container):
    rag, llm = _rag(container, "should not be called [1]")
    rag.settings = container.settings.model_copy(update={"min_similarity": 0.99})
    resp = rag.answer(ChatRequest(question="How do I revoke a token?"))
    assert resp.abstained and llm.seen == []


def test_model_can_decline(container):
    rag, _ = _rag(container, "INSUFFICIENT_EVIDENCE")
    resp = rag.answer(ChatRequest(question="How do I revoke a token?"))
    assert resp.abstained and resp.abstain_reason == "model_declined"


def test_uncited_llm_answer_is_rejected(container):
    rag, _ = _rag(container, "Tokens can be revoked with a POST request.")
    resp = rag.answer(ChatRequest(question="How do I revoke a token?"))
    assert resp.abstained and resp.abstain_reason == "no_valid_citations"


def test_invalid_citation_markers_are_stripped(container):
    rag, _ = _rag(container, "Call the revoke endpoint [1][99].")
    resp = rag.answer(ChatRequest(question="How do I revoke a token?"))
    assert not resp.abstained and "[99]" not in resp.answer and [c.index for c in resp.citations] == [1]
    assert check_response(resp).citations_valid


def test_prompt_marks_context_as_untrusted_and_includes_history(container):
    rag, llm = _rag(container, "Answer [1].")
    rag.answer(ChatRequest(question="and how do I revoke it?",
                           history=[ChatTurn(role="user", content="How do refresh tokens work?"),
                                    ChatTurn(role="assistant", content="They rotate [1].")]))
    inp = llm.seen[0]
    assert "untrusted" in inp.system and "INSUFFICIENT_EVIDENCE" in inp.system
    assert inp.messages[0]["content"].startswith("How do refresh tokens") and inp.messages[-1]["role"] == "user"
    assert "[1]" in inp.messages[-1]["content"]


def test_followup_query_uses_previous_question_for_retrieval(container):
    rag = container.rag
    q = rag._retrieval_query(ChatRequest(question="and how to revoke it?",
                                         history=[ChatTurn(role="user", content="How do refresh tokens work?")]))
    assert "refresh tokens" in q and "revoke" in q
    assert rag._retrieval_query(ChatRequest(question="A completely standalone long question here please")) == \
        "A completely standalone long question here please"


def test_source_type_filter(container):
    resp = container.rag.answer(ChatRequest(question="Why does AsyncClient return 401 after refreshing the token?",
                                            source_types=[SourceType.github_issue]))
    assert all(r.source_type == SourceType.github_issue for r in resp.retrieved)
    assert not resp.abstained and any("issue" in c.url for c in resp.citations)


def test_parse_citations_handles_groups_and_order():
    p = parse_citations("A [2]. B [1, 3]. C [7].", 3)
    assert p.cited == [2, 1, 3] and p.invalid == [7] and "[7]" not in p.text and "[1][3]" in p.text


def test_is_refusal():
    assert is_refusal("INSUFFICIENT_EVIDENCE") and is_refusal("`INSUFFICIENT_EVIDENCE`.")
    assert not is_refusal("The answer is 42 [1]")


def test_grounding_checker_detects_bad_answers(container):
    resp = container.rag.answer(ChatRequest(question="How long do OAuth access tokens last before they expire?"))
    bad = resp.model_copy(update={"answer": "Bananas ripen quickly in warm tropical weather [1]. Another uncited claim goes here."})
    rep = check_response(bad)
    assert not rep.ok and rep.supported_ratio < 1 and not rep.every_claim_cited
    ghost = resp.model_copy(update={"answer": "Something [5]."})
    assert not check_response(ghost).citations_valid
