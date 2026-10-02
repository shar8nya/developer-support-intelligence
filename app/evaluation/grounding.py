"""Citation-correctness and groundedness checks for a ChatResponse (used by tests and run_eval)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.evaluation.metrics import sentence_support
from app.schemas import ChatResponse

_MARKER_RE = re.compile(r"\[(\d+)\]")
# a "sentence" ends at . ! ? (plus any trailing [n] markers) followed by whitespace/end, or at a newline.
# A '.' inside "1.4.2" or "api.example.com" is not followed by whitespace, so it does not split.
_SENTENCE_RE = re.compile(r"[^\n]+?(?:[.!?](?:\s*\[\d+\])*(?=\s|$)|$)")


@dataclass
class GroundingReport:
    citations_valid: bool = True            # every [n] in the text maps to a returned citation
    all_cited_in_retrieved: bool = True     # every citation points at a chunk that was retrieved
    every_claim_cited: bool = True          # every non-trivial sentence carries a marker
    supported_ratio: float = 1.0            # share of cited sentences lexically supported by their source
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.citations_valid and self.all_cited_in_retrieved and self.every_claim_cited and self.supported_ratio >= 0.999


def check_response(resp: ChatResponse, min_overlap: float = 0.6) -> GroundingReport:
    rep = GroundingReport()
    if resp.abstained:
        if resp.citations:
            rep.citations_valid = False
            rep.problems.append("abstained answer must not carry citations")
        return rep

    by_index = {c.index: c for c in resp.citations}
    retrieved_ids = {r.chunk_id for r in resp.retrieved}
    full_text = {r.chunk_id: r.content for r in resp.retrieved}

    for n in {int(m) for m in _MARKER_RE.findall(resp.answer)}:
        if n not in by_index:
            rep.citations_valid = False
            rep.problems.append(f"marker [{n}] has no matching citation")
    for c in resp.citations:
        if c.chunk_id not in retrieved_ids:
            rep.all_cited_in_retrieved = False
            rep.problems.append(f"citation [{c.index}] not in retrieved set")
    if not resp.citations:
        rep.every_claim_cited = False
        rep.problems.append("non-abstained answer has no citations")

    supported = total = 0
    for sentence in _SENTENCE_RE.findall(resp.answer):
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue
        markers = [int(m) for m in _MARKER_RE.findall(sentence)]
        if not markers:
            rep.every_claim_cited = False
            rep.problems.append(f"uncited sentence: {sentence[:60]!r}")
            continue
        evidence = " ".join(full_text.get(by_index[m].chunk_id, "") for m in markers if m in by_index)
        total += 1
        if sentence_support(_MARKER_RE.sub("", sentence), evidence, min_overlap):
            supported += 1
        else:
            rep.problems.append(f"weakly supported: {sentence[:60]!r}")
    rep.supported_ratio = supported / total if total else 1.0
    return rep
