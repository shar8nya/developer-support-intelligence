"""Markdown-aware chunker.

Strategy
1. Split the document into sections at markdown headings, tracking the heading path
   ("Guide > Auth > Refresh tokens"), ignoring '#' inside fenced code blocks.
2. Split each section into blocks (paragraphs, fenced code blocks, list runs).
3. Greedily pack blocks into chunks up to `max_tokens`; oversized blocks are split by
   sentence/line. Consecutive chunks of one section overlap by ~`overlap_tokens`.
4. Every chunk is prefixed with its heading path so it is self-describing when retrieved.
"""
from __future__ import annotations

import re

from app.schemas import Chunk
from app.text_utils import estimate_tokens

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[])")


def _split_sections(text: str) -> list[tuple[list[str], str]]:
    """Return [(heading_path, section_body)] ignoring headings inside code fences."""
    sections: list[tuple[list[str], list[str]]] = []
    path: list[tuple[int, str]] = []
    current: list[str] = []
    current_path: list[str] = []
    in_fence = False

    def flush():
        body = "\n".join(current).strip()
        if body:
            sections.append((list(current_path), body))  # type: ignore[arg-type]

    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            flush()
            current = []
            level, title = len(m.group(1)), m.group(2).strip()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
            current_path = [t for _, t in path]
        else:
            current.append(line)
    flush()
    return sections  # type: ignore[return-value]


def _split_blocks(body: str) -> list[str]:
    blocks: list[str] = []
    buf: list[str] = []
    in_fence = False
    for line in body.split("\n"):
        if line.lstrip().startswith("```"):
            if not in_fence and buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            buf.append(line)
            if in_fence:  # closing fence
                blocks.append("\n".join(buf).strip())
                buf = []
            in_fence = not in_fence
            continue
        if in_fence:
            buf.append(line)
        elif line.strip() == "":
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b]


def _hard_split(text: str, max_tokens: int) -> list[str]:
    step = max(1, max_tokens * 4)
    return [text[i : i + step] for i in range(0, len(text), step)]


def _split_oversized(block: str, max_tokens: int) -> list[str]:
    """Split a block that is larger than the budget on lines (code) or sentences (prose)."""
    if estimate_tokens(block) <= max_tokens:
        return [block]
    multiline = "\n" in block
    units = block.split("\n") if multiline else _SENT_SPLIT_RE.split(block)
    joiner = "\n" if multiline else " "
    if len(units) == 1:
        return _hard_split(block, max_tokens)
    pieces: list[str] = []
    cur: list[str] = []
    for unit in units:
        if cur and estimate_tokens(joiner.join(cur + [unit])) > max_tokens:
            pieces.append(joiner.join(cur))
            cur = [unit]
        else:
            cur.append(unit)
    if cur:
        pieces.append(joiner.join(cur))
    out: list[str] = []
    for piece in pieces:  # a single unit may still be too large
        out.extend([piece] if estimate_tokens(piece) <= max_tokens * 1.5 else _hard_split(piece, max_tokens))
    return out


def _tail_overlap(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    chars = overlap_tokens * 4
    if len(text) <= chars:
        return text
    tail = text[-chars:]
    cut = tail.find(" ")
    return tail[cut + 1 :] if 0 <= cut < len(tail) // 2 else tail


def chunk_markdown(
    text: str,
    *,
    title: str = "",
    max_tokens: int = 350,
    overlap_tokens: int = 50,
    min_tokens: int = 8,
) -> list[Chunk]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    overlap_tokens = min(overlap_tokens, max_tokens // 2)
    chunks: list[Chunk] = []

    for heading_path, body in _split_sections(text) or [([], text.strip())]:
        if not body:
            continue
        path_str = " > ".join(heading_path) if heading_path else ""
        prefix_parts = [p for p in ([title] if title and title not in heading_path else []) + heading_path if p]
        prefix = " > ".join(prefix_parts)
        budget = max(20, max_tokens - estimate_tokens(prefix) - 2)

        pieces: list[str] = []
        for block in _split_blocks(body):
            pieces.extend(_split_oversized(block, budget))

        packed: list[str] = []
        cur = ""
        for piece in pieces:
            candidate = f"{cur}\n\n{piece}" if cur else piece
            if cur and estimate_tokens(candidate) > budget:
                packed.append(cur)
                overlap = _tail_overlap(cur, overlap_tokens)
                cur = f"{overlap}\n\n{piece}" if overlap and estimate_tokens(overlap + piece) <= budget else piece
            else:
                cur = candidate
        if cur:
            packed.append(cur)

        for content in packed:
            content = content.strip()
            if not content:
                continue
            full = f"{prefix}\n\n{content}" if prefix else content
            if estimate_tokens(full) < min_tokens and chunks and not heading_path:
                continue
            chunks.append(
                Chunk(chunk_index=len(chunks), content=full, heading_path=path_str,
                      token_count=estimate_tokens(full))
            )
    return chunks
