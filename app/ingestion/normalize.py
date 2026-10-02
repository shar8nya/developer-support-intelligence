"""Content extraction and normalisation for the different source formats."""
from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MDX_IMPORT_RE = re.compile(r"^\s*(import|export)\s+.*$", re.MULTILINE)
_JSX_SELF_CLOSING_RE = re.compile(r"<[A-Z][A-Za-z0-9]*[^>]*/>")
_JSX_TAG_RE = re.compile(r"</?[A-Z][A-Za-z0-9]*[^>]*>")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse a simple `key: value` YAML front matter block (no nested YAML)."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t", "-")):
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip().strip("'\"")
    return meta, text[match.end():]


def normalize_markdown(text: str, *, is_mdx: bool = False) -> str:
    """Normalise markdown: unify newlines, drop comments / MDX plumbing, collapse blank lines.

    Code blocks are preserved verbatim - they are often the most useful part of docs.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = _HTML_COMMENT_RE.sub("", text)
    if is_mdx:
        # Only touch text outside fenced code blocks.
        parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
        for i in range(0, len(parts), 2):
            p = _MDX_IMPORT_RE.sub("", parts[i])
            p = _JSX_SELF_CLOSING_RE.sub("", p)
            p = _JSX_TAG_RE.sub("", p)
            parts[i] = p
        text = "".join(parts)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _MULTI_BLANK_RE.sub("\n\n", text).strip()


def first_heading(text: str) -> str | None:
    match = re.search(r"^#\s+(.+?)\s*#*\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"}
    _BLOCK = {"p", "div", "section", "article", "li", "br", "tr", "table", "ul", "ol", "pre", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._heading: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and not self._skip_depth:
            self._heading = "#" * int(tag[1])
            self.parts.append("\n\n" + self._heading + " ")
        elif tag == "code" and not self._skip_depth:
            self.parts.append("`")
        elif tag in self._BLOCK and not self._skip_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading = None
            self.parts.append("\n\n")
        elif tag == "code" and not self._skip_depth:
            self.parts.append("`")
        elif tag in self._BLOCK and not self._skip_depth:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)


def html_to_markdown_text(html: str) -> tuple[str, str]:
    """Return (title, markdown-ish text) from an HTML page using only the stdlib."""
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return parser.title.strip(), normalize_markdown(text)


def content_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
