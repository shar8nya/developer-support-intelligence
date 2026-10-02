from app.ingestion.chunker import chunk_markdown
from app.ingestion.normalize import (
    content_hash, html_to_markdown_text, normalize_markdown, parse_frontmatter,
)
from app.text_utils import estimate_tokens, stem


def test_frontmatter_parsed_and_removed():
    meta, body = parse_frontmatter("---\ntitle: Hello\nurl: http://x\n---\n# Body\n")
    assert meta == {"title": "Hello", "url": "http://x"}
    assert body.startswith("# Body")


def test_normalize_collapses_blank_lines_and_strips_comments():
    out = normalize_markdown("a\r\n\r\n\r\n\r\nb <!-- hidden --> c   \n")
    assert out == "a\n\nb  c"


def test_mdx_plumbing_removed_but_code_preserved():
    text = "import X from 'x'\n\n<Note type=\"x\">hi</Note>\n\n```jsx\n<Note/>\n```\n"
    out = normalize_markdown(text, is_mdx=True)
    assert "import X" not in out and "<Note type" not in out
    assert "```jsx\n<Note/>\n```" in out


def test_html_extraction_drops_nav_and_scripts():
    html = "<html><title>T</title><nav>menu</nav><script>evil()</script><h1>Head</h1><p>Body text</p></html>"
    title, text = html_to_markdown_text(html)
    assert title == "T" and "# Head" in text and "Body text" in text
    assert "menu" not in text and "evil" not in text


def test_hash_changes_with_content():
    assert content_hash("a", "b") != content_hash("a", "c")
    assert content_hash("a", "b") == content_hash("a", "b")


def test_stemmer_conflates_variants():
    assert stem("retries") == stem("retry") == stem("retried")
    assert stem("expires") == stem("expired")
    assert stem("tokens") == stem("token")


def test_chunks_respect_size_and_carry_heading_path():
    text = "# Guide\n\n## Auth\n\n" + " ".join(f"Sentence {i} about tokens." for i in range(120))
    chunks = chunk_markdown(text, title="Doc", max_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(c.token_count <= 110 for c in chunks)
    assert all(c.heading_path == "Guide > Auth" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].content.startswith("Doc > Guide > Auth")


def test_code_block_hash_lines_are_not_headings():
    text = "# T\n\n```bash\n# just a comment\nls\n```\n\nafter"
    chunks = chunk_markdown(text, title="T")
    assert len(chunks) == 1 and "# just a comment" in chunks[0].content


def test_giant_code_block_is_split_without_loss():
    code = "\n".join(f"line_{i} = {i}" for i in range(500))
    chunks = chunk_markdown(f"# T\n\n```py\n{code}\n```", title="T", max_tokens=80, overlap_tokens=0)
    joined = "\n".join(c.content for c in chunks)
    assert all(f"line_{i} = {i}" in joined for i in (0, 250, 499))
    assert max(estimate_tokens(c.content) for c in chunks) < 130


def test_document_without_headings_is_chunked():
    chunks = chunk_markdown("Just a paragraph of text that has no headings at all.", title="Plain")
    assert len(chunks) == 1
