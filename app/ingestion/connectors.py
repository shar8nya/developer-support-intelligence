"""Source connectors. Each yields RawDocument objects; the pipeline does chunking/embedding/storage."""
from __future__ import annotations

import ipaddress
import json
import logging
import socket
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import httpx

from app.exceptions import IngestionError
from app.ingestion.normalize import first_heading, html_to_markdown_text, normalize_markdown, parse_frontmatter
from app.schemas import RawDocument, SourceType

log = logging.getLogger(__name__)
GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"
_TEXT_EXT = {".md", ".mdx", ".markdown", ".txt"}


class Connector(ABC):
    @abstractmethod
    def fetch(self) -> Iterator[RawDocument]: ...


# --------------------------------------------------------------------- helpers
def markdown_to_document(text: str, *, source_uri: str, fallback_title: str, url: Optional[str],
                         source_type: SourceType, is_mdx: bool = False,
                         extra_meta: Optional[dict[str, Any]] = None) -> RawDocument:
    meta, body = parse_frontmatter(text.replace("\r\n", "\n"))
    body = normalize_markdown(body, is_mdx=is_mdx)
    title = meta.get("title") or first_heading(body) or fallback_title
    metadata: dict[str, Any] = {k: v for k, v in meta.items() if k in {"description", "tags", "category", "product"}}
    metadata.update(extra_meta or {})
    return RawDocument(source_type=source_type, source_uri=source_uri, title=title, content=body,
                       url=meta.get("url") or url, metadata=metadata)


def issue_to_document(issue: dict[str, Any], comments: list[dict[str, Any]], repo: str,
                      max_comments: int = 10) -> RawDocument:
    """Turn a GitHub-API-shaped issue (+comments) into a normalised document."""
    labels = [l["name"] if isinstance(l, dict) else str(l) for l in issue.get("labels", [])]
    parts = [
        f"# {issue['title']}",
        f"Issue #{issue['number']} in {repo} — state: {issue.get('state', 'unknown')}"
        + (f" — labels: {', '.join(labels)}" if labels else ""),
        "## Description",
        (issue.get("body") or "_No description provided._").strip(),
    ]
    good = [c for c in comments if (c.get("body") or "").strip()][:max_comments]
    if good:
        parts.append("## Discussion")
        for c in good:
            user = (c.get("user") or {}).get("login", "unknown")
            parts.append(f"### Comment by @{user}\n{c['body'].strip()}")
    return RawDocument(
        source_type=SourceType.github_issue,
        source_uri=f"github:{repo}#issue-{issue['number']}",
        title=f"#{issue['number']}: {issue['title']}",
        content=normalize_markdown("\n\n".join(parts)),
        url=issue.get("html_url") or f"https://github.com/{repo}/issues/{issue['number']}",
        metadata={"repo": repo, "number": issue["number"], "state": issue.get("state"), "labels": labels,
                  "created_at": issue.get("created_at"), "comments": issue.get("comments", len(good))},
    )


def _github_headers(token: Optional[str]) -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
         "User-Agent": "developer-support-intelligence"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _check_github(resp: httpx.Response, what: str) -> None:
    if resp.status_code == 200:
        return
    if resp.status_code in (403, 429) and resp.headers.get("x-ratelimit-remaining") == "0":
        raise IngestionError(f"GitHub API rate limit reached while {what}. Set GITHUB_TOKEN to raise the limit.")
    if resp.status_code == 404:
        raise IngestionError(f"GitHub returned 404 while {what} (check repo/branch/path, or private repo needs a token).")
    raise IngestionError(f"GitHub API error {resp.status_code} while {what}: {resp.text[:200]}")


# ----------------------------------------------------------------- local files
class LocalConnector(Connector):
    """Directory of .md/.mdx/.txt files and .json files containing GitHub-API-shaped issues."""

    def __init__(self, path: Path):
        self.root = Path(path)
        if not self.root.exists():
            raise IngestionError(f"Path does not exist: {self.root}")

    def fetch(self) -> Iterator[RawDocument]:
        files = [self.root] if self.root.is_file() else sorted(p for p in self.root.rglob("*") if p.is_file())
        base = self.root.parent if self.root.is_file() else self.root
        for f in files:
            rel = f.relative_to(base).as_posix()
            suffix = f.suffix.lower()
            try:
                if suffix in _TEXT_EXT:
                    yield markdown_to_document(
                        f.read_text(encoding="utf-8"), source_uri=f"local:{rel}", fallback_title=f.stem.replace("-", " ").replace("_", " ").title(),
                        url=None, source_type=SourceType.docs, is_mdx=suffix == ".mdx")
                elif suffix == ".json":
                    data = json.loads(f.read_text(encoding="utf-8"))
                    for entry in (data if isinstance(data, list) else [data]):
                        yield issue_to_document(entry, entry.get("comments_data", []), entry.get("repo", "example/repo"))
            except (OSError, ValueError, KeyError) as exc:
                log.warning("skipping %s: %s", f, exc)


# ---------------------------------------------------------------- GitHub docs
class GitHubDocsConnector(Connector):
    def __init__(self, repo: str, branch: str = "master", path_prefix: str = "", max_files: int = 50,
                 token: Optional[str] = None, client: Optional[httpx.Client] = None):
        self.repo, self.branch, self.prefix, self.max_files = repo, branch, path_prefix.strip("/"), max_files
        self.client = client or httpx.Client(timeout=30, headers=_github_headers(token), follow_redirects=True)

    def _tree(self) -> list[dict]:
        r = self.client.get(f"{GITHUB_API}/repos/{self.repo}/git/trees/{self.branch}", params={"recursive": "1"})
        if r.status_code == 404:  # maybe the branch name is wrong -> use the repo's default branch
            meta = self.client.get(f"{GITHUB_API}/repos/{self.repo}")
            _check_github(meta, "looking up the repository")
            self.branch = meta.json()["default_branch"]
            r = self.client.get(f"{GITHUB_API}/repos/{self.repo}/git/trees/{self.branch}", params={"recursive": "1"})
        _check_github(r, "listing repository files")
        data = r.json()
        if data.get("truncated"):
            log.warning("GitHub tree was truncated; narrow path_prefix for full coverage")
        return data["tree"]

    def fetch(self) -> Iterator[RawDocument]:
        entries = [e for e in self._tree()
                   if e["type"] == "blob" and Path(e["path"]).suffix.lower() in {".md", ".mdx"}
                   and (not self.prefix or e["path"].startswith(self.prefix + "/") or e["path"] == self.prefix)]
        entries.sort(key=lambda e: e["path"])
        if not entries:
            raise IngestionError(f"No .md/.mdx files found under '{self.prefix}' in {self.repo}@{self.branch}")
        for e in entries[: self.max_files]:
            path = e["path"]
            r = self.client.get(f"{GITHUB_RAW}/{self.repo}/{self.branch}/{path}")
            if r.status_code != 200:
                log.warning("could not fetch %s (%s)", path, r.status_code)
                continue
            yield markdown_to_document(
                r.text, source_uri=f"github:{self.repo}/{path}",
                fallback_title=Path(path).stem.replace("-", " ").replace("_", " ").title(),
                url=f"https://github.com/{self.repo}/blob/{self.branch}/{path}",
                source_type=SourceType.docs, is_mdx=path.endswith(".mdx"),
                extra_meta={"repo": self.repo, "path": path, "branch": self.branch})


# -------------------------------------------------------------- GitHub issues
class GitHubIssuesConnector(Connector):
    def __init__(self, repo: str, state: str = "all", labels: Optional[str] = None, max_issues: int = 50,
                 include_comments: bool = True, token: Optional[str] = None,
                 client: Optional[httpx.Client] = None):
        self.repo, self.state, self.labels = repo, state, labels
        self.max_issues, self.include_comments = max_issues, include_comments
        self.client = client or httpx.Client(timeout=30, headers=_github_headers(token), follow_redirects=True)

    def fetch(self) -> Iterator[RawDocument]:
        seen, page = 0, 1
        while seen < self.max_issues:
            params: dict[str, Any] = {"state": self.state, "per_page": min(100, max(self.max_issues, 1)),
                                      "page": page, "sort": "updated", "direction": "desc"}
            if self.labels:
                params["labels"] = self.labels
            r = self.client.get(f"{GITHUB_API}/repos/{self.repo}/issues", params=params)
            _check_github(r, "listing issues")
            batch = r.json()
            if not batch:
                return
            for issue in batch:
                if "pull_request" in issue:      # the issues API also returns PRs
                    continue
                comments: list[dict] = []
                if self.include_comments and issue.get("comments", 0) > 0:
                    cr = self.client.get(issue["comments_url"], params={"per_page": 20})
                    if cr.status_code == 200:
                        comments = cr.json()
                    else:
                        log.warning("comments for #%s unavailable (%s)", issue["number"], cr.status_code)
                yield issue_to_document(issue, comments, self.repo)
                seen += 1
                if seen >= self.max_issues:
                    return
            page += 1


# ------------------------------------------------------------------ web pages
def assert_public_http_url(url: str) -> None:
    """SSRF guard: only http(s) to publicly routable hosts."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise IngestionError(f"Only http(s) URLs are allowed: {url}")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise IngestionError(f"Cannot resolve host {parsed.hostname}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise IngestionError(f"Refusing to fetch non-public address for {parsed.hostname}")


class UrlConnector(Connector):
    def __init__(self, urls: list[str], client: Optional[httpx.Client] = None, check_ssrf: bool = True):
        self.urls, self.check_ssrf = urls, check_ssrf
        self.client = client or httpx.Client(timeout=30, follow_redirects=True,
                                             headers={"User-Agent": "developer-support-intelligence"})

    def fetch(self) -> Iterator[RawDocument]:
        for url in self.urls:
            if self.check_ssrf:
                assert_public_http_url(url)
            r = self.client.get(url)
            if r.status_code != 200:
                raise IngestionError(f"GET {url} returned {r.status_code}")
            ctype = r.headers.get("content-type", "")
            if "html" in ctype:
                title, text = html_to_markdown_text(r.text)
            else:
                title, text = "", normalize_markdown(r.text)
            yield RawDocument(source_type=SourceType.web, source_uri=url,
                              title=title or first_heading(text) or url, content=text, url=url,
                              metadata={"content_type": ctype})
