"""Command-line ingestion (no API server needed).

Examples (from the project root, virtualenv active):
    python -m app.ingestion.cli local --path data/sample_docs
    python -m app.ingestion.cli github-docs --repo supabase/supabase --branch master \
           --path-prefix apps/docs/content/guides/auth --max-files 40
    python -m app.ingestion.cli github-issues --repo supabase/supabase --max-issues 30
    python -m app.ingestion.cli url --url https://example.com/page
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from app.config import get_settings
from app.container import build_container
from app.exceptions import AppError
from app.schemas import IngestRequest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ingest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("local", "github-docs", "github-issues", "url"):
        sp = sub.add_parser(name)
        sp.add_argument("--force", action="store_true", help="re-embed even if unchanged")
        if name == "local":
            sp.add_argument("--path", required=True)
        elif name in ("github-docs", "github-issues"):
            sp.add_argument("--repo", required=True, help="owner/name")
            if name == "github-docs":
                sp.add_argument("--branch", default="master")
                sp.add_argument("--path-prefix", default="")
                sp.add_argument("--max-files", type=int, default=50)
            else:
                sp.add_argument("--state", default="all", choices=["open", "closed", "all"])
                sp.add_argument("--labels")
                sp.add_argument("--max-issues", type=int, default=50)
                sp.add_argument("--no-comments", action="store_true")
        else:
            sp.add_argument("--url", action="append", required=True, dest="urls")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    fields = {"source": a.cmd.replace("-", "_"), "force": a.force}
    if a.cmd == "local":
        fields["path"] = a.path
    elif a.cmd == "github-docs":
        fields.update(repo=a.repo, branch=a.branch, path_prefix=a.path_prefix, max_files=a.max_files)
    elif a.cmd == "github-issues":
        fields.update(repo=a.repo, state=a.state, labels=a.labels, max_issues=a.max_issues,
                      include_comments=not a.no_comments)
    else:
        fields["urls"] = a.urls

    settings = get_settings()
    problems = settings.validate_live()
    if problems:
        print("Configuration problems:\n - " + "\n - ".join(problems), file=sys.stderr)
        return 2
    if settings.is_demo:
        print("NOTE: APP_MODE=demo uses an in-memory store that is discarded when this command exits. "
              "Set APP_MODE=live (with DATABASE_URL + API keys) to persist to Supabase.", file=sys.stderr)
    container = build_container(settings, auto_ingest_demo=False)
    try:
        stats = container.pipeline.run(IngestRequest(**fields))
    except AppError as exc:
        print(f"ERROR: {exc.message}", file=sys.stderr)
        return 1
    print(json.dumps(stats.model_dump(), indent=2))
    return 0 if stats.documents_failed < max(stats.documents_seen, 1) else 1


if __name__ == "__main__":
    sys.exit(main())
