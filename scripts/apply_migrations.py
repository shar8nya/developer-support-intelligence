"""Apply supabase/migrations/*.sql in order using DATABASE_URL (idempotent).

Usage (from the project root, virtualenv active):
    python scripts/apply_migrations.py
    python scripts/apply_migrations.py --database-url postgresql://...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = ap.parse_args()
    if not args.database_url:
        print("ERROR: set DATABASE_URL in .env or pass --database-url", file=sys.stderr)
        return 2
    files = sorted((ROOT / "supabase" / "migrations").glob("*.sql"))
    if not files:
        print("ERROR: no migration files found", file=sys.stderr)
        return 2
    with psycopg.connect(args.database_url, autocommit=True, prepare_threshold=None) as conn:
        for f in files:
            print(f"Applying {f.name} ...")
            conn.execute(f.read_text(encoding="utf-8"))
        row = conn.execute("select extversion::text from pg_extension where extname = 'vector'").fetchone()
        version = row[0].decode() if row and isinstance(row[0], bytes) else (row[0] if row else "NOT INSTALLED")
        print(f"Done. pgvector version: {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
