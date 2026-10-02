# Supabase / PostgreSQL migrations

Run in order (`scripts/apply_migrations.py` does this for you and is idempotent):

| File | Purpose |
|------|---------|
| `001_extensions_and_tables.sql` | `vector`, tables: documents, document_chunks, ingestion_jobs, chat_interactions, feedback |
| `002_indexes.sql` | HNSW (cosine) vector index, GIN full-text index, FK/filter indexes |
| `003_search_functions.sql` | `match_chunks()` (vector) and `hybrid_search_chunks()` (vector + full-text, RRF) |
| `004_rls.sql` | Enables RLS (no policies) so anon keys cannot touch the data |

Alternatively paste each file, in order, into the Supabase **SQL Editor**.
