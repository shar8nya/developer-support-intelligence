-- 002: indexes
-- Approximate nearest-neighbour index (cosine). HNSW needs no training data, so it is fine on an empty table.
create index if not exists document_chunks_embedding_hnsw
    on public.document_chunks using hnsw (embedding vector_cosine_ops);

-- Keyword search
create index if not exists document_chunks_fts_gin
    on public.document_chunks using gin (fts);

-- Joins / filters
create index if not exists document_chunks_document_id_idx on public.document_chunks (document_id);
create index if not exists documents_source_type_idx        on public.documents (source_type);
create index if not exists documents_content_hash_idx       on public.documents (content_hash);
create index if not exists feedback_interaction_id_idx      on public.feedback (interaction_id);
create index if not exists ingestion_jobs_created_at_idx    on public.ingestion_jobs (created_at desc);
