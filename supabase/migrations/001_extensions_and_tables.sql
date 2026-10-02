-- 001: extensions + core tables
-- Works on Supabase and on any PostgreSQL 14+ with the pgvector extension (>= 0.5 for HNSW).
-- NOTE: vector(1536) matches text-embedding-3-small. If you change EMBEDDING_DIMENSIONS,
--       change every vector(1536) in migrations 001 and 003 BEFORE running them.

create extension if not exists vector;
create extension if not exists pgcrypto;   -- gen_random_uuid() on PG < 13

create table if not exists public.documents (
    id           uuid primary key default gen_random_uuid(),
    source_type  text        not null check (source_type in ('docs','github_issue','web','local')),
    source_uri   text        not null unique,          -- stable identity (URL / repo path)
    title        text        not null,
    url          text,
    content_hash text        not null,                 -- sha256 of normalised content (dedupe)
    metadata     jsonb       not null default '{}'::jsonb,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now()
);

create table if not exists public.document_chunks (
    id            uuid primary key default gen_random_uuid(),
    document_id   uuid        not null references public.documents(id) on delete cascade,
    chunk_index   integer     not null,
    content       text        not null,
    heading_path  text        not null default '',
    token_count   integer     not null default 0,
    embedding     vector(1536) not null,
    metadata      jsonb       not null default '{}'::jsonb,
    -- generated column powering keyword (BM25-style) search for hybrid retrieval
    fts           tsvector generated always as (to_tsvector('english', content)) stored,
    created_at    timestamptz not null default now(),
    unique (document_id, chunk_index)
);

create table if not exists public.ingestion_jobs (
    id          uuid primary key default gen_random_uuid(),
    status      text        not null default 'queued'
                check (status in ('queued','running','succeeded','failed')),
    source      text        not null,
    params      jsonb       not null default '{}'::jsonb,
    stats       jsonb       not null default '{}'::jsonb,
    error       text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create table if not exists public.chat_interactions (
    id          uuid primary key default gen_random_uuid(),
    question    text        not null,
    answer      text        not null,
    abstained   boolean     not null default false,
    citations   jsonb       not null default '[]'::jsonb,
    latency_ms  integer,
    created_at  timestamptz not null default now()
);

create table if not exists public.feedback (
    id             uuid primary key default gen_random_uuid(),
    interaction_id uuid        not null references public.chat_interactions(id) on delete cascade,
    rating         smallint    not null check (rating in (-1, 1)),
    comment        text,
    created_at     timestamptz not null default now()
);
