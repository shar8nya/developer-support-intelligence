-- 003: similarity + hybrid search functions
-- Both return the dense cosine similarity for every row (used for abstention decisions)
-- and are callable from SQL, PostgREST (supabase.rpc) or the backend's psycopg connection.

drop function if exists public.match_chunks(vector, int, text[]);
drop function if exists public.hybrid_search_chunks(text, vector, int, text[], float, float, int);

-- Pure vector search --------------------------------------------------------
create or replace function public.match_chunks(
    query_embedding vector(1536),
    match_count     int     default 5,
    source_types    text[]  default null
)
returns table (
    chunk_id uuid, document_id uuid, content text, heading_path text, metadata jsonb,
    title text, url text, source_uri text, source_type text,
    similarity float, score float
)
language sql stable
as $$
    select c.id, c.document_id, c.content, c.heading_path, c.metadata,
           d.title, d.url, d.source_uri, d.source_type,
           (1 - (c.embedding <=> query_embedding))::float as similarity,
           (1 - (c.embedding <=> query_embedding))::float as score
    from public.document_chunks c
    join public.documents d on d.id = c.document_id
    where source_types is null or d.source_type = any(source_types)
    order by c.embedding <=> query_embedding
    limit least(match_count, 200);
$$;

-- Hybrid search: full-text rank + vector rank fused with Reciprocal Rank Fusion ------
create or replace function public.hybrid_search_chunks(
    query_text       text,
    query_embedding  vector(1536),
    match_count      int    default 5,
    source_types     text[] default null,
    full_text_weight float  default 1.0,
    semantic_weight  float  default 1.0,
    rrf_k            int    default 60
)
returns table (
    chunk_id uuid, document_id uuid, content text, heading_path text, metadata jsonb,
    title text, url text, source_uri text, source_type text,
    similarity float, score float
)
language sql stable
as $$
    with full_text as (
        select c.id,
               row_number() over (order by ts_rank_cd(c.fts, websearch_to_tsquery('english', query_text)) desc) as rank_ix
        from public.document_chunks c
        join public.documents d on d.id = c.document_id
        where c.fts @@ websearch_to_tsquery('english', query_text)
          and (source_types is null or d.source_type = any(source_types))
        order by rank_ix
        limit least(match_count, 200) * 2
    ),
    semantic as (
        select c.id,
               row_number() over (order by c.embedding <=> query_embedding) as rank_ix
        from public.document_chunks c
        join public.documents d on d.id = c.document_id
        where source_types is null or d.source_type = any(source_types)
        order by rank_ix
        limit least(match_count, 200) * 2
    ),
    fused as (
        select coalesce(f.id, s.id) as id,
               coalesce(1.0 / (rrf_k + f.rank_ix), 0.0) * full_text_weight +
               coalesce(1.0 / (rrf_k + s.rank_ix), 0.0) * semantic_weight as rrf
        from full_text f
        full outer join semantic s on f.id = s.id
    )
    select c.id, c.document_id, c.content, c.heading_path, c.metadata,
           d.title, d.url, d.source_uri, d.source_type,
           (1 - (c.embedding <=> query_embedding))::float as similarity,
           fused.rrf::float as score
    from fused
    join public.document_chunks c on c.id = fused.id
    join public.documents d on d.id = c.document_id
    order by fused.rrf desc
    limit least(match_count, 200);
$$;
