-- 004: Row Level Security
-- The FastAPI backend connects with the database owner ("postgres" role), which bypasses RLS.
-- Enabling RLS with NO policies means the public anon/authenticated keys exposed by
-- Supabase's REST API cannot read or write any of these tables. Add policies only if you
-- intentionally want to expose data to browser clients.
alter table public.documents          enable row level security;
alter table public.document_chunks    enable row level security;
alter table public.ingestion_jobs     enable row level security;
alter table public.chat_interactions  enable row level security;
alter table public.feedback           enable row level security;
