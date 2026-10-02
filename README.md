# Developer Support Intelligence — an AI-powered RAG support agent

Answers developer questions from **technical documentation** and **public GitHub issues**, with
**citations** to the original sources, and **abstains** when the evidence is insufficient.

Stack: Python · FastAPI · Supabase PostgreSQL + pgvector · Streamlit · Pydantic · OpenAI-compatible embeddings · OpenAI LLM.

### Why this project?

Developer support often means searching through documentation, GitHub issues, and scattered technical resources to find one reliable answer.

This project builds a support agent that:

- 🔎 Retrieves relevant documentation using vector + full-text hybrid search
- 🤖 Generates answers using an LLM grounded in retrieved evidence
- 🔗 Provides citations back to the original sources
- 🛑 Abstains when there isn't enough evidence instead of guessing
- 📚 Ingests documentation, GitHub issues, local files, and public web pages
- 💬 Provides a Streamlit chat interface
- 📊 Includes retrieval and answer-grounding evaluation

> **Two modes.**
> **Demo mode (default)** needs *no* API keys and *no* database: offline hashing embeddings, an extractive
> "answerer" (quotes relevant sentences with citations) and an in-memory store preloaded with a small **fictional** corpus ("Acme Tasks API").
> It exists so you can run everything immediately. It is *not* real AI quality.
> **Live mode** uses Supabase/Postgres + a real embedding model + a real LLM (requires your credentials).

---------------------------------------------------------------------

## Contents
1. [Quick start on Windows (demo, 3 minutes)](#quick-start-on-windows-demo-3-minutes)
2. [What was tested and what was not](#what-was-tested-and-what-was-not)
3. [Architecture](#architecture) · [Ingestion pipeline](#ingestion-pipeline) · [Query flow](#query-flow)
4. [Database schema](#database-schema)
5. [RAG pipeline explained](#rag-pipeline-explained)
6. [Live mode: Supabase + API keys](#live-mode-supabase--api-keys)
7. [Ingesting documents](#ingesting-documents)
8. [API reference](#api-reference)
9. [Evaluation](#evaluation) · [Running tests](#running-tests)
10. [Library choices](#library-choices) · [Project structure](#project-structure)
11. [Known limitations](#known-limitations) · [Security notes](#security-notes) · [Troubleshooting](#troubleshooting)

---------------------------------------------------------------------

## Quick start on Windows (demo, 3 minutes)

Prerequisite: **Python 3.10+** from python.org (tick *“Add python.exe to PATH”*). Extract the ZIP, open **Command Prompt** in the extracted `developer-support-intelligence` folder:

```bat
scripts\setup.bat
scripts\run_api.bat
```
Open a **second** Command Prompt in the same folder:
```bat
scripts\run_frontend.bat
```
Browser opens at http://localhost:8501 (API docs: http://127.0.0.1:8000/docs). Ask *“How long do OAuth access tokens last?”*, then *“What is the capital of France?”* (it should abstain).

<details><summary>Same thing manually (PowerShell)</summary>

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
copy .env.example .env
uvicorn app.main:app --port 8000      # terminal 1
streamlit run frontend/app.py         # terminal 2
```
macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`, the rest is identical (use `cp` instead of `copy`).
</details>

---------------------------------------------------------------------

## What was tested and what was not

Honest status, produced while building this project in a Linux sandbox (Python 3.12; fastapi 0.141, pydantic 2.13, psycopg 3.3, pgvector-python 0.5, streamlit 1.64, numpy 2.5):

| Area | Status |
|---|---|
| Chunking, normalisation, metrics, config, providers (HTTP mocked), connectors (HTTP mocked), API, RAG logic, evaluation, Streamlit UI (headless `AppTest`) | ✅ automated tests pass (`pytest`) |
| SQL migrations, `match_chunks`, `hybrid_search_chunks`, `SupabaseRepository`, ingest→chat→feedback over real HTTP | ✅ tested against **real PostgreSQL 16 + pgvector 0.6.0** (local, not Supabase-hosted) using the offline demo providers |
| Real GitHub markdown fetch/parse/chunk (`raw.githubusercontent.com`, Supabase docs `.mdx`) | ✅ parsing/chunking verified on real files |
| GitHub **API** connectors against live api.github.com | ⚠️ **not verified live** — the sandbox hit the anonymous rate limit (60 req/h). Verified only with mocked responses. Set `GITHUB_TOKEN`. |
| **OpenAI embeddings / OpenAI LLM / Anthropic LLM** | ⚠️ **not tested with real credentials.** Request/response handling is unit-tested against mocked HTTP; the actual endpoints were never called. |
| **Supabase-hosted** database (pooler URLs, hosted pgvector version) | ⚠️ not tested; local Postgres only. Code disables prepared statements for the transaction pooler, but that path is unexercised. |
| Answer quality with a real LLM / real embeddings | ⚠️ **unknown.** No evaluation numbers for live mode exist yet — run `python -m app.evaluation.run_eval --mode live` yourself. |
| Windows `.bat` scripts | ⚠️ written but **not executed on Windows** (sandbox is Linux). The equivalent commands were run. |

Features needing external credentials: live embeddings (`OPENAI_API_KEY`), live LLM (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`), Supabase (`DATABASE_URL`), higher GitHub limits (`GITHUB_TOKEN`, optional).

---------------------------------------------------------------------

## Architecture

```mermaid
flowchart LR
    subgraph Client
        UI[Streamlit UI<br/>frontend/app.py]
        CLI[Ingestion CLI<br/>app.ingestion.cli]
    end
    subgraph API[FastAPI backend]
        R[/health · /chat · /search<br/>/ingest · /ingest/id · /feedback/]
        RAG[RAGService]
        RET[RetrievalService<br/>vector · hybrid · rerank]
        PIPE[IngestionPipeline]
    end
    subgraph Providers[app/providers]
        EMB[Embedding provider<br/>OpenAI-compatible | demo hashing]
        LLM[LLM provider<br/>OpenAI | Anthropic | demo extractive]
    end
    subgraph Sources[Connectors]
        GD[GitHub docs .md/.mdx]
        GI[GitHub issues + comments]
        LOC[Local files] --- WEB[Web pages]
    end
    REPO[(Repository interface)]
    PG[(Supabase Postgres<br/>documents · document_chunks<br/>pgvector + tsvector)]
    MEM[(In-memory store<br/>demo / tests)]

    UI -->|HTTP| R
    R --> RAG --> RET --> EMB
    RAG --> LLM
    RET --> REPO
    R --> PIPE
    CLI --> PIPE
    PIPE --> Sources
    PIPE --> EMB
    PIPE --> REPO
    REPO --> PG
    REPO --> MEM
```

## Ingestion pipeline

```mermaid
flowchart TD
    A[Connector yields RawDocument] --> B[Normalise markdown / MDX / HTML]
    B --> C{"content_hash =<br/>sha256(title + text + chunk config + embedding model)"}
    C -->|same hash & not --force| D[Skip: unchanged]
    C -->|new or changed| E[Markdown-aware chunking<br/>heading path, code fences kept intact, overlap]
    E --> F[Batch-embed chunks]
    F --> G["One transaction:<br/>upsert document, delete old chunks, insert new chunks"]
    G --> H[Stats: new / updated / unchanged / failed]
```
Each document is isolated: one failing document is recorded in `stats.errors` and the run continues. Because the hash includes chunk size and embedding model, changing either re-indexes automatically.

## Query flow

```mermaid
sequenceDiagram
    participant U as User (Streamlit)
    participant A as FastAPI /chat
    participant R as RetrievalService
    participant D as Postgres (pgvector)
    participant L as LLM
    U->>A: question + history + options
    A->>R: search(query)
    R->>R: embed query
    R->>D: hybrid_search_chunks() (vector + full-text, RRF)
    D-->>R: candidates with similarity
    R->>R: optional lexical rerank
    R-->>A: top-k chunks
    alt best similarity < MIN_SIMILARITY
        A-->>U: abstain (LLM never called)
    else enough evidence
        A->>L: numbered context + strict system prompt
        L-->>A: answer with [n] markers or INSUFFICIENT_EVIDENCE
        A->>A: validate markers, drop invalid, reject if none valid
        A-->>U: answer + citations (title, URL, snippet) + retrieved passages
    end
    A->>D: log interaction (for feedback)
```

---------------------------------------------------------------------

## Database schema

```mermaid
erDiagram
    documents ||--o{ document_chunks : "has (ON DELETE CASCADE)"
    chat_interactions ||--o{ feedback : "receives (ON DELETE CASCADE)"
    documents {
        uuid id PK
        text source_type "docs|github_issue|web|local"
        text source_uri UK "stable identity"
        text title
        text url
        text content_hash "dedupe"
        jsonb metadata
        timestamptz created_at
        timestamptz updated_at
    }
    document_chunks {
        uuid id PK
        uuid document_id FK
        int chunk_index "unique with document_id"
        text content
        text heading_path
        int token_count
        vector_1536 embedding "HNSW cosine index"
        tsvector fts "generated, GIN index"
        jsonb metadata
    }
    ingestion_jobs {
        uuid id PK
        text status "queued|running|succeeded|failed"
        text source
        jsonb params
        jsonb stats
        text error
    }
    chat_interactions {
        uuid id PK
        text question
        text answer
        bool abstained
        jsonb citations
        int latency_ms
    }
    feedback {
        uuid id PK
        uuid interaction_id FK
        smallint rating "-1 or 1"
        text comment
    }
```
SQL lives in `supabase/migrations/` (001 tables, 002 indexes, 003 search functions, 004 RLS).
* `match_chunks(query_embedding, match_count, source_types)` — cosine vector search.
* `hybrid_search_chunks(query_text, query_embedding, match_count, source_types, full_text_weight, semantic_weight, rrf_k)` — full-text rank + vector rank fused with Reciprocal Rank Fusion. Both return the dense `similarity` (used for abstention) and a `score`.
* **Dimension:** `vector(1536)` fits `text-embedding-3-small`. For another size, edit every `vector(1536)` in migrations 001 and 003 **before** applying, and set `EMBEDDING_DIMENSIONS`.
* RLS is enabled with no policies: Supabase's public anon key cannot read these tables. The backend connects as the database owner.

---------------------------------------------------------------------

## RAG pipeline explained

1. **Chunking** (`app/ingestion/chunker.py`): splits at markdown headings, keeps fenced code blocks whole (splitting by line only if oversized), packs paragraphs up to `CHUNK_MAX_TOKENS` (~4 chars/token estimate) with `CHUNK_OVERLAP_TOKENS` overlap, and prefixes each chunk with `Title > Heading > Subheading` so it is self-describing.
2. **Retrieval** (`services/retrieval.py`): embed query → `hybrid_search_chunks` (or pure vector). With reranking on, 3× candidates are fetched and re-scored by `0.5·similarity + 0.4·query-term coverage + 0.1·title/heading coverage` (`services/reranker.py`, dependency-free; swap in a cross-encoder for higher quality).
3. **Evidence gate**: if the best chunk's cosine similarity is below `MIN_SIMILARITY`, the system abstains **without calling the LLM**.
4. **Generation** (`services/generation.py`): numbered context blocks, strict system prompt (answer only from context, cite `[n]`, reply `INSUFFICIENT_EVIDENCE` if unsure, treat context as untrusted data, caveat open GitHub issues).
5. **Citation validation**: markers not in the context are stripped; if the answer has no valid citation (and `REQUIRE_CITATIONS=true`) it is rejected and the system abstains. Each returned citation includes title, source URL, heading path and snippet.
6. **Follow-ups**: short follow-up questions (≤5 words) get the previous user question prepended for retrieval only.

`MIN_SIMILARITY` is embedding-model dependent — **tune it** on your own eval set.

---------------------------------------------------------------------

## Live mode: Supabase + API keys

1. Create a project at https://supabase.com. Note the DB password.
2. **Connection string:** Project Settings → Database → *Connection string* → **URI**. On IPv4-only networks (most Windows home connections) use the **Session pooler** string. Put your password in it (URL-encode special characters).
3. Edit `.env`:
   ```
   APP_MODE=live
   DATABASE_URL=postgresql://postgres.<ref>:<password>@<pooler-host>:5432/postgres
   OPENAI_API_KEY=sk-...           # for embeddings (and the LLM if LLM_PROVIDER=openai)
   LLM_PROVIDER=openai             # or anthropic (then set ANTHROPIC_API_KEY and LLM_MODEL)
   ```
4. Apply migrations (either option):
   * `scripts\apply_migrations.bat` (uses `DATABASE_URL`, idempotent), **or**
   * paste `supabase/migrations/001…004*.sql` in order into Supabase **SQL Editor** → Run.
5. Start the API (`scripts\run_api.bat`). `GET /health` should show `"database": "supabase:ok"` and no `problems`.
6. Ingest something (next section), then open the UI.

Changing embedding model/dimension later requires re-ingesting into a table whose `vector(N)` matches.

---------------------------------------------------------------------

## Ingesting documents

Three ways: the CLI, the API (`POST /ingest`), or the Streamlit sidebar (“➕ Ingest sources”).
In **demo mode** the store is in-memory, so CLI-ingested data disappears when the command exits — use live mode to persist.

```bat
:: local folder (must be inside .\data unless ALLOW_ANY_LOCAL_PATH=true); .md .mdx .txt and issue .json
python -m app.ingestion.cli local --path data/sample_docs

:: Supabase docs from GitHub (markdown/MDX guides)
python -m app.ingestion.cli github-docs --repo supabase/supabase --branch master --path-prefix apps/docs/content/guides/auth --max-files 40

:: public GitHub issues (pull requests are skipped; up to 10 comments per issue are included)
python -m app.ingestion.cli github-issues --repo supabase/supabase --state all --max-issues 50

:: web pages (public http/https only; private/loopback addresses are refused)
python -m app.ingestion.cli url --url https://example.com/docs/page
```
Re-running is safe: unchanged content is skipped (`documents_unchanged`), changed content replaces old chunks, `--force` re-embeds everything. Set `GITHUB_TOKEN` to avoid the 60 requests/hour anonymous limit.

Via API: `POST /ingest` → `202 {"job_id": ...}`, then poll `GET /ingest/{job_id}`.

---------------------------------------------------------------------

## API reference

Interactive docs: `/docs`. If `API_KEY` is set, send `X-API-Key` on everything except `/health`.

| Method & path | Purpose |
|---|---|
| `GET /health` | Mode, store, providers, doc/chunk counts, configuration problems |
| `POST /chat` | `{question, history?, top_k?, use_hybrid?, use_rerank?, source_types?}` → answer, `citations[]`, `retrieved[]`, `abstained`, `abstain_reason`, `interaction_id` |
| `POST /search` | Retrieval only (no LLM) |
| `POST /ingest` / `GET /ingest/{job_id}` | Start / poll background ingestion |
| `POST /feedback` | `{interaction_id, rating: 1 \| -1, comment?}` |

Errors are JSON: `{"error": {"code", "message"}}` (400 ingestion, 404 not found, 422 validation, 502 provider, 503 database/config, 401 bad key). Unexpected errors return a generic 500 without internals.

---------------------------------------------------------------------

## Evaluation

```bat
scripts\run_eval.bat                          & demo mode, offline, deterministic
scripts\run_eval.bat --mode live              & real embeddings + LLM (needs keys; uses an in-memory store, no DB)
scripts\run_eval.bat --k 1 3 5 10 --json
```
* Dataset: `app/evaluation/data/eval_dataset.jsonl` — 32 questions over the **sample corpus** (26 answerable with gold `relevant_docs` + `expected_keywords`, 6 deliberately unanswerable). Hand-written by the project author; small; it validates the pipeline, it is **not** a benchmark of Supabase docs. Build your own dataset (same JSONL format) for your corpus.
* Metrics (all computed at run time, nothing hard-coded; reports saved to `eval_results/`): **Recall@K**, **Hit@K**, **MRR** for vector vs hybrid vs hybrid+rerank; abstention accuracy; keyword coverage; **citation precision**; **sentence-support ratio** and **fully-grounded rate** (each cited sentence must be lexically supported by the chunk it cites; `app/evaluation/grounding.py`).
* The lexical support check is a *proxy* for groundedness, not an entailment model. For live LLMs consider an LLM-as-judge or human review.
* Demo-mode numbers describe the toy hashing embedder + extractive answerer on a tiny fictional corpus; do not read them as real-world quality.

## Running tests

```bat
scripts\run_tests.bat                   & everything (integration tests auto-skip without a database)
python -m pytest tests\test_rag_service.py -v
```
Real-database integration tests (disposable DB — tables are TRUNCATEd):
```bat
set TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/dsi_test
python scripts\apply_migrations.py --database-url %TEST_DATABASE_URL%
python -m pytest -m integration
```
Test groups: `test_chunker_and_normalize`, `test_metrics`, `test_providers`, `test_repository_memory`, `test_ingestion`, `test_rag_service` (grounding, citation correctness, abstention), `test_api` (endpoint integration), `test_evaluation`, `test_frontend` (headless Streamlit), `test_config`, `test_supabase_integration`.

---------------------------------------------------------------------

## Library choices

| Choice | Why |
|---|---|
| **FastAPI + Pydantic v2 + pydantic-settings** | Typed request/response schemas, automatic OpenAPI docs, env-based config. |
| **psycopg 3 + psycopg_pool + pgvector-python** | Talks to Supabase as plain Postgres (works with direct, session or transaction pooler and local Postgres), lets us call SQL functions, and does the document+chunks upsert in one transaction. Avoids the REST client's JSON-vector overhead. |
| **httpx** for OpenAI/Anthropic/GitHub | One HTTP stack, trivially mockable in tests, no vendor SDK version churn. Any OpenAI-compatible endpoint works via `OPENAI_BASE_URL`. |
| **pgvector HNSW + tsvector/GIN** | Approximate vector search and keyword search in one database; RRF fusion in SQL. |
| **`text-embedding-3-small` (1536-d)** | Good cost/quality default; dimension is configurable. |
| **numpy** | In-memory store maths. |
| **Streamlit** | Fast chat UI with built-in headless testing. |
| Hashing embedder + extractive LLM | Offline demo/CI without keys — explicitly *not* semantic. |
| stdlib HTML parser | No BeautifulSoup dependency for the simple web connector. |

## Project structure

```
developer-support-intelligence/
├─ app/
│  ├─ main.py  config.py  schemas.py  container.py  exceptions.py  text_utils.py
│  ├─ api/            routes_{health,chat,search,ingest,feedback}.py  deps.py
│  ├─ services/       rag.py  retrieval.py  generation.py  reranker.py
│  ├─ ingestion/      connectors.py  normalize.py  chunker.py  pipeline.py  cli.py
│  ├─ providers/      base.py  embeddings.py  llm.py  factory.py
│  ├─ repositories/   base.py  memory.py  supabase.py  factory.py
│  └─ evaluation/     metrics.py  grounding.py  run_eval.py  data/eval_dataset.jsonl
├─ frontend/          app.py  api_client.py
├─ supabase/          migrations/001..004.sql  README.md
├─ data/              sample_docs/ (fictional Acme Tasks docs)  sample_issues/ (fictional issues)
├─ scripts/           setup/run_api/run_frontend/run_tests/run_eval/apply_migrations (.bat), apply_migrations.py
├─ tests/  .env.example  requirements.txt  requirements-dev.txt  pytest.ini
```

---------------------------------------------------------------------

## Known limitations

* **Sample data is fictional** (Acme Tasks). Real Supabase docs/issues are *not* bundled; ingest them yourself (needs internet, ideally `GITHUB_TOKEN`).
* Demo mode embeddings are lexical (no synonyms/paraphrase understanding); the demo answerer only quotes sentences.
* No live-mode quality numbers; `MIN_SIMILARITY` and the reranker weights are untuned defaults.
* Token counts are estimated (~4 chars/token), not tokenizer-exact.
* Groundedness check is lexical; it can miss subtle unsupported claims and flag correct paraphrases.
* Reranker is lexical, not a cross-encoder.
* Follow-up handling is a simple heuristic, not an LLM query-rewriter. No streaming responses.
* Ingestion jobs run as in-process background tasks: restarting the API mid-job leaves it `running`; there is no queue, scheduling or incremental sync/deletion of removed upstream documents.
* GitHub issues: only the first 20 comments are fetched and 10 used; issue content can be stale/wrong (the prompt tells the model to caveat open issues).
* Web connector handles static HTML only (no JavaScript rendering).
* Single embedding dimension per database; changing it needs a migration and re-ingest.
* Chat interactions are stored (question, answer, citations) for feedback — consider privacy/retention before real use.
* Authentication is a single optional shared `API_KEY`; no per-user auth or rate limiting.
* Windows scripts untested on Windows; Supabase-hosted and vendor-API paths untested (see status table).

## Security notes
* `.env` is git-ignored; no keys are included. Rotate any key you paste somewhere by mistake.
* Local-file ingestion via the API is confined to `./data` (path traversal blocked); URL ingestion refuses non-public addresses (SSRF guard; note DNS-rebinding is not fully mitigated).
* Retrieved text is untrusted: the system prompt tells the model to ignore instructions inside it, but prompt injection is never fully preventable.
* Set `API_KEY` before exposing the API beyond localhost; restrict CORS/reverse-proxy as needed (no CORS is enabled by default).

## Troubleshooting
* **Cannot reach the API** in the UI → start `scripts\run_api.bat`; check *Connection* in the sidebar.
* **`DATABASE_URL is required`** → you set `APP_MODE=live`; fill `.env` or use `APP_MODE=demo`.
* **Supabase connection timeout** → use the Session pooler URI (IPv4).
* **`Embedding has N dims but EMBEDDING_DIMENSIONS=…`** → dimension must equal the `vector(N)` column.
* **Always abstains in live mode** → lower `MIN_SIMILARITY` (try 0.15–0.25) and check that ingestion succeeded (`/health` chunk count).
* **GitHub rate limit** → set `GITHUB_TOKEN` in `.env`.
* PowerShell blocks activation → `Set-ExecutionPolicy -Scope Process Bypass`.
