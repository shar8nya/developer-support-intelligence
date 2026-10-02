"""Streamlit UI for Developer Support Intelligence.

Run (with the API already running):  streamlit run frontend/app.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_client import ApiClient, ApiError  # noqa: E402

st.set_page_config(page_title="Developer Support Intelligence", page_icon="🛟", layout="wide")

SOURCE_LABELS = {"docs": "Documentation", "github_issue": "GitHub issues", "web": "Web pages", "local": "Local files"}
ICONS = {"docs": "📘", "github_issue": "🐛", "web": "🌐", "local": "📄"}

# ------------------------------------------------------------------ session state
st.session_state.setdefault("messages", [])          # [{role, content, response?}]
st.session_state.setdefault("feedback_sent", {})     # interaction_id -> rating
st.session_state.setdefault("search_results", None)


def get_api() -> ApiClient:
    return ApiClient(st.session_state.get("api_url", os.getenv("API_BASE_URL", "http://127.0.0.1:8000")),
                     st.session_state.get("api_key") or os.getenv("API_KEY"),
                     client=st.session_state.get("_http_client"))


# ------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛟 Dev Support Intelligence")
    st.caption("RAG assistant over documentation and GitHub issues")

    with st.expander("Connection", expanded=False):
        st.text_input("API base URL", value=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"), key="api_url")
        st.text_input("API key (if the backend requires one)", type="password", key="api_key")

    try:
        health = get_api().health()
        badge = "🟢" if health["status"] == "ok" else "🟠"
        st.markdown(f"{badge} **{health['mode'].upper()} mode** · {health['documents']} docs · {health['chunks']} chunks")
        st.caption(f"Store: `{health['database']}`  \nEmbeddings: `{health['embedding_provider']}`  \nLLM: `{health['llm_provider']}`")
        for p in health.get("problems", []):
            st.warning(p)
        if health["mode"] == "demo":
            st.info("Demo mode: offline embeddings and an extractive answerer over sample docs. "
                    "Not a real LLM - see the README to enable live mode.")
        if health["chunks"] == 0:
            st.warning("The index is empty. Ingest some sources below.")
    except ApiError as exc:
        st.error(exc.message)

    st.subheader("Retrieval settings")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 15, 5)
    use_hybrid = st.toggle("Hybrid search (keyword + vector)", value=True)
    use_rerank = st.toggle("Rerank results", value=True)
    source_filter = st.multiselect("Sources", list(SOURCE_LABELS), default=[],
                                   format_func=lambda s: SOURCE_LABELS[s], placeholder="All sources")
    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("➕ Ingest sources"):
        src = st.selectbox("Source type", ["local", "github_docs", "github_issues", "url"],
                           format_func=lambda s: {"local": "Local folder", "github_docs": "GitHub docs (markdown)",
                                                  "github_issues": "GitHub issues", "url": "Web page URL"}[s])
        payload: dict = {"source": src}
        if src == "local":
            payload["path"] = st.text_input("Folder (inside ./data)", "data/sample_docs")
        elif src in ("github_docs", "github_issues"):
            payload["repo"] = st.text_input("Repository (owner/name)", "supabase/supabase")
            if src == "github_docs":
                payload["branch"] = st.text_input("Branch", "master")
                payload["path_prefix"] = st.text_input("Path prefix", "apps/docs/content/guides/auth")
                payload["max_files"] = st.number_input("Max files", 1, 500, 30)
            else:
                payload["state"] = st.selectbox("State", ["all", "open", "closed"])
                payload["max_issues"] = st.number_input("Max issues", 1, 200, 30)
                payload["include_comments"] = st.checkbox("Include comments", True)
        else:
            payload["urls"] = [u.strip() for u in st.text_area("URLs (one per line)").splitlines() if u.strip()]
        payload["force"] = st.checkbox("Force re-embedding of unchanged content", False)
        if st.button("Start ingestion", use_container_width=True):
            try:
                job = get_api().ingest(payload)
                with st.spinner("Ingesting…"):
                    status = None
                    for _ in range(120):
                        status = get_api().ingest_status(job["job_id"])
                        if status["status"] in ("succeeded", "failed"):
                            break
                        time.sleep(1)
                if status and status["status"] == "succeeded":
                    st.success("Ingestion finished")
                else:
                    st.error(f"Ingestion {status['status'] if status else 'timed out'}: {(status or {}).get('error') or ''}")
                if status:
                    st.json(status["stats"])
            except ApiError as exc:
                st.error(exc.message)
def render_answer_with_clickable_citations(answer: str, citations: list[dict]) -> None:
    """Render answer text with [1], [2], etc. as clickable source links."""
    import re

    citation_urls = {
        str(c["index"]): c.get("url")
        for c in citations
        if c.get("url")
    }

    parts = re.split(r"(\[\d+\])", answer)

    rendered = ""

    for part in parts:
        match = re.fullmatch(r"\[(\d+)\]", part)

        if match:
            idx = match.group(1)
            url = citation_urls.get(idx)

            if url:
                rendered += f"[{idx}]({url})"
            else:
                rendered += part
        else:
            rendered += part

    st.markdown(rendered)

# ------------------------------------------------------------------------ helpers
def render_citations(resp: dict) -> None:
    if resp["abstained"]:
        return
    st.markdown("**Sources**")
    for c in resp["citations"]:
        title = f"[{c['index']}] {ICONS.get(c['source_type'], '📄')} {c['title']}"
        with st.expander(title):
            if c.get("heading_path"):
                st.caption(c["heading_path"])
            st.write(c["snippet"])
            if c.get("url"):
                st.markdown(f"[Open original source ↗]({c['url']})")
            st.caption(f"similarity {c['similarity']:.2f} · score {c['score']:.3f}")


def render_retrieved(resp: dict) -> None:
    if not resp["retrieved"]:
        return
    with st.expander(f"Retrieved passages ({len(resp['retrieved'])})"):
        for i, r in enumerate(resp["retrieved"], start=1):
            st.markdown(f"**{i}. {r['title']}** · `{r['source_type']}` · similarity {r['similarity']:.2f}")
            st.caption(r["heading_path"] or "")
            st.text(r["content"][:400] + ("…" if len(r["content"]) > 400 else ""))


def render_feedback(resp: dict, key: str) -> None:
    iid = resp["interaction_id"]
    sent = st.session_state.feedback_sent.get(iid)
    if sent:
        st.caption("Thanks for the feedback! " + ("👍" if sent == 1 else "👎"))
        return
    cols = st.columns([1, 1, 8])
    for col, (label, rating) in zip(cols[:2], (("👍", 1), ("👎", -1))):
        if col.button(label, key=f"fb_{rating}_{key}"):
            try:
                get_api().feedback(iid, rating)
                st.session_state.feedback_sent[iid] = rating
                st.rerun()
            except ApiError as exc:
                st.error(f"Could not save feedback: {exc.message}")


def render_assistant(resp: dict, key: str) -> None:
    if resp["abstained"]:
        st.warning(resp["answer"], icon="🤔")
    else:
        render_answer_with_clickable_citations(resp["answer"], resp["citations"])
    render_citations(resp)
    render_retrieved(resp)
    st.caption(f"{resp['latency_ms']} ms · {resp['mode']} mode")
    render_feedback(resp, key)


# ---------------------------------------------------------------------------- tabs
chat_tab, search_tab = st.tabs(["💬 Chat", "🔎 Search"])

with chat_tab:
    if not st.session_state.messages:
        st.markdown("#### Ask a question about the indexed documentation and issues")
        st.caption("Try: *How long do OAuth access tokens last?* · *Why does AsyncClient return 401 after refreshing?* · "
                   "*What is the capital of France?* (it should abstain)")

    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_assistant(m["response"], key=str(i))

    question = st.chat_input("Ask a technical question…")
    if question:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages][-6:]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the knowledge base and drafting an answer…"):
                    resp = get_api().chat({"question": question, "history": history, "top_k": top_k,
                                           "use_hybrid": use_hybrid, "use_rerank": use_rerank,
                                           "source_types": source_filter or None})
                st.session_state.messages.append({"role": "assistant", "content": resp["answer"], "response": resp})
                st.rerun()
            except ApiError as exc:
                st.session_state.messages.pop()  # drop the unanswered question
                st.error(f"Something went wrong: {exc.message}")

with search_tab:
    st.markdown("#### Semantic search (no LLM)")
    with st.form("search_form"):
        query = st.text_input("Search query", placeholder="e.g. webhook retry policy")
        submitted = st.form_submit_button("Search")
    if submitted:
        if len(query.strip()) < 2:
            st.warning("Please enter at least two characters.")
        else:
            try:
                with st.spinner("Searching…"):
                    st.session_state.search_results = get_api().search({
                        "query": query, "top_k": max(top_k, 5), "use_hybrid": use_hybrid,
                        "use_rerank": use_rerank, "source_types": source_filter or None})
            except ApiError as exc:
                st.session_state.search_results = None
                st.error(exc.message)
    res = st.session_state.search_results
    if res is not None:
        st.caption(f"{len(res['results'])} results in {res['latency_ms']} ms")
        if not res["results"]:
            st.info("No results. Is anything ingested?")
        for i, r in enumerate(res["results"], start=1):
            with st.expander(f"{i}. {ICONS.get(r['source_type'], '📄')} {r['title']} — similarity {r['similarity']:.2f}"):
                if r["heading_path"]:
                    st.caption(r["heading_path"])
                st.write(r["content"])
                if r.get("url"):
                    st.markdown(f"[Open original source ↗]({r['url']})")
