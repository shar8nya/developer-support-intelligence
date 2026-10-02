"""Reproducible evaluation. Nothing here is hard-coded: every number is computed from this run.

    python -m app.evaluation.run_eval                    # demo mode (offline, deterministic)
    python -m app.evaluation.run_eval --mode live        # real embeddings + LLM (needs API keys)
    python -m app.evaluation.run_eval --dataset my.jsonl --corpus data/my_docs --k 1 3 5 10

The corpus is ingested into an in-memory store (no database needed) using the configured
embedding provider. Results are printed and saved to eval_results/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import PROJECT_ROOT, Settings
from app.container import Container, build_container
from app.evaluation.grounding import check_response
from app.evaluation.metrics import (
    citation_precision, dedupe_keep_order, hit_at_k, keyword_coverage, mean, recall_at_k, reciprocal_rank,
)
from app.repositories.memory import InMemoryRepository
from app.schemas import ChatRequest, IngestRequest

DEFAULT_DATASET = Path(__file__).parent / "data" / "eval_dataset.jsonl"
DEFAULT_CORPUS = [PROJECT_ROOT / "data" / "sample_docs", PROJECT_ROOT / "data" / "sample_issues"]

CONFIGS = {
    "vector": dict(use_hybrid=False, use_rerank=False),
    "hybrid": dict(use_hybrid=True, use_rerank=False),
    "hybrid+rerank": dict(use_hybrid=True, use_rerank=True),
}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        for key in ("id", "question", "answerable"):
            if key not in r:
                raise ValueError(f"dataset row missing '{key}': {r}")
        if r["answerable"] and not r.get("relevant_docs"):
            raise ValueError(f"answerable row {r['id']} needs relevant_docs")
    return rows


def build_eval_container(mode: str, corpus: list[Path]) -> Container:
    settings = Settings(app_mode=mode)  # explicit argument beats .env
    problems = settings.validate_live() if mode == "live" else []
    problems = [p for p in problems if "DATABASE_URL" not in p]   # eval uses an in-memory store
    if problems:
        raise SystemExit("Cannot run live evaluation:\n - " + "\n - ".join(problems))
    container = build_container(settings, auto_ingest_demo=False, repo=InMemoryRepository())
    for path in corpus:
        container.pipeline.run(IngestRequest(source="local", path=str(path)))
    return container


def eval_retrieval(container: Container, rows: list[dict], ks: list[int]) -> dict[str, dict[str, float]]:
    answerable = [r for r in rows if r["answerable"]]
    out: dict[str, dict[str, float]] = {}
    max_k = max(ks)
    for name, cfg in CONFIGS.items():
        per_q = []
        for r in answerable:
            hits = container.retrieval.search(r["question"], top_k=max_k, **cfg)
            docs = dedupe_keep_order(h.source_uri for h in hits)
            per_q.append((docs, r["relevant_docs"]))
        metrics: dict[str, float] = {"MRR": mean([reciprocal_rank(d, rel) for d, rel in per_q])}
        for k in ks:
            metrics[f"Recall@{k}"] = mean([recall_at_k(d, rel, k) for d, rel in per_q])
            metrics[f"Hit@{k}"] = mean([hit_at_k(d, rel, k) for d, rel in per_q])
        out[name] = {k: round(v, 4) for k, v in metrics.items()}
    return out


def eval_answers(container: Container, rows: list[dict]) -> dict[str, Any]:
    per_question, abst_correct = [], []
    kw, cite_prec, supported, all_grounded = [], [], [], []
    for r in rows:
        resp = container.rag.answer(ChatRequest(question=r["question"]))
        abst_correct.append(resp.abstained == (not r["answerable"]))
        rec: dict[str, Any] = {"id": r["id"], "answerable": r["answerable"], "abstained": resp.abstained,
                               "abstain_reason": resp.abstain_reason}
        if r["answerable"] and not resp.abstained:
            cited_docs = dedupe_keep_order(
                next(x.source_uri for x in resp.retrieved if x.chunk_id == c.chunk_id) for c in resp.citations)
            g = check_response(resp)
            rec.update(keyword_coverage=keyword_coverage(resp.answer, r.get("expected_keywords", [])),
                       citation_precision=citation_precision(cited_docs, r["relevant_docs"]),
                       supported_ratio=g.supported_ratio, grounded=g.ok, problems=g.problems)
            kw.append(rec["keyword_coverage"])
            cite_prec.append(rec["citation_precision"])
            supported.append(g.supported_ratio)
            all_grounded.append(1.0 if g.ok else 0.0)
        per_question.append(rec)
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    by_id = {p["id"]: p for p in per_question}
    return {
        "abstention_accuracy": round(mean([1.0 if c else 0.0 for c in abst_correct]), 4),
        "answer_rate_on_answerable": round(mean([0.0 if by_id[r["id"]]["abstained"] else 1.0 for r in answerable]), 4),
        "correct_abstention_on_unanswerable": round(mean([1.0 if by_id[r["id"]]["abstained"] else 0.0 for r in unanswerable]), 4) if unanswerable else None,
        "keyword_coverage": round(mean(kw), 4),
        "citation_precision": round(mean(cite_prec), 4),
        "sentence_support_ratio": round(mean(supported), 4),
        "fully_grounded_answers": round(mean(all_grounded), 4),
        "n_answered": len(kw),
        "per_question": per_question,
    }


def run(mode: str, dataset: Path, corpus: list[Path], ks: list[int], out_dir: Optional[Path]) -> dict[str, Any]:
    rows = load_dataset(dataset)
    container = build_eval_container(mode, corpus)
    started = time.perf_counter()
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "embedding_provider": container.embedder.name,
        "llm_provider": f"{container.llm.name}:{container.llm.model}",
        "dataset": str(dataset), "n_questions": len(rows),
        "n_answerable": sum(1 for r in rows if r["answerable"]),
        "corpus": [str(c) for c in corpus], "corpus_stats": container.repo.stats(),
        "min_similarity": container.settings.min_similarity,
        "retrieval": eval_retrieval(container, rows, ks),
        "answers": eval_answers(container, rows),
    }
    report["runtime_seconds"] = round(time.perf_counter() - started, 2)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"eval_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["saved_to"] = str(path)
    return report


def print_report(rep: dict[str, Any]) -> None:
    print(f"\nMode: {rep['mode']}  | embeddings: {rep['embedding_provider']}  | llm: {rep['llm_provider']}")
    print(f"Corpus: {rep['corpus_stats']}  | questions: {rep['n_questions']} ({rep['n_answerable']} answerable)\n")
    cols = list(next(iter(rep["retrieval"].values())).keys())
    print(f"{'config':<16}" + "".join(f"{c:>10}" for c in cols))
    for name, m in rep["retrieval"].items():
        print(f"{name:<16}" + "".join(f"{m[c]:>10.3f}" for c in cols))
    a = rep["answers"]
    print("\nAnswer-level metrics (default config):")
    for key in ("abstention_accuracy", "answer_rate_on_answerable", "correct_abstention_on_unanswerable",
                "keyword_coverage", "citation_precision", "sentence_support_ratio", "fully_grounded_answers"):
        print(f"  {key:<38}{a[key]}")
    bad = [p for p in a["per_question"] if p["answerable"] and p["abstained"]]
    wrong = [p for p in a["per_question"] if not p["answerable"] and not p["abstained"]]
    if bad:
        print("\n  Answerable but abstained:", ", ".join(p["id"] for p in bad))
    if wrong:
        print("  Unanswerable but answered:", ", ".join(p["id"] for p in wrong))
    if rep.get("saved_to"):
        print(f"\nSaved: {rep['saved_to']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["demo", "live"], default="demo")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--corpus", type=Path, nargs="+", default=DEFAULT_CORPUS)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full JSON report")
    a = ap.parse_args(argv)
    rep = run(a.mode, a.dataset, a.corpus, a.k, None if a.no_save else a.out_dir)
    print(json.dumps(rep, indent=2)) if a.json else print_report(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
