"""The evaluation script must be reproducible and must not fabricate numbers."""
import json

import pytest

from app.evaluation import run_eval
from app.evaluation.run_eval import DEFAULT_DATASET, load_dataset


def test_dataset_is_well_formed_and_references_real_documents(container):
    rows = load_dataset(DEFAULT_DATASET)
    assert len(rows) >= 30 and len({r["id"] for r in rows}) == len(rows)
    assert any(not r["answerable"] for r in rows) and any(r["answerable"] for r in rows)
    for r in rows:
        for rel in r["relevant_docs"]:
            assert container.repo.get_document_by_uri(rel) is not None, f"{r['id']} references unknown doc {rel}"


def test_eval_is_deterministic_and_computed(tmp_path):
    a = run_eval.run("demo", DEFAULT_DATASET, run_eval.DEFAULT_CORPUS, [1, 3, 5], tmp_path)
    b = run_eval.run("demo", DEFAULT_DATASET, run_eval.DEFAULT_CORPUS, [1, 3, 5], None)
    assert a["retrieval"] == b["retrieval"]
    assert {"MRR", "Recall@1", "Recall@3", "Recall@5"} <= set(a["retrieval"]["hybrid"])
    saved = json.loads(open(a["saved_to"], encoding="utf-8").read())
    assert saved["n_questions"] == len(load_dataset(DEFAULT_DATASET))


def test_eval_quality_floors_for_demo_mode(tmp_path):
    """Regression floors (deliberately below current values) - catches broken retrieval/abstention."""
    rep = run_eval.run("demo", DEFAULT_DATASET, run_eval.DEFAULT_CORPUS, [1, 3, 5], None)
    h = rep["retrieval"]["hybrid"]
    assert h["Recall@5"] >= 0.85 and h["MRR"] >= 0.75
    a = rep["answers"]
    assert a["correct_abstention_on_unanswerable"] >= 0.8
    assert a["citation_precision"] >= 0.7
    assert a["fully_grounded_answers"] >= 0.9      # every answered response passes the citation/grounding checks


def test_eval_metrics_on_a_tiny_hand_built_case(container):
    rows = [{"id": "x", "question": "How do I revoke a token?", "answerable": True,
             "relevant_docs": ["local:authentication.md"], "expected_keywords": []}]
    m = run_eval.eval_retrieval(container, rows, [1, 3])
    assert m["hybrid"]["Recall@3"] == 1.0 and m["hybrid"]["MRR"] > 0.3


def test_dataset_validation(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "1", "question": "q", "answerable": true}\n')
    with pytest.raises(ValueError, match="relevant_docs"):
        load_dataset(p)
