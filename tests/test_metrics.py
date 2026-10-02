import pytest

from app.evaluation import metrics as m


def test_recall_at_k():
    assert m.recall_at_k(["a", "b", "c"], ["b", "z"], 3) == 0.5
    assert m.recall_at_k(["a", "b", "c"], ["b"], 1) == 0.0
    assert m.recall_at_k(["a"], [], 3) == 0.0


def test_hit_at_k():
    assert m.hit_at_k(["a", "b"], ["b"], 2) == 1.0
    assert m.hit_at_k(["a", "b"], ["b"], 1) == 0.0


def test_reciprocal_rank():
    assert m.reciprocal_rank(["x", "y", "z"], ["z"]) == pytest.approx(1 / 3)
    assert m.reciprocal_rank(["x"], ["q"]) == 0.0
    assert m.reciprocal_rank(["q"], ["q", "x"]) == 1.0


def test_mean_of_empty_is_zero():
    assert m.mean([]) == 0.0


def test_dedupe_keep_order():
    assert m.dedupe_keep_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_keyword_and_citation_metrics():
    assert m.keyword_coverage("Tokens last 60 minutes", ["60 minutes", "hours"]) == 0.5
    assert m.keyword_coverage("x", []) == 1.0
    assert m.citation_precision(["a", "b"], ["a"]) == 0.5
    assert m.citation_precision([], ["a"]) == 1.0


def test_sentence_support():
    ev = "Access tokens expire after 60 minutes."
    assert m.sentence_support("Access tokens expire after 60 minutes", ev)
    assert not m.sentence_support("Bananas are yellow fruit grown in tropical regions", ev)
