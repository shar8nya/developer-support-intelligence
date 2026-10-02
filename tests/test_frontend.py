"""Headless UI test using Streamlit's AppTest, wired to the real FastAPI app (demo mode) in-process."""
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app.main import create_app
from tests.conftest import PROJECT_ROOT

APP = str(PROJECT_ROOT / "frontend" / "app.py")


@pytest.fixture
def at(container):
    http = TestClient(create_app(container))
    http.__enter__()
    t = AppTest.from_file(APP, default_timeout=30)
    t.session_state["_http_client"] = http
    t.session_state["api_url"] = "http://testserver"
    yield t
    http.__exit__(None, None, None)


def test_chat_flow_shows_answer_and_citations(at):
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("How long do OAuth access tokens last before they expire?").run()
    assert not at.exception
    assert len(at.session_state["messages"]) == 2
    resp = at.session_state["messages"][1]["response"]
    assert not resp["abstained"] and "60 minutes" in resp["answer"]
    assert any("Authentication" in e.label for e in at.expander)


def test_abstention_is_shown_as_warning(at):
    at.run()
    at.chat_input[0].set_value("What is the capital of France?").run()
    assert not at.exception
    assert at.session_state["messages"][1]["response"]["abstained"]
    assert any("won't guess" in w.value for w in at.warning)


def test_feedback_button_records_rating(at):
    at.run()
    at.chat_input[0].set_value("How do I revoke a token?").run()
    up = next(b for b in at.button if b.key and b.key.startswith("fb_1_"))
    up.click().run()
    assert not at.exception and list(at.session_state["feedback_sent"].values()) == [1]


def test_search_tab(at):
    at.run()
    next(t for t in at.text_input if t.label == "Search query").set_value("webhook retries").run()
    next(b for b in at.button if b.label == "Search").click().run()
    assert not at.exception
    assert at.session_state["search_results"]["results"][0]["title"] == "Webhooks"


def test_unreachable_backend_shows_friendly_error():
    t = AppTest.from_file(APP, default_timeout=30)
    t.session_state["api_url"] = "http://127.0.0.1:9"    # nothing listens on the discard port
    t.run()
    assert not t.exception
    assert any("Cannot reach the API" in e.value for e in t.error)
