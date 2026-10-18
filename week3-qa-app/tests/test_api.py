"""tests/test_api.py: fast tests for the API. No API key, no network, no waiting.

    pytest -v
"""
import pytest
from fastapi.testclient import TestClient

from app import llm, retry
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Make the fake LLM and the retry backoff instant, so tests run in milliseconds."""
    async def instant(*args, **kwargs):
        return None
    monkeypatch.setattr(llm.asyncio, "sleep", instant)
    monkeypatch.setattr(retry.asyncio, "sleep", instant)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.parametrize("bad", [
    {"question": ""},                                   # empty
    {"question": "hi", "fail_rate": 2},                 # out of range
    {"question": "hi", "backend": "gpt-9"},             # unknown backend
    {"question": "hi", "temperature": 0.5},             # unknown field
])
def test_bad_requests_get_422(bad):
    assert client.post("/ask", json=bad).status_code == 422


def test_batched_answer_shape():
    r = client.post("/ask", json={"question": "What is RAG?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "backend", "retries", "latency_ms"}
    assert "retrieves" in body["answer"] and body["retries"] == 0


def test_retry_recovers_from_two_failures(monkeypatch):
    calls = {"n": 0}

    async def flaky(question, backend, fail_rate=0.0):          # mock: fails twice, then works
        calls["n"] += 1
        if calls["n"] <= 2:
            raise llm.TransientLLMError("boom")
        return "ok"

    monkeypatch.setattr(llm, "complete", flaky)
    r = client.post("/ask", json={"question": "x", "max_retries": 3})
    assert r.status_code == 200
    assert r.json()["retries"] == 2 and calls["n"] == 3


def test_retries_exhausted_gives_503():
    r = client.post("/ask", json={"question": "x", "fail_rate": 1.0, "max_retries": 2})
    assert r.status_code == 503
    assert "after 2 retries" in r.json()["detail"]


def test_stream_returns_plain_text_and_retry_header():
    r = client.post("/ask", json={"question": "What is streaming?", "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.headers["x-retries"] == "0"
    assert "pieces" in r.text


def test_stream_failure_before_first_word_is_a_proper_503():
    r = client.post("/ask", json={"question": "x", "stream": True, "fail_rate": 1.0, "max_retries": 1})
    assert r.status_code == 503


def test_real_backend_without_key_is_503(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/ask", json={"question": "x", "backend": "real"})
    assert r.status_code == 503
    assert "not configured" in r.json()["detail"]


def test_permanent_provider_error_is_502_and_not_retried(monkeypatch):
    calls = {"n": 0}

    async def wrong_key(question, backend, fail_rate=0.0):      # mock: like a 401 from the provider
        calls["n"] += 1
        raise llm.ProviderError("AuthenticationError: invalid API key")

    monkeypatch.setattr(llm, "complete", wrong_key)
    r = client.post("/ask", json={"question": "x", "max_retries": 3})
    assert r.status_code == 502 and calls["n"] == 1
