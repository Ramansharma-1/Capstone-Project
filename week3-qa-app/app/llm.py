"""llm.py: two interchangeable "brains" behind the same two functions.

    complete(question, backend, fail_rate) -> str          # whole answer at once
    stream(question, backend, fail_rate)   -> async iterator of text pieces

backend="fake": no API key, random latency, fails on purpose at `fail_rate` (to exercise retries).
backend="real": OpenAI chat model (OPENAI_API_KEY from .env; OPENAI_BASE_URL is honoured if set).
"""
import asyncio
import os
import random
from functools import lru_cache

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = "You are a concise, helpful assistant. Answer in at most 3 sentences."


class TransientLLMError(Exception):
    """A temporary failure worth retrying (timeouts, rate limits, server errors, the fake's failures)."""


class BackendNotConfigured(Exception):
    """The real backend was requested but no API key is available."""


class ProviderError(Exception):
    """A permanent error from the provider (wrong key, bad model name...). Retrying will not help."""


# ── Fake backend ─────────────────────────────────────────────────────────────────────────────
CANNED = {
    "rag": "RAG retrieves relevant passages from your documents and gives them to the model, "
           "so the answer is grounded in your sources rather than only in its training data.",
    "stream": "Streaming sends the answer in small pieces as it is produced, "
              "so the first words appear quickly while the rest is still being generated.",
    "retry": "A retry repeats a failed call after a short, growing wait, "
             "which recovers from temporary problems like timeouts or rate limits.",
    "fastapi": "FastAPI is a Python web framework that turns typed functions into HTTP endpoints, "
               "with validation and documentation generated automatically.",
}


def _fake_answer(question: str) -> str:
    q = question.lower()
    for key, answer in CANNED.items():
        if key in q:
            return answer
    return f"(fake answer) This is a simulated response to: {question}"


async def _fake_think(fail_rate: float) -> None:
    """Simulate the model's thinking time, then fail at random (only BEFORE any text is produced)."""
    await asyncio.sleep(random.uniform(0.3, 1.0))
    if random.random() < fail_rate:
        raise TransientLLMError("fake LLM: simulated temporary failure")


async def _fake_complete(question: str, fail_rate: float) -> str:
    await _fake_think(fail_rate)
    return _fake_answer(question)


async def _fake_stream(question: str, fail_rate: float):
    await _fake_think(fail_rate)
    for word in _fake_answer(question).split(" "):
        yield word + " "
        await asyncio.sleep(0.04)                    # the fake "types" at roughly 25 words per second


# ── Real backend (OpenAI) ────────────────────────────────────────────────────────────────────
def real_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def _client():
    from openai import AsyncOpenAI
    # max_retries=0: OUR retry loop decides; otherwise SDK retries x our retries multiply (Week 2).
    return AsyncOpenAI(timeout=30, max_retries=0)


def _messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]


def _translate(e: Exception) -> Exception:
    """Map OpenAI errors to ours: temporary ones are retried, permanent ones are reported at once."""
    import openai
    transient = (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError,
                 openai.InternalServerError)
    if isinstance(e, transient):
        return TransientLLMError(type(e).__name__)
    if isinstance(e, openai.APIError):                  # e.g. AuthenticationError (401), NotFoundError (404)
        return ProviderError(f"{type(e).__name__}: {getattr(e, 'message', e)}")
    return e


async def _real_complete(question: str) -> str:
    if not real_available():
        raise BackendNotConfigured("OPENAI_API_KEY is not set")
    try:
        resp = await _client().chat.completions.create(model=MODEL, messages=_messages(question),
                                                       max_tokens=300, temperature=0.3)
    except Exception as e:
        raise _translate(e) from e
    return resp.choices[0].message.content or ""


async def _real_stream(question: str):
    if not real_available():
        raise BackendNotConfigured("OPENAI_API_KEY is not set")
    try:
        stream = await _client().chat.completions.create(model=MODEL, messages=_messages(question),
                                                         max_tokens=300, temperature=0.3, stream=True)
        async for chunk in stream:                   # real tokens, as the model produces them
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        raise _translate(e) from e


# ── The two functions the API uses ───────────────────────────────────────────────────────────
async def complete(question: str, backend: str, fail_rate: float = 0.0) -> str:
    if backend == "real":
        return await _real_complete(question)
    return await _fake_complete(question, fail_rate)


def stream(question: str, backend: str, fail_rate: float = 0.0):
    if backend == "real":
        return _real_stream(question)
    return _fake_stream(question, fail_rate)
