"""fake_llm.py — a no-API async stand-in for a real LLM call.

Why this exists: some demos need *reliable, on-demand failures* (to show retry)
and want zero API quota. This module has the same Question -> Answer shape as a
real call and one async function, ``fake_ask_llm``, which:
  * sleeps a random latency (so async parallelism is visible),
  * returns a Pydantic ``Answer`` with a canned response,
  * raises ``FakeLLMError`` at a configurable ``fail_rate`` (to teach retry).

Swapping to the real ``AsyncOpenAI`` client is a one-line change — the interface
(Question -> Answer) is identical.
"""
from __future__ import annotations

import asyncio
import random

from pydantic import BaseModel


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    question: str
    text: str
    cost_usd: float
    retries: int = 0


class FakeLLMError(Exception):
    """Raised by fake_ask_llm when configured to fail (simulated transient error)."""


_CANNED = {
    "rag": "RAG grounds an LLM's answer in retrieved documents rather than its training data.",
    "vector": "Vector databases store embeddings and power semantic search, RAG retrieval, and recommendations.",
    "hallucin": "LLMs hallucinate when they produce confident text not grounded in evidence.",
    "pydantic": "Pydantic gives you typed classes that validate data at runtime — ideal for checking LLM output shapes.",
    "async": "Async lets one program progress many I/O-bound tasks at once by yielding to the event loop while awaiting.",
    "embedding": "An embedding is a fixed-length vector whose distance correlates with semantic similarity.",
    "agent": "An agent is a loop that uses an LLM to decide what to do next.",
    "backoff": "Exponential backoff doubles the wait between retries so a flaky API can recover.",
}


def _pick(text: str) -> str:
    lower = text.lower()
    for key, answer in _CANNED.items():
        if key in lower:
            return answer
    return f"(simulated answer) A grounded, concise response to: {text}"


async def fake_ask_llm(
    q: Question,
    *,
    min_latency: float = 0.3,
    max_latency: float = 1.2,
    fail_rate: float = 0.0,
) -> Answer:
    """Drop-in async stand-in for the real LLM call — no network, no API key."""
    await asyncio.sleep(random.uniform(min_latency, max_latency))
    if fail_rate > 0.0 and random.random() < fail_rate:
        raise FakeLLMError(f"simulated transient failure: {q.text[:50]}")
    return Answer(question=q.text, text=_pick(q.text), cost_usd=0.0001)
