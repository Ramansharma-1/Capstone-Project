"""retry.py: retry temporary failures with exponential backoff and full jitter (the Week 2 pattern).

Two helpers:
  call_with_retry(make_call, max_retries)   -> (result, retries_used)       for a normal call
  start_stream_with_retry(make_stream, ...) -> (first_piece, rest, retries) for a stream:
      a stream can only be retried BEFORE its first piece reaches the user, so we wait for
      the first piece here (with retries), and only then start sending.
"""
import asyncio
import logging
import random

from app.llm import TransientLLMError

log = logging.getLogger("qa.retry")
BASE_DELAY = 0.5        # seconds; doubles each attempt
MAX_DELAY = 4.0


def _backoff(attempt: int) -> float:
    return random.uniform(0, min(MAX_DELAY, BASE_DELAY * 2 ** attempt))   # full jitter


async def call_with_retry(make_call, max_retries: int):
    for attempt in range(max_retries + 1):
        try:
            return await make_call(), attempt
        except TransientLLMError as e:
            if attempt == max_retries:
                raise
            wait = _backoff(attempt)
            log.warning("retry attempt=%d error=%s wait_s=%.2f", attempt + 1, e, wait)
            await asyncio.sleep(wait)


async def start_stream_with_retry(make_stream, max_retries: int):
    for attempt in range(max_retries + 1):
        gen = make_stream()
        try:
            first = await anext(gen)                 # nothing has been sent to the user yet
            return first, gen, attempt
        except StopAsyncIteration:                   # an empty answer: still a valid stream
            return "", gen, attempt
        except TransientLLMError as e:
            await gen.aclose()
            if attempt == max_retries:
                raise
            wait = _backoff(attempt)
            log.warning("stream retry attempt=%d error=%s wait_s=%.2f", attempt + 1, e, wait)
            await asyncio.sleep(wait)
