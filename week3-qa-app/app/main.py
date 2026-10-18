"""main.py: the HTTP door. One endpoint, POST /ask, with options for backend, streaming, fail rate, retries.

    uvicorn app.main:app --reload --port 8000        (run from the project folder)
"""
import logging
import time
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

load_dotenv(Path(__file__).resolve().parent.parent / ".env")   # project-root .env (never committed)

from app import llm                                  # noqa: E402  (after load_dotenv, so env is ready)
from app.retry import call_with_retry, start_stream_with_retry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("qa.api")

app = FastAPI(title="Week 3 Q&A API", version="1.0.0")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    backend: Literal["fake", "real"] = "fake"
    stream: bool = False
    fail_rate: float = Field(default=0.0, ge=0.0, le=1.0)      # fake backend only
    max_retries: int = Field(default=2, ge=0, le=5)


class AskResponse(BaseModel):
    answer: str
    backend: str
    retries: int
    latency_ms: int


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config")
async def config():
    """Lets the UI know whether the real backend can be used."""
    return {"real_available": llm.real_available(), "model": llm.MODEL}


@app.post("/ask", response_model=AskResponse,
          responses={200: {"content": {"text/plain": {}}, "description": "JSON, or plain text when stream=true"}})
async def ask(req: AskRequest):
    t0 = time.perf_counter()
    log.info("ask backend=%s stream=%s fail_rate=%.2f max_retries=%d chars=%d",
             req.backend, req.stream, req.fail_rate, req.max_retries, len(req.question))   # no question text

    try:
        if not req.stream:
            answer, retries = await call_with_retry(
                lambda: llm.complete(req.question, req.backend, req.fail_rate), req.max_retries)
            latency = int((time.perf_counter() - t0) * 1000)
            log.info("answered retries=%d latency_ms=%d", retries, latency)
            return AskResponse(answer=answer, backend=req.backend, retries=retries, latency_ms=latency)

        # Streaming: get the FIRST piece (with retries) before sending anything, so failures can still be a 503.
        first, rest, retries = await start_stream_with_retry(
            lambda: llm.stream(req.question, req.backend, req.fail_rate), req.max_retries)
    except llm.BackendNotConfigured as e:
        raise HTTPException(status_code=503, detail=f"Real backend not configured: {e}")
    except llm.TransientLLMError as e:
        raise HTTPException(status_code=503, detail=f"LLM failed after {req.max_retries} retries: {e}")
    except llm.ProviderError as e:
        raise HTTPException(status_code=502, detail=f"LLM provider rejected the request (not retried): {e}")

    async def body():
        yield first
        try:
            async for piece in rest:
                yield piece
        except Exception as e:                       # the 200 is already sent: say so inside the stream
            log.error("stream interrupted error=%s", type(e).__name__)
            yield "\n\n[stream interrupted — please retry]"

    log.info("streaming retries=%d first_piece_ms=%d", retries, int((time.perf_counter() - t0) * 1000))
    return StreamingResponse(body(), media_type="text/plain",
                             headers={"X-Retries": str(retries), "X-Backend": req.backend})
