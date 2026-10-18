# Week 3 Q&A App

A small end-to-end AI app:
- a **Streamlit UI**;
- a **FastAPI** service with one `POST /ask` endpoint;
- a choice of **fake LLM** (no key) or **real LLM** (OpenAI);
- **streaming** or **non-streaming** answers;
- an adjustable **fail rate** (fake LLM) and **retries with backoff**;
- **tests** with the LLM mocked.

```
week3-qa-app/
├── app/
│   ├── llm.py      fake + real backends behind complete() and stream()
│   ├── retry.py    retry with exponential backoff (temporary errors only)
│   └── main.py     FastAPI: GET /health, GET /config, POST /ask
├── ui/app.py       Streamlit front end
├── tests/test_api.py
├── requirements.txt
├── .env.example    copy to .env and add your key (.env is never committed)
└── .gitignore
```

## 1. Install
```bash
pip install --break-system-packages -r requirements.txt     # on Vocareum
# or: pip install -r requirements.txt                       # in your own virtual environment
```

## 2. (Optional) add your OpenAI key for the real LLM
```bash
cp .env.example .env
# edit .env:  OPENAI_API_KEY=your-key-here
# if your provider gives you a different endpoint, also set OPENAI_BASE_URL in .env
```
The fake LLM works without any key.

## 3. Run the API (terminal 1)
```bash
python -m uvicorn app.main:app --reload --port 8000
```
Check it: `curl http://localhost:8000/health` → `{"status":"ok"}`. API docs: http://localhost:8000/docs

## 4. Run the UI (terminal 2)
```bash
streamlit run ui/app.py --server.port 8501
```
Open http://localhost:8501. In the sidebar, choose the backend, streaming, fail rate and max retries.

## 5. Run the tests
```bash
python -m pytest -v
```

## Try it with curl
```bash
# fake, whole answer
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What is RAG?"}'
# fake, streamed
curl -N -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What is streaming?", "stream": true}'
# fake, failing half the time, up to 3 retries
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What is a retry?", "fail_rate": 0.5, "max_retries": 3}'
# real model, streamed (needs OPENAI_API_KEY)
curl -N -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What is RAG?", "backend": "real", "stream": true}'
```

## The /ask request
| Field | Type | Default | Meaning |
|---|---|---|---|
| `question` | text, 1–2000 characters | required | The question |
| `backend` | `"fake"` or `"real"` | `"fake"` | Which LLM answers |
| `stream` | true / false | false | Stream as plain text, or return JSON |
| `fail_rate` | 0.0–1.0 | 0.0 | Chance that each fake call fails (fake only) |
| `max_retries` | 0–5 | 2 | Retries for temporary failures |

**Responses:**
- **200** — the answer: JSON `{answer, backend, retries, latency_ms}`, or streamed text with headers `X-Retries` and `X-Backend`.
- **422** — invalid request.
- **502** — the provider rejected the request (e.g. a wrong key). Not retried.
- **503** — still failing after all retries, or the real backend isn't configured.

**How retries work with streaming:** a stream can only be retried **before** its first word is sent. After that the 200 status is already on its way. So the API waits for the first piece (retrying if needed) before it starts streaming. If something fails mid-stream, the text ends with `[stream interrupted — please retry]`.
