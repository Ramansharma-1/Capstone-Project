import asyncio
from fastapi import FastAPI
from fake_llm import Question, Answer
from fake_llm import fake_ask_llm, FakeLLMError
from fastapi.responses import StreamingResponse


app = FastAPI(title = "Week 3 API demo")

@app.post("/ask_batched", response_model = Answer)
async def ask_question(q : Question):
    return await fake_ask_llm(q)


@app.get("/health")
async def get_health():
    return {"status" : "ok"}

async def _stream(q: Question):
    ans = await fake_ask_llm(q)
    for word in ans.text.split(" "):
        yield word + " "
        await asyncio.sleep(0.05)

@app.post("/ask")
async def ask(q: Question):
    return StreamingResponse(_stream(q), media_type="text/plain")


#Start server using "uvicorn in_class_demo:app --reload"


#On a different Terminal 
#Non streaming
#curl -N -X POST localhost:8000/ask_batched -H 'Content-Type: application/json' -d '{"text":"What is rag?"}'


#Streaming
#curl -N -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"text":"What is rag?"}'


#Homework : 
# 1. Generator understanding
# 2. Replace this with real LLM call (both one shot and streaming) 4 endpoints (2 for fake llm 2 for real llm)