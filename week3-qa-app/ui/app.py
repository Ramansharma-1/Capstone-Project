"""ui/app.py: Streamlit front end for the Week 3 Q&A API.

    streamlit run ui/app.py --server.port 8501       (with the API running on port 8000)
"""
import os
import time

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Week 3 Q&A", page_icon="💬", layout="centered")
st.title("Week 3 Q&A")
st.caption("FastAPI + Streamlit · fake or real LLM · streaming · retries")

# ── Sidebar: the options sent with every question ─────────────────────────────────────────────
try:
    cfg = requests.get(f"{API}/config", timeout=5).json()
except requests.exceptions.RequestException:
    st.error(f"Cannot reach the API at {API}. Start it with: uvicorn app.main:app --port 8000")
    st.stop()

with st.sidebar:
    st.header("Options")
    backend = st.radio("LLM backend", ["fake", "real"],
                       help="fake = no API key, simulated answers · real = OpenAI " + cfg["model"])
    if backend == "real" and not cfg["real_available"]:
        st.warning("OPENAI_API_KEY is not set on the API server. Add it to .env and restart the API.")
    stream = st.toggle("Stream the answer", value=True)
    fail_rate = st.slider("Fail rate (fake only)", 0.0, 1.0, 0.0, 0.1, disabled=(backend == "real"))
    max_retries = st.number_input("Max retries", min_value=0, max_value=5, value=2)

# ── Question form: only the Ask button sends a request ────────────────────────────────────────
with st.form("ask_form"):
    question = st.text_input("Your question", placeholder="e.g. What is RAG?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    payload = {"question": question, "backend": backend, "stream": stream,
               "fail_rate": fail_rate if backend == "fake" else 0.0, "max_retries": int(max_retries)}
    t0 = time.perf_counter()
    try:
        if not stream:
            with st.spinner("Waiting for the full answer..."):
                r = requests.post(f"{API}/ask", json=payload, timeout=60)
            if r.status_code != 200:
                st.error(f"{r.status_code}: {r.json().get('detail')}")
            else:
                data = r.json()
                st.markdown(data["answer"])
                st.caption(f"backend: {data['backend']} · retries: {data['retries']} · "
                           f"total: {time.perf_counter() - t0:.2f}s")
        else:
            placeholder, answer, first = st.empty(), "", None
            with requests.post(f"{API}/ask", json=payload, stream=True, timeout=(5, 60)) as r:
                if r.status_code != 200:
                    st.error(f"{r.status_code}: {r.json().get('detail')}")
                else:
                    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                        if first is None:
                            first = time.perf_counter() - t0
                        answer += chunk
                        placeholder.markdown(answer)
                    st.caption(f"backend: {r.headers.get('X-Backend')} · retries: {r.headers.get('X-Retries')} · "
                               f"first word: {first or 0:.2f}s · total: {time.perf_counter() - t0:.2f}s")
    except requests.exceptions.ConnectionError:
        st.error(f"Lost connection to the API at {API} (is uvicorn still running?)")
    except requests.exceptions.Timeout:
        st.error("The API took too long to respond.")
