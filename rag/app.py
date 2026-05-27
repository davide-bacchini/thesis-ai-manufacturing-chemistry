from __future__ import annotations

from pathlib import Path

import streamlit as st

from local_rag import LocalRAG, get_settings
from questions import SUGGESTED_QUESTIONS


st.set_page_config(page_title="Local RAG Thesis Chatbot", page_icon="📚", layout="wide")

st.title("Local RAG chatbot for thesis analyses")
st.caption("Fully local retrieval over your PDF analyses. No Vertex AI, no GCS bucket, no PROJECT_ID.")

settings = get_settings()

with st.sidebar:
    st.header("Settings")
    settings.index_dir = Path(st.text_input("Index folder", value=str(settings.index_dir)))
    settings.top_k = st.slider("Retrieved chunks", min_value=3, max_value=15, value=settings.top_k)
    settings.use_ollama = st.toggle("Use local Ollama LLM", value=settings.use_ollama)
    settings.ollama_model = st.text_input("Ollama model", value=settings.ollama_model)
    st.write("Embedding model:")
    st.code(settings.embedding_model)
    st.divider()
    st.write("To rebuild the index:")
    st.code("python build_index.py --data-dir data --index-dir index")

if not (settings.index_dir / "manifest.json").exists():
    st.error("No local index found. Run this first in the terminal:")
    st.code("python build_index.py --data-dir data --index-dir index")
    st.stop()

@st.cache_resource(show_spinner="Loading local vector index...")
def load_rag(index_dir: str, use_ollama: bool, ollama_model: str) -> LocalRAG:
    cached_settings = get_settings()
    cached_settings.index_dir = Path(index_dir)
    cached_settings.use_ollama = use_ollama
    cached_settings.ollama_model = ollama_model
    return LocalRAG(cached_settings)

rag = load_rag(str(settings.index_dir), settings.use_ollama, settings.ollama_model)

with st.expander("Indexed documents", expanded=False):
    manifest = rag.manifest
    st.write(f"Documents: {manifest.get('num_documents')}")
    st.write(f"Chunks: {manifest.get('num_chunks')}")
    for doc in manifest.get("documents", []):
        st.write(f"- {doc}")

st.subheader("Suggested managerial questions")
cols = st.columns(2)
for i, question in enumerate(SUGGESTED_QUESTIONS):
    with cols[i % 2]:
        if st.button(question, key=f"q_{i}"):
            st.session_state["question"] = question

question = st.text_area(
    "Ask a question",
    value=st.session_state.get("question", "Which companies are investing the most in generative AI and agentic systems?"),
    height=100,
)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Write a question first.")
        st.stop()

    with st.spinner("Retrieving relevant chunks and generating an answer..."):
        result = rag.answer(question.strip(), top_k=settings.top_k)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Retrieved evidence")
    for i, source in enumerate(result["sources"], start=1):
        title = f"{i}. {source['source']} | page {source['page']} | score {source['score']:.3f}"
        with st.expander(title):
            st.write(source["text"])
