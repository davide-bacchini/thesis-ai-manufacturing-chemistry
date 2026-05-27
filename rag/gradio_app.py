from __future__ import annotations

import os
import subprocess
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag_core import LocalRAG, get_settings


SYSTEM_PROMPT = """
You are a RAG chatbot for a bachelor's thesis about AI, data, software, GenAI, agentic systems, and digital capabilities in manufacturing, chemistry, rubber, tire, and automotive components companies.

Answer in the same language as the user's question.

Use only the retrieved context. Do not use outside knowledge.

The documents are final company analysis reports. Treat them as evidence-based summaries. When they refer to hiring analysis, remember that hiring signals are evidence of capability building, not proof of deployment, spending, headcount, internal ownership, or production maturity.

Your job is to answer the user's question directly and synthesize what matters. Do not list every initiative unless the user asks for all initiatives or detailed evidence.

Adapt the answer to the question:
- If the user asks a broad question, give a compact synthesis of the main capability areas.
- If the user asks about one capability, focus only on that capability.
- If the user asks for evidence, provide more granular evidence and citations.
- If the user asks for comparison, compare companies by capability area, evidence strength, maturity, and competitor implication.
- If the user asks a short factual question, answer briefly and cite the relevant source.
- If the user asks for a deep dive, provide a longer structured answer.

Do not force both a managerial overview and granular evidence in every answer. Use the level of detail that fits the question.

When synthesizing company activity:
- Group related examples under practical capability areas.
- Do not create long lists of disconnected projects.
- Do not turn report labels into capability names.
- Avoid generic labels such as “digital transformation,” “AI strategy,” or “innovation agenda” unless you explain the concrete business activity.
- Prefer concrete capability names when supported by the context, such as “AI for tire materials R&D,” “digital manufacturing quality control,” “fleet telematics and predictive services,” “GenAI for manufacturing operators,” “ADAS perception and planning,” or “predictive analytics for supply chain forecasting.”
- Explain what the company is doing, where it is applied, what problem it solves, and what the evidence does or does not prove.

Evidence discipline:
- Cite sources inline using this format: [Company Analysis.pdf, p. X].
- Place citations close to the claims they support.
- Do not cite only at the end.
- If evidence comes from hiring signals, say that the hiring evidence suggests capability building. Do not describe it as deployment.
- If evidence comes from partner or vendor evidence, say that partner evidence indicates use, deployment, or integration of a vendor-supported capability. Do not treat it as proof of internal ownership unless the context says so.
- If the retrieved context does not contain enough evidence, say so clearly.
- If no direct evidence is available for a claim, do not make the claim.

Maturity discipline:
- Separate capability building, vendor-supported deployment, internal development, production deployment, commercial deployment, and scaled deployment.
- Do not imply production deployment, global scale, achieved savings, ROI, or strategic priority unless the retrieved context supports it.
- Do not treat job postings as confirmed hires, budget, or production systems.
- Do not treat absence of hiring evidence as proof that the company has no activity in that area.

Style:
- Use simple, precise language.
- Be neutral and evidence-calibrated.
- Avoid buzzwords and promotional language.
- Avoid dramatic or inflated words such as “massive,” “revolutionary,” “world-class,” “game-changing,” “dominant,” “cutting-edge,” or “transformation” unless directly quoting or clearly attributing the source.
- Do not write like a technology catalog.
- Do not include a separate “sources retrieved” section unless the user asks.
- Do not repeat retrieval scores.
- Do not output tables unless the user explicitly asks for a table.

For broad company-level questions such as “What is Bridgestone doing?”:
- Start with the main answer in a few sentences.
- Then describe the main capability areas in short paragraphs.
- For each area, explain the business activity, evidence, maturity, and why it matters.
- End with what remains uncertain only if it is important for interpreting the evidence.
"""


def detect_project_id() -> str:
    project = os.getenv("PROJECT_ID")
    if project:
        return project

    result = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    project = result.stdout.strip()
    if project and project != "(unset)":
        return project

    raise RuntimeError("PROJECT_ID is not set. Copy .env.example to .env and set your own project id.")


def build_context(chunks: list[dict], max_chars: int = 24000) -> str:
    blocks = []
    used = 0

    for i, chunk in enumerate(chunks, start=1):
        block = (
            f"[{i}] Source: {chunk['source']}, page {chunk['page']}, "
            f"score {chunk['score']:.3f}\n"
            f"{chunk['text'].strip()}\n"
        )

        if used + len(block) > max_chars:
            break

        blocks.append(block)
        used += len(block)

    return "\n".join(blocks)


def extract_text(response) -> str:
    text = getattr(response, "text", None)
    if text and text.strip():
        return text.strip()

    try:
        candidates = getattr(response, "candidates", []) or []
        parts = candidates[0].content.parts
        recovered = "\n".join(
            getattr(part, "text", "")
            for part in parts
            if getattr(part, "text", "")
        )
        return recovered.strip()
    except Exception:
        return ""


def format_sources(chunks: list[dict]) -> str:
    seen = set()
    lines = []

    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key in seen:
            continue

        seen.add(key)
        lines.append(f"- {chunk['source']}, p. {chunk['page']}")

    return "\n".join(lines)


def answer_question(question: str, show_sources: bool = True) -> str:
    if not question.strip():
        return "Scrivi una domanda."

    retrieved = rag.retrieve(question, top_k=TOP_K)
    context = build_context(retrieved)

    prompt = f"""
Question:
{question}

Retrieved context:
{context}

Answer:
"""

    response = client.models.generate_content(
        model=VERTEX_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        ),
    )

    answer = extract_text(response)

    if not answer:
        answer = (
            "Gemini ha restituito una risposta vuota. "
            "Il retrieval locale ha funzionato, ma il modello non ha prodotto testo."
        )

    if show_sources:
        sources = format_sources(retrieved)
        answer = answer + "\n\n---\n\n### Retrieved sources\n" + sources

    return answer


def chat_fn(message, history):
    return answer_question(message, show_sources=True)


load_dotenv()

PROJECT_ID = detect_project_id()
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "global")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
TOP_K = int(os.getenv("TOP_K", "8"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))

settings = get_settings()
settings.index_dir = Path(os.getenv("INDEX_DIR", "rag/index"))

if not (settings.index_dir / "manifest.json").exists():
    raise FileNotFoundError(
        "Indice non trovato. Esegui prima: "
        "python rag/build_index.py --data-dir rag/data --index-dir rag/index"
    )

print("Loading local RAG index...")
rag = LocalRAG(settings)

print(f"Location: {VERTEX_LOCATION}")
print(f"Model: {VERTEX_MODEL}")
print(f"Chunks indexed: {rag.manifest.get('num_chunks')}")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=VERTEX_LOCATION,
)

examples = [
    "Which companies are investing in GenAI?",
    "What is Bridgestone doing in AI and digital capabilities?",
    "Compare Aptiv and Adient on GenAI and agentic systems.",
    "Which capabilities are already deployed and which are only capability building?",
    "What should a manager monitor in this sector?",
]

demo = gr.ChatInterface(
    fn=chat_fn,
    title="AI Chatbot",
    examples=examples,
)

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
