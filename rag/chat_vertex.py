from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from local_rag import LocalRAG, get_settings


SYSTEM_PROMPT = """
You are a RAG chatbot for a bachelor's thesis about AI, data, software, GenAI, agentic systems, and digital investments in automotive suppliers.

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
        if recovered.strip():
            return recovered.strip()
    except Exception:
        pass

    return ""


def ask_gemini(client, model: str, question: str, retrieved_chunks: list[dict]) -> str:
    context = build_context(retrieved_chunks)

    prompt = f"""
Question:
{question}

Retrieved context:
{context}

Answer:
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")),
        ),
    )

    answer = extract_text(response)

    if answer:
        return answer

    print("\nDEBUG: Gemini raw response:")
    print(response)

    return (
        "Gemini ha restituito una risposta vuota. "
        "Il retrieval locale ha funzionato, ma il modello non ha prodotto testo."
    )


def print_sources(chunks: list[dict]) -> None:
    print("\nFonti recuperate:")
    for i, chunk in enumerate(chunks, start=1):
        print(
            f"{i}. {chunk['source']} p.{chunk['page']} "
            f"score={chunk['score']:.3f}"
        )


def main() -> None:
    load_dotenv()

    settings = get_settings()
    settings.index_dir = Path(os.getenv("INDEX_DIR", "rag/index"))

    if not (settings.index_dir / "manifest.json").exists():
        raise FileNotFoundError(
            "Indice non trovato. Esegui prima: "
            "python rag/build_index.py --data-dir rag/data --index-dir rag/index"
        )

    project_id = detect_project_id()
    location = os.getenv("VERTEX_LOCATION", "global")
    model = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
    top_k = int(os.getenv("TOP_K", "8"))

    print("\nCarico indice locale...")
    rag = LocalRAG(settings)

    print(f"Location: {location}")
    print(f"Model: {model}")
    print(f"Chunks indicizzati: {rag.manifest.get('num_chunks')}")
    print("\nChat pronta. Scrivi una domanda. Per uscire: exit\n")

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    while True:
        question = input("Tu > ").strip()

        if question.lower() in {"exit", "quit", "q", "fine"}:
            print("Chiuso.")
            break

        if not question:
            continue

        retrieved = rag.retrieve(question, top_k=top_k)
        answer = ask_gemini(client, model, question, retrieved)

        print("\nGemini >")
        print(answer)
        print_sources(retrieved)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
