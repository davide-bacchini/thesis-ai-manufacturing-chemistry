from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


DEFAULT_AGENT = "deep-research-preview-04-2026"

DEEPSEARCH_PROMPT = r'''
Analyze the company using the uploaded annual report and the uploaded hiring-analysis PDF as the baseline evidence set.

The uploaded hiring-analysis PDF quantifies hiring signals in AI, data, software, cloud, automation, analytics, GenAI, agentic systems, connected products, digital platforms, and related technical capabilities. Treat the uploaded hiring report as baseline evidence of hiring signals, not as direct proof of spending, deployment scale, headcount, internal ownership, or production maturity.

Your role is to deepen the uploaded baseline with trustworthy public evidence and explain what the company is actually funding, building, deploying, scaling, buying, integrating, or exploring in AI, data, software, and digital capabilities.

The final deliverable must be exactly one static PDF report in English. The PDF must be fully self-contained and readable on its own.

The PDF must contain text only. Do not include tables, graphs, charts, screenshots, images, diagrams, visual exhibits, dashboards, spreadsheets, slide decks, websites, HTML pages, interactive elements, or appendices. Convert any relevant chart or table evidence from the uploaded PDFs into prose.

Do not add an executive introduction, abstract, methodology section, conclusion, appendix, or any section outside the four required sections and the required SOURCES section. Start directly with SECTION 1.

Keep the report focused. Prefer depth on the most evidence-supported capabilities over broad coverage of weak signals. Do not force a full subsection for a small hiring signal if public evidence is absent. In that case, summarize the limitation clearly.

CORE OBJECTIVE

Reconstruct the company's investments in AI, data, software, and digital capabilities as concretely as possible.

Focus on:
- what capabilities the company appears to be funding, building, deploying, scaling, buying, integrating, or exploring
- where these capabilities are applied in the business
- what operational, engineering, commercial, or strategic problem they solve
- how they work in practice, based on the evidence
- what the evidence says about investment intensity
- what this implies for a competitor

The report must read like a manager's briefing, not a technology catalog. Prefer concrete mechanisms, scope, operating use, business process, product application, outcomes, and evidence over labels.

EVIDENCE DISCIPLINE

No speculation.

Every important claim must be grounded in either uploaded documents or trustworthy public sources.

Separate uploaded-document evidence from public-source evidence in every subsection using these prose labels:
Uploaded-document evidence:
Public-source evidence:
Assessment:

Use these labels as short prose subheadings, not as a table.

Cite every non-trivial claim close to the sentence or paragraph it supports. Do not place citations only at the end.

Use numbered citations in the prose, for example [1], [2], and include a final SOURCES section after SECTION 4. The SOURCES section must list every cited source with its number, title, publisher when available, and URL when available. This source list is required so the public URL benchmark can map citation numbers to URLs.

Prioritize sources in this order:
1. Annual reports, 10-Ks, 20-Fs, SEC filings, regulated filings, and official financial statements
2. Official company investor presentations, earnings materials, press releases, product pages, sustainability reports, and technology pages
3. Official partner case studies from technology providers, cloud providers, software vendors, or industrial automation vendors
4. Reputable trade press and industry publications

If evidence comes from a vendor, partner, or trade publication, state that clearly in the prose. Do not treat vendor or partner evidence as fully equivalent to company-confirmed evidence.

Do not use generic technology sources as evidence that the company uses a technology. Generic sources may be used only as background, and only if needed to explain a term. If used, label them as background, not company evidence.

Do not treat a job posting as proof of production deployment. A job posting can support hiring intent, capability building, skills sought, business area, geography, timing, and possible application context. It cannot prove that the capability is deployed, scaled, internally owned, or strategically central.

Do not make claims about investment intensity from hiring data alone. Hiring data may indicate relative capability focus. Financial filings, capex, R&D, acquisitions, partnerships, product launches, deployed systems, and named operating programs are stronger evidence of investment intensity.

Distinguish between:
- using a technology
- buying or integrating a vendor platform
- developing a capability internally
- deploying a capability operationally
- scaling a capability across the organization
- achieving measurable business impact

These are different claims and require different levels of evidence.

If no strong evidence exists, write exactly:
"No direct public evidence found for X."

Do not treat absence of evidence in the uploaded hiring dataset as proof that the company has no activity in that area. Say "the uploaded hiring dataset does not show evidence of X" rather than "the company does not have X."

CAPABILITY NAMING RULE

Do not turn hiring categories into capability names.

A label such as "AI / ML modeling in customer offerings" is not a capability. Convert it into the concrete business activity supported by the evidence, such as "machine learning for ADAS perception and trajectory prediction," "predictive analytics for supply chain forecasting," "LLM applications for procurement support," or "AI for manufacturing quality inspection," only when the evidence supports that wording.

If the evidence does not support a concrete capability name, keep the wording conservative and say what is known.

MATURITY AND EVIDENCE STRENGTH

For each major capability, explain what the evidence supports in prose.

Use the following logic:
- If the source shows a commercial product, operating service, named customer deployment, or official company launch, the evidence may support production or commercial deployment.
- If the source shows use of a third-party platform or vendor solution, the evidence may support vendor-supported operational deployment, but not internal ownership.
- If company evidence shows rollout across plants, business units, regions, or functions, the evidence may support scaled internal deployment.
- If the source uses trial, pilot, proof-of-concept, demo, or early-stage language, classify the capability as pilot activity.
- If the evidence comes mainly from hiring signals, classify it as capability building.
- If evidence is limited or ambiguous, classify it as exploration or partial evidence.
- If no reliable company-specific evidence exists, say that no direct public evidence was found.

Do not assign maturity mechanically. Explain why the evidence supports the level you describe.

WRITING STYLE

Use simple, precise language.

The tone must be neutral, analytical, and evidence-calibrated.

Do not use promotional, dramatic, adversarial, or exaggerated language. Avoid rhetorical intensifiers and superlatives such as "massive," "revolutionary," "undeniable," "severe," or "world-class" unless the source itself uses the term and the report clearly attributes it.

Do not create certainty through language. If the evidence is direct, say what it shows. If the evidence is indirect, say what it suggests. If the evidence is only a hiring signal, say that it is consistent with capability building. If the evidence is from a vendor, say that partner evidence indicates use or deployment of a vendor-supported capability.

Do not write generic technology explanations unless they are needed to understand a specific company investment or application.

Do not write one-line capability summaries. Build each major capability as an evidence-based mini-case.

REQUIRED STRUCTURE

SECTION 1 - Quantitative investment baseline

Use the uploaded annual report and financial documents first.

Extract exact figures and year-over-year trends where available. Focus on:
- R&D and engineering expense
- capital expenditure
- capitalized software
- software and services revenue
- digital or cloud infrastructure
- data platforms
- connected services
- automation
- acquisitions, divestitures, impairments, restructuring, or integration costs relevant to AI, software, data, or digital capability
- disclosed technology partnerships or innovation infrastructure
- workforce or engineering headcount where relevant

For each figure:
- state the number
- state the period
- state the year-over-year change if available
- explain what the filing says it funds or reflects
- avoid inferring that a broad financial line item is AI-specific unless the document explicitly links it to AI, software, data, digital, automation, cloud, or related capability

Separate:
Uploaded-document evidence:
Public-source evidence:
Assessment:

If the company does not disclose AI-specific, cloud-specific, software-specific, or data-specific spend, say so clearly.

Do not imply that low disclosed software assets or low AI-specific disclosure means low real technology use. Explain only what the evidence supports.

SECTION 2 - Capability-by-capability analysis, including GenAI and agentic systems

Identify the major capability pockets from the uploaded hiring analysis and public evidence.

Focus on capability pockets that have either strong uploaded hiring evidence, strong public evidence, or both. If a capability appears only as a small hiring signal and no public evidence is found, summarize it briefly instead of giving it a full mini-case.

For each major capability pocket, explain:
- what the company is doing
- where in the business it is being applied
- what operational, engineering, commercial, or strategic problem it solves
- what technologies, platforms, systems, partners, data sources, or workflows are involved
- whether the evidence supports internal development, vendor-supported deployment, production deployment, commercial deployment, pilot activity, capability building, or exploration
- what public evidence says about scope, geography, teams, partners, users, and outcomes
- what the uploaded hiring evidence contributes
- what the uploaded hiring evidence does not prove

GenAI, RAG, LLMs, copilots, chatbots, autonomous agents, agentic systems, LangChain, LangGraph, MCP, prompt engineering, and related topics must be discussed inside the relevant capability subsection, not in a separate standalone section.

For GenAI or agentic capabilities, explain only what company-specific evidence supports:
- who the users are, if disclosed
- what workflow is supported
- what enterprise systems, data sources, tools, or platforms are connected, if disclosed
- whether evidence supports production deployment, vendor-supported operational deployment, pilot activity, capability building, exploration, or no direct public evidence
- what practical business effect is claimed or implied by the evidence

Do not use generic descriptions of RAG, MCP, Copilot Studio, LLMs, or agents as proof that the company uses them.

If the only evidence is job postings, describe it as hiring intent or capability building, not as deployment.

If the only evidence is a vendor or partner platform that includes GenAI features, state that the company has vendor-mediated exposure to GenAI. Do not state that the company has internal GenAI capability unless there is direct evidence.

If no direct company-specific public evidence exists for GenAI, RAG, or agentic systems beyond the uploaded hiring signals, write:
"No direct public evidence found for GenAI, RAG, or agentic systems beyond the uploaded hiring signals."

Separate within each capability:
Uploaded-document evidence:
Public-source evidence:
Assessment:

The assessment must be evidence-calibrated. Do not overstate maturity.

Do not call a capability internal unless there is evidence that the company owns, develops, or operates it internally. If the company is using a vendor platform, say that it is using a vendor-supported capability.

SECTION 3 - Talent, teams, and operating model

Use the uploaded hiring analysis and public evidence together to explain:
- where the company appears to be adding AI, data, software, cloud, automation, or digital capability
- which business areas the uploaded hiring evidence points to
- which geographies appear relevant in the uploaded hiring dataset
- whether roles are concentrated in IT, product engineering, manufacturing, R&D, supply chain, commercial functions, connected services, or other areas
- whether the company appears to rely on internal teams, external partners, acquisitions, vendors, or hybrid models
- what the seniority mix suggests, if available

Be careful:
- Do not treat posting volume as confirmed headcount.
- Do not treat job postings across different months, countries, or locations as duplicates unless the data proves duplication.
- Do not treat hiring concentration as spending concentration.
- Do not infer formal organizational structure unless public evidence supports it.
- Avoid absolute statements based only on the uploaded hiring dataset.
- Use "the uploaded hiring dataset does not show" rather than "the company does not have."

Separate:
Uploaded-document evidence:
Public-source evidence:
Assessment:

SECTION 4 - What the evidence implies for investment priorities and competitor monitoring

Synthesize the evidence into a small number of investment priorities.

For each priority:
- state the priority in concrete business terms
- identify the evidence base
- explain what the company seems to be trying to achieve
- explain what is known
- explain what remains uncertain
- explain the implication for a competitor

This section may include cautious, evidence-bound inference, but every inference must be clearly tied to evidence.

Competitor implications must be conditional and evidence-calibrated. Do not write as if a competitor outcome is guaranteed. Explain what could matter for competitors, under what conditions, and why the evidence supports that view.

Do not introduce new unsupported claims in this section.

Separate:
Uploaded-document evidence:
Public-source evidence:
Assessment:

FINAL QUALITY CHECK BEFORE WRITING

Before finalizing the PDF, check that:
- the report starts directly with SECTION 1
- the report contains the four required sections plus the final SOURCES section
- there is no executive introduction, abstract, methodology section, conclusion, appendix, or extra section
- every major claim has a citation close to the claim
- uploaded-document evidence and public-source evidence are separated in every subsection
- source type is clear when evidence comes from a vendor, partner, or trade publication
- no generic technical source is used as company-specific evidence
- no job posting is treated as proof of deployment
- no vendor case study is overstated as company-owned internal capability
- technology use is not confused with internal ownership
- no claim about production deployment, global scale, ROI, achieved savings, staff reduction, or strategic priority is made unless directly supported
- GenAI, RAG, and agentic claims are integrated into the relevant capability subsection
- the tone is precise, neutral, and evidence-calibrated
- the report avoids rhetorical inflation, promotional language, dramatic framing, and certainty that exceeds the evidence
- the report is text only
- there are no tables, graphs, charts, screenshots, images, diagrams, visual exhibits, appendices, dashboards, or interactive elements
- chart or table evidence from the uploaded files has been converted into prose
- no evidence is presented in tabular format
- the final output is exactly one static PDF report in English
'''.strip()


def get_api_key() -> str:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env")
    return key


def wait_for_operation(client: genai.Client, operation: Any, sleep_seconds: int = 5) -> Any:
    while not getattr(operation, "done", False):
        time.sleep(sleep_seconds)
        operation = client.operations.get(operation)
    if getattr(operation, "error", None):
        raise RuntimeError(f"Operation failed: {operation.error}")
    return operation


def upload_file_to_store(client: genai.Client, store_name: str, file_path: Path) -> None:
    print(f"Uploading and indexing: {file_path}")
    operation = client.file_search_stores.upload_to_file_search_store(
        file=str(file_path),
        file_search_store_name=store_name,
        config={"display_name": file_path.name},
    )
    wait_for_operation(client, operation)
    print(f"Indexed: {file_path.name}")


def get_attr_or_key(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value:
                return value
        if isinstance(obj, dict) and obj.get(name):
            return obj[name]
    return None


def extract_interaction_text(obj: Any) -> str:
    text = get_attr_or_key(obj, "output_text", "text")
    if text:
        return str(text).strip()

    outputs = get_attr_or_key(obj, "outputs") or []
    texts = []
    for output in outputs:
        output_text = get_attr_or_key(output, "text", "content")
        if output_text:
            texts.append(str(output_text))
    if texts:
        return "\n\n".join(texts).strip()

    candidates = get_attr_or_key(obj, "candidates") or []
    for candidate in candidates:
        content = get_attr_or_key(candidate, "content")
        parts = get_attr_or_key(content, "parts") if content else []
        for part in parts or []:
            part_text = get_attr_or_key(part, "text")
            if part_text:
                texts.append(str(part_text))
    return "\n\n".join(texts).strip()


def save_raw(obj: Any, path: Path) -> None:
    try:
        if hasattr(obj, "model_dump"):
            data = obj.model_dump()
        elif isinstance(obj, dict):
            data = obj
        else:
            data = repr(obj)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) if not isinstance(data, str) else data, encoding="utf-8")
    except Exception:
        path.write_text(repr(obj), encoding="utf-8")


def make_pdf(text: str, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        spaceAfter=7,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8,
    )
    subheading = ParagraphStyle(
        "Subheading",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        spaceBefore=8,
        spaceAfter=5,
    )

    story = []
    paragraph_buffer: list[str] = []

    def flush() -> None:
        nonlocal paragraph_buffer
        if not paragraph_buffer:
            return
        paragraph = " ".join(x.strip() for x in paragraph_buffer if x.strip())
        paragraph_buffer = []
        if not paragraph:
            return
        story.append(Paragraph(html.escape(paragraph), body))
        story.append(Spacer(1, 4))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue

        if line.startswith("SECTION ") or line == "SOURCES":
            flush()
            story.append(Paragraph(html.escape(line), heading))
            continue

        if line in {"Uploaded-document evidence:", "Public-source evidence:", "Assessment:"}:
            flush()
            story.append(Paragraph(html.escape(line), subheading))
            continue

        paragraph_buffer.append(line)

    flush()
    doc.build(story)


def sanitize_company_name(company: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", company).strip("_")
    return clean or "company"


def run_deepsearch(
    company: str,
    annual_report: Path,
    hiring_report: Path,
    output_dir: Path,
    agent: str,
    poll_seconds: int,
    save_raw_response: bool,
) -> None:
    client = genai.Client(api_key=get_api_key())

    if not annual_report.exists():
        raise FileNotFoundError(f"Annual report not found: {annual_report}")
    if not hiring_report.exists():
        raise FileNotFoundError(f"Hiring report not found: {hiring_report}")

    safe_company = sanitize_company_name(company)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating File Search Store...")
    store = client.file_search_stores.create(
        config={"display_name": f"{safe_company}_baseline_evidence"}
    )
    store_name = store.name
    print(f"File Search Store: {store_name}")

    upload_file_to_store(client, store_name, annual_report)
    upload_file_to_store(client, store_name, hiring_report)

    print(f"Starting Gemini Deep Research with agent: {agent}")
    interaction = client.interactions.create(
        agent=agent,
        input=DEEPSEARCH_PROMPT.replace("{company}", company),
        background=True,
        tools=[
            {"type": "google_search"},
            {"type": "file_search", "file_search_store_names": [store_name]},
        ],
    )

    interaction_id = get_attr_or_key(interaction, "id", "name", "interaction_id")
    if not interaction_id:
        raise RuntimeError(f"Could not find interaction id in: {interaction}")

    print(f"Interaction ID: {interaction_id}")

    while True:
        result = client.interactions.get(interaction_id)
        status = str(get_attr_or_key(result, "status", "state") or "").lower()
        print(f"Status: {status or 'unknown'}")

        if status in {"completed", "complete", "succeeded", "done"}:
            break
        if status in {"failed", "error", "cancelled", "canceled"}:
            raw_path = output_dir / f"{safe_company}_deepsearch_failed_raw.json"
            save_raw(result, raw_path)
            raise RuntimeError(f"Deep Research failed. Raw response saved to {raw_path}")

        time.sleep(poll_seconds)

    if save_raw_response:
        save_raw(result, output_dir / f"{safe_company}_deepsearch_raw.json")

    report_text = extract_interaction_text(result)
    if not report_text:
        raw_path = output_dir / f"{safe_company}_deepsearch_empty_raw.json"
        save_raw(result, raw_path)
        raise RuntimeError(f"Deep Research completed but no text was extracted. Raw response saved to {raw_path}")

    pdf_path = output_dir / f"{safe_company}_deepsearch_report.pdf"
    make_pdf(report_text, pdf_path)
    print(f"Saved PDF: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Google Gemini Deep Research for one company using two uploaded baseline PDFs.")
    parser.add_argument("--company", required=True)
    parser.add_argument("--annual-report", required=True)
    parser.add_argument("--hiring-report", required=True)
    parser.add_argument("--output-dir", default="deepsearch_outputs")
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--save-raw-response", action="store_true")
    args = parser.parse_args()

    run_deepsearch(
        company=args.company,
        annual_report=Path(args.annual_report),
        hiring_report=Path(args.hiring_report),
        output_dir=Path(args.output_dir),
        agent=args.agent,
        poll_seconds=args.poll_seconds,
        save_raw_response=args.save_raw_response,
    )


if __name__ == "__main__":
    main()
