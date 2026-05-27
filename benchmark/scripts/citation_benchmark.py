from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import trafilatura
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


SUPPORT_SCORE = {
    "fully_supported": 1.0,
    "mostly_supported": 0.8,
    "partially_supported": 0.5,
    "weakly_supported": 0.25,
    "not_supported": 0.0,
    "wrong_source": 0.0,
    "source_inaccessible": 0.0,
    "missing_citation": 0.0,
}

IMPORTANCE_WEIGHT = {
    "high": 3,
    "medium": 2,
    "low": 1,
}


EXTRACT_CLAIMS_PROMPT = """
You are auditing an evidence-based company analysis report.

Task:
Extract the most important verifiable claims from the report.

Do not extract every sentence. Extract claims that would materially affect report quality if wrong.

Prioritize claims about:
- AI capability
- deployment maturity
- production deployment
- scale
- ROI, savings, or performance impact
- vendor/platform usage
- GenAI, RAG, LLMs, agents, MLOps
- job posting interpretation
- source type
- competitor implication
- financial figures

Unit of analysis:
A claim must be a single verifiable statement. Split multi-part claims.

Important:
If a claim relies on hiring evidence, mark claim_type accordingly.
If a material claim has no citation number, set uncited_material_claim=true and citation_numbers=[].
Citation numbers are numeric references used in the report, for example 1, 12, 33.

Return strict JSON only.

Schema:
{
  "claims": [
    {
      "claim_id": "C001",
      "claim_text": "...",
      "section": "...",
      "claim_type": "ai_capability | deployment_maturity | production_deployment | scale | roi_or_savings | vendor_platform_usage | genai_rag_agents | job_posting_interpretation | source_type | competitor_implication | financial_figure | other",
      "importance": "high | medium | low",
      "citation_numbers": [1, 12],
      "uncited_material_claim": false,
      "why_this_claim_matters": "..."
    }
  ]
}
"""


EXTRACT_REFERENCES_PROMPT = """
You are extracting the bibliography and citation map from a company analysis report.

Task:
Create a citation-number to source map.

Extract references that look like:
1. ...
[1] ...
Source 1 ...
or citations listed in notes or bibliography.

For each citation number, extract:
- citation_number
- title
- publisher
- url
- raw_entry

If no URL is present, leave url empty.
If title or publisher is unclear, infer from the raw entry only. Do not use outside knowledge.

Return strict JSON only.

Schema:
{
  "references": [
    {
      "citation_number": 1,
      "title": "...",
      "publisher": "...",
      "url": "https://...",
      "raw_entry": "..."
    }
  ]
}
"""


JUDGE_PROMPT = """
You are a citation-level auditor.

You must judge whether the cited source excerpts support the claim.

Use only the provided claim and source excerpts.
Do not use outside knowledge.

Support labels:
- fully_supported: the source supports all material parts directly.
- mostly_supported: the source supports the main point, but a minor detail is missing.
- partially_supported: the source supports only an important part of the claim.
- weakly_supported: the source is directionally consistent but vague or indirect.
- not_supported: the source does not support the claim.
- wrong_source: the source is about a different company, topic, or irrelevant.
- source_inaccessible: no readable source text is available.
- missing_citation: the report made a material claim but provided no citation.

Source quality labels:
- official_company_filing
- official_company_source
- executive_or_technical_leader_statement
- vendor_or_partner_case_study
- trade_press
- job_posting
- academic_or_patent
- generic_background
- low_quality_or_unclear
- inaccessible_or_irrelevant

Source quality score:
5 = official filing or official company source
4 = named executive, technical leader, or reputable partner evidence
3 = vendor case study or reputable trade press
2 = job posting or indirect source
1 = generic blog, unclear source, or SEO content
0 = inaccessible, irrelevant, or wrong source

Important discipline:
- A vendor case study can support "vendor evidence indicates deployment/use/integration".
- A vendor case study does not prove internal ownership unless the excerpt says so.
- A job posting suggests capability building, not deployment, spending, production maturity, or confirmed headcount.
- Do not imply production deployment, global scale, ROI, or achieved savings unless directly supported.
- If the claim is too strong, mark overclaiming in claim_calibration.

Return strict JSON only.

Schema:
{
  "support_label": "fully_supported | mostly_supported | partially_supported | weakly_supported | not_supported | wrong_source | source_inaccessible | missing_citation",
  "supported_parts": "...",
  "unsupported_or_overstated_parts": "...",
  "source_quote": "short quote from the source excerpt, max 30 words",
  "required_rewrite": "...",
  "confidence": 0.0,
  "source_quality_label": "...",
  "source_quality_score": 0,
  "claim_calibration": "well_calibrated | slight_overclaim | strong_overclaim | unsupported | unclear",
  "risk_flags": {
    "vendor_overclaim": false,
    "job_posting_overclaim": false,
    "ai_scope_drift": false,
    "uncited_material_claim": false
  }
}
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

    raise RuntimeError("PROJECT_ID not found. Set it in .env.")


def read_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})

    return pages


def pages_to_text(pages: list[dict[str, Any]], max_chars: int = 180000) -> str:
    chunks = []
    used = 0

    for page in pages:
        block = f"\n\n--- PAGE {page['page']} ---\n{page['text']}"
        if used + len(block) > max_chars:
            break
        chunks.append(block)
        used += len(block)

    return "".join(chunks)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        raise


def gemini_json(client, model: str, system_prompt: str, user_payload: str, max_tokens: int = 16384) -> dict[str, Any]:
    def call_model(prompt: str):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )
        except TypeError:
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    max_output_tokens=max_tokens,
                ),
            )

    response = call_model(user_payload)

    raw = response.text or ""

    if not raw:
        try:
            raw = "\n".join(
                part.text
                for part in response.candidates[0].content.parts
                if getattr(part, "text", None)
            )
        except Exception:
            raw = ""

    if not raw:
        raise RuntimeError("Gemini returned an empty response.")

    try:
        return extract_json(raw)
    except Exception as first_error:
        Path("last_bad_gemini_json.txt").write_text(raw)

        repair_prompt = f"""
The previous answer was intended to be JSON but is invalid.

Repair it into valid strict JSON only.
Do not add explanations.
Do not use markdown.

Invalid JSON:
{raw[:12000]}

JSON parsing error:
{first_error}
"""

        repaired = call_model(repair_prompt)
        repaired_raw = repaired.text or ""

        if not repaired_raw:
            try:
                repaired_raw = "\n".join(
                    part.text
                    for part in repaired.candidates[0].content.parts
                    if getattr(part, "text", None)
                )
            except Exception:
                repaired_raw = ""

        if not repaired_raw:
            raise RuntimeError("Gemini returned empty response during JSON repair.")

        try:
            return extract_json(repaired_raw)
        except Exception:
            Path("last_bad_gemini_json_repair.txt").write_text(repaired_raw)
            raise


def extract_claims(client, model: str, report_text: str, max_claims: int) -> list[dict[str, Any]]:
    payload = f"""
Report text:
{report_text}

Extract at most {max_claims} claims.
"""
    data = gemini_json(client, model, EXTRACT_CLAIMS_PROMPT, payload)
    claims = data.get("claims", [])

    clean = []
    for i, claim in enumerate(claims, start=1):
        claim.setdefault("claim_id", f"C{i:03d}")
        claim.setdefault("citation_numbers", [])
        claim.setdefault("importance", "medium")
        claim.setdefault("claim_type", "other")
        claim.setdefault("uncited_material_claim", False)
        clean.append(claim)

    return clean


def extract_references(client, model: str, report_text: str) -> dict[int, dict[str, Any]]:
    data = gemini_json(client, model, EXTRACT_REFERENCES_PROMPT, report_text)
    refs = {}

    for ref in data.get("references", []):
        try:
            n = int(ref.get("citation_number"))
        except Exception:
            continue

        refs[n] = {
            "citation_number": n,
            "title": ref.get("title", ""),
            "publisher": ref.get("publisher", ""),
            "url": ref.get("url", ""),
            "raw_entry": ref.get("raw_entry", ""),
        }

    return refs


def crawl_url(url: str, timeout: int = 25) -> dict[str, Any]:
    if not url:
        return {
            "status": "missing_url",
            "url": url,
            "title": "",
            "text": "",
            "error": "No URL in bibliography entry.",
            "crawl_method": "none",
        }

    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                include_links=False,
            )
            metadata = trafilatura.extract_metadata(downloaded)
            title = metadata.title if metadata and metadata.title else ""

            if text and len(text.strip()) > 200:
                return {
                    "status": "accessible",
                    "url": url,
                    "title": title,
                    "text": text.strip(),
                    "error": "",
                    "crawl_method": "trafilatura",
                }

        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 citation-benchmark"},
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        if "application/pdf" in content_type.lower() or url.lower().endswith(".pdf"):
            tmp = Path("_tmp_source.pdf")
            tmp.write_bytes(response.content)
            pages = read_pdf_pages(tmp)
            tmp.unlink(missing_ok=True)
            text = pages_to_text(pages, max_chars=120000)
            return {
                "status": "accessible",
                "url": url,
                "title": "",
                "text": text.strip(),
                "error": "",
                "crawl_method": "requests_pdf_pypdf",
            }

        html = response.text
        text = trafilatura.extract(html, include_comments=False, include_tables=True)

        if text and len(text.strip()) > 100:
            return {
                "status": "accessible",
                "url": url,
                "title": "",
                "text": text.strip(),
                "error": "",
                "crawl_method": "requests_trafilatura",
            }

        return {
            "status": "too_short",
            "url": url,
            "title": "",
            "text": text.strip() if text else "",
            "error": "Extracted text too short.",
            "crawl_method": "requests_trafilatura",
        }

    except Exception as exc:
        return {
            "status": "inaccessible",
            "url": url,
            "title": "",
            "text": "",
            "error": str(exc),
            "crawl_method": "failed",
        }


def split_text(text: str, words_per_chunk: int = 280, overlap: int = 60) -> list[str]:
    words = text.split()
    chunks = []
    step = max(1, words_per_chunk - overlap)

    for start in range(0, len(words), step):
        part = words[start:start + words_per_chunk]
        if len(part) < 30:
            continue
        chunks.append(" ".join(part))

    return chunks


def relevant_excerpt(claim_text: str, source_text: str, top_k: int = 3, max_chars: int = 5500) -> str:
    if not source_text:
        return ""

    chunks = split_text(source_text)
    if not chunks:
        return source_text[:max_chars]

    corpus = [claim_text] + chunks

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
        matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        ranked = sims.argsort()[::-1][:top_k]
        selected = [chunks[i] for i in ranked]
    except Exception:
        selected = chunks[:top_k]

    excerpt = "\n\n--- EXCERPT ---\n\n".join(selected)
    return excerpt[:max_chars]


def judge_claim(client, model: str, claim: dict[str, Any], source_bundle: list[dict[str, Any]]) -> dict[str, Any]:
    if claim.get("uncited_material_claim") or not claim.get("citation_numbers"):
        payload = f"""
Claim:
{json.dumps(claim, ensure_ascii=False)}

Source excerpts:
No citation was provided for this material claim.
"""
        result = gemini_json(client, model, JUDGE_PROMPT, payload)
        result["support_label"] = "missing_citation"
        return result

    payload = f"""
Claim:
{json.dumps(claim, ensure_ascii=False)}

Cited source excerpts:
{json.dumps(source_bundle, ensure_ascii=False)[:28000]}
"""
    return gemini_json(client, model, JUDGE_PROMPT, payload)


def summarize_results(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], report_name: str) -> dict[str, Any]:
    total_weight = 0
    weighted = 0

    for row in rows:
        support = row.get("support_label", "not_supported")
        importance = row.get("importance", "medium")
        w = IMPORTANCE_WEIGHT.get(importance, 2)
        total_weight += w
        weighted += SUPPORT_SCORE.get(support, 0) * w

    score = 100 * weighted / total_weight if total_weight else 0

    def rate(labels):
        if not rows:
            return 0
        return sum(1 for r in rows if r.get("support_label") in labels) / len(rows)

    unique_sources = {}
    for src in source_rows:
        key = src.get("url") or f"citation_{src.get('citation_number')}"
        unique_sources[key] = src

    accessible = [
        s for s in unique_sources.values()
        if s.get("status") == "accessible"
    ]

    source_accessibility_rate = len(accessible) / len(unique_sources) if unique_sources else 0

    vendor_labels = {"vendor_or_partner_case_study"}
    official_labels = {"official_company_filing", "official_company_source"}

    vendor_count = sum(1 for r in rows if r.get("source_quality_label") in vendor_labels)
    official_count = sum(1 for r in rows if r.get("source_quality_label") in official_labels)

    vendor_overclaim_count = sum(1 for r in rows if r.get("vendor_overclaim"))
    job_posting_overclaim_count = sum(1 for r in rows if r.get("job_posting_overclaim"))
    ai_scope_drift_count = sum(1 for r in rows if r.get("ai_scope_drift"))

    high_risk = [
        r for r in rows
        if r.get("importance") == "high"
        and r.get("support_label") not in {"fully_supported", "mostly_supported"}
    ]

    return {
        "report": report_name,
        "claim_count": len(rows),
        "weighted_support_score": round(score, 2),
        "unsupported_claim_rate": round(rate({"not_supported", "wrong_source", "missing_citation"}), 3),
        "weak_evidence_rate": round(rate({"weakly_supported", "partially_supported"}), 3),
        "source_accessibility_rate": round(source_accessibility_rate, 3),
        "official_source_share": round(official_count / len(rows), 3) if rows else 0,
        "vendor_evidence_share": round(vendor_count / len(rows), 3) if rows else 0,
        "vendor_overclaim_rate": round(vendor_overclaim_count / len(rows), 3) if rows else 0,
        "job_posting_overclaim_rate": round(job_posting_overclaim_count / len(rows), 3) if rows else 0,
        "ai_scope_drift_rate": round(ai_scope_drift_count / len(rows), 3) if rows else 0,
        "high_risk_claim_count": len(high_risk),
    }


def process_pdf(client, model: str, pdf_path: Path, out_dir: Path, max_claims: int, interactive: bool) -> dict[str, Any]:
    report_name = pdf_path.name
    report_out = out_dir / pdf_path.stem
    report_out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing {report_name} ===")

    pages = read_pdf_pages(pdf_path)
    report_text = pages_to_text(pages)

    print("Extracting claims...")
    claims = extract_claims(client, model, report_text, max_claims=max_claims)

    print("Extracting reference map...")
    refs = extract_references(client, model, report_text)

    with open(report_out / "claims.json", "w") as f:
        json.dump(claims, f, indent=2, ensure_ascii=False)

    with open(report_out / "references.json", "w") as f:
        json.dump(refs, f, indent=2, ensure_ascii=False)

    source_cache = {}
    source_rows = []
    result_rows = []

    cited_numbers = sorted({
        int(n)
        for claim in claims
        for n in claim.get("citation_numbers", [])
        if str(n).isdigit()
    })

    print(f"Crawling {len(cited_numbers)} cited sources...")
    for n in tqdm(cited_numbers):
        ref = refs.get(n, {"citation_number": n, "title": "", "publisher": "", "url": "", "raw_entry": ""})
        url = ref.get("url", "")

        if url in source_cache:
            crawled = source_cache[url]
        else:
            crawled = crawl_url(url)
            source_cache[url] = crawled
            time.sleep(0.5)

        source_row = {
            "citation_number": n,
            "title": ref.get("title", ""),
            "publisher": ref.get("publisher", ""),
            "url": url,
            "raw_entry": ref.get("raw_entry", ""),
            "status": crawled.get("status", ""),
            "crawl_method": crawled.get("crawl_method", ""),
            "error": crawled.get("error", ""),
            "page_title": crawled.get("title", ""),
            "text_length": len(crawled.get("text", "")),
        }
        source_rows.append(source_row)

        with open(report_out / f"source_{n}.json", "w") as f:
            json.dump({**source_row, "text": crawled.get("text", "")}, f, indent=2, ensure_ascii=False)

    print(f"Judging {len(claims)} claims...")
    for claim in tqdm(claims):
        bundle = []

        for n in claim.get("citation_numbers", []):
            try:
                n = int(n)
            except Exception:
                continue

            ref = refs.get(n, {"citation_number": n, "title": "", "publisher": "", "url": "", "raw_entry": ""})
            crawled = source_cache.get(ref.get("url", ""), {
                "status": "missing_url",
                "text": "",
                "error": "Source not crawled.",
            })

            excerpt = relevant_excerpt(claim.get("claim_text", ""), crawled.get("text", ""))

            bundle.append({
                "citation_number": n,
                "title": ref.get("title", ""),
                "publisher": ref.get("publisher", ""),
                "url": ref.get("url", ""),
                "status": crawled.get("status", ""),
                "error": crawled.get("error", ""),
                "source_excerpt": excerpt,
            })

        try:
            judgement = judge_claim(client, model, claim, bundle)
        except Exception as exc:
            judgement = {
                "support_label": "not_supported",
                "supported_parts": "",
                "unsupported_or_overstated_parts": f"Judge failed: {exc}",
                "source_quote": "",
                "required_rewrite": "",
                "confidence": 0,
                "source_quality_label": "low_quality_or_unclear",
                "source_quality_score": 0,
                "claim_calibration": "unclear",
                "risk_flags": {},
            }

        flags = judgement.get("risk_flags", {}) or {}

        row = {
            "report": report_name,
            "claim_id": claim.get("claim_id"),
            "claim_text": claim.get("claim_text"),
            "section": claim.get("section"),
            "claim_type": claim.get("claim_type"),
            "importance": claim.get("importance"),
            "citation_numbers": ", ".join(map(str, claim.get("citation_numbers", []))),
            "uncited_material_claim": claim.get("uncited_material_claim", False),
            "why_this_claim_matters": claim.get("why_this_claim_matters"),
            "support_label": judgement.get("support_label"),
            "supported_parts": judgement.get("supported_parts"),
            "unsupported_or_overstated_parts": judgement.get("unsupported_or_overstated_parts"),
            "source_quote": judgement.get("source_quote"),
            "required_rewrite": judgement.get("required_rewrite"),
            "confidence": judgement.get("confidence"),
            "source_quality_label": judgement.get("source_quality_label"),
            "source_quality_score": judgement.get("source_quality_score"),
            "claim_calibration": judgement.get("claim_calibration"),
            "vendor_overclaim": flags.get("vendor_overclaim", False),
            "job_posting_overclaim": flags.get("job_posting_overclaim", False),
            "ai_scope_drift": flags.get("ai_scope_drift", False),
            "source_bundle": json.dumps(bundle, ensure_ascii=False),
        }

        result_rows.append(row)

        if interactive:
            print("\n" + "=" * 100)
            print(f"CLAIM {row['claim_id']}: {row['claim_text']}")
            print(f"SUPPORT: {row['support_label']} | QUALITY: {row['source_quality_label']} ({row['source_quality_score']})")
            print(f"UNSUPPORTED/OVERSTATED: {row['unsupported_or_overstated_parts']}")
            print(f"REWRITE: {row['required_rewrite']}")
            cmd = input("Enter = next, q = quit interactive review: ").strip().lower()
            if cmd == "q":
                interactive = False

    summary = summarize_results(result_rows, source_rows, report_name)

    with open(report_out / "results.json", "w") as f:
        json.dump(result_rows, f, indent=2, ensure_ascii=False)

    with open(report_out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    xlsx_path = report_out / f"{pdf_path.stem}_citation_benchmark.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(result_rows).to_excel(writer, sheet_name="claims_judged", index=False)
        pd.DataFrame(source_rows).to_excel(writer, sheet_name="sources", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)

    print(f"Saved: {xlsx_path}")
    print(f"Weighted support score: {summary['weighted_support_score']}")

    return summary


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pdf", default="")
    parser.add_argument("--out-dir", default="benchmark_outputs")
    parser.add_argument("--model", default=os.getenv("VERTEX_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--location", default=os.getenv("VERTEX_LOCATION", "global"))
    parser.add_argument("--max-claims", type=int, default=20)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    project_id = detect_project_id()

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=args.location,
    )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        pdfs = [Path(args.pdf)]
    else:
        pdfs = sorted(data_dir.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    summaries = []

    for pdf in pdfs:
        summary = process_pdf(
            client=client,
            model=args.model,
            pdf_path=pdf,
            out_dir=out_dir,
            max_claims=args.max_claims,
            interactive=args.interactive,
        )
        summaries.append(summary)

    summary_path = out_dir / "citation_benchmark_summary.xlsx"
    pd.DataFrame(summaries).to_excel(summary_path, index=False)

    print(f"\nGlobal summary saved: {summary_path}")


if __name__ == "__main__":
    main()
