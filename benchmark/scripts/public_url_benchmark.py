from __future__ import annotations

import argparse
import asyncio
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

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
except Exception:  # Crawl4AI is optional at import time but listed in requirements.
    AsyncWebCrawler = None
    BrowserConfig = None
    CacheMode = None
    CrawlerRunConfig = None
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


EXTRACT_CLAIMS_PROMPT = """
You are auditing an evidence-based company analysis report.

Task:
Extract every factual, checkable claim in the report that cites at least one public URL citation number.

Do not apply a materiality filter.
Do not extract only the most important claims.
Do not skip minor factual claims if they are cited with a public citation.
Do not extract generic commentary unless it makes a factual claim.
Do not extract claims that have no citation.
Do not extract claims that cite only non-public or local sources.

You will receive a list of citation numbers that have public URLs.
Only extract claims that cite at least one of those citation numbers.

A claim must be a single verifiable statement.
If one sentence contains multiple factual claims, split it into separate claims.

For each claim, preserve the citation numbers used by the report.

Return strict JSON only.

Schema:
{
  "claims": [
    {
      "claim_id": "C001",
      "claim_text": "...",
      "section": "...",
      "claim_type": "ai_capability | deployment_maturity | production_deployment | scale | roi_or_savings | vendor_platform_usage | genai_rag_agents | job_posting_interpretation | competitor_implication | financial_figure | other",
      "importance": "high | medium | low",
      "citation_numbers": [5, 12],
      "why_this_claim_matters": "..."
    }
  ]
}
"""


EXTRACT_REFERENCES_PROMPT = """
You are extracting the bibliography and citation map from a company analysis report.

Create a citation-number to source map.

For each citation number, extract:
- citation_number
- title
- publisher
- url
- raw_entry

Only use information present in the report.
If no URL is present, leave url empty.

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

Judge whether the cited public URL source excerpts support the claim.

Use only the claim and source excerpts.
Do not use outside knowledge.

Use one of four final labels:

supported:
The cited public source supports the material claim.

needs_review:
The source provides some support, but the claim is too strong, too broad, incomplete, or needs better wording.

unsupported:
The public source was checked, but it does not support the claim, or it is the wrong source.

not_verifiable:
The claim has no citation, no public URL, inaccessible URL, or not enough readable source text.

Important:
- A vendor case study can support "vendor evidence indicates use/deployment/integration".
- A vendor case study does not prove internal ownership unless it says so.
- Job postings suggest capability building, not deployment, spend, or production maturity.
- Do not imply production deployment, global scale, ROI, or achieved savings unless directly supported.

Return strict JSON only.

Schema:
{
  "status": "supported | needs_review | unsupported | not_verifiable",
  "reason": "...",
  "supported_parts": "...",
  "problematic_parts": "...",
  "short_source_quote": "max 30 words",
  "suggested_rewrite": "...",
  "source_type": "official_company_filing | official_company_source | vendor_or_partner | trade_press | job_posting | academic_or_patent | generic_or_unclear | inaccessible",
  "risk_flags": {
    "vendor_overclaim": false,
    "job_posting_overclaim": false,
    "ai_scope_drift": false
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
        pages.append({
            "page": i,
            "text": page.extract_text() or "",
        })

    return pages


def pages_to_text(pages: list[dict[str, Any]], max_chars: int = 180000) -> str:
    blocks = []
    used = 0

    for page in pages:
        block = f"\n\n--- PAGE {page['page']} ---\n{page['text']}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)

    return "".join(blocks)


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


def gemini_json(client, model: str, system_prompt: str, payload: str, max_tokens: int = 16384) -> dict[str, Any]:
    def call(prompt: str):
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

    response = call(payload)
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
        Path("last_bad_public_url_json.txt").write_text(raw)

        repair_payload = f"""
Repair this invalid JSON into strict valid JSON only.
No markdown. No explanation.

Invalid JSON:
{raw[:12000]}

Error:
{first_error}
"""
        repaired = call(repair_payload)
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

        return extract_json(repaired_raw)


def extract_claims(
    client,
    model: str,
    report_text: str,
    public_citation_numbers: list[int],
) -> list[dict[str, Any]]:
    public_citations = ", ".join(map(str, public_citation_numbers))

    payload = f"""
Report:
{report_text}

Public URL citation numbers:
{public_citations}

Extract every factual, checkable claim that cites at least one of these public URL citation numbers.
Do not limit the number of claims.
Do not extract claims that cite only citation numbers outside this public URL list.
Do not extract uncited claims.
"""

    data = gemini_json(client, model, EXTRACT_CLAIMS_PROMPT, payload)

    if isinstance(data, dict):
        claims = data.get("claims", [])
    elif isinstance(data, list):
        claims = data
    else:
        claims = []

    clean_claims = []
    public_set = set(int(x) for x in public_citation_numbers)

    for i, claim in enumerate(claims, start=1):
        claim.setdefault("claim_id", f"C{i:03d}")
        claim.setdefault("citation_numbers", [])
        claim.setdefault("claim_type", "other")
        claim.setdefault("importance", "medium")

        nums = []
        for n in claim.get("citation_numbers", []):
            try:
                nums.append(int(n))
            except Exception:
                pass

        # Safety filter: keep only claims with at least one public URL citation.
        if not any(n in public_set for n in nums):
            continue

        claim["citation_numbers"] = nums
        clean_claims.append(claim)

    for i, claim in enumerate(clean_claims, start=1):
        claim["claim_id"] = f"C{i:03d}"

    return clean_claims


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


def has_public_url(ref: dict[str, Any]) -> bool:
    url = (ref.get("url") or "").strip()
    return url.startswith("http://") or url.startswith("https://")


def claim_public_refs(claim: dict[str, Any], refs: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    public_refs = []

    for n in claim.get("citation_numbers", []):
        try:
            ref = refs.get(int(n), {})
        except Exception:
            continue

        if has_public_url(ref):
            public_refs.append(ref)

    return public_refs


def extract_pdf_from_url(url: str, timeout: int = 25) -> dict[str, Any]:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 public-url-benchmark"},
    )
    response.raise_for_status()

    tmp = Path("_tmp_public_source.pdf")
    tmp.write_bytes(response.content)
    pages = read_pdf_pages(tmp)
    tmp.unlink(missing_ok=True)
    text = pages_to_text(pages, max_chars=120000)

    if text and len(text.strip()) > 200:
        return {
            "url": url,
            "status": "accessible",
            "title": "",
            "text": text.strip(),
            "error": "",
            "crawl_method": "requests_pdf_pypdf",
        }

    return {
        "url": url,
        "status": "not_readable",
        "title": "",
        "text": text.strip() if text else "",
        "error": "Extracted PDF text too short.",
        "crawl_method": "requests_pdf_pypdf",
    }


async def crawl_url_with_crawl4ai(url: str) -> dict[str, Any]:
    if AsyncWebCrawler is None:
        raise RuntimeError("Crawl4AI is not installed or could not be imported.")

    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=20,
        excluded_tags=["script", "style", "nav", "footer"],
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

    success = bool(getattr(result, "success", False))
    markdown = getattr(result, "markdown", "") or ""
    cleaned_html = getattr(result, "cleaned_html", "") or ""
    text = markdown.strip() or trafilatura.extract(
        cleaned_html,
        include_comments=False,
        include_tables=True,
        include_links=False,
    ) or ""
    title = getattr(result, "metadata", {}) or {}
    if isinstance(title, dict):
        title = title.get("title", "")
    else:
        title = ""

    if success and text and len(text.strip()) > 200:
        return {
            "url": url,
            "status": "accessible",
            "title": title,
            "text": text.strip(),
            "error": "",
            "crawl_method": "crawl4ai",
        }

    error = getattr(result, "error_message", "") or "Crawl4AI extracted text too short."
    return {
        "url": url,
        "status": "not_readable",
        "title": title,
        "text": text.strip(),
        "error": error,
        "crawl_method": "crawl4ai",
    }


def crawl_url_with_trafilatura(url: str, timeout: int = 25) -> dict[str, Any]:
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
                "url": url,
                "status": "accessible",
                "title": title,
                "text": text.strip(),
                "error": "",
                "crawl_method": "trafilatura_fallback",
            }

    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 public-url-benchmark"},
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        return extract_pdf_from_url(url, timeout=timeout)

    text = trafilatura.extract(
        response.text,
        include_comments=False,
        include_tables=True,
        include_links=False,
    )

    if text and len(text.strip()) > 200:
        return {
            "url": url,
            "status": "accessible",
            "title": "",
            "text": text.strip(),
            "error": "",
            "crawl_method": "requests_trafilatura_fallback",
        }

    return {
        "url": url,
        "status": "not_readable",
        "title": "",
        "text": text.strip() if text else "",
        "error": "Extracted text too short.",
        "crawl_method": "requests_trafilatura_fallback",
    }


def crawl_url(url: str, timeout: int = 25) -> dict[str, Any]:
    try:
        if url.lower().split("?")[0].endswith(".pdf"):
            return extract_pdf_from_url(url, timeout=timeout)

        try:
            return asyncio.run(crawl_url_with_crawl4ai(url))
        except Exception as crawl4ai_error:
            fallback = crawl_url_with_trafilatura(url, timeout=timeout)
            if fallback.get("status") == "accessible":
                fallback["error"] = f"Crawl4AI fallback used after: {crawl4ai_error}"
            return fallback

    except Exception as exc:
        return {
            "url": url,
            "status": "inaccessible",
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
        if len(part) >= 30:
            chunks.append(" ".join(part))

    return chunks


def relevant_excerpt(claim_text: str, source_text: str, top_k: int = 3, max_chars: int = 5500) -> str:
    if not source_text:
        return ""

    chunks = split_text(source_text)

    if not chunks:
        return source_text[:max_chars]

    try:
        corpus = [claim_text] + chunks
        vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
        matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        ranked = sims.argsort()[::-1][:top_k]
        selected = [chunks[i] for i in ranked]
    except Exception:
        selected = chunks[:top_k]

    return "\n\n--- EXCERPT ---\n\n".join(selected)[:max_chars]


def judge_claim(client, model: str, claim: dict[str, Any], source_bundle: list[dict[str, Any]]) -> dict[str, Any]:
    if not source_bundle:
        return {
            "status": "not_verifiable",
            "reason": "The claim has no citation with a public URL.",
            "supported_parts": "",
            "problematic_parts": "No public URL source is available for this claim.",
            "short_source_quote": "",
            "suggested_rewrite": "Add a public URL source or verify this claim separately.",
            "source_type": "inaccessible",
            "risk_flags": {
                "vendor_overclaim": False,
                "job_posting_overclaim": False,
                "ai_scope_drift": False,
            },
        }

    readable = [s for s in source_bundle if s.get("status") == "accessible" and s.get("source_excerpt")]

    if not readable:
        return {
            "status": "not_verifiable",
            "reason": "The cited public URL exists but could not be accessed or read automatically.",
            "supported_parts": "",
            "problematic_parts": "No readable source text was available.",
            "short_source_quote": "",
            "suggested_rewrite": "Check the URL manually or replace it with a more accessible source.",
            "source_type": "inaccessible",
            "risk_flags": {
                "vendor_overclaim": False,
                "job_posting_overclaim": False,
                "ai_scope_drift": False,
            },
        }

    payload = f"""
Claim:
{json.dumps(claim, ensure_ascii=False)}

Public source excerpts:
{json.dumps(readable, ensure_ascii=False)[:28000]}
"""
    return gemini_json(client, model, JUDGE_PROMPT, payload)



EXCEL_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


def clean_excel_value(value):
    if isinstance(value, str):
        return EXCEL_ILLEGAL_RE.sub("", value)
    return value


def clean_df_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.map(clean_excel_value)



def process_pdf(client, model: str, pdf_path: Path, out_dir: Path, max_claims: int) -> dict[str, Any]:
    report_name = pdf_path.name
    report_dir = out_dir / pdf_path.stem
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing {report_name} ===")

    pages = read_pdf_pages(pdf_path)
    report_text = pages_to_text(pages)

    print("Extracting URL references...")
    refs = extract_references(client, model, report_text)

    public_ref_numbers = sorted([
        n for n, ref in refs.items()
        if has_public_url(ref)
    ])

    print(f"Public URL sources found: {len(public_ref_numbers)}")

    print("Extracting all factual claims with public URL citations...")
    claims = extract_claims(
        client=client,
        model=model,
        report_text=report_text,
        public_citation_numbers=public_ref_numbers,
    )

    print(f"Public URL claims extracted: {len(claims)}")

    source_cache = {}
    source_rows = []

    for n in tqdm(public_ref_numbers, desc="Crawling URLs"):
        ref = refs[n]
        url = ref["url"]

        if url not in source_cache:
            source_cache[url] = crawl_url(url)
            time.sleep(0.4)

        crawled = source_cache[url]

        source_rows.append({
            "report": report_name,
            "citation_number": n,
            "title": ref.get("title", ""),
            "publisher": ref.get("publisher", ""),
            "url": url,
            "status": crawled.get("status", ""),
            "crawl_method": crawled.get("crawl_method", ""),
            "error": crawled.get("error", ""),
            "text_length": len(crawled.get("text", "")),
        })

    rows = []
    ignored_rows = []

    print(f"Judging public-URL-verifiable claims out of {len(claims)} extracted claims...")
    for claim in tqdm(claims, desc="Judging claims"):
        public_refs = claim_public_refs(claim, refs)

        if not public_refs:
            ignored_rows.append({
                "report": report_name,
                "claim_id": claim.get("claim_id"),
                "claim_text": claim.get("claim_text"),
                "section": claim.get("section"),
                "claim_type": claim.get("claim_type"),
                "importance": claim.get("importance"),
                "citation_numbers": ", ".join(map(str, claim.get("citation_numbers", []))),
                "ignore_reason": "No public URL citation. Ignored in public URL benchmark.",
            })
            continue

        source_bundle = []

        for ref in public_refs:
            crawled = source_cache.get(ref["url"], {})
            excerpt = relevant_excerpt(claim.get("claim_text", ""), crawled.get("text", ""))

            source_bundle.append({
                "citation_number": ref.get("citation_number"),
                "title": ref.get("title", ""),
                "publisher": ref.get("publisher", ""),
                "url": ref.get("url", ""),
                "status": crawled.get("status", ""),
                "error": crawled.get("error", ""),
                "source_excerpt": excerpt,
            })

        try:
            judgement = judge_claim(client, model, claim, source_bundle)
        except Exception as exc:
            judgement = {
                "status": "not_verifiable",
                "reason": f"Judge failed: {exc}",
                "supported_parts": "",
                "problematic_parts": "",
                "short_source_quote": "",
                "suggested_rewrite": "",
                "source_type": "inaccessible",
                "risk_flags": {
                    "vendor_overclaim": False,
                    "job_posting_overclaim": False,
                    "ai_scope_drift": False,
                },
            }

        flags = judgement.get("risk_flags", {}) or {}

        rows.append({
            "report": report_name,
            "claim_id": claim.get("claim_id"),
            "claim_text": claim.get("claim_text"),
            "section": claim.get("section"),
            "claim_type": claim.get("claim_type"),
            "importance": claim.get("importance"),
            "citation_numbers": ", ".join(map(str, claim.get("citation_numbers", []))),
            "public_urls_used": "\n".join([s.get("url", "") for s in source_bundle]),
            "status": judgement.get("status", "not_verifiable"),
            "reason": judgement.get("reason", ""),
            "supported_parts": judgement.get("supported_parts", ""),
            "problematic_parts": judgement.get("problematic_parts", ""),
            "short_source_quote": judgement.get("short_source_quote", ""),
            "suggested_rewrite": judgement.get("suggested_rewrite", ""),
            "source_type": judgement.get("source_type", ""),
            "vendor_overclaim": flags.get("vendor_overclaim", False),
            "job_posting_overclaim": flags.get("job_posting_overclaim", False),
            "ai_scope_drift": flags.get("ai_scope_drift", False),
        })

    claims_df = pd.DataFrame(rows)
    ignored_df = pd.DataFrame(ignored_rows)
    sources_df = pd.DataFrame(source_rows)

    if claims_df.empty:
        summary_by_status = pd.DataFrame(columns=["status", "claim_count"])
        summary_by_type = pd.DataFrame(columns=["claim_type", "status", "claim_count"])
    else:
        summary_by_status = (
            claims_df.groupby("status", dropna=False)
            .size()
            .reset_index(name="claim_count")
            .sort_values("claim_count", ascending=False)
        )

        summary_by_type = (
            claims_df.groupby(["claim_type", "status"], dropna=False)
            .size()
            .reset_index(name="claim_count")
            .sort_values(["claim_type", "claim_count"], ascending=[True, False])
        )

    if sources_df.empty:
        source_status = pd.DataFrame()
    else:
        source_status = (
            sources_df.groupby("status", dropna=False)
            .size()
            .reset_index(name="source_count")
            .sort_values("source_count", ascending=False)
        )

    if claims_df.empty:
        claims_to_review = pd.DataFrame()
    else:
        claims_to_review = claims_df[
            claims_df["status"].isin(["needs_review", "unsupported", "not_verifiable"])
        ].copy()


    summary_by_status = clean_df_for_excel(summary_by_status)
    summary_by_type = clean_df_for_excel(summary_by_type)
    source_status = clean_df_for_excel(source_status)
    claims_to_review = clean_df_for_excel(claims_to_review)
    claims_df = clean_df_for_excel(claims_df)
    ignored_df = clean_df_for_excel(ignored_df)
    sources_df = clean_df_for_excel(sources_df)

    xlsx_path = report_dir / f"{pdf_path.stem}_public_url_benchmark.xlsx"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_by_status.to_excel(writer, sheet_name="summary_by_status", index=False)
        summary_by_type.to_excel(writer, sheet_name="summary_by_claim_type", index=False)
        source_status.to_excel(writer, sheet_name="source_status", index=False)
        claims_to_review.to_excel(writer, sheet_name="claims_to_review", index=False)
        claims_df.to_excel(writer, sheet_name="checked_claims", index=False)
        ignored_df.to_excel(writer, sheet_name="ignored_no_public_url", index=False)
        sources_df.to_excel(writer, sheet_name="all_public_sources", index=False)

    claims_df.to_json(report_dir / "public_url_claims.json", orient="records", indent=2)
    ignored_df.to_json(report_dir / "ignored_no_public_url_claims.json", orient="records", indent=2)
    sources_df.to_json(report_dir / "public_url_sources.json", orient="records", indent=2)

    print(f"Saved: {xlsx_path}")

    return {
        "report": report_name,
        "total_claims_extracted": len(claims),
        "claims_checked_with_public_urls": len(claims_df),
        "claims_ignored_no_public_url": len(ignored_df),
        "supported": int((claims_df["status"] == "supported").sum()) if not claims_df.empty else 0,
        "needs_review": int((claims_df["status"] == "needs_review").sum()) if not claims_df.empty else 0,
        "unsupported": int((claims_df["status"] == "unsupported").sum()) if not claims_df.empty else 0,
        "not_verifiable": int((claims_df["status"] == "not_verifiable").sum()) if not claims_df.empty else 0,
        "public_sources_found": len(public_ref_numbers),
        "public_sources_accessible": int((sources_df["status"] == "accessible").sum()) if not sources_df.empty else 0,
        "public_sources_not_accessible": int((sources_df["status"] != "accessible").sum()) if not sources_df.empty else 0,
        "output_file": str(xlsx_path),
    }


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pdf", default="")
    parser.add_argument("--out-dir", default="benchmark_outputs_public_url")
    parser.add_argument("--model", default=os.getenv("BENCHMARK_MODEL", "gemini-3.1-pro-preview"))
    parser.add_argument("--location", default=os.getenv("VERTEX_LOCATION", "global"))
    parser.add_argument("--max-claims", type=int, default=0, help=argparse.SUPPRESS)

    args = parser.parse_args()

    project_id = detect_project_id()

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=args.location,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        pdfs = [Path(args.pdf)]
    else:
        pdfs = sorted(Path(args.data_dir).glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError("No PDFs found.")

    summaries = []

    for pdf in pdfs:
        summaries.append(
            process_pdf(
                client=client,
                model=args.model,
                pdf_path=pdf,
                out_dir=out_dir,
                max_claims=args.max_claims,
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / "public_url_benchmark_summary.xlsx"

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary_by_report", index=False)

    print(f"\nGlobal summary saved: {summary_path}")


if __name__ == "__main__":
    main()
