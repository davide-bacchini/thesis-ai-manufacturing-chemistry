from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_metrics(out_dir: Path, output_file: Path) -> None:
    rows = []
    status_rows = []
    source_rows = []

    if not out_dir.exists():
        raise FileNotFoundError(f"Benchmark output folder not found: {out_dir}")

    for report_dir in sorted(out_dir.iterdir()):
        if not report_dir.is_dir():
            continue

        claims_path = report_dir / "public_url_claims.json"
        ignored_path = report_dir / "ignored_no_public_url_claims.json"
        sources_path = report_dir / "public_url_sources.json"

        if not claims_path.exists():
            continue

        checked_claims = load_json(claims_path)
        ignored_claims = load_json(ignored_path) if ignored_path.exists() else []
        sources = load_json(sources_path) if sources_path.exists() else []

        total_extracted = len(checked_claims) + len(ignored_claims)
        checked = len(checked_claims)
        ignored = len(ignored_claims)

        supported = sum(1 for c in checked_claims if c.get("status") == "supported")
        needs_review = sum(1 for c in checked_claims if c.get("status") == "needs_review")
        unsupported = sum(1 for c in checked_claims if c.get("status") == "unsupported")
        not_verifiable_url = sum(1 for c in checked_claims if c.get("status") == "not_verifiable")

        public_sources_found = len(sources)
        accessible_sources = sum(1 for s in sources if s.get("status") == "accessible")
        inaccessible_or_unreadable = public_sources_found - accessible_sources

        row = {
            "company_report": report_dir.name + ".pdf",
            "total_claims_extracted": total_extracted,
            "public_url_checked": checked,
            "ignored_no_public_url": ignored,
            "public_url_coverage": checked / total_extracted if total_extracted else 0,
            "supported": supported,
            "needs_review": needs_review,
            "unsupported": unsupported,
            "not_verifiable_url": not_verifiable_url,
            "supported_rate": supported / checked if checked else 0,
            "review_rate": needs_review / checked if checked else 0,
            "unsupported_rate": unsupported / checked if checked else 0,
            "not_verifiable_url_rate": not_verifiable_url / checked if checked else 0,
            "public_sources_found": public_sources_found,
            "accessible_public_sources": accessible_sources,
            "inaccessible_or_unreadable_public_sources": inaccessible_or_unreadable,
            "source_accessibility_rate": accessible_sources / public_sources_found if public_sources_found else 0,
        }
        rows.append(row)

        status_map = {
            "supported": {"count": supported, "rate": row["supported_rate"]},
            "needs_review": {"count": needs_review, "rate": row["review_rate"]},
            "unsupported": {"count": unsupported, "rate": row["unsupported_rate"]},
            "not_verifiable_url": {"count": not_verifiable_url, "rate": row["not_verifiable_url_rate"]},
        }

        for status, values in status_map.items():
            status_rows.append({
                "company_report": report_dir.name + ".pdf",
                "status": status,
                "claim_count": values["count"],
                "share_of_checked_claims": values["rate"],
            })

        for src in sources:
            source_rows.append({
                "company_report": report_dir.name + ".pdf",
                "citation_number": src.get("citation_number"),
                "title": src.get("title"),
                "publisher": src.get("publisher"),
                "url": src.get("url"),
                "status": src.get("status"),
                "crawl_method": src.get("crawl_method"),
                "text_length": src.get("text_length"),
                "error": src.get("error"),
            })

    metrics_df = pd.DataFrame(rows)
    status_df = pd.DataFrame(status_rows)
    sources_df = pd.DataFrame(source_rows)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="metrics_by_report", index=False)
        status_df.to_excel(writer, sheet_name="claim_status_long", index=False)
        sources_df.to_excel(writer, sheet_name="source_accessibility", index=False)

    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comparison metrics from public URL benchmark outputs.")
    parser.add_argument("--out-dir", default="benchmark/output_public_url", help="Folder containing per-report benchmark outputs.")
    parser.add_argument("--output-file", default="benchmark/results/public_url_metrics_comparison.xlsx", help="Excel file to write.")
    args = parser.parse_args()
    build_metrics(Path(args.out_dir), Path(args.output_file))


if __name__ == "__main__":
    main()
