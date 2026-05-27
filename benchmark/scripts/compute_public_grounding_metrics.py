from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_metrics(input_path: Path, output_path: Path) -> None:
    df = pd.read_excel(input_path, sheet_name="metrics_by_report")

    required = [
        "company_report",
        "supported",
        "needs_review",
        "unsupported",
        "not_verifiable_url",
        "public_sources_found",
        "accessible_public_sources",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    S = df["supported"]
    R = df["needs_review"]
    U = df["unsupported"]
    V = df["not_verifiable_url"]

    judged_claims = S + R + U
    all_public_claims = S + R + U + V

    df["Public Grounding Accuracy"] = (S + 0.5 * R) / judged_claims.replace(0, pd.NA)
    df["Strict Support Rate"] = S / judged_claims.replace(0, pd.NA)
    df["Review Burden Rate"] = R / judged_claims.replace(0, pd.NA)
    df["Unsupported Rate"] = U / judged_claims.replace(0, pd.NA)
    df["Public Verifiability Rate"] = judged_claims / all_public_claims.replace(0, pd.NA)
    df["Source Accessibility Rate"] = df["accessible_public_sources"] / df["public_sources_found"].replace(0, pd.NA)

    batch_S = df["supported"].sum()
    batch_R = df["needs_review"].sum()
    batch_U = df["unsupported"].sum()
    batch_V = df["not_verifiable_url"].sum()

    batch_judged_claims = batch_S + batch_R + batch_U
    batch_all_public_claims = batch_S + batch_R + batch_U + batch_V

    batch_sources = df["public_sources_found"].sum()
    batch_accessible_sources = df["accessible_public_sources"].sum()

    batch_metrics = pd.DataFrame([{
        "Public Grounding Accuracy": (batch_S + 0.5 * batch_R) / batch_judged_claims if batch_judged_claims else None,
        "Strict Support Rate": batch_S / batch_judged_claims if batch_judged_claims else None,
        "Review Burden Rate": batch_R / batch_judged_claims if batch_judged_claims else None,
        "Unsupported Rate": batch_U / batch_judged_claims if batch_judged_claims else None,
        "Public Verifiability Rate": batch_judged_claims / batch_all_public_claims if batch_all_public_claims else None,
        "Source Accessibility Rate": batch_accessible_sources / batch_sources if batch_sources else None,
    }])

    batch_counts = pd.DataFrame([{
        "Supported Claims": batch_S,
        "Claims Needing Review": batch_R,
        "Unsupported Claims": batch_U,
        "Not Verifiable URL Claims": batch_V,
        "Judged Public Claims": batch_judged_claims,
        "All Public URL Claims": batch_all_public_claims,
        "Public Sources Found": batch_sources,
        "Accessible Public Sources": batch_accessible_sources,
    }])

    company_metrics = df.rename(columns={
        "company_report": "Company Report",
        "public_url_checked": "Public URL Claims Checked",
        "supported": "Supported Claims",
        "needs_review": "Claims Needing Review",
        "unsupported": "Unsupported Claims",
        "not_verifiable_url": "Not Verifiable URL Claims",
        "public_sources_found": "Public Sources Found",
        "accessible_public_sources": "Accessible Public Sources",
    })

    cols = [
        "Company Report",
        "Public URL Claims Checked",
        "Supported Claims",
        "Claims Needing Review",
        "Unsupported Claims",
        "Not Verifiable URL Claims",
        "Public Grounding Accuracy",
        "Strict Support Rate",
        "Review Burden Rate",
        "Unsupported Rate",
        "Public Verifiability Rate",
        "Public Sources Found",
        "Accessible Public Sources",
        "Source Accessibility Rate",
    ]
    cols = [c for c in cols if c in company_metrics.columns]
    company_metrics = company_metrics[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        batch_metrics.to_excel(writer, sheet_name="Batch Metrics", index=False)
        batch_counts.to_excel(writer, sheet_name="Batch Counts", index=False)
        company_metrics.to_excel(writer, sheet_name="Company Metrics", index=False)

    print("Batch Metrics")
    print(batch_metrics.to_string(index=False))
    print(f"\nSaved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute final grounding metrics from the public URL comparison file.")
    parser.add_argument("--input", default="benchmark/results/public_url_metrics_comparison.xlsx", help="Input metrics comparison Excel file.")
    parser.add_argument("--output", default="benchmark/results/public_grounding_metrics.xlsx", help="Output Excel file.")
    args = parser.parse_args()
    compute_metrics(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
