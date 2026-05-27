# Thesis code repository

This repository contains the scripts used for the thesis pipeline. Data files, reports, benchmark outputs, and local indexes should be added locally by the user when the pipeline has to be rerun end to end.

## What is included

- `hiring_reports/report_generation.py` classifies job postings and generates hiring reports.
- `research_agent/run_google_deepsearch.py` expands hiring evidence with annual reports and public sources.
- `benchmark/scripts/` contains the public source grounding benchmark and metric scripts.
- `rag/` contains the local retrieval index builder and chatbot interfaces.
- `.env.example` lists the environment variables needed to run the scripts.
- `requirements.txt` lists the Python dependencies.

## Setup

Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Copy the environment template and fill in your own values.

```bash
cp .env.example .env
```

Do not commit `.env`. It is ignored by `.gitignore`.

To check the local setup, run:

```bash
python scripts/check_setup.py
```

## Expected local folder structure

```text
job_postings/
  pre_processed/          # input Excel files for each company
  processed/              # enriched Excel outputs
annual_reports/           # annual reports used by the research agent
hiring_reports/
  reports/                # generated hiring reports
final_reports/            # generated final company reports
benchmark/
  data/                   # final report PDFs copied here for benchmarking
  output_public_url/      # raw benchmark outputs
  results/                # summary Excel files
rag/
  data/                   # PDFs indexed by the RAG system
  index/                  # generated vector index
```

## Pipeline order

### 1. Job posting classification and hiring reports

Place one Excel file per company in `job_postings/pre_processed/`. Then run:

```bash
python hiring_reports/report_generation.py
```

The script expects Google Cloud / Vertex AI to be configured through `.env` or through your shell. It treats job postings as hiring evidence, not as proof of deployment or spending.

### 2. Research agent for final company reports

The research agent combines one hiring report with one annual report and searches additional public sources. Example:

```bash
python research_agent/run_google_deepsearch.py \
  --annual-report annual_reports/Company_Annual_Report.pdf \
  --hiring-report hiring_reports/reports/company_macro_ai_hiring_report.pdf \
  --company "Company Name" \
  --output-dir final_reports
```

The script requires either `GEMINI_API_KEY` or `GOOGLE_API_KEY`, unless it has been adapted to another authentication flow.

### 3. Public URL grounding benchmark

Copy the final report PDFs to `benchmark/data/`, then run:

```bash
python benchmark/scripts/run_missing_public_url_benchmark.py \
  --data-dir benchmark/data \
  --out-dir benchmark/output_public_url
```

Then create the comparison file:

```bash
python benchmark/scripts/make_public_url_metrics.py \
  --out-dir benchmark/output_public_url \
  --output-file benchmark/results/public_url_metrics_comparison.xlsx
```

Then compute the final grounding metrics:

```bash
python benchmark/scripts/compute_public_grounding_metrics.py \
  --input benchmark/results/public_url_metrics_comparison.xlsx \
  --output benchmark/results/public_grounding_metrics.xlsx
```

### 4. RAG chatbot

Copy evaluated final reports to `rag/data/`, then build the local index:

```bash
python rag/build_index.py --data-dir rag/data --index-dir rag/index
```

For the Vertex AI chat interface, run:

```bash
python rag/chat_vertex.py
```

For the Gradio interface, run:

```bash
python rag/gradio_app.py
```
