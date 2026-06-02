# Thesis code repository

## What is included

- `hiring_reports/report_generation.py` classifies job postings and generates hiring reports.
- `research_agent/run_google_deepsearch.py` expands hiring evidence with annual reports and public sources.
- `benchmark/scripts/` contains the public source grounding benchmark and metric scripts.
- `rag/build_index.py` builds the retrieval index from evaluated company reports.
- `rag/gradio_app.py` is the only chatbot interface kept in the repository.
- `rag/rag_core.py` contains shared indexing and retrieval utilities used by the index builder and the Gradio app.
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
  pre_processed/          # input Excel files, one workbook per company
  processed/              # enriched Excel outputs created by the hiring script
annual_reports/           # annual reports used by the research agent
hiring_reports/
  reports/                # generated hiring reports
final_reports/            # generated final company reports
benchmark/
  data/                   # final report PDFs copied here for benchmarking
  output_public_url/      # raw benchmark outputs
  results/                # summary Excel files
rag/
  data/                   # evaluated final report PDFs indexed by the RAG system
  index/                  # generated vector index
```

## Pipeline order

### 1. Job posting classification and hiring reports

Place one Excel file per company in `job_postings/pre_processed/`. Then run:

```bash
python hiring_reports/report_generation.py
```

The script reads from `job_postings/pre_processed/`, writes enriched workbooks to `job_postings/processed/`, and writes hiring reports under `hiring_reports/reports/`. It treats job postings as hiring evidence, not as proof of deployment, spending, headcount, or production maturity.

### 2. Research agent for final company reports

The research agent combines one hiring report with one annual report and searches additional public sources. Example:

```bash
python research_agent/run_google_deepsearch.py   --annual-report annual_reports/Company_Annual_Report.pdf   --hiring-report hiring_reports/reports/company/company_macro_ai_hiring_report.pdf   --company "Company Name"   --output-dir final_reports
```

The generated report includes inline numbered citations and a final `SOURCES` section. That source list is required by the public URL benchmark.

The script requires either `GEMINI_API_KEY` or `GOOGLE_API_KEY`, unless it has been adapted to another authentication flow.

### 3. Public URL grounding benchmark

Copy the final report PDFs to `benchmark/data/`, then run:

```bash
python benchmark/scripts/run_missing_public_url_benchmark.py   --data-dir benchmark/data   --out-dir benchmark/output_public_url
```

The benchmark uses Crawl4AI as the first extraction method for web pages, with PDF extraction and text fallback methods when needed. Its default judge model is controlled by `BENCHMARK_MODEL`, which is set to `gemini-3.1-pro-preview` in `.env.example`.

Then create the comparison file:

```bash
python benchmark/scripts/make_public_url_metrics.py   --out-dir benchmark/output_public_url   --output-file benchmark/results/public_url_metrics_comparison.xlsx
```

Then compute the final grounding metrics:

```bash
python benchmark/scripts/compute_public_grounding_metrics.py   --input benchmark/results/public_url_metrics_comparison.xlsx   --output benchmark/results/public_grounding_metrics.xlsx
```

### 4. Gradio RAG chatbot

Copy evaluated final reports to `rag/data/`, then build the local retrieval index:

```bash
python rag/build_index.py --data-dir rag/data --index-dir rag/index
```

Run the Gradio interface:

```bash
python rag/gradio_app.py
```
