from __future__ import annotations

import importlib.util
import os
from pathlib import Path

REQUIRED = [
    "pandas",
    "numpy",
    "openpyxl",
    "pypdf",
    "sentence_transformers",
    "dotenv",
    "google.genai",
    "reportlab",
    "matplotlib",
    "tqdm",
]

missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python packages:")
    for name in missing:
        print(f"- {name}")
    print("\nInstall them with: pip install -r requirements.txt")
else:
    print("All core Python packages are importable.")

for folder in [
    "job_postings/pre_processed",
    "job_postings/processed",
    "annual_reports",
    "hiring_reports/reports",
    "final_reports",
    "benchmark/data",
    "benchmark/results",
    "rag/data",
    "rag/index",
]:
    path = Path(folder)
    print(f"{folder}: {'exists' if path.exists() else 'missing'}")

for key in ["PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY"]:
    print(f"{key}: {'set' if os.getenv(key) else 'not set'}")
