#!/usr/bin/env python3
# Adaptive Vertex pipeline: dynamic workers, dynamic rate limit, reusable deps, dtype fixes, resume safe.


# Script for classifying job postings and generating hiring reports.
# Configure credentials through .env or your shell before running.


import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass




# ===== Notebook cell 1 =====

import os
import sys
from datetime import datetime
from pathlib import Path

print("Python executable:", sys.executable)
print("Recommended kernel: Python Gemini Vertex")

# Remove old throwaway local dependency folders from sys.path.
# The balanced script uses one reusable folder instead of reinstalling fresh packages every run.
sys.path = [p for p in sys.path if "/_pydeps_clean_" not in p]

# Reusable dependency folder. Set FORCE_INSTALL_DEPS=1 only if you want to refresh it.
PKG_DIR = os.path.abspath(os.getenv("PIPELINE_DEPS_DIR", "./_pydeps_vertex_pipeline"))
os.makedirs(PKG_DIR, exist_ok=True)
sys.path.insert(0, PKG_DIR)

import subprocess
import importlib.util

REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "matplotlib",
    "reportlab",
    "openpyxl",
    "tqdm",
    "PIL",
    "google.genai",
    "google.auth",
    "pydantic",
]

DEPS_READY_MARKER = os.path.join(PKG_DIR, ".deps_ready_v3")
FORCE_INSTALL_DEPS = os.getenv("FORCE_INSTALL_DEPS", "0").strip().lower() in {"1", "true", "yes"}
missing_modules = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]

if FORCE_INSTALL_DEPS or missing_modules or not os.path.exists(DEPS_READY_MARKER):
    print("Installing or refreshing local dependencies in:", PKG_DIR)
    if missing_modules:
        print("Missing modules:", ", ".join(missing_modules))
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--upgrade",
        f"--target={PKG_DIR}",
        "numpy>=1.26,<2.0",
        "pandas>=2.2,<3.0",
        "matplotlib>=3.8,<3.11",
        "reportlab>=4.0,<5.0",
        "openpyxl>=3.1,<4.0",
        "tqdm>=4.66,<5.0",
        "pillow>=10.0,<13.0",
        "google-genai>=1.51.0,<2.0.0",
        "google-auth>=2.51.0",
        "pydantic>=2,<3",
    ], check=True)
    Path(DEPS_READY_MARKER).write_text(datetime.now().isoformat(), encoding="utf-8")
else:
    print("Using cached local dependencies:", PKG_DIR)

# Make the isolated dependency folder visible to subprocesses as well.
os.environ["PYTHONPATH"] = PKG_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

print("Using:", PKG_DIR)
print("sys.path[0]:", sys.path[0])



# ===== Notebook cell 3 =====

import sys

for name in list(sys.modules):
    if (
        name == "PIL" or name.startswith("PIL.")
        or name == "matplotlib" or name.startswith("matplotlib.")
    ):
        del sys.modules[name]

print("Cleared cached PIL and matplotlib modules")



# ===== Notebook cell 5 =====

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import json
import os
import re
import tempfile
from textwrap import dedent, wrap
from typing import Literal, Optional
import time
import random
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer
from tqdm import tqdm

def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def env_optional_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    value = str(value).strip().lower()
    if value in {"none", "0", "false", "off", "no"}:
        return None
    return int(value)


COMPANIES_ROOT = Path(os.getenv("COMPANIES_ROOT", "companies")).expanduser()
COMPANIES_ROOT.mkdir(parents=True, exist_ok=True)

MODEL = os.getenv("MODEL_NAME", "gemini-3-flash-preview")
TEMPERATURE = 0.0
TOP_P = 1.0
BATCH_SIZE = env_int("BATCH_SIZE", 100)
MAX_WORKERS = env_int("MAX_WORKERS", 6)  # Starting workers when adaptive tuning is enabled.
MAX_RETRIES = env_int("MAX_RETRIES", 5)
SAVE_EVERY_BATCH = True
SLEEP_BETWEEN_BATCHES = env_int("SLEEP_BETWEEN_BATCHES", 2)
MAX_CALLS_PER_MINUTE = env_optional_int("MAX_CALLS_PER_MINUTE", 60)  # Starting calls/minute.
SHOW_PLOTS = False
RUN_ANALYSIS = env_bool("RUN_ANALYSIS", True)
SLEEP_BETWEEN_COMPANIES = env_int("SLEEP_BETWEEN_COMPANIES", 0)

# Adaptive tuning. The script changes these values after each batch.
ADAPTIVE_TUNING = env_bool("ADAPTIVE_TUNING", True)
MIN_WORKERS = env_int("MIN_WORKERS", 3)
MAX_WORKERS_CAP = env_int("MAX_WORKERS_CAP", max(MAX_WORKERS, 12))
MIN_CALLS_PER_MINUTE = env_int("MIN_CALLS_PER_MINUTE", 30)
MAX_CALLS_PER_MINUTE_CAP = env_int("MAX_CALLS_PER_MINUTE_CAP", max(MAX_CALLS_PER_MINUTE or 60, 120))
TUNE_UP_CLEAN_BATCHES = env_int("TUNE_UP_CLEAN_BATCHES", 2)
TUNE_DOWN_RETRY_RATE = env_float("TUNE_DOWN_RETRY_RATE", 0.08)
TUNE_DOWN_HARD_RETRY_RATE = env_float("TUNE_DOWN_HARD_RETRY_RATE", 0.18)
TUNE_UP_MAX_RETRY_RATE = env_float("TUNE_UP_MAX_RETRY_RATE", 0.01)

MAX_WORKERS = max(MIN_WORKERS, min(MAX_WORKERS, MAX_WORKERS_CAP))
if MAX_CALLS_PER_MINUTE is not None:
    MAX_CALLS_PER_MINUTE = max(MIN_CALLS_PER_MINUTE, min(MAX_CALLS_PER_MINUTE, MAX_CALLS_PER_MINUTE_CAP))

CURRENT_WORKERS = MAX_WORKERS
CURRENT_CALLS_PER_MINUTE = MAX_CALLS_PER_MINUTE
CLEAN_BATCH_STREAK = 0

print(f"Companies root: {COMPANIES_ROOT.resolve()}")
print(f"Model: {MODEL}")
print(f"Sleep between batches: {SLEEP_BETWEEN_BATCHES}s")
print(f"Sleep between companies: {SLEEP_BETWEEN_COMPANIES}s")
print(f"Starting calls per minute: {MAX_CALLS_PER_MINUTE}")
print(f"Starting workers: {MAX_WORKERS}")
print(f"Max retries: {MAX_RETRIES}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Run analysis: {RUN_ANALYSIS}")
print(f"Adaptive tuning: {ADAPTIVE_TUNING}")
if ADAPTIVE_TUNING:
    print(f"Adaptive worker range: {MIN_WORKERS}-{MAX_WORKERS_CAP}")
    print(f"Adaptive calls/min range: {MIN_CALLS_PER_MINUTE}-{MAX_CALLS_PER_MINUTE_CAP}")


class AdaptiveRateLimiter:
    """Thread-safe limiter with a shared cooldown after quota or rate errors."""

    def __init__(self, calls_per_minute: int):
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be positive")
        self.lock = threading.Lock()
        self.next_allowed_time = 0.0
        self.cooldown_until = 0.0
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute

    def wait(self):
        with self.lock:
            now = time.monotonic()
            target_time = max(self.next_allowed_time, self.cooldown_until)
            wait_time = target_time - now
            if wait_time > 0:
                time.sleep(wait_time)
            self.next_allowed_time = time.monotonic() + self.min_interval

    def cooldown(self, seconds: float):
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + max(0.0, seconds))

    def set_rate(self, calls_per_minute: int):
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be positive")
        with self.lock:
            self.calls_per_minute = calls_per_minute
            self.min_interval = 60.0 / calls_per_minute


RATE_LIMITER = AdaptiveRateLimiter(MAX_CALLS_PER_MINUTE) if MAX_CALLS_PER_MINUTE else None


class BatchStats:
    """Thread-safe counters used by the adaptive tuner after each batch."""

    def __init__(self):
        self.lock = threading.Lock()
        self.retryable_errors = 0
        self.quota_rate_errors = 0
        self.non_retryable_errors = 0
        self.final_errors = 0

    def record_retryable(self, exc: Exception):
        with self.lock:
            self.retryable_errors += 1
            if is_quota_or_rate_error(exc):
                self.quota_rate_errors += 1

    def record_non_retryable(self, exc: Exception):
        with self.lock:
            self.non_retryable_errors += 1

    def record_final_error(self):
        with self.lock:
            self.final_errors += 1

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "retryable_errors": self.retryable_errors,
                "quota_rate_errors": self.quota_rate_errors,
                "non_retryable_errors": self.non_retryable_errors,
                "final_errors": self.final_errors,
            }


def is_quota_or_rate_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "quota" in text
        or "rate" in text
    )


def is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        is_quota_or_rate_error(exc)
        or "503" in text
        or "500" in text
        or "deadline" in text
        or "timeout" in text
        or "temporarily unavailable" in text
    )


def sleep_for_retry(exc: Exception, attempt: int):
    is_quota_or_rate = is_quota_or_rate_error(exc)
    if is_quota_or_rate:
        # Keep the first retry useful, but avoid many workers hammering the endpoint at once.
        wait = min(180.0, 45.0 * (attempt + 1))
        if RATE_LIMITER is not None:
            RATE_LIMITER.cooldown(min(wait, 75.0))
    else:
        wait = min(60.0, 4.0 * (2 ** attempt))
    wait += random.uniform(0, 6)
    message = str(exc).replace("\n", " ")[:220]
    print(f"Retryable error on attempt {attempt + 1}/{MAX_RETRIES}. Sleeping {wait:.1f}s before retry. {message}")
    time.sleep(wait)


def _set_calls_per_minute(new_value: Optional[int]):
    """Update the global rate limiter without losing existing cooldown state."""
    global CURRENT_CALLS_PER_MINUTE, RATE_LIMITER

    if new_value is None:
        CURRENT_CALLS_PER_MINUTE = None
        RATE_LIMITER = None
        return

    new_value = max(MIN_CALLS_PER_MINUTE, min(int(new_value), MAX_CALLS_PER_MINUTE_CAP))
    CURRENT_CALLS_PER_MINUTE = new_value
    if RATE_LIMITER is None:
        RATE_LIMITER = AdaptiveRateLimiter(new_value)
    else:
        RATE_LIMITER.set_rate(new_value)


def tune_after_batch(batch_number: int, batch_total: int, ok_count: int, stats: BatchStats, elapsed_seconds: float):
    """Increase throughput after clean batches and decrease it after retry-heavy batches."""
    global CURRENT_WORKERS, CLEAN_BATCH_STREAK

    snapshot = stats.snapshot()
    retryable = snapshot["retryable_errors"]
    quota_rate = snapshot["quota_rate_errors"]
    non_retryable = snapshot["non_retryable_errors"]
    final_errors = snapshot["final_errors"]
    retry_rate = retryable / max(batch_total, 1)
    rows_per_minute = ok_count / max(elapsed_seconds / 60.0, 0.001)

    print(
        "Batch metrics | "
        f"ok={ok_count}/{batch_total} | final_errors={final_errors} | "
        f"retryable={retryable} | quota/rate={quota_rate} | "
        f"non_retryable={non_retryable} | retry_rate={retry_rate:.1%} | "
        f"throughput={rows_per_minute:.1f} rows/min"
    )

    if not ADAPTIVE_TUNING:
        return

    old_workers = CURRENT_WORKERS
    old_cpm = CURRENT_CALLS_PER_MINUTE
    reason = "keep"

    if final_errors > 0 or retry_rate >= TUNE_DOWN_HARD_RETRY_RATE:
        CURRENT_WORKERS = max(MIN_WORKERS, int(max(1, CURRENT_WORKERS * 0.65)))
        if CURRENT_CALLS_PER_MINUTE is not None:
            _set_calls_per_minute(max(MIN_CALLS_PER_MINUTE, int(CURRENT_CALLS_PER_MINUTE * 0.65)))
        CLEAN_BATCH_STREAK = 0
        reason = "hard decrease after failed or retry-heavy batch"
        if RATE_LIMITER is not None:
            RATE_LIMITER.cooldown(45)

    elif quota_rate > 0 or retry_rate >= TUNE_DOWN_RETRY_RATE:
        CURRENT_WORKERS = max(MIN_WORKERS, CURRENT_WORKERS - 1)
        if CURRENT_CALLS_PER_MINUTE is not None:
            _set_calls_per_minute(max(MIN_CALLS_PER_MINUTE, CURRENT_CALLS_PER_MINUTE - 10))
        CLEAN_BATCH_STREAK = 0
        reason = "soft decrease after quota/rate pressure"
        if RATE_LIMITER is not None:
            RATE_LIMITER.cooldown(20)

    elif final_errors == 0 and retry_rate <= TUNE_UP_MAX_RETRY_RATE:
        CLEAN_BATCH_STREAK += 1
        if CLEAN_BATCH_STREAK >= TUNE_UP_CLEAN_BATCHES:
            CURRENT_WORKERS = min(MAX_WORKERS_CAP, CURRENT_WORKERS + 1)
            if CURRENT_CALLS_PER_MINUTE is not None:
                _set_calls_per_minute(min(MAX_CALLS_PER_MINUTE_CAP, CURRENT_CALLS_PER_MINUTE + 10))
            CLEAN_BATCH_STREAK = 0
            reason = "increase after clean batches"
        else:
            reason = f"clean batch streak {CLEAN_BATCH_STREAK}/{TUNE_UP_CLEAN_BATCHES}"

    else:
        CLEAN_BATCH_STREAK = 0

    if old_workers != CURRENT_WORKERS or old_cpm != CURRENT_CALLS_PER_MINUTE or reason != "keep":
        print(
            "Adaptive tuning | "
            f"reason={reason} | workers {old_workers} -> {CURRENT_WORKERS} | "
            f"calls/min {old_cpm} -> {CURRENT_CALLS_PER_MINUTE}"
        )


# ===== Notebook cell 9 =====

PRIMARY_CATEGORIES = [
    "bi_analytics",
    "data_engineering",
    "cloud_platforms",
    "software_engineering",
    "enterprise_systems",
    "ai_ml_modeling",
    "llm_genai_applications",
    "agentic_ai_systems",
    "industrial_automation",
    "materials_science_rd",
    "non_technical",
    "unclear",
]

BUSINESS_AREAS = [
    "manufacturing_operations",
    "materials_research",
    "it_and_data",
    "supply_chain",
    "customer_offerings",
    "sales_marketing_customer",
    "corporate_functions",
    "unclear",
]

SCORE_VALUES = [0, 1, 2]
SCORE_VALUE_STRINGS = ["0", "1", "2"]

SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string"},
        "primary_category": {"type": "string", "enum": PRIMARY_CATEGORIES},
        "business_area": {"type": "string", "enum": BUSINESS_AREAS},
        "technical_relevance_score": {"type": "string", "enum": SCORE_VALUE_STRINGS},
        "technical_signals": {"type": "array", "items": {"type": "string"}},
        "investment_signal": {"type": "string", "nullable": True},
    },
    "required": [
        "job_id",
        "primary_category",
        "business_area",
        "technical_relevance_score",
        "technical_signals",
        "investment_signal",
    ],
    "additionalProperties": False,
}


PrimaryCategory = Literal[
    "bi_analytics",
    "data_engineering",
    "cloud_platforms",
    "software_engineering",
    "enterprise_systems",
    "ai_ml_modeling",
    "llm_genai_applications",
    "agentic_ai_systems",
    "industrial_automation",
    "materials_science_rd",
    "non_technical",
    "unclear",
]

BusinessArea = Literal[
    "manufacturing_operations",
    "materials_research",
    "it_and_data",
    "supply_chain",
    "customer_offerings",
    "sales_marketing_customer",
    "corporate_functions",
    "unclear",
]

TechnicalRelevanceScore = Literal["0", "1", "2"]


class JobClassification(BaseModel):
    """Expected structured output from Gemini."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    primary_category: PrimaryCategory
    business_area: BusinessArea
    technical_relevance_score: TechnicalRelevanceScore
    technical_signals: list[str]
    investment_signal: Optional[str] = None

SYSTEM_PROMPT = dedent("""
You are classifying job postings for an academic study of AI, digital, automation, and technical capability.

You will read the full row context:
- job_id
- company_name
- job_title
- seniority
- job_country_code
- year
- month
- job_description

Return exactly one JSON object matching the schema below.
Do not add commentary. Do not wrap the JSON in markdown.

Schema:
- job_id: string
- primary_category: one of [bi_analytics, data_engineering, cloud_platforms, software_engineering, enterprise_systems, ai_ml_modeling, llm_genai_applications, agentic_ai_systems, industrial_automation, materials_science_rd, non_technical, unclear]
- business_area: one of [manufacturing_operations, materials_research, it_and_data, supply_chain, customer_offerings, sales_marketing_customer, corporate_functions, unclear]
- technical_relevance_score: one of ["0", "1", "2"] as a string
- technical_signals: list of explicit technical signals only
- investment_signal: one sentence or null

Primary category definitions:
- bi_analytics: dashboards, reporting, KPI tracking, descriptive analytics, business analysis, visualization, insight generation, analytical decision support
- data_engineering: data pipelines, ETL, orchestration, warehouse engineering, data modeling, Spark, Databricks, scalable data foundations
- cloud_platforms: cloud infrastructure, DevOps, platform operations, database administration, Kubernetes, reliability engineering, SAP Basis, internal technical backbone, platform support
- software_engineering: backend, frontend, full stack, APIs, microservices, mobile, application development, product engineering not mainly centered on data or AI
- enterprise_systems: implementation, configuration, support, rollout, or integration of ERP, CRM, HR, finance, workflow, or other internal business systems
- ai_ml_modeling: predictive modeling, optimization, computer vision, machine learning, deep learning, statistical learning, applied AI, MLOps, or similar data driven AI methods
- llm_genai_applications: LLM applications, chatbots, copilots, RAG, prompt engineering, conversational AI, GenAI evaluation, LLM powered interfaces, but not clearly autonomous agent systems
- agentic_ai_systems: autonomous or semi autonomous agent systems, multi agent architectures, planning, tool use, orchestration, MCP, LangGraph, workflow agents, reasoning agents
- industrial_automation: PLC, SCADA, MES, robotics, controls, instrumentation, plant automation, industrial control systems
- materials_science_rd: materials research, mechanics, simulation, finite element analysis, scientific computing, product physics modeling, digital twin for product or process understanding
- non_technical: no meaningful technical or digital capability signal
- unclear: not enough evidence

Important distinctions:
- Do not classify a role as ai_ml_modeling only because it mentions modeling, simulation, prediction, optimization, or data science.
- If the posting is mainly about finite element analysis, scientific simulation, tire mechanics, product physics, or numerical modeling, use materials_science_rd unless AI or ML methods are explicit and central.
- Physics based, mechanics based, and finite element based modeling are not the same as AI or ML modeling.
- Use enterprise_systems for business system support or implementation.
- Use cloud_platforms for infrastructure and technical backbone work.
- Use bi_analytics for reporting and dashboard work.

Business area definitions:
- manufacturing_operations: plant, production, maintenance, quality, industrialization, process operations, factory systems, site operations
- materials_research: R and D centers, labs, product design, product performance, durability, formulation, simulation supporting product development
- it_and_data: central IT, data teams, cloud teams, enterprise digital platforms, analytics platforms, cybersecurity, internal shared digital systems
- supply_chain: sourcing, planning, logistics, inventory, procurement, fulfillment
- customer_offerings: products or services delivered to customers, including connected products and customer facing digital services
- sales_marketing_customer: marketing, CRM, pricing, commercial analytics, customer acquisition, customer insights, sales support, customer engagement
- corporate_functions: finance, HR, legal, communications, administration, shared internal business functions
- unclear: not enough evidence

Scoring:
- 2 = AI, ML, LLM, agentic AI, MLOps, or similar advanced AI capability is central and explicit in the role
- 1 = AI, ML, LLM, or related tools are explicitly mentioned but are not the core of the role
- 0 = no meaningful AI, ML, LLM, agentic AI, MLOps, or related capability signal

Important scoring rules:
- The score measures AI centrality, not general technical sophistication.
- Do not assign score 1 or 2 only because the role is technical.
- Scientific simulation, HPC, software engineering, industrial automation, ERP work, cloud work, and materials modeling should receive score 0 unless AI or ML methods are explicitly present.
- Prefer score 0 over score 1 when AI is weakly implied.

Rules for technical_signals:
- Keep only explicit technical signals relevant to software, data, cloud, AI, enterprise systems, industrial technology, or scientific simulation.
- Prefer specific tools, frameworks, platforms, methods, architectures, or industrial systems.
- Exclude broad business activities and soft skills.

Rules for investment_signal:
- Generate it only when technical_relevance_score = 2.
- Otherwise return null.
- Use one sentence with this structure when possible: Investment in [specific AI capability] for [application context] to [objective].
- Do not summarize the whole role.

Decision rules:
- Choose exactly one primary_category.
- Choose exactly one business_area.
- The posting may be written in any language. Classify based on meaning.
- Prefer unclear over over claiming.
- Return JSON only.
""").strip()



# ===== Notebook cell 11 =====

VERTEX_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not VERTEX_PROJECT_ID:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set. Copy .env.example to .env and set your own project id.")
VERTEX_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", VERTEX_PROJECT_ID)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", VERTEX_LOCATION)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


def build_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_LOCATION,
    )


client = build_client()
print(f"Vertex AI client ready | location={VERTEX_LOCATION} | model={MODEL}")



# ===== Notebook cell 13 =====

def nice(text: str) -> str:
    return str(text).replace("_", " ")


def nice_label(text: str, width: int = 16) -> str:
    return "\n".join(wrap(nice(text).title(), width=width))


def is_missing_scalar(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def clean_text(value) -> str:
    return "" if is_missing_scalar(value) else str(value).strip()


def parse_list_cell(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if is_missing_scalar(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                pass
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def extract_json_object(text: str) -> str:
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output")
    return text[start:end + 1]


def find_source_workbook(company_dir: Path) -> Path:
    files = [
        path for path in sorted(company_dir.glob("*.xlsx"))
        if not path.name.startswith("~$") and not path.stem.endswith("_enriched")
    ]
    if not files:
        enriched_only = [
            path for path in sorted(company_dir.glob("*.xlsx"))
            if not path.name.startswith("~$")
        ]
        if enriched_only:
            return enriched_only[0]
        raise FileNotFoundError(f"No .xlsx file found in {company_dir}")
    return files[0]


def get_output_workbook_path(source_path: Path) -> Path:
    if source_path.stem.endswith("_enriched"):
        return source_path
    return source_path.with_name(f"{source_path.stem}_enriched.xlsx")


def normalize_jobs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    jobs = df.copy()
    jobs.columns = (
        jobs.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.lower()
    )

    drop_cols = [
        "salary",
        "hiring_manager_full_name",
        "hiring_manager_first_name",
        "hiring_manager_role",
        "hiring_manager_linkedin_url",
        "company_url",
        "company_linkedin_url",
        "company_industry",
        "company_employee_count",
        "company_revenue_usd",
        "company_seo_description",
        "company_description",
        "company_city",
        "job_location",
        "is_remote",
        "url",
        "employment_status",
    ]
    jobs = jobs.drop(columns=[col for col in drop_cols if col in jobs.columns], errors="ignore")

    if "posted_date" in jobs.columns:
        jobs["posted_date"] = pd.to_datetime(jobs["posted_date"], errors="coerce")
        jobs["year"] = jobs.get("year", jobs["posted_date"].dt.year)
        jobs["month"] = jobs.get("month", jobs["posted_date"].dt.month)

    if "job_id" not in jobs.columns:
        jobs.insert(0, "job_id", range(1, len(jobs) + 1))
    jobs["job_id"] = jobs["job_id"].astype(str)

    subset_cols = [col for col in jobs.columns if col != "job_id"]
    jobs = jobs.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)
    jobs = jobs.drop(columns=["job_id"], errors="ignore")
    jobs.insert(0, "job_id", range(1, len(jobs) + 1))
    jobs["job_id"] = jobs["job_id"].astype(str)

    output_columns = [
        "primary_category",
        "business_area",
        "technical_relevance_score",
        "technical_signals",
        "investment_signal",
        "llm_status",
        "llm_error",
    ]
    for column in output_columns:
        if column not in jobs.columns:
            jobs[column] = pd.NA
        # Excel often loads empty output columns as float64. Force object so string labels can be assigned safely.
        jobs[column] = jobs[column].astype("object")

    if "technical_signals" in jobs.columns and "skills_signals" in jobs.columns:
        needs_fill = jobs["technical_signals"].isna() & jobs["skills_signals"].notna()
        jobs.loc[needs_fill, "technical_signals"] = jobs.loc[needs_fill, "skills_signals"]
    elif "skills_signals" in jobs.columns and "technical_signals" not in jobs.columns:
        jobs["technical_signals"] = jobs["skills_signals"]

    return jobs


def load_jobs_for_processing(company_dir: Path) -> tuple[Path, Path, pd.DataFrame]:
    source_path = find_source_workbook(company_dir)
    output_path = get_output_workbook_path(source_path)
    load_path = output_path if output_path.exists() else source_path
    jobs = pd.read_excel(load_path)
    jobs = normalize_jobs_dataframe(jobs)
    return source_path, output_path, jobs



# ===== Notebook cell 15 =====

PRIMARY_ENUM = set(PRIMARY_CATEGORIES)
BUSINESS_ENUM = set(BUSINESS_AREAS)
SCORE_ENUM = set(SCORE_VALUES)


def safe_strip(value):
    return value.strip() if isinstance(value, str) else value


def coerce_score(value):
    if is_missing_scalar(value):
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def is_json_list_cell(value) -> bool:
    if isinstance(value, list):
        return True
    if is_missing_scalar(value):
        return False
    if isinstance(value, str):
        try:
            return isinstance(json.loads(value), list)
        except Exception:
            return False
    return False


def validate_result(result, row: pd.Series) -> dict:
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    elif isinstance(result, str):
        result = json.loads(extract_json_object(result))

    if not isinstance(result, dict):
        raise ValueError(f"Result is not a dict: {type(result)}")

    if "technical_signals" not in result and "skills_signals" in result:
        result["technical_signals"] = result.pop("skills_signals")

    required_keys = {
        "job_id",
        "primary_category",
        "business_area",
        "technical_relevance_score",
        "technical_signals",
        "investment_signal",
    }
    missing = required_keys - set(result.keys())
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")

    result["job_id"] = str(result["job_id"]).strip()
    result["primary_category"] = safe_strip(result["primary_category"])
    result["business_area"] = safe_strip(result["business_area"])
    result["technical_relevance_score"] = coerce_score(result["technical_relevance_score"])

    if result["primary_category"] not in PRIMARY_ENUM:
        raise ValueError(f"Invalid primary_category: {result['primary_category']}")
    if result["business_area"] not in BUSINESS_ENUM:
        raise ValueError(f"Invalid business_area: {result['business_area']}")
    if result["technical_relevance_score"] not in SCORE_ENUM:
        raise ValueError(
            f"Invalid technical_relevance_score: {result['technical_relevance_score']}"
        )

    if not isinstance(result["technical_signals"], list):
        raise ValueError("technical_signals must be a list")

    result["technical_signals"] = [
        str(item).strip()
        for item in result["technical_signals"]
        if item is not None and str(item).strip()
    ]

    if result["technical_relevance_score"] != 2:
        result["investment_signal"] = None
    elif result["investment_signal"] is not None:
        result["investment_signal"] = str(result["investment_signal"]).strip() or None

    row_job_id = str(row["job_id"]).strip()
    if result["job_id"] != row_job_id:
        raise ValueError(f"job_id mismatch: result={result['job_id']} row={row_job_id}")

    return result


def row_needs_reprocessing(row: pd.Series) -> bool:
    primary_ok = pd.notna(row["primary_category"]) and row["primary_category"] in PRIMARY_ENUM
    business_ok = pd.notna(row["business_area"]) and row["business_area"] in BUSINESS_ENUM
    score_value = coerce_score(row["technical_relevance_score"])
    score_ok = score_value in SCORE_ENUM
    signals_ok = is_json_list_cell(row["technical_signals"])

    investment_value = row["investment_signal"]
    score_is_two = score_value == 2

    investment_ok = (
        score_is_two
        or is_missing_scalar(investment_value)
        or str(investment_value).strip() == ""
    )

    return not (primary_ok and business_ok and score_ok and signals_ok and investment_ok)


def classify_row(row: pd.Series) -> dict:
    payload = {
        "job_id": clean_text(row.get("job_id", "")),
        "company_name": clean_text(row.get("company_name", "")),
        "job_title": clean_text(row.get("job_title", "")),
        "seniority": clean_text(row.get("seniority", "")),
        "job_country_code": clean_text(row.get("job_country_code", "")),
        "year": None if is_missing_scalar(row.get("year")) else int(float(row.get("year"))),
        "month": None if is_missing_scalar(row.get("month")) else int(float(row.get("month"))),
        "job_description": clean_text(row.get("job_description", "")),
    }

    if RATE_LIMITER is not None:
        RATE_LIMITER.wait()

    response = client.models.generate_content(
        model=MODEL,
        contents=json.dumps(payload, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            response_mime_type="application/json",
            response_schema=JobClassification,
        ),
    )

    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed.model_dump() if hasattr(parsed, "model_dump") else parsed

    content = response.text

    return json.loads(extract_json_object(content))


def process_jobs_dataframe(jobs: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    pending_idx = jobs.index[jobs.apply(row_needs_reprocessing, axis=1)].tolist()
    print(f"Rows to process: {len(pending_idx)}")

    def process_one(idx: int, stats: BatchStats):
        row = jobs.loc[idx]
        time.sleep(random.uniform(0.2, 1.5))
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                result = validate_result(classify_row(row), row)
                return idx, result, None
            except Exception as exc:
                last_error = str(exc)
                if attempt >= MAX_RETRIES:
                    stats.record_final_error()
                    break
                if is_retryable_error(exc):
                    stats.record_retryable(exc)
                    sleep_for_retry(exc, attempt)
                else:
                    stats.record_non_retryable(exc)
                    time.sleep(2.0 * (attempt + 1))

        return idx, None, last_error

    if not pending_idx:
        return jobs

    total_batches = (len(pending_idx) - 1) // BATCH_SIZE + 1

    for start in range(0, len(pending_idx), BATCH_SIZE):
        batch_idx = pending_idx[start:start + BATCH_SIZE]
        batch_number = start // BATCH_SIZE + 1
        results = []

        print(f"\nRunning batch {batch_number} / {total_batches}")
        print(f"Rows in this batch: {len(batch_idx)}")
        print(f"Current tuning: workers={CURRENT_WORKERS}, calls/min={CURRENT_CALLS_PER_MINUTE}")
        batch_stats = BatchStats()
        batch_started_at = time.monotonic()

        with ThreadPoolExecutor(max_workers=CURRENT_WORKERS) as executor:
            futures = {executor.submit(process_one, idx, batch_stats): idx for idx in batch_idx}
            for future in tqdm(as_completed(futures), total=len(futures)):
                results.append(future.result())

        batch_elapsed = time.monotonic() - batch_started_at

        for idx, result, error in results:
            if error is None:
                jobs.at[idx, "primary_category"] = result["primary_category"]
                jobs.at[idx, "business_area"] = result["business_area"]
                jobs.at[idx, "technical_relevance_score"] = result["technical_relevance_score"]
                jobs.at[idx, "technical_signals"] = json.dumps(
                    result["technical_signals"],
                    ensure_ascii=False,
                )
                jobs.at[idx, "investment_signal"] = result["investment_signal"]
                jobs.at[idx, "llm_status"] = "ok"
                jobs.at[idx, "llm_error"] = pd.NA
            else:
                jobs.at[idx, "llm_status"] = "error"
                jobs.at[idx, "llm_error"] = error

        ok_count = sum(1 for _, _, error in results if error is None)
        tune_after_batch(batch_number, len(batch_idx), ok_count, batch_stats, batch_elapsed)

        if SAVE_EVERY_BATCH:
            jobs.to_excel(output_path, index=False)
            print(f"Saved after batch {batch_number} to {output_path}")

        if batch_number < total_batches and SLEEP_BETWEEN_BATCHES > 0:
            print(f"Sleeping {SLEEP_BETWEEN_BATCHES}s before next batch")
            time.sleep(SLEEP_BETWEEN_BATCHES)

    jobs.to_excel(output_path, index=False)
    return jobs



# ===== Notebook cell 17 =====


AI_FAMILIES = [
    "ai_ml_modeling",
    "llm_genai_applications",
    "agentic_ai_systems",
]

AI_FAMILY_LABELS = {
    "ai_ml_modeling": "AI / ML modeling",
    "llm_genai_applications": "LLM / GenAI apps",
    "agentic_ai_systems": "Agentic AI",
}

AI_FAMILY_COLORS = {
    "ai_ml_modeling": "#BFD7EA",
    "llm_genai_applications": "#86AED3",
    "agentic_ai_systems": "#2F6FAE",
}


def prepare_analysis_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    analysis_df = df.copy()

    if "technical_signals" not in analysis_df.columns and "skills_signals" in analysis_df.columns:
        analysis_df["technical_signals"] = analysis_df["skills_signals"]

    analysis_df["technical_relevance_score"] = pd.to_numeric(
        analysis_df["technical_relevance_score"], errors="coerce"
    )

    if "year" in analysis_df.columns:
        analysis_df["year"] = pd.to_numeric(analysis_df["year"], errors="coerce")
    if "month" in analysis_df.columns:
        analysis_df["month"] = pd.to_numeric(analysis_df["month"], errors="coerce")

    if "job_country_code" in analysis_df.columns:
        analysis_df["job_country_code"] = (
            analysis_df["job_country_code"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
            .str.upper()
        )

    if "technical_signals" in analysis_df.columns:
        analysis_df["_signals_list"] = analysis_df["technical_signals"].apply(parse_list_cell)
    else:
        analysis_df["_signals_list"] = [[] for _ in range(len(analysis_df))]

    return analysis_df


def empty_figure(message: str, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
    fig.patch.set_facecolor("white")
    return fig


def save_report_pdf(fig, title: str, note: str, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title_style",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#17324D"),
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "body_style",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#222222"),
        spaceAfter=8,
    )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=pdf_path.parent) as tmp:
        temp_png_path = Path(tmp.name)

    fig.savefig(temp_png_path, dpi=220, bbox_inches="tight", facecolor="white")

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )

    img = Image(str(temp_png_path))
    img._restrictSize(A4[0] - doc.leftMargin - doc.rightMargin, 13.0 * cm)

    paragraphs = []
    for block in str(note).strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        block = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        block = block.replace("\n", "<br/>")
        paragraphs.append(Paragraph(block, body_style))

    story = [Paragraph(title, title_style), Spacer(1, 0.2 * cm), img, Spacer(1, 0.45 * cm)]
    story.extend(paragraphs)
    doc.build(story)

    temp_png_path.unlink(missing_ok=True)


def select_until_coverage(series: pd.Series, coverage: float = 0.95) -> pd.Series:
    ranked = series.sort_values(ascending=False)
    if ranked.empty or ranked.sum() == 0:
        return ranked

    cumulative = ranked.cumsum() / ranked.sum()
    keep = cumulative <= coverage
    if keep.sum() == 0:
        keep.iloc[0] = True
    else:
        first_excluded = keep[keep == False].index[:1]
        if len(first_excluded) > 0:
            keep.loc[first_excluded[0]] = True
    return ranked[keep].sort_values(ascending=True)



# ===== Notebook cell 19 =====


def build_breadth_report(df: pd.DataFrame):
    score2 = df.loc[df["technical_relevance_score"] == 2].copy()
    title = "AI breadth across capability pockets"
    pdf_name = "ai_breadth_across_capability_pockets.pdf"

    if score2.empty:
        note = "No score-2 AI roles are available, so the breadth matrix cannot be built yet."
        return empty_figure(note, figsize=(12, 6)), title, note, pdf_name

    matrix = score2.groupby(["primary_category", "business_area"]).size().unstack(fill_value=0)
    if matrix.empty:
        note = "The score-2 subset is empty after grouping, so the breadth matrix cannot be built."
        return empty_figure(note, figsize=(12, 6)), title, note, pdf_name

    x = np.arange(len(matrix.columns))
    y = np.arange(len(matrix.index))
    x_labels = [nice_label(col, 18) for col in matrix.columns]
    y_labels = [nice_label(row, 18) for row in matrix.index]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for xi in x:
        ax.axvline(xi, color="#EAECEF", lw=1, zorder=0)
    for yi in y:
        ax.axhline(yi, color="#EAECEF", lw=1, zorder=0)

    max_value = matrix.to_numpy().max() if matrix.to_numpy().size else 1
    max_value = max(max_value, 1)

    for i, row_name in enumerate(matrix.index):
        for j, col_name in enumerate(matrix.columns):
            value = matrix.loc[row_name, col_name]
            if value > 0:
                size = 900 * (value / max_value) + 250
                ax.scatter(j, i, s=size, color="#A8C7E6", edgecolor="#6F9FCB", linewidth=1.5, alpha=0.95, zorder=3)
                ax.text(j, i, str(value), ha="center", va="center", fontsize=13, fontweight="bold", color="#17324D", zorder=4)
            else:
                ax.scatter(j, i, s=60, color="#F5F7FA", edgecolor="#D9E1E8", linewidth=0.8, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, pad=10)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    total_roles = int(matrix.values.sum())
    category_counts = matrix.sum(axis=1).sort_values(ascending=False)
    area_counts = matrix.sum(axis=0).sort_values(ascending=False)
    pair_counts = matrix.stack().reset_index(name="count")
    pair_counts = pair_counts[pair_counts["count"] > 0].sort_values("count", ascending=False)

    top_category_lines = [
        f"- {nice(idx)}: {int(val)} roles ({val / total_roles:.1%} of score-2 roles)"
        for idx, val in category_counts.head(5).items()
    ]
    top_area_lines = [
        f"- {nice(idx)}: {int(val)} roles ({val / total_roles:.1%} of score-2 roles)"
        for idx, val in area_counts.head(5).items()
    ]
    top_pair_lines = [
        f"- {nice(row.primary_category)} × {nice(row.business_area)}: {int(row['count'])} roles ({row['count'] / total_roles:.1%} of score-2 roles)"
        for _, row in pair_counts.head(8).iterrows()
    ]

    lead_pair = pair_counts.iloc[0]
    note = f"""
Score-2 role distribution note

This section refers to the bubble matrix built from postings with technical_relevance_score = 2.

The chart maps the count of AI-core roles across:
- primary_category on the y-axis
- business_area on the x-axis

Each bubble shows the number of postings in one category-area intersection. Larger bubbles mean more score-2 roles in that pocket.

Total score-2 roles in this matrix: {total_roles}

The largest primary categories are:
{chr(10).join(top_category_lines)}

The largest business areas are:
{chr(10).join(top_area_lines)}

The most important intersections in the matrix are:
{chr(10).join(top_pair_lines)}

Deterministic reading of the matrix:
- The score-2 signal is concentrated rather than evenly spread.
- The largest concentration sits in {nice(lead_pair['primary_category'])} × {nice(lead_pair['business_area'])}.
- The matrix shows where the company is hiring AI-core talent by combining capability type and business application area.
- This chart should be read as a distribution of AI-core hiring pockets, not as a direct measure of spending.
""".strip()

    return fig, title, note, pdf_name



# ===== Notebook cell 21 =====

def build_skills_seniority_report(df: pd.DataFrame):
    title = "AI skills and seniority"
    pdf_name = "ai_skills_and_seniority.pdf"
    score2 = df.loc[df["technical_relevance_score"] == 2].copy()

    if score2.empty:
        note = "No score-2 AI roles are available, so the skills and seniority view cannot be built yet."
        return empty_figure(note, figsize=(14, 8)), title, note, pdf_name

    score2["_seniority_bucket"] = (
        score2.get("seniority", pd.Series(index=score2.index))
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )

    area_total = df["business_area"].value_counts()
    area_score2_seniority = (
        score2.groupby(["business_area", "_seniority_bucket"])
        .size()
        .unstack(fill_value=0)
    )

    share_pct = area_score2_seniority.div(area_total, axis=0).fillna(0) * 100
    share_pct["total"] = share_pct.sum(axis=1)
    share_pct = share_pct.sort_values("total", ascending=True)

    seniority_cols = [col for col in share_pct.columns if col != "total"]
    palette = ["#DCECC9", "#A9D18E", "#6AA84F", "#B7C9E2", "#7EA6D8", "#4F81BD", "#D9D2E9", "#B4A7D6"]
    color_map = {col: palette[i % len(palette)] for i, col in enumerate(seniority_cols)}

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(
        2, 2,
        width_ratios=[1.05, 1.15],
        height_ratios=[1, 1],
        wspace=0.28,
        hspace=0.28
    )
    left_grid = grid[:, 0].subgridspec(3, 1, hspace=0.35)

    family_titles = {
        "ai_ml_modeling": "AI / ML modeling skills",
        "llm_genai_applications": "LLM / GenAI skills",
        "agentic_ai_systems": "Agentic AI skills",
    }
    family_colors = {
        "ai_ml_modeling": ("#D9EAF7", "#8FB6D8"),
        "llm_genai_applications": ("#CFE2F3", "#6FA8DC"),
        "agentic_ai_systems": ("#B6D7F0", "#3D85C6"),
    }

    for i, family in enumerate(AI_FAMILIES):
        ax = fig.add_subplot(left_grid[i, 0])

        top_signals = (
            score2.loc[score2["primary_category"] == family, "_signals_list"]
            .explode()
            .dropna()
            .value_counts()
            .head(8)
            .sort_values(ascending=True)
        )

        fill_color, edge_color = family_colors[family]

        if not top_signals.empty:
            ax.barh(
                top_signals.index,
                top_signals.values,
                color=fill_color,
                edgecolor=edge_color,
                height=0.6,
            )
            for j, value in enumerate(top_signals.values):
                ax.text(
                    value + 0.15,
                    j,
                    str(int(value)),
                    va="center",
                    ha="left",
                    fontsize=10,
                    color="#17324D",
                    fontweight="bold",
                )

        ax.set_title(family_titles[family], loc="left", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=10, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_right = fig.add_subplot(grid[:, 1])
    left = pd.Series(0, index=share_pct.index)

    for bucket in seniority_cols:
        ax_right.barh(
            share_pct.index.map(lambda value: nice(value).title()),
            share_pct[bucket].values,
            left=left.values,
            color=color_map[bucket],
            edgecolor="white",
            height=0.65,
            label=nice(bucket).title(),
        )
        left += share_pct[bucket]

    for i, total in enumerate(share_pct["total"].values):
        ax_right.text(
            total + 0.2,
            i,
            f"{total:.0f}%",
            va="center",
            ha="left",
            fontsize=11,
            color="#24411A",
            fontweight="bold",
        )

    ax_right.set_title("AI roles share by area and seniority", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax_right.set_xlabel("")
    ax_right.set_ylabel("")
    ax_right.set_xticks([])
    ax_right.tick_params(axis="y", labelsize=11, length=0)

    if seniority_cols:
        ax_right.legend(frameon=False, loc="lower right")

    for spine in ax_right.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    family_lines = []
    family_counts = score2["primary_category"].value_counts()
    for family, value in family_counts.head(6).items():
        family_lines.append(f"- {nice(family)}: {int(value)} roles ({value / len(score2):.1%} of score-2 roles)")

    signal_lines = []
    for family in AI_FAMILIES:
        top_signals = (
            score2.loc[score2["primary_category"] == family, "_signals_list"]
            .explode()
            .dropna()
            .value_counts()
            .head(8)
        )
        if not top_signals.empty:
            block = [f"- {signal}: {int(value)} mentions" for signal, value in top_signals.items()]
            signal_lines.append(f"{AI_FAMILY_LABELS[family]} skills:\n" + "\n".join(block))

    top_areas = share_pct.sort_values("total", ascending=False).head(6)

    area_lines = []
    for area, row in top_areas.iterrows():
        total_share = row["total"]
        seniority_breakdown = row.drop(labels=["total"])
        seniority_breakdown = seniority_breakdown[seniority_breakdown > 0].sort_values(ascending=False)
        seniority_text = ", ".join(
            [f"{nice(bucket)} {value:.1f}%" for bucket, value in seniority_breakdown.items()]
        )
        area_lines.append(
            f"- {nice(area)}: {total_share:.1f}% of postings in this area are score-2 AI roles; "
            f"seniority split within that share: {seniority_text}"
        )

    note = f"""
Score-2 skills and area-seniority note

This section refers to the chart composed of:
- three left panels showing the most frequent technical signals inside the main AI role families
- one right panel showing the share of score-2 AI roles within each business area, split by seniority

Total score-2 roles used in this chart: {len(score2)}

The main score-2 role families are:
{chr(10).join(family_lines)}

The most frequent technical signals by AI family are:
{chr(10).join(signal_lines) if signal_lines else '- No recurring technical signals were found.'}

The business areas with the highest concentration of score-2 AI roles are:
{chr(10).join(area_lines) if area_lines else '- No area-level concentration could be computed.'}

Deterministic reading of the chart:
- The left panels show which technical signals characterize each AI role family rather than the whole score-2 population combined.
- The right panel should be read as a share of postings within each business area in this dataset, not as a share of employees.
- The strongest area-level concentration of score-2 AI roles sits in the business areas at the top of the ranking above.
- The seniority split shows whether AI hiring in each area is concentrated more in junior, mid-level, senior, or staff roles.
""".strip()

    return fig, title, note, pdf_name



# ===== Notebook cell 23 =====


def build_trend_report(df: pd.DataFrame):
    title = "AI hiring trend"
    pdf_name = "ai_hiring_trend.pdf"

    trend_df = df[df["technical_relevance_score"].isin([0, 1, 2])].copy()
    if trend_df.empty or "year" not in trend_df.columns or "month" not in trend_df.columns:
        note = "The dataset does not contain enough year and month information to build the trend report."
        return empty_figure(note, figsize=(14, 9)), title, note, pdf_name

    trend_df["time_period_month"] = pd.to_datetime(
        dict(year=trend_df["year"], month=trend_df["month"], day=1),
        errors="coerce",
    )
    trend_df = trend_df.dropna(subset=["time_period_month", "year"])
    if trend_df.empty:
        note = "All year and month values are missing after cleaning, so the trend report cannot be built."
        return empty_figure(note, figsize=(14, 9)), title, note, pdf_name

    monthly_trend = trend_df.groupby(["time_period_month", "technical_relevance_score"]).size().unstack(fill_value=0).sort_index()
    yearly_total = trend_df.groupby("year").size().sort_index()
    yearly_ai = trend_df.loc[trend_df["technical_relevance_score"] == 2].groupby("year").size().sort_index()
    yearly_ai_share = ((yearly_ai / yearly_total) * 100).fillna(0).sort_index()

    ai_df = trend_df.loc[trend_df["technical_relevance_score"] == 2].copy()
    ai_mix = (
        ai_df[ai_df["primary_category"].isin(AI_FAMILIES)]
        .groupby(["year", "primary_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=AI_FAMILIES, fill_value=0)
        .sort_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), gridspec_kw={"width_ratios": [1.1, 1], "height_ratios": [1, 1]})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, wspace=0.22, hspace=0.35)

    if 0 in monthly_trend.columns:
        axes[0, 0].bar(monthly_trend.index, monthly_trend[0], width=20, color="#BFD7EA", edgecolor="#86AED3")
    axes[0, 0].set_title("No AI roles by month", loc="left", fontsize=15, fontweight="bold", pad=10)
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Number of job postings")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].set_axisbelow(True)
    for spine in axes[0, 0].spines.values():
        spine.set_visible(False)

    bottom = pd.Series(0, index=monthly_trend.index)
    monthly_labels = {1: "AI-literate roles", 2: "AI roles"}
    monthly_colors = {1: "#F4B183", 2: "#C55A11"}
    for score in [1, 2]:
        if score in monthly_trend.columns:
            axes[0, 1].bar(
                monthly_trend.index,
                monthly_trend[score],
                bottom=bottom,
                width=20,
                label=monthly_labels[score],
                color=monthly_colors[score],
                edgecolor="white",
            )
            bottom += monthly_trend[score]
    axes[0, 1].set_title("AI-related roles by month", loc="left", fontsize=15, fontweight="bold", pad=10)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Number of job postings")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].set_axisbelow(True)
    axes[0, 1].legend(frameon=False, loc="upper left")
    for spine in axes[0, 1].spines.values():
        spine.set_visible(False)

    axes[1, 0].bar(yearly_ai_share.index.astype(int).astype(str), yearly_ai_share.values, color="#C55A11", edgecolor="white", width=0.62)
    for i, value in enumerate(yearly_ai_share.values):
        axes[1, 0].text(i, value + 0.15, f"{value:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#7A3000")
    axes[1, 0].set_title("AI roles share by year", loc="left", fontsize=15, fontweight="bold", pad=10)
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("% of total job postings")
    axes[1, 0].set_ylim(0, max(yearly_ai_share.max() * 1.18, 1))
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].set_axisbelow(True)
    axes[1, 0].margins(x=0.08)
    for spine in axes[1, 0].spines.values():
        spine.set_visible(False)

    bottom = pd.Series(0, index=ai_mix.index)
    for family in ai_mix.columns:
        axes[1, 1].bar(
            ai_mix.index.astype(int).astype(str),
            ai_mix[family].values,
            bottom=bottom.values,
            color=AI_FAMILY_COLORS[family],
            edgecolor="white",
            width=0.62,
            label=AI_FAMILY_LABELS[family],
        )
        bottom += ai_mix[family]
    axes[1, 1].set_title("AI roles mix by year", loc="left", fontsize=15, fontweight="bold", pad=10)
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Number of AI roles")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].set_axisbelow(True)
    axes[1, 1].legend(frameon=False, loc="upper left")
    axes[1, 1].margins(x=0.08)
    for spine in axes[1, 1].spines.values():
        spine.set_visible(False)

    def peak_line(series: pd.Series, label: str) -> str:
        if series.empty or series.max() == 0:
            return f"- {label}: no postings in this series"
        peak_date = series.idxmax()
        peak_value = int(series.max())
        return f"- {label}: peak of {peak_value} postings in {peak_date.strftime('%Y-%m')}"

    monthly_lines = []
    if 0 in monthly_trend.columns:
        monthly_lines.append(peak_line(monthly_trend[0], "No AI roles"))
    if 1 in monthly_trend.columns:
        monthly_lines.append(peak_line(monthly_trend[1], "AI-literate roles"))
    if 2 in monthly_trend.columns:
        monthly_lines.append(peak_line(monthly_trend[2], "AI roles"))

    share_lines = [
        f"- {int(year)}: {value:.1f}% of total postings are AI roles"
        for year, value in yearly_ai_share.items()
    ]

    mix_lines = []
    for year, row in ai_mix.iterrows():
        total = int(row.sum())
        if total == 0:
            continue
        dominant_family = row.idxmax()
        dominant_value = int(row.max())
        parts = [f"{AI_FAMILY_LABELS[family]} {int(row[family])}" for family in ai_mix.columns if int(row[family]) > 0]
        mix_lines.append(
            f"- {int(year)}: {', '.join(parts)}; dominant family = {AI_FAMILY_LABELS[dominant_family]} ({dominant_value} roles)"
        )

    n_no_ai = int((trend_df["technical_relevance_score"] == 0).sum())
    n_ai_literate = int((trend_df["technical_relevance_score"] == 1).sum())
    n_ai = int((trend_df["technical_relevance_score"] == 2).sum())
    n_total = int(len(trend_df))

    note = f"""
AI hiring trend note

This section refers to the 2x2 trend chart built from postings classified with:
- technical_relevance_score = 0 for no AI signal
- technical_relevance_score = 1 for AI-literate or AI-adjacent roles
- technical_relevance_score = 2 for AI roles

Total postings included in this trend view: {n_total}
- No AI roles: {n_no_ai}
- AI-literate roles: {n_ai_literate}
- AI roles: {n_ai}

The four panels represent:
- top left: monthly count of roles with no AI signal
- top right: monthly count of AI-literate and AI roles
- bottom left: yearly share of AI roles over total postings
- bottom right: yearly mix of AI roles across AI / ML modeling, LLM / GenAI applications, and Agentic AI systems

Monthly peaks in hiring activity:
{chr(10).join(monthly_lines)}

Yearly AI-role share:
{chr(10).join(share_lines)}

Yearly mix of AI-role families:
{chr(10).join(mix_lines) if mix_lines else '- No score-2 AI family mix is available.'}

Deterministic reading of the chart:
- The top row separates the broad hiring base from the AI-related segment, so the chart can be read as a change in AI hiring intensity rather than only total hiring growth.
- The bottom-left panel shows whether AI roles become a larger or smaller share of the total hiring mix over time.
- The bottom-right panel shows which AI family dominates the AI-role mix in each year.
- This chart should be interpreted as a hiring-intensity and capability-mix view, not as a direct measure of investment spending.
""".strip()

    return fig, title, note, pdf_name



# ===== Notebook cell 25 =====


def build_geography_report(df: pd.DataFrame):
    title = "AI hiring geography"
    pdf_name = "ai_hiring_geography.pdf"

    if "job_country_code" not in df.columns:
        note = "The dataset does not contain job_country_code, so the geography report cannot be built."
        return empty_figure(note, figsize=(14, 9)), title, note, pdf_name

    score2 = df[df["technical_relevance_score"] == 2].copy()
    if score2.empty:
        note = "No score-2 AI roles are available, so the geography report cannot be built yet."
        return empty_figure(note, figsize=(14, 9)), title, note, pdf_name

    overall_country = df["job_country_code"].value_counts()
    ai_country = score2["job_country_code"].value_counts()

    country_summary = pd.DataFrame({"overall_hiring": overall_country, "ai_roles": ai_country}).fillna(0)
    country_summary["overall_hiring"] = country_summary["overall_hiring"].astype(int)
    country_summary["ai_roles"] = country_summary["ai_roles"].astype(int)
    country_summary["ai_intensity_pct"] = (country_summary["ai_roles"] / country_summary["overall_hiring"] * 100).fillna(0)

    top_overall = select_until_coverage(country_summary["overall_hiring"], coverage=0.95)
    top_ai = select_until_coverage(country_summary.loc[country_summary["ai_roles"] > 0, "ai_roles"], coverage=0.95)

    global_ai_rate = country_summary["ai_roles"].sum() / country_summary["overall_hiring"].sum()
    shrink_strength = max(1, int(country_summary["overall_hiring"].median()))
    country_summary["ai_intensity_smoothed_pct"] = (
        (country_summary["ai_roles"] + shrink_strength * global_ai_rate)
        / (country_summary["overall_hiring"] + shrink_strength)
        * 100
    )

    top_intensity_index = select_until_coverage(
        country_summary.loc[country_summary["ai_roles"] > 0, "ai_roles"],
        coverage=0.95,
    ).index
    top_intensity = country_summary.loc[top_intensity_index].sort_values("ai_intensity_smoothed_pct", ascending=True)

    mix_countries = top_ai.index.tolist()
    ai_mix_country = (
        score2[
            score2["primary_category"].isin(AI_FAMILIES)
            & score2["job_country_code"].isin(mix_countries)
        ]
        .groupby(["job_country_code", "primary_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=mix_countries, columns=AI_FAMILIES, fill_value=0)
    )
    ai_mix_country["total"] = ai_mix_country.sum(axis=1)
    ai_mix_country = ai_mix_country.sort_values("total", ascending=True)
    ai_mix_country_display = ai_mix_country.drop(columns="total")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1.05]})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, wspace=0.28, hspace=0.35)

    axes[0, 0].barh(top_overall.index, top_overall.values, color="#D9EAF7", edgecolor="#8FB6D8", height=0.6)
    for i, value in enumerate(top_overall.values):
        axes[0, 0].text(value + max(top_overall.max() * 0.01, 1), i, f"{int(value)}", va="center", ha="left", fontsize=10, color="#17324D", fontweight="bold")
    axes[0, 0].set_title("Top hiring countries", loc="left", fontsize=15, fontweight="bold", pad=10)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("")
    axes[0, 0].tick_params(axis="y", labelsize=11, length=0)
    for spine in axes[0, 0].spines.values():
        spine.set_visible(False)

    axes[0, 1].barh(top_ai.index, top_ai.values, color="#F4B183", edgecolor="#C55A11", height=0.6)
    for i, value in enumerate(top_ai.values):
        axes[0, 1].text(value + max(top_ai.max() * 0.02, 0.3), i, f"{int(value)}", va="center", ha="left", fontsize=10, color="#7A3000", fontweight="bold")
    axes[0, 1].set_title("Top countries for AI roles", loc="left", fontsize=15, fontweight="bold", pad=10)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")
    axes[0, 1].tick_params(axis="y", labelsize=11, length=0)
    for spine in axes[0, 1].spines.values():
        spine.set_visible(False)

    axes[1, 0].barh(top_intensity.index, top_intensity["ai_intensity_smoothed_pct"].values, color="#DCECC9", edgecolor="#6AA84F", height=0.6)
    for i, value in enumerate(top_intensity["ai_intensity_smoothed_pct"].values):
        axes[1, 0].text(value + max(top_intensity["ai_intensity_smoothed_pct"].max() * 0.02, 0.1), i, f"{value:.1f}%", va="center", ha="left", fontsize=10, color="#24411A", fontweight="bold")
    axes[1, 0].set_title("AI intensity by country", loc="left", fontsize=15, fontweight="bold", pad=10)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("")
    axes[1, 0].tick_params(axis="y", labelsize=11, length=0)
    for spine in axes[1, 0].spines.values():
        spine.set_visible(False)

    left = pd.Series(0, index=ai_mix_country_display.index)
    for family in ai_mix_country_display.columns:
        axes[1, 1].barh(
            ai_mix_country_display.index,
            ai_mix_country_display[family].values,
            left=left.values,
            color=AI_FAMILY_COLORS[family],
            edgecolor="white",
            height=0.62,
            label=AI_FAMILY_LABELS[family],
        )
        left += ai_mix_country_display[family]
    for i, total in enumerate(left.values):
        axes[1, 1].text(total + max(left.max() * 0.02, 0.2), i, f"{int(total)}", va="center", ha="left", fontsize=10, color="#17324D", fontweight="bold")
    axes[1, 1].set_title("AI role mix in top AI countries", loc="left", fontsize=15, fontweight="bold", pad=10)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("")
    axes[1, 1].tick_params(axis="y", labelsize=11, length=0)
    axes[1, 1].legend(frameon=False, loc="lower right")
    for spine in axes[1, 1].spines.values():
        spine.set_visible(False)

    overall_lines = [
        f"- {country}: {int(value)} postings ({value / len(df):.1%} of all postings)"
        for country, value in top_overall.sort_values(ascending=False).items()
    ]
    ai_lines = [
        f"- {country}: {int(value)} AI roles ({value / max(len(score2), 1):.1%} of score-2 roles)"
        for country, value in top_ai.sort_values(ascending=False).items()
    ]
    intensity_lines = [
        f"- {country}: {row['ai_intensity_smoothed_pct']:.1f}% AI intensity"
        for country, row in top_intensity.iterrows()
    ]

    mix_lines = []
    for country, row in ai_mix_country.sort_values("total", ascending=False).iterrows():
        total = int(row["total"])
        if total == 0:
            continue
        parts = [f"{AI_FAMILY_LABELS[family]} {int(row[family])}" for family in AI_FAMILIES if int(row[family]) > 0]
        dominant_family = row[AI_FAMILIES].idxmax()
        mix_lines.append(f"- {country}: {', '.join(parts)}; dominant family = {AI_FAMILY_LABELS[dominant_family]}")

    note = f"""
Geography note for AI hiring

This section refers to the 2x2 geography chart built from the posting dataset and the subset of postings with technical_relevance_score = 2.

The four panels represent:
- top left: countries with the largest overall hiring volume in the dataset
- top right: countries with the largest number of AI roles
- bottom left: countries with the highest AI intensity, defined as the share of AI roles within that country's posting base
- bottom right: mix of AI-role families in the main AI hiring countries

Total postings in the dataset: {len(df)}
Total AI roles used in the geography view: {len(score2)}

Overall hiring is concentrated in:
{chr(10).join(overall_lines)}

AI-role hiring is concentrated in:
{chr(10).join(ai_lines)}

The countries with the highest AI intensity are:
{chr(10).join(intensity_lines)}

Within the main AI hiring countries, the AI-role mix is:
{chr(10).join(mix_lines) if mix_lines else '- No country level AI family mix is available.'}

Deterministic reading of the chart:
- The top-left and top-right panels distinguish the broad hiring footprint from the narrower geography of AI-role hiring.
- The bottom-left panel should be read as an intensity measure, not just raw volume.
- A country can have large overall hiring but low AI intensity, or smaller overall hiring but higher AI concentration.
- The bottom-right panel shows whether the AI-role mix in each leading country is dominated by AI / ML modeling, LLM / GenAI applications, or Agentic AI systems.
- This chart should be interpreted as a geography-of-capability-building view, not as a direct measure of investment spending.
""".strip()

    return fig, title, note, pdf_name



# ===== Notebook cell 27 =====

def _escape_reportlab_text(text):
    text = "" if text is None else str(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _markdown_bold_to_reportlab(text):
    escaped = _escape_reportlab_text(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def save_text_report_pdf(report_text, pdf_path, title):
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MemoTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        spaceAfter=14,
    )
    body_style = ParagraphStyle(
        "MemoBody",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=15,
        spaceAfter=8,
    )

    story = [Paragraph(_escape_reportlab_text(title), title_style), Spacer(1, 0.2 * cm)]

    paragraphs = [p.strip() for p in str(report_text).split("\n\n") if p.strip()]
    for paragraph in paragraphs:
        story.append(Paragraph(_markdown_bold_to_reportlab(paragraph), body_style))
        story.append(Spacer(1, 0.15 * cm))

    doc.build(story)


def build_investment_signal_memo_report(df: pd.DataFrame):
    title = "AI investment signal report"
    pdf_name = "ai_investment_signal_report.pdf"

    score = pd.to_numeric(df.get("technical_relevance_score"), errors="coerce")
    memo_df = df.loc[score >= 2].copy()

    memo_df["investment_signal"] = (
        memo_df.get("investment_signal", pd.Series(index=memo_df.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    memo_df = memo_df.loc[memo_df["investment_signal"] != ""].copy()

    if memo_df.empty:
        return {
            "kind": "text",
            "title": title,
            "content": "No score-2 rows with non-empty investment signals were found, so this report could not be generated.",
            "pdf_name": pdf_name,
        }

    signal_counts = (
        memo_df["investment_signal"]
        .value_counts()
        .rename_axis("investment_signal")
        .reset_index(name="count")
    )

    signals_payload = signal_counts.to_dict(orient="records")

    system_prompt = """
You are analyzing investment signals extracted from job postings.

Use only the investment signals provided.
Do not use any other fields.
Do not invent technologies, business areas, motives, or applications that are not clearly supported by the signals.

Your task is to write a precise qualitative report on what the company is investing in.

Before writing, do this internally:
1. Group similar investment signals into broader non-overlapping investment areas.
2. Merge signals that refer to the same capability, application context, or objective.
3. Separate only those areas that are materially distinct.

Rules:
- Stay very close to the wording and meaning of the signals.
- Do not write one paragraph per raw signal.
- Consolidate overlapping signals into a smaller number of clearly differentiated investment areas.
- Each investment area must explain:
  1. the capability being built
  2. where it is being applied
  3. what objective it serves
- Prefer exact applications over abstract labels.
- Do not mention counts.
- Do not say repeated, multiple, several, appears, or similar frequency language.
- Do not add generic caveats.
- Do not use markdown headings like ##.
- Write in compact report style, with short titled paragraphs in bold.
- Avoid duplication across paragraphs.
- If two paragraphs describe the same investment, merge them.
- Keep application layer, infrastructure layer, and domain-specific AI applications separate only when the signals clearly support that distinction.
""".strip()

    user_prompt = f"""
Investment signals extracted from postings with technical relevance score equal to 2:

{json.dumps(signals_payload, ensure_ascii=False, indent=2)}
""".strip()

    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
            top_p=1,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )

    memo_text = response.text
    memo_text = "" if memo_text is None else str(memo_text).strip()

    if not memo_text:
        raise ValueError("The model returned an empty investment memo.")

    return {
        "kind": "text",
        "title": title,
        "content": memo_text,
        "pdf_name": pdf_name,
    }



# ===== Notebook cell 29 =====


# Writes a separate report generator script used only for graph PDFs.
# It runs in a fresh Python process, so matplotlib font-state errors in the notebook kernel do not affect graph generation.
REPORT_GENERATOR_SCRIPT = Path("report_generator_subprocess.py").resolve()
REPORT_GENERATOR_SCRIPT.write_text('\nimport sys\nimport os\nimport ast\nimport json\nimport re\nimport tempfile\nfrom pathlib import Path\nfrom textwrap import wrap\n\nos.environ.setdefault("MPLBACKEND", "Agg")\nimport matplotlib\nmatplotlib.use("Agg", force=True)\nimport matplotlib.pyplot as plt\nfrom matplotlib.figure import Figure\nimport numpy as np\nimport pandas as pd\n\n# Avoid tight layout and tight bbox inside subprocess. This preserves graph generation\n# while avoiding the FontProperties error that can occur in a polluted Jupyter kernel.\ndef _safe_no_tight_layout(*args, **kwargs):\n    return None\nplt.tight_layout = _safe_no_tight_layout\nFigure.tight_layout = _safe_no_tight_layout\n\nfrom reportlab.lib import colors\nfrom reportlab.lib.pagesizes import A4\nfrom reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet\nfrom reportlab.lib.units import cm\nfrom reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer\n\n\ndef nice(text: str) -> str:\n    return str(text).replace("_", " ")\n\n\ndef nice_label(text: str, width: int = 16) -> str:\n    return "\\n".join(wrap(nice(text).title(), width=width))\n\n\ndef is_missing_scalar(value) -> bool:\n    if value is None:\n        return True\n    if isinstance(value, (list, tuple, dict, set)):\n        return False\n    try:\n        return bool(pd.isna(value))\n    except Exception:\n        return False\n\n\ndef clean_text(value) -> str:\n    return "" if is_missing_scalar(value) else str(value).strip()\n\n\ndef parse_list_cell(value):\n    if isinstance(value, list):\n        return [str(item).strip() for item in value if str(item).strip()]\n    if is_missing_scalar(value):\n        return []\n    if isinstance(value, str):\n        value = value.strip()\n        if not value:\n            return []\n        for parser in (json.loads, ast.literal_eval):\n            try:\n                parsed = parser(value)\n                if isinstance(parsed, list):\n                    return [str(item).strip() for item in parsed if str(item).strip()]\n            except Exception:\n                pass\n        return [item.strip() for item in value.split(",") if item.strip()]\n    return []\n\n\ndef extract_json_object(text: str) -> str:\n    text = str(text).strip()\n    if text.startswith("```"):\n        text = re.sub(r"^```(?:json)?\\s*", "", text)\n        text = re.sub(r"\\s*```$", "", text)\n    start = text.find("{")\n    end = text.rfind("}")\n    if start == -1 or end == -1 or end < start:\n        raise ValueError("No JSON object found in model output")\n    return text[start:end + 1]\n\n\ndef find_source_workbook(company_dir: Path) -> Path:\n    files = [\n        path for path in sorted(company_dir.glob("*.xlsx"))\n        if not path.name.startswith("~$") and not path.stem.endswith("_enriched")\n    ]\n    if not files:\n        enriched_only = [\n            path for path in sorted(company_dir.glob("*.xlsx"))\n            if not path.name.startswith("~$")\n        ]\n        if enriched_only:\n            return enriched_only[0]\n        raise FileNotFoundError(f"No .xlsx file found in {company_dir}")\n    return files[0]\n\n\ndef get_output_workbook_path(source_path: Path) -> Path:\n    if source_path.stem.endswith("_enriched"):\n        return source_path\n    return source_path.with_name(f"{source_path.stem}_enriched.xlsx")\n\n\ndef normalize_jobs_dataframe(df: pd.DataFrame) -> pd.DataFrame:\n    jobs = df.copy()\n    jobs.columns = (\n        jobs.columns.astype(str)\n        .str.strip()\n        .str.replace(" ", "_", regex=False)\n        .str.lower()\n    )\n\n    drop_cols = [\n        "salary",\n        "hiring_manager_full_name",\n        "hiring_manager_first_name",\n        "hiring_manager_role",\n        "hiring_manager_linkedin_url",\n        "company_url",\n        "company_linkedin_url",\n        "company_industry",\n        "company_employee_count",\n        "company_revenue_usd",\n        "company_seo_description",\n        "company_description",\n        "company_city",\n        "job_location",\n        "is_remote",\n        "url",\n        "employment_status",\n    ]\n    jobs = jobs.drop(columns=[col for col in drop_cols if col in jobs.columns], errors="ignore")\n\n    if "posted_date" in jobs.columns:\n        jobs["posted_date"] = pd.to_datetime(jobs["posted_date"], errors="coerce")\n        jobs["year"] = jobs.get("year", jobs["posted_date"].dt.year)\n        jobs["month"] = jobs.get("month", jobs["posted_date"].dt.month)\n\n    if "job_id" not in jobs.columns:\n        jobs.insert(0, "job_id", range(1, len(jobs) + 1))\n    jobs["job_id"] = jobs["job_id"].astype(str)\n\n    subset_cols = [col for col in jobs.columns if col != "job_id"]\n    jobs = jobs.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)\n    jobs = jobs.drop(columns=["job_id"], errors="ignore")\n    jobs.insert(0, "job_id", range(1, len(jobs) + 1))\n    jobs["job_id"] = jobs["job_id"].astype(str)\n\n    for column in [\n        "primary_category",\n        "business_area",\n        "technical_relevance_score",\n        "technical_signals",\n        "investment_signal",\n        "llm_status",\n        "llm_error",\n    ]:\n        if column not in jobs.columns:\n            jobs[column] = pd.NA\n\n    if "technical_signals" in jobs.columns and "skills_signals" in jobs.columns:\n        needs_fill = jobs["technical_signals"].isna() & jobs["skills_signals"].notna()\n        jobs.loc[needs_fill, "technical_signals"] = jobs.loc[needs_fill, "skills_signals"]\n    elif "skills_signals" in jobs.columns and "technical_signals" not in jobs.columns:\n        jobs["technical_signals"] = jobs["skills_signals"]\n\n    return jobs\n\n\ndef load_jobs_for_processing(company_dir: Path) -> tuple[Path, Path, pd.DataFrame]:\n    source_path = find_source_workbook(company_dir)\n    output_path = get_output_workbook_path(source_path)\n    load_path = output_path if output_path.exists() else source_path\n    jobs = pd.read_excel(load_path)\n    jobs = normalize_jobs_dataframe(jobs)\n    return source_path, output_path, jobs\n\n\n\nAI_FAMILIES = [\n    "ai_ml_modeling",\n    "llm_genai_applications",\n    "agentic_ai_systems",\n]\n\nAI_FAMILY_LABELS = {\n    "ai_ml_modeling": "AI / ML modeling",\n    "llm_genai_applications": "LLM / GenAI apps",\n    "agentic_ai_systems": "Agentic AI",\n}\n\nAI_FAMILY_COLORS = {\n    "ai_ml_modeling": "#BFD7EA",\n    "llm_genai_applications": "#86AED3",\n    "agentic_ai_systems": "#2F6FAE",\n}\n\n\ndef prepare_analysis_dataframe(df: pd.DataFrame) -> pd.DataFrame:\n    analysis_df = df.copy()\n\n    if "technical_signals" not in analysis_df.columns and "skills_signals" in analysis_df.columns:\n        analysis_df["technical_signals"] = analysis_df["skills_signals"]\n\n    analysis_df["technical_relevance_score"] = pd.to_numeric(\n        analysis_df["technical_relevance_score"], errors="coerce"\n    )\n\n    if "year" in analysis_df.columns:\n        analysis_df["year"] = pd.to_numeric(analysis_df["year"], errors="coerce")\n    if "month" in analysis_df.columns:\n        analysis_df["month"] = pd.to_numeric(analysis_df["month"], errors="coerce")\n\n    if "job_country_code" in analysis_df.columns:\n        analysis_df["job_country_code"] = (\n            analysis_df["job_country_code"]\n            .fillna("Unknown")\n            .astype(str)\n            .str.strip()\n            .replace("", "Unknown")\n            .str.upper()\n        )\n\n    if "technical_signals" in analysis_df.columns:\n        analysis_df["_signals_list"] = analysis_df["technical_signals"].apply(parse_list_cell)\n    else:\n        analysis_df["_signals_list"] = [[] for _ in range(len(analysis_df))]\n\n    return analysis_df\n\n\ndef empty_figure(message: str, figsize=(10, 5)):\n    fig, ax = plt.subplots(figsize=figsize)\n    ax.axis("off")\n    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)\n    fig.patch.set_facecolor("white")\n    return fig\n\n\ndef save_report_pdf(fig, title: str, note: str, pdf_path: Path) -> None:\n    styles = getSampleStyleSheet()\n    title_style = ParagraphStyle(\n        "title_style",\n        parent=styles["Title"],\n        fontName="Helvetica-Bold",\n        fontSize=18,\n        leading=22,\n        textColor=colors.HexColor("#17324D"),\n        spaceAfter=10,\n    )\n    body_style = ParagraphStyle(\n        "body_style",\n        parent=styles["BodyText"],\n        fontName="Helvetica",\n        fontSize=10.5,\n        leading=15,\n        textColor=colors.HexColor("#222222"),\n        spaceAfter=8,\n    )\n\n    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:\n        temp_png_path = Path(tmp.name)\n\n    fig.savefig(temp_png_path, dpi=220, facecolor="white")\n\n    doc = SimpleDocTemplate(\n        str(pdf_path),\n        pagesize=A4,\n        leftMargin=1.8 * cm,\n        rightMargin=1.8 * cm,\n        topMargin=1.6 * cm,\n        bottomMargin=1.6 * cm,\n    )\n\n    img = Image(str(temp_png_path))\n    img._restrictSize(A4[0] - doc.leftMargin - doc.rightMargin, 13.0 * cm)\n\n    paragraphs = []\n    for block in str(note).strip().split("\\n\\n"):\n        block = block.strip()\n        if not block:\n            continue\n        block = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")\n        block = block.replace("\\n", "<br/>")\n        paragraphs.append(Paragraph(block, body_style))\n\n    story = [Paragraph(title, title_style), Spacer(1, 0.2 * cm), img, Spacer(1, 0.45 * cm)]\n    story.extend(paragraphs)\n    doc.build(story)\n\n    temp_png_path.unlink(missing_ok=True)\n\n\ndef select_until_coverage(series: pd.Series, coverage: float = 0.95) -> pd.Series:\n    ranked = series.sort_values(ascending=False)\n    if ranked.empty or ranked.sum() == 0:\n        return ranked\n\n    cumulative = ranked.cumsum() / ranked.sum()\n    keep = cumulative <= coverage\n    if keep.sum() == 0:\n        keep.iloc[0] = True\n    else:\n        first_excluded = keep[keep == False].index[:1]\n        if len(first_excluded) > 0:\n            keep.loc[first_excluded[0]] = True\n    return ranked[keep].sort_values(ascending=True)\n\n\n\ndef build_breadth_report(df: pd.DataFrame):\n    score2 = df.loc[df["technical_relevance_score"] == 2].copy()\n    title = "AI breadth across capability pockets"\n    pdf_name = "ai_breadth_across_capability_pockets.pdf"\n\n    if score2.empty:\n        note = "No score-2 AI roles are available, so the breadth matrix cannot be built yet."\n        return empty_figure(note, figsize=(12, 6)), title, note, pdf_name\n\n    matrix = score2.groupby(["primary_category", "business_area"]).size().unstack(fill_value=0)\n    if matrix.empty:\n        note = "The score-2 subset is empty after grouping, so the breadth matrix cannot be built."\n        return empty_figure(note, figsize=(12, 6)), title, note, pdf_name\n\n    x = np.arange(len(matrix.columns))\n    y = np.arange(len(matrix.index))\n    x_labels = [nice_label(col, 18) for col in matrix.columns]\n    y_labels = [nice_label(row, 18) for row in matrix.index]\n\n    fig, ax = plt.subplots(figsize=(12, 6.5))\n    fig.patch.set_facecolor("white")\n    ax.set_facecolor("white")\n\n    for xi in x:\n        ax.axvline(xi, color="#EAECEF", lw=1, zorder=0)\n    for yi in y:\n        ax.axhline(yi, color="#EAECEF", lw=1, zorder=0)\n\n    max_value = matrix.to_numpy().max() if matrix.to_numpy().size else 1\n    max_value = max(max_value, 1)\n\n    for i, row_name in enumerate(matrix.index):\n        for j, col_name in enumerate(matrix.columns):\n            value = matrix.loc[row_name, col_name]\n            if value > 0:\n                size = 900 * (value / max_value) + 250\n                ax.scatter(j, i, s=size, color="#A8C7E6", edgecolor="#6F9FCB", linewidth=1.5, alpha=0.95, zorder=3)\n                ax.text(j, i, str(value), ha="center", va="center", fontsize=13, fontweight="bold", color="#17324D", zorder=4)\n            else:\n                ax.scatter(j, i, s=60, color="#F5F7FA", edgecolor="#D9E1E8", linewidth=0.8, zorder=2)\n\n    ax.set_xticks(x)\n    ax.set_xticklabels(x_labels, fontsize=11)\n    ax.set_yticks(y)\n    ax.set_yticklabels(y_labels, fontsize=11)\n    ax.set_xlabel("")\n    ax.set_ylabel("")\n    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, pad=10)\n    ax.tick_params(axis="y", length=0)\n    ax.invert_yaxis()\n    for spine in ax.spines.values():\n        spine.set_visible(False)\n    plt.tight_layout()\n\n    total_roles = int(matrix.values.sum())\n    category_counts = matrix.sum(axis=1).sort_values(ascending=False)\n    area_counts = matrix.sum(axis=0).sort_values(ascending=False)\n    pair_counts = matrix.stack().reset_index(name="count")\n    pair_counts = pair_counts[pair_counts["count"] > 0].sort_values("count", ascending=False)\n\n    top_category_lines = [\n        f"- {nice(idx)}: {int(val)} roles ({val / total_roles:.1%} of score-2 roles)"\n        for idx, val in category_counts.head(5).items()\n    ]\n    top_area_lines = [\n        f"- {nice(idx)}: {int(val)} roles ({val / total_roles:.1%} of score-2 roles)"\n        for idx, val in area_counts.head(5).items()\n    ]\n    top_pair_lines = [\n        f"- {nice(row.primary_category)} × {nice(row.business_area)}: {int(row[\'count\'])} roles ({row[\'count\'] / total_roles:.1%} of score-2 roles)"\n        for _, row in pair_counts.head(8).iterrows()\n    ]\n\n    lead_pair = pair_counts.iloc[0]\n    note = f"""\nScore-2 role distribution note\n\nThis section refers to the bubble matrix built from postings with technical_relevance_score = 2.\n\nThe chart maps the count of AI-core roles across:\n- primary_category on the y-axis\n- business_area on the x-axis\n\nEach bubble shows the number of postings in one category-area intersection. Larger bubbles mean more score-2 roles in that pocket.\n\nTotal score-2 roles in this matrix: {total_roles}\n\nThe largest primary categories are:\n{chr(10).join(top_category_lines)}\n\nThe largest business areas are:\n{chr(10).join(top_area_lines)}\n\nThe most important intersections in the matrix are:\n{chr(10).join(top_pair_lines)}\n\nDeterministic reading of the matrix:\n- The score-2 signal is concentrated rather than evenly spread.\n- The largest concentration sits in {nice(lead_pair[\'primary_category\'])} × {nice(lead_pair[\'business_area\'])}.\n- The matrix shows where the company is hiring AI-core talent by combining capability type and business application area.\n- This chart should be read as a distribution of AI-core hiring pockets, not as a direct measure of spending.\n""".strip()\n\n    return fig, title, note, pdf_name\n\n\ndef build_skills_seniority_report(df: pd.DataFrame):\n    title = "AI skills and seniority"\n    pdf_name = "ai_skills_and_seniority.pdf"\n    score2 = df.loc[df["technical_relevance_score"] == 2].copy()\n\n    if score2.empty:\n        note = "No score-2 AI roles are available, so the skills and seniority view cannot be built yet."\n        return empty_figure(note, figsize=(14, 8)), title, note, pdf_name\n\n    score2["_seniority_bucket"] = (\n        score2.get("seniority", pd.Series(index=score2.index))\n        .fillna("unknown")\n        .astype(str)\n        .str.strip()\n    )\n\n    area_total = df["business_area"].value_counts()\n    area_score2_seniority = (\n        score2.groupby(["business_area", "_seniority_bucket"])\n        .size()\n        .unstack(fill_value=0)\n    )\n\n    share_pct = area_score2_seniority.div(area_total, axis=0).fillna(0) * 100\n    share_pct["total"] = share_pct.sum(axis=1)\n    share_pct = share_pct.sort_values("total", ascending=True)\n\n    seniority_cols = [col for col in share_pct.columns if col != "total"]\n    palette = ["#DCECC9", "#A9D18E", "#6AA84F", "#B7C9E2", "#7EA6D8", "#4F81BD", "#D9D2E9", "#B4A7D6"]\n    color_map = {col: palette[i % len(palette)] for i, col in enumerate(seniority_cols)}\n\n    fig = plt.figure(figsize=(16, 10))\n    grid = fig.add_gridspec(\n        2, 2,\n        width_ratios=[1.05, 1.15],\n        height_ratios=[1, 1],\n        wspace=0.28,\n        hspace=0.28\n    )\n    left_grid = grid[:, 0].subgridspec(3, 1, hspace=0.35)\n\n    family_titles = {\n        "ai_ml_modeling": "AI / ML modeling skills",\n        "llm_genai_applications": "LLM / GenAI skills",\n        "agentic_ai_systems": "Agentic AI skills",\n    }\n    family_colors = {\n        "ai_ml_modeling": ("#D9EAF7", "#8FB6D8"),\n        "llm_genai_applications": ("#CFE2F3", "#6FA8DC"),\n        "agentic_ai_systems": ("#B6D7F0", "#3D85C6"),\n    }\n\n    for i, family in enumerate(AI_FAMILIES):\n        ax = fig.add_subplot(left_grid[i, 0])\n\n        top_signals = (\n            score2.loc[score2["primary_category"] == family, "_signals_list"]\n            .explode()\n            .dropna()\n            .value_counts()\n            .head(8)\n            .sort_values(ascending=True)\n        )\n\n        fill_color, edge_color = family_colors[family]\n\n        if not top_signals.empty:\n            ax.barh(\n                top_signals.index,\n                top_signals.values,\n                color=fill_color,\n                edgecolor=edge_color,\n                height=0.6,\n            )\n            for j, value in enumerate(top_signals.values):\n                ax.text(\n                    value + 0.15,\n                    j,\n                    str(int(value)),\n                    va="center",\n                    ha="left",\n                    fontsize=10,\n                    color="#17324D",\n                    fontweight="bold",\n                )\n\n        ax.set_title(family_titles[family], loc="left", fontsize=13, fontweight="bold", pad=8)\n        ax.set_xlabel("")\n        ax.set_ylabel("")\n        ax.set_xticks([])\n        ax.tick_params(axis="y", labelsize=10, length=0)\n        for spine in ax.spines.values():\n            spine.set_visible(False)\n\n    ax_right = fig.add_subplot(grid[:, 1])\n    left = pd.Series(0, index=share_pct.index)\n\n    for bucket in seniority_cols:\n        ax_right.barh(\n            share_pct.index.map(lambda value: nice(value).title()),\n            share_pct[bucket].values,\n            left=left.values,\n            color=color_map[bucket],\n            edgecolor="white",\n            height=0.65,\n            label=nice(bucket).title(),\n        )\n        left += share_pct[bucket]\n\n    for i, total in enumerate(share_pct["total"].values):\n        ax_right.text(\n            total + 0.2,\n            i,\n            f"{total:.0f}%",\n            va="center",\n            ha="left",\n            fontsize=11,\n            color="#24411A",\n            fontweight="bold",\n        )\n\n    ax_right.set_title("AI roles share by area and seniority", loc="left", fontsize=15, fontweight="bold", pad=12)\n    ax_right.set_xlabel("")\n    ax_right.set_ylabel("")\n    ax_right.set_xticks([])\n    ax_right.tick_params(axis="y", labelsize=11, length=0)\n\n    if seniority_cols:\n        ax_right.legend(frameon=False, loc="lower right")\n\n    for spine in ax_right.spines.values():\n        spine.set_visible(False)\n\n    plt.tight_layout()\n\n    family_lines = []\n    family_counts = score2["primary_category"].value_counts()\n    for family, value in family_counts.head(6).items():\n        family_lines.append(f"- {nice(family)}: {int(value)} roles ({value / len(score2):.1%} of score-2 roles)")\n\n    signal_lines = []\n    for family in AI_FAMILIES:\n        top_signals = (\n            score2.loc[score2["primary_category"] == family, "_signals_list"]\n            .explode()\n            .dropna()\n            .value_counts()\n            .head(8)\n        )\n        if not top_signals.empty:\n            block = [f"- {signal}: {int(value)} mentions" for signal, value in top_signals.items()]\n            signal_lines.append(f"{AI_FAMILY_LABELS[family]} skills:\\n" + "\\n".join(block))\n\n    top_areas = share_pct.sort_values("total", ascending=False).head(6)\n\n    area_lines = []\n    for area, row in top_areas.iterrows():\n        total_share = row["total"]\n        seniority_breakdown = row.drop(labels=["total"])\n        seniority_breakdown = seniority_breakdown[seniority_breakdown > 0].sort_values(ascending=False)\n        seniority_text = ", ".join(\n            [f"{nice(bucket)} {value:.1f}%" for bucket, value in seniority_breakdown.items()]\n        )\n        area_lines.append(\n            f"- {nice(area)}: {total_share:.1f}% of postings in this area are score-2 AI roles; "\n            f"seniority split within that share: {seniority_text}"\n        )\n\n    note = f"""\nScore-2 skills and area-seniority note\n\nThis section refers to the chart composed of:\n- three left panels showing the most frequent technical signals inside the main AI role families\n- one right panel showing the share of score-2 AI roles within each business area, split by seniority\n\nTotal score-2 roles used in this chart: {len(score2)}\n\nThe main score-2 role families are:\n{chr(10).join(family_lines)}\n\nThe most frequent technical signals by AI family are:\n{chr(10).join(signal_lines) if signal_lines else \'- No recurring technical signals were found.\'}\n\nThe business areas with the highest concentration of score-2 AI roles are:\n{chr(10).join(area_lines) if area_lines else \'- No area-level concentration could be computed.\'}\n\nDeterministic reading of the chart:\n- The left panels show which technical signals characterize each AI role family rather than the whole score-2 population combined.\n- The right panel should be read as a share of postings within each business area in this dataset, not as a share of employees.\n- The strongest area-level concentration of score-2 AI roles sits in the business areas at the top of the ranking above.\n- The seniority split shows whether AI hiring in each area is concentrated more in junior, mid-level, senior, or staff roles.\n""".strip()\n\n    return fig, title, note, pdf_name\n\n\ndef build_trend_report(df: pd.DataFrame):\n    title = "AI hiring trend"\n    pdf_name = "ai_hiring_trend.pdf"\n\n    trend_df = df[df["technical_relevance_score"].isin([0, 1, 2])].copy()\n    if trend_df.empty or "year" not in trend_df.columns or "month" not in trend_df.columns:\n        note = "The dataset does not contain enough year and month information to build the trend report."\n        return empty_figure(note, figsize=(14, 9)), title, note, pdf_name\n\n    trend_df["time_period_month"] = pd.to_datetime(\n        dict(year=trend_df["year"], month=trend_df["month"], day=1),\n        errors="coerce",\n    )\n    trend_df = trend_df.dropna(subset=["time_period_month", "year"])\n    if trend_df.empty:\n        note = "All year and month values are missing after cleaning, so the trend report cannot be built."\n        return empty_figure(note, figsize=(14, 9)), title, note, pdf_name\n\n    monthly_trend = trend_df.groupby(["time_period_month", "technical_relevance_score"]).size().unstack(fill_value=0).sort_index()\n    yearly_total = trend_df.groupby("year").size().sort_index()\n    yearly_ai = trend_df.loc[trend_df["technical_relevance_score"] == 2].groupby("year").size().sort_index()\n    yearly_ai_share = ((yearly_ai / yearly_total) * 100).fillna(0).sort_index()\n\n    ai_df = trend_df.loc[trend_df["technical_relevance_score"] == 2].copy()\n    ai_mix = (\n        ai_df[ai_df["primary_category"].isin(AI_FAMILIES)]\n        .groupby(["year", "primary_category"])\n        .size()\n        .unstack(fill_value=0)\n        .reindex(columns=AI_FAMILIES, fill_value=0)\n        .sort_index()\n    )\n\n    fig, axes = plt.subplots(2, 2, figsize=(16, 11), gridspec_kw={"width_ratios": [1.1, 1], "height_ratios": [1, 1]})\n    fig.patch.set_facecolor("white")\n    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, wspace=0.22, hspace=0.35)\n\n    if 0 in monthly_trend.columns:\n        axes[0, 0].bar(monthly_trend.index, monthly_trend[0], width=20, color="#BFD7EA", edgecolor="#86AED3")\n    axes[0, 0].set_title("No AI roles by month", loc="left", fontsize=15, fontweight="bold", pad=10)\n    axes[0, 0].set_xlabel("")\n    axes[0, 0].set_ylabel("Number of job postings")\n    axes[0, 0].tick_params(axis="x", rotation=45)\n    axes[0, 0].grid(axis="y", alpha=0.25)\n    axes[0, 0].set_axisbelow(True)\n    for spine in axes[0, 0].spines.values():\n        spine.set_visible(False)\n\n    bottom = pd.Series(0, index=monthly_trend.index)\n    monthly_labels = {1: "AI-literate roles", 2: "AI roles"}\n    monthly_colors = {1: "#F4B183", 2: "#C55A11"}\n    for score in [1, 2]:\n        if score in monthly_trend.columns:\n            axes[0, 1].bar(\n                monthly_trend.index,\n                monthly_trend[score],\n                bottom=bottom,\n                width=20,\n                label=monthly_labels[score],\n                color=monthly_colors[score],\n                edgecolor="white",\n            )\n            bottom += monthly_trend[score]\n    axes[0, 1].set_title("AI-related roles by month", loc="left", fontsize=15, fontweight="bold", pad=10)\n    axes[0, 1].set_xlabel("")\n    axes[0, 1].set_ylabel("Number of job postings")\n    axes[0, 1].tick_params(axis="x", rotation=45)\n    axes[0, 1].grid(axis="y", alpha=0.25)\n    axes[0, 1].set_axisbelow(True)\n    axes[0, 1].legend(frameon=False, loc="upper left")\n    for spine in axes[0, 1].spines.values():\n        spine.set_visible(False)\n\n    axes[1, 0].bar(yearly_ai_share.index.astype(int).astype(str), yearly_ai_share.values, color="#C55A11", edgecolor="white", width=0.62)\n    for i, value in enumerate(yearly_ai_share.values):\n        axes[1, 0].text(i, value + 0.15, f"{value:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#7A3000")\n    axes[1, 0].set_title("AI roles share by year", loc="left", fontsize=15, fontweight="bold", pad=10)\n    axes[1, 0].set_xlabel("")\n    axes[1, 0].set_ylabel("% of total job postings")\n    axes[1, 0].set_ylim(0, max(yearly_ai_share.max() * 1.18, 1))\n    axes[1, 0].grid(axis="y", alpha=0.25)\n    axes[1, 0].set_axisbelow(True)\n    axes[1, 0].margins(x=0.08)\n    for spine in axes[1, 0].spines.values():\n        spine.set_visible(False)\n\n    bottom = pd.Series(0, index=ai_mix.index)\n    for family in ai_mix.columns:\n        axes[1, 1].bar(\n            ai_mix.index.astype(int).astype(str),\n            ai_mix[family].values,\n            bottom=bottom.values,\n            color=AI_FAMILY_COLORS[family],\n            edgecolor="white",\n            width=0.62,\n            label=AI_FAMILY_LABELS[family],\n        )\n        bottom += ai_mix[family]\n    axes[1, 1].set_title("AI roles mix by year", loc="left", fontsize=15, fontweight="bold", pad=10)\n    axes[1, 1].set_xlabel("")\n    axes[1, 1].set_ylabel("Number of AI roles")\n    axes[1, 1].grid(axis="y", alpha=0.25)\n    axes[1, 1].set_axisbelow(True)\n    axes[1, 1].legend(frameon=False, loc="upper left")\n    axes[1, 1].margins(x=0.08)\n    for spine in axes[1, 1].spines.values():\n        spine.set_visible(False)\n\n    def peak_line(series: pd.Series, label: str) -> str:\n        if series.empty or series.max() == 0:\n            return f"- {label}: no postings in this series"\n        peak_date = series.idxmax()\n        peak_value = int(series.max())\n        return f"- {label}: peak of {peak_value} postings in {peak_date.strftime(\'%Y-%m\')}"\n\n    monthly_lines = []\n    if 0 in monthly_trend.columns:\n        monthly_lines.append(peak_line(monthly_trend[0], "No AI roles"))\n    if 1 in monthly_trend.columns:\n        monthly_lines.append(peak_line(monthly_trend[1], "AI-literate roles"))\n    if 2 in monthly_trend.columns:\n        monthly_lines.append(peak_line(monthly_trend[2], "AI roles"))\n\n    share_lines = [\n        f"- {int(year)}: {value:.1f}% of total postings are AI roles"\n        for year, value in yearly_ai_share.items()\n    ]\n\n    mix_lines = []\n    for year, row in ai_mix.iterrows():\n        total = int(row.sum())\n        if total == 0:\n            continue\n        dominant_family = row.idxmax()\n        dominant_value = int(row.max())\n        parts = [f"{AI_FAMILY_LABELS[family]} {int(row[family])}" for family in ai_mix.columns if int(row[family]) > 0]\n        mix_lines.append(\n            f"- {int(year)}: {\', \'.join(parts)}; dominant family = {AI_FAMILY_LABELS[dominant_family]} ({dominant_value} roles)"\n        )\n\n    n_no_ai = int((trend_df["technical_relevance_score"] == 0).sum())\n    n_ai_literate = int((trend_df["technical_relevance_score"] == 1).sum())\n    n_ai = int((trend_df["technical_relevance_score"] == 2).sum())\n    n_total = int(len(trend_df))\n\n    note = f"""\nAI hiring trend note\n\nThis section refers to the 2x2 trend chart built from postings classified with:\n- technical_relevance_score = 0 for no AI signal\n- technical_relevance_score = 1 for AI-literate or AI-adjacent roles\n- technical_relevance_score = 2 for AI roles\n\nTotal postings included in this trend view: {n_total}\n- No AI roles: {n_no_ai}\n- AI-literate roles: {n_ai_literate}\n- AI roles: {n_ai}\n\nThe four panels represent:\n- top left: monthly count of roles with no AI signal\n- top right: monthly count of AI-literate and AI roles\n- bottom left: yearly share of AI roles over total postings\n- bottom right: yearly mix of AI roles across AI / ML modeling, LLM / GenAI applications, and Agentic AI systems\n\nMonthly peaks in hiring activity:\n{chr(10).join(monthly_lines)}\n\nYearly AI-role share:\n{chr(10).join(share_lines)}\n\nYearly mix of AI-role families:\n{chr(10).join(mix_lines) if mix_lines else \'- No score-2 AI family mix is available.\'}\n\nDeterministic reading of the chart:\n- The top row separates the broad hiring base from the AI-related segment, so the chart can be read as a change in AI hiring intensity rather than only total hiring growth.\n- The bottom-left panel shows whether AI roles become a larger or smaller share of the total hiring mix over time.\n- The bottom-right panel shows which AI family dominates the AI-role mix in each year.\n- This chart should be interpreted as a hiring-intensity and capability-mix view, not as a direct measure of investment spending.\n""".strip()\n\n    return fig, title, note, pdf_name\n\n\n\ndef build_geography_report(df: pd.DataFrame):\n    title = "AI hiring geography"\n    pdf_name = "ai_hiring_geography.pdf"\n\n    if "job_country_code" not in df.columns:\n        note = "The dataset does not contain job_country_code, so the geography report cannot be built."\n        return empty_figure(note, figsize=(14, 9)), title, note, pdf_name\n\n    score2 = df[df["technical_relevance_score"] == 2].copy()\n    if score2.empty:\n        note = "No score-2 AI roles are available, so the geography report cannot be built yet."\n        return empty_figure(note, figsize=(14, 9)), title, note, pdf_name\n\n    overall_country = df["job_country_code"].value_counts()\n    ai_country = score2["job_country_code"].value_counts()\n\n    country_summary = pd.DataFrame({"overall_hiring": overall_country, "ai_roles": ai_country}).fillna(0)\n    country_summary["overall_hiring"] = country_summary["overall_hiring"].astype(int)\n    country_summary["ai_roles"] = country_summary["ai_roles"].astype(int)\n    country_summary["ai_intensity_pct"] = (country_summary["ai_roles"] / country_summary["overall_hiring"] * 100).fillna(0)\n\n    top_overall = select_until_coverage(country_summary["overall_hiring"], coverage=0.95)\n    top_ai = select_until_coverage(country_summary.loc[country_summary["ai_roles"] > 0, "ai_roles"], coverage=0.95)\n\n    global_ai_rate = country_summary["ai_roles"].sum() / country_summary["overall_hiring"].sum()\n    shrink_strength = max(1, int(country_summary["overall_hiring"].median()))\n    country_summary["ai_intensity_smoothed_pct"] = (\n        (country_summary["ai_roles"] + shrink_strength * global_ai_rate)\n        / (country_summary["overall_hiring"] + shrink_strength)\n        * 100\n    )\n\n    top_intensity_index = select_until_coverage(\n        country_summary.loc[country_summary["ai_roles"] > 0, "ai_roles"],\n        coverage=0.95,\n    ).index\n    top_intensity = country_summary.loc[top_intensity_index].sort_values("ai_intensity_smoothed_pct", ascending=True)\n\n    mix_countries = top_ai.index.tolist()\n    ai_mix_country = (\n        score2[\n            score2["primary_category"].isin(AI_FAMILIES)\n            & score2["job_country_code"].isin(mix_countries)\n        ]\n        .groupby(["job_country_code", "primary_category"])\n        .size()\n        .unstack(fill_value=0)\n        .reindex(index=mix_countries, columns=AI_FAMILIES, fill_value=0)\n    )\n    ai_mix_country["total"] = ai_mix_country.sum(axis=1)\n    ai_mix_country = ai_mix_country.sort_values("total", ascending=True)\n    ai_mix_country_display = ai_mix_country.drop(columns="total")\n\n    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={"width_ratios": [1, 1.05]})\n    fig.patch.set_facecolor("white")\n    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, wspace=0.28, hspace=0.35)\n\n    axes[0, 0].barh(top_overall.index, top_overall.values, color="#D9EAF7", edgecolor="#8FB6D8", height=0.6)\n    for i, value in enumerate(top_overall.values):\n        axes[0, 0].text(value + max(top_overall.max() * 0.01, 1), i, f"{int(value)}", va="center", ha="left", fontsize=10, color="#17324D", fontweight="bold")\n    axes[0, 0].set_title("Top hiring countries", loc="left", fontsize=15, fontweight="bold", pad=10)\n    axes[0, 0].set_xticks([])\n    axes[0, 0].set_xlabel("")\n    axes[0, 0].set_ylabel("")\n    axes[0, 0].tick_params(axis="y", labelsize=11, length=0)\n    for spine in axes[0, 0].spines.values():\n        spine.set_visible(False)\n\n    axes[0, 1].barh(top_ai.index, top_ai.values, color="#F4B183", edgecolor="#C55A11", height=0.6)\n    for i, value in enumerate(top_ai.values):\n        axes[0, 1].text(value + max(top_ai.max() * 0.02, 0.3), i, f"{int(value)}", va="center", ha="left", fontsize=10, color="#7A3000", fontweight="bold")\n    axes[0, 1].set_title("Top countries for AI roles", loc="left", fontsize=15, fontweight="bold", pad=10)\n    axes[0, 1].set_xticks([])\n    axes[0, 1].set_xlabel("")\n    axes[0, 1].set_ylabel("")\n    axes[0, 1].tick_params(axis="y", labelsize=11, length=0)\n    for spine in axes[0, 1].spines.values():\n        spine.set_visible(False)\n\n    axes[1, 0].barh(top_intensity.index, top_intensity["ai_intensity_smoothed_pct"].values, color="#DCECC9", edgecolor="#6AA84F", height=0.6)\n    for i, value in enumerate(top_intensity["ai_intensity_smoothed_pct"].values):\n        axes[1, 0].text(value + max(top_intensity["ai_intensity_smoothed_pct"].max() * 0.02, 0.1), i, f"{value:.1f}%", va="center", ha="left", fontsize=10, color="#24411A", fontweight="bold")\n    axes[1, 0].set_title("AI intensity by country", loc="left", fontsize=15, fontweight="bold", pad=10)\n    axes[1, 0].set_xticks([])\n    axes[1, 0].set_xlabel("")\n    axes[1, 0].set_ylabel("")\n    axes[1, 0].tick_params(axis="y", labelsize=11, length=0)\n    for spine in axes[1, 0].spines.values():\n        spine.set_visible(False)\n\n    left = pd.Series(0, index=ai_mix_country_display.index)\n    for family in ai_mix_country_display.columns:\n        axes[1, 1].barh(\n            ai_mix_country_display.index,\n            ai_mix_country_display[family].values,\n            left=left.values,\n            color=AI_FAMILY_COLORS[family],\n            edgecolor="white",\n            height=0.62,\n            label=AI_FAMILY_LABELS[family],\n        )\n        left += ai_mix_country_display[family]\n    for i, total in enumerate(left.values):\n        axes[1, 1].text(total + max(left.max() * 0.02, 0.2), i, f"{int(total)}", va="center", ha="left", fontsize=10, color="#17324D", fontweight="bold")\n    axes[1, 1].set_title("AI role mix in top AI countries", loc="left", fontsize=15, fontweight="bold", pad=10)\n    axes[1, 1].set_xticks([])\n    axes[1, 1].set_xlabel("")\n    axes[1, 1].set_ylabel("")\n    axes[1, 1].tick_params(axis="y", labelsize=11, length=0)\n    axes[1, 1].legend(frameon=False, loc="lower right")\n    for spine in axes[1, 1].spines.values():\n        spine.set_visible(False)\n\n    overall_lines = [\n        f"- {country}: {int(value)} postings ({value / len(df):.1%} of all postings)"\n        for country, value in top_overall.sort_values(ascending=False).items()\n    ]\n    ai_lines = [\n        f"- {country}: {int(value)} AI roles ({value / max(len(score2), 1):.1%} of score-2 roles)"\n        for country, value in top_ai.sort_values(ascending=False).items()\n    ]\n    intensity_lines = [\n        f"- {country}: {row[\'ai_intensity_smoothed_pct\']:.1f}% AI intensity"\n        for country, row in top_intensity.iterrows()\n    ]\n\n    mix_lines = []\n    for country, row in ai_mix_country.sort_values("total", ascending=False).iterrows():\n        total = int(row["total"])\n        if total == 0:\n            continue\n        parts = [f"{AI_FAMILY_LABELS[family]} {int(row[family])}" for family in AI_FAMILIES if int(row[family]) > 0]\n        dominant_family = row[AI_FAMILIES].idxmax()\n        mix_lines.append(f"- {country}: {\', \'.join(parts)}; dominant family = {AI_FAMILY_LABELS[dominant_family]}")\n\n    note = f"""\nGeography note for AI hiring\n\nThis section refers to the 2x2 geography chart built from the posting dataset and the subset of postings with technical_relevance_score = 2.\n\nThe four panels represent:\n- top left: countries with the largest overall hiring volume in the dataset\n- top right: countries with the largest number of AI roles\n- bottom left: countries with the highest AI intensity, defined as the share of AI roles within that country\'s posting base\n- bottom right: mix of AI-role families in the main AI hiring countries\n\nTotal postings in the dataset: {len(df)}\nTotal AI roles used in the geography view: {len(score2)}\n\nOverall hiring is concentrated in:\n{chr(10).join(overall_lines)}\n\nAI-role hiring is concentrated in:\n{chr(10).join(ai_lines)}\n\nThe countries with the highest AI intensity are:\n{chr(10).join(intensity_lines)}\n\nWithin the main AI hiring countries, the AI-role mix is:\n{chr(10).join(mix_lines) if mix_lines else \'- No country level AI family mix is available.\'}\n\nDeterministic reading of the chart:\n- The top-left and top-right panels distinguish the broad hiring footprint from the narrower geography of AI-role hiring.\n- The bottom-left panel should be read as an intensity measure, not just raw volume.\n- A country can have large overall hiring but low AI intensity, or smaller overall hiring but higher AI concentration.\n- The bottom-right panel shows whether the AI-role mix in each leading country is dominated by AI / ML modeling, LLM / GenAI applications, or Agentic AI systems.\n- This chart should be interpreted as a geography-of-capability-building view, not as a direct measure of investment spending.\n""".strip()\n\n    return fig, title, note, pdf_name\n\n\n\nGRAPH_REPORT_BUILDERS = [\n    build_breadth_report,\n    build_skills_seniority_report,\n    build_trend_report,\n    build_geography_report,\n]\n\n\ndef generate_graph_reports(input_xlsx: Path, output_dir: Path):\n    raw_df = pd.read_excel(input_xlsx)\n    analysis_df = prepare_analysis_dataframe(raw_df)\n    output_dir.mkdir(parents=True, exist_ok=True)\n\n    saved = []\n    for build_report in GRAPH_REPORT_BUILDERS:\n        fig = None\n        try:\n            fig, title, note, pdf_name = build_report(analysis_df)\n            pdf_path = output_dir / pdf_name\n            save_report_pdf(fig, title, note, pdf_path)\n            saved.append(str(pdf_path))\n            print(f"Saved: {pdf_name}")\n        finally:\n            if fig is not None:\n                plt.close(fig)\n    return saved\n\n\nif __name__ == "__main__":\n    if len(sys.argv) != 3:\n        raise SystemExit("Usage: python report_generator_subprocess.py INPUT_XLSX OUTPUT_DIR")\n    input_xlsx = Path(sys.argv[1])\n    output_dir = Path(sys.argv[2])\n    saved = generate_graph_reports(input_xlsx, output_dir)\n    print(json.dumps({"saved": saved}, ensure_ascii=False))\n', encoding="utf-8")
print("Report generator subprocess script written to:", REPORT_GENERATOR_SCRIPT)



# ===== Notebook cell 30 =====


import subprocess
import sys

GRAPH_REPORT_PDF_NAMES = [
    "ai_breadth_across_capability_pockets.pdf",
    "ai_skills_and_seniority.pdf",
    "ai_hiring_trend.pdf",
    "ai_hiring_geography.pdf",
]


def run_graph_reports_in_subprocess(enriched_df: pd.DataFrame, company_dir: Path) -> list[Path]:
    """Generate graph PDFs in a fresh Python process to avoid matplotlib font-state errors in Jupyter."""
    if not REPORT_GENERATOR_SCRIPT.exists():
        raise FileNotFoundError(f"Missing report generator script: {REPORT_GENERATOR_SCRIPT}")

    temp_xlsx = None
    mpl_cache = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            temp_xlsx = Path(tmp.name)
        enriched_df.to_excel(temp_xlsx, index=False)

        env = os.environ.copy()
        if "PKG_DIR" in globals() and PKG_DIR:
            env["PYTHONPATH"] = str(PKG_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        mpl_cache = tempfile.mkdtemp(prefix="mplconfig_")
        env["MPLCONFIGDIR"] = mpl_cache
        env["MPLBACKEND"] = "Agg"

        cmd = [sys.executable, str(REPORT_GENERATOR_SCRIPT), str(temp_xlsx), str(company_dir)]
        completed = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            env=env,
            timeout=300,
        )

        if completed.stdout:
            print(completed.stdout.strip())
        if completed.stderr:
            print(completed.stderr.strip())

        if completed.returncode != 0:
            raise RuntimeError(f"Graph report subprocess failed with exit code {completed.returncode}")

        return [company_dir / name for name in GRAPH_REPORT_PDF_NAMES if (company_dir / name).exists()]

    finally:
        if temp_xlsx is not None:
            Path(temp_xlsx).unlink(missing_ok=True)


def run_analysis_reports(enriched_df: pd.DataFrame, company_dir: Path) -> list[Path]:
    pdf_paths = []

    try:
        graph_paths = run_graph_reports_in_subprocess(enriched_df, company_dir)
        pdf_paths.extend(graph_paths)
    except Exception as exc:
        print(f"Graph reports failed in subprocess -> {exc}")

    try:
        result = build_investment_signal_memo_report(enriched_df)
        title = result["title"]
        report_text = result["content"]
        pdf_name = result["pdf_name"]
        pdf_path = company_dir / pdf_name
        save_text_report_pdf(report_text, pdf_path, title)
        pdf_paths.append(pdf_path)
        print(f"Saved: {pdf_name}")
    except Exception as exc:
        print(f"Report failed: build_investment_signal_memo_report -> {exc}")

    return pdf_paths


def run_company_pipeline(company_dir: Path) -> dict:
    source_path, output_path, jobs = load_jobs_for_processing(company_dir)
    print()
    print(f"=== {company_dir.name} ===")
    print(f"Input workbook: {source_path.name}")
    print(f"Output workbook: {output_path.name}")
    print(f"Rows loaded: {len(jobs)}")

    enriched_jobs = process_jobs_dataframe(jobs, output_path)
    pdf_paths = []

    if RUN_ANALYSIS:
        pdf_paths = run_analysis_reports(enriched_jobs, company_dir)

    return {
        "company": company_dir.name,
        "input_workbook": str(source_path),
        "output_workbook": str(output_path),
        "rows": len(enriched_jobs),
        "pdf_count": len(pdf_paths),
        "pdf_files": [path.name for path in pdf_paths],
    }



# ===== Main loop: run all companies =====
def resolve_company_arg(arg: str) -> Path:
    raw = Path(arg).expanduser()
    if raw.exists():
        return raw
    candidate = COMPANIES_ROOT / arg
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find company folder from argument: {arg}")


def main():
    if len(sys.argv) > 1:
        company_dirs = [resolve_company_arg(arg) for arg in sys.argv[1:]]
    else:
        company_dirs = sorted(
            [
                path for path in COMPANIES_ROOT.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ]
        )

    if not company_dirs:
        print(f"No company folders found inside {COMPANIES_ROOT}/")
        return

    summary_rows = []
    total_companies = len(company_dirs)

    print(f"Found {total_companies} company folders.")

    for company_number, company_dir in enumerate(company_dirs, start=1):
        print()
        print("=" * 100)
        print(f"Company {company_number}/{total_companies}: {company_dir.name}")
        print("=" * 100)

        try:
            summary_rows.append(run_company_pipeline(company_dir))
        except Exception as exc:
            summary_rows.append(
                {
                    "company": company_dir.name,
                    "input_workbook": None,
                    "output_workbook": None,
                    "rows": None,
                    "pdf_count": 0,
                    "pdf_files": [],
                    "error": str(exc),
                }
            )
            print(f"Error in {company_dir.name}: {exc}")

        summary_df = pd.DataFrame(summary_rows)
        summary_path = Path("pipeline_summary_live.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Live summary saved to: {summary_path.resolve()}")

        if company_number < total_companies and SLEEP_BETWEEN_COMPANIES > 0:
            print(f"Sleeping {SLEEP_BETWEEN_COMPANIES}s before next company...")
            time.sleep(SLEEP_BETWEEN_COMPANIES)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("pipeline_summary_final.csv", index=False)

    print()
    print("Pipeline finished.")
    print(summary_df.to_string(index=False))
    print("Final summary saved to: pipeline_summary_final.csv")


if __name__ == "__main__":
    main()
