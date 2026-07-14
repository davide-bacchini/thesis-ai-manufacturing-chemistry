# An Overview of Modern AI Systems in Manufacturing and Chemistry

**Master's Thesis** — Davide Bacchini | Bocconi University | Supervised by Prof. Biggio

�� **[Read the full thesis (PDF)](./thesis.pdf)**

---

## What This Thesis Is About

Companies in manufacturing and chemistry are adopting AI across their operations — from predictive maintenance and quality control to materials discovery and digital twins. But understanding *what a specific company is actually doing with AI* is hard: academic papers show technical possibilities, not corporate reality. Internal programs, hiring decisions, and actual deployments remain hidden in fragmented public data.

This thesis tackles that problem in two parts:

1. **Literature review** — A structured overview of how LLMs, RAG, and Agentic AI are being applied in manufacturing and chemistry, from process monitoring to autonomous laboratories.

2. **Empirical analysis** — A working LLM pipeline that processes 91,544 job postings from 16 major firms (Michelin, Bridgestone, BASF, Adient, etc.), extracts AI investment signals, enriches them with annual reports and public sources via a research agent, and produces evaluated company reports queryable through a RAG chatbot.

---

## Business Impact: What Problem This Solves

Traditional competitive intelligence in industrial sectors relies on manual desk research — reading annual reports, tracking press releases, browsing job boards. This is slow, incomplete, and doesn't scale.

**This thesis demonstrates that:**

- Job postings are reliable predictors of what companies are building internally (hiring = investment signal)
- An LLM pipeline can automatically classify 91K+ postings by AI relevance, business area, and capability type
- A research agent can expand hiring signals with public evidence (press releases, interviews, investor comms) into structured company reports
- Those reports can be evaluated for factual grounding (72.4% grounding accuracy across 392 claims) and served via a RAG chatbot for natural language queries

**The result:** Instead of weeks of manual research per company, the pipeline produces evaluated intelligence reports in hours — each one telling you what AI capabilities a firm is building, where they're hiring, and what their strategic priorities appear to be.

---

## Technical Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  1. DATA COLLECTION                                              │
│  TheirStack API → 91,544 job postings (2022–present)             │
│  16 firms × NACE sectors 22.11, 22.19, 29.32                    │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  2. LLM CLASSIFICATION (Gemini 3 Flash Preview)                  │
│  For each job description → structured JSON:                     │
│    • primary_category (AI/ML, LLM/GenAI, Agentic, etc.)          │
│    • business_area (manufacturing, materials R&D, IT, etc.)      │
│    • technical_relevance_score (0 / 1 / 2)                       │
│    • technical_signals (tools, platforms, skills)                 │
│    • investment_signal (what AI capability is being built)        │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  3. HIRING REPORT GENERATION                                     │
│  Per company: trend, geography, seniority, capability pockets    │
│  → Investment signals grouped into capability areas              │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  4. RESEARCH AGENT (Google deep-research-preview-04-2026)        │
│  Hiring report + Annual report → Web search for public evidence  │
│  → Final company report with 4 sections:                         │
│    • Quantitative investment baseline                            │
│    • Capability-by-capability analysis (incl. GenAI & Agentic)   │
│    • Talent, teams, and operating model                          │
│    • Investment priorities and competitor monitoring              │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  5. EVALUATION BENCHMARK                                         │
│  Extract claims → Crawl cited URLs → TF-IDF selection            │
│  → LLM judge (gemini-3.1-pro) assigns:                          │
│    supported / needs_review / unsupported / not_verifiable       │
│  Metrics: Grounding accuracy, Strict support rate, etc.          │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  6. RAG CHATBOT                                                  │
│  Reports embedded with multilingual-e5-small → Vector index      │
│  Natural language queries about companies, technologies, trends  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Job postings processed | 91,544 |
| Companies analyzed | 16 |
| Public URL claims evaluated | 392 |
| Grounding accuracy | 72.4% |
| Strict support rate | 59.6% |
| Unsupported rate | 14.7% |
| Verifiability rate | 74.5% |

---

## Sectors & Companies Analyzed

**NACE sectors:** 22.11 (rubber tyres), 22.19 (other rubber products), 29.32 (motor vehicle parts)

**Companies include:** Michelin, Bridgestone, BASF, Adient, BorgWarner, Dana, Trelleborg, Continental, and others — all matched to their global corporate group for worldwide hiring coverage.

---

## Technologies Covered in the Review

| Technology | Industrial Application |
|---|---|
| CNNs | Visual quality control, defect detection |
| GNNs | Molecular property prediction, materials discovery (GNoME) |
| Hybrid models | Digital twins, process optimization |
| LLMs | Safety chatbots, Text-to-CAD, material property prediction |
| RAG | SOP generation, machine manuals Q&A, MOF design |
| Agentic AI | Self-driving labs, autonomous reaction optimization, plant diagnostics |

---

## How to Use

The thesis PDF contains the full analysis, methodology, and results. The pipeline code used for the empirical analysis is not included in this repository (proprietary data sources), but the architecture and prompts are fully described in Chapter 3.

To replicate or extend:
1. Obtain job posting data (e.g., from TheirStack or similar platforms)
2. Use the classification schema described in Section 3.2 with any capable LLM
3. Apply the evaluation benchmark (Section 3.3) to verify report quality
4. Index reports in any RAG framework (the thesis used multilingual-e5-small embeddings)

---

## Citation

If you reference this work, please cite:

```
Bacchini, D. (2025). An Overview of Modern AI Systems in Manufacturing and Chemistry.
Master's Thesis, Bocconi University. Supervised by Prof. Biggio.
```

---

## License

This thesis is shared for academic and educational purposes.
