# AI in Manufacturing and Chemistry — BSc Thesis

BSc Thesis, Bocconi University, 2026.

[📄 Full thesis PDF](./thesis.pdf)

---

## Summary

This thesis does two things:

1. Reviews how AI (LLMs, RAG, Agentic AI) is used in manufacturing and chemistry today.
2. Builds a pipeline that reads 91,544 job postings from 16 industrial firms, classifies them by AI relevance, and produces company reports on what each firm is building — then makes them queryable through a chatbot.

---

## The problem

You can't easily tell what a specific company is doing with AI from the outside. Academic papers show what's technically possible. Annual reports are vague. Press releases are marketing.

Job postings are different. When a company hires for "ML Engineer — predictive maintenance for tire manufacturing," that's a direct signal of what they're investing in.

---

## What the pipeline does

1. **Collects** 91,544 job postings from TheirStack (Workday, Indeed, LinkedIn) for 16 firms in rubber/automotive manufacturing
2. **Classifies** each posting with Gemini Flash: AI relevance score, business area, capability type, technical signals
3. **Generates hiring reports** per company (trend over time, geography, seniority, capability areas)
4. **Expands** with a research agent that searches public sources (press releases, interviews) and combines with annual reports
5. **Produces** a final company report per firm (investment baseline, capabilities, talent model, strategic priorities)
6. **Evaluates** each report: extracts claims, crawls cited URLs, checks if claims are actually supported by the source
7. **Indexes** reports in a RAG chatbot for natural language queries

---

## Results

- 91,544 postings processed across 16 companies
- 392 public-source claims evaluated
- 72.4% grounding accuracy (claims supported by cited evidence)
- 59.6% strict support rate
- 14.7% unsupported rate

---

## Companies and sectors

NACE 22.11, 22.19, 29.32 (tyres, rubber products, motor vehicle parts). Firms include Michelin, Bridgestone, BASF, Adient, BorgWarner, Dana, Trelleborg, Continental, and others.

---

## Literature review covers

- Predictive maintenance (PCA, UMAP, LSTMs)
- Visual quality control (CNNs)
- Molecular/materials discovery (GNNs, GNoME)
- Hybrid models and digital twins
- LLMs in manufacturing (safety chatbots, Text-to-CAD, material property prediction)
- RAG (SOP generation, machine manual Q&A, MOF band gap optimization)
- Agentic AI (self-driving labs, autonomous reaction optimization, plant diagnostics)

---

## Replication

The pipeline code is not in this repo (uses proprietary data). Chapter 3 describes the full architecture, prompts, and evaluation methodology. To replicate:

1. Get job posting data (TheirStack or similar)
2. Classify with the schema in Section 3.2
3. Run the evaluation benchmark (Section 3.3)
4. Index in any RAG framework (thesis used multilingual-e5-small)
