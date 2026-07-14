# An Overview of Modern AI Systems in Manufacturing and Chemistry

BSc Thesis, Bocconi University, 2026.

[📄 Full thesis (PDF)](./thesis.pdf)

---

## Abstract

This thesis examines the evolution of Artificial Intelligence applications in manufacturing and chemistry, from earlier task specific systems to more recent approaches based on Large Language Models, Retrieval Augmented Generation, and agentic AI. These technologies can be applied across several phases before the final product creation, from research and production to quality control, maintenance, supply chain, and business operations.

The literature provides many examples of traditional applications, often related to specific stages of production. However, these sources usually focus on the overall sector or on high level technical applications, while they provide limited evidence on what specific companies are currently developing, especially for newer trends. This may happen because firms could avoid disclosing information that reveals a competitive advantage.

For this reason, the empirical part relies on a pipeline to process public company data and understand from the outside what firms may be implementing. The analysis is based on 91,544 job postings from 16 companies, which are used as proxies for capability building, since their descriptions show which types of profiles and competences companies are looking for. Then, they are combined with annual reports, which provide evidence on investment priorities and reported initiatives. Both sources are expanded through a research agent, which is guided by the capabilities extracted from the job descriptions and searches for further public details online, including company pages, press releases, interviews, and investor communications.

The analysis produces company reports that can be queried against a RAG chatbot and used to compare firms and their capabilities. Their factual claims are evaluated by an LLM judge against the cited public sources, after the references in each report have been extracted, mapped to the claims that use them, and retrieved as readable text.

---

## Empirical analysis

The objective is to understand from publicly available data what companies in manufacturing and chemistry are building with AI. The firms were identified from Orbis by filtering for very large European companies in three NACE sectors (22.11, 22.19, 29.32: rubber tyres, other rubber products, and motor vehicle parts). Each firm was matched to its global corporate group so the analysis covers worldwide hiring activity. From this group, 16 firms with at least 2,000 job postings each were identified in TheirStack, producing the final dataset of 91,544 records from 2022 to the present.

### Pipeline

**Classification.** Each job description is processed by Gemini Flash, which returns a structured JSON with five fields: the type of role (business analytics, data engineering, AI/ML, agentic AI, etc.), the business area where the technology is applied (manufacturing, materials research, supply chain, etc.), a relevance score (0 = not AI related, 1 = adjacent, 2 = core AI role), the explicit technical tools mentioned, and an investment signal summarizing what the firm is trying to build.

**Hiring reports.** For each company, the classified postings are summarized into four views: hiring trend over time, geographic distribution of AI roles, seniority breakdown, and a map of where AI investment appears across different business areas and capability types.

**Capability reports.** The investment signals from core AI roles are grouped into broader capability areas per company (e.g., fleet analytics, demand forecasting, industrial automation, materials research). These become the input for the research agent.

**Research agent.** A Google deep research agent takes each company's capability report together with its annual report, and searches public sources (press releases, company pages, leadership interviews, investor communications) to expand and contextualize the hiring signals. The output is one structured company report per firm, organized into: quantitative investment baseline, capability by capability analysis, talent and operating model, and strategic priorities.

**Evaluation.** Each company report is evaluated for factual accuracy. The benchmark extracts every claim citing a public URL, crawls the cited source with Crawl4AI, selects the most relevant passages using TF-IDF similarity, and uses an LLM judge (Gemini Pro) to classify each claim as supported, needs_review, unsupported, or not_verifiable.

<p align="center">
  <img src="docs/evaluation-workflow.png" alt="Evaluation workflow" width="700">
</p>

**RAG chatbot.** The evaluated reports are embedded with multilingual-e5-small and indexed so they can be queried in natural language to compare companies, technologies, and sector trends.

---

## Evaluation results

392 public source claims were examined across the batch of company reports:

| Metric | Result |
|--------|--------|
| Claims judged | 292 |
| Supported | 174 |
| Needs review | 75 |
| Unsupported | 43 |
| Not verifiable | 100 |
| Grounding accuracy | 72.4% |
| Strict support rate | 59.6% |
| Unsupported rate | 14.7% |
| Verifiability rate | 74.5% |

---

## Limitations

Job postings signal a willingness to develop a capability, without demonstrating actual internal deployment. Some activities may also be understated as companies may not want to disclose them. In the absence of confirmation from sources such as investor disclosures, patents, or other formal documents, the results should not be interpreted as evidence of implementation.

---

## Companies analyzed

Michelin, Bridgestone, BASF, Adient, BorgWarner, Dana, Trelleborg, Continental, and others. All matched to their global corporate group for worldwide coverage.

---

## Replication

The pipeline code is not included in this repository (uses proprietary data from TheirStack). The full architecture, prompts, classification schema, and evaluation methodology are described in Chapter 3 of the thesis.
