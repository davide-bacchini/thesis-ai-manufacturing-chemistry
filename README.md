# Mapping AI Capability Building from Public Company Data

Large industrial companies rarely disclose in detail what they are building with AI. Annual reports describe strategy, but they often stay high level, while internal projects are usually not visible from the outside.

I developed a RAG chatbot that helps R&D managers, strategy teams, and analysts ask questions about what AI capabilities industrial companies appear to be developing, which business areas they relate to, and whether those signals are supported by public evidence.

The chatbot is built on evaluated company reports. To create them, I built a pipeline that starts from job postings available for each company, classifies them by technical relevance and business area, extracts the capabilities each company appears to be building, and then searches public sources such as annual reports, investor material, company pages, press releases, and interviews to support or contextualize those findings.

The analysis covers 91,544 job postings from 16 global manufacturing and chemistry groups.

BSc Thesis, Bocconi University, 2026. [📄 Full thesis (PDF)](./thesis.pdf)

## What the pipeline does

The pipeline starts from raw job descriptions and extracts structured signals about:

| Signal | Why it matters |
| :-- | :-- |
| Role type | Shows whether the company is hiring for analytics, data, software, AI, automation, or research roles |
| Business area | Shows where the capability is being built, such as manufacturing, supply chain, R&D, IT, or corporate functions |
| AI relevance | Separates generic roles from roles that directly contribute to AI capability building |
| Technical tools | Captures the platforms, methods, and systems mentioned in the posting |
| Investment signal | Summarizes what the company appears to be trying to build |

These signals are grouped by company to identify patterns in hiring, geography, seniority, and business focus.

The key output is a map of visible capability building. The pipeline does not assume that a capability is already deployed only because a company is hiring for it. Instead, job postings are treated as a first signal of where the company is investing talent and technical resources.

## From hiring signals to company reports

For each company, the pipeline first identifies the strongest capability signals from core technical roles. These signals are then used to guide the search for supporting evidence in public sources.

The company report connects three types of evidence:

| Source | Role in the analysis |
| :-- | :-- |
| Job postings | Show hiring demand and capability building signals |
| Annual reports | Show strategic priorities and management narrative |
| Public sources | Add evidence from company pages, press releases, interviews, and investor material |

The final reports explain which AI capabilities are visible for each company, which business areas they relate to, and whether public sources provide evidence that the company is actually discussing, investing in, or deploying similar technologies.

Examples of capability areas include predictive maintenance, industrial automation, demand forecasting, materials research, supply chain analytics, and connected product services.

## Reliability check

Because generated reports can easily overstate what the sources say, I added a claim level evaluation step.

Each factual claim with a public source is checked against the cited page. The system retrieves the source, extracts relevant passages, and classifies the claim as supported, needs review, unsupported, or not verifiable.

<p align="center">
  <img src="./evaluation-workflow.png" alt="Evaluation workflow" width="700">
</p>

Across the evaluated reports:

| Metric | Result |
| :-- | --: |
| Public source claims examined | 392 |
| Claims judged | 292 |
| Supported claims | 174 |
| Needs review | 75 |
| Unsupported | 43 |
| Not verifiable | 100 |
| Grounding accuracy | 72.4% |
| Strict support rate | 59.6% |

This step makes the reports more useful as evidence based company analysis rather than simple generated summaries.

## Searchable company reports

The evaluated company reports are indexed in a RAG chatbot, so users can query them in natural language.

The chatbot helps R&D managers and strategy teams understand what type of AI capabilities a company appears to be developing, which technologies are connected to specific business areas, and how different firms compare across manufacturing, R&D, supply chain, operations, and customer facing activities.

Example questions:

```text
Which companies show stronger signals in predictive maintenance?

How does Michelin compare with Bridgestone in AI related hiring?

Which firms are investing in materials research capabilities?

Where is AI more visible in operations than in customer facing products?
```

## Scope and limitations

The analysis covers 16 large industrial groups selected from rubber, tyres, and automotive components, including companies such as Michelin, Bridgestone, BASF, Adient, BorgWarner, Dana, Trelleborg, and Continental.

Job postings are treated as signals of capability building, not as proof of internal deployment. Some projects may also be missing because companies do not disclose strategically sensitive work.

The repository does not include the full pipeline code because the job posting data comes from TheirStack, a proprietary data source. It documents the architecture, prompts, classification schema, evaluation method, and thesis methodology.
