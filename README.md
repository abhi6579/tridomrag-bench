# TriDomRAG-Bench 🧠

> The first benchmark that simultaneously evaluates RAG systems across three high-stakes domains under a unified evaluation protocol.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-20%2F20%20passing-brightgreen.svg)](test_phase1.py)
[![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20OpenAI%20%7C%20Local-orange.svg)](src/llm_config.py)

---

## Author

**Abhinav Mishra**
M.Tech, Artificial Intelligence and Machine Learning (2024-2026)
NIET (NIMS Institute of Engineering and Technology), NIMS University, Jaipur, India
Supervised by: Dr. Vineet Mehan, Assistant Professor, Dept. of AI and ML, NIET
Email: abhinav06579@gmail.com | GitHub: https://github.com/abhi6579

---

## What is TriDomRAG-Bench?

RAG systems are increasingly deployed in high-stakes domains like healthcare, legal, and finance.
Yet existing benchmarks evaluate them in isolation, use metrics that fail on numerical answers,
and completely ignore domain-specific vocabulary.

TriDomRAG-Bench solves all three problems with a single unified framework.

### The 3 Gaps We Fill

| Gap | Problem | Our Solution |
|-----|---------|-------------|
| No tri-domain benchmark | Systems evaluated in isolation | 1,200 QA pairs across 3 domains |
| RAGAS fails on numerical answers | 83.5% of finance samples fail | Domain Hallucination Score (DHS) |
| Domain terminology ignored | Wrong vocabulary accepted as correct | T-score with SNOMED / Black Law / XBRL |

---

## Domain Hallucination Score (DHS)

DHS(a, C, a*, d) = alpha_d * F(a,C) + beta_d * A(a,C) + gamma_d * T(a,a*,d)

### Domain Weights

| Domain     | alpha (Faithfulness) | beta (Attribution) | gamma (Terminology) |
|------------|---------------------|--------------------|---------------------|
| Healthcare | 0.40                | 0.30               | 0.30                |
| Legal      | 0.35                | 0.45               | 0.20                |
| Finance    | 0.30                | 0.25               | 0.45                |

### Answer Type Handling

| Type   | Description       | Scoring Method                   |
|--------|-------------------|----------------------------------|
| Type-T | Textual answers   | NLI-based / token overlap        |
| Type-N | Numerical answers | Proximity formula with tolerance |
| Type-E | Entity answers    | Normalised string match          |

---

## Dataset

| Domain     | Source       | Samples | Answer Types       |
|------------|--------------|---------|--------------------|
| Healthcare | PubMedQA     | 400     | Textual, Entity    |
| Legal      | LegalBench   | 400     | Textual, Entity    |
| Finance    | FinanceBench | 400     | Numerical, Textual |

---

## 6 RAG Configurations

| Config          | Retriever   | Granularity    |
|-----------------|-------------|----------------|
| bm25_chunk      | BM25        | Chunk-level    |
| bm25_document   | BM25        | Document-level |
| dense_chunk     | Dense (BGE) | Chunk-level    |
| dense_document  | Dense (BGE) | Document-level |
| hybrid_chunk    | BM25+Dense  | Chunk-level    |
| hybrid_document | BM25+Dense  | Document-level |

Embedding Model: BAAI/bge-base-en-v1.5

---

## Results

Full results from 400-sample run using Llama-3.3-70B via Groq API.
Results table will be updated after full experiment run.

---

## Failure Taxonomy

| Code | Name                    | Description                             |
|------|-------------------------|-----------------------------------------|
| F1   | Wrong Chunk             | Retriever fetched irrelevant chunks     |
| F2   | Numerical Hallucination | LLM fabricated numbers not in context  |
| F3   | Terminology Mismatch    | Wrong domain vocabulary used            |
| F4   | Attribution Gap         | Answer not traceable to any source      |

---

## Project Structure

tridomrag-bench/
  core/
    domain_models.py        - DHS types, BenchmarkSample, DHSResult, FailureMode
    models.py               - base data models
    exceptions.py           - custom exceptions
  src/
    dataset_loader.py       - downloads PubMedQA / LegalBench / FinanceBench
    terminology_lexicon.py  - SNOMED / Black Law / XBRL term extraction
    dhs_metric.py           - DHS scoring engine (F + A + T)
    retrieval_configs.py    - BM25, Dense, Hybrid + 6 configs
    llm_config.py           - Groq / OpenAI / local LLM switch
    experiment_runner.py    - full pipeline orchestrator
  config/settings.py
  test_phase1.py            - 20/20 tests passing
  demo.py                   - 60-second live demo
  colab_run.ipynb           - one-click Google Colab notebook
  requirements.txt

---

## Quick Start

  git clone https://github.com/abhi6579/tridomrag-bench.git
  cd tridomrag-bench
  python -m venv venv && source venv/bin/activate
  pip install -r requirements.txt

Create .env file:
  LLM_PROVIDER=groq
  GROQ_API_KEY=gsk_...

Run tests:
  python test_phase1.py

Run experiments:
  python src/experiment_runner.py --mode dev --config hybrid_document --llm groq
  python src/experiment_runner.py --mode full --llm groq

---

## Run on Google Colab (Free GPU)

Open colab_run.ipynb in Google Colab for a one-click full run with free T4 GPU.
https://colab.research.google.com/github/abhi6579/tridomrag-bench/blob/main/colab_run.ipynb

---

## Requirements

| Package               | Version | Purpose            |
|-----------------------|---------|--------------------|
| torch                 | 2.11.0  | Deep learning      |
| transformers          | 5.5.0   | HuggingFace models |
| sentence-transformers | 5.3.0   | BGE embeddings     |
| chromadb              | 1.5.5   | Vector store       |
| datasets              | 4.8.4   | Dataset loading    |
| pydantic              | 2.12.5  | Data validation    |
| groq                  | 1.1.2   | Free LLM API       |
| rank-bm25             | 0.2.2   | Sparse retrieval   |
| python-dotenv         | 1.x     | .env loading       |

---

## LLM Provider Comparison

| Provider | Model         | Cost  | Quality   | Use Case      |
|----------|---------------|-------|-----------|---------------|
| Groq     | Llama-3.3-70B | FREE  | Excellent | Paper results |
| OpenAI   | GPT-3.5-Turbo | ~$1-2 | Excellent | Alternative   |
| Local    | distilgpt2    | Free  | Weak      | Dev only      |

---

## Paper

Title    : TriDomRAG-Bench: A Tri-Domain Benchmark for Evaluating Hallucination,
           Faithfulness, and Attribution in Domain-Specific RAG
Author   : Abhinav Mishra (sole author)
Supervisor: Dr. Vineet Mehan, NIET, NIMS University, Jaipur, India
Target   : EMNLP 2026 / ACL Findings
Status   : Pipeline complete, full run in progress. arXiv preprint coming soon.

---

## Citation

@article{mishra2026tridomrag,
  title  = {TriDomRAG-Bench: A Tri-Domain Benchmark for Evaluating Hallucination,
            Faithfulness, and Attribution in Domain-Specific RAG},
  author = {Mishra, Abhinav},
  year   = {2026},
  note   = {Supervised by Dr. Vineet Mehan, NIET, NIMS University, Jaipur, India}
}

---

## License

MIT License
Copyright (c) 2026 Abhinav Mishra
NIET (NIMS Institute of Engineering and Technology), NIMS University, Jaipur, India

---

Built with love by Abhinav Mishra | NIET, NIMS University, Jaipur, India
