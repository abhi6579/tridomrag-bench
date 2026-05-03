# TriDomRAG-Bench ??

> The first benchmark that simultaneously evaluates RAG systems across three high-stakes domains under a unified evaluation protocol.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-20%2F20%20passing-brightgreen.svg)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20OpenAI%20%7C%20AICredits-orange.svg)

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

| Domain | alpha (Faithfulness) | beta (Attribution) | gamma (Terminology) |
|--------|---------------------|-------------------|-------------------|
| Healthcare | 0.40 | 0.30 | 0.30 |
| Legal | 0.35 | 0.45 | 0.20 |
| Finance | 0.30 | 0.25 | 0.45 |

### Answer Type Handling

| Type | Description | Scoring Method |
|------|-------------|---------------|
| Type-T | Textual answers | NLI-based / token overlap |
| Type-N | Numerical answers | Proximity formula with tolerance |
| Type-E | Entity answers | Normalised string match |

---

## Dataset

| Domain | Source | Samples | Answer Types |
|--------|--------|---------|-------------|
| Healthcare | PubMedQA | 400 | Textual, Entity |
| Legal | LegalBench | 400 | Textual, Entity |
| Finance | FinanceBench | 400 | Numerical, Textual |

---

## 6 RAG Configurations

| Config | Retriever | Granularity |
|--------|-----------|-------------|
| bm25_chunk | BM25 | Chunk-level |
| bm25_document | BM25 | Document-level |
| dense_chunk | Dense (BGE) | Chunk-level |
| dense_document | Dense (BGE) | Document-level |
| hybrid_chunk | BM25+Dense | Chunk-level |
| hybrid_document | BM25+Dense | Document-level |

Embedding Model: BAAI/bge-base-en-v1.5

---

## Preliminary Results (10 samples, GPT-4o-mini via AICredits)

| Config | Healthcare | Legal | Finance | Avg DHS |
|--------|-----------|-------|---------|---------|
| bm25_chunk | 0.578 | 0.260 | 0.375 | 0.404 |
| bm25_document | 0.563 | 0.260 | 0.418 | 0.414 |
| dense_chunk | 0.519 | 0.260 | 0.383 | 0.387 |
| dense_document | 0.490 | 0.260 | 0.383 | 0.378 |
| hybrid_chunk | 0.521 | 0.260 | 0.427 | 0.403 |
| hybrid_document | 0.520 | 0.260 | 0.419 | 0.400 |

Full 400-sample results coming soon.

---

## Failure Taxonomy

| Code | Name | Description |
|------|------|-------------|
| F1 | Wrong Chunk | Retriever fetched irrelevant chunks |
| F2 | Numerical Hallucination | LLM fabricated numbers not in context |
| F3 | Terminology Mismatch | Wrong domain vocabulary used |
| F4 | Attribution Gap | Answer not traceable to any source |

---

## Project Structure

tridomrag-bench/
  core/
    domain_models.py       - DHS types, BenchmarkSample, DHSResult, FailureMode
    models.py              - base data models
    exceptions.py          - custom exceptions
  src/
    dataset_loader.py      - downloads PubMedQA / LegalBench / FinanceBench
    terminology_lexicon.py - SNOMED / Black Law / XBRL term extraction
    dhs_metric.py          - DHS scoring engine (F + A + T)
    retrieval_configs.py   - BM25, Dense, Hybrid + 6 configs
    llm_config.py          - Groq / OpenAI / AICredits LLM switch
    experiment_runner.py   - full pipeline orchestrator
  config/settings.py
  test_phase1.py           - 20/20 tests passing
  requirements.txt         - universal, works on Windows/Linux/Colab

---

## Quick Start

### Windows
  git clone https://github.com/abhi6579/tridomrag-bench.git
  cd tridomrag-bench
  py -3.12 -m venv venv
  venv\Scripts\activate
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt

### Linux / WSL / Colab
  git clone https://github.com/abhi6579/tridomrag-bench.git
  cd tridomrag-bench
  python3.12 -m venv venv
  source venv/bin/activate
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt

### Create .env file
  LLM_PROVIDER=openai
  OPENAI_API_KEY=sk-...
  OPENAI_BASE_URL=https://api.aicredits.in/v1
  OPENAI_MODEL=openai/gpt-4o-mini

  OR use Groq (free):
  LLM_PROVIDER=groq
  GROQ_API_KEY=gsk_...

### Run tests
  python test_phase1.py

### Run experiments
  python src/experiment_runner.py --mode dev --llm openai
  python src/experiment_runner.py --mode full --llm openai

---

## LLM Provider Options

| Provider | Model | Cost | Quality | How to get |
|----------|-------|------|---------|------------|
| AICredits | GPT-4o-mini | ~Rs 210 full run | Excellent | aicredits.in (UPI/Indian debit card) |
| Groq | Llama-3.3-70B | FREE | Excellent | console.groq.com |
| OpenAI | GPT-4o-mini | ~ full run | Excellent | platform.openai.com |
| Local | distilgpt2 | Free | Dev only | auto-downloaded |

---

## Paper

Title    : TriDomRAG-Bench: A Tri-Domain Benchmark for Evaluating Hallucination,
           Faithfulness, and Attribution in Domain-Specific RAG
Author   : Abhinav Mishra (sole author)
Supervisor: Dr. Vineet Mehan, NIET, NIMS University, Jaipur, India
Target   : EMNLP 2026 / ACL Findings
Status   : Pipeline complete, full 400-sample run in progress.

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

MIT License - Copyright (c) 2026 Abhinav Mishra
NIET (NIMS Institute of Engineering and Technology), NIMS University, Jaipur, India

---

Built with passion by Abhinav Mishra | NIET, NIMS University, Jaipur, India
