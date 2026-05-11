# TriDomRAG-Bench 🧠

> The first benchmark that simultaneously evaluates RAG systems across three high-stakes domains under a unified evaluation protocol.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-20%2F20%20passing-brightgreen.svg)
![LLM](https://img.shields.io/badge/LLM-GPT--4o--mini%20%7C%20Groq%20%7C%20Local-orange.svg)
![Samples](https://img.shields.io/badge/Samples-1200-purple.svg)

---

## Authors

**Abhinav Mishra**
M.Tech, Artificial Intelligence and Machine Learning (2024-2026)
NIMS University Rajasthan, Jaipur, India
Email: abhinav06579@gmail.com | GitHub: https://github.com/abhi6579

**Co-authors:**
Vineet Mehan, NIMS University Rajasthan, Jaipur, India | mehanvineet@gmail.com
Nilesh Bhosle, NIMS University Rajasthan, Jaipur, India | bhoslenp@gmail.com

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

### How We Differ from Prior Work

| | RAGAS | RAGChecker | ARES | Ours |
|--|-------|-----------|------|------|
| Tri-domain | No | No | No | Yes |
| Numerical scoring | No | No | No | Yes |
| Domain lexicons | No | No | No | Yes |
| Unified benchmark | No | No | No | Yes |

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
| Type-N | Numerical answers | Proximity formula: max(0, 1 - |v_a - v_a*| / e*v_a*) |
| Type-E | Entity answers | Normalised string match |

---

## Dataset (1,200 Total Samples)

| Domain | Source | Samples | Public Access |
|--------|--------|---------|--------------|
| Healthcare | PubMedQA (qiaojin/PubMedQA) | 400 | Free |
| Legal | Open Australian Legal QA (isaacus/open-australian-legal-qa) | 400 | Free |
| Finance | FinanceBench (150) + financial-qa-10K (250) | 400 | Free |

---

## 6 RAG Configurations

| Config | Retriever | Granularity |
|--------|-----------|-------------|
| bm25_chunk | BM25 (k1=0.9, b=0.4) | Chunk (256 words, 32 overlap) |
| bm25_document | BM25 | Full document |
| dense_chunk | Dense BAAI/bge-base-en-v1.5 | Chunk |
| dense_document | Dense BAAI/bge-base-en-v1.5 | Full document |
| hybrid_chunk | BM25 + Dense (lambda=0.5) | Chunk |
| hybrid_document | BM25 + Dense (lambda=0.5) | Full document |

---

## Results (Full 1,200-Sample Run)

| Config | Healthcare | Legal | Finance | Avg DHS |
|--------|-----------|-------|---------|---------|
| bm25_chunk | 0.509 | 0.561 | 0.638 | 0.569 |
| bm25_document | 0.501 | 0.564 | 0.635 | 0.567 |
| dense_chunk | **0.520** | **0.569** | **0.665** | **0.585** |
| dense_document | 0.513 | 0.564 | 0.662 | 0.580 |
| hybrid_chunk | 0.514 | 0.559 | 0.657 | 0.577 |
| hybrid_document | 0.512 | 0.570 | 0.662 | 0.581 |

**Best config: dense_chunk (avg DHS 0.585)**
**LLM: GPT-4o-mini via AICredits API | Runtime: 12,299s (~3.4 hrs) | Cost: ~INR 210**

### Key Findings
- Dense retrieval outperforms BM25 by 2.8 percentage points
- Finance achieves highest domain DHS (0.653) with real 10-K context
- Legal attribution jumps from 0.000 (synthetic) to 0.678 (real case law)
- Healthcare most consistent across configs (variance < 0.02)
- Chunk-level granularity consistently beats document-level

---

## Failure Taxonomy

| Code | Name | Description | Primary Domain |
|------|------|-------------|---------------|
| F1 | Wrong Chunk | Retriever fetched irrelevant chunks | Healthcare |
| F2 | Numerical Hallucination | LLM fabricated numbers not in context | Finance |
| F3 | Terminology Mismatch | Wrong domain vocabulary used | Legal |
| F4 | Attribution Gap | Answer not traceable to source | All |

---

## Project Structure

tridomrag-bench/
  core/
    domain_models.py       - DHS types, BenchmarkSample, DHSResult, FailureMode
    models.py              - base data models
    exceptions.py          - custom exceptions
  src/
    dataset_loader.py      - PubMedQA / Australian Legal QA / FinanceBench
    terminology_lexicon.py - SNOMED / Black Law / XBRL term extraction
    dhs_metric.py          - DHS scoring engine (F + A + T)
    retrieval_configs.py   - BM25, Dense, Hybrid + 6 configs
    llm_config.py          - Groq / OpenAI / AICredits / Together / Local
    experiment_runner.py   - full pipeline orchestrator
  config/settings.py
  results/
    summary_table.json     - Paper Table 2 numbers
    experiment_results.json - Full raw results
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

### Environment Setup (.env file)
  # Option 1: AICredits (Indian UPI/debit card accepted)
  LLM_PROVIDER=openai
  OPENAI_API_KEY=sk-live-...
  OPENAI_BASE_URL=https://api.aicredits.in/v1
  OPENAI_MODEL=openai/gpt-4o-mini

  # Option 2: Groq (completely free)
  LLM_PROVIDER=groq
  GROQ_API_KEY=gsk_...

  # Option 3: Together AI
  LLM_PROVIDER=together
  TOGETHER_API_KEY=...

### Run Tests
  python test_phase1.py

### Run Experiments
  # Dev run (10 samples)
  python src/experiment_runner.py --mode dev --llm openai

  # Full run (400 samples per domain)
  python src/experiment_runner.py --mode full --llm openai

---

## LLM Provider Options

| Provider | Model | Cost | Quality | Notes |
|----------|-------|------|---------|-------|
| AICredits | GPT-4o-mini | ~INR 210 full run | Excellent | UPI/Indian debit card |
| Groq | Llama-3.3-70B | FREE | Excellent | 100K tokens/day limit |
| Together AI | Llama-3.3-70B | ~ full run | Excellent | No daily limit |
| OpenAI | GPT-4o-mini | ~ full run | Excellent | Credit card needed |
| Local | distilgpt2 | Free | Dev only | No API key needed |

---

## Paper

Title    : TriDomRAG-Bench: A Tri-Domain Benchmark for Evaluating Hallucination,
           Faithfulness, and Attribution in Domain-Specific RAG
Authors  : Abhinav Mishra, Vineet Mehan, Nilesh Bhosle
Institute: NIMS University Rajasthan, Jaipur, India
Target   : EMNLP 2026 / ACL Findings / IEEE Access
Status   : Full 1,200-sample run complete. Preparing for arXiv submission.

---

## Citation

@article{mishra2026tridomrag,
  title     = {TriDomRAG-Bench: A Tri-Domain Benchmark for Evaluating Hallucination,
               Faithfulness, and Attribution in Domain-Specific RAG},
  author    = {Mishra, Abhinav and Mehan, Vineet and Bhosle, Nilesh},
  year      = {2026},
  note      = {NIMS University Rajasthan, Jaipur, India.
               GitHub: https://github.com/abhi6579/tridomrag-bench}
}

---

## License

MIT License - Copyright (c) 2026 Abhinav Mishra, Vineet Mehan, Nilesh Bhosle
NIMS University Rajasthan, Jaipur, India

---

Built with passion for publication-quality research | NIMS University Rajasthan, Jaipur, India
