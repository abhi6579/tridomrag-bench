"""
dataset_loader.py
=================
Downloads, formats, and saves QA pairs from three domains:
  - Healthcare : PubMedQA
  - Legal      : LegalBench
  - Finance    : FinanceBench

Output format (each domain):
  List[Dict] with keys:
    id, domain, question, answer, context, answer_type, difficulty, source_doc

Usage:
  from src.dataset_loader import DatasetLoader
  loader = DatasetLoader()
  loader.load_all()                  # downloads + saves all three
  data = loader.get_dataset("healthcare")   # returns List[Dict]
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datasets import load_dataset

logger = logging.getLogger(__name__)

# ── where datasets are saved locally ──────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DOMAINS  = ["healthcare", "legal", "finance"]

# ── answer type constants (matches paper Definition) ──────────────────────────
TYPE_T = "textual"    # free-form sentence
TYPE_N = "numerical"  # scalar, percentage, date
TYPE_E = "entity"     # named entity (drug, case, company)


class DatasetLoader:
    """
    Single entry-point to load all three benchmark datasets.
    Each dataset is cached as a JSON file under recallify/data/<domain>.json
    so you only download once.
    """

    def __init__(self, data_dir: Path = DATA_DIR, sample_size: int = 400):
        self.data_dir    = data_dir
        self.sample_size = sample_size          # 400 per domain = 1200 total
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def load_all(self) -> Dict[str, List[Dict]]:
        """Download + cache all three domains. Returns combined dict."""
        result = {}
        for domain in DOMAINS:
            logger.info(f"Loading {domain} dataset...")
            result[domain] = self._load_domain(domain)
            logger.info(f"  ✓ {domain}: {len(result[domain])} samples loaded")
        return result

    def get_dataset(self, domain: str) -> List[Dict]:
        """Load a single domain from cache (or download if missing)."""
        if domain not in DOMAINS:
            raise ValueError(f"Domain must be one of {DOMAINS}, got '{domain}'")
        return self._load_domain(domain)

    def get_all_combined(self) -> List[Dict]:
        """Returns all 1200 samples as a single flat list."""
        all_data = []
        for domain in DOMAINS:
            all_data.extend(self._load_domain(domain))
        return all_data

    def stats(self) -> Dict:
        """Print a quick stats table — call after load_all()."""
        stats = {}
        for domain in DOMAINS:
            cache_path = self._cache_path(domain)
            if cache_path.exists():
                data = self._read_cache(cache_path)
                type_counts = {TYPE_T: 0, TYPE_N: 0, TYPE_E: 0}
                diff_counts = {"basic": 0, "intermediate": 0, "expert": 0}
                for d in data:
                    type_counts[d.get("answer_type", TYPE_T)] += 1
                    diff_counts[d.get("difficulty", "basic")] += 1
                stats[domain] = {
                    "total"      : len(data),
                    "by_type"    : type_counts,
                    "by_difficulty": diff_counts,
                }
            else:
                stats[domain] = {"total": 0, "status": "not downloaded"}
        return stats

    # ─────────────────────────────────────────────────────────────────────────
    #  INTERNAL — per-domain loaders
    # ─────────────────────────────────────────────────────────────────────────

    def _load_domain(self, domain: str) -> List[Dict]:
        cache = self._cache_path(domain)
        if cache.exists():
            logger.info(f"  [{domain}] Cache hit → {cache}")
            return self._read_cache(cache)
        logger.info(f"  [{domain}] Cache miss → downloading...")
        loaders = {
            "healthcare": self._load_pubmedqa,
            "legal"     : self._load_legalbench,
            "finance"   : self._load_financebench,
        }
        data = loaders[domain]()
        self._write_cache(cache, data)
        return data

    # ── Healthcare ─────────────────────────────────────────────────────────

    def _load_pubmedqa(self) -> List[Dict]:
        """
        PubMedQA — biomedical research QA.
        HuggingFace: 'qiaojin/PubMedQA', config 'pqa_labeled'
        """
        try:
            ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled",
                              split="train")
        except Exception as e:
            logger.error(f"PubMedQA download failed: {e}")
            raise

        samples = []
        indices = random.sample(range(len(ds)), min(self.sample_size, len(ds)))

        for idx in indices:
            row     = ds[idx]
            context = " ".join(row["context"]["contexts"])
            answer  = row.get("long_answer", str(row.get("final_decision", "")))

            samples.append({
                "id"          : f"hc_{idx:04d}",
                "domain"      : "healthcare",
                "question"    : row["question"],
                "answer"      : answer,
                "context"     : context[:2000],
                "answer_type" : self._infer_type(answer),
                "difficulty"  : self._infer_difficulty(row["question"]),
                "source_doc"  : f"PubMed_{row.get('pubid', idx)}",
            })

        logger.info(f"  PubMedQA: {len(samples)} samples prepared")
        return samples

    # ── Legal ──────────────────────────────────────────────────────────────

    def _load_legalbench(self) -> List[Dict]:
        """
        LegalBench — legal reasoning tasks.
        Uses 'nguyen/legal_bench' tasks via HuggingFace.
        Falls back to synthetic samples if dataset is inaccessible.
        """
        samples = []
        want    = self.sample_size   # define BEFORE try so except can use it

        # Try public legal datasets in order of availability
        public_datasets = [
            ("isaacus/open-australian-legal-qa", "default", "train"),
            ("joelito/legal_mc_questions", None,   "test"),
            ("coastalcph/legalBench",      "abercrombie", "test"),
        ]

        for hf_path, config, split in public_datasets:
            if len(samples) >= self.sample_size:
                break
            try:
                ds = load_dataset(hf_path, config, split=split) \
                     if config else load_dataset(hf_path, split=split)
                want = self.sample_size - len(samples)
                indices = random.sample(range(len(ds)), min(want, len(ds)))
                for idx in indices:
                    row      = ds[idx]
                    question = str(row.get("question", row.get("input",  "")))
                    answer   = str(row.get("answer",   row.get("output", row.get("label", ""))))
                    # For isaacus/open-australian-legal-qa, use source text as context
                    if isinstance(row.get("source"), dict):
                        context = str(row["source"].get("text", ""))
                    else:
                        context = str(row.get("context", row.get("text", row.get("passage", ""))))
                    samples.append({
                        "id"          : f"lg_{idx:04d}",
                        "domain"      : "legal",
                        "question"    : question,
                        "answer"      : answer,
                        "context"     : context[:2000],
                        "answer_type" : self._infer_type(answer),
                        "difficulty"  : self._infer_difficulty(question),
                        "source_doc"  : f"LegalBench_{hf_path}",
                    })
                logger.info(f"  LegalBench via {hf_path}: {len(samples)} samples")
                break   # success — stop trying other datasets
            except Exception as e:
                logger.warning(f"  {hf_path} not accessible: {e}")

        # Fallback — use synthetic samples if all downloads failed
        if len(samples) < self.sample_size:
            needed = self.sample_size - len(samples)
            logger.warning(f"  Using {needed} fallback legal samples")
            samples.extend(self._fallback_legal(needed))

        samples = samples[:self.sample_size]
        logger.info(f"  LegalBench: {len(samples)} samples prepared")
        return samples

    # ── Finance ────────────────────────────────────────────────────────────

    def _load_financebench(self) -> List[Dict]:
        """
        FinanceBench + financial-qa-10K combined.
        """
        samples = []
        try:
            ds1 = load_dataset("PatronusAI/financebench", split="train")
            for idx in range(len(ds1)):
                row = ds1[idx]
                samples.append({
                    "id"          : f"fn_{idx:04d}",
                    "domain"      : "finance",
                    "question"    : str(row.get("question", "")),
                    "answer"      : str(row.get("answer", "")),
                    "context"     : (lambda e: " ".join([x.get("evidence_text","") for x in e]) if isinstance(e, list) else str(e))( row.get("evidence", row.get("context", "")))[:2000],
                    "answer_type" : self._infer_type(str(row.get("answer", ""))),
                    "difficulty"  : self._infer_difficulty(str(row.get("question", ""))),
                    "source_doc"  : str(row.get("doc_name", row.get("company", f"FinDoc_{idx}"))),
                })
            logger.info(f"  FinanceBench: {len(samples)} samples loaded")
        except Exception as e:
            logger.warning(f"FinanceBench failed: {e}")
        if len(samples) < self.sample_size:
            try:
                ds2 = load_dataset("virattt/financial-qa-10K", split="train")
                needed = self.sample_size - len(samples)
                indices = random.sample(range(len(ds2)), min(needed, len(ds2)))
                for i, idx in enumerate(indices):
                    row = ds2[idx]
                    samples.append({
                        "id"          : f"fq_{i:04d}",
                        "domain"      : "finance",
                        "question"    : str(row.get("question", "")),
                        "answer"      : str(row.get("answer", "")),
                        "context"     : str(row.get("context", ""))[:2000],
                        "answer_type" : self._infer_type(str(row.get("answer", ""))),
                        "difficulty"  : self._infer_difficulty(str(row.get("question", ""))),
                        "source_doc"  : f"{row.get('ticker', 'Unknown')}_{row.get('filing', '')}",
                    })
                logger.info(f"  financial-qa-10K: total {len(samples)} samples")
            except Exception as e:
                logger.warning(f"financial-qa-10K failed: {e}")
        if not samples:
            return self._fallback_finance(self.sample_size)
        logger.info(f"  Finance total: {len(samples)} samples")
        return samples

    def _infer_type(self, answer: str) -> str:
        """
        Heuristic answer-type classifier.
        Type-N: starts with digit / $ / % / contains numeric pattern
        Type-E: short (≤4 words), all title-case words → entity
        Type-T: everything else
        """
        import re
        a = answer.strip()

        # Numerical indicators
        num_pattern = re.compile(
            r"^\$|^\d|%$|\d+\.\d+|\d{1,3}(,\d{3})+|"
            r"(billion|million|thousand|percent|\bQ[1-4]\b)",
            re.IGNORECASE
        )
        if num_pattern.search(a):
            return TYPE_N

        # Entity: short answer, likely a name / case / drug
        words = a.split()
        if len(words) <= 4 and sum(1 for w in words if w[0].isupper()) >= len(words) - 1:
            return TYPE_E

        return TYPE_T

    def _infer_difficulty(self, question: str) -> str:
        """
        Heuristic difficulty tagger based on question length + keywords.
        Basic       : short, single concept
        Intermediate: medium, comparison / cause-effect
        Expert      : long, multi-step reasoning / calculate / analyse
        """
        q   = question.lower()
        wc  = len(question.split())
        expert_kw = [
            "calculate", "analyse", "analyze", "compare", "derive",
            "evaluate", "interpret", "differentiate", "synthesize",
            "what is the relationship", "why did", "how does",
            "what would happen", "what are the implications"
        ]
        if wc > 30 or any(k in q for k in expert_kw):
            return "expert"
        elif wc > 15:
            return "intermediate"
        return "basic"

    # ── Fallback samples (if HuggingFace download fails) ──────────────────

    def _fallback_legal(self, n: int) -> List[Dict]:
        """Minimal synthetic samples so pipeline doesn't break during dev."""
        templates = [
            {"question": "Does the clause restrict sublicensing?",
             "answer": "Yes", "context": "Licensee shall not sublicense..."},
            {"question": "What is the governing law of this agreement?",
             "answer": "Delaware", "context": "This agreement shall be governed by..."},
            {"question": "Can the contract be terminated without cause?",
             "answer": "Yes, with 30 days written notice.",
             "context": "Either party may terminate..."},
        ]
        samples = []
        for i in range(n):
            t = templates[i % len(templates)]
            samples.append({
                "id"          : f"lg_fallback_{i:04d}",
                "domain"      : "legal",
                "question"    : t["question"],
                "answer"      : t["answer"],
                "context"     : t["context"],
                "answer_type" : self._infer_type(t["answer"]),
                "difficulty"  : "basic",
                "source_doc"  : "FallbackLegal",
            })
        return samples

    def _fallback_finance(self, n: int) -> List[Dict]:
        templates = [
            {"question": "What was Apple's revenue in Q3 2023?",
             "answer": "$81.8 billion",
             "context": "Apple reported total net sales of $81.8 billion..."},
            {"question": "What is the net income for FY2022?",
             "answer": "$99.8 billion",
             "context": "Net income for fiscal year 2022 was $99.8 billion."},
        ]
        samples = []
        for i in range(n):
            t = templates[i % len(templates)]
            samples.append({
                "id"          : f"fn_fallback_{i:04d}",
                "domain"      : "finance",
                "question"    : t["question"],
                "answer"      : t["answer"],
                "context"     : t["context"],
                "answer_type" : self._infer_type(t["answer"]),
                "difficulty"  : "basic",
                "source_doc"  : "FallbackFinance",
            })
        return samples

    # ── Cache helpers ──────────────────────────────────────────────────────

    def _cache_path(self, domain: str) -> Path:
        return self.data_dir / f"{domain}.json"

    def _write_cache(self, path: Path, data: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"  Cached → {path}")

    def _read_cache(self, path: Path) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick test — run directly: python src/dataset_loader.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    loader = DatasetLoader(sample_size=10)   # small=10 for quick test
    print("\n── Loading all domains (10 samples each for speed) ──")
    all_data = loader.load_all()

    print("\n── Stats ──")
    pprint.pprint(loader.stats())

    print("\n── Sample healthcare record ──")
    pprint.pprint(all_data["healthcare"][0])

    print("\n── Sample finance record ──")
    pprint.pprint(all_data["finance"][0])

    print("\n✓ dataset_loader.py is working correctly.")