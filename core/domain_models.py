"""
domain_models.py
================
Extends recallify/core/models.py with DHS-specific types.
DO NOT replace models.py — import from both.

New types added:
  - AnswerType       : TYPE_T / TYPE_N / TYPE_E enum
  - DomainType       : healthcare / legal / finance enum
  - Difficulty       : basic / intermediate / expert enum
  - DHSComponents    : sub-scores F, A, T before weighting
  - DHSResult        : final DHS score + breakdown per sample
  - BenchmarkSample  : one QA pair from the benchmark
  - ExperimentResult : one RAG config run result
  - FailureMode      : taxonomy F1-F4

Usage:
  from core.domain_models import DHSResult, BenchmarkSample, FailureMode
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ─────────────────────────────────────────────────────────────────────────────
#  ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class AnswerType(str, Enum):
    TEXTUAL   = "textual"      # Type-T: free-form sentence
    NUMERICAL = "numerical"    # Type-N: scalar / %, date, money
    ENTITY    = "entity"       # Type-E: named entity


class DomainType(str, Enum):
    HEALTHCARE = "healthcare"
    LEGAL      = "legal"
    FINANCE    = "finance"


class Difficulty(str, Enum):
    BASIC        = "basic"
    INTERMEDIATE = "intermediate"
    EXPERT       = "expert"


class RetrievalStrategy(str, Enum):
    BM25   = "bm25"
    DENSE  = "dense"
    HYBRID = "hybrid"


class Granularity(str, Enum):
    CHUNK    = "chunk"
    DOCUMENT = "document"


class FailureMode(str, Enum):
    F1_WRONG_CHUNK_RETRIEVED  = "F1_wrong_chunk"       # retriever pulled wrong chunk
    F2_NUMERICAL_HALLUCINATION = "F2_numerical_halluc"  # numeric answer fabricated
    F3_TERMINOLOGY_MISMATCH   = "F3_terminology"        # wrong domain vocab used
    F4_ATTRIBUTION_GAP        = "F4_attribution"        # no traceable source span


# ─────────────────────────────────────────────────────────────────────────────
#  CORE DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkSample:
    """
    One QA pair from the tri-domain benchmark.
    Mirrors the JSON structure saved by dataset_loader.py.
    """
    id          : str
    domain      : DomainType
    question    : str
    answer      : str               # gold answer
    context     : str               # retrieved / reference context
    answer_type : AnswerType        = AnswerType.TEXTUAL
    difficulty  : Difficulty        = Difficulty.BASIC
    source_doc  : str               = ""

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkSample":
        return cls(
            id          = d["id"],
            domain      = DomainType(d["domain"]),
            question    = d["question"],
            answer      = d["answer"],
            context     = d["context"],
            answer_type = AnswerType(d.get("answer_type", "textual")),
            difficulty  = Difficulty(d.get("difficulty",  "basic")),
            source_doc  = d.get("source_doc", ""),
        )


@dataclass
class DHSComponents:
    """
    Raw sub-scores before domain-specific weighting.
    All values in [0, 1].
    """
    faithfulness  : float   # F — NLI-based or numerical proximity
    attribution   : float   # A — BM25 overlap + source indicator
    terminology   : float   # T — domain lexicon overlap

    def to_dict(self) -> Dict:
        return {
            "faithfulness" : round(self.faithfulness, 4),
            "attribution"  : round(self.attribution,  4),
            "terminology"  : round(self.terminology,  4),
        }


@dataclass
class DHSWeights:
    """
    Domain-specific weights (α, β, γ) — Table 3 of the paper.
    Must sum to 1.0.
    """
    alpha : float   # faithfulness weight
    beta  : float   # attribution weight
    gamma : float   # terminology weight

    def __post_init__(self):
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"DHSWeights must sum to 1.0, got {total:.4f}"
            )


# Paper Table 3 weights — use these directly
DOMAIN_WEIGHTS: Dict[DomainType, DHSWeights] = {
    DomainType.HEALTHCARE : DHSWeights(alpha=0.40, beta=0.30, gamma=0.30),
    DomainType.LEGAL      : DHSWeights(alpha=0.35, beta=0.45, gamma=0.20),
    DomainType.FINANCE    : DHSWeights(alpha=0.30, beta=0.25, gamma=0.45),
}


@dataclass
class DHSResult:
    """
    Final DHS evaluation result for one QA sample.
    This is what gets written to results CSV / JSON.
    """
    sample_id        : str
    domain           : DomainType
    question         : str
    generated_answer : str
    gold_answer      : str
    answer_type      : AnswerType
    components       : DHSComponents
    weights          : DHSWeights
    dhs_score        : float                      # final weighted score [0,1]
    failure_mode     : Optional[FailureMode] = None   # set if dhs_score < 0.5
    ragas_score      : Optional[float]       = None   # for comparison
    notes            : str                   = ""

    @property
    def passed(self) -> bool:
        """True if DHS >= 0.5 (system gave acceptable answer)."""
        return self.dhs_score >= 0.5

    def to_dict(self) -> Dict:
        return {
            "sample_id"        : self.sample_id,
            "domain"           : self.domain.value,
            "question"         : self.question,
            "generated_answer" : self.generated_answer,
            "gold_answer"      : self.gold_answer,
            "answer_type"      : self.answer_type.value,
            "faithfulness"     : self.components.faithfulness,
            "attribution"      : self.components.attribution,
            "terminology"      : self.components.terminology,
            "dhs_score"        : round(self.dhs_score, 4),
            "passed"           : self.passed,
            "failure_mode"     : self.failure_mode.value if self.failure_mode else None,
            "ragas_score"      : self.ragas_score,
            "notes"            : self.notes,
        }


@dataclass
class ExperimentConfig:
    """
    One of the 6 RAG configurations evaluated in the paper.
    """
    retrieval    : RetrievalStrategy
    granularity  : Granularity
    llm          : str              = "distilgpt2"    # or "gpt-3.5-turbo-0125"
    chunk_size   : int              = 256
    chunk_overlap: int              = 32
    top_k        : int              = 5
    hybrid_alpha : float            = 0.5             # BM25 vs dense balance

    @property
    def name(self) -> str:
        return f"{self.retrieval.value}_{self.granularity.value}"


@dataclass
class ExperimentResult:
    """
    Aggregated results for one ExperimentConfig across one domain.
    """
    config         : ExperimentConfig
    domain         : DomainType
    dhs_results    : List[DHSResult]   = field(default_factory=list)

    @property
    def avg_dhs(self) -> float:
        if not self.dhs_results:
            return 0.0
        return sum(r.dhs_score for r in self.dhs_results) / len(self.dhs_results)

    @property
    def avg_faithfulness(self) -> float:
        if not self.dhs_results:
            return 0.0
        return sum(r.components.faithfulness for r in self.dhs_results) / len(self.dhs_results)

    @property
    def avg_attribution(self) -> float:
        if not self.dhs_results:
            return 0.0
        return sum(r.components.attribution for r in self.dhs_results) / len(self.dhs_results)

    @property
    def avg_terminology(self) -> float:
        if not self.dhs_results:
            return 0.0
        return sum(r.components.terminology for r in self.dhs_results) / len(self.dhs_results)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.dhs_results if not r.passed)

    @property
    def failure_breakdown(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.dhs_results:
            if r.failure_mode:
                key = r.failure_mode.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    def summary(self) -> Dict:
        return {
            "config"           : self.config.name,
            "domain"           : self.domain.value,
            "n_samples"        : len(self.dhs_results),
            "avg_dhs"          : round(self.avg_dhs,          4),
            "avg_faithfulness" : round(self.avg_faithfulness,  4),
            "avg_attribution"  : round(self.avg_attribution,   4),
            "avg_terminology"  : round(self.avg_terminology,   4),
            "failures"         : self.failure_count,
            "failure_breakdown": self.failure_breakdown,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Weights must sum to 1
    w = DOMAIN_WEIGHTS[DomainType.FINANCE]
    assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6

    # BenchmarkSample round-trip
    sample = BenchmarkSample.from_dict({
        "id": "fn_0001", "domain": "finance",
        "question": "What was Apple net income Q3 2023?",
        "answer": "$19.88 billion", "context": "Apple reported...",
        "answer_type": "numerical", "difficulty": "basic",
        "source_doc": "AAPL_10Q_2023",
    })
    assert sample.answer_type == AnswerType.NUMERICAL
    assert sample.domain      == DomainType.FINANCE

    # DHSResult to dict
    result = DHSResult(
        sample_id="fn_0001", domain=DomainType.FINANCE,
        question=sample.question, generated_answer="$19.88 billion",
        gold_answer=sample.answer, answer_type=AnswerType.NUMERICAL,
        components=DHSComponents(0.94, 0.88, 0.91),
        weights=w, dhs_score=0.913,
    )
    d = result.to_dict()
    assert d["passed"] is True

    print("✓ All domain_models.py assertions passed.")
    import pprint; pprint.pprint(d)