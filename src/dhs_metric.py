"""
dhs_metric.py
=============
Implements the Domain Hallucination Score (DHS) — Equation (1) of the paper.

  DHS(a, C, a*, d) = α_d·F(a,C) + β_d·A(a,C) + γ_d·T(a,a*,d)

Three sub-scorers:
  F — Faithfulness  : NLI-based (Type-T) or numerical proximity (Type-N/E)
  A — Attribution   : BM25 overlap with source span present
  T — Terminology   : domain lexicon overlap between answer and gold

Usage:
  from src.dhs_metric import DHSMetric
  metric = DHSMetric()
  result = metric.score(
      sample_id        = "hc_0001",
      domain           = DomainType.HEALTHCARE,
      question         = "...",
      generated_answer = "...",
      gold_answer      = "...",
      context_chunks   = ["chunk1 text", "chunk2 text"],
      answer_type      = AnswerType.TEXTUAL,
  )
  print(result.dhs_score)
"""

import re
import math
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# local imports — works from recallify/ root
from core.domain_models import (
    AnswerType, DomainType, DHSComponents, DHSWeights,
    DHSResult, DOMAIN_WEIGHTS, FailureMode,
)
from src.terminology_lexicon import TerminologyLexicon


class DHSMetric:
    """
    Computes the Domain Hallucination Score for a single QA sample.

    For Type-T answers   → NLI faithfulness via cross-encoder
    For Type-N answers   → numerical proximity (Equation 2)
    For Type-E answers   → entity string match (normalised)
    """

    def __init__(self, use_nli: bool = True):
        """
        Args:
            use_nli: If True, load the NLI cross-encoder for Type-T faithfulness.
                     Set False during quick dev runs to skip the heavy model load.
        """
        self.lexicon  = TerminologyLexicon()
        self._nli     = None
        self._use_nli = use_nli

        # Domain tolerance thresholds ε_d — from paper Section 5.1
        self._epsilon = {
            DomainType.HEALTHCARE : 0.05,   # ±5% clinical values
            DomainType.LEGAL      : 0.01,   # ±1% legal exactness
            DomainType.FINANCE    : 0.02,   # ±2% financial rounding
        }

        if use_nli:
            self._load_nli()

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def score(
        self,
        sample_id        : str,
        domain           : DomainType,
        question         : str,
        generated_answer : str,
        gold_answer      : str,
        context_chunks   : List[str],
        answer_type      : AnswerType        = AnswerType.TEXTUAL,
        source_available : bool              = True,
        ragas_score      : Optional[float]   = None,
    ) -> DHSResult:
        """
        Compute full DHS for one sample. Returns a DHSResult object.

        Args:
            sample_id        : unique identifier (e.g. "hc_0001")
            domain           : DomainType enum
            question         : the original query
            generated_answer : answer produced by the RAG system
            gold_answer      : ground-truth answer
            context_chunks   : list of retrieved text chunks (strings)
            answer_type      : AnswerType enum (T / N / E)
            source_available : False if retriever returned no traceable source
            ragas_score      : optional RAGAS faithfulness for comparison table
        """
        weights = DOMAIN_WEIGHTS[domain]

        # ── Sub-scores ──────────────────────────────────────────────────
        F = self._faithfulness(generated_answer, gold_answer,
                               context_chunks, answer_type, domain)
        A = self._attribution(generated_answer, context_chunks, source_available)
        T = self._terminology(generated_answer, gold_answer, domain)

        components = DHSComponents(faithfulness=F, attribution=A, terminology=T)

        # ── Weighted DHS ─────────────────────────────────────────────────
        dhs = weights.alpha * F + weights.beta * A + weights.gamma * T
        dhs = min(max(dhs, 0.0), 1.0)           # clamp to [0,1]

        # ── Failure mode tagging ─────────────────────────────────────────
        failure = self._classify_failure(dhs, components, answer_type) \
                  if dhs < 0.5 else None

        return DHSResult(
            sample_id        = sample_id,
            domain           = domain,
            question         = question,
            generated_answer = generated_answer,
            gold_answer      = gold_answer,
            answer_type      = answer_type,
            components       = components,
            weights          = weights,
            dhs_score        = round(dhs, 4),
            failure_mode     = failure,
            ragas_score      = ragas_score,
        )

    def batch_score(self, samples: list, rag_outputs: list) -> List[DHSResult]:
        """
        Score a batch of (BenchmarkSample, generated_answer, context_chunks) tuples.

        Args:
            samples    : List[BenchmarkSample]
            rag_outputs: List[dict] with keys 'generated_answer', 'context_chunks'

        Returns:
            List[DHSResult]
        """
        results = []
        for sample, output in zip(samples, rag_outputs):
            try:
                r = self.score(
                    sample_id        = sample.id,
                    domain           = sample.domain,
                    question         = sample.question,
                    generated_answer = output.get("generated_answer", ""),
                    gold_answer      = sample.answer,
                    context_chunks   = output.get("context_chunks", []),
                    answer_type      = sample.answer_type,
                    ragas_score      = output.get("ragas_score"),
                )
                results.append(r)
            except Exception as e:
                logger.error(f"DHS scoring failed for {sample.id}: {e}")
        return results

    # ─────────────────────────────────────────────────────────────────────────
    #  SUB-SCORERS
    # ─────────────────────────────────────────────────────────────────────────

    # ── F: Faithfulness ──────────────────────────────────────────────────────

    def _faithfulness(
        self,
        generated : str,
        gold      : str,
        chunks    : List[str],
        atype     : AnswerType,
        domain    : DomainType,
    ) -> float:
        """
        Type-T → NLI cross-encoder decomposition (or token overlap fallback)
        Type-N → Equation (2): numerical proximity
        Type-E → normalised entity match
        """
        if atype == AnswerType.NUMERICAL:
            return self._numerical_faithfulness(generated, gold, domain)
        elif atype == AnswerType.ENTITY:
            return self._entity_faithfulness(generated, gold)
        else:
            # Type-T: try NLI, fall back to token overlap
            if self._nli is not None:
                return self._nli_faithfulness(generated, chunks)
            else:
                return self._token_overlap_faithfulness(generated, chunks)

    def _numerical_faithfulness(
        self,
        generated : str,
        gold      : str,
        domain    : DomainType,
    ) -> float:
        """
        Equation (2) from the paper:
          F_N = max(0, 1 - |v_a - v_a*| / (ε_d * v_a*))
        """
        v_gen  = self._extract_number(generated)
        v_gold = self._extract_number(gold)

        if v_gen is None or v_gold is None or v_gold == 0:
            # Can't parse numbers — fall back to string match
            return 1.0 if generated.strip().lower() == gold.strip().lower() else 0.0

        eps  = self._epsilon[domain]
        score = max(0.0, 1.0 - abs(v_gen - v_gold) / (eps * abs(v_gold)))
        return min(score, 1.0)

    def _entity_faithfulness(self, generated: str, gold: str) -> float:
        """Normalised string match for entity-type answers."""
        def normalise(s: str) -> str:
            return re.sub(r"[^\w\s]", "", s.lower()).strip()

        gen_norm  = normalise(generated)
        gold_norm = normalise(gold)

        if gen_norm == gold_norm:
            return 1.0
        # Partial match: gold tokens inside generated
        gold_tokens = set(gold_norm.split())
        gen_tokens  = set(gen_norm.split())
        if not gold_tokens:
            return 0.0
        overlap = gold_tokens & gen_tokens
        return len(overlap) / len(gold_tokens)

    def _nli_faithfulness(self, generated: str, chunks: List[str]) -> float:
        """
        Decompose generated answer into atomic claims,
        verify each against the context using NLI cross-encoder.
        Returns fraction of claims entailed.
        """
        if not chunks or not generated.strip():
            return 0.0

        # Split answer into sentences as atomic claims
        claims  = [s.strip() for s in re.split(r"[.!?]", generated) if s.strip()]
        context = " ".join(chunks)[:1500]       # cap context length

        if not claims:
            return 0.0

        entailed = 0
        for claim in claims:
            try:
                result = self._nli.predict([(context, claim)])
                # Labels: 0=contradiction, 1=neutral, 2=entailment
                if hasattr(result, "__iter__"):
                    label = int(result[0])
                else:
                    label = int(result)
                if label == 2:
                    entailed += 1
            except Exception:
                # If model fails, use token overlap for this claim
                entailed += self._token_overlap_faithfulness(claim, [context])

        return entailed / len(claims)

    def _token_overlap_faithfulness(
        self, generated: str, chunks: List[str]
    ) -> float:
        """
        Lightweight fallback: F1 token overlap between generated answer
        and the best matching context chunk.
        """
        def tokens(s: str):
            return set(re.sub(r"[^\w]", " ", s.lower()).split())

        gen_toks = tokens(generated)
        if not gen_toks:
            return 0.0

        best = 0.0
        for chunk in chunks:
            chunk_toks = tokens(chunk)
            if not chunk_toks:
                continue
            prec = len(gen_toks & chunk_toks) / len(gen_toks)
            rec  = len(gen_toks & chunk_toks) / len(chunk_toks)
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            best = max(best, f1)
        return best

    # ── A: Attribution ───────────────────────────────────────────────────────

    def _attribution(
        self,
        generated        : str,
        chunks           : List[str],
        source_available : bool,
    ) -> float:
        """
        Equation (3): max BM25 similarity between answer and context spans,
        penalised if no source document is traceable.
        """
        if not source_available:
            return 0.0
        if not chunks or not generated.strip():
            return 0.0

        best = max(self._bm25_similarity(generated, chunk) for chunk in chunks)
        return min(best, 1.0)

    def _bm25_similarity(self, query: str, doc: str,
                         k1: float = 1.5, b: float = 0.75) -> float:
        """
        Simplified single-doc BM25 score (normalised to [0,1]).
        Full BM25 requires a corpus; here we approximate with TF-IDF-style
        scoring between query tokens and document tokens.
        """
        def tokenize(s: str) -> List[str]:
            return re.sub(r"[^\w]", " ", s.lower()).split()

        q_tokens = tokenize(query)
        d_tokens = tokenize(doc)
        if not q_tokens or not d_tokens:
            return 0.0

        doc_len = len(d_tokens)
        avg_len = doc_len                        # single doc — avg = itself
        tf_map  = {}
        for t in d_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for qt in q_tokens:
            tf = tf_map.get(qt, 0)
            numerator   = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_len)
            score += numerator / (denominator + 1e-9)

        # Normalise: divide by max possible score (if all query tokens match)
        max_score = len(q_tokens) * (k1 + 1) / (1.0 + k1 * (1 - b + b) + 1e-9)
        return min(score / (max_score + 1e-9), 1.0)

    # ── T: Terminology ───────────────────────────────────────────────────────

    def _terminology(
        self,
        generated : str,
        gold      : str,
        domain    : DomainType,
    ) -> float:
        """
        Equation (4): Jaccard-style overlap of domain terms.
          T = |T(a) ∩ T(a*)| / max(|T(a)|, |T(a*)|)
        """
        gen_terms  = self.lexicon.extract_terms(generated, domain)
        gold_terms = self.lexicon.extract_terms(gold,      domain)

        if not gold_terms and not gen_terms:
            return 1.0     # neither answer uses domain terms → neutral
        if not gold_terms or not gen_terms:
            return 0.0

        intersection = gen_terms & gold_terms
        return len(intersection) / max(len(gen_terms), len(gold_terms))

    # ─────────────────────────────────────────────────────────────────────────
    #  FAILURE CLASSIFIER
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_failure(
        self,
        dhs        : float,
        components : DHSComponents,
        atype      : AnswerType,
    ) -> FailureMode:
        """
        Maps low DHS + component breakdown → one of F1–F4 failure modes.
        Priority order matches paper Table 6.
        """
        F, A, T = components.faithfulness, components.attribution, components.terminology

        if atype == AnswerType.NUMERICAL and F < 0.4:
            return FailureMode.F2_NUMERICAL_HALLUCINATION
        if A < 0.3:
            return FailureMode.F4_ATTRIBUTION_GAP
        if T < 0.35:
            return FailureMode.F3_TERMINOLOGY_MISMATCH
        return FailureMode.F1_WRONG_CHUNK_RETRIEVED

    # ─────────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_number(self, text: str) -> Optional[float]:
        """
        Extract the first numeric value from text, handling:
        $1.2B, 19.88 billion, 4,500,000, 83.5%, etc.
        """
        text = text.replace(",", "")

        multipliers = {
            "trillion": 1e12, "billion": 1e9,
            "million":  1e6,  "thousand": 1e3,
        }

        for word, mult in multipliers.items():
            pattern = rf"(\d+(?:\.\d+)?)\s*{word}"
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return float(m.group(1)) * mult

        # Plain number (possibly with $ or %)
        m = re.search(r"[\$]?(\d+(?:\.\d+)?)\s*[%]?", text)
        if m:
            return float(m.group(1))

        return None

    def _load_nli(self) -> None:
        """Load NLI cross-encoder (only when use_nli=True)."""
        try:
            from sentence_transformers import CrossEncoder
            self._nli = CrossEncoder(
                "cross-encoder/nli-deberta-v3-base",
                max_length=512,
            )
            logger.info("NLI cross-encoder loaded: cross-encoder/nli-deberta-v3-base")
        except Exception as e:
            logger.warning(
                f"NLI model load failed ({e}). "
                "Falling back to token-overlap faithfulness."
            )
            self._nli = None


# ─────────────────────────────────────────────────────────────────────────────
#  Quick self-test — python src/dhs_metric.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)

    # Use NLI=False for quick test (no heavy model download)
    metric = DHSMetric(use_nli=False)

    # ── Test 1: Finance / numerical ─────────────────────────────
    r1 = metric.score(
        sample_id        = "fn_0001",
        domain           = DomainType.FINANCE,
        question         = "What was Apple net income Q3 2023?",
        generated_answer = "$19.88 billion",
        gold_answer      = "$19.88 billion",
        context_chunks   = ["Apple reported net income of $19.88 billion for Q3 2023."],
        answer_type      = AnswerType.NUMERICAL,
    )
    print(f"\n[Finance/Numerical] DHS={r1.dhs_score:.3f}  passed={r1.passed}")
    pprint.pprint(r1.to_dict())
    assert r1.dhs_score > 0.7, "Perfect match should score high"

    # ── Test 2: Healthcare / textual ────────────────────────────
    r2 = metric.score(
        sample_id        = "hc_0001",
        domain           = DomainType.HEALTHCARE,
        question         = "What is the recommended treatment for HTN?",
        generated_answer = "Lifestyle modification and antihypertensive medication.",
        gold_answer      = "Antihypertensive medications and lifestyle changes.",
        context_chunks   = ["Treatment includes lifestyle modification and antihypertensives."],
        answer_type      = AnswerType.TEXTUAL,
    )
    print(f"\n[Healthcare/Textual] DHS={r2.dhs_score:.3f}  passed={r2.passed}")

    # ── Test 3: Failure mode tagging ────────────────────────────
    r3 = metric.score(
        sample_id        = "fn_0002",
        domain           = DomainType.FINANCE,
        question         = "What was revenue in Q2?",
        generated_answer = "$1.2 trillion",    # wildly wrong
        gold_answer      = "$19.88 billion",
        context_chunks   = ["Unrelated sentence about marketing."],
        answer_type      = AnswerType.NUMERICAL,
    )
    print(f"\n[Failure case] DHS={r3.dhs_score:.3f}  failure={r3.failure_mode}")
    assert r3.failure_mode is not None

    print("\n✓ All dhs_metric.py tests passed.")