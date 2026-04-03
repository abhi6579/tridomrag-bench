"""
retrieval_configs.py
====================
Defines the 6 RAG retrieval configurations evaluated in the paper:
  BM25   × Chunk
  BM25   × Document
  Dense  × Chunk
  Dense  × Document
  Hybrid × Chunk
  Hybrid × Document

Each config wraps the existing vector_store.py and adds BM25 support.

Usage:
  from src.retrieval_configs import get_all_configs, retrieve
  configs = get_all_configs()
  chunks  = retrieve(config=configs[0], query="what is hypertension?", corpus=docs)
"""

import re
import math
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

from core.domain_models import RetrievalStrategy, Granularity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  RETRIEVAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalConfig:
    strategy    : RetrievalStrategy
    granularity : Granularity
    top_k       : int   = 5
    chunk_size  : int   = 256    # tokens (approx words)
    chunk_overlap: int  = 32
    hybrid_alpha: float = 0.5    # 0=BM25 only, 1=Dense only

    @property
    def name(self) -> str:
        return f"{self.strategy.value}_{self.granularity.value}"


def get_all_configs() -> List[RetrievalConfig]:
    """Returns all 6 paper configurations."""
    configs = []
    for strategy in RetrievalStrategy:
        for granularity in Granularity:
            configs.append(RetrievalConfig(
                strategy=strategy,
                granularity=granularity,
            ))
    return configs


# ─────────────────────────────────────────────────────────────────────────────
#  CHUNKER
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 256,
               overlap: int = 32) -> List[str]:
    """
    Split text into overlapping word-level chunks.
    chunk_size and overlap are in words (approximates tokens).
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def prepare_corpus(
    samples     : List[Dict],
    granularity : Granularity,
    chunk_size  : int = 256,
    overlap     : int = 32,
) -> Tuple[List[str], List[Dict]]:
    """
    Given benchmark samples, build a retrieval corpus.

    Granularity.CHUNK    → split each context into chunks
    Granularity.DOCUMENT → use full context as one document

    Returns:
        texts    : List[str]  — corpus documents
        metadata : List[Dict] — parallel metadata (sample_id, domain, source)
    """
    texts    : List[str]  = []
    metadata : List[Dict] = []

    for sample in samples:
        ctx = sample.get("context", "")
        sid = sample.get("id",      "unknown")
        dom = sample.get("domain",  "unknown")
        src = sample.get("source_doc", "")

        if granularity == Granularity.CHUNK:
            chunks = chunk_text(ctx, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadata.append({
                    "sample_id" : sid,
                    "chunk_idx" : i,
                    "domain"    : dom,
                    "source_doc": src,
                })
        else:
            texts.append(ctx)
            metadata.append({
                "sample_id" : sid,
                "domain"    : dom,
                "source_doc": src,
            })

    return texts, metadata


# ─────────────────────────────────────────────────────────────────────────────
#  BM25 RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Pure-Python BM25 retriever — no external dependencies.
    Works offline, no GPU needed.
    """

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1   = k1
        self.b    = b
        self.docs : List[str]       = []
        self.meta : List[Dict]      = []
        self._tf  : List[Dict]      = []
        self._df  : Dict[str, int]  = {}
        self._avg_dl : float        = 0.0

    def index(self, texts: List[str], metadata: List[Dict]) -> None:
        self.docs = texts
        self.meta = metadata
        self._tf  = []
        self._df  = {}

        total_len = 0
        for doc in texts:
            tokens  = self._tokenize(doc)
            total_len += len(tokens)
            tf : Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for t in set(tokens):
                self._df[t] = self._df.get(t, 0) + 1

        self._avg_dl = total_len / max(len(texts), 1)
        logger.info(f"BM25 indexed {len(texts)} documents, vocab={len(self._df)}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Returns list of (text, score, metadata) sorted by score desc."""
        q_tokens = self._tokenize(query)
        N        = len(self.docs)
        scores   = []

        for idx, tf in enumerate(self._tf):
            dl    = sum(tf.values())
            score = 0.0
            for qt in q_tokens:
                if qt not in tf:
                    continue
                df_t = self._df.get(qt, 0)
                idf  = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
                num  = tf[qt] * (self.k1 + 1)
                den  = tf[qt] + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                score += idf * num / (den + 1e-9)
            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            (self.docs[i], scores[i], self.meta[i])
            for i, _ in ranked[:top_k]
        ]

    def _tokenize(self, text: str) -> List[str]:
        return re.sub(r"[^\w]", " ", text.lower()).split()


# ─────────────────────────────────────────────────────────────────────────────
#  DENSE RETRIEVER  (uses existing vector_store.py)
# ─────────────────────────────────────────────────────────────────────────────

class DenseRetriever:
    """
    Wraps sentence-transformers for dense retrieval.
    Model: BAAI/bge-base-en-v1.5 (paper setting)
    Falls back to all-MiniLM-L6-v2 if BGE not available.
    """

    MODEL_NAME = "BAAI/bge-base-en-v1.5"
    FALLBACK   = "all-MiniLM-L6-v2"

    def __init__(self):
        self.model     = None
        self.docs      : List[str]  = []
        self.meta      : List[Dict] = []
        self.embeddings             = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            try:
                self.model = SentenceTransformer(self.MODEL_NAME)
                logger.info(f"Dense model loaded: {self.MODEL_NAME}")
            except Exception:
                self.model = SentenceTransformer(self.FALLBACK)
                logger.info(f"Dense model loaded (fallback): {self.FALLBACK}")
        except ImportError:
            logger.warning("sentence-transformers not available — dense retrieval disabled")

    def index(self, texts: List[str], metadata: List[Dict]) -> None:
        if self.model is None:
            logger.warning("Dense model not loaded — skipping index")
            return
        self.docs  = texts
        self.meta  = metadata
        logger.info(f"Encoding {len(texts)} documents...")
        self.embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=False,
            normalize_embeddings=True,
        )
        logger.info("Dense index ready")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        if self.model is None or self.embeddings is None:
            return []
        import numpy as np
        q_emb  = self.model.encode([query], normalize_embeddings=True)[0]
        scores = (self.embeddings @ q_emb).tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            (self.docs[i], scores[i], self.meta[i])
            for i, _ in ranked[:top_k]
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  HYBRID RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 + Dense scores with linear interpolation.
      hybrid_score = (1 - alpha) * bm25_norm + alpha * dense_norm
    alpha=0.5 by default (paper setting).
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha  = alpha
        self.bm25   = BM25Retriever()
        self.dense  = DenseRetriever()
        self.docs   : List[str]  = []
        self.meta   : List[Dict] = []

    def index(self, texts: List[str], metadata: List[Dict]) -> None:
        self.docs = texts
        self.meta = metadata
        self.bm25.index(texts, metadata)
        self.dense.index(texts, metadata)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        # Get scores from both retrievers (full corpus)
        bm25_results  = {m["sample_id"] + str(i): (t, s, m)
                         for i, (t, s, m) in enumerate(
                             self.bm25.retrieve(query, top_k=len(self.docs) or 100))}
        dense_results = {m["sample_id"] + str(i): (t, s, m)
                         for i, (t, s, m) in enumerate(
                             self.dense.retrieve(query, top_k=len(self.docs) or 100))}

        # Normalise scores to [0,1]
        def normalise(scores: List[float]) -> List[float]:
            mn, mx = min(scores, default=0), max(scores, default=1)
            if mx == mn:
                return [0.5] * len(scores)
            return [(s - mn) / (mx - mn) for s in scores]

        bm25_scores  = [v[1] for v in bm25_results.values()]
        dense_scores = [v[1] for v in dense_results.values()]
        bm25_norm    = dict(zip(bm25_results.keys(),  normalise(bm25_scores)))
        dense_norm   = dict(zip(dense_results.keys(), normalise(dense_scores)))

        # Combine
        all_keys = set(bm25_norm) | set(dense_norm)
        combined = {}
        for k in all_keys:
            b = bm25_norm.get(k,  0.0)
            d = dense_norm.get(k, 0.0)
            combined[k] = (1 - self.alpha) * b + self.alpha * d

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for k, score in ranked:
            if k in bm25_results:
                t, _, m = bm25_results[k]
            else:
                t, _, m = dense_results[k]
            results.append((t, score, m))
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED RETRIEVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_retriever(config: RetrievalConfig):
    """Factory — returns the right retriever for a config."""
    if config.strategy == RetrievalStrategy.BM25:
        return BM25Retriever()
    elif config.strategy == RetrievalStrategy.DENSE:
        return DenseRetriever()
    else:
        return HybridRetriever(alpha=config.hybrid_alpha)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    corpus = [
        "Hypertension is a chronic medical condition where blood pressure is elevated.",
        "The contract shall be governed by the laws of Delaware.",
        "Apple reported net income of $19.88 billion for Q3 2023.",
        "Treatment of hypertension includes lifestyle changes and medication.",
        "Antihypertensive drugs reduce cardiovascular risk significantly.",
    ]
    meta = [{"sample_id": f"doc_{i}", "domain": "test"} for i in range(len(corpus))]

    print("\n── BM25 ──")
    bm25 = BM25Retriever()
    bm25.index(corpus, meta)
    for text, score, m in bm25.retrieve("hypertension treatment", top_k=3):
        print(f"  [{score:.3f}] {text[:60]}")

    print("\n── Dense ──")
    dense = DenseRetriever()
    dense.index(corpus, meta)
    for text, score, m in dense.retrieve("hypertension treatment", top_k=3):
        print(f"  [{score:.3f}] {text[:60]}")

    print("\n── Hybrid ──")
    hybrid = HybridRetriever()
    hybrid.index(corpus, meta)
    for text, score, m in hybrid.retrieve("hypertension treatment", top_k=3):
        print(f"  [{score:.3f}] {text[:60]}")

    print("\n── All 6 configs ──")
    for c in get_all_configs():
        print(f"  {c.name}")

    print("\n✓ retrieval_configs.py working correctly.")