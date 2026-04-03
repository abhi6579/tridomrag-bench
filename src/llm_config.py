"""
llm_config.py
=============
Unified LLM interface — switch between local and OpenAI with one env var.

  LLM_PROVIDER=local    → uses distilgpt2 via HuggingFace (default, no cost)
  LLM_PROVIDER=openai   → uses gpt-3.5-turbo-0125 via API (paper experiments)

Set in your .env file:
  LLM_PROVIDER=local
  OPENAI_API_KEY=sk-...   # only needed when LLM_PROVIDER=openai

Usage:
  from src.llm_config import get_llm, RAGPromptTemplate
  llm = get_llm()
  answer = llm.generate(question="...", context="...")
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

# ── Standard RAG prompt used for ALL experiments (paper Section 5.1) ─────────
RAG_PROMPT = """You are an expert in {domain}. Answer the following question using ONLY the provided context.
If the answer is not clearly stated in the context, respond with: INSUFFICIENT CONTEXT

Context:
{context}

Question: {question}

Answer:"""


# ─────────────────────────────────────────────────────────────────────────────
#  BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def generate(self, question: str, context: str,
                 domain: str = "general") -> str:
        """Generate an answer given question + context."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """True if the model is loaded and ready."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
#  LOCAL LLM  (distilgpt2 — free, offline)
# ─────────────────────────────────────────────────────────────────────────────

class LocalLLM(BaseLLM):
    """
    Uses HuggingFace distilgpt2 locally.
    No API key, no cost, works offline.
    Good for development and testing pipeline logic.
    """

    MODEL_NAME = "distilgpt2"

    def __init__(self, max_new_tokens: int = 150):
        self.max_new_tokens = max_new_tokens
        self._pipeline      = None
        self._load()

    def _load(self) -> None:
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self.MODEL_NAME,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=50256,
                truncation=True,
            )
            logger.info(f"Local LLM loaded: {self.MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")

    def generate(self, question: str, context: str,
                 domain: str = "general") -> str:
        if self._pipeline is None:
            return "ERROR: Local model not loaded"

        prompt = RAG_PROMPT.format(
            domain=domain,
            context=context[:800],      # keep prompt short for distilgpt2
            question=question,
        )
        try:
            output = self._pipeline(prompt)[0]["generated_text"]
            # Extract only the answer part (after "Answer:")
            if "Answer:" in output:
                answer = output.split("Answer:")[-1].strip()
                # Take only first sentence/line to avoid rambling
                answer = answer.split("\n")[0].strip()
                return answer if answer else "INSUFFICIENT CONTEXT"
            return output.strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "ERROR: Generation failed"

    def is_available(self) -> bool:
        return self._pipeline is not None


# ─────────────────────────────────────────────────────────────────────────────
#  OPENAI LLM  (gpt-3.5-turbo — for final paper results)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    """
    Uses OpenAI GPT-3.5-Turbo-0125.
    Requires OPENAI_API_KEY in .env
    Used ONLY for final paper experiment numbers.
    """

    MODEL_NAME = "gpt-3.5-turbo-0125"

    def __init__(self, max_tokens: int = 300, temperature: float = 0.0):
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._client     = None
        self._load()

    def _load(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — OpenAI LLM unavailable")
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI LLM ready: {self.MODEL_NAME}")
        except ImportError:
            logger.warning("openai package not installed. Run: pip install openai")

    def generate(self, question: str, context: str,
                 domain: str = "general") -> str:
        if self._client is None:
            return "ERROR: OpenAI client not available"

        prompt = RAG_PROMPT.format(
            domain=domain,
            context=context[:3000],
            question=question,
        )
        try:
            response = self._client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return "ERROR: OpenAI call failed"

    def is_available(self) -> bool:
        return self._client is not None


# ─────────────────────────────────────────────────────────────────────────────
#  FACTORY — get_llm()
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(provider: Optional[str] = None) -> BaseLLM:
    """
    Returns the right LLM based on LLM_PROVIDER env var.

    LLM_PROVIDER=local  (default) → LocalLLM
    LLM_PROVIDER=openai           → OpenAILLM

    To switch to OpenAI for paper results:
      Add to .env:  LLM_PROVIDER=openai
                    OPENAI_API_KEY=sk-...
    """
    provider = provider or os.getenv("LLM_PROVIDER", "local").lower()

    if provider == "openai":
        logger.info("Using OpenAI LLM (paper experiment mode)")
        llm = OpenAILLM()
        if not llm.is_available():
            logger.warning("OpenAI not available — falling back to local")
            return LocalLLM()
        return llm
    else:
        logger.info("Using local LLM (dev mode)")
        return LocalLLM()


# ─────────────────────────────────────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    llm = get_llm(provider="local")
    print(f"\nLLM available: {llm.is_available()}")
    print(f"LLM type: {type(llm).__name__}")

    if llm.is_available():
        answer = llm.generate(
            question="What is hypertension?",
            context="Hypertension is a chronic condition where blood pressure in the arteries is elevated.",
            domain="healthcare",
        )
        print(f"\nTest answer: {answer}")
        print("\n✓ llm_config.py working correctly.")
    else:
        print("⚠ LLM not loaded — check transformers install")