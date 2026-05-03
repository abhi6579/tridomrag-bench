"""
llm_config.py — supports local / openai / groq / together
LLM_PROVIDER=together + TOGETHER_API_KEY in .env for paper results
LLM_PROVIDER=groq    + GROQ_API_KEY in .env (dev only — token limits)
"""
import os, logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

RAG_PROMPT = """You are an expert in {domain}. Answer the following question using ONLY the provided context.
If the answer is not clearly stated in the context, respond with: INSUFFICIENT CONTEXT

Context:
{context}

Question: {question}

Answer:"""

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, question: str, context: str, domain: str = "general") -> str: ...
    @abstractmethod
    def is_available(self) -> bool: ...

class LocalLLM(BaseLLM):
    MODEL_NAME = "distilgpt2"
    def __init__(self, max_new_tokens: int = 150):
        self.max_new_tokens = max_new_tokens
        self._pipeline = None
        self._load()
    def _load(self):
        try:
            from transformers import pipeline
            self._pipeline = pipeline("text-generation", model=self.MODEL_NAME,
                max_new_tokens=self.max_new_tokens, pad_token_id=50256, truncation=True)
            logger.info(f"Local LLM loaded: {self.MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
    def generate(self, question, context, domain="general"):
        if self._pipeline is None: return "ERROR: Local model not loaded"
        prompt = RAG_PROMPT.format(domain=domain, context=context[:800], question=question)
        try:
            output = self._pipeline(prompt)[0]["generated_text"]
            if "Answer:" in output:
                answer = output.split("Answer:")[-1].strip().split("\n")[0].strip()
                return answer if answer else "INSUFFICIENT CONTEXT"
            return output.strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "ERROR: Generation failed"
    def is_available(self): return self._pipeline is not None

class GroqLLM(BaseLLM):
    MODEL_NAME = "llama-3.3-70b-versatile"
    def __init__(self, max_tokens: int = 300, temperature: float = 0.0):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        self._load()
    def _load(self):
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("GROQ_API_KEY not set — Groq LLM unavailable")
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            logger.info(f"Groq LLM ready: {self.MODEL_NAME}")
        except ImportError:
            logger.warning("groq package not installed. Run: pip install groq")
    def generate(self, question, context, domain="general"):
        import time
        time.sleep(0.5)
        if self._client is None: return "ERROR: Groq client not available"
        prompt = RAG_PROMPT.format(domain=domain, context=context[:3000], question=question)
        try:
            response = self._client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            return "ERROR: Groq call failed"
    def is_available(self): return self._client is not None

class OpenAILLM(BaseLLM):
    MODEL_NAME = "openai/gpt-4o-mini"
    def __init__(self, max_tokens: int = 300, temperature: float = 0.0):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        self._load()
    def _load(self):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set")
            return
        try:
            from openai import OpenAI
            base_url = os.getenv("OPENAI_BASE_URL", None)
            self.MODEL_NAME = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
            self._client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        except ImportError:
            logger.warning("openai package not installed")
    def generate(self, question, context, domain="general"):
        if self._client is None: return "ERROR: OpenAI client not available"
        prompt = RAG_PROMPT.format(domain=domain, context=context[:3000], question=question)
        try:
            response = self._client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens, temperature=self.temperature)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return "ERROR: OpenAI call failed"
    def is_available(self): return self._client is not None

class TogetherLLM(BaseLLM):
    # Same Llama-3.3-70B as Groq — results are directly comparable
    MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    BASE_URL   = "https://api.together.xyz/v1"

    def __init__(self, max_tokens: int = 300, temperature: float = 0.0):
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._client     = None
        self._load()

    def _load(self):
        api_key = os.getenv("TOGETHER_API_KEY", "")
        if not api_key:
            logger.warning("TOGETHER_API_KEY not set — Together LLM unavailable")
            return
        try:
            # Together AI is OpenAI-API-compatible — no extra package needed
            from openai import OpenAI
            self._client = OpenAI(
                api_key  = api_key,
                base_url = self.BASE_URL,
            )
            logger.info(f"Together LLM ready: {self.MODEL_NAME}")
        except ImportError:
            logger.warning("openai package not installed. Run: pip install openai")

    def generate(self, question: str, context: str, domain: str = "general") -> str:
        import time
        time.sleep(0.3)          # light throttle — Together has generous limits
        if self._client is None:
            return "ERROR: Together client not available"
        prompt = RAG_PROMPT.format(domain=domain, context=context[:3000], question=question)
        try:
            response = self._client.chat.completions.create(
                model       = self.MODEL_NAME,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = self.max_tokens,
                temperature = self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Together generation error: {e}")
            return "ERROR: Together call failed"

    def is_available(self) -> bool:
        return self._client is not None


def get_llm(provider: Optional[str] = None) -> BaseLLM:
    provider = (provider or os.getenv("LLM_PROVIDER", "local")).lower()

    if provider == "together":
        llm = TogetherLLM()
        if not llm.is_available():
            logger.warning("Together not available — falling back to local")
            return LocalLLM()
        return llm
    elif provider == "groq":
        llm = GroqLLM()
        if not llm.is_available():
            logger.warning("Groq not available — falling back to local")
            return LocalLLM()
        return llm
    elif provider == "openai":
        llm = OpenAILLM()
        if not llm.is_available():
            return LocalLLM()
        return llm
    else:
        return LocalLLM()