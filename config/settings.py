"""Application settings using Pydantic"""
from pydantic import BaseModel
from typing import Optional
import os
from pathlib import Path

class Settings(BaseModel):
    """Application configuration"""
    
    # LLM Settings
    LLM_MODEL: str = os.getenv("LLM_MODEL", "distilgpt2")
    LLM_MAX_LENGTH: int = int(os.getenv("LLM_MAX_LENGTH", "100"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # Embedding Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector Store Settings
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./data/chroma_db")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "recallify_documents")
    
    # Document Settings
    DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "./data/documents")
    SUPPORTED_FORMATS: list = ["txt", "pdf", "md"]
    
    # Search Settings
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))
    
    # System Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    class Config:
        """Pydantic config"""
        case_sensitive = False

# Singleton instance
settings = Settings()
