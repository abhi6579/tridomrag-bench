"""Application settings using Pydantic"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application configuration"""
    
    # LLM Settings
    LLM_MODEL: str = "distilgpt2"
    LLM_MAX_LENGTH: int = 100
    LLM_TEMPERATURE: float = 0.7
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector Store Settings
    CHROMA_PATH: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "recallify_documents"
    
    # Document Settings
    DOCUMENTS_PATH: str = "./data/documents"
    SUPPORTED_FORMATS: list = ["txt", "pdf", "md"]
    
    # Search Settings
    TOP_K_RESULTS: int = 3
    
    # System Settings
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False
    
    class Config:
        """Pydantic config"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Singleton instance
settings = Settings()
