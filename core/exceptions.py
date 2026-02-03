"""Custom exceptions for Recallify"""

class RecallifyException(Exception):
    """Base exception for all Recallify errors"""
    pass

class DocumentProcessingError(RecallifyException):
    """Raised when document processing fails"""
    pass

class EmbeddingError(RecallifyException):
    """Raised when embedding generation fails"""
    pass

class LLMError(RecallifyException):
    """Raised when LLM generation fails"""
    pass

class RetrievalError(RecallifyException):
    """Raised when retrieval fails"""
    pass

class ConfigurationError(RecallifyException):
    """Raised when configuration is invalid"""
    pass
