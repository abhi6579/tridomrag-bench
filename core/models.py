"""Pydantic models for type safety"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Document(BaseModel):
    """Document model"""
    id: str
    content: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True

class Query(BaseModel):
    """Query model"""
    question: str
    top_k: int = 3
    include_sources: bool = True

class RetrievedDocument(BaseModel):
    """Retrieved document with similarity score"""
    id: str
    content: str
    source: str
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RAGResponse(BaseModel):
    """RAG system response"""
    question: str
    answer: str
    retrieved_documents: List[RetrievedDocument]
    confidence: float = 0.0
    processing_time: float = 0.0
