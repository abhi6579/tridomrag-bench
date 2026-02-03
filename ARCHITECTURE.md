# Recallify Architecture Guidance & Refactoring Plan

## Current Architecture Issues & Recommendations

### 🔴 CRITICAL ISSUES

#### 1. **Wrong LLM Choice**
- **Current**: OpenAI (gpt-3.5-turbo) ❌
- **Project Spec**: distilgpt2 (local, free) ✅
- **Impact**: High API costs, external dependency, not aligned with thesis
- **Action**: Replace with Hugging Face transformers pipeline

#### 2. **Missing Core Components**
- `document_processor.py` is **EMPTY** - needs PDF/text parsing
- `app.py` is **EMPTY** - no Streamlit UI
- No configuration management
- No error handling/logging
- No database for metadata (DuckDB mentioned but not used)

#### 3. **Architectural Concerns**
- No dependency injection - tightly coupled components
- No abstraction layers - direct calls between modules
- No configuration system (hardcoded paths, models)
- No async support for long-running operations
- No caching mechanism for repeated queries

---

## Proposed Refactored Architecture

### **Clean Architecture Pattern**

```
recallify/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Config management (Pydantic)
│   └── constants.py         # App constants
│
├── core/
│   ├── __init__.py
│   ├── models.py            # Pydantic models (Document, Query, etc.)
│   └── exceptions.py        # Custom exceptions
│
├── services/
│   ├── __init__.py
│   ├── document_service.py  # Document loading & processing
│   ├── embedding_service.py # Embedding generation (HF)
│   ├── llm_service.py       # LLM generation (distilgpt2)
│   ├── retrieval_service.py # Vector search
│   └── rag_service.py       # RAG orchestration (Main)
│
├── infrastructure/
│   ├── __init__.py
│   ├── vector_store.py      # ChromaDB wrapper
│   ├── metadata_store.py    # DuckDB for metadata
│   └── cache.py             # Caching layer
│
├── ui/
│   └── app.py               # Streamlit interface
│
├── utils/
│   ├── __init__.py
│   ├── logger.py            # Logging setup
│   └── validators.py        # Input validation
│
├── tests/
│   ├── conftest.py
│   ├── test_services.py
│   └── test_integration.py
│
├── .env.example
├── config.yaml
└── requirements.txt
```

---

## Step-by-Step Refactoring Plan

### **Phase 1: Foundation (Week 1)**
1. ✅ Create config management system
2. ✅ Define Pydantic models
3. ✅ Create custom exceptions
4. ✅ Setup logging
5. ✅ Replace OpenAI with HuggingFace transformers

### **Phase 2: Core Services (Week 2)**
1. ✅ Implement DocumentService (PDF, TXT parsing)
2. ✅ Implement EmbeddingService (sentence-transformers)
3. ✅ Implement LLMService (distilgpt2)
4. ✅ Implement RetrievalService
5. ✅ Refactor RAGEngine → RAGService

### **Phase 3: Infrastructure (Week 3)**
1. ✅ Enhance VectorStore with metadata
2. ✅ Implement MetadataStore (DuckDB)
3. ✅ Add caching layer
4. ✅ Connection pooling for DB

### **Phase 4: UI & Integration (Week 4)**
1. ✅ Build Streamlit app
2. ✅ Add comprehensive tests
3. ✅ Error handling & validation
4. ✅ Performance monitoring

---

## Key Design Patterns

### **1. Dependency Injection**
```python
# Before: Tightly coupled
class RAGEngine:
    def __init__(self):
        self.llm = LLMHandler()  # Hard dependency

# After: Loose coupling
class RAGService:
    def __init__(self, llm_service: LLMService, retrieval_service: RetrievalService):
        self.llm = llm_service
        self.retrieval = retrieval_service
```

### **2. Service Layer Pattern**
- Services: Single responsibility, reusable business logic
- Infrastructure: Data access, external API calls
- Separation of concerns

### **3. Pydantic Models for Type Safety**
```python
class Document(BaseModel):
    id: str
    content: str
    source: str
    metadata: dict
    embedding: Optional[List[float]] = None
    
class Query(BaseModel):
    question: str
    top_k: int = 3
```

### **4. Configuration Management**
```python
class Settings(BaseSettings):
    MODEL_NAME: str = "distilgpt2"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_PATH: str = "./data/chroma_db"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
```

### **5. Error Handling**
```python
class RecallifyException(Exception):
    """Base exception"""

class DocumentProcessingError(RecallifyException):
    """Document loading failed"""

class EmbeddingError(RecallifyException):
    """Embedding generation failed"""
```

---

## Migration Strategy

### **Quick Wins (Start Here)**
1. Replace OpenAI with HuggingFace pipeline (2 hours)
2. Create config.py for settings (1 hour)
3. Add logging setup (1 hour)
4. Create requirements_new.txt (30 min)

### **Medium Term**
1. Refactor LLMHandler → LLMService
2. Implement DocumentProcessor (PDF parsing)
3. Add Pydantic models
4. Create service layer

### **Long Term**
1. Add DuckDB metadata store
2. Implement caching
3. Build Streamlit UI
4. Add comprehensive tests
5. Performance optimization

---

## Recommendations

| Aspect | Current | Recommended |
|--------|---------|-------------|
| LLM | OpenAI API | distilgpt2 (transformers) |
| Embeddings | None (ChromaDB default) | sentence-transformers |
| Config | Hardcoded | Pydantic Settings + .env |
| DB | ChromaDB only | ChromaDB + DuckDB |
| Caching | None | Redis/SQLite cache |
| Async | No | FastAPI for backend |
| Testing | Basic | pytest + pytest-asyncio |
| Logging | print() | Python logging module |
| Validation | None | Pydantic models |

---

## Next Steps

Which phase do you want to start with?

1. **Quick Foundation Fix**: Replace OpenAI → HF, add config
2. **Full Refactor**: Complete rewrite with clean architecture
3. **Specific Module**: Focus on document processor, RAG engine, or UI
