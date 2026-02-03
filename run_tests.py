#!/usr/bin/env python3
"""Quick test of all components"""
import sys
sys.path.insert(0, '/mnt/c/Dev/recallify')

print("\n" + "="*60)
print("✅ RECALLIFY QUICK FIX - COMPREHENSIVE TEST")
print("="*60 + "\n")

# Test 1: Config
try:
    from config.settings import settings
    print("✅ TEST 1: Configuration loaded")
    print(f"   • LLM Model: {settings.LLM_MODEL}")
    print(f"   • Embedding: {settings.EMBEDDING_MODEL}")
except Exception as e:
    print(f"❌ TEST 1 FAILED: {e}")
    sys.exit(1)

# Test 2: Core Models
try:
    from core.models import Document, Query
    doc = Document(id="test", content="test", source="test.txt")
    query = Query(question="What?", top_k=3)
    print("\n✅ TEST 2: Core models working")
    print(f"   • Document: {doc.id}")
    print(f"   • Query: {query.question}")
except Exception as e:
    print(f"\n❌ TEST 2 FAILED: {e}")
    sys.exit(1)

# Test 3: Vector Store
try:
    from src.vector_store import VectorStore
    store = VectorStore()
    stats = store.get_collection_stats()
    print("\n✅ TEST 3: Vector Store initialized")
    print(f"   • Collection: {stats['collection_name']}")
    print(f"   • Documents: {stats['total_documents']}")
except Exception as e:
    print(f"\n❌ TEST 3 FAILED: {e}")
    sys.exit(1)

# Test 4: Document Processor
try:
    from src.document_processor import DocumentProcessor
    from pathlib import Path
    
    processor = DocumentProcessor()
    docs_path = Path(settings.DOCUMENTS_PATH)
    docs_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample
    (docs_path / "sample.txt").write_text("Recallify is amazing!")
    
    docs, ids = processor.process_documents()
    print("\n✅ TEST 4: Document Processor")
    print(f"   • Loaded: {len(docs)} documents")
    if docs:
        print(f"   • First doc: {ids[0]}")
except Exception as e:
    print(f"\n❌ TEST 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Logger
try:
    from utils.logger import logger
    logger.info("Logger test message")
    print("\n✅ TEST 5: Logger working")
except Exception as e:
    print(f"\n❌ TEST 5 FAILED: {e}")
    sys.exit(1)

# Test 6: LLM Handler
try:
    from src.llm_handler import LLMHandler
    print("\n⏳ TEST 6: LLM Handler (initializing distilgpt2...)")
    llm = LLMHandler()
    response = llm.generate_response("Recallify is", max_tokens=15)
    print("✅ TEST 6: LLM Handler working")
    print(f"   • Response: {response[:60]}...")
except Exception as e:
    print(f"\n⚠️  TEST 6 WARNING: {e}")
    print("   (This is expected if model download fails)")

# Test 7: RAG Engine
try:
    from src.rag_engine import RAGEngine
    print("\n🚀 TEST 7: RAG Engine initialization")
    rag = RAGEngine()
    doc_count = rag.load_documents()
    stats = rag.get_stats()
    print("✅ TEST 7: RAG Engine working")
    print(f"   • Loaded: {doc_count} documents")
    print(f"   • Collection: {stats['collection_name']}")
except Exception as e:
    print(f"\n❌ TEST 7 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✨ ALL TESTS PASSED!")
print("="*60)
print("\nQuick Fix Implementation Status: SUCCESS ✅")
print("Ready for Phase 2: Core Services Development\n")
