"""RAG Engine - Main orchestration component"""
import time
from typing import List, Optional
from .vector_store import VectorStore
from .llm_handler import LLMHandler
from .document_processor import DocumentProcessor
from core.models import RAGResponse, RetrievedDocument
from core.exceptions import RetrievalError
from utils.logger import logger
from config.settings import settings

class RAGEngine:
    def __init__(self, llm_handler: Optional[LLMHandler] = None):
        """Initialize RAG components with dependency injection"""
        logger.info("🚀 Initializing RAG Engine...")
        
        try:
            self.vector_store = VectorStore()
            self.llm_handler = llm_handler or LLMHandler()
            self.document_processor = DocumentProcessor()
            logger.info("✅ RAG Engine ready!")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Engine: {str(e)}")
            raise
    
    def load_documents(self) -> int:
        """Load and index documents"""
        logger.info("\n📚 Loading documents...")
        try:
            documents, ids = self.document_processor.process_documents()
            
            if documents:
                self.vector_store.add_documents(documents, ids)
                logger.info(f"✅ Loaded {len(documents)} documents")
                return len(documents)
            return 0
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
    
    def retrieve_context(self, query: str, n_results: int = None) -> List[RetrievedDocument]:
        """Retrieve relevant documents"""
        n_results = n_results or settings.TOP_K_RESULTS
        try:
            results = self.vector_store.search(query, n_results=n_results)
            
            if results and results['documents']:
                # Convert to RetrievedDocument models
                retrieved = []
                for i, doc in enumerate(results['documents'][0]):
                    similarity = results['distances'][0][i] if results.get('distances') else 0.0
                    retrieved.append(RetrievedDocument(
                        id=results['ids'][0][i] if results.get('ids') else f"doc_{i}",
                        content=doc,
                        source="chroma_db",
                        similarity_score=1 - similarity,  # Convert distance to similarity
                        metadata=results.get('metadatas', [{}])[0][i] if results.get('metadatas') else {}
                    ))
                return retrieved
            return []
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            raise RetrievalError(f"Failed to retrieve context: {str(e)}")
    
    def generate_answer(self, query: str) -> tuple:
        """Generate answer using context + LLM"""
        try:
            context_docs = self.retrieve_context(query, n_results=settings.TOP_K_RESULTS)
            
            if not context_docs:
                return "No relevant documents found.", []
            
            context_text = "\n\n".join([doc.content for doc in context_docs])
            
            prompt = f"""Based on the following context, answer the question concisely.

Context:
{context_text}

Question: {query}

Answer:"""
            
            answer = self.llm_handler.generate_response(prompt, max_tokens=settings.LLM_MAX_LENGTH)
            return answer, context_docs
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def query(self, question: str) -> RAGResponse:
        """Main query method - process question and return RAG response"""
        start_time = time.time()
        logger.info(f"\n🔍 Query: {question}")
        
        try:
            answer, context_docs = self.generate_answer(question)
            processing_time = time.time() - start_time
            
            response = RAGResponse(
                question=question,
                answer=answer,
                retrieved_documents=context_docs,
                processing_time=processing_time
            )
            
            logger.info(f"📝 Answer: {answer}")
            logger.info(f"⏱️  Processing time: {processing_time:.2f}s")
            
            return response
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        return self.vector_store.get_collection_stats()

if __name__ == "__main__":
    rag = RAGEngine()
    rag.load_documents()
    rag.query("What is Recallify?")
    print(f"\n📊 Stats: {rag.get_stats()}")
