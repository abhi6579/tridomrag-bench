import chromadb
import os

class VectorStore:
    def __init__(self, persist_directory="./data/chroma_db"):
        """Initialize ChromaDB vector store"""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="recallify_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents, ids=None):
        """Add documents to vector store"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(documents=documents, ids=ids)
        print(f"✅ Added {len(documents)} documents to vector store")
    
    def search(self, query, n_results=3):
        """Search for similar documents"""
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results
    
    def get_collection_stats(self):
        """Get statistics about collection"""
        count = self.collection.count()
        return {"total_documents": count, "collection_name": "recallify_documents"}

if __name__ == "__main__":
    store = VectorStore()
    stats = store.get_collection_stats()
    print(f"Vector Store Stats: {stats}")
