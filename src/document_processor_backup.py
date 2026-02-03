"""Document processor for loading and processing documents"""
import os
from pathlib import Path
from typing import List, Tuple
from core.exceptions import DocumentProcessingError
from utils.logger import logger
from config.settings import settings

class DocumentProcessor:
    """Process documents from disk"""
    
    def __init__(self, documents_path: str = None):
        """Initialize document processor"""
        self.documents_path = documents_path or settings.DOCUMENTS_PATH
        logger.info(f"DocumentProcessor initialized with path: {self.documents_path}")
    
    def process_documents(self) -> Tuple[List[str], List[str]]:
        """
        Load and process documents from directory
        
        Returns:
            Tuple of (documents, document_ids)
        """
        try:
            documents = []
            doc_ids = []
            
            # Create documents path if it doesn't exist
            Path(self.documents_path).mkdir(parents=True, exist_ok=True)
            
            # Walk through all documents
            for idx, file_path in enumerate(Path(self.documents_path).rglob("*")):
                if file_path.is_file() and file_path.suffix.lower().lstrip(".") in settings.SUPPORTED_FORMATS:
                    try:
                        content = self._read_file(file_path)
                        if content:
                            documents.append(content)
                            doc_ids.append(f"doc_{idx}_{file_path.stem}")
                            logger.info(f"✅ Loaded: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to load {file_path.name}: {str(e)}")
            
            if documents:
                logger.info(f"📚 Loaded {len(documents)} documents successfully")
            else:
                logger.warning("⚠️  No documents found. Add .txt, .pdf, or .md files to data/documents/")
            
            return documents, doc_ids
        
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise DocumentProcessingError(f"Failed to process documents: {str(e)}")
    
    def _read_file(self, file_path: Path) -> str:
        """Read content from a file"""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == ".txt" or suffix == ".md":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            
            elif suffix == ".pdf":
                try:
                    import PyPDF2
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text()
                        return text
                except ImportError:
                    logger.warning("PyPDF2 not installed. Skipping PDF file.")
                    return ""
            
            return ""
        
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return ""

if __name__ == "__main__":
    processor = DocumentProcessor()
    docs, ids = processor.process_documents()
    print(f"Processed {len(docs)} documents")
