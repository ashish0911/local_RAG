"""
Vector database operations for storing and retrieving document embeddings.
"""

import os
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings

from config import RAGConfig


class VectorStore:
    """Manages vector database operations."""
    
    def __init__(self, config: RAGConfig):
        """Initialize with configuration."""
        self.config = config
        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model,
        )
        self._vector_db = None
    
    def create_or_load_vector_db(self, documents: Optional[List[Document]] = None) -> Chroma:
        """Create a new vector database or load an existing one."""
        if documents and not os.path.exists(self.config.chroma_dir):
            # Create a new vector database
            vector_db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.config.chroma_dir,
            )
            # No need to call persist() as Chroma automatically persists when persist_directory is set
            self._vector_db = vector_db
            return vector_db
        else:
            # Load existing vector database
            self._vector_db = Chroma(
                persist_directory=self.config.chroma_dir,
                embedding_function=self.embeddings,
            )
            return self._vector_db
    
    def get_retriever(self, top_k: Optional[int] = None):
        """Get a retriever from the vector database."""
        if self._vector_db is None:
            self._vector_db = self.create_or_load_vector_db()
        
        return self._vector_db.as_retriever(
            search_kwargs={"k": top_k or self.config.top_k}
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector database."""
        if self._vector_db is None:
            self._vector_db = self.create_or_load_vector_db(documents)
        else:
            self._vector_db.add_documents(documents)
    
    def clear(self) -> None:
        """Clear the vector database."""
        if os.path.exists(self.config.chroma_dir):
            import shutil
            shutil.rmtree(self.config.chroma_dir)
        self._vector_db = None
