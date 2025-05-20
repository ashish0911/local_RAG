"""
Document processing module for loading and chunking documents.
"""

import os
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document

from config import RAGConfig


class DocumentProcessor:
    """Handles document loading and chunking."""
    
    def __init__(self, config: RAGConfig):
        """Initialize with configuration."""
        self.config = config
    
    def load_documents(self, directory: Optional[str] = None) -> List[Document]:
        """Load documents from a directory."""
        directory = directory or self.config.documents_dir
        
        if not os.path.exists(directory):
            raise ValueError(f"Directory {directory} does not exist")
        
        loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        return loader.load()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    
    def process_documents(self, directory: Optional[str] = None) -> List[Document]:
        """Load and split documents in one step."""
        documents = self.load_documents(directory)
        return self.split_documents(documents)
