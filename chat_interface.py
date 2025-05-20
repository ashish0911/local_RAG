"""
Chat interface for interacting with the RAG system.
"""

import os
from typing import Optional

from config import RAGConfig
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_chain import RAGChain


class ChatInterface:
    """Manages the chat interface."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize with configuration."""
        self.config = config or RAGConfig()
        self.document_processor = DocumentProcessor(self.config)
        self.vector_store = VectorStore(self.config)
        self.rag_chain = RAGChain(self.config)
    
    def initialize(self) -> None:
        """Initialize the system, indexing documents if needed."""
        if not os.path.exists(self.config.chroma_dir):
            self.index_documents()
        else:
            self.vector_store.create_or_load_vector_db()
    
    def index_documents(self) -> None:
        """Index documents in the vector database."""
        print(f"Loading documents from {self.config.documents_dir}...")
        documents = self.document_processor.load_documents()
        print(f"Loaded {len(documents)} documents")
        
        print("Splitting documents into chunks...")
        chunks = self.document_processor.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        print("Creating vector database...")
        self.vector_store.create_or_load_vector_db(chunks)
        print(f"Vector database created with {len(chunks)} chunks")
    
    def query(self, query: str) -> str:
        """Query the system."""
        retriever = self.vector_store.get_retriever()
        return self.rag_chain.query(
            query, 
            retriever=lambda q: self.rag_chain.format_docs(retriever.invoke(q))
        )
    
    def run_chat_loop(self) -> None:
        """Run an interactive chat loop."""
        # First, initialize the system
        self.initialize()
        
        print("\n=== Local RAG Chat ===")
        print("Type 'exit' to quit\n")
        
        while True:
            user_input = input("Question: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            response = self.query(user_input)
            print("\nAnswer:", response, "\n")
