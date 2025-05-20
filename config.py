"""
Configuration settings for the Local RAG system.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class RAGConfig:
    """Configuration for the RAG system."""
    # Paths
    documents_dir: str = "data/documents"
    chroma_dir: str = "data/chroma_db"
    
    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # LLM settings
    model_name: str = "deepseek-r1:1.5b"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0
    
    # Retrieval settings
    top_k: int = 4
    
    # Custom prompt template
    prompt_template: Optional[str] = None
    
    def __post_init__(self):
        """Ensure directories exist."""
        os.makedirs(self.documents_dir, exist_ok=True)
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """Create a config from a dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__annotations__}


# Default configuration
default_config = RAGConfig()
