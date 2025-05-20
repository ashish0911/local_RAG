"""
RAG pipeline implementation for retrieval and generation.
"""

from typing import List, Optional, Callable, Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from config import RAGConfig


class RAGChain:
    """Manages the RAG pipeline."""
    
    DEFAULT_PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    def __init__(self, config: RAGConfig):
        """Initialize with configuration."""
        self.config = config
        self.llm = ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
        )
    
    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format documents for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_chain(self, retriever: Callable[[str], List[Document]]) -> Any:
        """Create a RAG chain with the given retriever."""
        # Define the prompt template
        template = self.config.prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the RAG chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(self, query: str, retriever: Callable[[str], List[Document]]) -> str:
        """Query the RAG chain."""
        chain = self.create_chain(retriever)
        return chain.invoke(query)
