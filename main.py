"""
Local RAG (Retrieval-Augmented Generation) system.

This is the main entry point for the application.
"""

import argparse
import json
import os
from typing import Dict, Any

from config import RAGConfig
from chat_interface import ChatInterface


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Local RAG System")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--documents-dir",
        type=str,
        help="Directory containing documents"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Ollama model name for generation"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Ollama model name for embeddings"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindexing of documents"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    if not os.path.exists(config_path):
        raise ValueError(f"Config file {config_path} does not exist")

    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main entry point."""
    args = parse_args()

    # Create configuration
    if args.config:
        config_dict = load_config(args.config)
        config = RAGConfig.from_dict(config_dict)
    else:
        config = RAGConfig()

    # Override config with command line arguments
    if args.documents_dir:
        config.documents_dir = args.documents_dir
    if args.model:
        config.model_name = args.model
    if args.embedding_model:
        config.embedding_model = args.embedding_model

    # Create chat interface
    chat = ChatInterface(config)

    # Force reindexing if requested
    if args.reindex and os.path.exists(config.chroma_dir):
        import shutil
        shutil.rmtree(config.chroma_dir)

    # Run chat loop
    chat.run_chat_loop()


if __name__ == "__main__":
    main()
