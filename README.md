# Local RAG

A lightweight, fully local Retrieval Augmented Generation (RAG) system that runs entirely on your machine. No cloud services, no API keys, just your data and your compute.

## What is Local RAG?

Local RAG is a privacy-focused implementation of the Retrieval Augmented Generation pattern. It enables you to:

- Run an LLM (Large Language Model) on your local machine
- Create and manage vector embeddings for your documents locally
- Search and retrieve relevant context from your personal knowledge base
- Generate responses augmented with your retrieved data
- Maintain complete privacy with no data leaving your machine

This project combines the power of modern LLMs with the flexibility of vector databases while respecting user privacy and maintaining data sovereignty.

## Current Status

- Loading chat and embedding models using ollama

## Hardware Requirements

- 8GB+ RAM
- 8-core CPU

## Software Requirements

- Python 3.13+
- Ollama (https://ollama.com/download)
- Langchain Ollama (https://github.com/ollama/langchain-ollama)

## Installation

```bash
# Install ollama
curl https://ollama.ai/install.sh | sh
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text

# Clone the repository
git clone https://github.com/your-repo/local-rag.git
cd local-rag
uv run main.py
```
