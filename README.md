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

- Complete modular RAG implementation with:
  - Document loading and processing
  - Text chunking with RecursiveCharacterTextSplitter
  - Vector embeddings using Ollama's nomic-embed-text model
  - Local vector database with ChromaDB
  - Retrieval and generation with Langchain
  - Interactive chat interface
  - Configurable via command line arguments or JSON config file

## Hardware Requirements

- 8GB+ RAM
- 8-core CPU

## Software Requirements

- Python 3.13+
- Ollama (https://ollama.com/download)
- Required Python packages (installed via pyproject.toml):
  - langchain-ollama
  - langchain-community
  - langchain-chroma
  - chromadb
  - langchain-core
  - langchain

## Installation

```bash
# Install ollama
curl https://ollama.ai/install.sh | sh
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text

# Clone the repository
git clone https://github.com/your-repo/local-rag.git
cd local-rag

# Install dependencies
uv pip install -e .

# Run the application
python main.py
```

## Usage

1. **Add documents**: Place your text files in the `data/documents` directory
2. **Run the application**: Execute `python main.py`
3. **Ask questions**: The system will automatically index your documents and allow you to ask questions about them
4. **Exit**: Type 'exit' or 'quit' to end the session

### Command Line Options

```bash
python main.py --help
```

Available options:
- `--config`: Path to a JSON configuration file
- `--documents-dir`: Directory containing documents
- `--model`: Ollama model name for generation
- `--embedding-model`: Ollama model name for embeddings
- `--reindex`: Force reindexing of documents

Example:
```bash
python main.py --model llama3.2 --embedding-model nomic-embed-text --reindex
```

## Project Structure

The codebase is organized into modular components:

- `main.py`: Entry point that ties everything together
- `config.py`: Configuration settings and management
- `document_processor.py`: Document loading and chunking
- `vector_store.py`: Vector database operations
- `rag_chain.py`: RAG pipeline implementation
- `chat_interface.py`: User interaction

## How It Works

1. **Document Processing**: The system loads text documents from the configured documents directory
2. **Chunking**: Documents are split into smaller chunks for better retrieval
3. **Embedding**: Each chunk is converted to a vector embedding using Ollama's embedding model
4. **Storage**: Embeddings are stored in a local ChromaDB vector database
5. **Retrieval**: When you ask a question, the system finds the most relevant chunks
6. **Generation**: The LLM generates an answer based on the retrieved context

## Customization

You can customize the system by:

1. **Using a configuration file**:
   Create a JSON file with your settings and pass it with `--config`:
   ```json
   {
     "documents_dir": "custom/docs",
     "model_name": "llama3.2",
     "chunk_size": 500
   }
   ```

2. **Using command line arguments**:
   Override specific settings with command line options.

3. **Modifying the RAGConfig class**:
   Edit the `config.py` file to add new configuration options.
