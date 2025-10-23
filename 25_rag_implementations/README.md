# 🚀 25 RAG Implementation Project

A comprehensive implementation of 25 different Retrieval-Augmented Generation (RAG) systems using Local Ollama LLM.

## Quick Start (Windows)

1. **Setup Environment**:
   ```cmd
   cd 25_rag_implementations
   venv\Scripts\activate
   ```

2. **Install Ollama** (if not already installed):
   - Visit: https://ollama.ai/download
   - Download Windows installer

3. **Test Installation**:
   ```cmd
   python main.py test
   ```

4. **Run a Query**:
   ```cmd
   python main.py query standard --query "What is machine learning?"
   ```

## Project Structure

```
25_rag_implementations/
├── common/                 # Shared components
│   ├── models/            # Base models and interfaces
│   ├── utils/             # Utility functions
│   └── config/            # Configuration management
├── rag_types/             # Individual RAG implementations
│   ├── 01_standard/       # Standard RAG
│   ├── 02_conversational/ # Conversational RAG
│   └── ...               # 23 more RAG types
├── data/                  # Test data and documents
├── tests/                 # Test suites
├── demos/                 # Demo applications
└── logs/                  # Performance logs
```

## Available Commands

- `python main.py query <rag_type> --query "^<your_query^>"` - Process a query
- `python main.py test` - Run comprehensive tests

## Development Status

- ✅ Project structure created
- 🚧 RAG implementations in progress
- 🚧 Testing framework being built

## Next Steps

Follow the Technical Implementation Plan to build each RAG type systematically.
