#!/bin/bash
# Quick Start Setup Script for 25 RAG Implementation Project

echo "🚀 25 RAG Implementation Project - Quick Setup"
echo "=============================================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $python_version"

# Create project directory structure
echo "📁 Creating project structure..."
mkdir -p 25_rag_implementations/{common/{models,utils,config},rag_types,data,tests,demos,logs}
mkdir -p 25_rag_implementations/rag_types/{01_standard,02_conversational,03_graph,04_fusion,05_adaptive}

# Create virtual environment
echo "🔧 Creating virtual environment..."
python -m venv 25_rag_implementations/venv

# Activate virtual environment (Linux/Mac)
source 25_rag_implementations/venv/bin/activate

# Create requirements.txt
echo "📦 Creating requirements.txt..."
cat > 25_rag_implementations/requirements.txt << EOF
# Core RAG Dependencies
ollama>=0.1.0
langchain>=0.1.0
chromadb>=0.4.0
sentence-transformers>=2.2.0

# Web Interfaces
streamlit>=1.28.0
gradio>=3.50.0
fastapi>=0.104.0
uvicorn>=0.24.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0

# NLP Libraries
spacy>=3.7.0
transformers>=4.35.0
torch>=2.1.0

# Graph Processing
networkx>=3.2.0
neo4j>=5.13.0

# Testing & Quality
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Monitoring & Logging
psutil>=5.9.0
wandb>=0.16.0

# Development Tools
jupyter>=1.0.0
black>=23.0.0
flake8>=6.1.0

# Vector Databases
pinecone-client>=2.2.0
weaviate-client>=3.25.0
qdrant-client>=1.6.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.4.0
click>=8.1.0
tqdm>=4.66.0
EOF

echo "⬇️ Installing Python dependencies..."
pip install -r 25_rag_implementations/requirements.txt

# Download spaCy model
echo "📚 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️ Ollama is not installed. Please install Ollama first:"
    echo "   Visit: https://ollama.ai/download"
    echo "   Or run: curl https://ollama.ai/install.sh | sh"
else
    echo "✅ Ollama is installed"
    
    # Pull essential models
    echo "📥 Pulling essential Ollama models (this may take a while)..."
    ollama pull llama2
    echo "✅ llama2 model ready"
    
    # Test Ollama
    echo "🧪 Testing Ollama installation..."
    test_response=$(ollama run llama2 "Hello, world!" --timeout 30)
    if [[ $? -eq 0 ]]; then
        echo "✅ Ollama is working correctly"
    else
        echo "⚠️ Ollama test failed. Please check your installation"
    fi
fi

# Create basic configuration file
echo "⚙️ Creating configuration file..."
cat > 25_rag_implementations/common/config/default_config.yaml << EOF
# Default Configuration for 25 RAG Implementation Project

ollama:
  host: "localhost:11434"
  default_model: "llama2"
  timeout: 60
  temperature: 0.7
  max_tokens: 512

embeddings:
  model: "all-MiniLM-L6-v2"
  device: "cpu"  # or "cuda" if GPU available
  batch_size: 32

vector_store:
  type: "chromadb"  # chromadb, pinecone, weaviate, qdrant
  persist_directory: "./vector_db"
  collection_prefix: "rag_"

retrieval:
  top_k: 5
  chunk_size: 512
  chunk_overlap: 50
  similarity_threshold: 0.7

generation:
  max_context_length: 4000
  response_max_tokens: 512
  temperature: 0.7
  top_p: 0.9

monitoring:
  enable_logging: true
  log_file: "./logs/performance.jsonl"
  log_level: "INFO"

testing:
  test_data_path: "./data/test_documents"
  benchmark_queries_path: "./data/benchmark_queries.json"
  performance_threshold:
    max_response_time: 10.0  # seconds
    min_confidence: 0.3
EOF

# Create project structure files
echo "📝 Creating project structure files..."

# Main entry point
cat > 25_rag_implementations/main.py << EOF
#!/usr/bin/env python3
"""
25 RAG Implementation Project - Main Entry Point
"""
import click
import yaml
from pathlib import Path
from common.config.config_manager import ConfigManager
from common.utils.logger import setup_logging

@click.group()
@click.option('--config', default='common/config/default_config.yaml', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """25 RAG Implementation Project CLI"""
    ctx.ensure_object(dict)
    
    # Load configuration
    config_path = Path(config)
    if not config_path.exists():
        click.echo(f"❌ Configuration file not found: {config}")
        raise click.Abort()
    
    with open(config_path, 'r') as f:
        ctx.obj['config'] = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(ctx.obj['config'].get('monitoring', {}).get('log_level', 'INFO'))

@cli.command()
@click.argument('rag_type', default='standard')
@click.option('--query', prompt='Enter your query', help='Query to process')
@click.pass_context
def query(ctx, rag_type, query):
    """Process a query using specified RAG type"""
    click.echo(f"🔍 Processing query with {rag_type} RAG...")
    click.echo(f"Query: {query}")
    
    # Implementation will be added as RAG types are developed
    click.echo("🚧 Implementation in progress...")

@cli.command()
@click.pass_context
def test(ctx):
    """Run comprehensive tests on all RAG implementations"""
    click.echo("🧪 Running comprehensive RAG tests...")
    
    from tests.test_framework import run_all_rag_tests
    run_all_rag_tests()

@cli.command()
@click.pass_context
def benchmark(ctx):
    """Run performance benchmarks"""
    click.echo("📊 Running performance benchmarks...")
    
    # Implementation will be added
    click.echo("🚧 Benchmark implementation in progress...")

@cli.command()
@click.option('--port', default=8501, help='Port for web interface')
@click.pass_context
def web(ctx, port):
    """Launch web interface"""
    click.echo(f"🌐 Launching web interface on port {port}...")
    
    # Implementation will be added
    click.echo("🚧 Web interface implementation in progress...")

if __name__ == '__main__':
    cli()
EOF

# Configuration manager
mkdir -p 25_rag_implementations/common/config
cat > 25_rag_implementations/common/config/config_manager.py << EOF
"""Configuration Management for RAG Implementations"""
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class OllamaConfig:
    host: str = "localhost:11434"
    default_model: str = "llama2"
    timeout: int = 60
    temperature: float = 0.7
    max_tokens: int = 512

@dataclass
class EmbeddingConfig:
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32

@dataclass
class VectorStoreConfig:
    type: str = "chromadb"
    persist_directory: str = "./vector_db"
    collection_prefix: str = "rag_"

class ConfigManager:
    def __init__(self, config_path: str = "common/config/default_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_ollama_config(self) -> OllamaConfig:
        """Get Ollama configuration"""
        ollama_config = self.config.get('ollama', {})
        return OllamaConfig(**ollama_config)
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration"""
        embedding_config = self.config.get('embeddings', {})
        return EmbeddingConfig(**embedding_config)
    
    def get_vector_store_config(self) -> VectorStoreConfig:
        """Get vector store configuration"""
        vs_config = self.config.get('vector_store', {})
        return VectorStoreConfig(**vs_config)
EOF

# Logger utility
mkdir -p 25_rag_implementations/common/utils
cat > 25_rag_implementations/common/utils/logger.py << EOF
"""Logging utilities for RAG implementations"""
import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)
EOF

# Create __init__.py files
touch 25_rag_implementations/common/__init__.py
touch 25_rag_implementations/common/models/__init__.py
touch 25_rag_implementations/common/utils/__init__.py
touch 25_rag_implementations/common/config/__init__.py
touch 25_rag_implementations/rag_types/__init__.py
touch 25_rag_implementations/tests/__init__.py

# Create README
cat > 25_rag_implementations/README.md << EOF
# 🚀 25 RAG Implementation Project

A comprehensive implementation of 25 different Retrieval-Augmented Generation (RAG) systems using Local Ollama LLM.

## Quick Start

1. **Setup Environment**:
   \`\`\`bash
   cd 25_rag_implementations
   source venv/bin/activate  # Linux/Mac
   # or
   venv\\Scripts\\activate  # Windows
   \`\`\`

2. **Install Ollama** (if not already installed):
   - Visit: https://ollama.ai/download
   - Or run: \`curl https://ollama.ai/install.sh | sh\`

3. **Test Installation**:
   \`\`\`bash
   python main.py test
   \`\`\`

4. **Run a Query**:
   \`\`\`bash
   python main.py query standard --query "What is machine learning?"
   \`\`\`

## Project Structure

\`\`\`
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
\`\`\`

## Available Commands

- \`python main.py query <rag_type> --query "<your_query>"\` - Process a query
- \`python main.py test\` - Run comprehensive tests
- \`python main.py benchmark\` - Run performance benchmarks
- \`python main.py web\` - Launch web interface

## Configuration

Edit \`common/config/default_config.yaml\` to customize:
- Ollama model settings
- Vector database configuration
- Embedding model selection
- Performance thresholds

## Development Status

- ✅ Project structure created
- ✅ Base framework implemented
- 🚧 RAG implementations in progress
- 🚧 Web interface in development
- 🚧 Testing framework being built

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your RAG type following the base interface
4. Add comprehensive tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details
EOF

echo "✅ Project setup complete!"
echo ""
echo "📋 Next Steps:"
echo "1. cd 25_rag_implementations"
echo "2. source venv/bin/activate  (Linux/Mac) or venv\\Scripts\\activate (Windows)"
echo "3. python main.py test"
echo "4. Start implementing RAG types following the Technical Implementation Plan"
echo ""
echo "🚀 You're ready to build 25 RAG systems!"