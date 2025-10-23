@echo off
REM Quick Start Setup Script for 25 RAG Implementation Project (Windows)

echo 🚀 25 RAG Implementation Project - Quick Setup (Windows)
echo =======================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    echo Visit: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version') do set python_version=%%i
echo ✅ Python version: %python_version%

REM Create project directory structure
echo 📁 Creating project structure...
mkdir 25_rag_implementations 2>nul
mkdir 25_rag_implementations\common 2>nul
mkdir 25_rag_implementations\common\models 2>nul
mkdir 25_rag_implementations\common\utils 2>nul
mkdir 25_rag_implementations\common\config 2>nul
mkdir 25_rag_implementations\rag_types 2>nul
mkdir 25_rag_implementations\rag_types\01_standard 2>nul
mkdir 25_rag_implementations\rag_types\02_conversational 2>nul
mkdir 25_rag_implementations\rag_types\03_graph 2>nul
mkdir 25_rag_implementations\rag_types\04_fusion 2>nul
mkdir 25_rag_implementations\rag_types\05_adaptive 2>nul
mkdir 25_rag_implementations\data 2>nul
mkdir 25_rag_implementations\tests 2>nul
mkdir 25_rag_implementations\demos 2>nul
mkdir 25_rag_implementations\logs 2>nul

REM Create virtual environment
echo 🔧 Creating virtual environment...
cd 25_rag_implementations
python -m venv venv

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate

REM Create requirements.txt
echo 📦 Creating requirements.txt...
(
echo # Core RAG Dependencies
echo ollama^>=0.1.0
echo langchain^>=0.1.0
echo chromadb^>=0.4.0
echo sentence-transformers^>=2.2.0
echo.
echo # Web Interfaces
echo streamlit^>=1.28.0
echo gradio^>=3.50.0
echo fastapi^>=0.104.0
echo uvicorn^>=0.24.0
echo.
echo # Data Processing
echo numpy^>=1.24.0
echo pandas^>=2.0.0
echo scipy^>=1.11.0
echo.
echo # NLP Libraries
echo spacy^>=3.7.0
echo transformers^>=4.35.0
echo torch^>=2.1.0
echo.
echo # Graph Processing
echo networkx^>=3.2.0
echo neo4j^>=5.13.0
echo.
echo # Testing ^& Quality
echo pytest^>=7.4.0
echo pytest-asyncio^>=0.21.0
echo pytest-cov^>=4.1.0
echo.
echo # Monitoring ^& Logging
echo psutil^>=5.9.0
echo wandb^>=0.16.0
echo.
echo # Development Tools
echo jupyter^>=1.0.0
echo black^>=23.0.0
echo flake8^>=6.1.0
echo.
echo # Vector Databases
echo pinecone-client^>=2.2.0
echo weaviate-client^>=3.25.0
echo qdrant-client^>=1.6.0
echo.
echo # Utilities
echo python-dotenv^>=1.0.0
echo pydantic^>=2.4.0
echo click^>=8.1.0
echo tqdm^>=4.66.0
) > requirements.txt

echo ⬇️ Installing Python dependencies...
pip install -r requirements.txt

REM Download spaCy model
echo 📚 Downloading spaCy English model...
python -m spacy download en_core_web_sm

REM Check if Ollama is installed
ollama --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Ollama is not installed. Please install Ollama first:
    echo    Visit: https://ollama.ai/download
    echo    Download the Windows installer
) else (
    echo ✅ Ollama is installed
    
    REM Pull essential models
    echo 📥 Pulling essential Ollama models ^(this may take a while^)...
    ollama pull llama2
    echo ✅ llama2 model ready
    
    REM Test Ollama
    echo 🧪 Testing Ollama installation...
    echo Hello, world! | ollama run llama2 >nul 2>&1
    if errorlevel 1 (
        echo ⚠️ Ollama test failed. Please check your installation
    ) else (
        echo ✅ Ollama is working correctly
    )
)

REM Create basic configuration file
echo ⚙️ Creating configuration file...
(
echo # Default Configuration for 25 RAG Implementation Project
echo.
echo ollama:
echo   host: "localhost:11434"
echo   default_model: "llama2"
echo   timeout: 60
echo   temperature: 0.7
echo   max_tokens: 512
echo.
echo embeddings:
echo   model: "all-MiniLM-L6-v2"
echo   device: "cpu"  # or "cuda" if GPU available
echo   batch_size: 32
echo.
echo vector_store:
echo   type: "chromadb"  # chromadb, pinecone, weaviate, qdrant
echo   persist_directory: "./vector_db"
echo   collection_prefix: "rag_"
echo.
echo retrieval:
echo   top_k: 5
echo   chunk_size: 512
echo   chunk_overlap: 50
echo   similarity_threshold: 0.7
echo.
echo generation:
echo   max_context_length: 4000
echo   response_max_tokens: 512
echo   temperature: 0.7
echo   top_p: 0.9
echo.
echo monitoring:
echo   enable_logging: true
echo   log_file: "./logs/performance.jsonl"
echo   log_level: "INFO"
echo.
echo testing:
echo   test_data_path: "./data/test_documents"
echo   benchmark_queries_path: "./data/benchmark_queries.json"
echo   performance_threshold:
echo     max_response_time: 10.0  # seconds
echo     min_confidence: 0.3
) > common\config\default_config.yaml

REM Create main entry point
echo 📝 Creating main application file...
(
echo #!/usr/bin/env python3
echo """
echo 25 RAG Implementation Project - Main Entry Point
echo """
echo import click
echo import yaml
echo from pathlib import Path
echo.
echo @click.group^(^)
echo @click.option^('--config', default='common/config/default_config.yaml', help='Configuration file path'^)
echo @click.pass_context
echo def cli^(ctx, config^):
echo     """25 RAG Implementation Project CLI"""
echo     ctx.ensure_object^(dict^)
echo     
echo     # Load configuration
echo     config_path = Path^(config^)
echo     if not config_path.exists^(^):
echo         click.echo^(f"❌ Configuration file not found: {config}"^)
echo         raise click.Abort^(^)
echo     
echo     with open^(config_path, 'r'^) as f:
echo         ctx.obj['config'^] = yaml.safe_load^(f^)
echo.
echo @cli.command^(^)
echo @click.argument^('rag_type', default='standard'^)
echo @click.option^('--query', prompt='Enter your query', help='Query to process'^)
echo @click.pass_context
echo def query^(ctx, rag_type, query^):
echo     """Process a query using specified RAG type"""
echo     click.echo^(f"🔍 Processing query with {rag_type} RAG..."^)
echo     click.echo^(f"Query: {query}"^)
echo     click.echo^("🚧 Implementation in progress..."^)
echo.
echo @cli.command^(^)
echo @click.pass_context
echo def test^(ctx^):
echo     """Run comprehensive tests on all RAG implementations"""
echo     click.echo^("🧪 Running comprehensive RAG tests..."^)
echo     click.echo^("🚧 Test implementation in progress..."^)
echo.
echo if __name__ == '__main__':
echo     cli^(^)
) > main.py

REM Create __init__.py files
echo. > common\__init__.py
echo. > common\models\__init__.py
echo. > common\utils\__init__.py
echo. > common\config\__init__.py
echo. > rag_types\__init__.py
echo. > tests\__init__.py

REM Create README
echo 📝 Creating README file...
(
echo # 🚀 25 RAG Implementation Project
echo.
echo A comprehensive implementation of 25 different Retrieval-Augmented Generation ^(RAG^) systems using Local Ollama LLM.
echo.
echo ## Quick Start ^(Windows^)
echo.
echo 1. **Setup Environment**:
echo    ```cmd
echo    cd 25_rag_implementations
echo    venv\Scripts\activate
echo    ```
echo.
echo 2. **Install Ollama** ^(if not already installed^):
echo    - Visit: https://ollama.ai/download
echo    - Download Windows installer
echo.
echo 3. **Test Installation**:
echo    ```cmd
echo    python main.py test
echo    ```
echo.
echo 4. **Run a Query**:
echo    ```cmd
echo    python main.py query standard --query "What is machine learning?"
echo    ```
echo.
echo ## Project Structure
echo.
echo ```
echo 25_rag_implementations/
echo ├── common/                 # Shared components
echo │   ├── models/            # Base models and interfaces
echo │   ├── utils/             # Utility functions
echo │   └── config/            # Configuration management
echo ├── rag_types/             # Individual RAG implementations
echo │   ├── 01_standard/       # Standard RAG
echo │   ├── 02_conversational/ # Conversational RAG
echo │   └── ...               # 23 more RAG types
echo ├── data/                  # Test data and documents
echo ├── tests/                 # Test suites
echo ├── demos/                 # Demo applications
echo └── logs/                  # Performance logs
echo ```
echo.
echo ## Available Commands
echo.
echo - `python main.py query ^<rag_type^> --query "^<your_query^>"` - Process a query
echo - `python main.py test` - Run comprehensive tests
echo.
echo ## Development Status
echo.
echo - ✅ Project structure created
echo - 🚧 RAG implementations in progress
echo - 🚧 Testing framework being built
echo.
echo ## Next Steps
echo.
echo Follow the Technical Implementation Plan to build each RAG type systematically.
) > README.md

echo ✅ Project setup complete!
echo.
echo 📋 Next Steps:
echo 1. cd 25_rag_implementations
echo 2. venv\Scripts\activate
echo 3. python main.py test
echo 4. Start implementing RAG types following the Technical Implementation Plan
echo.
echo 🚀 You're ready to build 25 RAG systems!
echo.
pause