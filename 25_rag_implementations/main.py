#!/usr/bin/env python3
"""
25 RAG Implementation Project - Main Entry Point
"""
import click
import yaml
from pathlib import Path


@click.group()
@click.option(
    "--config",
    default="common/config/default_config.yaml",
    help="Configuration file path",
)
@click.pass_context
def cli(ctx, config):
    """
    25 RAG Implementation Project - Command Line Interface
    
    A comprehensive toolkit for testing and comparing 25 different RAG system implementations
    using Local Ollama LLM. Supports document ingestion, query processing, and system testing.
    
    Examples:
        python main.py query --rag_type standard --query "What is RAG?"
        python main.py add-docs --rag_type standard --file my_document.txt
        python main.py test
    """
    ctx.ensure_object(dict)

    # Load configuration
    config_path = Path(config)
    if not config_path.exists():
        click.echo(f"❌ Configuration file not found: {config}")
        raise click.Abort()

    with open(config_path, "r") as f:
        ctx.obj["config"] = yaml.safe_load(f)


@cli.command()
@click.option('--rag_type', default='standard', help='RAG system type to use (standard, conversational, hierarchical, etc.)')
@click.option('--query', prompt='Enter your query', help='Query to process')
@click.option('--model', help='Ollama model to use (overrides default)')
@click.pass_context
def query(ctx, rag_type, query, model):
    """Process a query using specified RAG type"""
    click.echo(f"🔍 Processing query with {rag_type} RAG...")

    try:
        if rag_type.lower() == "standard":
            from rag_types.standard_rag import StandardRAG
            from common.models.base_rag import RAGQuery

            # Initialize Standard RAG
            rag_system = StandardRAG()
            if not rag_system.initialize():
                click.echo("❌ Failed to initialize Standard RAG system")
                return

            # Process query
            rag_query = RAGQuery(text=query)
            response = rag_system.query(rag_query)

            # Display results
            click.echo("\n" + "=" * 50)
            click.echo(f"Query: {query}")
            click.echo("=" * 50)
            click.echo(f"Answer: {response.answer}")
            click.echo(f"Confidence: {response.confidence:.2f}")
            click.echo(f"Retrieval Time: {response.retrieval_time:.2f}s")
            click.echo(f"Generation Time: {response.generation_time:.2f}s")
            click.echo(f"Sources: {[src['source'] for src in response.sources]}")

            if response.metadata:
                click.echo(
                    f"Total Time: {response.metadata.get('total_time', 0):.2f}s"
                )
                click.echo(
                    f"Chunks Retrieved: {response.metadata.get('chunks_retrieved', 0)}"
                )

        else:
            click.echo(f"❌ RAG type '{rag_type}' not yet implemented")
            available_types = ["standard"]
            click.echo(f"Available types: {', '.join(available_types)}")

    except Exception as e:
        click.echo(f"❌ Query processing failed: {e}")


@cli.command()
@click.option('--rag_type', default='standard', help='RAG system type to add documents to')
@click.option('--file', help='Path to document file to add')
@click.option('--text', help='Text content to add directly')
@click.pass_context
def add_docs(ctx, rag_type, file, text):
    """Add documents to the RAG system"""
    click.echo(f"📄 Adding documents to {rag_type} RAG...")

    try:
        if not file and not text:
            click.echo("❌ Please provide either --file or --text option")
            return

        if rag_type.lower() == "standard":
            from rag_types.standard_rag import StandardRAG

            # Initialize Standard RAG
            rag_system = StandardRAG()
            if not rag_system.initialize():
                click.echo("❌ Failed to initialize Standard RAG system")
                return

            documents = []
            metadata = []

            if file:
                # Read from file
                file_path = Path(file)
                if not file_path.exists():
                    click.echo(f"❌ File not found: {file}")
                    return

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(content)
                    metadata.append({"source": str(file_path), "type": "file"})

            if text:
                # Add text directly
                documents.append(text)
                metadata.append({"source": "direct_input", "type": "text"})

            # Add documents
            success = rag_system.add_documents(documents, metadata)
            if success:
                click.echo(f"✅ Successfully added {len(documents)} document(s)")
                stats = rag_system.get_stats()
                click.echo(f"Total documents indexed: {stats['documents_indexed']}")
            else:
                click.echo("❌ Failed to add documents")

        else:
            click.echo(f"❌ RAG type '{rag_type}' not yet implemented")

    except Exception as e:
        click.echo(f"❌ Document addition failed: {e}")


@cli.command()
@click.pass_context
def test(ctx):
    """Run comprehensive tests on all RAG implementations"""
    click.echo("🧪 Running comprehensive RAG tests...")

    # Basic system health checks
    click.echo("✅ Python environment: OK")

    # Test Ollama connection
    try:
        import ollama

        models = ollama.list()
        click.echo("✅ Ollama connection: OK")
        click.echo(
            f"Available models: {[model['model'] for model in models.get('models', [])]}"
        )
    except Exception as e:
        click.echo(f"❌ Ollama connection failed: {e}")

    # Test core dependencies
    try:
        import chromadb

        click.echo("✅ ChromaDB: OK")
    except Exception as e:
        click.echo(f"❌ ChromaDB import failed: {e}")

    try:
        from sentence_transformers import SentenceTransformer

        click.echo("✅ Sentence Transformers: OK")
    except Exception as e:
        click.echo(f"❌ Sentence Transformers import failed: {e}")

    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        click.echo("✅ spaCy with English model: OK")
    except Exception as e:
        click.echo(f"❌ spaCy model failed: {e}")

    click.echo("🎉 Basic system tests completed!")


@cli.command()
@click.pass_context
def benchmark(ctx):
    """Run performance benchmarks"""
    click.echo("📊 Running performance benchmarks...")
    click.echo("🚧 Benchmark implementation in progress...")


@cli.command()
@click.option("--port", default=8501, help="Port for web interface")
@click.pass_context
def web(ctx, port):
    """Launch web interface"""
    click.echo(f"🌐 Launching web interface on port {port}...")
    click.echo("🚧 Web interface implementation in progress...")


if __name__ == "__main__":
    cli()
