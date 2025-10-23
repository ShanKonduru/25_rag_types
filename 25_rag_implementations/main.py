#!/usr/bin/env python3
"""
25 RAG Implementation Project - Main Entry Point
"""
import click
import yaml
from pathlib import Path

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

@cli.command()
@click.argument('rag_type', default='standard')
@click.option('--query', prompt='Enter your query', help='Query to process')
@click.pass_context
def query(ctx, rag_type, query):
    """Process a query using specified RAG type"""
    click.echo(f"🔍 Processing query with {rag_type} RAG...")
    click.echo(f"Query: {query}")
    click.echo("🚧 Implementation in progress...")

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
        click.echo(f"Available models: {[model['model'] for model in models.get('models', [])]}")
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
@click.option('--port', default=8501, help='Port for web interface')
@click.pass_context
def web(ctx, port):
    """Launch web interface"""
    click.echo(f"🌐 Launching web interface on port {port}...")
    click.echo("🚧 Web interface implementation in progress...")

if __name__ == '__main__':
    cli()
