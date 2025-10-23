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

@dataclass
class RetrievalConfig:
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_threshold: float = 0.7

@dataclass
class GenerationConfig:
    max_context_length: int = 4000
    response_max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class ConfigManager:
    """Centralized configuration management for RAG systems"""
    
    def __init__(self, config_path: str = "common/config/default_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            print(f"Warning: Configuration file not found: {self.config_path}")
            print("Using default configuration")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                print(f"✅ Configuration loaded from {self.config_path}")
                return config
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is not available"""
        return {
            'ollama': {
                'host': 'localhost:11434',
                'default_model': 'llama2',
                'timeout': 60,
                'temperature': 0.7,
                'max_tokens': 512
            },
            'embeddings': {
                'model': 'all-MiniLM-L6-v2',
                'device': 'cpu',
                'batch_size': 32
            },
            'vector_store': {
                'type': 'chromadb',
                'persist_directory': './vector_db',
                'collection_prefix': 'rag_'
            },
            'retrieval': {
                'top_k': 5,
                'chunk_size': 512,
                'chunk_overlap': 50,
                'similarity_threshold': 0.7
            },
            'generation': {
                'max_context_length': 4000,
                'response_max_tokens': 512,
                'temperature': 0.7,
                'top_p': 0.9
            },
            'monitoring': {
                'enable_logging': True,
                'log_file': './logs/performance.jsonl',
                'log_level': 'INFO'
            }
        }
    
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
    
    def get_retrieval_config(self) -> RetrievalConfig:
        """Get retrieval configuration"""
        retrieval_config = self.config.get('retrieval', {})
        return RetrievalConfig(**retrieval_config)
    
    def get_generation_config(self) -> GenerationConfig:
        """Get generation configuration"""
        generation_config = self.config.get('generation', {})
        return GenerationConfig(**generation_config)
    
    def get_rag_config(self, rag_type: str) -> Dict[str, Any]:
        """Get configuration for a specific RAG type"""
        base_config = {
            'ollama_model': self.config['ollama']['default_model'],
            'ollama_host': self.config['ollama']['host'],
            'embedding_model': self.config['embeddings']['model'],
            'chroma_path': self.config['vector_store']['persist_directory'],
            'collection_suffix': rag_type,
            'chunk_size': self.config['retrieval']['chunk_size'],
            'chunk_overlap': self.config['retrieval']['chunk_overlap'],
            'top_k': self.config['retrieval']['top_k'],
            'max_tokens': self.config['generation']['response_max_tokens'],
            'temperature': self.config['generation']['temperature']
        }
        
        # Add RAG-specific configurations if they exist
        rag_specific = self.config.get(f'{rag_type}_config', {})
        base_config.update(rag_specific)
        
        return base_config
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            print(f"✅ Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        try:
            self._deep_update(self.config, updates)
            return self.save_config(self.config)
        except Exception as e:
            print(f"❌ Failed to update configuration: {e}")
            return False
    
    @staticmethod
    def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep update dictionary with nested values"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                ConfigManager._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value