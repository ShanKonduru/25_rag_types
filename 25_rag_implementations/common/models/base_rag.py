"""
Base RAG Interface and Core Classes
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time

class RAGType(Enum):
    STANDARD = "standard"
    CONVERSATIONAL = "conversational"  
    GRAPH = "graph"
    FUSION = "fusion"
    ADAPTIVE = "adaptive"
    AUTO = "auto"
    CORAG = "corag"
    CORRECTIVE = "corrective"
    AGENTIC = "agentic"
    REACT = "react"
    SELF = "self"
    REALM = "realm"
    RETRO = "retro"
    RAPTOR = "raptor"
    REVEAL = "reveal"
    MEMO = "memo"
    ATLAS = "atlas"
    ITERATIVE = "iterative"
    CONTREGEN = "contregen"
    CRAT = "crat"
    REPLUG = "replug"
    REFEED = "refeed"
    EACO = "eaco"
    RULE = "rule"
    SPECULATIVE = "speculative"

@dataclass
class RAGQuery:
    """Query object for RAG systems"""
    text: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 5
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass 
class RAGResponse:
    """Response object from RAG systems"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    retrieval_time: float
    generation_time: float
    metadata: Dict[str, Any]
    query_id: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class BaseRAG(ABC):
    """Abstract base class for all RAG implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_type = self._get_rag_type()
        self.name = f"{self.rag_type.value}_rag"
        self.initialized = False
        
        # Initialize components
        self._initialize()
    
    def _initialize(self):
        """Initialize all RAG components"""
        try:
            self.ollama_client = self._initialize_ollama()
            self.vector_store = self._initialize_vector_store()
            self.embeddings = self._initialize_embeddings()
            self.initialized = True
            print(f"✅ {self.name} initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize {self.name}: {e}")
            raise
    
    @abstractmethod
    def _get_rag_type(self) -> RAGType:
        """Return the RAG type for this implementation"""
        pass
    
    @abstractmethod
    def process_query(self, query: RAGQuery) -> RAGResponse:
        """Process a query and return response"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the RAG system"""
        pass
    
    def _initialize_ollama(self):
        """Initialize Ollama client - implemented by subclasses"""
        from common.ollama_service import OllamaService
        return OllamaService(
            model_name=self.config.get('ollama_model', 'llama2'),
            host=self.config.get('ollama_host', 'localhost:11434')
        )
    
    def _initialize_vector_store(self):
        """Initialize vector database - implemented by subclasses"""
        import chromadb
        client = chromadb.PersistentClient(
            path=self.config.get('chroma_path', './vector_db')
        )
        collection_name = f"{self.name}_{self.config.get('collection_suffix', 'default')}"
        return client.get_or_create_collection(name=collection_name)
    
    def _initialize_embeddings(self):
        """Initialize embedding model - implemented by subclasses"""
        from sentence_transformers import SentenceTransformer
        model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        return SentenceTransformer(model_name)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this RAG system"""
        return {
            'name': self.name,
            'type': self.rag_type.value,
            'initialized': self.initialized,
            'config': self.config,
            'supported_operations': ['query', 'add_documents']
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': time.time()
        }
        
        try:
            # Check Ollama
            if hasattr(self, 'ollama_client'):
                test_response = self.ollama_client.generate("Test", max_tokens=5)
                health['components']['ollama'] = 'healthy'
            else:
                health['components']['ollama'] = 'not_initialized'
        except Exception as e:
            health['components']['ollama'] = f'error: {str(e)}'
            health['status'] = 'degraded'
        
        try:
            # Check vector store
            if hasattr(self, 'vector_store'):
                self.vector_store.count()
                health['components']['vector_store'] = 'healthy'
            else:
                health['components']['vector_store'] = 'not_initialized'
        except Exception as e:
            health['components']['vector_store'] = f'error: {str(e)}'
            health['status'] = 'degraded'
        
        try:
            # Check embeddings
            if hasattr(self, 'embeddings'):
                test_embedding = self.embeddings.encode(["test"])
                health['components']['embeddings'] = 'healthy'
            else:
                health['components']['embeddings'] = 'not_initialized'
        except Exception as e:
            health['components']['embeddings'] = f'error: {str(e)}'
            health['status'] = 'degraded'
        
        return health