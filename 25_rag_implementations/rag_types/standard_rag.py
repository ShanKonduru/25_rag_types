"""Standard RAG Implementation - The Foundation RAG System"""
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import spacy

from common.models.base_rag import BaseRAG, RAGQuery, RAGResponse, RAGType
from common.ollama_service import OllamaService
from common.config.config_manager import ConfigManager

class StandardRAG(BaseRAG):
    """
    Standard RAG Implementation
    
    This is the foundational RAG system that implements:
    1. Document chunking and embedding
    2. Vector similarity search
    3. Context-aware generation with Ollama
    
    Architecture:
    Document → Chunks → Embeddings → Vector Store → Retrieval → Generation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path) if config_path else ConfigManager()
        self.config = self.config_manager.get_rag_config("standard_rag")
        
        # Initialize components
        self.ollama_service = None
        self.embedding_model = None
        self.vector_client = None
        self.collection = None
        self.nlp = None
        
        # Performance tracking
        self.stats = {
            'queries_processed': 0,
            'documents_indexed': 0,
            'avg_response_time': 0.0,
            'avg_retrieval_time': 0.0,
            'avg_generation_time': 0.0
        }
        
        print("🚀 Standard RAG initialized with configuration")
    
    def initialize(self) -> bool:
        """Initialize all components for Standard RAG"""
        try:
            print("🔧 Initializing Standard RAG components...")
            
            # Initialize Ollama service
            self.ollama_service = OllamaService(
                model_name=self.config['ollama_model'],
                host=self.config['ollama_host']
            )
            
            # Test Ollama connection
            if not self.ollama_service.is_available():
                print("❌ Ollama service not available")
                return False
            
            # Initialize embedding model (with fallback to Ollama embeddings)
            embedding_config = self.config_manager.get_embedding_config()
            try:
                self.embedding_model = SentenceTransformer(
                    embedding_config.model,
                    device=embedding_config.device
                )
                self.use_ollama_embeddings = False
                print(f"✅ SentenceTransformer model loaded: {embedding_config.model}")
            except Exception as e:
                print(f"⚠️ Failed to load SentenceTransformer: {e}")
                print("🔄 Using Ollama embeddings as fallback...")
                self.embedding_model = None
                self.use_ollama_embeddings = True
                # Check if embedding model is available in Ollama
                try:
                    embed_response = self.ollama_service.client.embeddings(
                        model="nomic-embed-text",
                        prompt="test"
                    )
                    print("✅ Ollama embeddings available: nomic-embed-text")
                except Exception as embed_e:
                    print(f"❌ Ollama embeddings not available: {embed_e}")
                    return False
            
            # Initialize ChromaDB
            vector_config = self.config_manager.get_vector_store_config()
            persist_dir = Path(vector_config.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.vector_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            collection_name = f"{vector_config.collection_prefix}standard"
            try:
                self.collection = self.vector_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"✅ Created new collection: {collection_name}")
            except Exception:
                self.collection = self.vector_client.get_collection(collection_name)
                print(f"✅ Using existing collection: {collection_name}")
            
            # Initialize spaCy for text processing
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy English model loaded")
            except OSError:
                print("⚠️ spaCy English model not found, using basic tokenization")
                self.nlp = None
            
            print("🎉 Standard RAG initialization complete!")
            return True
            
        except Exception as e:
            print(f"❌ Standard RAG initialization failed: {e}")
            return False
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Add documents to the vector store"""
        try:
            if not documents:
                print("⚠️ No documents provided")
                return False
            
            print(f"📄 Processing {len(documents)} documents...")
            
            # Process documents into chunks
            all_chunks = []
            all_metadata = []
            all_ids = []
            
            for i, doc in enumerate(documents):
                doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                chunks = self._chunk_document(doc)
                
                for j, chunk in enumerate(chunks):
                    import uuid
                    chunk_id = f"doc_{i}_chunk_{j}_{str(uuid.uuid4())[:8]}"
                    chunk_metadata = {
                        **doc_metadata,
                        'document_index': i,
                        'chunk_index': j,
                        'chunk_id': chunk_id,
                        'source': doc_metadata.get('source', f'document_{i}')
                    }
                    
                    all_chunks.append(chunk)
                    all_metadata.append(chunk_metadata)
                    all_ids.append(chunk_id)
            
            # Generate embeddings
            print(f"🔮 Generating embeddings for {len(all_chunks)} chunks...")
            if self.use_ollama_embeddings:
                embeddings = self._generate_ollama_embeddings(all_chunks)
            else:
                embeddings = self.embedding_model.encode(
                    all_chunks,
                    batch_size=self.config_manager.get_embedding_config().batch_size,
                    show_progress_bar=True
                )
            
            # Add to vector store
            # Handle different embedding formats
            if hasattr(embeddings, 'tolist'):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = embeddings
            
            self.collection.add(
                documents=all_chunks,
                embeddings=embeddings_list,
                metadatas=all_metadata,
                ids=all_ids
            )
            
            self.stats['documents_indexed'] += len(documents)
            print(f"✅ Added {len(documents)} documents ({len(all_chunks)} chunks) to vector store")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
            return False
    
    def query(self, query: RAGQuery) -> RAGResponse:
        """Process a query using Standard RAG pipeline"""
        import time
        start_time = time.time()
        
        try:
            print(f"🔍 Processing query: {query.text[:100]}...")
            
            # Step 1: Retrieve relevant chunks
            retrieval_start = time.time()
            relevant_chunks = self._retrieve_relevant_chunks(query.text)
            retrieval_time = time.time() - retrieval_start
            
            if not relevant_chunks:
                return RAGResponse(
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[],
                    confidence=0.0,
                    retrieval_time=retrieval_time,
                    generation_time=0.0,
                    metadata={
                        'total_time': time.time() - start_time,
                        'chunks_retrieved': 0
                    }
                )
            
            # Step 2: Prepare context
            context = self._prepare_context(relevant_chunks)
            
            # Step 3: Generate response
            generation_start = time.time()
            prompt = self._build_prompt(query.text, context)
            
            llm_response = self.ollama_service.generate(
                prompt=prompt,
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature']
            )
            generation_time = time.time() - generation_start
            
            # Extract sources
            sources = [
                {'source': chunk['metadata'].get('source', 'Unknown'), 'chunk_id': chunk['metadata'].get('chunk_id', '')}
                for chunk in relevant_chunks
            ]
            
            # Calculate confidence based on retrieval scores
            avg_score = sum(chunk['distance'] for chunk in relevant_chunks) / len(relevant_chunks)
            confidence = max(0.0, 1.0 - avg_score)  # Convert distance to confidence
            
            total_time = time.time() - start_time
            
            # Update statistics
            self.stats['queries_processed'] += 1
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['queries_processed'] - 1) + total_time) /
                self.stats['queries_processed']
            )
            self.stats['avg_retrieval_time'] = (
                (self.stats['avg_retrieval_time'] * (self.stats['queries_processed'] - 1) + retrieval_time) /
                self.stats['queries_processed']
            )
            self.stats['avg_generation_time'] = (
                (self.stats['avg_generation_time'] * (self.stats['queries_processed'] - 1) + generation_time) /
                self.stats['queries_processed']
            )
            
            return RAGResponse(
                answer=llm_response,
                sources=sources,
                confidence=confidence,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                metadata={
                    'total_time': total_time,
                    'chunks_retrieved': len(relevant_chunks)
                }
            )
            
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
            return RAGResponse(
                answer=f"An error occurred while processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                retrieval_time=0.0,
                generation_time=0.0,
                metadata={'error': str(e)}
            )
    
    def _chunk_document(self, document: str) -> List[str]:
        """Split document into chunks"""
        retrieval_config = self.config_manager.get_retrieval_config()
        chunk_size = retrieval_config.chunk_size
        chunk_overlap = retrieval_config.chunk_overlap
        
        # Simple sentence-aware chunking
        if self.nlp:
            doc = self.nlp(document)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback to basic sentence splitting
            import re
            sentences = re.split(r'[.!?]+', document)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Handle overlap
                if chunk_overlap > 0 and chunks:
                    overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                    current_chunk = overlap_text + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [document]  # Fallback to original document
    
    def _retrieve_relevant_chunks(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using vector similarity"""
        try:
            retrieval_config = self.config_manager.get_retrieval_config()
            
            # Generate query embedding
            if self.use_ollama_embeddings:
                query_embedding = self._generate_ollama_embeddings([question])
            else:
                query_embedding = self.embedding_model.encode([question])
            
            # Search vector store
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=retrieval_config.top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            relevant_chunks = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    chunk_data = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    }
                    
                    # Filter by similarity threshold
                    if chunk_data['distance'] <= 0.7:  # Cosine distance threshold
                        relevant_chunks.append(chunk_data)
            
            print(f"📚 Retrieved {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            return []
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved chunks"""
        generation_config = self.config_manager.get_generation_config()
        max_length = generation_config.max_context_length
        
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            content = chunk['content']
            source = chunk['metadata'].get('source', 'Unknown')
            
            # Format chunk with source information
            formatted_chunk = f"[Source: {source}]\n{content}\n"
            
            if current_length + len(formatted_chunk) <= max_length:
                context_parts.append(formatted_chunk)
                current_length += len(formatted_chunk)
            else:
                break
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for the LLM"""
        system_prompt = self.config.get('system_prompt', "You are a helpful AI assistant that provides accurate and relevant information based on the provided context.")
        
        prompt = f"""{system_prompt}

Context Information:
{context}

Question: {question}

Instructions: Please provide a comprehensive answer based on the context information above. If the context doesn't contain enough information to fully answer the question, please say so and provide what information you can based on the available context.

Answer:"""
        
        return prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        collection_info = {
            'total_chunks': self.collection.count() if self.collection else 0,
            'collection_name': self.collection.name if self.collection else 'Not initialized'
        }
        
        return {
            **self.stats,
            **collection_info,
            'config': self.config
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all components"""
        health = {}
        
        # Check Ollama
        health['ollama'] = self.ollama_service.is_available() if self.ollama_service else False
        
        # Check embedding model
        health['embeddings'] = self.embedding_model is not None
        
        # Check vector store
        try:
            health['vector_store'] = self.collection is not None and self.collection.count() >= 0
        except:
            health['vector_store'] = False
        
        # Check NLP
        health['nlp'] = self.nlp is not None
        
        health['overall'] = all(health.values())
        
        return health
    
    def get_rag_type(self) -> RAGType:
        """Return the RAG type"""
        return RAGType.STANDARD_RAG
    
    def _get_rag_type(self) -> RAGType:
        """Abstract method implementation"""
        return RAGType.STANDARD_RAG
    
    def process_query(self, query: RAGQuery) -> RAGResponse:
        """Abstract method implementation - delegates to query method"""
        return self.query(query)
    
    def _generate_ollama_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama"""
        embeddings = []
        for text in texts:
            try:
                response = self.ollama_service.client.embeddings(
                    model="nomic-embed-text",
                    prompt=text
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                print(f"❌ Failed to generate embedding: {e}")
                # Fallback to zero embedding
                embeddings.append([0.0] * 768)  # Standard embedding dimension
        return embeddings