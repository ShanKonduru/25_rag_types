"""
Ollama Integration Service for RAG Systems
"""
import ollama
from typing import Dict, List, Any, Generator, Optional
import json
import time

class OllamaService:
    """Service class for interacting with Ollama models"""
    
    def __init__(self, model_name: str = "llama2", host: str = "localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)
        self._ensure_model_available()
        self.generation_count = 0
        self.total_tokens = 0
    
    def _ensure_model_available(self):
        """Ensure the specified model is available locally"""
        try:
            models_response = self.client.list()
            # Handle different API response formats
            if hasattr(models_response, 'models'):
                models = models_response.models
                model_names = [model.model for model in models]
            elif isinstance(models_response, dict) and 'models' in models_response:
                models = models_response['models']
                model_names = [model['name'] for model in models]
            else:
                model_names = []
            
            if not any(name.startswith(self.model_name) for name in model_names):
                print(f"Model {self.model_name} not found. Available models:")
                for name in model_names:
                    print(f"  - {name}")
                raise Exception(f"Model {self.model_name} not available")
            else:
                print(f"✅ Ollama model {self.model_name} is available")
        except Exception as e:
            raise Exception(f"Failed to connect to Ollama: {e}")
    
    def is_available(self) -> bool:
        """Check if the Ollama service is available"""
        try:
            self.client.list()
            return True
        except Exception:
            return False
    
    def generate(self, prompt: str, context: str = "", model: str = None, max_tokens: int = None, temperature: float = None, **kwargs) -> str:
        """Generate response using Ollama"""
        start_time = time.time()
        
        # Prepare the full prompt
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt
        
        # Use provided model or default
        model_to_use = model or self.model_name
        
        # Prepare generation options
        options = {}
        if max_tokens:
            options['num_predict'] = max_tokens
        if temperature is not None:
            options['temperature'] = temperature
        
        try:
            response = self.client.generate(
                model=model_to_use,
                prompt=full_prompt,
                options=options,
                **kwargs
            )
            
            generation_time = time.time() - start_time
            tokens_generated = len(response['response'].split())
            
            # Update statistics
            self.generation_count += 1
            self.total_tokens += tokens_generated
            
            # Handle different response formats
            if hasattr(response, 'response'):
                response_text = response.response
            elif isinstance(response, dict) and 'response' in response:
                response_text = response['response']
            else:
                response_text = str(response)
            
            # Update statistics
            self.generation_count += 1
            self.total_tokens += len(response_text.split())
            
            return response_text
        except Exception as e:
            raise Exception(f"Ollama generation failed: {e}")
    
    def generate_stream(self, prompt: str, context: str = "", **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using Ollama"""
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt
        
        try:
            stream = self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']
        except Exception as e:
            raise Exception(f"Ollama streaming failed: {e}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Use Ollama chat format for conversation"""
        start_time = time.time()
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            
            generation_time = time.time() - start_time
            tokens_generated = len(response['message']['content'].split())
            
            # Update statistics
            self.generation_count += 1
            self.total_tokens += tokens_generated
            
            return {
                'text': response['message']['content'],
                'generation_time': generation_time,
                'model': self.model_name,
                'tokens_generated': tokens_generated,
                'role': response['message']['role'],
                'metadata': {
                    'eval_count': response.get('eval_count', 0),
                    'eval_duration': response.get('eval_duration', 0),
                    'load_duration': response.get('load_duration', 0),
                    'prompt_eval_count': response.get('prompt_eval_count', 0),
                    'prompt_eval_duration': response.get('prompt_eval_duration', 0),
                    'total_duration': response.get('total_duration', 0),
                }
            }
        except Exception as e:
            raise Exception(f"Ollama chat failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            models = self.client.list()['models']
            current_model = next(
                (model for model in models if model['name'].startswith(self.model_name)), 
                None
            )
            
            if current_model:
                return {
                    'name': current_model['name'],
                    'size': current_model.get('size', 'unknown'),
                    'digest': current_model.get('digest', 'unknown'),
                    'modified_at': current_model.get('modified_at', 'unknown'),
                    'format': current_model.get('format', 'unknown'),
                    'family': current_model.get('family', 'unknown'),
                    'parameter_size': current_model.get('parameter_size', 'unknown'),
                    'quantization_level': current_model.get('quantization_level', 'unknown')
                }
            else:
                return {'error': f'Model {self.model_name} not found'}
        except Exception as e:
            return {'error': f'Failed to get model info: {e}'}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for this service instance"""
        return {
            'model_name': self.model_name,
            'generation_count': self.generation_count,
            'total_tokens': self.total_tokens,
            'average_tokens_per_generation': (
                self.total_tokens / self.generation_count if self.generation_count > 0 else 0
            ),
            'host': self.host
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to Ollama and model availability"""
        try:
            # Test basic connection
            models = self.client.list()
            
            # Test generation
            test_response = self.generate("Hello", max_tokens=5)
            
            return {
                'status': 'success',
                'connection': 'ok',
                'model_available': True,
                'test_generation': 'ok',
                'available_models': [model['name'] for model in models['models']],
                'response_sample': test_response['text'][:50] + '...' if len(test_response['text']) > 50 else test_response['text']
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'connection': 'failed'
            }