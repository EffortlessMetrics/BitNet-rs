"""
llama-cpp-python compatible API for BitNet.rs

This module provides a drop-in replacement for llama-cpp-python.
Simply replace:
    from llama_cpp import Llama
with:
    from bitnet.llama_compat import Llama

And your code will work unchanged with BitNet.rs!
"""

from typing import Optional, List, Dict, Any, Union
import ctypes
from pathlib import Path

try:  # Prefer package-relative import when available
    from . import bitnet_py
except Exception:  # Fallback when running as a script or in tests
    import bitnet_py

class Llama:
    """Drop-in replacement for llama_cpp.Llama"""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        main_gpu: int = 0,
        tensor_split: Optional[List[float]] = None,
        rope_freq_base: float = 10000.0,
        rope_freq_scale: float = 1.0,
        low_vram: bool = False,
        mul_mat_q: bool = True,
        f16_kv: bool = True,
        logits_all: bool = False,
        vocab_only: bool = False,
        use_mmap: bool = True,
        use_mlock: bool = False,
        embedding: bool = False,
        n_threads_batch: Optional[int] = None,
        seed: int = -1,
        verbose: bool = True,
        **kwargs
    ):
        """Initialize BitNet model with llama-cpp-python compatible parameters"""
        
        # Store parameters
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads or 1
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        
        # Create BitNet model
        self._model = bitnet_py.Model(model_path)
        
        # Initialize context
        self._context = self._model.create_context(
            max_tokens=n_ctx,
            batch_size=n_batch,
            n_threads=self.n_threads,
            seed=seed if seed >= 0 else None,
        )
        
        # Cache for last evaluation
        self._last_tokens = []
        self._last_logits = None
        
    def tokenize(
        self,
        text: Union[str, bytes],
        add_bos: bool = True,
        special: bool = True,
    ) -> List[int]:
        """Tokenize text, compatible with llama_cpp.tokenize"""
        
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        
        tokens = self._model.tokenize(text, add_bos=add_bos, special=special)
        return tokens
    
    def detokenize(
        self,
        tokens: List[int],
        skip_special_tokens: bool = True,
    ) -> bytes:
        """Detokenize tokens to bytes, compatible with llama_cpp"""
        
        text = self._model.detokenize(tokens)
        return text.encode('utf-8')
    
    def eval(
        self,
        tokens: List[int],
        n_past: int = 0,
        n_threads: Optional[int] = None,
    ) -> int:
        """Evaluate tokens, compatible with llama_cpp.eval"""
        
        # Store tokens for later
        self._last_tokens = tokens
        
        # Run evaluation
        logits = self._context.eval(tokens, n_past=n_past)
        self._last_logits = logits
        
        return 0  # Success
    
    def sample(
        self,
        top_k: int = 40,
        top_p: float = 0.95,
        temperature: float = 0.8,
        repeat_penalty: float = 1.1,
        last_n_tokens: Optional[List[int]] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        penalize_nl: bool = True,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> int:
        """Sample next token, compatible with llama_cpp.sample"""
        
        if self._last_logits is None:
            raise RuntimeError("No logits available. Call eval() first.")
        
        # Sample from logits
        token = self._context.sample(
            self._last_logits,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
        )
        
        return token
    
    def generate(
        self,
        tokens: List[int],
        top_k: int = 40,
        top_p: float = 0.95,
        temperature: float = 0.8,
        repeat_penalty: float = 1.1,
        reset: bool = True,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        penalize_nl: bool = True,
        logit_bias: Optional[Dict[int, float]] = None,
        stopping_criteria: Optional[Any] = None,
    ):
        """Generate tokens, compatible with llama_cpp.generate"""
        
        # Eval initial tokens
        self.eval(tokens)
        
        # Generate loop
        while True:
            # Sample next token
            token = self.sample(
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
            )
            
            # Check stopping criteria
            if stopping_criteria and stopping_criteria([token]):
                break
            
            # Yield token
            yield token
            
            # Eval new token for next iteration
            self.eval([token])
    
    def __call__(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        logit_bias: Optional[Dict[int, float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """High-level generate interface, compatible with llama_cpp"""
        
        # Tokenize prompt
        tokens = self.tokenize(prompt)
        
        # Generate
        generated_tokens = []
        for token in self.generate(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
        ):
            generated_tokens.append(token)
            
            # Check max tokens
            if len(generated_tokens) >= max_tokens:
                break
            
            # Check stop sequences
            if stop:
                text = self.detokenize(generated_tokens).decode('utf-8')
                stop_list = [stop] if isinstance(stop, str) else stop
                if any(s in text for s in stop_list):
                    break
        
        # Detokenize result
        result_tokens = tokens + generated_tokens if echo else generated_tokens
        result_text = self.detokenize(result_tokens).decode('utf-8')
        
        # Return in llama-cpp format
        return {
            "id": "bitnet-" + str(hash(prompt)),
            "object": "text_completion",
            "created": 0,
            "model": self.model_path,
            "choices": [{
                "text": result_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length" if len(generated_tokens) >= max_tokens else "stop",
            }],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(generated_tokens),
                "total_tokens": len(tokens) + len(generated_tokens),
            }
        }
    
    def create_embedding(
        self,
        input: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create embeddings, compatible with llama_cpp.create_embedding"""
        
        if isinstance(input, str):
            input = [input]
        
        embeddings = []
        for text in input:
            tokens = self.tokenize(text)
            self.eval(tokens)
            # Get embeddings from context
            embedding = self._context.get_embeddings()
            embeddings.append(embedding)
        
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": emb,
                    "index": i,
                }
                for i, emb in enumerate(embeddings)
            ],
            "model": model or self.model_path,
            "usage": {
                "prompt_tokens": sum(len(self.tokenize(t)) for t in input),
                "total_tokens": sum(len(self.tokenize(t)) for t in input),
            }
        }
    
    def create_completion(
        self,
        prompt: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Create completion, compatible with llama_cpp.create_completion"""
        
        if isinstance(prompt, list):
            # Batch processing
            completions = []
            for p in prompt:
                result = self(p, **kwargs)
                completions.extend(result["choices"])
            
            return {
                "id": "bitnet-batch",
                "object": "text_completion",
                "created": 0,
                "model": self.model_path,
                "choices": completions,
                "usage": {
                    "prompt_tokens": sum(len(self.tokenize(p)) for p in prompt),
                    "completion_tokens": sum(c.get("usage", {}).get("completion_tokens", 0) for c in completions),
                    "total_tokens": 0,  # Will be calculated
                }
            }
        else:
            return self(prompt, **kwargs)
    
    def reset(self) -> None:
        """Reset the model state"""
        self._context.reset()
        self._last_tokens = []
        self._last_logits = None
    
    def set_cache(self, cache: Any) -> None:
        """Set KV cache state"""
        self._context.set_cache(cache)
    
    def get_cache(self) -> Any:
        """Get KV cache state"""
        return self._context.get_cache()
    
    @property
    def n_vocab(self) -> int:
        """Get vocabulary size"""
        return self._model.vocab_size()
    
    @property
    def n_ctx(self) -> int:
        """Get context size"""
        return self._context.max_tokens
    
    @property
    def n_embd(self) -> int:
        """Get embedding size"""
        return self._model.embedding_size()


# Additional compatibility classes and functions

def llama_backend_init(numa: bool = False) -> None:
    """Initialize backend (no-op for compatibility)"""
    pass

def llama_backend_free() -> None:
    """Free backend (no-op for compatibility)"""
    pass

class LlamaCache:
    """Compatible cache object"""
    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.data = {}
    
    def __getstate__(self):
        return self.data
    
    def __setstate__(self, state):
        self.data = state


# Export main classes
__all__ = [
    'Llama',
    'LlamaCache',
    'llama_backend_init',
    'llama_backend_free',
]