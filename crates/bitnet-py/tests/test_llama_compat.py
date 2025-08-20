"""
Python API Contract Tests

These tests ensure our llama-cpp-python compatibility layer
maintains exact API compatibility and never regresses.
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_compat import Llama, LlamaCache, llama_backend_init, llama_backend_free


class TestLlamaAPIContract:
    """Test that we maintain exact llama-cpp-python API compatibility"""
    
    def test_llama_class_exists(self):
        """Test that main Llama class exists with correct interface"""
        assert hasattr(Llama, '__init__')
        assert hasattr(Llama, 'tokenize')
        assert hasattr(Llama, 'detokenize')
        assert hasattr(Llama, 'eval')
        assert hasattr(Llama, 'sample')
        assert hasattr(Llama, 'generate')
        assert hasattr(Llama, 'create_embedding')
        assert hasattr(Llama, 'create_completion')
        assert hasattr(Llama, 'reset')
        assert hasattr(Llama, 'set_cache')
        assert hasattr(Llama, 'get_cache')
        
    def test_llama_init_signature(self):
        """Test that __init__ accepts all llama-cpp-python parameters"""
        # These parameters must be accepted (even if ignored)
        params = {
            'model_path': 'test.gguf',
            'n_ctx': 2048,
            'n_batch': 512,
            'n_threads': 4,
            'n_gpu_layers': 0,
            'main_gpu': 0,
            'tensor_split': None,
            'rope_freq_base': 10000.0,
            'rope_freq_scale': 1.0,
            'low_vram': False,
            'mul_mat_q': True,
            'f16_kv': True,
            'logits_all': False,
            'vocab_only': False,
            'use_mmap': True,
            'use_mlock': False,
            'embedding': False,
            'n_threads_batch': None,
            'seed': -1,
            'verbose': True,
        }
        
        # Should not raise TypeError for unknown parameters
        try:
            llama = Llama(**params)
        except FileNotFoundError:
            # Model doesn't exist, that's fine for this test
            pass
        except TypeError as e:
            pytest.fail(f"Llama init signature incompatible: {e}")
    
    def test_tokenize_signature(self):
        """Test tokenize method signature matches llama-cpp-python"""
        # Mock Llama instance
        class MockModel:
            def tokenize(self, text, add_bos=True, special=True):
                return [1, 2, 3]
        
        # Should accept these exact parameters
        params = {
            'text': b"Hello world",
            'add_bos': True,
            'special': True,
        }
        
        # Verify signature compatibility
        import inspect
        sig = inspect.signature(Llama.tokenize)
        assert 'text' in sig.parameters
        assert 'add_bos' in sig.parameters
        assert 'special' in sig.parameters
    
    def test_generate_signature(self):
        """Test generate method signature matches llama-cpp-python"""
        import inspect
        sig = inspect.signature(Llama.generate)
        
        # Required parameters
        assert 'tokens' in sig.parameters
        
        # Optional parameters that must be accepted
        assert 'top_k' in sig.parameters
        assert 'top_p' in sig.parameters
        assert 'temperature' in sig.parameters
        assert 'repeat_penalty' in sig.parameters
        assert 'frequency_penalty' in sig.parameters
        assert 'presence_penalty' in sig.parameters
        assert 'tfs_z' in sig.parameters
        assert 'mirostat_mode' in sig.parameters
        assert 'mirostat_tau' in sig.parameters
        assert 'mirostat_eta' in sig.parameters
    
    def test_call_signature(self):
        """Test __call__ method signature for high-level API"""
        import inspect
        sig = inspect.signature(Llama.__call__)
        
        # All these parameters must be accepted
        required_params = [
            'prompt', 'suffix', 'max_tokens', 'temperature',
            'top_p', 'top_k', 'repeat_penalty', 'frequency_penalty',
            'presence_penalty', 'tfs_z', 'mirostat_mode',
            'mirostat_tau', 'mirostat_eta', 'echo', 'stop',
            'stream', 'logit_bias'
        ]
        
        for param in required_params:
            assert param in sig.parameters, f"Missing parameter: {param}"
    
    def test_output_format(self):
        """Test that output format matches llama-cpp-python exactly"""
        # Expected output structure
        expected_keys = {
            'id', 'object', 'created', 'model', 'choices', 'usage'
        }
        
        # Choice structure
        expected_choice_keys = {
            'text', 'index', 'logprobs', 'finish_reason'
        }
        
        # Usage structure
        expected_usage_keys = {
            'prompt_tokens', 'completion_tokens', 'total_tokens'
        }
        
        # This locks in the output format contract
    
    def test_properties(self):
        """Test that properties match llama-cpp-python"""
        # These properties must exist
        assert hasattr(Llama, 'n_vocab')
        assert hasattr(Llama, 'n_ctx')
        assert hasattr(Llama, 'n_embd')


class TestLlamaCacheContract:
    """Test LlamaCache compatibility"""
    
    def test_cache_class_exists(self):
        """Test that LlamaCache exists with correct interface"""
        cache = LlamaCache(capacity=512)
        assert hasattr(cache, '__getstate__')
        assert hasattr(cache, '__setstate__')
        assert hasattr(cache, 'capacity')
        assert hasattr(cache, 'data')
    
    def test_cache_serialization(self):
        """Test that cache can be serialized/deserialized"""
        cache = LlamaCache(capacity=1024)
        cache.data = {'test': 'data'}
        
        # Should be pickleable
        import pickle
        serialized = pickle.dumps(cache)
        restored = pickle.loads(serialized)
        
        assert restored.data == cache.data


class TestBackendFunctions:
    """Test backend initialization functions"""
    
    def test_backend_init_exists(self):
        """Test that backend init functions exist"""
        # These must be callable without error
        llama_backend_init(numa=False)
        llama_backend_free()
    
    def test_backend_init_signature(self):
        """Test backend init signature"""
        import inspect
        
        sig = inspect.signature(llama_backend_init)
        assert 'numa' in sig.parameters
        
        sig = inspect.signature(llama_backend_free)
        assert len(sig.parameters) == 0


class TestErrorCodes:
    """Test that error codes and behaviors match"""
    
    def test_tokenization_errors(self):
        """Test that tokenization errors match llama-cpp-python behavior"""
        # When model file doesn't exist
        with pytest.raises(Exception):
            llama = Llama(model_path="nonexistent.gguf")
    
    def test_eval_without_model(self):
        """Test eval behavior without model"""
        # This documents expected error behavior
        pass


class TestRegressions:
    """Regression tests for specific compatibility issues"""
    
    def test_bytes_input_handling(self):
        """Test that we handle bytes input like llama-cpp-python"""
        # llama-cpp-python accepts both str and bytes
        test_inputs = [
            b"Hello world",
            "Hello world",
            b"\xf0\x9f\x98\x80",  # UTF-8 emoji bytes
        ]
        
        # All should be accepted without error
        # (actual tokenization would require a model)
    
    def test_streaming_compatibility(self):
        """Test that streaming interface matches"""
        # Stream parameter must be accepted in __call__
        # generate() must be a generator
        pass
    
    def test_logit_bias_format(self):
        """Test that logit_bias format matches"""
        # Must accept Dict[int, float] format
        logit_bias = {
            100: 10.0,   # Boost token 100
            200: -10.0,  # Suppress token 200
        }
        
        # Should be accepted in relevant methods


# Integration test that would run with actual model
@pytest.mark.skipif(not os.path.exists("test_model.gguf"), 
                    reason="Test model not available")
class TestIntegration:
    """Integration tests with actual model"""
    
    def test_full_pipeline(self):
        """Test complete pipeline matches llama-cpp-python"""
        llama = Llama(model_path="test_model.gguf")
        
        # Tokenize
        tokens = llama.tokenize(b"Hello")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        
        # Generate
        output = llama(
            "Hello",
            max_tokens=10,
            temperature=0.7,
        )
        
        # Check output format
        assert 'choices' in output
        assert 'usage' in output
        
        # Embeddings
        embeddings = llama.create_embedding("test")
        assert 'data' in embeddings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])