"""
Test suite for BitNet model loading functionality.
"""
import pytest
import torch
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Import BitNet modules
try:
    import model as bitnet_model
    from gpu.model import ModelArgs, Transformer, make_cache
except ImportError as e:
    pytest.skip(f"BitNet modules not available: {e}", allow_module_level=True)

class TestModelConfiguration:
    """Test model configuration loading and validation."""
    
    def test_model_args_default_values(self):
        """Test that ModelArgs has correct default values."""
        args = ModelArgs()
        
        assert args.dim == 2560
        assert args.n_layers == 30
        assert args.n_heads == 20
        assert args.n_kv_heads == 5
        assert args.vocab_size == 128256
        assert args.ffn_dim == 6912
        assert args.norm_eps == 1e-5
        assert args.rope_theta == 500000.0
        assert args.use_kernel == False
    
    def test_model_args_custom_values(self):
        """Test ModelArgs with custom values."""
        custom_args = ModelArgs(
            dim=1024,
            n_layers=12,
            n_heads=16,
            n_kv_heads=4,
            vocab_size=50000,
            use_kernel=True
        )
        
        assert custom_args.dim == 1024
        assert custom_args.n_layers == 12
        assert custom_args.n_heads == 16
        assert custom_args.n_kv_heads == 4
        assert custom_args.vocab_size == 50000
        assert custom_args.use_kernel == True
    
    def test_model_args_validation(self):
        """Test model configuration validation."""
        # Test that head dimensions are compatible
        args = ModelArgs(dim=128, n_heads=16)  # 128/16 = 8 head_dim
        transformer = Transformer(args)
        
        # Should not raise an error
        assert transformer is not None

class TestModelInitialization:
    """Test model initialization and structure."""
    
    def test_transformer_initialization(self):
        """Test Transformer model initialization."""
        args = ModelArgs(dim=512, n_layers=6, n_heads=8, vocab_size=1000)
        model = Transformer(args)
        
        # Check model structure
        assert hasattr(model, 'tok_embeddings')
        assert hasattr(model, 'layers')
        assert hasattr(model, 'norm')
        assert hasattr(model, 'output')
        
        # Check layer count
        assert len(model.layers) == args.n_layers
        
        # Check embedding dimensions
        assert model.tok_embeddings.num_embeddings == args.vocab_size
        assert model.tok_embeddings.embedding_dim == args.dim
        
        # Check output layer
        assert model.output.in_features == args.dim
        assert model.output.out_features == args.vocab_size
    
    def test_transformer_block_structure(self):
        """Test TransformerBlock structure."""
        args = ModelArgs(dim=512, n_layers=2, n_heads=8)
        model = Transformer(args)
        
        block = model.layers[0]
        
        # Check block components
        assert hasattr(block, 'attention')
        assert hasattr(block, 'feed_forward')
        assert hasattr(block, 'attention_norm')
        assert hasattr(block, 'ffn_norm')
    
    def test_attention_layer_structure(self):
        """Test Attention layer structure."""
        args = ModelArgs(dim=512, n_heads=8, n_kv_heads=4)
        model = Transformer(args)
        
        attention = model.layers[0].attention
        
        # Check attention components
        assert hasattr(attention, 'wqkv')
        assert hasattr(attention, 'wo')
        assert hasattr(attention, 'attn_sub_norm')
        
        # Check dimensions
        expected_qkv_dim = (args.n_heads + 2 * args.n_kv_heads) * (args.dim // args.n_heads)
        assert attention.wqkv.in_features == args.dim
        assert attention.wqkv.out_features == expected_qkv_dim
    
    def test_feedforward_structure(self):
        """Test FeedForward layer structure."""
        args = ModelArgs(dim=512, ffn_dim=2048)
        model = Transformer(args)
        
        ff = model.layers[0].feed_forward
        
        # Check feedforward components
        assert hasattr(ff, 'w13')
        assert hasattr(ff, 'w2')
        assert hasattr(ff, 'ffn_sub_norm')
        
        # Check dimensions
        assert ff.w13.in_features == args.dim
        assert ff.w13.out_features == 2 * args.ffn_dim
        assert ff.w2.in_features == args.ffn_dim
        assert ff.w2.out_features == args.dim

class TestCacheManagement:
    """Test KV cache creation and management."""
    
    def test_cache_creation(self):
        """Test cache creation with correct dimensions."""
        args = ModelArgs(dim=512, n_layers=6, n_heads=8, n_kv_heads=4)
        cache_length = 1024
        
        cache = make_cache(args, cache_length)
        
        # Check cache structure
        assert len(cache) == args.n_layers
        assert all(len(layer_cache) == 2 for layer_cache in cache)  # k and v caches
        
        # Check cache dimensions
        head_dim = args.dim // args.n_heads
        expected_shape = (1, cache_length, args.n_kv_heads, args.n_heads // args.n_kv_heads, head_dim)
        
        for layer_cache in cache:
            k_cache, v_cache = layer_cache
            assert k_cache.shape == expected_shape
            assert v_cache.shape == expected_shape
    
    def test_cache_device_placement(self, device):
        """Test cache creation on specific device."""
        args = ModelArgs(dim=256, n_layers=2, n_heads=4)
        cache_length = 512
        
        cache = make_cache(args, cache_length, device=device)
        
        for layer_cache in cache:
            k_cache, v_cache = layer_cache
            assert k_cache.device == device
            assert v_cache.device == device
    
    def test_cache_dtype(self):
        """Test cache creation with specific dtype."""
        args = ModelArgs(dim=256, n_layers=2, n_heads=4)
        cache_length = 512
        dtype = torch.float16
        
        cache = make_cache(args, cache_length, dtype=dtype)
        
        for layer_cache in cache:
            k_cache, v_cache = layer_cache
            assert k_cache.dtype == dtype
            assert v_cache.dtype == dtype
    
    def test_cache_prefix(self):
        """Test cache prefix functionality."""
        from gpu.model import cache_prefix
        
        args = ModelArgs(dim=256, n_layers=2, n_heads=4)
        cache_length = 1024
        prefix_length = 512
        
        full_cache = make_cache(args, cache_length)
        prefix_cache = cache_prefix(full_cache, prefix_length)
        
        # Check that prefix cache has correct length
        for full_layer, prefix_layer in zip(full_cache, prefix_cache):
            full_k, full_v = full_layer
            prefix_k, prefix_v = prefix_layer
            
            assert prefix_k.shape[1] == prefix_length
            assert prefix_v.shape[1] == prefix_length
            assert full_k.shape[1] == cache_length
            assert full_v.shape[1] == cache_length

class TestModelForward:
    """Test model forward pass functionality."""
    
    def test_transformer_forward_shape(self, device):
        """Test that forward pass produces correct output shape."""
        args = ModelArgs(dim=256, n_layers=2, n_heads=4, vocab_size=1000)
        model = Transformer(args).to(device)
        
        batch_size = 2
        seq_len = 10
        cache_len = 64
        
        # Create inputs
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        cache = make_cache(args, cache_len, device=device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(token_values, token_lengths, start_pos, cache, cache_len)
        
        # Check output shape
        expected_shape = (batch_size * seq_len, args.vocab_size)
        assert logits.shape == expected_shape
        assert logits.dtype == torch.float32
    
    def test_transformer_forward_deterministic(self, device):
        """Test that forward pass is deterministic."""
        args = ModelArgs(dim=128, n_layers=1, n_heads=2, vocab_size=100)
        model = Transformer(args).to(device)
        
        # Set model to eval mode for deterministic behavior
        model.eval()
        
        batch_size = 1
        seq_len = 5
        cache_len = 32
        
        # Create inputs
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Run forward pass twice
        cache1 = make_cache(args, cache_len, device=device)
        cache2 = make_cache(args, cache_len, device=device)
        
        with torch.no_grad():
            logits1 = model(token_values, token_lengths, start_pos, cache1, cache_len)
            logits2 = model(token_values, token_lengths, start_pos, cache2, cache_len)
        
        # Results should be identical
        torch.testing.assert_close(logits1, logits2, rtol=1e-6, atol=1e-6)
    
    @pytest.mark.slow
    def test_transformer_forward_memory_usage(self, device):
        """Test memory usage during forward pass."""
        if device.type != 'cuda':
            pytest.skip("Memory testing only relevant for CUDA")
        
        args = ModelArgs(dim=512, n_layers=4, n_heads=8, vocab_size=5000)
        model = Transformer(args).to(device)
        
        batch_size = 4
        seq_len = 128
        cache_len = 256
        
        # Clear cache and measure initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create inputs
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        cache = make_cache(args, cache_len, device=device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(token_values, token_lengths, start_pos, cache, cache_len)
        
        peak_memory = torch.cuda.memory_allocated()
        memory_used = peak_memory - initial_memory
        
        # Memory usage should be reasonable (less than 1GB for this small model)
        assert memory_used < 1024 * 1024 * 1024  # 1GB
        
        # Clean up
        del logits, cache, token_values, token_lengths, start_pos
        torch.cuda.empty_cache()

class TestModelSerialization:
    """Test model serialization and loading."""
    
    def test_model_state_dict_keys(self):
        """Test that model state dict has expected keys."""
        args = ModelArgs(dim=128, n_layers=2, n_heads=4, vocab_size=100)
        model = Transformer(args)
        
        state_dict = model.state_dict()
        
        # Check for expected keys
        expected_keys = [
            'tok_embeddings.weight',
            'norm.weight',
            'output.weight',
        ]
        
        for key in expected_keys:
            assert key in state_dict
        
        # Check layer keys
        for i in range(args.n_layers):
            layer_keys = [
                f'layers.{i}.attention.wqkv.weight',
                f'layers.{i}.attention.wo.weight',
                f'layers.{i}.attention.attn_sub_norm.weight',
                f'layers.{i}.feed_forward.w13.weight',
                f'layers.{i}.feed_forward.w2.weight',
                f'layers.{i}.feed_forward.ffn_sub_norm.weight',
                f'layers.{i}.attention_norm.weight',
                f'layers.{i}.ffn_norm.weight',
            ]
            
            for key in layer_keys:
                assert key in state_dict
    
    def test_model_save_load_consistency(self, temp_model_dir):
        """Test that model can be saved and loaded consistently."""
        args = ModelArgs(dim=128, n_layers=1, n_heads=4, vocab_size=100)
        original_model = Transformer(args)
        
        # Save model
        model_path = temp_model_dir / "test_model.pt"
        torch.save(original_model.state_dict(), model_path)
        
        # Load model
        loaded_model = Transformer(args)
        loaded_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # Compare state dicts
        original_state = original_model.state_dict()
        loaded_state = loaded_model.state_dict()
        
        for key in original_state:
            torch.testing.assert_close(original_state[key], loaded_state[key])
    
    def test_model_load_strict_mode(self, temp_model_dir):
        """Test model loading with strict mode."""
        args = ModelArgs(dim=128, n_layers=1, n_heads=4, vocab_size=100)
        model = Transformer(args)
        
        # Save partial state dict
        state_dict = model.state_dict()
        partial_state = {k: v for k, v in state_dict.items() if 'tok_embeddings' in k}
        
        model_path = temp_model_dir / "partial_model.pt"
        torch.save(partial_state, model_path)
        
        # Loading with strict=True should fail
        new_model = Transformer(args)
        with pytest.raises(RuntimeError):
            new_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        
        # Loading with strict=False should succeed
        new_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

@pytest.mark.slow
class TestModelPerformance:
    """Test model performance characteristics."""
    
    def test_forward_pass_timing(self, device):
        """Test forward pass timing for performance regression detection."""
        args = ModelArgs(dim=512, n_layers=6, n_heads=8, vocab_size=10000)
        model = Transformer(args).to(device)
        model.eval()
        
        batch_size = 4
        seq_len = 64
        cache_len = 128
        
        # Create inputs
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        cache = make_cache(args, cache_len, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(token_values, token_lengths, start_pos, cache, cache_len)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Time forward passes
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(token_values, token_lengths, start_pos, cache, cache_len)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Performance should be reasonable (less than 100ms per forward pass for this size)
        assert avg_time < 0.1, f"Forward pass too slow: {avg_time:.4f}s"
        
        # Store timing for regression detection
        print(f"Average forward pass time: {avg_time:.4f}s")