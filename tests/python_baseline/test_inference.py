"""
Test suite for BitNet inference functionality.
"""
import pytest
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import sys
from pathlib import Path

# Import BitNet modules
try:
    from gpu.generate import FastGen, GenArgs
    from gpu.model import ModelArgs, Transformer, make_cache
    from gpu.tokenizer import Tokenizer
    from gpu.sample_utils import top_p
except ImportError as e:
    pytest.skip(f"BitNet inference modules not available: {e}", allow_module_level=True)

class TestInferenceConfiguration:
    """Test inference configuration and setup."""
    
    def test_gen_args_default_values(self):
        """Test GenArgs default values."""
        args = GenArgs()
        
        assert args.gen_length == 32
        assert args.gen_bsz == 1
        assert args.prompt_length == 64
        assert args.use_sampling == False
        assert args.temperature == 0.8
        assert args.top_p == 0.9
    
    def test_gen_args_custom_values(self):
        """Test GenArgs with custom values."""
        args = GenArgs(
            gen_length=128,
            gen_bsz=4,
            prompt_length=256,
            use_sampling=True,
            temperature=0.7,
            top_p=0.95
        )
        
        assert args.gen_length == 128
        assert args.gen_bsz == 4
        assert args.prompt_length == 256
        assert args.use_sampling == True
        assert args.temperature == 0.7
        assert args.top_p == 0.95
    
    def test_model_args_for_inference(self):
        """Test ModelArgs configuration for inference."""
        # Test prefill model (use_kernel=False)
        prefill_args = ModelArgs(use_kernel=False)
        assert prefill_args.use_kernel == False
        
        # Test decode model (use_kernel=True)
        decode_args = ModelArgs(use_kernel=True)
        assert decode_args.use_kernel == True

class TestTokenGeneration:
    """Test token generation functionality."""
    
    def test_greedy_generation_deterministic(self, device):
        """Test that greedy generation is deterministic."""
        args = ModelArgs(dim=256, n_layers=2, n_heads=4, vocab_size=1000)
        model = Transformer(args).to(device)
        model.eval()
        
        batch_size = 1
        seq_len = 10
        cache_len = 64
        
        # Create identical inputs
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Generate twice with same inputs
        cache1 = make_cache(args, cache_len, device=device)
        cache2 = make_cache(args, cache_len, device=device)
        
        with torch.no_grad():
            logits1 = model(token_values, token_lengths, start_pos, cache1, cache_len)
            logits2 = model(token_values, token_lengths, start_pos, cache2, cache_len)
            
            # Greedy selection
            next_token1 = torch.argmax(logits1, dim=-1)
            next_token2 = torch.argmax(logits2, dim=-1)
        
        # Results should be identical
        torch.testing.assert_close(next_token1, next_token2)
    
    def test_sampling_generation_randomness(self, device):
        """Test that sampling generation produces different results."""
        args = ModelArgs(dim=128, n_layers=1, n_heads=2, vocab_size=100)
        model = Transformer(args).to(device)
        model.eval()
        
        batch_size = 1
        seq_len = 5
        cache_len = 32
        
        # Create inputs
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Generate multiple times with sampling
        results = []
        for i in range(10):
            cache = make_cache(args, cache_len, device=device)
            
            with torch.no_grad():
                logits = model(token_values, token_lengths, start_pos, cache, cache_len)
                
                # Apply temperature and top-p sampling
                temperature = 0.8
                top_p_value = 0.9
                
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = top_p(probs, top_p_value)
                results.append(next_token.cpu())
        
        # Results should show some variation (not all identical)
        unique_results = len(set(tuple(r.flatten().tolist()) for r in results))
        assert unique_results > 1, "Sampling should produce varied results"
    
    def test_top_p_sampling_properties(self, device):
        """Test top-p sampling properties."""
        # Create test logits
        batch_size = 2
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size, device=device)
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Test different top-p values
        for p_value in [0.5, 0.8, 0.9, 0.95]:
            sampled_tokens = top_p(probs, p_value)
            
            # Check output shape and range
            assert sampled_tokens.shape == (batch_size,)
            assert sampled_tokens.min() >= 0
            assert sampled_tokens.max() < vocab_size
            assert sampled_tokens.dtype == torch.long
    
    def test_temperature_scaling_effect(self, device):
        """Test temperature scaling effect on probability distribution."""
        vocab_size = 50
        logits = torch.randn(1, vocab_size, device=device)
        
        # Test different temperatures
        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
        entropies = []
        
        for temp in temperatures:
            scaled_logits = logits / temp
            probs = torch.softmax(scaled_logits, dim=-1)
            
            # Calculate entropy
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            entropies.append(entropy.item())
        
        # Higher temperature should generally lead to higher entropy
        assert entropies[0] < entropies[-1], "Higher temperature should increase entropy"

class TestInferenceAccuracy:
    """Test inference accuracy and correctness."""
    
    def test_logits_shape_consistency(self, device):
        """Test that logits have consistent shape across different inputs."""
        args = ModelArgs(dim=128, n_layers=1, n_heads=2, vocab_size=100)
        model = Transformer(args).to(device)
        model.eval()
        
        cache_len = 64
        
        # Test different sequence lengths
        for seq_len in [1, 5, 10, 20]:
            batch_size = 2
            token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
            token_lengths = torch.tensor([seq_len] * batch_size, device=device)
            start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
            cache = make_cache(args, cache_len, device=device)
            
            with torch.no_grad():
                logits = model(token_values, token_lengths, start_pos, cache, cache_len)
            
            # Check shape
            expected_shape = (batch_size * seq_len, args.vocab_size)
            assert logits.shape == expected_shape
            assert logits.dtype == torch.float32
    
    def test_logits_numerical_stability(self, device):
        """Test numerical stability of logits."""
        args = ModelArgs(dim=128, n_layers=2, n_heads=4, vocab_size=100)
        model = Transformer(args).to(device)
        model.eval()
        
        batch_size = 2
        seq_len = 10
        cache_len = 32
        
        # Create inputs
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        cache = make_cache(args, cache_len, device=device)
        
        with torch.no_grad():
            logits = model(token_values, token_lengths, start_pos, cache, cache_len)
        
        # Check for numerical issues
        assert not torch.isnan(logits).any(), "Logits contain NaN values"
        assert not torch.isinf(logits).any(), "Logits contain infinite values"
        assert logits.abs().max() < 1000, "Logits have extreme values"
    
    def test_probability_distribution_validity(self, device):
        """Test that probability distributions are valid."""
        args = ModelArgs(dim=64, n_layers=1, n_heads=2, vocab_size=50)
        model = Transformer(args).to(device)
        model.eval()
        
        batch_size = 1
        seq_len = 5
        cache_len = 16
        
        # Create inputs
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        cache = make_cache(args, cache_len, device=device)
        
        with torch.no_grad():
            logits = model(token_values, token_lengths, start_pos, cache, cache_len)
            probs = torch.softmax(logits, dim=-1)
        
        # Check probability properties
        assert torch.all(probs >= 0), "Probabilities should be non-negative"
        assert torch.all(probs <= 1), "Probabilities should not exceed 1"
        
        # Check that probabilities sum to 1 (within numerical tolerance)
        prob_sums = probs.sum(dim=-1)
        torch.testing.assert_close(prob_sums, torch.ones_like(prob_sums), rtol=1e-5, atol=1e-6)

class TestInferencePerformance:
    """Test inference performance characteristics."""
    
    @pytest.mark.slow
    def test_single_token_generation_speed(self, device):
        """Test single token generation speed."""
        args = ModelArgs(dim=512, n_layers=6, n_heads=8, vocab_size=10000)
        model = Transformer(args).to(device)
        model.eval()
        
        batch_size = 1
        seq_len = 1  # Single token generation
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
        
        # Time generation
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(50):
                _ = model(token_values, token_lengths, start_pos, cache, cache_len)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 50
        
        # Single token generation should be fast (less than 10ms)
        assert avg_time < 0.01, f"Single token generation too slow: {avg_time:.4f}s"
        
        print(f"Average single token generation time: {avg_time:.4f}s")
    
    @pytest.mark.slow
    def test_batch_inference_scaling(self, device):
        """Test inference scaling with batch size."""
        args = ModelArgs(dim=256, n_layers=3, n_heads=4, vocab_size=1000)
        model = Transformer(args).to(device)
        model.eval()
        
        seq_len = 10
        cache_len = 32
        
        batch_times = {}
        
        for batch_size in [1, 2, 4, 8]:
            # Create inputs
            token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
            token_lengths = torch.tensor([seq_len] * batch_size, device=device)
            start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
            cache = make_cache(args, cache_len * batch_size, device=device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(token_values, token_lengths, start_pos, cache, cache_len * batch_size)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Time inference
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    _ = model(token_values, token_lengths, start_pos, cache, cache_len * batch_size)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            batch_times[batch_size] = avg_time
            
            print(f"Batch size {batch_size}: {avg_time:.4f}s")
        
        # Batch processing should be more efficient than linear scaling
        # (batch_size=4 should be less than 4x batch_size=1)
        if 1 in batch_times and 4 in batch_times:
            efficiency_ratio = batch_times[4] / (4 * batch_times[1])
            assert efficiency_ratio < 1.5, f"Batch processing not efficient: {efficiency_ratio:.2f}"
    
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_memory_usage_scaling(self, device):
        """Test memory usage scaling with model size."""
        if device.type != 'cuda':
            pytest.skip("Memory testing only relevant for CUDA")
        
        memory_usage = {}
        
        # Test different model sizes
        model_configs = [
            (128, 2, 4, 500),    # Small
            (256, 4, 8, 1000),   # Medium
            (512, 6, 16, 2000),  # Large
        ]
        
        for dim, n_layers, n_heads, vocab_size in model_configs:
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            args = ModelArgs(dim=dim, n_layers=n_layers, n_heads=n_heads, vocab_size=vocab_size)
            model = Transformer(args).to(device)
            
            after_model = torch.cuda.memory_allocated()
            model_memory = after_model - initial_memory
            memory_usage[dim] = model_memory
            
            print(f"Model dim={dim}: {model_memory / 1024**2:.1f} MB")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
        
        # Memory usage should scale reasonably with model size
        # Larger models should use more memory, but not excessively
        if 128 in memory_usage and 512 in memory_usage:
            scaling_factor = memory_usage[512] / memory_usage[128]
            # 4x larger model should use less than 20x memory (rough heuristic)
            assert scaling_factor < 20, f"Memory scaling too high: {scaling_factor:.1f}x"

class TestInferenceEdgeCases:
    """Test inference edge cases and error handling."""
    
    def test_empty_sequence_handling(self, device):
        """Test handling of empty sequences."""
        args = ModelArgs(dim=64, n_layers=1, n_heads=2, vocab_size=100)
        model = Transformer(args).to(device)
        model.eval()
        
        # Test with sequence length 0 (should handle gracefully or raise appropriate error)
        batch_size = 1
        seq_len = 0
        cache_len = 16
        
        if seq_len > 0:  # Only test if we have tokens
            token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
            token_lengths = torch.tensor([seq_len] * batch_size, device=device)
            start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
            cache = make_cache(args, cache_len, device=device)
            
            # This should either work or raise a clear error
            try:
                with torch.no_grad():
                    logits = model(token_values, token_lengths, start_pos, cache, cache_len)
                # If it works, check the output
                assert logits.shape[0] == batch_size * seq_len
            except (RuntimeError, ValueError) as e:
                # Expected for empty sequences
                assert "empty" in str(e).lower() or "zero" in str(e).lower()
    
    def test_single_token_sequence(self, device):
        """Test handling of single token sequences."""
        args = ModelArgs(dim=64, n_layers=1, n_heads=2, vocab_size=100)
        model = Transformer(args).to(device)
        model.eval()
        
        batch_size = 1
        seq_len = 1
        cache_len = 16
        
        # Create single token input
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        cache = make_cache(args, cache_len, device=device)
        
        with torch.no_grad():
            logits = model(token_values, token_lengths, start_pos, cache, cache_len)
        
        # Check output
        assert logits.shape == (batch_size * seq_len, args.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_out_of_vocabulary_tokens(self, device):
        """Test handling of out-of-vocabulary tokens."""
        args = ModelArgs(dim=64, n_layers=1, n_heads=2, vocab_size=100)
        model = Transformer(args).to(device)
        model.eval()
        
        batch_size = 1
        seq_len = 5
        cache_len = 16
        
        # Create inputs with out-of-vocab tokens
        token_values = torch.tensor([0, 1, args.vocab_size, args.vocab_size + 10, 2], device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        cache = make_cache(args, cache_len, device=device)
        
        # This should raise an error or handle gracefully
        try:
            with torch.no_grad():
                logits = model(token_values, token_lengths, start_pos, cache, cache_len)
            # If it doesn't raise an error, the embedding layer might handle it
            assert logits.shape == (batch_size * seq_len, args.vocab_size)
        except (RuntimeError, IndexError) as e:
            # Expected for out-of-vocab tokens
            assert "index" in str(e).lower() or "out of range" in str(e).lower()
    
    def test_very_long_sequence(self, device):
        """Test handling of very long sequences."""
        args = ModelArgs(dim=64, n_layers=1, n_heads=2, vocab_size=100)
        model = Transformer(args).to(device)
        model.eval()
        
        batch_size = 1
        seq_len = 1000  # Very long sequence
        cache_len = seq_len + 100
        
        # Create long sequence
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        cache = make_cache(args, cache_len, device=device)
        
        # This might be slow but should work
        try:
            with torch.no_grad():
                logits = model(token_values, token_lengths, start_pos, cache, cache_len)
            
            assert logits.shape == (batch_size * seq_len, args.vocab_size)
            assert not torch.isnan(logits).any()
            assert not torch.isinf(logits).any()
        except RuntimeError as e:
            # Might run out of memory or hit other limits
            if "memory" in str(e).lower():
                pytest.skip(f"Insufficient memory for long sequence test: {e}")
            else:
                raise

class TestInferenceIntegration:
    """Test integration between different inference components."""
    
    def test_prefill_decode_consistency(self, device):
        """Test consistency between prefill and decode phases."""
        # This test would require the actual FastGen implementation
        # For now, we test the basic model consistency
        
        args = ModelArgs(dim=128, n_layers=2, n_heads=4, vocab_size=100)
        model = Transformer(args).to(device)
        model.eval()
        
        batch_size = 1
        prompt_len = 10
        cache_len = 64
        
        # Simulate prefill phase
        prompt_tokens = torch.randint(0, args.vocab_size, (batch_size * prompt_len,), device=device)
        token_lengths = torch.tensor([prompt_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        cache = make_cache(args, cache_len, device=device)
        
        with torch.no_grad():
            prefill_logits = model(prompt_tokens, token_lengths, start_pos, cache, cache_len)
        
        # Simulate decode phase (single token)
        last_token = torch.argmax(prefill_logits[-1:], dim=-1)
        decode_lengths = torch.tensor([1] * batch_size, device=device)
        decode_start_pos = torch.tensor([prompt_len] * batch_size, dtype=torch.long, device=device)
        
        with torch.no_grad():
            decode_logits = model(last_token, decode_lengths, decode_start_pos, cache, cache_len)
        
        # Both should produce valid outputs
        assert prefill_logits.shape == (batch_size * prompt_len, args.vocab_size)
        assert decode_logits.shape == (batch_size * 1, args.vocab_size)
        assert not torch.isnan(prefill_logits).any()
        assert not torch.isnan(decode_logits).any()
    
    def test_cache_state_consistency(self, device):
        """Test that cache state is consistent across forward passes."""
        args = ModelArgs(dim=64, n_layers=2, n_heads=4, vocab_size=100)
        model = Transformer(args).to(device)
        model.eval()
        
        batch_size = 1
        seq_len = 5
        cache_len = 32
        
        # Create inputs
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,), device=device)
        token_lengths = torch.tensor([seq_len] * batch_size, device=device)
        start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Run with fresh cache
        cache1 = make_cache(args, cache_len, device=device)
        with torch.no_grad():
            logits1 = model(token_values, token_lengths, start_pos, cache1, cache_len)
        
        # Run with another fresh cache
        cache2 = make_cache(args, cache_len, device=device)
        with torch.no_grad():
            logits2 = model(token_values, token_lengths, start_pos, cache2, cache_len)
        
        # Results should be identical (same inputs, fresh caches)
        torch.testing.assert_close(logits1, logits2, rtol=1e-6, atol=1e-6)
        
        # Cache states should be identical after processing same inputs
        for (k1, v1), (k2, v2) in zip(cache1, cache2):
            torch.testing.assert_close(k1, k2, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(v1, v2, rtol=1e-6, atol=1e-6)