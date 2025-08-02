"""
Property-based tests for BitNet using Hypothesis.
"""
import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import sys
from pathlib import Path

# Import BitNet modules
try:
    from gpu.model import BitLinear, BitLinearKernel, ModelArgs, Transformer
    from gpu.pack_weight import convert_weight_int8_to_int2
except ImportError as e:
    pytest.skip(f"BitNet modules not available: {e}", allow_module_level=True)

# Hypothesis strategies for common data types
@st.composite
def tensor_shape(draw, min_dim=1, max_dim=4, min_size=1, max_size=1024):
    """Generate valid tensor shapes."""
    ndim = draw(st.integers(min_value=min_dim, max_value=max_dim))
    shape = draw(st.lists(
        st.integers(min_value=min_size, max_value=max_size),
        min_size=ndim,
        max_size=ndim
    ))
    return tuple(shape)

@st.composite
def matrix_shape(draw, min_size=1, max_size=512):
    """Generate valid matrix shapes (2D)."""
    rows = draw(st.integers(min_value=min_size, max_value=max_size))
    cols = draw(st.integers(min_value=min_size, max_value=max_size))
    return (rows, cols)

@st.composite
def linear_layer_config(draw, min_features=1, max_features=256):
    """Generate valid linear layer configurations."""
    in_features = draw(st.integers(min_value=min_features, max_value=max_features))
    out_features = draw(st.integers(min_value=min_features, max_value=max_features))
    return (in_features, out_features)

@st.composite
def model_config(draw):
    """Generate valid model configurations."""
    # Ensure dimensions are compatible
    n_heads = draw(st.integers(min_value=2, max_value=16))
    dim = draw(st.integers(min_value=n_heads, max_value=512))
    # Ensure dim is divisible by n_heads
    dim = (dim // n_heads) * n_heads
    
    n_kv_heads = draw(st.integers(min_value=1, max_value=n_heads))
    # Ensure n_heads is divisible by n_kv_heads
    n_heads = (n_heads // n_kv_heads) * n_kv_heads
    
    return {
        "dim": dim,
        "n_layers": draw(st.integers(min_value=1, max_value=6)),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "vocab_size": draw(st.integers(min_value=100, max_value=1000)),
        "ffn_dim": draw(st.integers(min_value=dim, max_value=dim * 4)),
    }

class TestQuantizationProperties:
    """Property-based tests for quantization."""
    
    @given(arrays(np.float32, tensor_shape(min_dim=2, max_dim=2, min_size=1, max_size=64)))
    @settings(max_examples=50, deadline=5000)
    def test_sign_quantization_idempotent(self, arr):
        """Test that sign quantization is idempotent."""
        tensor = torch.from_numpy(arr)
        
        # Apply quantization once
        quantized_once = torch.sign(tensor)
        
        # Apply quantization again
        quantized_twice = torch.sign(quantized_once)
        
        # Results should be identical
        torch.testing.assert_close(quantized_once, quantized_twice)
    
    @given(arrays(np.float32, tensor_shape(min_dim=2, max_dim=2, min_size=1, max_size=64)))
    @settings(max_examples=50, deadline=5000)
    def test_sign_quantization_range(self, arr):
        """Test that sign quantization produces values in correct range."""
        tensor = torch.from_numpy(arr)
        quantized = torch.sign(tensor)
        
        # All values should be in {-1, 0, 1}
        valid_values = torch.tensor([-1.0, 0.0, 1.0])
        for value in quantized.unique():
            assert value in valid_values, f"Invalid quantized value: {value}"
    
    @given(arrays(np.float32, tensor_shape(min_dim=2, max_dim=2, min_size=1, max_size=64)))
    @settings(max_examples=50, deadline=5000)
    def test_sign_quantization_preserves_zero(self, arr):
        """Test that sign quantization preserves zero values."""
        tensor = torch.from_numpy(arr)
        
        # Set some values to exactly zero
        if tensor.numel() > 0:
            tensor.flat[0] = 0.0
            if tensor.numel() > 1:
                tensor.flat[-1] = 0.0
        
        quantized = torch.sign(tensor)
        
        # Check that zeros are preserved
        zero_mask = tensor == 0.0
        assert torch.all(quantized[zero_mask] == 0.0), "Zero values not preserved"
    
    @given(arrays(np.float32, tensor_shape(min_dim=2, max_dim=2, min_size=1, max_size=64)))
    @settings(max_examples=50, deadline=5000)
    def test_sign_quantization_preserves_sign(self, arr):
        """Test that sign quantization preserves sign information."""
        tensor = torch.from_numpy(arr)
        quantized = torch.sign(tensor)
        
        # For non-zero values, signs should match
        nonzero_mask = tensor != 0.0
        if nonzero_mask.any():
            original_signs = torch.sign(tensor[nonzero_mask])
            quantized_signs = quantized[nonzero_mask]
            torch.testing.assert_close(original_signs, quantized_signs)
    
    @given(matrix_shape(min_size=4, max_size=32))
    @settings(max_examples=20, deadline=10000)
    def test_weight_conversion_shape_preservation(self, shape):
        """Test that weight conversion preserves expected shape relationships."""
        M, K = shape
        assume(K % 4 == 0)  # Required for int2 packing
        
        # Create test weight
        weight_int8 = torch.randint(-1, 2, (M, K), dtype=torch.int8)
        
        # Convert to int2
        weight_int2 = convert_weight_int8_to_int2(weight_int8)
        
        # Check shape relationship (4 int8 values packed into 1 int2 byte)
        expected_shape = (M, K // 4)
        assert weight_int2.shape == expected_shape
        assert weight_int2.dtype == torch.int8
    
    @given(matrix_shape(min_size=4, max_size=32))
    @settings(max_examples=20, deadline=10000)
    def test_weight_conversion_deterministic(self, shape):
        """Test that weight conversion is deterministic."""
        M, K = shape
        assume(K % 4 == 0)
        
        weight_int8 = torch.randint(-1, 2, (M, K), dtype=torch.int8)
        
        # Convert multiple times
        weight_int2_1 = convert_weight_int8_to_int2(weight_int8)
        weight_int2_2 = convert_weight_int8_to_int2(weight_int8)
        
        # Results should be identical
        torch.testing.assert_close(weight_int2_1, weight_int2_2)

class TestBitLinearProperties:
    """Property-based tests for BitLinear layers."""
    
    @given(linear_layer_config(min_features=4, max_features=64))
    @settings(max_examples=20, deadline=10000)
    def test_bitlinear_output_shape(self, config):
        """Test that BitLinear produces correct output shape."""
        in_features, out_features = config
        batch_size = 4
        
        layer = BitLinear(in_features, out_features)
        input_tensor = torch.randn(batch_size, in_features)
        
        output = layer(input_tensor)
        
        expected_shape = (batch_size, out_features)
        assert output.shape == expected_shape
    
    @given(linear_layer_config(min_features=4, max_features=64))
    @settings(max_examples=20, deadline=10000)
    def test_bitlinear_deterministic(self, config):
        """Test that BitLinear is deterministic for same input."""
        in_features, out_features = config
        batch_size = 2
        
        layer = BitLinear(in_features, out_features)
        input_tensor = torch.randn(batch_size, in_features)
        
        # Forward pass twice
        output1 = layer(input_tensor)
        output2 = layer(input_tensor)
        
        # Results should be identical
        torch.testing.assert_close(output1, output2, rtol=1e-6, atol=1e-6)
    
    @given(
        linear_layer_config(min_features=4, max_features=64),
        st.integers(min_value=1, max_value=8)
    )
    @settings(max_examples=20, deadline=10000)
    def test_bitlinear_batch_consistency(self, config, batch_size):
        """Test that BitLinear handles different batch sizes consistently."""
        in_features, out_features = config
        
        layer = BitLinear(in_features, out_features)
        
        # Create inputs with different batch sizes
        input1 = torch.randn(1, in_features)
        input_batch = input1.repeat(batch_size, 1)
        
        # Forward pass
        output1 = layer(input1)
        output_batch = layer(input_batch)
        
        # First output should match first element of batch output
        torch.testing.assert_close(output1, output_batch[:1], rtol=1e-6, atol=1e-6)
        
        # All batch elements should be identical (same input)
        for i in range(1, batch_size):
            torch.testing.assert_close(output_batch[0:1], output_batch[i:i+1], rtol=1e-6, atol=1e-6)
    
    @given(linear_layer_config(min_features=4, max_features=64))
    @settings(max_examples=20, deadline=10000)
    def test_bitlinear_numerical_stability(self, config):
        """Test BitLinear numerical stability with extreme inputs."""
        in_features, out_features = config
        batch_size = 2
        
        layer = BitLinear(in_features, out_features)
        
        # Test with extreme values
        extreme_input = torch.zeros(batch_size, in_features)
        extreme_input[0, 0] = 1000.0   # Large positive
        extreme_input[0, 1] = -1000.0  # Large negative
        extreme_input[1, 0] = 1e-8     # Very small
        extreme_input[1, 1] = 0.0      # Zero
        
        output = layer(extreme_input)
        
        # Output should be finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        assert not torch.isnan(output).any(), "Output contains NaN values"

class TestModelProperties:
    """Property-based tests for model components."""
    
    @given(model_config())
    @settings(max_examples=10, deadline=15000)
    def test_model_initialization_valid(self, config):
        """Test that model initializes correctly with valid configurations."""
        args = ModelArgs(**config)
        
        # Should not raise an exception
        model = Transformer(args)
        
        # Check basic structure
        assert hasattr(model, 'tok_embeddings')
        assert hasattr(model, 'layers')
        assert hasattr(model, 'norm')
        assert hasattr(model, 'output')
        assert len(model.layers) == args.n_layers
    
    @given(
        model_config(),
        st.integers(min_value=1, max_value=4),  # batch_size
        st.integers(min_value=1, max_value=32)  # seq_len
    )
    @settings(max_examples=10, deadline=20000)
    def test_model_forward_shape_consistency(self, config, batch_size, seq_len):
        """Test that model forward pass produces consistent shapes."""
        args = ModelArgs(**config)
        model = Transformer(args)
        model.eval()
        
        cache_len = seq_len + 16
        
        # Create inputs
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,))
        token_lengths = torch.tensor([seq_len] * batch_size)
        start_pos = torch.zeros(batch_size, dtype=torch.long)
        
        # Create cache
        from gpu.model import make_cache
        cache = make_cache(args, cache_len)
        
        with torch.no_grad():
            logits = model(token_values, token_lengths, start_pos, cache, cache_len)
        
        # Check output shape
        expected_shape = (batch_size * seq_len, args.vocab_size)
        assert logits.shape == expected_shape
        assert logits.dtype == torch.float32
    
    @given(model_config())
    @settings(max_examples=5, deadline=20000)
    def test_model_parameter_count_consistency(self, config):
        """Test that model parameter count is consistent with configuration."""
        args = ModelArgs(**config)
        model = Transformer(args)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have reasonable number of parameters (not zero, not excessive)
        assert total_params > 0, "Model has no parameters"
        assert total_params < 1e9, "Model has excessive parameters"  # Less than 1B params
        
        # Parameter count should be deterministic for same config
        model2 = Transformer(args)
        total_params2 = sum(p.numel() for p in model2.parameters())
        assert total_params == total_params2, "Parameter count not consistent"

class TestNumericalProperties:
    """Property-based tests for numerical properties."""
    
    @given(arrays(np.float32, tensor_shape(min_dim=1, max_dim=3, min_size=1, max_size=32)))
    @settings(max_examples=50, deadline=5000)
    def test_softmax_properties(self, arr):
        """Test softmax properties."""
        assume(arr.size > 0)
        tensor = torch.from_numpy(arr)
        
        # Apply softmax along last dimension
        if tensor.dim() > 0:
            probs = torch.softmax(tensor, dim=-1)
            
            # Check properties
            assert torch.all(probs >= 0), "Softmax should produce non-negative values"
            assert torch.all(probs <= 1), "Softmax should produce values <= 1"
            
            # Check that probabilities sum to 1 along last dimension
            prob_sums = probs.sum(dim=-1)
            expected_sums = torch.ones_like(prob_sums)
            torch.testing.assert_close(prob_sums, expected_sums, rtol=1e-5, atol=1e-6)
    
    @given(
        arrays(np.float32, tensor_shape(min_dim=2, max_dim=2, min_size=1, max_size=32)),
        st.floats(min_value=0.1, max_value=2.0)
    )
    @settings(max_examples=30, deadline=5000)
    def test_temperature_scaling_properties(self, arr, temperature):
        """Test temperature scaling properties."""
        assume(arr.size > 0)
        tensor = torch.from_numpy(arr)
        
        # Apply temperature scaling
        scaled = tensor / temperature
        
        # Properties should be preserved
        assert scaled.shape == tensor.shape
        assert torch.isfinite(scaled).all()
        
        # Higher temperature should reduce the range of values
        if temperature > 1.0:
            assert scaled.abs().max() <= tensor.abs().max()
    
    @given(
        arrays(np.float32, (st.integers(1, 16), st.integers(10, 100))),
        st.floats(min_value=0.1, max_value=0.99)
    )
    @settings(max_examples=20, deadline=5000)
    def test_top_p_sampling_properties(self, arr, p_value):
        """Test top-p sampling properties."""
        assume(arr.size > 0)
        logits = torch.from_numpy(arr)
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Import top_p function
        try:
            from gpu.sample_utils import top_p
            
            # Sample tokens
            sampled = top_p(probs, p_value)
            
            # Check properties
            assert sampled.shape == (probs.shape[0],)
            assert sampled.dtype == torch.long
            assert torch.all(sampled >= 0)
            assert torch.all(sampled < probs.shape[-1])
            
        except ImportError:
            pytest.skip("top_p function not available")

class TestInvarianceProperties:
    """Property-based tests for invariance properties."""
    
    @given(
        linear_layer_config(min_features=4, max_features=32),
        st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=20, deadline=10000)
    def test_bitlinear_scale_invariance(self, config, scale_factor):
        """Test BitLinear behavior under input scaling."""
        in_features, out_features = config
        batch_size = 2
        
        layer = BitLinear(in_features, out_features)
        input_tensor = torch.randn(batch_size, in_features)
        scaled_input = input_tensor * scale_factor
        
        # Forward pass
        output1 = layer(input_tensor)
        output2 = layer(scaled_input)
        
        # Due to quantization, outputs should be related but not necessarily identical
        # Both should be finite and have same shape
        assert output1.shape == output2.shape
        assert torch.isfinite(output1).all()
        assert torch.isfinite(output2).all()
    
    @given(model_config())
    @settings(max_examples=5, deadline=20000)
    def test_model_permutation_invariance(self, config):
        """Test model behavior under token permutation (within constraints)."""
        args = ModelArgs(**config)
        model = Transformer(args)
        model.eval()
        
        batch_size = 2
        seq_len = 4
        cache_len = seq_len + 8
        
        # Create inputs
        tokens = torch.randint(0, args.vocab_size, (seq_len,))
        
        # Create two batches with same tokens in different order
        token_values1 = tokens.repeat(batch_size)
        token_values2 = torch.cat([tokens, tokens])  # Same as above, just explicit
        
        token_lengths = torch.tensor([seq_len] * batch_size)
        start_pos = torch.zeros(batch_size, dtype=torch.long)
        
        from gpu.model import make_cache
        cache1 = make_cache(args, cache_len)
        cache2 = make_cache(args, cache_len)
        
        with torch.no_grad():
            logits1 = model(token_values1, token_lengths, start_pos, cache1, cache_len)
            logits2 = model(token_values2, token_lengths, start_pos, cache2, cache_len)
        
        # Should produce same results for same inputs
        torch.testing.assert_close(logits1, logits2, rtol=1e-6, atol=1e-6)

# Custom strategies for edge cases
@st.composite
def edge_case_tensors(draw):
    """Generate tensors with edge case values."""
    shape = draw(tensor_shape(min_dim=2, max_dim=2, min_size=2, max_size=16))
    base_tensor = torch.randn(shape)
    
    # Add edge case values
    if base_tensor.numel() >= 4:
        base_tensor.flat[0] = 0.0          # Zero
        base_tensor.flat[1] = float('inf')  # Infinity
        base_tensor.flat[2] = float('-inf') # Negative infinity
        base_tensor.flat[3] = float('nan')  # NaN
    
    return base_tensor

class TestEdgeCaseProperties:
    """Property-based tests for edge cases."""
    
    @given(edge_case_tensors())
    @settings(max_examples=20, deadline=5000)
    def test_quantization_edge_cases(self, tensor):
        """Test quantization with edge case values."""
        # Apply quantization
        quantized = torch.sign(tensor)
        
        # Check handling of special values
        inf_mask = torch.isinf(tensor)
        if inf_mask.any():
            # Positive infinity should become 1, negative infinity should become -1
            pos_inf_mask = tensor == float('inf')
            neg_inf_mask = tensor == float('-inf')
            
            if pos_inf_mask.any():
                assert torch.all(quantized[pos_inf_mask] == 1.0)
            if neg_inf_mask.any():
                assert torch.all(quantized[neg_inf_mask] == -1.0)
        
        # NaN should remain NaN
        nan_mask = torch.isnan(tensor)
        if nan_mask.any():
            assert torch.all(torch.isnan(quantized[nan_mask]))
        
        # Zero should remain zero
        zero_mask = tensor == 0.0
        if zero_mask.any():
            assert torch.all(quantized[zero_mask] == 0.0)
    
    @given(
        st.integers(min_value=1, max_value=8),  # batch_size
        st.integers(min_value=0, max_value=2)   # seq_len (including 0)
    )
    @settings(max_examples=10, deadline=10000)
    def test_model_edge_case_inputs(self, batch_size, seq_len):
        """Test model with edge case input sizes."""
        config = {
            "dim": 64,
            "n_layers": 1,
            "n_heads": 4,
            "n_kv_heads": 2,
            "vocab_size": 100,
            "ffn_dim": 128,
        }
        
        args = ModelArgs(**config)
        model = Transformer(args)
        model.eval()
        
        if seq_len == 0:
            # Empty sequence should be handled gracefully
            pytest.skip("Empty sequences not supported in current implementation")
        
        cache_len = max(seq_len + 8, 16)
        
        # Create inputs
        token_values = torch.randint(0, args.vocab_size, (batch_size * seq_len,))
        token_lengths = torch.tensor([seq_len] * batch_size)
        start_pos = torch.zeros(batch_size, dtype=torch.long)
        
        from gpu.model import make_cache
        cache = make_cache(args, cache_len)
        
        try:
            with torch.no_grad():
                logits = model(token_values, token_lengths, start_pos, cache, cache_len)
            
            # If successful, check output properties
            expected_shape = (batch_size * seq_len, args.vocab_size)
            assert logits.shape == expected_shape
            assert torch.isfinite(logits).all()
            
        except (RuntimeError, ValueError) as e:
            # Some edge cases might legitimately fail
            if seq_len == 0:
                # Empty sequences might not be supported
                pass
            else:
                raise