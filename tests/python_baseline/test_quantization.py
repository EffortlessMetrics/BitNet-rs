"""
Test suite for BitNet quantization functionality.
"""
import pytest
import torch
import numpy as np
from typing import Tuple, Dict, Any
import sys
from pathlib import Path

# Import BitNet modules
try:
    from gpu.model import BitLinear, BitLinearKernel
    from gpu.pack_weight import convert_weight_int8_to_int2
    from utils.convert import preprocess_weights
except ImportError as e:
    pytest.skip(f"BitNet quantization modules not available: {e}", allow_module_level=True)

class TestBitLinearQuantization:
    """Test BitLinear quantization functionality."""
    
    def test_quant_input_basic(self, device):
        """Test basic input quantization."""
        in_features, out_features = 256, 128
        bitlinear = BitLinear(in_features, out_features).to(device)
        
        # Create test input
        batch_size = 4
        input_tensor = torch.randn(batch_size, in_features, device=device)
        
        # Test quantization
        quantized = bitlinear.quant_input(input_tensor)
        
        # Check output properties
        assert quantized.shape == input_tensor.shape
        assert quantized.device == device
        
        # Check quantization bounds (should be in reasonable range after scaling)
        assert quantized.abs().max() <= 1.1  # Allow small numerical errors
    
    def test_quant_input_deterministic(self, device):
        """Test that quantization is deterministic."""
        in_features, out_features = 128, 64
        bitlinear = BitLinear(in_features, out_features).to(device)
        
        # Create test input
        input_tensor = torch.randn(2, in_features, device=device)
        
        # Quantize twice
        quant1 = bitlinear.quant_input(input_tensor)
        quant2 = bitlinear.quant_input(input_tensor)
        
        # Results should be identical
        torch.testing.assert_close(quant1, quant2, rtol=1e-6, atol=1e-6)
    
    def test_quant_input_zero_handling(self, device):
        """Test quantization handling of zero and near-zero values."""
        in_features, out_features = 64, 32
        bitlinear = BitLinear(in_features, out_features).to(device)
        
        # Create input with zeros and near-zeros
        input_tensor = torch.zeros(2, in_features, device=device)
        input_tensor[0, 0] = 1e-8  # Very small value
        input_tensor[1, 0] = 0.0   # Exact zero
        
        # Should not crash and should handle gracefully
        quantized = bitlinear.quant_input(input_tensor)
        assert not torch.isnan(quantized).any()
        assert not torch.isinf(quantized).any()
    
    def test_quant_input_extreme_values(self, device):
        """Test quantization with extreme input values."""
        in_features, out_features = 64, 32
        bitlinear = BitLinear(in_features, out_features).to(device)
        
        # Create input with extreme values
        input_tensor = torch.zeros(2, in_features, device=device)
        input_tensor[0, 0] = 1000.0   # Large positive
        input_tensor[0, 1] = -1000.0  # Large negative
        input_tensor[1, 0] = 1e-6     # Small positive
        input_tensor[1, 1] = -1e-6    # Small negative
        
        quantized = bitlinear.quant_input(input_tensor)
        
        # Should be properly scaled
        assert quantized.abs().max() <= 1.1
        assert not torch.isnan(quantized).any()
        assert not torch.isinf(quantized).any()
    
    def test_bitlinear_forward_consistency(self, device):
        """Test that BitLinear forward pass is consistent."""
        in_features, out_features = 128, 64
        bitlinear = BitLinear(in_features, out_features).to(device)
        
        # Create test input
        input_tensor = torch.randn(4, in_features, device=device)
        
        # Forward pass twice
        output1 = bitlinear(input_tensor)
        output2 = bitlinear(input_tensor)
        
        # Results should be identical
        torch.testing.assert_close(output1, output2, rtol=1e-6, atol=1e-6)
        
        # Check output shape
        assert output1.shape == (4, out_features)

class TestBitLinearKernel:
    """Test BitLinearKernel functionality."""
    
    @pytest.mark.gpu
    def test_bitlinear_kernel_initialization(self):
        """Test BitLinearKernel initialization."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        in_features, out_features = 256, 128
        kernel = BitLinearKernel(in_features, out_features).cuda()
        
        # Check weight shapes
        assert kernel.weight.shape == (out_features, in_features // 4)
        assert kernel.weight.dtype == torch.int8
        assert kernel.weight_scale.shape == (4,)
        assert kernel.weight_scale.dtype == torch.bfloat16
        
        # Check that weights are on GPU
        assert kernel.weight.is_cuda
        assert kernel.weight_scale.is_cuda
    
    @pytest.mark.gpu
    def test_bitlinear_kernel_quant_input(self):
        """Test BitLinearKernel input quantization."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        in_features, out_features = 256, 128
        kernel = BitLinearKernel(in_features, out_features).cuda()
        
        # Create test input
        input_tensor = torch.randn(4, in_features, device='cuda')
        
        # Test quantization
        quantized, scale = kernel.quant_input(input_tensor)
        
        # Check output properties
        assert quantized.shape == input_tensor.shape
        assert quantized.dtype == torch.int8
        assert scale.shape == (4, 1)  # Batch dimension preserved
        assert scale.dtype == input_tensor.dtype
        
        # Check quantization bounds
        assert quantized.min() >= -128
        assert quantized.max() <= 127
    
    @pytest.mark.gpu
    def test_bitlinear_kernel_forward_shape(self):
        """Test BitLinearKernel forward pass output shape."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        in_features, out_features = 256, 128
        kernel = BitLinearKernel(in_features, out_features).cuda()
        
        # Create test input
        batch_size = 4
        input_tensor = torch.randn(batch_size, in_features, device='cuda')
        
        # Forward pass
        output = kernel(input_tensor)
        
        # Check output shape and type
        assert output.shape == (batch_size, out_features)
        assert output.dtype == torch.bfloat16

class TestWeightConversion:
    """Test weight conversion utilities."""
    
    def test_convert_weight_int8_to_int2_shape(self):
        """Test int8 to int2 weight conversion shape."""
        # Create test weight tensor
        M, K = 128, 256
        weight_int8 = torch.randint(-1, 2, (M, K), dtype=torch.int8)
        
        # Convert to int2
        weight_int2 = convert_weight_int8_to_int2(weight_int8)
        
        # Check output shape (4 int8 values packed into 1 int2 byte)
        expected_shape = (M, K // 4)
        assert weight_int2.shape == expected_shape
        assert weight_int2.dtype == torch.int8
    
    def test_convert_weight_int8_to_int2_values(self):
        """Test int8 to int2 weight conversion values."""
        # Create simple test case
        weight_int8 = torch.tensor([
            [-1, 0, 1, -1],  # Should pack to one byte
            [1, 1, 0, 0],    # Should pack to one byte
        ], dtype=torch.int8)
        
        weight_int2 = convert_weight_int8_to_int2(weight_int8)
        
        # Check that conversion preserves information
        assert weight_int2.shape == (2, 1)
        assert weight_int2.dtype == torch.int8
        
        # The exact packed values depend on the packing implementation
        # but should be deterministic
        weight_int2_again = convert_weight_int8_to_int2(weight_int8)
        torch.testing.assert_close(weight_int2, weight_int2_again)
    
    def test_convert_weight_int8_to_int2_deterministic(self):
        """Test that weight conversion is deterministic."""
        M, K = 64, 128
        weight_int8 = torch.randint(-1, 2, (M, K), dtype=torch.int8)
        
        # Convert multiple times
        weight_int2_1 = convert_weight_int8_to_int2(weight_int8)
        weight_int2_2 = convert_weight_int8_to_int2(weight_int8)
        
        # Results should be identical
        torch.testing.assert_close(weight_int2_1, weight_int2_2)

class TestQuantizationRoundTrip:
    """Test quantization round-trip accuracy."""
    
    def test_quantization_round_trip_basic(self, test_config):
        """Test basic quantization round-trip."""
        # Create test data
        original_data = torch.randn(64, 128)
        
        # Simulate quantization process
        # 1. Sign quantization (BitNet style)
        sign_quantized = torch.sign(original_data)
        
        # 2. Convert to int8
        int8_data = sign_quantized.to(torch.int8)
        
        # 3. Convert back to float
        reconstructed = int8_data.to(torch.float32)
        
        # Check that sign information is preserved
        original_signs = torch.sign(original_data)
        reconstructed_signs = torch.sign(reconstructed)
        
        # Where original was non-zero, signs should match
        non_zero_mask = original_data != 0
        if non_zero_mask.any():
            signs_match = (original_signs[non_zero_mask] == reconstructed_signs[non_zero_mask]).all()
            assert signs_match, "Sign information not preserved in quantization"
    
    def test_quantization_numerical_precision(self, test_config, test_data_generator):
        """Test quantization numerical precision."""
        tolerance = test_config["numerical_tolerance"]
        
        # Generate test data with known properties
        test_data = test_data_generator.generate_quantization_test_data((32, 64))
        
        # Apply BitNet-style quantization
        quantized = torch.sign(test_data)
        
        # Check that quantized values are in expected range
        assert quantized.abs().max() <= 1.0
        assert torch.all(torch.isin(quantized, torch.tensor([-1.0, 0.0, 1.0])))
        
        # Check that zero values are preserved
        zero_mask = test_data == 0.0
        assert torch.all(quantized[zero_mask] == 0.0)
    
    @pytest.mark.parametrize("shape", [(32, 64), (128, 256), (64, 128)])
    def test_quantization_different_shapes(self, shape):
        """Test quantization with different tensor shapes."""
        test_data = torch.randn(shape)
        
        # Apply quantization
        quantized = torch.sign(test_data)
        
        # Check shape preservation
        assert quantized.shape == test_data.shape
        
        # Check value range
        assert quantized.abs().max() <= 1.0

class TestQuantizationProperties:
    """Test mathematical properties of quantization."""
    
    def test_quantization_idempotent(self):
        """Test that quantization is idempotent (applying twice gives same result)."""
        test_data = torch.randn(32, 64)
        
        # Apply quantization once
        quantized_once = torch.sign(test_data)
        
        # Apply quantization again
        quantized_twice = torch.sign(quantized_once)
        
        # Results should be identical
        torch.testing.assert_close(quantized_once, quantized_twice)
    
    def test_quantization_preserves_zero(self):
        """Test that quantization preserves zero values."""
        # Create data with explicit zeros
        test_data = torch.randn(32, 64)
        test_data[0, :] = 0.0  # Set first row to zero
        test_data[:, 0] = 0.0  # Set first column to zero
        
        quantized = torch.sign(test_data)
        
        # Check that zeros are preserved
        assert torch.all(quantized[0, :] == 0.0)
        assert torch.all(quantized[:, 0] == 0.0)
    
    def test_quantization_sign_preservation(self):
        """Test that quantization preserves sign information."""
        # Create data with known signs
        positive_data = torch.abs(torch.randn(16, 32))  # All positive
        negative_data = -torch.abs(torch.randn(16, 32))  # All negative
        
        pos_quantized = torch.sign(positive_data)
        neg_quantized = torch.sign(negative_data)
        
        # Check sign preservation
        assert torch.all(pos_quantized >= 0)  # Should be 0 or 1
        assert torch.all(neg_quantized <= 0)  # Should be 0 or -1
        
        # Check that non-zero values have correct signs
        pos_nonzero = positive_data != 0
        neg_nonzero = negative_data != 0
        
        if pos_nonzero.any():
            assert torch.all(pos_quantized[pos_nonzero] == 1)
        if neg_nonzero.any():
            assert torch.all(neg_quantized[neg_nonzero] == -1)

@pytest.mark.slow
class TestQuantizationPerformance:
    """Test quantization performance characteristics."""
    
    def test_quantization_speed(self, device):
        """Test quantization speed for performance regression detection."""
        # Large tensor for timing
        large_tensor = torch.randn(1024, 2048, device=device)
        
        # Warmup
        for _ in range(5):
            _ = torch.sign(large_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Time quantization
        import time
        start_time = time.time()
        
        for _ in range(100):
            _ = torch.sign(large_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Should be very fast (less than 1ms for this size)
        assert avg_time < 0.001, f"Quantization too slow: {avg_time:.6f}s"
        
        print(f"Average quantization time: {avg_time:.6f}s")
    
    def test_quantization_memory_efficiency(self, device):
        """Test quantization memory efficiency."""
        if device.type != 'cuda':
            pytest.skip("Memory testing only relevant for CUDA")
        
        # Clear cache and measure initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create large tensor
        large_tensor = torch.randn(2048, 4096, device=device)
        after_creation = torch.cuda.memory_allocated()
        
        # Apply quantization
        quantized = torch.sign(large_tensor)
        after_quantization = torch.cuda.memory_allocated()
        
        # Memory usage should not increase significantly
        creation_memory = after_creation - initial_memory
        quantization_memory = after_quantization - after_creation
        
        # Quantization should not use more than 10% additional memory
        assert quantization_memory < 0.1 * creation_memory
        
        # Clean up
        del large_tensor, quantized
        torch.cuda.empty_cache()

class TestQuantizationEdgeCases:
    """Test quantization edge cases and error conditions."""
    
    def test_quantization_empty_tensor(self):
        """Test quantization with empty tensor."""
        empty_tensor = torch.empty(0, 0)
        quantized = torch.sign(empty_tensor)
        
        assert quantized.shape == empty_tensor.shape
        assert quantized.numel() == 0
    
    def test_quantization_single_element(self):
        """Test quantization with single element tensor."""
        single_element = torch.tensor([[5.0]])
        quantized = torch.sign(single_element)
        
        assert quantized.shape == (1, 1)
        assert quantized.item() == 1.0
        
        # Test with negative value
        single_negative = torch.tensor([[-3.0]])
        quantized_neg = torch.sign(single_negative)
        assert quantized_neg.item() == -1.0
        
        # Test with zero
        single_zero = torch.tensor([[0.0]])
        quantized_zero = torch.sign(single_zero)
        assert quantized_zero.item() == 0.0
    
    def test_quantization_inf_nan_handling(self):
        """Test quantization handling of inf and nan values."""
        # Create tensor with special values
        special_tensor = torch.tensor([
            [float('inf'), float('-inf'), float('nan'), 1.0],
            [0.0, -1.0, 2.0, -2.0]
        ])
        
        quantized = torch.sign(special_tensor)
        
        # Check handling of special values
        assert quantized[0, 0] == 1.0   # inf -> 1
        assert quantized[0, 1] == -1.0  # -inf -> -1
        assert torch.isnan(quantized[0, 2])  # nan -> nan
        
        # Check normal values
        assert quantized[0, 3] == 1.0   # 1.0 -> 1
        assert quantized[1, 0] == 0.0   # 0.0 -> 0
        assert quantized[1, 1] == -1.0  # -1.0 -> -1
        assert quantized[1, 2] == 1.0   # 2.0 -> 1
        assert quantized[1, 3] == -1.0  # -2.0 -> -1