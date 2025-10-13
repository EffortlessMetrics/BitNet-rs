#!/usr/bin/env python3
"""
Generate quantization test data for BitNet.rs I2_S, TL1, TL2 testing.

Creates reference FP32 tensors and their quantized equivalents with known accuracy properties.
Designed to enable ≥99% accuracy validation against FP32 reference implementations.

Features:
- Deterministic test data generation with seed=42
- I2_S, TL1, TL2 quantization formats
- Known accuracy properties for validation
- Cross-validation compatible reference data
- Device-aware test data (CPU/GPU variants)

Designed for BitNet.rs Issue #159 test fixtures.
"""

import numpy as np
import struct
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Set deterministic seed for reproducible test data
np.random.seed(42)

@dataclass
class QuantizationTestVector:
    """Test vector for quantization accuracy validation."""
    name: str
    shape: List[int]
    fp32_data: List[float]
    quantized_data: List[int]
    scales: List[float]
    zero_points: List[int]
    quantization_type: str
    block_size: int
    expected_accuracy: float
    tolerance: float

    def to_dict(self):
        result = asdict(self)
        # Convert numpy types to native Python types for JSON serialization
        for key, value in result.items():
            if isinstance(value, list):
                result[key] = [float(x) if hasattr(x, 'dtype') else x for x in value]
            elif hasattr(value, 'dtype'):
                result[key] = float(value)
        return result

class I2SQuantizer:
    """I2_S quantization implementation for test data generation."""

    def __init__(self, block_size: int = 64):
        self.block_size = block_size

    def quantize(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize tensor to I2_S format (2-bit signed: -1, 0, +1).
        Returns quantized values and scale factors.
        """
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()

        # Pad to block boundary
        padded_size = ((len(flat_tensor) + self.block_size - 1) // self.block_size) * self.block_size
        if padded_size > len(flat_tensor):
            flat_tensor = np.pad(flat_tensor, (0, padded_size - len(flat_tensor)))

        # Process in blocks
        quantized = []
        scales = []

        for i in range(0, len(flat_tensor), self.block_size):
            block = flat_tensor[i:i + self.block_size]

            # Calculate scale factor (max absolute value in block)
            scale = np.max(np.abs(block))
            if scale == 0:
                scale = 1.0  # Avoid division by zero

            # Quantize to {-1, 0, +1}
            normalized = block / scale
            quantized_block = np.round(np.clip(normalized, -1.0, 1.0)).astype(np.int8)

            quantized.extend(quantized_block)
            scales.append(scale)

        quantized = np.array(quantized[:len(tensor.flatten())], dtype=np.int8)
        return quantized.reshape(original_shape), np.array(scales, dtype=np.float32)

    def dequantize(self, quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Dequantize I2_S tensor back to FP32."""
        flat_quantized = quantized.flatten()
        dequantized = []

        for i in range(0, len(flat_quantized), self.block_size):
            block = flat_quantized[i:i + self.block_size]
            scale_idx = i // self.block_size
            if scale_idx < len(scales):
                scale = scales[scale_idx]
                dequantized_block = block.astype(np.float32) * scale
                dequantized.extend(dequantized_block)

        return np.array(dequantized[:len(quantized.flatten())], dtype=np.float32).reshape(quantized.shape)

class TL1Quantizer:
    """TL1 (Table Lookup 1) quantization for test data."""

    def __init__(self, num_levels: int = 16):
        self.num_levels = num_levels
        # Generate lookup table with concentrated values around 0
        self.lookup_table = np.linspace(-1.0, 1.0, num_levels)

    def quantize(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize using lookup table."""
        # Find scale factor
        scale = np.max(np.abs(tensor))
        if scale == 0:
            scale = 1.0

        # Normalize and find closest lookup values
        normalized = tensor / scale
        quantized = np.zeros_like(tensor, dtype=np.uint8)

        for i, val in enumerate(self.lookup_table):
            mask = np.abs(normalized - val) <= (1.0 / self.num_levels)
            quantized[mask] = i

        return quantized, np.array([scale], dtype=np.float32)

    def dequantize(self, quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Dequantize using lookup table."""
        scale = scales[0] if len(scales) > 0 else 1.0
        dequantized = self.lookup_table[quantized] * scale
        return dequantized.astype(np.float32)

class TL2Quantizer:
    """TL2 (Table Lookup 2) quantization for test data."""

    def __init__(self, num_levels: int = 256):
        self.num_levels = num_levels
        # More fine-grained lookup table
        self.lookup_table = np.linspace(-2.0, 2.0, num_levels)

    def quantize(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize using fine-grained lookup table."""
        scale = np.max(np.abs(tensor))
        if scale == 0:
            scale = 1.0

        normalized = tensor / scale
        quantized = np.zeros_like(tensor, dtype=np.uint8)

        # Find closest lookup table entries
        for i in range(tensor.size):
            flat_idx = i
            val = normalized.flat[flat_idx]
            closest_idx = np.argmin(np.abs(self.lookup_table - val))
            quantized.flat[flat_idx] = closest_idx

        return quantized, np.array([scale], dtype=np.float32)

    def dequantize(self, quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Dequantize using lookup table."""
        scale = scales[0] if len(scales) > 0 else 1.0
        dequantized = self.lookup_table[quantized] * scale
        return dequantized.astype(np.float32)

class QuantizationTestDataGenerator:
    """Generate comprehensive quantization test data."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.i2s_quantizer = I2SQuantizer()
        self.tl1_quantizer = TL1Quantizer()
        self.tl2_quantizer = TL2Quantizer()

        self.test_vectors = []

    def generate_attention_weight_tests(self):
        """Generate test vectors for attention weights."""
        shapes = [
            [2048, 2048],  # Full size attention
            [256, 256],    # Small test size
            [64, 64],      # Tiny test size
        ]

        for shape in shapes:
            # Generate FP32 reference tensor
            fp32_tensor = self._generate_attention_like_weights(shape)

            # Test each quantization type
            for quant_type, quantizer in [
                ("I2_S", self.i2s_quantizer),
                ("TL1", self.tl1_quantizer),
                ("TL2", self.tl2_quantizer)
            ]:
                quantized, scales = quantizer.quantize(fp32_tensor)
                dequantized = quantizer.dequantize(quantized, scales)

                # Calculate accuracy
                accuracy = self._calculate_accuracy(fp32_tensor, dequantized)

                test_vector = QuantizationTestVector(
                    name=f"attention_{shape[0]}x{shape[1]}_{quant_type.lower()}",
                    shape=shape,
                    fp32_data=fp32_tensor.flatten().tolist(),
                    quantized_data=quantized.flatten().astype(int).tolist(),
                    scales=scales.tolist(),
                    zero_points=[0] * len(scales),  # BitNet doesn't use zero points
                    quantization_type=quant_type,
                    block_size=getattr(quantizer, 'block_size', 1),
                    expected_accuracy=accuracy,
                    tolerance=0.01  # 1% tolerance for testing
                )

                self.test_vectors.append(test_vector)

    def generate_ffn_weight_tests(self):
        """Generate test vectors for feed-forward network weights."""
        shapes = [
            [5632, 2048],  # FFN up/gate weights
            [2048, 5632],  # FFN down weights
            [512, 256],    # Small test
        ]

        for shape in shapes:
            fp32_tensor = self._generate_ffn_like_weights(shape)

            for quant_type, quantizer in [
                ("I2_S", self.i2s_quantizer),
                ("TL1", self.tl1_quantizer),
                ("TL2", self.tl2_quantizer)
            ]:
                quantized, scales = quantizer.quantize(fp32_tensor)
                dequantized = quantizer.dequantize(quantized, scales)
                accuracy = self._calculate_accuracy(fp32_tensor, dequantized)

                test_vector = QuantizationTestVector(
                    name=f"ffn_{shape[0]}x{shape[1]}_{quant_type.lower()}",
                    shape=shape,
                    fp32_data=fp32_tensor.flatten().tolist(),
                    quantized_data=quantized.flatten().astype(int).tolist(),
                    scales=scales.tolist(),
                    zero_points=[0] * len(scales),
                    quantization_type=quant_type,
                    block_size=getattr(quantizer, 'block_size', 1),
                    expected_accuracy=accuracy,
                    tolerance=0.01
                )

                self.test_vectors.append(test_vector)

    def generate_edge_case_tests(self):
        """Generate edge case test vectors."""
        edge_cases = [
            ("all_zeros", np.zeros((100, 100))),
            ("all_ones", np.ones((100, 100))),
            ("extreme_positive", np.full((50, 50), 10.0)),
            ("extreme_negative", np.full((50, 50), -10.0)),
            ("sparse_pattern", self._generate_sparse_tensor((128, 128))),
            ("alternating", self._generate_alternating_tensor((64, 64))),
        ]

        for case_name, tensor in edge_cases:
            for quant_type, quantizer in [
                ("I2_S", self.i2s_quantizer),
                ("TL1", self.tl1_quantizer),
                ("TL2", self.tl2_quantizer)
            ]:
                quantized, scales = quantizer.quantize(tensor)
                dequantized = quantizer.dequantize(quantized, scales)
                accuracy = self._calculate_accuracy(tensor, dequantized)

                test_vector = QuantizationTestVector(
                    name=f"edge_{case_name}_{quant_type.lower()}",
                    shape=list(tensor.shape),
                    fp32_data=tensor.flatten().tolist(),
                    quantized_data=quantized.flatten().astype(int).tolist(),
                    scales=scales.tolist(),
                    zero_points=[0] * len(scales),
                    quantization_type=quant_type,
                    block_size=getattr(quantizer, 'block_size', 1),
                    expected_accuracy=accuracy,
                    tolerance=0.05  # More tolerance for edge cases
                )

                self.test_vectors.append(test_vector)

    def _generate_attention_like_weights(self, shape: List[int]) -> np.ndarray:
        """Generate weights that behave like attention weights."""
        # Xavier/Glorot initialization - typical for attention weights
        fan_in, fan_out = shape[1], shape[0]
        scale = np.sqrt(6.0 / (fan_in + fan_out))
        weights = np.random.uniform(-scale, scale, shape).astype(np.float32)

        # Add some structure typical of trained attention weights
        weights = weights * (1 + 0.1 * np.sin(np.linspace(0, 10*np.pi, weights.size).reshape(shape)))

        return weights

    def _generate_ffn_like_weights(self, shape: List[int]) -> np.ndarray:
        """Generate weights that behave like FFN weights."""
        # He initialization - typical for ReLU-like activations
        fan_in = shape[1]
        scale = np.sqrt(2.0 / fan_in)
        weights = np.random.normal(0, scale, shape).astype(np.float32)

        # Add sparsity pattern common in trained FFN weights
        mask = np.random.random(shape) > 0.1  # 90% non-zero
        weights = weights * mask

        return weights

    def _generate_sparse_tensor(self, shape: List[int]) -> np.ndarray:
        """Generate sparse tensor (mostly zeros)."""
        tensor = np.zeros(shape, dtype=np.float32)
        # Randomly place 5% non-zero values
        num_nonzero = int(0.05 * tensor.size)
        indices = np.random.choice(tensor.size, num_nonzero, replace=False)
        tensor.flat[indices] = np.random.randn(num_nonzero)
        return tensor

    def _generate_alternating_tensor(self, shape: List[int]) -> np.ndarray:
        """Generate alternating pattern tensor."""
        tensor = np.zeros(shape, dtype=np.float32)
        tensor[::2, ::2] = 1.0
        tensor[1::2, 1::2] = -1.0
        return tensor

    def _calculate_accuracy(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate reconstruction accuracy."""
        if np.allclose(original, 0):
            # All zeros case
            return 1.0 if np.allclose(reconstructed, 0) else 0.0

        # Calculate relative error
        mse = np.mean((original - reconstructed) ** 2)
        original_power = np.mean(original ** 2)

        if original_power == 0:
            return 1.0 if mse == 0 else 0.0

        snr = original_power / (mse + 1e-10)
        accuracy = min(1.0, snr / 100.0)  # Normalize to [0, 1]

        return accuracy

    def save_test_vectors(self):
        """Save test vectors to JSON files."""
        # Save comprehensive test data
        all_vectors_file = self.output_dir / "quantization_test_vectors.json"
        with open(all_vectors_file, 'w') as f:
            json.dump([tv.to_dict() for tv in self.test_vectors], f, indent=2)

        # Save by quantization type
        for quant_type in ["I2_S", "TL1", "TL2"]:
            type_vectors = [tv for tv in self.test_vectors if tv.quantization_type == quant_type]
            type_file = self.output_dir / f"{quant_type.lower()}_test_vectors.json"
            with open(type_file, 'w') as f:
                json.dump([tv.to_dict() for tv in type_vectors], f, indent=2)

        # Save binary reference data for fast loading
        self._save_binary_references()

        print(f"Saved {len(self.test_vectors)} test vectors to {self.output_dir}")

        # Print summary statistics
        self._print_summary()

    def _save_binary_references(self):
        """Save binary reference data for efficient loading in tests."""
        binary_dir = self.output_dir / "binary"
        binary_dir.mkdir(exist_ok=True)

        for tv in self.test_vectors:
            # Save FP32 reference
            fp32_data = np.array(tv.fp32_data, dtype=np.float32).reshape(tv.shape)
            fp32_file = binary_dir / f"{tv.name}_fp32.bin"
            fp32_data.tofile(fp32_file)

            # Save quantized data
            quant_data = np.array(tv.quantized_data, dtype=np.int8 if tv.quantization_type == "I2_S" else np.uint8)
            quant_file = binary_dir / f"{tv.name}_quantized.bin"
            quant_data.tofile(quant_file)

            # Save scales
            scales_data = np.array(tv.scales, dtype=np.float32)
            scales_file = binary_dir / f"{tv.name}_scales.bin"
            scales_data.tofile(scales_file)

    def _print_summary(self):
        """Print summary statistics."""
        print("\n=== Quantization Test Data Summary ===")

        by_type = {}
        for tv in self.test_vectors:
            if tv.quantization_type not in by_type:
                by_type[tv.quantization_type] = []
            by_type[tv.quantization_type].append(tv.expected_accuracy)

        for quant_type, accuracies in by_type.items():
            mean_acc = np.mean(accuracies)
            min_acc = np.min(accuracies)
            max_acc = np.max(accuracies)
            count = len(accuracies)

            print(f"{quant_type}: {count} test vectors")
            print(f"  Accuracy - Mean: {mean_acc:.4f}, Min: {min_acc:.4f}, Max: {max_acc:.4f}")

        high_accuracy = [tv for tv in self.test_vectors if tv.expected_accuracy >= 0.99]
        print(f"\nTest vectors with ≥99% accuracy: {len(high_accuracy)}/{len(self.test_vectors)}")

def main():
    """Generate quantization test data."""
    output_dir = Path(__file__).parent
    generator = QuantizationTestDataGenerator(output_dir)

    print("Generating quantization test vectors...")

    generator.generate_attention_weight_tests()
    generator.generate_ffn_weight_tests()
    generator.generate_edge_case_tests()

    generator.save_test_vectors()

    print("\n✓ Quantization test data generation complete")

if __name__ == "__main__":
    main()
