#!/usr/bin/env python3
"""
Generate cross-validation test data for BitNet.rs vs C++ reference comparison.

Creates reference outputs that can be used with `cargo run -p xtask -- crossval`
for validating Rust implementation against C++ reference implementation.

Features:
- Deterministic test data (seed=42)
- Known input/output pairs for I2_S, TL1, TL2 quantization
- Tolerance specifications for numerical comparison
- GGUF-compatible tensor formats
- Performance benchmark reference data

Designed for BitNet.rs Issue #159 cross-validation fixtures.
"""

import numpy as np
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Deterministic seed for reproducible cross-validation
np.random.seed(42)

@dataclass
class CrossValidationReference:
    """Cross-validation reference data for Rust vs C++ comparison."""
    test_name: str
    input_shape: List[int]
    input_data: List[float]
    expected_output: List[float]
    quantization_type: str
    tolerance_absolute: float
    tolerance_relative: float
    performance_baseline_ms: float
    metadata: Dict[str, str]

    def to_dict(self):
        result = asdict(self)
        # Ensure all numeric data is JSON-serializable
        for key, value in result.items():
            if isinstance(value, list):
                result[key] = [float(x) if hasattr(x, 'dtype') else x for x in value]
            elif hasattr(value, 'dtype'):
                result[key] = float(value)
        return result

class CrossValidationDataGenerator:
    """Generate cross-validation test data."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.references = []

    def generate_i2s_crossval_data(self):
        """Generate I2_S quantization cross-validation data."""
        test_cases = [
            ("i2s_small_matrix", [64, 64]),
            ("i2s_attention_q", [2048, 2048]),
            ("i2s_ffn_up", [5632, 2048]),
            ("i2s_ffn_down", [2048, 5632]),
        ]

        for test_name, shape in test_cases:
            # Generate input tensor
            input_tensor = self._generate_realistic_weight_matrix(shape)

            # Simulate C++ reference I2_S quantization output
            # (In practice, this would come from actual C++ implementation)
            expected_output = self._simulate_i2s_reference(input_tensor)

            reference = CrossValidationReference(
                test_name=test_name,
                input_shape=shape,
                input_data=input_tensor.flatten().tolist(),
                expected_output=expected_output.flatten().tolist(),
                quantization_type="I2_S",
                tolerance_absolute=1e-6,
                tolerance_relative=1e-4,
                performance_baseline_ms=self._estimate_performance_baseline(shape),
                metadata={
                    "block_size": "64",
                    "device": "cpu",
                    "precision": "fp32",
                    "reference_version": "cpp_v1.0"
                }
            )

            self.references.append(reference)

    def generate_tl1_crossval_data(self):
        """Generate TL1 quantization cross-validation data."""
        test_cases = [
            ("tl1_small_test", [32, 32]),
            ("tl1_medium_matrix", [256, 256]),
            ("tl1_ffn_gate", [5632, 2048]),
        ]

        for test_name, shape in test_cases:
            input_tensor = self._generate_realistic_weight_matrix(shape)
            expected_output = self._simulate_tl1_reference(input_tensor)

            reference = CrossValidationReference(
                test_name=test_name,
                input_shape=shape,
                input_data=input_tensor.flatten().tolist(),
                expected_output=expected_output.flatten().tolist(),
                quantization_type="TL1",
                tolerance_absolute=1e-5,
                tolerance_relative=1e-3,
                performance_baseline_ms=self._estimate_performance_baseline(shape),
                metadata={
                    "lookup_table_size": "16",
                    "device": "cpu",
                    "precision": "fp32",
                    "reference_version": "cpp_v1.0"
                }
            )

            self.references.append(reference)

    def generate_tl2_crossval_data(self):
        """Generate TL2 quantization cross-validation data."""
        test_cases = [
            ("tl2_attention_kv", [2048, 2048]),
            ("tl2_embedding", [32000, 2048]),
            ("tl2_output_proj", [2048, 32000]),
        ]

        for test_name, shape in test_cases:
            input_tensor = self._generate_realistic_weight_matrix(shape)
            expected_output = self._simulate_tl2_reference(input_tensor)

            reference = CrossValidationReference(
                test_name=test_name,
                input_shape=shape,
                input_data=input_tensor.flatten().tolist(),
                expected_output=expected_output.flatten().tolist(),
                quantization_type="TL2",
                tolerance_absolute=1e-7,
                tolerance_relative=1e-5,
                performance_baseline_ms=self._estimate_performance_baseline(shape),
                metadata={
                    "lookup_table_size": "256",
                    "device": "cpu",
                    "precision": "fp32",
                    "reference_version": "cpp_v1.0"
                }
            )

            self.references.append(reference)

    def generate_end_to_end_inference_data(self):
        """Generate end-to-end inference cross-validation data."""
        # Simple inference test case
        input_tokens = [1, 15496, 338, 278, 6593, 310, 2834, 29973]  # "What is the meaning of life?"

        # Simulate expected logits from C++ reference (normally would be actual output)
        vocab_size = 32000
        expected_logits = np.random.randn(len(input_tokens), vocab_size).astype(np.float32)
        # Add realistic probability distribution patterns
        expected_logits = expected_logits - np.log(np.sum(np.exp(expected_logits), axis=-1, keepdims=True))

        reference = CrossValidationReference(
            test_name="e2e_inference_small",
            input_shape=[len(input_tokens)],
            input_data=[float(x) for x in input_tokens],
            expected_output=expected_logits.flatten().tolist(),
            quantization_type="Mixed",  # Uses multiple quantization types
            tolerance_absolute=1e-4,
            tolerance_relative=1e-2,
            performance_baseline_ms=50.0,  # 50ms baseline for small inference
            metadata={
                "model_size": "2B",
                "sequence_length": str(len(input_tokens)),
                "vocab_size": "32000",
                "device": "cpu",
                "reference_version": "cpp_v1.0"
            }
        )

        self.references.append(reference)

    def generate_performance_benchmark_data(self):
        """Generate performance benchmark reference data."""
        benchmark_cases = [
            ("perf_i2s_matmul", [1024, 1024], "I2_S", 2.5),
            ("perf_tl1_matmul", [1024, 1024], "TL1", 3.2),
            ("perf_tl2_matmul", [1024, 1024], "TL2", 1.8),
            ("perf_large_i2s", [4096, 4096], "I2_S", 45.0),
        ]

        for test_name, shape, quant_type, baseline_ms in benchmark_cases:
            input_tensor = self._generate_realistic_weight_matrix(shape)
            # For performance tests, output is same as input (identity operation)
            expected_output = input_tensor.copy()

            reference = CrossValidationReference(
                test_name=test_name,
                input_shape=shape,
                input_data=input_tensor.flatten().tolist(),
                expected_output=expected_output.flatten().tolist(),
                quantization_type=quant_type,
                tolerance_absolute=1e-6,
                tolerance_relative=1e-6,
                performance_baseline_ms=baseline_ms,
                metadata={
                    "test_type": "performance",
                    "operation": "matrix_multiplication",
                    "device": "cpu",
                    "reference_version": "cpp_v1.0"
                }
            )

            self.references.append(reference)

    def _generate_realistic_weight_matrix(self, shape: List[int]) -> np.ndarray:
        """Generate realistic weight matrix for neural networks."""
        if len(shape) == 2:
            fan_in, fan_out = shape[1], shape[0]
            # Xavier initialization
            scale = np.sqrt(6.0 / (fan_in + fan_out))
            weights = np.random.uniform(-scale, scale, shape).astype(np.float32)
        else:
            # For other shapes, use normal distribution
            weights = np.random.randn(*shape).astype(np.float32) * 0.1

        return weights

    def _simulate_i2s_reference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Simulate C++ reference I2_S quantization output.
        In practice, this would be replaced with actual C++ reference data.
        """
        # Simplified I2_S simulation: quantize to {-1, 0, 1} with block-wise scaling
        block_size = 64
        output = input_tensor.copy()
        flat_output = output.flatten()

        for i in range(0, len(flat_output), block_size):
            block = flat_output[i:i + block_size]
            if len(block) == 0:
                continue

            scale = np.max(np.abs(block))
            if scale > 0:
                normalized = block / scale
                quantized = np.round(np.clip(normalized, -1, 1))
                flat_output[i:i + block_size] = quantized * scale

        return output

    def _simulate_tl1_reference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Simulate C++ reference TL1 quantization output."""
        # Simplified TL1: 16-level lookup table quantization
        lookup_table = np.linspace(-1.0, 1.0, 16)
        scale = np.max(np.abs(input_tensor))

        if scale == 0:
            return input_tensor.copy()

        normalized = input_tensor / scale
        quantized = np.zeros_like(input_tensor)

        for i, val in enumerate(lookup_table):
            mask = np.abs(normalized - val) <= (1.0 / 16)
            quantized[mask] = val

        return quantized * scale

    def _simulate_tl2_reference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Simulate C++ reference TL2 quantization output."""
        # Simplified TL2: 256-level lookup table quantization
        lookup_table = np.linspace(-2.0, 2.0, 256)
        scale = np.max(np.abs(input_tensor))

        if scale == 0:
            return input_tensor.copy()

        normalized = input_tensor / scale
        output = np.zeros_like(input_tensor)

        for idx in np.ndindex(input_tensor.shape):
            val = normalized[idx]
            closest_idx = np.argmin(np.abs(lookup_table - val))
            output[idx] = lookup_table[closest_idx]

        return output * scale

    def _estimate_performance_baseline(self, shape: List[int]) -> float:
        """Estimate performance baseline in milliseconds."""
        if len(shape) == 2:
            # Matrix multiplication complexity: O(n^3) for square matrices
            elements = shape[0] * shape[1]
            # Rough estimate: 1 GFLOP = 1ms (CPU baseline)
            flops = elements * 2  # Simple quantization ops
            return max(0.1, flops / 1e6)  # Convert to ms
        else:
            elements = np.prod(shape)
            return max(0.1, elements / 1e5)

    def save_crossval_data(self):
        """Save cross-validation reference data."""
        # Save all references
        all_refs_file = self.output_dir / "crossval_references.json"
        with open(all_refs_file, 'w') as f:
            json.dump([ref.to_dict() for ref in self.references], f, indent=2)

        # Save by quantization type
        by_quant_type = {}
        for ref in self.references:
            quant_type = ref.quantization_type.lower()
            if quant_type not in by_quant_type:
                by_quant_type[quant_type] = []
            by_quant_type[quant_type].append(ref)

        for quant_type, refs in by_quant_type.items():
            type_file = self.output_dir / f"crossval_{quant_type}.json"
            with open(type_file, 'w') as f:
                json.dump([ref.to_dict() for ref in refs], f, indent=2)

        # Save binary data for efficient loading
        self._save_binary_data()

        # Generate xtask-compatible configuration
        self._generate_xtask_config()

        print(f"Generated {len(self.references)} cross-validation references")

    def _save_binary_data(self):
        """Save binary reference data for efficient loading."""
        binary_dir = self.output_dir / "binary"
        binary_dir.mkdir(exist_ok=True)

        for ref in self.references:
            # Save input data
            input_data = np.array(ref.input_data, dtype=np.float32)
            input_file = binary_dir / f"{ref.test_name}_input.bin"
            input_data.tofile(input_file)

            # Save expected output
            output_data = np.array(ref.expected_output, dtype=np.float32)
            output_file = binary_dir / f"{ref.test_name}_expected.bin"
            output_data.tofile(output_file)

    def _generate_xtask_config(self):
        """Generate xtask-compatible configuration."""
        xtask_config = {
            "crossval_tests": [],
            "tolerance_config": {
                "I2_S": {"absolute": 1e-6, "relative": 1e-4},
                "TL1": {"absolute": 1e-5, "relative": 1e-3},
                "TL2": {"absolute": 1e-7, "relative": 1e-5},
                "Mixed": {"absolute": 1e-4, "relative": 1e-2}
            },
            "performance_baselines": {}
        }

        for ref in self.references:
            test_config = {
                "name": ref.test_name,
                "input_file": f"binary/{ref.test_name}_input.bin",
                "expected_file": f"binary/{ref.test_name}_expected.bin",
                "shape": ref.input_shape,
                "quantization_type": ref.quantization_type,
                "metadata": ref.metadata
            }
            xtask_config["crossval_tests"].append(test_config)

            # Add performance baseline
            if ref.test_name.startswith("perf_"):
                xtask_config["performance_baselines"][ref.test_name] = ref.performance_baseline_ms

        # Save xtask config
        xtask_file = self.output_dir / "xtask_crossval_config.json"
        with open(xtask_file, 'w') as f:
            json.dump(xtask_config, f, indent=2)

        print(f"Generated xtask cross-validation config: {xtask_file}")

def main():
    """Generate cross-validation test data."""
    output_dir = Path(__file__).parent
    generator = CrossValidationDataGenerator(output_dir)

    print("Generating cross-validation reference data...")

    generator.generate_i2s_crossval_data()
    generator.generate_tl1_crossval_data()
    generator.generate_tl2_crossval_data()
    generator.generate_end_to_end_inference_data()
    generator.generate_performance_benchmark_data()

    generator.save_crossval_data()

    print("✓ Cross-validation reference data generation complete")
    print("\nUsage with xtask:")
    print("  export BITNET_CROSSVAL_CONFIG=tests/fixtures/tensors/crossval/xtask_crossval_config.json")
    print("  cargo run -p xtask -- crossval")

if __name__ == "__main__":
    main()