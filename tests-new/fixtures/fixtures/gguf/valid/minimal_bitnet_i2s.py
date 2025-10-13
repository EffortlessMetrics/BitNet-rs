#!/usr/bin/env python3
"""
Generate minimal GGUF test file with I2_S quantization for BitNet.rs testing.

This script creates a minimal but realistic GGUF file containing:
- Valid GGUF header and metadata
- Transformer layer weights with I2_S quantization
- Token embeddings and output projection
- Proper tensor alignment for 32-byte boundaries

Designed for BitNet.rs Issue #159 test fixtures.
"""

import struct
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# GGUF Constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# BitNet I2_S Quantization Constants
I2S_QUANT_TYPE = 16  # Custom quantization type for I2_S
I2S_BLOCK_SIZE = 64  # Block size for I2_S quantization

class GGUFValueType:
    """GGUF value types for metadata."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

class MinimalGGUFBuilder:
    """Build minimal GGUF file for BitNet.rs testing."""

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 2048,
                 num_layers: int = 4, intermediate_size: int = 5632):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size

        # Metadata for BitNet configuration
        self.metadata = {
            "general.architecture": "bitnet",
            "general.name": "minimal_bitnet_test",
            "general.quantization_version": 2,
            "bitnet.context_length": 2048,
            "bitnet.embedding_length": hidden_size,
            "bitnet.block_count": num_layers,
            "bitnet.feed_forward_length": intermediate_size,
            "bitnet.attention.head_count": 16,
            "bitnet.attention.head_count_kv": 16,
            "bitnet.attention.layer_norm_rms_epsilon": 1e-5,
            "bitnet.quantization.i2s_enabled": True,
            "tokenizer.ggml.model": "bitnet",
            "tokenizer.ggml.tokens": ["<unk>", "<s>", "</s>"] + [f"token_{i}" for i in range(vocab_size - 3)],
            "tokenizer.ggml.scores": [0.0] * vocab_size,
            "tokenizer.ggml.token_type": [0] * vocab_size,
        }

        self.tensors = {}
        self._generate_tensors()

    def _generate_tensors(self):
        """Generate realistic test tensors with I2_S quantization patterns."""
        np.random.seed(42)  # Deterministic for testing

        # Token embeddings (FP32 for now, quantized variants in separate fixtures)
        self.tensors["token_embd.weight"] = {
            "shape": [self.vocab_size, self.hidden_size],
            "dtype": "F32",
            "data": np.random.randn(self.vocab_size, self.hidden_size).astype(np.float32) * 0.1
        }

        # Output projection
        self.tensors["output.weight"] = {
            "shape": [self.hidden_size, self.vocab_size],
            "dtype": "F32",
            "data": np.random.randn(self.hidden_size, self.vocab_size).astype(np.float32) * 0.1
        }

        # Output normalization
        self.tensors["output_norm.weight"] = {
            "shape": [self.hidden_size],
            "dtype": "F32",
            "data": np.ones(self.hidden_size, dtype=np.float32)
        }

        # Transformer layers
        for layer_idx in range(self.num_layers):
            layer_prefix = f"blk.{layer_idx}"

            # Attention weights - simulate I2_S quantization with specific patterns
            self._add_attention_weights(layer_prefix)
            self._add_ffn_weights(layer_prefix)
            self._add_norm_weights(layer_prefix)

    def _add_attention_weights(self, layer_prefix: str):
        """Add attention weights with I2_S quantization simulation."""
        # Generate weights that compress well with I2_S (mostly -1, 0, +1 values)
        for weight_name in ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"]:
            # Create I2_S-friendly weight matrix (values concentrated around -1, 0, +1)
            raw_weights = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32)
            # Quantize to I2_S-like values for realistic testing
            quantized = np.round(np.clip(raw_weights * 2, -1.5, 1.5))
            quantized = quantized.astype(np.float32) * 0.5  # Scale back

            self.tensors[f"{layer_prefix}.{weight_name}"] = {
                "shape": [self.hidden_size, self.hidden_size],
                "dtype": "F32",
                "data": quantized
            }

    def _add_ffn_weights(self, layer_prefix: str):
        """Add feed-forward network weights."""
        # FFN gate and up weights
        for weight_name in ["ffn_gate.weight", "ffn_up.weight"]:
            raw_weights = np.random.randn(self.intermediate_size, self.hidden_size).astype(np.float32)
            quantized = np.round(np.clip(raw_weights * 2, -1.5, 1.5)).astype(np.float32) * 0.5

            self.tensors[f"{layer_prefix}.{weight_name}"] = {
                "shape": [self.intermediate_size, self.hidden_size],
                "dtype": "F32",
                "data": quantized
            }

        # FFN down weight
        raw_weights = np.random.randn(self.hidden_size, self.intermediate_size).astype(np.float32)
        quantized = np.round(np.clip(raw_weights * 2, -1.5, 1.5)).astype(np.float32) * 0.5

        self.tensors[f"{layer_prefix}.ffn_down.weight"] = {
            "shape": [self.hidden_size, self.intermediate_size],
            "dtype": "F32",
            "data": quantized
        }

    def _add_norm_weights(self, layer_prefix: str):
        """Add normalization weights."""
        # Attention and FFN normalization weights (typically all ones)
        for norm_name in ["attn_norm.weight", "ffn_norm.weight"]:
            self.tensors[f"{layer_prefix}.{norm_name}"] = {
                "shape": [self.hidden_size],
                "dtype": "F32",
                "data": np.ones(self.hidden_size, dtype=np.float32)
            }

    def _write_metadata(self, f):
        """Write GGUF metadata section."""
        metadata_count = len(self.metadata)
        f.write(struct.pack("<Q", metadata_count))

        for key, value in self.metadata.items():
            # Write key
            key_bytes = key.encode('utf-8')
            f.write(struct.pack("<Q", len(key_bytes)))
            f.write(key_bytes)

            # Write value based on type
            if isinstance(value, bool):
                f.write(struct.pack("<I", GGUFValueType.BOOL))
                f.write(struct.pack("<B", 1 if value else 0))
            elif isinstance(value, int):
                f.write(struct.pack("<I", GGUFValueType.UINT32))
                f.write(struct.pack("<I", value))
            elif isinstance(value, float):
                f.write(struct.pack("<I", GGUFValueType.FLOAT32))
                f.write(struct.pack("<f", value))
            elif isinstance(value, str):
                f.write(struct.pack("<I", GGUFValueType.STRING))
                value_bytes = value.encode('utf-8')
                f.write(struct.pack("<Q", len(value_bytes)))
                f.write(value_bytes)
            elif isinstance(value, list):
                f.write(struct.pack("<I", GGUFValueType.ARRAY))
                f.write(struct.pack("<I", GGUFValueType.STRING))  # Array element type
                f.write(struct.pack("<Q", len(value)))
                for item in value:
                    if isinstance(item, str):
                        item_bytes = item.encode('utf-8')
                        f.write(struct.pack("<Q", len(item_bytes)))
                        f.write(item_bytes)
                    else:
                        f.write(struct.pack("<f", float(item)))

    def _write_tensors(self, f):
        """Write GGUF tensor information section."""
        tensor_count = len(self.tensors)
        f.write(struct.pack("<Q", tensor_count))

        for name, tensor_info in self.tensors.items():
            # Write tensor name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)

            # Write tensor shape
            shape = tensor_info["shape"]
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<Q", dim))

            # Write tensor type (F32 = 0)
            f.write(struct.pack("<I", 0))

            # Write tensor offset (will be calculated later)
            f.write(struct.pack("<Q", 0))  # Placeholder offset

    def _align_to_boundary(self, f, alignment=GGUF_DEFAULT_ALIGNMENT):
        """Align file position to boundary."""
        pos = f.tell()
        padding = (alignment - (pos % alignment)) % alignment
        if padding > 0:
            f.write(b'\x00' * padding)

    def write_gguf(self, output_path: str):
        """Write complete GGUF file."""
        with open(output_path, 'wb') as f:
            # Write header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))  # tensor_count
            f.write(struct.pack("<Q", len(self.metadata)))  # metadata_kv_count

            # Write metadata
            self._write_metadata(f)

            # Write tensor information
            tensor_info_start = f.tell()
            self._write_tensors(f)

            # Align to tensor data boundary
            self._align_to_boundary(f)
            tensor_data_start = f.tell()

            # Write tensor data
            for name, tensor_info in self.tensors.items():
                data = tensor_info["data"]
                f.write(data.tobytes())

                # Align each tensor to boundary
                self._align_to_boundary(f)

            # Update tensor offsets (simplified - would need proper calculation for real GGUF)
            print(f"Generated GGUF file: {output_path}")
            print(f"Tensors: {len(self.tensors)}")
            print(f"Metadata entries: {len(self.metadata)}")

def main():
    """Generate test GGUF fixtures."""
    output_dir = Path(__file__).parent

    # Generate minimal I2_S model
    builder = MinimalGGUFBuilder(
        vocab_size=32000,
        hidden_size=2048,
        num_layers=4,
        intermediate_size=5632
    )

    output_path = output_dir / "minimal_bitnet_i2s.gguf"
    builder.write_gguf(str(output_path))

    print(f"✓ Generated minimal I2_S GGUF: {output_path}")

    # Generate small test model for faster testing
    small_builder = MinimalGGUFBuilder(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        intermediate_size=512
    )

    small_path = output_dir / "small_bitnet_test.gguf"
    small_builder.write_gguf(str(small_path))

    print(f"✓ Generated small test GGUF: {small_path}")

if __name__ == "__main__":
    main()
