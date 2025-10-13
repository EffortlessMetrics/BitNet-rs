#!/usr/bin/env python3
"""
Generate corrupted GGUF files for error handling tests.

Creates various types of corrupted GGUF files to test BitNet.rs error handling:
- Invalid headers
- Misaligned tensor data
- Incomplete files
- Invalid metadata

Designed for BitNet.rs Issue #159 test fixtures.
"""

import struct
import os
from pathlib import Path

def create_invalid_magic():
    """Create GGUF with invalid magic number."""
    with open("invalid_magic.gguf", "wb") as f:
        f.write(b"XXXX")  # Invalid magic
        f.write(struct.pack("<I", 3))  # Version
        f.write(struct.pack("<Q", 0))  # tensor_count
        f.write(struct.pack("<Q", 0))  # metadata_count

def create_invalid_version():
    """Create GGUF with unsupported version."""
    with open("invalid_version.gguf", "wb") as f:
        f.write(struct.pack("<I", 0x46554747))  # Valid magic "GGUF"
        f.write(struct.pack("<I", 999))  # Invalid version
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", 0))

def create_truncated_header():
    """Create GGUF with truncated header."""
    with open("truncated_header.gguf", "wb") as f:
        f.write(struct.pack("<I", 0x46554747))  # Valid magic
        f.write(b"XX")  # Truncated version

def create_invalid_tensor_count():
    """Create GGUF with invalid tensor count."""
    with open("invalid_tensor_count.gguf", "wb") as f:
        f.write(struct.pack("<I", 0x46554747))  # Valid magic
        f.write(struct.pack("<I", 3))  # Version
        f.write(struct.pack("<Q", 0xFFFFFFFFFFFFFFFF))  # Invalid tensor count
        f.write(struct.pack("<Q", 0))  # metadata_count

def create_misaligned_tensor_data():
    """Create GGUF with misaligned tensor data."""
    with open("misaligned_tensors.gguf", "wb") as f:
        # Valid header
        f.write(struct.pack("<I", 0x46554747))
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))  # tensor_count
        f.write(struct.pack("<Q", 0))  # metadata_count

        # Single tensor info
        name = "test.weight"
        name_bytes = name.encode('utf-8')
        f.write(struct.pack("<Q", len(name_bytes)))
        f.write(name_bytes)

        # Tensor shape
        f.write(struct.pack("<I", 2))  # n_dimensions
        f.write(struct.pack("<Q", 10))  # dim 0
        f.write(struct.pack("<Q", 10))  # dim 1

        f.write(struct.pack("<I", 0))  # type (F32)
        f.write(struct.pack("<Q", 100))  # Invalid offset

        # Write some data at wrong offset
        f.write(b"invalid_tensor_data_not_aligned")

def create_incomplete_file():
    """Create GGUF file that cuts off unexpectedly."""
    with open("incomplete.gguf", "wb") as f:
        f.write(struct.pack("<I", 0x46554747))  # Valid magic
        f.write(struct.pack("<I", 3))  # Version
        f.write(struct.pack("<Q", 5))  # tensor_count
        f.write(struct.pack("<Q", 2))  # metadata_count

        # Start metadata but cut off
        f.write(b"partial_metadata")

def create_invalid_metadata():
    """Create GGUF with invalid metadata structure."""
    with open("invalid_metadata.gguf", "wb") as f:
        # Valid header
        f.write(struct.pack("<I", 0x46554747))
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 0))  # tensor_count
        f.write(struct.pack("<Q", 1))  # metadata_count

        # Invalid metadata entry
        f.write(struct.pack("<Q", 0))  # Key length 0 (invalid)
        f.write(struct.pack("<I", 999))  # Invalid value type

def create_zero_byte_file():
    """Create completely empty file."""
    with open("zero_byte.gguf", "wb") as f:
        pass  # Empty file

def create_random_data():
    """Create file with random binary data."""
    import random
    with open("random_data.gguf", "wb") as f:
        # Write 1KB of random data
        for _ in range(1024):
            f.write(bytes([random.randint(0, 255)]))

def create_missing_tensors():
    """Create GGUF that claims to have tensors but has incomplete tensor info."""
    with open("missing_tensors.gguf", "wb") as f:
        # Valid header claiming 3 tensors
        f.write(struct.pack("<I", 0x46554747))
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 3))  # tensor_count = 3
        f.write(struct.pack("<Q", 0))  # metadata_count

        # Only provide info for 1 tensor (missing 2)
        name = "partial.weight"
        name_bytes = name.encode('utf-8')
        f.write(struct.pack("<Q", len(name_bytes)))
        f.write(name_bytes)

        f.write(struct.pack("<I", 1))  # n_dimensions
        f.write(struct.pack("<Q", 100))  # dim 0
        f.write(struct.pack("<I", 0))  # type (F32)
        f.write(struct.pack("<Q", 0))  # offset

        # Missing tensor info for the other 2 tensors

def main():
    """Generate all corrupted GGUF test files."""
    output_dir = Path(__file__).parent
    os.chdir(output_dir)

    print("Generating corrupted GGUF files for error testing...")

    test_functions = [
        ("invalid_magic", create_invalid_magic),
        ("invalid_version", create_invalid_version),
        ("truncated_header", create_truncated_header),
        ("invalid_tensor_count", create_invalid_tensor_count),
        ("misaligned_tensors", create_misaligned_tensor_data),
        ("incomplete", create_incomplete_file),
        ("invalid_metadata", create_invalid_metadata),
        ("zero_byte", create_zero_byte_file),
        ("random_data", create_random_data),
        ("missing_tensors", create_missing_tensors),
    ]

    for test_name, test_func in test_functions:
        try:
            test_func()
            print(f"✓ Generated {test_name}.gguf")
        except Exception as e:
            print(f"✗ Failed to generate {test_name}.gguf: {e}")

    print(f"Generated {len(test_functions)} corrupted GGUF files in {output_dir}")

if __name__ == "__main__":
    main()
