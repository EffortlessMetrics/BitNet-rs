#!/usr/bin/env python3
"""
Generate minimal GGUF test fixtures for BitNet.rs tokenizer tests.

GGUF v3 format:
- Magic: "GGUF" (4 bytes)
- Version: 3 (u32 LE)
- Tensor count: 0 (u64 LE)
- Metadata KV count: N (u64 LE)
- KV pairs (no alignment field for early v3 variant)
"""

import struct
import sys
from pathlib import Path

def write_u32(f, val):
    f.write(struct.pack('<I', val))

def write_u64(f, val):
    f.write(struct.pack('<Q', val))

def write_string(f, s):
    """Write GGUF string: u64 length + bytes"""
    write_u64(f, len(s))
    f.write(s.encode('utf-8'))

def write_kv_u32(f, key, value):
    """Write metadata KV pair with u32 value"""
    write_string(f, key)
    write_u32(f, 4)  # Type = u32 (4, not 5 which is i32)
    write_u32(f, value)

def write_kv_string(f, key, value):
    """Write metadata KV pair with string value"""
    write_string(f, key)
    write_u32(f, 8)  # Type = string
    write_string(f, value)

def write_kv_string_array(f, key, values):
    """Write metadata KV pair with string array value"""
    write_string(f, key)
    write_u32(f, 9)  # Type = array
    write_u32(f, 8)  # Element type = string
    write_u64(f, len(values))
    for val in values:
        write_string(f, val)

def create_llama3_128k_gguf(output_path):
    """Create LLaMA-3 128K vocab GGUF fixture"""
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'GGUF')
        write_u32(f, 3)  # Version 3
        write_u64(f, 0)  # Tensor count
        write_u64(f, 5)  # Metadata KV count

        # Metadata
        write_kv_string(f, "general.architecture", "llama")
        write_kv_u32(f, "llama.vocab_size", 128256)
        write_kv_string(f, "tokenizer.ggml.model", "hf")
        write_kv_u32(f, "tokenizer.ggml.bos_token_id", 128000)
        write_kv_u32(f, "tokenizer.ggml.eos_token_id", 128001)

def create_llama2_32k_gguf(output_path):
    """Create LLaMA-2 32K vocab GGUF fixture"""
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'GGUF')
        write_u32(f, 3)  # Version 3
        write_u64(f, 0)  # Tensor count
        write_u64(f, 6)  # Metadata KV count

        # Metadata
        write_kv_string(f, "general.architecture", "llama")
        write_kv_u32(f, "llama.vocab_size", 32000)
        write_kv_string(f, "tokenizer.ggml.model", "sentencepiece")
        write_kv_string_array(f, "tokenizer.ggml.tokens", ["<s>", "</s>", "<unk>"])
        write_kv_u32(f, "tokenizer.ggml.bos_token_id", 1)
        write_kv_u32(f, "tokenizer.ggml.eos_token_id", 2)

def create_gpt2_50k_gguf(output_path):
    """Create GPT-2 50K vocab GGUF fixture"""
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'GGUF')
        write_u32(f, 3)  # Version 3
        write_u64(f, 0)  # Tensor count
        write_u64(f, 4)  # Metadata KV count

        # Metadata
        write_kv_string(f, "general.architecture", "gpt2")
        write_kv_u32(f, "gpt2.vocab_size", 50257)
        write_kv_string(f, "tokenizer.ggml.model", "gpt2")
        write_kv_u32(f, "tokenizer.ggml.eos_token_id", 50256)

def create_llama3_with_hf_tokenizer(output_path):
    """Create LLaMA-3 with HF tokenizer GGUF fixture"""
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'GGUF')
        write_u32(f, 3)  # Version 3
        write_u64(f, 0)  # Tensor count
        write_u64(f, 6)  # Metadata KV count

        # Metadata
        write_kv_string(f, "general.architecture", "llama")
        write_kv_u32(f, "llama.vocab_size", 128256)
        write_kv_string(f, "tokenizer.ggml.model", "hf")
        write_kv_string_array(f, "tokenizer.ggml.tokens", ["<s>", "</s>", "<unk>"])
        write_kv_u32(f, "tokenizer.ggml.bos_token_id", 128000)
        write_kv_u32(f, "tokenizer.ggml.eos_token_id", 128001)

def create_corrupted_embedded_tokenizer(output_path):
    """Create GGUF with valid magic but corrupted embedded data"""
    with open(output_path, 'wb') as f:
        # Valid GGUF header
        f.write(b'GGUF')
        write_u32(f, 3)  # Version 3
        write_u64(f, 0)  # Tensor count
        write_u64(f, 3)  # Metadata KV count

        # Valid architecture but corrupted tokenizer data
        write_kv_string(f, "general.architecture", "llama")
        write_kv_u32(f, "llama.vocab_size", 32000)
        # Corrupted: Invalid tokenizer.ggml.model value
        write_kv_string(f, "tokenizer.ggml.model", "CORRUPTED_INVALID_TYPE")

def create_llama2_with_sentencepiece(output_path):
    """Create LLaMA-2 with SentencePiece GGUF fixture"""
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'GGUF')
        write_u32(f, 3)  # Version 3
        write_u64(f, 0)  # Tensor count
        write_u64(f, 6)  # Metadata KV count

        # Metadata
        write_kv_string(f, "general.architecture", "llama")
        write_kv_u32(f, "llama.vocab_size", 32000)
        write_kv_string(f, "tokenizer.ggml.model", "sentencepiece")
        write_kv_string_array(f, "tokenizer.ggml.tokens", ["<s>", "</s>", "<unk>"])
        write_kv_u32(f, "tokenizer.ggml.bos_token_id", 1)
        write_kv_u32(f, "tokenizer.ggml.eos_token_id", 2)

def create_bitnet_gguf(output_path):
    """Create BitNet GGUF fixture"""
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'GGUF')
        write_u32(f, 3)  # Version 3
        write_u64(f, 0)  # Tensor count
        write_u64(f, 2)  # Metadata KV count

        # Metadata
        write_kv_string(f, "general.architecture", "bitnet")
        write_kv_u32(f, "bitnet.vocab_size", 32000)

def main():
    fixtures_dir = Path(__file__).parent.parent / "crates/bitnet-tokenizers/tests/fixtures/gguf"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating GGUF test fixtures...")

    # Core fixtures for AC1 tests
    create_llama3_128k_gguf(fixtures_dir / "llama3-128k.gguf")
    print(f"✓ Created llama3-128k.gguf")

    create_llama2_32k_gguf(fixtures_dir / "llama2-32k.gguf")
    print(f"✓ Created llama2-32k.gguf")

    create_gpt2_50k_gguf(fixtures_dir / "gpt2-50k.gguf")
    print(f"✓ Created gpt2-50k.gguf")

    create_llama3_with_hf_tokenizer(fixtures_dir / "llama3-with-hf-tokenizer.gguf")
    print(f"✓ Created llama3-with-hf-tokenizer.gguf")

    create_corrupted_embedded_tokenizer(fixtures_dir / "corrupted-embedded-tokenizer.gguf")
    print(f"✓ Created corrupted-embedded-tokenizer.gguf")

    create_llama2_with_sentencepiece(fixtures_dir / "llama2-with-sentencepiece.gguf")
    print(f"✓ Created llama2-with-sentencepiece.gguf")

    create_bitnet_gguf(fixtures_dir / "bitnet-b1.58-2B.gguf")
    print(f"✓ Created bitnet-b1.58-2B.gguf")

    print("\nAll fixtures generated successfully!")

if __name__ == "__main__":
    main()
