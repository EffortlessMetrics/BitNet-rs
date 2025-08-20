#!/usr/bin/env python3
"""
Inspect GGUF model metadata to understand tokenizer configuration
"""

import struct
import json
from pathlib import Path

def read_string(f):
    """Read a GGUF string (length-prefixed)"""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def read_gguf_header(filepath):
    """Read and parse GGUF header to extract metadata"""
    with open(filepath, 'rb') as f:
        # Read magic number
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Not a GGUF file (magic: {magic})")
        
        # Read version
        version = struct.unpack('<I', f.read(4))[0]
        print(f"GGUF Version: {version}")
        
        # Read tensor count and metadata KV count
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Tensor count: {tensor_count}")
        print(f"Metadata KV count: {metadata_kv_count}")
        
        # Read metadata key-value pairs
        metadata = {}
        tokenizer_metadata = {}
        
        for i in range(metadata_kv_count):
            key = read_string(f)
            
            # Read value type
            vtype = struct.unpack('<I', f.read(4))[0]
            
            # Parse value based on type
            if vtype == 0:  # UINT8
                value = struct.unpack('B', f.read(1))[0]
            elif vtype == 1:  # INT8
                value = struct.unpack('b', f.read(1))[0]
            elif vtype == 2:  # UINT16
                value = struct.unpack('<H', f.read(2))[0]
            elif vtype == 3:  # INT16
                value = struct.unpack('<h', f.read(2))[0]
            elif vtype == 4:  # UINT32
                value = struct.unpack('<I', f.read(4))[0]
            elif vtype == 5:  # INT32
                value = struct.unpack('<i', f.read(4))[0]
            elif vtype == 6:  # FLOAT32
                value = struct.unpack('<f', f.read(4))[0]
            elif vtype == 7:  # BOOL
                value = struct.unpack('?', f.read(1))[0]
            elif vtype == 8:  # STRING
                value = read_string(f)
            elif vtype == 9:  # ARRAY
                array_type = struct.unpack('<I', f.read(4))[0]
                array_len = struct.unpack('<Q', f.read(8))[0]
                
                if array_type == 8:  # STRING array
                    value = [read_string(f) for _ in range(array_len)]
                else:
                    # Skip other array types for now
                    value = f"Array of type {array_type} with {array_len} elements"
                    if array_type == 0:  # UINT8
                        f.read(array_len)
                    elif array_type == 5:  # INT32
                        f.read(array_len * 4)
                    else:
                        print(f"Unknown array type: {array_type}")
            elif vtype == 10:  # UINT64
                value = struct.unpack('<Q', f.read(8))[0]
            elif vtype == 11:  # INT64
                value = struct.unpack('<q', f.read(8))[0]
            elif vtype == 12:  # FLOAT64
                value = struct.unpack('<d', f.read(8))[0]
            else:
                value = f"Unknown type {vtype}"
            
            metadata[key] = value
            
            # Collect tokenizer-specific metadata
            if 'tokenizer' in key or 'vocab' in key or 'token' in key:
                tokenizer_metadata[key] = value
        
        return metadata, tokenizer_metadata

def main():
    model_path = "/home/steven/code/Rust/BitNet-rs/models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
    
    print("=" * 60)
    print("GGUF Model Tokenizer Analysis")
    print("=" * 60)
    print(f"Model: {Path(model_path).name}\n")
    
    try:
        metadata, tokenizer_metadata = read_gguf_header(model_path)
        
        print("\n=== Tokenizer-Related Metadata ===")
        for key, value in sorted(tokenizer_metadata.items()):
            if isinstance(value, list) and len(value) > 10:
                print(f"{key}: [list with {len(value)} items]")
                # Show first few items if it's tokens
                if 'tokens' in key:
                    print(f"  First 5: {value[:5]}")
                    print(f"  Last 5: {value[-5:]}")
            elif isinstance(value, str) and len(value) > 100:
                print(f"{key}: {value[:100]}...")
            else:
                print(f"{key}: {value}")
        
        # Analyze specific tokenizer issues
        print("\n=== Tokenizer Configuration Analysis ===")
        
        # Check model type
        if 'tokenizer.ggml.model' in tokenizer_metadata:
            model_type = tokenizer_metadata['tokenizer.ggml.model']
            print(f"Tokenizer model type: {model_type}")
            
            if model_type != 'llama':
                print(f"  ⚠️  WARNING: Non-standard tokenizer type (expected 'llama')")
        
        # Check vocab size
        if 'llama.vocab_size' in metadata:
            vocab_size = metadata['llama.vocab_size']
            print(f"Vocabulary size: {vocab_size}")
            
            if vocab_size > 128256:
                print(f"  ⚠️  WARNING: Large vocabulary size may cause issues")
        
        # Check for BPE merges
        if 'tokenizer.ggml.merges' in tokenizer_metadata:
            merges = tokenizer_metadata['tokenizer.ggml.merges']
            if isinstance(merges, list):
                print(f"BPE merges: {len(merges)} merge rules")
            else:
                print(f"BPE merges: {merges}")
        
        # Check special tokens
        special_tokens = {
            'bos': 'tokenizer.ggml.bos_token_id',
            'eos': 'tokenizer.ggml.eos_token_id',
            'unk': 'tokenizer.ggml.unknown_token_id',
            'sep': 'tokenizer.ggml.separator_token_id',
            'pad': 'tokenizer.ggml.padding_token_id'
        }
        
        print("\nSpecial tokens:")
        for name, key in special_tokens.items():
            if key in tokenizer_metadata:
                token_id = tokenizer_metadata[key]
                print(f"  {name}: {token_id}")
                
                # Try to get the actual token string
                tokens_key = 'tokenizer.ggml.tokens'
                if tokens_key in tokenizer_metadata and isinstance(tokenizer_metadata[tokens_key], list):
                    if 0 <= token_id < len(tokenizer_metadata[tokens_key]):
                        token_str = tokenizer_metadata[tokens_key][token_id]
                        print(f"    -> '{token_str}'")
        
        # Check for known issues
        print("\n=== Potential Issues ===")
        issues_found = False
        
        # Issue 1: Missing tokenizer model
        if 'tokenizer.ggml.model' not in tokenizer_metadata:
            print("❌ Missing tokenizer.ggml.model - C++ may not recognize tokenizer type")
            issues_found = True
        
        # Issue 2: Missing merges for BPE
        if 'tokenizer.ggml.tokens' in tokenizer_metadata:
            has_tokens = True
            tokens = tokenizer_metadata['tokenizer.ggml.tokens']
            if isinstance(tokens, list) and len(tokens) > 50000:  # Large vocab typically uses BPE
                if 'tokenizer.ggml.merges' not in tokenizer_metadata:
                    print("❌ Large vocabulary but missing BPE merges")
                    issues_found = True
        
        # Issue 3: Special token mismatch
        if 'tokenizer.ggml.eos_token_id' in tokenizer_metadata:
            eos_id = tokenizer_metadata['tokenizer.ggml.eos_token_id']
            if 'llama.eos_token_id' in metadata:
                llama_eos = metadata['llama.eos_token_id']
                if eos_id != llama_eos:
                    print(f"❌ EOS token mismatch: tokenizer={eos_id}, model={llama_eos}")
                    issues_found = True
        
        if not issues_found:
            print("✅ No obvious tokenizer configuration issues found")
        
        # Save full metadata for inspection
        with open("gguf_metadata.json", "w") as f:
            # Convert any bytes to string for JSON serialization
            clean_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, bytes):
                    clean_metadata[k] = v.decode('utf-8', errors='ignore')
                elif isinstance(v, list) and len(v) > 100:
                    clean_metadata[k] = f"[{len(v)} items]"
                else:
                    clean_metadata[k] = v
            
            json.dump(clean_metadata, f, indent=2)
        
        print("\nFull metadata saved to gguf_metadata.json")
        
    except Exception as e:
        print(f"Error reading GGUF file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()