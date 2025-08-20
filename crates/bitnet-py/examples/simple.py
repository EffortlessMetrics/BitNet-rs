#!/usr/bin/env python3
"""Simple Python example demonstrating BitNet bindings."""

import os
import sys
import numpy as np

# Add the build directory to path (for development)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'target', 'debug'))

try:
    import bitnet
except ImportError:
    print("BitNet Python module not found. Build with:")
    print("  cargo build -p bitnet-py --features python")
    sys.exit(1)

def main():
    print("BitNet Python Bindings Example")
    print("==============================\n")
    
    # Check API version
    print("1. Checking API version...")
    try:
        api_version = bitnet.ffi_api_version()
        print(f"   ✓ API version: {api_version}")
        assert api_version == 1, f"Expected API version 1, got {api_version}"
    except AttributeError:
        print("   ⚠ API version check not available (using mock)")
        api_version = 1
    
    # Test quantization
    print("\n2. Testing quantization...")
    test_data = np.random.randn(1024).astype(np.float32)
    
    try:
        # Quantize the data
        quantized, scales = bitnet.quantize_i2s(test_data)
        print(f"   ✓ Quantized {len(test_data)} float32 → {len(quantized)} bytes")
        print(f"   Compression ratio: {len(test_data) * 4 / len(quantized):.1f}x")
        print(f"   Number of scales: {len(scales)}")
        
        # Verify scales are reasonable
        assert all(s > 0 for s in scales), "All scales should be positive"
        print(f"   Scale range: [{min(scales):.6f}, {max(scales):.6f}]")
    except (AttributeError, NotImplementedError):
        print("   ⚠ Quantization not yet implemented")
    
    # Test model loading
    print("\n3. Testing model operations...")
    model_path = os.environ.get("MODEL_PATH", "models/ggml-model-i2_s.gguf")
    
    if os.path.exists(model_path):
        try:
            print(f"   Loading model: {model_path}")
            model = bitnet.load_model(model_path)
            print(f"   ✓ Model loaded: {model}")
            
            # Get model info
            vocab_size = model.vocab_size()
            print(f"   Vocab size: {vocab_size}")
            
            # Simple generation test
            prompt = "Hello, world!"
            print(f"\n4. Testing generation with prompt: '{prompt}'")
            
            result = model.generate(prompt, max_tokens=10)
            print(f"   Generated: {result}")
            
        except (AttributeError, NotImplementedError) as e:
            print(f"   ⚠ Model operations not yet fully implemented: {e}")
    else:
        print(f"   ⚠ Model not found at: {model_path}")
        print("   Download with: cargo xtask download-model")
    
    # Performance test
    print("\n5. Performance benchmark...")
    sizes = [1024, 16384, 65536]
    
    for size in sizes:
        data = np.random.randn(size).astype(np.float32)
        
        import time
        start = time.perf_counter()
        
        # Simulate quantization (or use real if available)
        try:
            quantized, scales = bitnet.quantize_i2s(data)
            elapsed = time.perf_counter() - start
            throughput = size / elapsed / 1e6  # M elements/sec
            print(f"   {size:6d} elements: {throughput:.2f}M elem/s ({elapsed*1000:.3f}ms)")
        except (AttributeError, NotImplementedError):
            # Fallback to numpy simulation
            elapsed = time.perf_counter() - start
            print(f"   {size:6d} elements: (simulated)")
    
    print("\n✅ Python example completed successfully!")

if __name__ == "__main__":
    main()