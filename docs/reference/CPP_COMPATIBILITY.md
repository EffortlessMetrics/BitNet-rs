# C++ Compatibility Guide

This document explains how to fix C++ compatibility issues with BitNet models.

## Common Issues and Solutions

### 1. Library Loading Errors (libllama.so not found)

**Symptom:**
```
error while loading shared libraries: libllama.so: cannot open shared object file
```

**Root Cause:** The C++ libraries are built as shared libraries but not in the system library path.

**Solutions:**

#### Solution A: Set Library Paths (Recommended for Development)
```bash
# Linux
export LD_LIBRARY_PATH="$HOME/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src:$HOME/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src:$LD_LIBRARY_PATH"

# macOS
export DYLD_LIBRARY_PATH="$HOME/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src:$HOME/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src:$DYLD_LIBRARY_PATH"
```

#### Solution B: Static Linking (Recommended for CI)
```bash
cd ~/.cache/bitnet_cpp
rm -rf build
cmake -B build -DBUILD_SHARED_LIBS=OFF -DLLAMA_STATIC=ON
cmake --build build -j$(nproc)
```

#### Solution C: System Installation (Production)
```bash
sudo cp ~/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src/*.so /usr/local/lib/
sudo cp ~/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src/*.so /usr/local/lib/
sudo ldconfig
```

### 2. GGUF Version Incompatibility

**Symptom:**
```
llama_load_model_from_file: unsupported GGUF version
invalid or unsupported tensor
```

**Root Cause:** The BitNet models use GGUF v3, but older C++ implementations only support v2.

**Solutions:**

#### Solution A: Update C++ Implementation
```bash
cargo xtask fetch-cpp --tag main --force --clean
```

#### Solution B: Convert Model to GGUF v2
```bash
# Build the converter tool
cargo build -p bitnet-compat --release

# Convert the model
cargo run -p bitnet-compat -- convert \
  --input models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --output models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s-v2.gguf \
  --target-version 2
```

#### Solution C: Use Soft-Fail in CI
```bash
# This allows CI to pass even if C++ fails
export CROSSVAL_ALLOW_CPP_FAIL=1
```

### 3. Tokenizer Metadata Issues

**Symptom:**
```
missing tokenizer.ggml.pre
tokenizer metadata not found
```

**Root Cause:** BitNet models may have incomplete tokenizer metadata that C++ requires.

**Solutions:**

#### Solution A: Fix Metadata with Compat Tool
```bash
cargo run -p bitnet-compat -- fix-metadata \
  --input models/model.gguf \
  --output models/model-fixed.gguf
```

#### Solution B: Use Rust Implementation
The Rust implementation handles missing metadata gracefully and doesn't require all fields.

### 4. Quantization Format Issues

**Symptom:**
```
unsupported quantization type
invalid tensor data
```

**Root Cause:** BitNet uses 1-bit quantization that may not be supported in older C++ versions.

**Solutions:**

#### Solution A: Update C++ to Latest
```bash
# Get the latest version with BitNet support
cargo xtask fetch-cpp --repo https://github.com/microsoft/BitNet.git --tag main
```

#### Solution B: Ensure Correct Build Flags
```bash
cd ~/.cache/bitnet_cpp
cmake -B build \
  -DLLAMA_BITNET=ON \
  -DGGML_BITNET=ON \
  -DLLAMA_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Testing Compatibility

### Quick Test Script
```bash
#!/bin/bash
# test_compat.sh

MODEL="$1"
CPP_DIR="$HOME/.cache/bitnet_cpp"

# Set library paths
export LD_LIBRARY_PATH="${CPP_DIR}/build/3rdparty/llama.cpp/src:${CPP_DIR}/build/3rdparty/llama.cpp/ggml/src"

# Test C++ loading
${CPP_DIR}/build/bin/llama-cli -m "$MODEL" -n 1 -p "test" --log-disable

if [ $? -eq 0 ]; then
    echo "✅ C++ compatible"
else
    echo "❌ C++ incompatible"
fi
```

### Full Cross-Validation Test
```bash
# With proper library paths
export LD_LIBRARY_PATH="$HOME/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src:$HOME/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src"
export CROSSVAL_GGUF="path/to/model.gguf"

# Run tests
cargo test -p bitnet-crossval --features crossval
```

## CI Configuration

### GitHub Actions Example
```yaml
env:
  # Enable soft-fail for C++ compatibility issues
  CROSSVAL_ALLOW_CPP_FAIL: "1"
  
  # Set library paths
  LD_LIBRARY_PATH: "$HOME/.cache/bitnet_cpp/build/3rdparty/llama.cpp/src:$HOME/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src"
```

### Local Development
Add to your `.bashrc` or `.zshrc`:
```bash
# BitNet C++ libraries
export BITNET_CPP_DIR="$HOME/.cache/bitnet_cpp"
export LD_LIBRARY_PATH="${BITNET_CPP_DIR}/build/3rdparty/llama.cpp/src:${BITNET_CPP_DIR}/build/3rdparty/llama.cpp/ggml/src:$LD_LIBRARY_PATH"
```

## Troubleshooting Checklist

1. **Check GGUF version:**
   ```bash
   hexdump -C model.gguf | head -1
   # Look for version at byte 4-7
   ```

2. **Verify C++ build:**
   ```bash
   ldd ~/.cache/bitnet_cpp/build/bin/llama-cli
   # Check all libraries are found
   ```

3. **Test minimal model:**
   ```bash
   cargo xtask gen-mini-gguf
   # Test with tests/models/mini.gguf first
   ```

4. **Enable verbose logging:**
   ```bash
   export RUST_LOG=debug
   export LLAMA_LOG_LEVEL=debug
   ```

## When All Else Fails

If C++ compatibility cannot be achieved:

1. **Use Rust-only mode:** The Rust implementation is fully functional without C++
2. **Enable soft-fail:** Set `CROSSVAL_ALLOW_CPP_FAIL=1` in CI
3. **File an issue:** Report incompatibilities to help improve both implementations

## Summary

Most C++ compatibility issues fall into three categories:
1. **Library paths** - Easily fixed with environment variables
2. **GGUF versions** - Fixed by updating C++ or converting models
3. **Missing metadata** - Fixed with compat tools or using Rust

The soft-fail mechanism (`CROSSVAL_ALLOW_CPP_FAIL=1`) ensures CI remains green while these issues are resolved.