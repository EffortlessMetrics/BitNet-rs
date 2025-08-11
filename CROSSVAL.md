# Cross-Validation Documentation

This document describes how to run cross-validation tests between BitNet.rs and Microsoft's official BitNet C++ implementation to ensure drop-in compatibility and identical results.

## Quick Start

```bash
# One-command cross-validation (fetches, builds, and tests)
./scripts/crossval.sh /path/to/model.gguf

# Or set environment and run tests directly
export CROSSVAL_GGUF=/path/to/model.gguf
export BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp
cargo test --features crossval
```

## Prerequisites

- Rust 1.70+ with cargo
- C++ compiler (g++ or clang++)
- CMake 3.14+
- Git with submodule support
- bindgen dependencies (libclang)
- ~2GB disk space for C++ build
- A BitNet GGUF model file

## Setup Process

### 1. Fetch and Build Microsoft BitNet C++

The fetch script clones the official Microsoft BitNet repository and builds it:

```bash
# Build pinned stable release (recommended)
./ci/fetch_bitnet_cpp.sh --tag b1-65-ggml

# Or use latest main branch
./ci/fetch_bitnet_cpp.sh --tag main --force
```

This script:
- Clones https://github.com/microsoft/BitNet.git with submodules
- Handles the known `bitnet-lut-kernels.h` header issue
- Builds shared libraries for FFI
- Verifies all critical files are present

The C++ implementation is cached at `$HOME/.cache/bitnet_cpp`.

### 2. Set Environment Variables

```bash
# Point to C++ implementation
export BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp

# Deterministic execution (REQUIRED for parity)
export OMP_NUM_THREADS=1
export GGML_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Library paths (optional with RPATH support)
export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src:$LD_LIBRARY_PATH

# Your test model
export CROSSVAL_GGUF=/path/to/model.gguf
```

### 3. Build Rust with Cross-Validation

```bash
# Build the crossval crate with C++ support
cargo build --features crossval -p bitnet-crossval --release
```

## Running Tests

### All Parity Tests

```bash
./scripts/crossval.sh /path/to/model.gguf
```

### Individual Tests

```bash
# Tokenization parity
cargo test --features crossval -p bitnet-crossval test_tokenization_parity -- --nocapture

# Single-step logits parity
cargo test --features crossval -p bitnet-crossval test_single_step_logits -- --nocapture

# Multi-step generation parity
cargo test --features crossval -p bitnet-crossval test_multi_step_generation -- --nocapture

# Batch processing parity
cargo test --features crossval -p bitnet-crossval test_batch_processing -- --nocapture
```

## Test Coverage

The cross-validation suite tests:

1. **Model Loading**: Verifies model properties match (vocab size, context, embedding dim)
2. **Tokenization Parity**: Uses C++ tokenizer for both to ensure identical tokens
3. **Single-Step Logits**: Forward pass produces identical logits (tolerance: 1e-4)
4. **Multi-Step Generation**: Greedy decoding produces identical token sequences
5. **Batch Processing**: All positions in a batch have matching logits

### Tolerance Settings

- **Logit comparison**: `1e-4` (can tighten to `5e-5` for stricter validation)
- **Token comparison**: Exact match required
- **Greedy sampling**: Argmax must select same token ID

## Deterministic Execution

For reproducible parity testing:

- **Threading**: Single thread (`n_threads=1`, `OMP_NUM_THREADS=1`, `GGML_NUM_THREADS=1`)
- **Hardware**: CPU-only (no GPU/Metal/BLAS acceleration)
- **Sampling**: Greedy (argmax, no randomness)
- **Context**: Fixed seed (`seed=0`), `logits_all=true` for per-position comparison
- **Build**: Release mode with same optimization flags

## Architecture

```
bitnet-sys/              # FFI bindings to Microsoft BitNet
├── build.rs            # Links to C++ libs with RPATH, generates bindings
├── src/
│   ├── lib.rs          # Module exports
│   └── wrapper.rs      # Safe wrappers using official llama.cpp API
│
crossval/                # Cross-validation framework
├── src/
│   └── lib.rs          # Comparison utilities
└── tests/
    └── parity.rs       # Deterministic parity tests with per-step validation

ci/
└── fetch_bitnet_cpp.sh # Fetch & build script with strict validation

scripts/
└── crossval.sh         # One-command runner with all flags set
```

## Troubleshooting

### Missing Headers

If you see `bitnet-lut-kernels.h not found`:
- The fetch script automatically copies from preset kernels
- This is a known issue with the Microsoft repository structure

### Library Not Found

If linking fails:
```bash
# Verify libraries were built
find $HOME/.cache/bitnet_cpp/build -name "*.so" -o -name "*.dylib"

# Should find:
# - libllama.so / libllama.dylib
# - libggml.so / libggml.dylib
```

### Parity Failures

If tests show differences:
1. **Check environment**: All threading vars must be `1`
2. **Verify model**: Must be exact same GGUF file
3. **Check step number**: Early divergence = fundamental issue
4. **Compare top-5**: Test shows highest probability tokens
5. **Isolate failure**: Run single-step test first

Debug commands:
```bash
# Maximum verbosity
RUST_BACKTRACE=1 cargo test --features crossval -p bitnet-crossval \
  test_single_step_logits -- --nocapture

# Check just tokenization
cargo test --features crossval -p bitnet-crossval \
  test_tokenization_parity -- --nocapture
```

### Clean Rebuild

```bash
# Clean C++ build
rm -rf $HOME/.cache/bitnet_cpp/build
./ci/fetch_bitnet_cpp.sh --clean

# Clean Rust build
cargo clean
cargo build --features crossval
```

## CI Integration

For GitHub Actions:

```yaml
- name: Setup cross-validation
  run: |
    ./ci/fetch_bitnet_cpp.sh --tag b1-65-ggml
    echo "BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp" >> $GITHUB_ENV
    echo "OMP_NUM_THREADS=1" >> $GITHUB_ENV
    echo "GGML_NUM_THREADS=1" >> $GITHUB_ENV
    
- name: Run parity tests
  run: |
    cargo test --features crossval -p bitnet-crossval --release \
      -- --test-threads=1
```

### Caching

```yaml
- name: Cache BitNet C++
  uses: actions/cache@v3
  with:
    path: ~/.cache/bitnet_cpp
    key: bitnet-cpp-b1-65-ggml-${{ runner.os }}
```

## Implementation Details

### Pinned Version

We pin to tag `b1-65-ggml` (BitNet v1.0 release) for stability. This version:
- Has working llama.cpp integration
- Supports GGUF format
- Includes BitNet b1.58 kernels

### Key APIs Used

- `llama_batch_add()`: Official batch construction (not direct field access)
- `llama_get_logits_ith()`: Per-position logits with `logits_all=true`
- `llama_tokenize()`: C++ tokenizer for exact match
- `llama_context_default_params()`: Deterministic context setup

### Safety

- RAII with Drop traits for cleanup
- Safe wrappers around all C calls
- No manual memory management
- Fail-fast on missing dependencies

## References

- [Microsoft BitNet Repository](https://github.com/microsoft/BitNet)
- [BitNet Paper](https://arxiv.org/abs/2402.17764)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)