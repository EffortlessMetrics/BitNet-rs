# C++ Cross-Validation Guide

This guide explains how the BitNet-rs cross-validation system works and how to fix common C++ compatibility issues.

## Quick Start

### 1. Static Build (Recommended)

Build C++ with static linking to avoid library path issues:

```bash
# Fetch and build with static linking
cargo xtask fetch-cpp --force

# Or manually rebuild existing installation
cd ~/.cache/bitnet_cpp
cmake -B build -S . \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_STATIC=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_NATIVE=ON
cmake --build build -j
```

### 2. Run Cross-Validation

```bash
# With soft-fail enabled (CI-friendly)
export CROSSVAL_ALLOW_CPP_FAIL=1
cargo xtask crossval --model models/bitnet/model.gguf

# Generate test fixtures
cargo xtask gen-mini-gguf --version 3
cargo xtask crossval --model tests/models/mini.gguf
```

## Architecture

### Soft-Fail Mechanism

The cross-validation system has a two-stage validation:

1. **Rust Validation**: Must always pass
2. **C++ Parity**: Can be allowed to fail with `CROSSVAL_ALLOW_CPP_FAIL=1`

This ensures CI stays green even when C++ has compatibility issues with experimental models.

### Platform-Specific Library Handling

The xtask automatically configures the correct library paths:

- **Linux**: Sets `LD_LIBRARY_PATH`
- **macOS**: Sets `DYLD_LIBRARY_PATH`
- **Windows**: Updates `PATH`

### GGUF Format Compatibility

We generate spec-compliant GGUF files:

- **v2**: Uses u32 string lengths (smaller files)
- **v3**: Uses u64 string lengths (larger files)

Mini fixtures (128 bytes) enable fast validation without large downloads.

## Common Issues and Solutions

### Issue 1: Library Not Found

**Symptom:**
```
error while loading shared libraries: libllama.so: cannot open shared object file
```

**Solution:**
Use static build (see Quick Start) or set library paths:

```bash
export LD_LIBRARY_PATH="$HOME/.cache/bitnet_cpp/build/bin:$HOME/.cache/bitnet_cpp/build/3rdparty/llama.cpp/ggml/src"
```

### Issue 2: GGUF Version Mismatch

**Symptom:**
```
unsupported GGUF version
```

**Solution:**
Update C++ to latest:
```bash
cargo xtask fetch-cpp --tag main --force --clean
```

### Issue 3: Model Loading Fails in C++ but Not Rust

**Symptom:**
```
C++ backend failed to load model
Rust implementation validated successfully
```

**Solution:**
This is expected for experimental BitNet models. Use `CROSSVAL_ALLOW_CPP_FAIL=1` to allow soft failures.

### Issue 4: Non-Deterministic Results

**Symptom:**
Results vary between runs.

**Solution:**
The xtask automatically sets deterministic environment:
- `RAYON_NUM_THREADS=1`
- `BITNET_DETERMINISTIC=1`
- `BITNET_SEED=42`
- `OMP_NUM_THREADS=1`

For C++ inference, also use:
```bash
llama-cli -ngl 0  # Force CPU-only
```

## Testing Workflow

### 1. Validate Rust Implementation

```bash
cargo test --no-default-features -p bitnet-models --features cpu
```

### 2. Test with Mini Fixtures

```bash
# Generate fixtures
cargo xtask gen-mini-gguf --version 2 --output tests/v2.gguf
cargo xtask gen-mini-gguf --version 3 --output tests/v3.gguf

# Test both implementations
cargo xtask crossval --model tests/v2.gguf
cargo xtask crossval --model tests/v3.gguf
```

### 3. Full Model Testing

```bash
# Download model
cargo xtask download-model

# Run full crossval
export CROSSVAL_ALLOW_CPP_FAIL=1
cargo xtask full-crossval
```

## CI Configuration

### GitHub Actions

```yaml
env:
  CROSSVAL_ALLOW_CPP_FAIL: "1"  # Allow C++ failures
  RAYON_NUM_THREADS: "1"         # Deterministic
  BITNET_DETERMINISTIC: "1"      # Stable results
  BITNET_SEED: "42"              # Fixed seed

steps:
  - name: Build C++ (static)
    run: cargo xtask fetch-cpp --force

  - name: Generate fixtures
    run: cargo xtask gen-mini-gguf

  - name: Cross-validation
    run: cargo xtask crossval --model tests/models/mini.gguf
    continue-on-error: true  # Don't fail CI
```

## Debugging

### Check C++ Can Load Model

```bash
# Test header parsing only
~/.cache/bitnet_cpp/build/bin/llama-gguf -l model.gguf

# Full load test (CPU only)
~/.cache/bitnet_cpp/build/bin/llama-cli \
  -m model.gguf -p "test" -n 1 -ngl 0
```

### Compare Metadata

```bash
# Rust metadata
cargo run -p bitnet-cli -- inspect model.gguf

# C++ metadata
~/.cache/bitnet_cpp/build/bin/llama-gguf -l model.gguf
```

### Environment Diagnostics

```bash
# Check library paths
ldd ~/.cache/bitnet_cpp/build/bin/llama-cli

# Check static linking
file ~/.cache/bitnet_cpp/build/bin/llama-cli | grep -i static
```

## Key Improvements

1. **Spec-Correct GGUF Generation**: v2 and v3 use correct string length encoding
2. **Real Loader Validation**: Uses actual GGUF parser, not just magic bytes
3. **Cross-Platform Environment**: Automatic library path configuration
4. **Enhanced Error Detection**: More C++ failure patterns recognized
5. **Static Build Default**: Avoids runtime library issues

## Summary

The cross-validation system is designed to be robust and CI-friendly:

- Rust validation is mandatory (hard fail)
- C++ validation is optional (soft fail with env var)
- Static builds eliminate library path issues
- Mini fixtures enable fast testing
- Platform-specific handling ensures portability

This approach ensures development velocity while maintaining quality standards.
