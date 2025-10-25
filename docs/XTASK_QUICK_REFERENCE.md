# XTask Quick Reference Guide

## Command Matrix

| Command | Feature Gate | FFI Used | Subprocess | Always Works |
|---------|--------------|----------|-----------|--------------|
| `download-model` | None | No | No | ✓ |
| `tokenizer` | None | No | No | ✓ |
| `fetch-cpp` | None | No | Yes (git/cmake) | ✓ |
| `setup-cpp-auto` | None | No | Yes (fetch-cpp) | ✓ |
| `trace-diff` | None | No | Yes (python3) | ✓ |
| `verify-receipt` | None | No | No | ✓ |
| `gate mapper` | None | No | No | ✓ |
| `crossval` | `crossval` | In subprocess | Yes (cargo test) | ✓ |
| `crossval-per-token` | `inference` | Yes (bitnet-sys) | No | If `--features inference` |
| `benchmark` | `inference` | Optional | No | If `--features inference` |
| `infer` | `inference` | Optional | No | If `--features inference` |
| `compare-metrics` | None | No | No | ✓ |
| `detect-breaking` | None | No | No | ✓ |

## Feature Flags

### Minimum Build
```bash
cargo build --no-default-features --features cpu
# Available: All "Always Works" commands
# Cannot: crossval-per-token, benchmark, infer
```

### Development Build (Recommended)
```bash
cargo build --no-default-features --features cpu,inference
# Available: All commands except some crossval features
# Note: Includes FFI support if C++ reference available
```

### Full Cross-Validation Build (CI/Testing)
```bash
cargo build --no-default-features --features cpu,crossval-all
# Available: ALL commands
# Enables: inference, crossval, ffi
```

### GPU Support
```bash
cargo build --no-default-features --features cpu,gpu,inference
# Adds: GPU acceleration for benchmark/infer commands
```

## FFI Availability Checks

### crossval-per-token Command

```rust
// Requires: --features inference
// Inside function (lazy import):
if !bitnet_sys::is_available() {
    bail!("C++ FFI not available. Compile with --features crossval or set BITNET_CPP_DIR");
}
```

**Dependencies**:
- bitnet_sys (for C++ wrapper)
- bitnet_crossval (for comparison)
- bitnet_inference (for Rust inference)
- BITNET_CPP_DIR env var (for C++ reference)

### crossval Command

```rust
// Subprocess-based, no FFI in xtask itself
// Spawns: cargo test -p bitnet-crossval --features crossval
// FFI happens inside bitnet-crossval tests
```

**Dependencies**:
- `cargo` (for subprocess)
- bitnet-crossval crate (in subprocess)
- BITNET_CPP_DIR env var (passed to subprocess)

## Environment Variables

### FFI-Related
```bash
BITNET_CPP_DIR=/path/to/bitnet.cpp    # C++ reference location
BITNET_DETERMINISTIC=1                 # Reproducible inference
BITNET_SEED=42                         # Deterministic RNG
```

### Testing
```bash
CROSSVAL_ALLOW_CPP_FAIL=1             # Allow C++ failures for xfail models
RAYON_NUM_THREADS=1                    # Single-threaded execution
RUST_BACKTRACE=1                       # Error diagnostics
```

### Authentication
```bash
HF_TOKEN=<token>                       # Hugging Face API token
```

## Usage Examples

### Example 1: Verify Receipt (No FFI Needed)
```bash
cargo run -p xtask -- verify-receipt --path ci/inference.json
# Works with --no-default-features --features cpu
# Pure JSON schema validation, no FFI
```

### Example 2: Setup C++ Reference (No FFI Needed)
```bash
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
# Works with --no-default-features --features cpu
# Just emits environment variables, no FFI
```

### Example 3: Compare Logits with FFI (Requires Feature)
```bash
cargo run -p xtask --no-default-features --features cpu,inference -- \
  crossval-per-token \
    --model models/model.gguf \
    --tokenizer models/tokenizer.json \
    --prompt "Test" \
    --max-tokens 4
# Requires: --features inference
# Uses: bitnet-sys FFI (if available) + bitnet-crossval
```

### Example 4: Cross-Validation Tests (Subprocess)
```bash
BITNET_CPP_DIR=~/.cache/bitnet_cpp cargo run -p xtask -- \
  crossval --model models/model.gguf
# Works with any feature set
# FFI happens in bitnet-crossval subprocess, not in xtask
```

### Example 5: Benchmark with Receipt Generation (Optional FFI)
```bash
cargo run -p xtask --no-default-features --features cpu,inference -- \
  benchmark \
    --model models/model.gguf \
    --tokenizer models/tokenizer.json \
    --tokens 128 \
    --json ci/inference.json
# Optional FFI (via bitnet-inference)
# Generates receipt with kernel IDs
```

## Architecture Decision Tree

```
User runs: cargo run -p xtask -- [COMMAND] [OPTIONS]

├─ Is command in "Always Works" list?
│  ├─ Yes → Execute directly (no feature checks)
│  │        (download-model, fetch-cpp, verify-receipt, etc.)
│  │
│  └─ No → Continue...
│
├─ Is --features inference enabled?
│  ├─ No → Print error "feature not enabled"
│  │
│  └─ Yes → Continue...
│
├─ Is this a crossval-per-token command?
│  ├─ Yes → Check bitnet_sys::is_available()
│  │        ├─ Yes (FFI available) → Use C++ wrapper
│  │        └─ No (FFI not available) → Bail with helpful error
│  │
│  └─ No → Continue to benchmark/infer...
│
└─ Is this benchmark or infer?
   ├─ Yes → Try to use bitnet-inference
   │        ├─ Success → Real inference + receipts
   │        └─ Fail → Fallback to mock if --allow-mock
   │
   └─ No → Unknown command
```

## Dependency Graph

```
xtask (top-level, no FFI imports here!)
│
├─ Always available (no feature gate)
│  ├─ bitnet-kernels (for GPU detection)
│  ├─ bitnet-common (for Device enum)
│  ├─ bitnet-models (for GGUF reading)
│  ├─ bitnet-tokenizers (for tokenization)
│  └─ reqwest (for HTTP downloads)
│
├─ --features inference (optional, enables FFI)
│  ├─ bitnet-inference (Rust inference)
│  ├─ bitnet (core library)
│  ├─ bitnet-sys (C++ FFI) ← lazy import!
│  ├─ bitnet-crossval (comparison tools) ← lazy import!
│  └─ tokio, futures (async runtime)
│
├─ --features crossval (optional)
│  └─ bitnet-crossval ← used in subprocess only
│
└─ --features ffi (optional)
   └─ bitnet-sys ← lazy import!
```

## Error Messages & Solutions

### "C++ FFI not available"
```
Error: C++ FFI not available. Compile with --features crossval or set BITNET_CPP_DIR

Solution:
1. Build with --features inference or --features crossval-all
2. OR set BITNET_CPP_DIR=/path/to/bitnet.cpp
3. OR run: eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
```

### "Tokenizer required"
```
Error: tokenizer required. This model expects the **LLaMA-3 tokenizer (128,256)**.

Solution:
1. Pass --tokenizer /path/to/tokenizer.json
2. OR use --allow-mock (for testing only)
3. OR run: cargo run -p xtask -- tokenizer --into models/
```

### "compute_path must be 'real'"
```
Error: Receipt verification failed - compute_path must be 'real'

Solution:
1. This receipt was generated with mock inference
2. Use real inference by ensuring --features inference is enabled
3. Run: cargo run -p xtask -- benchmark --model ... --json ci/inference.json
```

### "GPU kernel verification required but no GPU kernels found"
```
Error: GPU kernel verification required (backend is 'cuda') but no GPU kernels found

Solution:
1. Build with GPU support: cargo build --features gpu
2. Verify CUDA is installed: nvidia-smi
3. Check that inference actually uses GPU (not CPU fallback)
```

## Development Workflow

### Building with Different Features

```bash
# Minimal (CPU only, no FFI)
cargo build --no-default-features --features cpu

# Development (with inference)
cargo build --no-default-features --features cpu,inference

# Full (with all FFI)
cargo build --no-default-features --features cpu,inference,crossval,ffi

# With GPU
cargo build --no-default-features --features cpu,gpu,inference

# Shorthand for full build
cargo build --no-default-features --features cpu,crossval-all
```

### Testing Feature Combinations

```bash
# Test with minimal features
cargo test --no-default-features --features cpu -- --test-threads=1

# Test with inference (FFI optional)
cargo test --no-default-features --features cpu,inference -- --test-threads=1

# Test with all features
cargo nextest run --no-default-features --features cpu,crossval-all
```

### Debugging FFI Issues

```bash
# Check if FFI is available at runtime
BITNET_CPP_DIR=~/.cache/bitnet_cpp cargo run -p xtask -- \
  crossval-per-token \
    --model models/model.gguf \
    --tokenizer models/tokenizer.json \
    --prompt "Test" \
    --format json

# If FFI not available, set up C++ reference first
eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

# Then retry cross-validation
cargo run -p xtask --features crossval-all -- crossval --model models/model.gguf
```

## Performance Tips

1. **Avoid subprocess overhead**: Use `crossval-per-token` instead of `crossval` for single-model testing
2. **Cache model downloads**: Use same `--out` directory for multiple runs
3. **Use release builds**: `cargo build --release` for production inference
4. **Set RAYON_NUM_THREADS=1**: For deterministic testing
5. **Warm up GPU**: First benchmark run trains GPU drivers, subsequent runs faster

---

**Last Updated**: 2025-10-24
**Architecture Version**: 0.1.0-qna-mvp
