# CLAUDE.md

Essential guidance for working with the BitNet.rs neural network inference codebase.

## Quick Reference

### Essential Commands
```bash
# Build (default features are EMPTY - always specify features)
cargo build --no-default-features --features cpu     # CPU inference
cargo build --no-default-features --features gpu     # GPU inference

# Test
cargo test --workspace --no-default-features --features cpu

# Quality
cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings

# Development workflow
cargo run -p xtask -- download-model
cargo run -p xtask -- infer --model path/to/model.gguf --prompt "Test"

# Model export and validation (clean GGUF with F16 LayerNorm)
just model-clean <model_dir> <tokenizer.json>        # Export + validate
just model-validate <model.gguf> <tokenizer.json>    # Validate existing
just model-inspect-ln <model.gguf>                   # Check LayerNorm stats

# SafeTensors to GGUF converter (Rust st2gguf - preferred)
just st2gguf-convert <input> <output>                # Convert with LayerNorm enforcement
cargo run -p bitnet-st2gguf -- --help                # See all options
st2gguf --input model.safetensors --output model.gguf --strict  # Strict validation
```

## Core Architecture

### Design Principles
1. **Feature-Gated**: Default features are **EMPTY** - always specify `--features cpu|gpu`
2. **Zero-Copy**: Memory-mapped models, efficient lifetime management
3. **Device-Aware**: Automatic GPU/CPU selection with graceful fallback
4. **Cross-Validated**: Systematic comparison with C++ reference

### Key Crates
- `bitnet` (root): Main library with unified API
- `bitnet-inference`: Autoregressive generation engine
- `bitnet-quantization`: 1-bit quantization (I2_S, TL1, TL2)
- `bitnet-kernels`: SIMD/CUDA compute kernels
- `bitnet-models`: GGUF/SafeTensors model loading
- `bitnet-tokenizers`: Universal tokenizer with auto-discovery
- `bitnet-st2gguf`: SafeTensors to GGUF converter with LayerNorm preservation
- `bitnet-cli`: Command-line interface and utilities
- `crossval`: C++ reference validation framework

## Key Configurations

### MSRV: 1.90.0 (Rust 2024 edition)

### Feature Flags
- `cpu`: SIMD-optimized CPU inference (AVX2/AVX-512/NEON)
- `gpu`: CUDA acceleration with mixed precision (FP16/BF16)
- `cuda`: Backward-compatible alias for `gpu` (temporary - prefer `gpu` in new code)
- `ffi`: C++ FFI bridge for gradual migration
- `crossval`: Cross-validation against Microsoft BitNet C++

**Important**: Always use unified GPU predicate in code:
```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_function() { /* ... */ }
```
Use `bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime}` for runtime checks.

### Quantization Support
- **I2_S**: Production 2-bit signed quantization (99%+ accuracy vs FP32)
- **TL1/TL2**: Table lookup quantization with device-aware selection
- **IQ2_S**: GGML-compatible via FFI bridge

## Documentation Structure

### Getting Started
- `docs/quickstart.md`: 5-minute setup guide
- `docs/getting-started.md`: Comprehensive introduction
- `docs/explanation/FEATURES.md`: Feature flag documentation

### Development
- `docs/development/build-commands.md`: Comprehensive build reference
- `docs/development/gpu-development.md`: CUDA development guide
- `docs/development/test-suite.md`: Testing framework
- `docs/development/validation-framework.md`: Quality assurance
- `docs/development/xtask.md`: Developer tooling
- `docs/howto/export-clean-gguf.md`: Clean GGUF export and validation

### Architecture
- `docs/architecture-overview.md`: System design and components
- `docs/reference/quantization-support.md`: Quantization algorithms
- `docs/gpu-kernel-architecture.md`: CUDA kernel design
- `docs/tokenizer-architecture.md`: Universal tokenizer system

### Operations
- `docs/performance-benchmarking.md`: Performance testing
- `docs/health-endpoints.md`: Monitoring and observability
- `docs/GPU_SETUP.md`: GPU configuration
- `docs/environment-variables.md`: Runtime configuration
- `docs/baselines/`: Model baselines and fingerprints

## Common Workflows

### Development
```bash
# Standard development cycle
cargo test --workspace --no-default-features --features cpu
cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings

# Cross-validation (when changing inference)
# First-time setup: provision model
cargo run -p xtask -- download-model
# Tests auto-discover model in models/ directory
cargo test -p bitnet-models --no-default-features --features crossval
# Or use custom model path
export BITNET_GGUF="path/to/model.gguf"
cargo run -p xtask -- crossval

# Model operations
cargo run -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf
cargo run -p bitnet-cli -- compat-check model.gguf
```

### Troubleshooting
- FFI linker errors: Use `--no-default-features --features cpu` or `cargo xtask fetch-cpp`
- CUDA issues: Ensure CUDA toolkit installed and `nvcc` in PATH
- Model validation: `cargo run -p bitnet-cli -- compat-check model.gguf`
- GPU detection: Run `cargo run -p xtask -- preflight` to check GPU compilation and runtime availability
- Silent CPU fallback: Check receipts for GPU kernel IDs (`gemm_*`, `i2s_gpu_*`); use `BITNET_GPU_FAKE` for testing
- Feature gate mismatches: Always use `#[cfg(any(feature = "gpu", feature = "cuda"))]` pattern
- LayerNorm validation errors: If you see "suspicious LayerNorm gamma" warnings, your GGUF has quantized LN weights (should be FP16/FP32)
  - **RMS-based validation**: Validator checks LayerNorm gamma RMS (root mean square) in envelope [0.5, 2.0] around expected ≈1.0
  - **Proper fix**: Regenerate GGUF with LayerNorm weights in float format (not quantized)
  - **Diagnosis**: Use `cargo run -p bitnet-cli -- inspect --ln-stats model.gguf` to examine LayerNorm statistics
  - **Temporary workaround**: Policy-driven corrections for known-bad models (see `docs/explanation/correction-policy.md`)
    - Requires both `BITNET_CORRECTION_POLICY=/path/to/policy.yml` and `BITNET_ALLOW_RUNTIME_CORRECTIONS=1`
    - CI blocks correction flags - use only for fingerprinted known-bad models
  - **Strict mode**: `BITNET_STRICT_MODE=1` will fail immediately on suspicious LN weights

## Environment Variables
- `BITNET_DETERMINISTIC=1 BITNET_SEED=42`: Reproducible inference
- `BITNET_GGUF`: Model path override for cross-validation and inference (auto-discovers `models/` if not set)
- `RAYON_NUM_THREADS=1`: Single-threaded determinism
- `BITNET_GPU_FAKE=cuda|none`: Override GPU detection for deterministic testing (Issue #439)
- `BITNET_STRICT_MODE=1`: Enable strict validation including LayerNorm gamma RMS statistics (fails on suspicious weights)
- `BITNET_CORRECTION_POLICY=/path/to/policy.yml`: Policy file for model-specific corrections (requires `BITNET_ALLOW_RUNTIME_CORRECTIONS=1`)
- `BITNET_ALLOW_RUNTIME_CORRECTIONS=1`: Enable runtime corrections for known-bad models (CI blocks this flag)

## Repository Contracts
- **Always specify features**: `--no-default-features --features cpu|gpu`
- **Use xtask for operations**: `cargo run -p xtask --` instead of scripts
- **Check compatibility**: Review `COMPATIBILITY.md` before API changes
- **Never modify GGUF in-place**: Use `bitnet-compat export-fixed` for new files

For comprehensive documentation, see the `docs/` directory organized by audience and use case.