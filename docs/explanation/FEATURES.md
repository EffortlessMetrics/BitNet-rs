# BitNet.rs Feature Flags Documentation

This document provides comprehensive documentation for all feature flags available in the BitNet.rs project.

## Overview

BitNet.rs uses Cargo feature flags to control compilation of optional functionality. **Default features are intentionally empty** to prevent unwanted dependencies and allow precise control over the build.

## Quick Reference

```bash
# CPU-only build (most common)
cargo build --no-default-features --features cpu

# GPU-enabled build
cargo build --no-default-features --features gpu

# Full feature set
cargo build --no-default-features --features full
```

## Core Features

### `cpu`
**Purpose:** Enable CPU inference with optimized kernels
**Dependencies:** None (pure Rust)
**When to use:** For CPU-only deployments without GPU requirements

```bash
cargo build --no-default-features --features cpu
```

Enables:
- Fallback scalar kernels
- Platform-specific SIMD optimizations (if available)
- Multi-threaded CPU inference
- Automatic CPU feature detection

### `gpu`
**Purpose:** Enable advanced GPU acceleration with device-aware quantization
**Dependencies:** CUDA toolkit 11.0+, cudarc crate
**When to use:** For GPU-accelerated inference with automatic CPU fallback

```bash
cargo build --no-default-features --features gpu
```

Enables:
- Device-aware quantization with automatic GPU detection
- Comprehensive GPU quantization kernels (I2S, TL1, TL2)
- Intelligent fallback to optimized CPU kernels
- CUDA kernel compilation with bit-packing optimizations
- GPU memory management with automatic cleanup
- Multi-GPU device selection and management
- Thread-safe concurrent GPU operations
- Performance monitoring and device statistics

Requirements:
- NVIDIA GPU with compute capability 6.0+
- CUDA toolkit installed and `nvcc` in PATH
- Set `LD_LIBRARY_PATH` to include CUDA libraries

### `cuda`
**Purpose:** Backward-compatible alias for `gpu` feature
**Dependencies:** Same as `gpu` feature
**When to use:** For backward compatibility with existing build scripts

```bash
cargo build --no-default-features --features cuda
```

**Important Notes:**
- This is an alias for the `gpu` feature. New projects should use `gpu` for clarity.
- The unified predicate `#[cfg(any(feature = "gpu", feature = "cuda"))]` ensures both features work identically.
- Planned for removal in a future release; prefer `gpu` in new code.

### `ffi`
**Purpose:** Enable C++ FFI bridge for cross-validation
**Dependencies:** cc, bindgen, C++ compiler
**When to use:** For comparing against original C++ implementation

```bash
cargo build --no-default-features --features ffi
```

Enables:
- FFI bindings to BitNet C++ implementation
- Cross-validation testing capabilities
- Performance comparison tools

Note: Requires building or linking the C++ BitNet library.

## SIMD Features

### `avx2`
**Purpose:** Enable x86_64 AVX2 SIMD optimizations
**Dependencies:** x86_64 CPU with AVX2 support
**When to use:** Automatically detected on compatible CPUs

### `avx512`
**Purpose:** Enable x86_64 AVX-512 SIMD optimizations
**Dependencies:** x86_64 CPU with AVX-512F and AVX-512BW support
**When to use:** For Intel server/workstation CPUs (Skylake-X, Ice Lake, and newer)

```bash
cargo build --no-default-features --features "cpu,avx512"
```

Enables:
- AVX-512 matrix multiplication kernels with vectorized quantization
- 64-element K-dimension processing in 16x16 blocks
- Runtime feature detection with automatic AVX2 fallback
- Masked loads for tail handling and memory-safe bounds checking
- Up to 2x theoretical throughput improvement over AVX2

Requirements:
- x86_64 CPU with AVX-512F (Foundation) and AVX-512BW (Byte and Word) instruction sets
- Intel Skylake-X, Ice Lake, Tiger Lake, or newer architecture
- Automatic runtime detection - no manual configuration needed

### `neon`
**Purpose:** Enable ARM NEON SIMD optimizations
**Dependencies:** ARM64/AArch64 CPU
**When to use:** For ARM-based systems (Apple Silicon, ARM servers)

## Development Features

### `crossval`
**Purpose:** Enable cross-validation against C++ implementation
**Dependencies:** `ffi` feature, C++ BitNet implementation
**When to use:** For validating Rust implementation correctness

```bash
cargo test --no-default-features --features "cpu,ffi,crossval"
```

### `full`
**Purpose:** Enable all features for maximum functionality
**Dependencies:** All of the above
**When to use:** For development and testing

```bash
cargo build --no-default-features --features full
```

## Feature Combinations

### Common Configurations

**Production CPU deployment:**
```bash
cargo build --no-default-features --release --no-default-features --features cpu
```

**Production GPU deployment with device-aware quantization:**
```bash
cargo build --no-default-features --release --no-default-features --features "cpu,gpu"
```

**Development with GPU validation:**
```bash
cargo build --no-default-features --features "cpu,gpu,ffi,crossval"
```

**Maximum performance with GPU acceleration:**
```bash
cargo build --no-default-features --release --features "cpu,gpu,avx512"
```

## Workspace Crate Features

### bitnet-kernels
- `cpu`: CPU kernel implementations with SIMD optimizations
- `gpu`: Advanced GPU kernels with device-aware quantization
- `cuda`: Backward-compatible alias for `gpu`
- `ffi`: FFI bridge to C++ kernels
- `ffi-bridge`: Build-time FFI compilation

### bitnet-inference
- `cpu`: CPU inference engine with optimized kernels
- `gpu`: GPU inference engine with device-aware quantization
- `cuda`: Backward-compatible alias for `gpu`
- `async`: Async/await support (always enabled)

### bitnet-cli
- Inherits features from dependencies
- No crate-specific features

### bitnet-server
- `cpu`: CPU inference endpoints
- `gpu`: GPU inference endpoints with device-aware quantization
- `cuda`: Backward-compatible alias for `gpu`

## Feature Resolution

The workspace uses Cargo's resolver version 2 to prevent feature unification across dependencies:

```toml
[workspace]
resolver = "2"
```

This ensures:
- Dev-dependencies don't affect production builds
- Features are resolved per-package
- No unexpected feature activation

## Troubleshooting

### Issue: Build fails with undefined references
**Cause:** FFI enabled without C++ library
**Solution:** Either disable FFI or build the C++ library:
```bash
# Option 1: Disable FFI
cargo build --no-default-features --features cpu

# Option 2: Build C++ library
cargo xtask fetch-cpp
cargo build --no-default-features --features "cpu,ffi"
```

### Issue: CUDA features not working
**Cause:** CUDA toolkit not installed or not in PATH
**Solution:** Install CUDA toolkit and set environment:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: Features seem to be ignored
**Cause:** Default features interfering
**Solution:** Always use `--no-default-features`:
```bash
cargo build --no-default-features --features cpu
```

### Issue: Unexpected dependencies pulled in
**Cause:** Feature unification in workspace
**Solution:** Ensure `resolver = "2"` in workspace Cargo.toml

## Platform Support Matrix

| Platform | CPU | GPU/CUDA | AVX2 | AVX512 | NEON | FFI | Device-Aware |
|----------|-----|----------|------|--------|------|-----|---------------|
| Linux x86_64 | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Linux ARM64 | ✅ | ✅* | ❌ | ❌ | ✅ | ✅ | ✅ |
| macOS x86_64 | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅** |
| macOS ARM64 | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅** |
| Windows x86_64 | ✅ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ✅ |

✅ Supported
❌ Not supported
⚠️ Experimental
\* CUDA on ARM64 requires NVIDIA Jetson or similar
\*\* Device-aware on macOS falls back to optimized CPU kernels

## Device-Aware Quantization Features

The `gpu` feature enables advanced device-aware quantization with the following capabilities:

### Supported Quantization Algorithms
- **I2S (2-bit Signed)**: Native GPU kernels with optimized bit-packing
- **TL1 (Table Lookup 1)**: GPU-accelerated table lookup quantization
- **TL2 (Table Lookup 2)**: Advanced table lookup with GPU optimization

### Device Selection and Fallback
- **Automatic Device Detection**: Scans for available CUDA devices at runtime
- **Graceful Fallback**: Seamlessly falls back to optimized CPU kernels on GPU failure
- **Device Affinity**: Maintains consistent device usage throughout quantization pipeline
- **Error Recovery**: Comprehensive error handling with detailed failure reporting

### Performance Optimizations
- **CUDA Kernel Integration**: Custom CUDA kernels with atomic operations
- **Memory Management**: Automatic GPU memory allocation and cleanup
- **Concurrent Operations**: Thread-safe GPU operations with proper synchronization
- **Performance Monitoring**: Built-in statistics and device usage tracking

### Testing and Validation
```bash
# Test device-aware quantization
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu --test gpu_quantization

# Test GPU vs CPU accuracy
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy --ignored

# Test automatic fallback
cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_gpu_quantization_fallback --ignored
```

## Performance Impact

Enabling features affects performance and binary size:

| Feature | Performance Impact | Binary Size Impact | Notes |
|---------|-------------------|--------------------|-------|
| `cpu` | Baseline | +2 MB | Optimized SIMD kernels |
| `gpu` | 2-15x faster | +12 MB | Device-aware quantization |
| `cuda` | Same as `gpu` | Same as `gpu` | Backward-compatible alias |
| `avx2` | 1.5-2x faster | +0.5 MB | x86_64 SIMD optimization |
| `avx512` | 2-3x faster | +1 MB | Modern Intel and AMD CPUs |
| `neon` | 1.5-2x faster | +0.5 MB | ARM64 SIMD optimization |
| `ffi` | No impact | +4 MB | Cross-validation only |

## Best Practices

1. **Production builds:** Use minimal features needed
2. **Development:** Use `full` for comprehensive testing
3. **CI/CD:** Test with multiple feature combinations
4. **Benchmarking:** Compare with and without SIMD features
5. **Deployment:** Document required features in README

## Future Features

Planned features for future releases:

- `rocm`: AMD GPU support via ROCm
- `metal`: Apple Metal GPU support
- `vulkan`: Cross-platform GPU via Vulkan
- `wgpu`: WebGPU support for browsers
- `onnx`: ONNX model format support
- `tflite`: TensorFlow Lite support

## Strict Mode

**Feature:** `BITNET_STRICT_MODE` environment variable
**Purpose:** Prevent silent FP32 fallbacks in quantized inference and ensure honest performance claims
**Since:** BitNet.rs v0.1.0 (Issue #453)

### Rationale

**Problem:** Quantized neural network inference can silently fall back to FP32 dequantization when native quantized kernels are unavailable. This produces correct results but misleading performance metrics:

```
Expected: I2S quantized GPU inference → 80 tok/s
Reality:  FP32 CPU fallback → 12 tok/s (claimed as "quantized")
Impact:   Production deployment expects 80 tok/s, gets 12 tok/s
```

**Solution:** Strict mode implements three-tier validation to detect and prevent silent fallbacks:

1. **Tier 1 (Development):** Debug assertions panic immediately on fallback detection
2. **Tier 2 (Production):** Strict mode returns `Err(BitNetError::StrictMode(...))` instead of falling back
3. **Tier 3 (Verification):** Receipt validation ensures claims match evidence

### Design Goals

1. **Honest Receipts:** Performance claims backed by actual computation paths
2. **Early Detection:** Catch fallbacks during development (debug assertions)
3. **Production Safety:** Prevent silent degradation in production deployments
4. **Audit Trail:** Receipt validation provides post-inference verification
5. **Zero Overhead:** <1% performance impact in release builds

### Use Cases

**Development (Tier 1):**
```bash
# Debug builds catch fallbacks immediately
cargo test --no-default-features --features cpu -p bitnet-inference

# If fallback occurs:
# thread 'test' panicked at 'fallback to FP32 in debug mode: layer=blk.0.attn_q, qtype=I2S, reason=kernel_unavailable'
```

**Production (Tier 2):**
```bash
# Strict mode prevents silent fallbacks
BITNET_STRICT_MODE=1 \
cargo run --release -p bitnet-cli --no-default-features --features cpu -- \
  infer --model model.gguf --prompt "Test" --max-tokens 16

# If kernel unavailable: Fails with detailed error
# Otherwise: Succeeds with guaranteed quantized computation
```

**Verification (Tier 3):**
```bash
# Verify receipts match claims
cargo run -p xtask -- benchmark --model model.gguf --tokens 128
cargo run -p xtask -- verify-receipt --require-quantized-kernels ci/inference.json

# Validates:
# - compute_path="real" has quantized kernel IDs
# - GPU claims use GPU kernels (not CPU fallback)
# - Performance metrics are realistic (not suspicious)
```

### Three-Tier Validation Strategy

**Tier 1: Debug Assertions**
- **When:** Compile-time (debug builds only)
- **Where:** `QuantizedLinear::forward`, `BitNetAttention::compute_qkv_projections`
- **Behavior:** Panic with detailed message
- **Overhead:** Zero in release builds (compiled out)
- **Purpose:** Catch issues during development

**Tier 2: Strict Mode Enforcement**
- **When:** Runtime (release builds with `BITNET_STRICT_MODE=1`)
- **Where:** Quantization fallback paths in inference layers
- **Behavior:** Return `Err(BitNetError::StrictMode(...))`
- **Overhead:** <1% (single boolean check per forward pass)
- **Purpose:** Prevent silent fallbacks in production

**Tier 3: Receipt Validation**
- **When:** Post-inference (offline verification)
- **Where:** `xtask verify-receipt` command
- **Behavior:** Exit code 1 if claims don't match evidence
- **Overhead:** Zero (verification happens after inference)
- **Purpose:** Audit trail for performance baselines

### Configuration

**Primary Strict Mode:**
```bash
export BITNET_STRICT_MODE=1  # Enables all checks
```

**Granular Controls:**
```bash
export BITNET_STRICT_FAIL_ON_MOCK=1              # Fail on mock computation
export BITNET_STRICT_REQUIRE_QUANTIZATION=1     # Require real quantization kernels
export BITNET_STRICT_VALIDATE_PERFORMANCE=1     # Validate performance metrics
export BITNET_CI_ENHANCED_STRICT=1              # CI enhanced logging
```

### Error Messages

Strict mode errors provide actionable debugging context:

```
Error: Strict mode: FP32 fallback rejected - qtype=I2S, device=Cuda(0), layer_dims=[2048, 2048], reason=kernel_unavailable
```

**Error includes:**
- Quantization type that was attempted (I2S, TL1, TL2)
- Device where inference was attempted (CPU, GPU)
- Layer dimensions (for debugging model architecture)
- Specific reason for fallback (kernel_unavailable, device_mismatch, gpu_oom, etc.)

### Receipt Honesty Validation

Strict mode extends to receipt verification, ensuring performance claims are backed by evidence:

**Quantized Kernel Patterns:**
- **GPU:** `gemm_*`, `i2s_gpu_*`, `wmma_*`
- **CPU (I2S):** `i2s_gemv`, `quantized_matmul_i2s`
- **CPU (TL1/ARM):** `tl1_neon_*`, `tl1_lookup_*`
- **CPU (TL2/x86):** `tl2_avx_*`, `tl2_avx512_*`

**Fallback Kernel Patterns:**
- **Dequantization:** `dequant_*`
- **FP32 Computation:** `fp32_matmul`, `fp32_gemm`
- **Generic Fallback:** `fallback_*`, `scalar_*`
- **Mock/Test:** `mock_*`, `test_stub`

**Validation:**
```bash
# Verify quantized kernels used
cargo run -p xtask -- verify-receipt --require-quantized-kernels ci/inference.json

# Verify GPU kernels for GPU claims
cargo run -p xtask -- verify-receipt --require-gpu-kernels ci/inference.json

# Validate performance metrics
cargo run -p xtask -- verify-receipt --validate-performance ci/inference.json
```

### Integration with Deterministic Inference

Combine strict mode with deterministic inference for maximum reproducibility:

```bash
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

cargo run -p bitnet-cli --no-default-features --features cpu -- \
  infer --model model.gguf --prompt "Test" --max-tokens 16 --seed 42

# Outputs will be:
# 1. Identical across runs (deterministic)
# 2. Using real quantized kernels (strict mode)
# 3. Verified via receipt (honest computation)
```

### Related Documentation

- **Tutorial:** [Getting Started with Strict Mode](../tutorials/strict-mode-quantization-validation.md)
- **How-To:** [Running Strict Mode Validation Workflows](../how-to/strict-mode-validation-workflows.md)
- **How-To:** [Verifying Receipt Honesty](../how-to/receipt-verification.md)
- **Reference:** [Quantization Support - Strict Mode](../reference/quantization-support.md#strict-quantization-guards)
- **Reference:** [Environment Variables - Strict Mode](../environment-variables.md#strict-mode-variables)
- **Reference:** [Validation Gates - Receipt Honesty](../reference/validation-gates.md#receipt-honesty-validation)
- **Explanation:** [Strict Quantization Guards Specification](./strict-quantization-guards.md)

### Trade-offs

**Benefits:**
- Honest performance baselines (no false claims)
- Early detection of fallback scenarios (debug assertions)
- Production safety (strict mode enforcement)
- Audit trail (receipt validation)
- Minimal overhead (<1% in release builds)

**Costs:**
- Requires explicit feature flags (`--features cpu|gpu`)
- May fail in environments without proper kernel support
- CI pipelines need updated workflows
- Development must handle strict mode errors

**Recommendation:** Enable strict mode in production deployments and CI/CD pipelines. Disable only when explicitly testing fallback behavior.

## Contributing

When adding new features:

1. Update this document
2. Add feature to relevant Cargo.toml files
3. Gate code with `#[cfg(feature = "name")]`
4. Add tests for feature-gated code
5. Update CI to test new feature
6. If adding runtime guards, consider strict mode integration
