# Quantization Support

This document describes the quantization formats and device-aware acceleration supported by BitNet.rs.

## Supported Quantization Formats

BitNet-rs supports multiple quantization formats with advanced device-aware acceleration:

### I2_S - Native Rust Implementation (Production Ready - Issue #261)

- Native Rust implementation with intelligent GPU/CPU selection and automatic fallback
- Device-aware quantization with CUDA kernel acceleration and CPU SIMD optimization
- **Accuracy**: ≥99.8% correlation with FP32 reference (production requirement, validated in AC3)
- **Performance**: CPU 10-20 tok/s (architecture-dependent: AVX-512 > AVX2 > NEON), GPU 50-100 tok/s with mixed precision
- 2-bit signed quantization with optimized bit-packing (4 values per byte)
- **Strict Mode**: Use `BITNET_STRICT_MODE=1` to prevent mock fallbacks and ensure real quantized computation
- **Real Computation**: Native quantized GEMV kernel eliminates FP32 dequantization staging (Issue #261 - AC3)
- **QLinear Layer Integration**: Replaces standard Linear layers in transformer architecture (Issue #261 - AC5)

### TL1 - Table Lookup Quantization (ARM Optimized - Issue #261)

- Table lookup quantization optimized for ARM NEON architecture
- **Accuracy**: ≥99.6% correlation with FP32 reference (validated in AC3)
- **Performance**: 12-18 tok/s on ARM NEON (realistic baseline from AC7)
- **Device-Aware Selection**: Automatic ARM NEON vectorization with scalar fallback
- Memory-efficient lookup tables (16-256 entries, cache-friendly)
- Parallel processing with configurable block sizes
- **Real Computation**: Direct table lookup matmul without FP32 staging (Issue #261)
- **Safe LUT Index Calculation**: Uses `bitnet_kernels::tl_lut::lut_index()` with checked arithmetic and overflow protection

### TL2 - Advanced Table Lookup (x86 Optimized - Issue #261)

- Advanced table lookup quantization optimized for x86 AVX2/AVX-512
- **Accuracy**: ≥99.6% correlation with FP32 reference (validated in AC3)
- **Performance**: 10-15 tok/s on x86 AVX (realistic baseline from AC7)
- **SIMD Optimization**: AVX2 (32-byte) and AVX-512 (64-byte) vectorization
- Enhanced vectorized operations (256-4096 entry tables) for large tensor processing
- CPU feature detection with graceful fallback to scalar implementation
- **Real Computation**: Direct table lookup matmul without FP32 staging (Issue #261)
- **Safe LUT Index Calculation**: Uses `bitnet_kernels::tl_lut::lut_index()` with checked arithmetic and overflow protection

### IQ2_S - GGML-Compatible

- GGML-compatible quantization with 82-byte block layout and 4-level [-2,-1,1,2] mapping

### Standard Formats (Planned)

- Q4_0, Q5_0, Q8_0, etc. (planned for future releases)

## Table Lookup (TL) Helper API

The `bitnet_kernels::tl_lut` module provides safe, bounds-checked index calculation for TL1/TL2 quantization kernels.

### `lut_index` Function

Calculate validated index into table lookup buffer with overflow protection.

**Signature:**

```rust
pub fn lut_index(
    block_idx: usize,
    elem_in_block: usize,
    block_bytes: usize,
    elems_per_block: usize,
    lut_len: usize,
) -> Result<usize>
```

- **Parameters:**

- `block_idx`: Block index in quantized buffer
- `elem_in_block`: Element position within block (0..elems_per_block)
- `block_bytes`: Size of each block in bytes
- `elems_per_block`: Number of elements per quantized block
- `lut_len`: Total length of LUT buffer (for bounds checking)

**Returns:** Validated LUT index or error if overflow/out-of-bounds

- **Safety Guarantees:**

- Validates `elem_in_block < elems_per_block` (bounds check)
- Uses checked arithmetic to prevent integer overflow
- Validates final index `< lut_len` before returning
- 100% mutation testing coverage (6/6 mutants killed, Issue #462)

**Example Usage:**

```rust
use bitnet_kernels::tl_lut::lut_index;

// Calculate LUT index for block 0, element 0
let idx = lut_index(0, 0, 32, 128, 1024)?;
assert_eq!(idx, 0);

// Calculate LUT index for block 1, element 8
// Formula: 1 * 32 + (8 / 8) = 32 + 1 = 33
let idx = lut_index(1, 8, 32, 128, 1024)?;
assert_eq!(idx, 33);

// Bounds check prevents out-of-range access
let result = lut_index(0, 128, 32, 128, 1024);
#assert!(result.is_err()); // elem_in_block >= elems_per_block
#```

**Testing Commands:**
```bash
# Run TL LUT helper tests
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu tl_lut

# Specific test cases
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_lut_index_basic
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_lut_index_overflow_detection
cargo test --no-default-features -p bitnet-kernels --no-default-features --features cpu test_lut_index_boundary_validation
```

**See also:** Issue #462 for TL LUT helper implementation and mutation testing results.

## Device-Aware Operations

All quantizers support device-aware operations with:

- **Automatic GPU acceleration**: CUDA kernels with performance monitoring (50-100 tok/s)
- **Transparent CPU fallback**: Graceful degradation with maintained accuracy (10-20 tok/s)
- **Memory optimization**: GPU memory leak detection and efficient allocation
- **Feature gating**: Proper `#[cfg(feature = "gpu")]` guards for CPU-only builds
- **Strict Mode Enforcement**: `BITNET_STRICT_MODE=1` prevents mock fallbacks
- **FFI Bridge Support**: C++ kernel integration for I2S, TL1, and TL2 quantization (requires `--features ffi`)
- **Cross-Validation**: <5% performance variance from C++ reference implementation

## FFI Quantization Bridge

The FFI bridge enables gradual migration from C++ to Rust while maintaining functionality:

- **Quantization Types**: Full support for I2S, TL1, and TL2 via C++ kernels
- **Performance Comparison**: Built-in tools to compare FFI vs Rust quantization
- **Migration Path**: Systematic approach to replace C++ kernels with native Rust
- **Safety**: Safe Rust wrappers with proper error handling and memory management
- **Testing**: Comprehensive test suite ensuring FFI/Rust quantization parity

## Mixed Precision GPU Acceleration

BitNet.rs provides native CUDA mixed precision support for enhanced GPU performance:

- ### Supported Precision Modes

- **FP32**: Full precision (reference implementation)
- **FP16**: Half-precision floating point with Tensor Core acceleration (compute capability 6.1+)
- **BF16**: Brain floating point format for modern architectures (compute capability 8.0+)
- **Auto**: Automatic precision selection based on device capabilities

- ### Device-Aware Precision Selection

- **Automatic Detection**: Hardware capability detection determines optimal precision
- **Device ID Tracking**: GPU kernels expose device ID for multi-GPU debugging scenarios (PR #201)
- **Capability Querying**: Direct access to FP16/BF16 support via `supports_fp16()` and `supports_bf16()` methods (PR #201)
- **Graceful Fallback**: Automatic CPU fallback when GPU operations fail
- **Performance Monitoring**: Comprehensive metrics for each precision mode
- **Memory Tracking**: GPU memory allocation and deallocation monitoring
- **Tensor Core Optimization**: Leverages WMMA API for maximum performance (CC 7.0+)

- ### Mixed Precision Features

- **Native CUDA Kernels**: Custom PTX kernels optimized for each precision mode
- **Matrix Multiplication**: Optimized matmul operations with device-specific launch parameters
- **Precision Conversion**: Efficient FP32↔FP16↔BF16 conversion utilities
- **Memory Optimization**: Vectorized memory operations and bandwidth optimization
- **Error Handling**: Comprehensive error propagation with detailed diagnostics

## Testing Commands

### Device-Aware Quantization Testing

```bash
# Test device-aware quantization with strict mode (prevents mock fallbacks)
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-quantization --no-default-features --features gpu test_dequantize_cpu_and_gpu_paths

# GPU kernel validation with numerical accuracy testing
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_gpu_vs_cpu_quantization_accuracy

# Enhanced GPU validation with performance metrics and error handling
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_cuda_validation_comprehensive

# Validate quantization accuracy targets (I2S >99.8%, TL1/TL2 >99.6%)
cargo test --no-default-features -p bitnet-quantization --no-default-features --features cpu test_quantization_accuracy_targets
```

### Mixed Precision Testing

```bash
# Test mixed precision with strict mode (no mock GPU fallbacks)
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_kernel_creation

# Test FP16/BF16 matrix multiplication accuracy against FP32 reference
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_mixed_precision_matmul_accuracy

# Test precision mode validation and automatic fallback
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_precision_mode_validation

# Benchmark mixed precision performance with strict mode (realistic baselines)
BITNET_STRICT_MODE=1 cargo bench --no-default-features -p bitnet-kernels --bench mixed_precision_bench --no-default-features --features gpu

# Test device-aware precision selection and optimization
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-kernels --no-default-features --features gpu test_precision_detection_optimization
```

### FFI Quantization Testing

```bash
# FFI quantization bridge validation with strict mode
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-kernels --features ffi test_ffi_quantize_matches_rust

# FFI kernel creation and availability testing
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-kernels --features ffi test_ffi_kernel_creation

# FFI performance comparison against C++ reference (cross-validation)
BITNET_STRICT_MODE=1 cargo test --no-default-features -p bitnet-kernels --features ffi --release test_performance_comparison_structure

# Cross-validation with C++ reference implementation
BITNET_GGUF="path/to/model.gguf" BITNET_STRICT_MODE=1 cargo run -p xtask -- crossval
```

### SIMD Testing

```bash
# SIMD kernel validation and performance testing
cargo test --no-default-features -p bitnet-quantization --test simd_compatibility --no-default-features --features cpu
cargo bench --no-default-features -p bitnet-quantization --bench simd_comparison --no-default-features --features cpu

# SIMD vs scalar parity testing
cargo test --no-default-features --features cpu -p bitnet-quantization test_i2s_simd_scalar_parity
cargo test --no-default-features --features cpu -p bitnet-quantization test_simd_performance_baseline
```

## Strict Mode Enforcement (Issue #261 - AC2, AC6)

BitNet.rs provides comprehensive strict mode controls to eliminate mock inference paths and ensure production-ready quantized computation:

### Primary Strict Mode Configuration

```bash
# Enable strict mode for production deployments
BITNET_STRICT_MODE=1 cargo run -p xtask -- infer --model model.gguf --prompt "Test"

# This enables ALL strict mode checks:
# - fail_on_mock: Fails when mock computation detected
# - require_quantization: Requires real I2S/TL1/TL2 kernels
# - validate_performance: Rejects suspicious metrics (>150 tok/s)
```

### Granular Strict Mode Controls

```bash
# Fail immediately on mock detection (Issue #261 - AC2)
BITNET_STRICT_FAIL_ON_MOCK=1 \
cargo test -p bitnet-inference --features cpu

# Require real quantization kernels (Issue #261 - AC3)
BITNET_STRICT_REQUIRE_QUANTIZATION=1 \
cargo test -p bitnet-quantization --features cpu

# Validate performance metrics (Issue #261 - AC6)
BITNET_STRICT_VALIDATE_PERFORMANCE=1 \
cargo run -p xtask -- benchmark --model model.gguf

# CI enhanced strict mode (Issue #261 - AC6)
CI=1 BITNET_CI_ENHANCED_STRICT=1 BITNET_STRICT_MODE=1 \
cargo test --workspace --features cpu
```

### Strict Mode API Usage

```rust
use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};

// Production inference with strict mode
std::env::set_var("BITNET_STRICT_MODE", "1");
let enforcer = StrictModeEnforcer::new_detailed();

// Validate inference path (fails on mock usage)
enforcer.validate_inference_path(&inference_path)?;

// Validate quantization kernel availability
enforcer.validate_kernel_availability(&kernel_scenario)?;

// Validate performance metrics (rejects >150 tok/s as suspicious)
enforcer.validate_performance_metrics(&performance_metrics)?;
```

### Performance Validation Thresholds

Strict mode validates performance metrics against realistic baselines:

| Metric | Threshold | Reasoning |
|--------|-----------|-----------|
| Throughput | ≤150 tok/s | Values >150 tok/s flag potential mock computation |
| Computation Type | Must be `Real` | Rejects `Mock` computation type |
| Quantization Accuracy | I2S ≥99.8%, TL1/TL2 ≥99.6% | Validates against FP32 reference |
| Device Utilization | GPU >80% | Ensures efficient GPU utilization |

### CI Integration

```yaml
# .github/workflows/performance-tracking.yml
- name: Run strict mode tests
  env:
    BITNET_STRICT_MODE: "1"
    BITNET_CI_ENHANCED_STRICT: "1"
    BITNET_DETERMINISTIC: "1"
    BITNET_SEED: "42"
  run: |
    cargo test --workspace --features cpu
    cargo run -p xtask -- crossval
```

## Strict Quantization Guards (Issue #453)

BitNet.rs provides comprehensive strict quantization guards to prevent silent FP32 fallback in quantized layers.

This three-tier validation strategy ensures production-grade quantized inference with honest performance claims.

### Three-Tier Validation Strategy

#### Tier 1: Debug Assertions (Development)

- **Purpose:** Catch FP32 fallback immediately during development
- **Scope:** Debug builds only (`#[cfg(debug_assertions)]`)
- **Behavior:** Panic with detailed error message
- **Overhead:** Zero in release builds (compiled out)

```bash
# Debug builds automatically include assertions
cargo test --no-default-features --features cpu -p bitnet-inference

# If fallback occurs:
# thread 'test' panicked at 'fallback to FP32 in debug mode: layer=blk.0.attn_q, qtype=I2S, reason=kernel_unavailable'
```

#### Tier 2: Strict Mode Enforcement (Production)

- **Purpose:** Reject FP32 fallback in production deployments
- **Scope:** Release builds with `BITNET_STRICT_MODE=1`
- **Behavior:** Return `Err(BitNetError::StrictMode(...))`
- **Overhead:** <1% (single boolean check per forward pass)

```bash
# Production inference with strict mode
BITNET_STRICT_MODE=1 \
cargo run --release -p bitnet-cli --no-default-features --features cpu -- \
  infer \
  --model model.gguf \
  --prompt "Test" \
  --max-tokens 16

# If kernel unavailable: Fails with detailed error
# Otherwise: Succeeds with guaranteed quantized computation
```

#### Tier 3: Receipt Validation (Verification)

- **Purpose:** Validate receipts accurately reflect computation path
- **Scope:** Post-inference verification (`xtask verify-receipt`)
- **Behavior:** Exit code 1 if receipt claims don't match kernel IDs
- **Overhead:** Zero (offline verification)

```bash
# Run benchmark
cargo run -p xtask -- benchmark --model model.gguf --tokens 128

# Verify receipt honesty
cargo run -p xtask -- verify-receipt ci/inference.json

# Checks:
# - compute_path="real" matches actual kernel IDs
# - GPU claims require GPU kernel IDs (gemm_*, i2s_gpu_*)
# - CPU claims require CPU kernel IDs (i2s_gemv, tl1_neon_*, tl2_avx_*)
```

### Strict Mode Configuration

**Primary Strict Mode:**

```bash
# Enable all strict mode checks
export BITNET_STRICT_MODE=1

# This enables:
# - fail_on_mock: Fails when mock computation detected
# - require_quantization: Requires real I2S/TL1/TL2 kernels
# - enforce_quantized_inference: Rejects FP32 fallback in quantized layers
# - validate_performance: Rejects suspicious metrics (>150 tok/s)
```

**Granular Strict Mode Controls:**

```bash
# Fail immediately on mock detection (Issue #453 - AC2)
export BITNET_STRICT_FAIL_ON_MOCK=1

# Require real quantization kernels (Issue #453 - AC3)
export BITNET_STRICT_REQUIRE_QUANTIZATION=1

# Validate performance metrics (Issue #453 - AC6)
export BITNET_STRICT_VALIDATE_PERFORMANCE=1

# CI enhanced strict mode (Issue #453 - AC6)
export CI=1
export BITNET_CI_ENHANCED_STRICT=1
```

### Strict Mode Error Messages

Strict mode errors provide actionable context for debugging:

```text
Error: Strict mode: FP32 fallback rejected - qtype=I2S, device=Cuda(0), layer_dims=[2048, 2048], reason=kernel_unavailable
       ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^  ^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
       (1)           (2)                     (3)        (4)          (5)                 (6)
```

1. **Strict mode:** Indicates strict mode validation failure
2. **FP32 fallback rejected:** System tried to fall back to FP32 but strict mode prevented it
3. **qtype=I2S:** The quantization type that was attempted
4. **device=Cuda(0):** The device where inference was attempted
5. **layer_dims=[2048, 2048]:** Layer dimensions (in_features × out_features)
6. **reason=kernel_unavailable:** Why fallback was needed

### Common Fallback Reasons and Solutions

| Reason | Meaning | Solution |
|--------|---------|----------|
| `kernel_unavailable` | Feature not compiled | `cargo build --no-default-features --features cpu` or `--features gpu` |
| `device_mismatch` | Tensor on wrong device | Ensure model loaded on same device as inference |
| `unsupported_dimensions` | Layer size not supported | Check model architecture compatibility |
| `gpu_oom` | GPU out of memory | Reduce batch size or use smaller model |
| `simd_unavailable` | SIMD features not detected | Rebuild with `RUSTFLAGS="-C target-cpu=native"` |

### Receipt Honesty Validation

Strict mode extends to receipt validation, ensuring performance claims are backed by evidence:

**Quantized Kernel ID Patterns:**

- **GPU Kernels:** `gemm_*`, `wmma_*`, `cuda_*`, `i2s_gpu_*`, `tl1_gpu_*`, `tl2_gpu_*`
- **CPU Kernels (I2S):** `i2s_gemv`, `i2s_matmul_*`, `quantized_matmul_i2s`
- **CPU Kernels (TL1/ARM):** `tl1_neon_*`, `tl1_lookup_*`
- **CPU Kernels (TL2/x86):** `tl2_avx_*`, `tl2_avx512_*`

**Fallback Kernel ID Patterns:**

- **Dequantization:** `dequant_*`, `dequant_i2s_to_fp32`
- **FP32 Computation:** `fp32_matmul`, `fp32_gemm`
- **Generic Fallback:** `fallback_*`, `scalar_*`
- **Mock/Test:** `mock_*`, `test_stub`

**Validation Commands:**

```bash
# Verify quantized kernels are used
cargo run -p xtask -- verify-receipt --require-quantized-kernels ci/inference.json

# Verify GPU kernels for GPU claims
cargo run -p xtask -- verify-receipt --require-gpu-kernels ci/inference.json

# Validate performance metrics
cargo run -p xtask -- verify-receipt --validate-performance ci/inference.json
```

### Programmatic Usage

```rust
use bitnet_common::strict_mode::{StrictModeConfig, StrictModeEnforcer};
use bitnet_common::{Device, QuantizationType, Result};

// Production inference with strict mode
std::env::set_var("BITNET_STRICT_MODE", "1");
let enforcer = StrictModeEnforcer::new_detailed();

// Validate inference path (fails on mock usage)
enforcer.validate_inference_path(&inference_path)?;

// Validate quantization kernel availability
enforcer.validate_kernel_availability(&kernel_scenario)?;

// Validate quantization fallback (Issue #453 - AC3)
enforcer.validate_quantization_fallback(
    QuantizationType::I2S,
    Device::Cpu,
    &[2048, 2048],  // layer_dims
    "kernel_unavailable"
)?;

// Validate performance metrics (rejects >150 tok/s as suspicious)
enforcer.validate_performance_metrics(&performance_metrics)?;
```

**Integration in Quantized Linear:**

```rust
// crates/bitnet-inference/src/layers/quantized_linear.rs

async fn forward_i2s(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
    let has_native = bitnet_kernels::is_quantized_kernel_available(
        QuantizationType::I2S,
        self.device,
        (self.in_features, self.out_features)
    );

    // Debug assertions (Tier 1 - Issue #453 - AC1)
    #[cfg(debug_assertions)]
    if !has_native {
        panic!("fallback to FP32 in debug mode: layer={}, qtype=I2S, reason=kernel_unavailable", self.name);
    }

    // Strict mode enforcement (Tier 2 - Issue #453 - AC3)
    if !has_native {
        let strict_mode = StrictModeEnforcer::new();
        if strict_mode.get_config().enforce_quantized_inference {
            return Err(BitNetError::StrictMode(format!(
                "FP32 fallback rejected - qtype=I2S, device={:?}, layer_dims=[{}, {}], reason=kernel_unavailable",
                self.device, self.in_features, self.out_features
            )));
        }
    }

    // Use native quantized matmul (no dequantization)
    if has_native {
        self.quantized_matmul_i2s(&input_2d, provider).await
    } else {
        log::warn!("Using FP32 fallback - should not happen in production");
        self.fallback_i2s_matmul(&input_2d).await
    }
}
```

### Testing Strict Mode

**Unit Tests with AC Traceability:**

```bash
# AC1: Debug assertions in QuantizedLinear::forward
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac1_debug_assert_i2s_fallback -- --nocapture

# AC3: Strict mode rejects FP32 fallback
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac3_strict_mode_rejects_fallback -- --nocapture

# AC5: 16-token decode in strict mode
BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac5_16_token_decode_cpu_strict_mode --test strict_quantization_test

# AC6: Receipt validation for quantized computation claims
cargo test -p xtask test_ac6_receipt_quantized_kernels_valid -- --nocapture
```

**Integration Tests:**

```bash
# CPU strict mode validation
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu --test strict_quantization_test

# GPU strict mode validation (requires GPU)
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features gpu --test strict_quantization_test

# Cross-validation with strict mode
BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo run -p xtask -- crossval
```

### Deterministic Inference with Strict Mode

Combine strict mode with deterministic inference for maximum reproducibility:

```bash
# Enable strict mode + deterministic inference
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Run inference
cargo run -p bitnet-cli --no-default-features --features cpu -- \
  infer \
  --model model.gguf \
  --prompt "Test prompt" \
  --max-tokens 16 \
  --seed 42

# Outputs will be:
# 1. Identical across runs (deterministic)
# 2. Using real quantized kernels (strict mode)
# 3. Verified via receipt (honest computation)
```

### Receipt Schema for Strict Mode

Receipts generated with strict mode include additional validation fields:

```json
{
  "schema_version": "1.0.0",
  "backend": "cpu",
  "compute_path": "real",
  "kernels": [
    "i2s_gemv",
    "quantized_matmul_i2s"
  ],
  "tokens_per_second": 18.5,
  "tokens_generated": 128,
  "environment": {
    "BITNET_STRICT_MODE": "1",
    "BITNET_DETERMINISTIC": "1",
    "BITNET_SEED": "42"
  },
  "timestamp": "2025-10-14T12:34:56.789Z"
}
```

For more information, see:

- **Tutorial:** [Getting Started with Strict Mode](../tutorials/strict-mode-quantization-validation.md) - Learning-oriented introduction
- **How-To:** [Running Strict Mode Validation Workflows](../how-to/strict-mode-validation-workflows.md) - Problem-oriented workflows
- **How-To:** [Verifying Receipt Honesty](../how-to/receipt-verification.md) - Receipt validation guide
- **Reference:** [Environment Variables](../environment-variables.md#strict-mode-variables) - Complete strict mode variable documentation
- **Reference:** [Validation Gates](./validation-gates.md#receipt-honesty-validation) - Receipt honesty technical reference
- **Explanation:** [Strict Mode Rationale](../explanation/FEATURES.md#strict-mode) - Design rationale
- **Explanation:** [Strict Quantization Guards Specification](../explanation/strict-quantization-guards.md) - Complete feature specification
- **Development:** [GPU Development Guide](../development/gpu-development.md) - GPU-specific quantization details
- **Development:** [Build Commands](../development/build-commands.md) - Build commands for different quantization features
- **Architecture:** [FFI Threading Architecture](../ffi-threading-architecture.md) - FFI bridge details
