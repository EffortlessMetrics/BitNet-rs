# Test Fixtures for Issue #453: Strict Quantization Guards

This directory contains comprehensive test fixtures for validating strict quantization guards in BitNet-rs neural network inference.

## Fixture Organization

### Quantization Test Data (`quantization_test_data.rs`)

Realistic quantization matrices with ground truth data for I2S, TL1, and TL2 algorithms.

**Key Functions:**
- `i2s_matrix_small()` - 128×128 I2S matrix for unit testing
- `i2s_matrix_medium()` - 512×512 I2S matrix for integration testing
- `i2s_matrix_large()` - 2048×2048 I2S matrix for stress testing
- `i2s_matrix_fallback_scenario()` - I2S with unavailable kernel (strict mode rejection)
- `i2s_gpu_matrix_fp16()` - GPU I2S with FP16 mixed precision (1024×1024)
- `i2s_gpu_matrix_bf16()` - GPU I2S with BF16 mixed precision (1024×1024)
- `tl1_matrix_neon()` - TL1 ARM NEON table lookup (256×256)
- `tl1_matrix_fallback()` - TL1 with unavailable NEON kernel
- `tl2_matrix_avx2()` - TL2 x86 AVX2 table lookup (512×512)
- `tl2_matrix_avx512()` - TL2 x86 AVX-512 table lookup (1024×1024)
- `tl2_matrix_fallback()` - TL2 with unavailable AVX kernel

**Ground Truth Data:**
- `ground_truth_fp32_small()` - FP32 reference weights (128×128)
- `ground_truth_fp32_large()` - FP32 reference weights (2048×2048)

**Accuracy Metrics:**
- `I2S_ACCURACY_METRICS` - 99.8%+ correlation, MSE ≤ 1e-3
- `TL1_ACCURACY_METRICS` - 99.6%+ correlation, MSE ≤ 1e-2
- `TL2_ACCURACY_METRICS` - 99.6%+ correlation, MSE ≤ 1e-2

**Feature Gates:**
- `#[cfg(feature = "cpu")]` - CPU quantization matrices
- `#[cfg(feature = "gpu")]` - GPU quantization matrices with mixed precision

**Usage Example:**
```rust
use crate::fixtures::quantization_test_data::*;

let matrix = i2s_matrix_medium();
assert_eq!(matrix.shape, (512, 512));
assert_eq!(matrix.quantization_type, QuantizationType::I2S);
assert!(matrix.target_correlation >= 0.9985);
```

---

### Device Capabilities (`device_capabilities.rs`)

Mock GPU/CPU devices with realistic compute capabilities for testing kernel availability and fallback scenarios.

**GPU Devices:**
- `nvidia_gtx_1080ti()` - Pascal (Compute 6.1), FP16, no Tensor Cores
- `nvidia_rtx_2080ti()` - Turing (Compute 7.5), FP16, Tensor Cores
- `nvidia_a100()` - Ampere (Compute 8.0), FP16/BF16, Tensor Cores
- `nvidia_rtx_3090()` - Ampere (Compute 8.6), FP16/BF16, Tensor Cores
- `nvidia_rtx_4090()` - Ada Lovelace (Compute 8.9), FP16/BF16, Tensor Cores
- `nvidia_h100()` - Hopper (Compute 9.0), FP16/BF16, Tensor Cores
- `nvidia_gpu_unavailable()` - GPU unavailable status for strict mode testing

**CPU Devices:**
- `intel_cpu_avx2()` - Intel Core i7-9700K with AVX2
- `intel_cpu_avx512()` - Intel Xeon Platinum 8280 with AVX-512
- `amd_cpu_avx2()` - AMD Ryzen 9 5950X with AVX2 (no AVX-512)
- `arm_cpu_neon()` - ARM Cortex-A72 with NEON
- `arm_cpu_neon_sve()` - ARM Neoverse V1 with NEON + SVE
- `cpu_no_simd()` - Generic CPU without SIMD (fallback testing)
- `cpu_unavailable()` - CPU unavailable status for strict mode testing

**Fallback Scenarios:**
- `fallback_i2s_kernel_unavailable()` - I2S GPU kernel unavailable
- `fallback_tl1_neon_unavailable()` - TL1 NEON kernel unavailable
- `fallback_tl2_avx_unavailable()` - TL2 AVX kernel unavailable
- `fallback_compute_capability_too_low()` - GPU compute capability < 7.0
- `fallback_insufficient_gpu_memory()` - GPU memory insufficient
- `fallback_unsupported_dimensions()` - Tensor dimensions not aligned
- `fallback_device_mismatch()` - Model on GPU, kernel on CPU
- `fallback_missing_simd_features()` - Required SIMD features unavailable

**Feature Gates:**
- `#[cfg(feature = "cpu")]` - CPU device mocks
- `#[cfg(feature = "gpu")]` - GPU device mocks

**Usage Example:**
```rust
use crate::fixtures::device_capabilities::*;

let gpu = nvidia_a100();
assert_eq!(gpu.compute_capability, (8, 0));
assert!(gpu.supports_bf16);
assert!(supports_bf16_tensor_cores(&gpu));

let scenario = fallback_i2s_kernel_unavailable();
assert_eq!(scenario.expected_strict_mode_behavior, StrictModeBehavior::RejectWithError);
```

---

### Mock Kernels (`mock_kernels.rs`)

Mock kernel registry with ADR-012 naming conventions for testing GPU/CPU kernel selection.

**GPU Kernels (ADR-012 Naming):**
- `gemm_fp16` - FP16 GEMM with I2S quantization (Volta 7.0+)
- `gemm_bf16` - BF16 GEMM with I2S quantization (Ampere 8.0+)
- `wmma_matmul` - Tensor Core FP16 matmul (Volta 7.0+)
- `wmma_bf16` - Tensor Core BF16 matmul (Ampere 8.0+)
- `i2s_gpu_quantize` - I2S GPU quantization (Pascal 6.1+)
- `i2s_gpu_pack` - I2S GPU bit-packing (Pascal 6.1+)
- `i2s_gpu_matmul` - I2S GPU matrix multiplication (Volta 7.0+)
- `tl1_gpu_pack` - TL1 GPU table lookup packing (Pascal 6.1+)
- `tl2_gpu_pack` - TL2 GPU table lookup packing (Pascal 6.1+)
- `cuda_sync` - CUDA stream synchronization

**CPU Kernels (ADR-012 Naming):**
- `i2s_gemv` - I2S CPU GEMV (general vector multiplication)
- `tl1_neon_pack` - TL1 ARM NEON table lookup packing
- `tl1_neon_matmul` - TL1 ARM NEON matrix multiplication
- `tl2_avx_matmul` - TL2 x86 AVX2 table lookup matmul
- `tl2_avx512_pack` - TL2 x86 AVX-512 table lookup packing
- `quantized_matmul_i2s` - CPU I2S quantized matrix multiplication

**Fallback Kernels:**
- `dequant_i2s` - Fallback I2S dequantization (FP32)
- `fp32_matmul` - Fallback FP32 matrix multiplication
- `scalar_matmul` - Fallback scalar matmul (no SIMD)

**Kernel Registry API:**
```rust
use crate::fixtures::mock_kernels::*;

let registry = MockKernelRegistry::new();

// Check kernel availability
assert!(registry.is_kernel_available("gemm_fp16", DeviceType::Gpu));
assert!(registry.has_native_quantized_kernel(QuantizationType::I2S, DeviceType::Gpu));

// Pattern matching helpers
assert!(is_gpu_kernel("gemm_fp16"));
assert!(is_quantized_kernel("i2s_gemv"));
assert!(is_fallback_kernel("dequant_i2s"));

// Generate kernel IDs
let kernel_id = generate_kernel_id(QuantizationType::I2S, DeviceType::Gpu, KernelPrecision::FP16);
assert_eq!(kernel_id, "gemm_fp16");
```

**Feature Gates:**
- `#[cfg(feature = "cpu")]` - CPU kernel definitions
- `#[cfg(feature = "gpu")]` - GPU kernel definitions

---

### Receipt JSON Fixtures (`test_receipts/`)

Realistic inference receipts following schema v1.1.0 with proper kernel IDs.

**Valid Receipts:**
- `valid_i2s.json` - GPU I2S inference with native quantized kernels (87.5 tok/s)
- `cpu_quantized.json` - CPU I2S/TL2 inference with AVX2 (42.3 tok/s)
- `gpu_mixed_precision_fp16.json` - GPU FP16 mixed precision I2S (127.5 tok/s)
- `gpu_mixed_precision_bf16.json` - GPU BF16 mixed precision I2S on A100 (135.2 tok/s)
- `cpu_tl1_neon.json` - ARM NEON TL1 inference (38.7 tok/s)
- `cpu_tl2_avx512.json` - x86 AVX-512 TL2 inference (52.8 tok/s)

**Fallback Receipts:**
- `fallback_fp32.json` - FP32 fallback due to kernel unavailable (35.0 tok/s)
- `invalid_false_claim.json` - False quantization claim (strict mode should reject)

**Backward Compatibility:**
- `v1_0_backward_compat.json` - Schema v1.0.0 receipt for compatibility testing

**Receipt Schema v1.1.0:**
```json
{
  "schema_version": "1.1.0",
  "compute_path": "real",
  "backend": "cuda",
  "kernels": ["gemm_fp16", "i2s_gpu_quantize", "wmma_matmul"],
  "kernel_path": "native_quantized",
  "quantization": {
    "types_used": ["I2S"],
    "fallback_count": 0,
    "device_aware_selection": true,
    "precision": "FP16"
  },
  "performance": {
    "kernel_times_ms": { "gemm_fp16": 5.2, ... },
    "tokens_per_second": 127.5
  }
}
```

**Usage Example:**
```rust
let receipt: InferenceReceipt = serde_json::from_str(
    include_str!("fixtures/test_receipts/gpu_mixed_precision_fp16.json")
)?;
assert_eq!(receipt.schema_version, "1.1.0");
assert!(receipt.kernels.contains(&"gemm_fp16".to_string()));
```

---

### Cross-Validation Reference Data (`crossval/reference_outputs.json`)

Ground truth outputs from Microsoft BitNet C++ reference implementation for numerical accuracy validation.

**Test Cases:**
1. `i2s_cpu_16_token_decode` - I2S CPU 16-token decode (99.85% correlation)
2. `i2s_gpu_fp16_16_token_decode` - I2S GPU FP16 16-token decode (99.78% correlation)
3. `tl1_neon_matmul` - TL1 ARM NEON matmul (99.65% correlation)
4. `tl2_avx2_matmul` - TL2 x86 AVX2 matmul (99.62% correlation)
5. `i2s_attention_qkv_projections` - I2S attention projections (all ≥99.86% correlation)
6. `i2s_fallback_fp32_comparison` - FP32 fallback comparison (99.99% correlation)
7. `i2s_gpu_bf16_tensor_cores` - BF16 Tensor Core inference (99.72% correlation)
8. `deterministic_inference_comparison` - Deterministic inference validation (99.89% correlation)

**Accuracy Thresholds:**
- I2S quantization: ≥99.8% correlation, tolerance 1e-3
- TL1 quantization: ≥99.6% correlation, tolerance 1e-2
- TL2 quantization: ≥99.6% correlation, tolerance 1e-2
- FP16 mixed precision: tolerance 2e-3
- BF16 mixed precision: tolerance 3e-3

**Usage Example:**
```rust
let crossval: serde_json::Value = serde_json::from_str(
    include_str!("fixtures/crossval/reference_outputs.json")
)?;

let test_case = &crossval["test_cases"][0];
assert_eq!(test_case["test_id"], "i2s_cpu_16_token_decode");
assert!(test_case["correlation_score"].as_f64().unwrap() >= 0.998);
```

---

### Mock Quantized Model (`mock_quantized_model.rs`)

Mock implementations of quantized layers for testing without actual model files.

**Components:**
- `MockQuantizedLinear` - Mock quantized linear layer with forward pass
- `MockBitNetAttention` - Mock attention with Q/K/V/O projections
- `MockBitNetModel` - Mock full model with multiple layers
- `MockTokenizer` - Simple mock tokenizer for integration testing
- `MockReceipt` - Mock receipt generation for testing

**Test Helpers:**
- `create_test_model_with_fallback()` - Model with forced fallback scenario
- `create_test_model_quantized()` - Model with all quantized kernels available
- `run_mock_inference()` - Run mock inference loop

---

## Testing Workflow

### 1. Unit Testing (Small Matrices)
```bash
cargo test -p bitnet-inference --no-default-features --features cpu -- test_i2s_small
```

### 2. Integration Testing (Medium Matrices)
```bash
cargo test -p bitnet-inference --no-default-features --features cpu -- test_i2s_medium
```

### 3. GPU Testing (Mixed Precision)
```bash
cargo test -p bitnet-inference --no-default-features --features gpu -- test_i2s_gpu_fp16
```

### 4. Strict Mode Testing (Fallback Rejection)
```bash
BITNET_STRICT_MODE=1 cargo test -p bitnet-inference --no-default-features --features cpu -- test_fallback_rejection
```

### 5. Cross-Validation Testing
```bash
cargo test -p bitnet-inference --no-default-features --features cpu -- test_crossval_correlation
```

### 6. Receipt Validation Testing
```bash
cargo run -p xtask -- verify-receipt
```

---

## Feature Flag Matrix

| Fixture | CPU | GPU | Notes |
|---------|-----|-----|-------|
| `quantization_test_data.rs` | ✅ | ✅ | Full feature coverage |
| `device_capabilities.rs` | ✅ | ✅ | Device-specific mocks |
| `mock_kernels.rs` | ✅ | ✅ | Kernel registry for both devices |
| `test_receipts/*.json` | ✅ | ✅ | Receipts for CPU/GPU inference |
| `crossval/reference_outputs.json` | ✅ | ✅ | Cross-validation for both |
| `mock_quantized_model.rs` | ✅ | ❌ | CPU-only mock model |

---

## Deterministic Testing

All fixtures support deterministic generation with:
- `BITNET_DETERMINISTIC=1` - Enable deterministic mode
- `BITNET_SEED=42` - Set deterministic seed

Example:
```bash
BITNET_DETERMINISTIC=1 BITNET_SEED=42 cargo test -p bitnet-inference
```

---

## Strict Mode Testing

Test strict mode enforcement with:
- `BITNET_STRICT_MODE=1` - Enable strict quantization guards
- Expected behavior: Reject FP32 fallback, return `Err` with detailed context

Example:
```bash
BITNET_STRICT_MODE=1 cargo test -p bitnet-inference -- test_strict_mode_rejection
```

---

## Coverage Summary

- ✅ **I2S Quantization:** Small/medium/large matrices, CPU/GPU, FP16/BF16
- ✅ **TL1 Quantization:** ARM NEON table lookup, fallback scenarios
- ✅ **TL2 Quantization:** x86 AVX2/AVX-512 table lookup
- ✅ **GPU Devices:** Pascal, Turing, Ampere, Ada, Hopper compute capabilities
- ✅ **CPU Devices:** Intel AVX2/AVX-512, AMD AVX2, ARM NEON/SVE
- ✅ **Fallback Scenarios:** Kernel unavailable, device mismatch, SIMD missing
- ✅ **Receipt Validation:** Schema v1.0.0/v1.1.0, kernel ID patterns (ADR-012)
- ✅ **Cross-Validation:** 9 test cases, 99.6-99.99% correlation
- ✅ **Attention Projections:** Q/K/V/O projection quantization validation

---

## References

- **Issue #453:** Strict Quantization Guards specification
- **ADR-012:** Kernel ID naming conventions
- **Schema v1.1.0:** Inference receipt schema with quantization section
- **CLAUDE.md:** BitNet-rs development guide
- **docs/explanation/strict-quantization-guards.md:** Full specification

---

**Last Updated:** 2025-10-14
**Maintainer:** BitNet-rs Test Fixture Architect
