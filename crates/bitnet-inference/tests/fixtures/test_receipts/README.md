# Test Receipt Fixtures for Issue #453

This directory contains sample receipt JSON files for testing strict quantization guard validation (AC6).

## Files

### `valid_i2s.json`
**Purpose:** Valid receipt with native quantized I2S GPU kernels (schema v1.1.0)

**Key Features:**
- `kernel_path: "native_quantized"`
- GPU quantized kernel IDs: `gemm_fp16`, `i2s_gpu_quantize`, `wmma_matmul`
- `fallback_count: 0`
- `compute_path: "real"`

**Expected Validation:** ✅ PASS

---

### `fallback_fp32.json`
**Purpose:** Valid receipt with explicit FP32 fallback (honest reporting)

**Key Features:**
- `kernel_path: "fp32_fallback"`
- Fallback kernel IDs: `dequant_i2s`, `fp32_matmul`
- `fallback_count: 16`
- Lower tokens_per_second (35.0 vs 87.5)

**Expected Validation:** ✅ PASS (explicit fallback is acceptable)

---

### `invalid_false_claim.json`
**Purpose:** Invalid receipt with false quantization claim

**Key Features:**
- `kernel_path: "native_quantized"` (CLAIMS quantization)
- Fallback kernel IDs: `dequant_i2s`, `fp32_matmul`, `fallback_gemm` (ACTUAL fallback)
- Mismatch between claim and evidence

**Expected Validation:** ❌ FAIL (kernel_path mismatch)

---

### `v1_0_backward_compat.json`
**Purpose:** Backward compatibility test with schema v1.0.0

**Key Features:**
- `schema_version: "1.0.0"`
- No `kernel_path` field (must infer from `kernels` array)
- Contains quantized kernel IDs: `gemm_fp16`, `i2s_gpu_quantize`

**Expected Validation:** ✅ PASS (infer native_quantized from kernels)

---

### `cpu_quantized.json`
**Purpose:** Valid CPU inference with quantized kernels

**Key Features:**
- `backend: "cpu"`
- CPU quantized kernel IDs: `i2s_gemv`, `quantized_matmul_i2s`, `tl2_avx_matmul`
- Mixed quantization types: I2S + TL2

**Expected Validation:** ✅ PASS

---

## Usage in Tests

### Loading Fixtures
```rust
use std::fs;

let receipt_json = fs::read_to_string(
    "tests/fixtures/test_receipts/valid_i2s.json"
)?;
let receipt: Receipt = serde_json::from_str(&receipt_json)?;
```

### Validation Testing
```rust
// AC6: Valid receipt test
let receipt = load_receipt("tests/fixtures/test_receipts/valid_i2s.json")?;
let result = verify_quantization_claims(&receipt);
assert!(result.is_ok());

// AC6: Invalid receipt test
let receipt = load_receipt("tests/fixtures/test_receipts/invalid_false_claim.json")?;
let result = verify_quantization_claims(&receipt);
assert!(result.is_err());
```

---

## Kernel ID Naming Conventions

### Quantized Kernels (Native 1/2-bit Arithmetic)
**GPU:**
- `gemm_*` - GPU GEMM kernels (FP16/BF16 with quantized weights)
- `wmma_*` - Tensor Core kernels (mixed precision)
- `i2s_gpu_*` - I2S GPU quantization operations
- `tl1_gpu_*` - TL1 GPU quantization
- `tl2_gpu_*` - TL2 GPU quantization

**CPU:**
- `i2s_gemv` - I2S CPU GEMV (SIMD-optimized)
- `tl1_neon_*` - ARM NEON TL1 kernels
- `tl2_avx_*` - x86 AVX TL2 kernels
- `quantized_matmul_*` - Generic quantized matmul

### FP32 Fallback Kernels (Dequantization + FP32 Arithmetic)
- `dequant_*` - Explicit dequantization to FP32
- `fp32_matmul` - Standard FP32 matrix multiplication
- `scalar_*` - Scalar fallback (no SIMD, no quantization)
- `fallback_*` - Explicit fallback naming
- `mock_*` - Mock kernels (testing only)

---

## Schema Version Compatibility

### v1.0.0 (Existing)
- No `kernel_path` field
- No `quantization` section
- Inference required from `kernels` array

### v1.1.0 (New - Issue #453)
- Optional `kernel_path` field: `"native_quantized"` | `"fp32_fallback"`
- Optional `quantization` section with `types_used`, `fallback_count`, `device_aware_selection`
- Backward compatible (old readers ignore new fields)

---

## Related Documentation
- **Feature Spec:** `docs/explanation/strict-quantization-guards.md#ac6-receipt-validation`
- **API Contracts:** `docs/reference/strict-mode-api.md#receipt-schema-v110`
- **Validation Logic:** `xtask/src/main.rs::verify_quantization_claims`
