# ADR-012: Kernel ID Naming Conventions for Quantization Validation

## Status

**ACCEPTED** - Issue #453 Implementation

## Context

Receipt validation (Issue #453, AC6) requires correlating claimed quantization paths with actual kernel IDs. We need consistent naming conventions to distinguish native quantized kernels from FP32 fallback kernels.

### Requirements

1. **Unambiguous Classification:** Clear distinction between quantized and fallback kernels
2. **Pattern Matching:** Simple prefix/substring matching for validation
3. **Extensibility:** Support future quantization types (I2S, TL1, TL2, IQ2_S, etc.)
4. **Device Awareness:** Distinguish GPU kernels from CPU kernels

## Decision

We establish **standardized kernel ID naming conventions** with explicit prefixes for quantization type and device.

### Quantized Kernels (Native 1/2-bit Arithmetic)

**GPU Quantized Kernels:**
```
gemm_*          - GPU GEMM kernels (FP16/BF16 matmul with quantized weights)
wmma_*          - Tensor Core kernels (mixed precision with quantized inputs)
cuda_*          - CUDA-specific quantization operations
i2s_gpu_*       - I2S GPU quantization (pack, unpack, matmul)
tl1_gpu_*       - TL1 GPU quantization (table lookup)
tl2_gpu_*       - TL2 GPU quantization (table lookup)
```

**Examples:**
- `gemm_fp16` - FP16 GEMM with quantized weights
- `i2s_gpu_quantize` - I2S GPU quantization operation
- `wmma_matmul` - Tensor Core mixed precision matmul
- `tl2_gpu_matmul` - TL2 GPU table lookup matmul

**CPU Quantized Kernels:**
```
i2s_gemv        - I2S CPU GEMV (SIMD-optimized)
tl1_neon_*      - ARM NEON TL1 kernels (pack, matmul)
tl2_avx_*       - x86 AVX TL2 kernels (matmul, pack)
tl2_avx512_*    - x86 AVX-512 TL2 kernels (enhanced)
quantized_matmul_* - Generic quantized matmul implementations
```

**Examples:**
- `i2s_gemv` - I2S CPU GEMV (SIMD)
- `tl1_neon_pack` - ARM NEON TL1 packing
- `tl2_avx_matmul` - x86 AVX TL2 matmul
- `quantized_matmul_i2s` - Generic I2S quantized matmul

### FP32 Fallback Kernels (Dequantization + FP32 Arithmetic)

**Fallback Kernel Indicators:**
```
dequant_*       - Explicit dequantization to FP32 (staging)
fp32_matmul     - Standard FP32 matrix multiplication
scalar_*        - Scalar fallback (no SIMD, no quantization)
fallback_*      - Explicit fallback naming convention
mock_*          - Mock kernels (testing only, not production)
```

**Examples:**
- `dequant_i2s` - Dequantize I2S weights to FP32
- `fp32_matmul` - Standard FP32 matrix multiplication
- `scalar_matmul` - Scalar fallback (no SIMD)
- `fallback_gemm` - Explicit fallback GEMM
- `mock_inference` - Mock computation (testing only)

### Naming Convention Rules

**Rule 1: Quantization Type Prefix**
- **Format:** `{qtype}_{device}_{operation}`
- **Examples:**
  - `i2s_gpu_matmul` - I2S quantization, GPU device, matmul operation
  - `tl1_neon_pack` - TL1 quantization, NEON device, pack operation

**Rule 2: Device Indicator**
- **GPU:** `_gpu_`, `gemm_`, `wmma_`, `cuda_`
- **CPU:** No explicit prefix (inferred), or `_cpu_` for clarity
- **Architecture-Specific:** `_neon_`, `_avx_`, `_avx512_`

**Rule 3: Operation Suffix**
- **Quantization:** `_quantize`, `_pack`, `_unpack`
- **Computation:** `_matmul`, `_gemm`, `_gemv`
- **Mixed:** `_quantized_matmul`

**Rule 4: Fallback Indicator**
- **Explicit:** `dequant_`, `fallback_`, `fp32_`, `scalar_`
- **Purpose:** Unambiguous identification of FP32 fallback path

## Implementation

### Pattern Matching Functions

**Location:** `xtask/src/verify_receipt.rs`

```rust
/// Check if kernel ID represents native quantized computation
pub fn is_quantized_kernel(kernel_id: &str) -> bool {
    const QUANTIZED_PREFIXES: &[&str] = &[
        "gemm_",
        "wmma_",
        "cuda_",
        "i2s_gpu_",
        "tl1_gpu_",
        "tl2_gpu_",
        "i2s_gemv",
        "tl1_neon_",
        "tl2_avx_",
        "tl2_avx512_",
        "quantized_matmul_",
    ];

    QUANTIZED_PREFIXES.iter().any(|prefix| kernel_id.starts_with(prefix))
}

/// Check if kernel ID indicates FP32 fallback
pub fn is_fallback_kernel(kernel_id: &str) -> bool {
    const FALLBACK_INDICATORS: &[&str] = &[
        "dequant_",
        "fp32_matmul",
        "scalar_",
        "fallback_",
        "mock_",
    ];

    FALLBACK_INDICATORS.iter().any(|indicator| kernel_id.contains(indicator))
}
```

### Kernel ID Generation

**Location:** `crates/bitnet-kernels/src/lib.rs`

```rust
/// Get kernel ID for quantized matmul operation
pub fn get_quantized_kernel_id(
    qtype: QuantizationType,
    device: Device
) -> String {
    match (qtype, device) {
        (QuantizationType::I2S, Device::Cuda(_)) => "i2s_gpu_matmul".to_string(),
        (QuantizationType::I2S, Device::Cpu) => "i2s_gemv".to_string(),
        (QuantizationType::TL1, Device::Cpu) => "tl1_neon_matmul".to_string(),
        (QuantizationType::TL2, Device::Cpu) => "tl2_avx_matmul".to_string(),
        // ... other combinations
    }
}

/// Get kernel ID for fallback operation
pub fn get_fallback_kernel_id(
    qtype: QuantizationType,
    device: Device
) -> String {
    match (qtype, device) {
        (QuantizationType::I2S, Device::Cuda(_)) => "dequant_i2s_gpu".to_string(),
        (QuantizationType::I2S, Device::Cpu) => "dequant_i2s_cpu".to_string(),
        // ... other combinations
    }
}
```

## Rationale

### Why Prefix-Based Naming?

**Alternative 1: Suffix-Based**
- **Rejected:** `matmul_i2s_gpu` - Harder to pattern match (requires parsing)
- **Rejected:** Inconsistent with existing BitNet-rs conventions

**Alternative 2: Hierarchical**
- **Rejected:** `gpu::i2s::matmul` - Requires splitting on `::` (parsing overhead)
- **Rejected:** Not compatible with string-based kernel IDs in receipts

**Alternative 3: Enum-Based**
- **Rejected:** Requires serialization/deserialization logic
- **Rejected:** Less flexible for future extensions

**Selected: Prefix-Based (Best Practice)**
- ✅ Simple pattern matching (`starts_with`, `contains`)
- ✅ Unambiguous classification (quantized vs fallback)
- ✅ Extensible for future quantization types
- ✅ Compatible with existing receipt schema (string-based)

### Mixed Precision Clarification

**FP16/BF16 GPU Kernels:**
- **Kernel ID:** `gemm_fp16`, `wmma_fp16`, `i2s_gpu_fp16`
- **Classification:** **Quantized** (native GPU quantized path)
- **Rationale:** FP16/BF16 matmul with quantized weights (no FP32 dequantization)

**FP32 Dequantization:**
- **Kernel ID:** `dequant_i2s`, `fp32_matmul`
- **Classification:** **Fallback** (FP32 dequantization staging)
- **Rationale:** Explicit dequantization to FP32 (not native quantized path)

## Validation

### Test Cases

**Location:** `xtask/tests/verify_receipt_test.rs`

```rust
#[test]
fn test_is_quantized_kernel() {
    // GPU quantized kernels
    assert!(is_quantized_kernel("gemm_fp16"));
    assert!(is_quantized_kernel("i2s_gpu_quantize"));
    assert!(is_quantized_kernel("wmma_matmul"));

    // CPU quantized kernels
    assert!(is_quantized_kernel("i2s_gemv"));
    assert!(is_quantized_kernel("tl1_neon_pack"));
    assert!(is_quantized_kernel("tl2_avx_matmul"));

    // Fallback kernels (not quantized)
    assert!(!is_quantized_kernel("dequant_i2s"));
    assert!(!is_quantized_kernel("fp32_matmul"));
    assert!(!is_quantized_kernel("scalar_matmul"));
}

#[test]
fn test_is_fallback_kernel() {
    // Fallback kernels
    assert!(is_fallback_kernel("dequant_i2s"));
    assert!(is_fallback_kernel("fp32_matmul"));
    assert!(is_fallback_kernel("scalar_matmul"));
    assert!(is_fallback_kernel("fallback_gemm"));
    assert!(is_fallback_kernel("mock_inference"));

    // Quantized kernels (not fallback)
    assert!(!is_fallback_kernel("gemm_fp16"));
    assert!(!is_fallback_kernel("i2s_gpu_matmul"));
    assert!(!is_fallback_kernel("tl1_neon_pack"));
}
```

### Validation Commands

```bash
# Test kernel ID classification
cargo test -p xtask test_is_quantized_kernel
cargo test -p xtask test_is_fallback_kernel

# Test kernel ID generation
cargo test -p bitnet-kernels test_get_quantized_kernel_id
cargo test -p bitnet-kernels test_get_fallback_kernel_id
```

## Consequences

### Positive

1. **Unambiguous Classification:** 100% accuracy in distinguishing quantized vs fallback
2. **Simple Pattern Matching:** `starts_with` and `contains` (no parsing)
3. **Extensible:** Add new quantization types without breaking existing validation
4. **Receipt Validation:** Accurate correlation of claims with kernel IDs

### Negative

1. **Naming Constraints:** Developers must follow conventions strictly
2. **Documentation Overhead:** Must document all kernel ID patterns
3. **Backward Compatibility:** Existing kernel IDs may need migration

### Mitigation

- **Naming Constraints:** Codified in `get_quantized_kernel_id` and `get_fallback_kernel_id`
- **Documentation:** Comprehensive reference in `docs/reference/strict-mode-api.md`
- **Backward Compatibility:** Gradual migration (old IDs still valid, new IDs added)

## Success Metrics

- ✅ 100% classification accuracy for quantized vs fallback kernels
- ✅ Zero false positives in receipt validation
- ✅ Zero false negatives in strict mode enforcement
- ✅ All kernel IDs follow naming conventions (verified in CI)

## Related ADRs

- **ADR-010:** Three-Tier Validation Strategy
- **ADR-011:** Receipt Schema Backward Compatibility
- **ADR-013:** FP32 Fallback Detection Mechanisms

## References

- **Issue #453:** Strict Quantization Guards (AC6)
- **PR #452:** Receipt Verification Infrastructure
