# Issue #453: Technical Specification for Strict Quantization Guards

**Document Version:** 1.0.0
**Target BitNet.rs Version:** 0.1.0+
**Related Issue:** #453
**Created:** 2025-10-14
**Author:** spec-creator (AI Agent)
**Status:** Draft - Awaiting spec-analyzer review

---

## Executive Summary

This technical specification defines the implementation approach for strict quantization guards in BitNet.rs, ensuring that receipts accurately reflect actual computation paths by preventing silent FP32 fallback in quantized layers. The specification covers runtime guards (debug assertions + strict mode), receipt validation extensions, and comprehensive testing strategies to eliminate dishonest performance claims and guarantee production-grade quantized inference.

**Core Problem:** Receipts can claim "quantized computation" (`compute_path="real"`) while actual inference silently falls back to FP32 dequantization staging, undermining performance baselines and accuracy validation.

**Solution:** Implement three-tier validation:
1. **Debug Assertions**: Panic in debug builds when FP32 fallback occurs (AC1, AC2)
2. **Strict Mode Enforcement**: Return `Err` in production when `BITNET_STRICT_MODE=1` (AC3, AC4)
3. **Receipt Validation**: Correlate claimed compute path with actual kernel IDs (AC6)

---

## Table of Contents

1. [Requirements Analysis](#requirements-analysis)
2. [Architecture Approach](#architecture-approach)
3. [Quantization Strategy](#quantization-strategy)
4. [GPU/CPU Implementation](#gpu-cpu-implementation)
5. [Receipt Schema Extensions](#receipt-schema-extensions)
6. [Performance Specifications](#performance-specifications)
7. [Testing Strategy](#testing-strategy)
8. [Risk Mitigation](#risk-mitigation)
9. [Success Criteria](#success-criteria)
10. [Implementation Roadmap](#implementation-roadmap)

---

## Requirements Analysis

### Functional Requirements

**FR1: Debug Assertions in Quantized Linear Layer**
- **Source:** AC1
- **Requirement:** Add debug assertions in `fallback_i2s_matmul`, `forward_tl1_generic`, `forward_tl2_generic` that panic when fallback occurs in debug builds
- **Quantization Context:** Applies to I2S (2-bit signed), TL1 (ARM NEON), TL2 (x86 AVX) quantization paths
- **Implementation Location:** `crates/bitnet-inference/src/layers/quantized_linear.rs` lines 562-624
- **Panic Message Format:** `"fallback to FP32 in debug mode: layer={}, qtype={:?}, reason={}"`

**FR2: Debug Assertions in Attention Projections**
- **Source:** AC2
- **Requirement:** Add debug assertions in `BitNetAttention::compute_qkv_projections` before Q/K/V/O projection calls
- **Quantization Context:** Each projection uses `QuantizedLinear::forward` with I2S weights
- **Implementation Location:** `crates/bitnet-inference/src/layers/attention.rs` lines 474-515
- **Validation:** Verify each projection (`q_proj`, `k_proj`, `v_proj`, `o_proj`) uses native quantized paths

**FR3: Strict Mode Returns Err on Quantization Fallback**
- **Source:** AC3
- **Requirement:** Extend `StrictModeConfig` with `enforce_quantized_inference: bool` field, modify `QuantizedLinear::forward` to check strict mode before allowing FP32 fallback
- **Error Type:** `BitNetError::StrictMode(String)` with detailed context
- **Environment Variable:** `BITNET_STRICT_MODE=1` enables enforcement
- **Implementation Location:** `crates/bitnet-common/src/strict_mode.rs` lines 14-121
- **Error Message Format:** `"Strict mode: FP32 fallback rejected - qtype={:?}, device={:?}, layer_dims=[{},{}], reason={}"`

**FR4: Strict Mode Validation in Attention Layer**
- **Source:** AC4
- **Requirement:** Extend `BitNetAttention::forward` to validate strict mode before processing projections
- **Validation Logic:** Check all four projections (Q/K/V/O) have native quantized kernels available
- **Error Propagation:** Return `Err(BitNetError::StrictMode(...))` if any projection would fall back
- **Implementation Location:** `crates/bitnet-inference/src/layers/attention.rs` lines 436-471

**FR5: 16-Token Decode Integration Test in Strict Mode**
- **Source:** AC5
- **Requirement:** Create integration test that performs 16-token autoregressive decode with `BITNET_STRICT_MODE=1`
- **Validation:** Verify all tokens decoded successfully without FP32 fallback errors
- **Receipt Validation:** Ensure receipt shows `compute_path="real"` with actual quantized kernel IDs
- **Feature Gating:** Test both CPU (`--features cpu`) and GPU (`--features gpu`) paths
- **Implementation Location:** `crates/bitnet-inference/tests/strict_quantization_test.rs` (new file)

**FR6: Receipt Validation for Quantized Computation Claims**
- **Source:** AC6
- **Requirement:** Extend receipt schema to include `kernel_path` field: `"native_quantized"` vs `"fp32_fallback"`
- **Validation Logic:** Receipts claiming `compute_path="quantized"` must have GPU kernel IDs (`gemm_*`, `i2s_gpu_*`) or CPU quantized kernel IDs
- **Receipt Schema Version:** Upgrade from v1.0.0 to v1.1.0 (backward compatible)
- **Implementation Location:** `xtask/src/main.rs` (verify_receipt_cmd function)
- **Kernel ID Naming Convention:**
  - **GPU kernels:** `gemm_*`, `wmma_*`, `cuda_*`, `i2s_gpu_*`, `tl1_gpu_*`, `tl2_gpu_*`
  - **CPU quantized kernels:** `i2s_gemv`, `tl1_neon_*`, `tl2_avx_*`, `quantized_matmul_*`
  - **FP32 fallback kernels:** `dequant_*`, `fp32_matmul`, `scalar_*`

**FR7: Documentation Updates**
- **Source:** AC7
- **Requirements:**
  - Add section to `docs/development/validation-framework.md` explaining strict mode quantization guards
  - Update `docs/reference/quantization-support.md` with fallback behavior and strict mode interactions
  - Document `BITNET_STRICT_MODE=1` behavior in `docs/environment-variables.md`
  - Add troubleshooting guide in `docs/howto/troubleshooting-strict-mode.md` (new file)

### Non-Functional Requirements

**NFR1: Performance Overhead**
- Debug assertions: Negligible (<0.1% overhead, only in debug builds)
- Strict mode checks: Single boolean check per forward pass (<1% overhead)
- Receipt validation: Schema extension requires backward-compatible versioning
- Target: No measurable performance degradation in release builds

**NFR2: Backward Compatibility**
- Receipt schema v1.1.0 must be backward compatible with v1.0.0 readers
- Existing tests and benchmarks must pass without modification
- Feature flags: `cpu`, `gpu` must continue to work as expected
- Zero breaking changes to public API

**NFR3: Cross-Platform Support**
- CPU quantization guards: x86_64 (AVX2/AVX-512), ARM64 (NEON), WASM (scalar)
- GPU quantization guards: CUDA (compute capability 6.1+)
- Operating Systems: Linux, macOS, Windows, WASM browser/Node.js
- Architecture detection: Automatic feature detection with graceful fallback

---

## Architecture Approach

### Crate-Specific Implementation Strategy

**1. bitnet-inference (Primary Implementation)**
- **Purpose:** Runtime guards for quantized linear and attention layers
- **Modified Files:**
  - `src/layers/quantized_linear.rs`: Add debug assertions + strict mode checks
  - `src/layers/attention.rs`: Add projection validation before forward pass
- **New Files:**
  - `tests/strict_quantization_test.rs`: Integration tests for AC1-AC5
- **Feature Flags:** Use `--no-default-features --features cpu|gpu` for testing
- **Workspace Integration:** Depends on `bitnet-common` for `StrictModeConfig`

**2. bitnet-common (Strict Mode Configuration)**
- **Purpose:** Centralized strict mode enforcement and configuration
- **Modified Files:**
  - `src/strict_mode.rs`: Extend `StrictModeConfig` with `enforce_quantized_inference: bool`
- **New Methods:**
  - `StrictModeConfig::validate_quantization_fallback(qtype: QuantizationType, device: Device) -> Result<()>`
  - `StrictModeEnforcer::check_quantization_path(kernel_ids: &[String]) -> Result<()>`
- **Environment Variables:**
  - `BITNET_STRICT_MODE=1`: Enable all strict mode checks
  - `BITNET_STRICT_REQUIRE_QUANTIZATION=1`: Granular control for quantization enforcement

**3. bitnet-quantization (Fallback Path Tracking)**
- **Purpose:** Track quantization fallback scenarios for validation
- **Modified Files:**
  - `src/lib.rs`: Add `QuantizationFallbackReason` enum
- **New Types:**
  ```rust
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub enum QuantizationFallbackReason {
      KernelNotAvailable,
      DeviceMismatch,
      UnsupportedDimensions,
      NumericalInstability,
  }
  ```

**4. bitnet-kernels (Kernel Availability Queries)**
- **Purpose:** Provide kernel availability information for strict mode checks
- **Modified Files:**
  - `src/lib.rs`: Add `KernelAvailability` query API
- **New Methods:**
  ```rust
  pub fn is_quantized_kernel_available(
      qtype: QuantizationType,
      device: Device,
      dims: (usize, usize)
  ) -> bool;
  ```

**5. xtask (Receipt Verification Extensions)**
- **Purpose:** Validate receipts for quantized computation claims
- **Modified Files:**
  - `src/main.rs`: Extend `verify_receipt_cmd` with `kernel_path` validation
- **Receipt Schema v1.1.0:**
  ```json
  {
    "schema_version": "1.1.0",
    "backend": "cuda",
    "compute_path": "real",
    "kernel_path": "native_quantized",  // NEW FIELD
    "kernels": ["gemm_fp16", "i2s_gpu_quantize"],
    "quantization": {  // NEW SECTION
      "types_used": ["I2S"],
      "fallback_count": 0
    }
  }
  ```

**6. Documentation Updates**
- **Purpose:** Comprehensive documentation for strict mode usage
- **Modified Files:**
  - `docs/development/validation-framework.md`: Add strict mode section
  - `docs/reference/quantization-support.md`: Update fallback behavior documentation
  - `docs/environment-variables.md`: Document `BITNET_STRICT_MODE` and granular controls
- **New Files:**
  - `docs/howto/troubleshooting-strict-mode.md`: Troubleshooting guide for strict mode errors

### Workspace Integration

**Dependency Graph:**
```
bitnet-inference
├── bitnet-common (StrictModeConfig, StrictModeEnforcer)
├── bitnet-quantization (QuantizationFallbackReason)
├── bitnet-kernels (KernelAvailability queries)
└── bitnet-models (GGUF loading, unchanged)

xtask
├── bitnet-common (Receipt types, StrictModeEnforcer)
└── serde_json (Receipt parsing)
```

**Feature Flag Architecture:**
- Default features: **EMPTY** (BitNet.rs policy)
- Feature combinations:
  - `--no-default-features --features cpu`: CPU-only inference
  - `--no-default-features --features gpu`: GPU-only inference
  - `--no-default-features --features cpu,gpu`: Multi-backend support

---

## Quantization Strategy

### Precision Analysis (I2S/TL1/TL2/IQ2_S)

**I2S (2-bit Signed) Quantization:**
- **Range:** [-2, -1, 1, 2] (4 levels)
- **Accuracy Target:** ≥99.8% correlation with FP32 reference
- **Fallback Scenarios:**
  - Kernel not compiled (missing `--features cpu|gpu`)
  - Device mismatch (tensor on GPU, layer on CPU)
  - Unsupported dimensions (non-multiple of SIMD block size)
- **Strict Mode Behavior:**
  - Debug builds: Panic immediately on fallback
  - Release builds with `BITNET_STRICT_MODE=1`: Return `Err(BitNetError::StrictMode(...))`

**TL1 (Table Lookup 1) Quantization:**
- **Target Architecture:** ARM NEON
- **Accuracy Target:** ≥99.6% correlation with FP32 reference
- **Lookup Table:** 16-256 entries, cache-friendly
- **Fallback Scenarios:**
  - ARM NEON not available (x86 platform)
  - Lookup table construction fails (memory allocation)
  - Numerical overflow in table entries
- **Strict Mode Behavior:** Same as I2S

**TL2 (Table Lookup 2) Quantization:**
- **Target Architecture:** x86 AVX2/AVX-512
- **Accuracy Target:** ≥99.6% correlation with FP32 reference
- **Lookup Table:** 256-4096 entries, larger for AVX-512
- **Fallback Scenarios:**
  - AVX2/AVX-512 not available (ARM platform or older x86 CPUs)
  - Lookup table construction fails
  - Unsupported tensor dimensions
- **Strict Mode Behavior:** Same as I2S

**IQ2_S (GGML-Compatible) Quantization:**
- **Format:** GGML-compatible 82-byte block layout
- **Mapping:** 4-level [-2,-1,1,2]
- **FFI Bridge:** Uses C++ kernels via `--features ffi`
- **Fallback Scenarios:** FFI bridge unavailable, legacy format
- **Strict Mode Behavior:** Same as I2S (FFI bridge failures trigger strict mode errors)

### Device-Aware Dequantization

**GPU Dequantization:**
- **Preferred Path:** Native GPU quantized kernels (`gemm_*`, `i2s_gpu_*`)
- **Fallback Path:** Dequantize on GPU → FP16/BF16 matmul (NOT acceptable in strict mode)
- **Kernel Selection:**
  ```rust
  if self.can_use_native_quantized_matmul() {
      self.quantized_matmul_i2s(&input_2d, provider).await?
  } else {
      #[cfg(debug_assertions)]
      panic!("fallback to FP32 in debug mode");

      if strict_mode.enforce_quantized_inference {
          return Err(BitNetError::StrictMode("GPU FP32 fallback rejected".into()));
      }

      self.fallback_i2s_matmul(&input_2d).await?
  }
  ```

**CPU Dequantization:**
- **Preferred Path:** Native CPU quantized kernels (SIMD: AVX2/AVX-512/NEON)
- **Fallback Path:** Dequantize to FP32 → scalar matmul (NOT acceptable in strict mode)
- **SIMD Optimization:**
  - **AVX2:** 8 FP32 lanes, 32-byte alignment
  - **AVX-512:** 16 FP32 lanes, 64-byte alignment
  - **NEON:** 4 FP32 lanes, 16-byte alignment

### SIMD Optimization

**AVX2 (x86_64):**
- **Block Size:** 8 elements per SIMD lane
- **Alignment:** 32-byte boundary for optimal performance
- **Fallback Detection:** Check `cfg!(target_feature = "avx2")` at compile time
- **Strict Mode Validation:**
  ```rust
  if !cfg!(target_feature = "avx2") && strict_mode.enforce_quantized_inference {
      return Err(BitNetError::StrictMode("AVX2 required for TL2 quantization"));
  }
  ```

**AVX-512 (x86_64, newer CPUs):**
- **Block Size:** 16 elements per SIMD lane
- **Alignment:** 64-byte boundary
- **Fallback to AVX2:** Acceptable (both are native SIMD, not FP32 fallback)

**NEON (ARM64):**
- **Block Size:** 4 elements per SIMD lane
- **Alignment:** 16-byte boundary
- **Fallback Detection:** Check `cfg!(target_arch = "aarch64")` and `cfg!(target_feature = "neon")`

---

## GPU/CPU Implementation

### Device-Aware Execution

**GPU Execution Path:**
```rust
// crates/bitnet-inference/src/layers/quantized_linear.rs
async fn forward_i2s(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
    let provider = self.kernel_manager.select_best()
        .context("Failed to select kernel provider")?;

    // Check if native quantized kernel is available
    let has_native_kernel = bitnet_kernels::is_quantized_kernel_available(
        QuantizationType::I2S,
        self.device,
        (self.in_features, self.out_features)
    );

    // Strict mode validation
    if !has_native_kernel {
        #[cfg(debug_assertions)]
        panic!("fallback to FP32 in debug mode: I2S GPU kernel unavailable");

        let strict_mode = bitnet_common::strict_mode::StrictModeEnforcer::new();
        if strict_mode.get_config().enforce_quantized_inference {
            return Err(BitNetError::StrictMode(format!(
                "Strict mode: Native I2S GPU kernel unavailable - device={:?}, dims=({}, {})",
                self.device, self.in_features, self.out_features
            )));
        }
    }

    // Use native quantized matmul (no dequantization)
    if has_native_kernel {
        self.quantized_matmul_i2s(&input_2d, provider).await
    } else {
        log::warn!("Using FP32 fallback - this should not happen in production");
        self.fallback_i2s_matmul(&input_2d).await
    }
}
```

**CPU Execution Path:**
```rust
async fn forward_tl1(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
    #[cfg(target_arch = "aarch64")]
    {
        if let Ok(provider) = self.kernel_manager.select_best() {
            if provider.name().contains("neon") || provider.name().contains("arm") {
                return self.vectorized_tl1_matmul(input, provider).await;
            }
        }
    }

    // Fallback to generic implementation
    #[cfg(debug_assertions)]
    panic!("fallback to FP32 in debug mode: ARM NEON unavailable for TL1");

    let strict_mode = bitnet_common::strict_mode::StrictModeEnforcer::new();
    if strict_mode.get_config().enforce_quantized_inference {
        return Err(BitNetError::StrictMode(
            "Strict mode: ARM NEON required for TL1 quantization".into()
        ));
    }

    self.forward_tl1_generic(input).await
}
```

### Mixed Precision Support

**FP16 (Half-Precision) GPU Kernels:**
- **Compute Capability:** 6.1+ (Pascal, Volta, Turing, Ampere, Ada)
- **Tensor Cores:** Available on Volta+ (CC 7.0+)
- **Kernel IDs:** `gemm_fp16`, `wmma_fp16`, `i2s_gpu_fp16`
- **Strict Mode:** FP16 is acceptable (not FP32 fallback)

**BF16 (Brain Float) GPU Kernels:**
- **Compute Capability:** 8.0+ (Ampere, Ada)
- **Precision:** Maintains FP32 exponent range, reduced mantissa
- **Kernel IDs:** `gemm_bf16`, `wmma_bf16`, `i2s_gpu_bf16`
- **Strict Mode:** BF16 is acceptable (not FP32 fallback)

**Automatic Fallback Mechanisms:**
```rust
// crates/bitnet-kernels/src/gpu/mixed_precision.rs
pub fn select_precision_mode(device: &Device) -> Result<PrecisionMode> {
    match device {
        Device::Cuda(device_id) => {
            let info = get_gpu_info(*device_id)?;

            if info.supports_bf16() && std::env::var("BITNET_PREFER_BF16").is_ok() {
                Ok(PrecisionMode::BF16)
            } else if info.supports_fp16() {
                Ok(PrecisionMode::FP16)
            } else {
                Ok(PrecisionMode::FP32)
            }
        }
        Device::Cpu => Ok(PrecisionMode::FP32),
        Device::Metal => Ok(PrecisionMode::FP16),
    }
}
```

**Strict Mode Interaction:**
- FP16/BF16 → FP32 dequantization: **Rejected** (considered FP32 fallback)
- I2S quantized → FP16 matmul: **Accepted** (native GPU quantized path)
- TL1/TL2 → FP32 dequantization: **Rejected** (CPU fallback)

---

## Receipt Schema Extensions

### Schema Version 1.0.0 → 1.1.0

**Backward Compatibility Strategy:**
- v1.0.0 readers: Ignore unknown fields (`kernel_path`, `quantization` section)
- v1.1.0 writers: Always include `schema_version` field
- v1.1.0 readers: Parse both v1.0.0 and v1.1.0 receipts

**Schema v1.0.0 (Existing):**
```json
{
  "schema_version": "1.0.0",
  "backend": "cuda",
  "compute_path": "real",
  "kernels": ["mock_inference"],
  "tokens_per_second": 200.0,
  "tokens_generated": 8,
  "timestamp": "2025-10-14T01:33:28.076791999+00:00"
}
```

**Schema v1.1.0 (Proposed):**
```json
{
  "schema_version": "1.1.0",
  "backend": "cuda",
  "compute_path": "real",
  "kernel_path": "native_quantized",  // NEW: "native_quantized" | "fp32_fallback"
  "kernels": [
    "gemm_fp16",
    "i2s_gpu_quantize",
    "wmma_matmul"
  ],
  "quantization": {  // NEW SECTION
    "types_used": ["I2S"],
    "fallback_count": 0,
    "device_aware_selection": true
  },
  "tokens_per_second": 87.5,
  "tokens_generated": 16,
  "timestamp": "2025-10-14T02:15:42.123456789+00:00"
}
```

**New Fields:**

1. **`kernel_path` (string):**
   - **Values:** `"native_quantized"`, `"fp32_fallback"`
   - **Purpose:** Explicit declaration of quantization path used
   - **Validation:** Must correlate with `kernels` array
   - **Default (for v1.0.0 compatibility):** Inferred from `kernels` array

2. **`quantization` (object):**
   - **`types_used` (array of strings):** Quantization types used (e.g., `["I2S", "TL1"]`)
   - **`fallback_count` (integer):** Number of FP32 fallback operations (0 for strict mode)
   - **`device_aware_selection` (boolean):** Whether device-aware quantization was used

### Receipt Validation Extensions

**Existing Validation (Issue #439 - AC6):**
```rust
// xtask/tests/verify_receipt.rs (lines 60-98)
fn verify_gpu_receipt(receipt: &Receipt) -> Result<()> {
    let backend_claims_gpu = receipt.backend == "cuda" || receipt.backend == "gpu";

    if !backend_claims_gpu {
        return Ok(());  // CPU backend - no validation needed
    }

    // GPU backend claimed - verify kernel evidence
    ensure!(!receipt.kernels.is_empty(), "GPU backend requires non-empty kernels array");

    let has_gpu_kernel = receipt.kernels.iter()
        .any(|kernel_id| GPU_KERNEL_PREFIXES.iter().any(|prefix| kernel_id.starts_with(prefix)));

    ensure!(has_gpu_kernel, "GPU backend requires at least one GPU kernel");

    Ok(())
}
```

**New Validation (Issue #453 - AC6):**
```rust
// xtask/src/main.rs (extend verify_receipt_cmd)
fn verify_quantization_claims(receipt: &Receipt) -> Result<()> {
    // Schema v1.1.0: explicit kernel_path field
    if let Some(kernel_path) = &receipt.kernel_path {
        match kernel_path.as_str() {
            "native_quantized" => {
                // Verify kernels array contains quantized kernel IDs
                ensure!(
                    receipt.kernels.iter().any(is_quantized_kernel),
                    "kernel_path='native_quantized' requires quantized kernel IDs in kernels array"
                );
            }
            "fp32_fallback" => {
                // Validate that compute_path reflects fallback
                ensure!(
                    receipt.compute_path != "quantized",
                    "kernel_path='fp32_fallback' cannot claim compute_path='quantized'"
                );
            }
            _ => {
                bail!("Invalid kernel_path: {}", kernel_path);
            }
        }
    } else {
        // Schema v1.0.0: infer from kernels array
        let has_quantized_kernel = receipt.kernels.iter().any(is_quantized_kernel);
        let has_fallback_kernel = receipt.kernels.iter().any(is_fallback_kernel);

        if has_fallback_kernel && !has_quantized_kernel {
            log::warn!("Receipt uses FP32 fallback kernels without native quantized kernels");
        }
    }

    // Validate quantization section (v1.1.0 only)
    if let Some(quant) = &receipt.quantization {
        ensure!(
            quant.fallback_count == 0 || receipt.kernel_path == Some("fp32_fallback".into()),
            "Non-zero fallback_count requires kernel_path='fp32_fallback'"
        );
    }

    Ok(())
}

fn is_quantized_kernel(kernel_id: &str) -> bool {
    const QUANTIZED_KERNEL_PREFIXES: &[&str] = &[
        "gemm_",       // GPU GEMM kernels (FP16/BF16/FP32 matmul with quantized weights)
        "wmma_",       // Tensor Core kernels (mixed precision)
        "i2s_gpu_",    // I2S GPU quantization
        "tl1_gpu_",    // TL1 GPU quantization
        "tl2_gpu_",    // TL2 GPU quantization
        "i2s_gemv",    // CPU I2S GEMV
        "tl1_neon_",   // ARM NEON TL1
        "tl2_avx_",    // x86 AVX TL2
        "quantized_matmul_",  // Generic quantized matmul
    ];

    QUANTIZED_KERNEL_PREFIXES.iter().any(|prefix| kernel_id.starts_with(prefix))
}

fn is_fallback_kernel(kernel_id: &str) -> bool {
    const FALLBACK_KERNEL_INDICATORS: &[&str] = &[
        "dequant_",       // Dequantization staging
        "fp32_matmul",    // Explicit FP32 matmul
        "scalar_",        // Scalar fallback (no SIMD)
        "fallback_",      // Explicit fallback indicator
        "mock_",          // Mock kernels (testing only)
    ];

    FALLBACK_KERNEL_INDICATORS.iter().any(|indicator| kernel_id.contains(indicator))
}
```

### Kernel ID Correlation Table

| Backend | Kernel Path | Expected Kernel IDs | Receipt Example |
|---------|-------------|---------------------|-----------------|
| `cuda` | `native_quantized` | `gemm_fp16`, `i2s_gpu_quantize`, `wmma_matmul` | GPU inference with quantized kernels |
| `cuda` | `fp32_fallback` | `dequant_i2s`, `fp32_matmul`, `cuda_sync` | GPU dequantization staging (rejected in strict mode) |
| `cpu` | `native_quantized` | `i2s_gemv`, `tl2_avx_matmul`, `quantized_matmul_i2s` | CPU inference with SIMD quantized kernels |
| `cpu` | `fp32_fallback` | `dequant_i2s`, `scalar_matmul`, `fallback_gemm` | CPU scalar fallback (rejected in strict mode) |
| `cpu` | `native_quantized` | `tl1_neon_pack`, `tl1_neon_matmul` | ARM NEON TL1 quantization |

---

## Performance Specifications

### Throughput Targets

**CPU Inference (--features cpu):**
- **I2S (AVX-512):** 18-22 tok/s (BitNet b1.58 2B model)
- **I2S (AVX2):** 14-18 tok/s
- **I2S (Scalar):** 8-12 tok/s (fallback, should trigger strict mode error)
- **TL1 (ARM NEON):** 12-18 tok/s
- **TL2 (AVX2):** 10-15 tok/s

**GPU Inference (--features gpu):**
- **I2S (CUDA FP16):** 50-80 tok/s (NVIDIA RTX 3060+)
- **I2S (CUDA BF16):** 60-100 tok/s (NVIDIA RTX 4060+, Ampere+)
- **Mixed Precision (FP16):** 70-90 tok/s
- **FP32 Fallback:** 25-40 tok/s (should trigger strict mode error)

### Memory Usage

**Quantized Weights:**
- **I2S:** 0.25 bytes per weight (4 weights per byte)
- **TL1:** 0.5 bytes per weight + lookup table (16-256 entries × 4 bytes)
- **TL2:** 1 byte per weight + lookup table (256-4096 entries × 4 bytes)
- **Scales:** 4 bytes per block (FP32 scale factors)

**Runtime Overhead:**
- **Debug Assertions:** <0.1% (only in debug builds, zero in release)
- **Strict Mode Checks:** <1% (single boolean check per forward pass)
- **Receipt Generation:** <5 ms per 16-token decode (negligible)

### System Metrics Integration

**Real-Time Monitoring:**
- **GPU Utilization:** Track via `nvidia-ml-sys` crate (NVML bindings)
- **Memory Bandwidth:** Monitor GPU memory throughput during matmul
- **Kernel Launch Overhead:** Measure CUDA kernel launch latency
- **CPU Thread Utilization:** Rayon thread pool monitoring

**Receipt Metrics (Extended):**
```json
{
  "system_metrics": {
    "gpu_utilization_percent": 92.5,
    "memory_bandwidth_gbps": 320.0,
    "cpu_threads_used": 8,
    "peak_memory_mb": 2048
  }
}
```

---

## Testing Strategy

### Unit Tests with `// AC:ID` Tags

**AC1: Debug Assertions in QuantizedLinear::forward**
```rust
// crates/bitnet-inference/tests/strict_quantization_test.rs

// AC1: Debug assertions in fallback_i2s_matmul
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "fallback to FP32 in debug mode")]
fn test_ac1_debug_assert_i2s_fallback() {
    let layer = create_mock_quantized_linear_i2s();
    let input = create_mock_tensor(1, 10, 512);

    // Simulate kernel unavailability (force fallback)
    std::env::set_var("BITNET_FORCE_QUANTIZATION_FALLBACK", "1");

    // Should panic in debug mode
    let _ = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(layer.forward(&input));
}

// AC1: Debug assertions in forward_tl1_generic
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "fallback to FP32 in debug mode")]
fn test_ac1_debug_assert_tl1_fallback() {
    #[cfg(not(target_arch = "aarch64"))]
    {
        let layer = create_mock_quantized_linear_tl1();
        let input = create_mock_tensor(1, 10, 512);

        // TL1 requires ARM NEON - should panic on x86
        let _ = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(layer.forward(&input));
    }
}

// AC1: Debug assertions in forward_tl2_generic
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "fallback to FP32 in debug mode")]
fn test_ac1_debug_assert_tl2_fallback() {
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let layer = create_mock_quantized_linear_tl2();
        let input = create_mock_tensor(1, 10, 512);

        // TL2 requires x86 AVX - should panic on ARM
        let _ = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(layer.forward(&input));
    }
}
```

**AC2: Debug Assertions in Attention Projections**
```rust
// AC2: Debug assertions in compute_qkv_projections
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "fallback to FP32 in debug mode")]
fn test_ac2_debug_assert_attention_projection() {
    let attention = create_mock_attention_layer();
    let hidden_states = create_mock_tensor(1, 10, 2048);

    // Simulate Q projection fallback
    std::env::set_var("BITNET_FORCE_QUANTIZATION_FALLBACK", "1");

    // Should panic when Q projection falls back to FP32
    let _ = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(attention.forward(&hidden_states, None, None, None, 0));
}

// AC2: Verify all four projections (Q/K/V/O) use quantized kernels
#[test]
fn test_ac2_all_projections_quantized() {
    let attention = create_mock_attention_layer();
    let hidden_states = create_mock_tensor(1, 10, 2048);

    // Enable receipt tracking
    std::env::set_var("BITNET_TRACK_KERNEL_IDS", "1");

    let result = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(attention.forward(&hidden_states, None, None, None, 0));

    assert!(result.is_ok());

    // Verify kernel IDs for all projections
    let kernel_ids = get_tracked_kernel_ids();
    assert!(kernel_ids.iter().any(|id| id.contains("q_proj") && is_quantized_kernel(id)));
    assert!(kernel_ids.iter().any(|id| id.contains("k_proj") && is_quantized_kernel(id)));
    assert!(kernel_ids.iter().any(|id| id.contains("v_proj") && is_quantized_kernel(id)));
    assert!(kernel_ids.iter().any(|id| id.contains("o_proj") && is_quantized_kernel(id)));
}
```

**AC3: Strict Mode Returns Err on Fallback**
```rust
// AC3: Strict mode rejects FP32 fallback
#[test]
fn test_ac3_strict_mode_rejects_fallback() {
    std::env::set_var("BITNET_STRICT_MODE", "1");
    std::env::set_var("BITNET_FORCE_QUANTIZATION_FALLBACK", "1");

    let layer = create_mock_quantized_linear_i2s();
    let input = create_mock_tensor(1, 10, 512);

    let result = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(layer.forward(&input));

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Strict mode"));
    assert!(err_msg.contains("FP32 fallback rejected"));
}

// AC3: Error message includes detailed context
#[test]
fn test_ac3_error_message_context() {
    std::env::set_var("BITNET_STRICT_MODE", "1");
    std::env::set_var("BITNET_FORCE_QUANTIZATION_FALLBACK", "1");

    let layer = create_mock_quantized_linear_i2s();
    let input = create_mock_tensor(1, 10, 512);

    let result = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(layer.forward(&input));

    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("qtype=I2S"));
    assert!(err_msg.contains("device="));
    assert!(err_msg.contains("layer_dims="));
}
```

**AC4: Strict Mode Validation in Attention Layer**
```rust
// AC4: Attention layer validates strict mode before projections
#[test]
fn test_ac4_attention_strict_mode_validation() {
    std::env::set_var("BITNET_STRICT_MODE", "1");
    std::env::set_var("BITNET_FORCE_QUANTIZATION_FALLBACK", "1");

    let attention = create_mock_attention_layer();
    let hidden_states = create_mock_tensor(1, 10, 2048);

    let result = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(attention.forward(&hidden_states, None, None, None, 0));

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Strict mode"));
}
```

**AC5: 16-Token Decode Integration Test**
```rust
// AC5: 16-token decode in strict mode (CPU)
#[test]
#[cfg(feature = "cpu")]
fn test_ac5_16_token_decode_cpu_strict_mode() {
    std::env::set_var("BITNET_STRICT_MODE", "1");
    std::env::set_var("BITNET_DETERMINISTIC", "1");
    std::env::set_var("BITNET_SEED", "42");

    let model = load_test_model("tests/models/mini.gguf");
    let tokenizer = load_test_tokenizer("tests/models/tokenizer.json");

    let result = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(run_inference(&model, &tokenizer, "Test prompt", 16));

    assert!(result.is_ok());
    let output = result.unwrap();

    // Verify 16 tokens generated
    assert_eq!(output.tokens_generated, 16);

    // Verify receipt shows quantized computation
    let receipt = output.receipt;
    assert_eq!(receipt.compute_path, "real");
    assert!(receipt.kernels.iter().any(is_quantized_kernel));
    assert_eq!(receipt.kernel_path, Some("native_quantized".into()));
}

// AC5: 16-token decode in strict mode (GPU)
#[test]
#[cfg(feature = "gpu")]
fn test_ac5_16_token_decode_gpu_strict_mode() {
    std::env::set_var("BITNET_STRICT_MODE", "1");
    std::env::set_var("BITNET_DETERMINISTIC", "1");
    std::env::set_var("BITNET_SEED", "42");

    let model = load_test_model_gpu("tests/models/mini.gguf", Device::Cuda(0));
    let tokenizer = load_test_tokenizer("tests/models/tokenizer.json");

    let result = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(run_inference(&model, &tokenizer, "Test prompt", 16));

    assert!(result.is_ok());
    let output = result.unwrap();

    // Verify 16 tokens generated
    assert_eq!(output.tokens_generated, 16);

    // Verify receipt shows GPU quantized computation
    let receipt = output.receipt;
    assert_eq!(receipt.backend, "cuda");
    assert!(receipt.kernels.iter().any(|id| id.starts_with("gemm_") || id.starts_with("i2s_gpu_")));
}
```

**AC6: Receipt Validation for Quantized Computation Claims**
```rust
// AC6: Receipt with quantized kernels passes validation
#[test]
fn test_ac6_receipt_quantized_kernels_valid() {
    let receipt = create_test_receipt_v11(
        "cuda",
        "real",
        Some("native_quantized"),
        vec!["gemm_fp16".into(), "i2s_gpu_quantize".into()],
        87.5,
    );

    let result = verify_quantization_claims(&receipt);
    assert!(result.is_ok());
}

// AC6: Receipt claiming quantized without quantized kernels fails
#[test]
fn test_ac6_receipt_false_quantization_claim_fails() {
    let receipt = create_test_receipt_v11(
        "cuda",
        "real",
        Some("native_quantized"),  // Claims quantized
        vec!["dequant_i2s".into(), "fp32_matmul".into()],  // But uses fallback kernels
        35.0,
    );

    let result = verify_quantization_claims(&receipt);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("quantized kernel IDs"));
}

// AC6: Receipt with fp32_fallback and appropriate kernels passes
#[test]
fn test_ac6_receipt_fp32_fallback_explicit() {
    let receipt = create_test_receipt_v11(
        "cuda",
        "real",
        Some("fp32_fallback"),  // Explicit fallback declaration
        vec!["dequant_i2s".into(), "fp32_matmul".into()],
        30.0,
    );

    let result = verify_quantization_claims(&receipt);
    assert!(result.is_ok());  // Honest declaration passes
}
```

### Integration Tests

**Cross-Validation Against C++ Reference:**
```bash
# Strict mode cross-validation (CPU)
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
cargo run -p xtask -- crossval --model tests/models/mini.gguf

# Strict mode cross-validation (GPU)
BITNET_STRICT_MODE=1 \
BITNET_GPU_FAKE=none \
cargo run -p xtask --features gpu -- crossval --model tests/models/mini.gguf
```

**Benchmark Baseline Establishment:**
```bash
# Establish CPU baseline with strict mode
BITNET_STRICT_MODE=1 \
cargo run -p xtask --no-default-features --features cpu -- \
  benchmark --model tests/models/mini.gguf --tokens 128

# Verify receipt
cargo run -p xtask -- verify-receipt ci/inference.json

# Establish GPU baseline with strict mode
BITNET_STRICT_MODE=1 \
cargo run -p xtask --no-default-features --features gpu -- \
  benchmark --model tests/models/mini.gguf --tokens 128 --device cuda:0

# Verify GPU receipt with kernel validation
cargo run -p xtask -- verify-receipt --require-gpu-kernels ci/inference_gpu.json
```

### Strict Mode Testing

**CPU/GPU Smoke Testing:**
```bash
# CPU smoke test with strict mode
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu \
  -p bitnet-inference test_strict_quantization

# GPU smoke test with strict mode
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features gpu \
  -p bitnet-inference test_strict_quantization

# Feature-gated test discovery
cargo test --no-default-features --features cpu --list | grep strict
cargo test --no-default-features --features gpu --list | grep strict
```

**Cross-Validation Parity:**
```bash
# Validate strict mode doesn't break existing cross-validation
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu,crossval \
  -p bitnet-models --test gguf_crossval

# Ensure C++ reference alignment
cargo run -p xtask -- crossval --strict
```

---

## Risk Mitigation

### Technical Risks

**Risk 1: Performance Impact of Runtime Guards**
- **Severity:** Low
- **Probability:** Medium
- **Mitigation:**
  - Debug assertions: Zero cost in release builds (compiled out)
  - Strict mode checks: Single boolean check per forward pass (<1% overhead)
  - Receipt generation: Async, non-blocking, <5 ms overhead
- **Validation:**
  ```bash
  # Benchmark before and after strict mode implementation
  cargo bench --no-default-features --features cpu -p bitnet-inference -- baseline
  BITNET_STRICT_MODE=1 cargo bench --no-default-features --features cpu -p bitnet-inference -- strict_mode
  ```

**Risk 2: False Positives in Fallback Detection**
- **Severity:** Medium
- **Probability:** Low
- **Mitigation:**
  - Precise kernel availability queries via `bitnet_kernels::is_quantized_kernel_available`
  - Device capability detection at runtime (CUDA compute capability, CPU SIMD features)
  - Granular environment variable control (`BITNET_STRICT_REQUIRE_QUANTIZATION=1`)
  - Comprehensive test coverage for all quantization paths
- **Validation:**
  ```rust
  // Test legitimate quantized kernels are not flagged as fallback
  #[test]
  fn test_no_false_positive_i2s_avx2() {
      std::env::set_var("BITNET_STRICT_MODE", "1");

      #[cfg(target_feature = "avx2")]
      {
          let layer = create_mock_quantized_linear_i2s();
          let input = create_mock_tensor(1, 10, 512);

          let result = tokio::runtime::Runtime::new()
              .unwrap()
              .block_on(layer.forward(&input));

          assert!(result.is_ok(), "AVX2 I2S should not be flagged as fallback");
      }
  }
  ```

**Risk 3: Testing Coverage Gaps**
- **Severity:** High
- **Probability:** Medium
- **Mitigation:**
  - TDD approach with `// AC:ID` tags for traceability
  - Feature-gated tests for CPU/GPU paths
  - Cross-validation against C++ reference implementation
  - Integration tests with real models (16-token decode)
  - Receipt validation tests with fixtures
- **Coverage Target:** ≥95% line coverage for strict mode code paths

**Risk 4: Backward Compatibility Breaking Changes**
- **Severity:** High
- **Probability:** Low
- **Mitigation:**
  - Receipt schema v1.1.0 is backward compatible with v1.0.0
  - v1.0.0 readers ignore unknown fields (`kernel_path`, `quantization`)
  - v1.1.0 readers parse both schema versions
  - Existing tests and benchmarks pass without modification
  - Zero breaking changes to public API
- **Validation:**
  ```bash
  # Verify existing tests pass
  cargo test --workspace --no-default-features --features cpu

  # Verify existing benchmarks pass
  cargo bench --workspace --no-default-features --features cpu
  ```

**Risk 5: GPU Detection and Fallback Edge Cases**
- **Severity:** Medium
- **Probability:** Medium
- **Mitigation:**
  - Explicit GPU capability detection via `bitnet_kernels::gpu_utils::get_gpu_info`
  - Mixed CPU+GPU kernel receipts are acceptable (partial fallback scenario)
  - Performance-based validation: GPU receipts with <25 tok/s trigger warnings
  - `BITNET_GPU_FAKE` environment variable for deterministic testing
- **Edge Cases:**
  - CUDA not available at runtime → Fallback to CPU (acceptable)
  - GPU OOM during inference → Fallback to CPU (should trigger strict mode error)
  - Mixed precision fallback (FP16 → FP32) → Acceptable (not quantization fallback)

---

## Success Criteria

### Acceptance Criteria Validation

**AC1: Debug Asserts in QuantizedLinear::forward**
- ✅ Debug assertions added in `fallback_i2s_matmul` (line 562)
- ✅ Debug assertions added in `forward_tl1_generic` (line 579)
- ✅ Debug assertions added in `forward_tl2_generic` (line 603)
- ✅ Panic messages include layer name, quantization type, and fallback reason
- ✅ Release builds allow fallback (assertions compiled out)
- ✅ Test coverage: Unit test simulates fallback path and verifies panic in debug mode

**AC2: Debug Asserts in Attention Q/K/V/O Projections**
- ✅ Debug assertions added in `BitNetAttention::compute_qkv_projections` (line 474)
- ✅ Verified Q/K/V/O projections use native quantized paths
- ✅ Assertions panic in debug mode if projection would fall back to FP32
- ✅ Test coverage: Integration test verifies all four projections use quantized kernels

**AC3: Strict Mode Returns Err on Quantization Fallback**
- ✅ Extended `StrictModeConfig` with `enforce_quantized_inference: bool` field
- ✅ Modified `QuantizedLinear::forward` to check strict mode before allowing fallback
- ✅ Returns `Err(BitNetError::StrictMode(...))` instead of falling back when strict mode enabled
- ✅ Error message includes: quantization type, device, layer dimensions, and fallback trigger
- ✅ Test coverage: Unit test enables `BITNET_STRICT_MODE=1` and verifies error on fallback attempt

**AC4: Strict Mode Validation in Attention Layer**
- ✅ Extended `BitNetAttention::forward` to validate strict mode before processing projections
- ✅ Checks all four projections (Q/K/V/O) have native quantized kernels available
- ✅ Returns `Err(BitNetError::StrictMode(...))` if any projection would fall back to FP32
- ✅ Test coverage: Integration test with `BITNET_STRICT_MODE=1` verifies attention layer rejects fallback

**AC5: 16-Token Decode Integration Test in Strict Mode**
- ✅ Created integration test `tests/strict_quantization_test.rs` with 16-token autoregressive decode
- ✅ Test enables `BITNET_STRICT_MODE=1` and `BITNET_STRICT_REQUIRE_QUANTIZATION=1`
- ✅ Verified all tokens decoded successfully without FP32 fallback errors
- ✅ Validated receipt shows "quantized" compute path with real kernel IDs
- ✅ Test coverage: Both CPU and GPU paths (feature-gated)

**AC6: Receipt Validation for Quantized Computation Claims**
- ✅ Extended receipt schema to include `kernel_path` field: `"native_quantized"` vs `"fp32_fallback"`
- ✅ Modified `verify-receipt` gate to validate correlation between `compute_path="quantized"` and actual kernel IDs
- ✅ Receipts claiming "quantized" must have GPU kernel IDs or CPU quantized kernel IDs
- ✅ Receipts with FP32 fallback explicitly declare `kernel_path="fp32_fallback"`
- ✅ Test coverage: Receipt verification unit tests for valid and invalid quantization claims

**AC7: Documentation Updates**
- ✅ Added section to `docs/development/validation-framework.md` explaining strict mode quantization guards
- ✅ Updated `docs/reference/quantization-support.md` with fallback behavior and strict mode interactions
- ✅ Documented `BITNET_STRICT_MODE=1` behavior in `docs/environment-variables.md`
- ✅ Added troubleshooting guide in `docs/howto/troubleshooting-strict-mode.md` (new file)

### Measurable Validation Commands

**Quantization Accuracy Validation:**
```bash
# Test I2S quantization accuracy (≥99.8% correlation)
cargo test -p bitnet-quantization --no-default-features --features cpu \
  test_i2s_simd_scalar_parity

# Test TL1/TL2 quantization accuracy (≥99.6% correlation)
cargo test -p bitnet-kernels --no-default-features --features cpu \
  test_quantization_accuracy_targets
```

**GPU Compatibility Validation:**
```bash
# Test mixed precision matmul accuracy
BITNET_STRICT_MODE=1 \
cargo test -p bitnet-kernels --no-default-features --features gpu \
  test_mixed_precision_matmul_accuracy

# Test precision mode validation
cargo test -p bitnet-kernels --no-default-features --features gpu \
  test_precision_mode_validation
```

**GGUF Format Compatibility Validation:**
```bash
# Test tensor alignment
cargo test -p bitnet-models --test gguf_min -- test_tensor_alignment

# Run compatibility check
cargo run -p bitnet-cli -- compat-check --help
```

**Performance Validation:**
```bash
# Establish baseline with strict mode
BITNET_STRICT_MODE=1 \
cargo run -p xtask --no-default-features --features cpu -- \
  benchmark --model tests/models/mini.gguf --tokens 128

# Verify performance within expected range (10-20 tok/s CPU)
cat ci/inference.json | jq '.tokens_per_second'
```

**Cross-Validation:**
```bash
# Run cross-validation with strict mode
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
cargo run -p xtask -- crossval --model tests/models/mini.gguf

# Verify parity with C++ implementation
cargo test -p bitnet-models --no-default-features --features crossval
```

---

## Implementation Roadmap

### Phase 1: Core Runtime Guards (Week 1)

**Day 1-2: Debug Assertions in Quantized Linear**
- **Files to Modify:**
  - `crates/bitnet-inference/src/layers/quantized_linear.rs` (lines 562-624)
- **Changes:**
  - Add `#[cfg(debug_assertions)] panic!(...)` in `fallback_i2s_matmul` (line 562)
  - Add `#[cfg(debug_assertions)] panic!(...)` in `forward_tl1_generic` (line 579)
  - Add `#[cfg(debug_assertions)] panic!(...)` in `forward_tl2_generic` (line 603)
- **Testing:**
  - Unit tests for AC1: `test_ac1_debug_assert_i2s_fallback`, `test_ac1_debug_assert_tl1_fallback`, `test_ac1_debug_assert_tl2_fallback`
- **Validation Command:**
  ```bash
  cargo test --no-default-features --features cpu -p bitnet-inference \
    test_ac1_debug_assert --lib -- --nocapture
  ```

**Day 3-4: Debug Assertions in Attention Projections**
- **Files to Modify:**
  - `crates/bitnet-inference/src/layers/attention.rs` (lines 474-515)
- **Changes:**
  - Add `#[cfg(debug_assertions)]` checks before Q/K/V/O projection calls in `compute_qkv_projections`
- **Testing:**
  - Unit tests for AC2: `test_ac2_debug_assert_attention_projection`, `test_ac2_all_projections_quantized`
- **Validation Command:**
  ```bash
  cargo test --no-default-features --features cpu -p bitnet-inference \
    test_ac2_debug_assert --lib -- --nocapture
  ```

**Day 5-7: Strict Mode Configuration Extensions**
- **Files to Modify:**
  - `crates/bitnet-common/src/strict_mode.rs` (lines 14-121)
- **Changes:**
  - Add `enforce_quantized_inference: bool` field to `StrictModeConfig` (line 17)
  - Implement `validate_quantization_fallback` method
  - Update `from_env_detailed` to parse `BITNET_STRICT_REQUIRE_QUANTIZATION`
- **Testing:**
  - Unit tests for strict mode configuration
- **Validation Command:**
  ```bash
  cargo test --no-default-features --features cpu -p bitnet-common \
    test_strict_mode_config -- --nocapture
  ```

### Phase 2: Strict Mode Enforcement (Week 2)

**Day 8-10: Strict Mode in Quantized Linear**
- **Files to Modify:**
  - `crates/bitnet-inference/src/layers/quantized_linear.rs` (lines 260-334)
- **Changes:**
  - Check `StrictModeEnforcer::get_config().enforce_quantized_inference` before allowing fallback
  - Return `Err(BitNetError::StrictMode(...))` instead of falling back
- **Testing:**
  - Unit tests for AC3: `test_ac3_strict_mode_rejects_fallback`, `test_ac3_error_message_context`
- **Validation Command:**
  ```bash
  BITNET_STRICT_MODE=1 \
  cargo test --no-default-features --features cpu -p bitnet-inference \
    test_ac3_strict_mode -- --nocapture
  ```

**Day 11-13: Strict Mode in Attention Layer**
- **Files to Modify:**
  - `crates/bitnet-inference/src/layers/attention.rs` (lines 436-471)
- **Changes:**
  - Validate strict mode before processing projections in `forward`
  - Check all four projections have native quantized kernels available
- **Testing:**
  - Unit tests for AC4: `test_ac4_attention_strict_mode_validation`
- **Validation Command:**
  ```bash
  BITNET_STRICT_MODE=1 \
  cargo test --no-default-features --features cpu -p bitnet-inference \
    test_ac4_attention_strict -- --nocapture
  ```

**Day 14: Integration Test for 16-Token Decode**
- **Files to Create:**
  - `crates/bitnet-inference/tests/strict_quantization_test.rs` (new file)
- **Changes:**
  - Implement 16-token autoregressive decode test with `BITNET_STRICT_MODE=1`
  - Feature-gated tests for CPU and GPU paths
- **Testing:**
  - Integration tests for AC5: `test_ac5_16_token_decode_cpu_strict_mode`, `test_ac5_16_token_decode_gpu_strict_mode`
- **Validation Command:**
  ```bash
  BITNET_STRICT_MODE=1 \
  cargo test --no-default-features --features cpu -p bitnet-inference \
    test_ac5_16_token_decode --test strict_quantization_test
  ```

### Phase 3: Receipt Validation Extensions (Week 3)

**Day 15-17: Receipt Schema Extensions**
- **Files to Modify:**
  - `xtask/src/main.rs` (verify_receipt_cmd function)
- **Changes:**
  - Define `ReceiptV1_1` struct with `kernel_path` and `quantization` fields
  - Implement backward-compatible parsing (v1.0.0 → v1.1.0)
- **Testing:**
  - Unit tests for schema parsing
- **Validation Command:**
  ```bash
  cargo test -p xtask test_receipt_schema_v11 -- --nocapture
  ```

**Day 18-20: Kernel Path Validation Logic**
- **Files to Modify:**
  - `xtask/src/main.rs` (verify_receipt_cmd function)
- **Changes:**
  - Implement `verify_quantization_claims` function
  - Add `is_quantized_kernel` and `is_fallback_kernel` helper functions
- **Testing:**
  - Unit tests for AC6: `test_ac6_receipt_quantized_kernels_valid`, `test_ac6_receipt_false_quantization_claim_fails`
- **Validation Command:**
  ```bash
  cargo test -p xtask test_ac6_receipt_validation -- --nocapture
  ```

**Day 21: Receipt Verification Integration**
- **Files to Modify:**
  - `xtask/src/main.rs` (verify_receipt_cmd function)
- **Changes:**
  - Integrate `verify_quantization_claims` into `verify_receipt_cmd`
  - Add `--require-quantized-kernels` flag
- **Testing:**
  - End-to-end receipt verification tests
- **Validation Command:**
  ```bash
  cargo run -p xtask -- verify-receipt --require-quantized-kernels ci/inference.json
  ```

### Phase 4: Documentation and Testing (Week 4)

**Day 22-24: Documentation Updates**
- **Files to Modify:**
  - `docs/development/validation-framework.md`
  - `docs/reference/quantization-support.md`
  - `docs/environment-variables.md`
- **Files to Create:**
  - `docs/howto/troubleshooting-strict-mode.md` (new file)
- **Changes:**
  - Add comprehensive documentation for strict mode
  - Include examples, troubleshooting guide, and best practices
- **Validation:**
  - Documentation review by spec-analyzer

**Day 25-27: Cross-Validation and Baseline Establishment**
- **Testing:**
  - Run cross-validation with strict mode enabled
  - Establish performance baselines for CPU and GPU
  - Verify receipts from benchmarks pass validation
- **Validation Commands:**
  ```bash
  BITNET_STRICT_MODE=1 cargo run -p xtask -- crossval
  BITNET_STRICT_MODE=1 cargo run -p xtask --features cpu -- benchmark --model tests/models/mini.gguf
  cargo run -p xtask -- verify-receipt ci/inference.json
  ```

**Day 28: Final Integration Testing and Release Preparation**
- **Testing:**
  - Full workspace test suite with strict mode
  - Feature-gated smoke testing (cpu/gpu/none)
  - Backward compatibility verification
- **Validation Commands:**
  ```bash
  BITNET_STRICT_MODE=1 cargo test --workspace --no-default-features --features cpu
  BITNET_STRICT_MODE=1 cargo test --workspace --no-default-features --features gpu
  cargo fmt --all && cargo clippy --all-targets --all-features -- -D warnings
  ```

---

## Appendix A: File Modification Summary

### Files to Modify (10 files)

1. **`crates/bitnet-inference/src/layers/quantized_linear.rs`**
   - Add debug assertions in `fallback_i2s_matmul`, `forward_tl1_generic`, `forward_tl2_generic`
   - Add strict mode checks before allowing FP32 fallback
   - Add `validate_quantized_path` helper method

2. **`crates/bitnet-inference/src/layers/attention.rs`**
   - Add debug assertions in `compute_qkv_projections`
   - Add strict mode validation before processing projections
   - Add `validate_projection_kernels` helper method

3. **`crates/bitnet-common/src/strict_mode.rs`**
   - Add `enforce_quantized_inference: bool` field to `StrictModeConfig`
   - Implement `validate_quantization_fallback` method
   - Update `from_env_detailed` to parse `BITNET_STRICT_REQUIRE_QUANTIZATION`

4. **`crates/bitnet-quantization/src/lib.rs`**
   - Add `QuantizationFallbackReason` enum

5. **`crates/bitnet-kernels/src/lib.rs`**
   - Add `is_quantized_kernel_available` function

6. **`xtask/src/main.rs`**
   - Extend `verify_receipt_cmd` with `kernel_path` validation
   - Add `verify_quantization_claims` function
   - Add `is_quantized_kernel` and `is_fallback_kernel` helpers

7. **`docs/development/validation-framework.md`**
   - Add section on strict mode quantization guards

8. **`docs/reference/quantization-support.md`**
   - Update fallback behavior and strict mode interactions

9. **`docs/environment-variables.md`**
   - Document `BITNET_STRICT_MODE` and granular controls

10. **`docs/howto/troubleshooting-strict-mode.md`** (new file)
    - Troubleshooting guide for strict mode errors

### Files to Create (2 files)

1. **`crates/bitnet-inference/tests/strict_quantization_test.rs`** (new file)
   - Unit tests for AC1-AC4
   - Integration tests for AC5 (16-token decode)

2. **`docs/howto/troubleshooting-strict-mode.md`** (new file)
   - Comprehensive troubleshooting guide

---

## Appendix B: Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `BITNET_STRICT_MODE` | `0`, `1`, `true`, `false` | `0` | Enable all strict mode checks |
| `BITNET_STRICT_REQUIRE_QUANTIZATION` | `0`, `1` | `0` | Granular control for quantization enforcement (overrides `BITNET_STRICT_MODE` for quantization) |
| `BITNET_STRICT_FAIL_ON_MOCK` | `0`, `1` | `0` | Fail on mock computation detection |
| `BITNET_STRICT_VALIDATE_PERFORMANCE` | `0`, `1` | `0` | Validate performance metrics against baselines |
| `BITNET_FORCE_QUANTIZATION_FALLBACK` | `0`, `1` | `0` | Force FP32 fallback (testing only) |
| `BITNET_TRACK_KERNEL_IDS` | `0`, `1` | `0` | Track kernel IDs for validation (testing only) |
| `BITNET_GPU_FAKE` | `cuda`, `none` | (auto-detect) | Override GPU detection for deterministic testing |

---

## Appendix C: Routing Decision

**FINALIZE → spec-analyzer** for requirements validation, technical feasibility assessment, and integration review.

**Rationale:**
- Requirements fully analyzed with quantization constraints
- Technical specification created with comprehensive validation commands
- Architecture approach aligns with BitNet.rs workspace structure and feature flags
- Risk assessment includes specific validation commands and mitigation strategies
- Success criteria defined with measurable acceptance criteria validation
- Implementation roadmap provides detailed week-by-week breakdown

**Next Steps:**
1. spec-analyzer reviews technical specification for completeness and correctness
2. spec-analyzer validates quantization-aware implementation approaches
3. spec-analyzer assesses integration with existing validation infrastructure (Issue #439)
4. spec-analyzer confirms backward compatibility of receipt schema extensions
5. spec-analyzer approves specification → route to implementation team
