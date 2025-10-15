# Strict Quantization Guards: Feature Specification

**Document Version:** 1.0.0
**Target BitNet.rs Version:** 0.1.0+
**Related Issue:** #453
**Related PR:** (TBD)
**Created:** 2025-10-14
**Status:** Approved - Ready for Implementation
**Type:** Explanation (Diátaxis)
**Audience:** BitNet.rs developers implementing quantization validation

---

## Executive Summary

This specification defines strict quantization guards for BitNet.rs neural network inference, ensuring that receipts accurately reflect actual computation paths by preventing silent FP32 fallback in quantized layers. The feature implements three-tier validation (debug assertions, strict mode enforcement, receipt validation) to guarantee production-grade quantized inference with honest performance claims.

**Core Problem:** Receipts can claim "quantized computation" (`compute_path="real"`) while actual inference silently falls back to FP32 dequantization staging, undermining performance baselines and accuracy validation.

**Solution:** Runtime guards that detect and reject FP32 fallback, ensuring receipts accurately reflect the actual computation path used during inference.

**Neural Network Context:** BitNet.rs inference pipeline (Model Loading → Quantization → Inference → Output) requires honest compute paths for production deployment confidence and cross-validation accuracy.

---

## Table of Contents

1. [User Story and Motivation](#user-story-and-motivation)
2. [Acceptance Criteria](#acceptance-criteria)
3. [Technical Architecture](#technical-architecture)
4. [BitNet.rs Quantization Integration](#bitnetrs-quantization-integration)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Testing Strategy](#testing-strategy)
7. [Documentation Requirements](#documentation-requirements)
8. [Success Metrics](#success-metrics)
9. [Related Work](#related-work)

---

## User Story and Motivation

### Primary User Story

**As a** BitNet.rs inference engineer,
**I want** runtime guards in strict mode that prevent silent FP32 fallback in quantized layers and attention projections,
**So that** receipts accurately reflect the actual computation path and I can trust performance baselines for production deployments.

### Business Value

**Neural Network Inference Workflow:**
1. **Model Loading**: Load GGUF/SafeTensors model with quantized weights
2. **Quantization**: Apply 1-bit/2-bit quantization (I2S, TL1, TL2)
3. **Inference**: Execute forward pass through quantized layers
4. **Output**: Generate tokens with performance receipt

**Problem Without Strict Guards:**
- Silent FP32 fallback in quantized layers → misleading receipts
- Performance baselines become unreliable (claiming 50 tok/s GPU, actually 12 tok/s CPU)
- Cross-validation false positives (comparing CPU perf to GPU baseline)
- Production deployment risk (unexpected performance degradation)

**Solution Impact:**
- **Honest Receipts**: Accurate reflection of actual computation path
- **Reliable Baselines**: Trustworthy performance benchmarks
- **Production Confidence**: Guaranteed quantized inference in strict mode
- **Early Detection**: Debug assertions catch fallback during development

### Motivation

**Why Now?** PR #452 established receipt verification infrastructure (schema v1.0.0, kernel recording, CI integration). Strict quantization guards extend this foundation to validate that quantized computation claims are backed by actual quantized kernels, not FP32 fallback.

**Neural Network Validation Context:**
- **I2S Quantization**: 99.8% correlation with FP32 reference (target)
- **TL1/TL2 Quantization**: 99.6% correlation with FP32 reference (target)
- **GPU Kernels**: Mixed precision (FP16/BF16) with quantized weights
- **CPU Kernels**: SIMD-optimized (AVX2/AVX-512/NEON) quantized matmul

---

## Acceptance Criteria

### AC1: Debug Assertions in QuantizedLinear::forward

**Requirement:** Add debug assertions in `fallback_i2s_matmul`, `forward_tl1_generic`, `forward_tl2_generic` that panic when fallback occurs in debug builds.

**Implementation Location:**
- File: `crates/bitnet-inference/src/layers/quantized_linear.rs`
- Lines: 562-624 (fallback paths)

**Panic Message Format:**
```rust
panic!("fallback to FP32 in debug mode: layer={}, qtype={:?}, reason={}",
       layer_name, quantization_type, fallback_reason);
```

**Validation Command:**
```bash
# AC1: Test I2S fallback detection in debug mode
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac1_debug_assert_i2s_fallback -- --nocapture

# AC1: Test TL1 fallback detection (ARM-specific)
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac1_debug_assert_tl1_fallback -- --nocapture

# AC1: Test TL2 fallback detection (x86-specific)
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac1_debug_assert_tl2_fallback -- --nocapture
```

**Success Criteria:**
- ✅ Debug builds panic immediately on FP32 fallback
- ✅ Release builds allow fallback (assertions compiled out)
- ✅ Panic message includes layer name, quantization type, and fallback reason
- ✅ Test coverage: Unit test simulates fallback path and verifies panic

---

### AC2: Debug Assertions in Attention Q/K/V/O Projections

**Requirement:** Add debug assertions in `BitNetAttention::compute_qkv_projections` before Q/K/V/O projection calls.

**Implementation Location:**
- File: `crates/bitnet-inference/src/layers/attention.rs`
- Lines: 474-515 (projection computation)

**Validation Logic:**
```rust
#[cfg(debug_assertions)]
{
    // Verify each projection uses native quantized kernels (no FP32 fallback)
    debug_assert!(self.q_proj.has_native_quantized_kernel(),
                  "Q projection would fall back to FP32 in debug mode");
    debug_assert!(self.k_proj.has_native_quantized_kernel(),
                  "K projection would fall back to FP32 in debug mode");
    debug_assert!(self.v_proj.has_native_quantized_kernel(),
                  "V projection would fall back to FP32 in debug mode");
    debug_assert!(self.o_proj.has_native_quantized_kernel(),
                  "O projection would fall back to FP32 in debug mode");
}
```

**Validation Command:**
```bash
# AC2: Test attention projection fallback detection
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac2_debug_assert_attention_projection -- --nocapture

# AC2: Verify all four projections use quantized kernels
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac2_all_projections_quantized -- --nocapture
```

**Success Criteria:**
- ✅ Debug assertions added in `compute_qkv_projections`
- ✅ All four projections (Q/K/V/O) validated before forward pass
- ✅ Panic in debug mode if any projection would fall back to FP32
- ✅ Test coverage: Integration test verifies all projections use quantized kernels

---

### AC3: Strict Mode Returns Err on Quantization Fallback

**Requirement:** Extend `StrictModeConfig` with `enforce_quantized_inference: bool` field. Modify `QuantizedLinear::forward` to check strict mode before allowing FP32 fallback.

**Configuration Extension:**
```rust
// crates/bitnet-common/src/strict_mode.rs
pub struct StrictModeConfig {
    pub enabled: bool,
    pub fail_on_mock: bool,
    pub require_quantization: bool,        // Existing
    pub enforce_quantized_inference: bool, // NEW: Reject FP32 fallback
    pub validate_performance: bool,
    // ... other fields
}
```

**Error Type:**
```rust
// crates/bitnet-common/src/error.rs
pub enum BitNetError {
    StrictMode(String), // Extend with detailed context
    // ... other variants
}
```

**Error Message Format:**
```
Strict mode: FP32 fallback rejected - qtype=I2S, device=Cuda(0),
layer_dims=[2048, 2048], reason=kernel_unavailable
```

**Environment Variable:**
```bash
# Enable strict mode (all checks)
BITNET_STRICT_MODE=1

# Granular control (quantization-specific)
BITNET_STRICT_REQUIRE_QUANTIZATION=1
```

**Validation Command:**
```bash
# AC3: Test strict mode rejects FP32 fallback
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac3_strict_mode_rejects_fallback -- --nocapture

# AC3: Verify error message includes detailed context
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac3_error_message_context -- --nocapture
```

**Success Criteria:**
- ✅ `StrictModeConfig` extended with `enforce_quantized_inference` field
- ✅ `QuantizedLinear::forward` checks strict mode before allowing fallback
- ✅ Returns `Err(BitNetError::StrictMode(...))` instead of falling back
- ✅ Error message includes: quantization type, device, layer dimensions, reason
- ✅ Test coverage: Unit test enables `BITNET_STRICT_MODE=1` and verifies error

---

### AC4: Strict Mode Validation in Attention Layer

**Requirement:** Extend `BitNetAttention::forward` to validate strict mode before processing projections. Check all four projections (Q/K/V/O) have native quantized kernels available.

**Implementation Strategy:**
```rust
// crates/bitnet-inference/src/layers/attention.rs
async fn forward(&self, hidden_states: &BitNetTensor, ...) -> Result<BitNetTensor> {
    // Strict mode validation: Check all projections have quantized kernels
    let strict_mode = StrictModeEnforcer::new();
    if strict_mode.get_config().enforce_quantized_inference {
        self.validate_projections_quantized()?;
    }

    // Proceed with forward pass
    let (q, k, v) = self.compute_qkv_projections(hidden_states).await?;
    // ... rest of attention computation
}

fn validate_projections_quantized(&self) -> Result<()> {
    let projections = [
        ("Q", &self.q_proj),
        ("K", &self.k_proj),
        ("V", &self.v_proj),
        ("O", &self.o_proj),
    ];

    for (name, proj) in &projections {
        if !proj.has_native_quantized_kernel() {
            return Err(BitNetError::StrictMode(format!(
                "Strict mode: {} projection would fall back to FP32 - qtype={:?}, device={:?}",
                name, proj.quantization_type, proj.device
            )));
        }
    }

    Ok(())
}
```

**Validation Command:**
```bash
# AC4: Test attention strict mode validation
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac4_attention_strict_mode_validation -- --nocapture
```

**Success Criteria:**
- ✅ `BitNetAttention::forward` validates strict mode before projections
- ✅ All four projections checked for native quantized kernel availability
- ✅ Returns `Err(BitNetError::StrictMode(...))` if any projection would fall back
- ✅ Test coverage: Integration test with `BITNET_STRICT_MODE=1` verifies rejection

---

### AC5: 16-Token Decode Integration Test in Strict Mode

**Requirement:** Create integration test that performs 16-token autoregressive decode with `BITNET_STRICT_MODE=1`. Verify all tokens decoded successfully without FP32 fallback errors.

**Test Structure:**
```rust
// crates/bitnet-inference/tests/strict_quantization_test.rs

/// AC5: 16-token decode in strict mode (CPU)
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

    assert!(result.is_ok(), "16-token decode should succeed in strict mode");
    let output = result.unwrap();

    // Verify 16 tokens generated
    assert_eq!(output.tokens_generated, 16);

    // Verify receipt shows quantized computation
    let receipt = output.receipt;
    assert_eq!(receipt.compute_path, "real");
    assert_eq!(receipt.kernel_path, Some("native_quantized".into()));
    assert!(receipt.kernels.iter().any(is_quantized_kernel));
}

/// AC5: 16-token decode in strict mode (GPU)
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

    assert!(result.is_ok(), "GPU 16-token decode should succeed");
    let output = result.unwrap();

    assert_eq!(output.tokens_generated, 16);

    // Verify GPU quantized kernels used
    let receipt = output.receipt;
    assert_eq!(receipt.backend, "cuda");
    assert!(receipt.kernels.iter().any(|id|
        id.starts_with("gemm_") || id.starts_with("i2s_gpu_")));
}
```

**Validation Command:**
```bash
# AC5: CPU integration test
BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac5_16_token_decode_cpu_strict_mode --test strict_quantization_test

# AC5: GPU integration test
BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo test --no-default-features --features gpu -p bitnet-inference \
  test_ac5_16_token_decode_gpu_strict_mode --test strict_quantization_test
```

**Success Criteria:**
- ✅ Integration test performs 16-token autoregressive decode
- ✅ Test enables `BITNET_STRICT_MODE=1` and `BITNET_STRICT_REQUIRE_QUANTIZATION=1`
- ✅ All tokens decoded successfully without FP32 fallback errors
- ✅ Receipt shows `compute_path="real"` with actual quantized kernel IDs
- ✅ Test coverage: Both CPU (`--features cpu`) and GPU (`--features gpu`) paths

---

### AC6: Receipt Validation for Quantized Computation Claims

**Requirement:** Extend receipt schema to include `kernel_path` field: `"native_quantized"` vs `"fp32_fallback"`. Receipts claiming `compute_path="quantized"` must have GPU kernel IDs or CPU quantized kernel IDs.

**Receipt Schema Extension (v1.0.0 → v1.1.0):**
```json
{
  "schema_version": "1.1.0",
  "backend": "cuda",
  "compute_path": "real",
  "kernel_path": "native_quantized",  // NEW FIELD
  "kernels": ["gemm_fp16", "i2s_gpu_quantize", "wmma_matmul"],
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

**Kernel ID Naming Convention:**

**Quantized Kernels (Native 1/2-bit Arithmetic):**
- **GPU Kernels:** `gemm_*`, `wmma_*`, `cuda_*`, `i2s_gpu_*`, `tl1_gpu_*`, `tl2_gpu_*`
- **CPU Kernels:** `i2s_gemv`, `tl1_neon_*`, `tl2_avx_*`, `quantized_matmul_*`

**FP32 Fallback Kernels (Dequantization + FP32 Arithmetic):**
- **Fallback Indicators:** `dequant_*`, `fp32_matmul`, `scalar_*`, `fallback_*`

**Validation Logic:**
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
                    "kernel_path='native_quantized' requires quantized kernel IDs"
                );
            }
            "fp32_fallback" => {
                // Validate that compute_path reflects fallback
                ensure!(
                    receipt.compute_path != "quantized",
                    "kernel_path='fp32_fallback' cannot claim compute_path='quantized'"
                );
            }
            _ => bail!("Invalid kernel_path: {}", kernel_path),
        }
    } else {
        // Schema v1.0.0: infer from kernels array
        let has_quantized = receipt.kernels.iter().any(is_quantized_kernel);
        let has_fallback = receipt.kernels.iter().any(is_fallback_kernel);

        if has_fallback && !has_quantized {
            log::warn!("Receipt uses FP32 fallback without quantized kernels");
        }
    }

    Ok(())
}

fn is_quantized_kernel(kernel_id: &str) -> bool {
    const QUANTIZED_PREFIXES: &[&str] = &[
        "gemm_", "wmma_", "i2s_gpu_", "tl1_gpu_", "tl2_gpu_",
        "i2s_gemv", "tl1_neon_", "tl2_avx_", "quantized_matmul_"
    ];
    QUANTIZED_PREFIXES.iter().any(|prefix| kernel_id.starts_with(prefix))
}

fn is_fallback_kernel(kernel_id: &str) -> bool {
    const FALLBACK_INDICATORS: &[&str] = &[
        "dequant_", "fp32_matmul", "scalar_", "fallback_", "mock_"
    ];
    FALLBACK_INDICATORS.iter().any(|ind| kernel_id.contains(ind))
}
```

**Validation Command:**
```bash
# AC6: Receipt with quantized kernels (valid)
cargo test -p xtask test_ac6_receipt_quantized_kernels_valid -- --nocapture

# AC6: Receipt claiming quantized without evidence (invalid)
cargo test -p xtask test_ac6_receipt_false_quantization_claim_fails -- --nocapture

# AC6: Receipt with explicit fp32_fallback (valid)
cargo test -p xtask test_ac6_receipt_fp32_fallback_explicit -- --nocapture

# AC6: End-to-end receipt verification
cargo run -p xtask -- benchmark --model tests/models/mini.gguf --tokens 128
cargo run -p xtask -- verify-receipt ci/inference.json
cargo run -p xtask -- verify-receipt --require-quantized-kernels ci/inference.json
```

**Success Criteria:**
- ✅ Receipt schema v1.1.0 defined with `kernel_path` and `quantization` fields
- ✅ Backward compatible with v1.0.0 (optional fields, ignored by old readers)
- ✅ `verify_quantization_claims` function validates kernel ID correlation
- ✅ Receipts claiming "quantized" must have actual quantized kernel IDs
- ✅ Test coverage: Receipt verification tests for valid/invalid claims

---

### AC7: Documentation Updates

**Requirement:** Comprehensive documentation for strict mode quantization guards.

**Modified Files:**
1. **`docs/development/validation-framework.md`**
   - Add section: "Strict Mode Quantization Guards"
   - Explain debug assertions, strict mode enforcement, receipt validation
   - Include troubleshooting guide for common fallback scenarios

2. **`docs/reference/quantization-support.md`**
   - Update section: "Fallback Behavior and Strict Mode Interactions"
   - Document I2S/TL1/TL2 fallback scenarios
   - Explain device-aware quantization selection

3. **`docs/environment-variables.md`**
   - Document `BITNET_STRICT_MODE=1` behavior
   - Document `BITNET_STRICT_REQUIRE_QUANTIZATION=1` granular control
   - Include examples of strict mode usage

**New File:**
4. **`docs/howto/troubleshooting-strict-mode.md`**
   - Comprehensive troubleshooting guide for strict mode errors
   - Common scenarios: missing GPU kernels, CPU fallback, SIMD unavailable
   - Resolution strategies: feature flag checks, device detection, model validation

**Validation Command:**
```bash
# AC7: Documentation tests
cargo test --doc --workspace --no-default-features --features cpu

# AC7: Link validation
mdbook test docs/

# AC7: Specific documentation modules
cargo test --doc -p bitnet-common strict_mode
```

**Success Criteria:**
- ✅ Validation framework documentation includes strict mode section
- ✅ Quantization support documentation updated with fallback behavior
- ✅ Environment variables documented with examples
- ✅ Troubleshooting guide created with common scenarios and resolutions

---

## Technical Architecture

### Three-Tier Validation Strategy

**Tier 1: Debug Assertions (Development)**
- **Purpose:** Catch FP32 fallback immediately during development
- **Scope:** Debug builds only (`#[cfg(debug_assertions)]`)
- **Behavior:** Panic with detailed error message
- **Overhead:** Zero in release builds (compiled out)
- **Target:** Developers running local tests

**Tier 2: Strict Mode Enforcement (Production)**
- **Purpose:** Reject FP32 fallback in production deployments
- **Scope:** Release builds with `BITNET_STRICT_MODE=1`
- **Behavior:** Return `Err(BitNetError::StrictMode(...))`
- **Overhead:** <1% (single boolean check per forward pass)
- **Target:** Production inference servers, CI baselines

**Tier 3: Receipt Validation (Verification)**
- **Purpose:** Validate receipts accurately reflect computation path
- **Scope:** Post-inference verification (`xtask verify-receipt`)
- **Behavior:** Exit code 1 if receipt claims don't match kernel IDs
- **Overhead:** Zero (offline verification)
- **Target:** CI gates, performance baseline validation

### Crate-Specific Implementation

**1. bitnet-inference (Primary Implementation)**
- **Purpose:** Runtime guards for quantized linear and attention layers
- **Modified Files:**
  - `src/layers/quantized_linear.rs`: Debug assertions + strict mode checks
  - `src/layers/attention.rs`: Projection validation before forward pass
- **New Files:**
  - `tests/strict_quantization_test.rs`: Integration tests for AC1-AC5

**2. bitnet-common (Strict Mode Configuration)**
- **Purpose:** Centralized strict mode enforcement and configuration
- **Modified Files:**
  - `src/strict_mode.rs`: Extend `StrictModeConfig` with `enforce_quantized_inference`
- **New Methods:**
  - `validate_quantization_fallback(qtype, device) -> Result<()>`
  - `check_quantization_path(kernel_ids: &[String]) -> Result<()>`

**3. bitnet-kernels (Kernel Availability Queries)**
- **Purpose:** Provide kernel availability information for strict mode checks
- **New Methods:**
  ```rust
  pub fn is_quantized_kernel_available(
      qtype: QuantizationType,
      device: Device,
      dims: (usize, usize)
  ) -> bool;
  ```

**4. xtask (Receipt Verification Extensions)**
- **Purpose:** Validate receipts for quantized computation claims
- **Modified Files:**
  - `src/main.rs`: Extend `verify_receipt_cmd` with `kernel_path` validation
- **Receipt Schema v1.1.0:**
  - Add `kernel_path` field: `"native_quantized"` | `"fp32_fallback"`
  - Add `quantization` section with `types_used`, `fallback_count`

---

## BitNet.rs Quantization Integration

### Quantization Types and Accuracy Targets

**I2S (2-bit Signed) Quantization:**
- **Range:** [-2, -1, 1, 2] (4 levels)
- **Accuracy Target:** ≥99.8% correlation with FP32 reference
- **GPU Kernels:** `i2s_gpu_quantize`, `i2s_gpu_pack`, `i2s_gpu_matmul`
- **CPU Kernels:** `i2s_gemv`, `quantized_matmul_i2s`
- **Fallback Scenarios:**
  - Kernel not compiled (missing `--features cpu|gpu`)
  - Device mismatch (tensor on GPU, layer on CPU)
  - Unsupported dimensions (non-multiple of SIMD block size)

**TL1 (Table Lookup 1) Quantization:**
- **Target Architecture:** ARM NEON
- **Accuracy Target:** ≥99.6% correlation with FP32 reference
- **Lookup Table:** 16-256 entries, cache-friendly
- **CPU Kernels:** `tl1_neon_pack`, `tl1_neon_matmul`
- **Fallback Scenarios:**
  - ARM NEON not available (x86 platform)
  - Lookup table construction fails (memory allocation)

**TL2 (Table Lookup 2) Quantization:**
- **Target Architecture:** x86 AVX2/AVX-512
- **Accuracy Target:** ≥99.6% correlation with FP32 reference
- **Lookup Table:** 256-4096 entries, larger for AVX-512
- **CPU Kernels:** `tl2_avx_matmul`, `tl2_avx512_pack`
- **Fallback Scenarios:**
  - AVX2/AVX-512 not available (ARM platform or older x86 CPUs)
  - Unsupported tensor dimensions

### Device-Aware Execution

**GPU Execution Path:**
```rust
async fn forward_i2s(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
    let provider = self.kernel_manager.select_best()?;

    // Check native quantized kernel availability
    let has_native = bitnet_kernels::is_quantized_kernel_available(
        QuantizationType::I2S,
        self.device,
        (self.in_features, self.out_features)
    );

    // Strict mode validation
    if !has_native {
        #[cfg(debug_assertions)]
        panic!("fallback to FP32 in debug mode: I2S GPU kernel unavailable");

        let strict_mode = StrictModeEnforcer::new();
        if strict_mode.get_config().enforce_quantized_inference {
            return Err(BitNetError::StrictMode(format!(
                "Native I2S GPU kernel unavailable - device={:?}, dims=({}, {})",
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

    let strict_mode = StrictModeEnforcer::new();
    if strict_mode.get_config().enforce_quantized_inference {
        return Err(BitNetError::StrictMode(
            "ARM NEON required for TL1 quantization".into()
        ));
    }

    self.forward_tl1_generic(input).await
}
```

### Mixed Precision Support

**FP16/BF16 GPU Kernels:**
- **Compute Capability:** 6.1+ (Pascal, Volta, Turing, Ampere, Ada)
- **Kernel IDs:** `gemm_fp16`, `wmma_fp16`, `i2s_gpu_fp16`
- **Strict Mode:** FP16/BF16 is acceptable (not FP32 fallback)
- **Clarification:** FP16/BF16 → FP32 dequantization is **rejected** (considered FP32 fallback)
- **Acceptable:** I2S quantized → FP16 matmul (native GPU quantized path)

---

## Implementation Roadmap

### Phase 1: Core Runtime Guards (Week 1)

**Day 1-2: Debug Assertions in Quantized Linear**
- Modify: `crates/bitnet-inference/src/layers/quantized_linear.rs` (lines 562-624)
- Add `#[cfg(debug_assertions)] panic!(...)` in fallback paths
- Test: AC1 unit tests (`test_ac1_debug_assert_*`)

**Day 3-4: Debug Assertions in Attention Projections**
- Modify: `crates/bitnet-inference/src/layers/attention.rs` (lines 474-515)
- Add `#[cfg(debug_assertions)]` checks before Q/K/V/O projection calls
- Test: AC2 unit tests (`test_ac2_debug_assert_*`)

**Day 5-7: Strict Mode Configuration Extensions**
- Modify: `crates/bitnet-common/src/strict_mode.rs` (lines 14-121)
- Add `enforce_quantized_inference: bool` field
- Implement `validate_quantization_fallback` method
- Test: Strict mode configuration tests

### Phase 2: Strict Mode Enforcement (Week 2)

**Day 8-10: Strict Mode in Quantized Linear**
- Modify: `crates/bitnet-inference/src/layers/quantized_linear.rs` (lines 260-334)
- Check `enforce_quantized_inference` before allowing fallback
- Return `Err(BitNetError::StrictMode(...))` instead of falling back
- Test: AC3 unit tests (`test_ac3_strict_mode_*`)

**Day 11-13: Strict Mode in Attention Layer**
- Modify: `crates/bitnet-inference/src/layers/attention.rs` (lines 436-471)
- Validate strict mode before processing projections
- Check all four projections have native quantized kernels
- Test: AC4 unit tests (`test_ac4_attention_strict_*`)

**Day 14: Integration Test for 16-Token Decode**
- Create: `crates/bitnet-inference/tests/strict_quantization_test.rs`
- Implement 16-token autoregressive decode with `BITNET_STRICT_MODE=1`
- Feature-gated tests for CPU and GPU paths
- Test: AC5 integration tests (`test_ac5_16_token_decode_*`)

### Phase 3: Receipt Validation Extensions (Week 3)

**Day 15-17: Receipt Schema Extensions**
- Modify: `xtask/src/main.rs` (verify_receipt_cmd function)
- Define `ReceiptV1_1` struct with `kernel_path` and `quantization` fields
- Implement backward-compatible parsing (v1.0.0 → v1.1.0)
- Test: Schema parsing tests

**Day 18-20: Kernel Path Validation Logic**
- Modify: `xtask/src/main.rs` (verify_receipt_cmd function)
- Implement `verify_quantization_claims` function
- Add `is_quantized_kernel` and `is_fallback_kernel` helpers
- Test: AC6 receipt verification tests

**Day 21: Receipt Verification Integration**
- Integrate `verify_quantization_claims` into `verify_receipt_cmd`
- Add `--require-quantized-kernels` flag
- Test: End-to-end receipt verification

### Phase 4: Documentation and Testing (Week 4)

**Day 22-24: Documentation Updates**
- Modify: `docs/development/validation-framework.md`
- Modify: `docs/reference/quantization-support.md`
- Modify: `docs/environment-variables.md`
- Create: `docs/howto/troubleshooting-strict-mode.md`

**Day 25-27: Cross-Validation and Baseline Establishment**
- Run cross-validation with strict mode enabled
- Establish performance baselines for CPU and GPU
- Verify receipts from benchmarks pass validation

**Day 28: Final Integration Testing**
- Full workspace test suite with strict mode
- Feature-gated smoke testing (cpu/gpu/none)
- Backward compatibility verification

---

## Testing Strategy

### Unit Tests with `// AC:ID` Tags

All unit tests include `// AC:ID` comment tags for traceability to acceptance criteria.

**Debug Assertions Tests:**
```rust
// AC1: Debug assertions in fallback_i2s_matmul
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "fallback to FP32 in debug mode")]
fn test_ac1_debug_assert_i2s_fallback() { /* ... */ }
```

**Strict Mode Tests:**
```rust
// AC3: Strict mode rejects FP32 fallback
#[test]
fn test_ac3_strict_mode_rejects_fallback() {
    std::env::set_var("BITNET_STRICT_MODE", "1");
    // ... test implementation
}
```

### Integration Tests

**16-Token Decode:**
- Feature-gated: `#[cfg(feature = "cpu")]`, `#[cfg(feature = "gpu")]`
- Deterministic: `BITNET_DETERMINISTIC=1 BITNET_SEED=42`
- Strict mode: `BITNET_STRICT_MODE=1`
- Receipt validation: Verify `kernel_path="native_quantized"`

### Cross-Validation Requirements

```bash
# Ensure strict mode doesn't break C++ reference alignment
BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo run -p xtask -- crossval --model tests/models/mini.gguf
```

---

## Documentation Requirements

### Updated Files

1. **`docs/development/validation-framework.md`**: Strict mode section
2. **`docs/reference/quantization-support.md`**: Fallback behavior
3. **`docs/environment-variables.md`**: `BITNET_STRICT_MODE` documentation

### New File

4. **`docs/howto/troubleshooting-strict-mode.md`**: Comprehensive troubleshooting guide

---

## Success Metrics

### Functional Metrics

- ✅ All 7 acceptance criteria validated with measurable commands
- ✅ Debug assertions catch fallback in development (100% detection)
- ✅ Strict mode rejects fallback in production (100% enforcement)
- ✅ Receipt validation correlates claims with kernel IDs (100% accuracy)

### Performance Metrics

- ✅ Debug assertions: <0.1% overhead (only in debug builds, zero in release)
- ✅ Strict mode checks: <1% overhead (single boolean check per forward pass)
- ✅ Receipt generation: <5 ms per 16-token decode (negligible)
- ✅ No measurable performance degradation in release builds

### Quality Metrics

- ✅ Test coverage: ≥95% line coverage for strict mode code paths
- ✅ Cross-validation: Maintains C++ reference parity (1e-5 tolerance)
- ✅ Backward compatibility: Zero breaking changes to public API
- ✅ Documentation: Complete guides for all validation procedures

---

## Related Work

### Foundation

- **PR #452**: Receipt Verification Infrastructure
  - Schema v1.0.0
  - Kernel recording
  - CI integration
  - `xtask verify-receipt` command

### Integration Points

- **Issue #261**: Native I2S/TL1/TL2 quantization implementation
- **Issue #439**: GPU detection override for deterministic testing
- **Issue #260**: Mock inference elimination

### Future Work

- **Issue #454**: GPU kernel verification with CUDA runtime validation
- **Issue #455**: Performance regression detection with statistical baselines
- **Issue #456**: Cross-validation automation with C++ reference

---

## Appendix A: File Modification Summary

### Files to Modify (10 files)

1. `crates/bitnet-inference/src/layers/quantized_linear.rs`
2. `crates/bitnet-inference/src/layers/attention.rs`
3. `crates/bitnet-common/src/strict_mode.rs`
4. `crates/bitnet-quantization/src/lib.rs`
5. `crates/bitnet-kernels/src/lib.rs`
6. `xtask/src/main.rs`
7. `docs/development/validation-framework.md`
8. `docs/reference/quantization-support.md`
9. `docs/environment-variables.md`

### Files to Create (2 files)

1. `crates/bitnet-inference/tests/strict_quantization_test.rs`
2. `docs/howto/troubleshooting-strict-mode.md`

---

## Appendix B: Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `BITNET_STRICT_MODE` | `0`, `1`, `true`, `false` | `0` | Enable all strict mode checks |
| `BITNET_STRICT_REQUIRE_QUANTIZATION` | `0`, `1` | `0` | Granular control for quantization enforcement |
| `BITNET_STRICT_FAIL_ON_MOCK` | `0`, `1` | `0` | Fail on mock computation detection |
| `BITNET_STRICT_VALIDATE_PERFORMANCE` | `0`, `1` | `0` | Validate performance metrics against baselines |
| `BITNET_FORCE_QUANTIZATION_FALLBACK` | `0`, `1` | `0` | Force FP32 fallback (testing only) |
| `BITNET_TRACK_KERNEL_IDS` | `0`, `1` | `0` | Track kernel IDs for validation (testing only) |
| `BITNET_GPU_FAKE` | `cuda`, `none` | (auto-detect) | Override GPU detection for deterministic testing |

---

**Document Status:** Approved - Ready for Implementation
**Next Steps:** Implementation team begins Phase 1 (Core Runtime Guards)
