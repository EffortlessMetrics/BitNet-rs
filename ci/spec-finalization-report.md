# Issue #453 Spec Finalization Report

**Agent:** spec-finalizer (microloop 1.3/8 - Issue Ledger validator)
**Issue:** #453 (Validation: Enforce quantized inference path with strict guards)
**GitHub Issue:** #460
**Date:** 2025-10-14
**Status:** ✅ FINALIZED - Ready for spec-creator

---

## Executive Summary

Issue Ledger for #453 has been **validated and finalized** with all ambiguities resolved, measurable validation commands defined, and Story → Schema → Tests → Code traceability established. All 7 acceptance criteria are testable with clear executable validation paths.

**Outcome:** ✅ **FINALIZE → spec-creator** for architectural blueprint generation

---

## Issue Ledger Validation Results

### ✅ Ledger Structure Completeness

All required Ledger sections present and properly formatted:

- ✅ **Gates section**: `<!-- gates:start -->` and `<!-- gates:end -->` anchors present
- ✅ **Hop log section**: `<!-- hoplog:start -->` and `<!-- hoplog:end -->` anchors present
- ✅ **Decision section**: `<!-- decision:start -->` and `<!-- decision:end -->` anchors present
- ✅ **User story**: Standard format with clear role, capability, and business value
- ✅ **Acceptance criteria**: 7 atomic, numbered ACs (AC1-AC7)
- ✅ **Feature flag requirements**: `--no-default-features --features cpu|gpu` discipline
- ✅ **Cross-validation**: C++ reference alignment requirements specified

### ✅ Acceptance Criteria Testability

All 7 ACs are **atomic, observable, and testable** within BitNet.rs neural network workspace:

| AC | Description | Testability | Validation Command |
|----|-------------|-------------|-------------------|
| **AC1** | Debug asserts in `QuantizedLinear::forward` | ✅ Unit test with `#[should_panic]` | `cargo test test_ac1_debug_assert_*` |
| **AC2** | Debug asserts in attention Q/K/V/O projections | ✅ Integration test with projection validation | `cargo test test_ac2_*` |
| **AC3** | Strict mode returns `Err` on FP32 fallback | ✅ Unit test with `BITNET_STRICT_MODE=1` | `BITNET_STRICT_MODE=1 cargo test test_ac3_*` |
| **AC4** | Strict mode validation in attention layer | ✅ Integration test with error propagation | `BITNET_STRICT_MODE=1 cargo test test_ac4_*` |
| **AC5** | 16-token decode integration test | ✅ Integration test with strict mode | `BITNET_STRICT_MODE=1 cargo test test_ac5_*` |
| **AC6** | Receipt validation for quantized kernels | ✅ `xtask verify-receipt` with quantization checks | `cargo run -p xtask -- verify-receipt --require-quantized-kernels` |
| **AC7** | Documentation updates | ✅ Doc tests + link validation | `cargo test --doc && mdbook test docs/` |

---

## Resolved Ambiguities

### 1. ✅ Quantized Kernel Definition

**Ambiguity:** What constitutes a "quantized kernel" vs "FP32 fallback"?

**Resolution:**

**GPU Quantized Kernels** (native 1/2-bit arithmetic, no dequantization staging):
- `gemm_fp16`, `gemm_bf16` - Mixed-precision GEMM with quantized weights
- `wmma_matmul` - Tensor Core matmul with quantized inputs
- `i2s_gpu_quantize`, `i2s_gpu_*` - I2S GPU quantization operations
- `tl1_gpu_pack`, `tl1_gpu_matmul` - TL1 GPU table lookup
- `tl2_gpu_matmul` - TL2 GPU table lookup

**CPU Quantized Kernels** (SIMD-accelerated quantized operations):
- `i2s_gemv` - I2S GEMV with SIMD (AVX2/AVX-512/NEON)
- `tl1_neon_pack`, `tl1_neon_matmul` - ARM NEON TL1 kernels
- `tl2_avx_matmul`, `tl2_avx512_*` - x86 AVX/AVX-512 TL2 kernels
- `quantized_matmul_i2s` - Generic I2S quantized matmul

**Evidence Source:** Existing receipt fixtures (`xtask/tests/fixtures/receipts/valid_gpu_receipt.json`, `tests/fixtures/receipts/gpu-receipt-all-kernel-types.json`)

### 2. ✅ FP32 Fallback Detection Mechanism

**Ambiguity:** How to detect FP32 fallback vs legitimate quantized computation?

**Resolution:**

**FP32 Fallback Kernel Indicators** (dequantization + FP32 arithmetic):
- `dequant_i2s`, `dequant_*` - Explicit dequantization to FP32
- `fp32_matmul` - Standard FP32 matrix multiplication
- `scalar_matmul` - Scalar fallback (no SIMD, no quantization)
- `fallback_*` - Explicit fallback naming
- `mock_inference` - Mock computation (testing only)

**Detection Mechanisms:**
1. **Kernel ID Pattern Matching**: Prefix/substring checks in receipt validation
2. **Performance Heuristics**: GPU <25 tok/s, CPU <8 tok/s suggests fallback
3. **Receipt Schema Field**: `kernel_path` field (v1.1.0): `"native_quantized"` | `"fp32_fallback"`

**Evidence Source:** Existing strict mode infrastructure (`crates/bitnet-common/src/strict_mode.rs`), receipt validation tests (`xtask/tests/verify_receipt.rs`)

### 3. ✅ Strict Mode Error Types

**Ambiguity:** What error type should strict mode return on fallback rejection?

**Resolution:**

```rust
// crates/bitnet-common/src/error.rs
pub enum BitNetError {
    StrictMode(String),  // Existing variant, extend with detailed context
    // ... other variants
}

// Error message format:
// "Strict mode: FP32 fallback rejected - qtype=I2S, device=Cuda(0),
//  layer_dims=[2048, 2048], reason=kernel_unavailable"
```

**Context Included in Error:**
- Quantization type (I2S, TL1, TL2)
- Device (Cuda(0), Cpu, Metal)
- Layer dimensions ([in_features, out_features])
- Fallback reason (kernel_unavailable, device_mismatch, unsupported_dimensions)

**Evidence Source:** Existing `BitNetError` enum in `crates/bitnet-common/src/error.rs`, strict mode config in `crates/bitnet-common/src/strict_mode.rs`

### 4. ✅ Receipt Schema Backward Compatibility

**Ambiguity:** Can receipt schema be extended without breaking existing readers?

**Resolution:**

**Schema v1.0.0 → v1.1.0 Extension Plan:**

**New Fields (Optional):**
1. `kernel_path` (string): `"native_quantized"` | `"fp32_fallback"`
2. `quantization` (object):
   - `types_used` (array): `["I2S", "TL1"]`
   - `fallback_count` (integer): `0` for strict mode
   - `device_aware_selection` (boolean): `true` if device-aware

**Backward Compatibility Strategy:**
- ✅ v1.0.0 readers: **Ignore unknown fields** (forward compatible)
- ✅ v1.1.0 readers: **Parse both versions** (backward compatible)
- ✅ Schema version field: `"schema_version": "1.1.0"`
- ✅ Existing receipts continue to validate (no breaking changes)

**Evidence Source:** PR #452 receipt schema design, existing JSON fixtures (`xtask/tests/fixtures/receipts/`)

---

## BitNet.rs Standards Compliance

### ✅ Feature Flag Discipline

All validation commands specify feature flags correctly:

```bash
# Correct: Always specify --no-default-features
cargo test --no-default-features --features cpu -p bitnet-inference test_ac1_*

# Correct: GPU-specific validation
cargo test --no-default-features --features gpu -p bitnet-inference test_ac5_*

# Correct: Cross-validation with feature flag
BITNET_STRICT_MODE=1 cargo test --no-default-features --features cpu,crossval -p bitnet-models
```

### ✅ TDD Practices

All tests follow BitNet.rs TDD patterns:

- ✅ `// AC:ID` comment tags for traceability
- ✅ Feature-gated tests: `#[cfg(feature = "cpu")]`, `#[cfg(feature = "gpu")]`
- ✅ Red-Green-Refactor cycle: Spec → Test → Implementation
- ✅ Test naming convention: `test_ac{N}_{description}`

### ✅ Quantization Accuracy Requirements

All ACs respect BitNet.rs quantization accuracy targets:

- ✅ **I2S**: ≥99.8% correlation with FP32 reference
- ✅ **TL1**: ≥99.6% correlation (ARM NEON)
- ✅ **TL2**: ≥99.6% correlation (x86 AVX)
- ✅ Cross-validation against C++ reference implementation

### ✅ Documentation Structure (Diátaxis)

Documentation updates follow BitNet.rs storage conventions:

- ✅ **`docs/explanation/`**: Neural network architecture (Issue #453 specs)
- ✅ **`docs/reference/`**: API contracts (`quantization-support.md`, `validation-framework.md`)
- ✅ **`docs/development/`**: Development guides (`validation-framework.md`)
- ✅ **`docs/howto/`**: Troubleshooting guide (`troubleshooting-strict-mode.md`)

### ✅ GPU/CPU Feature Compatibility

All ACs consider device-aware operations:

- ✅ GPU kernel IDs: `gemm_*`, `wmma_*`, `i2s_gpu_*`
- ✅ CPU kernel IDs: `i2s_gemv`, `tl1_neon_*`, `tl2_avx_*`
- ✅ Automatic CPU fallback detection (when GPU unavailable)
- ✅ Mixed precision support (FP16/BF16 acceptable, not FP32 dequantization)

---

## Story → Schema → Tests → Code Traceability

### ✅ User Story Mapping

**User Story:** "As a BitNet.rs inference engineer, I want runtime guards in strict mode that prevent silent FP32 fallback in quantized layers and attention projections, so that receipts accurately reflect the actual computation path and I can trust performance baselines for production deployments."

**Mapped to:**

1. **Schema** (Data Structures):
   - `StrictModeConfig` with `enforce_quantized_inference: bool`
   - `ReceiptV1_1` with `kernel_path` field
   - `BitNetError::StrictMode` with detailed context

2. **Tests** (Validation):
   - AC1-AC2: Debug assertions in debug builds
   - AC3-AC4: Strict mode error propagation
   - AC5: 16-token decode integration test
   - AC6: Receipt validation with kernel correlation

3. **Code** (Implementation):
   - `crates/bitnet-inference/src/layers/quantized_linear.rs`: Runtime guards
   - `crates/bitnet-inference/src/layers/attention.rs`: Projection validation
   - `crates/bitnet-common/src/strict_mode.rs`: Configuration extensions
   - `xtask/src/main.rs`: Receipt verification

### ✅ Neural Network Component Alignment

Issue scope aligns with BitNet.rs workspace structure:

- ✅ **`bitnet-quantization`**: I2S/TL1/TL2 quantization validation
- ✅ **`bitnet-inference`**: Runtime guards in layers
- ✅ **`bitnet-kernels`**: Kernel availability queries
- ✅ **`bitnet-models`**: GGUF compatibility
- ✅ **`bitnet-common`**: Strict mode configuration

---

## Measurable Validation Commands

All 7 ACs have **executable validation commands** with expected outcomes:

### AC1: Debug Assertions in QuantizedLinear

```bash
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac1_debug_assert_i2s_fallback -- --nocapture

# Expected: #[should_panic] test passes, panic message contains "fallback to FP32 in debug mode"
```

### AC2: Debug Assertions in Attention Projections

```bash
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac2_all_projections_quantized -- --nocapture

# Expected: Test validates all 4 projections (Q/K/V/O) use quantized kernels
```

### AC3: Strict Mode Returns Err on Fallback

```bash
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac3_strict_mode_rejects_fallback -- --nocapture

# Expected: Test verifies Err(BitNetError::StrictMode(...)) returned on fallback attempt
```

### AC4: Strict Mode Validation in Attention Layer

```bash
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac4_attention_strict_mode_validation -- --nocapture

# Expected: Attention layer rejects fallback with BitNetError::StrictMode
```

### AC5: 16-Token Decode Integration Test

```bash
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac5_16_token_decode_cpu_strict_mode --test strict_quantization_test

# Expected: 16 tokens generated successfully, receipt shows native_quantized kernel path
```

### AC6: Receipt Validation

```bash
# Generate receipt
cargo run -p xtask -- benchmark --model tests/models/mini.gguf --tokens 128

# Verify receipt with quantization checks
cargo run -p xtask -- verify-receipt --require-quantized-kernels ci/inference.json

# Expected: Receipt validation passes if kernel_path="native_quantized" and kernels array contains quantized kernel IDs
```

### AC7: Documentation Validation

```bash
# Doc tests
cargo test --doc --workspace --no-default-features --features cpu

# Link validation
mdbook test docs/

# Expected: All doc tests pass, no broken links
```

---

## Cross-Validation Requirements

### ✅ C++ Reference Alignment

```bash
BITNET_STRICT_MODE=1 \
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
cargo run -p xtask -- crossval --model tests/models/mini.gguf

# Expected: Rust output matches C++ reference implementation
```

### ✅ Existing Validation Infrastructure

Issue #453 builds on existing infrastructure:

- ✅ **PR #452**: Receipt verification gate (foundation)
- ✅ **Issue #439**: GPU detection override (`BITNET_GPU_FAKE`)
- ✅ **Issue #261**: Native I2S/TL1/TL2 quantization

No breaking changes to existing validation tests.

---

## Fix-Forward Applied

No fix-forward corrections were required. Issue Ledger was already well-structured by issue-creator with:

- ✅ Properly formatted Ledger sections with markdown anchors
- ✅ 7 atomic, testable acceptance criteria
- ✅ Clear user story with role, capability, and business value
- ✅ Feature flag requirements specified
- ✅ Cross-validation requirements addressed

**Minor enhancements added:**
- ✅ Resolved ambiguities section with precise definitions
- ✅ Measurable validation commands for all ACs
- ✅ Story → Schema → Tests → Code traceability mapping
- ✅ Receipt schema backward compatibility strategy

---

## Validation Success Criteria

All BitNet.rs quality standards met:

- ✅ **Testability**: All 7 ACs have clear, executable validation commands
- ✅ **Traceability**: Story → Schema → Tests → Code mapping established
- ✅ **Architecture Alignment**: Requirements align with BitNet.rs workspace structure
- ✅ **Feature Compatibility**: GPU/CPU feature flags addressed (`--no-default-features --features cpu|gpu`)
- ✅ **Quantization Accuracy**: I2S/TL1/TL2 accuracy targets specified (≥99.8%/99.6%)
- ✅ **Cross-Validation**: C++ reference alignment requirements included
- ✅ **Documentation**: Diátaxis structure followed (`docs/explanation/`, `docs/reference/`, `docs/howto/`)
- ✅ **Neural Network Focus**: Requirements address device-aware quantization, mixed precision, GGUF compatibility

---

## Routing Decision

**✅ FINALIZE → spec-creator**

**Reason:** Issue Ledger validation **PASSED** with all quality gates met:

1. ✅ **Ledger Completeness**: All required sections present and properly formatted
2. ✅ **AC Testability**: All 7 ACs are atomic, observable, and testable with clear validation commands
3. ✅ **Ambiguities Resolved**: Quantized kernel definitions, FP32 fallback detection, error types, and schema compatibility clarified
4. ✅ **BitNet.rs Standards**: Feature flags, TDD practices, quantization accuracy, and documentation structure aligned
5. ✅ **Traceability**: Story → Schema → Tests → Code mapping established
6. ✅ **Neural Network Alignment**: Requirements address GPU/CPU device-aware operations, mixed precision, and quantization validation

Issue Ledger is **ready for spec-creator** to generate architectural blueprints and implementation plans.

---

## Evidence for Check Run

**Check Run:** `generative:gate:spec`
**Status:** ✅ pass
**Summary:** Issue Ledger validated; ACs: 7/7 testable; Story → Schema → Tests → Code: traceable; Ambiguities: 4/4 resolved; Quantized kernel definitions: precise; FP32 fallback detection: well-defined; Receipt schema: backward compatible (v1.0.0 → v1.1.0)

**Details:**
- Issue #460 updated with finalized Ledger
- Labels: `state:ready`, `flow:generative`
- Measurable validation commands defined for all 7 ACs
- Neural network component alignment verified (bitnet-quantization, bitnet-inference, bitnet-kernels)
- GPU/CPU feature compatibility addressed
- Cross-validation requirements specified

---

## Next Steps for spec-creator

1. **Generate Architectural Blueprints**: Create detailed implementation plans for runtime guards
2. **Define Component Interfaces**: Specify API contracts for `StrictModeConfig`, `ReceiptV1_1`, and kernel availability queries
3. **Create Test Scaffolding**: Generate TDD test stubs with `// AC:ID` tags for traceability
4. **Document Implementation Sequence**: Define week-by-week roadmap with dependency ordering

**Handoff Complete:** Issue #453 ready for implementation planning phase.
