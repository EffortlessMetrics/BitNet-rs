# Issue #453: [Validation] Enforce quantized inference path with strict guards

## Context

BitNet.rs provides receipt-based proof of computation (PR #452) but lacks runtime guarantees that quantized inference actually uses native quantized kernels instead of silently falling back to FP32 dequantization. This gap means receipts can claim "quantized computation" while the actual inference path uses mock FP32 operations, undermining both performance claims and correctness validation.

### Problem Background

The current implementation allows silent fallback to FP32 staging in two critical locations:
1. **`QuantizedLinear::forward`**: Can dequantize weights and use standard FP32 matrix multiplication when native quantized kernels fail
2. **Attention projections (Q/K/V/O)**: Can fall back to FP32 computation in `BitNetAttention::compute_qkv_projections`

These fallback paths are necessary for graceful degradation but create a validation gap: receipts prove *that* computation occurred but not *how* it occurred. Without strict mode guards, production inference can silently degrade from native I2S/TL1/TL2 quantized operations to FP32 fallback, invalidating performance baselines and accuracy claims.

### Affected Components

- **`bitnet-inference/src/layers/quantized_linear.rs`**: Core quantized linear layer with fallback paths
- **`bitnet-inference/src/layers/attention.rs`**: Multi-head attention with quantized projections
- **`bitnet-common/src/strict_mode.rs`**: Existing strict mode enforcement infrastructure
- **Receipt verification system**: Needs correlation between claimed compute path and actual kernel usage

### Inference Pipeline Impact

**Model Loading → Quantization → Kernels → Inference → Output**

This issue specifically affects:
- **Kernels stage**: Ensures native quantized kernels (I2S, TL1, TL2) are actually used
- **Inference stage**: Prevents silent FP32 fallback during forward passes
- **Output stage**: Guarantees receipt claims match actual computation path

### Quantization Correctness Requirements

- **I2S quantization**: Must use native `quantized_matmul_i2s` without FP32 dequantization staging
- **TL1/TL2 quantization**: Must use native `quantized_matmul_tl1/tl2` kernels
- **Attention projections**: Q/K/V/O projections must use quantized linear layers without fallback
- **Receipt validation**: Claimed "quantized" computation must correlate with actual quantized kernel IDs

## User Story

As a BitNet.rs inference engineer, I want runtime guards in strict mode that prevent silent FP32 fallback in quantized layers and attention projections, so that receipts accurately reflect the actual computation path and I can trust performance baselines for production deployments.

## Acceptance Criteria

AC1: **Debug asserts in `QuantizedLinear::forward`**
- Add debug assertions in `fallback_i2s_matmul`, `forward_tl1_generic`, `forward_tl2_generic` that panic when fallback occurs in debug builds
- Assertions should include layer name, quantization type, and fallback reason in panic message
- Release builds should still allow fallback for production resilience
- Test coverage: Unit test that simulates fallback path and verifies panic in debug mode

AC2: **Debug asserts in attention Q/K/V/O projections**
- Add debug assertions in `BitNetAttention::compute_qkv_projections` before projection calls
- Verify each projection (q_proj, k_proj, v_proj, o_proj) uses native quantized paths
- Assertions should panic in debug mode if projection would fall back to FP32
- Test coverage: Integration test that verifies all four projections use quantized kernels

AC3: **Strict mode returns `Err` on quantization fallback**
- Extend `StrictModeConfig` with `enforce_quantized_inference: bool` field (enabled when `BITNET_STRICT_MODE=1`)
- Modify `QuantizedLinear::forward` to check strict mode before allowing FP32 fallback
- Return `Err(BitNetError::StrictMode(...))` instead of falling back when strict mode is enabled
- Error message should include: quantization type, device, layer dimensions, and fallback trigger
- Test coverage: Unit test that enables `BITNET_STRICT_MODE=1` and verifies error on fallback attempt

AC4: **Strict mode validation in attention layer**
- Extend `BitNetAttention::forward` to validate strict mode before processing projections
- Check that all four projections (Q/K/V/O) have native quantized kernels available
- Return `Err(BitNetError::StrictMode(...))` if any projection would fall back to FP32
- Test coverage: Integration test with `BITNET_STRICT_MODE=1` that verifies attention layer rejects fallback

AC5: **16-token decode integration test in strict mode**
- Create integration test `tests/strict_quantization_test.rs` that performs 16-token autoregressive decode
- Test should enable `BITNET_STRICT_MODE=1` and `BITNET_STRICT_REQUIRE_QUANTIZATION=1`
- Verify all tokens are decoded successfully without FP32 fallback errors
- Test should validate receipt shows "quantized" compute path with real kernel IDs
- Test coverage: Both CPU and GPU paths (feature-gated)

AC6: **Receipt validation for quantized computation claims**
- Extend receipt schema to include `kernel_path` field: "native_quantized" vs "fp32_fallback"
- Modify `verify-receipt` gate to validate correlation between `compute_path="quantized"` and actual kernel IDs
- Receipts claiming "quantized" must have GPU kernel IDs (`gemm_*`, `i2s_gpu_*`) or CPU quantized kernel IDs
- Receipts with FP32 fallback must explicitly declare `kernel_path="fp32_fallback"`
- Test coverage: Receipt verification unit tests for valid and invalid quantization claims

AC7: **Documentation updates**
- Add section to `docs/development/validation-framework.md` explaining strict mode quantization guards
- Update `docs/reference/quantization-support.md` with fallback behavior and strict mode interactions
- Document environment variable `BITNET_STRICT_MODE=1` behavior in `docs/environment-variables.md`
- Add troubleshooting guide for strict mode errors in `docs/howto/troubleshooting-strict-mode.md`

## Technical Implementation Notes

### Affected Crates
- **bitnet-inference**: Primary implementation location for runtime guards
- **bitnet-common**: Strict mode configuration extensions
- **bitnet-quantization**: Fallback path tracking and validation
- **bitnet-kernels**: Kernel availability queries for strict mode checks
- **xtask**: Receipt verification extensions for kernel path validation

### Pipeline Stages
- **Quantization stage**: Validate native quantized kernels are available before layer creation
- **Inference stage**: Runtime checks in forward passes to prevent FP32 fallback
- **Output stage**: Receipt generation must accurately reflect actual computation path

### Performance Considerations
- **Debug mode overhead**: Assertions add negligible overhead (only in debug builds)
- **Strict mode overhead**: Single boolean check per forward pass (< 1% overhead)
- **Receipt validation**: Schema extension requires backward-compatible versioning
- **GPU/CPU considerations**: Device-aware validation (different kernel IDs for GPU vs CPU)

### Quantization Requirements
- **I2S validation**: Ensure `quantized_matmul_i2s` is used without dequantization staging
- **TL1 validation**: ARM NEON optimization with fallback detection
- **TL2 validation**: x86 AVX optimization with fallback detection
- **Accuracy preservation**: ≥99.8% correlation for I2S, ≥99.6% for TL1/TL2 (existing requirement)

### Cross-Validation
- **C++ reference alignment**: Strict mode should match C++ implementation's quantization path enforcement
- **Baseline validation**: Ensure strict mode doesn't change accuracy when native kernels are available
- **Performance parity**: Validate that strict mode enforcement doesn't degrade performance

### Feature Flags
- **CPU feature**: `--no-default-features --features cpu` must support strict mode quantization checks
- **GPU feature**: `--no-default-features --features gpu` must validate GPU quantized kernel availability
- **Graceful degradation**: Release builds without strict mode should still allow fallback for resilience

### GGUF Compatibility
- **Model loading**: Strict mode should validate quantization types in GGUF metadata
- **Tensor alignment**: Ensure quantized layers align with GGUF tensor formats (I2S, IQ2_S, etc.)
- **Metadata validation**: Cross-reference GGUF `general.file_type` with actual quantization paths used

### Testing Strategy

**TDD with `// AC:ID` tags:**
```rust
// AC1: Debug assertions in QuantizedLinear::forward
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "fallback to FP32 in debug mode")]
fn test_quantized_linear_debug_assert_on_fallback() {
    // Test code here
}

// AC3: Strict mode returns Err on fallback
#[test]
fn test_strict_mode_rejects_fp32_fallback() {
    std::env::set_var("BITNET_STRICT_MODE", "1");
    // Test code here
}

// AC5: 16-token decode integration test
#[test]
fn test_16_token_decode_strict_mode() {
    std::env::set_var("BITNET_STRICT_MODE", "1");
    // Test code here
}
```

**CPU/GPU smoke testing:**
- Run `cargo test --no-default-features --features cpu -p bitnet-inference test_strict_quantization`
- Run `cargo test --no-default-features --features gpu -p bitnet-inference test_strict_quantization`

**Cross-validation:**
- Validate that strict mode enforcement doesn't break existing cross-validation tests
- Ensure C++ reference implementation uses equivalent quantization path validation

**Benchmark baseline establishment:**
- Establish baseline performance metrics with strict mode enabled
- Validate that performance doesn't degrade compared to current implementation
- Document expected performance in `docs/baselines/strict-mode-performance.md`

### Error Handling
- **Graceful error messages**: Provide clear context when strict mode rejects fallback
- **Recovery guidance**: Error messages should suggest disabling strict mode if intentional fallback is needed
- **Logging**: Log all strict mode validations when `BITNET_STRICT_MODE=1` is set
- **CI integration**: CI should fail fast on strict mode errors (prevent silent degradation)

### Deterministic Inference
- **Seed preservation**: Strict mode should preserve deterministic inference with `BITNET_DETERMINISTIC=1`
- **GPU determinism**: Validate that GPU quantized kernels maintain determinism with proper seeding
- **Reproducibility**: Ensure receipts from strict mode runs are reproducible across runs

### Cross-Platform Support
- **WASM compatibility**: Strict mode should gracefully handle WASM environments (CPU-only quantization)
- **ARM support**: TL1 validation should work correctly on ARM NEON architectures
- **x86 support**: TL2 validation should work correctly on x86 AVX2/AVX-512 architectures

### System Metrics
- **Performance monitoring**: Track strict mode validation overhead in production
- **Receipt statistics**: Aggregate receipt data to detect fallback patterns in deployed systems
- **Alerting**: Production systems should alert when strict mode detects FP32 fallback attempts

## Success Paths

**Flow successful: spec created** → route to spec-analyzer for requirements validation and technical feasibility assessment

**Flow successful: runtime guards implemented** → route to spec-analyzer with implementation evidence for correctness validation

**Flow successful: receipt verification extended** → route to spec-analyzer with schema validation for backward compatibility review

**Flow successful: integration tests passing** → route to spec-analyzer with test evidence for deployment readiness assessment

**Flow successful: cross-validation maintained** → route to spec-analyzer with C++ reference alignment for production qualification

## Related Issues and PRs

- **PR #452**: Receipt Verification Gate (provides foundation for receipt-based validation)
- **Issue #261**: Native I2S/TL1/TL2 quantization implementation (establishes quantized kernel infrastructure)
- **Issue #439**: GPU detection override (`BITNET_GPU_FAKE`) for deterministic testing

## Files to Modify

### Primary Implementation
1. **`crates/bitnet-inference/src/layers/quantized_linear.rs`**
   - Add debug assertions in `fallback_i2s_matmul`, `forward_tl1_generic`, `forward_tl2_generic`
   - Extend `forward` method to check strict mode before allowing fallback
   - Add `validate_quantized_path` helper method

2. **`crates/bitnet-inference/src/layers/attention.rs`**
   - Add debug assertions in `compute_qkv_projections` before each projection call
   - Extend `forward` method to validate strict mode for attention layer
   - Add `validate_projection_kernels` helper method

3. **`crates/bitnet-common/src/strict_mode.rs`**
   - Add `enforce_quantized_inference: bool` field to `StrictModeConfig`
   - Implement `validate_quantization_fallback` method
   - Extend `PerformanceMetrics` with `kernel_path` field

### Testing Infrastructure
4. **`crates/bitnet-inference/tests/strict_quantization_test.rs`** (new file)
   - Unit tests for AC1 (debug assertions in QuantizedLinear)
   - Unit tests for AC2 (debug assertions in attention projections)
   - Unit tests for AC3 (strict mode rejects fallback)
   - Integration test for AC4 (strict mode in attention layer)
   - Integration test for AC5 (16-token decode in strict mode)

5. **`crates/xtask/src/commands/verify_receipt.rs`**
   - Extend receipt schema with `kernel_path` field
   - Add validation logic for quantized computation claims
   - Unit tests for AC6 (receipt validation)

### Documentation
6. **`docs/development/validation-framework.md`**
   - Add section on strict mode quantization guards (AC7)

7. **`docs/reference/quantization-support.md`**
   - Document fallback behavior and strict mode interactions (AC7)

8. **`docs/environment-variables.md`**
   - Document `BITNET_STRICT_MODE=1` behavior for quantization enforcement (AC7)

9. **`docs/howto/troubleshooting-strict-mode.md`** (new file)
   - Troubleshooting guide for strict mode errors (AC7)

### Receipt Schema
10. **`ci/inference.json` schema** (extended schema, backward compatible)
    - Add `kernel_path` field: "native_quantized" | "fp32_fallback"
    - Update schema version to 1.1.0
    - Maintain backward compatibility with existing receipts

## Routing

**FINALIZE → spec-analyzer** for requirements validation, technical feasibility assessment, and integration review.
