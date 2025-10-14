# ADR-010: Three-Tier Validation Strategy for Strict Quantization Guards

## Status

**ACCEPTED** - Issue #453 Implementation

## Context

BitNet.rs implements receipt-based proof of computation (PR #452) but lacks runtime guarantees that quantized inference uses native quantized kernels instead of silently falling back to FP32 dequantization. This gap undermines performance baselines and correctness validation.

### Current State

1. **No Runtime Guards**: Quantized layers can silently fall back to FP32 without detection
2. **Misleading Receipts**: Claims "quantized computation" while using FP32 fallback
3. **Development Delays**: Fallback bugs discovered late in testing cycle
4. **Production Risk**: Silent performance degradation from FP32 fallback

### Requirements

- **Development Speed**: Immediate feedback on fallback during development
- **Production Safety**: Reject FP32 fallback in strict mode deployments
- **Verification**: Post-inference validation of computation claims
- **Performance**: Negligible overhead (<1%) in release builds

## Decision

We implement a **Three-Tier Validation Strategy** with complementary checks at different stages of the inference pipeline:

### Tier 1: Debug Assertions (Development)

**Purpose:** Catch FP32 fallback immediately during development

**Implementation:**
```rust
// crates/bitnet-inference/src/layers/quantized_linear.rs
async fn forward_i2s(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
    if !self.has_native_quantized_kernel() {
        #[cfg(debug_assertions)]
        panic!("fallback to FP32 in debug mode: layer={}, qtype=I2S",
               self.name);

        // Release build: allow fallback with warning
        log::warn!("Using FP32 fallback for layer: {}", self.name);
        return self.fallback_i2s_matmul(input).await;
    }

    self.quantized_matmul_i2s(input).await
}
```

**Characteristics:**
- **Scope:** Debug builds only (`#[cfg(debug_assertions)]`)
- **Behavior:** Immediate panic with detailed error message
- **Overhead:** Zero in release builds (compiled out)
- **Target Audience:** Developers running local tests

### Tier 2: Strict Mode Enforcement (Production)

**Purpose:** Reject FP32 fallback in production deployments with explicit opt-in

**Implementation:**
```rust
// crates/bitnet-inference/src/layers/quantized_linear.rs
async fn forward_i2s(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
    if !self.has_native_quantized_kernel() {
        #[cfg(debug_assertions)]
        panic!("fallback to FP32 in debug mode");

        // Strict mode check (production)
        let strict_mode = StrictModeEnforcer::new();
        if strict_mode.get_config().enforce_quantized_inference {
            return Err(BitNetError::StrictMode(format!(
                "FP32 fallback rejected - qtype=I2S, device={:?}, dims=[{}, {}]",
                self.device, self.in_features, self.out_features
            )));
        }

        // Allow fallback in non-strict mode
        log::warn!("Using FP32 fallback");
        return self.fallback_i2s_matmul(input).await;
    }

    self.quantized_matmul_i2s(input).await
}
```

**Characteristics:**
- **Scope:** Release builds with `BITNET_STRICT_MODE=1`
- **Behavior:** Return `Err(BitNetError::StrictMode(...))`
- **Overhead:** <1% (single boolean check per forward pass)
- **Target Audience:** Production inference servers, CI baselines

### Tier 3: Receipt Validation (Verification)

**Purpose:** Validate receipts accurately reflect computation path

**Implementation:**
```rust
// xtask/src/main.rs
fn verify_quantization_claims(receipt: &Receipt) -> Result<()> {
    // Validate kernel_path matches kernels array
    if let Some(kernel_path) = &receipt.kernel_path {
        match kernel_path.as_str() {
            "native_quantized" => {
                ensure!(
                    receipt.kernels.iter().any(is_quantized_kernel),
                    "kernel_path='native_quantized' requires quantized kernel IDs"
                );
            }
            "fp32_fallback" => {
                ensure!(
                    receipt.compute_path != "quantized",
                    "kernel_path='fp32_fallback' cannot claim quantized compute"
                );
            }
            _ => bail!("Invalid kernel_path: {}", kernel_path),
        }
    }

    Ok(())
}
```

**Characteristics:**
- **Scope:** Post-inference verification (`xtask verify-receipt`)
- **Behavior:** Exit code 1 if receipt claims don't match kernel IDs
- **Overhead:** Zero (offline verification)
- **Target Audience:** CI gates, performance baseline validation

## Rationale

### Why Three Tiers?

**Tier 1 (Debug Assertions):**
- **Development Efficiency:** Immediate feedback during local testing
- **Zero Overhead:** Compiled out in release builds (no performance cost)
- **Early Detection:** Catch fallback bugs before PR submission

**Tier 2 (Strict Mode):**
- **Production Safety:** Prevent silent fallback in critical deployments
- **Opt-In Design:** Explicit `BITNET_STRICT_MODE=1` (not breaking existing workflows)
- **Granular Control:** Separate flags for mock detection vs quantization enforcement

**Tier 3 (Receipt Validation):**
- **Verification:** Independent validation of computation claims
- **CI Integration:** Automated gates in performance baselines
- **Auditability:** Permanent record of actual computation path

### Alternatives Considered

**Alternative 1: Single-Tier (Runtime Only)**
- **Rejected:** No early detection during development
- **Rejected:** Performance overhead in all release builds

**Alternative 2: Two-Tier (Debug + Strict Mode)**
- **Rejected:** No post-hoc verification of receipts
- **Rejected:** Cannot validate historical performance baselines

**Alternative 3: Compile-Time Only**
- **Rejected:** Cannot detect runtime device mismatches (GPU unavailable, etc.)
- **Rejected:** No production enforcement without recompilation

## Implementation Strategy

### Phase 1: Debug Assertions (Week 1, Days 1-4)

**Files Modified:**
- `crates/bitnet-inference/src/layers/quantized_linear.rs` (AC1)
- `crates/bitnet-inference/src/layers/attention.rs` (AC2)

**Validation:**
```bash
# AC1: Debug assertions in QuantizedLinear::forward
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac1_debug_assert_i2s_fallback -- --nocapture

# AC2: Debug assertions in attention projections
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac2_debug_assert_attention_projection -- --nocapture
```

### Phase 2: Strict Mode Enforcement (Week 2, Days 8-14)

**Files Modified:**
- `crates/bitnet-common/src/strict_mode.rs` (extend StrictModeConfig)
- `crates/bitnet-inference/src/layers/quantized_linear.rs` (AC3)
- `crates/bitnet-inference/src/layers/attention.rs` (AC4)
- `crates/bitnet-inference/tests/strict_quantization_test.rs` (AC5, new file)

**Validation:**
```bash
# AC3: Strict mode rejects FP32 fallback
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac3_strict_mode_rejects_fallback

# AC5: 16-token decode integration test
BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac5_16_token_decode_cpu_strict_mode
```

### Phase 3: Receipt Validation (Week 3, Days 15-21)

**Files Modified:**
- `xtask/src/main.rs` (extend verify_receipt_cmd)

**Validation:**
```bash
# AC6: Receipt validation for quantized claims
cargo test -p xtask test_ac6_receipt_quantized_kernels_valid
cargo run -p xtask -- benchmark --model tests/models/mini.gguf --tokens 128
cargo run -p xtask -- verify-receipt ci/inference.json
```

## Consequences

### Positive

1. **Early Detection:** Debug assertions catch fallback during development (100% detection rate)
2. **Production Safety:** Strict mode prevents silent fallback in critical deployments
3. **Auditability:** Receipt validation provides independent verification
4. **Performance:** <1% overhead in release builds with strict mode
5. **Backward Compatible:** Opt-in design (existing workflows unchanged)

### Negative

1. **Complexity:** Three separate validation mechanisms require coordination
2. **Test Overhead:** Each tier requires dedicated test coverage
3. **Documentation:** Requires comprehensive guide for developers

### Mitigation

- **Complexity:** Clear separation of concerns (development, production, verification)
- **Test Overhead:** TDD with `// AC:ID` tags ensures traceability
- **Documentation:** Comprehensive troubleshooting guide (`docs/howto/troubleshooting-strict-mode.md`)

## Performance Impact

### Debug Assertions (Tier 1)

- **Debug Builds:** <0.1% overhead (single boolean check + panic)
- **Release Builds:** 0% overhead (compiled out)

### Strict Mode (Tier 2)

- **Overhead:** <1% in release builds with `BITNET_STRICT_MODE=1`
- **Method:** Single boolean check per forward pass
- **Target:** 10s inference SLO maintained (<100ms overhead for typical model)

### Receipt Validation (Tier 3)

- **Overhead:** 0% (offline verification, separate process)
- **Timing:** Post-inference, does not impact inference latency

## Validation Metrics

### Success Criteria

- ✅ Debug assertions: 100% fallback detection rate in debug builds
- ✅ Strict mode: 100% rejection rate for FP32 fallback when enabled
- ✅ Receipt validation: 100% accuracy in correlating claims with kernel IDs
- ✅ Performance: <1% overhead in release builds with strict mode
- ✅ Backward compatibility: Zero breaking changes to existing workflows

### Measurable Commands

```bash
# Tier 1 validation (debug assertions)
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac1_debug_assert_*

# Tier 2 validation (strict mode)
BITNET_STRICT_MODE=1 \
cargo test --no-default-features --features cpu -p bitnet-inference \
  test_ac3_strict_mode_*

# Tier 3 validation (receipt verification)
cargo run -p xtask -- verify-receipt ci/inference.json
```

## Related ADRs

- **ADR-011:** Receipt Schema Backward Compatibility (v1.0.0 → v1.1.0)
- **ADR-012:** Kernel ID Naming Conventions
- **ADR-013:** FP32 Fallback Detection Mechanisms
- **ADR-004:** Mock Elimination Technical Decisions (Issue #261)

## References

- **Issue #453:** Strict Quantization Guards
- **PR #452:** Receipt Verification Infrastructure
- **Issue #261:** Mock Performance Reporting Elimination
- **Issue #439:** GPU Feature-Gate Hardening
