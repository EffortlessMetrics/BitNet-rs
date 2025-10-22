# TDD Scaffold Implementation Guide: Neural Network Tests

**Issue**: #248
**File**: `crates/bitnet-inference/tests/neural_network_test_scaffolding.rs`
**Priority**: HIGH - Actionable Now
**Estimated Complexity**: High (requires full inference pipeline)

## Overview

This file contains 5 high-level neural network tests that validate the complete BitNet inference pipeline from quantized linear layers through autoregressive generation. Some tests have been partially implemented, others are still scaffolds.

## Scaffolds to Implement

### 1. AC1: Quantized Linear Layer Forward Pass (Lines 39-60)
**Test**: `test_ac1_quantized_linear_layer_forward_pass`
**Status**: #[ignore] - TDD placeholder with panic
**Goal**: Validate I2S, TL1, TL2 quantization maintains >99% accuracy

**Current state**:
- Basic test structure exists
- `test_i2s_quantization()` helper is called
- Panics with "not yet implemented" message

**What needs implementation**:
1. Complete `test_i2s_quantization()` helper to use real quantized GEMV kernels
2. Add `test_tl1_quantization()` helper for TL1 (4-bit) validation
3. Add `test_tl2_quantization()` helper for TL2 (8-bit) validation
4. Implement accuracy calculation using MSE or cosine similarity
5. Remove panic and validate all three quantization typesAnd 

**Success criteria**:
```rust
// All three quantization types should pass
assert!(i2s_result.accuracy > 0.99);
assert!(tl1_result.accuracy > 0.99);
assert!(tl2_result.accuracy > 0.99);
```

**Implementation notes**:
- Use `bitnet-kernels` quantized_linear APIs
- Create reference FP32 computation for comparison
- Calculate accuracy: `1.0 - (MSE / signal_power)`

### 2. AC2: Multi-Head Attention Mechanism (Lines 67-93)
**Test**: `test_ac2_multi_head_attention_mechanism`
**Status**: PARTIALLY IMPLEMENTED - No #[ignore]
**Goal**: Validate attention with quantized Q, K, V projections

**Current state**:
- Test already passing (removed from previous sprint)
- Uses real `BitNetAttention` with quantized projections
- Validates output shapes

**What's still needed** (if any):
- Verify RoPE integration works correctly
- Add attention mask validation
- Add KV cache testing (if not covered elsewhere)

**This test may already be complete** - verify by running:
```bash
cargo test -p bitnet-inference --no-default-features --features cpu \
  test_ac2_multi_head_attention_mechanism -- --exact
```

### 3. AC3: Autoregressive Token Generation (Lines 99-126)
**Test**: `test_ac3_autoregressive_token_generation`
**Status**: PARTIALLY IMPLEMENTED - No #[ignore]
**Goal**: Validate temperature, top-k, nucleus sampling with deterministic seeding

**Current state**:
- Test already passing (removed from previous sprint)
- Uses `AutoregressiveGenerator` with real sampling
- Validates token generation

**What's still needed** (if any):
- Verify all sampling methods work (greedy, temperature, top-k, nucleus)
- Add deterministic seeding tests
- Validate repetition penalty

**This test may already be complete** - verify by running:
```bash
cargo test -p bitnet-inference --no-default-features --features cpu \
  test_ac3_autoregressive_token_generation -- --exact
```

### 4. AC4: Cross-Validation Accuracy Preservation (Lines 132-161)
**Test**: `test_ac4_cross_validation_accuracy_preservation`
**Status**: NOT IGNORED but panics - "not yet implemented"
**Goal**: Validate >99% accuracy vs C++ reference using xtask crossval

**Current state**:
- Test structure exists
- Checks for `BITNET_CROSSVAL_ENABLED` environment variable
- Panics with "not yet implemented" message

**What needs implementation**:
1. Implement `test_cross_validation_accuracy()` helper
2. Integrate with `cargo run -p xtask -- crossval` infrastructure
3. Run inference on same prompt in Rust and C++
4. Compare outputs using cosine similarity and token accuracy
5. Validate accuracy ≥ 99% and correlation ≥ 99.9%

**Success criteria**:
```rust
assert!(crossval_result.accuracy >= 0.99);
assert!(crossval_result.correlation >= 0.999);
```

**Implementation notes**:
- Requires `BITNET_CPP_DIR` environment variable pointing to C++ reference
- Use `feature = "crossval"` gate
- May need to call C++ inference via FFI or subprocess
- See `crossval` crate for existing infrastructure

**Blockers**:
- Requires C++ reference setup (`BITNET_CPP_DIR`)
- May be blocked by Issue #469 (tokenizer parity)

### 5. AC5: Performance Target Validation (Lines 167-204)
**Test**: `test_ac5_performance_targets_validation`
**Status**: #[ignore] - TDD placeholder
**Goal**: Validate 5-15 tok/sec CPU, 2-5x GPU speedup

**Current state**:
- Test structure exists
- Calls `test_performance_targets()` helper
- Panics with "not yet implemented" message

**What needs implementation**:
1. Implement `test_performance_targets()` helper
2. Run CPU inference and measure tokens/sec
3. Run GPU inference (if available) and measure speedup
4. Measure memory usage during inference
5. Validate against target thresholds

**Success criteria**:
```rust
// CPU performance
assert!(perf_result.cpu_tokens_per_sec >= 5.0);
assert!(perf_result.cpu_tokens_per_sec <= 50.0); // Upper bound for sanity

// GPU performance (if available)
if gpu_available {
    assert!(perf_result.gpu_speedup >= 2.0);
    assert!(perf_result.gpu_speedup <= 10.0); // Upper bound for sanity
}

// Memory usage
assert!(perf_result.memory_usage_mb <= 4096); // 4GB max for 2B model
```

**Implementation notes**:
- Use `std::time::Instant` for timing
- Use `sysinfo` crate for memory tracking
- Run with small model (2B parameters) for CI
- Allow flexible thresholds (5-15 tok/sec is wide range)
- GPU speedup is optional (skip if no GPU)

### Additional Tests Mentioned in File

#### AC6: Real vs Mock Inference Comparison (Lines 205-234)
**Status**: NOT IGNORED - already implemented
**Goal**: Validate real inference produces different outputs than mocks

This test appears complete and passing.

#### AC7: Deterministic Inference Validation (Lines 236-269)
**Status**: NOT IGNORED - already implemented
**Goal**: Validate deterministic inference with BITNET_SEED

This test appears complete and passing.

#### AC8: Mock Implementation Replacement Validation (Lines 270-296)
**Status**: #[ignore] - TDD placeholder
**Goal**: Validate all mock implementations are replaced with real ones

**Current state**:
- Test structure exists
- Calls `validate_no_mock_implementations()` helper
- Panics with "not yet implemented"

**What needs implementation**:
1. Implement `validate_no_mock_implementations()` helper
2. Search codebase for mock patterns:
   - Functions named `*_mock` or `*_stub`
   - Comments containing "mock" or "stub"
   - `unimplemented!()` calls in production code
3. Validate all critical paths use real implementations
4. Report any remaining mocks

**Success criteria**:
```rust
let validation_result = validate_no_mock_implementations()?;
assert!(validation_result.all_replaced,
    "Found {} mock implementations still in use",
    validation_result.mocks_found);
```

**Implementation notes**:
- Use regex to search source files
- Focus on production code (not test files)
- Allow mocks in test-only modules

#### AC9: Comprehensive Integration Test (Lines 297-327)
**Status**: #[ignore] - TDD placeholder
**Goal**: End-to-end transformer pipeline validation

**Current state**:
- Test structure exists
- Calls `test_comprehensive_integration()` helper
- Panics with "not yet implemented"

**What needs implementation**:
1. Implement `test_comprehensive_integration()` helper
2. Test complete transformer pipeline:
   - Token embedding
   - All transformer layers (attention + FFN)
   - Layer normalization
   - Output projection
   - Token generation
3. Validate intermediate activations at each layer
4. Check for numerical stability (no NaN/Inf)
5. Validate output token probabilities sum to ~1.0

**Success criteria**:
```rust
// Complete pipeline validation
assert!(integration_result.all_layers_passed);
assert!(!integration_result.has_nan_or_inf);
assert!((integration_result.prob_sum - 1.0).abs() < 0.01);
```

**Implementation notes**:
- Use small model for speed (100M-500M params)
- Test both forward pass and generation
- Validate shapes at each layer
- Check activation statistics (mean, std, min, max)

#### AC10: Error Handling Robustness (Lines 328-365)
**Status**: #[ignore] - TDD placeholder
**Goal**: Validate graceful error handling for edge cases

**Current state**:
- Test structure exists
- Calls `test_error_handling_robustness()` helper
- Panics with "not yet implemented"

**What needs implementation**:
1. Implement `test_error_handling_robustness()` helper
2. Test error scenarios:
   - Empty input
   - Invalid token IDs
   - Out-of-vocabulary tokens
   - Mismatched tensor shapes
   - OOM conditions (large batch)
   - NaN/Inf in input
   - Device unavailable (GPU fallback)
3. Validate proper error messages
4. Ensure no panics or crashes

**Success criteria**:
```rust
// All error cases should return Result::Err (not panic)
for error_case in error_cases {
    let result = test_error_case(error_case);
    assert!(result.is_err(), "Should fail gracefully for: {:?}", error_case);
}
```

**Implementation notes**:
- Use `#[should_panic]` for cases that SHOULD panic
- Most cases should return `Result<_, Error>` not panic
- Test error message clarity
- Validate cleanup (no resource leaks)

## Implementation Strategy

### Phase 1: Complete AC1 (Quantized Linear Layers)
- Focus on getting all three quantization types working
- This unblocks many other tests
- Priority: **CRITICAL**

### Phase 2: Performance and Integration (AC5, AC9)
- Implement performance measurement infrastructure
- Build end-to-end integration test
- Priority: **HIGH**

### Phase 3: Error Handling and Validation (AC8, AC10)
- Add error handling tests
- Validate mock replacement
- Priority: **MEDIUM**

### Phase 4: Cross-Validation (AC4)
- Requires C++ reference setup
- May be blocked by Issue #469
- Priority: **MEDIUM** (can defer if blocked)

## Testing Commands

```bash
# Run all neural network tests
cargo test -p bitnet-inference --no-default-features --features cpu \
  --test neural_network_test_scaffolding

# Run specific AC test
cargo test -p bitnet-inference --no-default-features --features cpu \
  test_ac1_quantized_linear_layer_forward_pass -- --exact

# Run with logging
RUST_LOG=debug cargo test -p bitnet-inference --features cpu \
  --test neural_network_test_scaffolding -- --nocapture

# Run with cross-validation (requires setup)
BITNET_CROSSVAL_ENABLED=1 BITNET_CPP_DIR=/path/to/bitnet.cpp \
  cargo test -p bitnet-inference --features cpu,crossval \
  test_ac4_cross_validation_accuracy_preservation -- --exact
```

## Dependencies

**Existing infrastructure**:
- `NeuralNetworkTestConfig` (already defined)
- `create_mock_tensor_data()` helper (already defined)
- `bitnet-inference` crate with InferenceEngine
- `bitnet-kernels` crate with quantized ops
- `crossval` crate for C++ reference comparison

**Crates needed**:
- `tokio` (already in use for async tests)
- `anyhow` (already in use for error handling)
- `sysinfo` (for memory tracking in AC5)
- `regex` (for mock detection in AC8)

## Success Metrics

- All 10 AC tests passing without #[ignore]
- No panics in production code paths
- Performance targets met (5-15 tok/sec CPU)
- Cross-validation accuracy ≥99% (if C++ available)
- Comprehensive error handling (no crashes on invalid input)

## References

- Issue #248: Neural Network Inference Implementation
- Issue #469: Tokenizer Parity (may block AC4)
- Issue #254: Shape Mismatch (should be resolved)
- `bitnet-inference` crate documentation
- `docs/architecture-overview.md`
