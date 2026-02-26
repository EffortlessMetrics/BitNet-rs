# TDD Scaffold Implementation Guide: GGUF Property Tests

**Issue**: #159
**File**: `crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs`
**Priority**: HIGH - Actionable Now
**Estimated Complexity**: Medium

## Overview

This file contains 10 property-based tests for GGUF weight loading that validate quantization accuracy, error bounds, deterministic behavior, and extreme value handling. All tests are currently ignored with TDD placeholder comments.

## Scaffolds to Implement

### 1. I2S Quantization Error Bounds (Lines 224-257)
**Test**: `prop_i2s_quantization_error_bounds`
**Status**: #[ignore] - TDD placeholder
**Goal**: Validate I2S quantization error bounds are consistent

**What needs implementation**:
- Implement `test_i2s_quantization_error_bounds()` helper function
- Perform round-trip I2S quantization (FP32 → I2S → FP32)
- Calculate max_error and mean_error metrics
- Validate max_error ≤ 1.0 and mean_error ≤ 0.1

**Success criteria**:
```rust
// Should pass with real I2S quantization
assert!(max_error <= 1.0);
assert!(mean_error <= 0.1);
```

### 2. I2S Deterministic Quantization (Lines 262-294)
**Test**: `prop_i2s_quantization_deterministic`
**Status**: #[ignore] - TDD placeholder
**Goal**: Validate I2S quantization is deterministic with same seed

**What needs implementation**:
- Implement `test_i2s_quantization_deterministic()` helper
- Set `BITNET_DETERMINISTIC=1` and `BITNET_SEED`
- Run quantization twice with same seed
- Validate outputs are identical

**Success criteria**:
```rust
// Should produce identical results
assert_eq!(output1, output2);
```

### 3. TL1 Numerical Stability (Lines 305-336)
**Test**: `prop_tl1_quantization_numerical_stability`
**Status**: #[ignore] - TDD placeholder
**Goal**: Validate TL1 quantization maintains numerical stability

**What needs implementation**:
- Implement `test_tl1_quantization_stability()` helper
- Perform TL1 (4-bit) quantization with round-trip
- Calculate accuracy metric using MSE
- Calculate stability_metric (should be finite)
- Validate accuracy ≥ 99% threshold

**Success criteria**:
```rust
assert!(accuracy >= 0.99);
assert!(stability_metric.is_finite());
```

### 4. TL1 Sparsity Preservation (Lines 340-369)
**Test**: `prop_tl1_quantization_sparsity_preservation`
**Status**: NO #[ignore] - but depends on helpers
**Goal**: Validate TL1 preserves tensor sparsity patterns

**What needs implementation**:
- Implement `test_tl1_sparsity_preservation()` helper
- Create sparse weights using `create_sparse_weights()`
- Quantize to TL1 and back
- Measure preserved sparsity ratio
- Validate sparsity error ≤ 0.1

**Note**: This test doesn't fail on error, just logs warning

### 5. TL2 Extreme Value Handling (Lines 381-420)
**Test**: `prop_tl2_extreme_value_handling`
**Status**: #[ignore] - TDD placeholder
**Goal**: Validate TL2 handles extreme values correctly

**What needs implementation**:
- Implement `test_tl2_extreme_values()` helper
- Create weights with extreme values (min/max float range)
- Perform TL2 (8-bit) quantization
- Validate no NaN/Inf in output
- Validate accuracy ≥ 90% even for extreme values

**Success criteria**:
```rust
assert!(!output.contains(&f32::NAN));
assert!(!output.contains(&f32::INFINITY));
assert!(accuracy >= 0.90);
```

### 6. TL2 Block Size Scaling (Lines 421-465)
**Test**: `prop_tl2_block_size_scaling`
**Status**: #[ignore] - TDD placeholder
**Goal**: Validate TL2 accuracy scales with block size

**What needs implementation**:
- Implement `test_tl2_block_size_scaling()` helper
- Test multiple block sizes (64, 128, 256, 512)
- Validate accuracy improves or stays stable with larger blocks
- Check that accuracy doesn't degrade

**Success criteria**:
```rust
// Accuracy should be monotonic or stable
for i in 1..accuracies.len() {
    assert!(accuracies[i] >= accuracies[i-1] - 0.01); // Allow 1% tolerance
}
```

### 7. Zero-Copy Efficiency (Lines 550-588)
**Test**: `prop_zero_copy_efficiency`
**Status**: #[ignore] - TDD placeholder
**Goal**: Validate zero-copy loading doesn't allocate extra memory

**What needs implementation**:
- Implement `test_zero_copy_loading()` helper
- Use memory-mapped GGUF file
- Track memory usage during loading
- Validate memory overhead ≤ 10% of file size

**Success criteria**:
```rust
let memory_overhead = (memory_after - memory_before) / file_size;
assert!(memory_overhead <= 0.10);
```

### 8. Extreme Range Handling (Lines 713-751)
**Test**: `prop_extreme_range_handling`
**Status**: #[ignore] - TDD placeholder
**Goal**: Validate loader handles extreme tensor ranges

**What needs implementation**:
- Implement `test_extreme_range_loading()` helper
- Create GGUF with extreme value ranges
- Load and validate all tensors
- Check for numerical stability (no NaN/Inf)

**Success criteria**:
```rust
assert!(all_values_finite);
assert!(no_overflow_detected);
```

### 9. Sparsity Preservation Validation (Lines 752-801)
**Test**: `prop_sparsity_preservation_validation`
**Status**: #[ignore] - TDD placeholder
**Goal**: Validate quantization preserves sparsity patterns

**What needs implementation**:
- Implement `test_sparsity_preservation()` helper
- Create sparse GGUF weights (50-90% zeros)
- Quantize and load
- Measure sparsity preservation
- Validate within 10% of original

**Success criteria**:
```rust
let sparsity_error = (original_sparsity - loaded_sparsity).abs();
assert!(sparsity_error <= 0.10);
```

### 10. Architecture Validation (Lines 802-831)
**Test**: `prop_architecture_validation`
**Status**: #[ignore] - TDD placeholder
**Goal**: Validate loaded tensors match architecture requirements

**What needs implementation**:
- Implement `test_architecture_validation()` helper
- Load GGUF file
- Validate tensor shapes match transformer architecture
- Check layer counts, hidden sizes, attention heads
- Validate weight matrix dimensions

**Success criteria**:
```rust
assert_eq!(actual_num_layers, expected_num_layers);
assert_eq!(actual_hidden_size, expected_hidden_size);
assert_eq!(actual_num_heads, expected_num_heads);
```

## Implementation Strategy

### Phase 1: Quantization Helpers (Tests 1-6)
1. Implement round-trip quantization for I2S, TL1, TL2
2. Add MSE-based accuracy metrics
3. Add deterministic seeding support
4. Test with property-based random inputs

### Phase 2: Memory and Loading (Tests 7, 9)
1. Implement memory tracking using `sysinfo` crate
2. Add zero-copy validation
3. Implement sparsity measurement helpers

### Phase 3: Extreme Values and Architecture (Tests 8, 10)
1. Add NaN/Inf detection
2. Implement architecture metadata parsing
3. Add tensor shape validation

## Testing Commands

```bash
# Run individual property test (after removing #[ignore])
cargo test -p bitnet-models --no-default-features --features cpu \
  prop_i2s_quantization_error_bounds -- --exact

# Run all property tests
cargo test -p bitnet-models --no-default-features --features cpu \
  --test gguf_weight_loading_property_tests

# Run with proptest verbose output
RUST_LOG=proptest=debug cargo test -p bitnet-models --features cpu \
  --test gguf_weight_loading_property_tests -- --nocapture
```

## Dependencies

**Existing infrastructure**:
- `PropertyTestConfig` (already defined)
- `arbitrary_weight_tensor()` strategy (already defined)
- `arbitrary_tensor_shape()` strategy (already defined)
- `create_sparse_weights()` helper (needs implementation)

**Crates needed**:
- `proptest` (already in Cargo.toml)
- `sysinfo` (for memory tracking)
- BitNet-rs quantization APIs from `bitnet-quantization` crate

## Success Metrics

- All 10 property tests passing without #[ignore]
- Property tests run 100+ random cases each
- No panics or crashes on random inputs
- Accuracy metrics meet thresholds (≥99% for I2S/TL1, ≥90% for TL2)
- Memory overhead ≤10% for zero-copy loading
- Deterministic behavior with BITNET_SEED

## References

- Issue #159: GGUF Weight Loading
- `bitnet-quantization` crate for quantization APIs
- `proptest` documentation for property-based testing patterns
