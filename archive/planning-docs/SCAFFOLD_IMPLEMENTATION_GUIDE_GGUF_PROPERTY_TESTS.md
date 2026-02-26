# GGUF Property Tests - TDD Scaffold Implementation Guide

**File**: `crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs`
**Issue**: #159
**Total Scaffolds**: 13 ignored tests
**Priority**: HIGH - All actionable now

---

## Overview

This file contains 13 property-based tests (using proptest) for GGUF weight loading that validate quantization accuracy, memory efficiency, and numerical stability. All tests follow TDD patterns and are currently in "Red" phase with stub helper functions.

**Pattern**: Each test has:
1. A `#[test] #[ignore]` annotation with TDD placeholder comment
2. A property test using arbitrary strategies
3. A helper function that needs real implementation
4. Assertions with specific thresholds (99% accuracy, memory ratios, etc.)

---

## Scaffold 1: I2S Quantization Round-Trip (Line 179)

**Test**: `prop_i2s_quantization_preserves_distribution`
**Helper**: `test_i2s_quantization_roundtrip` (line 868)
**Status**: Stub - returns error

**What to Implement**:
```rust
fn test_i2s_quantization_roundtrip(
    weight_data: &[f32],
    shape: &[usize],
    params: &QuantizationTestParams,
) -> Result<f32> {
    // 1. Import I2SQuantizer from bitnet_quantization
    // 2. Create BitNetTensor from weight_data using bitnet_quantization::utils::create_tensor_from_f32
    // 3. Quantize using I2SQuantizer::new().quantize_tensor()
    // 4. Dequantize back to f32
    // 5. Calculate MSE and return accuracy = 1.0 - (MSE / signal_power)
    // 6. Ensure accuracy >= 0.99 (99% threshold)
}
```

**Dependencies**:
- `use bitnet_quantization::{I2SQuantizer, Quantize};`
- `use bitnet_quantization::utils::create_tensor_from_f32;`
- `use candle_core::Device;`

**Validation**: Accuracy >= 0.99

---

## Scaffold 2: I2S Error Bounds (Line 215)

**Test**: `prop_i2s_quantization_error_bounds`
**Helper**: `test_i2s_quantization_error_bounds` (line 879)
**Status**: Stub - returns error

**What to Implement**:
```rust
fn test_i2s_quantization_error_bounds(weight_data: &[f32], shape: &[usize]) -> Result<(f32, f32)> {
    // 1. Quantize and dequantize using I2S
    // 2. Calculate element-wise absolute differences
    // 3. Return (max_error, mean_error)
    // 4. max_error should be <= 1.0
    // 5. mean_error should be <= 0.1
}
```

**Validation**: max_error <= 1.0, mean_error <= 0.1

---

## Scaffold 3: I2S Deterministic Quantization (Line 253)

**Test**: `prop_i2s_quantization_deterministic`
**Helper**: `test_i2s_quantization_deterministic` (line 886)
**Status**: Stub - returns error

**What to Implement**:
```rust
fn test_i2s_quantization_deterministic(
    weight_data: &[f32],
    shape: &[usize],
    seed: u64,
) -> Result<Vec<f32>> {
    // 1. Set environment: BITNET_DETERMINISTIC=1, BITNET_SEED=seed
    // 2. Quantize twice with same seed
    // 3. Compare outputs element-wise
    // 4. Return dequantized result
    // 5. Ensure outputs are identical
}
```

**Validation**: Two runs with same seed produce identical results

---

## Scaffold 4: TL1 Numerical Stability (Line 296)

**Test**: `prop_tl1_quantization_numerical_stability`
**Helper**: `test_tl1_quantization_stability` (line 897)
**Status**: Stub - returns error

**What to Implement**:
```rust
fn test_tl1_quantization_stability(weight_data: &[f32], shape: &[usize]) -> Result<(f32, f32)> {
    // 1. Import TL1Quantizer from bitnet_quantization
    // 2. Quantize using TL1 (4-bit table lookup)
    // 3. Calculate accuracy (MSE-based)
    // 4. Calculate stability metric (variance of errors)
    // 5. Return (accuracy, stability_metric)
    // 6. Ensure accuracy >= 0.99, stability.is_finite()
}
```

**Dependencies**:
- `use bitnet_quantization::{TL1Quantizer, Quantize};`

**Validation**: accuracy >= 0.99, stability_metric.is_finite()

---

## Scaffold 5: TL1 Sparsity Preservation (Line 332)

**Test**: `prop_tl1_quantization_sparsity_preservation`
**Helper**: `test_tl1_sparsity_preservation` (line 904)
**Status**: Stub - returns error

**What to Implement**:
```rust
fn test_tl1_sparsity_preservation(
    weight_data: &[f32],
    shape: &[usize],
    target_sparsity: f32,
) -> Result<f32> {
    // 1. Create sparse weights using create_sparse_weights()
    // 2. Quantize with TL1
    // 3. Dequantize back
    // 4. Count zeros in output
    // 5. Calculate preserved_sparsity = zero_count / total_count
    // 6. Ensure |preserved_sparsity - target_sparsity| <= 0.1
}
```

**Validation**: sparsity_error <= 0.1

---

## Scaffold 6: Memory Usage Linear Scaling (Line 515)

**Test**: `prop_memory_usage_linear_scaling`
**Helper**: `test_memory_usage_scaling` (line 1036)
**Status**: Stub - returns error

**What to Implement**:
```rust
fn test_memory_usage_scaling(
    data1: &[f32],
    data2: &[f32],
    scale_factor: usize,
) -> Result<(usize, usize, f32)> {
    // 1. Use sysinfo crate to get process memory before/after
    // 2. Load data1, measure memory (memory1)
    // 3. Load data2 (scale_factor × larger), measure memory (memory2)
    // 4. Calculate actual_scale = memory2 / memory1
    // 5. Validate scale_error <= 0.2 (20% tolerance)
}
```

**Dependencies**:
- `use sysinfo::{System, SystemExt, ProcessExt};`

**Validation**: scale_error <= 0.2

---

## Scaffold 7: Zero-Copy Memory Efficiency (Line 551)

**Test**: `prop_zero_copy_memory_efficiency`
**Helper**: `test_zero_copy_efficiency` (line 1047)
**Status**: Stub - returns error

**What to Implement**:
```rust
fn test_zero_copy_efficiency(
    weight_data: &[f32],
    alignment: usize,
) -> Result<(usize, usize, bool)> {
    // 1. Measure memory with copy-based loading
    // 2. Measure memory with zero-copy (mmap) loading
    // 3. Calculate memory ratio = zero_copy_memory / copy_memory
    // 4. Ensure memory_ratio <= 0.8 (20% savings)
    // 5. Return (copy_memory, zero_copy_memory, copy_saved)
}
```

**Validation**: memory_ratio <= 0.8

---

## Scaffold 8: NaN/Inf Edge Case Handling (Line 592)

**Test**: `prop_quantization_handles_nan_inf`
**Helper**: `test_edge_case_handling` (line 1112)
**Status**: Stub - returns error

**What to Implement**:
```rust
fn test_edge_case_handling(weight_data: &[f32], shape: &[usize]) -> Result<(bool, bool, Vec<f32>)> {
    // 1. Inject NaN and Inf values into weight_data
    // 2. Quantize using I2S (should handle gracefully)
    // 3. Dequantize back
    // 4. Validate all outputs are finite
    // 5. Return (nan_handled, inf_handled, finite_output)
}
```

**Validation**: All output values are finite

---

## Scaffold 9: Distribution Preservation (Line 629)

**Test**: `prop_quantization_preserves_distribution`
**Helper**: `test_distribution_preservation` (line 1119)
**Status**: Stub - returns error

**What to Implement**:
```rust
fn test_distribution_preservation(
    weight_data: &[f32],
    shape: &[usize],
) -> Result<(bool, bool, f32)> {
    // 1. Calculate original mean and variance
    // 2. Quantize and dequantize
    // 3. Calculate dequantized mean and variance
    // 4. Check if mean preserved (within 10% tolerance)
    // 5. Check if variance preserved (within 20% tolerance)
    // 6. Calculate correlation coefficient
    // 7. Return (mean_preserved, variance_preserved, correlation)
}
```

**Validation**: mean within 10%, variance within 20%, correlation >= 0.95

---

## Scaffold 10: Extreme Dynamic Range (Line 710)

**Test**: `prop_extreme_dynamic_range`
**Helper**: `test_extreme_dynamic_range_handling` (line 1142)
**Status**: Stub - returns error

**What to Implement**:
```rust
fn test_extreme_dynamic_range_handling(
    weight_data: &[f32],
    shape: &[usize],
) -> Result<(f32, f32, bool)> {
    // 1. Find min/max in weight_data (extreme values)
    // 2. Quantize with clipping/saturation
    // 3. Dequantize back
    // 4. Calculate dynamic_range = max - min
    // 5. Calculate accuracy (allow lower threshold: 0.85)
    // 6. Ensure clipping_handled = all outputs are finite
    // 7. Return (dynamic_range, accuracy, clipping_handled)
}
```

**Validation**: accuracy >= 0.85, dynamic_range.is_finite()

---

## Scaffold 11: Sparse Tensor Handling (Line 749)

**Test**: `prop_sparse_tensor_handling`
**Helper**: `test_sparse_tensor_preservation` (line 1153)
**Status**: **PARTIALLY IMPLEMENTED** - needs completion

**What to Implement**:
```rust
fn test_sparse_tensor_preservation(
    weight_data: &[f32],
    shape: &[usize],
    target_sparsity: f32,
) -> Result<(f32, f32)> {
    // Already partially implemented with I2S quantization
    // Need to:
    // 1. Complete sparsity calculation
    // 2. Calculate compression ratio
    // 3. Ensure sparsity_error <= 0.15
    // 4. Ensure compression_ratio >= 1.0
}
```

**Validation**: sparsity_error <= 0.15, compression_ratio >= 1.0

---

## Scaffold 12: Model Architecture Support (Line 798)

**Test**: `prop_model_architecture_support`
**Helper**: `test_architecture_compatibility` (line 1194)
**Status**: Stub - returns error

**What to Implement**:
```rust
fn test_architecture_compatibility(arch: &ModelArchitecture) -> Result<(bool, f32)> {
    // 1. Parse architecture configuration
    // 2. Validate tensor shapes match architecture
    // 3. Run quantization on architecture-specific shapes
    // 4. Calculate accuracy
    // 5. Return (supported, accuracy)
    // 6. Ensure accuracy >= 0.99
}
```

**Validation**: supported = true, accuracy >= 0.99

---

## Scaffold 13: Custom Quantization Parameters (Line 828)

**Test**: `prop_custom_quantization_params`
**Helper**: `test_custom_quantization_params` (line 1202)
**Status**: Stub - returns error

**What to Implement**:
```rust
fn test_custom_quantization_params(
    weight_data: &[f32],
    shape: &[usize],
    scales: &[f32],
    zero_points: &[i32],
) -> Result<f32> {
    // 1. Create custom quantizer with provided scales/zero_points
    // 2. Quantize with custom parameters
    // 3. Dequantize back
    // 4. Calculate accuracy
    // 5. Return accuracy
    // 6. Ensure accuracy >= 0.99
}
```

**Validation**: accuracy >= 0.99

---

## Implementation Strategy

**For Each Scaffold**:
1. Read the helper function stub (lines 868-1220)
2. Implement real quantization logic using bitnet_quantization APIs
3. Remove #[ignore] attribute from corresponding test
4. Run test to validate (should pass or correctly identify limitations)
5. Commit with clear message

**Common Patterns**:
- Use `bitnet_quantization::utils::create_tensor_from_f32()`
- Use `I2SQuantizer`, `TL1Quantizer`, `TL2Quantizer` from bitnet_quantization
- Calculate MSE-based accuracy: `1.0 - (MSE / signal_power)`
- Use `sysinfo` crate for memory tracking
- Handle edge cases gracefully (NaN, Inf, extreme ranges)

**BitNet-rs Standards**:
- Feature-gated: `#[cfg(feature = "cpu")]`
- Error handling with `anyhow::Result`
- Property-based testing with proptest
- TDD patterns: implement to make Red → Green

---

## Expected Outcomes

**After Implementation**:
- 13 tests transition from #[ignore] to active
- 9-11 tests should pass (some may reveal real limitations)
- 2-4 tests may correctly identify edge cases needing work
- Comprehensive property-based validation coverage

**Success Criteria**:
- All helper functions have real implementations
- Tests exercise production quantization APIs
- Failures reveal actual limitations, not missing code
- Coverage includes I2S, TL1, TL2 quantization
- Memory efficiency and numerical stability validated

---

## Files to Modify

**Primary**: `crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs`
- Remove 13 `#[ignore]` attributes
- Implement 13 helper functions (lines 868-1220)

**No New Files Needed** - all infrastructure exists

---

## Estimated Complexity

| Scaffold | Complexity | Time | Dependencies |
|----------|-----------|------|--------------|
| 1. I2S Round-Trip | Medium | 15 min | bitnet_quantization |
| 2. I2S Error Bounds | Low | 10 min | bitnet_quantization |
| 3. I2S Deterministic | Medium | 15 min | env vars |
| 4. TL1 Stability | Medium | 15 min | bitnet_quantization |
| 5. TL1 Sparsity | Medium | 15 min | create_sparse_weights |
| 6. Memory Scaling | High | 20 min | sysinfo |
| 7. Zero-Copy | High | 20 min | sysinfo, mmap |
| 8. NaN/Inf Handling | Low | 10 min | edge case injection |
| 9. Distribution | Medium | 15 min | statistics |
| 10. Extreme Range | Medium | 15 min | clipping |
| 11. Sparse Tensor | Low | 10 min | **partial impl** |
| 12. Architecture | High | 20 min | architecture parsing |
| 13. Custom Params | Medium | 15 min | custom quantizer |

**Total Estimated Time**: 3-4 hours (sequential) or 20-30 minutes (parallel with 13 agents)
