# TDD Scaffold Implementation Guide: GGUF Property Tests

**Issue**: #159  
**File**: `crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs`  
**Total Scaffolds**: 8 ignored tests  
**Priority**: HIGH  
**Sprint**: 4 - TDD Scaffold Completion

## Overview

This file implements property-based tests for GGUF weight loading quantization accuracy, numerical stability, and edge case handling. Uses the `proptest` framework to generate comprehensive test cases and validate quantization properties across multiple formats (I2S, TL1, TL2).

**Test Categories**:
- **I2S Quantization Properties** (3 tests)
- **TL2 Quantization Properties** (2 tests)
- **Memory Efficiency** (1 test)
- **Edge Cases & Numerical Stability** (2 tests)

**Current Status**: 8 tests marked `#[ignore]` with TDD placeholders. Helper functions are mostly stubbed with `unimplemented!()` markers.

---

## Scaffold 1: `prop_i2s_quantization_preserves_distribution`

**Lines**: 178-210  
**Status**: Stub (helper unimplemented)  
**Issue**: #159  
**Priority**: HIGH

### Current Implementation

```rust
#[test]
#[ignore] // Issue #159: TDD placeholder - I2S quantization integration needed
#[cfg(feature = "cpu")]
fn prop_i2s_quantization_preserves_distribution(
    weight_data in arbitrary_weight_tensor(),
    shape in arbitrary_tensor_shape(),
    params in arbitrary_quantization_params()
) { ... }
```

**Test Structure**: Exists with proptest generators  
**Helper Function**: `test_i2s_quantization_roundtrip()` - **STUBBED** (line 873-881)

```rust
fn test_i2s_quantization_roundtrip(
    weight_data: &[f32],
    shape: &[usize],
    params: &QuantizationTestParams,
) -> Result<f32> {
    // TODO: Implement I2S quantization round-trip test
    let _ = (weight_data, shape, params);
    Err(anyhow::anyhow!("I2S quantization integration not implemented"))
}
```

### What Needs Implementation

1. **Implement `test_i2s_quantization_roundtrip()` helper**:
   - Create tensor from weight_data using `bitnet_quantization::utils::create_tensor_from_f32()`
   - Quantize using `I2SQuantizer::new().quantize_tensor()`
   - Dequantize back using `quantized.dequantize()`
   - Calculate accuracy metric (MSE-based or correlation)
   - Return accuracy as f32 (0.0 to 1.0 range)

2. **Apply quantization params**:
   - Use `QuantizationTestParams` (block_size, scale, offset) to configure quantization
   - May need to create custom quantizer configuration

3. **Distribution validation**:
   - Compare original vs dequantized distributions
   - Check mean, variance, and correlation preservation

### Required APIs

- `bitnet_quantization::I2SQuantizer::new()` - Instantiate I2S quantizer
- `bitnet_quantization::Quantize::quantize_tensor()` - Quantize tensor
- `bitnet_quantization::utils::create_tensor_from_f32()` - Convert Vec<f32> to Tensor
- `bitnet_quantization::utils::extract_f32_data()` - Extract f32 from Tensor
- Statistical helpers: `calculate_mse()`, `calculate_signal_power()`, `calculate_correlation()` - Already exist (lines 1627-1659)

### Acceptance Criteria

- [ ] `test_i2s_quantization_roundtrip()` successfully quantizes and dequantizes
- [ ] Returns accuracy ≥ 0.99 (99% threshold from config)
- [ ] Handles arbitrary tensor shapes and sizes
- [ ] Respects quantization parameters (block_size, scale, offset)
- [ ] Proptest passes 100 iterations without failure

### Implementation Complexity

**Medium** - Requires understanding I2S quantization API and integrating with existing helper functions. Statistical validation is already implemented.

### Dependencies

- **Depends on**: I2S quantization API completion
- **Blocks**: Cross-quantization consistency tests
- **Related Tests**: `prop_i2s_quantization_error_bounds` (already passing), `prop_i2s_quantization_deterministic`

---

## Scaffold 2: `prop_i2s_quantization_deterministic`

**Lines**: 252-284  
**Status**: Stub (helper unimplemented)  
**Issue**: #159  
**Priority**: MEDIUM

### Current Implementation

```rust
#[test]
#[ignore] // Issue #159: TDD placeholder - I2S deterministic implementation needed
#[cfg(feature = "cpu")]
fn prop_i2s_quantization_deterministic(
    weight_data in arbitrary_weight_tensor(),
    shape in arbitrary_tensor_shape(),
    seed in 0u64..1000
) { ... }
```

**Test Structure**: Exists with seed-based determinism validation  
**Helper Function**: `test_i2s_quantization_deterministic()` - **STUBBED** (line 936-944)

```rust
fn test_i2s_quantization_deterministic(
    weight_data: &[f32],
    shape: &[usize],
    seed: u64,
) -> Result<Vec<f32>> {
    // TODO: Implement I2S deterministic quantization test
    let _ = (weight_data, shape, seed);
    Err(anyhow::anyhow!("I2S deterministic test not implemented"))
}
```

### What Needs Implementation

1. **Implement `test_i2s_quantization_deterministic()` helper**:
   - Set deterministic environment variables (`BITNET_DETERMINISTIC=1`, `BITNET_SEED=<seed>`)
   - Create tensor from weight_data
   - Quantize using I2S
   - Dequantize back
   - Extract and return dequantized data as Vec<f32>

2. **Determinism validation**:
   - Test calls helper twice with same seed
   - Assert exact equality of outputs (`prop_assert_eq!`)

3. **Environment cleanup**:
   - Ensure env vars are properly set/unset
   - Avoid test pollution

### Required APIs

- `std::env::set_var()` - Set environment variables for determinism
- `bitnet_quantization::I2SQuantizer::new()` - Instantiate quantizer
- `bitnet_quantization::Quantize::quantize_tensor()` - Quantize tensor
- `bitnet_quantization::utils::create_tensor_from_f32()` - Tensor creation
- `bitnet_quantization::utils::extract_f32_data()` - Extract dequantized data

### Acceptance Criteria

- [ ] `test_i2s_quantization_deterministic()` returns dequantized data
- [ ] Identical outputs for same seed across multiple runs
- [ ] Environment variables properly configure determinism
- [ ] No test pollution from env var changes
- [ ] Proptest validates determinism across 100 seed values

### Implementation Complexity

**Medium** - Requires environment variable handling and deterministic quantization support. Core quantization logic similar to Scaffold 1.

### Dependencies

- **Depends on**: I2S quantization determinism support, environment variable handling
- **Blocks**: Reproducibility validation for other quantization formats
- **Related Tests**: `prop_i2s_quantization_preserves_distribution`

---

## Scaffold 3: `prop_tl2_quantization_extreme_values`

**Lines**: 370-405  
**Status**: Stub (helper unimplemented)  
**Issue**: #159  
**Priority**: HIGH

### Current Implementation

```rust
#[test]
#[ignore] // Issue #159: TDD placeholder - TL2 extreme value handling needed
#[cfg(feature = "cpu")]
fn prop_tl2_quantization_extreme_values(
    base_data in arbitrary_weight_tensor(),
    shape in arbitrary_tensor_shape(),
    extreme_multiplier in 1.0f32..1000.0f32
) { ... }
```

**Test Structure**: Exists with extreme value generation  
**Helper Function**: `test_tl2_extreme_value_handling()` - **STUBBED** (line 1060-1064)

```rust
fn test_tl2_extreme_value_handling(weight_data: &[f32], shape: &[usize]) -> Result<(f32, bool)> {
    // TODO: Implement TL2 extreme value handling test
    let _ = (weight_data, shape);
    Err(anyhow::anyhow!("TL2 extreme value handling not implemented"))
}
```

### What Needs Implementation

1. **Implement `test_tl2_extreme_value_handling()` helper**:
   - Create tensor from extreme weight_data
   - Quantize using `TL2Quantizer::new().quantize_tensor()`
   - Dequantize back
   - Calculate accuracy (MSE-based)
   - Check if overflow/clipping handled gracefully (all outputs finite)
   - Return `(accuracy: f32, overflow_handled: bool)`

2. **Extreme value handling**:
   - Test multiplies base values by extreme_multiplier (1-1000x)
   - Validate TL2 quantization doesn't panic on extreme values
   - Ensure outputs are finite (no NaN/Inf propagation)

3. **Lower accuracy threshold**:
   - Test expects ≥90% accuracy (vs 99% for normal cases)
   - Reflects reduced precision for extreme values

### Required APIs

- `bitnet_quantization::TL2Quantizer::new()` - Instantiate TL2 quantizer
- `bitnet_quantization::Quantize::quantize_tensor()` - Quantize tensor
- `bitnet_quantization::utils::create_tensor_from_f32()` - Tensor creation
- `bitnet_quantization::utils::extract_f32_data()` - Extract dequantized data
- Statistical helpers: `calculate_mse()`, `calculate_signal_power()` - Already exist

### Acceptance Criteria

- [ ] `test_tl2_extreme_value_handling()` quantizes extreme values without panic
- [ ] Returns accuracy ≥ 0.9 (90% threshold for extreme values)
- [ ] `overflow_handled = true` (all outputs are finite)
- [ ] Handles multipliers from 1x to 1000x
- [ ] Proptest validates extreme value stability across 100 iterations

### Implementation Complexity

**Medium** - Requires TL2 quantization integration and overflow handling validation. Similar pattern to I2S tests.

### Dependencies

- **Depends on**: TL2 quantization API completion, extreme value handling
- **Blocks**: TL2 block size scaling tests
- **Related Tests**: `prop_tl2_quantization_block_size_scaling`

---

## Scaffold 4: `prop_tl2_quantization_block_size_scaling`

**Lines**: 410-441  
**Status**: Stub (helper unimplemented)  
**Issue**: #159  
**Priority**: MEDIUM

### Current Implementation

```rust
#[test]
#[ignore] // Issue #159: TDD placeholder - TL2 block size scaling needed
#[cfg(feature = "cpu")]
fn prop_tl2_quantization_block_size_scaling(
    weight_data in arbitrary_weight_tensor(),
    shape in arbitrary_tensor_shape(),
    block_size in 8usize..128
) { ... }
```

**Test Structure**: Exists with block size variation  
**Helper Function**: `test_tl2_block_size_effects()` - **STUBBED** (line 1067-1075)

```rust
fn test_tl2_block_size_effects(
    weight_data: &[f32],
    shape: &[usize],
    block_size: usize,
) -> Result<f32> {
    // TODO: Implement TL2 block size analysis
    let _ = (weight_data, shape, block_size);
    Err(anyhow::anyhow!("TL2 block size analysis not implemented"))
}
```

### What Needs Implementation

1. **Implement `test_tl2_block_size_effects()` helper**:
   - Create tensor from weight_data
   - Configure TL2 quantizer with specified block_size
   - Quantize and dequantize
   - Calculate accuracy metric
   - Return accuracy as f32

2. **Block size configuration**:
   - Test validates block_size from 8 to 128
   - Larger block sizes should provide better accuracy
   - Adjust min_accuracy threshold: ≥95% for block_size ≥32, ≥90% otherwise

3. **Block size compatibility**:
   - Ensure block_size ≤ tensor size (already handled by test adjustment)

### Required APIs

- `bitnet_quantization::TL2Quantizer::new()` - Instantiate quantizer
- `bitnet_quantization::TL2Quantizer::with_block_size()` - Configure block size (may need implementation)
- `bitnet_quantization::Quantize::quantize_tensor()` - Quantize tensor
- `bitnet_quantization::utils::create_tensor_from_f32()` - Tensor creation
- `bitnet_quantization::utils::extract_f32_data()` - Extract data
- Statistical helpers: `calculate_mse()`, `calculate_signal_power()` - Already exist

### Acceptance Criteria

- [ ] `test_tl2_block_size_effects()` quantizes with custom block_size
- [ ] Returns accuracy ≥ 0.95 for block_size ≥ 32
- [ ] Returns accuracy ≥ 0.9 for block_size < 32
- [ ] Handles block sizes from 8 to 128
- [ ] Validates block size scaling hypothesis (larger = better accuracy)

### Implementation Complexity

**Medium** - Requires TL2 quantizer block size configuration. May need API extension for custom block size.

### Dependencies

- **Depends on**: TL2 quantization API, configurable block size support
- **Blocks**: Block alignment optimization tests
- **Related Tests**: `prop_tl2_quantization_extreme_values`, `prop_block_aligned_quantization`

---

## Scaffold 5: `prop_memory_usage_linear_scaling`

**Lines**: 504-535  
**Status**: Stub (helper unimplemented)  
**Issue**: #159  
**Priority**: LOW

### Current Implementation

```rust
#[test]
#[ignore] // Issue #159: TDD placeholder - memory usage scaling implementation needed
#[cfg(feature = "cpu")]
fn prop_memory_usage_linear_scaling(
    base_size in (64usize..1024),
    scale_factor in (1usize..8)
) { ... }
```

**Test Structure**: Exists with size scaling validation  
**Helper Function**: `test_memory_usage_scaling()` - **STUBBED** (line 1089-1097)

```rust
fn test_memory_usage_scaling(
    data1: &[f32],
    data2: &[f32],
    expected_scale: usize,
) -> Result<(usize, usize, f32)> {
    // TODO: Implement memory usage scaling test
    let _ = (data1, data2, expected_scale);
    Err(anyhow::anyhow!("Memory usage scaling not implemented"))
}
```

### What Needs Implementation

1. **Implement `test_memory_usage_scaling()` helper**:
   - Measure memory usage for data1 (size = base_size)
   - Measure memory usage for data2 (size = base_size * scale_factor)
   - Calculate actual scale: memory2 / memory1
   - Return `(memory1: usize, memory2: usize, actual_scale: f32)`

2. **Memory measurement**:
   - Use `sysinfo` crate (already imported) to track memory
   - Similar pattern to `test_zero_copy_efficiency()` (line 1100-1168)
   - Create temporary files or allocate memory for tensors

3. **Linear scaling validation**:
   - Test expects actual_scale ≈ expected_scale (within 20% tolerance)
   - Validates memory usage grows linearly with tensor size

### Required APIs

- `sysinfo::System::new_with_specifics()` - Initialize system info tracker
- `sysinfo::System::refresh_memory()` - Refresh memory stats
- `sysinfo::System::used_memory()` - Get current memory usage
- `bitnet_quantization::I2SQuantizer::new()` - Quantize tensors
- `std::fs::File` / `tempfile` - Create temporary files for memory testing

### Acceptance Criteria

- [ ] `test_memory_usage_scaling()` measures memory usage accurately
- [ ] Returns actual_scale ≈ expected_scale (within 20% tolerance)
- [ ] Validates linear scaling for scale_factors 1x to 8x
- [ ] Handles tensor sizes from 64 to 8192 elements
- [ ] Proptest validates scaling across 100 iterations

### Implementation Complexity

**Low-Medium** - Memory measurement pattern exists in `test_zero_copy_efficiency()`. Requires adaptation for scaling validation.

### Dependencies

- **Depends on**: Memory tracking infrastructure (already exists)
- **Blocks**: None (standalone validation)
- **Related Tests**: `prop_zero_copy_memory_efficiency` (uses similar memory tracking)

---

## Scaffold 6: `prop_quantization_handles_nan_inf`

**Lines**: 580-612  
**Status**: Stub (helper unimplemented)  
**Issue**: #159  
**Priority**: HIGH

### Current Implementation

```rust
#[test]
#[ignore] // Issue #159: TDD placeholder - NaN/Inf handling implementation needed
#[cfg(feature = "cpu")]
fn prop_quantization_handles_nan_inf(
    weight_data in arbitrary_weight_tensor_with_edge_cases(),
    shape in arbitrary_tensor_shape()
) { ... }
```

**Test Structure**: Exists with edge case generator (NaN, Inf, denormals)  
**Helper Function**: `test_edge_case_handling()` - **STUBBED** (line 1226-1230)

```rust
fn test_edge_case_handling(weight_data: &[f32], shape: &[usize]) -> Result<(bool, bool, Vec<f32>)> {
    // TODO: Implement edge case handling test
    let _ = (weight_data, shape);
    Err(anyhow::anyhow!("Edge case handling not implemented"))
}
```

### What Needs Implementation

1. **Implement `test_edge_case_handling()` helper**:
   - Check if input contains NaN values
   - Check if input contains Inf values
   - Quantize using I2S (or other quantization method)
   - Dequantize back
   - Extract dequantized data
   - Validate all outputs are finite (no NaN/Inf propagation)
   - Return `(nan_handled: bool, inf_handled: bool, finite_output: Vec<f32>)`

2. **Edge case handling**:
   - Test uses `arbitrary_weight_tensor_with_edge_cases()` generator
   - Generates NaN, Inf, denormals, near-zero values, extreme values
   - Quantization should sanitize/clip these values

3. **Finite output validation**:
   - Test asserts `finite_output.iter().all(|&x| x.is_finite())`
   - No NaN/Inf should survive quantization round-trip

### Required APIs

- `bitnet_quantization::I2SQuantizer::new()` - Instantiate quantizer
- `bitnet_quantization::Quantize::quantize_tensor()` - Quantize tensor
- `bitnet_quantization::utils::create_tensor_from_f32()` - Tensor creation
- `bitnet_quantization::utils::extract_f32_data()` - Extract data
- `f32::is_finite()` - Check for NaN/Inf

### Acceptance Criteria

- [ ] `test_edge_case_handling()` processes NaN/Inf without panic
- [ ] Returns `nan_handled = true` (NaN values sanitized)
- [ ] Returns `inf_handled = true` (Inf values sanitized)
- [ ] All dequantized outputs are finite (no NaN/Inf propagation)
- [ ] Proptest validates edge case handling across 100 iterations

### Implementation Complexity

**Medium** - Requires edge case handling in quantization pipeline. May need input sanitization or clipping logic.

### Dependencies

- **Depends on**: Quantization edge case handling, input sanitization
- **Blocks**: Distribution preservation tests
- **Related Tests**: `prop_extreme_dynamic_range` (similar clipping validation)

---

## Scaffold 7: `prop_quantization_preserves_distribution`

**Lines**: 616-652  
**Status**: Stub (helper unimplemented)  
**Issue**: #159  
**Priority**: HIGH

### Current Implementation

```rust
#[test]
#[ignore] // Issue #159: TDD placeholder - distribution preservation validation needed
#[cfg(feature = "cpu")]
fn prop_quantization_preserves_distribution(
    weight_data in arbitrary_weight_tensor(),
    shape in arbitrary_tensor_shape()
) { ... }
```

**Test Structure**: Exists with distribution validation  
**Helper Function**: `test_distribution_preservation()` - **STUBBED** (line 1233-1241)

```rust
fn test_distribution_preservation(
    weight_data: &[f32],
    shape: &[usize],
) -> Result<(bool, bool, f32)> {
    // TODO: Implement distribution preservation test
    // Returns (mean_preserved, variance_preserved, correlation)
    let _ = (weight_data, shape);
    Err(anyhow::anyhow!("Distribution preservation not implemented"))
}
```

### What Needs Implementation

1. **Implement `test_distribution_preservation()` helper**:
   - Create tensor from weight_data
   - Quantize using I2S
   - Dequantize back
   - Calculate original and dequantized statistics:
     - Mean (using `calculate_mean()` - line 1564-1573)
     - Variance (using `calculate_variance()` - line 1576-1587)
     - Correlation (using `calculate_correlation()` - line 1591-1615)
   - Check if mean preserved (within tolerance)
   - Check if variance preserved (within tolerance)
   - Return `(mean_preserved: bool, variance_preserved: bool, correlation: f32)`

2. **Distribution metrics**:
   - Mean preservation: `|mean_orig - mean_deq| / mean_orig < tolerance` (e.g., 5%)
   - Variance preservation: `|var_orig - var_deq| / var_orig < tolerance` (e.g., 10%)
   - Correlation: Pearson coefficient ≥ 0.99

3. **Statistical helpers**:
   - All helper functions already exist (lines 1564-1615)
   - Just need to integrate them

### Required APIs

- `bitnet_quantization::I2SQuantizer::new()` - Instantiate quantizer
- `bitnet_quantization::Quantize::quantize_tensor()` - Quantize tensor
- `bitnet_quantization::utils::create_tensor_from_f32()` - Tensor creation
- `bitnet_quantization::utils::extract_f32_data()` - Extract data
- Statistical helpers: `calculate_mean()`, `calculate_variance()`, `calculate_correlation()` - Already exist

### Acceptance Criteria

- [ ] `test_distribution_preservation()` calculates distribution metrics
- [ ] Returns `mean_preserved = true` (mean within tolerance)
- [ ] Returns `variance_preserved = true` (variance within tolerance)
- [ ] Returns `correlation ≥ 0.99` (99% threshold)
- [ ] Proptest validates distribution preservation across 100 iterations

### Implementation Complexity

**Low-Medium** - Statistical helpers already exist. Just need to integrate with quantization pipeline and define tolerances.

### Dependencies

- **Depends on**: I2S quantization API
- **Blocks**: None (standalone validation)
- **Related Tests**: `prop_i2s_quantization_preserves_distribution` (different approach)

---

## Scaffold 8: `prop_block_aligned_quantization`

**Lines**: 656-693  
**Status**: Stub (helper unimplemented)  
**Issue**: #159  
**Priority**: MEDIUM

### Current Implementation

```rust
#[test]
#[ignore] // Issue #159: TDD placeholder - block alignment optimization needed
#[cfg(feature = "cpu")]
fn prop_block_aligned_quantization(
    block_size in prop::sample::select(vec![32usize, 64, 128, 256]),
    shape in arbitrary_tensor_shape()
) { ... }
```

**Test Structure**: Exists with block alignment generation  
**Helper Function**: `test_block_aligned_efficiency()` - **STUBBED** (line 1244-1253)

```rust
fn test_block_aligned_efficiency(
    weight_data: &[f32],
    shape: &[usize],
    block_size: usize,
) -> Result<(f32, f32)> {
    // TODO: Implement block alignment efficiency test
    // Returns (accuracy, efficiency_gain)
    let _ = (weight_data, shape, block_size);
    Err(anyhow::anyhow!("Block alignment efficiency not implemented"))
}
```

### What Needs Implementation

1. **Implement `test_block_aligned_efficiency()` helper**:
   - Create tensor from block-aligned weight_data
   - Quantize using I2S with block_size configuration
   - Dequantize back
   - Calculate accuracy (MSE-based)
   - Calculate efficiency_gain (e.g., time savings or memory reduction)
   - Return `(accuracy: f32, efficiency_gain: f32)`

2. **Block alignment validation**:
   - Test creates block-aligned shapes (multiples of block_size)
   - Validates quantization efficiency improves with alignment
   - Efficiency_gain ≥ 0.0 (no degradation)

3. **Efficiency metric**:
   - Could measure quantization time difference
   - Or memory reduction ratio
   - Or operation count reduction

### Required APIs

- `bitnet_quantization::I2SQuantizer::new()` - Instantiate quantizer
- `bitnet_quantization::I2SQuantizer::with_block_size()` - Configure block size (may need implementation)
- `bitnet_quantization::Quantize::quantize_tensor()` - Quantize tensor
- `bitnet_quantization::utils::create_tensor_from_f32()` - Tensor creation
- `bitnet_quantization::utils::extract_f32_data()` - Extract data
- Statistical helpers: `calculate_mse()`, `calculate_signal_power()` - Already exist
- Timing: `std::time::Instant` - Measure quantization time

### Acceptance Criteria

- [ ] `test_block_aligned_efficiency()` quantizes block-aligned tensors
- [ ] Returns accuracy ≥ 0.99 (99% threshold)
- [ ] Returns efficiency_gain ≥ 0.0 (no degradation)
- [ ] Validates block sizes 32, 64, 128, 256
- [ ] Proptest validates alignment benefits across 100 iterations

### Implementation Complexity

**Medium** - Requires block size configuration and efficiency measurement. May need timing infrastructure or memory profiling.

### Dependencies

- **Depends on**: I2S block size configuration, efficiency measurement infrastructure
- **Blocks**: None (standalone validation)
- **Related Tests**: `prop_tl2_quantization_block_size_scaling` (similar block size validation)

---

## Implementation Order Recommendation

1. **Scaffold 7: `prop_quantization_preserves_distribution`** - **START HERE**
   - **Why**: Statistical helpers already exist, just need integration. Validates core quantization quality.
   - **Complexity**: Low-Medium
   - **Impact**: High (validates core quantization property)

2. **Scaffold 1: `prop_i2s_quantization_preserves_distribution`**
   - **Why**: Similar to Scaffold 7 but uses quantization params. Builds on statistical foundation.
   - **Complexity**: Medium
   - **Impact**: High (validates I2S with custom params)

3. **Scaffold 6: `prop_quantization_handles_nan_inf`**
   - **Why**: Edge case handling is critical for robustness. Independent of other tests.
   - **Complexity**: Medium
   - **Impact**: High (prevents runtime panics)

4. **Scaffold 3: `prop_tl2_quantization_extreme_values`**
   - **Why**: TL2 extreme value handling, similar pattern to Scaffold 6.
   - **Complexity**: Medium
   - **Impact**: High (validates TL2 robustness)

5. **Scaffold 2: `prop_i2s_quantization_deterministic`**
   - **Why**: Determinism validation, depends on I2S API stability.
   - **Complexity**: Medium
   - **Impact**: Medium (reproducibility guarantee)

6. **Scaffold 4: `prop_tl2_quantization_block_size_scaling`**
   - **Why**: TL2 block size validation, may require API extension.
   - **Complexity**: Medium
   - **Impact**: Medium (validates TL2 scalability)

7. **Scaffold 8: `prop_block_aligned_quantization`**
   - **Why**: Block alignment optimization, requires efficiency measurement.
   - **Complexity**: Medium
   - **Impact**: Medium (performance optimization validation)

8. **Scaffold 5: `prop_memory_usage_linear_scaling`**
   - **Why**: Memory scaling validation, independent but lower priority.
   - **Complexity**: Low-Medium
   - **Impact**: Low (nice-to-have memory validation)

---

## Common Patterns

### Quantization Round-Trip Pattern

All scaffolds follow a similar pattern for quantization testing:

```rust
// 1. Create tensor from weight data
let tensor = bitnet_quantization::utils::create_tensor_from_f32(
    weight_data.to_vec(),
    shape,
    &candle_core::Device::Cpu,
)?;

// 2. Quantize using appropriate quantizer
let quantizer = I2SQuantizer::new(); // or TL1Quantizer, TL2Quantizer
let quantized = quantizer.quantize_tensor(&tensor)?;

// 3. Dequantize back to f32
let dequantized = quantized.dequantize()?;

// 4. Extract data for comparison
let original_data = bitnet_quantization::utils::extract_f32_data(&tensor)?;
let dequantized_data = bitnet_quantization::utils::extract_f32_data(&dequantized)?;

// 5. Calculate accuracy metric
let mse = calculate_mse(&original_data, &dequantized_data);
let signal_power = calculate_signal_power(&original_data);
let accuracy = if signal_power > 1e-10 {
    1.0 - (mse / signal_power)
} else {
    if mse < 1e-6 { 1.0 } else { 0.0 }
};
```

### Statistical Validation Pattern

Use existing helper functions for statistical validation:

```rust
// Mean preservation
let mean_orig = calculate_mean(&original_data);
let mean_deq = calculate_mean(&dequantized_data);
let mean_preserved = (mean_orig - mean_deq).abs() / mean_orig.max(1e-10) < 0.05; // 5% tolerance

// Variance preservation
let var_orig = calculate_variance(&original_data);
let var_deq = calculate_variance(&dequantized_data);
let variance_preserved = (var_orig - var_deq).abs() / var_orig.max(1e-10) < 0.10; // 10% tolerance

// Correlation
let correlation = calculate_correlation(&original_data, &dequantized_data);
// Expect correlation ≥ 0.99
```

### Edge Case Handling Pattern

For NaN/Inf handling:

```rust
// Check for edge cases in input
let has_nan = weight_data.iter().any(|x| x.is_nan());
let has_inf = weight_data.iter().any(|x| x.is_infinite());

// Quantize (should sanitize edge cases)
let quantized = quantizer.quantize_tensor(&tensor)?;
let dequantized = quantized.dequantize()?;
let dequantized_data = extract_f32_data(&dequantized)?;

// Validate all outputs are finite
let nan_handled = has_nan; // Mark as handled if input had NaN
let inf_handled = has_inf; // Mark as handled if input had Inf
let all_finite = dequantized_data.iter().all(|x| x.is_finite());

Ok((nan_handled, inf_handled, dequantized_data))
```

### Memory Measurement Pattern

See `test_zero_copy_efficiency()` (line 1100-1168) for reference:

```rust
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

// Initialize system info tracker
let mut sys = System::new_with_specifics(
    RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
);

// Measure memory before
sys.refresh_memory();
let memory_before = sys.used_memory();

// Perform operation (quantization, allocation, etc.)
let tensor = /* ... */;

// Measure memory after
sys.refresh_memory();
let memory_after = sys.used_memory();
let memory_used = memory_after.saturating_sub(memory_before);
```

---

## Test Execution Notes

### Running Ignored Tests

```bash
# Run all property tests (including ignored)
cargo test -p bitnet-models --test gguf_weight_loading_property_tests --no-default-features --features cpu -- --ignored --include-ignored

# Run specific ignored test
cargo test -p bitnet-models --test gguf_weight_loading_property_tests --no-default-features --features cpu prop_i2s_quantization_preserves_distribution -- --ignored

# Run with verbose output (shows proptest iterations)
cargo test -p bitnet-models --test gguf_weight_loading_property_tests --no-default-features --features cpu -- --ignored --nocapture
```

### Proptest Configuration

All tests use `PropertyTestConfig::default()`:

```rust
PropertyTestConfig {
    accuracy_threshold: 0.99,      // 99% accuracy threshold
    numerical_tolerance: 1e-5,     // Numerical comparison tolerance
    max_tensor_size: 8192,         // Max elements in tensor
    min_tensor_size: 32,           // Min elements in tensor
    test_iterations: 100,          // Proptest iterations (default)
}
```

To adjust proptest iterations, set environment variable:

```bash
PROPTEST_CASES=200 cargo test -p bitnet-models --test gguf_weight_loading_property_tests
```

### Test-Specific Environment Variables

- `BITNET_DETERMINISTIC=1` - Enable deterministic inference (Scaffold 2)
- `BITNET_SEED=<seed>` - Set random seed for determinism (Scaffold 2)

---

## API Requirements Summary

### Existing APIs (Ready to Use)

- ✅ `bitnet_quantization::I2SQuantizer::new()` - I2S quantizer instantiation
- ✅ `bitnet_quantization::TL1Quantizer::new()` - TL1 quantizer instantiation
- ✅ `bitnet_quantization::TL2Quantizer::new()` - TL2 quantizer instantiation (needs validation)
- ✅ `bitnet_quantization::Quantize::quantize_tensor()` - Quantize tensor
- ✅ `bitnet_quantization::QuantizedTensor::dequantize()` - Dequantize tensor
- ✅ `bitnet_quantization::utils::create_tensor_from_f32()` - Tensor creation
- ✅ `bitnet_quantization::utils::extract_f32_data()` - Extract f32 data
- ✅ `bitnet_models::loader::MmapFile::open()` - Memory-mapped file (for memory tests)
- ✅ Statistical helpers: `calculate_mse()`, `calculate_signal_power()`, `calculate_mean()`, `calculate_variance()`, `calculate_correlation()` - All exist in file

### APIs Needing Extension

- ⚠️ `I2SQuantizer::with_quantization_params()` - Configure custom block_size, scale, offset (Scaffold 1)
- ⚠️ `TL2Quantizer::with_block_size()` - Configure custom block size (Scaffold 4, 8)
- ⚠️ Edge case handling in quantizers - Sanitize NaN/Inf inputs (Scaffold 6)
- ⚠️ Determinism support - Respect `BITNET_DETERMINISTIC` and `BITNET_SEED` env vars (Scaffold 2)

### External Dependencies

- ✅ `proptest` - Property-based testing framework (already imported)
- ✅ `anyhow` - Error handling (already imported)
- ✅ `sysinfo` - Memory tracking (already used in `test_zero_copy_efficiency()`)
- ✅ `tempfile` - Temporary files for memory tests (already used)
- ✅ `candle_core` - Tensor operations (already imported via bitnet_quantization)

---

## Blockers and Risks

### Active Blockers

1. **TL2 Quantizer API Maturity**: Scaffolds 3 and 4 depend on TL2 quantizer completion and API stability.
2. **Custom Quantization Params**: Scaffold 1 requires API extension for custom block_size/scale/offset configuration.
3. **Determinism Support**: Scaffold 2 requires environment variable handling in quantizers.
4. **Edge Case Handling**: Scaffold 6 requires NaN/Inf sanitization in quantization pipeline.

### Risks

1. **Proptest Performance**: Property tests with large tensors (8192 elements) may be slow. Consider reducing `max_tensor_size` or using `PROPTEST_CASES` env var.
2. **Memory Measurement Flakiness**: Memory tests (Scaffold 5) may be flaky on different systems due to OS memory management variations.
3. **Accuracy Thresholds**: 99% accuracy threshold may be too strict for some quantization formats. Monitor test failures and adjust thresholds if needed.
4. **Environment Variable Pollution**: Scaffold 2 uses `std::env::set_var()` which can pollute test environment. Ensure proper cleanup.

---

## Next Steps

1. **Start with Scaffold 7** - Implement `test_distribution_preservation()` using existing statistical helpers.
2. **Validate I2S API** - Ensure `I2SQuantizer` API is stable and complete before implementing Scaffolds 1-2.
3. **Validate TL2 API** - Ensure `TL2Quantizer` API is stable before implementing Scaffolds 3-4.
4. **Add API Extensions** - Implement custom block_size/scale configuration if needed.
5. **Test Iteratively** - Remove `#[ignore]` from each test as implementation completes, run proptest to validate.
6. **Monitor Performance** - Track proptest execution time, adjust `max_tensor_size` if tests are too slow.

---

## Success Criteria

- [ ] All 8 ignored tests pass without `#[ignore]` markers
- [ ] Proptest runs 100 iterations for each test without failure
- [ ] Accuracy thresholds met: ≥99% for normal cases, ≥90% for extreme values
- [ ] Edge cases (NaN/Inf) handled without panics
- [ ] Determinism validated with same seed producing identical outputs
- [ ] Memory usage scales linearly (within 20% tolerance)
- [ ] Zero-copy efficiency validated (≤80% memory ratio)
- [ ] Block alignment improves efficiency (gain ≥ 0.0)
- [ ] CI pipeline passes all property tests without `--ignored` flag

---

**End of Guide**
