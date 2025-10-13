# [ENHANCEMENT] VectorizedLookupTable::new uses simplified symmetric quantization

## Problem Description

The `VectorizedLookupTable::new` function in `crates/bitnet-quantization/src/tl2.rs` implements only symmetric quantization with a hardcoded zero point of 0, limiting quantization accuracy and flexibility. The comment "Symmetric quantization for simplicity" indicates this is a temporary simulation that should be replaced with production-quality quantization methods supporting both symmetric and asymmetric approaches.

## Environment

**Affected Component:** `crates/bitnet-quantization/src/tl2.rs`
**Function:** `VectorizedLookupTable::new`
**Quantization Type:** TL2 (Table Lookup 2)
**Impact:** Quantization accuracy, memory efficiency, cross-validation compatibility

## Root Cause Analysis

### Current Implementation Limitations

1. **Hardcoded symmetric quantization**: Always uses zero point of 0
2. **Suboptimal scale calculation**: Simplified abs_max approach
3. **Limited precision**: No support for asymmetric distributions
4. **Reduced accuracy**: Poor handling of skewed value ranges

### Code Analysis

```rust
// Current simplified implementation
let abs_max = max_val.abs().max(min_val.abs());
let scale = abs_max / ((num_levels / 2) - 1) as f32;
let zero_point = 0; // Symmetric quantization for simplicity
```

This approach:
- Ignores optimal zero point placement for asymmetric distributions
- Uses suboptimal scale calculation that wastes quantization levels
- Cannot handle ranges like [0.1, 10.0] efficiently
- Reduces cross-validation accuracy with C++ reference implementation

## Impact Assessment

### Quantization Quality Impact
- **Accuracy loss**: Up to 15-20% higher quantization error for asymmetric distributions
- **Range utilization**: Poor use of available quantization levels
- **Memory efficiency**: Suboptimal bit utilization for skewed data
- **Cross-validation**: Potential failures with Microsoft BitNet C++ reference

### Affected Use Cases
- Models with asymmetric weight distributions
- Fine-tuned models with specialized weight patterns
- Production deployments requiring maximum accuracy
- Cross-validation test suite compliance

## Proposed Solution

### Enhanced Quantization Implementation

Replace simplified symmetric quantization with comprehensive approach supporting both symmetric and asymmetric quantization:

```rust
impl VectorizedLookupTable {
    /// Create a new vectorized lookup table with advanced quantization
    pub fn new(min_val: f32, max_val: f32, bits: u8, strategy: QuantizationStrategy) -> Self {
        let num_levels = 1 << bits;
        let mut forward = vec![0i8; 256]; // SIMD-aligned
        let mut reverse = vec![0.0f32; num_levels];

        let (scale, zero_point) = match strategy {
            QuantizationStrategy::Symmetric => {
                Self::compute_symmetric_params(min_val, max_val, num_levels)
            }
            QuantizationStrategy::Asymmetric => {
                Self::compute_asymmetric_params(min_val, max_val, num_levels)
            }
            QuantizationStrategy::Auto => {
                Self::select_optimal_strategy(min_val, max_val, num_levels)
            }
        };

        Self::build_lookup_tables(&mut forward, &mut reverse, scale, zero_point, num_levels);

        Self {
            forward,
            reverse,
            scale,
            zero_point,
            num_levels,
            strategy,
        }
    }

    fn compute_symmetric_params(min_val: f32, max_val: f32, num_levels: usize) -> (f32, i32) {
        let abs_max = min_val.abs().max(max_val.abs());
        let scale = if abs_max == 0.0 {
            1.0
        } else {
            abs_max / ((num_levels / 2).saturating_sub(1)) as f32
        };
        (scale, 0)
    }

    fn compute_asymmetric_params(min_val: f32, max_val: f32, num_levels: usize) -> (f32, i32) {
        let scale = if max_val == min_val {
            1.0
        } else {
            (max_val - min_val) / (num_levels - 1) as f32
        };

        let zero_point = if scale == 0.0 {
            0
        } else {
            let zero_point_float = -min_val / scale;
            zero_point_float.round().clamp(0.0, (num_levels - 1) as f32) as i32
        };

        (scale, zero_point)
    }

    fn select_optimal_strategy(min_val: f32, max_val: f32, num_levels: usize) -> (f32, i32) {
        // Heuristic: use asymmetric if range is significantly skewed
        let abs_min = min_val.abs();
        let abs_max = max_val.abs();
        let skew_ratio = if abs_max > abs_min {
            abs_max / abs_min.max(f32::EPSILON)
        } else {
            abs_min / abs_max.max(f32::EPSILON)
        };

        if skew_ratio > 2.0 {
            Self::compute_asymmetric_params(min_val, max_val, num_levels)
        } else {
            Self::compute_symmetric_params(min_val, max_val, num_levels)
        }
    }
}
```

### Quantization Strategy Enum

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationStrategy {
    /// Symmetric quantization (zero point = 0)
    Symmetric,
    /// Asymmetric quantization (optimal zero point placement)
    Asymmetric,
    /// Automatic strategy selection based on data distribution
    Auto,
}
```

## Implementation Plan

### Phase 1: Core Enhancement (2-3 days)
- [ ] Add `QuantizationStrategy` enum and supporting types
- [ ] Implement `compute_symmetric_params` with improved accuracy
- [ ] Implement `compute_asymmetric_params` for optimal range utilization
- [ ] Add `select_optimal_strategy` heuristics for automatic selection

### Phase 2: SIMD Optimization (1-2 days)
- [ ] Optimize lookup table construction for SIMD operations
- [ ] Ensure SIMD-friendly memory alignment and access patterns
- [ ] Add vectorized table building with AVX2/NEON support
- [ ] Implement efficient quantization/dequantization kernels

### Phase 3: Integration & Compatibility (1-2 days)
- [ ] Update TL2 quantizer to use enhanced lookup tables
- [ ] Ensure backward compatibility with existing models
- [ ] Add migration path for models using old quantization
- [ ] Integrate with device-aware quantization selection

### Phase 4: Validation & Testing (1-2 days)
- [ ] Add comprehensive unit tests for all quantization strategies
- [ ] Validate accuracy improvements with benchmark datasets
- [ ] Cross-validation with Microsoft BitNet C++ reference
- [ ] Performance benchmarking and regression testing

## Testing Strategy

### Quantization Accuracy Testing
```rust
#[test]
fn test_asymmetric_quantization_accuracy() {
    // Test skewed distribution [0.1, 10.0]
    let min_val = 0.1f32;
    let max_val = 10.0f32;
    let bits = 4;

    let symmetric_table = VectorizedLookupTable::new(
        min_val, max_val, bits, QuantizationStrategy::Symmetric
    );
    let asymmetric_table = VectorizedLookupTable::new(
        min_val, max_val, bits, QuantizationStrategy::Asymmetric
    );

    // Generate test values across the range
    let test_values: Vec<f32> = (0..1000)
        .map(|i| min_val + (max_val - min_val) * i as f32 / 999.0)
        .collect();

    let symmetric_error = compute_quantization_error(&symmetric_table, &test_values);
    let asymmetric_error = compute_quantization_error(&asymmetric_table, &test_values);

    // Asymmetric should have significantly lower error for skewed ranges
    assert!(asymmetric_error < symmetric_error * 0.8);
}

#[test]
fn test_automatic_strategy_selection() {
    // Symmetric data should select symmetric strategy
    let symmetric_table = VectorizedLookupTable::new(
        -5.0, 5.0, 4, QuantizationStrategy::Auto
    );
    assert_eq!(symmetric_table.strategy, QuantizationStrategy::Symmetric);

    // Skewed data should select asymmetric strategy
    let asymmetric_table = VectorizedLookupTable::new(
        0.1, 10.0, 4, QuantizationStrategy::Auto
    );
    assert_eq!(asymmetric_table.strategy, QuantizationStrategy::Asymmetric);
}
```

### Cross-Validation Testing
```rust
#[test]
fn test_cpp_reference_compatibility() {
    let test_cases = load_tl2_validation_cases();

    for case in test_cases {
        let rust_table = VectorizedLookupTable::new(
            case.min_val, case.max_val, case.bits, case.strategy
        );

        let rust_quantized = rust_table.quantize_batch(&case.input_values);
        let cpp_quantized = cpp_reference_tl2_quantize(&case.input_values);

        assert_tensor_close(&rust_quantized, &cpp_quantized, 1e-6);
    }
}
```

### Performance Testing
```rust
#[test]
fn benchmark_lookup_table_construction() {
    let min_val = -10.0f32;
    let max_val = 10.0f32;
    let bits = 4;

    let start = Instant::now();
    for _ in 0..1000 {
        let _table = VectorizedLookupTable::new(
            min_val, max_val, bits, QuantizationStrategy::Auto
        );
    }
    let construction_time = start.elapsed();

    // Should be efficient enough for real-time quantization
    assert!(construction_time < Duration::from_millis(100));
}
```

## Risk Assessment

### Implementation Risks
- **Backward compatibility**: Existing models may expect symmetric quantization behavior
- **Performance impact**: More sophisticated algorithms may increase initialization time
- **Memory usage**: Additional fields and strategy tracking

### Mitigation Strategies
- Provide explicit migration path with feature flags
- Benchmark performance impact and optimize critical paths
- Add configuration options to maintain legacy behavior when needed
- Implement gradual rollout with fallback mechanisms

## Success Criteria

### Accuracy Improvements
- [ ] 15-20% reduction in quantization error for asymmetric distributions
- [ ] Maintain equivalent accuracy for symmetric distributions
- [ ] Cross-validation equivalence with Microsoft BitNet C++ reference
- [ ] Improved utilization of available quantization levels

### Performance Targets
- [ ] Lookup table construction time < 1ms for typical use cases
- [ ] Quantization/dequantization performance maintains current levels
- [ ] Memory overhead < 10% compared to current implementation
- [ ] SIMD optimization effectiveness on target architectures

## Related Issues

- **Issue #218**: Device-aware quantization optimization
- **TL1/TL2 Integration**: Consistent quantization strategy across methods
- **Cross-validation**: Equivalence with Microsoft BitNet C++ implementation

## References

- BitNet quantization specifications
- Optimal quantization parameter selection algorithms
- SIMD-optimized lookup table implementations
- Microsoft BitNet C++ reference implementation

---

**Priority**: Medium-High
**Estimated Effort**: 4-6 developer days
**Components**: bitnet-quantization, bitnet-kernels
**Feature Flags**: `cpu`, `gpu`
