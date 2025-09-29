# [FEATURE] Implement dynamic growth for rotary embeddings to support variable sequence lengths

## Problem Description

The `RotaryEmbedding::apply` function in `crates/bitnet-models/src/transformer.rs` currently has a fixed maximum sequence length limitation without dynamic growth capability. When sequences exceed the pre-computed `max_seq_len`, the system fails rather than adapting to the longer sequences. This limitation prevents handling variable-length inputs and reduces the model's flexibility for production use cases.

## Environment

**Affected Component:** `crates/bitnet-models/src/transformer.rs`
**Function:** `RotaryEmbedding::apply`
**Impact:** Sequence length flexibility, production robustness, memory efficiency
**Related Features:** `cpu`, `gpu` (both affected by sequence length limitations)

## Root Cause Analysis

### Current Implementation Limitations

1. **Fixed sequence length**: Pre-computed sin/cos tables have fixed maximum length
2. **Hard failure**: No graceful handling of longer sequences
3. **Memory inefficiency**: Large fixed tables waste memory for short sequences
4. **Inflexibility**: Cannot adapt to varying inference requirements

### Code Analysis

The current implementation lacks dynamic growth logic:

```rust
pub fn apply(&self, x: &Tensor, position: usize) -> Result<Tensor> {
    // Current implementation assumes sequence fits within max_seq_len
    let cos = self.cos.narrow(0, position, seq_len)?; // Fails if position + seq_len > max_seq_len
    let sin = self.sin.narrow(0, position, seq_len)?;
    // ... rest of implementation
}
```

Issues:
- No bounds checking with graceful expansion
- Fixed sin/cos tables cannot accommodate longer sequences
- Memory allocated for maximum possible length regardless of actual usage
- No consideration for batch processing with varying sequence lengths

## Impact Assessment

### Functional Impact
- **Sequence length restrictions**: Cannot process sequences longer than initial max_seq_len
- **Batch processing limitations**: Mixed-length batches fail when any sequence exceeds limit
- **Production constraints**: Real-world inputs may require longer sequences than anticipated
- **Model flexibility**: Reduces applicability to diverse use cases

### Performance Impact
- **Memory waste**: Fixed allocation for maximum length regardless of usage
- **Cache efficiency**: Large fixed tables may impact cache performance
- **Initialization overhead**: Computing large sin/cos tables upfront

## Proposed Solution

### Dynamic Growth Implementation

Implement intelligent dynamic growth that expands rotary embeddings on-demand:

```rust
pub struct RotaryEmbedding {
    cos: Arc<Mutex<Tensor>>,
    sin: Arc<Mutex<Tensor>>,
    dim: usize,
    base: f64,
    current_max_len: Arc<Mutex<usize>>,
    device: Device,
    growth_strategy: GrowthStrategy,
}

#[derive(Debug, Clone)]
pub enum GrowthStrategy {
    /// Double the size when growth is needed
    Exponential,
    /// Grow by a fixed increment
    Linear { increment: usize },
    /// Grow to exact required size
    Exact,
    /// Grow with a safety margin
    WithMargin { margin_factor: f32 },
}

impl RotaryEmbedding {
    pub fn new(dim: usize, base: f64, initial_max_len: usize, device: &Device, growth_strategy: GrowthStrategy) -> Result<Self> {
        let (cos, sin) = Self::compute_sincos_tables(dim, base, initial_max_len, device)?;

        Ok(Self {
            cos: Arc::new(Mutex::new(cos)),
            sin: Arc::new(Mutex::new(sin)),
            dim,
            base,
            current_max_len: Arc::new(Mutex::new(initial_max_len)),
            device: device.clone(),
            growth_strategy,
        })
    }

    pub fn apply(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        let seq_len = if x.dims().len() == 4 {
            x.dims4()?.2 // [B, H, T, D]
        } else {
            x.dims3()?.1 // [B, T, D]
        };

        let required_len = position + seq_len;

        // Check if we need to grow the tables
        {
            let current_max = *self.current_max_len.lock().unwrap();
            if required_len > current_max {
                self.grow_tables(required_len)?;
            }
        }

        // Apply rotary embeddings with grown tables
        self.apply_with_tables(x, position, seq_len)
    }

    fn grow_tables(&self, required_len: usize) -> Result<()> {
        let mut cos_guard = self.cos.lock().unwrap();
        let mut sin_guard = self.sin.lock().unwrap();
        let mut current_max_guard = self.current_max_len.lock().unwrap();

        // Double-check after acquiring locks
        if required_len <= *current_max_guard {
            return Ok(());
        }

        let new_max_len = self.calculate_new_max_len(required_len, *current_max_guard);

        log::debug!(
            "Growing rotary embedding tables from {} to {} (required: {})",
            *current_max_guard,
            new_max_len,
            required_len
        );

        let (new_cos, new_sin) = Self::compute_sincos_tables(self.dim, self.base, new_max_len, &self.device)?;

        *cos_guard = new_cos;
        *sin_guard = new_sin;
        *current_max_guard = new_max_len;

        Ok(())
    }

    fn calculate_new_max_len(&self, required_len: usize, current_max: usize) -> usize {
        match &self.growth_strategy {
            GrowthStrategy::Exponential => {
                let mut new_len = current_max;
                while new_len < required_len {
                    new_len *= 2;
                }
                new_len
            }
            GrowthStrategy::Linear { increment } => {
                ((required_len + increment - 1) / increment) * increment
            }
            GrowthStrategy::Exact => required_len,
            GrowthStrategy::WithMargin { margin_factor } => {
                (required_len as f32 * (1.0 + margin_factor)).ceil() as usize
            }
        }
    }

    fn apply_with_tables(&self, x: &Tensor, position: usize, seq_len: usize) -> Result<Tensor> {
        let cos_guard = self.cos.lock().unwrap();
        let sin_guard = self.sin.lock().unwrap();

        if x.dims().len() == 4 {
            let (batch, n_heads, _seq_len, head_dim) = x.dims4()?;
            let half_dim = head_dim / 2;

            // Reshape to separate real and imaginary parts
            let x_reshaped = x.reshape(&[batch, n_heads, seq_len, half_dim, 2])?;
            let x0 = x_reshaped.narrow(4, 0, 1)?.squeeze(4)?;
            let x1 = x_reshaped.narrow(4, 1, 1)?.squeeze(4)?;

            // Get cos/sin for the required range
            let cos = cos_guard.narrow(0, position, seq_len)?
                .unsqueeze(0)?
                .unsqueeze(1)?
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;
            let sin = sin_guard.narrow(0, position, seq_len)?
                .unsqueeze(0)?
                .unsqueeze(1)?
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;

            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            let rotated = Tensor::stack(&[x0_rot, x1_rot], 4)?
                .reshape(&[batch, n_heads, seq_len, head_dim])?;

            Ok(rotated)
        } else {
            // 3D implementation
            let (_batch, _seq, dim) = x.dims3()?;
            let half_dim = dim / 2;

            let x_reshaped = x.reshape(&[x.dims()[0], x.dims()[1], half_dim, 2])?;
            let x0 = x_reshaped.narrow(3, 0, 1)?.squeeze(3)?;
            let x1 = x_reshaped.narrow(3, 1, 1)?.squeeze(3)?;

            let cos = cos_guard.narrow(0, position, seq_len)?;
            let sin = sin_guard.narrow(0, position, seq_len)?;

            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            let rotated = Tensor::stack(&[x0_rot, x1_rot], 3)?
                .reshape(&[x.dims()[0], x.dims()[1], dim])?;

            Ok(rotated)
        }
    }

    fn compute_sincos_tables(dim: usize, base: f64, max_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let half_dim = dim / 2;
        let inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f64 / dim as f64))
            .collect();

        let mut cos_vals = Vec::with_capacity(max_len * half_dim);
        let mut sin_vals = Vec::with_capacity(max_len * half_dim);

        for pos in 0..max_len {
            for &freq in &inv_freq {
                let angle = pos as f64 * freq;
                cos_vals.push(angle.cos() as f32);
                sin_vals.push(angle.sin() as f32);
            }
        }

        let cos = Tensor::from_vec(cos_vals, &[max_len, half_dim], device)?;
        let sin = Tensor::from_vec(sin_vals, &[max_len, half_dim], device)?;

        Ok((cos, sin))
    }
}
```

### Configuration Integration

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct RotaryEmbeddingConfig {
    pub initial_max_len: usize,
    pub growth_strategy: GrowthStrategy,
    pub growth_threshold: f32, // Trigger growth when utilization exceeds this
    pub max_absolute_len: Option<usize>, // Hard limit to prevent runaway growth
}

impl Default for RotaryEmbeddingConfig {
    fn default() -> Self {
        Self {
            initial_max_len: 2048,
            growth_strategy: GrowthStrategy::WithMargin { margin_factor: 0.25 },
            growth_threshold: 0.8,
            max_absolute_len: Some(32768), // Reasonable production limit
        }
    }
}
```

## Implementation Plan

### Phase 1: Core Dynamic Growth (2-3 days)
- [ ] Implement `GrowthStrategy` enum and calculation logic
- [ ] Add thread-safe dynamic table growth mechanism
- [ ] Implement bounds checking and automatic expansion
- [ ] Add configuration system for growth parameters

### Phase 2: Performance Optimization (1-2 days)
- [ ] Optimize sin/cos table computation for large sizes
- [ ] Add memory-efficient incremental growth
- [ ] Implement lazy growth with utilization monitoring
- [ ] Add device-aware memory management for GPU tensors

### Phase 3: Integration & Safety (1-2 days)
- [ ] Integrate with transformer model initialization
- [ ] Add safety limits to prevent excessive memory usage
- [ ] Implement graceful degradation for memory pressure
- [ ] Add monitoring and metrics for growth events

### Phase 4: Testing & Validation (1-2 days)
- [ ] Add comprehensive unit tests for growth scenarios
- [ ] Validate numerical accuracy after growth
- [ ] Performance benchmarking vs fixed-size tables
- [ ] Memory usage profiling and optimization

## Testing Strategy

### Dynamic Growth Testing
```rust
#[test]
fn test_rotary_embedding_growth() {
    let device = Device::Cpu;
    let rotary = RotaryEmbedding::new(
        128, 10000.0, 64, &device,
        GrowthStrategy::Exponential
    ).unwrap();

    // Test normal operation within initial bounds
    let x_small = Tensor::randn(&[1, 4, 32, 128], &device).unwrap();
    let result = rotary.apply(&x_small, 0).unwrap();
    assert_eq!(result.dims(), x_small.dims());

    // Test growth trigger
    let x_large = Tensor::randn(&[1, 4, 200, 128], &device).unwrap();
    let result = rotary.apply(&x_large, 0).unwrap();
    assert_eq!(result.dims(), x_large.dims());

    // Verify tables have grown
    assert!(*rotary.current_max_len.lock().unwrap() >= 200);
}

#[test]
fn test_growth_strategies() {
    let test_cases = vec![
        (GrowthStrategy::Exponential, 100, 64, 128),
        (GrowthStrategy::Linear { increment: 50 }, 100, 64, 100),
        (GrowthStrategy::Exact, 100, 64, 100),
        (GrowthStrategy::WithMargin { margin_factor: 0.5 }, 100, 64, 150),
    ];

    for (strategy, required, current, expected) in test_cases {
        let device = Device::Cpu;
        let rotary = RotaryEmbedding::new(64, 10000.0, current, &device, strategy).unwrap();
        let new_max = rotary.calculate_new_max_len(required, current);
        assert_eq!(new_max, expected, "Strategy: {:?}", rotary.growth_strategy);
    }
}
```

### Numerical Accuracy Testing
```rust
#[test]
fn test_growth_preserves_accuracy() {
    let device = Device::Cpu;
    let rotary_fixed = RotaryEmbedding::new(64, 10000.0, 1000, &device, GrowthStrategy::Exact).unwrap();
    let rotary_dynamic = RotaryEmbedding::new(64, 10000.0, 100, &device, GrowthStrategy::Exponential).unwrap();

    let x = Tensor::randn(&[1, 4, 500, 64], &device).unwrap();

    let result_fixed = rotary_fixed.apply(&x, 0).unwrap();
    let result_dynamic = rotary_dynamic.apply(&x, 0).unwrap();

    // Should produce identical results
    let diff = (result_fixed - result_dynamic).unwrap().abs().unwrap().max(0).unwrap();
    let max_diff: f32 = diff.to_scalar().unwrap();
    assert!(max_diff < 1e-6, "Numerical accuracy degraded after growth: {}", max_diff);
}
```

### Performance Benchmarking
```rust
#[test]
fn benchmark_growth_performance() {
    let device = Device::Cpu;
    let rotary = RotaryEmbedding::new(128, 10000.0, 64, &device, GrowthStrategy::Exponential).unwrap();

    // Measure growth time
    let start = Instant::now();
    let x = Tensor::randn(&[1, 8, 1000, 128], &device).unwrap();
    let _result = rotary.apply(&x, 0).unwrap();
    let growth_time = start.elapsed();

    // Growth should complete quickly
    assert!(growth_time < Duration::from_millis(100));

    // Subsequent operations should be fast
    let start = Instant::now();
    let _result = rotary.apply(&x, 0).unwrap();
    let apply_time = start.elapsed();

    assert!(apply_time < Duration::from_millis(10));
}
```

## Risk Assessment

### Implementation Risks
- **Thread safety**: Concurrent access to growing tables requires careful synchronization
- **Memory usage**: Uncontrolled growth could lead to excessive memory consumption
- **Performance impact**: Growth operations may cause temporary latency spikes

### Mitigation Strategies
- Use efficient locking strategies with minimal contention
- Implement safety limits and monitoring for memory usage
- Add configurable growth policies with conservative defaults
- Provide fallback mechanisms for memory-constrained environments

## Success Criteria

### Functionality Improvements
- [ ] Support for arbitrary sequence lengths without pre-configuration
- [ ] Graceful handling of variable-length inputs in production
- [ ] Memory-efficient operation for typical use cases
- [ ] Thread-safe concurrent access to dynamic tables

### Performance Targets
- [ ] Growth operation completion < 100ms for typical expansions
- [ ] Memory overhead < 20% compared to optimal fixed-size allocation
- [ ] No performance regression for sequences within initial bounds
- [ ] Concurrent access performance equivalent to fixed tables

## Related Issues

- **Memory Management**: Integration with GPU memory allocation systems
- **Performance Optimization**: SIMD/CUDA acceleration for sin/cos computation
- **Configuration System**: Runtime parameter adjustment capabilities

## References

- Rotary Position Embedding (RoPE) paper and implementation details
- Thread-safe tensor operations in production environments
- Memory management best practices for dynamic data structures
- Performance optimization for trigonometric function computation

---

**Priority**: Medium-High
**Estimated Effort**: 4-6 developer days
**Components**: bitnet-models, bitnet-kernels
**Feature Flags**: `cpu`, `gpu`