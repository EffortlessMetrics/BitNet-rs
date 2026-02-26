# [ARCH] Rotary Embedding Dynamic Growth for Long Sequence Support

## Problem Description

The `RotaryEmbedding::apply` implementation lacks dynamic growth capability, limiting BitNet-rs to sequences shorter than the pre-allocated `max_seq_len`. This architectural limitation prevents the model from processing long documents, conversations, or variable-length inputs that exceed the initial sequence length constraints.

## Environment

- **Component**: `bitnet-models` crate
- **File**: `crates/bitnet-models/src/transformer.rs`
- **Rust Version**: 1.90.0+ (2024 edition)
- **Model Impact**: All transformer models using rotary positional embeddings
- **Sequence Limits**: Currently hard-capped at initialization `max_seq_len`

## Current Implementation Analysis

### Fixed Sequence Length Limitation
```rust
impl RotaryEmbedding {
    pub fn apply(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        if x.dims().len() == 4 {
            let (batch, n_heads, seq_len, head_dim) = x.dims4()?;

            // PROBLEM: No dynamic growth when seq_len > self.max_seq_len
            let cos = self.cos.narrow(0, position, seq_len)?  // Fails for long sequences
                .unsqueeze(0)?
                .unsqueeze(1)?
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;

            // Similar issue with sin cache
            let sin = self.sin.narrow(0, position, seq_len)?  // Hard limit
                .unsqueeze(0)?
                .unsqueeze(1)?
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;
        }
        // No error handling or growth mechanism
    }
}
```

### Missing Architecture Components
- No cache expansion mechanism for cos/sin tables
- No memory reallocation strategy for longer sequences
- No efficient incremental growth algorithm
- No configuration for growth policies

## Root Cause Analysis

1. **Static Allocation**: Cos/sin caches allocated once at initialization
2. **Hard Limits**: No mechanism to exceed `max_seq_len` constraint
3. **Memory Management**: No strategy for dynamic tensor resizing
4. **Performance Impact**: Complete failure rather than graceful extension
5. **Architectural Gap**: Missing long-sequence support in design

## Impact Assessment

**Severity**: High - Limits model usability for long-sequence scenarios

**Affected Use Cases**:
- Long document processing (>2K tokens)
- Extended conversations and chatbots
- Code analysis with long contexts
- Scientific paper analysis
- Real-time inference with accumulating context

**User Impact**:
- Hard failures on longer inputs
- Need to chunk inputs artificially
- Loss of long-range attention benefits
- Reduced model effectiveness

## Proposed Solution

### Dynamic Growth Architecture with Efficient Caching

```rust
use candle_core::{Tensor, Device};
use std::sync::{Arc, RwLock};

/// Configuration for rotary embedding growth behavior
#[derive(Debug, Clone)]
pub struct RotaryGrowthConfig {
    /// Initial maximum sequence length
    pub initial_max_seq_len: usize,
    /// Growth strategy when exceeding limits
    pub growth_strategy: GrowthStrategy,
    /// Maximum allowed sequence length (safety limit)
    pub absolute_max_seq_len: usize,
    /// Growth increment size
    pub growth_increment: usize,
    /// Whether to enable dynamic growth
    pub enable_dynamic_growth: bool,
}

#[derive(Debug, Clone)]
pub enum GrowthStrategy {
    /// Double the cache size when limit exceeded
    Double,
    /// Add fixed increment to cache size
    Increment(usize),
    /// Grow to exact required size plus buffer
    Exact { buffer_size: usize },
    /// Custom growth function
    Custom(fn(current_size: usize, required_size: usize) -> usize),
}

impl Default for RotaryGrowthConfig {
    fn default() -> Self {
        Self {
            initial_max_seq_len: 2048,
            growth_strategy: GrowthStrategy::Double,
            absolute_max_seq_len: 32768, // 32K tokens max
            growth_increment: 1024,
            enable_dynamic_growth: true,
        }
    }
}

/// Thread-safe rotary embedding cache with dynamic growth
#[derive(Debug)]
pub struct DynamicRotaryCache {
    /// Cosine cache [max_seq_len, dim/2]
    cos: Arc<RwLock<Tensor>>,
    /// Sine cache [max_seq_len, dim/2]
    sin: Arc<RwLock<Tensor>>,
    /// Current maximum sequence length
    current_max_seq_len: Arc<RwLock<usize>>,
    /// Growth configuration
    config: RotaryGrowthConfig,
    /// Model dimension
    dim: usize,
    /// Rotation base frequency
    base: f32,
    /// Target device for tensor operations
    device: Device,
}

impl DynamicRotaryCache {
    pub fn new(
        dim: usize,
        config: RotaryGrowthConfig,
        base: f32,
        device: Device,
    ) -> Result<Self, ModelError> {
        let half_dim = dim / 2;

        // Initialize with configured starting size
        let (cos, sin) = Self::compute_cos_sin_tables(
            config.initial_max_seq_len,
            half_dim,
            base,
            &device,
        )?;

        Ok(Self {
            cos: Arc::new(RwLock::new(cos)),
            sin: Arc::new(RwLock::new(sin)),
            current_max_seq_len: Arc::new(RwLock::new(config.initial_max_seq_len)),
            config,
            dim,
            base,
            device,
        })
    }

    /// Get cos/sin tensors, growing cache if necessary
    pub fn get_cos_sin(&self, required_seq_len: usize) -> Result<(Tensor, Tensor), ModelError> {
        let current_max = *self.current_max_seq_len.read().unwrap();

        // Check if growth is needed
        if required_seq_len > current_max {
            if !self.config.enable_dynamic_growth {
                return Err(ModelError::SequenceTooLong {
                    required: required_seq_len,
                    max_allowed: current_max,
                    suggestion: "Enable dynamic growth or increase initial_max_seq_len".to_string(),
                });
            }

            // Perform dynamic growth
            self.grow_cache(required_seq_len)?;
        }

        // Return cloned tensors (efficient with Arc backing)
        let cos = self.cos.read().unwrap().clone();
        let sin = self.sin.read().unwrap().clone();

        Ok((cos, sin))
    }

    /// Dynamically grow the cos/sin caches
    fn grow_cache(&self, required_seq_len: usize) -> Result<(), ModelError> {
        let current_max = *self.current_max_seq_len.read().unwrap();

        if required_seq_len <= current_max {
            return Ok(); // Already sufficient
        }

        if required_seq_len > self.config.absolute_max_seq_len {
            return Err(ModelError::SequenceExceedsAbsoluteLimit {
                required: required_seq_len,
                absolute_limit: self.config.absolute_max_seq_len,
            });
        }

        // Calculate new cache size based on growth strategy
        let new_max_seq_len = self.calculate_new_cache_size(current_max, required_seq_len);

        tracing::info!(
            "Growing rotary embedding cache: {} -> {} (required: {})",
            current_max,
            new_max_seq_len,
            required_seq_len
        );

        // Compute new cos/sin tables
        let half_dim = self.dim / 2;
        let (new_cos, new_sin) = Self::compute_cos_sin_tables(
            new_max_seq_len,
            half_dim,
            self.base,
            &self.device,
        )?;

        // Update caches atomically
        {
            let mut cos_guard = self.cos.write().unwrap();
            let mut sin_guard = self.sin.write().unwrap();
            let mut max_len_guard = self.current_max_seq_len.write().unwrap();

            *cos_guard = new_cos;
            *sin_guard = new_sin;
            *max_len_guard = new_max_seq_len;
        }

        Ok(())
    }

    /// Calculate new cache size based on growth strategy
    fn calculate_new_cache_size(&self, current_size: usize, required_size: usize) -> usize {
        let candidate_size = match &self.config.growth_strategy {
            GrowthStrategy::Double => {
                let mut new_size = current_size;
                while new_size < required_size {
                    new_size *= 2;
                }
                new_size
            }
            GrowthStrategy::Increment(increment) => {
                let increments_needed = (required_size - current_size + increment - 1) / increment;
                current_size + increments_needed * increment
            }
            GrowthStrategy::Exact { buffer_size } => {
                required_size + buffer_size
            }
            GrowthStrategy::Custom(growth_fn) => {
                growth_fn(current_size, required_size)
            }
        };

        // Respect absolute maximum
        std::cmp::min(candidate_size, self.config.absolute_max_seq_len)
    }

    /// Compute cos/sin lookup tables for given sequence length
    fn compute_cos_sin_tables(
        max_seq_len: usize,
        half_dim: usize,
        base: f32,
        device: &Device,
    ) -> Result<(Tensor, Tensor), ModelError> {
        // Create position indices [0, 1, 2, ..., max_seq_len-1]
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions_tensor = Tensor::from_vec(positions, (max_seq_len, 1), device)?;

        // Create frequency indices and compute inverse frequencies
        let freq_indices: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
        let inv_freq: Vec<f32> = freq_indices
            .iter()
            .map(|&i| 1.0 / base.powf(i / half_dim as f32))
            .collect();
        let inv_freq_tensor = Tensor::from_vec(inv_freq, (1, half_dim), device)?;

        // Compute angles: positions × inv_frequencies
        let angles = positions_tensor.matmul(&inv_freq_tensor)?; // [max_seq_len, half_dim]

        // Compute cos and sin
        let cos = angles.cos()?;
        let sin = angles.sin()?;

        Ok((cos, sin))
    }

    /// Get current cache statistics
    pub fn cache_stats(&self) -> RotaryCacheStats {
        let current_max = *self.current_max_seq_len.read().unwrap();
        let element_count = current_max * (self.dim / 2);
        let memory_usage = element_count * 2 * std::mem::size_of::<f32>(); // cos + sin tables

        RotaryCacheStats {
            current_max_seq_len: current_max,
            initial_max_seq_len: self.config.initial_max_seq_len,
            absolute_max_seq_len: self.config.absolute_max_seq_len,
            memory_usage_bytes: memory_usage,
            growth_enabled: self.config.enable_dynamic_growth,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RotaryCacheStats {
    pub current_max_seq_len: usize,
    pub initial_max_seq_len: usize,
    pub absolute_max_seq_len: usize,
    pub memory_usage_bytes: usize,
    pub growth_enabled: bool,
}

/// Enhanced RotaryEmbedding with dynamic growth support
pub struct RotaryEmbedding {
    /// Dynamic cache with growth capability
    cache: DynamicRotaryCache,
    /// Model dimension
    dim: usize,
}

impl RotaryEmbedding {
    pub fn new(
        dim: usize,
        config: RotaryGrowthConfig,
        base: f32,
        device: Device,
    ) -> Result<Self, ModelError> {
        let cache = DynamicRotaryCache::new(dim, config, base, device)?;

        Ok(Self {
            cache,
            dim,
        })
    }

    /// Apply rotary embeddings with automatic cache growth
    pub fn apply(&self, x: &Tensor, position: usize) -> Result<Tensor, ModelError> {
        if x.dims().len() == 4 {
            let (batch, n_heads, seq_len, head_dim) = x.dims4()?;
            let half_dim = head_dim / 2;

            // Ensure cache can handle the required sequence length
            let required_len = position + seq_len;
            let (cos_cache, sin_cache) = self.cache.get_cos_sin(required_len)?;

            // Reshape to separate real and imaginary parts
            let x_reshaped = x.reshape(&[batch, n_heads, seq_len, half_dim, 2])?;
            let x0 = x_reshaped.narrow(4, 0, 1)?.squeeze(4)?;
            let x1 = x_reshaped.narrow(4, 1, 1)?.squeeze(4)?;

            // Extract cos/sin for the current position and sequence
            let cos = cos_cache
                .narrow(0, position, seq_len)?
                .unsqueeze(0)?  // Add batch dim
                .unsqueeze(1)?  // Add heads dim
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;

            let sin = sin_cache
                .narrow(0, position, seq_len)?
                .unsqueeze(0)?
                .unsqueeze(1)?
                .broadcast_as(&[batch, n_heads, seq_len, half_dim])?;

            // Apply rotary transformation
            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            // Reconstruct tensor
            let rotated = Tensor::stack(&[x0_rot, x1_rot], 4)?
                .reshape(&[batch, n_heads, seq_len, head_dim])?;

            Ok(rotated)
        } else {
            // Handle 3D case with similar growth support
            let (_batch, _seq, dim) = x.dims3()?;
            let half_dim = dim / 2;

            let required_len = position + 1;
            let (cos_cache, sin_cache) = self.cache.get_cos_sin(required_len)?;

            let x_reshaped = x.reshape(&[x.dims()[0], x.dims()[1], half_dim, 2])?;
            let x0 = x_reshaped.narrow(3, 0, 1)?.squeeze(3)?;
            let x1 = x_reshaped.narrow(3, 1, 1)?.squeeze(3)?;

            let cos = cos_cache.narrow(0, position, 1)?;
            let sin = sin_cache.narrow(0, position, 1)?;

            let x0_rot = (x0.mul(&cos)? - x1.mul(&sin)?)?;
            let x1_rot = (x0.mul(&sin)? + x1.mul(&cos)?)?;

            let rotated = Tensor::stack(&[x0_rot, x1_rot], 3)?
                .reshape(&[x.dims()[0], x.dims()[1], dim])?;

            Ok(rotated)
        }
    }

    /// Get cache statistics for monitoring
    pub fn cache_stats(&self) -> RotaryCacheStats {
        self.cache.cache_stats()
    }

    /// Pre-warm cache to specific sequence length
    pub fn prewarm_cache(&self, seq_len: usize) -> Result<(), ModelError> {
        self.cache.get_cos_sin(seq_len)?;
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Sequence length {required} exceeds maximum allowed {max_allowed}: {suggestion}")]
    SequenceTooLong {
        required: usize,
        max_allowed: usize,
        suggestion: String,
    },

    #[error("Sequence length {required} exceeds absolute limit {absolute_limit}")]
    SequenceExceedsAbsoluteLimit {
        required: usize,
        absolute_limit: usize,
    },

    #[error("Tensor operation failed: {0}")]
    TensorError(#[from] candle_core::Error),

    #[error("Invalid rotary embedding dimension: {0}")]
    InvalidDimension(usize),
}
```

## Implementation Plan

### Phase 1: Dynamic Cache Infrastructure (Week 1)
- [ ] Implement `DynamicRotaryCache` with thread-safe growth
- [ ] Add growth strategy configuration and policies
- [ ] Create efficient cos/sin table computation
- [ ] Add memory usage tracking and monitoring

### Phase 2: Integration & Growth Logic (Week 2)
- [ ] Replace static `RotaryEmbedding` with dynamic version
- [ ] Implement automatic growth detection and execution
- [ ] Add pre-warming and cache management APIs
- [ ] Integrate with existing transformer models

### Phase 3: Testing & Optimization (Week 3)
- [ ] Add comprehensive unit tests for growth scenarios
- [ ] Test memory efficiency and performance impact
- [ ] Validate numerical accuracy across cache growths
- [ ] Benchmark growth overhead vs. static allocation

### Phase 4: Production Features (Week 4)
- [ ] Add configuration validation and safety limits
- [ ] Implement monitoring and alerting for growth events
- [ ] Add graceful degradation for memory constraints
- [ ] Documentation and usage examples

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_cache_growth() {
        let config = RotaryGrowthConfig {
            initial_max_seq_len: 128,
            growth_strategy: GrowthStrategy::Double,
            absolute_max_seq_len: 2048,
            enable_dynamic_growth: true,
            ..Default::default()
        };

        let cache = DynamicRotaryCache::new(512, config, 10000.0, Device::Cpu).unwrap();

        // Should start with initial size
        assert_eq!(cache.cache_stats().current_max_seq_len, 128);

        // Request larger sequence - should trigger growth
        let (cos, sin) = cache.get_cos_sin(256).unwrap();
        assert!(cache.cache_stats().current_max_seq_len >= 256);

        // Verify cache contents are valid
        assert_eq!(cos.dims(), &[cache.cache_stats().current_max_seq_len, 256]);
        assert_eq!(sin.dims(), &[cache.cache_stats().current_max_seq_len, 256]);
    }

    #[test]
    fn test_rotary_embedding_long_sequence() {
        let config = RotaryGrowthConfig {
            initial_max_seq_len: 64,
            growth_strategy: GrowthStrategy::Exact { buffer_size: 32 },
            absolute_max_seq_len: 1024,
            enable_dynamic_growth: true,
            ..Default::default()
        };

        let rope = RotaryEmbedding::new(512, config, 10000.0, Device::Cpu).unwrap();

        // Test with sequence longer than initial cache
        let batch_size = 2;
        let n_heads = 8;
        let seq_len = 200;  // Exceeds initial 64
        let head_dim = 64;

        let x = Tensor::randn(0f32, 1f32, (batch_size, n_heads, seq_len, head_dim), &Device::Cpu).unwrap();
        let result = rope.apply(&x, 0).unwrap();

        // Should complete successfully with correct output shape
        assert_eq!(result.dims(), &[batch_size, n_heads, seq_len, head_dim]);

        // Cache should have grown
        assert!(rope.cache_stats().current_max_seq_len >= 200);
    }

    #[test]
    fn test_growth_strategies() {
        let strategies = vec![
            GrowthStrategy::Double,
            GrowthStrategy::Increment(128),
            GrowthStrategy::Exact { buffer_size: 64 },
        ];

        for strategy in strategies {
            let config = RotaryGrowthConfig {
                initial_max_seq_len: 100,
                growth_strategy: strategy,
                absolute_max_seq_len: 1000,
                enable_dynamic_growth: true,
                ..Default::default()
            };

            let cache = DynamicRotaryCache::new(256, config, 10000.0, Device::Cpu).unwrap();

            // Request growth
            cache.get_cos_sin(250).unwrap();

            // Should have grown appropriately
            let stats = cache.cache_stats();
            assert!(stats.current_max_seq_len >= 250);
            assert!(stats.current_max_seq_len <= 1000);
        }
    }

    #[test]
    fn test_absolute_limit_enforcement() {
        let config = RotaryGrowthConfig {
            initial_max_seq_len: 64,
            absolute_max_seq_len: 256,
            enable_dynamic_growth: true,
            ..Default::default()
        };

        let cache = DynamicRotaryCache::new(128, config, 10000.0, Device::Cpu).unwrap();

        // Should succeed within limit
        cache.get_cos_sin(200).unwrap();

        // Should fail beyond absolute limit
        let result = cache.get_cos_sin(500);
        assert!(result.is_err());

        match result.unwrap_err() {
            ModelError::SequenceExceedsAbsoluteLimit { required, absolute_limit } => {
                assert_eq!(required, 500);
                assert_eq!(absolute_limit, 256);
            }
            _ => panic!("Expected SequenceExceedsAbsoluteLimit error"),
        }
    }

    #[test]
    fn test_disabled_growth() {
        let config = RotaryGrowthConfig {
            initial_max_seq_len: 128,
            enable_dynamic_growth: false,
            ..Default::default()
        };

        let cache = DynamicRotaryCache::new(256, config, 10000.0, Device::Cpu).unwrap();

        // Should fail when growth disabled
        let result = cache.get_cos_sin(256);
        assert!(result.is_err());

        match result.unwrap_err() {
            ModelError::SequenceTooLong { required, max_allowed, suggestion } => {
                assert_eq!(required, 256);
                assert_eq!(max_allowed, 128);
                assert!(suggestion.contains("Enable dynamic growth"));
            }
            _ => panic!("Expected SequenceTooLong error"),
        }
    }

    #[test]
    fn test_numerical_accuracy_across_growth() {
        let config = RotaryGrowthConfig {
            initial_max_seq_len: 64,
            enable_dynamic_growth: true,
            ..Default::default()
        };

        let rope = RotaryEmbedding::new(128, config, 10000.0, Device::Cpu).unwrap();

        // Apply to initial sequence
        let x_short = Tensor::randn(0f32, 1f32, (1, 4, 32, 32), &Device::Cpu).unwrap();
        let result_short = rope.apply(&x_short, 0).unwrap();

        // Trigger cache growth
        let x_long = Tensor::randn(0f32, 1f32, (1, 4, 100, 32), &Device::Cpu).unwrap();
        let _result_long = rope.apply(&x_long, 0).unwrap();

        // Apply to short sequence again - should be identical
        let result_short_after_growth = rope.apply(&x_short, 0).unwrap();

        // Results should be numerically identical
        let diff = (result_short - result_short_after_growth).unwrap().abs().unwrap();
        let max_diff = diff.max(0).unwrap().max(1).unwrap().max(2).unwrap().to_scalar::<f32>().unwrap();
        assert!(max_diff < 1e-6, "Maximum difference after growth: {}", max_diff);
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{black_box, Criterion};

    pub fn bench_dynamic_vs_static_rope(c: &mut Criterion) {
        let device = Device::Cpu;

        // Static rope with large initial cache
        let static_rope = RotaryEmbedding::new(
            512,
            RotaryGrowthConfig {
                initial_max_seq_len: 2048,
                enable_dynamic_growth: false,
                ..Default::default()
            },
            10000.0,
            device.clone(),
        ).unwrap();

        // Dynamic rope with small initial cache
        let dynamic_rope = RotaryEmbedding::new(
            512,
            RotaryGrowthConfig {
                initial_max_seq_len: 128,
                enable_dynamic_growth: true,
                ..Default::default()
            },
            10000.0,
            device.clone(),
        ).unwrap();

        let batch_size = 4;
        let n_heads = 8;
        let seq_len = 512;
        let head_dim = 64;

        let x = Tensor::randn(0f32, 1f32, (batch_size, n_heads, seq_len, head_dim), &device).unwrap();

        // Benchmark static rope
        c.bench_function("rope_static", |b| {
            b.iter(|| {
                static_rope.apply(black_box(&x), 0).unwrap()
            })
        });

        // Pre-warm dynamic rope
        dynamic_rope.prewarm_cache(seq_len).unwrap();

        // Benchmark dynamic rope (after warm-up)
        c.bench_function("rope_dynamic_warmed", |b| {
            b.iter(|| {
                dynamic_rope.apply(black_box(&x), 0).unwrap()
            })
        });

        // Benchmark dynamic rope with growth
        let dynamic_rope_cold = RotaryEmbedding::new(
            512,
            RotaryGrowthConfig {
                initial_max_seq_len: 64,
                enable_dynamic_growth: true,
                ..Default::default()
            },
            10000.0,
            device,
        ).unwrap();

        c.bench_function("rope_dynamic_with_growth", |b| {
            b.iter(|| {
                dynamic_rope_cold.apply(black_box(&x), 0).unwrap()
            })
        });
    }
}
```

## Success Criteria

- [ ] **Long Sequence Support**: Handle sequences up to 32K tokens seamlessly
- [ ] **Growth Performance**: < 10ms overhead for cache growth operations
- [ ] **Memory Efficiency**: < 50% memory overhead compared to static allocation
- [ ] **Numerical Accuracy**: Identical results before/after cache growth
- [ ] **Thread Safety**: Concurrent access safe during growth operations
- [ ] **Configurability**: Flexible growth policies for different use cases

## Related Issues

- #XXX: Memory management optimization for long sequences
- #XXX: Attention mechanism scalability improvements
- #XXX: KV-cache dynamic growth coordination
- #XXX: Long sequence inference optimization

## Implementation Notes

This dynamic growth implementation enables BitNet-rs to handle arbitrary sequence lengths while maintaining optimal memory usage and performance characteristics. The thread-safe design allows concurrent inference while supporting automatic cache expansion as needed.
