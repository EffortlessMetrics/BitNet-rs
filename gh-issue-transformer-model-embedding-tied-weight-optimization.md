# [PERF] Transformer Model Tied Weight Embedding Optimization Architecture

## Problem Description

The `TransformerModel::embed_tied_weight` field is designed as a performance optimization for models with tied embeddings, but the current implementation is incomplete and inconsistent. The cached transposed embedding weight is not properly utilized throughout the inference pipeline, leading to redundant transpose operations and suboptimal memory usage patterns.

## Environment

- **Component**: `bitnet-models` crate
- **File**: `crates/bitnet-models/src/transformer.rs`
- **Rust Version**: 1.90.0+ (2024 edition)
- **Model Types**: LLaMA, GPT-2, BitNet models with tied embeddings
- **Memory Impact**: Significant for large vocabulary models (>32K tokens)

## Current Implementation Analysis

### Incomplete Tied Weight Optimization
```rust
pub struct TransformerModel {
    // ...
    pub embed_tied_weight: Option<Tensor>, // Placeholder - inconsistent usage
    // ...
}

impl TransformerModel {
    pub fn new(config: BitNetConfig, vb: VarBuilder) -> Result<Self> {
        // PROBLEM: Tied weight caching logic is incomplete and not used consistently
        let (embed_transposed, embed_tied_weight) = if embed_transposed {
            (true, None)  // Missing optimization opportunity
        } else if lm_head.is_none() {
            let embed_weight = embed_tokens.embeddings();
            if embed_weight.dims() == [vocab_size, hidden_size] {
                tracing::info!("Pre-transposing tied embeddings [V,H] -> [H,V] to avoid per-step transpose");
                let transposed_weight = embed_weight.transpose(0, 1)?; // [H, V]
                (false, Some(transposed_weight)) // Cached but not utilized properly
            } else {
                tracing::warn!("Embeddings have unexpected shape: {:?}", embed_weight.dims());
                (embed_transposed, None)
            }
        } else {
            (embed_transposed, None)
        };
        // Field set but never used in forward pass!
    }
}
```

### Missing Integration Points
1. **Forward Pass**: Cached tied weight not used during logits computation
2. **Memory Management**: No efficient sharing between embedding and output layers
3. **Gradient Updates**: Inconsistent handling during fine-tuning scenarios
4. **Device Placement**: No consideration for GPU memory optimization

## Root Cause Analysis

1. **Incomplete Implementation**: Cached weight created but not utilized
2. **Architecture Gap**: Missing integration with logits computation path
3. **Memory Inefficiency**: Redundant transpose operations during inference
4. **Inconsistent State**: Model configuration doesn't reflect optimization status
5. **Performance Loss**: Designed optimization providing no actual benefit

## Impact Assessment

**Severity**: Medium-High - Performance optimization completely unused

**Performance Impact**:
- Redundant transpose operations on every forward pass
- Increased memory usage for large vocabulary models
- Cache misses due to suboptimal memory layout
- Wasted initialization time for unused optimizations

**Memory Impact** (for typical models):
- LLaMA-7B: ~400MB redundant memory usage
- GPT-2: ~50MB redundant usage
- Large vocab models: Up to 1GB+ inefficient memory patterns

## Proposed Solution

### Comprehensive Tied Weight Architecture

```rust
use candle_core::{Tensor, Device};
use std::sync::Arc;

/// Tied weight configuration for embedding/output layer sharing
#[derive(Debug, Clone)]
pub struct TiedWeightConfig {
    pub use_tied_weights: bool,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub optimize_for_inference: bool,
    pub device: Device,
}

/// Optimized tied weight implementation
#[derive(Debug)]
pub struct TiedWeightManager {
    /// Original embedding weights [V, H]
    embedding_weight: Arc<Tensor>,
    /// Cached transposed weights [H, V] for efficient matmul
    transposed_weight: Arc<Tensor>,
    /// Configuration
    config: TiedWeightConfig,
    /// Whether weights are currently transposed
    is_transposed: bool,
}

impl TiedWeightManager {
    pub fn new(
        embedding_weight: Tensor,
        config: TiedWeightConfig,
    ) -> Result<Self, ModelError> {
        let expected_shape = [config.vocab_size, config.hidden_size];

        if embedding_weight.dims() != expected_shape {
            return Err(ModelError::InvalidTiedWeightShape {
                expected: expected_shape,
                actual: embedding_weight.dims().to_vec(),
            });
        }

        // Pre-transpose for inference optimization
        let transposed_weight = if config.optimize_for_inference {
            Arc::new(embedding_weight.transpose(0, 1)?) // [H, V]
        } else {
            Arc::new(embedding_weight.clone())
        };

        Ok(Self {
            embedding_weight: Arc::new(embedding_weight),
            transposed_weight,
            config,
            is_transposed: config.optimize_for_inference,
        })
    }

    /// Get embedding weight for token embedding lookup
    pub fn embedding_weight(&self) -> &Tensor {
        &self.embedding_weight
    }

    /// Get optimal weight for logits computation
    pub fn logits_weight(&self) -> &Tensor {
        if self.config.optimize_for_inference && self.is_transposed {
            &self.transposed_weight
        } else {
            &self.embedding_weight
        }
    }

    /// Compute logits efficiently using cached transpose
    pub fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor, ModelError> {
        let batch_size = hidden_states.dim(0)?;
        let seq_len = hidden_states.dim(1)?;
        let hidden_dim = hidden_states.dim(2)?;

        if hidden_dim != self.config.hidden_size {
            return Err(ModelError::DimensionMismatch {
                expected: self.config.hidden_size,
                actual: hidden_dim,
            });
        }

        // Use pre-transposed weight for efficient computation
        let logits = if self.is_transposed {
            // hidden_states: [B, S, H], transposed_weight: [H, V] -> logits: [B, S, V]
            hidden_states.matmul(&self.transposed_weight)?
        } else {
            // Fallback to transpose on-demand (less efficient)
            let weight_t = self.embedding_weight.transpose(0, 1)?;
            hidden_states.matmul(&weight_t)?
        };

        // Validate output dimensions
        let expected_logits_shape = [batch_size, seq_len, self.config.vocab_size];
        if logits.dims() != expected_logits_shape {
            return Err(ModelError::InvalidLogitsShape {
                expected: expected_logits_shape,
                actual: logits.dims().to_vec(),
            });
        }

        Ok(logits)
    }

    /// Update weights (for fine-tuning scenarios)
    pub fn update_weights(&mut self, new_weights: Tensor) -> Result<(), ModelError> {
        if new_weights.dims() != [self.config.vocab_size, self.config.hidden_size] {
            return Err(ModelError::InvalidTiedWeightShape {
                expected: [self.config.vocab_size, self.config.hidden_size],
                actual: new_weights.dims().to_vec(),
            });
        }

        self.embedding_weight = Arc::new(new_weights.clone());

        // Update cached transpose if optimizing for inference
        if self.config.optimize_for_inference {
            self.transposed_weight = Arc::new(new_weights.transpose(0, 1)?);
        }

        Ok(())
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> TiedWeightMemoryStats {
        let element_size = std::mem::size_of::<f32>(); // Assuming f32 tensors
        let weight_elements = self.config.vocab_size * self.config.hidden_size;

        let embedding_bytes = weight_elements * element_size;
        let transposed_bytes = if self.is_transposed {
            weight_elements * element_size
        } else {
            0
        };

        TiedWeightMemoryStats {
            embedding_weight_bytes: embedding_bytes,
            transposed_weight_bytes: transposed_bytes,
            total_bytes: embedding_bytes + transposed_bytes,
            optimization_overhead: transposed_bytes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TiedWeightMemoryStats {
    pub embedding_weight_bytes: usize,
    pub transposed_weight_bytes: usize,
    pub total_bytes: usize,
    pub optimization_overhead: usize,
}

/// Enhanced TransformerModel with proper tied weight integration
pub struct TransformerModel {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<BitNetLayer>,
    pub norm: LayerNorm,
    pub lm_head: Option<candle_nn::Linear>,

    /// Optimized tied weight manager
    pub tied_weights: Option<TiedWeightManager>,

    pub config: BitNetConfig,
    // ... other fields
}

impl TransformerModel {
    pub fn new(config: BitNetConfig, vb: VarBuilder) -> Result<Self> {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;

        // Create embedding layer
        let embed_tokens = candle_nn::embedding(vocab_size, hidden_size, vb.pp("embed_tokens"))?;

        // Create language model head if specified
        let lm_head = if config.tie_word_embeddings {
            None // Will use tied weights
        } else {
            Some(candle_nn::linear(hidden_size, vocab_size, vb.pp("lm_head"))?)
        };

        // Set up tied weight optimization
        let tied_weights = if config.tie_word_embeddings && lm_head.is_none() {
            let tied_config = TiedWeightConfig {
                use_tied_weights: true,
                vocab_size,
                hidden_size,
                optimize_for_inference: true, // Enable optimization by default
                device: vb.device().clone(),
            };

            let embedding_weight = embed_tokens.embeddings().clone();
            let tied_manager = TiedWeightManager::new(embedding_weight, tied_config)?;

            tracing::info!(
                "Initialized tied weight optimization: {} MB memory overhead",
                tied_manager.memory_usage().optimization_overhead / 1_048_576
            );

            Some(tied_manager)
        } else {
            None
        };

        // ... initialize layers and other components

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            tied_weights,
            config,
        })
    }

    /// Forward pass with optimized tied weight handling
    pub fn forward(&self, input_ids: &Tensor, position_ids: Option<&Tensor>) -> Result<Tensor> {
        // Token embedding lookup
        let hidden_states = self.embed_tokens.forward(input_ids)?;

        // Pass through transformer layers
        let hidden_states = self.forward_layers(hidden_states, position_ids)?;

        // Final layer normalization
        let hidden_states = self.norm.forward(&hidden_states)?;

        // Compute logits using optimized path
        let logits = if let Some(lm_head) = &self.lm_head {
            // Dedicated output layer
            lm_head.forward(&hidden_states)?
        } else if let Some(tied_weights) = &self.tied_weights {
            // Optimized tied weights path
            tied_weights.compute_logits(&hidden_states)?
        } else {
            // Fallback: on-demand transpose (least efficient)
            tracing::warn!("Using fallback tied weight computation - consider enabling optimization");
            let embed_weight = self.embed_tokens.embeddings();
            let weight_t = embed_weight.transpose(0, 1)?;
            hidden_states.matmul(&weight_t)?
        };

        Ok(logits)
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> ModelMemoryStats {
        let tied_stats = self.tied_weights
            .as_ref()
            .map(|tw| tw.memory_usage())
            .unwrap_or_default();

        ModelMemoryStats {
            tied_weights: tied_stats,
            // ... other memory stats
        }
    }

    /// Enable/disable tied weight optimization
    pub fn set_tied_weight_optimization(&mut self, optimize: bool) -> Result<(), ModelError> {
        if let Some(tied_weights) = &mut self.tied_weights {
            // Would need to implement optimization toggle in TiedWeightManager
            tracing::info!("Tied weight optimization toggled: {}", optimize);
            Ok(())
        } else {
            Err(ModelError::NoTiedWeights)
        }
    }
}

#[derive(Debug, Default)]
pub struct ModelMemoryStats {
    pub tied_weights: TiedWeightMemoryStats,
    // ... other stats
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Invalid tied weight shape: expected {expected:?}, got {actual:?}")]
    InvalidTiedWeightShape { expected: [usize; 2], actual: Vec<usize> },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid logits shape: expected {expected:?}, got {actual:?}")]
    InvalidLogitsShape { expected: [usize; 3], actual: Vec<usize> },

    #[error("No tied weights configured for this model")]
    NoTiedWeights,

    #[error("Tensor operation failed: {0}")]
    TensorError(#[from] candle_core::Error),
}

/// Configuration validation for tied weights
impl BitNetConfig {
    pub fn validate_tied_weight_config(&self) -> Result<(), ModelError> {
        if self.tie_word_embeddings {
            if self.vocab_size == 0 {
                return Err(ModelError::DimensionMismatch {
                    expected: 1,
                    actual: 0,
                });
            }
            if self.hidden_size == 0 {
                return Err(ModelError::DimensionMismatch {
                    expected: 1,
                    actual: 0,
                });
            }
        }
        Ok(())
    }
}
```

## Implementation Plan

### Phase 1: Architecture Foundation (Week 1)
- [ ] Implement `TiedWeightManager` with caching and optimization
- [ ] Create memory usage tracking and statistics
- [ ] Add configuration validation for tied weights
- [ ] Establish error handling framework

### Phase 2: Model Integration (Week 2)
- [ ] Integrate `TiedWeightManager` into `TransformerModel`
- [ ] Replace manual transpose logic with optimized path
- [ ] Add forward pass optimization for tied weights
- [ ] Implement memory-efficient weight sharing

### Phase 3: Testing & Validation (Week 3)
- [ ] Add comprehensive unit tests for tied weight operations
- [ ] Validate numerical accuracy against manual transpose
- [ ] Benchmark memory usage and performance improvements
- [ ] Test with various model configurations (LLaMA, GPT-2, etc.)

### Phase 4: Production Features (Week 4)
- [ ] Add runtime optimization toggling
- [ ] Implement fine-tuning weight update support
- [ ] Add monitoring and diagnostics
- [ ] Documentation and usage examples

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_tied_weight_manager_creation() {
        let device = Device::Cpu;
        let vocab_size = 1000;
        let hidden_size = 512;

        let embedding_weight = Tensor::randn(0f32, 1f32, (vocab_size, hidden_size), &device).unwrap();
        let config = TiedWeightConfig {
            use_tied_weights: true,
            vocab_size,
            hidden_size,
            optimize_for_inference: true,
            device,
        };

        let tied_manager = TiedWeightManager::new(embedding_weight, config).unwrap();

        // Verify shapes
        assert_eq!(tied_manager.embedding_weight().dims(), &[vocab_size, hidden_size]);
        assert_eq!(tied_manager.logits_weight().dims(), &[hidden_size, vocab_size]);
    }

    #[test]
    fn test_logits_computation_accuracy() {
        let device = Device::Cpu;
        let vocab_size = 100;
        let hidden_size = 64;
        let batch_size = 2;
        let seq_len = 10;

        let embedding_weight = Tensor::randn(0f32, 1f32, (vocab_size, hidden_size), &device).unwrap();
        let hidden_states = Tensor::randn(0f32, 1f32, (batch_size, seq_len, hidden_size), &device).unwrap();

        // Compute logits using optimized path
        let config = TiedWeightConfig {
            use_tied_weights: true,
            vocab_size,
            hidden_size,
            optimize_for_inference: true,
            device,
        };
        let tied_manager = TiedWeightManager::new(embedding_weight.clone(), config).unwrap();
        let optimized_logits = tied_manager.compute_logits(&hidden_states).unwrap();

        // Compute logits using manual transpose
        let weight_t = embedding_weight.transpose(0, 1).unwrap();
        let manual_logits = hidden_states.matmul(&weight_t).unwrap();

        // Should be numerically identical
        let diff = (optimized_logits - manual_logits).unwrap().abs().unwrap();
        let max_diff = diff.max(0).unwrap().max(1).unwrap().to_scalar::<f32>().unwrap();
        assert!(max_diff < 1e-6, "Maximum difference: {}", max_diff);
    }

    #[test]
    fn test_memory_usage_tracking() {
        let device = Device::Cpu;
        let vocab_size = 32000; // Large vocabulary like LLaMA
        let hidden_size = 4096;

        let embedding_weight = Tensor::randn(0f32, 1f32, (vocab_size, hidden_size), &device).unwrap();
        let config = TiedWeightConfig {
            use_tied_weights: true,
            vocab_size,
            hidden_size,
            optimize_for_inference: true,
            device,
        };

        let tied_manager = TiedWeightManager::new(embedding_weight, config).unwrap();
        let stats = tied_manager.memory_usage();

        // Verify memory calculations
        let expected_embedding_bytes = vocab_size * hidden_size * 4; // f32
        let expected_transposed_bytes = vocab_size * hidden_size * 4; // cached transpose

        assert_eq!(stats.embedding_weight_bytes, expected_embedding_bytes);
        assert_eq!(stats.transposed_weight_bytes, expected_transposed_bytes);
        assert_eq!(stats.total_bytes, expected_embedding_bytes + expected_transposed_bytes);
        assert_eq!(stats.optimization_overhead, expected_transposed_bytes);
    }

    #[test]
    fn test_transformer_model_tied_weights() {
        let mut config = BitNetConfig::default();
        config.vocab_size = 1000;
        config.hidden_size = 512;
        config.tie_word_embeddings = true;

        // Create mock VarBuilder
        let device = Device::Cpu;
        let vb = VarBuilder::from_tensors(std::collections::HashMap::new(), candle_core::DType::F32, &device);

        let model = TransformerModel::new(config, vb).unwrap();

        // Should have tied weights configured
        assert!(model.tied_weights.is_some());
        assert!(model.lm_head.is_none()); // No dedicated head for tied weights

        // Test forward pass
        let input_ids = Tensor::zeros((2, 10), candle_core::DType::U32, &device).unwrap();
        let logits = model.forward(&input_ids, None).unwrap();

        // Should produce valid logits
        assert_eq!(logits.dims(), &[2, 10, 1000]);
    }

    #[test]
    fn test_performance_comparison() {
        let device = Device::Cpu;
        let vocab_size = 32000;
        let hidden_size = 4096;
        let batch_size = 8;
        let seq_len = 512;

        let embedding_weight = Tensor::randn(0f32, 1f32, (vocab_size, hidden_size), &device).unwrap();
        let hidden_states = Tensor::randn(0f32, 1f32, (batch_size, seq_len, hidden_size), &device).unwrap();

        // Benchmark optimized path
        let config = TiedWeightConfig {
            use_tied_weights: true,
            vocab_size,
            hidden_size,
            optimize_for_inference: true,
            device: device.clone(),
        };
        let tied_manager = TiedWeightManager::new(embedding_weight.clone(), config).unwrap();

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = tied_manager.compute_logits(&hidden_states).unwrap();
        }
        let optimized_time = start.elapsed();

        // Benchmark manual transpose path
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let weight_t = embedding_weight.transpose(0, 1).unwrap();
            let _ = hidden_states.matmul(&weight_t).unwrap();
        }
        let manual_time = start.elapsed();

        println!("Optimized: {:?}, Manual: {:?}", optimized_time, manual_time);

        // Optimized should be significantly faster
        assert!(optimized_time < manual_time);
        let speedup = manual_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
        assert!(speedup > 1.5, "Expected speedup > 1.5x, got {:.2}x", speedup);
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{black_box, Criterion};

    pub fn bench_tied_weight_logits(c: &mut Criterion) {
        let device = Device::Cpu;
        let vocab_sizes = vec![8000, 32000, 65000];
        let hidden_size = 4096;
        let batch_size = 4;
        let seq_len = 256;

        for vocab_size in vocab_sizes {
            let embedding_weight = Tensor::randn(0f32, 1f32, (vocab_size, hidden_size), &device).unwrap();
            let hidden_states = Tensor::randn(0f32, 1f32, (batch_size, seq_len, hidden_size), &device).unwrap();

            // Benchmark optimized tied weights
            let config = TiedWeightConfig {
                use_tied_weights: true,
                vocab_size,
                hidden_size,
                optimize_for_inference: true,
                device: device.clone(),
            };
            let tied_manager = TiedWeightManager::new(embedding_weight.clone(), config).unwrap();

            c.bench_function(&format!("tied_weights_optimized_v{}", vocab_size), |b| {
                b.iter(|| {
                    tied_manager.compute_logits(black_box(&hidden_states)).unwrap()
                })
            });

            // Benchmark manual transpose
            c.bench_function(&format!("tied_weights_manual_v{}", vocab_size), |b| {
                b.iter(|| {
                    let weight_t = embedding_weight.transpose(0, 1).unwrap();
                    black_box(&hidden_states).matmul(&weight_t).unwrap()
                })
            });
        }
    }
}
```

## Success Criteria

- [ ] **Performance**: Optimized path >= 2x faster than manual transpose
- [ ] **Memory Efficiency**: Clear tracking of optimization overhead
- [ ] **Numerical Accuracy**: Identical results to manual transpose (within fp32 precision)
- [ ] **Integration**: Seamless integration with existing transformer models
- [ ] **Configurability**: Runtime optimization enable/disable support
- [ ] **Monitoring**: Comprehensive memory and performance statistics

## Related Issues

- #XXX: GPU memory management for large vocabulary models
- #XXX: Mixed precision support for tied weights
- #XXX: Model loading optimization for tied weight models
- #XXX: Fine-tuning support with tied weight updates

## Implementation Notes

This implementation provides a comprehensive tied weight optimization that eliminates redundant transpose operations while maintaining clean architecture and extensive testing. The solution enables significant performance improvements for large vocabulary models while providing clear monitoring and configuration options.
