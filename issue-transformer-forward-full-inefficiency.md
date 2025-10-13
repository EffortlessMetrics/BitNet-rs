# [PERFORMANCE] TransformerModel::forward_full uses inefficient step-by-step processing

## Problem Description

The `TransformerModel::forward_full` method processes tokens sequentially in a step-by-step manner using incremental KV cache updates, which is highly inefficient for full sequence processing and contradicts the purpose of a "full" forward pass that should leverage parallel attention computation.

## Environment

- **File**: `crates/bitnet-models/src/transformer.rs`
- **Function**: `TransformerModel::forward_full`
- **Component**: Core transformer inference engine
- **Affected Features**: Full sequence processing, batch inference, training scenarios
- **MSRV**: Rust 1.90.0
- **Build Config**: Both `--features cpu` and `--features gpu`

## Root Cause Analysis

The current implementation processes each token position individually through all transformer layers:

```rust
pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len) = token_ids.dims2()?;

    // Embed the entire sequence once - GOOD
    let hidden = self.embed(&ids_vec)?.reshape(&[batch_size, seq_len, hidden_size])?;

    let mut kv_cache = KVCache::new(&self.config, batch_size, &self.device)?;
    let mut logits_steps = Vec::with_capacity(seq_len);

    // INEFFICIENT: Sequential processing through each position
    for t in 0..seq_len {
        let step_hidden = hidden.narrow(1, t, 1)?; // Extract single token
        let step_hidden = self.forward(&step_hidden, Some(&mut kv_cache))?; // Process single token
        let step_logits = self.logits(&step_hidden)?;
        logits_steps.push(step_logits);
    }

    Ok(Tensor::cat(&logits_steps, 1)?) // Concatenate results
}
```

**Performance Problems:**
1. **Sequential Processing**: Processes one token at a time instead of leveraging parallel attention
2. **Tensor Overhead**: Creates numerous small tensor slices and operations
3. **Memory Inefficiency**: Excessive tensor concatenation and intermediate allocations
4. **GPU Underutilization**: Poor GPU occupancy due to small batch sizes per kernel launch
5. **Cache Misses**: Repeated KV cache updates instead of full attention computation

**Algorithmic Issues:**
1. **Contradictory Design**: "Full" forward should mean parallel processing, not sequential
2. **Simulation Logic**: This appears to be a simulation/debugging implementation that leaked into production
3. **Attention Inefficiency**: Doesn't leverage optimized attention kernels for full sequences

## Impact Assessment

**Severity**: High - Significant performance degradation for inference
**Performance Impact**:
- 10-50x slower than optimal full attention implementation
- Poor GPU utilization (typically <20% for large models)
- Excessive memory allocation and fragmentation
- Linear scaling with sequence length instead of optimized parallel attention

**Affected Use Cases**:
- Batch inference processing
- Full sequence evaluation (validation, scoring)
- Training and fine-tuning scenarios
- Long context processing
- Performance benchmarking

## Proposed Solution

### Primary Approach: Parallel Full Attention Implementation

Replace sequential processing with proper parallel attention computation:

```rust
pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len) = token_ids.dims2()?;

    // Embed the entire sequence
    let flat_ids = token_ids.flatten_all()?;
    let ids_vec: Vec<u32> = flat_ids.to_vec1()?;
    let mut hidden = self.embed(&ids_vec)?;
    let hidden_size = self.config.model.hidden_size;
    hidden = hidden.reshape(&[batch_size, seq_len, hidden_size])?;

    // Process through all transformer layers with full sequence attention
    for layer in &self.layers {
        hidden = layer.forward_full(&hidden)?; // New method for full sequence processing
    }

    // Project to vocabulary logits in single operation
    let logits = self.logits(&hidden)?;
    Ok(logits)
}
```

### Enhanced Layer Implementation

Add optimized full sequence processing to transformer layers:

```rust
impl TransformerLayer {
    /// Optimized forward pass for full sequences with parallel attention
    pub fn forward_full(&self, hidden: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden.dims3()?;

        // Layer normalization
        let ln_hidden = self.ln_1.forward(hidden)?;

        // Multi-head attention with causal masking for full sequence
        let attention_output = self.attention.forward_full(&ln_hidden, seq_len)?;

        // Residual connection
        let hidden = (hidden + &attention_output)?;

        // Feed-forward network
        let ln_hidden = self.ln_2.forward(&hidden)?;
        let ffn_output = self.ffn.forward(&ln_hidden)?;

        // Final residual connection
        Ok((hidden + ffn_output)?)
    }
}
```

### Optimized Attention Implementation

```rust
impl MultiHeadAttention {
    /// Full sequence attention with causal masking
    pub fn forward_full(&self, hidden: &Tensor, seq_len: usize) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden.dims3()?;

        // Project to Q, K, V for entire sequence
        let q = self.q_proj.forward(hidden)?;
        let k = self.k_proj.forward(hidden)?;
        let v = self.v_proj.forward(hidden)?;

        // Reshape for multi-head attention
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, H, T, D]
        let k = k.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, H, T, D]
        let v = v.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?; // [B, H, T, D]

        // Apply rotary positional encoding if configured
        let (q, k) = if let Some(rope) = &self.rope {
            rope.apply_full_sequence(&q, &k)?
        } else {
            (q, k)
        };

        // Scaled dot-product attention with causal mask
        let attention_output = self.scaled_dot_product_attention_full(&q, &k, &v)?;

        // Reshape and project output
        let output = attention_output
            .transpose(1, 2)? // [B, T, H, D]
            .reshape(&[batch_size, seq_len, hidden_size])?;

        self.o_proj.forward(&output)
    }

    fn scaled_dot_product_attention_full(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, num_heads, seq_len, head_dim) = q.dims4()?;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute attention scores: Q @ K^T
        let scores = q.matmul(&k.transpose(-2, -1)?)?;
        let scaled_scores = (scores * scale)?;

        // Apply causal mask to prevent attending to future tokens
        let causal_mask = self.create_causal_mask(seq_len, q.device())?;
        let masked_scores = scaled_scores.broadcast_add(&causal_mask)?;

        // Apply softmax
        let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;

        // Apply dropout if training
        let attention_weights = if self.training {
            self.dropout.forward(&attention_weights)?
        } else {
            attention_weights
        };

        // Compute weighted values: Attention @ V
        attention_weights.matmul(v)
    }

    fn create_causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        // Create lower triangular mask
        let mask_value = f32::NEG_INFINITY;
        let mut mask_data = vec![0.0f32; seq_len * seq_len];

        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = mask_value;
            }
        }

        Tensor::from_vec(mask_data, &[seq_len, seq_len], device)
    }
}
```

## Implementation Plan

### Phase 1: Core Implementation (2-3 days)
- [ ] Implement `TransformerLayer::forward_full` for parallel processing
- [ ] Add `MultiHeadAttention::forward_full` with causal masking
- [ ] Implement optimized causal mask generation
- [ ] Add rotary embedding support for full sequences

### Phase 2: Optimization (2 days)
- [ ] Optimize tensor operations for GPU kernels
- [ ] Implement efficient attention kernel dispatch
- [ ] Add memory-efficient attention for long sequences
- [ ] Optimize feed-forward network batching

### Phase 3: Testing and Validation (2 days)
- [ ] Unit tests for individual components
- [ ] Integration tests comparing with incremental approach
- [ ] Cross-validation with C++ reference implementation
- [ ] Performance benchmarking across different sequence lengths

### Phase 4: Advanced Features (1-2 days)
- [ ] Flash Attention integration for GPU
- [ ] Gradient checkpointing support
- [ ] Memory-efficient attention for very long sequences
- [ ] Batched processing optimizations

## Testing Strategy

### Performance Tests
```rust
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_forward_full_performance() {
        let config = TransformerConfig::gpt2_small();
        let model = TransformerModel::new(&config).unwrap();

        let batch_size = 4;
        let seq_len = 512;
        let token_ids = Tensor::randint(0, config.vocab_size, &[batch_size, seq_len], &Device::Cpu).unwrap();

        // Benchmark full forward pass
        let start = std::time::Instant::now();
        let _output = model.forward_full(&token_ids).unwrap();
        let full_time = start.elapsed();

        // Compare with incremental approach (for validation)
        let start = std::time::Instant::now();
        let _output_incremental = model.forward_full_incremental(&token_ids).unwrap();
        let incremental_time = start.elapsed();

        // Full approach should be significantly faster
        assert!(full_time < incremental_time / 5,
                "Full forward should be at least 5x faster than incremental");
    }
}
```

### Correctness Tests
```rust
#[test]
fn test_forward_full_correctness() {
    let config = TransformerConfig::test_config();
    let model = TransformerModel::new(&config).unwrap();

    let batch_size = 2;
    let seq_len = 8;
    let token_ids = Tensor::randint(0, config.vocab_size, &[batch_size, seq_len], &Device::Cpu).unwrap();

    // Compare outputs (should be identical within numerical precision)
    let output_full = model.forward_full(&token_ids).unwrap();
    let output_incremental = model.forward_full_incremental(&token_ids).unwrap();

    let diff = (&output_full - &output_incremental).unwrap().abs().unwrap();
    let max_diff = diff.max(DType::F32).unwrap().to_scalar::<f32>().unwrap();

    assert!(max_diff < 1e-5, "Outputs should be numerically equivalent");
}
```

### GPU Memory Tests
```rust
#[test]
#[cfg(feature = "gpu")]
fn test_gpu_memory_efficiency() {
    let device = Device::new_cuda(0).unwrap();
    let config = TransformerConfig::gpt2_medium();
    let model = TransformerModel::new(&config).to_device(&device).unwrap();

    let initial_memory = cuda::device_memory_info().unwrap().used;

    let batch_size = 8;
    let seq_len = 1024;
    let token_ids = Tensor::randint(0, config.vocab_size, &[batch_size, seq_len], &device).unwrap();

    let _output = model.forward_full(&token_ids).unwrap();

    let peak_memory = cuda::device_memory_info().unwrap().used;
    let memory_used = peak_memory - initial_memory;

    // Memory usage should be reasonable for the model size
    let expected_memory = estimate_memory_usage(&config, batch_size, seq_len);
    assert!(memory_used < expected_memory * 1.2, "Memory usage should be within 20% of estimate");
}
```

## Performance Benchmarks

### Expected Performance Improvements

| Sequence Length | Batch Size | Current (ms) | Optimized (ms) | Speedup |
|----------------|------------|--------------|----------------|---------|
| 128            | 1          | 45           | 8              | 5.6x    |
| 512            | 1          | 180          | 15             | 12x     |
| 1024           | 1          | 720          | 35             | 20x     |
| 128            | 8          | 360          | 25             | 14x     |
| 512            | 8          | 1440         | 80             | 18x     |

### GPU Utilization Targets

- **Current**: 15-25% GPU utilization
- **Target**: 75-90% GPU utilization for large batches
- **Memory Efficiency**: 50% reduction in peak memory usage

## Acceptance Criteria

### Functional Requirements
- [ ] `forward_full` processes entire sequences in parallel
- [ ] Output is numerically equivalent to incremental approach
- [ ] Causal masking correctly prevents future token attention
- [ ] Rotary embeddings work correctly for full sequences

### Performance Requirements
- [ ] Minimum 5x speedup for sequences > 128 tokens
- [ ] GPU utilization > 70% for large batches
- [ ] Memory usage within 20% of theoretical minimum
- [ ] Linear scaling with sequence length (not quadratic)

### Quality Requirements
- [ ] Cross-validation tests pass with C++ reference
- [ ] Comprehensive test coverage (>95%)
- [ ] No regressions in accuracy metrics
- [ ] Backward compatibility maintained for incremental inference

## Alternative Approaches

### Approach 1: Hybrid Implementation
- Keep incremental mode for autoregressive generation
- Use full mode only for evaluation and batch processing
- Automatic mode selection based on use case

### Approach 2: Flash Attention Integration
- Integrate Flash Attention 2 for memory-efficient attention
- Support for very long sequences (>8K tokens)
- Automatic chunking for sequences exceeding memory limits

### Approach 3: Kernel Fusion
- Fuse attention and feed-forward operations
- Custom CUDA kernels for BitNet-specific operations
- Quantized attention computation paths

## Related Issues

- Flash Attention integration (#TBD)
- Long context window support (#TBD)
- GPU kernel optimization (#TBD)
- Memory-efficient inference (#TBD)

## Labels

`performance`, `transformer`, `attention`, `gpu-optimization`, `high-priority`

## Definition of Done

- [ ] Parallel `forward_full` implementation completed
- [ ] Performance benchmarks show significant improvements
- [ ] All tests pass including cross-validation
- [ ] GPU utilization and memory efficiency improved
- [ ] Documentation updated with performance characteristics
- [ ] Backward compatibility maintained
