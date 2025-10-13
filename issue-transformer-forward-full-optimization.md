# [Performance] Transformer Forward Pass Optimization for Full Sequence Processing

## Problem Description

The `TransformerModel::forward_full` function processes tokens step-by-step with KV cache updates, which is inefficient for full sequence processing. This approach prevents vectorization and optimal memory usage for non-autoregressive inference scenarios.

## Environment

- **Affected Crates**: `bitnet-models`
- **Primary Files**: `crates/bitnet-models/src/transformer.rs`
- **Build Configuration**: `--no-default-features --features cpu,inference`
- **Performance Impact**: 2-3x slower than optimal batch processing

## Root Cause Analysis

### Current Step-by-Step Processing

```rust
// Current: Inefficient token-by-token processing
for t in 0..seq_len {
    let step_hidden = hidden.narrow(1, t, 1)?;
    let step_hidden = self.forward(&step_hidden, Some(&mut kv_cache))?;
    let step_logits = self.logits(&step_hidden)?;
    logits_steps.push(step_logits);
}
```

### Performance Bottlenecks

1. **Sequential Processing**: No parallel computation opportunities
2. **Cache Overhead**: Unnecessary KV cache updates for full sequences
3. **Memory Fragmentation**: Multiple small tensor allocations
4. **Suboptimal Vectorization**: Limited SIMD utilization

## Impact Assessment

- **Severity**: High - Significantly impacts inference performance
- **Performance Impact**: 2-3x slower than batch processing
- **Memory Usage**: Higher due to fragmented allocations
- **Scalability**: Poor scaling with sequence length

## Proposed Solution

### Optimized Full Sequence Processing

```rust
pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len) = token_ids.dims2()?;

    // Embed entire sequence efficiently
    let flat_ids = token_ids.flatten_all()?;
    let ids_vec: Vec<u32> = flat_ids.to_vec1()?;
    let mut hidden = self.embed(&ids_vec)?;
    hidden = hidden.reshape(&[batch_size, seq_len, self.config.model.hidden_size])?;

    // Process all layers with causal masking
    for layer in &self.layers {
        hidden = layer.forward_full(&hidden)?;
    }

    // Project to vocabulary logits
    self.logits(&hidden)
}
```

## Implementation Plan

### Phase 1: Batch Processing Implementation (Week 1-2)
- [ ] Implement efficient full-sequence layer processing
- [ ] Add causal attention mask support
- [ ] Optimize tensor memory allocation patterns
- [ ] Create comprehensive performance benchmarks

### Phase 2: Memory Optimization (Week 2-3)
- [ ] Implement in-place tensor operations where possible
- [ ] Add memory pooling for intermediate tensors
- [ ] Optimize gradient computation for training scenarios
- [ ] Add memory usage profiling and monitoring

## Acceptance Criteria

### Performance Requirements
- [ ] >2x speedup compared to step-by-step processing
- [ ] Linear scaling with sequence length
- [ ] Optimal memory usage patterns
- [ ] Efficient SIMD utilization

### Quality Requirements
- [ ] Numerical accuracy maintained vs step-by-step approach
- [ ] Comprehensive testing across different sequence lengths
- [ ] Cross-validation with reference implementations
- [ ] Memory leak prevention verified

## Related Issues

- BitNet.rs #251: Production-ready inference server
- BitNet.rs #218: Device-aware quantization system
