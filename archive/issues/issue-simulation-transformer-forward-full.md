# [Simulation] TransformerModel::forward_full uses inefficient step-by-step processing instead of parallel batch inference

## Problem Description

The `TransformerModel::forward_full` function in `crates/bitnet-models/src/transformer.rs` processes tokens sequentially in a step-by-step manner using a KV cache, which significantly impacts performance for batch inference and full sequence processing. This approach is more suitable for autoregressive generation than full sequence forward passes.

## Environment

- **File**: `crates/bitnet-models/src/transformer.rs`
- **Function**: `TransformerModel::forward_full`
- **Feature Flags**: Affects both `cpu` and `gpu` inference modes
- **Crate**: `bitnet-models`

## Current Implementation Analysis

The current implementation processes each token position individually:

```rust
pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
    // Token ids expected shape: [B,T]
    let (batch_size, seq_len) = token_ids.dims2()?;

    // Embed the entire sequence once.
    let flat_ids = token_ids.flatten_all()?;
    let ids_vec: Vec<u32> = flat_ids.to_vec1()?;
    let hidden = self.embed(&ids_vec)?;
    let hidden_size = self.config.model.hidden_size;
    let hidden = hidden.reshape(&[batch_size, seq_len, hidden_size])?;

    // Create per-layer KV cache so that rotary/absolute positional
    // encodings use the proper positions during iterative decoding.
    let mut kv_cache = KVCache::new(&self.config, batch_size, &self.device)?;

    // Collect logits for each position.
    let mut logits_steps = Vec::with_capacity(seq_len);
    for t in 0..seq_len {  // Sequential processing bottleneck
        // Select the current token's embedding: [B,1,H]
        let step_hidden = hidden.narrow(1, t, 1)?;

        // Run through all layers using the incremental path which applies
        // positional encoding per layer and causal masking internally.
        let step_hidden = self.forward(&step_hidden, Some(&mut kv_cache))?;

        // Project to vocabulary logits for this step.
        let step_logits = self.logits(&step_hidden)?;
        logits_steps.push(step_logits);
    }

    // Concatenate logits from all steps: [B,T,V]
    Ok(Tensor::cat(&logits_steps, 1)?)
}
```

## Root Cause Analysis

1. **Sequential Processing**: The function processes one token at a time instead of leveraging parallelizable batch operations
2. **Inefficient Memory Access**: Creates and concatenates individual tensor slices instead of processing the full tensor
3. **Suboptimal KV Cache Usage**: Uses incremental KV cache updates designed for autoregressive generation, not full sequence processing
4. **Missing Vectorization**: Doesn't take advantage of SIMD/GPU vectorized operations for batch processing
5. **Performance Degradation**: O(n) sequential operations instead of O(1) parallel operations

## Impact Assessment

**Severity**: High - Performance Critical
**Affected Components**:
- Batch inference performance (significant slowdown)
- GPU utilization efficiency (underutilized parallel compute)
- Memory bandwidth efficiency (multiple small tensor operations)
- Training/fine-tuning workflows (full sequence processing)

**Performance Impact**:
- ~10-100x slower than optimal batch processing
- Poor GPU compute utilization
- Increased memory allocation overhead
- Higher latency for batch inference

## Proposed Solution

### Primary Implementation: Parallel Full-Sequence Processing

Replace the sequential step-by-step approach with efficient full-sequence batch processing:

```rust
pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
    // Token ids expected shape: [B,T]
    let (batch_size, seq_len) = token_ids.dims2()?;

    // Embed the entire sequence once
    let flat_ids = token_ids.flatten_all()?;
    let ids_vec: Vec<u32> = flat_ids.to_vec1()?;
    let mut hidden = self.embed(&ids_vec)?;
    let hidden_size = self.config.model.hidden_size;
    hidden = hidden.reshape(&[batch_size, seq_len, hidden_size])?;

    // Create causal attention mask for full sequence [B, T, T]
    let causal_mask = self.create_causal_mask(batch_size, seq_len)?;

    // Process through all layers with full sequence parallelization
    for (layer_idx, layer) in self.layers.iter().enumerate() {
        // Apply layer with causal mask for full sequence
        hidden = layer.forward_full_sequence(&hidden, &causal_mask, layer_idx)?;
    }

    // Project entire sequence to vocabulary logits: [B,T,V]
    let logits = self.logits(&hidden)?;

    Ok(logits)
}

// Helper method for causal mask creation
fn create_causal_mask(&self, batch_size: usize, seq_len: usize) -> Result<Tensor> {
    // Create lower triangular mask to prevent attention to future tokens
    let mask = Tensor::tril(&Tensor::ones((seq_len, seq_len), DType::F32, &self.device)?)?;
    // Expand for batch dimension: [1, 1, T, T] -> [B, 1, T, T]
    let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
    let mask = mask.expand(&[batch_size, 1, seq_len, seq_len])?;
    Ok(mask)
}
```

### Alternative Approach: Hybrid Processing

For scenarios requiring compatibility with existing incremental processing:

```rust
pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len) = token_ids.dims2()?;

    // Use parallel processing for sequences above threshold
    if seq_len > PARALLEL_THRESHOLD {
        return self.forward_full_parallel(token_ids);
    }

    // Fall back to incremental processing for short sequences
    self.forward_full_incremental(token_ids)
}
```

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Implement `create_causal_mask` utility function
- [ ] Add `forward_full_sequence` method to transformer layers
- [ ] Update attention mechanism to handle full-sequence causal masking
- [ ] Add device-aware mask creation (CPU vs GPU optimized)

### Phase 2: Layer Updates
- [ ] Update `BitNetAttention` to support full-sequence processing
- [ ] Modify `MultiHeadAttention` for batch causal attention
- [ ] Update feed-forward layers for batch processing
- [ ] Implement layer normalization optimizations

### Phase 3: Integration & Optimization
- [ ] Integrate new approach into `TransformerModel`
- [ ] Add configuration option for processing mode selection
- [ ] Implement memory-efficient attention for long sequences
- [ ] Add performance monitoring and metrics

### Phase 4: Testing & Validation
- [ ] Cross-validate outputs against reference implementation
- [ ] Performance benchmarking across different sequence lengths
- [ ] Memory usage profiling and optimization
- [ ] Integration testing with BitNet.rs inference pipeline

## Testing Strategy

### Functional Testing
```rust
#[test]
fn test_forward_full_equivalence() {
    // Verify output equivalence between sequential and parallel implementations
    let model = create_test_transformer();
    let input = create_test_batch();

    let sequential_output = model.forward_full_sequential(&input)?;
    let parallel_output = model.forward_full(&input)?;

    assert_tensors_close(&sequential_output, &parallel_output, 1e-5);
}

#[test]
fn test_causal_mask_correctness() {
    // Verify causal mask prevents future token attention
    let model = create_test_transformer();
    let mask = model.create_causal_mask(2, 4)?;

    // Verify lower triangular structure
    assert_causal_mask_structure(&mask);
}
```

### Performance Testing
```rust
#[test]
fn benchmark_forward_full_performance() {
    // Compare performance across different sequence lengths
    for seq_len in [32, 64, 128, 256, 512, 1024] {
        let input = create_batch_input(batch_size=8, seq_len);

        let start = Instant::now();
        let _output = model.forward_full(&input)?;
        let duration = start.elapsed();

        println!("Seq len {}: {:.2}ms", seq_len, duration.as_millis());
    }
}
```

## Related Issues/PRs

- Cross-validation with Microsoft BitNet C++ reference implementation
- GPU memory optimization for long sequences
- Integration with BitNet.rs inference server batch processing
- Performance monitoring and metrics collection

## Acceptance Criteria

- [ ] Full sequence processing implemented without step-by-step iteration
- [ ] Causal attention mask correctly prevents future token access
- [ ] Performance improvement of at least 5x for batch sizes > 1
- [ ] Memory usage remains constant or improves
- [ ] All existing tests pass with new implementation
- [ ] Cross-validation maintains numerical accuracy within 1e-5
- [ ] GPU utilization improves for batch inference workloads
- [ ] Backward compatibility maintained through configuration options

## Notes

This change is critical for production-ready batch inference performance and GPU utilization efficiency. The current sequential approach significantly underutilizes parallel compute capabilities and impacts overall system throughput.

Priority should be given to maintaining numerical accuracy while achieving performance gains, with comprehensive testing against the reference implementation to ensure correctness.
