# [Performance] Transformer Forward Full Sequential Processing Optimization

## Problem Description

The `TransformerModel::forward_full` function uses inefficient step-by-step token processing with KV cache, rather than batch processing the entire sequence, causing significant performance degradation for full sequence inference scenarios.

## Environment

- **File**: `crates/bitnet-models/src/transformer.rs`
- **Function**: `TransformerModel::forward_full`
- **Component**: Transformer Model Forward Pass
- **Rust Version**: 1.90.0+ (2024 edition)
- **Performance Impact**: High for batch inference and training

## Root Cause Analysis

The current implementation processes tokens sequentially in a loop, which is inefficient for scenarios where the full sequence is available:

### **Current Implementation:**
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

    // Create per-layer KV cache
    let mut kv_cache = KVCache::new(&self.config, batch_size, &self.device)?;

    // Collect logits for each position.
    let mut logits_steps = Vec::with_capacity(seq_len);
    for t in 0..seq_len {  // ← INEFFICIENT SEQUENTIAL PROCESSING
        // Select the current token's embedding: [B,1,H]
        let step_hidden = hidden.narrow(1, t, 1)?;

        // Run through all layers using the incremental path
        let step_hidden = self.forward(&step_hidden, Some(&mut kv_cache))?;

        // Project to vocabulary logits for this step.
        let step_logits = self.logits(&step_hidden)?;
        logits_steps.push(step_logits);
    }

    // Concatenate logits from all steps: [B,T,V]
    Ok(Tensor::cat(&logits_steps, 1)?)
}
```

### **Performance Issues:**
1. **Sequential Processing**: Processes one token at a time instead of batching
2. **Repeated Layer Traversal**: Runs through all layers `seq_len` times
3. **Memory Fragmentation**: Creates intermediate tensors for each step
4. **Cache Overhead**: KV cache operations for each individual step
5. **No Parallelization**: Cannot leverage tensor parallelism for batch operations

### **Expected vs Actual Complexity:**
- **Current**: O(seq_len × num_layers × ops_per_layer)
- **Optimal**: O(num_layers × ops_per_sequence)
- **Performance Impact**: ~seq_len × slowdown for batch processing

## Impact Assessment

### **Severity**: High
### **Affected Operations**: Batch inference, training, long sequence processing
### **Business Impact**: Significantly degraded performance for production workloads

**Current Limitations:**
- Cannot efficiently process multiple sequences simultaneously
- Poor scaling with sequence length
- Suboptimal GPU utilization due to small batch sizes
- Memory usage inefficiency from repeated allocations

## Proposed Solution

### **Primary Approach**: Batch-Optimized Forward Pass

Implement a batch-optimized forward pass that processes the entire sequence through all layers in a single pass, using causal masking to maintain autoregressive constraints.

### **Implementation Strategy:**

#### **1. Batch-Optimized Forward Pass**
```rust
impl TransformerModel {
    pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
        // Token ids expected shape: [B,T]
        let (batch_size, seq_len) = token_ids.dims2()?;

        // Embed the entire sequence once.
        let flat_ids = token_ids.flatten_all()?;
        let ids_vec: Vec<u32> = flat_ids.to_vec1()?;
        let mut hidden = self.embed(&ids_vec)?;
        let hidden_size = self.config.model.hidden_size;
        hidden = hidden.reshape(&[batch_size, seq_len, hidden_size])?;

        // Apply positional encodings
        hidden = self.apply_positional_encoding(&hidden)?;

        // Create causal attention mask for the full sequence
        let attention_mask = self.create_causal_mask(batch_size, seq_len)?;

        // Process through all transformer layers with batch operations
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.process_layer_batch(&hidden, layer, &attention_mask, layer_idx)?;
        }

        // Final layer normalization
        hidden = self.final_layer_norm.forward(&hidden)?;

        // Project to vocabulary logits for all positions at once
        let logits = self.logits(&hidden)?;

        Ok(logits)
    }

    fn process_layer_batch(
        &self,
        input: &Tensor,
        layer: &TransformerLayer,
        attention_mask: &Tensor,
        layer_idx: usize,
    ) -> Result<Tensor> {
        // Pre-attention layer normalization
        let normed_input = layer.layer_norm_pre().forward(input)?;

        // Multi-head attention with causal masking
        let attention_output = self.batch_attention(
            &normed_input,
            layer,
            attention_mask,
            layer_idx,
        )?;

        // Residual connection
        let attention_residual = input.add(&attention_output)?;

        // Pre-FFN layer normalization
        let ffn_input = layer.layer_norm_post().forward(&attention_residual)?;

        // Feed-forward network
        let ffn_output = layer.feed_forward().forward(&ffn_input)?;

        // Final residual connection
        let output = attention_residual.add(&ffn_output)?;

        Ok(output)
    }

    fn batch_attention(
        &self,
        input: &Tensor,
        layer: &TransformerLayer,
        attention_mask: &Tensor,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = input.dims3()?;
        let num_heads = self.config.model.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        // Compute Q, K, V projections for the entire sequence
        let q = layer.query_projection().forward(input)?;
        let k = layer.key_projection().forward(input)?;
        let v = layer.value_projection().forward(input)?;

        // Reshape for multi-head attention: [B, H, T, D]
        let q = q.reshape(&[batch_size, seq_len, num_heads, head_dim])?
            .transpose(1, 2)?;
        let k = k.reshape(&[batch_size, seq_len, num_heads, head_dim])?
            .transpose(1, 2)?;
        let v = v.reshape(&[batch_size, seq_len, num_heads, head_dim])?
            .transpose(1, 2)?;

        // Apply rotary position embeddings if configured
        let (q, k) = if self.config.model.use_rotary_embeddings {
            self.apply_rotary_embeddings(&q, &k, layer_idx)?
        } else {
            (q, k)
        };

        // Compute attention scores: [B, H, T, T]
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)?)?;
        let scaled_scores = scores.mul(scale)?;

        // Apply causal mask
        let masked_scores = self.apply_causal_mask(&scaled_scores, attention_mask)?;

        // Apply softmax
        let attention_weights = masked_scores.softmax(-1)?;

        // Apply attention to values: [B, H, T, D]
        let attention_output = attention_weights.matmul(&v)?;

        // Reshape back to [B, T, H*D]
        let attention_output = attention_output.transpose(1, 2)?
            .reshape(&[batch_size, seq_len, hidden_size])?;

        // Final output projection
        let output = layer.output_projection().forward(&attention_output)?;

        Ok(output)
    }

    fn create_causal_mask(&self, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        // Create lower triangular mask: [T, T]
        let mut mask_data = vec![f32::NEG_INFINITY; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = 0.0;
            }
        }

        let mask = Tensor::from_slice(&mask_data, &[seq_len, seq_len], &self.device)?;

        // Expand for batch and heads: [B, 1, T, T]
        let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
        let mask = mask.expand(&[batch_size, 1, seq_len, seq_len])?;

        Ok(mask)
    }

    fn apply_causal_mask(&self, scores: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Add mask to scores (0 for allowed, -inf for masked)
        scores.add(mask)
    }

    fn apply_positional_encoding(&self, hidden: &Tensor) -> Result<Tensor> {
        match &self.positional_encoding {
            PositionalEncoding::Absolute(pos_emb) => {
                let (batch_size, seq_len, hidden_size) = hidden.dims3()?;
                let pos_encoding = pos_emb.narrow(0, 0, seq_len)?;
                let pos_encoding = pos_encoding.unsqueeze(0)?
                    .expand(&[batch_size, seq_len, hidden_size])?;
                hidden.add(&pos_encoding)
            }
            PositionalEncoding::Rotary(_) => {
                // Rotary embeddings are applied in attention
                Ok(hidden.clone())
            }
            PositionalEncoding::None => Ok(hidden.clone()),
        }
    }

    fn apply_rotary_embeddings(
        &self,
        q: &Tensor,
        k: &Tensor,
        layer_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        if let PositionalEncoding::Rotary(rope) = &self.positional_encoding {
            let rotated_q = rope.apply(q, layer_idx)?;
            let rotated_k = rope.apply(k, layer_idx)?;
            Ok((rotated_q, rotated_k))
        } else {
            Ok((q.clone(), k.clone()))
        }
    }
}
```

#### **2. Memory-Efficient Batch Processing**
```rust
impl TransformerModel {
    pub fn forward_full_chunked(&self, token_ids: &Tensor, chunk_size: usize) -> Result<Tensor> {
        let (batch_size, seq_len) = token_ids.dims2()?;

        if seq_len <= chunk_size {
            // Small enough to process in one batch
            return self.forward_full(token_ids);
        }

        // Process in overlapping chunks to maintain context
        let overlap = chunk_size / 4; // 25% overlap
        let mut all_logits = Vec::new();

        for start in (0..seq_len).step_by(chunk_size - overlap) {
            let end = (start + chunk_size).min(seq_len);
            let chunk_tokens = token_ids.narrow(1, start, end - start)?;

            let chunk_logits = self.forward_full(&chunk_tokens)?;

            if start == 0 {
                // First chunk: keep all logits
                all_logits.push(chunk_logits);
            } else {
                // Subsequent chunks: skip overlap region
                let non_overlap_start = if start + overlap < end { overlap } else { 0 };
                let non_overlap_logits = chunk_logits.narrow(
                    1,
                    non_overlap_start,
                    chunk_logits.dim(1)? - non_overlap_start,
                )?;
                all_logits.push(non_overlap_logits);
            }
        }

        // Concatenate all chunk results
        Tensor::cat(&all_logits, 1)
    }

    pub fn forward_full_with_checkpointing(&self, token_ids: &Tensor) -> Result<Tensor> {
        // Gradient checkpointing for memory efficiency during training
        let checkpoint_layers = self.config.model.num_layers / 4; // Checkpoint every 4 layers

        let (batch_size, seq_len) = token_ids.dims2()?;
        let flat_ids = token_ids.flatten_all()?;
        let ids_vec: Vec<u32> = flat_ids.to_vec1()?;
        let mut hidden = self.embed(&ids_vec)?;
        hidden = hidden.reshape(&[batch_size, seq_len, self.config.model.hidden_size])?;

        let attention_mask = self.create_causal_mask(batch_size, seq_len)?;

        // Process layers with selective checkpointing
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if layer_idx % checkpoint_layers == 0 {
                // Create checkpoint for gradient computation
                hidden = self.checkpoint_layer_computation(&hidden, layer, &attention_mask, layer_idx)?;
            } else {
                // Normal forward pass
                hidden = self.process_layer_batch(&hidden, layer, &attention_mask, layer_idx)?;
            }
        }

        hidden = self.final_layer_norm.forward(&hidden)?;
        let logits = self.logits(&hidden)?;

        Ok(logits)
    }
}
```

#### **3. GPU-Optimized Implementation**
```rust
impl TransformerModel {
    pub fn forward_full_gpu_optimized(&self, token_ids: &Tensor) -> Result<Tensor> {
        // Ensure tensors are on GPU
        let token_ids = token_ids.to_device(&self.device)?;

        // Use fused operations where possible
        let (batch_size, seq_len) = token_ids.dims2()?;

        // Fused embedding + positional encoding
        let mut hidden = self.fused_embedding_positional(&token_ids)?;

        // Pre-allocate attention mask on GPU
        let attention_mask = self.create_causal_mask_gpu(batch_size, seq_len)?;

        // Process layers with GPU-optimized kernels
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = self.process_layer_gpu_fused(&hidden, layer, &attention_mask, layer_idx)?;
        }

        // Fused final norm + logits
        let logits = self.fused_final_norm_logits(&hidden)?;

        Ok(logits)
    }

    fn fused_embedding_positional(&self, token_ids: &Tensor) -> Result<Tensor> {
        // Custom kernel that combines embedding lookup and positional encoding
        let (batch_size, seq_len) = token_ids.dims2()?;
        let flat_ids = token_ids.flatten_all()?;
        let ids_vec: Vec<u32> = flat_ids.to_vec1()?;

        // Use custom CUDA kernel for fused operation
        #[cfg(feature = "gpu")]
        {
            self.launch_fused_embedding_kernel(&ids_vec, batch_size, seq_len)
        }
        #[cfg(not(feature = "gpu"))]
        {
            // Fallback to separate operations
            let hidden = self.embed(&ids_vec)?;
            let hidden = hidden.reshape(&[batch_size, seq_len, self.config.model.hidden_size])?;
            self.apply_positional_encoding(&hidden)
        }
    }

    fn process_layer_gpu_fused(
        &self,
        input: &Tensor,
        layer: &TransformerLayer,
        attention_mask: &Tensor,
        layer_idx: usize,
    ) -> Result<Tensor> {
        // Use fused attention kernel that combines:
        // 1. Layer norm
        // 2. QKV projection
        // 3. Attention computation
        // 4. Output projection
        // 5. Residual connection

        #[cfg(feature = "gpu")]
        {
            self.launch_fused_attention_kernel(input, layer, attention_mask, layer_idx)
        }
        #[cfg(not(feature = "gpu"))]
        {
            // Fallback to standard implementation
            self.process_layer_batch(input, layer, attention_mask, layer_idx)
        }
    }
}
```

## Implementation Plan

### **Phase 1: Core Batch Processing (Week 1-2)**
- Replace sequential loop with batch tensor operations
- Implement causal masking for full sequence attention
- Add proper positional encoding handling
- Optimize memory allocation patterns

### **Phase 2: Performance Optimization (Week 3)**
- Add chunked processing for memory efficiency
- Implement gradient checkpointing for training
- Optimize tensor reshaping and memory layout
- Add SIMD optimizations where applicable

### **Phase 3: GPU Acceleration (Week 4)**
- Implement GPU-optimized batch operations
- Add fused kernel operations
- Optimize GPU memory transfers
- Implement mixed precision support

## Testing Strategy

### **Performance Tests:**
```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn benchmark_forward_full_vs_sequential() {
        let model = create_test_transformer_model();
        let batch_size = 4;
        let seq_len = 512;
        let token_ids = create_test_tokens(batch_size, seq_len);

        // Benchmark old sequential method
        let start = Instant::now();
        let _sequential_result = model.forward_full_sequential(&token_ids).unwrap();
        let sequential_time = start.elapsed();

        // Benchmark new batch method
        let start = Instant::now();
        let _batch_result = model.forward_full(&token_ids).unwrap();
        let batch_time = start.elapsed();

        println!("Sequential time: {:?}", sequential_time);
        println!("Batch time: {:?}", batch_time);

        // Batch should be significantly faster
        assert!(batch_time < sequential_time / 2);
    }

    #[test]
    fn test_correctness_vs_sequential() {
        let model = create_test_transformer_model();
        let batch_size = 2;
        let seq_len = 32;
        let token_ids = create_test_tokens(batch_size, seq_len);

        let sequential_result = model.forward_full_sequential(&token_ids).unwrap();
        let batch_result = model.forward_full(&token_ids).unwrap();

        // Results should be identical within numerical precision
        let diff = sequential_result.sub(&batch_result).unwrap();
        let max_diff = diff.abs().unwrap().max().unwrap().to_scalar::<f32>().unwrap();

        assert!(max_diff < 1e-5, "Results differ by more than tolerance: {}", max_diff);
    }
}
```

### **Memory Tests:**
```rust
#[test]
fn test_memory_efficiency() {
    let model = create_test_transformer_model();
    let batch_size = 8;
    let seq_len = 1024;
    let token_ids = create_test_tokens(batch_size, seq_len);

    let initial_memory = get_memory_usage();

    let _result = model.forward_full(&token_ids).unwrap();

    let peak_memory = get_memory_usage();
    let memory_used = peak_memory - initial_memory;

    // Should use reasonable amount of memory
    let expected_max = batch_size * seq_len * model.config.model.hidden_size * 4 * 10; // 10x model size max
    assert!(memory_used < expected_max);
}
```

## Success Metrics

### **Performance:**
- [ ] >10x speedup for full sequence processing compared to sequential
- [ ] Linear scaling with sequence length (not quadratic)
- [ ] >80% GPU utilization for large batches
- [ ] Memory usage scales linearly with sequence length

### **Correctness:**
- [ ] Mathematical equivalence to sequential processing
- [ ] Proper causal masking maintains autoregressive properties
- [ ] Gradient computation correctness for training
- [ ] Numerical stability across different sequence lengths

### **Scalability:**
- [ ] Supports sequences up to 8K tokens efficiently
- [ ] Batch sizes up to 32 without memory issues
- [ ] Graceful degradation for very long sequences
- [ ] Memory-efficient processing for training workloads

## Acceptance Criteria

- [ ] `forward_full` processes entire sequence in single pass through layers
- [ ] Causal masking correctly prevents future token access
- [ ] Performance improves significantly over sequential processing
- [ ] Memory usage is efficient and predictable
- [ ] GPU utilization is maximized for batch operations
- [ ] Numerical results match sequential implementation
- [ ] Support for long sequences via chunking
- [ ] Documentation explains performance benefits and usage

## Labels

- `performance-optimization`
- `transformer-architecture`
- `batch-processing`
- `gpu-acceleration`
- `memory-efficiency`

## Related Issues

- **Dependencies**: Issue #XXX (Attention Mechanism Optimization)
- **Related**: Issue #XXX (GPU Memory Management), Issue #XXX (SIMD Optimization)
- **Enables**: Efficient batch inference, training support, long sequence processing