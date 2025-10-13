# [Attention] Grouped Query Attention (GQA) Production Implementation

## Problem Description

The `BitNetAttention::apply_gqa` function contains only a simplified placeholder implementation that clones key and value tensors instead of performing proper Grouped Query Attention computation. This prevents the attention mechanism from achieving the memory efficiency and computational benefits that GQA is designed to provide.

## Environment

- **File**: `crates/bitnet-inference/src/layers/attention.rs`
- **Function**: `BitNetAttention::apply_gqa`
- **Component**: Multi-Head Attention System
- **Rust Version**: 1.90.0+ (2024 edition)
- **Features**: `cpu`, `gpu` (affects both backends)

## Root Cause Analysis

The current GQA implementation is a placeholder that simply clones tensors without performing the essential grouped query attention computation:

### **Current Implementation:**
```rust
fn apply_gqa(
    &self,
    key_states: &BitNetTensor,
    value_states: &BitNetTensor,
) -> Result<(BitNetTensor, BitNetTensor)> {
    // For GQA, we repeat the key and value states for each group
    // This is a simplified implementation
    Ok((key_states.clone(), value_states.clone()))
}
```

### **Problems Identified:**
1. **No Group Computation**: Missing the core GQA logic of grouping query heads
2. **Memory Inefficiency**: Cloning tensors instead of sharing KV heads across groups
3. **Performance Loss**: Not achieving the computational benefits of GQA
4. **Incomplete Attention**: Missing proper attention head management
5. **Architecture Mismatch**: Not compatible with models that rely on GQA for efficiency

### **GQA Theory:**
Grouped Query Attention reduces the number of key-value heads while maintaining multiple query heads:
- **Traditional MHA**: Each attention head has its own Q, K, V
- **GQA**: Multiple query heads share the same key-value heads within groups
- **Benefits**: Reduces memory usage and computational overhead while maintaining performance

## Impact Assessment

### **Severity**: Medium-High
### **Affected Operations**: Multi-head attention computation in neural networks
### **Business Impact**: Reduced model efficiency and compatibility issues

**Current Limitations:**
- Cannot achieve memory efficiency benefits of GQA
- Incompatible with models designed for grouped query attention
- Missing performance optimizations for large-scale attention computation
- Inefficient memory usage during inference

## Proposed Solution

### **Primary Approach**: Complete GQA Implementation

Implement a production-ready Grouped Query Attention mechanism that properly handles head grouping, key-value sharing, and efficient tensor operations.

### **Implementation Strategy:**

#### **1. Enhanced Configuration Support**
```rust
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_sequence_length: usize,
    pub use_gqa: bool,
}

impl AttentionConfig {
    pub fn num_query_groups(&self) -> usize {
        assert!(self.num_attention_heads % self.num_key_value_heads == 0);
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(anyhow::anyhow!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads,
                self.num_key_value_heads
            ));
        }
        Ok(())
    }
}
```

#### **2. Production GQA Implementation**
```rust
impl BitNetAttention {
    fn apply_gqa(
        &self,
        key_states: &BitNetTensor,
        value_states: &BitNetTensor,
    ) -> Result<(BitNetTensor, BitNetTensor)> {
        if !self.config.use_gqa {
            // For non-GQA models, return as-is
            return Ok((key_states.clone(), value_states.clone()));
        }

        let batch_size = key_states.shape()[0];
        let seq_len = key_states.shape()[1];
        let num_kv_heads = self.config.num_key_value_heads;
        let num_q_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim;
        let num_kv_groups = self.config.num_query_groups();

        // Validate input tensor shapes
        self.validate_kv_shapes(key_states, value_states)?;

        // Reshape key and value states to separate heads
        let key_reshaped = key_states.reshape(&[batch_size, seq_len, num_kv_heads, head_dim])?;
        let value_reshaped = value_states.reshape(&[batch_size, seq_len, num_kv_heads, head_dim])?;

        // Expand key and value heads to match query heads via repetition
        let expanded_keys = self.expand_kv_heads(&key_reshaped, num_kv_groups)?;
        let expanded_values = self.expand_kv_heads(&value_reshaped, num_kv_groups)?;

        // Reshape back to expected format
        let final_key_shape = [batch_size, seq_len, num_q_heads * head_dim];
        let final_value_shape = [batch_size, seq_len, num_q_heads * head_dim];

        let final_keys = expanded_keys.reshape(&final_key_shape)?;
        let final_values = expanded_values.reshape(&final_value_shape)?;

        Ok((final_keys, final_values))
    }

    fn expand_kv_heads(
        &self,
        kv_tensor: &BitNetTensor,
        num_kv_groups: usize,
    ) -> Result<BitNetTensor> {
        let shape = kv_tensor.shape();
        let [batch_size, seq_len, num_kv_heads, head_dim] = shape[0..4] else {
            return Err(anyhow::anyhow!("Invalid KV tensor shape: {:?}", shape));
        };

        match kv_tensor {
            BitNetTensor::Candle(tensor) => {
                // Use candle's repeat_interleave for efficient expansion
                let expanded = tensor.repeat_interleave(num_kv_groups, 2)?;
                Ok(BitNetTensor::Candle(expanded))
            },
            BitNetTensor::Raw(data, shape, dtype) => {
                // Manual expansion for raw tensors
                self.expand_raw_tensor(data, shape, dtype, num_kv_groups)
            },
        }
    }

    fn expand_raw_tensor(
        &self,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
        num_kv_groups: usize,
    ) -> Result<BitNetTensor> {
        match dtype {
            DType::F32 => self.expand_f32_tensor(data, shape, num_kv_groups),
            DType::F16 => self.expand_f16_tensor(data, shape, num_kv_groups),
            DType::I8 => self.expand_i8_tensor(data, shape, num_kv_groups),
            _ => Err(anyhow::anyhow!("Unsupported dtype for GQA expansion: {:?}", dtype)),
        }
    }

    fn expand_f32_tensor(
        &self,
        data: &[u8],
        shape: &[usize],
        num_kv_groups: usize,
    ) -> Result<BitNetTensor> {
        let f32_data = bytemuck::cast_slice::<u8, f32>(data);
        let [batch_size, seq_len, num_kv_heads, head_dim] = shape[0..4] else {
            return Err(anyhow::anyhow!("Invalid shape for F32 expansion"));
        };

        let mut expanded_data = Vec::with_capacity(
            batch_size * seq_len * num_kv_heads * num_kv_groups * head_dim
        );

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..num_kv_heads {
                    // Repeat each head num_kv_groups times
                    let head_start = ((b * seq_len + s) * num_kv_heads + h) * head_dim;
                    let head_data = &f32_data[head_start..head_start + head_dim];

                    for _ in 0..num_kv_groups {
                        expanded_data.extend_from_slice(head_data);
                    }
                }
            }
        }

        let expanded_shape = [batch_size, seq_len, num_kv_heads * num_kv_groups, head_dim];
        let expanded_bytes = bytemuck::cast_slice::<f32, u8>(&expanded_data);

        Ok(BitNetTensor::Raw(
            expanded_bytes.to_vec(),
            expanded_shape.to_vec(),
            DType::F32,
        ))
    }

    fn validate_kv_shapes(
        &self,
        key_states: &BitNetTensor,
        value_states: &BitNetTensor,
    ) -> Result<()> {
        let key_shape = key_states.shape();
        let value_shape = value_states.shape();

        if key_shape != value_shape {
            return Err(anyhow::anyhow!(
                "Key and value tensors must have the same shape: key={:?}, value={:?}",
                key_shape,
                value_shape
            ));
        }

        if key_shape.len() < 3 {
            return Err(anyhow::anyhow!(
                "KV tensors must have at least 3 dimensions, got: {:?}",
                key_shape
            ));
        }

        Ok(())
    }
}
```

#### **3. Optimized GQA Computation**
```rust
impl BitNetAttention {
    pub fn forward_with_gqa(
        &self,
        query: &BitNetTensor,
        key: &BitNetTensor,
        value: &BitNetTensor,
        attention_mask: Option<&BitNetTensor>,
    ) -> Result<BitNetTensor> {
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        // Reshape tensors for multi-head attention
        let q_reshaped = query.reshape(&[batch_size, seq_len, num_q_heads, head_dim])?;
        let k_reshaped = key.reshape(&[batch_size, seq_len, num_kv_heads, head_dim])?;
        let v_reshaped = value.reshape(&[batch_size, seq_len, num_kv_heads, head_dim])?;

        // Apply GQA expansion if needed
        let (k_expanded, v_expanded) = if self.config.use_gqa {
            self.apply_gqa(&k_reshaped, &v_reshaped)?
        } else {
            (k_reshaped, v_reshaped)
        };

        // Compute attention efficiently with grouped heads
        let attention_output = self.compute_grouped_attention(
            &q_reshaped,
            &k_expanded,
            &v_expanded,
            attention_mask,
        )?;

        // Reshape back to expected output format
        let output_shape = [batch_size, seq_len, num_q_heads * head_dim];
        attention_output.reshape(&output_shape)
    }

    fn compute_grouped_attention(
        &self,
        query: &BitNetTensor,
        key: &BitNetTensor,
        value: &BitNetTensor,
        attention_mask: Option<&BitNetTensor>,
    ) -> Result<BitNetTensor> {
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];
        let num_heads = query.shape()[2];
        let head_dim = query.shape()[3];

        // Transpose for efficient matrix multiplication: [B, H, S, D]
        let q_transposed = query.transpose(1, 2)?; // [B, H, S, D]
        let k_transposed = key.transpose(1, 2)?;   // [B, H, S, D]
        let v_transposed = value.transpose(1, 2)?; // [B, H, S, D]

        // Compute attention scores: [B, H, S, S]
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scores = self.batched_matmul(&q_transposed, &k_transposed.transpose(-2, -1)?)?;
        let scaled_scores = scores.mul_scalar(scale)?;

        // Apply attention mask if provided
        let masked_scores = if let Some(mask) = attention_mask {
            self.apply_attention_mask(&scaled_scores, mask)?
        } else {
            scaled_scores
        };

        // Apply softmax
        let attention_weights = masked_scores.softmax(-1)?;

        // Apply attention to values: [B, H, S, D]
        let attention_output = self.batched_matmul(&attention_weights, &v_transposed)?;

        // Transpose back to [B, S, H, D]
        attention_output.transpose(1, 2)
    }

    fn batched_matmul(
        &self,
        a: &BitNetTensor,
        b: &BitNetTensor,
    ) -> Result<BitNetTensor> {
        match (a, b) {
            (BitNetTensor::Candle(a_tensor), BitNetTensor::Candle(b_tensor)) => {
                let result = a_tensor.matmul(b_tensor)?;
                Ok(BitNetTensor::Candle(result))
            },
            _ => {
                // Fallback to manual implementation for raw tensors
                self.manual_batched_matmul(a, b)
            }
        }
    }
}
```

#### **4. Memory-Efficient KV Cache Integration**
```rust
impl BitNetAttention {
    pub fn forward_with_kv_cache(
        &self,
        query: &BitNetTensor,
        kv_cache: &mut KVCache,
        layer_idx: usize,
        position: usize,
    ) -> Result<BitNetTensor> {
        // Check if we can use cached KV states
        if let Some((cached_k, cached_v)) = kv_cache.get_cached_kv(layer_idx, position) {
            // Use cached KV states with GQA
            return self.forward_with_gqa(&query, &cached_k, &cached_v, None);
        }

        // Compute new KV states
        let key_states = self.compute_key_states(query)?;
        let value_states = self.compute_value_states(query)?;

        // Store in cache for future use
        kv_cache.update(layer_idx, position, key_states.clone(), value_states.clone())?;

        // Apply GQA and compute attention
        self.forward_with_gqa(&query, &key_states, &value_states, None)
    }

    fn compute_key_states(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        // Apply key projection with proper head configuration
        let projected = self.key_proj.forward(input)?;

        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        projected.reshape(&[batch_size, seq_len, num_kv_heads, head_dim])
    }

    fn compute_value_states(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        // Apply value projection with proper head configuration
        let projected = self.value_proj.forward(input)?;

        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        projected.reshape(&[batch_size, seq_len, num_kv_heads, head_dim])
    }
}
```

## Implementation Plan

### **Phase 1: Core GQA Logic (Week 1)**

#### **Task 1.1: Configuration Enhancement**
```rust
// Update AttentionConfig to support GQA parameters
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,  // New: separate KV head count
    pub head_dim: usize,
    pub use_gqa: bool,               // New: enable GQA
    pub max_sequence_length: usize,
}
```

#### **Task 1.2: Tensor Expansion Implementation**
```rust
// Implement efficient tensor expansion for KV heads
fn expand_kv_heads(&self, kv_tensor: &BitNetTensor, num_groups: usize) -> Result<BitNetTensor> {
    // Implementation using repeat_interleave or manual expansion
}
```

### **Phase 2: Attention Computation (Week 2)**

#### **Task 2.1: Grouped Attention Forward Pass**
```rust
// Implement complete forward pass with GQA
fn forward_with_gqa(
    &self,
    query: &BitNetTensor,
    key: &BitNetTensor,
    value: &BitNetTensor,
    attention_mask: Option<&BitNetTensor>,
) -> Result<BitNetTensor> {
    // Full implementation with efficient grouped computation
}
```

#### **Task 2.2: Efficient Matrix Operations**
```rust
// Optimize matrix multiplications for grouped attention
fn compute_grouped_attention(&self, ...) -> Result<BitNetTensor> {
    // Efficient batched operations
}
```

### **Phase 3: Integration and Optimization (Week 3)**

#### **Task 3.1: KV Cache Integration**
```rust
// Integrate GQA with existing KV cache system
fn forward_with_kv_cache(&self, ...) -> Result<BitNetTensor> {
    // Efficient caching for GQA
}
```

#### **Task 3.2: Performance Optimization**
- SIMD optimizations for tensor expansion
- Memory layout optimization for cache efficiency
- GPU kernel optimization for CUDA backend

## Testing Strategy

### **Unit Tests:**
```rust
#[cfg(test)]
mod gqa_tests {
    use super::*;

    #[test]
    fn test_gqa_tensor_expansion() {
        let config = AttentionConfig {
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 64,
            use_gqa: true,
            max_sequence_length: 512,
        };

        let attention = BitNetAttention::new(config).unwrap();

        // Create test KV tensors with 8 heads
        let batch_size = 2;
        let seq_len = 10;
        let kv_heads = 8;
        let head_dim = 64;

        let key_tensor = create_test_tensor(&[batch_size, seq_len, kv_heads, head_dim]);
        let value_tensor = create_test_tensor(&[batch_size, seq_len, kv_heads, head_dim]);

        let (expanded_keys, expanded_values) = attention.apply_gqa(&key_tensor, &value_tensor).unwrap();

        // Should expand to 32 heads (4 groups × 8 heads)
        assert_eq!(expanded_keys.shape(), &[batch_size, seq_len, 32 * head_dim]);
        assert_eq!(expanded_values.shape(), &[batch_size, seq_len, 32 * head_dim]);
    }

    #[test]
    fn test_gqa_attention_computation() {
        let config = AttentionConfig {
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 32,
            use_gqa: true,
            max_sequence_length: 128,
        };

        let attention = BitNetAttention::new(config).unwrap();

        let batch_size = 1;
        let seq_len = 8;

        let query = create_test_tensor(&[batch_size, seq_len, 16 * 32]);
        let key = create_test_tensor(&[batch_size, seq_len, 4 * 32]);
        let value = create_test_tensor(&[batch_size, seq_len, 4 * 32]);

        let output = attention.forward_with_gqa(&query, &key, &value, None).unwrap();

        assert_eq!(output.shape(), query.shape());
    }

    #[test]
    fn test_gqa_vs_mha_consistency() {
        // Test that GQA produces similar results to MHA when using same number of heads
        let gqa_config = AttentionConfig {
            num_attention_heads: 8,
            num_key_value_heads: 8,
            head_dim: 64,
            use_gqa: true,
            max_sequence_length: 64,
        };

        let mha_config = AttentionConfig {
            num_attention_heads: 8,
            num_key_value_heads: 8,
            head_dim: 64,
            use_gqa: false,
            max_sequence_length: 64,
        };

        // Compare outputs...
    }
}
```

### **Integration Tests:**
```rust
#[test]
fn test_gqa_end_to_end_inference() {
    let model = create_test_bitnet_model_with_gqa();
    let input_tokens = vec![1, 2, 3, 4, 5];

    let output = model.forward(&input_tokens).unwrap();

    // Verify output is valid and attention works correctly
    assert!(!output.is_empty());
    assert!(output.len() > input_tokens.len());
}

#[test]
fn test_gqa_memory_efficiency() {
    // Test that GQA uses less memory than equivalent MHA
    let gqa_memory = measure_memory_usage_with_gqa();
    let mha_memory = measure_memory_usage_with_mha();

    assert!(gqa_memory < mha_memory);
}
```

## Alternative Approaches

### **Alternative 1: Simple Head Repetition**
**Approach**: Use basic tensor repetition without optimization
**Pros**: Simpler implementation
**Cons**: Higher memory usage, slower computation

### **Alternative 2: Dynamic Head Grouping**
**Approach**: Dynamically determine grouping based on input
**Pros**: More flexible
**Cons**: More complex, potential runtime overhead

### **Alternative 3: External Library Integration**
**Approach**: Use existing attention libraries with GQA support
**Pros**: Battle-tested implementation
**Cons**: Additional dependencies, less control over optimization

**Selected Approach**: Primary production implementation provides the best balance of performance, memory efficiency, and maintainability.

## Performance Considerations

### **Memory Efficiency:**
- **GQA Memory Reduction**: ~25-50% reduction compared to full MHA
- **Cache Efficiency**: Better cache utilization due to shared KV heads
- **Peak Memory**: Lower peak memory usage during large sequence processing

### **Computational Efficiency:**
- **FLOPs Reduction**: ~15-30% reduction in attention computation
- **Parallelization**: Better GPU utilization with grouped computation
- **Cache Performance**: Improved memory access patterns

## Success Metrics

### **Functionality:**
- [ ] Correct GQA computation with proper head grouping
- [ ] Memory usage reduction compared to full MHA
- [ ] Compatibility with existing attention mechanisms
- [ ] Integration with KV cache system

### **Performance:**
- [ ] 15-30% reduction in attention computation time
- [ ] 25-50% reduction in attention memory usage
- [ ] No accuracy degradation compared to full MHA
- [ ] Efficient scaling with number of heads and sequence length

### **Quality:**
- [ ] Unit test coverage >95% for GQA functionality
- [ ] Integration tests validate end-to-end inference
- [ ] Performance benchmarks meet target metrics
- [ ] Cross-validation with reference implementations

## Acceptance Criteria

- [ ] `apply_gqa` performs proper grouped query attention computation
- [ ] Tensor expansion correctly maps KV heads to query head groups
- [ ] Attention computation produces mathematically correct results
- [ ] Memory usage is reduced compared to full multi-head attention
- [ ] Performance improves for large models and long sequences
- [ ] Integration with KV cache maintains correctness
- [ ] Unit tests validate all GQA operations
- [ ] Documentation explains GQA configuration and usage

## Labels

- `attention-mechanism`
- `memory-optimization`
- `performance`
- `neural-networks`
- `cpu-gpu-common`

## Related Issues

- **Dependencies**: Issue #XXX (Attention Mechanism Optimization)
- **Related**: Issue #XXX (KV Cache Implementation), Issue #XXX (Memory Management)
- **Enables**: Efficient large model inference, reduced memory requirements
