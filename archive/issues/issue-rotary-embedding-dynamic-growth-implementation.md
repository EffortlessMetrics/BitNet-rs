# [Attention] Rotary Embedding Dynamic Growth Implementation

## Problem Description

The `RotaryEmbedding::apply` function has a placeholder comment suggesting dynamic growth when sequence length exceeds `max_seq_len`, but this functionality is not implemented, limiting model flexibility for longer sequences.

## Environment

- **File**: `crates/bitnet-models/src/transformer.rs`
- **Function**: `RotaryEmbedding::apply`
- **Component**: Rotary Position Embedding System

## Root Cause Analysis

### **Current Issue:**
The function contains logic that would fail when sequence length exceeds the precomputed `max_seq_len`, with a comment suggesting dynamic growth is needed:

```rust
// Comment in apply function suggests:
// "Sequence length {} exceeds max_seq_len {} (consider dynamic growth)"
```

### **Problems:**
1. **Fixed Sequence Limit**: Cannot handle sequences longer than initial `max_seq_len`
2. **No Dynamic Expansion**: Missing logic to grow sin/cos caches
3. **Runtime Failures**: Long sequences cause errors instead of graceful handling

## Proposed Solution

Implement dynamic cache expansion for rotary embeddings:

```rust
impl RotaryEmbedding {
    pub fn apply(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        let seq_len = x.dim(-2)?;
        let required_len = position + seq_len;

        // Check if we need to expand the cache
        if required_len > self.max_seq_len {
            self.expand_cache(required_len)?;
        }

        // Rest of the existing implementation...
        self.apply_rotary_internal(x, position)
    }

    fn expand_cache(&mut self, new_max_len: usize) -> Result<()> {
        let new_len = (new_max_len * 2).next_power_of_two(); // Grow exponentially

        // Recompute sin and cos for extended length
        let (new_cos, new_sin) = self.compute_rotation_matrices(new_len)?;

        self.cos = new_cos;
        self.sin = new_sin;
        self.max_seq_len = new_len;

        Ok(())
    }

    fn compute_rotation_matrices(&self, max_len: usize) -> Result<(Tensor, Tensor)> {
        // Recompute rotation matrices for the new maximum length
        let base = 10000.0f32;
        let inv_freq: Vec<f32> = (0..self.head_dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f32 / self.head_dim as f32))
            .collect();

        let positions: Vec<f32> = (0..max_len).map(|i| i as f32).collect();

        // Compute outer product and create sin/cos tables
        let mut cos_vals = Vec::with_capacity(max_len * self.head_dim / 2);
        let mut sin_vals = Vec::with_capacity(max_len * self.head_dim / 2);

        for &pos in &positions {
            for &freq in &inv_freq {
                let angle = pos * freq;
                cos_vals.push(angle.cos());
                sin_vals.push(angle.sin());
            }
        }

        let cos = Tensor::from_slice(&cos_vals, &[max_len, self.head_dim / 2], &self.device)?;
        let sin = Tensor::from_slice(&sin_vals, &[max_len, self.head_dim / 2], &self.device)?;

        Ok((cos, sin))
    }
}
```

## Implementation Plan

### **Week 1: Dynamic Expansion**
- Implement cache expansion logic
- Add exponential growth strategy
- Create rotation matrix recomputation

### **Week 2: Optimization and Testing**
- Add caching strategies for frequently used lengths
- Implement memory-efficient expansion
- Add comprehensive tests for various sequence lengths

## Success Metrics

- [ ] Support for sequences up to 16K tokens without preallocation
- [ ] Graceful handling of sequence length expansion
- [ ] Minimal performance overhead for cache expansion
- [ ] Memory-efficient growth strategy

## Labels

- `rotary-embeddings`
- `dynamic-allocation`
- `sequence-length`
- `memory-efficiency`
