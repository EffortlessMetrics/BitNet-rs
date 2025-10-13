# [ENHANCEMENT] Improve attention mask handling with advanced masking strategies

## Problem Description
The `BitNetAttention::compute_attention` function in `crates/bitnet-inference/src/layers/attention.rs` uses simplified attention mask application that may not handle all edge cases or advanced masking scenarios.

## Environment
- **File**: `crates/bitnet-inference/src/layers/attention.rs`
- **Function**: `BitNetAttention::compute_attention`
- **Current State**: Basic broadcast_add masking

## Root Cause Analysis
```rust
let masked_scores = if let Some(mask) = attention_mask {
    let mask_candle = mask.to_candle()?;
    let _mask_value = -1e9; // Large negative value for masked positions
    scaled_scores.broadcast_add(&mask_candle).context("Failed to apply attention mask")?
} else {
    scaled_scores
};
```

**Issues:**
1. Hardcoded mask value (-1e9) may cause numerical instability
2. Simple broadcast_add doesn't handle different mask types
3. No support for causal masking optimization
4. Potential precision issues with large negative values

## Proposed Solution
```rust
impl BitNetAttention {
    fn compute_attention(&self, q: &BitNetTensor, k: &BitNetTensor, v: &BitNetTensor, attention_mask: Option<&BitNetTensor>) -> Result<BitNetTensor> {
        // ... existing QK computation ...

        let masked_scores = if let Some(mask) = attention_mask {
            self.apply_attention_mask(&scaled_scores, mask)?
        } else {
            scaled_scores
        };

        // ... rest of function ...
    }

    fn apply_attention_mask(&self, scores: &Tensor, mask: &BitNetTensor) -> Result<Tensor> {
        let mask_candle = mask.to_candle()?;

        // Use appropriate mask value based on dtype
        let mask_value = match scores.dtype() {
            DType::F32 => -3.4028235e38f32, // Near -inf for f32
            DType::F16 => -65504.0f32,      // Near -inf for f16
            DType::BF16 => -3.38953e38f32,  // Near -inf for bf16
            _ => -1e9f32,
        };

        // Create mask tensor with proper value
        let neg_inf_mask = mask_candle.full_like(mask_value)?;

        // Apply mask: where mask==0, use -inf; where mask==1, use original scores
        scores.where_cond(&mask_candle.eq(1.0)?, &neg_inf_mask)
    }

    fn apply_causal_mask(&self, scores: &Tensor, seq_len: usize) -> Result<Tensor> {
        // Optimized causal masking for autoregressive generation
        let causal_mask = self.create_causal_mask(seq_len, scores.device())?;
        self.apply_attention_mask(scores, &BitNetTensor::from_candle(causal_mask)?)
    }

    fn create_causal_mask(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        // Create lower triangular mask efficiently
        let mask = Tensor::tril2(seq_len, DType::F32, device)?;
        Ok(mask)
    }
}
```

## Implementation Plan
### Phase 1: Robust Masking (1 day)
- [ ] Implement dtype-aware mask values
- [ ] Add proper mask application with where_cond
- [ ] Create causal mask optimization

### Phase 2: Advanced Features (1 day)
- [ ] Add sliding window attention support
- [ ] Implement block-sparse attention patterns
- [ ] Add mask validation and error handling

## Acceptance Criteria
- [ ] Numerically stable masking across data types
- [ ] Support for various mask patterns (causal, padding, custom)
- [ ] Optimized causal masking for generation
- [ ] Comprehensive test coverage for edge cases

**Labels**: `enhancement`, `attention`, `numerical-stability`, `P2-medium`
**Effort**: 2 days
