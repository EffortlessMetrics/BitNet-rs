# [Attention] Implement Comprehensive Attention Masking in BitNetAttention::compute_attention

## Problem Description

The `BitNetAttention::compute_attention` function in `crates/bitnet-inference/src/layers/attention.rs` currently implements a simplified attention mask application that may not handle all edge cases or advanced masking scenarios required for production neural network inference.

## Environment

- **Component**: `crates/bitnet-inference/src/layers/attention.rs`
- **Function**: `BitNetAttention::compute_attention`
- **Feature Context**: Core inference functionality (both `cpu` and `gpu` features)
- **Impact**: All models using attention mechanisms (BitNet-1.58B, BitNet-3B)

## Current Implementation Analysis

```rust
// Apply attention mask if provided
let masked_scores = if let Some(mask) = attention_mask {
    let mask_candle = mask.to_candle()?;
    let _mask_value = -1e9; // Large negative value for masked positions
    scaled_scores.broadcast_add(&mask_candle).context("Failed to apply attention mask")?
} else {
    scaled_scores
};
```

**Issues Identified:**
1. **Hardcoded mask value**: Uses `-1e9` which may not be optimal for all precision types (FP16, BF16, FP32)
2. **Broadcast addition**: Simple addition may not handle complex mask patterns correctly
3. **No mask type validation**: Doesn't distinguish between causal masks, padding masks, or custom attention patterns
4. **Missing numerical stability**: No protection against NaN propagation in masked positions
5. **Inefficient masking**: Applies mask after scaling, potentially wasting computation

## Impact Assessment

**Severity**: Medium-High
**Affected Users**: All users running inference with models requiring attention masking
**Performance Impact**:
- Suboptimal memory usage during attention computation
- Potential numerical instability in edge cases
- Incorrect attention patterns for certain model architectures

## Root Cause Analysis

The current implementation appears to be a placeholder that handles basic masking scenarios but lacks the sophistication required for:
1. **Causal attention**: Required for autoregressive language models
2. **Padding masks**: Essential for variable-length sequences
3. **Mixed precision**: Different optimal mask values for FP16 vs FP32
4. **Complex attention patterns**: Custom masks for specific model architectures

## Proposed Solution

### 1. Enhanced Mask Application Strategy

Implement a comprehensive masking system that:
- Detects mask type automatically
- Uses precision-appropriate mask values
- Applies masks efficiently before unnecessary computation
- Handles edge cases gracefully

### 2. Implementation Plan

```rust
impl BitNetAttention {
    fn compute_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&AttentionMask>,
        kv_cache: Option<&mut KVCache>,
    ) -> Result<Tensor> {
        // ... existing QKV computation ...

        let scaled_scores = self.scale_attention_scores(attention_scores)?;

        // Enhanced attention mask application
        let masked_scores = self.apply_attention_mask(scaled_scores, attention_mask)?;

        // ... rest of attention computation ...
    }

    fn apply_attention_mask(
        &self,
        scores: Tensor,
        mask: Option<&AttentionMask>,
    ) -> Result<Tensor> {
        let Some(mask) = mask else {
            return Ok(scores);
        };

        match mask.mask_type() {
            MaskType::Causal => self.apply_causal_mask(scores, mask),
            MaskType::Padding => self.apply_padding_mask(scores, mask),
            MaskType::Custom => self.apply_custom_mask(scores, mask),
        }
    }

    fn apply_causal_mask(&self, scores: Tensor, mask: &AttentionMask) -> Result<Tensor> {
        let mask_value = self.get_mask_value_for_dtype(scores.dtype())?;
        let mask_tensor = mask.to_candle()?;

        // Use where_cond for efficient conditional masking
        scores.where_cond(
            &mask_tensor.eq(0.0)?,
            &Tensor::full(mask_value, scores.shape(), scores.device())?,
        )
    }

    fn apply_padding_mask(&self, scores: Tensor, mask: &AttentionMask) -> Result<Tensor> {
        let mask_value = self.get_mask_value_for_dtype(scores.dtype())?;
        let mask_tensor = mask.to_candle()?;

        // Expand mask to match attention scores dimensions
        let expanded_mask = mask_tensor.unsqueeze(1)?.unsqueeze(1)?
            .expand(scores.shape())?;

        scores.where_cond(
            &expanded_mask.eq(0.0)?,
            &Tensor::full(mask_value, scores.shape(), scores.device())?,
        )
    }

    fn apply_custom_mask(&self, scores: Tensor, mask: &AttentionMask) -> Result<Tensor> {
        let mask_tensor = mask.to_candle()?;

        // For custom masks, use the provided mask values directly
        scores * &mask_tensor
    }

    fn get_mask_value_for_dtype(&self, dtype: DType) -> Result<f64> {
        match dtype {
            DType::F16 => Ok(-65504.0), // Largest negative value for FP16
            DType::BF16 => Ok(-3.38e38), // Appropriate for BF16
            DType::F32 => Ok(-3.4028235e38), // Largest negative for FP32
            DType::F64 => Ok(-1.7976931348623157e308), // Largest negative for FP64
            _ => bail!("Unsupported dtype for attention masking: {:?}", dtype),
        }
    }
}
```

### 3. Enhanced AttentionMask Type

```rust
#[derive(Debug, Clone)]
pub struct AttentionMask {
    mask: Tensor,
    mask_type: MaskType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskType {
    /// Causal mask for autoregressive models (lower triangular)
    Causal,
    /// Padding mask for variable-length sequences
    Padding,
    /// Custom attention pattern
    Custom,
}

impl AttentionMask {
    pub fn causal(seq_len: usize, device: &Device) -> Result<Self> {
        let mask = Tensor::tril2(seq_len, DType::F32, device)?;
        Ok(Self {
            mask,
            mask_type: MaskType::Causal,
        })
    }

    pub fn padding(lengths: &[usize], max_len: usize, device: &Device) -> Result<Self> {
        let batch_size = lengths.len();
        let mut mask_data = vec![0.0f32; batch_size * max_len];

        for (batch_idx, &len) in lengths.iter().enumerate() {
            for pos in 0..len.min(max_len) {
                mask_data[batch_idx * max_len + pos] = 1.0;
            }
        }

        let mask = Tensor::from_vec(mask_data, (batch_size, max_len), device)?;
        Ok(Self {
            mask,
            mask_type: MaskType::Padding,
        })
    }

    pub fn custom(mask: Tensor) -> Self {
        Self {
            mask,
            mask_type: MaskType::Custom,
        }
    }

    pub fn mask_type(&self) -> MaskType {
        self.mask_type
    }

    pub fn to_candle(&self) -> Result<&Tensor> {
        Ok(&self.mask)
    }
}
```

## Implementation Breakdown

### Phase 1: Core Masking Infrastructure
- [ ] Implement `AttentionMask` enum and types
- [ ] Add precision-aware mask value calculation
- [ ] Create unit tests for mask type detection

### Phase 2: Mask Application Methods
- [ ] Implement `apply_causal_mask` with triangular masking
- [ ] Implement `apply_padding_mask` with sequence length handling
- [ ] Implement `apply_custom_mask` for arbitrary patterns
- [ ] Add performance benchmarks for each method

### Phase 3: Integration and Optimization
- [ ] Update `BitNetAttention::compute_attention` to use new system
- [ ] Add GPU kernel optimizations for mask application
- [ ] Implement SIMD optimizations for CPU masking

### Phase 4: Testing and Validation
- [ ] Add comprehensive unit tests for all mask types
- [ ] Create integration tests with real model attention patterns
- [ ] Add cross-validation tests against reference implementations
- [ ] Performance regression testing

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_application() {
        let device = Device::Cpu;
        let seq_len = 4;
        let mask = AttentionMask::causal(seq_len, &device).unwrap();

        // Test that mask correctly blocks future positions
        let scores = Tensor::ones((1, 1, seq_len, seq_len), DType::F32, &device).unwrap();
        let attention = BitNetAttention::new(128, 8).unwrap();
        let masked = attention.apply_attention_mask(scores, Some(&mask)).unwrap();

        // Verify upper triangular positions are masked
        let values = masked.to_vec2::<f32>().unwrap();
        assert!(values[0][1] < -1000.0); // Future position should be masked
        assert_eq!(values[1][1], 1.0); // Current position should not be masked
    }

    #[test]
    fn test_padding_mask_variable_lengths() {
        let device = Device::Cpu;
        let lengths = vec![2, 3, 1];
        let max_len = 4;
        let mask = AttentionMask::padding(&lengths, max_len, &device).unwrap();

        // Test that padding positions are correctly masked
        // ... verification logic
    }

    #[test]
    fn test_mask_value_precision_appropriate() {
        let attention = BitNetAttention::new(128, 8).unwrap();

        // Test FP16 mask value doesn't overflow
        let fp16_mask = attention.get_mask_value_for_dtype(DType::F16).unwrap();
        assert!(fp16_mask > -70000.0 && fp16_mask < -60000.0);

        // Test F32 uses larger range
        let f32_mask = attention.get_mask_value_for_dtype(DType::F32).unwrap();
        assert!(f32_mask < -1e30);
    }
}
```

### Integration Tests
```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    fn test_attention_with_real_model_patterns() {
        // Test with actual BitNet model attention patterns
        // Verify correctness against reference implementation
    }

    #[test]
    fn test_mixed_precision_attention_masking() {
        // Test attention masking works correctly with FP16/BF16
    }
}
```

### Cross-Validation Tests
```rust
#[cfg(feature = "crossval")]
mod crossval_tests {
    #[test]
    fn test_attention_mask_compatibility_with_reference() {
        // Cross-validate attention computation with C++ reference
        // Ensure masked attention outputs match within tolerance
    }
}
```

## Performance Considerations

1. **Memory Efficiency**: Use in-place masking where possible to avoid tensor copies
2. **GPU Optimization**: Implement fused attention+masking kernels for GPU execution
3. **SIMD CPU**: Use vectorized operations for mask application on CPU
4. **Cache Efficiency**: Pre-compute commonly used masks (causal masks for different lengths)

## Risk Assessment

**Low Risk Changes:**
- Adding new mask types and utility functions
- Implementing precision-aware mask values

**Medium Risk Changes:**
- Modifying core attention computation flow
- Changing mask application timing

**Mitigation Strategies:**
- Comprehensive test coverage including edge cases
- Cross-validation against reference implementations
- Performance regression testing
- Feature flag for gradual rollout

## Acceptance Criteria

- [ ] All mask types (causal, padding, custom) work correctly
- [ ] Attention masking works with all supported precisions (FP16, BF16, FP32)
- [ ] Performance regression < 5% compared to current implementation
- [ ] Cross-validation tests pass with reference C++ implementation
- [ ] Comprehensive test coverage (>95% line coverage for attention module)
- [ ] Documentation updated with masking examples and best practices

## Related Issues/PRs

- **Related to**: GPU acceleration optimization efforts
- **Depends on**: Tensor operations performance improvements
- **Blocks**: Advanced model architecture support
- **References**: Attention mechanism standardization across the codebase

## Additional Context

This enhancement is crucial for supporting advanced transformer architectures and ensuring numerical stability across different hardware configurations. The implementation should maintain backward compatibility while providing the foundation for future attention mechanism improvements.
