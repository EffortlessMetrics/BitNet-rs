# [Attention] Enhance attention mask application with robust edge case handling

## Problem Description

The `BitNetAttention::compute_attention` function uses simplified attention mask broadcasting that may not handle complex masking scenarios, causal attention patterns, or advanced attention mechanisms required for production-grade inference.

## Root Cause Analysis

### Current Implementation
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

### Issues Identified
1. **Simplified Masking**: Basic broadcast addition doesn't handle all mask types
2. **Fixed Mask Value**: -1e9 may not be optimal for all data types
3. **No Causal Masking**: Missing built-in causal attention support
4. **Limited Flexibility**: Doesn't support different masking strategies

## Proposed Solution

### Enhanced Attention Masking

```rust
impl BitNetAttention {
    pub fn compute_attention(
        &self,
        query: &BitNetTensor,
        key: &BitNetTensor,
        value: &BitNetTensor,
        attention_mask: Option<&AttentionMask>,
        position_ids: Option<&[usize]>,
    ) -> Result<BitNetTensor> {
        let attention_scores = self.compute_attention_scores(query, key)?;
        let masked_scores = self.apply_comprehensive_masking(
            attention_scores,
            attention_mask,
            position_ids,
            query.size(1), // sequence length
        )?;

        let attention_weights = softmax(&masked_scores, -1)?;
        let context = attention_weights.matmul(&value.to_candle()?)?;

        Ok(BitNetTensor::from_candle(context))
    }

    fn apply_comprehensive_masking(
        &self,
        scores: CandleTensor,
        attention_mask: Option<&AttentionMask>,
        position_ids: Option<&[usize]>,
        seq_len: usize,
    ) -> Result<CandleTensor> {
        let mut masked_scores = scores;

        // Apply causal masking if enabled
        if self.config.is_causal {
            masked_scores = self.apply_causal_mask(masked_scores, seq_len)?;
        }

        // Apply custom attention mask if provided
        if let Some(mask) = attention_mask {
            masked_scores = self.apply_attention_mask(masked_scores, mask)?;
        }

        // Apply position-based masking if needed
        if let Some(positions) = position_ids {
            masked_scores = self.apply_position_mask(masked_scores, positions)?;
        }

        Ok(masked_scores)
    }

    fn apply_causal_mask(&self, scores: CandleTensor, seq_len: usize) -> Result<CandleTensor> {
        let mask_value = self.get_optimal_mask_value(&scores.dtype())?;
        let causal_mask = self.create_causal_mask(seq_len, scores.device())?;

        scores.where_cond(&causal_mask, &mask_value)
            .context("Failed to apply causal mask")
    }

    fn apply_attention_mask(&self, scores: CandleTensor, mask: &AttentionMask) -> Result<CandleTensor> {
        match mask {
            AttentionMask::Boolean(bool_mask) => {
                self.apply_boolean_mask(scores, bool_mask)
            },
            AttentionMask::Additive(add_mask) => {
                self.apply_additive_mask(scores, add_mask)
            },
            AttentionMask::Multiplicative(mult_mask) => {
                self.apply_multiplicative_mask(scores, mult_mask)
            },
        }
    }

    fn get_optimal_mask_value(&self, dtype: &DType) -> Result<CandleTensor> {
        let mask_value = match dtype {
            DType::F16 => -65504.0, // Near -inf for f16
            DType::F32 => -1e38,    // Near -inf for f32
            DType::BF16 => -3.4e38, // Near -inf for bf16
            _ => -1e9,              // Conservative fallback
        };

        Ok(Tensor::new(&[mask_value], &Device::Cpu)?)
    }

    fn create_causal_mask(&self, seq_len: usize, device: &Device) -> Result<CandleTensor> {
        // Create lower triangular mask for causal attention
        let mut mask_data = vec![false; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask_data[i * seq_len + j] = true;
            }
        }

        let mask_tensor = Tensor::from_slice(&mask_data, (seq_len, seq_len), device)?;
        Ok(mask_tensor)
    }
}

#[derive(Debug, Clone)]
pub enum AttentionMask {
    Boolean(CandleTensor),      // True for valid positions
    Additive(CandleTensor),     // Added to attention scores
    Multiplicative(CandleTensor), // Multiplied with attention scores
}

#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub is_causal: bool,
    pub mask_strategy: MaskStrategy,
    pub attention_dropout: f64,
    pub scale_factor: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum MaskStrategy {
    Standard,
    Optimized,
    Custom(Box<dyn MaskApplier>),
}

pub trait MaskApplier: Send + Sync {
    fn apply_mask(&self, scores: CandleTensor, mask: &AttentionMask) -> Result<CandleTensor>;
}
```

## Acceptance Criteria

- [ ] Robust causal attention masking
- [ ] Support for multiple mask types (boolean, additive, multiplicative)
- [ ] Data type-aware mask value optimization
- [ ] Position-based masking capabilities
- [ ] Comprehensive test coverage for edge cases

## Priority: Medium

Improves attention mechanism robustness and supports advanced transformer architectures.
