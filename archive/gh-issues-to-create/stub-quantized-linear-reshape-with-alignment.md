# Stub code: `QuantizedLinear::reshape_with_alignment` in `quantized_linear.rs` has a placeholder for padding

The `QuantizedLinear::reshape_with_alignment` function in `crates/bitnet-inference/src/layers/quantized_linear.rs` has a comment "Note: In a full implementation, this would add actual padding". It doesn't actually add alignment padding. This is a form of stubbing.

**File:** `crates/bitnet-inference/src/layers/quantized_linear.rs`

**Function:** `QuantizedLinear::reshape_with_alignment`

**Code:**
```rust
    fn reshape_with_alignment(
        &self,
        tensor: &candle_core::Tensor,
        batch_size: usize,
        feature_size: usize,
    ) -> Result<candle_core::Tensor> {
        let reshaped = tensor
            .reshape(&[batch_size, feature_size])
            .context("Failed to reshape tensor with alignment")?;

        // Add alignment padding if needed for vectorization
        if self.alignment_padding > 0 {
            // Note: In a full implementation, this would add actual padding
            // For now, we just return the reshaped tensor
        }

        Ok(reshaped)
    }
```

## Proposed Fix

The `QuantizedLinear::reshape_with_alignment` function should be implemented to add actual alignment padding if needed for vectorization. This would involve creating a new tensor with padding and copying the original tensor's data into it.

### Example Implementation

```rust
    fn reshape_with_alignment(
        &self,
        tensor: &candle_core::Tensor,
        batch_size: usize,
        feature_size: usize,
    ) -> Result<candle_core::Tensor> {
        let reshaped = tensor
            .reshape(&[batch_size, feature_size])
            .context("Failed to reshape tensor with alignment")?;

        // Add alignment padding if needed for vectorization
        if self.alignment_padding > 0 {
            let padded_size = feature_size + self.alignment_padding;
            let mut padded_data = vec![0.0; batch_size * padded_size];
            for i in 0..batch_size {
                padded_data[i * padded_size..(i * padded_size) + feature_size]
                    .copy_from_slice(&reshaped.to_vec2::<f32>()?[i]);
            }
            candle_core::Tensor::from_vec(padded_data, &[batch_size, padded_size], tensor.device())
                .context("Failed to create padded tensor")
        } else {
            Ok(reshaped)
        }
    }
```
