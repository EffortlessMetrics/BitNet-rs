# Potential bug: `KVCache::slice_cache_tensor` in `attention.rs` returns full tensor for `seq_len == 0`

The `KVCache::slice_cache_tensor` function in `crates/bitnet-inference/src/layers/attention.rs` returns the full tensor if `seq_len == 0`. This might be an issue if `seq_len` is 0 and the caller expects an empty tensor.

**File:** `crates/bitnet-inference/src/layers/attention.rs`

**Function:** `KVCache::slice_cache_tensor`

**Code:**
```rust
    fn slice_cache_tensor(&self, tensor: &BitNetTensor, seq_len: usize) -> Result<BitNetTensor> {
        if seq_len == 0 {
            return Ok(tensor.clone()); // Return full tensor if no slicing needed
        }

        let tensor_candle = tensor.to_candle()?;
        let shape = tensor_candle.shape();

        if shape.dims().is_empty() || seq_len >= shape.dims()[0] {
            return Ok(tensor.clone());
        }

        // Slice first dimension to sequence length
        let sliced = tensor_candle.narrow(0, 0, seq_len).context("Failed to slice cache tensor")?;
        Ok(BitNetTensor::new(sliced))
    }
```

## Proposed Fix

The `KVCache::slice_cache_tensor` function should return an empty tensor if `seq_len` is 0. This will ensure that the function behaves as expected when `seq_len` is 0.

### Example Implementation

```rust
    fn slice_cache_tensor(&self, tensor: &BitNetTensor, seq_len: usize) -> Result<BitNetTensor> {
        if seq_len == 0 {
            // Return an empty tensor with the correct shape
            let tensor_candle = tensor.to_candle()?;
            let shape = tensor_candle.shape();
            let empty_shape = &[0; shape.dims().len()]; // Create an empty shape
            return Ok(BitNetTensor::zeros(empty_shape, tensor_candle.dtype(), tensor_candle.device())?);
        }

        // ... rest of the function ...
    }
```
