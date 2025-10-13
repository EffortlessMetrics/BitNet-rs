# [Bug] KVCache::slice_cache_tensor returns full tensor for seq_len=0 instead of empty tensor

## Problem Description

The `KVCache::slice_cache_tensor` function in `crates/bitnet-inference/src/layers/attention.rs` returns the full tensor when `seq_len == 0`, which may be unexpected behavior when callers expect an empty tensor for zero-length sequences.

## Environment

- **File**: `crates/bitnet-inference/src/layers/attention.rs`
- **Function**: `KVCache::slice_cache_tensor`
- **Component**: Attention layer and KV cache management
- **Bug Type**: Logic error in tensor slicing
- **MSRV**: Rust 1.90.0

## Reproduction Steps

1. Call `slice_cache_tensor` with `seq_len = 0`
2. Observe that the full tensor is returned instead of an empty tensor
3. Note potential inconsistency with caller expectations

```rust
let cache = KVCache::new(config)?;
let tensor = BitNetTensor::zeros(&[10, 512], DType::F32, &Device::Cpu)?;
let result = cache.slice_cache_tensor(&tensor, 0)?;

// Expected: Empty tensor with shape [0, 512]
// Actual: Full tensor with shape [10, 512]
```

## Root Cause Analysis

```rust
fn slice_cache_tensor(&self, tensor: &BitNetTensor, seq_len: usize) -> Result<BitNetTensor> {
    if seq_len == 0 {
        return Ok(tensor.clone()); // Returns full tensor - potentially incorrect
    }
    // ... rest of slicing logic
}
```

**Issues:**
- Inconsistent behavior for zero-length sequences
- May cause unexpected memory usage
- Could lead to computation errors in attention mechanisms
- Breaks the contract that slicing with 0 length should return empty

## Proposed Solution

Return appropriately shaped empty tensor for zero-length sequences:

```rust
fn slice_cache_tensor(&self, tensor: &BitNetTensor, seq_len: usize) -> Result<BitNetTensor> {
    if seq_len == 0 {
        // Return empty tensor with correct shape
        let tensor_candle = tensor.to_candle()?;
        let shape = tensor_candle.shape();

        if shape.dims().is_empty() {
            return Ok(tensor.clone()); // Keep scalar tensors as-is
        }

        // Create empty tensor: [0, dim1, dim2, ...]
        let mut empty_dims = shape.dims().to_vec();
        empty_dims[0] = 0;

        let empty_tensor = BitNetTensor::zeros(
            &empty_dims,
            tensor_candle.dtype(),
            tensor_candle.device()
        )?;

        return Ok(empty_tensor);
    }

    let tensor_candle = tensor.to_candle()?;
    let shape = tensor_candle.shape();

    if shape.dims().is_empty() || seq_len >= shape.dims()[0] {
        return Ok(tensor.clone());
    }

    // Slice first dimension to sequence length
    let sliced = tensor_candle.narrow(0, 0, seq_len)
        .context("Failed to slice cache tensor")?;
    Ok(BitNetTensor::new(sliced))
}
```

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_cache_tensor_zero_length() {
        let cache = KVCache::new(KVCacheConfig::default()).unwrap();
        let tensor = BitNetTensor::zeros(&[10, 512], DType::F32, &Device::Cpu).unwrap();

        let result = cache.slice_cache_tensor(&tensor, 0).unwrap();

        // Should return empty tensor with shape [0, 512]
        assert_eq!(result.shape(), &[0, 512]);
        assert_eq!(result.dtype(), DType::F32);
    }

    #[test]
    fn test_slice_cache_tensor_normal_cases() {
        let cache = KVCache::new(KVCacheConfig::default()).unwrap();
        let tensor = BitNetTensor::zeros(&[10, 512], DType::F32, &Device::Cpu).unwrap();

        // Test partial slice
        let result = cache.slice_cache_tensor(&tensor, 5).unwrap();
        assert_eq!(result.shape(), &[5, 512]);

        // Test full slice
        let result = cache.slice_cache_tensor(&tensor, 10).unwrap();
        assert_eq!(result.shape(), &[10, 512]);

        // Test over-slice (should return full tensor)
        let result = cache.slice_cache_tensor(&tensor, 15).unwrap();
        assert_eq!(result.shape(), &[10, 512]);
    }
}
```

## Impact Assessment

**Severity**: Low-Medium - May cause unexpected behavior but not crashes
**Type**: Logic bug fix

**Affected Areas:**
- Attention computation with variable sequence lengths
- Memory usage in KV cache operations
- Batch processing with mixed sequence lengths

## Labels

`bug`, `attention`, `kv-cache`, `tensor-operations`, `low-priority`
