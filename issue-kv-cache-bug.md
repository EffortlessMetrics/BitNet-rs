# [Bug] KVCache::slice_cache_tensor returns full tensor when seq_len=0 instead of empty tensor

## Problem Description

The `KVCache::slice_cache_tensor` function in `crates/bitnet-inference/src/layers/attention.rs` has incorrect behavior when `seq_len == 0`. Instead of returning an appropriately shaped empty tensor, it returns the full input tensor, which can cause unexpected behavior in attention mechanisms and memory usage calculations.

## Environment

- **File**: `crates/bitnet-inference/src/layers/attention.rs`
- **Function**: `KVCache::slice_cache_tensor`
- **Crate**: `bitnet-inference`
- **Related Components**: Attention layers, KV cache management, tensor operations

## Current Implementation Analysis

The problematic behavior occurs in the zero-length case:

```rust
fn slice_cache_tensor(&self, tensor: &BitNetTensor, seq_len: usize) -> Result<BitNetTensor> {
    if seq_len == 0 {
        return Ok(tensor.clone()); // BUG: Returns full tensor instead of empty tensor
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

## Root Cause Analysis

1. **Incorrect Zero-Length Handling**: When `seq_len == 0`, the function returns `tensor.clone()` instead of an empty tensor
2. **Semantic Inconsistency**: The function name suggests slicing behavior, but zero-length case breaks this contract
3. **Memory Inefficiency**: Returning full tensor when empty tensor expected wastes memory
4. **Potential Attention Bugs**: Attention mechanisms expecting empty tensors may compute incorrect results
5. **Cache Logic Errors**: KV cache management may incorrectly calculate memory usage and cache states

## Impact Assessment

**Severity**: Medium-High - Functional Correctness & Memory Efficiency
**Affected Components**:
- KV cache tensor slicing operations
- Attention layer computations
- Memory usage calculations
- Autoregressive generation with variable sequence lengths

**Potential Issues**:
- Incorrect attention scores when sequence length is zero
- Memory leaks in KV cache management
- Wrong tensor shapes passed to subsequent operations
- Inconsistent behavior between empty and non-empty sequences

## Proposed Solution

### Primary Fix: Return Properly Shaped Empty Tensor

Modify the function to return an appropriately shaped empty tensor when `seq_len == 0`:

```rust
fn slice_cache_tensor(&self, tensor: &BitNetTensor, seq_len: usize) -> Result<BitNetTensor> {
    let tensor_candle = tensor.to_candle()?;
    let shape = tensor_candle.shape();

    if seq_len == 0 {
        // Return empty tensor with correct shape: [0, dim1, dim2, ...]
        let mut empty_shape = shape.dims().to_vec();
        if !empty_shape.is_empty() {
            empty_shape[0] = 0; // Set sequence dimension to 0
        }

        let empty_tensor = Tensor::zeros(
            &empty_shape,
            tensor_candle.dtype(),
            tensor_candle.device()
        ).context("Failed to create empty tensor")?;

        return Ok(BitNetTensor::new(empty_tensor));
    }

    if shape.dims().is_empty() || seq_len >= shape.dims()[0] {
        return Ok(tensor.clone());
    }

    // Slice first dimension to sequence length
    let sliced = tensor_candle.narrow(0, 0, seq_len).context("Failed to slice cache tensor")?;
    Ok(BitNetTensor::new(sliced))
}
```

### Alternative Approach: Early Validation

Add validation and explicit error handling for edge cases:

```rust
fn slice_cache_tensor(&self, tensor: &BitNetTensor, seq_len: usize) -> Result<BitNetTensor> {
    let tensor_candle = tensor.to_candle()?;
    let shape = tensor_candle.shape();

    // Validate input tensor has proper dimensionality
    if shape.dims().is_empty() {
        return Err(anyhow::anyhow!("Cannot slice scalar tensor"));
    }

    let tensor_seq_len = shape.dims()[0];

    match seq_len {
        0 => {
            // Create empty tensor with same shape except sequence dimension
            let mut empty_shape = shape.dims().to_vec();
            empty_shape[0] = 0;

            let empty_tensor = Tensor::zeros(
                &empty_shape,
                tensor_candle.dtype(),
                tensor_candle.device()
            ).context("Failed to create empty cache tensor")?;

            Ok(BitNetTensor::new(empty_tensor))
        }
        len if len >= tensor_seq_len => {
            // Return full tensor if requested length >= actual length
            Ok(tensor.clone())
        }
        len => {
            // Slice to requested length
            let sliced = tensor_candle.narrow(0, 0, len)
                .context("Failed to slice cache tensor")?;
            Ok(BitNetTensor::new(sliced))
        }
    }
}
```

## Implementation Plan

### Phase 1: Bug Fix Implementation
- [ ] Implement correct empty tensor creation for `seq_len == 0` case
- [ ] Add comprehensive error handling and validation
- [ ] Include debug logging for tensor operations
- [ ] Test edge cases (empty inputs, zero lengths, invalid shapes)

### Phase 2: Integration Testing
- [ ] Test integration with attention layer operations
- [ ] Validate KV cache behavior with variable sequence lengths
- [ ] Check memory usage and performance impact
- [ ] Test autoregressive generation scenarios

### Phase 3: Validation & Documentation
- [ ] Add comprehensive unit tests for all edge cases
- [ ] Update function documentation to clarify behavior
- [ ] Add integration tests with attention mechanisms
- [ ] Performance benchmark tensor slicing operations

## Testing Strategy

### Unit Testing
```rust
#[test]
fn test_slice_cache_tensor_zero_length() {
    let kv_cache = create_test_kv_cache();
    let input_tensor = BitNetTensor::new(
        Tensor::ones((4, 8, 64), DType::F32, &Device::Cpu).unwrap()
    );

    let result = kv_cache.slice_cache_tensor(&input_tensor, 0).unwrap();
    let result_candle = result.to_candle().unwrap();

    // Should return empty tensor with shape [0, 8, 64]
    assert_eq!(result_candle.shape().dims(), &[0, 8, 64]);
    assert_eq!(result_candle.elem_count(), 0);
}

#[test]
fn test_slice_cache_tensor_normal_cases() {
    let kv_cache = create_test_kv_cache();
    let input_tensor = BitNetTensor::new(
        Tensor::ones((4, 8, 64), DType::F32, &Device::Cpu).unwrap()
    );

    // Test normal slicing
    let result = kv_cache.slice_cache_tensor(&input_tensor, 2).unwrap();
    assert_eq!(result.to_candle().unwrap().shape().dims(), &[2, 8, 64]);

    // Test full tensor return
    let result = kv_cache.slice_cache_tensor(&input_tensor, 4).unwrap();
    assert_eq!(result.to_candle().unwrap().shape().dims(), &[4, 8, 64]);

    // Test over-length request
    let result = kv_cache.slice_cache_tensor(&input_tensor, 10).unwrap();
    assert_eq!(result.to_candle().unwrap().shape().dims(), &[4, 8, 64]);
}
```

## Related Issues/PRs

- KV cache memory management optimization
- Attention layer edge case handling
- Tensor operation error handling and validation
- Autoregressive generation stability improvements

## Acceptance Criteria

- [ ] `slice_cache_tensor` returns properly shaped empty tensor when `seq_len == 0`
- [ ] Empty tensor has correct shape `[0, dim1, dim2, ...]` preserving all dimensions except sequence
- [ ] Empty tensor preserves original dtype and device
- [ ] Function behavior is consistent across all sequence length cases
- [ ] No memory leaks or excessive allocations
- [ ] Comprehensive error handling for invalid inputs
- [ ] All existing functionality remains unchanged
- [ ] Performance impact is negligible (<1% overhead)
- [ ] Integration with attention layers works correctly
- [ ] Comprehensive test coverage for all edge cases

## Notes

This bug affects the correctness of KV cache operations, particularly in scenarios with variable sequence lengths or when processing empty sequences. The fix should be conservative and maintain backward compatibility while correcting the semantic behavior.
