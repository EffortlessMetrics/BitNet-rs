# [BUG] KV Cache slice_cache_tensor returns full tensor for seq_len=0

## Problem Description

The `KVCache::slice_cache_tensor` function in the attention layer contains a logic bug that returns the full tensor when `seq_len == 0`, potentially violating caller expectations and causing memory inefficiency or incorrect attention computations.

## Environment

- **File**: `crates/bitnet-inference/src/layers/attention.rs`
- **Function**: `KVCache::slice_cache_tensor`
- **Component**: Attention mechanism KV cache management
- **Affected Features**: Both CPU and GPU inference paths
- **MSRV**: Rust 1.90.0

## Root Cause Analysis

The current implementation has a conditional check that returns the full tensor when sequence length is zero:

```rust
fn slice_cache_tensor(&self, tensor: &BitNetTensor, seq_len: usize) -> Result<BitNetTensor> {
    if seq_len == 0 {
        return Ok(tensor.clone()); // BUG: Returns full tensor for seq_len=0
    }
    // ... rest of slicing logic
}
```

**Technical Issues:**
1. **Semantic Inconsistency**: When `seq_len == 0`, callers expect an empty tensor, not the full tensor
2. **Memory Waste**: Returning full tensors when no sequence data is needed wastes memory
3. **Attention Computation Errors**: May cause incorrect attention scores when processing empty sequences
4. **Cache State Corruption**: Could lead to inconsistent KV cache states in multi-step inference

## Impact Assessment

**Severity**: Medium - Affects attention computation correctness
**Affected Components**:
- Multi-head attention layers
- KV cache management
- Incremental inference (autoregressive generation)
- Batch processing with variable sequence lengths

**Performance Impact**:
- Unnecessary memory allocation and copying
- Potential GPU memory pressure from oversized tensors
- Inefficient cache utilization

## Reproduction Steps

1. Create a KV cache instance in attention layer
2. Call `slice_cache_tensor` with `seq_len = 0`
3. Observe that full tensor is returned instead of empty tensor
4. Verify memory usage is higher than expected
5. Check attention computation results for edge cases

**Expected Behavior**: Return empty tensor with appropriate shape
**Actual Behavior**: Returns full tensor clone

## Proposed Solution

### Primary Approach: Create Empty Tensor for seq_len=0

Modify the function to create an appropriately shaped empty tensor when `seq_len == 0`:

```rust
fn slice_cache_tensor(&self, tensor: &BitNetTensor, seq_len: usize) -> Result<BitNetTensor> {
    if seq_len == 0 {
        // Create empty tensor with correct shape for zero sequence length
        let tensor_candle = tensor.to_candle()?;
        let shape = tensor_candle.shape();
        let mut empty_dims = shape.dims().to_vec();

        // Set sequence dimension (first dimension) to 0
        if !empty_dims.is_empty() {
            empty_dims[0] = 0;
        }

        return Ok(BitNetTensor::zeros(
            &empty_dims,
            tensor_candle.dtype(),
            tensor_candle.device()
        )?);
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

### Alternative Approach: Validate Input Parameters

Add input validation to ensure `seq_len == 0` is handled explicitly:

```rust
fn slice_cache_tensor(&self, tensor: &BitNetTensor, seq_len: usize) -> Result<BitNetTensor> {
    let tensor_candle = tensor.to_candle()?;
    let shape = tensor_candle.shape();

    // Handle empty sequence case explicitly
    if seq_len == 0 {
        let mut empty_shape = shape.dims().to_vec();
        if !empty_shape.is_empty() {
            empty_shape[0] = 0; // Zero out sequence dimension
        }
        return BitNetTensor::zeros(&empty_shape, tensor_candle.dtype(), tensor_candle.device());
    }

    // Handle other edge cases
    if shape.dims().is_empty() {
        return Err(anyhow::anyhow!("Cannot slice tensor with empty shape"));
    }

    if seq_len >= shape.dims()[0] {
        return Ok(tensor.clone()); // Return full tensor if seq_len exceeds tensor size
    }

    // Normal slicing path
    let sliced = tensor_candle.narrow(0, 0, seq_len)
        .context("Failed to slice cache tensor")?;
    Ok(BitNetTensor::new(sliced))
}
```

## Implementation Plan

### Phase 1: Fix Implementation (1-2 days)
- [ ] Modify `slice_cache_tensor` to handle `seq_len == 0` correctly
- [ ] Add comprehensive input validation
- [ ] Update function documentation with edge case behavior

### Phase 2: Testing (1 day)
- [ ] Add unit tests for `seq_len == 0` case
- [ ] Add tests for various tensor shapes and sequence lengths
- [ ] Test memory usage to ensure no leaks
- [ ] Add property-based tests for edge cases

### Phase 3: Integration Testing (1 day)
- [ ] Test with multi-head attention layers
- [ ] Verify KV cache consistency in incremental inference
- [ ] Test batch processing with mixed sequence lengths
- [ ] Validate GPU memory usage patterns

### Phase 4: Performance Validation (0.5 days)
- [ ] Benchmark attention computation with various sequence lengths
- [ ] Verify memory efficiency improvements
- [ ] Test with large models and batch sizes

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_cache_tensor_zero_seq_len() {
        let cache = KVCache::new(&config, 1, &device).unwrap();
        let tensor = BitNetTensor::ones(&[10, 64], DType::F32, &device).unwrap();

        let result = cache.slice_cache_tensor(&tensor, 0).unwrap();
        let result_candle = result.to_candle().unwrap();

        assert_eq!(result_candle.shape().dims(), &[0, 64]);
        assert_eq!(result_candle.elem_count(), 0);
    }

    #[test]
    fn test_slice_cache_tensor_normal_cases() {
        let cache = KVCache::new(&config, 1, &device).unwrap();
        let tensor = BitNetTensor::ones(&[10, 64], DType::F32, &device).unwrap();

        // Test normal slicing
        let result = cache.slice_cache_tensor(&tensor, 5).unwrap();
        assert_eq!(result.to_candle().unwrap().shape().dims(), &[5, 64]);

        // Test seq_len >= tensor size
        let result = cache.slice_cache_tensor(&tensor, 15).unwrap();
        assert_eq!(result.to_candle().unwrap().shape().dims(), &[10, 64]);
    }
}
```

### Integration Tests
```rust
#[test]
fn test_attention_with_zero_sequence() {
    let attention = MultiHeadAttention::new(&config).unwrap();
    let empty_input = BitNetTensor::zeros(&[1, 0, 512], DType::F32, &device).unwrap();

    let result = attention.forward(&empty_input, None).unwrap();
    assert_eq!(result.to_candle().unwrap().shape().dims(), &[1, 0, 512]);
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] `slice_cache_tensor` returns empty tensor when `seq_len == 0`
- [ ] Empty tensor has correct shape (sequence dimension = 0, other dimensions preserved)
- [ ] Function handles all edge cases without panicking
- [ ] Memory usage is optimal for zero-length sequences

### Performance Requirements
- [ ] No memory leaks in empty tensor creation
- [ ] Minimal performance overhead for edge case handling
- [ ] GPU memory usage reduced for zero-length sequences

### Quality Requirements
- [ ] Comprehensive test coverage (>95%) including edge cases
- [ ] Clear documentation of function behavior
- [ ] No regressions in existing attention functionality
- [ ] Cross-validation with C++ reference implementation passes

## Related Issues

- KV cache management optimization (#TBD)
- Attention layer performance improvements (#TBD)
- Memory efficiency for variable sequence lengths (#TBD)

## Labels

`bug`, `attention`, `kv-cache`, `memory-efficiency`, `medium-priority`

## Definition of Done

- [ ] Code changes implemented and reviewed
- [ ] All tests pass including new edge case tests
- [ ] Documentation updated with function behavior clarification
- [ ] Performance benchmarks show no regression
- [ ] Cross-validation tests pass
- [ ] Memory usage verified to be optimal