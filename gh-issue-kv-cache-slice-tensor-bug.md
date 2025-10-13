# [BUG] KV Cache slice_cache_tensor Returns Full Tensor for seq_len=0 Instead of Empty Tensor

## Summary

The `KVCache::slice_cache_tensor` method in `crates/bitnet-inference/src/layers/attention.rs` has incorrect behavior when `seq_len == 0`. It returns the full tensor instead of an appropriately shaped empty tensor, which can cause memory inefficiency, incorrect attention computations, and inconsistent behavior in autoregressive generation scenarios.

## Environment

- **File**: `crates/bitnet-inference/src/layers/attention.rs`
- **Method**: `KVCache::slice_cache_tensor` (lines 87-102)
- **Affected Components**: Multi-head attention, autoregressive generation, KV caching
- **Feature Flags**: All (affects both `cpu` and `gpu` features)
- **MSRV**: 1.90.0+

## Problem Description

### Current Behavior
```rust
fn slice_cache_tensor(&self, tensor: &BitNetTensor, seq_len: usize) -> Result<BitNetTensor> {
    if seq_len == 0 {
        return Ok(tensor.clone()); // ❌ Returns full tensor - INCORRECT
    }
    // ... rest of implementation
}
```

When `seq_len == 0`, the method returns a clone of the full input tensor instead of returning an empty tensor with appropriate dimensions.

### Expected Behavior
When `seq_len == 0`, the method should return an empty tensor that:
1. Has the same number of dimensions as the input tensor
2. Has 0 elements in the sequence dimension (typically the first dimension)
3. Preserves other dimensions (heads, features) from the original tensor
4. Uses the same dtype and device as the input tensor

## Root Cause Analysis

### Technical Investigation

1. **Memory Management Issue**: Returning the full tensor for `seq_len=0` wastes memory and defeats the purpose of slicing for empty sequences.

2. **Attention Computation Impact**: In autoregressive generation:
   - Initial generation step may have `seq_len=0` (no previous tokens)
   - Returning full tensor instead of empty affects attention score computation
   - Can lead to incorrect attention patterns or unexpected behavior

3. **Cache Consistency Problem**: The KV cache system expects sliced tensors to reflect actual sequence lengths:
   - `current_len = 0` should correspond to empty cache tensors
   - Inconsistent tensor shapes between cache states can cause errors

4. **Tensor Operation Assumptions**: Downstream operations may assume tensor dimensions match actual sequence lengths:
   - Concatenation operations may fail with mismatched dimensions
   - Memory allocation calculations become incorrect

### Code Flow Analysis

The bug manifests in this call chain:
```
KVCache::get() -> slice_cache_tensor() -> (incorrect empty tensor handling)
                     ↓
BitNetAttention::forward() -> (receives incorrectly sized tensors)
                     ↓
Attention computation with wrong tensor dimensions
```

## Impact Assessment

### Severity: **Medium-High**
- **Memory Impact**: Unnecessary memory usage in edge cases
- **Correctness Impact**: Potential incorrect attention computation results
- **Performance Impact**: Wasted compute cycles on unnecessary tensor operations
- **Reliability Impact**: Inconsistent behavior between different sequence lengths

### Affected Use Cases
1. **Autoregressive Generation**: Initial token generation with empty cache
2. **Batch Processing**: Sequences with zero-length prefix handling
3. **Dynamic Caching**: Cache invalidation scenarios requiring empty state
4. **Testing**: Unit tests expecting correct empty tensor behavior

## Proposed Solution

### Primary Solution: Return Properly Shaped Empty Tensor

```rust
fn slice_cache_tensor(&self, tensor: &BitNetTensor, seq_len: usize) -> Result<BitNetTensor> {
    let tensor_candle = tensor.to_candle()?;
    let shape = tensor_candle.shape();

    if seq_len == 0 {
        // Create empty tensor with correct shape: [0, ...other_dims]
        let mut empty_shape = shape.dims().to_vec();
        if !empty_shape.is_empty() {
            empty_shape[0] = 0; // Set sequence dimension to 0
        }

        return BitNetTensor::zeros(&empty_shape, tensor_candle.dtype(), tensor_candle.device())
            .context("Failed to create empty cache tensor");
    }

    if shape.dims().is_empty() || seq_len >= shape.dims()[0] {
        return Ok(tensor.clone());
    }

    // Slice first dimension to sequence length
    let sliced = tensor_candle.narrow(0, 0, seq_len)
        .context("Failed to slice cache tensor")?;
    Ok(BitNetTensor::new(sliced))
}
```

### Alternative Solutions Considered

#### Option B: Early Return with Validation
```rust
if seq_len == 0 {
    // Validate input tensor has proper shape for slicing
    if shape.dims().is_empty() {
        return Err(anyhow::anyhow!("Cannot slice empty-dimensional tensor"));
    }

    // Return zero-length slice of first dimension
    let empty_slice = tensor_candle.narrow(0, 0, 0)
        .context("Failed to create empty slice")?;
    return Ok(BitNetTensor::new(empty_slice));
}
```

#### Option C: Dedicated Empty Tensor Factory
```rust
// Add helper method to KVCache
fn create_empty_cache_tensor(&self, reference_tensor: &BitNetTensor) -> Result<BitNetTensor> {
    let candle_tensor = reference_tensor.to_candle()?;
    let shape = candle_tensor.shape();

    if shape.dims().is_empty() {
        return BitNetTensor::zeros(&[0], candle_tensor.dtype(), candle_tensor.device());
    }

    let mut empty_shape = shape.dims().to_vec();
    empty_shape[0] = 0;
    BitNetTensor::zeros(&empty_shape, candle_tensor.dtype(), candle_tensor.device())
}

// In slice_cache_tensor:
if seq_len == 0 {
    return self.create_empty_cache_tensor(tensor);
}
```

**Recommendation**: Use **Primary Solution** as it's most direct and maintains consistency with existing patterns.

## Implementation Plan

### Phase 1: Core Fix Implementation
- [ ] **Task 1.1**: Implement corrected `slice_cache_tensor` method
  - Update logic for `seq_len == 0` case
  - Add proper error handling for edge cases
  - Ensure consistent device/dtype handling

- [ ] **Task 1.2**: Add comprehensive unit tests
  - Test `seq_len == 0` behavior with various tensor shapes
  - Test edge cases: empty input tensors, single-dimension tensors
  - Test device consistency (CPU/GPU)
  - Test dtype preservation

### Phase 2: Integration Testing
- [ ] **Task 2.1**: Update KV cache integration tests
  - Test cache behavior with zero-length sequences
  - Verify memory usage efficiency
  - Test interaction with attention mechanisms

- [ ] **Task 2.2**: Add autoregressive generation tests
  - Test initial generation step (empty cache)
  - Verify correct attention computation with empty KV cache
  - Test cache state transitions from empty to populated

### Phase 3: Performance & Memory Validation
- [ ] **Task 3.1**: Memory usage benchmarks
  - Compare memory usage before/after fix
  - Validate no memory leaks in empty tensor handling
  - Test memory pooling efficiency with corrected behavior

- [ ] **Task 3.2**: Cross-validation testing
  - Verify behavior matches C++ reference implementation
  - Test against attention accuracy benchmarks
  - Validate quantization accuracy with corrected cache behavior

### Phase 4: Documentation & Error Handling
- [ ] **Task 4.1**: Update documentation
  - Document expected behavior for empty sequence lengths
  - Add examples of correct usage patterns
  - Update KV cache architecture documentation

- [ ] **Task 4.2**: Enhance error handling
  - Add meaningful error messages for invalid inputs
  - Add debug logging for cache operations
  - Improve error context for troubleshooting

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod kv_cache_tests {
    use super::*;
    use bitnet_common::{Device, Tensor};

    #[test]
    fn test_slice_cache_tensor_empty_sequence() -> Result<()> {
        let kv_cache = KVCache::new(100, 1, 8, 64, &Device::Cpu)?;

        // Create test tensor [10, 8, 64]
        let test_tensor = BitNetTensor::zeros(&[10, 8, 64], DType::F32, &Device::Cpu)?;

        // Test seq_len = 0
        let result = kv_cache.slice_cache_tensor(&test_tensor, 0)?;

        // Should return empty tensor with shape [0, 8, 64]
        assert_eq!(result.shape(), &[0, 8, 64]);
        assert_eq!(result.dtype(), DType::F32);
        assert_eq!(result.device(), &Device::Cpu);

        Ok(())
    }

    #[test]
    fn test_slice_cache_tensor_various_shapes() -> Result<()> {
        let kv_cache = KVCache::new(100, 1, 8, 64, &Device::Cpu)?;

        // Test different tensor shapes
        let shapes = vec![
            vec![5, 8, 64],      // Standard 3D
            vec![1, 8, 64],      // Single sequence
            vec![0, 8, 64],      // Already empty
            vec![5, 1, 1],       // Minimal dimensions
        ];

        for shape in shapes {
            let tensor = BitNetTensor::zeros(&shape, DType::F32, &Device::Cpu)?;
            let result = kv_cache.slice_cache_tensor(&tensor, 0)?;

            let mut expected_shape = shape.clone();
            if !expected_shape.is_empty() {
                expected_shape[0] = 0;
            }

            assert_eq!(result.shape(), expected_shape);
        }

        Ok(())
    }

    #[test]
    fn test_slice_cache_tensor_memory_efficiency() -> Result<()> {
        let kv_cache = KVCache::new(1000, 1, 32, 128, &Device::Cpu)?;

        // Large tensor
        let large_tensor = BitNetTensor::zeros(&[1000, 32, 128], DType::F32, &Device::Cpu)?;

        // Slice to empty should not reference large tensor data
        let empty_result = kv_cache.slice_cache_tensor(&large_tensor, 0)?;

        // Verify empty tensor is actually empty (0 elements)
        assert_eq!(empty_result.shape()[0], 0);

        // Memory usage should be minimal for empty tensor
        let empty_elements: usize = empty_result.shape().iter().product();
        assert_eq!(empty_elements, 0);

        Ok(())
    }
}
```

### Integration Tests
```rust
#[tokio::test]
async fn test_attention_with_empty_kv_cache() -> Result<()> {
    // Test attention computation with initially empty KV cache
    let config = AttentionConfig::default();
    let mut kv_cache = KVCache::new(2048, 12, config.num_attention_heads, config.head_dim, &Device::Cpu)?;

    // Create attention layer (placeholder - actual implementation needed)
    // let attention = BitNetAttention::new(config, weights...)?;

    // Input for single token (initial generation step)
    let input = BitNetTensor::zeros(&[1, 1, config.hidden_size], DType::F32, &Device::Cpu)?;

    // Forward pass should handle empty cache correctly
    // let output = attention.forward(&input, None, None, Some(&mut kv_cache), 0).await?;

    // Verify cache is populated correctly after forward pass
    let (k_cache, v_cache) = kv_cache.get(0)?;

    // Cache should now contain data for the single token
    assert_eq!(k_cache.shape()[0], 1); // sequence length = 1
    assert_eq!(v_cache.shape()[0], 1);

    Ok(())
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] **AC1**: `slice_cache_tensor(tensor, 0)` returns empty tensor with shape `[0, ...other_dims]`
- [ ] **AC2**: Empty tensor preserves original tensor's dtype and device
- [ ] **AC3**: Method handles all valid tensor shapes correctly (1D, 2D, 3D, 4D)
- [ ] **AC4**: No memory leaks or excessive memory usage for empty tensors
- [ ] **AC5**: Integration with attention mechanism works correctly for empty cache

### Performance Requirements
- [ ] **AC6**: Empty tensor creation is efficient (< 1ms for typical shapes)
- [ ] **AC7**: Memory usage for empty tensors is minimal (proportional to header size only)
- [ ] **AC8**: No performance regression for non-zero sequence lengths

### Quality Requirements
- [ ] **AC9**: All existing tests continue to pass
- [ ] **AC10**: New unit tests cover edge cases and error conditions
- [ ] **AC11**: Integration tests validate end-to-end behavior
- [ ] **AC12**: Cross-validation tests confirm correctness against reference

### Error Handling Requirements
- [ ] **AC13**: Clear error messages for invalid inputs
- [ ] **AC14**: Graceful handling of empty or malformed input tensors
- [ ] **AC15**: Proper context in error reporting for debugging

## Related Issues and Components

### Related Components
- **Primary**: `crates/bitnet-inference/src/layers/attention.rs`
- **Secondary**: `crates/bitnet-common/src/tensor.rs` (BitNetTensor implementation)
- **Testing**: `crates/bitnet-inference/tests/ac2_multi_head_attention.rs`
- **Server**: `crates/bitnet-server/src/caching/kv_cache.rs` (server-side caching)

### Cross-References
- **Issue #248**: Multi-head attention implementation
- **Issue #251**: Production-ready inference server
- **AC2 Test Suite**: Multi-head attention mechanism tests
- **KV Cache Architecture**: `docs/gpu-kernel-architecture.md`

### Dependencies
- **BitNet Quantization**: Ensure quantized tensors work correctly with empty tensors
- **Device Abstraction**: CPU/GPU device handling for empty tensors
- **Memory Management**: Efficient allocation/deallocation of empty tensors

## Labels and Classification

**Labels**: `bug`, `priority-high`, `component-inference`, `component-attention`, `memory-management`, `correctness`

**Priority**: High (affects correctness of attention mechanism)

**Difficulty**: Medium (clear fix, but requires thorough testing)

**Area**: Core Inference Engine

---

**Assignee**: TBD
**Milestone**: TBD
**Estimated Effort**: 2-3 days (implementation + testing)
