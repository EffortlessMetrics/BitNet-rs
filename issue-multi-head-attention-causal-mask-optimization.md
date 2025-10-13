# [SIMULATION] Multi-Head Attention Causal Mask Creation Uses Inefficient Vector Manipulation

## Problem Description

The `MultiHeadAttention::create_causal_mask` function in `crates/bitnet-models/src/transformer.rs` creates causal masks using inefficient vector manipulation with nested loops and individual element assignment. This approach is significantly slower than optimized tensor operations and creates unnecessary performance bottlenecks in attention computation.

## Environment

- **File**: `crates/bitnet-models/src/transformer.rs`
- **Function**: `MultiHeadAttention::create_causal_mask` (lines 12-24)
- **Component**: Multi-head attention mechanism
- **Build Configuration**: All feature configurations
- **Context**: Transformer model forward passes with causal attention

## Root Cause Analysis

### Technical Issues

1. **Inefficient Vector Manipulation**:
   ```rust
   let mut mask_vec = vec![0.0f32; q_len * k_len];
   for i in 0..q_len {
       let start = past_len + i + 1;
       for j in start..k_len {
           mask_vec[i * k_len + j] = f32::NEG_INFINITY;  // Individual assignment
       }
   }
   ```
   - Nested loops for element-wise assignment
   - No vectorization or parallel processing
   - Poor cache performance with scattered memory writes

2. **Missing Tensor Operation Optimization**:
   - Could use efficient tensor creation and masking operations
   - No utilization of Candle's optimized tensor operations
   - Missing batch processing for multiple sequences

3. **Suboptimal Memory Usage**:
   - Creates full-size vector before tensor conversion
   - Unnecessary intermediate allocations
   - No memory reuse for repeated mask creation

4. **Lack of Caching Opportunities**:
   - Recreates identical masks for same sequence lengths
   - No static mask precomputation for common sizes
   - Missing optimization for incremental generation

### Impact Assessment

- **Performance**: Significant overhead in attention computation
- **Memory**: Unnecessary allocations and poor memory access patterns
- **Scalability**: Poor performance with large sequence lengths
- **Efficiency**: Suboptimal compared to tensor-native operations

## Reproduction Steps

1. Run transformer inference with attention masking:
   ```bash
   cargo run -p bitnet-cli -- infer --model model.gguf --prompt "test sequence"
   ```

2. Profile attention mask creation performance:
   ```rust
   let device = Device::Cpu;
   let attention = MultiHeadAttention::new(config, &device)?;

   let start = std::time::Instant::now();
   let mask = attention.create_causal_mask(512, 512, &device)?;
   let duration = start.elapsed();
   // Observe poor performance vs optimized tensor operations
   ```

3. **Expected**: Efficient tensor-based mask creation
4. **Actual**: Slow vector manipulation with nested loops

## Proposed Solution

### Primary Approach: Tensor-Native Causal Mask Creation

Implement efficient causal mask creation using optimized tensor operations:

```rust
impl MultiHeadAttention {
    fn create_causal_mask(&self, q_len: usize, k_len: usize, device: &Device) -> Result<Tensor> {
        // Use efficient tensor operations instead of vector manipulation
        self.create_causal_mask_optimized(q_len, k_len, device)
    }

    fn create_causal_mask_optimized(&self, q_len: usize, k_len: usize, device: &Device) -> Result<Tensor> {
        let past_len = k_len.saturating_sub(q_len);

        // Method 1: Use tensor operations with triu
        if q_len == k_len && past_len == 0 {
            // Standard causal mask - use efficient triu operation
            let ones = Tensor::ones(&[q_len, k_len], DType::F32, device)?;
            let mask = ones.triu(1)?; // Upper triangular with offset 1
            let neg_inf = Tensor::full(f32::NEG_INFINITY, &[q_len, k_len], device)?;
            let zeros = Tensor::zeros(&[q_len, k_len], DType::F32, device)?;

            // Convert 1s to NEG_INFINITY, 0s stay as 0
            mask.where_cond(&neg_inf, &zeros)
        } else {
            // Complex case with past tokens - use optimized indexing
            self.create_causal_mask_with_past(q_len, k_len, past_len, device)
        }
    }

    fn create_causal_mask_with_past(
        &self,
        q_len: usize,
        k_len: usize,
        past_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        // Create position indices tensors
        let q_indices = Tensor::arange(0, q_len as i64, device)?.unsqueeze(1)?; // [q_len, 1]
        let k_indices = Tensor::arange(0, k_len as i64, device)?.unsqueeze(0)?; // [1, k_len]

        // Compute allowed positions: k_pos <= past_len + q_pos
        let past_offset = Tensor::full(past_len as i64, &[1, 1], device)?;
        let allowed_positions = k_indices.broadcast_le(&q_indices.broadcast_add(&past_offset)?)?;

        // Convert boolean mask to float mask
        let zeros = Tensor::zeros(&[q_len, k_len], DType::F32, device)?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, &[q_len, k_len], device)?;

        // Use where operation: allowed ? 0.0 : NEG_INFINITY
        allowed_positions.where_cond(&zeros, &neg_inf)
    }

    // Optimized version with caching for common patterns
    fn create_causal_mask_cached(&self, q_len: usize, k_len: usize, device: &Device) -> Result<Tensor> {
        // Check cache for precomputed masks
        if let Some(cached_mask) = self.mask_cache.get(&(q_len, k_len)) {
            return Ok(cached_mask.to_device(device)?);
        }

        let mask = self.create_causal_mask_optimized(q_len, k_len, device)?;

        // Cache mask for future use (up to reasonable size limits)
        if q_len <= 2048 && k_len <= 2048 {
            self.mask_cache.insert((q_len, k_len), mask.clone());
        }

        Ok(mask)
    }

    // SIMD-optimized version for CPU
    #[cfg(target_arch = "x86_64")]
    fn create_causal_mask_simd(&self, q_len: usize, k_len: usize, device: &Device) -> Result<Tensor> {
        if !device.is_cpu() {
            return self.create_causal_mask_optimized(q_len, k_len, device);
        }

        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx") {
            unsafe { self.create_causal_mask_avx(q_len, k_len, device) }
        } else {
            self.create_causal_mask_optimized(q_len, k_len, device)
        }
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn create_causal_mask_avx(&self, q_len: usize, k_len: usize, device: &Device) -> Result<Tensor> {
        let mut mask_data = vec![0.0f32; q_len * k_len];
        let past_len = k_len.saturating_sub(q_len);

        // Process 8 elements at a time with AVX
        let neg_inf_vec = _mm256_set1_ps(f32::NEG_INFINITY);

        for i in 0..q_len {
            let start = past_len + i + 1;
            let row_offset = i * k_len;
            let mut j = start;

            // Vectorized processing for the row
            while j + 8 <= k_len {
                _mm256_storeu_ps(
                    mask_data.as_mut_ptr().add(row_offset + j),
                    neg_inf_vec,
                );
                j += 8;
            }

            // Handle remaining elements
            for k in j..k_len {
                mask_data[row_offset + k] = f32::NEG_INFINITY;
            }
        }

        Ok(Tensor::from_vec(mask_data, &[q_len, k_len], device)?)
    }

    // GPU-optimized version
    #[cfg(feature = "gpu")]
    fn create_causal_mask_gpu(&self, q_len: usize, k_len: usize, device: &Device) -> Result<Tensor> {
        if !device.is_cuda() {
            return self.create_causal_mask_optimized(q_len, k_len, device);
        }

        // Use CUDA kernel for large masks
        if q_len * k_len > 16384 {
            self.create_causal_mask_cuda_kernel(q_len, k_len, device)
        } else {
            self.create_causal_mask_optimized(q_len, k_len, device)
        }
    }

    #[cfg(feature = "gpu")]
    fn create_causal_mask_cuda_kernel(&self, q_len: usize, k_len: usize, device: &Device) -> Result<Tensor> {
        // Launch CUDA kernel for parallel mask creation
        // Implementation would use cuBLAS or custom CUDA kernel
        use candle_core::cuda_backend::cudarc::driver::*;

        let ctx = device.cuda_context()?;
        let func = ctx.get_func("create_causal_mask_kernel")?;

        // Allocate GPU memory
        let mask_gpu = device.zeros(&[q_len, k_len], DType::F32)?;

        // Launch kernel
        let grid_dim = (q_len.div_ceil(16), k_len.div_ceil(16), 1);
        let block_dim = (16, 16, 1);

        unsafe {
            func.launch(
                grid_dim,
                block_dim,
                0, // shared memory
                &[
                    mask_gpu.as_ptr(),
                    &q_len as *const usize as *const std::ffi::c_void,
                    &k_len as *const usize as *const std::ffi::c_void,
                    &(k_len.saturating_sub(q_len)) as *const usize as *const std::ffi::c_void,
                ],
            )?;
        }

        Ok(mask_gpu)
    }
}

// Enhanced attention implementation with optimized masking
impl MultiHeadAttention {
    pub fn forward_with_optimized_mask(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Compute Q, K, V projections
        let query = self.query_proj.forward(x)?;
        let key = self.key_proj.forward(x)?;
        let value = self.value_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = self.reshape_for_attention(&query)?;
        let k = self.reshape_for_attention(&key)?;
        let v = self.reshape_for_attention(&value)?;

        // Create or use provided causal mask
        let causal_mask = if let Some(mask) = mask {
            mask.clone()
        } else {
            // Use optimized mask creation
            self.create_causal_mask_cached(seq_len, seq_len + kv_cache.len(), x.device())?
        };

        // Efficient attention computation with optimized mask
        let attention_output = self.scaled_dot_product_attention(&q, &k, &v, Some(&causal_mask))?;

        // Output projection
        self.output_proj.forward(&attention_output)
    }
}

// Mask cache for performance optimization
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref MASK_CACHE: Mutex<HashMap<(usize, usize), Tensor>> = Mutex::new(HashMap::new());
}

impl MultiHeadAttention {
    fn get_cached_mask(&self, q_len: usize, k_len: usize, device: &Device) -> Option<Tensor> {
        let cache = MASK_CACHE.lock().ok()?;
        cache.get(&(q_len, k_len))?.to_device(device).ok()
    }

    fn cache_mask(&self, q_len: usize, k_len: usize, mask: &Tensor) {
        if let Ok(mut cache) = MASK_CACHE.lock() {
            if cache.len() < 100 { // Limit cache size
                cache.insert((q_len, k_len), mask.clone());
            }
        }
    }
}
```

### Alternative Approaches

1. **Precomputed Static Masks**: Generate common mask sizes at initialization
2. **Lazy Evaluation**: Compute masks only when needed with memoization
3. **Streaming Masks**: Generate masks incrementally for autoregressive generation

## Implementation Plan

### Phase 1: Tensor-Native Implementation (Priority: Critical)
- [ ] Replace vector manipulation with tensor operations
- [ ] Implement efficient triu-based masking for standard cases
- [ ] Add support for complex past token scenarios
- [ ] Comprehensive correctness testing

### Phase 2: Performance Optimization (Priority: High)
- [ ] Add mask caching for common sequence lengths
- [ ] Implement SIMD-optimized CPU version
- [ ] Add GPU kernel for large mask creation
- [ ] Performance benchmarking and validation

### Phase 3: Advanced Features (Priority: Medium)
- [ ] Support for different attention patterns (sliding window, etc.)
- [ ] Memory-efficient streaming mask generation
- [ ] Integration with KV cache for incremental updates
- [ ] Batch mask creation for multiple sequences

### Phase 4: Integration & Testing (Priority: High)
- [ ] Integration with existing attention mechanisms
- [ ] Cross-validation with reference implementations
- [ ] Performance regression testing
- [ ] Memory usage optimization

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_causal_mask_correctness() {
    let device = Device::Cpu;
    let attention = MultiHeadAttention::new_for_test();

    let mask_original = attention.create_causal_mask_original(4, 4, &device).unwrap();
    let mask_optimized = attention.create_causal_mask_optimized(4, 4, &device).unwrap();

    assert_tensor_eq(&mask_original, &mask_optimized, 1e-6);
}

#[test]
fn test_causal_mask_with_past() {
    let device = Device::Cpu;
    let attention = MultiHeadAttention::new_for_test();

    // Test with past tokens (incremental generation)
    let mask = attention.create_causal_mask_optimized(2, 5, &device).unwrap();

    // Verify correct masking pattern
    let expected = tensor![[0.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY],
                           [0.0, 0.0, 0.0, 0.0, 0.0]];
    assert_tensor_eq(&mask, &expected, 1e-6);
}

#[test]
fn test_mask_cache_performance() {
    let device = Device::Cpu;
    let attention = MultiHeadAttention::new_for_test();

    // First call should compute and cache
    let start = Instant::now();
    let _mask1 = attention.create_causal_mask_cached(512, 512, &device).unwrap();
    let first_time = start.elapsed();

    // Second call should use cache
    let start = Instant::now();
    let _mask2 = attention.create_causal_mask_cached(512, 512, &device).unwrap();
    let cached_time = start.elapsed();

    // Cached version should be much faster
    assert!(cached_time < first_time / 10);
}
```

### Performance Benchmarks
```bash
# Benchmark mask creation performance
cargo bench --no-default-features --features cpu causal_mask_benchmarks

# Test with different sequence lengths
cargo run -p xtask -- benchmark --component attention_mask --sizes 64,128,256,512,1024

# GPU performance validation
cargo test --no-default-features --features gpu test_gpu_mask_performance
```

## Acceptance Criteria

### Performance Requirements
- [ ] At least 5x speedup for common sequence lengths (64-512)
- [ ] At least 10x speedup for large sequences (>1024) with GPU
- [ ] Cache hit rate >90% for repeated sequence lengths
- [ ] Memory usage within 50% of current implementation

### Functional Requirements
- [ ] 100% mathematical correctness vs original implementation
- [ ] Support for arbitrary sequence lengths and past token counts
- [ ] Proper handling of edge cases (empty sequences, single tokens)
- [ ] Device-agnostic operation (CPU/GPU)

### Quality Requirements
- [ ] 100% unit test coverage for all mask patterns
- [ ] Performance regression testing in CI
- [ ] Memory leak detection and validation
- [ ] Cross-validation with reference attention implementations

## Related Issues

- Issue #251: Production-Ready Inference Server (attention performance critical)
- Multi-head attention optimization and SIMD acceleration
- KV cache integration and incremental generation
- Memory management optimization for transformer operations

## Dependencies

- Candle tensor operations (`triu`, `where_cond`, indexing)
- SIMD intrinsics for CPU optimization (`std::arch`)
- CUDA kernels for GPU acceleration (optional)
- Caching and memoization utilities

## Migration Impact

- **API Compatibility**: Maintains existing function signature
- **Performance**: Significant improvement (5-10x expected)
- **Memory**: Potential reduction through caching and optimization
- **Behavior**: Identical mathematical results with better performance

---

**Labels**: `performance`, `simulation`, `attention-mechanism`, `tensor-optimization`, `caching`
**Assignee**: Core team member with tensor operations and attention mechanism experience
**Milestone**: Optimized Attention Performance (v0.3.0)
**Estimated Effort**: 1-2 weeks for implementation and comprehensive testing
