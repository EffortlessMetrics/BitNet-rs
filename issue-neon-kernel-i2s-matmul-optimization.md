# [SIMD] NEON Kernel I2S Matrix Multiplication Optimization

## Problem Description

The `NeonKernel::matmul_i2s_neon` function contains basic NEON intrinsics implementation but lacks full optimization for I2S matrix multiplication, resulting in suboptimal performance on ARM64 platforms.

## Environment

- **File**: `crates/bitnet-kernels/src/cpu/arm.rs`
- **Function**: `NeonKernel::matmul_i2s_neon`
- **Component**: ARM64 NEON SIMD Kernels
- **Target**: ARM64 with NEON support

## Root Cause Analysis

### **Current Implementation:**
```rust
#[target_feature(enable = "neon")]
unsafe fn matmul_i2s_neon(
    &self,
    a: &[i8],
    b: &[u8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    // Initialize output to zero
    c.fill(0.0);

    // Process in blocks optimized for NEON
    const BLOCK_M: usize = 4;
    const BLOCK_N: usize = 4;
    const BLOCK_K: usize = 16;

    for i in (0..m).step_by(BLOCK_M) {
        for j in (0..n).step_by(BLOCK_N) {
            // Accumulator for 4x4 block
            let mut acc = [vdupq_n_f32(0.0); 4];

            for l in (0..k).step_by(BLOCK_K) {
                // ... NEON intrinsics ...
            }

            // Store results
            // ... incomplete implementation
        }
    }

    Ok(())
}
```

### **Issues:**
1. **Incomplete Implementation**: Missing detailed NEON intrinsic operations
2. **Suboptimal Block Sizes**: Not tuned for ARM64 cache characteristics
3. **Limited Vectorization**: Not fully utilizing NEON 128-bit vectors
4. **No Type Conversion Optimization**: Inefficient i8/u8 to f32 conversion

## Proposed Solution

Implement fully optimized NEON kernel with proper vectorization:

```rust
#[target_feature(enable = "neon")]
unsafe fn matmul_i2s_neon(
    &self,
    a: &[i8],
    b: &[u8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    // Optimized block sizes for ARM64
    const BLOCK_M: usize = 8;
    const BLOCK_N: usize = 8;
    const BLOCK_K: usize = 32;

    c.fill(0.0);

    for i in (0..m).step_by(BLOCK_M) {
        for j in (0..n).step_by(BLOCK_N) {
            for l in (0..k).step_by(BLOCK_K) {
                self.neon_micro_kernel(
                    &a[i * k + l..],
                    &b[l * n + j..],
                    &mut c[i * n + j..],
                    (m - i).min(BLOCK_M),
                    (n - j).min(BLOCK_N),
                    (k - l).min(BLOCK_K),
                    k, n
                )?;
            }
        }
    }

    Ok(())
}

#[target_feature(enable = "neon")]
unsafe fn neon_micro_kernel(
    &self,
    a_block: &[i8],
    b_block: &[u8],
    c_block: &mut [f32],
    block_m: usize,
    block_n: usize,
    block_k: usize,
    k_stride: usize,
    n_stride: usize,
) -> Result<()> {
    // Implement optimized NEON micro-kernel
    // Load 16 elements at a time using vld1q_s8/vld1q_u8
    // Convert to wider types using vmovl_s8/vmovl_u8
    // Perform dot products using vmlal_s16
    // Accumulate in f32 using vcvtq_f32_s32
    Ok(())
}
```

## Implementation Plan

### **Week 1: Core Optimization**
- Implement optimized NEON micro-kernel
- Add proper i8/u8 to f32 vectorized conversion
- Optimize block sizes for ARM64 cache hierarchy

### **Week 2: Performance Tuning**
- Add loop unrolling and prefetching
- Implement register blocking strategies
- Add benchmark comparisons with scalar version

## Success Metrics

- [ ] >4x speedup over scalar implementation
- [ ] Proper utilization of NEON 128-bit vectors
- [ ] Optimized memory access patterns for ARM64
- [ ] Numerical accuracy maintained within 1e-6

## Labels

- `neon-simd`
- `arm64-optimization`
- `i2s-quantization`
- `performance`