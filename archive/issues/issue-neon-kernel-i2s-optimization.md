# [ARM NEON] Optimize I2S matrix multiplication kernel for production performance

## Problem Description

The `NeonKernel::matmul_i2s_neon` function in `crates/bitnet-kernels/src/cpu/arm.rs` currently uses a basic implementation with limited NEON intrinsics optimization. While it includes block-based processing and basic NEON operations, it lacks the advanced optimizations necessary for production-grade ARM inference performance.

## Environment
- **OS**: ARM64 Linux, macOS Apple Silicon, Android ARM64
- **Hardware**: ARM Cortex-A78, Apple M1/M2/M3, Qualcomm Snapdragon
- **MSRV**: Rust 1.90.0
- **Feature Flags**: `--no-default-features --features cpu`
- **Target**: `aarch64-unknown-linux-gnu`, `aarch64-apple-darwin`

## Reproduction Steps

1. Build BitNet.rs with ARM NEON support:
   ```bash
   cargo build --no-default-features --features cpu --target aarch64-unknown-linux-gnu
   ```

2. Run I2S quantization benchmark on ARM hardware:
   ```bash
   cargo run -p xtask -- benchmark --quantization i2s --backend cpu
   ```

3. Observe performance metrics compared to optimized implementations

**Expected Results**:
- ARM NEON kernel should achieve >80% of theoretical peak throughput
- Performance should be competitive with optimized BLAS implementations

**Actual Results**:
- Current implementation achieves ~40-50% of theoretical peak
- Significant performance gap compared to x86_64 AVX2 kernels

## Root Cause Analysis

### Current Implementation Limitations

The existing `matmul_i2s_neon` function has several optimization opportunities:

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

    // Basic blocking with limited NEON utilization
    for i in (0..m).step_by(BLOCK_M) {
        for j in (0..n).step_by(BLOCK_N) {
            let mut acc = [vdupq_n_f32(0.0); 4];

            for l in (0..k).step_by(BLOCK_K) {
                // ... basic NEON intrinsics ...
            }

            // Inefficient result storage
            for ii in 0..(BLOCK_M.min(m - i)) {
                for jj in 0..(BLOCK_N.min(n - j)) {
                    if i + ii < m && j + jj < n {
                        let sum = vaddvq_f32(acc[jj]);
                        c[(i + ii) * n + (j + jj)] += sum;
                    }
                }
            }
        }
    }

    Ok(())
}
```

### Performance Bottlenecks Identified

1. **Suboptimal Block Sizes**: Current 4x4x16 blocking doesn't maximize NEON register utilization
2. **Limited Vectorization**: Missing advanced NEON intrinsics for I2S quantization operations
3. **Memory Access Patterns**: Non-contiguous memory accesses reduce cache efficiency
4. **Accumulator Management**: Inefficient accumulator handling and reduction
5. **Missing Specializations**: No ARM-specific optimizations for different CPU variants

## Impact Assessment

- **Severity**: High (production performance)
- **Performance Impact**:
  - 40-60% performance loss on ARM devices compared to optimal implementation
  - Reduced competitiveness on mobile and edge deployment targets
  - Poor user experience on ARM-based servers and workstations

- **Affected Use Cases**:
  - Mobile inference on Android/iOS devices
  - Edge deployment on ARM SBCs (Raspberry Pi, Jetson Nano)
  - Apple Silicon Mac deployment
  - ARM64 server inference

## Proposed Solution

Implement production-grade ARM NEON optimizations for I2S matrix multiplication with advanced intrinsics, optimal blocking strategies, and device-specific tuning.

### Technical Implementation Plan

#### 1. Advanced NEON Intrinsics Integration

```rust
#[target_feature(enable = "neon")]
unsafe fn matmul_i2s_neon_optimized(
    &self,
    a: &[i8],
    b: &[u8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    // Optimized block sizes for NEON register usage
    const BLOCK_M: usize = 8;
    const BLOCK_N: usize = 8;
    const BLOCK_K: usize = 32;

    // Pre-allocate NEON registers efficiently
    let mut acc: [[float32x4_t; 2]; 8] = [[vdupq_n_f32(0.0); 2]; 8];

    for i in (0..m).step_by(BLOCK_M) {
        for j in (0..n).step_by(BLOCK_N) {
            // Reset accumulators
            for row in 0..8 {
                acc[row][0] = vdupq_n_f32(0.0);
                acc[row][1] = vdupq_n_f32(0.0);
            }

            for l in (0..k).step_by(BLOCK_K) {
                self.neon_kernel_8x8x32(&a, &b, &mut acc, i, j, l, m, n, k)?;
            }

            // Optimized result storage with vector operations
            self.store_neon_results(&acc, c, i, j, m, n)?;
        }
    }

    Ok(())
}

#[inline(always)]
unsafe fn neon_kernel_8x8x32(
    &self,
    a: &[i8],
    b: &[u8],
    acc: &mut [[float32x4_t; 2]; 8],
    i: usize,
    j: usize,
    l: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    // Load A matrix block with efficient addressing
    let a_base = (i * k + l) as isize;
    let b_base = (l * n + j) as isize;

    // Unrolled loop for maximum NEON utilization
    for kk in 0..32 {
        if l + kk >= k { break; }

        // Load 8 elements from A (8 rows)
        let a_vec = self.load_a_vector_i8(a, a_base + kk as isize, k)?;

        // Load 8 elements from B (8 cols), convert u8 to i8
        let b_vec = self.load_b_vector_u8_to_i8(b, b_base + (kk * n) as isize)?;

        // Perform vectorized multiply-accumulate for I2S quantization
        for row in 0..8 {
            let a_splat = vdupq_n_f32(a_vec[row] as f32);

            // Process two float32x4_t vectors for 8 columns
            acc[row][0] = vmlaq_f32(acc[row][0], a_splat,
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(
                    vmovl_s8(vget_low_s8(b_vec[0]))
                )))
            );

            acc[row][1] = vmlaq_f32(acc[row][1], a_splat,
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(
                    vmovl_s8(vget_high_s8(b_vec[0]))
                )))
            );
        }
    }

    Ok(())
}
```

#### 2. Device-Specific Optimizations

```rust
pub struct NeonKernelConfig {
    pub cpu_variant: ArmCpuVariant,
    pub cache_sizes: CacheSizes,
    pub optimal_block_sizes: BlockSizes,
}

#[derive(Debug, Clone)]
pub enum ArmCpuVariant {
    CortexA78,
    CortexX1,
    AppleM1,
    AppleM2,
    AppleM3,
    QualcommKryo,
}

impl NeonKernel {
    pub fn new_with_cpu_detection() -> Result<Self> {
        let cpu_variant = Self::detect_cpu_variant()?;
        let config = Self::optimize_for_cpu(&cpu_variant)?;

        Ok(Self {
            config,
            cache_line_size: Self::detect_cache_line_size(),
        })
    }

    fn optimize_for_cpu(variant: &ArmCpuVariant) -> Result<NeonKernelConfig> {
        match variant {
            ArmCpuVariant::AppleM1 | ArmCpuVariant::AppleM2 => {
                // Apple Silicon specific optimizations
                NeonKernelConfig {
                    cpu_variant: variant.clone(),
                    cache_sizes: CacheSizes::apple_silicon(),
                    optimal_block_sizes: BlockSizes {
                        m: 12, n: 12, k: 64  // Optimized for Apple's wide execution units
                    },
                }
            },
            ArmCpuVariant::CortexA78 => {
                // Cortex-A78 specific optimizations
                NeonKernelConfig {
                    cpu_variant: variant.clone(),
                    cache_sizes: CacheSizes::cortex_a78(),
                    optimal_block_sizes: BlockSizes {
                        m: 8, n: 8, k: 32
                    },
                }
            },
            _ => {
                // Conservative defaults
                NeonKernelConfig::default()
            }
        }
    }
}
```

#### 3. I2S-Specific NEON Optimizations

```rust
impl NeonKernel {
    /// Optimized I2S quantization with NEON intrinsics
    #[target_feature(enable = "neon")]
    unsafe fn quantize_i2s_neon(&self, input: &[f32], output: &mut [i8], scale: f32) -> Result<()> {
        let scale_vec = vdupq_n_f32(scale);
        let chunks = input.chunks_exact(16);
        let remainder = chunks.remainder();

        for (i, chunk) in chunks.enumerate() {
            // Load 16 f32 values
            let v0 = vld1q_f32(chunk.as_ptr());
            let v1 = vld1q_f32(chunk.as_ptr().add(4));
            let v2 = vld1q_f32(chunk.as_ptr().add(8));
            let v3 = vld1q_f32(chunk.as_ptr().add(12));

            // Scale and quantize to I2S range
            let scaled0 = vmulq_f32(v0, scale_vec);
            let scaled1 = vmulq_f32(v1, scale_vec);
            let scaled2 = vmulq_f32(v2, scale_vec);
            let scaled3 = vmulq_f32(v3, scale_vec);

            // Convert to i32 with rounding
            let quant0 = vcvtnq_s32_f32(scaled0);
            let quant1 = vcvtnq_s32_f32(scaled1);
            let quant2 = vcvtnq_s32_f32(scaled2);
            let quant3 = vcvtnq_s32_f32(scaled3);

            // Clamp to I2S range [-1, 0, 1] with vectorized operations
            let clamped = self.clamp_i2s_neon(quant0, quant1, quant2, quant3)?;

            // Pack and store
            vst1_s8(output.as_mut_ptr().add(i * 16), clamped);
        }

        // Handle remainder
        self.quantize_i2s_scalar(remainder, &mut output[chunks.len() * 16..], scale)?;

        Ok(())
    }

    #[inline(always)]
    unsafe fn clamp_i2s_neon(
        &self,
        v0: int32x4_t,
        v1: int32x4_t,
        v2: int32x4_t,
        v3: int32x4_t,
    ) -> Result<int8x16_t> {
        // I2S values: -1, 0, 1
        let min_val = vdupq_n_s32(-1);
        let max_val = vdupq_n_s32(1);

        // Clamp each vector
        let clamped0 = vmaxq_s32(vminq_s32(v0, max_val), min_val);
        let clamped1 = vmaxq_s32(vminq_s32(v1, max_val), min_val);
        let clamped2 = vmaxq_s32(vminq_s32(v2, max_val), min_val);
        let clamped3 = vmaxq_s32(vminq_s32(v3, max_val), min_val);

        // Pack to int8x16_t
        let packed_low = vqmovn_s16(vcombine_s16(vqmovn_s32(clamped0), vqmovn_s32(clamped1)));
        let packed_high = vqmovn_s16(vcombine_s16(vqmovn_s32(clamped2), vqmovn_s32(clamped3)));

        Ok(vcombine_s8(packed_low, packed_high))
    }
}
```

#### 4. Performance Monitoring and Validation

```rust
pub struct NeonPerformanceMetrics {
    pub gflops: f64,
    pub cache_hit_rate: f64,
    pub vectorization_efficiency: f64,
    pub memory_bandwidth_utilization: f64,
}

impl NeonKernel {
    pub fn benchmark_matmul_i2s(&self, m: usize, n: usize, k: usize) -> Result<NeonPerformanceMetrics> {
        let iterations = 100;
        let mut total_time = Duration::ZERO;

        // Generate test data
        let a = self.generate_test_i8_matrix(m, k)?;
        let b = self.generate_test_u8_matrix(k, n)?;
        let mut c = vec![0.0f32; m * n];

        // Warmup
        for _ in 0..10 {
            self.matmul_i2s_neon(&a, &b, &mut c, m, n, k)?;
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            self.matmul_i2s_neon(&a, &b, &mut c, m, n, k)?;
        }
        total_time = start.elapsed();

        // Calculate metrics
        let ops_per_iteration = (2 * m * n * k) as f64;
        let total_ops = ops_per_iteration * iterations as f64;
        let avg_time_seconds = total_time.as_secs_f64() / iterations as f64;
        let gflops = (total_ops / 1e9) / avg_time_seconds;

        Ok(NeonPerformanceMetrics {
            gflops,
            cache_hit_rate: self.estimate_cache_hit_rate(m, n, k)?,
            vectorization_efficiency: self.calculate_vectorization_efficiency()?,
            memory_bandwidth_utilization: self.estimate_memory_bandwidth_usage(m, n, k, avg_time_seconds)?,
        })
    }
}
```

## Implementation Plan

### Phase 1: Core NEON Optimization (Week 1-2)
- [ ] Implement advanced NEON intrinsics for I2S operations
- [ ] Optimize block sizes for different ARM architectures
- [ ] Add vectorized quantization/dequantization functions
- [ ] Implement efficient accumulator management

### Phase 2: Device-Specific Tuning (Week 3)
- [ ] Add CPU variant detection at runtime
- [ ] Implement architecture-specific optimizations
- [ ] Optimize for Apple Silicon (M1/M2/M3) characteristics
- [ ] Add Cortex-A78/X1 specific tuning

### Phase 3: Performance Validation (Week 4)
- [ ] Implement comprehensive benchmarking suite
- [ ] Add cross-validation with reference implementations
- [ ] Performance regression testing
- [ ] Memory usage optimization

### Phase 4: Integration & Testing (Week 5)
- [ ] Integrate with BitNet.rs inference pipeline
- [ ] Add feature flag conditional compilation
- [ ] Comprehensive test suite for ARM devices
- [ ] Documentation and usage examples

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_matmul_correctness() {
        let kernel = NeonKernel::new().unwrap();
        let m = 128;
        let n = 256;
        let k = 512;

        let a = generate_test_i8_matrix(m, k);
        let b = generate_test_u8_matrix(k, n);
        let mut c_neon = vec![0.0f32; m * n];
        let mut c_reference = vec![0.0f32; m * n];

        // NEON implementation
        kernel.matmul_i2s_neon(&a, &b, &mut c_neon, m, n, k).unwrap();

        // Reference implementation
        reference_matmul_i2s(&a, &b, &mut c_reference, m, n, k);

        // Compare results with tolerance
        for (neon_val, ref_val) in c_neon.iter().zip(c_reference.iter()) {
            assert!((neon_val - ref_val).abs() < 1e-5);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_performance_targets() {
        let kernel = NeonKernel::new().unwrap();
        let metrics = kernel.benchmark_matmul_i2s(512, 512, 512).unwrap();

        // Performance targets for ARM optimization
        assert!(metrics.gflops > 5.0, "GFLOPS too low: {}", metrics.gflops);
        assert!(metrics.vectorization_efficiency > 0.8, "Vectorization efficiency too low");
    }
}
```

### Cross-Validation Tests
```rust
#[cfg(feature = "crossval")]
mod crossval_tests {
    #[test]
    fn test_neon_vs_reference_implementation() {
        // Compare against C++ reference on same ARM hardware
        let test_cases = generate_comprehensive_test_cases();

        for case in test_cases {
            let rust_result = neon_matmul_i2s(case.a, case.b, case.m, case.n, case.k);
            let cpp_result = cpp_reference_matmul_i2s(case.a, case.b, case.m, case.n, case.k);

            assert_results_match(&rust_result, &cpp_result, 1e-6);
        }
    }
}
```

## Performance Targets

### Target Metrics (ARM Cortex-A78)
- **GFLOPS**: >8.0 for 512x512x512 matrix multiplication
- **Memory Bandwidth**: >80% of theoretical peak
- **Cache Efficiency**: >90% L1 cache hit rate for blocked operations
- **Vectorization**: >85% NEON utilization

### Target Metrics (Apple M1/M2)
- **GFLOPS**: >15.0 for 512x512x512 matrix multiplication
- **Memory Bandwidth**: >85% of unified memory bandwidth
- **Cache Efficiency**: >95% L1 cache hit rate
- **Vectorization**: >90% NEON utilization

## Acceptance Criteria

- [ ] NEON kernel achieves target GFLOPS on test hardware
- [ ] Results match reference implementation within tolerance (1e-6)
- [ ] Performance is competitive with optimized BLAS libraries
- [ ] Memory usage remains within acceptable bounds
- [ ] All unit and integration tests pass
- [ ] Cross-validation tests pass with C++ reference
- [ ] Performance regression tests demonstrate improvement
- [ ] Code is well-documented with inline comments
- [ ] Conditional compilation works correctly across targets

## Dependencies

- ARM NEON intrinsics (std::arch::aarch64)
- Target feature detection at compile time
- Benchmarking infrastructure
- Cross-validation framework integration
- Test hardware for validation (ARM64 devices)

## Related Issues

- ARM CPU backend optimization
- I2S quantization accuracy validation
- Cross-platform performance parity
- Mobile inference optimization

## Labels
- `optimization`
- `arm64`
- `neon`
- `performance`
- `quantization`
- `priority-high`
- `cpu-kernels`
