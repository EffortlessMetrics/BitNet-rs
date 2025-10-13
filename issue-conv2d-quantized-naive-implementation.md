# [SIMULATION] Quantized Conv2D Implementation Uses Naive Nested Loops with On-the-Fly Dequantization

## Problem Description

The `conv2d_quantized` function in `crates/bitnet-kernels/src/convolution.rs` implements quantized 2D convolution using a naive approach with nested loops and on-the-fly dequantization. This implementation is suboptimal for performance and does not leverage modern quantized convolution optimizations like quantized GEMM, im2col transformations, or SIMD acceleration.

## Environment

- **File**: `crates/bitnet-kernels/src/convolution.rs`
- **Function**: `conv2d_quantized` (lines 11-41)
- **Component**: BitNet quantized convolution kernels
- **Build Configuration**: `--features cpu` and `--features gpu`
- **Quantization Types**: I2_S, TL1, TL2 via `QuantizationType` enum

## Root Cause Analysis

### Technical Issues

1. **Naive Nested Loop Implementation**:
   ```rust
   for batch in 0..n {
       for out_ch in 0..oc {
           let scale = weight_scales[out_ch];
           for in_ch in 0..ic {
               for y in 0..oh {
                   for x in 0..ow {
                       // ... nested convolution loops ...
   ```
   - 6-level nested loops leading to poor cache performance
   - No vectorization or SIMD utilization
   - Memory access patterns not optimized for modern CPU architectures

2. **On-the-Fly Dequantization Inefficiency**:
   - Quantized weights are dequantized for every convolution operation
   - No batching of dequantization operations
   - Missing SIMD-optimized dequantization kernels

3. **Missing Optimized Algorithm Patterns**:
   - No im2col transformation for GEMM-based convolution
   - Lacks Winograd or FFT-based convolution optimizations
   - No utilization of quantized BLAS operations

4. **Lack of Device-Specific Optimizations**:
   - No CPU/GPU dispatch for optimal implementation
   - Missing AVX2/AVX-512 optimized kernels for CPU
   - No CUDA kernel integration for GPU acceleration

### Impact Assessment

- **Performance**: 10-100x slower than optimized implementations
- **Scalability**: Poor performance with larger models and batch sizes
- **Memory Efficiency**: Suboptimal memory access patterns
- **Power Consumption**: Higher due to inefficient computation

## Reproduction Steps

1. Build BitNet.rs with kernel features:
   ```bash
   cargo build --no-default-features --features cpu
   ```

2. Create test input for Conv2D operation:
   ```bash
   cargo test --no-default-features --features cpu test_conv2d_quantized_performance
   ```

3. Profile the convolution performance:
   ```bash
   cargo run -p xtask -- benchmark --kernel conv2d_quantized
   ```

4. **Expected**: Optimized quantized convolution using GEMM or advanced algorithms
5. **Actual**: Naive nested loops with poor performance characteristics

## Proposed Solution

### Primary Approach: Optimized Quantized Convolution Pipeline

Implement a modern quantized convolution pipeline using proven optimization techniques:

```rust
pub fn conv2d_quantized(
    input: &[f32],
    weight_quantized: &[u8],
    weight_scales: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    params: Conv2DParams,
    input_dims: (usize, usize, usize, usize),
    weight_dims: (usize, usize, usize, usize),
    qtype: QuantizationType,
) -> Result<()> {
    let (n, ic, ih, iw) = input_dims;
    let (oc, _, kh, kw) = weight_dims;
    let oh = (ih + 2 * params.padding - kh) / params.stride + 1;
    let ow = (iw + 2 * params.padding - kw) / params.stride + 1;

    // Select optimal implementation based on problem size and hardware
    let impl_strategy = select_optimal_implementation(
        input_dims, weight_dims, &params, qtype
    )?;

    match impl_strategy {
        ConvImpl::QuantizedGemm => {
            conv2d_quantized_gemm(
                input, weight_quantized, weight_scales, bias, output,
                params, input_dims, weight_dims, qtype
            )
        },
        ConvImpl::DirectOptimized => {
            conv2d_quantized_direct_simd(
                input, weight_quantized, weight_scales, bias, output,
                params, input_dims, weight_dims, qtype
            )
        },
        ConvImpl::Winograd => {
            conv2d_quantized_winograd(
                input, weight_quantized, weight_scales, bias, output,
                params, input_dims, weight_dims, qtype
            )
        },
    }
}

fn conv2d_quantized_gemm(
    input: &[f32],
    weight_quantized: &[u8],
    weight_scales: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    params: Conv2DParams,
    input_dims: (usize, usize, usize, usize),
    weight_dims: (usize, usize, usize, usize),
    qtype: QuantizationType,
) -> Result<()> {
    let (n, ic, ih, iw) = input_dims;
    let (oc, _, kh, kw) = weight_dims;
    let oh = (ih + 2 * params.padding - kh) / params.stride + 1;
    let ow = (iw + 2 * params.padding - kw) / params.stride + 1;

    // Transform convolution to GEMM via im2col
    let col_buffer_size = n * oh * ow * ic * kh * kw;
    let mut im2col_buffer = vec![0.0f32; col_buffer_size];

    // Optimized im2col transformation
    im2col_cpu_optimized(
        input, &mut im2col_buffer, ic, ih, iw, kh, kw,
        params.padding, params.stride, params.dilation, oh, ow
    )?;

    // Batch dequantize weights for efficient GEMM
    let weight_buffer_size = oc * ic * kh * kw;
    let mut weight_fp32 = vec![0.0f32; weight_buffer_size];

    batch_dequantize_weights_simd(
        weight_quantized, weight_scales, &mut weight_fp32,
        oc, ic * kh * kw, qtype
    )?;

    // Perform optimized GEMM: output = weight_fp32 * im2col_buffer^T
    cblas_sgemm_wrapper(
        &weight_fp32, &im2col_buffer, output,
        oc, n * oh * ow, ic * kh * kw,
        bias
    )?;

    Ok(())
}

fn conv2d_quantized_direct_simd(
    input: &[f32],
    weight_quantized: &[u8],
    weight_scales: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    params: Conv2DParams,
    input_dims: (usize, usize, usize, usize),
    weight_dims: (usize, usize, usize, usize),
    qtype: QuantizationType,
) -> Result<()> {
    // SIMD-optimized direct convolution for small kernels
    let (n, ic, ih, iw) = input_dims;
    let (oc, _, kh, kw) = weight_dims;
    let oh = (ih + 2 * params.padding - kh) / params.stride + 1;
    let ow = (iw + 2 * params.padding - kw) / params.stride + 1;

    // Tile-based computation for cache efficiency
    const TILE_SIZE: usize = 64;

    for batch in 0..n {
        for out_ch_start in (0..oc).step_by(TILE_SIZE) {
            let out_ch_end = (out_ch_start + TILE_SIZE).min(oc);

            for y_start in (0..oh).step_by(TILE_SIZE) {
                let y_end = (y_start + TILE_SIZE).min(oh);

                for x_start in (0..ow).step_by(TILE_SIZE) {
                    let x_end = (x_start + TILE_SIZE).min(ow);

                    // Process tile with SIMD optimization
                    process_conv_tile_simd(
                        input, weight_quantized, weight_scales, output,
                        batch, out_ch_start..out_ch_end,
                        y_start..y_end, x_start..x_end,
                        params, input_dims, weight_dims, qtype
                    )?;
                }
            }
        }
    }

    // Add bias if present
    if let Some(bias) = bias {
        add_bias_simd(output, bias, n, oc, oh, ow)?;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn process_conv_tile_simd(
    input: &[f32],
    weight_quantized: &[u8],
    weight_scales: &[f32],
    output: &mut [f32],
    batch: usize,
    out_ch_range: std::ops::Range<usize>,
    y_range: std::ops::Range<usize>,
    x_range: std::ops::Range<usize>,
    params: Conv2DParams,
    input_dims: (usize, usize, usize, usize),
    weight_dims: (usize, usize, usize, usize),
    qtype: QuantizationType,
) -> Result<()> {
    use std::arch::x86_64::*;

    if is_x86_feature_detected!("avx2") {
        unsafe {
            process_conv_tile_avx2(
                input, weight_quantized, weight_scales, output,
                batch, out_ch_range, y_range, x_range,
                params, input_dims, weight_dims, qtype
            )
        }
    } else {
        process_conv_tile_scalar(
            input, weight_quantized, weight_scales, output,
            batch, out_ch_range, y_range, x_range,
            params, input_dims, weight_dims, qtype
        )
    }
}

fn select_optimal_implementation(
    input_dims: (usize, usize, usize, usize),
    weight_dims: (usize, usize, usize, usize),
    params: &Conv2DParams,
    qtype: QuantizationType,
) -> Result<ConvImpl> {
    let (n, ic, ih, iw) = input_dims;
    let (oc, _, kh, kw) = weight_dims;

    // Heuristics for selecting optimal implementation
    let total_ops = n * oc * ic * kh * kw * ih * iw;
    let kernel_size = kh * kw;

    if kernel_size == 1 {
        // 1x1 convolution → always use GEMM
        Ok(ConvImpl::QuantizedGemm)
    } else if kernel_size <= 9 && params.stride == 1 && params.dilation == 1 {
        // Small kernels with stride 1 → consider Winograd
        if kh == 3 && kw == 3 && params.padding == 1 {
            Ok(ConvImpl::Winograd)
        } else {
            Ok(ConvImpl::DirectOptimized)
        }
    } else if total_ops > 10_000_000 {
        // Large operations → use GEMM
        Ok(ConvImpl::QuantizedGemm)
    } else {
        // Default to direct optimized
        Ok(ConvImpl::DirectOptimized)
    }
}

enum ConvImpl {
    QuantizedGemm,
    DirectOptimized,
    Winograd,
}
```

### Alternative Approaches

1. **CUDA Kernel Integration**: Leverage cuDNN quantized convolution
2. **External Library Integration**: Use QNNPACK or similar optimized libraries
3. **Mixed Precision**: Combine different quantization precisions for optimal performance

## Implementation Plan

### Phase 1: GEMM-based Implementation (Priority: Critical)
- [ ] Implement im2col transformation with SIMD optimization
- [ ] Add batch weight dequantization with vectorization
- [ ] Integrate with optimized BLAS library (OpenBLAS/MKL)
- [ ] Add comprehensive correctness testing

### Phase 2: Direct SIMD Optimization (Priority: High)
- [ ] Implement AVX2/AVX-512 optimized kernels
- [ ] Add tile-based computation for cache efficiency
- [ ] Implement NEON optimization for ARM architectures
- [ ] Add performance benchmarking suite

### Phase 3: Advanced Algorithms (Priority: Medium)
- [ ] Implement Winograd convolution for 3x3 kernels
- [ ] Add FFT-based convolution for large kernels
- [ ] Implement channel-wise quantization support
- [ ] Add dynamic algorithm selection based on problem size

### Phase 4: GPU Integration (Priority: Medium)
- [ ] CUDA kernel implementation with Tensor Cores
- [ ] ROCm/HIP support for AMD GPUs
- [ ] Mixed CPU/GPU execution for optimal performance
- [ ] Memory management optimization for GPU

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_conv2d_quantized_correctness() {
    // Test against reference floating-point implementation
    let input = create_test_input(1, 32, 64, 64);
    let weights = create_test_weights(64, 32, 3, 3);
    let quantized_weights = quantize_weights(&weights, QuantizationType::I2_S);

    let mut output_quantized = vec![0.0; 1 * 64 * 62 * 62];
    let mut output_reference = vec![0.0; 1 * 64 * 62 * 62];

    conv2d_quantized(&input, &quantized_weights.0, &quantized_weights.1,
                    None, &mut output_quantized, conv_params,
                    (1, 32, 64, 64), (64, 32, 3, 3), QuantizationType::I2_S)?;

    conv2d_reference(&input, &weights, None, &mut output_reference, conv_params)?;

    assert_tensor_close(&output_quantized, &output_reference, 1e-3);
}

#[test]
fn test_conv2d_quantized_performance() {
    // Performance benchmark against current implementation
    let input = create_large_test_input();
    let (weights, scales) = create_large_quantized_weights();

    let start = Instant::now();
    conv2d_quantized(&input, &weights, &scales, None, &mut output,
                    params, input_dims, weight_dims, qtype)?;
    let duration = start.elapsed();

    // Should be at least 5x faster than naive implementation
    assert!(duration < Duration::from_millis(100));
}
```

### Integration Tests
```bash
# Correctness validation
cargo test --no-default-features --features cpu test_conv2d_integration

# Performance validation
cargo run -p xtask -- benchmark --kernel conv2d --quantization i2s

# Cross-validation with reference
cargo run -p xtask -- crossval --component conv2d
```

### Performance Benchmarks
- Small kernels (1x1, 3x3): >10x speedup vs naive
- Large feature maps: >20x speedup via GEMM optimization
- Memory usage: <2x theoretical minimum
- SIMD utilization: >80% of theoretical peak

## Acceptance Criteria

### Functional Requirements
- [ ] Mathematical correctness within quantization error bounds (±1e-3)
- [ ] Support for all BitNet quantization types (I2_S, TL1, TL2)
- [ ] Proper handling of padding, stride, and dilation parameters
- [ ] Bias addition support with numerical stability

### Performance Requirements
- [ ] At least 10x speedup over current naive implementation
- [ ] SIMD instruction utilization >80% on AVX2-capable hardware
- [ ] Memory usage within 2x of theoretical minimum
- [ ] Scalable performance with batch size and tensor dimensions

### Quality Requirements
- [ ] 100% unit test coverage for convolution correctness
- [ ] Performance regression testing in CI
- [ ] Memory leak detection and validation
- [ ] Cross-platform compatibility (x86_64, ARM64)

## Related Issues

- Issue #251: Production-Ready Inference Server (convolution performance critical)
- Performance optimization tracking for quantized operations
- SIMD kernel development for BitNet quantization
- Memory management optimization for large tensors

## Dependencies

- `std::arch` for SIMD intrinsics (AVX2, AVX-512, NEON)
- External BLAS library (OpenBLAS, Intel MKL, or BLIS)
- BitNet quantization utilities (`QuantizationType`, dequantization functions)
- Memory alignment utilities for SIMD operations

## Migration Impact

- **API Compatibility**: Maintains existing function signature
- **Performance**: Significant improvement (10-100x expected)
- **Memory**: Increased temporary buffer usage for optimization
- **Build Dependencies**: May require external BLAS library linking

---

**Labels**: `critical`, `simulation`, `performance`, `quantization`, `convolution`, `simd-optimization`
**Assignee**: Core team member with quantized kernel optimization experience
**Milestone**: High-Performance Quantized Kernels (v0.3.0)
**Estimated Effort**: 3-4 weeks for full implementation and testing
