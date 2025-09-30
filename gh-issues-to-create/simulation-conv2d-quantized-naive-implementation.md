# Simulation: `conv2d_quantized` in `convolution.rs` is a naive implementation

The `conv2d_quantized` function in `crates/bitnet-kernels/src/convolution.rs` is a naive implementation that performs quantized convolution with dequantization on-the-fly using nested loops. This is a form of simulation.

**File:** `crates/bitnet-kernels/src/convolution.rs`

**Function:** `conv2d_quantized`

**Code:**
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
    // ...

    // Perform quantized convolution with dequantization on-the-fly
    for batch in 0..n {
        #[allow(clippy::needless_range_loop)]
        for out_ch in 0..oc {
            let scale = weight_scales[out_ch];

            for in_ch in 0..ic {
                for y in 0..oh {
                    for x in 0..ow {
                        // ... nested loops for convolution ...
                    }
                }
            }
        }
    }

    Ok(())
}
```

## Proposed Fix

The `conv2d_quantized` function should be implemented to use optimized quantized convolution algorithms. This would involve using:

1.  **Quantized GEMM-based convolution:** Convert the quantized convolution operation into a quantized General Matrix Multiply (GEMM) operation, which can be highly optimized using specialized quantized BLAS libraries or SIMD intrinsics.
2.  **Optimized dequantization:** Perform dequantization efficiently, potentially using SIMD instructions, before or during the matrix multiplication.

### Example Implementation

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
    // ...

    // Example: Use an optimized quantized GEMM-based convolution
    // Quantized im2col transformation
    let quantized_im2col_input = quantized_im2col(input, input_dims, weight_dims, params, qtype);

    // Perform quantized GEMM
    quantized_gemm(&quantized_im2col_input, weight_quantized, weight_scales, output, ...);

    // ...

    Ok(())
}
```
