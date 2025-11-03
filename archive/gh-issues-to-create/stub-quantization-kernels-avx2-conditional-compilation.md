# Stub code: `QuantizationKernels::quantize_avx2` and `dequantize_avx2` in `simd_ops.rs` have conditional compilation

The `QuantizationKernels::quantize_avx2` and `dequantize_avx2` functions in `crates/bitnet-quantization/src/simd_ops.rs` are conditionally compiled with `#[target_feature(enable = "avx2")]`. If AVX2 is not available, they fall back to scalar implementation. This is a form of stubbing.

**File:** `crates/bitnet-quantization/src/simd_ops.rs`

**Functions:**
* `QuantizationKernels::quantize_avx2`
* `QuantizationKernels::dequantize_avx2`

**Code:**
```rust
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_avx2_block(&self, data: &[f32], output: &mut [i8], scale: f32, bits: u8) {
        // ...
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_avx2_block(&self, quantized: &[i8], output: &mut [f32], scale: f32) {
        // ...
    }
```

## Proposed Fix

The `QuantizationKernels::quantize_avx2` and `dequantize_avx2` functions should be implemented to use AVX2 intrinsics directly without conditional compilation. This would involve:

1.  **Removing conditional compilation:** Remove the `#[target_feature(enable = "avx2")]` attributes.
2.  **Implementing AVX2 intrinsics:** Implement the AVX2 intrinsics directly in the functions.
3.  **Providing a clear error message:** If AVX2 is not available, provide a clear error message instead of falling back to scalar implementation.

### Example Implementation

```rust
    #[cfg(target_arch = "x86_64")]
    fn quantize_avx2(
        &self,
        data: &[f32],
        scales: &[f32],
        block_size: usize,
        bits: u8,
    ) -> Result<Vec<i8>> {
        if !is_x86_feature_detected!("avx2") {
            return Err(anyhow::anyhow!("AVX2 not available"));
        }

        // ... AVX2 implementation ...
    }
```
