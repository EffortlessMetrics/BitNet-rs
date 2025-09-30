# Stub code: `QuantizationKernels::quantize_neon` and `dequantize_neon` in `simd_ops.rs` have conditional compilation

The `QuantizationKernels::quantize_neon` and `dequantize_neon` functions in `crates/bitnet-quantization/src/simd_ops.rs` are conditionally compiled with `#[target_feature(enable = "neon")]`. If NEON is not available, they fall back to scalar implementation. This is a form of stubbing.

**File:** `crates/bitnet-quantization/src/simd_ops.rs`

**Functions:**
* `QuantizationKernels::quantize_neon`
* `QuantizationKernels::dequantize_neon`

**Code:**
```rust
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn quantize_neon_block(&self, data: &[f32], output: &mut [i8], scale: f32, bits: u8) {
        // ...
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn dequantize_neon_block(&self, quantized: &[i8], output: &mut [f32], scale: f32) {
        // ...
    }
```

## Proposed Fix

The `QuantizationKernels::quantize_neon` and `dequantize_neon` functions should be implemented to use NEON intrinsics directly without conditional compilation. This would involve:

1.  **Removing conditional compilation:** Remove the `#[target_feature(enable = "neon")]` attributes.
2.  **Implementing NEON intrinsics:** Implement the NEON intrinsics directly in the functions.
3.  **Providing a clear error message:** If NEON is not available, provide a clear error message instead of falling back to scalar implementation.

### Example Implementation

```rust
    #[cfg(target_arch = "aarch64")]
    fn quantize_neon(
        &self,
        data: &[f32],
        scales: &[f32],
        block_size: usize,
        bits: u8,
    ) -> Result<Vec<i8>> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return Err(anyhow::anyhow!("NEON not available"));
        }

        // ... NEON implementation ...
    }
```
