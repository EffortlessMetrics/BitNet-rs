# Stub code: `QuantizedLinear::vectorized_tl1_matmul` and `vectorized_tl2_matmul` in `quantized_linear.rs` fallback to generic

The `QuantizedLinear::vectorized_tl1_matmul` and `vectorized_tl2_matmul` functions in `crates/bitnet-inference/src/layers/quantized_linear.rs` fallback to `forward_tl1_generic` and `forward_tl2_generic` respectively. This suggests that the vectorized kernels are not fully implemented. This is a form of stubbing.

**File:** `crates/bitnet-inference/src/layers/quantized_linear.rs`

**Functions:**
* `QuantizedLinear::vectorized_tl1_matmul`
* `QuantizedLinear::vectorized_tl2_matmul`

**Code:**
```rust
    #[cfg(target_arch = "aarch64")]
    async fn vectorized_tl1_matmul(
        &self,
        input: &BitNetTensor,
        _provider: &dyn bitnet_kernels::KernelProvider,
    ) -> Result<BitNetTensor> {
        // Would use NEON-optimized TL1 kernel
        self.forward_tl1_generic(input).await
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    async fn vectorized_tl2_matmul(
        &self,
        input: &BitNetTensor,
        _provider: &dyn bitnet_kernels::KernelProvider,
    ) -> Result<BitNetTensor> {
        // Would use AVX-optimized TL2 kernel
        self.forward_tl2_generic(input).await
    }
```

## Proposed Fix

The `QuantizedLinear::vectorized_tl1_matmul` and `vectorized_tl2_matmul` functions should be implemented to use their respective vectorized kernels. This would involve calling the NEON-optimized TL1 kernel and AVX-optimized TL2 kernel respectively.

### Example Implementation

```rust
    #[cfg(target_arch = "aarch64")]
    async fn vectorized_tl1_matmul(
        &self,
        input: &BitNetTensor,
        provider: &dyn bitnet_kernels::KernelProvider,
    ) -> Result<BitNetTensor> {
        // Use NEON-optimized TL1 kernel
        provider.matmul_tl1_neon(input, &self.weights)
    }
```
