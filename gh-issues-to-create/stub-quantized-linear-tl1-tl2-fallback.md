# Stub code: `QuantizedLinear::quantized_matmul_tl1` and `quantized_matmul_tl2` in `quantized_linear.rs` fallback to `matmul_i2s`

The `QuantizedLinear::quantized_matmul_tl1` and `quantized_matmul_tl2` functions in `crates/bitnet-inference/src/layers/quantized_linear.rs` fallback to `provider.matmul_i2s`. This suggests that the TL1 and TL2 kernels are not fully implemented. This is a form of stubbing.

**File:** `crates/bitnet-inference/src/layers/quantized_linear.rs`

**Functions:**
* `QuantizedLinear::quantized_matmul_tl1`
* `QuantizedLinear::quantized_matmul_tl2`

**Code:**
```rust
    async fn quantized_matmul_tl1(
        &self,
        input: &candle_core::Tensor,
        provider: &dyn bitnet_kernels::KernelProvider,
    ) -> Result<candle_core::Tensor> {
        // ...
        // Call quantized kernel (fallback to I2S kernel for now)
        provider
            .matmul_i2s(
                &quantized_input,
                &weight_data,
                &mut output_data,
                batch_size,
                out_features,
                in_features,
            )
            .context("Native TL1 quantized matmul failed")?;
        // ...
    }

    async fn quantized_matmul_tl2(
        &self,
        input: &candle_core::Tensor,
        provider: &dyn bitnet_kernels::KernelProvider,
    ) -> Result<candle_core::Tensor> {
        // ...
        // Call quantized kernel (fallback to I2S kernel for now)
        provider
            .matmul_i2s(
                &quantized_input,
                &weight_data,
                &mut output_data,
                batch_size,
                out_features,
                in_features,
            )
            .context("Native TL2 quantized matmul failed")?;
        // ...
    }
```

## Proposed Fix

The `QuantizedLinear::quantized_matmul_tl1` and `quantized_matmul_tl2` functions should be implemented to use their respective TL1 and TL2 kernels. This would involve calling `provider.matmul_tl1` and `provider.matmul_tl2` instead of `provider.matmul_i2s`.

### Example Implementation

```rust
    async fn quantized_matmul_tl1(
        &self,
        input: &candle_core::Tensor,
        provider: &dyn bitnet_kernels::KernelProvider,
    ) -> Result<candle_core::Tensor> {
        // ...
        // Call native TL1 quantized kernel
        provider
            .matmul_tl1(
                &quantized_input,
                &weight_data,
                &mut output_data,
                batch_size,
                out_features,
                in_features,
            )
            .context("Native TL1 quantized matmul failed")?;
        // ...
    }
```
