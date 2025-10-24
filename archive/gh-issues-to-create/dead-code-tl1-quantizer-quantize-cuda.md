# Dead code: `TL1Quantizer::quantize_cuda` in `tl1.rs` is never used

The `TL1Quantizer::quantize_cuda` method is defined but not used. This is a form of dead code.

**File:** `crates/bitnet-quantization/src/tl1.rs`

**Function:** `TL1Quantizer::quantize_cuda`

**Code:**
```rust
    #[cfg(feature = "cuda")]
    fn quantize_cuda(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor> {
        use bitnet_kernels::gpu::cuda::CudaKernel;
        let data = extract_f32_data(tensor)?;
        let shape = tensor.shape().to_vec();
        let num_blocks = data.len().div_ceil(self.config.block_size);
        let mut scales = vec![0f32; num_blocks];
        let packed_len = (data.len() * self.config.precision_bits as usize).div_ceil(8);
        let mut packed_data = vec![0u8; packed_len];
        let kernel = CudaKernel::new()?;
        kernel.quantize(&data, &mut packed_data, &mut scales, QuantizationType::TL1)?;
        Ok(QuantizedTensor::new_with_params(
            packed_data,
            scales,
            None,
            shape,
            QuantizationType::TL1,
            self.config.block_size,
        ))
    }
```

## Proposed Fix

If the `TL1Quantizer::quantize_cuda` method is not intended to be used, it should be removed to reduce the size of the codebase and improve maintainability. If it is intended to be used, it should be integrated into the quantization process.

### Example Integration

```rust
    pub fn quantize(&self, tensor: &BitNetTensor, device: &Device) -> Result<QuantizedTensor> {
        if !device.is_cpu() {
            #[cfg(feature = "cuda")]
            {
                if device.is_cuda()
                    && bitnet_kernels::gpu::cuda::is_cuda_available()
                    && let Ok(res) = self.quantize_cuda(tensor)
                {
                    return Ok(res);
                }
            }
        }

        // ...
    }
```
