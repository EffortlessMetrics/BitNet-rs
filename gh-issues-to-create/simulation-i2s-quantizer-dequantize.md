# Simulation: `I2SQuantizer::dequantize` in `i2s.rs` is a simplified implementation

The `I2SQuantizer::dequantize` function in `crates/bitnet-quantization/src/i2s.rs` uses `unpack_2bit_values` and `dequantize_simd`. It's not clear if these are fully optimized or simplified. This could be a form of simulation.

**File:** `crates/bitnet-quantization/src/i2s.rs`

**Function:** `I2SQuantizer::dequantize`

**Code:**
```rust
    pub fn dequantize(&self, tensor: &QuantizedTensor, device: &Device) -> Result<BitNetTensor> {
        if tensor.qtype != QuantizationType::I2S {
            return Err(
                QuantizationError::UnsupportedType { qtype: tensor.qtype.to_string() }.into()
            );
        }

        // Security: Validate quantized tensor before dequantization
        validate_quantized_tensor(tensor, limits)?;

        // Security: Validate tensor element count before unpacking
        let expected_elements = tensor.shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                BitNetError::Security(SecurityError::MemoryBomb {
                    reason: "Dequantization shape element count overflow".to_string(),
                })
            })
        })?;

        let tensor_numel = tensor.numel();
        if tensor_numel != expected_elements {
            return Err(BitNetError::Security(SecurityError::MalformedData {
                reason: format!(
                    "Tensor numel {} does not match shape element count {}",
                    tensor_numel, expected_elements
                ),
            }));
        }

        // Unpack 2-bit values with safety checks
        let quantized_data = unpack_2bit_values(&tensor.data, tensor_numel);

        // Security: Validate unpacked data length
        validate_unpacked_data_consistency(&quantized_data, tensor_numel)?;

        // Dequantize in parallel blocks with safety checks
        let dequantized_data =
            self.kernels.dequantize_simd(&quantized_data, &tensor.scales, self.block_size)?;

        // Create tensor on requested device
        create_tensor_from_f32(dequantized_data, &tensor.shape, device)
    }
```

## Proposed Fix

The `I2SQuantizer::dequantize` function should be implemented to use fully optimized and accurate I2S dequantization. This would involve:

1.  **Using optimized `unpack_2bit_values`:** Ensure `unpack_2bit_values` is highly optimized for performance.
2.  **Using optimized `dequantize_simd`:** Ensure `dequantize_simd` is highly optimized for performance and accuracy, potentially using platform-specific SIMD intrinsics.

### Example Implementation

```rust
    pub fn dequantize(&self, tensor: &QuantizedTensor, device: &Device) -> Result<BitNetTensor> {
        // ...

        // Unpack 2-bit values with safety checks
        let quantized_data = unpack_2bit_values_optimized(&tensor.data, tensor_numel);

        // Security: Validate unpacked data length
        validate_unpacked_data_consistency(&quantized_data, tensor_numel)?;

        // Dequantize in parallel blocks with safety checks
        let dequantized_data =
            self.kernels.dequantize_simd_optimized(&quantized_data, &tensor.scales, self.block_size)?;

        // ...
    }
```
