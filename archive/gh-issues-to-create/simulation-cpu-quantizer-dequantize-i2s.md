# Simulation: `CPUQuantizer::dequantize_i2s` in `device_aware_quantizer.rs` is a simplified implementation

The `CPUQuantizer::dequantize_i2s` function in `crates/bitnet-quantization/src/device_aware_quantizer.rs` has a comment "Simplified I2S dequantization". It performs a simplified I2S dequantization. This is a form of simulation.

**File:** `crates/bitnet-quantization/src/device_aware_quantizer.rs`

**Function:** `CPUQuantizer::dequantize_i2s`

**Code:**
```rust
    pub fn dequantize_i2s(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        debug!("Performing I2S dequantization on CPU");

        if tensor.qtype != QuantizationType::I2S {
            return Err(bitnet_common::BitNetError::Quantization(
                QuantizationError::UnsupportedType { qtype: tensor.qtype.to_string() },
            ));
        }

        let mut dequantized = Vec::new();
        let block_size = tensor.block_size;
        let num_blocks = tensor.scales.len();

        for block_idx in 0..num_blocks {
            let scale = tensor.scales[block_idx];
            let start_byte = block_idx * block_size.div_ceil(4); // 4 values per byte

            for byte_idx in 0..block_size.div_ceil(4) {
                if start_byte + byte_idx >= tensor.data.len() {
                    break;
                }

                let packed = tensor.data[start_byte + byte_idx];
                for bit_idx in 0..4 {
                    let quantized = ((packed >> (bit_idx * 2)) & 0x03) as i8;
                    let signed_val = match quantized {
                        0 => 0i8,
                        1 => 1i8,
                        2 => -1i8, // 2 in 2-bit represents -1
                        3 => 0i8,  // Invalid value, treat as 0
                        _ => 0i8,
                    };

                    let dequantized_val = signed_val as f32 * scale;
                    dequantized.push(dequantized_val);

                    if dequantized.len() >= tensor.numel() {
                        break;
                    }
                }
            }
        }

        // Trim to exact size
        dequantized.truncate(tensor.numel());
        Ok(dequantized)
    }
```

## Proposed Fix

The `CPUQuantizer::dequantize_i2s` function should be implemented to perform a more accurate and optimized I2S dequantization. This would involve:

1.  **Using a proper dequantization algorithm:** Implement a more advanced I2S dequantization algorithm.
2.  **Optimizing unpacking:** Optimize the unpacking of 2-bit values from bytes for better performance.

### Example Implementation

```rust
    pub fn dequantize_i2s(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        debug!("Performing I2S dequantization on CPU");

        if tensor.qtype != QuantizationType::I2S {
            return Err(bitnet_common::BitNetError::Quantization(
                QuantizationError::UnsupportedType { qtype: tensor.qtype.to_string() },
            ));
        }

        let mut dequantized = Vec::new();
        let block_size = tensor.block_size;
        let num_blocks = tensor.scales.len();

        for block_idx in 0..num_blocks {
            let scale = tensor.scales[block_idx];
            let start_byte = block_idx * block_size.div_ceil(4); // 4 values per byte

            for byte_idx in 0..block_size.div_ceil(4) {
                if start_byte + byte_idx >= tensor.data.len() {
                    break;
                }

                let packed = tensor.data[start_byte + byte_idx];
                for bit_idx in 0..4 {
                    let quantized = ((packed >> (bit_idx * 2)) & 0x03) as i8;
                    let signed_val = match quantized {
                        0 => 0i8,
                        1 => 1i8,
                        2 => -1i8, // 2 in 2-bit represents -1
                        3 => 0i8,  // Invalid value, treat as 0
                        _ => 0i8,
                    };

                    let dequantized_val = signed_val as f32 * scale;
                    dequantized.push(dequantized_val);

                    if dequantized.len() >= tensor.numel() {
                        break;
                    }
                }
            }
        }

        // Trim to exact size
        dequantized.truncate(tensor.numel());
        Ok(dequantized)
    }
```
