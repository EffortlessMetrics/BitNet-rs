# Simulation: `CPUQuantizer::dequantize_tl1` in `device_aware_quantizer.rs` is a simplified implementation

The `CPUQuantizer::dequantize_tl1` function in `crates/bitnet-quantization/src/device_aware_quantizer.rs` has a comment "Simplified TL1 dequantization". It performs a simplified TL1 dequantization. This is a form of simulation.

**File:** `crates/bitnet-quantization/src/device_aware_quantizer.rs`

**Function:** `CPUQuantizer::dequantize_tl1`

**Code:**
```rust
    pub fn dequantize_tl1(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        debug!("Performing TL1 dequantization on CPU");

        if tensor.qtype != QuantizationType::TL1 {
            return Err(bitnet_common::BitNetError::Quantization(
                QuantizationError::UnsupportedType { qtype: tensor.qtype.to_string() },
            ));
        }

        let mut dequantized = Vec::new();
        let block_size = tensor.block_size;
        let num_blocks = tensor.scales.len();

        for block_idx in 0..num_blocks {
            let scale = tensor.scales[block_idx];
            let start = block_idx * block_size;
            let end = (start + block_size).min(tensor.data.len());

            for i in start..end {
                let quantized = tensor.data[i] as f32;
                let normalized = (quantized / 7.5) - 1.0;
                let dequantized_val = normalized * scale;
                dequantized.push(dequantized_val);
            }
        }

        Ok(dequantized)
    }
```

## Proposed Fix

The `CPUQuantizer::dequantize_tl1` function should be implemented to perform a more accurate and optimized TL1 dequantization. This would involve:

1.  **Using a proper dequantization algorithm:** Implement a more advanced TL1 dequantization algorithm that uses a lookup table.
2.  **Optimizing unpacking:** Optimize the unpacking of 4-bit values from bytes for better performance.

### Example Implementation

```rust
    pub fn dequantize_tl1(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        debug!("Performing TL1 dequantization on CPU");

        if tensor.qtype != QuantizationType::TL1 {
            return Err(bitnet_common::BitNetError::Quantization(
                QuantizationError::UnsupportedType { qtype: tensor.qtype.to_string() },
            ));
        }

        let mut dequantized = Vec::new();
        let block_size = tensor.block_size;
        let num_blocks = tensor.scales.len();

        for block_idx in 0..num_blocks {
            let scale = tensor.scales[block_idx];
            let start = block_idx * block_size;
            let end = (start + block_size).min(tensor.data.len());

            for i in start..end {
                let quantized = tensor.data[i];
                let dequantized_val = self.lookup_table.dequantize(quantized) * scale; // Assuming a lookup table
                dequantized.push(dequantized_val);
            }
        }

        Ok(dequantized)
    }
```
