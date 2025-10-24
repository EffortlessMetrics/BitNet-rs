# Simulation: `TL1Quantizer::dequantize_scalar` in `tl1.rs` is a simplified implementation

The `TL1Quantizer::dequantize_scalar` function in `crates/bitnet-quantization/src/tl1.rs` performs a simplified dequantization. This is a form of simulation.

**File:** `crates/bitnet-quantization/src/tl1.rs`

**Function:** `TL1Quantizer::dequantize_scalar`

**Code:**
```rust
    fn dequantize_scalar(
        &self,
        quantized: &[i8],
        scales: &[f32],
        zero_points: &[i32],
    ) -> Result<Vec<f32>> {
        let mut dequantized = vec![0.0f32; quantized.len()];

        dequantized
            .par_chunks_mut(self.config.block_size)
            .zip(quantized.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .zip(zero_points.par_iter())
            .for_each(|(((dequant_block, quant_block), &scale), &zero_point)| {
                for (i, &value) in quant_block.iter().enumerate() {
                    let adjusted = if self.config.use_asymmetric {
                        value as i32 - zero_point
                    } else {
                        value as i32
                    };
                    dequant_block[i] = adjusted as f32 * scale;
                }
            });

        Ok(dequantized)
    }
```

## Proposed Fix

The `TL1Quantizer::dequantize_scalar` function should be implemented to perform a more accurate and optimized TL1 dequantization. This would involve using a proper lookup table for dequantization.

### Example Implementation

```rust
    fn dequantize_scalar(
        &self,
        quantized: &[i8],
        scales: &[f32],
        zero_points: &[i32],
    ) -> Result<Vec<f32>> {
        let mut dequantized = vec![0.0f32; quantized.len()];

        dequantized
            .par_chunks_mut(self.config.block_size)
            .zip(quantized.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .zip(zero_points.par_iter())
            .for_each(|(((dequant_block, quant_block), &scale), &zero_point)| {
                // Reconstruct block-specific lookup table
                let block_table = LookupTable::new(
                    // ... min_val, max_val from scale and zero_point ...
                    0.0, 0.0, // Placeholder
                    self.config.precision_bits,
                    self.config.use_asymmetric,
                );

                for (i, &value) in quant_block.iter().enumerate() {
                    dequant_block[i] = block_table.dequantize(value) * scale;
                }
            });

        Ok(dequantized)
    }
```
