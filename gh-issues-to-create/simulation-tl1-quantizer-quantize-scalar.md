# Simulation: `TL1Quantizer::quantize_scalar` in `tl1.rs` is a simplified implementation

The `TL1Quantizer::quantize_scalar` function in `crates/bitnet-quantization/src/tl1.rs` creates a block-specific lookup table for each block. This might be inefficient. This is a form of simulation.

**File:** `crates/bitnet-quantization/src/tl1.rs`

**Function:** `TL1Quantizer::quantize_scalar`

**Code:**
```rust
    fn quantize_scalar(
        &self,
        data: &[f32],
        _lookup_table: &LookupTable,
        scales: &[f32],
    ) -> Result<Vec<i8>> {
        let mut quantized = vec![0i8; data.len()];

        quantized
            .par_chunks_mut(self.config.block_size)
            .zip(data.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .for_each(|((quant_block, data_block), &_scale)| {
                // Create block-specific lookup table
                let block_min = data_block.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
                let block_max = data_block.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                let block_table = LookupTable::new(
                    block_min,
                    block_max,
                    self.config.precision_bits,
                    self.config.use_asymmetric,
                );

                for (i, &value) in data_block.iter().enumerate() {
                    quant_block[i] = block_table.quantize(value);
                }
            });

        Ok(quantized)
    }
```

## Proposed Fix

The `TL1Quantizer::quantize_scalar` function should be implemented to use a single lookup table for the entire tensor, or a more efficient way to manage block-specific lookup tables. This would involve pre-calculating the lookup table once and reusing it for all blocks.

### Example Implementation

```rust
    fn quantize_scalar(
        &self,
        data: &[f32],
        lookup_table: &LookupTable,
        scales: &[f32],
    ) -> Result<Vec<i8>> {
        let mut quantized = vec![0i8; data.len()];

        quantized
            .par_chunks_mut(self.config.block_size)
            .zip(data.par_chunks(self.config.block_size))
            .zip(scales.par_iter())
            .for_each(|((quant_block, data_block), &scale)| {
                for (i, &value) in data_block.iter().enumerate() {
                    quant_block[i] = lookup_table.quantize(value);
                }
            });

        Ok(quantized)
    }
```
