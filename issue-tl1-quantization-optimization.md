# [Quantization] Optimize TL1 Dequantization with Lookup Tables

## Problem Description

The `CPUQuantizer::dequantize_tl1` method in `/home/steven/code/Rust/BitNet-rs/crates/bitnet-quantization/src/device_aware_quantizer.rs` uses simplified linear dequantization instead of efficient lookup table methods, missing significant performance optimizations for TL1 quantization.

## Current Implementation
```rust
// Simplified TL1 dequantization
let normalized = (quantized / 7.5) - 1.0;
let dequantized_val = normalized * scale;
```

## Proposed Solution
Implement optimized lookup table dequantization:

```rust
pub fn dequantize_tl1(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
    let mut dequantized = Vec::with_capacity(tensor.data.len());
    let lookup_table = &self.tl1_lookup_table;

    for block_idx in 0..tensor.scales.len() {
        let scale = tensor.scales[block_idx];
        let block_start = block_idx * tensor.block_size;
        let block_end = (block_start + tensor.block_size).min(tensor.data.len());

        // SIMD-optimized lookup table dequantization
        for &quantized_val in &tensor.data[block_start..block_end] {
            let dequantized_val = lookup_table.dequantize(quantized_val) * scale;
            dequantized.push(dequantized_val);
        }
    }

    Ok(dequantized)
}
```

## Acceptance Criteria
- [ ] Lookup table-based TL1 dequantization
- [ ] SIMD optimization for batch operations
- [ ] Memory-efficient table storage
- [ ] 3-5x performance improvement over linear method
- [ ] Bit-accurate results matching reference implementation