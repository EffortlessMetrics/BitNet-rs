# [IMPLEMENTATION] Replace simplified TL1 dequantization with production-grade table lookup implementation

## Problem Description

The `CPUQuantizer::dequantize_tl1` function in `crates/bitnet-quantization/src/device_aware_quantizer.rs` uses a simplified mathematical approximation instead of proper table lookup (TL1) dequantization, impacting accuracy and performance.

## Environment
- **File**: `crates/bitnet-quantization/src/device_aware_quantizer.rs`
- **Function**: `CPUQuantizer::dequantize_tl1`
- **Current State**: Simplified simulation instead of real TL1 algorithm

## Root Cause Analysis

Current simplified implementation:
```rust
let normalized = (quantized / 7.5) - 1.0;  // Mathematical approximation
let dequantized_val = normalized * scale;
```

**Issues:**
1. Uses mathematical approximation instead of lookup table
2. Hardcoded magic numbers (7.5) lack scientific basis
3. No proper 4-bit unpacking from packed bytes
4. Missing TL1-specific optimization features
5. Doesn't match TL1 specification requirements

## Proposed Solution

Implement production-grade TL1 dequantization with proper lookup tables:

```rust
pub struct TL1LookupTable {
    dequant_values: [f32; 16],  // 4-bit = 16 possible values
}

impl TL1LookupTable {
    fn new() -> Self {
        // TL1 standard quantization levels
        let dequant_values = [
            -1.0, -0.8571, -0.7143, -0.5714, -0.4286, -0.2857, -0.1429, 0.0,
            0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571, 1.0, 1.1429
        ];
        Self { dequant_values }
    }

    fn dequantize(&self, quantized: u8) -> f32 {
        self.dequant_values[quantized as usize & 0xF]
    }
}

impl CPUQuantizer {
    pub fn dequantize_tl1(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        let lookup_table = TL1LookupTable::new();
        let mut dequantized = Vec::with_capacity(tensor.original_shape.iter().product());

        // Process blocks with proper 4-bit unpacking
        for (block_idx, &scale) in tensor.scales.iter().enumerate() {
            let byte_start = block_idx * tensor.block_size / 2;  // 2 values per byte
            let byte_end = byte_start + tensor.block_size / 2;

            for byte_idx in byte_start..byte_end {
                if byte_idx >= tensor.data.len() { break; }

                let packed_byte = tensor.data[byte_idx];

                // Unpack two 4-bit values from one byte
                let val1 = packed_byte & 0xF;
                let val2 = (packed_byte >> 4) & 0xF;

                dequantized.push(lookup_table.dequantize(val1) * scale);
                dequantized.push(lookup_table.dequantize(val2) * scale);
            }
        }

        Ok(dequantized)
    }
}
```

## Implementation Plan

### Phase 1: TL1 Algorithm Research (1 day)
- [ ] Research TL1 quantization specification
- [ ] Identify optimal lookup table values
- [ ] Define bit packing/unpacking format

### Phase 2: Core Implementation (2 days)
- [ ] Implement TL1LookupTable structure
- [ ] Add proper 4-bit value unpacking
- [ ] Implement vectorized operations for performance
- [ ] Add comprehensive error handling

### Phase 3: Testing & Validation (1 day)
- [ ] Create accuracy tests vs reference implementation
- [ ] Add performance benchmarks
- [ ] Validate against TL1 specification
- [ ] Cross-validate with GPU implementation

## Acceptance Criteria
- [ ] Proper TL1 lookup table implementation
- [ ] Accurate 4-bit unpacking from bytes
- [ ] ≥99% accuracy vs reference TL1 implementation
- [ ] Performance improvement over current simulation
- [ ] Comprehensive test coverage

**Labels**: `implementation`, `quantization`, `performance`, `P2-medium`
**Effort**: 4 days