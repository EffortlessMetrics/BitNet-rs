# [Quantization] Implement production-grade TL1 dequantization algorithm

## Problem Description

The `CPUQuantizer::dequantize_tl1` function in `crates/bitnet-quantization/src/device_aware_quantizer.rs` contains a simplified TL1 dequantization implementation that lacks the sophistication required for production use. The current implementation uses basic linear scaling instead of proper table lookup quantization, potentially compromising accuracy and performance.

## Environment

- **File**: `crates/bitnet-quantization/src/device_aware_quantizer.rs`
- **Function**: `CPUQuantizer::dequantize_tl1`
- **Quantization Type**: TL1 (Table Lookup 1) quantization
- **Architecture**: Device-aware quantization with CPU optimization

## Root Cause Analysis

### Current Implementation
```rust
pub fn dequantize_tl1(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
    debug!("Performing TL1 dequantization on CPU");

    if tensor.qtype != QuantizationType::TL1 {
        return Err(BitNetError::Quantization(QuantizationError::UnsupportedType));
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
            let normalized = (quantized / 7.5) - 1.0; // Simplified scaling
            let dequantized_val = normalized * scale;
            dequantized.push(dequantized_val);
        }
    }

    Ok(dequantized)
}
```

### Issues Identified
1. **Missing Lookup Table**: TL1 should use precomputed lookup tables, not linear scaling
2. **Hardcoded Magic Numbers**: The `7.5` scaling factor is arbitrary and model-agnostic
3. **No 4-bit Unpacking**: TL1 typically uses 4-bit quantization with proper bit unpacking
4. **Performance Suboptimal**: Linear scaling per value instead of vectorized lookup
5. **Accuracy Concerns**: Simplified approach may not match BitNet paper specifications

## Impact Assessment

- **Severity**: High - Core quantization accuracy affected
- **Model Compatibility**: High - TL1 models may have reduced accuracy
- **Performance Impact**: Medium - Inefficient dequantization affects inference speed
- **BitNet Compliance**: High - May not match reference implementation

## Proposed Solution

### Production-Grade TL1 Implementation

```rust
impl CPUQuantizer {
    pub fn dequantize_tl1(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        debug!("Performing production TL1 dequantization on CPU");

        if tensor.qtype != QuantizationType::TL1 {
            return Err(BitNetError::Quantization(
                QuantizationError::UnsupportedType { qtype: tensor.qtype.to_string() }
            ));
        }

        // Build or retrieve TL1 lookup table
        let lookup_table = self.get_or_build_tl1_lookup_table()?;

        let mut dequantized = Vec::with_capacity(tensor.original_shape.iter().product());
        let block_size = tensor.block_size;
        let num_blocks = tensor.scales.len();

        // Process in blocks for better cache locality
        for block_idx in 0..num_blocks {
            let scale = tensor.scales[block_idx];
            let zero_point = tensor.zero_points.get(block_idx).copied().unwrap_or(8);

            let data_start = block_idx * (block_size / 2); // 4-bit packed
            let data_end = ((block_idx + 1) * (block_size / 2)).min(tensor.data.len());

            // Unpack 4-bit values and dequantize using lookup table
            for byte_idx in data_start..data_end {
                let packed_byte = tensor.data[byte_idx];

                // Extract two 4-bit values from each byte
                let low_nibble = packed_byte & 0x0F;
                let high_nibble = (packed_byte >> 4) & 0x0F;

                // Dequantize using lookup table
                if let Some(val1) = lookup_table.get(low_nibble as usize) {
                    let adjusted_val1 = (val1 - zero_point as f32) * scale;
                    dequantized.push(adjusted_val1);
                }

                if let Some(val2) = lookup_table.get(high_nibble as usize) {
                    let adjusted_val2 = (val2 - zero_point as f32) * scale;
                    dequantized.push(adjusted_val2);
                }
            }
        }

        // Trim to original tensor size
        dequantized.truncate(tensor.original_shape.iter().product());

        debug!("TL1 dequantization completed: {} values", dequantized.len());
        Ok(dequantized)
    }

    fn get_or_build_tl1_lookup_table(&self) -> Result<&[f32]> {
        // Use cached lookup table if available
        if let Some(table) = &self.tl1_lookup_table {
            return Ok(table);
        }

        // Build TL1 lookup table according to BitNet specification
        self.build_tl1_lookup_table()
    }

    fn build_tl1_lookup_table(&mut self) -> Result<&[f32]> {
        debug!("Building TL1 lookup table");

        // TL1 uses 4-bit quantization (16 levels)
        let mut table = Vec::with_capacity(16);

        // Build table according to TL1 specification
        // This should match the quantization scheme used in the reference implementation
        for i in 0..16 {
            let quantized_level = i as f32;
            let normalized = match self.tolerance_config.tl1_method {
                TL1Method::Uniform => {
                    // Uniform quantization: map [0,15] to [-1,1]
                    (quantized_level / 7.5) - 1.0
                },
                TL1Method::NonUniform => {
                    // Non-uniform quantization with better precision near zero
                    self.compute_non_uniform_tl1_level(quantized_level)
                },
                TL1Method::Optimized => {
                    // Optimized levels based on weight distribution analysis
                    TL1_OPTIMIZED_LEVELS[i]
                }
            };
            table.push(normalized);
        }

        self.tl1_lookup_table = Some(table);
        Ok(self.tl1_lookup_table.as_ref().unwrap())
    }

    fn compute_non_uniform_tl1_level(&self, level: f32) -> f32 {
        // Non-uniform quantization with finer granularity near zero
        // Based on empirical analysis of weight distributions
        const LEVELS: [f32; 16] = [
            -1.0, -0.8, -0.6, -0.4, -0.25, -0.15, -0.05, -0.01,
            0.01, 0.05, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0
        ];
        LEVELS[level as usize.min(15)]
    }
}

// Optimized TL1 levels based on weight distribution analysis
const TL1_OPTIMIZED_LEVELS: [f32; 16] = [
    -1.0, -0.75, -0.5, -0.3, -0.2, -0.1, -0.05, -0.01,
    0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0
];

#[derive(Debug, Clone)]
pub enum TL1Method {
    Uniform,      // Standard uniform quantization
    NonUniform,   // Non-uniform with better zero precision
    Optimized,    // Empirically optimized levels
}

// Add to ToleranceConfig
impl ToleranceConfig {
    pub fn with_tl1_method(mut self, method: TL1Method) -> Self {
        self.tl1_method = method;
        self
    }
}
```

### SIMD Optimized Version

```rust
#[cfg(target_arch = "x86_64")]
impl CPUQuantizer {
    fn dequantize_tl1_simd(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        use std::arch::x86_64::*;

        if !is_x86_feature_detected!("avx2") {
            return self.dequantize_tl1_scalar(tensor);
        }

        debug!("Using AVX2 optimized TL1 dequantization");

        let lookup_table = self.get_or_build_tl1_lookup_table()?;
        let mut dequantized = Vec::with_capacity(tensor.original_shape.iter().product());

        // Process data in SIMD-friendly chunks
        unsafe {
            self.dequantize_tl1_avx2(tensor, lookup_table, &mut dequantized)?;
        }

        Ok(dequantized)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_tl1_avx2(
        &self,
        tensor: &QuantizedTensor,
        lookup_table: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<()> {
        // SIMD implementation for 8x speedup on compatible CPUs
        // Process multiple 4-bit values simultaneously
        // Implementation details...
        Ok(())
    }
}
```

## Implementation Plan

### Phase 1: Core Algorithm Implementation (Week 1)
- [ ] Implement proper 4-bit unpacking for TL1 data
- [ ] Create TL1 lookup table generation
- [ ] Add support for different TL1 quantization methods
- [ ] Implement block-wise processing with proper scaling

### Phase 2: Optimization and SIMD (Week 2)
- [ ] Add SIMD optimization for x86_64 (AVX2)
- [ ] Implement ARM NEON optimization
- [ ] Add lookup table caching
- [ ] Optimize memory access patterns

### Phase 3: Validation and Testing (Week 3)
- [ ] Cross-validate against C++ reference implementation
- [ ] Add comprehensive accuracy testing
- [ ] Performance benchmarking against current implementation
- [ ] Test with real TL1-quantized models

### Phase 4: Integration and Documentation (Week 4)
- [ ] Integrate with device-aware quantizer
- [ ] Update TL2 implementation with similar improvements
- [ ] Add comprehensive documentation
- [ ] Create usage examples and benchmarks

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tl1_lookup_table_generation() {
        let mut quantizer = CPUQuantizer::new(ToleranceConfig::default());
        let table = quantizer.build_tl1_lookup_table().unwrap();

        assert_eq!(table.len(), 16);
        assert!(table[0] < table[15]); // Monotonic
        assert!(table[7].abs() < 0.1); // Near zero
    }

    #[test]
    fn test_tl1_dequantization_accuracy() {
        let original_data = generate_test_weight_distribution(1024);
        let quantized = quantize_tl1_reference(&original_data);

        let quantizer = CPUQuantizer::new(ToleranceConfig::default());
        let dequantized = quantizer.dequantize_tl1(&quantized).unwrap();

        let mse = calculate_mse(&original_data, &dequantized);
        assert!(mse < 0.01, "TL1 dequantization MSE too high: {}", mse);
    }

    #[test]
    fn test_tl1_performance() {
        let large_tensor = create_large_tl1_tensor(1024 * 1024);
        let quantizer = CPUQuantizer::new(ToleranceConfig::default());

        let start = Instant::now();
        let _result = quantizer.dequantize_tl1(&large_tensor).unwrap();
        let duration = start.elapsed();

        // Should process at least 1M values per second
        let throughput = large_tensor.data.len() as f64 / duration.as_secs_f64();
        assert!(throughput > 1_000_000.0);
    }
}
```

## Acceptance Criteria

- [ ] Proper 4-bit unpacking and lookup table implementation
- [ ] Support for multiple TL1 quantization methods
- [ ] SIMD optimization for supported architectures
- [ ] Cross-validation accuracy within 1e-4 tolerance
- [ ] Performance improvement over current implementation
- [ ] Comprehensive test coverage including edge cases
- [ ] Integration with existing quantization pipeline

## Priority: High

This directly affects the accuracy and performance of TL1-quantized models, which are a key component of the BitNet quantization strategy.