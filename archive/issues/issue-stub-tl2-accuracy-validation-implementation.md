# [STUB] AccuracyValidator::validate_tl_accuracy incorrectly uses TL1 dequantization for TL2 validation

## Problem Description

The `AccuracyValidator::validate_tl_accuracy` method falls back to using `dequantize_tl1` for TL2 quantization validation, compromising accuracy assessment and potentially masking TL2-specific quantization issues.

## Environment

**File**: `crates/bitnet-quantization/src/device_aware_quantizer.rs`
**Component**: TL2 Quantization Accuracy Validation
**Issue Type**: Stub Implementation / Incorrect Validation Logic

## Root Cause Analysis

**Current Implementation:**
```rust
pub fn validate_tl_accuracy(
    &self,
    original: &[f32],
    quantized: &QuantizedTensor,
) -> Result<AccuracyReport> {
    let cpu_quantizer = CPUQuantizer::new(self.tolerance_config.clone());
    let dequantized = match quantized.qtype {
        QuantizationType::TL1 => cpu_quantizer.dequantize_tl1(quantized)?,
        QuantizationType::TL2 => {
            // TL2 would have its own implementation
            cpu_quantizer.dequantize_tl1(quantized)?  // ❌ INCORRECT
        }
        _ => {
            return Err(bitnet_common::BitNetError::Quantization(
                QuantizationError::UnsupportedType { qtype: quantized.qtype.to_string() },
            ));
        }
    };
    // ... accuracy calculation
}
```

**Analysis:**
1. **Validation Corruption**: TL2 tensors are being validated using TL1 dequantization logic
2. **Masked Errors**: TL2-specific quantization errors may go undetected
3. **Incorrect Metrics**: Accuracy reports for TL2 do not reflect actual TL2 performance
4. **Testing Compromise**: Validation framework cannot properly assess TL2 quantization quality

## Impact Assessment

**Severity**: High
**Affected Areas**:
- TL2 quantization accuracy assessment
- Production validation confidence
- Quantization algorithm comparison and selection
- Testing framework integrity

**Validation Impact**:
- False accuracy reports for TL2 quantization
- Inability to detect TL2-specific degradation
- Compromised quantization algorithm evaluation
- Potential deployment of suboptimal TL2 configurations

**Business Impact**:
- Reduced confidence in TL2 quantization deployment
- Possible quality degradation in production
- Ineffective quantization algorithm selection

## Proposed Solution

### Option 1: Implement Proper TL2 Dequantization (Recommended)

Add correct TL2 dequantization method and integrate into validation:

```rust
impl CPUQuantizer {
    pub fn dequantize_tl2(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>> {
        ensure_tl2_quantization(quantized)?;

        let mut result = Vec::with_capacity(quantized.original_shape().iter().product());

        // TL2-specific dequantization using lookup table 2
        let lookup_table = self.build_tl2_lookup_table(quantized)?;

        for &packed_value in &quantized.data {
            let indices = unpack_tl2_indices(packed_value);

            for index in indices {
                let dequantized_value = lookup_table[index as usize];
                result.push(dequantized_value);
            }
        }

        // Apply TL2-specific scaling
        self.apply_tl2_scaling(&mut result, &quantized.scales)?;

        Ok(result)
    }

    fn build_tl2_lookup_table(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>> {
        // TL2 uses a different lookup table construction than TL1
        let table_size = 1 << quantized.bits_per_weight; // 2^bits_per_weight entries
        let mut table = vec![0.0f32; table_size];

        // TL2-specific table initialization
        for i in 0..table_size {
            table[i] = self.compute_tl2_table_entry(i, quantized)?;
        }

        Ok(table)
    }

    fn compute_tl2_table_entry(&self, index: usize, quantized: &QuantizedTensor) -> Result<f32> {
        // TL2-specific table entry computation
        // This differs from TL1 in the range mapping and distribution
        let normalized_index = (index as f32 - 127.5) / 127.5; // For 8-bit indices

        // TL2 uses a different activation function/mapping
        Ok(normalized_index.tanh()) // Example: TL2 might use tanh activation
    }

    fn apply_tl2_scaling(&self, values: &mut [f32], scales: &[f32]) -> Result<()> {
        // TL2-specific scaling application
        let block_size = values.len() / scales.len();

        for (block_idx, &scale) in scales.iter().enumerate() {
            let start = block_idx * block_size;
            let end = (start + block_size).min(values.len());

            for value in &mut values[start..end] {
                *value *= scale;
                // TL2 might have additional post-scaling transformations
                *value = self.apply_tl2_post_scaling(*value)?;
            }
        }

        Ok(())
    }
}

// Updated validation method
impl AccuracyValidator {
    pub fn validate_tl_accuracy(
        &self,
        original: &[f32],
        quantized: &QuantizedTensor,
    ) -> Result<AccuracyReport> {
        let cpu_quantizer = CPUQuantizer::new(self.tolerance_config.clone());

        let dequantized = match quantized.qtype {
            QuantizationType::TL1 => cpu_quantizer.dequantize_tl1(quantized)?,
            QuantizationType::TL2 => cpu_quantizer.dequantize_tl2(quantized)?, // ✅ CORRECT
            _ => {
                return Err(bitnet_common::BitNetError::Quantization(
                    QuantizationError::UnsupportedType {
                        qtype: quantized.qtype.to_string()
                    },
                ));
            }
        };

        let mut report = AccuracyReport::new();
        report.update_errors(original, &dequantized);

        // Add TL2-specific validation metrics
        if matches!(quantized.qtype, QuantizationType::TL2) {
            report.add_tl2_specific_metrics(&dequantized)?;
        }

        info!(
            "TL accuracy validation: type={:?}, relative_error={:.2e}, passed={}",
            quantized.qtype, report.relative_error, report.passed
        );

        Ok(report)
    }
}
```

### Option 2: Add Temporary Error for Unsupported TL2

If TL2 dequantization is not ready, explicitly fail rather than use incorrect logic:

```rust
QuantizationType::TL2 => {
    return Err(bitnet_common::BitNetError::Quantization(
        QuantizationError::UnsupportedOperation {
            operation: "TL2 dequantization".to_string(),
            reason: "TL2 dequantization not yet implemented".to_string(),
        }
    ));
}
```

## Implementation Plan

### Task 1: Implement TL2 Dequantization Logic
- [ ] Research TL2 quantization algorithm specifications
- [ ] Implement `dequantize_tl2` method in `CPUQuantizer`
- [ ] Add TL2-specific lookup table construction
- [ ] Implement TL2-specific scaling and post-processing

### Task 2: Add TL2-Specific Validation
- [ ] Add TL2-specific accuracy metrics to `AccuracyReport`
- [ ] Implement TL2 quantization quality assessment
- [ ] Add comparative analysis between TL1 and TL2 performance
- [ ] Update validation thresholds for TL2-specific characteristics

### Task 3: Update Error Handling
- [ ] Add proper error types for TL2 validation failures
- [ ] Implement detailed error messages for TL2-specific issues
- [ ] Add validation for TL2 tensor format requirements
- [ ] Update error recovery strategies

### Task 4: Comprehensive Testing
- [ ] Add unit tests for TL2 dequantization correctness
- [ ] Implement round-trip tests (quantize -> dequantize -> compare)
- [ ] Add comparison tests between TL1 and TL2 validation
- [ ] Test edge cases specific to TL2 algorithm

## Testing Strategy

### TL2 Dequantization Tests
```rust
#[test]
fn test_tl2_dequantization_correctness() {
    let cpu_quantizer = CPUQuantizer::new(ToleranceConfig::default());

    // Create known TL2 quantized tensor
    let original = vec![0.1, -0.3, 0.7, -0.9, 0.5];
    let quantized = create_tl2_test_tensor(&original);

    let result = cpu_quantizer.dequantize_tl2(&quantized);
    assert!(result.is_ok());

    let dequantized = result.unwrap();
    assert_eq!(dequantized.len(), original.len());

    // Verify dequantization quality
    for (orig, dequant) in original.iter().zip(dequantized.iter()) {
        let error = (orig - dequant).abs();
        assert!(error < 0.1, "TL2 dequantization error too large: {}", error);
    }
}

#[test]
fn test_tl2_validation_accuracy() {
    let validator = AccuracyValidator::new(ToleranceConfig::default());
    let original = generate_test_weights(1000);

    // Create TL2 quantized tensor
    let quantizer = TL2Quantizer::new(128);
    let quantized = quantizer.quantize(&original).unwrap();

    let result = validator.validate_tl_accuracy(&original, &quantized);
    assert!(result.is_ok());

    let report = result.unwrap();
    assert_eq!(report.quantization_type, Some(QuantizationType::TL2));
    assert!(report.passed);
    assert!(report.relative_error < 0.05);
}

#[test]
fn test_tl1_vs_tl2_validation_differences() {
    let validator = AccuracyValidator::new(ToleranceConfig::default());
    let original = generate_test_weights(1000);

    // Compare TL1 and TL2 validation
    let tl1_quantizer = TL1Quantizer::new(128);
    let tl2_quantizer = TL2Quantizer::new(128);

    let tl1_quantized = tl1_quantizer.quantize(&original).unwrap();
    let tl2_quantized = tl2_quantizer.quantize(&original).unwrap();

    let tl1_report = validator.validate_tl_accuracy(&original, &tl1_quantized).unwrap();
    let tl2_report = validator.validate_tl_accuracy(&original, &tl2_quantized).unwrap();

    // Verify they produce different results (proving TL2 validation is distinct)
    assert_ne!(tl1_report.relative_error, tl2_report.relative_error);
    assert_ne!(tl1_report.snr_db, tl2_report.snr_db);
}
```

### Error Handling Tests
```rust
#[test]
fn test_tl2_validation_error_handling() {
    let validator = AccuracyValidator::new(ToleranceConfig::default());
    let original = vec![1.0, 2.0, 3.0];

    // Test with corrupted TL2 tensor
    let mut corrupted_tensor = create_tl2_test_tensor(&original);
    corrupted_tensor.data.clear(); // Corrupt the data

    let result = validator.validate_tl_accuracy(&original, &corrupted_tensor);
    assert!(result.is_err());

    // Verify error type and message
    if let Err(BitNetError::Quantization(QuantizationError::InvalidTensorFormat { .. })) = result {
        // Expected error type
    } else {
        panic!("Expected InvalidTensorFormat error");
    }
}
```

## Related Issues/PRs

- Depends on TL2 quantization algorithm implementation
- Related to comprehensive quantization validation framework
- Part of TL1/TL2 quantization comparison and selection

## Acceptance Criteria

- [ ] TL2 quantization uses proper `dequantize_tl2` method for validation
- [ ] TL2 validation produces different and correct results compared to TL1
- [ ] Error handling properly manages TL2-specific validation failures
- [ ] Round-trip tests (quantize -> dequantize -> validate) pass for TL2
- [ ] Performance metrics show TL2 validation completes within acceptable time
- [ ] Documentation explains TL2 validation approach and differences from TL1

## Risk Assessment

**Medium Risk**: Implementing TL2 dequantization requires understanding of the TL2 algorithm specifics.

**Mitigation Strategies**:
- Start with explicit error for unsupported TL2 validation if algorithm details are unclear
- Implement comprehensive testing to verify TL2 dequantization correctness
- Add validation against known TL2 reference implementations
- Provide fallback behavior and clear error messages for edge cases
