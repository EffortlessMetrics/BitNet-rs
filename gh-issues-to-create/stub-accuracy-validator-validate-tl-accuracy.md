# Stub code: `AccuracyValidator::validate_tl_accuracy` in `device_aware_quantizer.rs` falls back to `dequantize_tl1` for TL2

The `AccuracyValidator::validate_tl_accuracy` function in `crates/bitnet-quantization/src/device_aware_quantizer.rs` falls back to `cpu_quantizer.dequantize_tl1(quantized)?` for TL2 quantization. It has a comment "TL2 would have its own implementation". This is a form of stubbing.

**File:** `crates/bitnet-quantization/src/device_aware_quantizer.rs`

**Function:** `AccuracyValidator::validate_tl_accuracy`

**Code:**
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
                cpu_quantizer.dequantize_tl1(quantized)?
            }
            _ => {
                return Err(bitnet_common::BitNetError::Quantization(
                    QuantizationError::UnsupportedType { qtype: quantized.qtype.to_string() },
                ));
            }
        };

        // ...
    }
```

## Proposed Fix

The `AccuracyValidator::validate_tl_accuracy` function should be implemented to use a proper TL2 dequantization. This would involve calling `cpu_quantizer.dequantize_tl2(quantized)?` instead of `cpu_quantizer.dequantize_tl1(quantized)?`.

### Example Implementation

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
                cpu_quantizer.dequantize_tl2(quantized)?
            }
            _ => {
                return Err(bitnet_common::BitNetError::Quantization(
                    QuantizationError::UnsupportedType { qtype: quantized.qtype.to_string() },
                ));
            }
        };

        // ...
    }
```
