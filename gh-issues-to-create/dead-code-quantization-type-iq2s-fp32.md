# Dead code: `QuantizationType::IQ2S`, `FP32` in `device_aware_quantizer.rs` are defined but not used

The `QuantizationType` enum in `crates/bitnet-quantization/src/device_aware_quantizer.rs` defines `IQ2S` and `FP32`, but these variants are not used in the `quantize_with_validation` function. This is a form of dead code.

**File:** `crates/bitnet-quantization/src/device_aware_quantizer.rs`

**Enum Variants:**
* `QuantizationType::IQ2S`
* `QuantizationType::FP32`

**Code:**
```rust
pub enum QuantizationType {
    /// 2-bit signed quantization (BitNet native)
    I2S,
    /// Table lookup quantization 1
    TL1,
    /// Table lookup quantization 2
    TL2,
    /// IQ2_S quantization (GGML compatible)
    IQ2S,
    /// Full precision (reference)
    FP32,
}
```

## Proposed Fix

If `QuantizationType::IQ2S` and `FP32` are not intended to be used, they should be removed to reduce the size of the codebase and improve maintainability. If they are intended to be used, they should be integrated into the `quantize_with_validation` function.

### Example Implementation

```rust
    pub fn quantize_with_validation(
        &self,
        weights: &[f32],
        quant_type: QuantizationType,
    ) -> Result<QuantizedTensor> {
        let start_time = Instant::now();

        let quantized = match quant_type {
            QuantizationType::I2S => self.cpu_backend.quantize_i2s(weights)?,
            QuantizationType::TL1 => self.cpu_backend.quantize_tl1(weights)?,
            QuantizationType::TL2 => self.cpu_backend.quantize_tl1(weights)?, // Simplified
            QuantizationType::IQ2S => self.cpu_backend.quantize_iq2s(weights)?,
            QuantizationType::FP32 => self.cpu_backend.quantize_fp32(weights)?,
        };

        // ...
    }
```
