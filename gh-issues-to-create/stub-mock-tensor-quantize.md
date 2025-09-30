# Stub code: `MockTensor::quantize` in `property_tests.rs` directly calls quantizers

The `MockTensor::quantize` method in `crates/bitnet-quantization/src/property_tests.rs` directly calls `crate::i2s::quantize_i2s`, `crate::tl1::quantize_tl1`, and `crate::tl2::quantize_tl2`. This bypasses the `QuantizerTrait` and `DeviceAwareQuantizer`, which might hide issues with the quantizer selection logic. This is a form of stubbing.

**File:** `crates/bitnet-quantization/src/property_tests.rs`

**Function:** `MockTensor::quantize`

**Code:**
```rust
#[cfg(test)]
impl Quantize for MockTensor {
    fn quantize(&self, qtype: QuantizationType) -> Result<QuantizedTensor, crate::BitNetError> {
        match qtype {
            QuantizationType::I2S => crate::i2s::quantize_i2s(self),
            QuantizationType::TL1 => crate::tl1::quantize_tl1(self),
            QuantizationType::TL2 => crate::tl2::quantize_tl2(self),
        }
    }
}
```

## Proposed Fix

The `MockTensor::quantize` method should use the `DeviceAwareQuantizer` to quantize the tensor. This will ensure that the quantizer selection logic is properly tested.

### Example Implementation

```rust
#[cfg(test)]
impl Quantize for MockTensor {
    fn quantize(&self, qtype: QuantizationType) -> Result<QuantizedTensor, crate::BitNetError> {
        let quantizer = crate::device_aware_quantizer::DeviceAwareQuantizer::new();
        quantizer.quantize_with_validation(self.as_slice::<f32>().unwrap(), qtype)
    }
}
```
