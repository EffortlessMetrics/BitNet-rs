# Simulation: `quantize_value` and `dequantize_value` in `utils.rs` are simplified implementations

The `quantize_value` and `dequantize_value` functions in `crates/bitnet-quantization/src/utils.rs` perform simple clamping and rounding for quantization. They might not be the most optimal or accurate quantization methods. This is a form of simulation.

**File:** `crates/bitnet-quantization/src/utils.rs`

**Functions:**
* `quantize_value`
* `dequantize_value`

**Code:**
```rust
#[inline]
pub fn quantize_value(value: f32, scale: f32, bits: u8) -> i8 {
    let max_val = (1 << (bits - 1)) - 1;
    let min_val = -(1 << (bits - 1));

    // Fast path for typical values
    if value.is_finite() && scale != 0.0 && scale.is_finite() {
        let normalized = value / scale;
        let quantized = normalized.round() as i32;
        return quantized.clamp(min_val, max_val) as i8;
    }

    // Fallback for edge cases
    0i8
}

#[inline]
pub fn dequantize_value(quantized: i8, scale: f32) -> f32 {
    // Fast path for typical values
    if scale.is_finite() {
        quantized as f32 * scale
    } else {
        0.0 // Safe fallback for invalid scale
    }
}
```

## Proposed Fix

The `quantize_value` and `dequantize_value` functions should be implemented to use more optimal and accurate quantization methods. This would involve using more sophisticated quantization algorithms that take into account the distribution of the input data.

### Example Implementation

```rust
#[inline]
pub fn quantize_value(value: f32, scale: f32, bits: u8) -> i8 {
    // Example: Use a more sophisticated quantization algorithm
    let max_val = (1 << (bits - 1)) - 1;
    let min_val = -(1 << (bits - 1));

    if value.is_finite() && scale != 0.0 && scale.is_finite() {
        let normalized = value / scale;
        let quantized = if normalized > 0.5 {
            1i32
        } else if normalized < -0.5 {
            -1i32
        } else {
            0i32
        };
        return quantized.clamp(min_val, max_val) as i8;
    }

    0i8
}
```
