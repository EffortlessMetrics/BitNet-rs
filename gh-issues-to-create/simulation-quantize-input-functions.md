# Simulation: `utils::quantize_input` functions in `quantized_linear.rs` are simple quantizations

The `utils::quantize_input_i2s`, `quantize_input_tl1`, `quantize_input_tl2` functions in `crates/bitnet-inference/src/layers/quantized_linear.rs` perform simple clamping and rounding for quantization. They might not be the most optimal or accurate quantization methods. This is a form of simulation.

**File:** `crates/bitnet-inference/src/layers/quantized_linear.rs`

**Functions:**
* `utils::quantize_input_i2s`
* `utils::quantize_input_tl1`
* `utils::quantize_input_tl2`

**Code:**
```rust
    pub fn quantize_input_i2s(input: &[f32], _features: usize) -> Result<Vec<i8>> {
        // Simple quantization to I2S range [-2, 1]
        let quantized: Vec<i8> = input
            .iter()
            .map(|&x| {
                let clamped = x.clamp(-2.0, 1.0);
                clamped.round() as i8
            })
            .collect();

        Ok(quantized)
    }
```

## Proposed Fix

The `utils::quantize_input` functions should be implemented to use more optimal and accurate quantization methods. This would involve using more sophisticated quantization algorithms that take into account the distribution of the input data.

### Example Implementation

```rust
    pub fn quantize_input_i2s(input: &[f32], _features: usize) -> Result<Vec<i8>> {
        // Example: Use a more sophisticated quantization algorithm
        let quantized: Vec<i8> = input
            .iter()
            .map(|&x| {
                // Implement a more advanced quantization algorithm here
                // For example, a min-max quantization or a k-means based quantization
                let scale = 1.0; // Example scale
                let zero_point = 0; // Example zero point
                (x / scale + zero_point as f32).round().clamp(-2.0, 1.0) as i8
            })
            .collect();

        Ok(quantized)
    }
```
