# Simulation: `VectorizedLookupTable::new` in `tl2.rs` has simplified scale and zero point calculation

The `VectorizedLookupTable::new` function in `crates/bitnet-quantization/src/tl2.rs` has a comment "Symmetric quantization for simplicity". It uses a simplified scale and zero point calculation. This is a form of simulation.

**File:** `crates/bitnet-quantization/src/tl2.rs`

**Function:** `VectorizedLookupTable::new`

**Code:**
```rust
impl VectorizedLookupTable {
    /// Create a new vectorized lookup table
    pub fn new(min_val: f32, max_val: f32, bits: u8) -> Self {
        let num_levels = 1 << bits;
        let mut forward = vec![0i8; 256]; // Aligned to 256 for SIMD
        let mut reverse = vec![0.0f32; num_levels];

        // Calculate scale and zero point
        let abs_max = max_val.abs().max(min_val.abs());
        let scale = abs_max / ((num_levels / 2) - 1) as f32;
        let zero_point = 0; // Symmetric quantization for simplicity

        // Build reverse lookup table
        for (i, rev) in reverse.iter_mut().enumerate().take(num_levels) {
            let quantized = i as i32 - (num_levels / 2) as i32;
            *rev = quantized as f32 * scale;
        }

        // Build forward lookup table with SIMD-friendly layout
        for (i, fwd) in forward.iter_mut().enumerate().take(256) {
            let float_val = (i as f32 - 128.0) * scale / 128.0; // Normalize to [-1, 1] range
            let quantized = ((float_val / scale).round() as i32)
                .saturating_add((num_levels / 2) as i32)
                .clamp(0, (num_levels - 1) as i32) as i8;
            *fwd = quantized;
        }

        Self { forward, reverse, scale, _zero_point: zero_point, _num_levels: num_levels }
    }
```

## Proposed Fix

The `VectorizedLookupTable::new` function should be implemented to use a more accurate and optimized scale and zero point calculation. This would involve implementing asymmetric quantization and using a more sophisticated method for calculating the scale and zero point.

### Example Implementation

```rust
impl VectorizedLookupTable {
    /// Create a new vectorized lookup table
    pub fn new(min_val: f32, max_val: f32, bits: u8, use_asymmetric: bool) -> Self {
        let num_levels = 1 << bits;
        let mut forward = vec![0i8; 256]; // Aligned to 256 for SIMD
        let mut reverse = vec![0.0f32; num_levels];

        let (scale, zero_point) = if use_asymmetric {
            let scale = if max_val == min_val {
                1.0
            } else {
                (max_val - min_val) / (num_levels - 1) as f32
            };
            let zero_point = if scale == 0.0 { 0 } else { (-min_val / scale).round() as i32 };
            (scale, zero_point)
        } else {
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = if abs_max == 0.0 {
                1.0
            } else {
                abs_max / ((num_levels / 2).saturating_sub(1)) as f32
            };
            (scale, 0)
        };

        // ... rest of the function ...
    }
}
```
