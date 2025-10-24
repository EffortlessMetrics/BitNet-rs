# Simulation: `pack_2bit_values` and `unpack_2bit_values` in `utils.rs` are simplified implementations

The `pack_2bit_values` and `unpack_2bit_values` functions in `crates/bitnet-quantization/src/utils.rs` perform simple bit manipulation for packing and unpacking. They might not be the most optimal or efficient methods. This is a form of simulation.

**File:** `crates/bitnet-quantization/src/utils.rs`

**Functions:**
* `pack_2bit_values`
* `unpack_2bit_values`

**Code:**
```rust
pub fn pack_2bit_values(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(4));

    for chunk in values.chunks(4) {
        let mut byte = 0u8;
        for (i, &val) in chunk.iter().enumerate() {
            // Clamp to 2-bit signed range [-2, 1]
            let clamped = val.clamp(-2, 1);
            // Convert to unsigned 2-bit [0, 3]
            let unsigned = (clamped + 2) as u8;
            byte |= unsigned << (i * 2);
        }
        packed.push(byte);
    }

    packed
}

pub fn unpack_2bit_values(packed: &[u8], output_len: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(output_len);

    for &byte in packed {
        for i in 0..4 {
            if values.len() >= output_len {
                break;
            }
            let unsigned = (byte >> (i * 2)) & 0x3;
            let signed = unsigned as i8 - 2; // Convert back to signed [-2, 1]
            values.push(signed);
        }
    }

    values
}
```

## Proposed Fix

The `pack_2bit_values` and `unpack_2bit_values` functions should be implemented to use more optimal and efficient methods. This would involve using SIMD instructions or other low-level optimizations to pack and unpack the values.

### Example Implementation

```rust
pub fn pack_2bit_values(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(4));

    #[cfg(target_feature = "sse2")]
    unsafe {
        // ... SIMD implementation ...
    }
    #[cfg(not(target_feature = "sse2"))]
    {
        for chunk in values.chunks(4) {
            let mut byte = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                let clamped = val.clamp(-2, 1);
                let unsigned = (clamped + 2) as u8;
                byte |= unsigned << (i * 2);
            }
            packed.push(byte);
        }
    }

    packed
}
```
