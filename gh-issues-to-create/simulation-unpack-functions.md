# Simulation: `utils::unpack` functions in `quantized_linear.rs` are simple unpackings

The `utils::unpack_2bit_values`, `unpack_tl1_values`, `unpack_tl2_values` functions in `crates/bitnet-inference/src/layers/quantized_linear.rs` perform simple bit manipulation for unpacking. They might not be the most optimal or efficient unpacking methods. This is a form of simulation.

**File:** `crates/bitnet-inference/src/layers/quantized_linear.rs`

**Functions:**
* `utils::unpack_2bit_values`
* `utils::unpack_tl1_values`
* `utils::unpack_tl2_values`

**Code:**
```rust
    pub fn unpack_2bit_values(packed: &[u8], count: usize) -> Vec<i8> {
        let mut unpacked = Vec::with_capacity(count);

        for &byte in packed.iter() {
            if unpacked.len() >= count {
                break;
            }

            // Extract 4 values from each byte (2 bits each)
            for shift in [0, 2, 4, 6] {
                if unpacked.len() >= count {
                    break;
                }
                let value = ((byte >> shift) & 0b11) as i8;
                // Convert from [0,3] to [-2,1] range for I2S
                unpacked.push(value - 2);
            }
        }

        unpacked.truncate(count);
        unpacked
    }
```

## Proposed Fix

The `utils::unpack` functions should be implemented to use more optimal and efficient unpacking methods. This would involve using SIMD instructions or other low-level optimizations to unpack the values.

### Example Implementation

```rust
    pub fn unpack_2bit_values(packed: &[u8], count: usize) -> Vec<i8> {
        let mut unpacked = Vec::with_capacity(count);

        // Example: Using SIMD instructions for faster unpacking
        #[cfg(target_feature = "sse2")]
        unsafe {
            // ... SIMD implementation ...
        }
        #[cfg(not(target_feature = "sse2"))]
        {
            for &byte in packed.iter() {
                if unpacked.len() >= count {
                    break;
                }

                // Extract 4 values from each byte (2 bits each)
                for shift in [0, 2, 4, 6] {
                    if unpacked.len() >= count {
                        break;
                    }
                    let value = ((byte >> shift) & 0b11) as i8;
                    // Convert from [0,3] to [-2,1] range for I2S
                    unpacked.push(value - 2);
                }
            }
        }

        unpacked.truncate(count);
        unpacked
    }
```
