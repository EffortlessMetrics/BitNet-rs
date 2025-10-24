# Stub code: `TL2Config::default` in `tl2.rs` has conditional compilation

The `TL2Config::default` function in `crates/bitnet-quantization/src/tl2.rs` has conditional compilation for `x86` and `x86_64` architectures. If neither is detected, it sets `use_avx512` and `use_avx2` to `false`. This is a form of stubbing.

**File:** `crates/bitnet-quantization/src/tl2.rs`

**Function:** `TL2Config::default`

**Code:**
```rust
impl Default for TL2Config {
    fn default() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let (use_avx512, use_avx2) =
            (is_x86_feature_detected!("avx512f"), is_x86_feature_detected!("avx2"));
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let (use_avx512, use_avx2) = (false, false);

        Self {
            block_size: 128, // Larger blocks for x86 vectorization
            lookup_table_size: 256,
            use_avx512,
            use_avx2,
            precision_bits: 2,
            vectorized_tables: true,
        }
    }
}
```

## Proposed Fix

The `TL2Config::default` function should not have conditional compilation for `x86` and `x86_64` architectures. Instead, the CPU feature detection should be performed at runtime and the `use_avx512` and `use_avx2` fields should be set accordingly.

### Example Implementation

```rust
impl Default for TL2Config {
    fn default() -> Self {
        let (use_avx512, use_avx2) = if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
            (is_x86_feature_detected!("avx512f"), is_x86_feature_detected!("avx2"))
        } else {
            (false, false)
        };

        Self {
            block_size: 128, // Larger blocks for x86 vectorization
            lookup_table_size: 256,
            use_avx512,
            use_avx2,
            precision_bits: 2,
            vectorized_tables: true,
        }
    }
}
```
