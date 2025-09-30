# Stub code: `QuantizedLinear::new_tl1` and `new_tl2` have unused `_lookup_table` parameter

The `_lookup_table` parameter in `QuantizedLinear::new_tl1` and `new_tl2` in `crates/bitnet-inference/src/layers/quantized_linear.rs` is marked with `_` indicating it's unused. This suggests that the lookup table is not actually being used in the TL1 and TL2 implementations. This is a form of stubbing.

**File:** `crates/bitnet-inference/src/layers/quantized_linear.rs`

**Functions:**
* `QuantizedLinear::new_tl1`
* `QuantizedLinear::new_tl2`

**Code:**
```rust
    pub fn new_tl1(
        weights: QuantizedTensor,
        _lookup_table: LookupTable,
        device: Device,
    ) -> Result<Self> {
        // ...
    }

    pub fn new_tl2(
        weights: QuantizedTensor,
        _lookup_table: LookupTable,
        device: Device,
    ) -> Result<Self> {
        // ...
    }
```

## Proposed Fix

The `_lookup_table` parameter should be used in the TL1 and TL2 implementations. This would involve using the lookup table to perform the quantization and dequantization operations.

### Example Implementation

```rust
    pub fn new_tl1(
        weights: QuantizedTensor,
        lookup_table: LookupTable,
        device: Device,
    ) -> Result<Self> {
        // ...
        layer.lookup_table = Some(lookup_table);
        // ...
    }
```
