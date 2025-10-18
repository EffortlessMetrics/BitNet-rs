# GGML I2_S (QK=256) LUT Verification

This document describes how to verify the correct code→float mapping for GGML's I2_S quantization format using the C++ reference helper.

## Purpose

The Rust kernel in `crates/bitnet-models/src/quant/i2s_qk256.rs` needs to match llama.cpp's dequantization exactly. The 2-bit code→float mapping may vary depending on:

1. **Base LUT values**: What do codes 0,1,2,3 map to?
   - Symmetric: {-3, -1, +1, +3}?
   - Zero-point: {-1, 0, +1, +2}?
   - Other: {-2, -1, +1, +2}?

2. **Scale factors**: Does GGML apply a global/per-tensor/per-row scale?

3. **Zero-point offset**: Is there an offset added after decoding?

## How to Verify

### Step 1: Link against llama.cpp

The helper `i2s_qk256_dumper.cc` needs llama.cpp's actual dequantization function. Two options:

**Option A: Copy implementation**
```bash
# Copy ggml-quants.h and the relevant dequant_row_iq2_s function
# from llama.cpp into this file, then compile standalone
g++ -std=c++17 -O2 i2s_qk256_dumper.cc -o i2s_qk256_dumper
```

**Option B: Link against llama.cpp**
```bash
# Compile llama.cpp first, then link
cd /path/to/llama.cpp
make
cd /path/to/BitNet-rs/crates/bitnet-sys/csrc
g++ -std=c++17 -O2 \
  -I/path/to/llama.cpp/include \
  i2s_qk256_dumper.cc \
  /path/to/llama.cpp/libggml.a \
  -o i2s_qk256_dumper
```

### Step 2: Run the helper

```bash
./i2s_qk256_dumper path/to/model.gguf
```

This will output:
- Dequantized values for codes 0, 1, 2, 3 (test blocks 1-4)
- Pattern validation (test block 5)

### Step 3: Compare with Rust kernel

Check `crates/bitnet-models/src/quant/i2s_qk256.rs`:

```rust
pub fn code_to_f32(code: u8) -> f32 {
    const LUT: [f32; 4] = [-3.0, -1.0, 1.0, 3.0]; // Update if needed!
    LUT[code as usize]
}
```

**If the C++ helper shows different values**, update the LUT in Rust to match **exactly**.

### Step 4: Verify with real model weights

To test with actual GGUF weights:

```bash
# Modify the helper to:
# 1. Parse GGUF file
# 2. Extract first few blocks from Q/K/V/O or FFN tensors
# 3. Dequantize and print values

# Then compare with Rust implementation:
cargo test -p bitnet-models --test i2s_qk256_parity -- --nocapture
```

## Expected Outcomes

| Code | Likely Mapping (verify!) |
|------|--------------------------|
| 0    | -3.0 or -2.0             |
| 1    | -1.0                     |
| 2    | +1.0                     |
| 3    | +3.0 or +2.0             |

**IMPORTANT**: Do NOT assume the placeholder LUT is correct. Run the helper and verify!

## Troubleshooting

**Q: Helper shows all zeros**
- Check if `dequantize_row_iq2_s_reference` is linked correctly
- Verify the `block_iq2_s` struct matches llama.cpp's definition

**Q: Values have unexpected scale**
- GGML may apply a per-tensor or global scale factor
- Check for `d` (scale) field in `block_iq2_s` struct
- Look for scale metadata in GGUF tensor info

**Q: Pattern doesn't match**
- Verify bit packing order (LSB-first vs MSB-first)
- Check if endianness affects reading

## After Verification

Once the LUT is confirmed:

1. Update `code_to_f32()` in `i2s_qk256.rs` if needed
2. Add the confirmed mapping to docstring
3. Run `cargo test -p bitnet-models` to verify all tests pass
4. Delete or archive this helper (no longer needed)
