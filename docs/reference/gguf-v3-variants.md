# GGUF v3 Format Variants

## Overview

BitNet-rs supports two GGUF v3 format variants found in the wild:

1. **Standard GGUF v3** - Specification-compliant format with explicit alignment and data_offset fields
2. **Early v3 Variant** - Non-standard format used by some models (e.g., Microsoft BitNet models)

## Format Details

### Standard GGUF v3 (Recommended)

**Header Structure:**
```
Offset | Size | Field
-------|------|-------------------
0      | 4    | magic ("GGUF")
4      | 4    | version (3)
8      | 8    | tensor_count (u64)
16     | 8    | metadata_kv_count (u64)
24     | 4    | alignment (u32)     ← Present in standard v3
28     | 8    | data_offset (u64)   ← Present in standard v3
36     | ...  | metadata KV pairs
```

**Characteristics:**
- ✅ Full GGUF v3 specification compliance
- ✅ Explicit 32-byte alignment field
- ✅ Explicit data_offset field pointing to tensor data start
- ✅ Better for interoperability with other GGUF tools
- ✅ Produced by BitNet-rs `st2gguf` export tool

**Detection:**
- `header.version >= 3`
- `header.data_offset > 0`
- `header.is_standard_v3() == true`

### Early v3 Variant (Legacy)

**Header Structure:**
```
Offset | Size | Field
-------|------|-------------------
0      | 4    | magic ("GGUF")
4      | 4    | version (3)
8      | 8    | tensor_count (u64)
16     | 8    | metadata_kv_count (u64)
24     | ...  | metadata KV pairs  ← No alignment/data_offset fields
```

**Characteristics:**
- ⚠️ Non-standard GGUF v3 format
- ⚠️ Missing alignment and data_offset fields
- ⚠️ Loader must compute alignment (defaults to 32 bytes)
- ⚠️ Loader must compute data_offset from KV section end
- ℹ️ Used by some existing models (e.g., `microsoft/bitnet-b1.58-2B-4T-gguf`)

**Detection:**
- `header.version >= 3`
- `header.data_offset == 0`
- `header.is_early_v3_variant() == true`

**Handling:**
BitNet-rs loader automatically detects early variant by checking if bytes at offset 24 look like a string length (start of KV pair) rather than alignment field. When detected:
- Sets `alignment = 32` (standard default)
- Sets `data_offset = 0` (marker for early variant)
- Computes actual data offset from `align_up(kv_end, 32)`

## Compatibility

### Loading Models

BitNet-rs **automatically handles both variants** without user intervention:

```bash
# Works for both standard and early variant
cargo run -p bitnet-cli -- run \
  --model models/model.gguf \
  --prompt "Test" \
  --max-tokens 16
```

**Logging:**
- Standard v3: `GGUF v3 (standard, align=32, data_offset=<value>)`
- Early variant: `GGUF v3 (early variant, missing alignment/data_offset fields)`

### Exporting Models

BitNet-rs `st2gguf` tool **always produces standard GGUF v3**:

```bash
# Export SafeTensors to proper GGUF v3
cargo run -p bitnet-st2gguf -- \
  --input model.safetensors \
  --output model.gguf
```

**Validation:**
```rust
use bitnet_models::formats::gguf::GgufReader;

let file = File::open("model.gguf")?;
let mmap = unsafe { memmap2::Mmap::map(&file)? };
let reader = GgufReader::new(&mmap)?;

assert!(reader.header.is_standard_v3());
assert_eq!(reader.header.alignment, 32);
assert!(reader.header.data_offset > 0);
```

## Migration Guide

### Converting Early Variant to Standard v3

If you have an early variant model and want to convert it to standard v3:

1. **Check current format:**
   ```bash
   cargo run -p bitnet-cli -- compat-check model.gguf
   ```

2. **Re-export (if you have SafeTensors source):**
   ```bash
   cargo run -p bitnet-st2gguf -- \
     --input original.safetensors \
     --output standard-v3.gguf
   ```

3. **Verify standard v3:**
   ```bash
   # Should show "GGUF v3 (standard, ...)"
   RUST_LOG=debug cargo run -p bitnet-cli -- run \
     --model standard-v3.gguf \
     --prompt "Test" \
     --max-tokens 4
   ```

**Note:** Direct conversion from early variant GGUF to standard v3 GGUF is not currently supported. Re-export from SafeTensors if available.

## Implementation Details

### Detection Heuristic

The loader uses a heuristic to distinguish between standard v3 and early variant:

1. Read bytes at offset 24 (where alignment field should be)
2. Interpret as u64 string length
3. If length is reasonable (0-256) and followed by ASCII text matching key pattern `[A-Za-z0-9._-]`, assume early variant
4. Otherwise, parse as standard v3 alignment + data_offset

### Code Example

```rust
use bitnet_models::formats::gguf::{GgufHeader, GgufReader};
use std::fs::File;

// Load any GGUF v3 file (standard or early variant)
let file = File::open("model.gguf")?;
let mmap = unsafe { memmap2::Mmap::map(&file)? };
let reader = GgufReader::new(&mmap)?;

// Check variant
if reader.header.is_standard_v3() {
    println!("Standard GGUF v3 detected");
    println!("Alignment: {}", reader.header.alignment);
    println!("Data offset: {}", reader.header.data_offset);
} else if reader.header.is_early_v3_variant() {
    println!("Early v3 variant detected (non-standard)");
    println!("Using default alignment: 32");
    println!("Data offset computed from KV section end");
}

// Get human-readable description
println!("{}", reader.header.format_description());
```

## Recommendations

### For Model Authors

- **New models**: Use BitNet-rs `st2gguf` to produce standard GGUF v3
- **Existing early variant models**: Can be used as-is (loader handles transparently)
- **Re-export when possible**: Convert early variant to standard v3 for better interoperability

### For Tool Developers

- **Implement both variants**: Support both standard and early variant for maximum compatibility
- **Prefer standard v3**: Always write standard v3 when creating new files
- **Fail gracefully**: Provide clear error messages when encountering unknown variants

### For Users

- **Trust the loader**: BitNet-rs automatically detects and handles both variants
- **Check logs**: `RUST_LOG=debug` shows which variant was detected
- **Report issues**: If you encounter loading problems, check the format description in logs

## References

- GGUF Specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- BitNet-rs GGUF Loader: `crates/bitnet-models/src/formats/gguf/`
- Export Tool: `crates/bitnet-st2gguf/`
