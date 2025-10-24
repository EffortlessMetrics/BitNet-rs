# GGUF v3 Early Variant Detection Fix - Summary

## Issue

Original warning message during model loading:
```
GGUF v3 early variant detected (missing alignment/data_offset) - handling gracefully
```

This appeared when loading Microsoft BitNet models and other non-standard GGUF v3 files.

## Root Cause

Two GGUF v3 format variants exist in the wild:

1. **Standard GGUF v3** - Spec-compliant with explicit `alignment` (u32) and `data_offset` (u64) fields
2. **Early v3 Variant** - Non-standard format missing these fields, going directly to KV pairs

The Microsoft BitNet model (`microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`) uses the early variant.

## Verification

### Microsoft Model (Early Variant)
```
Offset | Bytes                  | Field
-------|------------------------|-------------------
0x00   | 47 47 55 46           | magic ("GGUF")
0x04   | 03 00 00 00           | version (3)
0x08   | 4c 01 00 00 00 00 00 00 | tensor_count (332)
0x10   | 18 00 00 00 00 00 00 00 | kv_count (24)
0x18   | 14 00 00 00 00 00 00 00 | KV pair (key length = 20) ← Missing alignment/data_offset!
0x20   | 67 65 6e 65 72 61 6c... | "general.architec..." (first KV key)
```

### BitNet.rs Exported Model (Standard v3)
```
Offset | Bytes                  | Field
-------|------------------------|-------------------
0x00   | 47 47 55 46           | magic ("GGUF")
0x04   | 03 00 00 00           | version (3)
0x08   | 3a 01 00 00 00 00 00 00 | tensor_count (314)
0x10   | 0a 00 00 00 00 00 00 00 | kv_count (10)
0x18   | 20 00 00 00           | alignment (32) ✅
0x1c   | 00 5f 00 00 00 00 00 00 | data_offset (24320) ✅
0x24   | 0c 00 00 00 00 00 00 00 | KV pair (key length = 12)
```

## Changes Implemented

### 1. Improved Detection Logging (`crates/bitnet-models/src/formats/gguf/types.rs`)

**Before:**
```rust
tracing::warn!(
    "GGUF v3 early variant detected (missing alignment/data_offset) - handling gracefully"
);
```

**After:**
```rust
tracing::debug!(
    "GGUF v3 early variant detected (missing alignment/data_offset fields) - using defaults (align=32, data_offset=computed). \
    This is a non-standard format used by some models. Consider re-exporting with proper v3 structure for full compliance."
);
```

- Changed from `warn!` to `debug!` (reduces noise)
- Added actionable guidance for users
- More detailed explanation

### 2. Added Format Introspection API (`crates/bitnet-models/src/formats/gguf/types.rs`)

```rust
impl GgufHeader {
    /// Check if this header represents a standard GGUF v3 file
    pub fn is_standard_v3(&self) -> bool {
        self.version >= 3 && self.data_offset > 0
    }

    /// Check if this header represents an early v3 variant
    pub fn is_early_v3_variant(&self) -> bool {
        self.version >= 3 && self.data_offset == 0
    }

    /// Get a description of the GGUF format variant
    pub fn format_description(&self) -> String {
        match self.version {
            2 => "GGUF v2 (legacy)".to_string(),
            3 if self.is_standard_v3() => {
                format!("GGUF v3 (standard, align={}, data_offset={})",
                    self.alignment, self.data_offset)
            }
            3 => "GGUF v3 (early variant, missing alignment/data_offset fields)".to_string(),
            v => format!("GGUF v{} (unknown)", v),
        }
    }
}
```

### 3. Added Format Detection Logging (`crates/bitnet-models/src/formats/gguf/reader.rs`)

```rust
// Log format variant for diagnostic purposes
tracing::debug!("Detected GGUF format: {}", header.format_description());
```

Users can now see format details with `RUST_LOG=debug`.

### 4. Validation Tests (`crates/bitnet-models/src/formats/gguf/tests.rs`)

Added comprehensive tests:
- `test_gguf_v3_early_variant_detection` - Validates early variant detection
- `test_gguf_v3_standard_detection` - Validates standard v3 detection
- `test_gguf_v3_invalid_alignment_clamped` - Tests error handling

**Test Results:**
```
running 3 tests
test formats::gguf::tests::test_gguf_v3_invalid_alignment_clamped ... ok
test formats::gguf::tests::test_gguf_v3_standard_detection ... ok
test formats::gguf::tests::test_gguf_v3_early_variant_detection ... ok
```

### 5. Writer Validation (`crates/bitnet-st2gguf/src/writer.rs`)

Added test to verify our export tool produces standard v3:

```rust
#[test]
fn test_writer_produces_standard_v3_format() -> Result<()> {
    // ... creates GGUF file ...

    let header = &reader.header;
    assert!(header.is_standard_v3());
    assert_eq!(header.alignment, 32);
    assert!(header.data_offset > 0);
    assert_eq!(header.data_offset % header.alignment as u64, 0);

    Ok(())
}
```

**Test Result:**
```
test writer::tests::test_writer_produces_standard_v3_format ... ok
```

### 6. Documentation (`docs/reference/gguf-v3-variants.md`)

Comprehensive documentation covering:
- Format variant details with byte-level layouts
- Detection heuristics and implementation
- Migration guide for converting early variant to standard v3
- Recommendations for model authors, tool developers, and users
- Code examples

### 7. Made Header Public (`crates/bitnet-models/src/formats/gguf/reader.rs`)

```rust
pub struct GgufReader<'a> {
    data: &'a [u8],
    pub header: GgufHeader,  // Now public for introspection
    // ...
}
```

Allows users to check format variant programmatically.

## Verification

### Model Status

| Model | Format | Status |
|-------|--------|--------|
| `microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` | Early v3 variant | ✅ Works (detected automatically) |
| `models/clean/clean-f16.gguf` | Standard v3 | ✅ Proper alignment/data_offset |
| `models/clean/clean-f16-fixed.gguf` | Standard v3 | ✅ Proper alignment/data_offset |

### User Experience

**Before:**
```
WARN  GGUF v3 early variant detected (missing alignment/data_offset) - handling gracefully
```
(Warning level logs, no context on what this means)

**After:**
```bash
# Default (no debug logging)
# (No warning - clean output)

# With debug logging
RUST_LOG=debug cargo run -p bitnet-cli -- run --model model.gguf --prompt "Test"
```
```
DEBUG Detected GGUF format: GGUF v3 (early variant, missing alignment/data_offset fields)
DEBUG GGUF v3 early variant detected (missing alignment/data_offset fields) - using defaults...
```

## Compatibility

### Loader Behavior

| Scenario | Loader Action | User Impact |
|----------|---------------|-------------|
| Standard v3 | Uses explicit `alignment` and `data_offset` fields | ✅ Optimal |
| Early v3 variant | Uses default `alignment=32`, computes `data_offset` | ✅ Transparent |
| Invalid alignment | Clamps to 32 (power of 2) | ✅ Graceful degradation |
| data_offset=0 in v3 | Treats as early variant, computes offset | ✅ Fallback |

### Export Behavior

BitNet.rs `st2gguf` tool **always** produces standard GGUF v3:
- ✅ Explicit `alignment` field (32 bytes)
- ✅ Explicit `data_offset` field (properly aligned)
- ✅ All tests passing

## Testing

### Unit Tests
```
cargo test -p bitnet-models --lib -- formats::gguf::tests::test_gguf_v3
```
Result: **3 tests passing**

### Integration Tests
```
cargo test -p bitnet-st2gguf test_writer_produces_standard_v3_format
```
Result: **2 tests passing** (lib + main)

### Regression Tests
All existing GGUF loading tests still pass - no breaking changes.

## Migration Path

For users with early variant models:

1. **No action required** - Loader handles transparently
2. **Optional re-export** - For full compliance:
   ```bash
   # If you have SafeTensors source
   cargo run -p bitnet-st2gguf -- \
     --input model.safetensors \
     --output standard-v3.gguf
   ```
3. **Verify with debug logging**:
   ```bash
   RUST_LOG=debug cargo run -p bitnet-cli -- run \
     --model model.gguf --prompt "Test" --max-tokens 4
   ```

## Files Modified

1. `crates/bitnet-models/src/formats/gguf/types.rs` - Detection & API
2. `crates/bitnet-models/src/formats/gguf/reader.rs` - Logging & public header
3. `crates/bitnet-models/src/formats/gguf/tests.rs` - Validation tests
4. `crates/bitnet-st2gguf/src/writer.rs` - Export validation test
5. `docs/reference/gguf-v3-variants.md` - Comprehensive documentation

## Recommendations

### For Users
- ✅ No action needed - loader handles both variants automatically
- ℹ️ Use `RUST_LOG=debug` to see which variant is detected
- ℹ️ Consider re-exporting from SafeTensors for full compliance

### For Model Authors
- ✅ Use BitNet.rs `st2gguf` for new models (produces standard v3)
- ℹ️ Existing early variant models work fine, but standard v3 is preferred
- ℹ️ Re-export when convenient for better interoperability

### For Tool Developers
- ✅ Implement support for both variants (like BitNet.rs does)
- ✅ Always write standard v3 when creating new files
- ✅ Provide clear diagnostics for format detection

## Summary

**Problem:** Warning logs for non-standard GGUF v3 models
**Solution:** Better detection, logging, validation, and documentation
**Status:** ✅ Complete - All tests passing, both variants supported
**User Impact:** Cleaner logs, better diagnostics, transparent handling
