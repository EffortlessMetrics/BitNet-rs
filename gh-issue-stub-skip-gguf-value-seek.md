# [GGUF Parser] Fix incomplete type handling in GGUF value seeking

## Problem Description

The `skip_gguf_value_seek` function in `crates/bitnet-models/src/gguf_min.rs` has a commented out case for uint8/int8 types, indicating incomplete GGUF type support that could lead to parsing failures for certain GGUF files.

## Root Cause Analysis

### Current Implementation
```rust
fn skip_gguf_value_seek<R: Read + Seek>(r: &mut R, ty: u32) -> Result<()> {
    match ty {
        /* ~ changed by cargo-mutants ~ */ // uint8 | int8  <-- MISSING CASE
        2 | 3 => skip_n_seek(r, 2)?, // uint16 | int16
        4..=6 => skip_n_seek(r, 4)?, // uint32 | int32 | float32
        7 => skip_n_seek(r, 1)?,     // bool
        // ... rest of implementation
    }
}
```

### Issues Identified
1. **Missing Type Support**: uint8 (type 0) and int8 (type 1) are not handled
2. **Parsing Failures**: GGUF files with 8-bit metadata will fail to parse
3. **Incomplete Specification**: Function doesn't handle all GGUF data types

## Proposed Solution

### Complete GGUF Type Implementation

```rust
fn skip_gguf_value_seek<R: Read + Seek>(r: &mut R, ty: u32) -> Result<()> {
    // GGUF scalar sizes (see llama.cpp specification)
    match ty {
        0 | 1 => skip_n_seek(r, 1)?,     // uint8 | int8
        2 | 3 => skip_n_seek(r, 2)?,     // uint16 | int16
        4..=6 => skip_n_seek(r, 4)?,     // uint32 | int32 | float32
        7 => skip_n_seek(r, 1)?,         // bool
        8 => {
            // string: u64 len + bytes
            let n = read_u64(r)?;
            skip_n_seek(r, n)?;
        }
        9 => {
            // array: elem_ty + count + values
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_gguf_value_seek(r, elem_ty)?;
            }
        }
        10..=12 => skip_n_seek(r, 8)?,   // uint64 | int64 | float64
        _ => bail!("unknown GGUF kv type id {ty}"),
    }
    Ok(())
}

// Also add comprehensive GGUF type constants for clarity
mod gguf_types {
    pub const UINT8: u32 = 0;
    pub const INT8: u32 = 1;
    pub const UINT16: u32 = 2;
    pub const INT16: u32 = 3;
    pub const UINT32: u32 = 4;
    pub const INT32: u32 = 5;
    pub const FLOAT32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const UINT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const FLOAT64: u32 = 12;
}

fn skip_gguf_value_seek<R: Read + Seek>(r: &mut R, ty: u32) -> Result<()> {
    use gguf_types::*;

    match ty {
        UINT8 | INT8 => skip_n_seek(r, 1)?,
        UINT16 | INT16 => skip_n_seek(r, 2)?,
        UINT32 | INT32 | FLOAT32 => skip_n_seek(r, 4)?,
        BOOL => skip_n_seek(r, 1)?,
        STRING => {
            let n = read_u64(r)?;
            skip_n_seek(r, n)?;
        }
        ARRAY => {
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_gguf_value_seek(r, elem_ty)?;
            }
        }
        UINT64 | INT64 | FLOAT64 => skip_n_seek(r, 8)?,
        _ => bail!("unknown GGUF kv type id {ty}"),
    }
    Ok(())
}
```

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skip_all_gguf_types() {
        // Test all supported GGUF types
        let test_cases = vec![
            (0, 1),  // uint8
            (1, 1),  // int8
            (2, 2),  // uint16
            (3, 2),  // int16
            (4, 4),  // uint32
            (5, 4),  // int32
            (6, 4),  // float32
            (7, 1),  // bool
            (10, 8), // uint64
            (11, 8), // int64
            (12, 8), // float64
        ];

        for (type_id, expected_size) in test_cases {
            let mut data = create_test_gguf_value(type_id, expected_size);
            let result = skip_gguf_value_seek(&mut data, type_id);
            assert!(result.is_ok(), "Failed to skip type {}", type_id);
        }
    }
}
```

## Acceptance Criteria

- [ ] Complete support for all GGUF scalar types (0-12)
- [ ] Proper handling of uint8 and int8 metadata
- [ ] Clear type constants for maintainability
- [ ] Comprehensive test coverage
- [ ] No regression in existing GGUF parsing

## Priority: High

Critical for GGUF compatibility - missing type support can cause parsing failures.
