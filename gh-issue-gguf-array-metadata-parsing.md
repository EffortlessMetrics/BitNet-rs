# [GGUF] Implement Array Metadata Value Parsing

## Problem Description

The `read_metadata_value` function in `crates/bitnet-models/src/gguf_parity.rs` returns a placeholder "[array]" string for GGUF array types instead of parsing the actual array contents, limiting metadata inspection capabilities.

## Environment

- **Component**: `crates/bitnet-models/src/gguf_parity.rs`
- **Function**: `read_metadata_value`
- **Impact**: GGUF metadata analysis and debugging

## Current Implementation

```rust
9 => {
    // ARRAY
    // For now, just return a placeholder
    skip_array(reader)?;
    Ok("[array]".to_string())
}
```

## Proposed Solution

Parse array contents properly:

```rust
9 => {
    // ARRAY
    let elem_type = read_u32(reader)?;
    let count = read_u64(reader)?;
    let mut values = Vec::new();
    for _ in 0..count {
        values.push(read_gguf_value(reader, elem_type)?);
    }
    Ok(format!("{:?}", values))
}
```

## Acceptance Criteria

- [ ] Properly parses GGUF array metadata
- [ ] Returns meaningful array content representation
- [ ] Handles nested arrays correctly
- [ ] Compatible with GGUF specification

## Related Issues

- **Related to**: GGUF format compliance improvements
- **Blocks**: Complete metadata inspection functionality