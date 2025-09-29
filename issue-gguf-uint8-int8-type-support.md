# [Models] Fix Missing GGUF uint8/int8 Type Handling

## Problem Description

The `skip_gguf_value_seek` function in `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/src/gguf_min.rs` has commented out support for GGUF types 0 and 1 (uint8/int8), potentially causing parsing failures for GGUF files containing these data types.

## Current Implementation
```rust
match ty {
    /* ~ changed by cargo-mutants ~ */ // uint8 | int8
    2 | 3 => skip_n_seek(r, 2)?, // uint16 | int16
    // ... rest of match arms
}
```

## Proposed Solution
Restore support for uint8 and int8 types:

```rust
match ty {
    0 | 1 => skip_n_seek(r, 1)?, // uint8 | int8
    2 | 3 => skip_n_seek(r, 2)?, // uint16 | int16
    4..=6 => skip_n_seek(r, 4)?, // uint32 | int32 | float32
    7 => skip_n_seek(r, 1)?,     // bool
    // ... rest of implementation
}
```

## Acceptance Criteria
- [ ] Complete GGUF type coverage including uint8/int8
- [ ] Validation against GGUF specification
- [ ] Test cases for all supported data types
- [ ] Backward compatibility with existing GGUF files
- [ ] Performance validation for large files