# [GGUF] Add Missing uint8/int8 Type Support in skip_gguf_value_seek

## Problem Description

The `skip_gguf_value_seek` function in `crates/bitnet-models/src/gguf_min.rs` has commented out support for uint8 and int8 types (type IDs 0 and 1), causing GGUF files containing these types to fail parsing.

## Environment

- **Component**: `crates/bitnet-models/src/gguf_min.rs`
- **Function**: `skip_gguf_value_seek`
- **Impact**: GGUF file parsing for models using uint8/int8 metadata

## Current Implementation

```rust
match ty {
    /* ~ changed by cargo-mutants ~ */ // uint8 | int8  <-- Missing case
    2 | 3 => skip_n_seek(r, 2)?, // uint16 | int16
    // ...
}
```

## Proposed Solution

Uncomment and implement the uint8/int8 case:

```rust
match ty {
    0 | 1 => skip_n_seek(r, 1)?, // uint8 | int8
    2 | 3 => skip_n_seek(r, 2)?, // uint16 | int16
    // ...
}
```

## Acceptance Criteria

- [ ] GGUF files with uint8/int8 metadata parse successfully
- [ ] Function handles all valid GGUF type IDs (0-12)
- [ ] Compatible with GGUF specification and llama.cpp reference

## Related Issues

- **Blocks**: Complete GGUF format support
- **References**: GGUF specification compliance