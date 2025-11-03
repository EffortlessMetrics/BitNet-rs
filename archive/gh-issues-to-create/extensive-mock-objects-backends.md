# Extensive mock objects in `backends.rs` test module

The `tests` module in `crates/bitnet-inference/src/backends.rs` defines `MockModel`. While this is correctly placed within the `#[cfg(test)]` block, it is quite extensive and could be simplified or moved to a dedicated test utilities module if it is reused across multiple test files.

**File:** `crates/bitnet-inference/src/backends.rs`

**Structs:**
* `MockModel`

## Description

The `MockModel` struct is used to test the `CpuBackend` and `GpuBackend` without relying on real implementations of the `Model` trait. While this is a valid testing strategy, the mock object is quite extensive and could be simplified.

Additionally, if this mock object is reused across multiple test files, it should be moved to a dedicated test utilities module to avoid code duplication.

## Proposed Fix

1.  **Simplify mock object:** The mock object should be simplified to only implement the methods that are actually used by the `CpuBackend` and `GpuBackend`. This will reduce the size of the mock object and make it easier to maintain.

2.  **Move mock object to a dedicated test utilities module:** If the mock object is reused across multiple test files, it should be moved to a dedicated test utilities module (e.g., `crates/bitnet-inference/tests/utils.rs`). This will avoid code duplication and make it easier to manage the mock object.

### Example Implementation

```rust
// In crates/bitnet-inference/tests/utils.rs

pub struct MockModel {
    config: BitNetConfig,
}

impl Model for MockModel {
    // ... simplified implementation ...
}
```
