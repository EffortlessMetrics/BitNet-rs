# Extensive mock objects in `engine.rs` test module

The `tests` module in `crates/bitnet-inference/src/engine.rs` defines `MockModel` and `MockTokenizer`. While these are correctly placed within the `#[cfg(test)]` block, they are quite extensive and could be simplified or moved to a dedicated test utilities module if they are reused across multiple test files.

**File:** `crates/bitnet-inference/src/engine.rs`

**Structs:**
* `MockModel`
* `MockTokenizer`

## Description

The `MockModel` and `MockTokenizer` structs are used to test the `InferenceEngine` without relying on real implementations of the `Model` and `Tokenizer` traits. While this is a valid testing strategy, the mock objects are quite extensive and could be simplified.

Additionally, if these mock objects are reused across multiple test files, they should be moved to a dedicated test utilities module to avoid code duplication.

## Proposed Fix

1.  **Simplify mock objects:** The mock objects should be simplified to only implement the methods that are actually used by the `InferenceEngine`. This will reduce the size of the mock objects and make them easier to maintain.

2.  **Move mock objects to a dedicated test utilities module:** If the mock objects are reused across multiple test files, they should be moved to a dedicated test utilities module (e.g., `crates/bitnet-inference/tests/utils.rs`). This will avoid code duplication and make it easier to manage the mock objects.

### Example Implementation

```rust
// In crates/bitnet-inference/tests/utils.rs

pub struct MockModel {
    config: BitNetConfig,
}

impl Model for MockModel {
    // ... simplified implementation ...
}

pub struct MockTokenizer {
    // ... simplified implementation ...
}
```
