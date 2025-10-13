# Stub code: `load_minimal` function in `minimal.rs` uses dummy weights and has an incomplete GGUF loader

The `load_minimal` function in `crates/bitnet-models/src/minimal.rs` has a `LoadMode::Dummy` that generates deterministic dummy weights for testing. This is a form of stubbing and should be replaced with a more robust solution.

Additionally, the `LoadMode::Gguf` is not fully implemented, as the comment says "Load from GGUF file (not yet implemented)".

**File:** `crates/bitnet-models/src/minimal.rs`

**Function:** `load_minimal`

## Description

The `load_minimal` function is intended to be a minimal model loader for testing and CI. However, it has two issues:

1.  **`LoadMode::Dummy` is a stub:** The `LoadMode::Dummy` variant generates deterministic dummy weights for testing. This is a form of stubbing and should be used only in tests.

2.  **`LoadMode::Gguf` is incomplete:** The `LoadMode::Gguf` variant is not fully implemented. It calls `crate::gguf_min::load_two`, but it does not handle all the necessary tensors for a complete model.

## Proposed Fix

1.  **Move `LoadMode::Dummy` to a test helper:** The `LoadMode::Dummy` variant should be moved to a `#[cfg(test)]` module and the `load_minimal` function should be split into two functions: `load_minimal_from_gguf` and `load_minimal_dummy_for_testing`.

2.  **Complete the `LoadMode::Gguf` implementation:** The `load_minimal_from_gguf` function should be updated to load all the necessary tensors for a complete model. This will involve using the `gguf_simple` module to load the GGUF file and then extracting the required tensors.

### Example Implementation

```rust
// In crates/bitnet-models/src/minimal.rs

pub fn load_minimal_from_gguf(path: &Path) -> Result<MinimalWeights> {
    let (config, tensor_map) = crate::gguf_simple::load_gguf(path, Device::Cpu)?;

    let tok_embeddings = tensor_map.get("token_embd.weight").unwrap().to_vec1::<f32>()?;
    let lm_head = tensor_map.get("output.weight").unwrap().to_vec1::<f32>()?;

    Ok(MinimalWeights {
        tok_embeddings,
        lm_head,
        vocab: config.model.vocab_size,
        dim: config.model.hidden_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn load_minimal_dummy_for_testing(vocab: usize, dim: usize) -> Result<MinimalWeights> {
        // ... dummy weight generation ...
    }
}
```
