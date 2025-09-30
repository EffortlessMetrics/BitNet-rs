# Stub code: `load_gguf_minimal` and `create_mock_tensor_layout` create mock tensors

The `load_gguf_minimal` and `create_mock_tensor_layout` functions in `crates/bitnet-models/src/gguf_simple.rs` create mock tensors. This is a form of stubbing and should be replaced with a more robust solution.

**File:** `crates/bitnet-models/src/gguf_simple.rs`

**Functions:**
* `load_gguf_minimal`
* `create_mock_tensor_layout`

## Description

The `load_gguf_minimal` function is intended to be a fallback for when the enhanced GGUF parser fails. However, instead of returning an error when it encounters an incomplete GGUF file, it creates mock tensors for the missing transformer layers. This can lead to unexpected behavior and make it difficult to debug issues with GGUF files.

The `create_mock_tensor_layout` function is used to create a mock tensor layout for test compatibility. However, it is not clearly marked as a test helper and could be mistakenly used in production code.

## Proposed Fix

1.  **`load_gguf_minimal` should return an error for incomplete GGUF files:** Instead of creating mock tensors, the `load_gguf_minimal` function should return a `BitNetError::Validation` error with a descriptive message when it encounters an incomplete GGUF file. This will make it easier to identify and debug issues with GGUF files.

2.  **`create_mock_tensor_layout` should be marked as a test helper:** The `create_mock_tensor_layout` function should be moved to a `#[cfg(test)]` module and its name should be changed to `create_mock_tensor_layout_for_testing` to make it clear that it is a test helper.

### Example Implementation

```rust
// In crates/bitnet-models/src/gguf_simple.rs

fn load_gguf_minimal(
    path: &Path,
    device: Device,
) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
    let two = match crate::gguf_min::load_two(path) {
        Ok(two_tensors) => two_tensors,
        Err(_) => {
            return Err(BitNetError::Validation(
                "Failed to parse GGUF file with minimal parser".to_string(),
            ));
        }
    };

    // ... existing code ...

    // Check for missing tensors and return an error
    for layer in 0..num_layers {
        let prefix = format!("blk.{}", layer);
        if !tensor_map.contains_key(&format!("{}.attn_q.weight", prefix)) {
            return Err(BitNetError::Validation(
                "Incomplete GGUF file: missing attention weights".to_string(),
            ));
        }
        // ... and so on for other tensors
    }

    Ok((config, tensor_map))
}

#[cfg(test)]
mod tests {
    fn create_mock_tensor_layout_for_testing(
        device: Device,
    ) -> Result<(bitnet_common::BitNetConfig, HashMap<String, CandleTensor>)> {
        // ... existing code ...
    }
}
```