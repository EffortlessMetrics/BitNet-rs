# Stub code: `GpuBackend::tokenize` in `gpu.rs` is a placeholder

The `GpuBackend::tokenize` function in `crates/bitnet-inference/src/gpu.rs` is a placeholder that just converts characters to u32. It doesn't use a proper tokenizer. This is a form of stubbing and should be replaced with a real implementation.

**File:** `crates/bitnet-inference/src/gpu.rs`

**Function:** `GpuBackend::tokenize`

**Code:**
```rust
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Placeholder implementation - in practice would use a proper tokenizer
        Ok(text.chars().map(|c| c as u32).collect())
    }
```

## Proposed Fix

The `GpuBackend::tokenize` function should be implemented to use a proper tokenizer. This would involve using the `bitnet_tokenizers` crate to encode the input text into a sequence of token IDs.

### Example Implementation

```rust
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let tokenizer = bitnet_tokenizers::get_tokenizer(); // Assuming a global tokenizer instance
        tokenizer.encode(text, true, true)
    }
```
