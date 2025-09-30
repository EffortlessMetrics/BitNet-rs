# Stub code: `CpuBackend::detokenize` in `cpu.rs` is a placeholder

The `CpuBackend::detokenize` function in `crates/bitnet-inference/src/cpu.rs` is a placeholder that just converts u32 to characters. It doesn't use a proper tokenizer. This is a form of stubbing and should be replaced with a real implementation.

**File:** `crates/bitnet-inference/src/cpu.rs`

**Function:** `CpuBackend::detokenize`

**Code:**
```rust
    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        // Placeholder implementation - in practice would use a proper tokenizer
        Ok(tokens.iter().map(|&t| char::from(t as u8)).collect())
    }
```

## Proposed Fix

The `CpuBackend::detokenize` function should be implemented to use a proper tokenizer. This would involve using the `bitnet_tokenizers` crate to decode the sequence of token IDs into a string.

### Example Implementation

```rust
    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        let tokenizer = bitnet_tokenizers::get_tokenizer(); // Assuming a global tokenizer instance
        tokenizer.decode(tokens)
    }
```
