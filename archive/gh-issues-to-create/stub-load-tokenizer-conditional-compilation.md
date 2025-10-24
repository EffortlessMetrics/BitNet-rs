# Stub code: `load_tokenizer` in `loader.rs` has conditional compilation for `spm`

The `load_tokenizer` function in `crates/bitnet-tokenizers/src/loader.rs` has conditional compilation for `spm` tokenizer. If the `spm` feature is not enabled, it returns an error. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/loader.rs`

**Function:** `load_tokenizer`

**Code:**
```rust
pub fn load_tokenizer(path: &Path) -> Result<Box<dyn Tokenizer + Send + Sync>> {
    // ...
    match ext {
        // ...
        "model" => {
            #[cfg(feature = "spm")]
            {
                match crate::sp_tokenizer::SpTokenizer::from_file(path) {
                    Ok(t) => Ok(t),
                    Err(e) => anyhow::bail!("Failed to load SentencePiece model: {}", e),
                }
            }
            #[cfg(not(feature = "spm"))]
            {
                anyhow::bail!("SentencePiece support not compiled in. Enable the 'spm' feature.")
            }
        }
        // ...
    }
}
```

## Proposed Fix

The `load_tokenizer` function should not have conditional compilation for `spm` tokenizer. Instead, SentencePiece support should be integrated directly into the `load_tokenizer` function without relying on feature flags. This would involve:

1.  **Removing conditional compilation:** Remove the `#[cfg(feature = "spm")]` attributes.
2.  **Integrating SentencePiece:** Integrate the SentencePiece tokenizer directly into the `load_tokenizer` function.
3.  **Providing a clear error message:** If SentencePiece support is not available, provide a clear error message instead of returning an error.

### Example Implementation

```rust
pub fn load_tokenizer(path: &Path) -> Result<Box<dyn Tokenizer + Send + Sync>> {
    // ...
    match ext {
        // ...
        "model" => {
            match crate::sp_tokenizer::SpTokenizer::from_file(path) {
                Ok(t) => Ok(t),
                Err(e) => anyhow::bail!("Failed to load SentencePiece model: {}", e),
            }
        }
        // ...
    }
}
```
