# Stub code: `load_tokenizer_from_gguf_reader` in `loader.rs` has conditional compilation for `spm`

The `load_tokenizer_from_gguf_reader` function in `crates/bitnet-tokenizers/src/loader.rs` has conditional compilation for `spm` tokenizer. If the `spm` feature is not enabled, it returns an error. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/loader.rs`

**Function:** `load_tokenizer_from_gguf_reader`

**Code:**
```rust
pub fn load_tokenizer_from_gguf_reader(
    reader: &bitnet_models::GgufReader,
) -> Result<Box<dyn Tokenizer + Send + Sync>> {
    // Check if the GGUF contains an embedded tokenizer (try both binary and array formats)
    if let Some(bytes) = reader.get_bin_or_u8_array("tokenizer.ggml.model") {
        #[cfg(feature = "spm")]
        {
            let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
            let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
            return Ok(crate::sp_tokenizer::SpTokenizer::from_gguf_blob(&bytes, bos, eos)?);
        }
        #[cfg(not(feature = "spm"))]
        {
            let _ = bytes;
            return Err(anyhow::anyhow!(
                "SentencePiece support not compiled in. Enable the 'spm' feature."
            ));
        }
    }

    Err(anyhow::anyhow!("GGUF missing tokenizer.ggml.model"))
}
```

## Proposed Fix

The `load_tokenizer_from_gguf_reader` function should not have conditional compilation for `spm` tokenizer. Instead, SentencePiece support should be integrated directly into the `load_tokenizer_from_gguf_reader` function without relying on feature flags. This would involve:

1.  **Removing conditional compilation:** Remove the `#[cfg(feature = "spm")]` attributes.
2.  **Integrating SentencePiece:** Integrate the SentencePiece tokenizer directly into the `load_tokenizer_from_gguf_reader` function.
3.  **Providing a clear error message:** If SentencePiece support is not available, provide a clear error message instead of returning an error.

### Example Implementation

```rust
pub fn load_tokenizer_from_gguf_reader(
    reader: &bitnet_models::GgufReader,
) -> Result<Box<dyn Tokenizer + Send + Sync>> {
    // Check if the GGUF contains an embedded tokenizer (try both binary and array formats)
    if let Some(bytes) = reader.get_bin_or_u8_array("tokenizer.ggml.model") {
        let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        return Ok(crate::sp_tokenizer::SpTokenizer::from_gguf_blob(&bytes, bos, eos)?);
    }

    Err(anyhow::anyhow!("GGUF missing tokenizer.ggml.model"))
}
```
