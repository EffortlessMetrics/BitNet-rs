# Stub code: `load_tokenizer_from_gguf` in `loader.rs` has conditional compilation for `spm`

The `load_tokenizer_from_gguf` function in `crates/bitnet-tokenizers/src/loader.rs` has conditional compilation for `spm` tokenizer. If the `spm` feature is not enabled, it returns an error. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/loader.rs`

**Function:** `load_tokenizer_from_gguf`

**Code:**
```rust
pub fn load_tokenizer_from_gguf(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<Box<dyn Tokenizer + Send + Sync>> {
    use base64::Engine;

    // Check if we have a SentencePiece model embedded
    if let Some(model_blob) = metadata.get("tokenizer.ggml.model") {
        // Decode to raw bytes (either base64 string or u8 array)
        let bytes: Option<Vec<u8>> = if let Some(blob_str) = model_blob.as_str() {
            Some(base64::engine::general_purpose::STANDARD.decode(blob_str)?)
        } else {
            model_blob.as_array().map(|blob_array| {
                blob_array.iter().filter_map(|v| v.as_u64().map(|b| b as u8)).collect()
            })
        };
        if let Some(bytes) = bytes {
            #[cfg(feature = "spm")]
            {
                let bos = metadata
                    .get("tokenizer.ggml.bos_token_id")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                let eos = metadata
                    .get("tokenizer.ggml.eos_token_id")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
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
    }

    // If no embedded tokenizer, fail
    anyhow::bail!("GGUF file does not contain an embedded tokenizer (tokenizer.ggml.model)")
}
```

## Proposed Fix

The `load_tokenizer_from_gguf` function should not have conditional compilation for `spm` tokenizer. Instead, SentencePiece support should be integrated directly into the `load_tokenizer_from_gguf` function without relying on feature flags. This would involve:

1.  **Removing conditional compilation:** Remove the `#[cfg(feature = "spm")]` attributes.
2.  **Integrating SentencePiece:** Integrate the SentencePiece tokenizer directly into the `load_tokenizer_from_gguf` function.
3.  **Providing a clear error message:** If SentencePiece support is not available, provide a clear error message instead of returning an error.

### Example Implementation

```rust
pub fn load_tokenizer_from_gguf(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<Box<dyn Tokenizer + Send + Sync>> {
    use base64::Engine;

    // Check if we have a SentencePiece model embedded
    if let Some(model_blob) = metadata.get("tokenizer.ggml.model") {
        // Decode to raw bytes (either base64 string or u8 array)
        let bytes: Option<Vec<u8>> = if let Some(blob_str) = model_blob.as_str() {
            Some(base64::engine::general_purpose::STANDARD.decode(blob_str)?)
        } else {
            model_blob.as_array().map(|blob_array| {
                blob_array.iter().filter_map(|v| v.as_u64().map(|b| b as u8)).collect()
            })
        };
        if let Some(bytes) = bytes {
            let bos = metadata
                .get("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let eos = metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            return Ok(crate::sp_tokenizer::SpTokenizer::from_gguf_blob(&bytes, bos, eos)?);
        }
    }

    // If no embedded tokenizer, fail
    anyhow::bail!("GGUF file does not contain an embedded tokenizer (tokenizer.ggml.model)")
}
```
