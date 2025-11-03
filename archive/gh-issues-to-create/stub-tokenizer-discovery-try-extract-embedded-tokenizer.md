# Stub code: `TokenizerDiscovery::try_extract_embedded_tokenizer` in `discovery.rs` has a simplified implementation

The `TokenizerDiscovery::try_extract_embedded_tokenizer` function in `crates/bitnet-tokenizers/src/discovery.rs` has comments "This is a simplified implementation - in production this would parse the model format" and "In production, this would parse the JSON and create an HfTokenizer". It creates a `BasicTokenizer` instead of a proper tokenizer. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/discovery.rs`

**Function:** `TokenizerDiscovery::try_extract_embedded_tokenizer`

**Code:**
```rust
    pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer>>> {
        debug!("Attempting to extract embedded tokenizer from GGUF metadata");

        // Check if tokenizer model is embedded as bytes
        if let Some(tokenizer_model) = self.gguf_reader.get_array_metadata("tokenizer.ggml.model") {
            debug!("Found embedded tokenizer.ggml.model ({} bytes)", tokenizer_model.len());

            // Try to create tokenizer from the embedded data
            // This is a simplified implementation - in production this would parse the model format
            if tokenizer_model.len() > 100 {
                // Sanity check for reasonable size
                let basic_tokenizer = crate::BasicTokenizer::with_config(
                    self.vocab_size,
                    Some(1), // BOS token
                    Some(2), // EOS token
                    Some(0), // PAD token
                );

                debug!("Created basic tokenizer from GGUF metadata");
                return Ok(Some(Arc::new(basic_tokenizer)));
            }
        }

        // Check for tokenizer vocab embedded in metadata
        if let Some(vocab) = self.gguf_reader.get_string_array_metadata("tokenizer.ggml.tokens")
            && vocab.len() == self.vocab_size
        {
            debug!("Found embedded vocabulary with {} tokens", vocab.len());

            // Create tokenizer with embedded vocabulary
            let basic_tokenizer = crate::BasicTokenizer::with_config(
                self.vocab_size,
                self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id"),
                self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id"),
                self.gguf_reader.get_u32_metadata("tokenizer.ggml.pad_token_id"),
            );

            debug!("Created tokenizer with embedded vocabulary");
            return Ok(Some(Arc::new(basic_tokenizer)));
        }

        // Check for HuggingFace tokenizer.json embedded as string
        if let Some(tokenizer_json) = self.gguf_reader.get_string_metadata("tokenizer.json") {
            debug!("Found embedded tokenizer.json ({} chars)", tokenizer_json.len());

            // In production, this would parse the JSON and create an HfTokenizer
            // For now, create a basic tokenizer with inferred parameters
            let basic_tokenizer = crate::BasicTokenizer::with_config(
                self.vocab_size,
                Some(1), // BOS token
                Some(2), // EOS token
                Some(0), // PAD token
            );

            debug!("Created tokenizer from embedded JSON metadata");
            return Ok(Some(Arc::new(basic_tokenizer)));
        }

        debug!("No embedded tokenizer found in GGUF metadata");
        Ok(None)
    }
```

## Proposed Fix

The `TokenizerDiscovery::try_extract_embedded_tokenizer` function should be implemented to parse the embedded tokenizer data and create a proper tokenizer instance (e.g., `HfTokenizer` or `SpmTokenizer`). This would involve:

1.  **Parsing the embedded data:** Parse the embedded tokenizer data (e.g., JSON for HuggingFace tokenizers, or SentencePiece model bytes).
2.  **Creating a proper tokenizer:** Create a proper tokenizer instance based on the parsed data.

### Example Implementation

```rust
    pub fn try_extract_embedded_tokenizer(&self) -> Result<Option<Arc<dyn Tokenizer>>> {
        debug!("Attempting to extract embedded tokenizer from GGUF metadata");

        // Check for HuggingFace tokenizer.json embedded as string
        if let Some(tokenizer_json) = self.gguf_reader.get_string_metadata("tokenizer.json") {
            debug!("Found embedded tokenizer.json ({} chars)", tokenizer_json.len());

            let hf_tokenizer = crate::hf_tokenizer::HfTokenizer::from_json_string(&tokenizer_json)?;
            return Ok(Some(Arc::new(hf_tokenizer)));
        }

        // Check if tokenizer model is embedded as bytes (for SentencePiece)
        if let Some(tokenizer_model_bytes) = self.gguf_reader.get_array_metadata("tokenizer.ggml.model") {
            debug!("Found embedded tokenizer.ggml.model ({} bytes)", tokenizer_model_bytes.len());

            let bos = self.gguf_reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
            let eos = self.gguf_reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
            let spm_tokenizer = crate::sp_tokenizer::SpTokenizer::from_gguf_blob(&tokenizer_model_bytes, bos, eos)?;
            return Ok(Some(Arc::new(spm_tokenizer)));
        }

        debug!("No embedded tokenizer found in GGUF metadata");
        Ok(None)
    }
```
