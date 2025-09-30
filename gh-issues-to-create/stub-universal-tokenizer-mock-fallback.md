# Stub code: Extensive use of `MockTokenizer` as a fallback in `universal.rs`

The `detect_and_create_backend` function in `crates/bitnet-tokenizers/src/universal.rs` extensively uses `MockTokenizer` as a fallback for various tokenizer types (gpt2, bpe, llama, tiktoken, etc.) and when SentencePiece support is not compiled or a path is missing. This is a form of stubbing and can hide real issues with tokenizer loading.

**File:** `crates/bitnet-tokenizers/src/universal.rs`

**Function:** `UniversalTokenizer::detect_and_create_backend`

**Code:**
```rust
    fn detect_and_create_backend(config: &TokenizerConfig) -> Result<TokenizerBackend> {
        match config.model_type.as_str() {
            "gpt2" | "bpe" | "llama" | "llama3" | "tiktoken" | "gpt4" | "cl100k" | "falcon" => {
                // ...
                debug!("Creating mock tokenizer for {}", config.model_type);
                Ok(TokenizerBackend::Mock(MockTokenizer::new()))
            }
            // ...
            unknown => {
                // ...
                warn!("Unknown tokenizer type: {}, using mock fallback", unknown);
                Ok(TokenizerBackend::Mock(MockTokenizer::new()))
            }
        }
    }
```

## Proposed Fix

The extensive use of `MockTokenizer` as a fallback should be replaced with proper error handling or actual implementations for the supported tokenizer types. This would involve:

1.  **Implementing actual tokenizers:** Implement actual tokenizers for the supported types (gpt2, bpe, llama, tiktoken, etc.) instead of falling back to `MockTokenizer`.
2.  **Returning errors for unsupported types:** For unknown tokenizer types, return a `BitNetError::Config` error instead of falling back to `MockTokenizer`.
3.  **Providing clear error messages:** Provide clear and actionable error messages when a tokenizer cannot be loaded.

### Example Implementation

```rust
    fn detect_and_create_backend(config: &TokenizerConfig) -> Result<TokenizerBackend> {
        match config.model_type.as_str() {
            "gpt2" => {
                // Implement GPT2 tokenizer
                Ok(TokenizerBackend::Gpt2(Gpt2Tokenizer::new()))
            }
            "llama" => {
                // Implement Llama tokenizer
                Ok(TokenizerBackend::Llama(LlamaTokenizer::new()))
            }
            // ...
            unknown => {
                Err(BitNetError::Inference(InferenceError::TokenizationFailed {
                    reason: format!("Unsupported tokenizer type: {}", unknown),
                }))
            }
        }
    }
```
