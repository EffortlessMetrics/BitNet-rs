# Stub code: `TokenizerBackend::SentencePiece` in `universal.rs` is conditionally compiled

The `TokenizerBackend::SentencePiece` variant and its usage are conditionally compiled with `#[cfg(feature = "spm")]`. If the `spm` feature is not enabled, it falls back to `MockTokenizer`. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/universal.rs`

**Enum Variant:** `TokenizerBackend::SentencePiece`

**Code:**
```rust
#[allow(clippy::large_enum_variant)]
enum TokenizerBackend {
    #[cfg(feature = "spm")]
    SentencePiece(crate::SpmTokenizer),
    Mock(MockTokenizer),
}

// ...

            #[cfg(feature = "spm")]
            "smp" | "sentencepiece" => {
                // ...
            }
            #[cfg(not(feature = "spm"))]
            "smp" | "sentencepiece" => {
                // ...
                warn!("SentencePiece support not compiled, using mock");
                Ok(TokenizerBackend::Mock(MockTokenizer::new()))
            }
```

## Proposed Fix

The `TokenizerBackend::SentencePiece` should not be conditionally compiled. Instead, SentencePiece support should be integrated directly into the `UniversalTokenizer` without relying on feature flags. This would involve:

1.  **Removing conditional compilation:** Remove the `#[cfg(feature = "spm")]` attributes.
2.  **Integrating SentencePiece:** Integrate the SentencePiece tokenizer directly into the `UniversalTokenizer`.
3.  **Providing a clear error message:** If SentencePiece support is not available, provide a clear error message instead of falling back to `MockTokenizer`.

### Example Implementation

```rust
enum TokenizerBackend {
    SentencePiece(crate::SpmTokenizer),
    Mock(MockTokenizer),
}

// ...

            "smp" | "sentencepiece" => {
                if let Some(path) = &config.pre_tokenizer {
                    let spm = SpmTokenizer::from_file(Path::new(path)).map_err(|e| {
                        BitNetError::Model(ModelError::LoadingFailed {
                            reason: format!("Failed to load SentencePiece tokenizer: {}", e),
                        })
                    })?;
                    Ok(TokenizerBackend::SentencePiece(spm))
                } else {
                    Err(BitNetError::Inference(InferenceError::TokenizationFailed {
                        reason: "SentencePiece tokenizer requires model path".to_string(),
                    }))
                }
            }
```
