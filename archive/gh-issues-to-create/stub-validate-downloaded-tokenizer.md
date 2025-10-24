# Stub code: `validate_downloaded_tokenizer` in `download.rs` has a placeholder for `warn!`

The `validate_downloaded_tokenizer` function in `crates/bitnet-tokenizers/src/download.rs` uses `warn!` when a downloaded tokenizer is missing a required field or has a vocabulary size mismatch. It doesn't return an error. This might hide real issues with downloaded tokenizers. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/download.rs`

**Function:** `validate_downloaded_tokenizer`

**Code:**
```rust
    pub fn validate_downloaded_tokenizer(
        &self,
        path: &Path,
        info: &TokenizerDownloadInfo,
    ) -> Result<()> {
        // ...

            // Basic validation for HuggingFace tokenizer format
            if let Some(obj) = json_value.as_object() {
                // Check for required fields
                let required_fields = ["model", "normalizer", "pre_tokenizer"];
                for field in &required_fields {
                    if !obj.contains_key(*field) {
                        warn!("Downloaded tokenizer missing field: {}", field);
                    }
                }

                // If vocab size is specified, validate it
                if let Some(expected_vocab) = info.expected_vocab
                    && let Some(model) = obj.get("model").and_then(|m| m.as_object())
                    && let Some(vocab) = model.get("vocab").and_then(|v| v.as_object())
                {
                    let actual_vocab = vocab.len();
                    if actual_vocab != expected_vocab {
                        warn!(
                            "Vocabulary size mismatch: expected {}, got {}",
                            expected_vocab, actual_vocab
                        );
                    } else {
                        debug!("Vocabulary size validated: {}", actual_vocab);
                    }
                }
            }
        }

        info!(
            "Downloaded tokenizer validation successful: {} ({} bytes)",
            path.display(),
            file_size
        );
        Ok(())
    }
```

## Proposed Fix

The `validate_downloaded_tokenizer` function should return an error if a downloaded tokenizer is missing a required field or has a vocabulary size mismatch. This will ensure that the function behaves correctly and doesn't hide real issues with downloaded tokenizers.

### Example Implementation

```rust
    pub fn validate_downloaded_tokenizer(
        &self,
        path: &Path,
        info: &TokenizerDownloadInfo,
    ) -> Result<()> {
        // ...

            // Basic validation for HuggingFace tokenizer format
            if let Some(obj) = json_value.as_object() {
                // Check for required fields
                let required_fields = ["model", "normalizer", "pre_tokenizer"];
                for field in &required_fields {
                    if !obj.contains_key(*field) {
                        return Err(BitNetError::Model(ModelError::LoadingFailed {
                            reason: format!("Downloaded tokenizer missing required field: {}", field),
                        }));
                    }
                }

                // If vocab size is specified, validate it
                if let Some(expected_vocab) = info.expected_vocab
                    && let Some(model) = obj.get("model").and_then(|m| m.as_object())
                    && let Some(vocab) = model.get("vocab").and_then(|v| v.as_object())
                {
                    let actual_vocab = vocab.len();
                    if actual_vocab != expected_vocab {
                        return Err(BitNetError::Model(ModelError::LoadingFailed {
                            reason: format!(
                                "Vocabulary size mismatch: expected {}, got {}",
                                expected_vocab, actual_vocab
                            ),
                        }));
                    } else {
                        debug!("Vocabulary size validated: {}", actual_vocab);
                    }
                }
            }
        }

        info!(
            "Downloaded tokenizer validation successful: {} ({} bytes)",
            path.display(),
            file_size
        );
        Ok(())
    }
```
