# Stub code: `from_path` in `lib.rs` has conditional compilation for `Spm`

The `from_path` function in `crates/bitnet-tokenizers/src/lib.rs` has conditional compilation for `Spm` tokenizer. If the `spm` feature is not enabled, it returns an error. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/lib.rs`

**Function:** `from_path`

**Code:**
```rust
pub fn from_path(path: &Path) -> Result<(Arc<dyn Tokenizer>, TokenizerFileKind)> {
    use bitnet_common::{BitNetError, ModelError};

    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();

    match ext.as_str() {
        "json" => {
            let t = HfTokenizer::from_file(path).map_err(|e| {
                BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Failed to load HF tokenizer: {}", e),
                })
            })?;
            Ok((Arc::new(t), TokenizerFileKind::HfJson))
        }
        "model" => {
            #[cfg(feature = "spm")]
            {
                let t = SpmTokenizer::from_file(path).map_err(|e| {
                    BitNetError::Model(ModelError::LoadingFailed {
                        reason: format!("Failed to load SPM tokenizer: {}", e),
                    })
                })?;
                Ok((Arc::new(t), TokenizerFileKind::Spm))
            }
            #[cfg(not(feature = "spm"))]
            {
                Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: "Build with `--features spm` to load SentencePiece .model files"
                        .to_string(),
                }))
            }
        }
        _ => Err(BitNetError::Model(ModelError::LoadingFailed {
            reason: format!(
                "Unsupported tokenizer file (expected *.json or *.model): {}",
                path.display()
            ),
        })),
    }
}
```

## Proposed Fix

The `from_path` function should not have conditional compilation for `Spm` tokenizer. Instead, SentencePiece support should be integrated directly into the `from_path` function without relying on feature flags. This would involve:

1.  **Removing conditional compilation:** Remove the `#[cfg(feature = "spm")]` attributes.
2.  **Integrating SentencePiece:** Integrate the SentencePiece tokenizer directly into the `from_path` function.
3.  **Providing a clear error message:** If SentencePiece support is not available, provide a clear error message instead of returning an error.

### Example Implementation

```rust
pub fn from_path(path: &Path) -> Result<(Arc<dyn Tokenizer>, TokenizerFileKind)> {
    use bitnet_common::{BitNetError, ModelError};

    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();

    match ext.as_str() {
        "json" => {
            let t = HfTokenizer::from_file(path).map_err(|e| {
                BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Failed to load HF tokenizer: {}", e),
                })
            })?;
            Ok((Arc::new(t), TokenizerFileKind::HfJson))
        }
        "model" => {
            let t = SpmTokenizer::from_file(path).map_err(|e| {
                BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Failed to load SPM tokenizer: {}", e),
                })
            })?;
            Ok((Arc::new(t), TokenizerFileKind::Spm))
        }
        _ => Err(BitNetError::Model(ModelError::LoadingFailed {
            reason: format!(
                "Unsupported tokenizer file (expected *.json or *.model): {}",
                path.display()
            ),
        })),
    }
}
```
