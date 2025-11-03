# Hardcoded value: `UniversalTokenizer::from_gguf` in `universal.rs` has hardcoded `gpt2` fallback

The `UniversalTokenizer::from_gguf` function in `crates/bitnet-tokenizers/src/universal.rs` uses `reader.get_string_metadata("tokenizer.ggml.model").unwrap_or_else(|| "gpt2".into())` which hardcodes "gpt2" as a fallback. This might not be appropriate for all GGUF models. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/universal.rs`

**Function:** `UniversalTokenizer::from_gguf`

**Code:**
```rust
    pub fn from_gguf(path: &Path) -> Result<Self> {
        use bitnet_models::{GgufReader, loader::MmapFile};

        let mmap = MmapFile::open(path)?;
        let reader = GgufReader::new(mmap.as_slice())?;

        let model_type =
            reader.get_string_metadata("tokenizer.ggml.model").unwrap_or_else(|| "gpt2".into());

        // ...
    }
```

## Proposed Fix

The `UniversalTokenizer::from_gguf` function should not hardcode "gpt2" as a fallback. Instead, it should return an error if the `tokenizer.ggml.model` metadata is not found or is not a recognized model type. This will ensure that the function behaves correctly for all GGUF models.

### Example Implementation

```rust
    pub fn from_gguf(path: &Path) -> Result<Self> {
        use bitnet_models::{GgufReader, loader::MmapFile};

        let mmap = MmapFile::open(path)?;
        let reader = GgufReader::new(mmap.as_slice())?;

        let model_type = reader.get_string_metadata("tokenizer.ggml.model").ok_or_else(|| {
            BitNetError::Inference(InferenceError::TokenizationFailed {
                reason: "Missing tokenizer.ggml.model metadata in GGUF file".to_string(),
            })
        })?;

        // ...
    }
```
