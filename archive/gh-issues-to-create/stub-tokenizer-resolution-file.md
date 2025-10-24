# Stub code: `TokenizerResolution::File` in `fallback.rs` is a placeholder

The `TokenizerResolution::File` variant's `into_tokenizer` method in `crates/bitnet-tokenizers/src/fallback.rs` creates a basic tokenizer. It doesn't actually load a tokenizer from the file path. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/fallback.rs`

**Enum Variant:** `TokenizerResolution::File`

**Code:**
```rust
    pub fn into_tokenizer(self) -> Result<Arc<dyn Tokenizer>> {
        match self {
            TokenizerResolution::File(_path) => {
                // Try to load tokenizer from file
                // For now, create a basic tokenizer - in production this would parse the JSON
                let basic_tokenizer = crate::BasicTokenizer::new();
                Ok(Arc::new(basic_tokenizer))
            }
            // ...
        }
    }
```

## Proposed Fix

The `TokenizerResolution::File` variant's `into_tokenizer` method should be implemented to actually load a tokenizer from the file path. This would involve using the `crate::loader::load_tokenizer` function to load the tokenizer from the specified path.

### Example Implementation

```rust
    pub fn into_tokenizer(self) -> Result<Arc<dyn Tokenizer>> {
        match self {
            TokenizerResolution::File(path) => {
                // Load tokenizer from file
                let tokenizer = crate::loader::load_tokenizer(&path)?;
                Ok(Arc::new(tokenizer))
            }
            // ...
        }
    }
```
