# Stub code: `spm_tokenizer.rs` is conditionally compiled

The entire `spm_tokenizer.rs` file is conditionally compiled with `#[cfg(feature = "spm")]`. If the `spm` feature is not enabled, this tokenizer is not available. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/spm_tokenizer.rs`

**Code:**
```rust
#[cfg(feature = "spm")]
use anyhow::Result as AnyhowResult;
// ... rest of the file ...
```

## Proposed Fix

The `spm_tokenizer.rs` file should not be conditionally compiled. Instead, SentencePiece support should be integrated directly into the tokenizer without relying on feature flags. This would involve:

1.  **Removing conditional compilation:** Remove the `#[cfg(feature = "spm")]` attributes.
2.  **Integrating SentencePiece:** Integrate the SentencePiece tokenizer directly into the `spm_tokenizer.rs` file.
3.  **Providing a clear error message:** If SentencePiece support is not available, provide a clear error message instead of returning an error.

### Example Implementation

```rust
use anyhow::Result as AnyhowResult;
use bitnet_common::{BitNetError, ModelError, Result};
use std::path::Path;

pub struct SpmTokenizer {
    inner: sentencepiece::SentencePieceProcessor,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    id2piece: Box<[String]>,
}

// ... rest of the file ...
```
