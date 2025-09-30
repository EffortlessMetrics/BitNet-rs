# Stub code: `sp_tokenizer.rs` is conditionally compiled

The entire `sp_tokenizer.rs` file is conditionally compiled with `#[cfg(feature = "spm")]`. If the `spm` feature is not enabled, this tokenizer is not available. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/sp_tokenizer.rs`

**Code:**
```rust
#[cfg(feature = "spm")]
use crate::Tokenizer;
// ... rest of the file ...
```

## Proposed Fix

The `sp_tokenizer.rs` file should not be conditionally compiled. Instead, SentencePiece support should be integrated directly into the tokenizer without relying on feature flags. This would involve:

1.  **Removing conditional compilation:** Remove the `#[cfg(feature = "spm")]` attributes.
2.  **Integrating SentencePiece:** Integrate the SentencePiece tokenizer directly into the `sp_tokenizer.rs` file.
3.  **Providing a clear error message:** If SentencePiece support is not available, provide a clear error message instead of returning an error.

### Example Implementation

```rust
use crate::Tokenizer;
use bitnet_common::{BitNetError, Result};
use sentencepiece::SentencePieceProcessor;

pub struct SpTokenizer {
    sp: SentencePieceProcessor,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

// ... rest of the file ...
```
