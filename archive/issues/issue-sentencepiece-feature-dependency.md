# [Tokenizers] Remove Conditional SentencePiece Feature Dependency

## Problem Description

The `load_tokenizer_from_gguf_reader` function in `/home/steven/code/Rust/BitNet-rs/crates/bitnet-tokenizers/src/loader.rs` uses conditional compilation for SentencePiece support, creating inconsistent behavior where GGUF tokenizer loading fails at runtime rather than being handled gracefully.

## Current Implementation
```rust
#[cfg(feature = "spm")]
{
    return Ok(crate::sp_tokenizer::SpTokenizer::from_gguf_blob(&bytes, bos, eos)?);
}
#[cfg(not(feature = "spm"))]
{
    return Err(anyhow::anyhow!(
        "SentencePiece support not compiled in. Enable the 'spm' feature."
    ));
}
```

## Proposed Solution
Implement dynamic feature detection with graceful fallbacks:

```rust
// Try SentencePiece first if available
if let Ok(tokenizer) = SpTokenizer::try_from_gguf_blob(&bytes, bos, eos) {
    return Ok(tokenizer);
}

// Fallback to other tokenizer types
if let Ok(tokenizer) = BpeTokenizer::try_from_gguf_blob(&bytes) {
    return Ok(tokenizer);
}

Err(anyhow::anyhow!(
    "No compatible tokenizer found for GGUF format. Consider enabling additional tokenizer features."
))
```

## Acceptance Criteria
- [ ] Eliminate conditional compilation for core tokenizer functionality
- [ ] Graceful fallback between different tokenizer types
- [ ] Clear error messages with actionable suggestions
- [ ] Consistent API behavior regardless of feature flags
- [ ] Support for multiple tokenizer formats in single build
