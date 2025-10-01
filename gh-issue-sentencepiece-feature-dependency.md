# [Tokenizer] Remove SentencePiece Feature Flag Dependency

## Problem Description

The `load_tokenizer_from_gguf_reader` function in `crates/bitnet-tokenizers/src/loader.rs` uses conditional compilation for SentencePiece support, requiring users to rebuild with specific features for GGUF models containing SentencePiece tokenizers.

## Environment

- **Component**: `crates/bitnet-tokenizers/src/loader.rs`
- **Function**: `load_tokenizer_from_gguf_reader`
- **Feature**: `spm` feature flag
- **Impact**: GGUF tokenizer loading capabilities

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

Remove conditional compilation and integrate SentencePiece support:

```rust
// Always attempt to load SentencePiece tokenizer
let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
return Ok(crate::sp_tokenizer::SpTokenizer::from_gguf_blob(&bytes, bos, eos)?);
```

## Implementation Tasks

- [ ] Remove conditional compilation for SentencePiece
- [ ] Make SentencePiece support always available
- [ ] Add runtime error handling for unsupported formats
- [ ] Update build configuration

## Acceptance Criteria

- [ ] GGUF models with SentencePiece tokenizers load without feature flags
- [ ] No build-time requirements for basic GGUF tokenizer support
- [ ] Clear error messages for genuinely unsupported tokenizer formats

## Related Issues

- **Related to**: Feature flag architecture improvements
- **Blocks**: Seamless GGUF model loading