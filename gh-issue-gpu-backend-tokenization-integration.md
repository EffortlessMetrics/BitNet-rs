# [GPU] Integrate Proper Tokenizer in GpuBackend

## Problem Description

The `GpuBackend::tokenize` function in `crates/bitnet-inference/src/gpu.rs` uses a placeholder implementation that converts characters to u32 values instead of using proper tokenization. This prevents GPU inference from working with real tokenizers and language models.

## Environment

- **Component**: `crates/bitnet-inference/src/gpu.rs`
- **Function**: `GpuBackend::tokenize`
- **Impact**: GPU inference pipeline tokenization

## Current Implementation

```rust
fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
    // Placeholder implementation - in practice would use a proper tokenizer
    Ok(text.chars().map(|c| c as u32).collect())
}
```

## Proposed Solution

Integrate with the bitnet-tokenizers crate to provide proper tokenization:

```rust
fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
    let tokenizer = self.get_tokenizer()?;
    tokenizer.encode(text, true, true).map_err(Into::into)
}
```

## Implementation Tasks

- [ ] Add tokenizer field to GpuBackend struct
- [ ] Implement proper tokenizer integration
- [ ] Add GPU-optimized tokenization if available
- [ ] Update existing GPU inference pipeline

## Acceptance Criteria

- [ ] Uses proper tokenizer instead of character mapping
- [ ] Compatible with bitnet-tokenizers API
- [ ] Maintains GPU inference performance
- [ ] Handles tokenization errors gracefully

## Related Issues

- **Depends on**: Tokenizer integration framework
- **Blocks**: GPU inference functionality