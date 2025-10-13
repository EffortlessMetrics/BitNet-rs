# [Tokenizers] Eliminate conditional compilation stubs in download functionality

## Problem Description

The `download_file` function in `crates/bitnet-tokenizers/src/download.rs` uses conditional compilation (`#[cfg(feature = "downloads")]`) with a stub implementation that always returns an error when the feature is disabled. This creates inconsistent behavior and poor user experience.

## Environment

- **File:** `crates/bitnet-tokenizers/src/download.rs`
- **Function:** `download_file`
- **Issue:** Conditional compilation with error-only stub

## Current Implementation

```rust
#[cfg(not(feature = "downloads"))]
#[allow(dead_code)]
async fn download_file(&self, _url: &str, _path: &Path) -> Result<()> {
    Err(BitNetError::Config(
        "Download feature not enabled. Build with --features downloads".to_string(),
    ))
}
```

## Proposed Solution

1. **Always Compile Download Logic**: Remove conditional compilation flags
2. **Runtime Feature Detection**: Check for network availability at runtime
3. **Graceful Fallbacks**: Provide alternative strategies when downloads fail
4. **Clear Error Messages**: Give users actionable guidance

## Implementation Plan

- [ ] Remove `#[cfg(feature = "downloads")]` conditional compilation
- [ ] Implement runtime network availability detection
- [ ] Add graceful fallback strategies for offline scenarios
- [ ] Improve error messages with actionable guidance
- [ ] Update documentation for download behavior

---

**Labels:** `tokenizers`, `network`, `user-experience`, `enhancement`
