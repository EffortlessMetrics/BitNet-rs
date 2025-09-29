# [Downloads] Improve Conditional Download Feature Architecture

## Problem Description

The `download_file` function in `crates/bitnet-tokenizers/src/download.rs` uses conditional compilation with feature flags, which creates inconsistent behavior and forces users to rebuild with specific features enabled for download functionality.

## Environment

- **Component**: `crates/bitnet-tokenizers/src/download.rs`
- **Function**: `download_file`
- **Feature**: `downloads` feature flag
- **Impact**: Runtime tokenizer download capabilities

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

Replace conditional compilation with runtime feature detection:

```rust
async fn download_file(&self, url: &str, path: &Path) -> Result<()> {
    if !self.download_enabled() {
        return Err(BitNetError::Config(
            "Download functionality disabled. Enable with download configuration.".to_string(),
        ));
    }

    // Actual download implementation
    self.perform_download(url, path).await
}

fn download_enabled(&self) -> bool {
    self.config.enable_downloads && self.has_network_access()
}
```

## Implementation Tasks

- [ ] Remove conditional compilation attributes
- [ ] Add runtime download capability detection
- [ ] Implement graceful fallback mechanisms
- [ ] Add configuration-based download control

## Acceptance Criteria

- [ ] No build-time feature requirements for basic functionality
- [ ] Runtime detection of download capabilities
- [ ] Clear error messages for unavailable features
- [ ] Consistent behavior across different build configurations

## Related Issues

- **Related to**: Configuration system improvements
- **Blocks**: Seamless tokenizer acquisition