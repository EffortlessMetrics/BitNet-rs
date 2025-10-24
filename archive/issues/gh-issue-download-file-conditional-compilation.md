# [Build System] Fix conditional compilation issues in download_file function

## Problem Description

The `download_file` function in `crates/bitnet-models/src/utils.rs` has conditional compilation attributes that may cause build issues when the `download` feature is not enabled, potentially breaking the build or causing undefined behavior.

## Root Cause Analysis

Conditional compilation for download functionality needs proper feature gating to ensure:
1. Clean compilation when download features are disabled
2. Proper error handling when functionality is unavailable
3. Clear documentation of feature requirements

## Proposed Solution

### Proper Feature Gating

```rust
#[cfg(feature = "download")]
pub async fn download_file(url: &str, path: &Path) -> Result<()> {
    // Full download implementation
    use reqwest;
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;

    let response = reqwest::get(url).await?;
    let mut file = File::create(path).await?;
    let content = response.bytes().await?;
    file.write_all(&content).await?;

    Ok(())
}

#[cfg(not(feature = "download"))]
pub async fn download_file(_url: &str, _path: &Path) -> Result<()> {
    Err(anyhow::anyhow!(
        "Download functionality disabled. Enable 'download' feature to use this function."
    ))
}
```

### Feature Documentation

```rust
//! # Model Download Utilities
//!
//! This module provides utilities for downloading model files from remote sources.
//!
//! ## Features
//!
//! - `download`: Enables HTTP download functionality (requires `reqwest` and `tokio`)
//!
//! ## Usage
//!
//! Add to `Cargo.toml`:
//! ```toml
//! [dependencies]
//! bitnet-models = { version = "0.1", features = ["download"] }
//! ```
```

## Acceptance Criteria

- [ ] Clean compilation with and without download feature
- [ ] Clear error messages when functionality is disabled
- [ ] Proper feature documentation
- [ ] No undefined behavior in feature-disabled builds

## Priority: Medium

Build system reliability improvement that ensures consistent compilation across different feature configurations.
