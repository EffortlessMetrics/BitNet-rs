# [Tokenizers] Improve Conditional Download Feature Architecture

## Problem Description

The `download_file` method in `/home/steven/code/Rust/BitNet-rs/crates/bitnet-tokenizers/src/download.rs:289` uses conditional compilation (`#[cfg(not(feature = "downloads"))]`) to return an error when the downloads feature is disabled. While this pattern is functionally correct, it creates a brittle architecture where the same API has fundamentally different behavior based on compile-time features, leading to potential runtime surprises and inconsistent user experience.

## Environment

- **File**: `crates/bitnet-tokenizers/src/download.rs`
- **Method**: `SmartTokenizerDownload::download_file` (lines 205, 289)
- **Feature Flag**: `downloads`
- **MSRV**: Rust 1.90.0
- **Dependencies**: `reqwest` (optional)

## Current Implementation Analysis

### Existing Code Structure
```rust
impl SmartTokenizerDownload {
    #[cfg(feature = "downloads")]
    async fn download_file(&self, url: &str, path: &Path) -> Result<()> {
        // Full implementation with reqwest client
        // ... actual download logic ...
    }

    #[cfg(not(feature = "downloads"))]
    #[allow(dead_code)]
    async fn download_file(&self, _url: &str, _path: &Path) -> Result<()> {
        Err(BitNetError::Config(
            "Download feature not enabled. Build with --features downloads".to_string(),
        ))
    }
}
```

### Architectural Issues

1. **Inconsistent API Behavior**: Same method signature, completely different semantics
2. **Runtime Error Discovery**: Users only discover missing functionality at runtime
3. **Complex Conditional Compilation**: Multiple `#[cfg]` blocks throughout the struct
4. **Testing Complexity**: Different behavior requires separate test paths
5. **Documentation Confusion**: API docs don't clearly indicate feature-dependent behavior

## Root Cause Analysis

1. **Feature Flag Overuse**: Using feature flags for core functionality that could be better abstracted
2. **Monolithic Design**: Single struct handles both download and non-download scenarios
3. **Error Handling Pattern**: Runtime errors for compile-time decisions
4. **Missing Abstraction**: No clear separation between download and cache-only operations

## Impact Assessment

### Severity: Medium
### Affected Components: Tokenizer discovery, user experience, testing

**User Experience Impact:**
- Confusing error messages for users without download feature
- Inconsistent behavior between different build configurations
- Difficult to provide clear documentation about feature requirements

**Development Impact:**
- Complex conditional compilation throughout codebase
- Testing requires multiple feature flag combinations
- API evolution becomes more complex with feature dependencies

**Deployment Impact:**
- Build configuration choices affect runtime behavior
- Potential for misconfigured deployments missing essential functionality

## Proposed Solution

### Primary Approach: Trait-Based Download Abstraction

Replace conditional compilation with a trait-based architecture that provides clear separation between download and cache-only behaviors while maintaining a consistent API.

#### Implementation Plan

**1. Download Strategy Trait**

```rust
//! Download strategy abstraction for tokenizer acquisition

use crate::discovery::TokenizerDownloadInfo;
use bitnet_common::Result;
use std::path::{Path, PathBuf};
use async_trait::async_trait;

/// Trait for tokenizer acquisition strategies
#[async_trait]
pub trait TokenizerAcquisition: Send + Sync {
    /// Acquire tokenizer files based on download info
    async fn acquire_tokenizer(&self, info: &TokenizerDownloadInfo) -> Result<PathBuf>;

    /// Check if acquisition method is available
    fn is_available(&self) -> bool;

    /// Get human-readable description of acquisition method
    fn description(&self) -> &'static str;

    /// Get configuration requirements for this acquisition method
    fn requirements(&self) -> Vec<String>;
}

/// Network-based tokenizer downloading
#[cfg(feature = "downloads")]
pub struct NetworkAcquisition {
    client: reqwest::Client,
    cache_dir: PathBuf,
}

/// Cache-only tokenizer acquisition (no downloads)
pub struct CacheOnlyAcquisition {
    cache_dir: PathBuf,
}

/// Combined acquisition strategy with fallback
pub struct HybridAcquisition {
    strategies: Vec<Box<dyn TokenizerAcquisition>>,
}
```

**2. Network-Based Implementation**

```rust
#[cfg(feature = "downloads")]
impl NetworkAcquisition {
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent("BitNet-rs/0.1.0")
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| {
                BitNetError::Config(format!("HTTP client initialization failed: {}", e))
            })?;

        Ok(Self { client, cache_dir })
    }

    async fn download_file_internal(&self, url: &str, path: &Path) -> Result<()> {
        use std::io::Write;

        debug!("Downloading: {} -> {}", url, path.display());

        // Check if partial download exists
        let temp_path = path.with_extension("download");
        let start_pos = if temp_path.exists() {
            temp_path.metadata()?.len()
        } else {
            0
        };

        // Create request with range header for resume
        let mut request = self.client.get(url);
        if start_pos > 0 {
            request = request.header("Range", format!("bytes={}-", start_pos));
            info!("Resuming download from byte {}", start_pos);
        }

        let response = request.send().await
            .map_err(|e| BitNetError::Model(ModelError::NetworkError(e.to_string())))?;

        if !response.status().is_success() {
            return Err(BitNetError::Model(ModelError::NetworkError(
                format!("HTTP {}: {}", response.status(), url)
            )));
        }

        // Handle content and write to file
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(start_pos > 0)
            .write(true)
            .open(&temp_path)?;

        let content = response.bytes().await
            .map_err(|e| BitNetError::Model(ModelError::NetworkError(e.to_string())))?;

        file.write_all(&content)?;
        file.sync_all()?;

        // Atomically move completed download
        std::fs::rename(&temp_path, path)?;

        info!("Downloaded {} ({} bytes)", path.display(), content.len());
        Ok(())
    }
}

#[cfg(feature = "downloads")]
#[async_trait]
impl TokenizerAcquisition for NetworkAcquisition {
    async fn acquire_tokenizer(&self, info: &TokenizerDownloadInfo) -> Result<PathBuf> {
        info!("Acquiring tokenizer via network for repo: {}", info.repo);

        // Check cache first
        if let Some(cached_path) = self.find_cached_tokenizer(&info.cache_key) {
            debug!("Using cached tokenizer: {}", cached_path.display());
            return Ok(cached_path);
        }

        // Download all required files
        let target_dir = self.cache_dir.join(&info.cache_key);
        std::fs::create_dir_all(&target_dir)?;

        let mut primary_path = None;

        for (i, filename) in info.files.iter().enumerate() {
            let url = format!("https://huggingface.co/{}/resolve/main/{}", info.repo, filename);
            let file_path = target_dir.join(filename);

            debug!(
                "Downloading file {}/{}: {} -> {}",
                i + 1,
                info.files.len(),
                url,
                file_path.display()
            );

            self.download_file_internal(&url, &file_path).await?;

            // Use first .json file as primary, or first file if no json
            if primary_path.is_none() && (filename.ends_with(".json") || i == 0) {
                primary_path = Some(file_path);
            }
        }

        primary_path.ok_or_else(|| {
            BitNetError::Model(ModelError::TokenizerError(
                "No tokenizer files downloaded".to_string()
            ))
        })
    }

    fn is_available(&self) -> bool {
        true // Network acquisition is always available when feature is enabled
    }

    fn description(&self) -> &'static str {
        "Network download from HuggingFace Hub"
    }

    fn requirements(&self) -> Vec<String> {
        vec![
            "Internet connection".to_string(),
            "HuggingFace Hub access".to_string(),
        ]
    }
}
```

**3. Cache-Only Implementation**

```rust
impl CacheOnlyAcquisition {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    fn find_cached_tokenizer(&self, cache_key: &str) -> Option<PathBuf> {
        let cache_dir = self.cache_dir.join(cache_key);
        if !cache_dir.exists() {
            return None;
        }

        // Look for tokenizer files in order of preference
        let candidates = vec!["tokenizer.json", "tokenizer.model", "vocab.txt"];

        for candidate in &candidates {
            let path = cache_dir.join(candidate);
            if path.exists() {
                return Some(path);
            }
        }

        // Return any file in the directory as fallback
        std::fs::read_dir(&cache_dir).ok()?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .find(|path| path.is_file())
    }
}

#[async_trait]
impl TokenizerAcquisition for CacheOnlyAcquisition {
    async fn acquire_tokenizer(&self, info: &TokenizerDownloadInfo) -> Result<PathBuf> {
        info!("Acquiring tokenizer from cache for repo: {}", info.repo);

        if let Some(cached_path) = self.find_cached_tokenizer(&info.cache_key) {
            debug!("Found cached tokenizer: {}", cached_path.display());
            Ok(cached_path)
        } else {
            Err(BitNetError::Model(ModelError::TokenizerNotFound {
                cache_key: info.cache_key.clone(),
                suggestions: vec![
                    "Pre-download tokenizer files manually".to_string(),
                    "Build with --features downloads for automatic downloading".to_string(),
                    format!("Place tokenizer files in: {}",
                        self.cache_dir.join(&info.cache_key).display()),
                ],
            }))
        }
    }

    fn is_available(&self) -> bool {
        true // Cache-only is always available
    }

    fn description(&self) -> &'static str {
        "Cache-only acquisition (no downloads)"
    }

    fn requirements(&self) -> Vec<String> {
        vec![
            "Pre-cached tokenizer files".to_string(),
            "Manual tokenizer placement".to_string(),
        ]
    }
}
```

**4. Hybrid Strategy Implementation**

```rust
impl HybridAcquisition {
    pub fn new() -> Self {
        let mut strategies: Vec<Box<dyn TokenizerAcquisition>> = Vec::new();

        // Add network acquisition if available
        #[cfg(feature = "downloads")]
        {
            if let Ok(cache_dir) = CacheManager::cache_directory() {
                if let Ok(network) = NetworkAcquisition::new(cache_dir.clone()) {
                    strategies.push(Box::new(network));
                }
            }
        }

        // Always add cache-only as fallback
        if let Ok(cache_dir) = CacheManager::cache_directory() {
            strategies.push(Box::new(CacheOnlyAcquisition::new(cache_dir)));
        }

        Self { strategies }
    }

    pub fn with_strategies(strategies: Vec<Box<dyn TokenizerAcquisition>>) -> Self {
        Self { strategies }
    }

    pub fn available_strategies(&self) -> Vec<String> {
        self.strategies.iter()
            .filter(|s| s.is_available())
            .map(|s| s.description().to_string())
            .collect()
    }
}

#[async_trait]
impl TokenizerAcquisition for HybridAcquisition {
    async fn acquire_tokenizer(&self, info: &TokenizerDownloadInfo) -> Result<PathBuf> {
        let mut last_error = None;

        for strategy in &self.strategies {
            if !strategy.is_available() {
                debug!("Skipping unavailable strategy: {}", strategy.description());
                continue;
            }

            debug!("Trying acquisition strategy: {}", strategy.description());

            match strategy.acquire_tokenizer(info).await {
                Ok(path) => {
                    info!("Successfully acquired tokenizer using: {}", strategy.description());
                    return Ok(path);
                }
                Err(e) => {
                    warn!("Strategy '{}' failed: {}", strategy.description(), e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            BitNetError::Config("No acquisition strategies available".to_string())
        }))
    }

    fn is_available(&self) -> bool {
        self.strategies.iter().any(|s| s.is_available())
    }

    fn description(&self) -> &'static str {
        "Hybrid acquisition with multiple strategies"
    }

    fn requirements(&self) -> Vec<String> {
        let mut all_requirements = Vec::new();
        for strategy in &self.strategies {
            if strategy.is_available() {
                all_requirements.extend(strategy.requirements());
            }
        }
        all_requirements.sort();
        all_requirements.dedup();
        all_requirements
    }
}
```

**5. Refactored SmartTokenizerDownload**

```rust
/// Enhanced tokenizer download system with pluggable acquisition strategies
pub struct SmartTokenizerDownload {
    acquisition: Box<dyn TokenizerAcquisition>,
    cache_dir: PathBuf,
}

impl SmartTokenizerDownload {
    /// Create with automatic strategy selection
    pub fn new() -> Result<Self> {
        let cache_dir = CacheManager::cache_directory()?;
        let acquisition = Box::new(HybridAcquisition::new());

        Ok(Self {
            acquisition,
            cache_dir,
        })
    }

    /// Create with specific acquisition strategy
    pub fn with_strategy(acquisition: Box<dyn TokenizerAcquisition>) -> Result<Self> {
        let cache_dir = CacheManager::cache_directory()?;

        Ok(Self {
            acquisition,
            cache_dir,
        })
    }

    /// Create cache-only instance (no downloads)
    pub fn cache_only() -> Result<Self> {
        let cache_dir = CacheManager::cache_directory()?;
        let acquisition = Box::new(CacheOnlyAcquisition::new(cache_dir.clone()));

        Ok(Self {
            acquisition,
            cache_dir,
        })
    }

    /// Create network-enabled instance
    #[cfg(feature = "downloads")]
    pub fn with_downloads() -> Result<Self> {
        let cache_dir = CacheManager::cache_directory()?;
        let acquisition = Box::new(NetworkAcquisition::new(cache_dir.clone())?);

        Ok(Self {
            acquisition,
            cache_dir,
        })
    }

    /// Get information about available acquisition methods
    pub fn acquisition_info(&self) -> AcquisitionInfo {
        AcquisitionInfo {
            description: self.acquisition.description().to_string(),
            available: self.acquisition.is_available(),
            requirements: self.acquisition.requirements(),
        }
    }

    /// Download tokenizer using configured acquisition strategy
    pub async fn download_tokenizer(&self, info: &TokenizerDownloadInfo) -> Result<PathBuf> {
        info!("Acquiring tokenizer for repo: {}", info.repo);

        // Check if acquisition method is available
        if !self.acquisition.is_available() {
            return Err(BitNetError::Config(format!(
                "Acquisition method '{}' is not available. Requirements: {:?}",
                self.acquisition.description(),
                self.acquisition.requirements()
            )));
        }

        // Check offline mode
        if Self::is_offline_mode() && !self.is_cache_only_strategy() {
            return Err(BitNetError::Config(format!(
                "Cannot use '{}' in offline mode. Use cache-only strategy instead.",
                self.acquisition.description()
            )));
        }

        self.acquisition.acquire_tokenizer(info).await
    }

    fn is_cache_only_strategy(&self) -> bool {
        self.acquisition.description().contains("cache-only") ||
        self.acquisition.description().contains("Cache-only")
    }

    // ... existing helper methods ...
}

#[derive(Debug, Clone)]
pub struct AcquisitionInfo {
    pub description: String,
    pub available: bool,
    pub requirements: Vec<String>,
}
```

**6. Enhanced Error Types**

```rust
// In bitnet_common/src/error.rs

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    // ... existing variants ...

    /// Tokenizer not found with helpful suggestions
    TokenizerNotFound {
        cache_key: String,
        suggestions: Vec<String>,
    },

    /// Network error during download
    NetworkError(String),

    /// Acquisition strategy not available
    AcquisitionUnavailable {
        strategy: String,
        requirements: Vec<String>,
    },
}
```

### Alternative Solutions Considered

1. **Runtime Feature Detection**: Check for download capability at runtime
2. **Separate Crates**: Split download functionality into separate crate
3. **Plugin Architecture**: Dynamic loading of download strategies

## Implementation Breakdown

### Phase 1: Trait Definition (Week 1)
- [ ] Define `TokenizerAcquisition` trait with async methods
- [ ] Create basic error types for acquisition failures
- [ ] Add trait documentation and examples
- [ ] Implement trait for existing functionality

### Phase 2: Strategy Implementations (Week 1-2)
- [ ] Implement `NetworkAcquisition` with full download logic
- [ ] Implement `CacheOnlyAcquisition` for offline scenarios
- [ ] Create `HybridAcquisition` with fallback logic
- [ ] Add configuration and initialization methods

### Phase 3: Integration (Week 2)
- [ ] Refactor `SmartTokenizerDownload` to use trait-based strategies
- [ ] Update constructor methods for different strategies
- [ ] Migrate existing conditional compilation logic
- [ ] Update error handling and reporting

### Phase 4: Enhanced Features (Week 2-3)
- [ ] Add acquisition method introspection
- [ ] Implement strategy selection helpers
- [ ] Create configuration validation
- [ ] Add performance monitoring for different strategies

### Phase 5: Testing and Migration (Week 3)
- [ ] Create comprehensive test suite for all strategies
- [ ] Add integration tests for feature flag combinations
- [ ] Performance testing across strategies
- [ ] Migration guide and documentation updates

## Testing Strategy

### Strategy Testing
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_only_acquisition() {
        let cache_dir = tempfile::tempdir().unwrap().path().to_path_buf();
        let acquisition = CacheOnlyAcquisition::new(cache_dir.clone());

        let info = TokenizerDownloadInfo {
            repo: "test/model".to_string(),
            cache_key: "test-key".to_string(),
            files: vec!["tokenizer.json".to_string()],
            expected_vocab: None,
        };

        // Should fail without cached file
        assert!(acquisition.acquire_tokenizer(&info).await.is_err());

        // Create cached file
        let cache_file = cache_dir.join("test-key").join("tokenizer.json");
        std::fs::create_dir_all(cache_file.parent().unwrap()).unwrap();
        std::fs::write(&cache_file, "{}").unwrap();

        // Should succeed with cached file
        let result = acquisition.acquire_tokenizer(&info).await.unwrap();
        assert_eq!(result, cache_file);
    }

    #[cfg(feature = "downloads")]
    #[tokio::test]
    async fn test_network_acquisition() {
        // Test network acquisition with mock server
        let mut server = mockito::Server::new_async().await;
        let mock = server.mock("GET", "/test/tokenizer.json")
            .with_status(200)
            .with_body("{}")
            .create_async().await;

        // Test implementation...
    }

    #[tokio::test]
    async fn test_hybrid_acquisition_fallback() {
        let hybrid = HybridAcquisition::new();
        assert!(hybrid.is_available());

        // Test that it tries multiple strategies
        let strategies = hybrid.available_strategies();
        assert!(!strategies.is_empty());
    }
}
```

### Feature Flag Testing
```bash
# Test with downloads enabled
cargo test --features downloads acquisition_with_downloads

# Test without downloads (cache-only)
cargo test --no-default-features acquisition_cache_only

# Test hybrid behavior
cargo test acquisition_hybrid_strategies
```

## Acceptance Criteria

- [ ] Trait-based architecture eliminates conditional compilation in core logic
- [ ] Clear separation between download and cache-only functionality
- [ ] Consistent API behavior regardless of feature flags
- [ ] Helpful error messages with actionable suggestions
- [ ] Strategy introspection and selection capabilities
- [ ] Zero regression in existing functionality
- [ ] Comprehensive test coverage for all acquisition strategies
- [ ] Performance parity with existing implementation
- [ ] Clear documentation of acquisition strategies and requirements

## Dependencies

### New Dependencies
```toml
[dependencies]
async-trait = "0.1"

[dev-dependencies]
mockito = "1.0"
tempfile = "3.0"
```

## Related Issues

- Tokenizer discovery and caching improvements
- Error handling standardization
- Feature flag architecture review
- Testing infrastructure for conditional features

## BitNet-Specific Considerations

- **Model Compatibility**: Different acquisition strategies should maintain tokenizer compatibility
- **Cache Management**: Consistent caching behavior across all acquisition methods
- **Offline Deployment**: Cache-only strategy supports air-gapped deployments
- **Performance**: Network acquisition should not impact inference performance
- **Error Recovery**: Clear guidance for users when acquisition fails

This refactoring transforms a brittle conditional compilation pattern into a flexible, testable, and user-friendly acquisition system that provides clear guarantees about behavior while maintaining the same external API.
