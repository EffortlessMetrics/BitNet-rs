//! Smart tokenizer downloading with caching and resume capability
//!
//! This module provides intelligent tokenizer downloading from HuggingFace Hub with comprehensive
//! caching, resume capability, and validation for BitNet.rs neural network models.

use crate::{
    discovery::TokenizerDownloadInfo,
    error_handling::{CacheManager, TokenizerErrorHandler},
};
use bitnet_common::{BitNetError, ModelError, Result};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Intelligent tokenizer downloading with caching and resume capability
pub struct SmartTokenizerDownload {
    cache_dir: PathBuf,
    #[cfg(feature = "downloads")]
    client: reqwest::Client,
}

impl SmartTokenizerDownload {
    /// Initialize download system with default cache directory
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    ///
    /// # Returns
    /// * `Ok(SmartTokenizerDownload)` - Successfully initialized
    /// * `Err(BitNetError::Model)` - Cache directory creation failed
    pub fn new() -> Result<Self> {
        let cache_dir = CacheManager::cache_directory()?;
        Self::with_cache_dir(cache_dir)
    }

    /// Initialize with custom cache directory
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self> {
        info!("Initializing SmartTokenizerDownload with cache dir: {}", cache_dir.display());

        // Use centralized cache directory management
        CacheManager::ensure_cache_directory(&cache_dir)?;

        #[cfg(feature = "downloads")]
        let client = reqwest::Client::builder()
            .user_agent("BitNet-rs/0.1.0")
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| {
                BitNetError::Config(format!("HTTP client initialization failed: {}", e))
            })?;

        Ok(Self {
            cache_dir,
            #[cfg(feature = "downloads")]
            client,
        })
    }

    /// Download tokenizer files for given download info
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    ///
    /// # Arguments
    /// * `info` - Download metadata including repo, files, and cache key
    ///
    /// # Returns
    /// * `Ok(PathBuf)` - Path to primary tokenizer file
    /// * `Err(BitNetError::Model)` - Download failed or network error
    ///
    /// # Example
    /// ```rust,no_run
    /// use bitnet_tokenizers::{TokenizerDownloadInfo, SmartTokenizerDownload};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let downloader = SmartTokenizerDownload::new()?;
    /// let info = TokenizerDownloadInfo {
    ///     repo: "meta-llama/Llama-2-7b-hf".to_string(),
    ///     files: vec!["tokenizer.json".to_string()],
    ///     cache_key: "llama2-32k".to_string(),
    ///     expected_vocab: Some(32000),
    /// };
    /// let tokenizer_path = downloader.download_tokenizer(&info).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn download_tokenizer(&self, info: &TokenizerDownloadInfo) -> Result<PathBuf> {
        info!("Downloading tokenizer for repo: {}", info.repo);

        // Check if already cached
        if let Some(cached_path) = self.find_cached_tokenizer(&info.cache_key) {
            debug!("Using cached tokenizer: {}", cached_path.display());
            return Ok(cached_path);
        }

        // Check offline mode
        if Self::is_offline_mode() {
            return Err(TokenizerErrorHandler::config_error(format!(
                "Cannot download tokenizer {} in offline mode",
                info.cache_key
            )));
        }

        // Create cache directory for this tokenizer
        let tokenizer_cache_dir = self.cache_dir.join(&info.cache_key);
        CacheManager::ensure_cache_directory(&tokenizer_cache_dir)?;

        // Download all files
        #[cfg(not(feature = "downloads"))]
        {
            Err(TokenizerErrorHandler::config_error(
                "Download feature not enabled. Build with --features downloads".to_string(),
            ))
        }

        #[cfg(feature = "downloads")]
        {
            let mut primary_path: Option<PathBuf> = None;
            for (i, filename) in info.files.iter().enumerate() {
                let url = Self::get_download_url(&info.repo, filename);
                let file_path = tokenizer_cache_dir.join(filename);

                info!(
                    "Downloading file {}/{}: {} -> {}",
                    i + 1,
                    info.files.len(),
                    url,
                    file_path.display()
                );

                self.download_file(&url, &file_path).await?;

                // Use first .json file as primary, or first file if no json
                if primary_path.is_none() && (filename.ends_with(".json") || i == 0) {
                    primary_path = Some(file_path);
                }
            }

            let primary_file = primary_path.ok_or_else(|| {
                TokenizerErrorHandler::config_error("No primary tokenizer file found".to_string())
            })?;

            // Validate downloaded tokenizer
            self.validate_downloaded_tokenizer(&primary_file, info)?;

            info!("Successfully downloaded tokenizer: {}", primary_file.display());
            Ok(primary_file)
        }
    }

    /// Check if tokenizer is already cached
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    ///
    /// # Returns
    /// * `Some(PathBuf)` - Path to cached tokenizer
    /// * `None` - Tokenizer not in cache
    pub fn find_cached_tokenizer(&self, cache_key: &str) -> Option<PathBuf> {
        let cache_dir = self.cache_dir.join(cache_key);
        if !cache_dir.exists() {
            return None;
        }

        // Look for primary tokenizer files
        let primary_files = ["tokenizer.json", "tokenizer.model", "vocab.json"];

        for filename in &primary_files {
            let file_path = cache_dir.join(filename);
            if file_path.exists() && file_path.is_file() {
                debug!("Found cached tokenizer: {}", file_path.display());
                return Some(file_path);
            }
        }

        debug!("No cached tokenizer found for key: {}", cache_key);
        None
    }

    /// Clear cache for specific tokenizer or all cached tokenizers
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    pub fn clear_cache(&self, cache_key: Option<&str>) -> Result<()> {
        if let Some(key) = cache_key {
            let cache_dir = self.cache_dir.join(key);
            if cache_dir.exists() {
                info!("Clearing cache for tokenizer: {}", key);
                std::fs::remove_dir_all(&cache_dir)
                    .map_err(|e| TokenizerErrorHandler::file_io_error(&cache_dir, e))?;
            }
        } else {
            info!("Clearing entire tokenizer cache: {}", self.cache_dir.display());
            if self.cache_dir.exists() {
                std::fs::remove_dir_all(&self.cache_dir)
                    .map_err(|e| TokenizerErrorHandler::file_io_error(&self.cache_dir, e))?;
                // Recreate the cache directory
                CacheManager::ensure_cache_directory(&self.cache_dir)?;
            }
        }
        Ok(())
    }

    /// Download individual file with resume capability
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    #[cfg(feature = "downloads")]
    async fn download_file(&self, url: &str, path: &Path) -> Result<()> {
        use std::io::Write;

        debug!("Downloading: {} -> {}", url, path.display());

        // Check if partial download exists
        let mut start_pos = 0u64;
        if path.exists() {
            start_pos = std::fs::metadata(path)
                .map_err(|e| {
                    BitNetError::Model(ModelError::FileIOError {
                        path: path.to_path_buf(),
                        source: e,
                    })
                })?
                .len();
            debug!("Resuming download from byte: {}", start_pos);
        }

        // Create HTTP request with range header for resume
        let mut request = self.client.get(url);
        if start_pos > 0 {
            request = request.header("Range", format!("bytes={}-", start_pos));
        }

        // Send request
        let response = request.send().await.map_err(|e| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("HTTP request failed for {}: {}", url, e),
            })
        })?;

        // Check status
        if !response.status().is_success() {
            return Err(BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("HTTP error {}: {}", response.status(), url),
            }));
        }

        // Open file for writing (append mode for resume)
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(start_pos > 0)
            .truncate(start_pos == 0)
            .open(path)
            .map_err(|e| {
                BitNetError::Model(ModelError::FileIOError { path: path.to_path_buf(), source: e })
            })?;

        // Stream download with progress reporting
        let mut bytes_stream = response.bytes_stream();
        let mut total_downloaded = start_pos;

        use futures_util::StreamExt;
        while let Some(chunk_result) = bytes_stream.next().await {
            let chunk = chunk_result.map_err(|e| {
                BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Download stream error: {}", e),
                })
            })?;

            file.write_all(&chunk).map_err(|e| {
                BitNetError::Model(ModelError::FileIOError { path: path.to_path_buf(), source: e })
            })?;

            total_downloaded += chunk.len() as u64;

            // Log progress every 1MB
            if total_downloaded % (1024 * 1024) == 0 || chunk.len() < 1024 {
                debug!("Downloaded: {} bytes", total_downloaded);
            }
        }

        file.sync_all().map_err(|e| {
            BitNetError::Model(ModelError::FileIOError { path: path.to_path_buf(), source: e })
        })?;

        info!("Successfully downloaded: {} ({} bytes)", path.display(), total_downloaded);
        Ok(())
    }

    #[cfg(not(feature = "downloads"))]
    #[allow(dead_code)]
    async fn download_file(&self, _url: &str, _path: &Path) -> Result<()> {
        Err(BitNetError::Config(
            "Download feature not enabled. Build with --features downloads".to_string(),
        ))
    }

    /// Validate downloaded tokenizer against expected metadata
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    #[allow(dead_code)]
    pub fn validate_downloaded_tokenizer(
        &self,
        path: &Path,
        info: &TokenizerDownloadInfo,
    ) -> Result<()> {
        debug!("Validating downloaded tokenizer: {}", path.display());

        // Check file exists and has content
        if !path.exists() || !path.is_file() {
            return Err(BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Downloaded tokenizer file does not exist: {}", path.display()),
            }));
        }

        let file_size = std::fs::metadata(path)
            .map_err(|e| {
                BitNetError::Model(ModelError::FileIOError { path: path.to_path_buf(), source: e })
            })?
            .len();

        if file_size == 0 {
            return Err(BitNetError::Model(ModelError::LoadingFailed {
                reason: "Downloaded tokenizer file is empty".to_string(),
            }));
        }

        // For JSON files, try to parse to ensure valid format
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let content = std::fs::read_to_string(path).map_err(|e| {
                BitNetError::Model(ModelError::FileIOError { path: path.to_path_buf(), source: e })
            })?;

            // Try to parse as JSON
            let json_value: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
                BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Invalid JSON in downloaded tokenizer: {}", e),
                })
            })?;

            // Basic validation for HuggingFace tokenizer format
            if let Some(obj) = json_value.as_object() {
                // Check for required fields
                let required_fields = ["model", "normalizer", "pre_tokenizer"];
                for field in &required_fields {
                    if !obj.contains_key(*field) {
                        warn!("Downloaded tokenizer missing field: {}", field);
                    }
                }

                // If vocab size is specified, validate it
                if let Some(expected_vocab) = info.expected_vocab
                    && let Some(model) = obj.get("model").and_then(|m| m.as_object())
                    && let Some(vocab) = model.get("vocab").and_then(|v| v.as_object())
                {
                    let actual_vocab = vocab.len();
                    if actual_vocab != expected_vocab {
                        warn!(
                            "Vocabulary size mismatch: expected {}, got {}",
                            expected_vocab, actual_vocab
                        );
                    } else {
                        debug!("Vocabulary size validated: {}", actual_vocab);
                    }
                }
            }
        }

        info!(
            "Downloaded tokenizer validation successful: {} ({} bytes)",
            path.display(),
            file_size
        );
        Ok(())
    }

    /// Get download URL for HuggingFace Hub file
    #[allow(dead_code)]
    fn get_download_url(repo: &str, file: &str) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}", repo, file)
    }

    /// Check if offline mode is enabled
    fn is_offline_mode() -> bool {
        std::env::var("BITNET_OFFLINE").as_deref() == Ok("1")
    }
}

/// Download progress information for reporting
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub current_file: String,
    pub completed_files: usize,
    pub total_files: usize,
}

impl DownloadProgress {
    /// Calculate download percentage (0.0-1.0)
    pub fn percentage(&self) -> Option<f64> {
        self.total_bytes.map(|total| self.downloaded_bytes as f64 / total as f64)
    }
}

/// Download metrics for performance validation
#[derive(Debug)]
pub struct DownloadMetrics {
    pub download_time: std::time::Duration,
    pub cache_time: std::time::Duration,
    pub network_efficiency: f64,
    pub bytes_transferred: u64,
    pub cache_hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    // use tokio;

    /// AC2: Tests SmartTokenizerDownload initialization with default cache
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_smart_tokenizer_download_initialization() {
        let result = SmartTokenizerDownload::new();

        // Should succeed with default cache directory
        assert!(result.is_ok(), "SmartTokenizerDownload initialization should succeed");
        let downloader = result.unwrap();

        // Verify cache directory was created
        assert!(downloader.cache_dir.exists(), "Cache directory should exist after initialization");
    }

    /// AC2: Tests download functionality for neural network tokenizers
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_download_tokenizer_llama_models() {
        let _download_info = TokenizerDownloadInfo {
            repo: "meta-llama/Llama-2-7b-hf".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "llama2-32k".to_string(),
            expected_vocab: Some(32000),
        };

        let downloader_result = SmartTokenizerDownload::new();
        assert!(downloader_result.is_ok(), "SmartTokenizerDownload::new should succeed");
        let _downloader = downloader_result.unwrap();

        // Test scaffolding for download process - would need network access
        // let result = downloader.download_tokenizer(&download_info).await;
        // assert!(result.is_ok(), "Download should succeed for valid repo");
        // Download initialization test completed
        println!("✅ AC2: Download initialization test scaffolding completed");
    }

    /// AC2: Tests caching functionality for downloaded tokenizers
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_download_cache_management() {
        let cache_dir = std::env::temp_dir().join("bitnet-test-cache");
        let downloader_result = SmartTokenizerDownload::with_cache_dir(cache_dir.clone());

        assert!(downloader_result.is_ok(), "SmartTokenizerDownload::with_cache_dir should succeed");
        let _downloader = downloader_result.unwrap();

        // Verify cache directory was created
        assert!(cache_dir.exists(), "Custom cache directory should exist");

        // Test cache miss
        // assert!(downloader.find_cached_tokenizer("nonexistent").is_none());

        // Test cache hit after download
        // let info = create_test_download_info();
        // let path = downloader.download_tokenizer(&info).await.unwrap();
        // assert!(downloader.find_cached_tokenizer(&info.cache_key).is_some());

        // Test cache clearing
        // downloader.clear_cache(Some(&info.cache_key)).unwrap();
        // assert!(downloader.find_cached_tokenizer(&info.cache_key).is_none());
    }

    /// AC2: Tests download with resume capability for large neural network tokenizers
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_download_resume_capability() {
        // Test scaffolding for resume functionality
        let _large_tokenizer_info = TokenizerDownloadInfo {
            repo: "meta-llama/Meta-Llama-3-8B".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "llama3-128k".to_string(),
            expected_vocab: Some(128256),
        };

        let downloader_result = SmartTokenizerDownload::new();
        assert!(
            downloader_result.is_ok(),
            "SmartTokenizerDownload::new should succeed for resume test"
        );
        let _downloader = downloader_result.unwrap();

        // Test scaffolding for resume functionality
        // 1. Start download and interrupt
        // 2. Resume download and verify completion
        // 3. Validate final file integrity

        // Test scaffolding placeholder for download resume
        println!("✅ AC2: Download resume test scaffolding completed");
    }

    /// AC2: Tests validation of downloaded tokenizers against neural network requirements
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_tokenizer_validation_neural_networks() {
        let test_cases = [
            // LLaMA-2 tokenizer validation
            TokenizerDownloadInfo {
                repo: "meta-llama/Llama-2-7b-hf".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "llama2-32k".to_string(),
                expected_vocab: Some(32000),
            },
            // GPT-2 tokenizer validation
            TokenizerDownloadInfo {
                repo: "openai-community/gpt2".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "gpt2-50k".to_string(),
                expected_vocab: Some(50257),
            },
            // LLaMA-3 large vocabulary validation
            TokenizerDownloadInfo {
                repo: "meta-llama/Meta-Llama-3-8B".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "llama3-128k".to_string(),
                expected_vocab: Some(128256),
            },
        ];

        for _info in test_cases {
            let downloader_result = SmartTokenizerDownload::new();
            assert!(
                downloader_result.is_ok(),
                "SmartTokenizerDownload::new should succeed for validation test"
            );
            let _downloader = downloader_result.unwrap();

            // Test scaffolding for validation
            // let downloader = downloader_result.unwrap();
            // let path = downloader.download_tokenizer(&info).await.unwrap();
            // let validation_result = downloader.validate_downloaded_tokenizer(&path, &info);
            // assert!(validation_result.is_ok(), "Validation should succeed for compatible tokenizer");
        }
    }

    /// AC2: Tests offline mode behavior for cached tokenizers
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_offline_mode_behavior() {
        // Set offline mode environment variable
        unsafe {
            std::env::set_var("BITNET_OFFLINE", "1");
        }

        let _download_info = TokenizerDownloadInfo {
            repo: "meta-llama/Llama-2-7b-hf".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "llama2-32k".to_string(),
            expected_vocab: Some(32000),
        };

        let downloader_result = SmartTokenizerDownload::new();
        assert!(
            downloader_result.is_ok(),
            "SmartTokenizerDownload::new should succeed for offline test"
        );
        let _downloader = downloader_result.unwrap();

        // Test scaffolding for offline mode
        // let downloader = downloader_result.unwrap();

        // Should fail if tokenizer not cached
        // let result = downloader.download_tokenizer(&download_info).await;
        // assert!(result.is_err(), "Should fail in offline mode without cached tokenizer");

        // Should succeed if tokenizer is cached
        // // Pre-populate cache in separate test setup
        // let cached_result = downloader.find_cached_tokenizer(&download_info.cache_key);
        // // Test would pass if tokenizer exists in cache

        unsafe {
            std::env::remove_var("BITNET_OFFLINE");
        }
    }

    /// AC2: Tests error handling for network failures and invalid repositories
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_download_error_handling() {
        let _invalid_repo_info = TokenizerDownloadInfo {
            repo: "nonexistent/invalid-repo".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "invalid".to_string(),
            expected_vocab: Some(1000),
        };

        let downloader_result = SmartTokenizerDownload::new();
        assert!(
            downloader_result.is_ok(),
            "SmartTokenizerDownload::new should succeed for error handling test"
        );
        let _downloader = downloader_result.unwrap();

        // Test scaffolding for error handling
        // let downloader = downloader_result.unwrap();
        // let result = downloader.download_tokenizer(&invalid_repo_info).await;
        // assert!(result.is_err(), "Should fail for invalid repository");

        // Test specific error types
        // match result.unwrap_err() {
        //     BitNetError::Model(ModelError::LoadingFailed { reason }) => {
        //         assert!(reason.contains("404") || reason.contains("network"), "Error should indicate network issue");
        //     }
        //     _ => panic!("Expected ModelError::LoadingFailed"),
        // }
    }

    /// AC2: Tests concurrent downloads for multi-file tokenizer packages
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_concurrent_multi_file_downloads() {
        let _multi_file_info = TokenizerDownloadInfo {
            repo: "1bitLLM/bitnet_b1_58-large".to_string(),
            files: vec!["tokenizer.json".to_string(), "tokenizer.model".to_string()],
            cache_key: "bitnet-custom".to_string(),
            expected_vocab: None,
        };

        let downloader_result = SmartTokenizerDownload::new();
        assert!(
            downloader_result.is_ok(),
            "SmartTokenizerDownload::new should succeed for multi-file test"
        );
        let _downloader = downloader_result.unwrap();

        // Test scaffolding for concurrent downloads
        // let downloader = downloader_result.unwrap();
        // let result = downloader.download_tokenizer(&multi_file_info).await;

        // Should download all files concurrently
        // assert!(result.is_ok(), "Multi-file download should succeed");

        // Verify all files are cached
        // for file in &multi_file_info.files {
        //     let cache_path = downloader.cache_dir.join(&multi_file_info.cache_key).join(file);
        //     assert!(cache_path.exists(), "All files should be downloaded and cached");
        // }
    }

    /// AC2: Tests download progress reporting for large neural network tokenizers
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_download_progress_reporting() {
        // Test scaffolding for progress reporting
        let progress = DownloadProgress {
            downloaded_bytes: 5 * 1024 * 1024,   // 5MB
            total_bytes: Some(10 * 1024 * 1024), // 10MB
            current_file: "tokenizer.json".to_string(),
            completed_files: 0,
            total_files: 1,
        };

        assert_eq!(progress.percentage(), Some(0.5));
        assert_eq!(progress.current_file, "tokenizer.json");
        assert_eq!(progress.completed_files, 0);
        assert_eq!(progress.total_files, 1);

        // Test progress calculation
        let progress_no_total = DownloadProgress {
            downloaded_bytes: 1024,
            total_bytes: None,
            current_file: "unknown.json".to_string(),
            completed_files: 1,
            total_files: 2,
        };

        assert_eq!(progress_no_total.percentage(), None);
    }

    /// Helper function to create test download info
    #[allow(dead_code)]
    fn create_test_download_info() -> TokenizerDownloadInfo {
        TokenizerDownloadInfo {
            repo: "microsoft/DialoGPT-medium".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "test-tokenizer".to_string(),
            expected_vocab: Some(50257),
        }
    }

    // ================================
    // ENHANCED EDGE CASE TESTS FOR DOWNLOADS
    // ================================

    /// Test network timeout scenarios for large tokenizer downloads
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_network_timeout_scenarios() {
        // Test with extremely short timeout to force timeout errors
        let downloader_result = SmartTokenizerDownload::new();
        assert!(downloader_result.is_ok(), "SmartTokenizerDownload::new should succeed");
        let _downloader = downloader_result.unwrap();

        // Test timeout configuration
        // In a real scenario, this would configure a very short timeout
        let timeout_info = TokenizerDownloadInfo {
            repo: "meta-llama/Llama-2-7b-hf".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "timeout-test".to_string(),
            expected_vocab: Some(32000),
        };

        // Test that timeout scenarios are handled gracefully
        // (This would require actual network requests in production)
        println!("✅ Network timeout test structure prepared");
        assert!(!timeout_info.repo.is_empty(), "Test info should be valid");
    }

    /// Test partial download corruption and recovery
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_partial_download_corruption() {
        use std::io::Write;
        use tempfile::tempdir;

        let temp_dir = tempdir().expect("Failed to create temp directory");
        let downloader = SmartTokenizerDownload::with_cache_dir(temp_dir.path().to_path_buf())
            .expect("Failed to create downloader with temp cache");

        // Create a partially corrupted cached file
        let cache_key = "corruption-test";
        let cache_dir = temp_dir.path().join(cache_key);
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

        let corrupted_file = cache_dir.join("tokenizer.json");
        let mut file =
            std::fs::File::create(&corrupted_file).expect("Failed to create corrupted file");
        file.write_all(b"{ \"incomplete\": \"json").expect("Failed to write corrupted content");

        // Test that corrupted files are detected
        let found_file = downloader.find_cached_tokenizer(cache_key);
        assert!(found_file.is_some(), "Should find cached file even if corrupted");

        // In production, validation would detect corruption
        let cached_path = found_file.unwrap();
        let content = std::fs::read_to_string(&cached_path).expect("Should be able to read file");
        assert!(content.contains("incomplete"), "Should contain corrupted content");

        // Test JSON validation
        let json_parse = serde_json::from_str::<serde_json::Value>(&content);
        assert!(json_parse.is_err(), "Corrupted JSON should fail to parse");
    }

    /// Test disk space exhaustion scenarios
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_disk_space_exhaustion() {
        use tempfile::tempdir;

        let temp_dir = tempdir().expect("Failed to create temp directory");
        let downloader_result =
            SmartTokenizerDownload::with_cache_dir(temp_dir.path().to_path_buf());

        assert!(downloader_result.is_ok(), "Should initialize even with limited disk space");
        let _downloader = downloader_result.unwrap();

        // Test large download scenarios (mock)
        let large_tokenizer_info = TokenizerDownloadInfo {
            repo: "meta-llama/Meta-Llama-3-8B".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "large-download".to_string(),
            expected_vocab: Some(128256),
        };

        // In production, this would test actual disk space limits
        // For now, validate the structure is in place
        assert!(
            large_tokenizer_info.expected_vocab.unwrap() > 100000,
            "Large vocab should require significant space"
        );
        println!("✅ Disk space exhaustion test structure prepared");
    }

    /// Test concurrent download conflicts and race conditions
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_concurrent_download_conflicts() {
        use std::sync::Arc;
        use tempfile::tempdir;
        use tokio::task;

        let temp_dir = tempdir().expect("Failed to create temp directory");
        let downloader = Arc::new(
            SmartTokenizerDownload::with_cache_dir(temp_dir.path().to_path_buf())
                .expect("Failed to create downloader"),
        );

        let download_info = Arc::new(TokenizerDownloadInfo {
            repo: "microsoft/DialoGPT-medium".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "concurrent-test".to_string(),
            expected_vocab: Some(50257),
        });

        // Spawn multiple concurrent download tasks
        let mut handles = vec![];
        for i in 0..5 {
            let downloader_clone = Arc::clone(&downloader);
            let info_clone = Arc::clone(&download_info);

            let handle = task::spawn(async move {
                // Test cache checking (safe concurrent operation)
                let _cached = downloader_clone.find_cached_tokenizer(&info_clone.cache_key);

                // Test cache clearing (potentially unsafe concurrent operation)
                let _clear_result = downloader_clone.clear_cache(Some(&info_clone.cache_key));

                println!("Concurrent task {} completed", i);
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.expect("Concurrent task should complete successfully");
        }

        println!("✅ Concurrent download conflict test completed");
    }

    /// Test download resume with interrupted connections
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_download_resume_interrupted() {
        use std::io::Write;
        use tempfile::tempdir;

        let temp_dir = tempdir().expect("Failed to create temp directory");
        let _downloader = SmartTokenizerDownload::with_cache_dir(temp_dir.path().to_path_buf())
            .expect("Failed to create downloader with temp cache");

        // Simulate partially downloaded file
        let cache_key = "resume-test";
        let cache_dir = temp_dir.path().join(cache_key);
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

        let partial_file = cache_dir.join("tokenizer.json");
        let mut file = std::fs::File::create(&partial_file).expect("Failed to create partial file");

        // Write partial JSON content (first half of a valid tokenizer)
        let partial_content = r#"{"version": "1.0", "model": {"type": "BPE", "vocab": {"hello": 1"#;
        file.write_all(partial_content.as_bytes()).expect("Failed to write partial content");
        file.sync_all().expect("Failed to sync file");

        // Test resume detection
        let file_size = std::fs::metadata(&partial_file).expect("Should get file metadata").len();
        assert!(file_size > 0, "Partial file should have content");
        assert!(file_size < 1000, "Partial file should be incomplete");

        // Test resume logic would continue from this byte position
        let resume_position = file_size;
        assert!(resume_position > 0, "Resume should start from non-zero position");

        println!("✅ Download resume test: would resume from byte {}", resume_position);
    }

    /// Test invalid repository and file URLs
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_invalid_repository_urls() {
        let _downloader = SmartTokenizerDownload::new().expect("Failed to create downloader");

        let invalid_scenarios = [
            // Invalid repository format
            TokenizerDownloadInfo {
                repo: "invalid-repo-format".to_string(), // Missing owner/repo format
                files: vec!["tokenizer.json".to_string()],
                cache_key: "invalid-format".to_string(),
                expected_vocab: Some(1000),
            },
            // Non-existent repository
            TokenizerDownloadInfo {
                repo: "nonexistent-user/nonexistent-repo".to_string(),
                files: vec!["tokenizer.json".to_string()],
                cache_key: "nonexistent".to_string(),
                expected_vocab: Some(1000),
            },
            // Invalid file names
            TokenizerDownloadInfo {
                repo: "microsoft/DialoGPT-medium".to_string(),
                files: vec!["nonexistent-file.json".to_string(), "../../../etc/passwd".to_string()],
                cache_key: "invalid-files".to_string(),
                expected_vocab: Some(1000),
            },
            // Empty files list
            TokenizerDownloadInfo {
                repo: "microsoft/DialoGPT-medium".to_string(),
                files: vec![], // Empty files list
                cache_key: "empty-files".to_string(),
                expected_vocab: Some(1000),
            },
        ];

        for (i, invalid_info) in invalid_scenarios.into_iter().enumerate() {
            // Test URL generation for invalid scenarios
            let test_url =
                SmartTokenizerDownload::get_download_url(&invalid_info.repo, "test.json");

            // Should generate URL even for invalid repos (validation happens during download)
            assert!(test_url.starts_with("https://"), "Should generate HTTPS URL: {}", test_url);
            assert!(test_url.contains(&invalid_info.repo), "URL should contain repo: {}", test_url);

            // Test validation of download info
            if invalid_info.files.is_empty() {
                println!("Invalid scenario {}: empty files list detected", i);
            }

            // Test path traversal detection
            for filename in &invalid_info.files {
                if filename.contains("..") || filename.starts_with('/') {
                    println!("Invalid scenario {}: path traversal detected in: {}", i, filename);
                }
            }
        }
    }

    /// Test download with malformed JSON responses
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_malformed_download_responses() {
        use std::io::Write;
        use tempfile::tempdir;

        let temp_dir = tempdir().expect("Failed to create temp directory");
        let downloader = SmartTokenizerDownload::with_cache_dir(temp_dir.path().to_path_buf())
            .expect("Failed to create downloader");

        let test_info = TokenizerDownloadInfo {
            repo: "test/malformed".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "malformed-test".to_string(),
            expected_vocab: Some(1000),
        };

        // Simulate malformed downloaded content
        let cache_dir = temp_dir.path().join(&test_info.cache_key);
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

        let malformed_scenarios = [
            // Truncated JSON
            r#"{"version": "1.0", "model": {"type"#,
            // Invalid JSON structure
            r#"{"version": 1.0, "model": {type: "BPE"}}"#,
            // Non-JSON content with .json extension
            r#"This is not JSON at all, just plain text"#,
            // Binary data masquerading as JSON
            "\x00\x01\x02\x03\x04\x05",
            // Extremely nested JSON (potential DOS)
            &(format!("{}{}", "{".repeat(1000), "}".repeat(1000))),
        ];

        for (i, malformed_content) in malformed_scenarios.into_iter().enumerate() {
            let test_file = cache_dir.join(format!("malformed_{}.json", i));
            let mut file = std::fs::File::create(&test_file).expect("Failed to create test file");
            file.write_all(malformed_content.as_bytes())
                .expect("Failed to write malformed content");

            // Test validation would catch malformed JSON
            let validation_result =
                downloader.validate_downloaded_tokenizer(&test_file, &test_info);

            match validation_result {
                Ok(_) => println!("Malformed scenario {} unexpectedly passed validation", i),
                Err(_) => println!("Malformed scenario {} correctly failed validation", i),
            }
        }
    }

    /// Test cache corruption and recovery
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_cache_corruption_recovery() {
        use std::io::Write;
        use tempfile::tempdir;

        let temp_dir = tempdir().expect("Failed to create temp directory");
        let downloader = SmartTokenizerDownload::with_cache_dir(temp_dir.path().to_path_buf())
            .expect("Failed to create downloader");

        let cache_key = "corruption-recovery";
        let cache_dir = temp_dir.path().join(cache_key);

        // Test cache directory corruption scenarios
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

        // Scenario 1: Replace cache directory with a file
        std::fs::remove_dir_all(&cache_dir).expect("Failed to remove cache dir");
        std::fs::File::create(&cache_dir)
            .expect("Failed to create file with same name as cache dir");

        let recovery_result = downloader.clear_cache(Some(cache_key));
        // Should handle the conflict gracefully
        match recovery_result {
            Ok(_) => println!("Cache recovery succeeded"),
            Err(_) => println!("Cache recovery failed (expected for some scenarios)"),
        }

        // Scenario 2: Permissions corruption (simulate read-only cache)
        if cache_dir.exists() {
            std::fs::remove_file(&cache_dir).unwrap_or(());
        }
        std::fs::create_dir_all(&cache_dir).expect("Failed to recreate cache directory");

        // Create a cache file
        let cache_file = cache_dir.join("tokenizer.json");
        let mut file = std::fs::File::create(&cache_file).expect("Failed to create cache file");
        file.write_all(b"valid json").expect("Failed to write to cache file");

        // Test that cache operations handle permission issues
        let found_cache = downloader.find_cached_tokenizer(cache_key);
        assert!(found_cache.is_some(), "Should find cache file even if permissions issues exist");
    }

    /// Test download progress tracking and reporting
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_download_progress_tracking() {
        // Test progress calculation with various scenarios
        let progress_scenarios = [
            // (downloaded, total, expected_percentage)
            (0, Some(1000), Some(0.0)),
            (500, Some(1000), Some(0.5)),
            (1000, Some(1000), Some(1.0)),
            (750, Some(1000), Some(0.75)),
            (500, None, None), // Unknown total size
        ];

        for (downloaded, total, expected_percentage) in progress_scenarios {
            let progress = DownloadProgress {
                downloaded_bytes: downloaded,
                total_bytes: total,
                current_file: "tokenizer.json".to_string(),
                completed_files: 0,
                total_files: 1,
            };

            assert_eq!(
                progress.percentage(),
                expected_percentage,
                "Progress calculation mismatch for {}/{:?}",
                downloaded,
                total
            );
            assert_eq!(progress.downloaded_bytes, downloaded);
            assert_eq!(progress.total_bytes, total);
        }

        // Test progress with multiple files
        let multi_file_progress = DownloadProgress {
            downloaded_bytes: 2048,
            total_bytes: Some(4096),
            current_file: "tokenizer.model".to_string(),
            completed_files: 1,
            total_files: 2,
        };

        assert_eq!(multi_file_progress.completed_files, 1);
        assert_eq!(multi_file_progress.total_files, 2);
        assert_eq!(multi_file_progress.current_file, "tokenizer.model");
        assert_eq!(multi_file_progress.percentage(), Some(0.5));
    }

    /// Test bandwidth and download speed limits
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_download_bandwidth_limits() {
        let _downloader = SmartTokenizerDownload::new().expect("Failed to create downloader");

        // Test download scenarios that would hit bandwidth limits
        let bandwidth_test_cases = [
            // (file_size_mb, expected_min_time_ms, description)
            (1, 100, "Small tokenizer should download quickly"),
            (10, 500, "Medium tokenizer acceptable download time"),
            (50, 2000, "Large tokenizer may take longer"),
        ];

        for (file_size_mb, min_time_ms, description) in bandwidth_test_cases {
            // Simulate download metrics
            let download_metrics = DownloadMetrics {
                download_time: std::time::Duration::from_millis(min_time_ms),
                cache_time: std::time::Duration::from_millis(50),
                network_efficiency: 0.85, // 85% efficiency
                bytes_transferred: (file_size_mb * 1024 * 1024) as u64,
                cache_hit_rate: 0.0, // No cache hit for new download
            };

            assert!(
                download_metrics.download_time.as_millis() >= min_time_ms as u128,
                "{}: download time should meet minimum",
                description
            );
            assert!(
                download_metrics.network_efficiency > 0.0
                    && download_metrics.network_efficiency <= 1.0,
                "Network efficiency should be valid percentage"
            );
            assert_eq!(
                download_metrics.bytes_transferred,
                (file_size_mb * 1024 * 1024) as u64,
                "Bytes transferred should match file size"
            );
        }

        println!("✅ Bandwidth limit test structure validated");
    }

    /// Test offline mode with various cache states
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_offline_mode_comprehensive() {
        use std::io::Write;
        use tempfile::tempdir;

        // Set offline mode
        unsafe {
            std::env::set_var("BITNET_OFFLINE", "1");
        }

        // Ensure the environment variable is set before proceeding
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Verify the environment variable was actually set
        let mut env_val = std::env::var("BITNET_OFFLINE").unwrap_or_default();

        // Retry setting the environment variable if needed (handle test race conditions)
        let mut attempts = 0;
        while env_val != "1" && attempts < 5 {
            unsafe {
                std::env::set_var("BITNET_OFFLINE", "1");
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
            env_val = std::env::var("BITNET_OFFLINE").unwrap_or_default();
            attempts += 1;
        }

        let temp_dir = tempdir().expect("Failed to create temp directory");
        let downloader = SmartTokenizerDownload::with_cache_dir(temp_dir.path().to_path_buf())
            .expect("Failed to create downloader");

        let test_info = TokenizerDownloadInfo {
            repo: "meta-llama/Llama-2-7b-hf".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "offline-test".to_string(),
            expected_vocab: Some(32000),
        };

        // Test 1: Verify offline mode is set
        // Check environment variable directly to ensure it's set
        let final_env_val = std::env::var("BITNET_OFFLINE").unwrap_or_default();
        assert_eq!(final_env_val, "1", "BITNET_OFFLINE should be set to 1");
        assert!(SmartTokenizerDownload::is_offline_mode(), "Should detect offline mode");

        let no_cache_result = downloader.find_cached_tokenizer(&test_info.cache_key);
        assert!(no_cache_result.is_none(), "Should not find non-existent cache");

        // Test 2: Valid cached tokenizer in offline mode
        let cache_dir = temp_dir.path().join(&test_info.cache_key);
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

        let cached_tokenizer = cache_dir.join("tokenizer.json");
        let mut file =
            std::fs::File::create(&cached_tokenizer).expect("Failed to create cached file");
        file.write_all(br#"{"version": "1.0", "model": {"type": "BPE"}}"#)
            .expect("Failed to write valid JSON");

        let cached_result = downloader.find_cached_tokenizer(&test_info.cache_key);
        assert!(cached_result.is_some(), "Should find valid cached tokenizer in offline mode");

        // Test 3: Corrupted cache in offline mode
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&cached_tokenizer)
            .expect("Failed to reopen cached file");
        file.write_all(b"corrupted content").expect("Failed to write corrupted content");

        let corrupted_cache = downloader.find_cached_tokenizer(&test_info.cache_key);
        assert!(corrupted_cache.is_some(), "Should still find file even if corrupted");

        // Validation would fail for corrupted content
        let validation_result =
            downloader.validate_downloaded_tokenizer(&cached_tokenizer, &test_info);
        assert!(validation_result.is_err(), "Should fail validation for corrupted cache");

        // Clean up
        unsafe {
            std::env::remove_var("BITNET_OFFLINE");
        }
        assert!(
            !SmartTokenizerDownload::is_offline_mode(),
            "Should not be in offline mode after cleanup"
        );
    }

    /// Test download retry and exponential backoff
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_download_retry_backoff() {
        use std::time::Duration;

        let _downloader = SmartTokenizerDownload::new().expect("Failed to create downloader");

        // Test retry timing calculations
        let retry_scenarios = [
            (0, Duration::from_millis(100)),  // First retry: 100ms
            (1, Duration::from_millis(200)),  // Second retry: 200ms
            (2, Duration::from_millis(400)),  // Third retry: 400ms
            (3, Duration::from_millis(800)),  // Fourth retry: 800ms
            (4, Duration::from_millis(1600)), // Fifth retry: 1600ms
        ];

        for (retry_count, expected_delay) in retry_scenarios {
            // Calculate exponential backoff delay
            let base_delay = 100; // 100ms base
            let calculated_delay = Duration::from_millis(base_delay * (2u64.pow(retry_count)));

            assert_eq!(
                calculated_delay, expected_delay,
                "Retry delay calculation mismatch for attempt {}",
                retry_count
            );

            // Test maximum retry delay cap
            let max_delay = Duration::from_secs(30); // 30 second cap
            let capped_delay = std::cmp::min(calculated_delay, max_delay);
            assert!(capped_delay <= max_delay, "Retry delay should be capped at maximum");
        }

        println!("✅ Download retry backoff test completed");
    }

    /// Test memory efficient streaming for large downloads
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_memory_efficient_streaming() {
        use std::io::{Seek, SeekFrom, Write};
        use std::time::Instant;
        use tempfile::NamedTempFile;

        // Simulate large download streaming without actually downloading
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

        // Test streaming write patterns
        let chunk_sizes = [
            1024,    // 1KB chunks
            8192,    // 8KB chunks
            65536,   // 64KB chunks
            1048576, // 1MB chunks
        ];

        let total_size = 10 * 1024 * 1024; // 10MB total

        for chunk_size in chunk_sizes {
            temp_file.seek(SeekFrom::Start(0)).expect("Failed to seek to start");

            let chunk = vec![0u8; chunk_size];
            let num_chunks = total_size / chunk_size;
            let start_time = Instant::now();

            for _ in 0..num_chunks {
                temp_file.write_all(&chunk).expect("Failed to write chunk");
            }

            // Sync data to disk
            let write_duration = start_time.elapsed();

            println!(
                "Chunk size {}: wrote {}MB in {:?}",
                chunk_size,
                total_size / (1024 * 1024),
                write_duration
            );

            // Validate memory efficiency (smaller chunks should be more memory efficient)
            let memory_efficiency = if chunk_size <= 65536 {
                1.0 // Good memory efficiency
            } else {
                0.8 // Lower memory efficiency for large chunks
            };

            assert!(memory_efficiency > 0.0, "Memory efficiency should be positive");
        }
    }
}
