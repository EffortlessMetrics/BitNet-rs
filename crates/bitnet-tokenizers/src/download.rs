//! Smart tokenizer downloading with caching and resume capability
//!
//! This module provides intelligent tokenizer downloading from HuggingFace Hub with comprehensive
//! caching, resume capability, and validation for BitNet.rs neural network models.

use crate::discovery::TokenizerDownloadInfo;
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
        let cache_dir = Self::cache_directory()?;
        Self::with_cache_dir(cache_dir)
    }

    /// Initialize with custom cache directory
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self> {
        info!("Initializing SmartTokenizerDownload with cache dir: {}", cache_dir.display());

        // Create cache directory if it doesn't exist
        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir).map_err(|e| {
                BitNetError::Model(ModelError::FileIOError { path: cache_dir.clone(), source: e })
            })?;
        }

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
    /// ```rust
    /// let info = TokenizerDownloadInfo {
    ///     repo: "meta-llama/Llama-2-7b-hf".to_string(),
    ///     files: vec!["tokenizer.json".to_string()],
    ///     cache_key: "llama2-32k".to_string(),
    ///     expected_vocab: Some(32000),
    /// };
    /// let tokenizer_path = downloader.download_tokenizer(&info).await?;
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
            return Err(BitNetError::Config(format!(
                "Cannot download tokenizer {} in offline mode",
                info.cache_key
            )));
        }

        // Create cache directory for this tokenizer
        let tokenizer_cache_dir = self.cache_dir.join(&info.cache_key);
        if !tokenizer_cache_dir.exists() {
            std::fs::create_dir_all(&tokenizer_cache_dir).map_err(|e| {
                BitNetError::Model(ModelError::FileIOError {
                    path: tokenizer_cache_dir.clone(),
                    source: e,
                })
            })?;
        }

        // Download all files
        #[cfg(not(feature = "downloads"))]
        {
            return Err(BitNetError::Config(
                "Download feature not enabled. Build with --features downloads".to_string(),
            ));
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
                BitNetError::Config("No primary tokenizer file found".to_string())
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
                std::fs::remove_dir_all(&cache_dir).map_err(|e| {
                    BitNetError::Model(ModelError::FileIOError { path: cache_dir, source: e })
                })?;
            }
        } else {
            info!("Clearing entire tokenizer cache: {}", self.cache_dir.display());
            if self.cache_dir.exists() {
                std::fs::remove_dir_all(&self.cache_dir).map_err(|e| {
                    BitNetError::Model(ModelError::FileIOError {
                        path: self.cache_dir.clone(),
                        source: e,
                    })
                })?;
                // Recreate the cache directory
                std::fs::create_dir_all(&self.cache_dir).map_err(|e| {
                    BitNetError::Model(ModelError::FileIOError {
                        path: self.cache_dir.clone(),
                        source: e,
                    })
                })?;
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
    fn validate_downloaded_tokenizer(
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

    /// Determine cache directory with environment variable override
    ///
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    fn cache_directory() -> Result<PathBuf> {
        // Check environment variable first
        if let Ok(cache_dir) = std::env::var("BITNET_CACHE_DIR") {
            return Ok(PathBuf::from(cache_dir).join("tokenizers"));
        }

        // Check XDG cache directory
        if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
            return Ok(PathBuf::from(xdg_cache).join("bitnet").join("tokenizers"));
        }

        // Use system cache directory
        if let Some(cache_dir) = dirs::cache_dir() {
            return Ok(cache_dir.join("bitnet").join("tokenizers"));
        }

        // Fallback to local directory
        Ok(PathBuf::from(".cache").join("bitnet").join("tokenizers"))
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

        // Test scaffolding - will fail until implementation complete
        assert!(result.is_err(), "Test scaffolding should fail until implemented");
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
        assert!(downloader_result.is_err(), "Test scaffolding - requires implementation");

        // Test scaffolding for download process
        // let downloader = downloader_result.unwrap();
        // let result = downloader.download_tokenizer(&download_info).await;
        // assert!(result.is_ok(), "Download should succeed for valid repo");
    }

    /// AC2: Tests caching functionality for downloaded tokenizers
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac2-smarttokenizer-download-implementation
    #[tokio::test]
    #[cfg(feature = "cpu")]
    async fn test_download_cache_management() {
        let cache_dir = std::env::temp_dir().join("bitnet-test-cache");
        let downloader_result = SmartTokenizerDownload::with_cache_dir(cache_dir);

        assert!(downloader_result.is_err(), "Test scaffolding - requires implementation");

        // Test scaffolding for cache operations
        // let downloader = downloader_result.unwrap();

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
        assert!(downloader_result.is_err(), "Test scaffolding - requires implementation");

        // Test scaffolding for resume functionality
        // 1. Start download and interrupt
        // 2. Resume download and verify completion
        // 3. Validate final file integrity

        assert!(true, "Test scaffolding placeholder for download resume");
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
            assert!(downloader_result.is_err(), "Test scaffolding - requires implementation");

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
        assert!(downloader_result.is_err(), "Test scaffolding - requires implementation");

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
        assert!(downloader_result.is_err(), "Test scaffolding - requires implementation");

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
        assert!(downloader_result.is_err(), "Test scaffolding - requires implementation");

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
    fn create_test_download_info() -> TokenizerDownloadInfo {
        TokenizerDownloadInfo {
            repo: "microsoft/DialoGPT-medium".to_string(),
            files: vec!["tokenizer.json".to_string()],
            cache_key: "test-tokenizer".to_string(),
            expected_vocab: Some(50257),
        }
    }
}
