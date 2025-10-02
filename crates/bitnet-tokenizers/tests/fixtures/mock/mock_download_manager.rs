//! Mock Download Manager for Testing Smart Tokenizer Downloads
//!
//! Provides mock download functionality for testing without network access.

#![cfg(test)]
#![cfg(feature = "cpu")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Mock download manager for testing AC4 smart download integration
#[derive(Debug, Clone)]
pub struct MockDownloadManager {
    /// Simulated cache of downloaded files
    pub cache: HashMap<String, PathBuf>,
    /// Simulate network failures for testing
    pub simulate_network_failure: bool,
    /// Simulate download delays (milliseconds)
    pub simulated_delay_ms: u64,
}

impl MockDownloadManager {
    /// Create a new mock download manager
    pub fn new() -> Self {
        Self { cache: HashMap::new(), simulate_network_failure: false, simulated_delay_ms: 0 }
    }

    /// Simulate downloading a tokenizer file
    pub async fn download_tokenizer(
        &mut self,
        repo: &str,
        file: &str,
        cache_dir: &Path,
    ) -> Result<PathBuf, MockDownloadError> {
        // Simulate network delay
        if self.simulated_delay_ms > 0 {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.simulated_delay_ms)).await;
        }

        // Simulate network failure
        if self.simulate_network_failure {
            return Err(MockDownloadError::NetworkError(format!(
                "Failed to download {} from {}",
                file, repo
            )));
        }

        // Check cache
        let cache_key = format!("{}/{}", repo, file);
        if let Some(cached_path) = self.cache.get(&cache_key) {
            return Ok(cached_path.clone());
        }

        // Simulate successful download
        let download_path = cache_dir.join(repo.replace('/', "_")).join(file);

        // Create mock file
        if let Some(parent) = download_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                MockDownloadError::IoError(format!("Failed to create cache directory: {}", e))
            })?;
        }

        // Write mock tokenizer data
        let mock_data = self.generate_mock_tokenizer_data(repo, file);
        std::fs::write(&download_path, mock_data)
            .map_err(|e| MockDownloadError::IoError(format!("Failed to write mock file: {}", e)))?;

        // Add to cache
        self.cache.insert(cache_key, download_path.clone());

        Ok(download_path)
    }

    /// Generate mock tokenizer data based on repo and file
    fn generate_mock_tokenizer_data(&self, repo: &str, file: &str) -> Vec<u8> {
        if file.ends_with(".json") {
            // Mock HuggingFace tokenizer JSON
            let vocab_size = if repo.contains("llama-3") || repo.contains("Llama-3") {
                128256
            } else if repo.contains("llama") || repo.contains("Llama") {
                32000
            } else if repo.contains("gpt2") {
                50257
            } else {
                32000
            };

            serde_json::to_vec(&serde_json::json!({
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab": (0..vocab_size).map(|i| (format!("token_{}", i), i)).collect::<HashMap<String, usize>>()
                },
                "added_tokens": []
            }))
            .unwrap_or_default()
        } else if file.ends_with(".model") {
            // Mock SentencePiece model
            b"SPM_MOCK_DATA".to_vec()
        } else {
            // Unknown file type
            b"MOCK_DATA".to_vec()
        }
    }

    /// Set network failure simulation
    pub fn set_network_failure(&mut self, fail: bool) {
        self.simulate_network_failure = fail;
    }

    /// Set download delay simulation
    pub fn set_delay(&mut self, delay_ms: u64) {
        self.simulated_delay_ms = delay_ms;
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for MockDownloadManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock download errors
#[derive(Debug, Clone)]
pub enum MockDownloadError {
    NetworkError(String),
    IoError(String),
    InvalidRepo(String),
}

impl std::fmt::Display for MockDownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MockDownloadError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            MockDownloadError::IoError(msg) => write!(f, "IO error: {}", msg),
            MockDownloadError::InvalidRepo(msg) => write!(f, "Invalid repository: {}", msg),
        }
    }
}

impl std::error::Error for MockDownloadError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_mock_download_success() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = MockDownloadManager::new();

        let result = manager
            .download_tokenizer("meta-llama/Llama-2-7b-hf", "tokenizer.json", temp_dir.path())
            .await;

        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.exists());
    }

    #[tokio::test]
    async fn test_mock_download_cache() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = MockDownloadManager::new();

        // First download
        let path1 = manager
            .download_tokenizer("meta-llama/Llama-2-7b-hf", "tokenizer.json", temp_dir.path())
            .await
            .unwrap();

        // Second download (should use cache)
        let path2 = manager
            .download_tokenizer("meta-llama/Llama-2-7b-hf", "tokenizer.json", temp_dir.path())
            .await
            .unwrap();

        assert_eq!(path1, path2);
        assert_eq!(manager.cache.len(), 1);
    }

    #[tokio::test]
    async fn test_mock_download_network_failure() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = MockDownloadManager::new();
        manager.set_network_failure(true);

        let result = manager
            .download_tokenizer("meta-llama/Llama-2-7b-hf", "tokenizer.json", temp_dir.path())
            .await;

        assert!(result.is_err());
    }
}
