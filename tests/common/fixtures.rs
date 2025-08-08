use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tokio::fs;
use tracing::{debug, info, warn};

use crate::{
    config::FixtureConfig,
    errors::{FixtureError, FixtureResult, TestResult},
    utils::format_bytes,
};

/// Manages test fixtures including models, datasets, and other test data
pub struct FixtureManager {
    config: FixtureConfig,
    cache_dir: PathBuf,
    fixtures: HashMap<String, FixtureInfo>,
    downloads: HashMap<String, DownloadInfo>,
}

impl FixtureManager {
    /// Create a new fixture manager with the given configuration
    pub async fn new(config: &FixtureConfig) -> TestResult<Self> {
        let cache_dir = std::env::var("BITNET_TEST_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("tests/cache"));

        // Create cache directory if it doesn't exist
        fs::create_dir_all(&cache_dir).await?;

        info!(
            "Initializing fixture manager with cache dir: {:?}",
            cache_dir
        );

        let mut manager = Self {
            config: config.clone(),
            cache_dir,
            fixtures: HashMap::new(),
            downloads: HashMap::new(),
        };

        // Load built-in fixtures
        manager.load_builtin_fixtures().await?;

        // Load custom fixtures from config
        manager.load_custom_fixtures().await?;

        // Perform initial cleanup if needed
        manager.cleanup_old_fixtures().await?;

        Ok(manager)
    }

    /// Get a model fixture by name, downloading if necessary
    pub async fn get_model_fixture(&self, name: &str) -> FixtureResult<PathBuf> {
        debug!("Requesting model fixture: {}", name);

        if let Some(fixture) = self.fixtures.get(name) {
            let path = self.cache_dir.join(&fixture.filename);

            // Check if file exists and has correct checksum
            if path.exists() {
                if self.verify_checksum(&path, &fixture.checksum).await? {
                    debug!("Found cached fixture: {} at {:?}", name, path);
                    return Ok(path);
                } else {
                    warn!("Checksum mismatch for cached fixture: {}", name);
                    // Remove corrupted file
                    let _ = fs::remove_file(&path).await;
                }
            }
        }

        // Download if not cached or checksum mismatch
        if self.config.auto_download {
            self.download_fixture(name).await
        } else {
            Err(FixtureError::not_found(format!(
                "Fixture '{}' not found and auto_download is disabled",
                name
            )))
        }
    }

    /// Get a dataset fixture by name
    pub async fn get_dataset_fixture(&self, name: &str) -> FixtureResult<PathBuf> {
        // For now, treat datasets the same as models
        // In the future, we might want different handling
        self.get_model_fixture(name).await
    }

    /// Get fixture information without downloading
    pub fn get_fixture_info(&self, name: &str) -> Option<&FixtureInfo> {
        self.fixtures.get(name)
    }

    /// List all available fixtures
    pub fn list_fixtures(&self) -> Vec<&FixtureInfo> {
        self.fixtures.values().collect()
    }

    /// Check if a fixture is cached locally
    pub async fn is_cached(&self, name: &str) -> bool {
        if let Some(fixture) = self.fixtures.get(name) {
            let path = self.cache_dir.join(&fixture.filename);
            if path.exists() {
                return self
                    .verify_checksum(&path, &fixture.checksum)
                    .await
                    .unwrap_or(false);
            }
        }
        false
    }

    /// Get shared fixture that can be used across multiple test suites
    /// This method ensures the fixture is available and returns a reference that can be shared
    pub async fn get_shared_fixture(&self, name: &str) -> FixtureResult<SharedFixture> {
        let path = self.get_model_fixture(name).await?;
        let info = self
            .fixtures
            .get(name)
            .ok_or_else(|| FixtureError::unknown(name))?;

        Ok(SharedFixture {
            name: name.to_string(),
            path,
            info: info.clone(),
            reference_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(1)),
        })
    }

    /// Register a fixture for shared use across test suites
    pub async fn register_shared_fixture(
        &mut self,
        name: &str,
        url: &str,
        checksum: &str,
    ) -> FixtureResult<()> {
        let filename = url.split('/').last().unwrap_or(name).to_string();

        let fixture_info = FixtureInfo {
            name: name.to_string(),
            filename: filename.clone(),
            checksum: checksum.to_string(),
            size: 0, // Will be determined on download
            description: format!("Shared fixture: {}", name),
            model_type: ModelType::Unknown,
            format: ModelFormat::Unknown,
        };

        let download_info = DownloadInfo {
            url: url.to_string(),
            filename,
            checksum: checksum.to_string(),
        };

        self.fixtures.insert(name.to_string(), fixture_info);
        self.downloads.insert(name.to_string(), download_info);

        debug!("Registered shared fixture: {}", name);
        Ok(())
    }

    /// Preload fixtures for better performance
    pub async fn preload_fixtures(&self, names: &[&str]) -> FixtureResult<Vec<PathBuf>> {
        let mut paths = Vec::new();

        for name in names {
            match self.get_model_fixture(name).await {
                Ok(path) => {
                    paths.push(path);
                    debug!("Preloaded fixture: {}", name);
                }
                Err(e) => {
                    warn!("Failed to preload fixture {}: {}", name, e);
                    return Err(e);
                }
            }
        }

        info!("Successfully preloaded {} fixtures", paths.len());
        Ok(paths)
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> FixtureResult<CacheStats> {
        let mut total_size = 0u64;
        let mut file_count = 0usize;
        let mut oldest_file: Option<SystemTime> = None;
        let mut newest_file: Option<SystemTime> = None;

        let mut entries = fs::read_dir(&self.cache_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_file() {
                let metadata = entry.metadata().await?;
                total_size += metadata.len();
                file_count += 1;

                if let Ok(modified) = metadata.modified() {
                    match oldest_file {
                        None => oldest_file = Some(modified),
                        Some(current_oldest) if modified < current_oldest => {
                            oldest_file = Some(modified)
                        }
                        _ => {}
                    }

                    match newest_file {
                        None => newest_file = Some(modified),
                        Some(current_newest) if modified > current_newest => {
                            newest_file = Some(modified)
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(CacheStats {
            total_size,
            file_count,
            oldest_file,
            newest_file,
            cache_dir: self.cache_dir.clone(),
        })
    }

    /// Clean up old fixtures based on configuration
    pub async fn cleanup_old_fixtures(&self) -> FixtureResult<CleanupStats> {
        let cutoff = SystemTime::now() - self.config.cleanup_interval;
        let mut removed_count = 0usize;
        let mut removed_size = 0u64;

        debug!(
            "Cleaning up fixtures older than {:?}",
            self.config.cleanup_interval
        );

        let mut entries = fs::read_dir(&self.cache_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_file() {
                let metadata = entry.metadata().await?;
                if let Ok(modified) = metadata.modified() {
                    if modified < cutoff {
                        let file_size = metadata.len();
                        match fs::remove_file(entry.path()).await {
                            Ok(_) => {
                                removed_count += 1;
                                removed_size += file_size;
                                debug!("Removed old fixture: {:?}", entry.path());
                            }
                            Err(e) => {
                                warn!("Failed to remove old fixture {:?}: {}", entry.path(), e);
                            }
                        }
                    }
                }
            }
        }

        if removed_count > 0 {
            info!(
                "Cleaned up {} old fixtures, freed {}",
                removed_count,
                format_bytes(removed_size)
            );
        }

        Ok(CleanupStats {
            removed_count,
            removed_size,
        })
    }

    /// Clean up fixtures based on cache size limit
    pub async fn cleanup_by_size(&self) -> FixtureResult<CleanupStats> {
        if self.config.max_cache_size == 0 {
            return Ok(CleanupStats {
                removed_count: 0,
                removed_size: 0,
            });
        }

        let stats = self.get_cache_stats().await?;
        if !stats.is_over_limit(self.config.max_cache_size) {
            return Ok(CleanupStats {
                removed_count: 0,
                removed_size: 0,
            });
        }

        debug!("Cache size limit exceeded, cleaning up oldest files");

        // Collect files with their modification times
        let mut files_with_times = Vec::new();
        let mut entries = fs::read_dir(&self.cache_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_file() {
                let metadata = entry.metadata().await?;
                if let Ok(modified) = metadata.modified() {
                    files_with_times.push((entry.path(), modified, metadata.len()));
                }
            }
        }

        // Sort by modification time (oldest first)
        files_with_times.sort_by_key(|(_, time, _)| *time);

        let mut removed_count = 0;
        let mut removed_size = 0;
        let mut current_size = stats.total_size;
        let target_size = (self.config.max_cache_size as f64 * 0.8) as u64; // Clean to 80% of limit

        for (path, _, size) in files_with_times {
            if current_size <= target_size {
                break;
            }

            match fs::remove_file(&path).await {
                Ok(_) => {
                    removed_count += 1;
                    removed_size += size;
                    current_size -= size;
                    debug!("Removed file for size cleanup: {:?}", path);
                }
                Err(e) => {
                    warn!("Failed to remove file {:?}: {}", path, e);
                }
            }
        }

        if removed_count > 0 {
            info!(
                "Size-based cleanup removed {} files, freed {}",
                removed_count,
                format_bytes(removed_size)
            );
        }

        Ok(CleanupStats {
            removed_count,
            removed_size,
        })
    }

    /// Validate all cached fixtures
    pub async fn validate_cache(&self) -> FixtureResult<ValidationStats> {
        let mut valid_count = 0;
        let mut invalid_count = 0;
        let mut missing_count = 0;
        let mut invalid_files = Vec::new();

        for (name, fixture) in &self.fixtures {
            let path = self.cache_dir.join(&fixture.filename);

            if !path.exists() {
                missing_count += 1;
                continue;
            }

            match self.verify_checksum(&path, &fixture.checksum).await {
                Ok(true) => {
                    valid_count += 1;
                }
                Ok(false) => {
                    invalid_count += 1;
                    invalid_files.push(name.clone());
                    warn!("Invalid checksum for fixture: {}", name);
                }
                Err(e) => {
                    invalid_count += 1;
                    invalid_files.push(name.clone());
                    warn!("Failed to validate fixture {}: {}", name, e);
                }
            }
        }

        Ok(ValidationStats {
            valid_count,
            invalid_count,
            missing_count,
            invalid_files,
        })
    }

    /// Remove invalid fixtures from cache
    pub async fn remove_invalid_fixtures(&self) -> FixtureResult<CleanupStats> {
        let validation = self.validate_cache().await?;
        let mut removed_count = 0;
        let mut removed_size = 0;

        for name in &validation.invalid_files {
            if let Some(fixture) = self.fixtures.get(name) {
                let path = self.cache_dir.join(&fixture.filename);
                if path.exists() {
                    match fs::metadata(&path).await {
                        Ok(metadata) => match fs::remove_file(&path).await {
                            Ok(_) => {
                                removed_count += 1;
                                removed_size += metadata.len();
                                info!("Removed invalid fixture: {}", name);
                            }
                            Err(e) => {
                                warn!("Failed to remove invalid fixture {}: {}", name, e);
                            }
                        },
                        Err(e) => {
                            warn!("Failed to get metadata for {}: {}", name, e);
                        }
                    }
                }
            }
        }

        Ok(CleanupStats {
            removed_count,
            removed_size,
        })
    }

    /// Download a fixture from its configured source
    async fn download_fixture(&self, name: &str) -> FixtureResult<PathBuf> {
        let download_info = self
            .downloads
            .get(name)
            .ok_or_else(|| FixtureError::unknown(name))?;

        let target_path = self.cache_dir.join(&download_info.filename);
        let temp_path = self
            .cache_dir
            .join(format!("{}.tmp", &download_info.filename));

        info!("Downloading fixture '{}' from {}", name, download_info.url);

        // Create HTTP client with timeout and retry logic
        let client = reqwest::Client::builder()
            .timeout(self.config.download_timeout)
            .user_agent("BitNet.rs-TestFramework/0.1.0")
            .build()
            .map_err(|e| FixtureError::download(&download_info.url, e.to_string()))?;

        // Retry download up to 3 times
        let mut last_error = None;
        for attempt in 1..=3 {
            match self
                .attempt_download(&client, download_info, &temp_path)
                .await
            {
                Ok(bytes) => {
                    // Verify checksum
                    let mut hasher = Sha256::new();
                    hasher.update(&bytes);
                    let hash = format!("{:x}", hasher.finalize());

                    if hash != download_info.checksum {
                        // Clean up temp file
                        let _ = fs::remove_file(&temp_path).await;
                        return Err(FixtureError::checksum_mismatch(
                            &download_info.filename,
                            &download_info.checksum,
                            &hash,
                        ));
                    }

                    // Move temp file to final location atomically
                    fs::rename(&temp_path, &target_path).await.map_err(|e| {
                        FixtureError::cache(format!("Failed to move downloaded file: {}", e))
                    })?;

                    info!(
                        "Successfully downloaded fixture '{}' to {:?} (attempt {})",
                        name, target_path, attempt
                    );

                    return Ok(target_path);
                }
                Err(e) => {
                    warn!("Download attempt {} failed for '{}': {}", attempt, name, e);
                    last_error = Some(e);

                    // Clean up temp file if it exists
                    let _ = fs::remove_file(&temp_path).await;

                    if attempt < 3 {
                        // Wait before retry with exponential backoff
                        let delay = Duration::from_secs(2_u64.pow(attempt - 1));
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            FixtureError::download(
                &download_info.url,
                "All download attempts failed".to_string(),
            )
        }))
    }

    /// Attempt to download a fixture once
    async fn attempt_download(
        &self,
        client: &reqwest::Client,
        download_info: &DownloadInfo,
        temp_path: &Path,
    ) -> FixtureResult<Vec<u8>> {
        let response = client
            .get(&download_info.url)
            .send()
            .await
            .map_err(|e| FixtureError::download(&download_info.url, e.to_string()))?;

        if !response.status().is_success() {
            return Err(FixtureError::download(
                &download_info.url,
                format!("HTTP {}", response.status()),
            ));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| FixtureError::download(&download_info.url, e.to_string()))?;

        // Write to temp file first
        fs::write(temp_path, &bytes)
            .await
            .map_err(|e| FixtureError::cache(format!("Failed to write temp file: {}", e)))?;

        Ok(bytes.to_vec())
    }

    /// Verify file checksum
    async fn verify_checksum(&self, path: &Path, expected: &str) -> FixtureResult<bool> {
        let bytes = fs::read(path).await.map_err(|e| {
            FixtureError::validation(format!("Failed to read {}: {}", path.display(), e))
        })?;

        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());

        Ok(hash == expected)
    }

    /// Load built-in fixture definitions
    async fn load_builtin_fixtures(&mut self) -> TestResult<()> {
        // Define built-in test fixtures
        let builtin_fixtures = vec![
            FixtureInfo {
                name: "tiny-model".to_string(),
                filename: "tiny-bitnet.gguf".to_string(),
                checksum: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                    .to_string(), // Empty file for testing
                size: 1024,
                description: "Tiny BitNet model for basic testing".to_string(),
                model_type: ModelType::BitNet,
                format: ModelFormat::Gguf,
            },
            FixtureInfo {
                name: "small-model".to_string(),
                filename: "small-bitnet.gguf".to_string(),
                checksum: "d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35"
                    .to_string(), // "hello" hash for testing
                size: 1024 * 1024, // 1MB
                description: "Small BitNet model for integration testing".to_string(),
                model_type: ModelType::BitNet,
                format: ModelFormat::Gguf,
            },
            FixtureInfo {
                name: "test-dataset".to_string(),
                filename: "test-prompts.json".to_string(),
                checksum: "aec070645fe53ee3b3763059376134f058cc337247c978add178b6ccdfb0019f"
                    .to_string(), // "world" hash for testing
                size: 4096,
                description: "Test prompts for validation".to_string(),
                model_type: ModelType::Dataset,
                format: ModelFormat::Json,
            },
        ];

        // Add corresponding download info (using placeholder URLs for now)
        let builtin_downloads = vec![
            DownloadInfo {
                url: "https://example.com/fixtures/tiny-bitnet.gguf".to_string(),
                filename: "tiny-bitnet.gguf".to_string(),
                checksum: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                    .to_string(),
            },
            DownloadInfo {
                url: "https://example.com/fixtures/small-bitnet.gguf".to_string(),
                filename: "small-bitnet.gguf".to_string(),
                checksum: "d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35"
                    .to_string(),
            },
            DownloadInfo {
                url: "https://example.com/fixtures/test-prompts.json".to_string(),
                filename: "test-prompts.json".to_string(),
                checksum: "aec070645fe53ee3b3763059376134f058cc337247c978add178b6ccdfb0019f"
                    .to_string(),
            },
        ];

        // Register fixtures
        for fixture in builtin_fixtures {
            self.fixtures.insert(fixture.name.clone(), fixture);
        }

        for download in builtin_downloads {
            self.downloads.insert(
                download
                    .filename
                    .strip_suffix(".gguf")
                    .or_else(|| download.filename.strip_suffix(".json"))
                    .unwrap_or(&download.filename)
                    .to_string(),
                download,
            );
        }

        debug!("Loaded {} built-in fixtures", self.fixtures.len());
        Ok(())
    }

    /// Load custom fixtures from configuration
    async fn load_custom_fixtures(&mut self) -> TestResult<()> {
        for custom_fixture in &self.config.custom_fixtures {
            let fixture_info = FixtureInfo {
                name: custom_fixture.name.clone(),
                filename: custom_fixture
                    .url
                    .split('/')
                    .last()
                    .unwrap_or(&custom_fixture.name)
                    .to_string(),
                checksum: custom_fixture.checksum.clone(),
                size: 0, // Unknown size for custom fixtures
                description: custom_fixture
                    .description
                    .clone()
                    .unwrap_or_else(|| format!("Custom fixture: {}", custom_fixture.name)),
                model_type: ModelType::Unknown,
                format: ModelFormat::Unknown,
            };

            let download_info = DownloadInfo {
                url: custom_fixture.url.clone(),
                filename: fixture_info.filename.clone(),
                checksum: custom_fixture.checksum.clone(),
            };

            self.fixtures
                .insert(custom_fixture.name.clone(), fixture_info);
            self.downloads
                .insert(custom_fixture.name.clone(), download_info);
        }

        if !self.config.custom_fixtures.is_empty() {
            debug!(
                "Loaded {} custom fixtures",
                self.config.custom_fixtures.len()
            );
        }

        Ok(())
    }
}

/// Information about a test fixture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureInfo {
    pub name: String,
    pub filename: String,
    pub checksum: String,
    pub size: u64,
    pub description: String,
    pub model_type: ModelType,
    pub format: ModelFormat,
}

/// Download information for a fixture
#[derive(Debug, Clone)]
pub struct DownloadInfo {
    pub url: String,
    pub filename: String,
    pub checksum: String,
}

/// Type of model or data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    BitNet,
    Transformer,
    Dataset,
    Unknown,
}

/// Format of the model or data file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
    Json,
    Csv,
    Unknown,
}

/// Statistics about the fixture cache
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_size: u64,
    pub file_count: usize,
    pub oldest_file: Option<SystemTime>,
    pub newest_file: Option<SystemTime>,
    pub cache_dir: PathBuf,
}

impl CacheStats {
    /// Get cache utilization as a percentage (if max size is configured)
    pub fn utilization_percent(&self, max_size: u64) -> Option<f64> {
        if max_size > 0 {
            Some((self.total_size as f64 / max_size as f64) * 100.0)
        } else {
            None
        }
    }

    /// Check if cache is over the size limit
    pub fn is_over_limit(&self, max_size: u64) -> bool {
        max_size > 0 && self.total_size > max_size
    }
}

/// Statistics about cache cleanup
#[derive(Debug, Clone)]
pub struct CleanupStats {
    pub removed_count: usize,
    pub removed_size: u64,
}

/// A shared fixture that can be used across multiple test suites
#[derive(Debug, Clone)]
pub struct SharedFixture {
    pub name: String,
    pub path: PathBuf,
    pub info: FixtureInfo,
    pub reference_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl SharedFixture {
    /// Increment reference count
    pub fn add_ref(&self) {
        self.reference_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Decrement reference count and return current count
    pub fn release(&self) -> usize {
        self.reference_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed)
            - 1
    }

    /// Get current reference count
    pub fn ref_count(&self) -> usize {
        self.reference_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if this is the last reference
    pub fn is_last_ref(&self) -> bool {
        self.ref_count() <= 1
    }
}

/// Statistics about fixture validation
#[derive(Debug, Clone)]
pub struct ValidationStats {
    pub valid_count: usize,
    pub invalid_count: usize,
    pub missing_count: usize,
    pub invalid_files: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FixtureConfig;
    use tempfile::TempDir;

    async fn create_test_fixture_manager() -> (FixtureManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = FixtureConfig::default();
        config.auto_download = false; // Disable auto-download for tests

        // Override cache dir with temp dir
        std::env::set_var("BITNET_TEST_CACHE", temp_dir.path());

        let manager = FixtureManager::new(&config).await.unwrap();
        (manager, temp_dir)
    }

    #[tokio::test]
    async fn test_fixture_manager_creation() {
        let (manager, _temp_dir) = create_test_fixture_manager().await;

        // Should have built-in fixtures
        assert!(!manager.fixtures.is_empty());
        assert!(manager.get_fixture_info("tiny-model").is_some());
        assert!(manager.get_fixture_info("small-model").is_some());
        assert!(manager.get_fixture_info("test-dataset").is_some());
    }

    #[tokio::test]
    async fn test_list_fixtures() {
        let (manager, _temp_dir) = create_test_fixture_manager().await;

        let fixtures = manager.list_fixtures();
        assert!(fixtures.len() >= 3); // At least the built-in fixtures

        let names: Vec<&str> = fixtures.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"tiny-model"));
        assert!(names.contains(&"small-model"));
        assert!(names.contains(&"test-dataset"));
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let (manager, temp_dir) = create_test_fixture_manager().await;

        // Create a test file in cache
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, b"test content").await.unwrap();

        let stats = manager.get_cache_stats().await.unwrap();
        assert_eq!(stats.file_count, 1);
        assert_eq!(stats.total_size, 12); // "test content" is 12 bytes
        assert!(stats.oldest_file.is_some());
        assert!(stats.newest_file.is_some());
    }

    #[tokio::test]
    async fn test_is_cached() {
        let (manager, _temp_dir) = create_test_fixture_manager().await;

        // Should not be cached initially
        assert!(!manager.is_cached("tiny-model").await);
        assert!(!manager.is_cached("nonexistent").await);
    }

    #[tokio::test]
    async fn test_cleanup_old_fixtures() {
        let (manager, temp_dir) = create_test_fixture_manager().await;

        // Create an old test file
        let old_file = temp_dir.path().join("old.txt");
        fs::write(&old_file, b"old content").await.unwrap();

        // Set file time to be old (this is platform-specific and might not work in all test environments)
        // For now, just test that cleanup runs without error
        let stats = manager.cleanup_old_fixtures().await.unwrap();
        assert_eq!(stats.removed_count, 0); // File is not old enough yet
    }

    #[tokio::test]
    async fn test_get_fixture_info() {
        let (manager, _temp_dir) = create_test_fixture_manager().await;

        let info = manager.get_fixture_info("tiny-model").unwrap();
        assert_eq!(info.name, "tiny-model");
        assert_eq!(info.filename, "tiny-bitnet.gguf");
        assert!(!info.checksum.is_empty());
        assert!(!info.description.is_empty());

        assert!(manager.get_fixture_info("nonexistent").is_none());
    }
}
