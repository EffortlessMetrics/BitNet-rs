use super::errors::{TestError, TestResult};
use super::results::TestResult;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::fs;
use tracing::{debug, info, warn};

/// Test result cache for storing and retrieving test results
pub struct TestCache {
    cache_dir: PathBuf,
    config: CacheConfig,
    metadata: CacheMetadata,
}

/// Configuration for test caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable test result caching
    pub enabled: bool,
    /// Maximum age of cached results in seconds
    pub max_age_seconds: u64,
    /// Maximum cache size in bytes
    pub max_size_bytes: u64,
    /// Enable incremental testing
    pub incremental_testing: bool,
    /// Enable smart test selection
    pub smart_selection: bool,
    /// Cache compression level (0-9)
    pub compression_level: u32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_age_seconds: 24 * 60 * 60,      // 24 hours
            max_size_bytes: 1024 * 1024 * 1024, // 1 GB
            incremental_testing: true,
            smart_selection: true,
            compression_level: 6,
        }
    }
}

/// Metadata about the test cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Version of the cache format
    pub version: String,
    /// Last cleanup time
    pub last_cleanup: SystemTime,
    /// Total cached results
    pub total_results: usize,
    /// Cache statistics
    pub stats: CacheStats,
}

impl Default for CacheMetadata {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            last_cleanup: UNIX_EPOCH,
            total_results: 0,
            stats: CacheStats::default(),
        }
    }
}

/// Statistics about cache usage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size_bytes: u64,
    pub entry_count: usize,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Cached test result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedTestResult {
    /// The actual test result
    pub result: TestResultData,
    /// Hash of test inputs/configuration
    pub input_hash: String,
    /// Hash of source code that affects this test
    pub source_hash: String,
    /// When this result was cached
    pub cached_at: SystemTime,
    /// Dependencies that affect this test
    pub dependencies: HashSet<PathBuf>,
    /// Test execution environment
    pub environment: TestEnvironment,
}

/// Test execution environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEnvironment {
    pub rust_version: String,
    pub target_triple: String,
    pub features: Vec<String>,
    pub environment_vars: HashMap<String, String>,
}

/// Key for identifying cached test results
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CacheKey {
    pub test_name: String,
    pub suite_name: String,
    pub input_hash: String,
    pub source_hash: String,
}

impl CacheKey {
    pub fn to_filename(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let hash = hasher.finish();

        format!("{:016x}.cache", hash)
    }
}

impl TestCache {
    /// Create a new test cache
    pub async fn new(cache_dir: PathBuf, config: CacheConfig) -> TestResult<Self> {
        fs::create_dir_all(&cache_dir).await?;

        let metadata_path = cache_dir.join("metadata.json");
        let metadata = if metadata_path.exists() {
            let content = fs::read_to_string(&metadata_path).await?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            CacheMetadata::default()
        };

        let mut cache = Self {
            cache_dir,
            config,
            metadata,
        };

        // Perform initial cleanup if needed
        cache.cleanup_if_needed().await?;

        Ok(cache)
    }

    /// Get a cached test result
    pub async fn get(&mut self, key: &CacheKey) -> TestResult<Option<CachedTestResult>> {
        if !self.config.enabled {
            return Ok(None);
        }

        let filename = key.to_filename();
        let path = self.cache_dir.join(&filename);

        if !path.exists() {
            self.metadata.stats.misses += 1;
            return Ok(None);
        }

        // Check if cache entry is expired
        let metadata = fs::metadata(&path).await?;
        let age = SystemTime::now()
            .duration_since(metadata.modified()?)
            .unwrap_or(Duration::ZERO);

        if age.as_secs() > self.config.max_age_seconds {
            debug!("Cache entry expired: {}", filename);
            let _ = fs::remove_file(&path).await;
            self.metadata.stats.misses += 1;
            return Ok(None);
        }

        // Read and deserialize cached result
        match self.read_cached_result(&path).await {
            Ok(cached_result) => {
                debug!("Cache hit: {}", key.test_name);
                self.metadata.stats.hits += 1;
                Ok(Some(cached_result))
            }
            Err(e) => {
                warn!("Failed to read cached result {}: {}", filename, e);
                let _ = fs::remove_file(&path).await;
                self.metadata.stats.misses += 1;
                Ok(None)
            }
        }
    }

    /// Store a test result in the cache
    pub async fn put(&mut self, key: CacheKey, result: CachedTestResult) -> TestResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let filename = key.to_filename();
        let path = self.cache_dir.join(&filename);

        self.write_cached_result(&path, &result).await?;

        self.metadata.total_results += 1;
        self.metadata.stats.entry_count += 1;

        // Update cache size
        if let Ok(metadata) = fs::metadata(&path).await {
            self.metadata.stats.size_bytes += metadata.len();
        }

        debug!("Cached test result: {}", key.test_name);

        // Cleanup if cache is getting too large
        self.cleanup_if_needed().await?;

        Ok(())
    }

    /// Check if a test result is cached and valid
    pub async fn is_cached(&self, key: &CacheKey) -> bool {
        if !self.config.enabled {
            return false;
        }

        let filename = key.to_filename();
        let path = self.cache_dir.join(&filename);

        if !path.exists() {
            return false;
        }

        // Check if cache entry is expired
        if let Ok(metadata) = fs::metadata(&path).await {
            let age = SystemTime::now()
                .duration_since(metadata.modified().unwrap_or(UNIX_EPOCH))
                .unwrap_or(Duration::ZERO);

            age.as_secs() <= self.config.max_age_seconds
        } else {
            false
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.metadata.stats
    }

    /// Clear all cached results
    pub async fn clear(&mut self) -> TestResult<()> {
        info!("Clearing test cache");

        let mut entries = fs::read_dir(&self.cache_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_file() {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "cache") {
                    let _ = fs::remove_file(&path).await;
                }
            }
        }

        self.metadata = CacheMetadata::default();
        self.save_metadata().await?;

        Ok(())
    }

    /// Cleanup expired and oversized cache entries
    pub async fn cleanup_if_needed(&mut self) -> TestResult<()> {
        let now = SystemTime::now();
        let last_cleanup_age = now
            .duration_since(self.metadata.last_cleanup)
            .unwrap_or(Duration::ZERO);

        // Only cleanup every hour
        if last_cleanup_age.as_secs() < 3600 {
            return Ok(());
        }

        self.cleanup().await?;
        self.metadata.last_cleanup = now;
        self.save_metadata().await?;

        Ok(())
    }

    /// Perform cache cleanup
    async fn cleanup(&mut self) -> TestResult<()> {
        info!("Performing cache cleanup");

        let mut entries = Vec::new();
        let mut dir_entries = fs::read_dir(&self.cache_dir).await?;

        while let Some(entry) = dir_entries.next_entry().await? {
            if entry.file_type().await?.is_file() {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "cache") {
                    if let Ok(metadata) = fs::metadata(&path).await {
                        entries.push((path, metadata));
                    }
                }
            }
        }

        let now = SystemTime::now();
        let mut removed_count = 0;
        let mut removed_size = 0u64;

        // Remove expired entries
        for (path, metadata) in &entries {
            let age = now
                .duration_since(metadata.modified().unwrap_or(UNIX_EPOCH))
                .unwrap_or(Duration::ZERO);

            if age.as_secs() > self.config.max_age_seconds {
                if let Ok(()) = fs::remove_file(path).await {
                    removed_count += 1;
                    removed_size += metadata.len();
                }
            }
        }

        // Remove oldest entries if cache is too large
        let total_size: u64 = entries.iter().map(|(_, m)| m.len()).sum();
        if total_size > self.config.max_size_bytes {
            // Sort by modification time (oldest first)
            let mut entries_by_time: Vec<_> = entries
                .into_iter()
                .filter_map(|(path, metadata)| {
                    metadata.modified().ok().map(|time| (path, metadata, time))
                })
                .collect();

            entries_by_time.sort_by_key(|(_, _, time)| *time);

            let mut current_size = total_size;
            let target_size = (self.config.max_size_bytes as f64 * 0.8) as u64;

            for (path, metadata, _) in entries_by_time {
                if current_size <= target_size {
                    break;
                }

                if fs::remove_file(&path).await.is_ok() {
                    removed_count += 1;
                    removed_size += metadata.len();
                    current_size -= metadata.len();
                }
            }
        }

        if removed_count > 0 {
            info!(
                "Cache cleanup completed: removed {} entries ({} bytes)",
                removed_count, removed_size
            );
            self.metadata.stats.evictions += removed_count;
            self.metadata.stats.size_bytes =
                self.metadata.stats.size_bytes.saturating_sub(removed_size);
            self.metadata.stats.entry_count = self
                .metadata
                .stats
                .entry_count
                .saturating_sub(removed_count as usize);
        }

        Ok(())
    }

    /// Read a cached result from disk
    async fn read_cached_result(&self, path: &Path) -> TestResult<CachedTestResult> {
        let data = fs::read(path).await?;

        // Decompress if needed
        let decompressed = if self.config.compression_level > 0 {
            use flate2::read::GzDecoder;
            use std::io::Read;

            let mut decoder = GzDecoder::new(&data[..]);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| TestError::cache(format!("Decompression failed: {}", e)))?;
            decompressed
        } else {
            data
        };

        serde_json::from_slice(&decompressed)
            .map_err(|e| TestError::cache(format!("Failed to deserialize cached result: {}", e)))
    }

    /// Write a cached result to disk
    async fn write_cached_result(&self, path: &Path, result: &CachedTestResult) -> TestResult<()> {
        let serialized = serde_json::to_vec(result)
            .map_err(|e| TestError::cache(format!("Failed to serialize result: {}", e)))?;

        // Compress if needed
        let data = if self.config.compression_level > 0 {
            use flate2::write::GzEncoder;
            use flate2::Compression;
            use std::io::Write;

            let mut encoder =
                GzEncoder::new(Vec::new(), Compression::new(self.config.compression_level));
            encoder
                .write_all(&serialized)
                .map_err(|e| TestError::cache(format!("Compression failed: {}", e)))?;
            encoder
                .finish()
                .map_err(|e| TestError::cache(format!("Compression finish failed: {}", e)))?
        } else {
            serialized
        };

        fs::write(path, data).await?;
        Ok(())
    }

    /// Save cache metadata
    async fn save_metadata(&self) -> TestResult<()> {
        let metadata_path = self.cache_dir.join("metadata.json");
        let content = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| TestError::cache(format!("Failed to serialize metadata: {}", e)))?;

        fs::write(metadata_path, content).await?;
        Ok(())
    }
}

/// Utility functions for cache key generation
pub mod cache_keys {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Generate a hash for test inputs
    pub fn hash_test_inputs(test_name: &str, config: &str, features: &[String]) -> String {
        let mut hasher = DefaultHasher::new();
        test_name.hash(&mut hasher);
        config.hash(&mut hasher);
        features.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Generate a hash for source code dependencies
    pub async fn hash_source_dependencies(dependencies: &[PathBuf]) -> TestResult<String> {
        let mut hasher = DefaultHasher::new();

        for dep in dependencies {
            if dep.exists() {
                // Hash file path and modification time
                dep.hash(&mut hasher);

                if let Ok(metadata) = fs::metadata(dep).await {
                    if let Ok(modified) = metadata.modified() {
                        if let Ok(duration) = modified.duration_since(UNIX_EPOCH) {
                            duration.as_secs().hash(&mut hasher);
                        }
                    }
                    metadata.len().hash(&mut hasher);
                }
            }
        }

        Ok(format!("{:016x}", hasher.finish()))
    }

    /// Get current test environment
    pub fn get_test_environment() -> TestEnvironment {
        TestEnvironment {
            rust_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
            target_triple: std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()),
            features: std::env::var("CARGO_FEATURES")
                .unwrap_or_default()
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            environment_vars: [
                "CARGO_PKG_VERSION",
                "CARGO_PKG_NAME",
                "BITNET_TEST_CONFIG",
                "BITNET_TEST_PARALLEL",
            ]
            .iter()
            .filter_map(|&key| std::env::var(key).ok().map(|val| (key.to_string(), val)))
            .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_cache_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();
        let config = CacheConfig::default();

        let mut cache = TestCache::new(cache_dir, config).await.unwrap();

        let key = CacheKey {
            test_name: "test_example".to_string(),
            suite_name: "example_suite".to_string(),
            input_hash: "input123".to_string(),
            source_hash: "source456".to_string(),
        };

        // Initially should not be cached
        assert!(!cache.is_cached(&key).await);
        assert!(cache.get(&key).await.unwrap().is_none());

        // Create a test result to cache
        let test_result = TestResultData::passed(
            "test_example".to_string(),
            Default::default(),
            Duration::from_millis(100),
        );

        let cached_result = CachedTestResult {
            result: test_result,
            input_hash: "input123".to_string(),
            source_hash: "source456".to_string(),
            cached_at: SystemTime::now(),
            dependencies: HashSet::new(),
            environment: cache_keys::get_test_environment(),
        };

        // Store in cache
        cache.put(key.clone(), cached_result.clone()).await.unwrap();

        // Should now be cached
        assert!(cache.is_cached(&key).await);
        let retrieved = cache.get(&key).await.unwrap().unwrap();
        assert_eq!(retrieved.result.test_name, "test_example");
        assert_eq!(retrieved.input_hash, "input123");
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().to_path_buf();
        let mut config = CacheConfig::default();
        config.max_age_seconds = 1; // 1 second expiration

        let mut cache = TestCache::new(cache_dir, config).await.unwrap();

        let key = CacheKey {
            test_name: "test_expiry".to_string(),
            suite_name: "example_suite".to_string(),
            input_hash: "input123".to_string(),
            source_hash: "source456".to_string(),
        };

        let test_result = TestResultData::passed(
            "test_expiry".to_string(),
            Default::default(),
            Duration::from_millis(100),
        );

        let cached_result = CachedTestResult {
            result: test_result,
            input_hash: "input123".to_string(),
            source_hash: "source456".to_string(),
            cached_at: SystemTime::now(),
            dependencies: HashSet::new(),
            environment: cache_keys::get_test_environment(),
        };

        // Store in cache
        cache.put(key.clone(), cached_result).await.unwrap();
        assert!(cache.is_cached(&key).await);

        // Wait for expiration
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Should no longer be cached
        assert!(!cache.is_cached(&key).await);
        assert!(cache.get(&key).await.unwrap().is_none());
    }

    #[test]
    fn test_cache_key_generation() {
        let key1 = CacheKey {
            test_name: "test1".to_string(),
            suite_name: "suite1".to_string(),
            input_hash: "input1".to_string(),
            source_hash: "source1".to_string(),
        };

        let key2 = CacheKey {
            test_name: "test2".to_string(),
            suite_name: "suite1".to_string(),
            input_hash: "input1".to_string(),
            source_hash: "source1".to_string(),
        };

        // Different keys should generate different filenames
        assert_ne!(key1.to_filename(), key2.to_filename());

        // Same key should generate same filename
        let key1_copy = key1.clone();
        assert_eq!(key1.to_filename(), key1_copy.to_filename());
    }
}
