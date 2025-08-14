use super::cache::{CacheConfig, TestCache};
use super::errors::TestError;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::fs;
use tracing::{debug, info, warn};

/// GitHub Actions cache integration for test results and data
pub struct GitHubCacheManager {
    config: GitHubCacheConfig,
    workspace_root: PathBuf,
}

/// Configuration for GitHub Actions cache integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubCacheConfig {
    /// Enable GitHub Actions cache integration
    pub enabled: bool,
    /// Cache key prefix
    pub key_prefix: String,
    /// Cache version (increment to invalidate all caches)
    pub version: String,
    /// Paths to cache
    pub cache_paths: Vec<PathBuf>,
    /// Maximum cache size in MB
    pub max_size_mb: u64,
    /// Cache restore keys (fallback keys)
    pub restore_keys: Vec<String>,
    /// Environment variables to include in cache key
    pub key_env_vars: Vec<String>,
}

impl Default for GitHubCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            key_prefix: "bitnet-test".to_string(),
            version: "v1".to_string(),
            cache_paths: vec![
                PathBuf::from("tests/cache"),
                PathBuf::from("target/debug/deps"),
                PathBuf::from("target/release/deps"),
            ],
            max_size_mb: 1024, // 1 GB
            restore_keys: vec!["bitnet-test-v1-".to_string()],
            key_env_vars: vec![
                "CARGO_PKG_VERSION".to_string(),
                "RUSTC_VERSION".to_string(),
                "TARGET".to_string(),
            ],
        }
    }
}

/// Cache key information for GitHub Actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheKeyInfo {
    /// Primary cache key
    pub primary_key: String,
    /// Restore keys (fallback)
    pub restore_keys: Vec<String>,
    /// Paths to cache
    pub paths: Vec<PathBuf>,
}

/// Cache operation result
#[derive(Debug, Clone)]
pub struct CacheResult {
    /// Whether cache was hit
    pub cache_hit: bool,
    /// Cache key used
    pub key: String,
    /// Size of cached data in bytes
    pub size_bytes: u64,
    /// Time taken for operation
    pub duration: std::time::Duration,
}

impl GitHubCacheManager {
    /// Create a new GitHub Actions cache manager
    pub fn new(config: GitHubCacheConfig, workspace_root: PathBuf) -> Self {
        Self {
            config,
            workspace_root,
        }
    }

    /// Generate cache key for test data
    pub async fn generate_test_cache_key(&self) -> TestResult<CacheKeyInfo> {
        let mut key_components = vec![self.config.key_prefix.clone(), self.config.version.clone()];

        // Add environment variables to key
        for env_var in &self.config.key_env_vars {
            if let Ok(value) = std::env::var(env_var) {
                key_components.push(format!("{}={}", env_var, value));
            }
        }

        // Add hash of Cargo.lock for dependency changes
        if let Ok(cargo_lock_hash) = self.hash_file("Cargo.lock").await {
            key_components.push(format!("deps={}", cargo_lock_hash));
        }

        // Add hash of test configuration files
        let config_files = ["tests/config.toml", "bitnet-test.toml"];
        for config_file in &config_files {
            if let Ok(hash) = self.hash_file(config_file).await {
                key_components.push(format!("config={}", hash));
            }
        }

        let primary_key = key_components.join("-");

        // Generate restore keys (progressively less specific)
        let mut restore_keys = Vec::new();
        for i in (1..key_components.len()).rev() {
            restore_keys.push(key_components[..i].join("-"));
        }

        Ok(CacheKeyInfo {
            primary_key,
            restore_keys,
            paths: self.config.cache_paths.clone(),
        })
    }

    /// Generate cache key for test fixtures
    pub async fn generate_fixture_cache_key(&self) -> TestResult<CacheKeyInfo> {
        let mut key_components = vec![
            self.config.key_prefix.clone(),
            "fixtures".to_string(),
            self.config.version.clone(),
        ];

        // Add hash of fixture configuration
        if let Ok(fixtures_hash) = self.hash_fixture_config().await {
            key_components.push(format!("fixtures={}", fixtures_hash));
        }

        let primary_key = key_components.join("-");
        let restore_keys = vec![
            format!(
                "{}-fixtures-{}-",
                self.config.key_prefix, self.config.version
            ),
            format!("{}-fixtures-", self.config.key_prefix),
        ];

        Ok(CacheKeyInfo {
            primary_key,
            restore_keys,
            paths: vec![PathBuf::from("tests/cache/fixtures")],
        })
    }

    /// Restore cache from GitHub Actions
    pub async fn restore_cache(&self, key_info: &CacheKeyInfo) -> TestResult<CacheResult> {
        if !self.config.enabled || !self.is_github_actions() {
            return Ok(CacheResult {
                cache_hit: false,
                key: "disabled".to_string(),
                size_bytes: 0,
                duration: std::time::Duration::ZERO,
            });
        }

        info!("Restoring cache with key: {}", key_info.primary_key);

        let start_time = std::time::Instant::now();

        // Use GitHub Actions cache action
        let mut cmd = Command::new("gh");
        cmd.args(&["cache", "restore"]).arg(&key_info.primary_key);

        // Add restore keys
        for restore_key in &key_info.restore_keys {
            cmd.arg("--restore-keys").arg(restore_key);
        }

        // Add paths
        for path in &key_info.paths {
            cmd.arg("--path").arg(path);
        }

        cmd.current_dir(&self.workspace_root);

        let output = cmd
            .output()
            .map_err(|e| TestError::cache(format!("Failed to run cache restore: {}", e)))?;

        let duration = start_time.elapsed();
        let cache_hit = output.status.success();

        if cache_hit {
            info!("Cache restored successfully in {:?}", duration);
            let size_bytes = self.calculate_cache_size(&key_info.paths).await?;

            Ok(CacheResult {
                cache_hit: true,
                key: key_info.primary_key.clone(),
                size_bytes,
                duration,
            })
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            debug!("Cache miss: {}", stderr);

            Ok(CacheResult {
                cache_hit: false,
                key: key_info.primary_key.clone(),
                size_bytes: 0,
                duration,
            })
        }
    }

    /// Save cache to GitHub Actions
    pub async fn save_cache(&self, key_info: &CacheKeyInfo) -> TestResult<CacheResult> {
        if !self.config.enabled || !self.is_github_actions() {
            return Ok(CacheResult {
                cache_hit: false,
                key: "disabled".to_string(),
                size_bytes: 0,
                duration: std::time::Duration::ZERO,
            });
        }

        info!("Saving cache with key: {}", key_info.primary_key);

        let start_time = std::time::Instant::now();

        // Check cache size before saving
        let size_bytes = self.calculate_cache_size(&key_info.paths).await?;
        let size_mb = size_bytes / (1024 * 1024);

        if size_mb > self.config.max_size_mb {
            warn!(
                "Cache size ({} MB) exceeds limit ({} MB), skipping save",
                size_mb, self.config.max_size_mb
            );
            return Ok(CacheResult {
                cache_hit: false,
                key: key_info.primary_key.clone(),
                size_bytes,
                duration: start_time.elapsed(),
            });
        }

        // Use GitHub Actions cache action
        let mut cmd = Command::new("gh");
        cmd.args(&["cache", "save"]).arg(&key_info.primary_key);

        // Add paths
        for path in &key_info.paths {
            if path.exists() {
                cmd.arg("--path").arg(path);
            }
        }

        cmd.current_dir(&self.workspace_root);

        let output = cmd
            .output()
            .map_err(|e| TestError::cache(format!("Failed to run cache save: {}", e)))?;

        let duration = start_time.elapsed();

        if output.status.success() {
            info!(
                "Cache saved successfully in {:?} ({} MB)",
                duration, size_mb
            );
            Ok(CacheResult {
                cache_hit: true,
                key: key_info.primary_key.clone(),
                size_bytes,
                duration,
            })
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("Failed to save cache: {}", stderr);

            Ok(CacheResult {
                cache_hit: false,
                key: key_info.primary_key.clone(),
                size_bytes,
                duration,
            })
        }
    }

    /// Setup cache for test execution
    pub async fn setup_test_cache(&self) -> TestResult<TestCache> {
        // Generate cache key for test data
        let key_info = self.generate_test_cache_key().await?;

        // Try to restore cache
        let restore_result = self.restore_cache(&key_info).await?;

        if restore_result.cache_hit {
            info!("Test cache restored from GitHub Actions");
        } else {
            info!("No test cache found, starting fresh");
        }

        // Create test cache with restored data
        let cache_dir = self.workspace_root.join("tests/cache");
        let cache_config = CacheConfig::default();

        TestCache::new(cache_dir, cache_config).await
    }

    /// Cleanup and save cache after test execution
    pub async fn cleanup_test_cache(&self, cache: &mut TestCache) -> TestResult<()> {
        // Perform cache cleanup
        cache.cleanup_if_needed().await?;

        // Generate cache key
        let key_info = self.generate_test_cache_key().await?;

        // Save cache to GitHub Actions
        let save_result = self.save_cache(&key_info).await?;

        if save_result.cache_hit {
            info!("Test cache saved to GitHub Actions");
        } else {
            warn!("Failed to save test cache");
        }

        Ok(())
    }

    /// Setup fixture cache
    pub async fn setup_fixture_cache(&self) -> TestResult<()> {
        let key_info = self.generate_fixture_cache_key().await?;
        let restore_result = self.restore_cache(&key_info).await?;

        if restore_result.cache_hit {
            info!("Fixture cache restored from GitHub Actions");
        } else {
            info!("No fixture cache found");
        }

        Ok(())
    }

    /// Save fixture cache
    pub async fn save_fixture_cache(&self) -> TestResult<()> {
        let key_info = self.generate_fixture_cache_key().await?;
        let save_result = self.save_cache(&key_info).await?;

        if save_result.cache_hit {
            info!("Fixture cache saved to GitHub Actions");
        }

        Ok(())
    }

    /// Check if running in GitHub Actions
    fn is_github_actions(&self) -> bool {
        std::env::var("GITHUB_ACTIONS").is_ok()
    }

    /// Calculate total size of cache paths
    async fn calculate_cache_size(&self, paths: &[PathBuf]) -> TestResult<u64> {
        let mut total_size = 0u64;

        for path in paths {
            if path.exists() {
                total_size += self.calculate_directory_size(path).await?;
            }
        }

        Ok(total_size)
    }

    /// Calculate size of a directory recursively
    fn calculate_directory_size<'a>(
        &'a self,
        path: &'a Path,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = TestResult<u64>> + Send + '_>> {
        Box::pin(async move {
            let mut total_size = 0u64;

            if path.is_file() {
                if let Ok(metadata) = fs::metadata(path).await {
                    return Ok(metadata.len());
                }
            } else if path.is_dir() {
                let mut entries = fs::read_dir(path).await?;

                while let Some(entry) = entries.next_entry().await? {
                    let entry_path = entry.path();
                    total_size += self.calculate_directory_size(&entry_path).await?;
                }
            }

            Ok(total_size)
        })
    }

    /// Hash a file for cache key generation
    async fn hash_file(&self, file_path: &str) -> TestResult<String> {
        let path = self.workspace_root.join(file_path);

        if !path.exists() {
            return Ok("missing".to_string());
        }

        let content = fs::read(&path).await?;
        let hash = sha2::Sha256::digest(&content);
        Ok(format!("{:x}", hash)[..8].to_string()) // Use first 8 chars
    }

    /// Hash fixture configuration for cache key
    async fn hash_fixture_config(&self) -> TestResult<String> {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();

        // Hash fixture configuration files
        let config_files = [
            "tests/fixtures.toml",
            "tests/config.toml",
            "bitnet-test.toml",
        ];

        for config_file in &config_files {
            let path = self.workspace_root.join(config_file);
            if path.exists() {
                let content = fs::read(&path).await?;
                hasher.update(&content);
            }
        }

        // Hash environment variables that affect fixtures
        let fixture_env_vars = [
            "BITNET_TEST_FIXTURE_BASE_URL",
            "BITNET_TEST_AUTO_DOWNLOAD",
            "BITNET_TEST_MAX_CACHE_SIZE",
        ];

        for env_var in &fixture_env_vars {
            if let Ok(value) = std::env::var(env_var) {
                hasher.update(env_var.as_bytes());
                hasher.update(value.as_bytes());
            }
        }

        let hash = hasher.finalize();
        Ok(format!("{:x}", hash)[..8].to_string())
    }
}

/// Utility functions for GitHub Actions cache integration
pub mod github_actions {
    use super::*;

    /// Set up GitHub Actions cache for test execution
    pub async fn setup_cache(
        workspace_root: PathBuf,
    ) -> TestResult<(GitHubCacheManager, TestCache)> {
        let config = GitHubCacheConfig::default();
        let cache_manager = GitHubCacheManager::new(config, workspace_root);

        // Setup fixture cache first
        cache_manager.setup_fixture_cache().await?;

        // Setup test cache
        let test_cache = cache_manager.setup_test_cache().await?;

        Ok((cache_manager, test_cache))
    }

    /// Cleanup and save caches after test execution
    pub async fn cleanup_cache(
        cache_manager: &GitHubCacheManager,
        test_cache: &mut TestCache,
    ) -> TestResult<()> {
        // Save test cache
        cache_manager.cleanup_test_cache(test_cache).await?;

        // Save fixture cache
        cache_manager.save_fixture_cache().await?;

        Ok(())
    }

    /// Generate cache configuration for GitHub Actions workflow
    pub fn generate_workflow_cache_config() -> serde_yaml::Value {
        serde_yaml::from_str(r#"
name: Cache Test Data
uses: actions/cache@v4
with:
  path: |
    tests/cache
    target/debug/deps
    target/release/deps
  key: bitnet-test-v1-${{ runner.os }}-${{ hashFiles('Cargo.lock') }}-${{ hashFiles('tests/config.toml') }}
  restore-keys: |
    bitnet-test-v1-${{ runner.os }}-${{ hashFiles('Cargo.lock') }}-
    bitnet-test-v1-${{ runner.os }}-
    bitnet-test-v1-
"#).unwrap()
    }

    /// Generate fixture cache configuration for GitHub Actions workflow
    pub fn generate_fixture_cache_config() -> serde_yaml::Value {
        serde_yaml::from_str(r#"
name: Cache Test Fixtures
uses: actions/cache@v4
with:
  path: tests/cache/fixtures
  key: bitnet-fixtures-v1-${{ hashFiles('tests/fixtures.toml') }}-${{ hashFiles('tests/config.toml') }}
  restore-keys: |
    bitnet-fixtures-v1-${{ hashFiles('tests/fixtures.toml') }}-
    bitnet-fixtures-v1-
"#).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_cache_key_generation() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_root = temp_dir.path().to_path_buf();

        // Create a Cargo.lock file
        let cargo_lock = workspace_root.join("Cargo.lock");
        fs::write(&cargo_lock, "# Test Cargo.lock").await.unwrap();

        let config = GitHubCacheConfig::default();
        let cache_manager = GitHubCacheManager::new(config, workspace_root);

        let key_info = cache_manager.generate_test_cache_key().await.unwrap();

        assert!(key_info.primary_key.starts_with("bitnet-test-v1"));
        assert!(!key_info.restore_keys.is_empty());
        assert!(!key_info.paths.is_empty());
    }

    #[tokio::test]
    async fn test_fixture_cache_key() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_root = temp_dir.path().to_path_buf();

        let config = GitHubCacheConfig::default();
        let cache_manager = GitHubCacheManager::new(config, workspace_root);

        let key_info = cache_manager.generate_fixture_cache_key().await.unwrap();

        assert!(key_info.primary_key.contains("fixtures"));
        assert!(!key_info.restore_keys.is_empty());
    }

    #[test]
    fn test_github_actions_detection() {
        let temp_dir = TempDir::new().unwrap();
        let workspace_root = temp_dir.path().to_path_buf();
        let config = GitHubCacheConfig::default();
        let cache_manager = GitHubCacheManager::new(config, workspace_root);

        // Should be false in test environment
        assert!(!cache_manager.is_github_actions());

        // Set environment variable
        std::env::set_var("GITHUB_ACTIONS", "true");
        assert!(cache_manager.is_github_actions());
        std::env::remove_var("GITHUB_ACTIONS");
    }

    #[tokio::test]
    async fn test_directory_size_calculation() {
        let temp_dir = TempDir::new().unwrap();
        let test_dir = temp_dir.path().join("test");
        fs::create_dir_all(&test_dir).await.unwrap();

        // Create some test files
        fs::write(test_dir.join("file1.txt"), "Hello")
            .await
            .unwrap();
        fs::write(test_dir.join("file2.txt"), "World!")
            .await
            .unwrap();

        let config = GitHubCacheConfig::default();
        let cache_manager = GitHubCacheManager::new(config, temp_dir.path().to_path_buf());

        let size = cache_manager
            .calculate_directory_size(&test_dir)
            .await
            .unwrap();
        assert_eq!(size, 11); // "Hello" (5) + "World!" (6)
    }
}
