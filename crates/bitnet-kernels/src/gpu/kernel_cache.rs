//! Persistent cache for compiled OpenCL kernels.
//!
//! Caches compiled kernel binaries to disk so subsequent launches skip
//! `clBuildProgram`. The cache key is a hash of the kernel source,
//! device name, driver version, and build options. Stored under
//! `$BITNET_OPENCL_CACHE_DIR`, `$XDG_CACHE_HOME/bitnet/opencl_kernels/`,
//! or `%LOCALAPPDATA%/bitnet/opencl_kernels/`.
//!
//! Set `BITNET_OPENCL_NO_CACHE=1` to disable caching entirely.

use log::{debug, info};
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Statistics about the kernel cache.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheStats {
    /// Number of cached entries.
    pub entry_count: usize,
    /// Total size in bytes of all cached binaries.
    pub total_size_bytes: u64,
    /// Age of the oldest entry, if any.
    pub oldest_entry_age: Option<Duration>,
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KernelCache: {} entries, {:.1} KB total",
            self.entry_count,
            self.total_size_bytes as f64 / 1024.0
        )?;
        if let Some(age) = self.oldest_entry_age {
            write!(f, ", oldest {:.0}s ago", age.as_secs_f64())?;
        }
        Ok(())
    }
}

/// Persistent cache for compiled OpenCL kernel binaries.
///
/// Stores compiled GPU programs on disk keyed by a hash of
/// (source, device_name, driver_version, build_options).
pub struct KernelCache {
    cache_dir: PathBuf,
}

impl fmt::Debug for KernelCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelCache")
            .field("cache_dir", &self.cache_dir)
            .finish()
    }
}

impl KernelCache {
    /// Create a new kernel cache using the default or overridden directory.
    ///
    /// Resolution order:
    /// 1. `$BITNET_OPENCL_CACHE_DIR` (if set)
    /// 2. `$XDG_CACHE_HOME/bitnet/opencl_kernels/` (Unix)
    /// 3. `%LOCALAPPDATA%/bitnet/opencl_kernels/` (Windows)
    /// 4. `$HOME/.cache/bitnet/opencl_kernels/`
    pub fn new() -> io::Result<Self> {
        if Self::is_disabled() {
            let dir = std::env::temp_dir().join("bitnet_opencl_cache_disabled");
            return Ok(Self { cache_dir: dir });
        }
        let dir = Self::resolve_cache_dir()?;
        fs::create_dir_all(&dir)?;
        info!("OpenCL kernel cache directory: {}", dir.display());
        Ok(Self { cache_dir: dir })
    }

    /// Create a kernel cache rooted at a specific directory (for testing).
    pub fn with_dir(dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self { cache_dir: dir })
    }

    /// Returns `true` if caching is disabled via `BITNET_OPENCL_NO_CACHE=1`.
    pub fn is_disabled() -> bool {
        std::env::var("BITNET_OPENCL_NO_CACHE")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    /// Compute a cache key from kernel source, device name, driver version,
    /// and optional build options.
    pub fn cache_key(
        source: &str,
        device_name: &str,
        driver_version: &str,
        build_options: &str,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        device_name.hash(&mut hasher);
        driver_version.hash(&mut hasher);
        build_options.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Try to load a pre-compiled binary for the given cache key.
    pub fn load(&self, key: &str) -> Option<Vec<u8>> {
        if Self::is_disabled() {
            return None;
        }
        let path = self.entry_path(key);
        match fs::read(&path) {
            Ok(data) => {
                debug!("Kernel cache hit: {}", key);
                Some(data)
            }
            Err(_) => {
                debug!("Kernel cache miss: {}", key);
                None
            }
        }
    }

    /// Store a compiled binary under the given cache key.
    pub fn store(&self, key: &str, binary: &[u8]) -> io::Result<()> {
        if Self::is_disabled() {
            return Ok(());
        }
        let path = self.entry_path(key);
        fs::create_dir_all(self.cache_dir.as_path())?;
        fs::write(&path, binary)?;
        debug!("Kernel cache store: {} ({} bytes)", key, binary.len());
        Ok(())
    }

    /// Remove all cached kernel binaries.
    pub fn clear(&self) -> io::Result<()> {
        if self.cache_dir.exists() {
            for entry in fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                if entry.path().extension().is_some_and(|e| e == "bin") {
                    fs::remove_file(entry.path())?;
                }
            }
            info!("Kernel cache cleared");
        }
        Ok(())
    }

    /// Gather statistics about the cache.
    pub fn stats(&self) -> CacheStats {
        let mut count = 0usize;
        let mut total_size = 0u64;
        let mut oldest: Option<SystemTime> = None;

        if let Ok(entries) = fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "bin") {
                    count += 1;
                    if let Ok(meta) = fs::metadata(&path) {
                        total_size += meta.len();
                        if let Ok(modified) = meta.modified() {
                            oldest = Some(match oldest {
                                Some(prev) if modified < prev => modified,
                                Some(prev) => prev,
                                None => modified,
                            });
                        }
                    }
                }
            }
        }

        let oldest_entry_age =
            oldest.and_then(|t| SystemTime::now().duration_since(t).ok());

        CacheStats {
            entry_count: count,
            total_size_bytes: total_size,
            oldest_entry_age,
        }
    }

    /// Return the path to the cache directory.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Check whether a key exists in the cache.
    pub fn contains(&self, key: &str) -> bool {
        self.entry_path(key).exists()
    }

    /// List all cache keys currently stored.
    pub fn keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "bin") {
                    if let Some(stem) = path.file_stem() {
                        keys.push(stem.to_string_lossy().into_owned());
                    }
                }
            }
        }
        keys
    }

    fn entry_path(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("{key}.bin"))
    }

    fn resolve_cache_dir() -> io::Result<PathBuf> {
        if let Ok(dir) = std::env::var("BITNET_OPENCL_CACHE_DIR") {
            return Ok(PathBuf::from(dir));
        }
        if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
            return Ok(
                PathBuf::from(xdg).join("bitnet").join("opencl_kernels")
            );
        }
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return Ok(
                PathBuf::from(local).join("bitnet").join("opencl_kernels")
            );
        }
        if let Ok(home) = std::env::var("HOME") {
            return Ok(PathBuf::from(home)
                .join(".cache")
                .join("bitnet")
                .join("opencl_kernels"));
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Could not determine cache directory: \
             set BITNET_OPENCL_CACHE_DIR, XDG_CACHE_HOME, LOCALAPPDATA, or HOME",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::TempDir;

    fn temp_cache() -> (TempDir, KernelCache) {
        let dir = TempDir::new().unwrap();
        let cache = KernelCache::with_dir(dir.path().to_path_buf()).unwrap();
        (dir, cache)
    }

    #[test]
    fn test_cache_miss_returns_none() {
        let (_dir, cache) = temp_cache();
        assert!(cache.load("nonexistent_key").is_none());
    }

    #[test]
    fn test_store_then_load_roundtrip() {
        let (_dir, cache) = temp_cache();
        let binary = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02];
        cache.store("test_kernel", &binary).unwrap();
        let loaded = cache.load("test_kernel").unwrap();
        assert_eq!(loaded, binary);
    }

    #[test]
    fn test_cache_hit_after_store() {
        let (_dir, cache) = temp_cache();
        assert!(!cache.contains("my_key"));
        cache.store("my_key", &[1, 2, 3]).unwrap();
        assert!(cache.contains("my_key"));
    }

    #[test]
    fn test_clear_removes_all_entries() {
        let (_dir, cache) = temp_cache();
        cache.store("a", &[1]).unwrap();
        cache.store("b", &[2]).unwrap();
        assert_eq!(cache.stats().entry_count, 2);
        cache.clear().unwrap();
        assert_eq!(cache.stats().entry_count, 0);
    }

    #[test]
    fn test_stats_empty_cache() {
        let (_dir, cache) = temp_cache();
        let stats = cache.stats();
        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.total_size_bytes, 0);
        assert!(stats.oldest_entry_age.is_none());
    }

    #[test]
    fn test_stats_with_entries() {
        let (_dir, cache) = temp_cache();
        cache.store("k1", &[0u8; 100]).unwrap();
        cache.store("k2", &[0u8; 200]).unwrap();
        let stats = cache.stats();
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.total_size_bytes, 300);
        assert!(stats.oldest_entry_age.is_some());
    }

    #[test]
    fn test_cache_key_deterministic() {
        let k1 = KernelCache::cache_key("src", "dev", "1.0", "");
        let k2 = KernelCache::cache_key("src", "dev", "1.0", "");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_varies_with_source() {
        let k1 = KernelCache::cache_key("src_a", "dev", "1.0", "");
        let k2 = KernelCache::cache_key("src_b", "dev", "1.0", "");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_varies_with_device() {
        let k1 = KernelCache::cache_key("src", "Arc A770", "1.0", "");
        let k2 = KernelCache::cache_key("src", "Arc A750", "1.0", "");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_varies_with_driver() {
        let k1 = KernelCache::cache_key("src", "dev", "1.0", "");
        let k2 = KernelCache::cache_key("src", "dev", "2.0", "");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_varies_with_build_options() {
        let k1 =
            KernelCache::cache_key("src", "dev", "1.0", "-cl-fast-relaxed-math");
        let k2 = KernelCache::cache_key("src", "dev", "1.0", "");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_overwrite_existing_entry() {
        let (_dir, cache) = temp_cache();
        cache.store("k", &[1, 2, 3]).unwrap();
        cache.store("k", &[4, 5]).unwrap();
        let loaded = cache.load("k").unwrap();
        assert_eq!(loaded, vec![4, 5]);
    }

    #[test]
    fn test_keys_listing() {
        let (_dir, cache) = temp_cache();
        cache.store("alpha", &[1]).unwrap();
        cache.store("beta", &[2]).unwrap();
        let mut keys = cache.keys();
        keys.sort();
        assert_eq!(keys, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_cache_display_stats() {
        let stats = CacheStats {
            entry_count: 3,
            total_size_bytes: 2048,
            oldest_entry_age: Some(Duration::from_secs(120)),
        };
        let s = stats.to_string();
        assert!(s.contains("3 entries"));
        assert!(s.contains("KB"));
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_env_override_cache_dir() {
        let dir = TempDir::new().unwrap();
        let custom_path = dir.path().join("custom_cache");
        temp_env::with_var(
            "BITNET_OPENCL_CACHE_DIR",
            Some(custom_path.to_str().unwrap()),
            || {
                temp_env::with_var(
                    "BITNET_OPENCL_NO_CACHE",
                    None::<&str>,
                    || {
                        let cache = KernelCache::new().unwrap();
                        assert_eq!(
                            cache.cache_dir(),
                            custom_path.as_path()
                        );
                    },
                );
            },
        );
    }

    #[test]
    #[serial(bitnet_env)]
    fn test_no_cache_env_disables_store_and_load() {
        temp_env::with_var("BITNET_OPENCL_NO_CACHE", Some("1"), || {
            let dir = TempDir::new().unwrap();
            let cache =
                KernelCache::with_dir(dir.path().to_path_buf()).unwrap();
            cache.store("key", &[1, 2, 3]).unwrap();
            assert!(cache.load("key").is_none());
        });
    }

    #[test]
    fn test_store_large_binary() {
        let (_dir, cache) = temp_cache();
        let binary = vec![0xABu8; 1_000_000];
        cache.store("large", &binary).unwrap();
        let loaded = cache.load("large").unwrap();
        assert_eq!(loaded.len(), 1_000_000);
    }

    #[test]
    fn test_clear_on_empty_cache() {
        let (_dir, cache) = temp_cache();
        cache.clear().unwrap();
        assert_eq!(cache.stats().entry_count, 0);
    }
}