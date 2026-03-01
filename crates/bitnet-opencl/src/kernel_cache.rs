use std::collections::HashMap;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

/// Key used to look up a cached kernel binary.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    pub kernel_name: String,
    pub device_id: String,
    pub compiler_options: String,
}

/// A cached compiled kernel binary with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub binary_data: Vec<u8>,
    pub source_hash: u64,
    pub timestamp: u64,
    pub device_name: String,
}

/// Statistics about the cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// Thread-safe GPU kernel compilation cache with optional disk persistence.
#[derive(Clone)]
pub struct KernelCache {
    inner: Arc<RwLock<CacheInner>>,
}

struct CacheInner {
    entries: HashMap<CacheKey, CacheEntry>,
    cache_dir: Option<PathBuf>,
    enabled: bool,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl KernelCache {
    /// Create a new cache. Reads `BITNET_KERNEL_CACHE` and
    /// `BITNET_KERNEL_CACHE_DIR` environment variables.
    #[must_use]
    pub fn new() -> Self {
        let enabled = std::env::var("BITNET_KERNEL_CACHE").map_or(true, |v| v != "0");

        let cache_dir = if enabled { resolve_cache_dir() } else { None };

        Self {
            inner: Arc::new(RwLock::new(CacheInner {
                entries: HashMap::new(),
                cache_dir,
                enabled,
                hits: 0,
                misses: 0,
                evictions: 0,
            })),
        }
    }

    /// Create a cache with an explicit directory and enabled flag.
    #[must_use]
    pub fn with_config(cache_dir: Option<PathBuf>, enabled: bool) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CacheInner {
                entries: HashMap::new(),
                cache_dir,
                enabled,
                hits: 0,
                misses: 0,
                evictions: 0,
            })),
        }
    }

    /// Look up a cached kernel. Returns `Some(entry)` on cache hit if the
    /// `source_hash` matches, otherwise `None`.
    pub fn get(&self, key: &CacheKey, source_hash: u64) -> Option<CacheEntry> {
        let mut inner = self.inner.write().expect("cache lock poisoned");
        if !inner.enabled {
            inner.misses += 1;
            return None;
        }

        // Try in-memory first.
        if let Some(entry) = inner.entries.get(key).cloned() {
            if entry.source_hash == source_hash {
                inner.hits += 1;
                return Some(entry);
            }
            // Source changed → stale entry.
            inner.entries.remove(key);
            inner.evictions += 1;
        }

        // Try disk.
        if let Some(entry) = Self::load_from_disk(inner.cache_dir.as_ref(), key)
            && entry.source_hash == source_hash
        {
            inner.entries.insert(key.clone(), entry.clone());
            inner.hits += 1;
            return Some(entry);
        }

        inner.misses += 1;
        None
    }

    /// Get a cached binary or compile via the provided closure.
    pub fn get_or_compile<F, E>(
        &self,
        key: &CacheKey,
        source_hash: u64,
        compile_fn: F,
    ) -> Result<CacheEntry, E>
    where
        F: FnOnce() -> Result<CacheEntry, E>,
    {
        if let Some(entry) = self.get(key, source_hash) {
            return Ok(entry);
        }

        let entry = compile_fn()?;
        self.insert(key.clone(), entry.clone());
        Ok(entry)
    }

    /// Insert a compiled kernel into the cache.
    pub fn insert(&self, key: CacheKey, entry: CacheEntry) {
        let mut inner = self.inner.write().expect("cache lock poisoned");
        if !inner.enabled {
            return;
        }
        Self::save_to_disk(inner.cache_dir.as_ref(), &key, &entry);
        inner.entries.insert(key, entry);
    }

    /// Invalidate a single cache entry.
    pub fn invalidate(&self, key: &CacheKey) {
        let mut inner = self.inner.write().expect("cache lock poisoned");
        if inner.entries.remove(key).is_some() {
            inner.evictions += 1;
        }
        Self::remove_from_disk(inner.cache_dir.as_ref(), key);
    }

    /// Remove all cached entries (in-memory and on disk).
    pub fn clear(&self) {
        let mut inner = self.inner.write().expect("cache lock poisoned");
        let count = inner.entries.len() as u64;
        inner.entries.clear();
        inner.evictions += count;

        if let Some(dir) = &inner.cache_dir {
            let _ = fs::remove_dir_all(dir);
        }
    }

    /// Return cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.read().expect("cache lock poisoned");
        CacheStats {
            entries: inner.entries.len(),
            hits: inner.hits,
            misses: inner.misses,
            evictions: inner.evictions,
        }
    }

    /// Whether the cache is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.inner.read().expect("cache lock poisoned").enabled
    }

    // ── disk helpers ──────────────────────────────────────────────

    fn disk_path(cache_dir: Option<&PathBuf>, key: &CacheKey) -> Option<PathBuf> {
        let dir = cache_dir?;
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        Some(dir.join(format!("{:016x}.json", hasher.finish())))
    }

    fn load_from_disk(cache_dir: Option<&PathBuf>, key: &CacheKey) -> Option<CacheEntry> {
        let path = Self::disk_path(cache_dir, key)?;
        let data = fs::read_to_string(&path).ok()?;
        serde_json::from_str(&data).ok()
    }

    fn save_to_disk(cache_dir: Option<&PathBuf>, key: &CacheKey, entry: &CacheEntry) {
        if let Some(path) = Self::disk_path(cache_dir, key) {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if let Ok(json) = serde_json::to_string_pretty(entry) {
                let _ = fs::write(&path, json);
            }
        }
    }

    fn remove_from_disk(cache_dir: Option<&PathBuf>, key: &CacheKey) {
        if let Some(path) = Self::disk_path(cache_dir, key) {
            let _ = fs::remove_file(path);
        }
    }
}

impl Default for KernelCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a stable hash for kernel source code.
#[must_use]
pub fn hash_source(source: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

/// Resolve the kernel cache directory from env or platform defaults.
fn resolve_cache_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("BITNET_KERNEL_CACHE_DIR") {
        return Some(PathBuf::from(dir));
    }
    dirs::cache_dir().map(|d| d.join("bitnet").join("kernels"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key(name: &str) -> CacheKey {
        CacheKey {
            kernel_name: name.to_string(),
            device_id: "test-device-0".to_string(),
            compiler_options: "-O2".to_string(),
        }
    }

    fn test_entry(hash: u64) -> CacheEntry {
        CacheEntry {
            binary_data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            source_hash: hash,
            timestamp: 1_700_000_000,
            device_name: "TestGPU".to_string(),
        }
    }

    #[test]
    fn cache_hit_returns_stored_binary() {
        let cache = KernelCache::with_config(None, true);
        let key = test_key("matmul");
        let entry = test_entry(42);
        cache.insert(key.clone(), entry.clone());

        let result = cache.get(&key, 42);
        assert!(result.is_some());
        assert_eq!(result.unwrap().binary_data, entry.binary_data);
    }

    #[test]
    fn cache_miss_returns_none() {
        let cache = KernelCache::with_config(None, true);
        let key = test_key("softmax");
        assert!(cache.get(&key, 1).is_none());
    }

    #[test]
    fn source_change_invalidates_entry() {
        let cache = KernelCache::with_config(None, true);
        let key = test_key("relu");
        cache.insert(key.clone(), test_entry(10));

        // Different source hash → miss + eviction.
        assert!(cache.get(&key, 99).is_none());
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn different_compiler_options_are_separate() {
        let cache = KernelCache::with_config(None, true);
        let key_a = CacheKey {
            kernel_name: "gemm".into(),
            device_id: "d0".into(),
            compiler_options: "-O0".into(),
        };
        let key_b = CacheKey {
            kernel_name: "gemm".into(),
            device_id: "d0".into(),
            compiler_options: "-O3".into(),
        };
        cache.insert(key_a.clone(), test_entry(1));
        cache.insert(key_b.clone(), test_entry(1));

        assert!(cache.get(&key_a, 1).is_some());
        assert!(cache.get(&key_b, 1).is_some());
        assert_eq!(cache.stats().entries, 2);
    }

    #[test]
    fn disabled_cache_always_misses() {
        let cache = KernelCache::with_config(None, false);
        let key = test_key("conv");
        cache.insert(key.clone(), test_entry(5));

        assert!(cache.get(&key, 5).is_none());
        assert!(!cache.is_enabled());
    }

    #[test]
    fn cache_clear_removes_all() {
        let cache = KernelCache::with_config(None, true);
        cache.insert(test_key("a"), test_entry(1));
        cache.insert(test_key("b"), test_entry(2));
        assert_eq!(cache.stats().entries, 2);

        cache.clear();
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().evictions, 2);
    }

    #[test]
    fn stats_track_hits_and_misses() {
        let cache = KernelCache::with_config(None, true);
        let key = test_key("pool");
        cache.insert(key.clone(), test_entry(7));

        let _ = cache.get(&key, 7); // hit
        let _ = cache.get(&key, 7); // hit
        let _ = cache.get(&test_key("missing"), 1); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn invalidate_removes_single_entry() {
        let cache = KernelCache::with_config(None, true);
        let key = test_key("norm");
        cache.insert(key.clone(), test_entry(3));
        cache.invalidate(&key);

        assert!(cache.get(&key, 3).is_none());
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn get_or_compile_hits_cache() {
        let cache = KernelCache::with_config(None, true);
        let key = test_key("add");
        cache.insert(key.clone(), test_entry(10));

        let result: Result<CacheEntry, String> = cache.get_or_compile(&key, 10, || {
            panic!("should not compile on cache hit");
        });
        assert!(result.is_ok());
    }

    #[test]
    fn get_or_compile_compiles_on_miss() {
        let cache = KernelCache::with_config(None, true);
        let key = test_key("sub");
        let result: Result<CacheEntry, String> =
            cache.get_or_compile(&key, 20, || Ok(test_entry(20)));
        assert!(result.is_ok());
        // Now cached.
        assert!(cache.get(&key, 20).is_some());
    }

    #[test]
    fn hash_source_deterministic() {
        let a = hash_source("__kernel void f() {}");
        let b = hash_source("__kernel void f() {}");
        assert_eq!(a, b);
    }

    #[test]
    fn hash_source_differs_for_different_inputs() {
        let a = hash_source("__kernel void f() {}");
        let b = hash_source("__kernel void g() {}");
        assert_ne!(a, b);
    }

    #[test]
    fn thread_safe_concurrent_access() {
        use std::thread;

        let cache = KernelCache::with_config(None, true);
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let cache = cache.clone();
                thread::spawn(move || {
                    let key = test_key(&format!("kernel_{i}"));
                    cache.insert(key.clone(), test_entry(i));
                    cache.get(&key, i)
                })
            })
            .collect();

        for h in handles {
            assert!(h.join().unwrap().is_some());
        }
        assert_eq!(cache.stats().entries, 8);
    }

    #[test]
    fn cache_key_equality() {
        let a = test_key("x");
        let b = test_key("x");
        assert_eq!(a, b);
    }

    #[test]
    fn cache_entry_serialization_roundtrip() {
        let entry = test_entry(42);
        let json = serde_json::to_string(&entry).unwrap();
        let decoded: CacheEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.binary_data, entry.binary_data);
        assert_eq!(decoded.source_hash, entry.source_hash);
    }
}
