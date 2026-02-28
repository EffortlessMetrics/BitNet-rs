//! Disk-based model cache with configurable eviction policies.
//!
//! Provides [`ModelCache`] for caching model files on disk with SHA-256
//! integrity verification, memory-mapped loading, and concurrent access
//! via read-write locking.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ── Configuration ─────────────────────────────────────────────────────────

/// Eviction policy for the model cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used — evict the entry accessed longest ago.
    LRU,
    /// Least Frequently Used — evict the entry with fewest accesses.
    LFU,
    /// First In, First Out — evict the oldest entry by creation time.
    FIFO,
    /// Size-Based — evict the largest entry first.
    SizeBased,
    /// Time-To-Live — evict entries older than `ttl` duration.
    TTL,
}

/// Configuration for [`ModelCache`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum total size of cached model files in bytes.
    pub max_size_bytes: u64,
    /// Directory where cached models and the index are stored.
    pub cache_dir: PathBuf,
    /// Which eviction policy to use when the cache is full.
    pub eviction_policy: EvictionPolicy,
    /// Whether to verify SHA-256 checksums on retrieval.
    pub integrity_check: bool,
    /// TTL duration in seconds (only used with [`EvictionPolicy::TTL`]).
    pub ttl_secs: Option<u64>,
}

impl CacheConfig {
    /// Create a config with sensible defaults for the given directory.
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            max_size_bytes: 10 * 1024 * 1024 * 1024, // 10 GiB
            cache_dir: cache_dir.into(),
            eviction_policy: EvictionPolicy::LRU,
            integrity_check: true,
            ttl_secs: None,
        }
    }
}

// ── Cache entry metadata ──────────────────────────────────────────────────

/// Metadata for a single cached model file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// SHA-256 hex digest of the model data.
    pub model_hash: String,
    /// Size of the cached file in bytes.
    pub size: u64,
    /// Unix timestamp of last access.
    pub last_accessed: u64,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Unix timestamp of when this entry was created.
    pub created_at: u64,
    /// Relative filename within the cache directory.
    pub filename: String,
}

// ── Persisted index ───────────────────────────────────────────────────────

/// On-disk JSON index mapping model keys to their [`CacheEntry`] metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheIndex {
    pub entries: HashMap<String, CacheEntry>,
}

// ── Statistics ────────────────────────────────────────────────────────────

/// Runtime hit/miss statistics for the cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_bytes: u64,
}

impl CacheStats {
    /// Hit rate as a fraction in `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        #[allow(clippy::cast_precision_loss)]
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    /// Miss rate as a fraction in `[0.0, 1.0]`.
    pub fn miss_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        #[allow(clippy::cast_precision_loss)]
        if total == 0 { 0.0 } else { self.misses as f64 / total as f64 }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────

/// Errors produced by [`ModelCache`] operations.
#[derive(Debug)]
pub enum CacheError {
    Io(io::Error),
    Json(serde_json::Error),
    IntegrityMismatch { expected: String, actual: String },
    EntryNotFound(String),
    EntryTooLarge { size: u64, max: u64 },
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "cache I/O error: {e}"),
            Self::Json(e) => write!(f, "cache index error: {e}"),
            Self::IntegrityMismatch { expected, actual } => {
                write!(
                    f,
                    "integrity check failed: expected {expected}, got {actual}"
                )
            }
            Self::EntryNotFound(key) => {
                write!(f, "cache entry not found: {key}")
            }
            Self::EntryTooLarge { size, max } => {
                write!(f, "entry {size} bytes exceeds max {max} bytes")
            }
        }
    }
}

impl std::error::Error for CacheError {}

impl From<io::Error> for CacheError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for CacheError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

// ── ModelCache ────────────────────────────────────────────────────────────

/// Disk-based model cache with eviction, integrity, and mmap support.
pub struct ModelCache {
    config: CacheConfig,
    index: RwLock<CacheIndex>,
    stats: RwLock<CacheStats>,
}

impl ModelCache {
    /// Open (or create) a cache in the configured directory.
    pub fn open(config: CacheConfig) -> Result<Self, CacheError> {
        fs::create_dir_all(&config.cache_dir)?;
        let index = Self::load_index(&config.cache_dir)?;
        let total_bytes =
            index.entries.values().map(|e| e.size).sum::<u64>();
        Ok(Self {
            config,
            index: RwLock::new(index),
            stats: RwLock::new(CacheStats {
                total_bytes,
                ..Default::default()
            }),
        })
    }

    /// Insert model data into the cache under `key`.
    pub fn put(
        &self,
        key: &str,
        data: &[u8],
    ) -> Result<(), CacheError> {
        let size = data.len() as u64;
        if size > self.config.max_size_bytes {
            return Err(CacheError::EntryTooLarge {
                size,
                max: self.config.max_size_bytes,
            });
        }

        // Evict until there is room.
        self.evict_until_fits(size)?;

        let hash = sha256_hex(data);
        let filename = format!("{hash}.bin");
        let path = self.config.cache_dir.join(&filename);
        let mut file = fs::File::create(&path)?;
        file.write_all(data)?;
        file.sync_all()?;

        let now = now_unix();
        let entry = CacheEntry {
            model_hash: hash,
            size,
            last_accessed: now,
            access_count: 0,
            created_at: now,
            filename,
        };

        {
            let mut idx = self.index.write().unwrap();
            idx.entries.insert(key.to_owned(), entry);
            Self::persist_index(&self.config.cache_dir, &idx)?;
            drop(idx);
        }
        {
            let mut st = self.stats.write().unwrap();
            st.total_bytes += size;
        }
        Ok(())
    }

    /// Retrieve model data for `key`, verifying integrity if configured.
    pub fn get(&self, key: &str) -> Result<Vec<u8>, CacheError> {
        let (file_path, expected_hash) = {
            let mut idx = self.index.write().unwrap();
            let entry = idx
                .entries
                .get_mut(key)
                .ok_or_else(|| CacheError::EntryNotFound(key.into()))?;
            entry.last_accessed = now_unix();
            entry.access_count += 1;
            let p = self.config.cache_dir.join(&entry.filename);
            let h = entry.model_hash.clone();
            Self::persist_index(&self.config.cache_dir, &idx)?;
            drop(idx);
            (p, h)
        };

        let data = fs::read(&file_path)?;

        if self.config.integrity_check {
            let actual = sha256_hex(&data);
            if actual != expected_hash {
                self.stats.write().unwrap().misses += 1;
                return Err(CacheError::IntegrityMismatch {
                    expected: expected_hash,
                    actual,
                });
            }
        }

        self.stats.write().unwrap().hits += 1;
        Ok(data)
    }

    /// Memory-map a cached model for zero-copy access.
    ///
    /// # Safety
    /// The caller must ensure the file is not modified while mapped.
    pub fn mmap(&self, key: &str) -> Result<Mmap, CacheError> {
        let path = {
            let mut idx = self.index.write().unwrap();
            let entry = idx
                .entries
                .get_mut(key)
                .ok_or_else(|| CacheError::EntryNotFound(key.into()))?;
            entry.last_accessed = now_unix();
            entry.access_count += 1;
            let p = self.config.cache_dir.join(&entry.filename);
            Self::persist_index(&self.config.cache_dir, &idx)?;
            drop(idx);
            p
        };
        let file = fs::File::open(&path)?;
        // SAFETY: the caller guarantees no concurrent mutation.
        let map = unsafe { Mmap::map(&file)? };
        self.stats.write().unwrap().hits += 1;
        Ok(map)
    }

    /// Remove a single entry from the cache.
    pub fn remove(&self, key: &str) -> Result<(), CacheError> {
        let mut idx = self.index.write().unwrap();
        if let Some(entry) = idx.entries.remove(key) {
            let path = self.config.cache_dir.join(&entry.filename);
            let _ = fs::remove_file(path);
            let mut st = self.stats.write().unwrap();
            st.total_bytes = st.total_bytes.saturating_sub(entry.size);
            drop(st);
            Self::persist_index(&self.config.cache_dir, &idx)?;
        }
        drop(idx);
        Ok(())
    }

    /// Check whether `key` is present in the cache.
    pub fn contains(&self, key: &str) -> bool {
        self.index.read().unwrap().entries.contains_key(key)
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.index.read().unwrap().entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.index.read().unwrap().entries.is_empty()
    }

    /// List all keys currently in the cache.
    pub fn keys(&self) -> Vec<String> {
        self.index.read().unwrap().entries.keys().cloned().collect()
    }

    /// Return a snapshot of the current statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Warm the cache by preloading data for the given keys.
    ///
    /// Entries that are already present are touched (`last_accessed` updated).
    /// Items in `preload` that map to `Some(data)` are inserted.
    pub fn warm(
        &self,
        preload: &[(&str, Option<&[u8]>)],
    ) -> Result<(), CacheError> {
        for &(key, data) in preload {
            if self.contains(key) {
                // Touch the entry.
                let mut idx = self.index.write().unwrap();
                if let Some(e) = idx.entries.get_mut(key) {
                    e.last_accessed = now_unix();
                }
                Self::persist_index(&self.config.cache_dir, &idx)?;
                drop(idx);
            } else if let Some(d) = data {
                self.put(key, d)?;
            }
        }
        Ok(())
    }

    /// Purge all entries from the cache, removing files and resetting stats.
    pub fn clear(&self) -> Result<(), CacheError> {
        let mut idx = self.index.write().unwrap();
        for entry in idx.entries.values() {
            let path = self.config.cache_dir.join(&entry.filename);
            let _ = fs::remove_file(path);
        }
        idx.entries.clear();
        Self::persist_index(&self.config.cache_dir, &idx)?;
        drop(idx);
        let mut st = self.stats.write().unwrap();
        st.total_bytes = 0;
        st.evictions = 0;
        drop(st);
        Ok(())
    }

    /// Verify integrity of every cached entry. Returns keys whose files
    /// are missing or have a hash mismatch.
    pub fn verify_all(&self) -> Vec<String> {
        let idx = self.index.read().unwrap();
        let entries: Vec<_> = idx
            .entries
            .iter()
            .map(|(k, e)| (k.clone(), e.filename.clone(), e.model_hash.clone()))
            .collect();
        drop(idx);
        let mut bad = Vec::new();
        for (key, filename, expected_hash) in &entries {
            let path = self.config.cache_dir.join(filename);
            match fs::read(&path) {
                Ok(data) => {
                    if sha256_hex(&data) != *expected_hash {
                        bad.push(key.clone());
                    }
                }
                Err(_) => bad.push(key.clone()),
            }
        }
        bad
    }

    /// Get metadata for a specific key without touching access stats.
    pub fn entry_metadata(
        &self,
        key: &str,
    ) -> Option<CacheEntry> {
        self.index.read().unwrap().entries.get(key).cloned()
    }

    // ── private helpers ──────────────────────────────────────────────

    fn index_path(dir: &Path) -> PathBuf {
        dir.join("cache_index.json")
    }

    fn load_index(dir: &Path) -> Result<CacheIndex, CacheError> {
        let path = Self::index_path(dir);
        if !path.exists() {
            return Ok(CacheIndex::default());
        }
        let mut buf = String::new();
        fs::File::open(&path)?.read_to_string(&mut buf)?;
        Ok(serde_json::from_str(&buf)?)
    }

    fn persist_index(
        dir: &Path,
        index: &CacheIndex,
    ) -> Result<(), CacheError> {
        let path = Self::index_path(dir);
        let json = serde_json::to_string_pretty(index)?;
        fs::write(&path, json.as_bytes())?;
        Ok(())
    }

    /// Evict entries according to the configured policy until
    /// `needed` bytes can fit within `max_size_bytes`.
    fn evict_until_fits(
        &self,
        needed: u64,
    ) -> Result<(), CacheError> {
        loop {
            let current: u64 = {
                let idx = self.index.read().unwrap();
                idx.entries.values().map(|e| e.size).sum()
            };
            if current + needed <= self.config.max_size_bytes {
                break;
            }
            let victim = self.pick_victim()?;
            self.remove(&victim)?;
            self.stats.write().unwrap().evictions += 1;
        }
        Ok(())
    }

    /// Select a victim key according to the active eviction policy.
    fn pick_victim(&self) -> Result<String, CacheError> {
        let idx = self.index.read().unwrap();
        if idx.entries.is_empty() {
            return Err(CacheError::EntryNotFound(
                "no entries to evict".into(),
            ));
        }
        let key = match self.config.eviction_policy {
            EvictionPolicy::LRU => idx
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone()),
            EvictionPolicy::LFU => idx
                .entries
                .iter()
                .min_by_key(|(_, e)| e.access_count)
                .map(|(k, _)| k.clone()),
            EvictionPolicy::FIFO => idx
                .entries
                .iter()
                .min_by_key(|(_, e)| e.created_at)
                .map(|(k, _)| k.clone()),
            EvictionPolicy::SizeBased => idx
                .entries
                .iter()
                .max_by_key(|(_, e)| e.size)
                .map(|(k, _)| k.clone()),
            EvictionPolicy::TTL => {
                let now = now_unix();
                let ttl = self.config.ttl_secs.unwrap_or(3600);
                idx.entries
                    .iter()
                    .filter(|(_, e)| now.saturating_sub(e.created_at) > ttl)
                    .min_by_key(|(_, e)| e.created_at)
                    .or_else(|| {
                        // fallback: oldest entry if none expired
                        idx.entries
                            .iter()
                            .min_by_key(|(_, e)| e.created_at)
                    })
                    .map(|(k, _)| k.clone())
            }
        };
        key.ok_or_else(|| {
            CacheError::EntryNotFound("no victim found".into())
        })
    }
}

// ── Utility functions ─────────────────────────────────────────────────────

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config(dir: &Path) -> CacheConfig {
        CacheConfig {
            max_size_bytes: 4096,
            cache_dir: dir.to_path_buf(),
            eviction_policy: EvictionPolicy::LRU,
            integrity_check: true,
            ttl_secs: None,
        }
    }

    fn open_cache(dir: &Path) -> ModelCache {
        ModelCache::open(test_config(dir)).unwrap()
    }

    // ── basic CRUD ───────────────────────────────────────────────

    #[test]
    fn put_and_get() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"hello model").unwrap();
        let data = cache.get("m1").unwrap();
        assert_eq!(data, b"hello model");
    }

    #[test]
    fn get_missing_key_returns_error() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        assert!(cache.get("nope").is_err());
    }

    #[test]
    fn contains_after_put() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        assert!(!cache.contains("m1"));
        cache.put("m1", b"data").unwrap();
        assert!(cache.contains("m1"));
    }

    #[test]
    fn remove_entry() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"data").unwrap();
        cache.remove("m1").unwrap();
        assert!(!cache.contains("m1"));
    }

    #[test]
    fn remove_missing_is_ok() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        assert!(cache.remove("nope").is_ok());
    }

    #[test]
    fn len_and_is_empty() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        cache.put("a", b"1").unwrap();
        cache.put("b", b"2").unwrap();
        assert_eq!(cache.len(), 2);
        assert!(!cache.is_empty());
    }

    #[test]
    fn keys_returns_all_keys() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("x", b"1").unwrap();
        cache.put("y", b"2").unwrap();
        let mut keys = cache.keys();
        keys.sort();
        assert_eq!(keys, vec!["x", "y"]);
    }

    #[test]
    fn put_overwrites_existing_key() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"old").unwrap();
        cache.put("m1", b"new").unwrap();
        assert_eq!(cache.get("m1").unwrap(), b"new");
    }

    #[test]
    fn put_empty_data() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("empty", b"").unwrap();
        assert_eq!(cache.get("empty").unwrap(), b"");
    }

    // ── clear ────────────────────────────────────────────────────

    #[test]
    fn clear_removes_all_entries() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("a", b"1").unwrap();
        cache.put("b", b"2").unwrap();
        cache.clear().unwrap();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().total_bytes, 0);
    }

    // ── integrity ────────────────────────────────────────────────

    #[test]
    fn integrity_check_detects_corruption() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"good data").unwrap();

        // Corrupt the file on disk.
        let idx = cache.index.read().unwrap();
        let entry = &idx.entries["m1"];
        let path = tmp.path().join(&entry.filename);
        drop(idx);
        fs::write(&path, b"bad data").unwrap();

        let err = cache.get("m1").unwrap_err();
        assert!(
            matches!(err, CacheError::IntegrityMismatch { .. }),
            "expected IntegrityMismatch, got {err}"
        );
    }

    #[test]
    fn integrity_check_disabled_skips_hash() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.integrity_check = false;
        let cache = ModelCache::open(cfg).unwrap();
        cache.put("m1", b"original").unwrap();

        // Corrupt the file.
        let idx = cache.index.read().unwrap();
        let path = tmp.path().join(&idx.entries["m1"].filename);
        drop(idx);
        fs::write(&path, b"tampered").unwrap();

        // Should succeed because integrity_check is off.
        let data = cache.get("m1").unwrap();
        assert_eq!(data, b"tampered");
    }

    #[test]
    fn verify_all_reports_bad_entries() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("good", b"ok").unwrap();
        cache.put("bad", b"data").unwrap();

        let idx = cache.index.read().unwrap();
        let path = tmp.path().join(&idx.entries["bad"].filename);
        drop(idx);
        fs::write(&path, b"corrupted").unwrap();

        let bad = cache.verify_all();
        assert!(bad.contains(&"bad".to_owned()));
        assert!(!bad.contains(&"good".to_owned()));
    }

    #[test]
    fn verify_all_reports_missing_files() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"data").unwrap();

        let idx = cache.index.read().unwrap();
        let path = tmp.path().join(&idx.entries["m1"].filename);
        drop(idx);
        fs::remove_file(path).unwrap();

        let bad = cache.verify_all();
        assert!(bad.contains(&"m1".to_owned()));
    }

    // ── SHA-256 ──────────────────────────────────────────────────

    #[test]
    fn sha256_hex_deterministic() {
        let h1 = sha256_hex(b"hello");
        let h2 = sha256_hex(b"hello");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn sha256_hex_different_inputs() {
        assert_ne!(sha256_hex(b"a"), sha256_hex(b"b"));
    }

    #[test]
    fn sha256_hex_empty() {
        let h = sha256_hex(b"");
        assert_eq!(h.len(), 64);
    }

    // ── eviction: LRU ────────────────────────────────────────────

    #[test]
    fn lru_eviction_removes_least_recently_used() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.max_size_bytes = 30;
        cfg.eviction_policy = EvictionPolicy::LRU;
        let cache = ModelCache::open(cfg).unwrap();

        cache.put("a", b"1234567890").unwrap(); // 10 bytes
        cache.put("b", b"1234567890").unwrap(); // 10 bytes
        // Force distinct timestamps to avoid same-second ambiguity.
        {
            let mut idx = cache.index.write().unwrap();
            idx.entries.get_mut("a").unwrap().last_accessed = 200;
            idx.entries.get_mut("b").unwrap().last_accessed = 100;
        }
        // Insert "c" — should evict "b" (last_accessed=100).
        cache.put("c", b"12345678901").unwrap(); // 11 bytes
        assert!(cache.contains("a"));
        assert!(!cache.contains("b"));
        assert!(cache.contains("c"));
    }

    // ── eviction: LFU ────────────────────────────────────────────

    #[test]
    fn lfu_eviction_removes_least_frequently_used() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.max_size_bytes = 30;
        cfg.eviction_policy = EvictionPolicy::LFU;
        let cache = ModelCache::open(cfg).unwrap();

        cache.put("a", b"1234567890").unwrap();
        cache.put("b", b"1234567890").unwrap();
        // Give "a" more accesses so "b" is least-frequently-used.
        {
            let mut idx = cache.index.write().unwrap();
            idx.entries.get_mut("a").unwrap().access_count = 10;
            idx.entries.get_mut("b").unwrap().access_count = 0;
        }
        // Insert "c" — should evict "b" (fewer accesses).
        cache.put("c", b"12345678901").unwrap();
        assert!(cache.contains("a"));
        assert!(!cache.contains("b"));
        assert!(cache.contains("c"));
    }

    // ── eviction: FIFO ───────────────────────────────────────────

    #[test]
    fn fifo_eviction_removes_oldest() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.max_size_bytes = 30;
        cfg.eviction_policy = EvictionPolicy::FIFO;
        let cache = ModelCache::open(cfg).unwrap();

        cache.put("a", b"1234567890").unwrap();
        cache.put("b", b"1234567890").unwrap();
        // Force distinct creation timestamps.
        {
            let mut idx = cache.index.write().unwrap();
            idx.entries.get_mut("a").unwrap().created_at = 100;
            idx.entries.get_mut("b").unwrap().created_at = 200;
        }
        cache.put("c", b"12345678901").unwrap();
        // "a" was created first (100), so it should be evicted.
        assert!(!cache.contains("a"));
        assert!(cache.contains("b"));
        assert!(cache.contains("c"));
    }

    // ── eviction: SizeBased ──────────────────────────────────────

    #[test]
    fn size_based_eviction_removes_largest() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.max_size_bytes = 40;
        cfg.eviction_policy = EvictionPolicy::SizeBased;
        let cache = ModelCache::open(cfg).unwrap();

        cache.put("small", b"abc").unwrap(); // 3 bytes
        cache.put("large", b"1234567890123456789012345").unwrap(); // 25
        cache.put("medium", b"12345678901234").unwrap(); // 14 — needs room
        // "large" (25 bytes) should be evicted first.
        assert!(cache.contains("small"));
        assert!(!cache.contains("large"));
        assert!(cache.contains("medium"));
    }

    // ── eviction: TTL ────────────────────────────────────────────

    #[test]
    fn ttl_eviction_prefers_expired_entries() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.max_size_bytes = 30;
        cfg.eviction_policy = EvictionPolicy::TTL;
        cfg.ttl_secs = Some(0); // everything is immediately expired
        let cache = ModelCache::open(cfg).unwrap();

        cache.put("a", b"1234567890").unwrap();
        cache.put("b", b"1234567890").unwrap();
        cache.put("c", b"12345678901").unwrap();
        // Both "a" and "b" are expired; oldest should be evicted.
        assert_eq!(cache.len(), 2);
        assert!(cache.contains("c"));
    }

    // ── entry too large ──────────────────────────────────────────

    #[test]
    fn put_rejects_entry_exceeding_max_size() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.max_size_bytes = 10;
        let cache = ModelCache::open(cfg).unwrap();

        let big = vec![0u8; 20];
        let err = cache.put("big", &big).unwrap_err();
        assert!(matches!(err, CacheError::EntryTooLarge { .. }));
    }

    // ── memory-mapped access ─────────────────────────────────────

    #[test]
    fn mmap_returns_correct_data() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"mmap data here").unwrap();
        let map = cache.mmap("m1").unwrap();
        assert_eq!(&map[..], b"mmap data here");
    }

    #[test]
    fn mmap_missing_key_errors() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        assert!(cache.mmap("nope").is_err());
    }

    // ── cache warming ────────────────────────────────────────────

    #[test]
    fn warm_inserts_new_entries() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache
            .warm(&[("a", Some(b"data_a")), ("b", Some(b"data_b"))])
            .unwrap();
        assert!(cache.contains("a"));
        assert!(cache.contains("b"));
    }

    #[test]
    fn warm_touches_existing_entries() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("a", b"data").unwrap();
        let before = cache
            .entry_metadata("a")
            .unwrap()
            .last_accessed;
        // Small sleep to ensure timestamp advances.
        std::thread::sleep(std::time::Duration::from_millis(50));
        cache.warm(&[("a", None)]).unwrap();
        let after = cache
            .entry_metadata("a")
            .unwrap()
            .last_accessed;
        assert!(after >= before);
    }

    #[test]
    fn warm_skips_none_for_missing_key() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.warm(&[("missing", None)]).unwrap();
        assert!(!cache.contains("missing"));
    }

    // ── statistics ───────────────────────────────────────────────

    #[test]
    fn stats_initial_zeros() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        let s = cache.stats();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.evictions, 0);
    }

    #[test]
    fn stats_hit_on_get() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"data").unwrap();
        let _ = cache.get("m1").unwrap();
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn stats_miss_on_integrity_failure() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"good").unwrap();

        let idx = cache.index.read().unwrap();
        let path = tmp.path().join(&idx.entries["m1"].filename);
        drop(idx);
        fs::write(&path, b"corrupt").unwrap();

        let _ = cache.get("m1"); // will fail
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn stats_eviction_count() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.max_size_bytes = 20;
        let cache = ModelCache::open(cfg).unwrap();
        cache.put("a", b"1234567890").unwrap();
        cache.put("b", b"1234567890").unwrap();
        cache.put("c", b"1234567890").unwrap(); // triggers eviction
        assert!(cache.stats().evictions > 0);
    }

    #[test]
    fn stats_total_bytes_tracks_puts() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("a", b"12345").unwrap(); // 5
        cache.put("b", b"67890").unwrap(); // 5
        assert_eq!(cache.stats().total_bytes, 10);
    }

    #[test]
    fn stats_total_bytes_decreases_on_remove() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("a", b"12345").unwrap();
        cache.remove("a").unwrap();
        assert_eq!(cache.stats().total_bytes, 0);
    }

    #[test]
    fn stats_hit_rate_empty() {
        let s = CacheStats::default();
        assert!(s.hit_rate().abs() < f64::EPSILON);
        assert!(s.miss_rate().abs() < f64::EPSILON);
    }

    #[test]
    fn stats_hit_rate_calculation() {
        let s = CacheStats {
            hits: 3,
            misses: 1,
            evictions: 0,
            total_bytes: 0,
        };
        assert!((s.hit_rate() - 0.75).abs() < f64::EPSILON);
        assert!((s.miss_rate() - 0.25).abs() < f64::EPSILON);
    }

    // ── persistence ──────────────────────────────────────────────

    #[test]
    fn index_persists_across_reopens() {
        let tmp = TempDir::new().unwrap();
        {
            let cache = open_cache(tmp.path());
            cache.put("m1", b"persist me").unwrap();
        }
        let cache = open_cache(tmp.path());
        assert!(cache.contains("m1"));
        assert_eq!(cache.get("m1").unwrap(), b"persist me");
    }

    #[test]
    fn index_file_is_valid_json() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"data").unwrap();
        let json = fs::read_to_string(
            tmp.path().join("cache_index.json"),
        )
        .unwrap();
        let idx: CacheIndex = serde_json::from_str(&json).unwrap();
        assert!(idx.entries.contains_key("m1"));
    }

    // ── entry metadata ───────────────────────────────────────────

    #[test]
    fn entry_metadata_returns_none_for_missing() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        assert!(cache.entry_metadata("nope").is_none());
    }

    #[test]
    fn entry_metadata_has_correct_size() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"12345").unwrap();
        let meta = cache.entry_metadata("m1").unwrap();
        assert_eq!(meta.size, 5);
    }

    #[test]
    fn entry_metadata_hash_matches_data() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        let data = b"hash me";
        cache.put("m1", data).unwrap();
        let meta = cache.entry_metadata("m1").unwrap();
        assert_eq!(meta.model_hash, sha256_hex(data));
    }

    #[test]
    fn access_count_increments_on_get() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"data").unwrap();
        assert_eq!(
            cache.entry_metadata("m1").unwrap().access_count,
            0
        );
        let _ = cache.get("m1").unwrap();
        assert_eq!(
            cache.entry_metadata("m1").unwrap().access_count,
            1
        );
        let _ = cache.get("m1").unwrap();
        assert_eq!(
            cache.entry_metadata("m1").unwrap().access_count,
            2
        );
    }

    #[test]
    fn last_accessed_updates_on_get() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"data").unwrap();
        let before = cache
            .entry_metadata("m1")
            .unwrap()
            .last_accessed;
        std::thread::sleep(std::time::Duration::from_millis(50));
        let _ = cache.get("m1").unwrap();
        let after = cache
            .entry_metadata("m1")
            .unwrap()
            .last_accessed;
        assert!(after >= before);
    }

    // ── concurrent access ────────────────────────────────────────

    #[test]
    fn concurrent_reads_do_not_panic() {
        let tmp = TempDir::new().unwrap();
        let cache =
            std::sync::Arc::new(open_cache(tmp.path()));
        cache.put("m1", b"concurrent data").unwrap();
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let c = cache.clone();
                std::thread::spawn(move || {
                    for _ in 0..10 {
                        let _ = c.get("m1");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(cache.stats().hits > 0);
    }

    #[test]
    fn concurrent_puts_do_not_corrupt_index() {
        let tmp = TempDir::new().unwrap();
        let cache =
            std::sync::Arc::new(open_cache(tmp.path()));
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let c = cache.clone();
                std::thread::spawn(move || {
                    let key = format!("k{i}");
                    let data = format!("data{i}");
                    c.put(&key, data.as_bytes()).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(cache.len(), 8);
    }

    #[test]
    fn concurrent_mixed_operations() {
        let tmp = TempDir::new().unwrap();
        let cache =
            std::sync::Arc::new(open_cache(tmp.path()));
        cache.put("shared", b"data").unwrap();
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let c = cache.clone();
                std::thread::spawn(move || {
                    if i % 2 == 0 {
                        let _ = c.get("shared");
                    } else {
                        let k = format!("new{i}");
                        let _ = c.put(&k, b"new");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Cache should be consistent.
        assert!(!cache.is_empty());
    }

    // ── config / policy enums ────────────────────────────────────

    #[test]
    fn cache_config_new_defaults() {
        let cfg = CacheConfig::new("/tmp/test");
        assert_eq!(cfg.eviction_policy, EvictionPolicy::LRU);
        assert!(cfg.integrity_check);
        assert!(cfg.max_size_bytes > 0);
    }

    #[test]
    fn eviction_policy_serialization() {
        let json = serde_json::to_string(&EvictionPolicy::LFU).unwrap();
        let back: EvictionPolicy =
            serde_json::from_str(&json).unwrap();
        assert_eq!(back, EvictionPolicy::LFU);
    }

    #[test]
    fn cache_config_roundtrip() {
        let cfg = CacheConfig::new("/tmp/test_cache");
        let json = serde_json::to_string(&cfg).unwrap();
        let back: CacheConfig =
            serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_size_bytes, cfg.max_size_bytes);
    }

    // ── edge cases ───────────────────────────────────────────────

    #[test]
    fn multiple_evictions_until_fit() {
        let tmp = TempDir::new().unwrap();
        let mut cfg = test_config(tmp.path());
        cfg.max_size_bytes = 20;
        let cache = ModelCache::open(cfg).unwrap();

        cache.put("a", b"12345").unwrap(); // 5
        cache.put("b", b"12345").unwrap(); // 5
        cache.put("c", b"12345").unwrap(); // 5
        // Now at 15/20. Insert 10 → needs to evict 5.
        cache.put("d", b"1234567890").unwrap(); // 10
        assert!(cache.len() <= 3);
        assert!(cache.contains("d"));
    }

    #[test]
    fn clear_then_reuse() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("a", b"data").unwrap();
        cache.clear().unwrap();
        cache.put("b", b"new_data").unwrap();
        assert!(!cache.contains("a"));
        assert!(cache.contains("b"));
    }

    #[test]
    fn mmap_hit_counted_in_stats() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("m1", b"data").unwrap();
        let _ = cache.mmap("m1").unwrap();
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn stats_total_bytes_after_reopen() {
        let tmp = TempDir::new().unwrap();
        {
            let cache = open_cache(tmp.path());
            cache.put("a", b"12345").unwrap();
            cache.put("b", b"67890").unwrap();
        }
        let cache = open_cache(tmp.path());
        assert_eq!(cache.stats().total_bytes, 10);
    }

    #[test]
    fn error_display_formats() {
        let e = CacheError::EntryNotFound("k1".into());
        assert!(e.to_string().contains("k1"));
        let e = CacheError::EntryTooLarge { size: 100, max: 50 };
        assert!(e.to_string().contains("100"));
        let e = CacheError::IntegrityMismatch {
            expected: "aaa".into(),
            actual: "bbb".into(),
        };
        assert!(e.to_string().contains("aaa"));
    }

    #[test]
    fn same_data_different_keys() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        cache.put("k1", b"same").unwrap();
        cache.put("k2", b"same").unwrap();
        assert_eq!(cache.get("k1").unwrap(), cache.get("k2").unwrap());
    }

    #[test]
    fn binary_data_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        let data: Vec<u8> = (0..=255).collect();
        cache.put("bin", &data).unwrap();
        assert_eq!(cache.get("bin").unwrap(), data);
    }

    #[test]
    fn large_key_name() {
        let tmp = TempDir::new().unwrap();
        let cache = open_cache(tmp.path());
        let key = "a".repeat(1000);
        cache.put(&key, b"data").unwrap();
        assert!(cache.contains(&key));
    }
}
