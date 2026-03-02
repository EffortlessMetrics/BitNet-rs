//! Persistent kernel compilation cache for Intel Arc A770.
//!
//! Caches compiled OpenCL program binaries to disk to avoid recompilation on
//! startup.  Provides versioning (device + driver), TTL-based invalidation,
//! integrity checking, and LRU eviction when the cache exceeds its size budget.

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// CacheKey
// ---------------------------------------------------------------------------

/// Uniquely identifies a compiled kernel binary by source, device, driver, and
/// compile options.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Deterministic 64-bit hash of the kernel source text.
    pub kernel_source_hash: u64,
    /// Device name string (e.g. `"Intel(R) Arc(TM) A770 Graphics"`).
    pub device_name: String,
    /// Driver version string (e.g. `"23.35.27191.42"`).
    pub driver_version: String,
    /// Compile options passed to `clBuildProgram` (e.g. `"-cl-mad-enable"`).
    pub compile_options: String,
}

impl CacheKey {
    pub fn new(
        kernel_source_hash: u64,
        device_name: impl Into<String>,
        driver_version: impl Into<String>,
        compile_options: impl Into<String>,
    ) -> Self {
        Self {
            kernel_source_hash,
            device_name: device_name.into(),
            driver_version: driver_version.into(),
            compile_options: compile_options.into(),
        }
    }

    /// Build a key by hashing raw kernel source text.
    pub fn from_source(
        source: &str,
        device_name: impl Into<String>,
        driver_version: impl Into<String>,
        compile_options: impl Into<String>,
    ) -> Self {
        Self::new(hash_source(source), device_name, driver_version, compile_options)
    }

    /// Deterministic filename for on-disk persistence.
    pub fn filename(&self) -> String {
        format!("{:016x}.bin", self.kernel_source_hash)
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheKey(hash={:016x}, device={}, driver={}, opts={})",
            self.kernel_source_hash, self.device_name, self.driver_version, self.compile_options,
        )
    }
}

// ---------------------------------------------------------------------------
// CacheEntry
// ---------------------------------------------------------------------------

/// A cached kernel binary together with bookkeeping metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cache key that produced this entry.
    pub key: CacheKey,
    /// The compiled binary data.
    pub binary_data: Vec<u8>,
    /// When the binary was compiled.
    pub compiled_at: SystemTime,
    /// Size of `binary_data` in bytes (convenience copy).
    pub size_bytes: usize,
    /// Number of cache hits for this entry.
    pub hit_count: u64,
}

impl CacheEntry {
    pub fn new(key: CacheKey, binary_data: Vec<u8>) -> Self {
        let size_bytes = binary_data.len();
        Self { key, binary_data, compiled_at: SystemTime::now(), size_bytes, hit_count: 0 }
    }

    /// Returns `true` if the entry has exceeded its time-to-live.
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.compiled_at.elapsed().map(|age| age > ttl).unwrap_or(false)
    }

    /// Record a cache hit.
    pub fn touch(&mut self) {
        self.hit_count += 1;
    }
}

// ---------------------------------------------------------------------------
// CacheConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for [`CompileCache`].
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Directory used for on-disk persistence (informational in CPU-ref mode).
    pub cache_dir: String,
    /// Maximum total size of all cached binaries in megabytes.
    pub max_size_mb: usize,
    /// Time-to-live in days. `0` means entries never expire.
    pub ttl_days: u32,
    /// Whether to run integrity checks on cached binaries.
    pub integrity_check: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: ".opencl_compile_cache".into(),
            max_size_mb: 256,
            ttl_days: 30,
            integrity_check: true,
        }
    }
}

impl CacheConfig {
    /// Maximum total size in bytes.
    pub fn max_size_bytes(&self) -> usize {
        self.max_size_mb * 1024 * 1024
    }

    /// TTL as a [`Duration`]. Returns `None` when `ttl_days == 0` (no expiry).
    pub fn ttl(&self) -> Option<Duration> {
        if self.ttl_days == 0 {
            None
        } else {
            Some(Duration::from_secs(u64::from(self.ttl_days) * 86_400))
        }
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Runtime statistics for the compile cache.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_size: usize,
    pub entry_count: usize,
}

impl CacheStats {
    /// Hit-rate as a fraction in `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheStats(hits={}, misses={}, evictions={}, entries={}, size={}B, \
             hit_rate={:.1}%)",
            self.hits,
            self.misses,
            self.evictions,
            self.entry_count,
            self.total_size,
            self.hit_rate() * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// IntegrityChecker
// ---------------------------------------------------------------------------

/// Validates cached binaries via hash verification and size checks.
#[derive(Debug, Clone)]
pub struct IntegrityChecker {
    /// Whether integrity checking is enabled.
    enabled: bool,
}

impl IntegrityChecker {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Compute a 64-bit integrity hash over `data`.
    pub fn compute_hash(data: &[u8]) -> u64 {
        use std::hash::DefaultHasher;
        let mut h = DefaultHasher::new();
        data.hash(&mut h);
        h.finish()
    }

    /// Validate that `data` matches `expected_hash` and `expected_size`.
    /// Returns `Ok(())` on success or a human-readable error string.
    pub fn validate(
        &self,
        data: &[u8],
        expected_hash: u64,
        expected_size: usize,
    ) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        if data.len() != expected_size {
            return Err(format!(
                "size mismatch: expected {} bytes, got {}",
                expected_size,
                data.len()
            ));
        }
        let actual_hash = Self::compute_hash(data);
        if actual_hash != expected_hash {
            return Err(format!(
                "hash mismatch: expected {:016x}, got {:016x}",
                expected_hash, actual_hash
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CacheWarmer
// ---------------------------------------------------------------------------

/// Pre-compiles frequently used kernels at startup.
///
/// In the CPU reference implementation the "compilation" is simulated by
/// invoking a user-supplied callback.
#[derive(Debug)]
pub struct CacheWarmer {
    /// Kernel sources to pre-compile, ordered by priority (highest first).
    entries: Vec<WarmEntry>,
}

/// A single kernel to pre-compile during warm-up.
#[derive(Debug, Clone)]
pub struct WarmEntry {
    /// Human-readable name for logging.
    pub name: String,
    /// Kernel source text.
    pub source: String,
    /// Priority (higher = compiled earlier).
    pub priority: u32,
}

impl CacheWarmer {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Register a kernel for pre-compilation.
    pub fn add(&mut self, name: impl Into<String>, source: impl Into<String>, priority: u32) {
        self.entries.push(WarmEntry { name: name.into(), source: source.into(), priority });
    }

    /// Return the registered entries sorted by descending priority.
    pub fn sorted_entries(&self) -> Vec<&WarmEntry> {
        let mut sorted: Vec<_> = self.entries.iter().collect();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));
        sorted
    }

    /// Run the warm-up phase.
    ///
    /// For each registered kernel (highest priority first), the `compile_fn`
    /// is called to produce a binary.  The binary is then stored into `cache`.
    /// Returns the number of kernels that were compiled (not already cached).
    pub fn warm(
        &self,
        cache: &mut CompileCache,
        device_name: &str,
        driver_version: &str,
        compile_options: &str,
        mut compile_fn: impl FnMut(&str) -> Vec<u8>,
    ) -> usize {
        let mut compiled = 0usize;
        for entry in self.sorted_entries() {
            let key =
                CacheKey::from_source(&entry.source, device_name, driver_version, compile_options);
            if cache.get(&key).is_some() {
                continue;
            }
            let binary = compile_fn(&entry.source);
            cache.put(key, binary);
            compiled += 1;
        }
        compiled
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for CacheWarmer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CompileCache
// ---------------------------------------------------------------------------

/// Persistent kernel compilation cache with LRU eviction.
///
/// This is a CPU reference implementation — all "persistence" is in-memory.
/// The design mirrors what a real implementation would do with file I/O.
pub struct CompileCache {
    config: CacheConfig,
    /// Map from key → (entry, integrity_hash).
    entries: HashMap<CacheKey, (CacheEntry, u64)>,
    /// LRU order: front = least recently used.
    lru_order: Vec<CacheKey>,
    stats: CacheStats,
    checker: IntegrityChecker,
}

impl CompileCache {
    pub fn new(config: CacheConfig) -> Self {
        let checker = IntegrityChecker::new(config.integrity_check);
        Self {
            config,
            entries: HashMap::new(),
            lru_order: Vec::new(),
            stats: CacheStats::default(),
            checker,
        }
    }

    /// Look up a cached binary. Returns `None` on miss or integrity failure.
    pub fn get(&mut self, key: &CacheKey) -> Option<&CacheEntry> {
        if !self.entries.contains_key(key) {
            self.stats.misses += 1;
            return None;
        }

        // Check TTL — compute in a block to release the borrow before mutating.
        if let Some(ttl) = self.config.ttl() {
            let expired = {
                let (entry, _) = self.entries.get(key).unwrap();
                entry.is_expired(ttl)
            };
            if expired {
                let size = self.entries.remove(key).map(|(e, _)| e.size_bytes).unwrap_or(0);
                self.lru_order.retain(|k| k != key);
                self.stats.total_size = self.stats.total_size.saturating_sub(size);
                self.stats.entry_count = self.entries.len();
                self.stats.misses += 1;
                return None;
            }
        }

        // Integrity check — compute in a block to release the borrow.
        if self.checker.enabled {
            let valid = {
                let (entry, hash) = self.entries.get(key).unwrap();
                let actual = IntegrityChecker::compute_hash(&entry.binary_data);
                actual == *hash && entry.binary_data.len() == entry.size_bytes
            };
            if !valid {
                let size = self.entries.remove(key).map(|(e, _)| e.size_bytes).unwrap_or(0);
                self.lru_order.retain(|k| k != key);
                self.stats.total_size = self.stats.total_size.saturating_sub(size);
                self.stats.entry_count = self.entries.len();
                self.stats.misses += 1;
                return None;
            }
        }

        // Touch the entry and update LRU order.
        {
            let (entry, _) = self.entries.get_mut(key).unwrap();
            entry.touch();
        }
        self.lru_order.retain(|k| k != key);
        self.lru_order.push(key.clone());
        self.stats.hits += 1;
        self.entries.get(key).map(|(e, _)| e)
    }

    /// Insert or replace a binary in the cache.
    pub fn put(&mut self, key: CacheKey, binary_data: Vec<u8>) {
        let hash = IntegrityChecker::compute_hash(&binary_data);
        let entry = CacheEntry::new(key.clone(), binary_data);
        let entry_size = entry.size_bytes;

        // Remove existing entry with the same key first.
        if let Some((old, _)) = self.entries.remove(&key) {
            self.stats.total_size = self.stats.total_size.saturating_sub(old.size_bytes);
            self.lru_order.retain(|k| k != &key);
        }

        // Evict until we have room.
        while self.stats.total_size + entry_size > self.config.max_size_bytes()
            && !self.entries.is_empty()
        {
            self.evict_lru();
        }
        while self.entries.len() >= self.max_entries() && !self.entries.is_empty() {
            self.evict_lru();
        }

        self.entries.insert(key.clone(), (entry, hash));
        self.lru_order.push(key);
        self.stats.total_size += entry_size;
        self.stats.entry_count = self.entries.len();
    }

    /// Remove a specific entry.
    pub fn invalidate(&mut self, key: &CacheKey) {
        if let Some((entry, _)) = self.entries.remove(key) {
            self.stats.total_size = self.stats.total_size.saturating_sub(entry.size_bytes);
            self.lru_order.retain(|k| k != key);
            self.stats.entry_count = self.entries.len();
        }
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
        self.stats.total_size = 0;
        self.stats.entry_count = 0;
    }

    /// Snapshot of current statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    // -- private helpers ----------------------------------------------------

    /// Practical upper bound on entries: 4096 or whatever fits in size budget.
    fn max_entries(&self) -> usize {
        4096
    }

    fn evict_lru(&mut self) {
        if let Some(victim) = self.lru_order.first().cloned() {
            if let Some((entry, _)) = self.entries.remove(&victim) {
                self.stats.total_size = self.stats.total_size.saturating_sub(entry.size_bytes);
                self.stats.evictions += 1;
            }
            self.lru_order.remove(0);
            self.stats.entry_count = self.entries.len();
        }
    }
}

impl fmt::Debug for CompileCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompileCache")
            .field("config", &self.config)
            .field("entries", &self.entries.len())
            .field("stats", &self.stats)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute a deterministic 64-bit hash of kernel source text.
pub fn hash_source(source: &str) -> u64 {
    use std::hash::DefaultHasher;
    let mut h = DefaultHasher::new();
    source.hash(&mut h);
    h.finish()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ------------------------------------------------------------

    fn make_key(id: u64) -> CacheKey {
        CacheKey::new(id, "Intel Arc A770", "23.35.27191", "-cl-mad-enable")
    }

    fn make_binary(size: usize) -> Vec<u8> {
        vec![0xAB; size]
    }

    fn default_cache() -> CompileCache {
        CompileCache::new(CacheConfig { ttl_days: 0, ..Default::default() })
    }

    fn small_cache(max_mb: usize) -> CompileCache {
        CompileCache::new(CacheConfig { max_size_mb: max_mb, ttl_days: 0, ..Default::default() })
    }

    // -----------------------------------------------------------------------
    // CacheKey tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_key_determinism() {
        let a = CacheKey::from_source("kernel void f(){}", "Dev", "1.0", "");
        let b = CacheKey::from_source("kernel void f(){}", "Dev", "1.0", "");
        assert_eq!(a, b, "identical source must produce identical keys");
    }

    #[test]
    fn test_cache_key_different_source() {
        let a = CacheKey::from_source("kernel void f(){}", "Dev", "1.0", "");
        let b = CacheKey::from_source("kernel void g(){}", "Dev", "1.0", "");
        assert_ne!(a, b);
    }

    #[test]
    fn test_cache_key_different_device() {
        let a = CacheKey::new(1, "A770", "1.0", "");
        let b = CacheKey::new(1, "A750", "1.0", "");
        assert_ne!(a, b);
    }

    #[test]
    fn test_cache_key_different_driver() {
        let a = CacheKey::new(1, "A770", "1.0", "");
        let b = CacheKey::new(1, "A770", "2.0", "");
        assert_ne!(a, b);
    }

    #[test]
    fn test_cache_key_different_options() {
        let a = CacheKey::new(1, "Dev", "1.0", "-O0");
        let b = CacheKey::new(1, "Dev", "1.0", "-O2");
        assert_ne!(a, b);
    }

    #[test]
    fn test_cache_key_display() {
        let k = make_key(0xFF);
        let s = format!("{k}");
        assert!(s.contains("00000000000000ff"));
        assert!(s.contains("Intel Arc A770"));
    }

    #[test]
    fn test_cache_key_filename() {
        let k = make_key(0xDEAD);
        assert_eq!(k.filename(), "000000000000dead.bin");
    }

    #[test]
    fn test_cache_key_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        let key = make_key(42);
        let h1 = {
            let mut h = DefaultHasher::new();
            key.hash(&mut h);
            h.finish()
        };
        let h2 = {
            let mut h = DefaultHasher::new();
            key.hash(&mut h);
            h.finish()
        };
        assert_eq!(h1, h2);
    }

    // -----------------------------------------------------------------------
    // CacheEntry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_entry_size_bytes() {
        let entry = CacheEntry::new(make_key(1), vec![0; 128]);
        assert_eq!(entry.size_bytes, 128);
    }

    #[test]
    fn test_entry_touch_increments_hit_count() {
        let mut entry = CacheEntry::new(make_key(1), vec![]);
        assert_eq!(entry.hit_count, 0);
        entry.touch();
        entry.touch();
        assert_eq!(entry.hit_count, 2);
    }

    #[test]
    fn test_entry_not_expired_generous_ttl() {
        let entry = CacheEntry::new(make_key(1), vec![]);
        assert!(!entry.is_expired(Duration::from_secs(3600)));
    }

    #[test]
    fn test_entry_expired_backdated() {
        let mut entry = CacheEntry::new(make_key(1), vec![]);
        entry.compiled_at = SystemTime::now() - Duration::from_secs(100);
        assert!(entry.is_expired(Duration::from_secs(1)));
    }

    // -----------------------------------------------------------------------
    // CacheConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = CacheConfig::default();
        assert_eq!(cfg.max_size_mb, 256);
        assert_eq!(cfg.ttl_days, 30);
        assert!(cfg.integrity_check);
    }

    #[test]
    fn test_config_max_size_bytes() {
        let cfg = CacheConfig { max_size_mb: 1, ..Default::default() };
        assert_eq!(cfg.max_size_bytes(), 1024 * 1024);
    }

    #[test]
    fn test_config_ttl_none_when_zero() {
        let cfg = CacheConfig { ttl_days: 0, ..Default::default() };
        assert!(cfg.ttl().is_none());
    }

    #[test]
    fn test_config_ttl_some_when_nonzero() {
        let cfg = CacheConfig { ttl_days: 7, ..Default::default() };
        assert_eq!(cfg.ttl().unwrap(), Duration::from_secs(7 * 86_400));
    }

    // -----------------------------------------------------------------------
    // Cache hit / miss
    // -----------------------------------------------------------------------

    #[test]
    fn test_put_and_get() {
        let mut cache = default_cache();
        let key = make_key(1);
        cache.put(key.clone(), make_binary(64));
        let entry = cache.get(&key).expect("should hit");
        assert_eq!(entry.binary_data, make_binary(64));
    }

    #[test]
    fn test_miss_on_empty_cache() {
        let mut cache = default_cache();
        assert!(cache.get(&make_key(999)).is_none());
    }

    #[test]
    fn test_miss_increments_stat() {
        let mut cache = default_cache();
        cache.get(&make_key(1));
        cache.get(&make_key(2));
        assert_eq!(cache.stats().misses, 2);
    }

    #[test]
    fn test_hit_increments_stat() {
        let mut cache = default_cache();
        let key = make_key(1);
        cache.put(key.clone(), make_binary(8));
        cache.get(&key);
        cache.get(&key);
        assert_eq!(cache.stats().hits, 2);
    }

    #[test]
    fn test_overwrite_existing_key() {
        let mut cache = default_cache();
        let key = make_key(1);
        cache.put(key.clone(), vec![1]);
        cache.put(key.clone(), vec![2]);
        let entry = cache.get(&key).unwrap();
        assert_eq!(entry.binary_data, vec![2]);
    }

    // -----------------------------------------------------------------------
    // LRU eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_lru_eviction_by_size() {
        // 1 MB cache → two 600 KB entries should trigger eviction.
        let mut cache = small_cache(1);
        cache.put(make_key(1), make_binary(600 * 1024));
        cache.put(make_key(2), make_binary(600 * 1024));
        // Key 1 should be evicted.
        assert!(cache.get(&make_key(1)).is_none());
        assert!(cache.get(&make_key(2)).is_some());
    }

    #[test]
    fn test_lru_eviction_preserves_recently_used() {
        let mut cache = small_cache(1);
        cache.put(make_key(1), make_binary(400 * 1024));
        cache.put(make_key(2), make_binary(400 * 1024));
        // Touch key 1 to make it recent.
        cache.get(&make_key(1));
        // Insert key 3 → should evict key 2 (LRU).
        cache.put(make_key(3), make_binary(400 * 1024));
        assert!(cache.get(&make_key(2)).is_none(), "key 2 should be evicted (LRU)");
        assert!(cache.get(&make_key(1)).is_some(), "key 1 should survive (recently used)");
    }

    #[test]
    fn test_eviction_stats_counted() {
        let mut cache = small_cache(1);
        cache.put(make_key(1), make_binary(600 * 1024));
        cache.put(make_key(2), make_binary(600 * 1024));
        assert!(cache.stats().evictions >= 1);
    }

    #[test]
    fn test_many_evictions() {
        let mut cache = small_cache(1);
        for i in 0..20 {
            cache.put(make_key(i), make_binary(200 * 1024));
        }
        // Only the most recent entries that fit in 1 MB should remain.
        assert!(cache.len() <= 5);
        assert!(cache.stats().evictions > 0);
    }

    // -----------------------------------------------------------------------
    // TTL-based expiration
    // -----------------------------------------------------------------------

    #[test]
    fn test_ttl_expiration() {
        let mut cache = CompileCache::new(CacheConfig { ttl_days: 1, ..Default::default() });
        let key = make_key(1);
        cache.put(key.clone(), make_binary(8));
        // Backdate the entry.
        if let Some((entry, _)) = cache.entries.get_mut(&key) {
            entry.compiled_at = SystemTime::now() - Duration::from_secs(2 * 86_400);
        }
        assert!(cache.get(&key).is_none(), "expired entry should be a miss");
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_no_ttl_never_expires() {
        let mut cache = CompileCache::new(CacheConfig { ttl_days: 0, ..Default::default() });
        let key = make_key(1);
        cache.put(key.clone(), make_binary(8));
        // Backdate heavily.
        if let Some((entry, _)) = cache.entries.get_mut(&key) {
            entry.compiled_at = SystemTime::now() - Duration::from_secs(365 * 86_400);
        }
        assert!(cache.get(&key).is_some(), "no-TTL entry must not expire");
    }

    // -----------------------------------------------------------------------
    // Integrity checking
    // -----------------------------------------------------------------------

    #[test]
    fn test_integrity_check_valid() {
        let data = vec![1, 2, 3, 4];
        let hash = IntegrityChecker::compute_hash(&data);
        let checker = IntegrityChecker::new(true);
        assert!(checker.validate(&data, hash, data.len()).is_ok());
    }

    #[test]
    fn test_integrity_check_hash_mismatch() {
        let checker = IntegrityChecker::new(true);
        let result = checker.validate(&[1, 2, 3], 0xBAD, 3);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("hash mismatch"));
    }

    #[test]
    fn test_integrity_check_size_mismatch() {
        let data = vec![1, 2, 3];
        let hash = IntegrityChecker::compute_hash(&data);
        let checker = IntegrityChecker::new(true);
        let result = checker.validate(&data, hash, 999);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("size mismatch"));
    }

    #[test]
    fn test_integrity_check_disabled() {
        let checker = IntegrityChecker::new(false);
        // Even with wrong hash and size, should pass when disabled.
        assert!(checker.validate(&[1, 2, 3], 0, 0).is_ok());
    }

    #[test]
    fn test_corrupted_binary_evicted_on_get() {
        let mut cache = CompileCache::new(CacheConfig {
            integrity_check: true,
            ttl_days: 0,
            ..Default::default()
        });
        let key = make_key(1);
        cache.put(key.clone(), vec![1, 2, 3, 4]);
        // Corrupt the stored binary.
        if let Some((entry, _)) = cache.entries.get_mut(&key) {
            entry.binary_data[0] = 0xFF;
        }
        assert!(cache.get(&key).is_none(), "corrupted binary should be evicted");
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_integrity_disabled_no_eviction_on_corruption() {
        let mut cache = CompileCache::new(CacheConfig {
            integrity_check: false,
            ttl_days: 0,
            ..Default::default()
        });
        let key = make_key(1);
        cache.put(key.clone(), vec![1, 2, 3, 4]);
        // Corrupt the stored binary.
        if let Some((entry, _)) = cache.entries.get_mut(&key) {
            entry.binary_data[0] = 0xFF;
        }
        assert!(
            cache.get(&key).is_some(),
            "with integrity disabled, corrupted entry should still be returned"
        );
    }

    // -----------------------------------------------------------------------
    // Cache warming
    // -----------------------------------------------------------------------

    #[test]
    fn test_warmer_priority_ordering() {
        let mut warmer = CacheWarmer::new();
        warmer.add("low", "src_low", 1);
        warmer.add("high", "src_high", 10);
        warmer.add("mid", "src_mid", 5);
        let sorted = warmer.sorted_entries();
        assert_eq!(sorted[0].name, "high");
        assert_eq!(sorted[1].name, "mid");
        assert_eq!(sorted[2].name, "low");
    }

    #[test]
    fn test_warmer_compiles_missing_kernels() {
        let mut cache = default_cache();
        let mut warmer = CacheWarmer::new();
        warmer.add("k1", "source1", 1);
        warmer.add("k2", "source2", 2);
        let compiled = warmer.warm(&mut cache, "Dev", "1.0", "", |_src| vec![0xBE, 0xEF]);
        assert_eq!(compiled, 2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_warmer_skips_cached_kernels() {
        let mut cache = default_cache();
        let mut warmer = CacheWarmer::new();
        warmer.add("k1", "source1", 1);
        // Pre-populate k1.
        let key = CacheKey::from_source("source1", "Dev", "1.0", "");
        cache.put(key, vec![0xAA]);
        let compiled = warmer.warm(&mut cache, "Dev", "1.0", "", |_src| vec![0xBE, 0xEF]);
        assert_eq!(compiled, 0, "already-cached kernel should be skipped");
    }

    #[test]
    fn test_warmer_empty() {
        let warmer = CacheWarmer::new();
        assert!(warmer.is_empty());
        assert_eq!(warmer.len(), 0);
    }

    // -----------------------------------------------------------------------
    // Stats tracking accuracy
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_total_size() {
        let mut cache = default_cache();
        cache.put(make_key(1), make_binary(100));
        cache.put(make_key(2), make_binary(200));
        assert_eq!(cache.stats().total_size, 300);
    }

    #[test]
    fn test_stats_entry_count() {
        let mut cache = default_cache();
        cache.put(make_key(1), make_binary(8));
        cache.put(make_key(2), make_binary(8));
        cache.put(make_key(3), make_binary(8));
        assert_eq!(cache.stats().entry_count, 3);
    }

    #[test]
    fn test_stats_hit_rate() {
        let mut cache = default_cache();
        let key = make_key(1);
        cache.put(key.clone(), make_binary(8));
        cache.get(&key); // hit
        cache.get(&make_key(999)); // miss
        let rate = cache.stats().hit_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_display() {
        let stats = CacheStats { hits: 10, misses: 5, ..Default::default() };
        let s = format!("{stats}");
        assert!(s.contains("hits=10"));
        assert!(s.contains("misses=5"));
    }

    // -----------------------------------------------------------------------
    // Concurrent access simulation
    // -----------------------------------------------------------------------

    #[test]
    fn test_sequential_concurrent_simulation() {
        // Simulate interleaved reads and writes from multiple "threads".
        let mut cache = default_cache();
        for i in 0..50 {
            cache.put(make_key(i), make_binary(64));
        }
        // "Thread A" reads even keys, "Thread B" reads odd keys.
        for i in (0..50).step_by(2) {
            cache.get(&make_key(i));
        }
        for i in (1..50).step_by(2) {
            cache.get(&make_key(i));
        }
        assert_eq!(cache.stats().hits, 50);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_cache_stats() {
        let cache = default_cache();
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
        assert_eq!(cache.stats().total_size, 0);
    }

    #[test]
    fn test_single_entry_eviction() {
        // A cache that can only hold 1 byte → every put after the first evicts.
        let mut cache =
            CompileCache::new(CacheConfig { max_size_mb: 0, ttl_days: 0, ..Default::default() });
        // max_size_bytes() == 0, so nothing can be stored beyond the first
        // entry once another arrives.
        cache.put(make_key(1), vec![0xAA]);
        cache.put(make_key(2), vec![0xBB]);
        assert!(cache.len() <= 1);
    }

    #[test]
    fn test_invalidate_removes_entry() {
        let mut cache = default_cache();
        let key = make_key(1);
        cache.put(key.clone(), make_binary(32));
        cache.invalidate(&key);
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_invalidate_nonexistent_noop() {
        let mut cache = default_cache();
        cache.invalidate(&make_key(999)); // must not panic
    }

    #[test]
    fn test_clear_empties_cache() {
        let mut cache = default_cache();
        for i in 0..10 {
            cache.put(make_key(i), make_binary(16));
        }
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().total_size, 0);
    }

    #[test]
    fn test_debug_format() {
        let cache = default_cache();
        let dbg = format!("{cache:?}");
        assert!(dbg.contains("CompileCache"));
    }

    // -----------------------------------------------------------------------
    // Property tests — hash collision resistance
    // -----------------------------------------------------------------------

    #[test]
    fn test_hash_source_deterministic() {
        let h1 = hash_source("kernel void matmul(){}");
        let h2 = hash_source("kernel void matmul(){}");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_source_distinct_inputs() {
        // 100 distinct inputs should produce 100 distinct hashes.
        let hashes: Vec<u64> =
            (0..100).map(|i| hash_source(&format!("kernel void f{i}(){{}}"))).collect();
        let unique: std::collections::HashSet<u64> = hashes.iter().copied().collect();
        assert_eq!(unique.len(), 100, "no collisions expected among 100 inputs");
    }

    #[test]
    fn test_hash_source_empty_string() {
        // Must not panic.
        let _ = hash_source("");
    }

    #[test]
    fn test_hash_source_whitespace_sensitivity() {
        let a = hash_source("kernel void f(){}");
        let b = hash_source("kernel void  f(){}"); // extra space
        assert_ne!(a, b, "hash should be whitespace-sensitive");
    }
}
