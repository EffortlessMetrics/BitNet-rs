//! CUDA kernel compilation cache with pluggable eviction policies.
//!
//! Caches compiled PTX kernels to avoid redundant compilation, with support for
//! LRU, LFU, FIFO, and size-aware eviction strategies.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur during cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// Attempted to insert a duplicate key.
    DuplicateKey(String),
    /// Cache has zero capacity — cannot insert.
    ZeroCapacity,
    /// An I/O or persistence error (stringified).
    PersistenceError(String),
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateKey(k) => write!(f, "duplicate cache key: {k}"),
            Self::ZeroCapacity => write!(f, "cache capacity is zero"),
            Self::PersistenceError(msg) => write!(f, "persistence error: {msg}"),
        }
    }
}

impl std::error::Error for CacheError {}

// ---------------------------------------------------------------------------
// Eviction policy
// ---------------------------------------------------------------------------

/// Strategy used to choose which entry to evict when the cache is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used — evicts the entry with the oldest `last_used`.
    LRU,
    /// Least Frequently Used — evicts the entry with the lowest `hit_count`.
    LFU,
    /// First In First Out — evicts the oldest inserted entry.
    FIFO,
    /// Size Aware — evicts the entry with the largest `compile_time_ms`
    /// (a proxy for compilation cost / resource weight).
    SizeAware,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`KernelCache`].
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries the cache may hold.
    pub max_entries: usize,
    /// Which eviction strategy to use when the cache is full.
    pub eviction_policy: EvictionPolicy,
    /// Optional path for on-disk persistence (not implemented yet).
    pub persist_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

/// A single cached kernel compilation result.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelCacheEntry {
    /// Unique key identifying this kernel variant.
    pub key: String,
    /// Hash of the compiled PTX blob.
    pub ptx_hash: String,
    /// Wall-clock compilation time in milliseconds.
    pub compile_time_ms: u64,
    /// Number of cache hits for this entry.
    pub hit_count: u64,
    /// Timestamp of the last access (millis since UNIX epoch).
    pub last_used: u64,
}

impl KernelCacheEntry {
    /// Helper that touches `last_used` to "now" and increments `hit_count`.
    fn touch(&mut self) {
        self.hit_count += 1;
        self.last_used = now_ms();
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a [`KernelCache`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_compile_time_ms: u64,
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

/// Thread-local CUDA kernel compilation cache.
///
/// Stores [`KernelCacheEntry`] values keyed by a `String` identifier.  When the
/// cache exceeds `max_entries` the configured [`EvictionPolicy`] determines
/// which entry is removed.
pub struct KernelCache {
    config: CacheConfig,
    entries: HashMap<String, KernelCacheEntry>,
    /// FIFO insertion order — only meaningful for `EvictionPolicy::FIFO`.
    insertion_order: VecDeque<String>,
    stats: CacheStats,
}

impl KernelCache {
    /// Create a new, empty cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
            stats: CacheStats::default(),
        }
    }

    /// Look up a kernel by key, recording a hit/miss.
    pub fn lookup(&mut self, key: &str) -> Option<&KernelCacheEntry> {
        if self.entries.contains_key(key) {
            self.stats.hits += 1;
            // Touch the entry (update hit_count / last_used).
            self.entries.get_mut(key).unwrap().touch();
            self.entries.get(key)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a new entry. Returns `CacheError::DuplicateKey` if the key
    /// already exists, or `CacheError::ZeroCapacity` if `max_entries == 0`.
    ///
    /// If the cache is full the configured eviction policy is applied first.
    pub fn insert(&mut self, key: String, entry: KernelCacheEntry) -> Result<(), CacheError> {
        if self.config.max_entries == 0 {
            return Err(CacheError::ZeroCapacity);
        }
        if self.entries.contains_key(&key) {
            return Err(CacheError::DuplicateKey(key));
        }
        // Evict until there is room.
        while self.entries.len() >= self.config.max_entries {
            self.evict();
        }
        self.stats.total_compile_time_ms += entry.compile_time_ms;
        self.insertion_order.push_back(key.clone());
        self.entries.insert(key, entry);
        Ok(())
    }

    /// Evict a single entry according to the configured policy.
    /// Returns the removed entry, or `None` if the cache is empty.
    pub fn evict(&mut self) -> Option<KernelCacheEntry> {
        let victim_key = self.select_victim()?;
        self.remove_key(&victim_key)
    }

    /// Return aggregate cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    /// Remove all entries and reset stats.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.insertion_order.clear();
        self.stats = CacheStats::default();
    }

    /// Bulk-load entries (e.g. from a persisted snapshot). Existing entries
    /// with the same key are silently skipped.
    pub fn warm_up(&mut self, entries: Vec<(String, KernelCacheEntry)>) {
        for (key, entry) in entries {
            if self.entries.len() >= self.config.max_entries {
                break;
            }
            if self.entries.contains_key(&key) {
                continue;
            }
            self.stats.total_compile_time_ms += entry.compile_time_ms;
            self.insertion_order.push_back(key.clone());
            self.entries.insert(key, entry);
        }
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Access the current configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Pick the victim key according to the eviction policy.
    fn select_victim(&self) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        match self.config.eviction_policy {
            EvictionPolicy::LRU => {
                self.entries.values().min_by_key(|e| e.last_used).map(|e| e.key.clone())
            }
            EvictionPolicy::LFU => {
                self.entries.values().min_by_key(|e| e.hit_count).map(|e| e.key.clone())
            }
            EvictionPolicy::FIFO => self.insertion_order.front().cloned(),
            EvictionPolicy::SizeAware => {
                self.entries.values().max_by_key(|e| e.compile_time_ms).map(|e| e.key.clone())
            }
        }
    }

    /// Remove a key from both the map and the insertion deque, updating stats.
    fn remove_key(&mut self, key: &str) -> Option<KernelCacheEntry> {
        if let Some(entry) = self.entries.remove(key) {
            self.insertion_order.retain(|k| k != key);
            self.stats.evictions += 1;
            Some(entry)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

/// Build a [`KernelCacheEntry`] with sensible defaults for testing.
#[cfg(test)]
fn make_entry(key: &str, compile_time_ms: u64) -> KernelCacheEntry {
    KernelCacheEntry {
        key: key.to_string(),
        ptx_hash: format!("hash_{key}"),
        compile_time_ms,
        hit_count: 0,
        last_used: now_ms(),
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a default LRU config with the given capacity.
    fn lru_config(cap: usize) -> CacheConfig {
        CacheConfig { max_entries: cap, eviction_policy: EvictionPolicy::LRU, persist_path: None }
    }

    fn lfu_config(cap: usize) -> CacheConfig {
        CacheConfig { max_entries: cap, eviction_policy: EvictionPolicy::LFU, persist_path: None }
    }

    fn fifo_config(cap: usize) -> CacheConfig {
        CacheConfig { max_entries: cap, eviction_policy: EvictionPolicy::FIFO, persist_path: None }
    }

    fn size_config(cap: usize) -> CacheConfig {
        CacheConfig {
            max_entries: cap,
            eviction_policy: EvictionPolicy::SizeAware,
            persist_path: None,
        }
    }

    fn entry(key: &str, compile_ms: u64) -> KernelCacheEntry {
        make_entry(key, compile_ms)
    }

    fn entry_with_hits(key: &str, compile_ms: u64, hits: u64) -> KernelCacheEntry {
        KernelCacheEntry {
            key: key.to_string(),
            ptx_hash: format!("hash_{key}"),
            compile_time_ms: compile_ms,
            hit_count: hits,
            last_used: now_ms(),
        }
    }

    fn entry_with_last_used(key: &str, last_used: u64) -> KernelCacheEntry {
        KernelCacheEntry {
            key: key.to_string(),
            ptx_hash: format!("hash_{key}"),
            compile_time_ms: 10,
            hit_count: 0,
            last_used,
        }
    }

    // ---------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------

    #[test]
    fn new_cache_is_empty() {
        let c = KernelCache::new(lru_config(8));
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn new_cache_stats_are_zero() {
        let c = KernelCache::new(lru_config(8));
        let s = c.stats();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.total_compile_time_ms, 0);
    }

    #[test]
    fn config_accessible() {
        let c = KernelCache::new(lru_config(16));
        assert_eq!(c.config().max_entries, 16);
        assert_eq!(c.config().eviction_policy, EvictionPolicy::LRU);
    }

    // ---------------------------------------------------------------
    // Insert basics
    // ---------------------------------------------------------------

    #[test]
    fn insert_single_entry() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("a".into(), entry("a", 100)).unwrap();
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn insert_updates_compile_time_stat() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("a".into(), entry("a", 50)).unwrap();
        c.insert("b".into(), entry("b", 30)).unwrap();
        assert_eq!(c.stats().total_compile_time_ms, 80);
    }

    #[test]
    fn insert_duplicate_key_returns_error() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("a".into(), entry("a", 10)).unwrap();
        let res = c.insert("a".into(), entry("a", 20));
        assert_eq!(res, Err(CacheError::DuplicateKey("a".into())));
    }

    #[test]
    fn insert_zero_capacity_returns_error() {
        let mut c = KernelCache::new(lru_config(0));
        let res = c.insert("a".into(), entry("a", 10));
        assert_eq!(res, Err(CacheError::ZeroCapacity));
    }

    #[test]
    fn insert_at_capacity_triggers_eviction() {
        let mut c = KernelCache::new(lru_config(2));
        c.insert("a".into(), entry_with_last_used("a", 1)).unwrap();
        c.insert("b".into(), entry_with_last_used("b", 2)).unwrap();
        c.insert("c".into(), entry_with_last_used("c", 3)).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c.stats().evictions, 1);
    }

    // ---------------------------------------------------------------
    // Lookup
    // ---------------------------------------------------------------

    #[test]
    fn lookup_existing_key_returns_entry() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("k".into(), entry("k", 10)).unwrap();
        let e = c.lookup("k").unwrap();
        assert_eq!(e.key, "k");
    }

    #[test]
    fn lookup_missing_key_returns_none() {
        let mut c = KernelCache::new(lru_config(4));
        assert!(c.lookup("nope").is_none());
    }

    #[test]
    fn lookup_increments_hit_count() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("k".into(), entry("k", 5)).unwrap();
        c.lookup("k");
        c.lookup("k");
        let e = c.lookup("k").unwrap();
        assert_eq!(e.hit_count, 3);
    }

    #[test]
    fn lookup_records_hit_stat() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("k".into(), entry("k", 5)).unwrap();
        c.lookup("k");
        assert_eq!(c.stats().hits, 1);
    }

    #[test]
    fn lookup_records_miss_stat() {
        let mut c = KernelCache::new(lru_config(4));
        c.lookup("missing");
        assert_eq!(c.stats().misses, 1);
    }

    #[test]
    fn lookup_updates_last_used() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("k".into(), entry_with_last_used("k", 1)).unwrap();
        let _ = c.lookup("k");
        let e = c.lookup("k").unwrap();
        assert!(e.last_used > 1);
    }

    // ---------------------------------------------------------------
    // Evict — empty cache
    // ---------------------------------------------------------------

    #[test]
    fn evict_empty_cache_returns_none() {
        let mut c = KernelCache::new(lru_config(4));
        assert!(c.evict().is_none());
    }

    // ---------------------------------------------------------------
    // LRU eviction
    // ---------------------------------------------------------------

    #[test]
    fn lru_evicts_oldest_used() {
        let mut c = KernelCache::new(lru_config(3));
        c.insert("a".into(), entry_with_last_used("a", 10)).unwrap();
        c.insert("b".into(), entry_with_last_used("b", 20)).unwrap();
        c.insert("c".into(), entry_with_last_used("c", 30)).unwrap();
        let victim = c.evict().unwrap();
        assert_eq!(victim.key, "a");
    }

    #[test]
    fn lru_evicts_correct_after_touch() {
        let mut c = KernelCache::new(lru_config(3));
        c.insert("a".into(), entry_with_last_used("a", 10)).unwrap();
        c.insert("b".into(), entry_with_last_used("b", 20)).unwrap();
        c.insert("c".into(), entry_with_last_used("c", 30)).unwrap();
        // Touch "a" so it becomes most recently used.
        c.lookup("a");
        let victim = c.evict().unwrap();
        assert_eq!(victim.key, "b");
    }

    #[test]
    fn lru_overflow_insert_evicts_least_recent() {
        let mut c = KernelCache::new(lru_config(2));
        c.insert("a".into(), entry_with_last_used("a", 1)).unwrap();
        c.insert("b".into(), entry_with_last_used("b", 2)).unwrap();
        // This should evict "a" (oldest last_used).
        c.insert("c".into(), entry_with_last_used("c", 3)).unwrap();
        assert!(c.lookup("a").is_none());
    }

    // ---------------------------------------------------------------
    // LFU eviction
    // ---------------------------------------------------------------

    #[test]
    fn lfu_evicts_least_frequently_used() {
        let mut c = KernelCache::new(lfu_config(3));
        c.insert("a".into(), entry_with_hits("a", 10, 5)).unwrap();
        c.insert("b".into(), entry_with_hits("b", 10, 1)).unwrap();
        c.insert("c".into(), entry_with_hits("c", 10, 3)).unwrap();
        let victim = c.evict().unwrap();
        assert_eq!(victim.key, "b");
    }

    #[test]
    fn lfu_after_lookups() {
        let mut c = KernelCache::new(lfu_config(3));
        c.insert("a".into(), entry("a", 10)).unwrap();
        c.insert("b".into(), entry("b", 10)).unwrap();
        c.insert("c".into(), entry("c", 10)).unwrap();
        // "a" gets 3 hits, "c" gets 1, "b" stays at 0 → evict "b".
        c.lookup("a");
        c.lookup("a");
        c.lookup("a");
        c.lookup("c");
        let victim = c.evict().unwrap();
        assert_eq!(victim.key, "b");
    }

    #[test]
    fn lfu_overflow_insert_evicts_lowest_hit() {
        let mut c = KernelCache::new(lfu_config(2));
        c.insert("a".into(), entry_with_hits("a", 10, 10)).unwrap();
        c.insert("b".into(), entry_with_hits("b", 10, 2)).unwrap();
        c.insert("c".into(), entry("c", 10)).unwrap();
        // "b" (hit_count=2) should be evicted over "a" (hit_count=10).
        assert!(c.lookup("b").is_none());
        assert!(c.lookup("a").is_some());
    }

    // ---------------------------------------------------------------
    // FIFO eviction
    // ---------------------------------------------------------------

    #[test]
    fn fifo_evicts_first_inserted() {
        let mut c = KernelCache::new(fifo_config(3));
        c.insert("a".into(), entry("a", 10)).unwrap();
        c.insert("b".into(), entry("b", 10)).unwrap();
        c.insert("c".into(), entry("c", 10)).unwrap();
        let victim = c.evict().unwrap();
        assert_eq!(victim.key, "a");
    }

    #[test]
    fn fifo_ignores_access_pattern() {
        let mut c = KernelCache::new(fifo_config(3));
        c.insert("a".into(), entry("a", 10)).unwrap();
        c.insert("b".into(), entry("b", 10)).unwrap();
        c.insert("c".into(), entry("c", 10)).unwrap();
        // Even though we access "a" heavily it should still be evicted first.
        for _ in 0..10 {
            c.lookup("a");
        }
        let victim = c.evict().unwrap();
        assert_eq!(victim.key, "a");
    }

    #[test]
    fn fifo_multiple_evictions_preserve_order() {
        let mut c = KernelCache::new(fifo_config(4));
        for k in &["w", "x", "y", "z"] {
            c.insert((*k).into(), entry(k, 10)).unwrap();
        }
        assert_eq!(c.evict().unwrap().key, "w");
        assert_eq!(c.evict().unwrap().key, "x");
        assert_eq!(c.evict().unwrap().key, "y");
        assert_eq!(c.evict().unwrap().key, "z");
    }

    #[test]
    fn fifo_overflow_insert_evicts_oldest() {
        let mut c = KernelCache::new(fifo_config(2));
        c.insert("a".into(), entry("a", 10)).unwrap();
        c.insert("b".into(), entry("b", 10)).unwrap();
        c.insert("c".into(), entry("c", 10)).unwrap();
        assert!(c.lookup("a").is_none());
        assert!(c.lookup("b").is_some());
    }

    // ---------------------------------------------------------------
    // SizeAware eviction
    // ---------------------------------------------------------------

    #[test]
    fn size_aware_evicts_largest_compile_time() {
        let mut c = KernelCache::new(size_config(3));
        c.insert("small".into(), entry("small", 10)).unwrap();
        c.insert("big".into(), entry("big", 500)).unwrap();
        c.insert("med".into(), entry("med", 100)).unwrap();
        let victim = c.evict().unwrap();
        assert_eq!(victim.key, "big");
    }

    #[test]
    fn size_aware_overflow_evicts_heaviest() {
        let mut c = KernelCache::new(size_config(2));
        c.insert("light".into(), entry("light", 5)).unwrap();
        c.insert("heavy".into(), entry("heavy", 999)).unwrap();
        c.insert("new".into(), entry("new", 50)).unwrap();
        // "heavy" should have been evicted.
        assert!(c.lookup("heavy").is_none());
    }

    #[test]
    fn size_aware_evicts_correctly_after_repeated_inserts() {
        let mut c = KernelCache::new(size_config(2));
        c.insert("a".into(), entry("a", 100)).unwrap();
        c.insert("b".into(), entry("b", 200)).unwrap();
        // Insert "c" (50) → evict "b" (200).
        c.insert("c".into(), entry("c", 50)).unwrap();
        assert!(c.lookup("b").is_none());
        // Insert "d" (150) → evict "a" (100) since a>c.
        c.insert("d".into(), entry("d", 150)).unwrap();
        assert!(c.lookup("a").is_none());
        assert!(c.lookup("c").is_some());
    }

    // ---------------------------------------------------------------
    // Clear
    // ---------------------------------------------------------------

    #[test]
    fn clear_empties_cache() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("a".into(), entry("a", 10)).unwrap();
        c.insert("b".into(), entry("b", 20)).unwrap();
        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn clear_resets_stats() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("a".into(), entry("a", 10)).unwrap();
        c.lookup("a");
        c.lookup("miss");
        c.clear();
        let s = c.stats();
        assert_eq!(s, CacheStats::default());
    }

    #[test]
    fn clear_allows_reinsertion_of_same_key() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("a".into(), entry("a", 10)).unwrap();
        c.clear();
        c.insert("a".into(), entry("a", 20)).unwrap();
        assert_eq!(c.len(), 1);
    }

    // ---------------------------------------------------------------
    // Warm-up
    // ---------------------------------------------------------------

    #[test]
    fn warm_up_loads_entries() {
        let mut c = KernelCache::new(lru_config(4));
        c.warm_up(vec![("a".into(), entry("a", 10)), ("b".into(), entry("b", 20))]);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn warm_up_respects_capacity() {
        let mut c = KernelCache::new(lru_config(2));
        c.warm_up(vec![
            ("a".into(), entry("a", 10)),
            ("b".into(), entry("b", 20)),
            ("c".into(), entry("c", 30)),
        ]);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn warm_up_skips_duplicates() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("a".into(), entry("a", 10)).unwrap();
        c.warm_up(vec![("a".into(), entry("a", 99)), ("b".into(), entry("b", 20))]);
        assert_eq!(c.len(), 2);
        // Original "a" should be unchanged.
        let e = c.lookup("a").unwrap();
        assert_eq!(e.compile_time_ms, 10);
    }

    #[test]
    fn warm_up_updates_compile_time_stat() {
        let mut c = KernelCache::new(lru_config(8));
        c.warm_up(vec![("a".into(), entry("a", 10)), ("b".into(), entry("b", 30))]);
        assert_eq!(c.stats().total_compile_time_ms, 40);
    }

    #[test]
    fn warm_up_empty_vec_is_noop() {
        let mut c = KernelCache::new(lru_config(8));
        c.warm_up(vec![]);
        assert!(c.is_empty());
    }

    // ---------------------------------------------------------------
    // Stats tracking
    // ---------------------------------------------------------------

    #[test]
    fn stats_track_hits_and_misses() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("a".into(), entry("a", 10)).unwrap();
        c.lookup("a"); // hit
        c.lookup("a"); // hit
        c.lookup("b"); // miss
        let s = c.stats();
        assert_eq!(s.hits, 2);
        assert_eq!(s.misses, 1);
    }

    #[test]
    fn stats_track_evictions() {
        let mut c = KernelCache::new(lru_config(1));
        c.insert("a".into(), entry_with_last_used("a", 1)).unwrap();
        c.insert("b".into(), entry_with_last_used("b", 2)).unwrap();
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn stats_total_compile_time_accumulates() {
        let mut c = KernelCache::new(lru_config(8));
        c.insert("a".into(), entry("a", 100)).unwrap();
        c.insert("b".into(), entry("b", 200)).unwrap();
        c.insert("c".into(), entry("c", 300)).unwrap();
        assert_eq!(c.stats().total_compile_time_ms, 600);
    }

    // ---------------------------------------------------------------
    // Edge cases
    // ---------------------------------------------------------------

    #[test]
    fn capacity_one_works() {
        let mut c = KernelCache::new(lru_config(1));
        c.insert("a".into(), entry_with_last_used("a", 1)).unwrap();
        c.insert("b".into(), entry_with_last_used("b", 2)).unwrap();
        assert_eq!(c.len(), 1);
        assert!(c.lookup("a").is_none());
        assert!(c.lookup("b").is_some());
    }

    #[test]
    fn large_capacity_no_eviction() {
        let mut c = KernelCache::new(lru_config(1000));
        for i in 0..100 {
            let k = format!("k{i}");
            c.insert(k.clone(), entry(&k, 1)).unwrap();
        }
        assert_eq!(c.len(), 100);
        assert_eq!(c.stats().evictions, 0);
    }

    #[test]
    fn mixed_hits_misses_sequence() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("a".into(), entry("a", 10)).unwrap();
        c.lookup("a"); // hit
        c.lookup("b"); // miss
        c.insert("b".into(), entry("b", 20)).unwrap();
        c.lookup("b"); // hit
        c.lookup("c"); // miss
        let s = c.stats();
        assert_eq!(s.hits, 2);
        assert_eq!(s.misses, 2);
    }

    #[test]
    fn evict_single_entry_cache() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("only".into(), entry("only", 42)).unwrap();
        let v = c.evict().unwrap();
        assert_eq!(v.key, "only");
        assert!(c.is_empty());
    }

    #[test]
    fn insert_after_eviction_succeeds() {
        let mut c = KernelCache::new(lru_config(2));
        c.insert("a".into(), entry_with_last_used("a", 1)).unwrap();
        c.insert("b".into(), entry_with_last_used("b", 2)).unwrap();
        c.evict(); // remove "a"
        c.insert("c".into(), entry("c", 10)).unwrap();
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn duplicate_key_does_not_mutate_cache() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("a".into(), entry("a", 10)).unwrap();
        let _ = c.insert("a".into(), entry("a", 999));
        assert_eq!(c.stats().total_compile_time_ms, 10);
        assert_eq!(c.len(), 1);
    }

    // ---------------------------------------------------------------
    // CacheError Display / PartialEq
    // ---------------------------------------------------------------

    #[test]
    fn cache_error_display_duplicate() {
        let e = CacheError::DuplicateKey("foo".into());
        assert_eq!(e.to_string(), "duplicate cache key: foo");
    }

    #[test]
    fn cache_error_display_zero_capacity() {
        assert_eq!(CacheError::ZeroCapacity.to_string(), "cache capacity is zero");
    }

    #[test]
    fn cache_error_display_persistence() {
        let e = CacheError::PersistenceError("disk full".into());
        assert!(e.to_string().contains("disk full"));
    }

    #[test]
    fn cache_error_eq() {
        assert_eq!(CacheError::ZeroCapacity, CacheError::ZeroCapacity);
        assert_ne!(CacheError::DuplicateKey("a".into()), CacheError::DuplicateKey("b".into()),);
    }

    // ---------------------------------------------------------------
    // EvictionPolicy PartialEq / Clone / Debug
    // ---------------------------------------------------------------

    #[test]
    fn eviction_policy_eq() {
        assert_eq!(EvictionPolicy::LRU, EvictionPolicy::LRU);
        assert_ne!(EvictionPolicy::LRU, EvictionPolicy::LFU);
    }

    #[test]
    fn eviction_policy_clone() {
        let p = EvictionPolicy::SizeAware;
        let p2 = p;
        assert_eq!(p, p2);
    }

    #[test]
    fn eviction_policy_debug() {
        let s = format!("{:?}", EvictionPolicy::FIFO);
        assert_eq!(s, "FIFO");
    }

    // ---------------------------------------------------------------
    // KernelCacheEntry Clone / PartialEq
    // ---------------------------------------------------------------

    #[test]
    fn entry_clone_and_eq() {
        let e1 = entry("x", 42);
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    // ---------------------------------------------------------------
    // CacheStats Default
    // ---------------------------------------------------------------

    #[test]
    fn cache_stats_default_is_zero() {
        let s = CacheStats::default();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.total_compile_time_ms, 0);
    }

    // ---------------------------------------------------------------
    // Persist path in config
    // ---------------------------------------------------------------

    #[test]
    fn config_persist_path_none_by_default() {
        let cfg = lru_config(4);
        assert!(cfg.persist_path.is_none());
    }

    #[test]
    fn config_persist_path_some() {
        let cfg = CacheConfig {
            max_entries: 4,
            eviction_policy: EvictionPolicy::LRU,
            persist_path: Some(PathBuf::from("/tmp/cache.bin")),
        };
        assert_eq!(cfg.persist_path.unwrap(), PathBuf::from("/tmp/cache.bin"));
    }

    // ---------------------------------------------------------------
    // Stress / overflow
    // ---------------------------------------------------------------

    #[test]
    fn overflow_many_inserts() {
        let cap = 10;
        let mut c = KernelCache::new(fifo_config(cap));
        for i in 0..100 {
            let k = format!("k{i}");
            c.insert(k.clone(), entry(&k, 1)).unwrap();
        }
        assert_eq!(c.len(), cap);
        assert_eq!(c.stats().evictions, 90);
    }

    #[test]
    fn lru_stress_interleaved_lookups() {
        let mut c = KernelCache::new(lru_config(5));
        for i in 0..5u64 {
            let k = format!("k{i}");
            c.insert(k.clone(), entry_with_last_used(&k, i)).unwrap();
        }
        // Access k4, k3, k2, k1 — k0 stays oldest.
        for i in (1..5).rev() {
            c.lookup(&format!("k{i}"));
        }
        c.insert("new".into(), entry("new", 1)).unwrap();
        assert!(c.lookup("k0").is_none());
    }

    // ---------------------------------------------------------------
    // Multiple evictions in sequence
    // ---------------------------------------------------------------

    #[test]
    fn lfu_sequential_evictions() {
        let mut c = KernelCache::new(lfu_config(4));
        c.insert("a".into(), entry_with_hits("a", 10, 1)).unwrap();
        c.insert("b".into(), entry_with_hits("b", 10, 4)).unwrap();
        c.insert("c".into(), entry_with_hits("c", 10, 2)).unwrap();
        c.insert("d".into(), entry_with_hits("d", 10, 3)).unwrap();
        // Eviction order should be: a(1), c(2), d(3), b(4)
        assert_eq!(c.evict().unwrap().key, "a");
        assert_eq!(c.evict().unwrap().key, "c");
        assert_eq!(c.evict().unwrap().key, "d");
        assert_eq!(c.evict().unwrap().key, "b");
        assert!(c.evict().is_none());
    }

    #[test]
    fn size_aware_sequential_evictions() {
        let mut c = KernelCache::new(size_config(3));
        c.insert("s".into(), entry("s", 10)).unwrap();
        c.insert("m".into(), entry("m", 50)).unwrap();
        c.insert("l".into(), entry("l", 200)).unwrap();
        assert_eq!(c.evict().unwrap().key, "l");
        assert_eq!(c.evict().unwrap().key, "m");
        assert_eq!(c.evict().unwrap().key, "s");
    }

    // ---------------------------------------------------------------
    // Warm-up then operate
    // ---------------------------------------------------------------

    #[test]
    fn warm_up_then_insert_and_evict() {
        let mut c = KernelCache::new(fifo_config(3));
        c.warm_up(vec![("a".into(), entry("a", 10)), ("b".into(), entry("b", 20))]);
        c.insert("c".into(), entry("c", 30)).unwrap();
        // FIFO: "a" was first in warm-up.
        c.insert("d".into(), entry("d", 40)).unwrap();
        assert!(c.lookup("a").is_none());
    }

    #[test]
    fn warm_up_with_capacity_one() {
        let mut c = KernelCache::new(lru_config(1));
        c.warm_up(vec![("a".into(), entry("a", 10)), ("b".into(), entry("b", 20))]);
        assert_eq!(c.len(), 1);
    }

    // ---------------------------------------------------------------
    // len / is_empty consistency
    // ---------------------------------------------------------------

    #[test]
    fn len_after_evict() {
        let mut c = KernelCache::new(lru_config(4));
        c.insert("a".into(), entry("a", 10)).unwrap();
        c.insert("b".into(), entry("b", 20)).unwrap();
        c.evict();
        assert_eq!(c.len(), 1);
        assert!(!c.is_empty());
    }

    #[test]
    fn is_empty_after_all_evicted() {
        let mut c = KernelCache::new(lru_config(2));
        c.insert("a".into(), entry("a", 10)).unwrap();
        c.evict();
        assert!(c.is_empty());
    }
}

// =========================================================================
// Property-based tests
// =========================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_policy() -> impl Strategy<Value = EvictionPolicy> {
        prop_oneof![
            Just(EvictionPolicy::LRU),
            Just(EvictionPolicy::LFU),
            Just(EvictionPolicy::FIFO),
            Just(EvictionPolicy::SizeAware),
        ]
    }

    proptest! {
        /// Inserting N items into a cache of capacity C never exceeds C.
        #[test]
        fn cache_never_exceeds_capacity(
            cap in 1usize..64,
            n in 1usize..200,
            policy in arb_policy(),
        ) {
            let cfg = CacheConfig {
                max_entries: cap,
                eviction_policy: policy,
                persist_path: None,
            };
            let mut c = KernelCache::new(cfg);
            for i in 0..n {
                let k = format!("k{i}");
                let e = KernelCacheEntry {
                    key: k.clone(),
                    ptx_hash: format!("h{i}"),
                    compile_time_ms: (i as u64) + 1,
                    hit_count: 0,
                    last_used: i as u64,
                };
                let _ = c.insert(k, e);
            }
            prop_assert!(c.len() <= cap);
        }

        /// hits + misses == total lookups.
        #[test]
        fn hits_plus_misses_eq_lookups(
            inserts in 0usize..30,
            lookups in proptest::collection::vec(0usize..50, 0..60),
        ) {
            let mut c = KernelCache::new(CacheConfig {
                max_entries: 20,
                eviction_policy: EvictionPolicy::LRU,
                persist_path: None,
            });
            for i in 0..inserts {
                let k = format!("k{i}");
                let _ = c.insert(k.clone(), KernelCacheEntry {
                    key: k, ptx_hash: String::new(),
                    compile_time_ms: 1, hit_count: 0, last_used: 0,
                });
            }
            for idx in &lookups {
                c.lookup(&format!("k{idx}"));
            }
            let s = c.stats();
            prop_assert_eq!(s.hits + s.misses, lookups.len() as u64);
        }

        /// After clear(), the cache is empty with zero stats.
        #[test]
        fn clear_resets_everything(
            n in 0usize..50,
            policy in arb_policy(),
        ) {
            let cfg = CacheConfig {
                max_entries: 30,
                eviction_policy: policy,
                persist_path: None,
            };
            let mut c = KernelCache::new(cfg);
            for i in 0..n {
                let k = format!("k{i}");
                let _ = c.insert(k.clone(), KernelCacheEntry {
                    key: k, ptx_hash: String::new(),
                    compile_time_ms: 1, hit_count: 0, last_used: i as u64,
                });
            }
            c.clear();
            prop_assert!(c.is_empty());
            prop_assert_eq!(c.stats(), CacheStats::default());
        }

        /// warm_up never exceeds capacity.
        #[test]
        fn warm_up_respects_cap(
            cap in 1usize..32,
            n in 0usize..100,
        ) {
            let cfg = CacheConfig {
                max_entries: cap,
                eviction_policy: EvictionPolicy::FIFO,
                persist_path: None,
            };
            let mut c = KernelCache::new(cfg);
            let entries: Vec<_> = (0..n)
                .map(|i| {
                    let k = format!("w{i}");
                    (k.clone(), KernelCacheEntry {
                        key: k,
                        ptx_hash: String::new(),
                        compile_time_ms: 1,
                        hit_count: 0,
                        last_used: 0,
                    })
                })
                .collect();
            c.warm_up(entries);
            prop_assert!(c.len() <= cap);
        }

        /// evictions == (inserts - len) when no manual evict() calls.
        #[test]
        fn evictions_bookkeeping(
            cap in 1usize..20,
            n in 0usize..80,
            policy in arb_policy(),
        ) {
            let cfg = CacheConfig {
                max_entries: cap,
                eviction_policy: policy,
                persist_path: None,
            };
            let mut c = KernelCache::new(cfg);
            let mut inserted = 0u64;
            for i in 0..n {
                let k = format!("k{i}");
                let e = KernelCacheEntry {
                    key: k.clone(),
                    ptx_hash: String::new(),
                    compile_time_ms: (i as u64) + 1,
                    hit_count: 0,
                    last_used: i as u64,
                };
                if c.insert(k, e).is_ok() {
                    inserted += 1;
                }
            }
            let s = c.stats();
            prop_assert_eq!(s.evictions, inserted - c.len() as u64);
        }
    }
}
