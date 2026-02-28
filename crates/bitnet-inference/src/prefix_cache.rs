//! # Prompt Prefix Cache
//!
//! Caches KV states for common prompt prefixes to avoid recomputation.
//! Uses a trie (prefix tree) for efficient token-sequence matching so that
//! the longest cached prefix is returned in O(n) time where n is the
//! query length.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Result, bail};
use tracing::debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Eviction strategy used when the cache exceeds its capacity limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used — evicts the entry whose last access is oldest.
    LRU,
    /// Least Frequently Used — evicts the entry with the fewest hits.
    LFU,
    /// First In, First Out — evicts the oldest entry by creation time.
    FIFO,
    /// Time To Live — evicts entries older than a configurable duration.
    TTL,
}

/// Configuration for [`PrefixCache`].
#[derive(Debug, Clone)]
pub struct PrefixCacheConfig {
    /// Maximum number of cached entries.
    pub max_entries: usize,
    /// Maximum total memory (bytes) consumed by cached states.
    pub max_memory_bytes: usize,
    /// Strategy for selecting victims when limits are exceeded.
    pub eviction_policy: EvictionPolicy,
    /// Minimum token-prefix length eligible for caching.
    pub min_prefix_length: usize,
    /// TTL duration in seconds (only used with [`EvictionPolicy::TTL`]).
    pub ttl_seconds: u64,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            max_memory_bytes: 512 * 1024 * 1024, // 512 MiB
            eviction_policy: EvictionPolicy::LRU,
            min_prefix_length: 4,
            ttl_seconds: 3600,
        }
    }
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// A single cached prefix and its associated opaque KV state.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The token prefix that was cached.
    pub token_prefix: Vec<u32>,
    /// Opaque bytes representing the cached KV state.
    pub cached_state: Vec<u8>,
    /// Number of cache hits for this entry.
    pub hits: u64,
    /// Timestamp of the most recent access (lookup or insert).
    pub last_access: Instant,
    /// Timestamp when the entry was first created.
    pub created_at: Instant,
    /// Size of `cached_state` in bytes.
    pub size_bytes: usize,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Aggregated statistics for a [`PrefixCache`] instance.
#[derive(Debug, Clone)]
pub struct PrefixCacheStats {
    /// Fraction of lookups that produced a match (0.0–1.0).
    pub hit_rate: f64,
    /// Fraction of lookups that found no match (0.0–1.0).
    pub miss_rate: f64,
    /// Total number of entries evicted since the cache was created.
    pub eviction_count: u64,
    /// Current memory consumed by all cached states.
    pub memory_usage: usize,
    /// Average prefix-match length across all successful lookups.
    pub avg_prefix_match_length: f64,
}

// ---------------------------------------------------------------------------
// Trie node (internal)
// ---------------------------------------------------------------------------

/// A node in the token-sequence trie.
#[derive(Debug)]
struct TrieNode {
    children: HashMap<u32, TrieNode>,
    /// If `Some`, this node terminates a cached prefix and stores the
    /// unique entry id.
    entry_id: Option<u64>,
}

impl TrieNode {
    fn new() -> Self {
        Self { children: HashMap::new(), entry_id: None }
    }
}

// ---------------------------------------------------------------------------
// PrefixCache
// ---------------------------------------------------------------------------

/// A trie-backed cache that stores opaque KV states keyed by token-sequence
/// prefixes.
///
/// # Example
/// ```
/// use bitnet_inference::prefix_cache::{PrefixCache, PrefixCacheConfig};
///
/// let mut cache = PrefixCache::new(PrefixCacheConfig::default());
/// let tokens = vec![1, 2, 3, 4, 5];
/// let state = vec![0u8; 64];
/// cache.insert(&tokens, state.clone()).unwrap();
///
/// let result = cache.lookup(&[1, 2, 3, 4, 5, 6]);
/// assert!(result.is_some());
/// let (matched_len, entry) = result.unwrap();
/// assert_eq!(matched_len, 5);
/// ```
pub struct PrefixCache {
    config: PrefixCacheConfig,
    root: TrieNode,
    entries: HashMap<u64, CacheEntry>,
    next_id: u64,
    current_memory: usize,

    // Statistics
    total_lookups: u64,
    total_hits: u64,
    total_prefix_match_len: u64,
    eviction_count: u64,
}

impl PrefixCache {
    /// Create a new, empty `PrefixCache`.
    pub fn new(config: PrefixCacheConfig) -> Self {
        Self {
            config,
            root: TrieNode::new(),
            entries: HashMap::new(),
            next_id: 0,
            current_memory: 0,
            total_lookups: 0,
            total_hits: 0,
            total_prefix_match_len: 0,
            eviction_count: 0,
        }
    }

    /// Look up the longest cached prefix that matches the beginning of
    /// `tokens`.
    ///
    /// Returns `Some((matched_prefix_len, entry))` when a match is found, or
    /// `None` if no cached prefix matches.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<(usize, CacheEntry)> {
        self.total_lookups += 1;

        let mut node = &self.root;
        let mut best: Option<(usize, u64)> = None;

        for (depth, &token) in tokens.iter().enumerate() {
            match node.children.get(&token) {
                Some(child) => {
                    if let Some(id) = child.entry_id {
                        best = Some((depth + 1, id));
                    }
                    node = child;
                }
                None => break,
            }
        }

        if let Some((matched_len, id)) = best {
            if let Some(entry) = self.entries.get_mut(&id) {
                entry.hits += 1;
                entry.last_access = Instant::now();
                let snapshot = entry.clone();
                self.total_hits += 1;
                self.total_prefix_match_len += matched_len as u64;
                debug!("prefix cache hit: matched {matched_len} tokens (entry {id})");
                return Some((matched_len, snapshot));
            }
        }

        debug!("prefix cache miss");
        None
    }

    /// Insert a new cached state keyed by `tokens`.
    ///
    /// If the prefix is shorter than [`PrefixCacheConfig::min_prefix_length`]
    /// an error is returned. If capacity limits are exceeded the configured
    /// eviction policy is applied before the insert.
    pub fn insert(&mut self, tokens: &[u32], state: Vec<u8>) -> Result<()> {
        if tokens.len() < self.config.min_prefix_length {
            bail!(
                "prefix length {} is below minimum {}",
                tokens.len(),
                self.config.min_prefix_length,
            );
        }

        let state_size = state.len();

        // Evict until we have room.
        while self.entries.len() >= self.config.max_entries
            || self.current_memory + state_size > self.config.max_memory_bytes
        {
            if !self.evict() {
                bail!("unable to free space for new prefix cache entry");
            }
        }

        // Remove an existing entry for the exact same prefix, if any.
        if let Some(old_id) = self.trie_lookup_exact(tokens) {
            self.remove_entry(old_id);
        }

        let id = self.next_id;
        self.next_id += 1;

        let now = Instant::now();
        let entry = CacheEntry {
            token_prefix: tokens.to_vec(),
            cached_state: state,
            hits: 0,
            last_access: now,
            created_at: now,
            size_bytes: state_size,
        };

        self.current_memory += state_size;
        self.entries.insert(id, entry);

        // Walk / create trie path.
        let mut node = &mut self.root;
        for &token in tokens {
            node = node.children.entry(token).or_insert_with(TrieNode::new);
        }
        node.entry_id = Some(id);

        debug!(
            "prefix cache insert: {len} tokens, {state_size} bytes (id {id})",
            len = tokens.len()
        );
        Ok(())
    }

    /// Evict one entry according to the configured policy. Returns `true` if
    /// an entry was removed.
    pub fn evict(&mut self) -> bool {
        let victim_id = match self.config.eviction_policy {
            EvictionPolicy::LRU => {
                self.entries.iter().min_by_key(|(_, e)| e.last_access).map(|(&id, _)| id)
            }
            EvictionPolicy::LFU => {
                self.entries.iter().min_by_key(|(_, e)| (e.hits, e.last_access)).map(|(&id, _)| id)
            }
            EvictionPolicy::FIFO => {
                self.entries.iter().min_by_key(|(_, e)| e.created_at).map(|(&id, _)| id)
            }
            EvictionPolicy::TTL => {
                let now = Instant::now();
                let ttl = std::time::Duration::from_secs(self.config.ttl_seconds);
                self.entries
                    .iter()
                    .filter(|(_, e)| now.duration_since(e.created_at) >= ttl)
                    .min_by_key(|(_, e)| e.created_at)
                    .map(|(&id, _)| id)
                    // Fall back to oldest if nothing has expired yet.
                    .or_else(|| {
                        self.entries.iter().min_by_key(|(_, e)| e.created_at).map(|(&id, _)| id)
                    })
            }
        };

        if let Some(id) = victim_id {
            self.remove_entry(id);
            self.eviction_count += 1;
            debug!("prefix cache eviction: entry {id}");
            true
        } else {
            false
        }
    }

    /// Invalidate (remove) the cached entry whose prefix matches `tokens`
    /// exactly.
    pub fn invalidate(&mut self, tokens: &[u32]) {
        if let Some(id) = self.trie_lookup_exact(tokens) {
            self.remove_entry(id);
            debug!("prefix cache invalidate: {len} tokens", len = tokens.len());
        }
    }

    /// Remove all entries and reset statistics.
    pub fn clear(&mut self) {
        self.root = TrieNode::new();
        self.entries.clear();
        self.current_memory = 0;
        self.total_lookups = 0;
        self.total_hits = 0;
        self.total_prefix_match_len = 0;
        self.eviction_count = 0;
        debug!("prefix cache cleared");
    }

    /// Return current aggregate statistics.
    pub fn stats(&self) -> PrefixCacheStats {
        let (hit_rate, miss_rate) = if self.total_lookups > 0 {
            let hr = self.total_hits as f64 / self.total_lookups as f64;
            (hr, 1.0 - hr)
        } else {
            (0.0, 0.0)
        };

        let avg_prefix_match_length = if self.total_hits > 0 {
            self.total_prefix_match_len as f64 / self.total_hits as f64
        } else {
            0.0
        };

        PrefixCacheStats {
            hit_rate,
            miss_rate,
            eviction_count: self.eviction_count,
            memory_usage: self.current_memory,
            avg_prefix_match_length,
        }
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Walk the trie for `tokens` and return the entry id stored at the
    /// terminal node (exact match only).
    fn trie_lookup_exact(&self, tokens: &[u32]) -> Option<u64> {
        let mut node = &self.root;
        for &token in tokens {
            node = node.children.get(&token)?;
        }
        node.entry_id
    }

    /// Remove an entry by id, cleaning up memory accounting and the
    /// corresponding trie path.
    fn remove_entry(&mut self, id: u64) {
        if let Some(entry) = self.entries.remove(&id) {
            self.current_memory -= entry.size_bytes;
            self.trie_remove(&entry.token_prefix, id);
        }
    }

    /// Remove the trie leaf for `tokens` (and prune empty ancestors).
    fn trie_remove(&mut self, tokens: &[u32], id: u64) {
        Self::trie_remove_recursive(&mut self.root, tokens, 0, id);
    }

    /// Returns `true` if the caller should remove this node from its parent.
    fn trie_remove_recursive(node: &mut TrieNode, tokens: &[u32], depth: usize, id: u64) -> bool {
        if depth == tokens.len() {
            if node.entry_id == Some(id) {
                node.entry_id = None;
            }
            return node.children.is_empty() && node.entry_id.is_none();
        }

        let token = tokens[depth];
        let should_remove = if let Some(child) = node.children.get_mut(&token) {
            Self::trie_remove_recursive(child, tokens, depth + 1, id)
        } else {
            return false;
        };

        if should_remove {
            node.children.remove(&token);
        }

        node.children.is_empty() && node.entry_id.is_none()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> PrefixCacheConfig {
        PrefixCacheConfig { min_prefix_length: 1, ..Default::default() }
    }

    // -- basic insert & lookup ------------------------------------------

    #[test]
    fn test_insert_and_lookup_exact() {
        let mut cache = PrefixCache::new(default_cfg());
        let tokens = vec![10, 20, 30, 40];
        let state = vec![0xAB; 128];

        cache.insert(&tokens, state.clone()).unwrap();

        let result = cache.lookup(&tokens);
        assert!(result.is_some());
        let (matched_len, entry) = result.unwrap();
        assert_eq!(matched_len, 4);
        assert_eq!(entry.cached_state, state);
    }

    #[test]
    fn test_insert_and_lookup_with_longer_query() {
        let mut cache = PrefixCache::new(default_cfg());
        cache.insert(&[1, 2, 3], vec![1; 32]).unwrap();

        let result = cache.lookup(&[1, 2, 3, 4, 5]);
        assert!(result.is_some());
        let (matched_len, _) = result.unwrap();
        assert_eq!(matched_len, 3);
    }

    // -- prefix matching (longest match) --------------------------------

    #[test]
    fn test_longest_prefix_match() {
        let mut cache = PrefixCache::new(default_cfg());
        cache.insert(&[1, 2], vec![0xAA; 16]).unwrap();
        cache.insert(&[1, 2, 3, 4], vec![0xBB; 16]).unwrap();

        // Query [1,2,3,4,5] should match the longer prefix [1,2,3,4].
        let (matched_len, entry) = cache.lookup(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(matched_len, 4);
        assert_eq!(entry.cached_state, vec![0xBB; 16]);
    }

    #[test]
    fn test_partial_prefix_match() {
        let mut cache = PrefixCache::new(default_cfg());
        cache.insert(&[1, 2, 3, 4, 5], vec![0xCC; 16]).unwrap();

        // Query only matches first 3 tokens — there is a cached entry at
        // depth 5 but the query stops at 3 so no entry is reached.
        let result = cache.lookup(&[1, 2, 3]);
        assert!(result.is_none());
    }

    // -- overlapping prefixes -------------------------------------------

    #[test]
    fn test_overlapping_prefixes() {
        let mut cache = PrefixCache::new(default_cfg());
        cache.insert(&[1, 2, 3], vec![0x01; 8]).unwrap();
        cache.insert(&[1, 2, 4], vec![0x02; 8]).unwrap();

        let (len_a, entry_a) = cache.lookup(&[1, 2, 3, 99]).unwrap();
        assert_eq!(len_a, 3);
        assert_eq!(entry_a.cached_state, vec![0x01; 8]);

        let (len_b, entry_b) = cache.lookup(&[1, 2, 4, 99]).unwrap();
        assert_eq!(len_b, 3);
        assert_eq!(entry_b.cached_state, vec![0x02; 8]);
    }

    // -- empty cache lookup ---------------------------------------------

    #[test]
    fn test_empty_cache_lookup() {
        let mut cache = PrefixCache::new(default_cfg());
        assert!(cache.lookup(&[1, 2, 3]).is_none());
        assert!(cache.is_empty());
    }

    // -- LRU eviction ---------------------------------------------------

    #[test]
    fn test_lru_eviction() {
        let cfg = PrefixCacheConfig {
            max_entries: 2,
            eviction_policy: EvictionPolicy::LRU,
            min_prefix_length: 1,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(cfg);

        cache.insert(&[1], vec![0x01; 8]).unwrap();
        cache.insert(&[2], vec![0x02; 8]).unwrap();

        // Access entry [1] so it becomes recently used.
        let _ = cache.lookup(&[1]);

        // Insert a third entry — should evict [2] (least recently used).
        cache.insert(&[3], vec![0x03; 8]).unwrap();

        assert!(cache.lookup(&[1]).is_some(), "entry [1] should survive (recently used)");
        assert!(cache.lookup(&[2]).is_none(), "entry [2] should be evicted");
        assert!(cache.lookup(&[3]).is_some(), "entry [3] should exist");
    }

    // -- LFU eviction ---------------------------------------------------

    #[test]
    fn test_lfu_eviction() {
        let cfg = PrefixCacheConfig {
            max_entries: 2,
            eviction_policy: EvictionPolicy::LFU,
            min_prefix_length: 1,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(cfg);

        cache.insert(&[1], vec![0x01; 8]).unwrap();
        cache.insert(&[2], vec![0x02; 8]).unwrap();

        // Access entry [1] several times to boost its frequency.
        for _ in 0..5 {
            let _ = cache.lookup(&[1]);
        }

        // Insert a third — should evict [2] (least frequently used).
        cache.insert(&[3], vec![0x03; 8]).unwrap();

        assert!(cache.lookup(&[1]).is_some(), "entry [1] should survive (high frequency)");
        assert!(cache.lookup(&[2]).is_none(), "entry [2] should be evicted");
    }

    // -- memory limit enforcement ---------------------------------------

    #[test]
    fn test_memory_limit_enforcement() {
        let cfg = PrefixCacheConfig {
            max_entries: 100,
            max_memory_bytes: 64,
            eviction_policy: EvictionPolicy::LRU,
            min_prefix_length: 1,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(cfg);

        cache.insert(&[1], vec![0; 32]).unwrap();
        cache.insert(&[2], vec![0; 32]).unwrap();

        // Total would be 96 > 64 so one entry must have been evicted.
        assert!(cache.len() <= 2);
        assert!(cache.stats().memory_usage <= 64);
    }

    // -- max entries limit ----------------------------------------------

    #[test]
    fn test_max_entries_limit() {
        let cfg = PrefixCacheConfig {
            max_entries: 3,
            eviction_policy: EvictionPolicy::FIFO,
            min_prefix_length: 1,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(cfg);

        for i in 0..5u32 {
            cache.insert(&[i], vec![i as u8; 8]).unwrap();
        }

        assert!(cache.len() <= 3);
    }

    // -- hit / miss statistics ------------------------------------------

    #[test]
    fn test_hit_miss_statistics() {
        let mut cache = PrefixCache::new(default_cfg());
        cache.insert(&[1, 2, 3], vec![0; 16]).unwrap();

        // 1 hit
        let _ = cache.lookup(&[1, 2, 3]);
        // 1 miss
        let _ = cache.lookup(&[9, 8, 7]);

        let stats = cache.stats();
        assert!((stats.hit_rate - 0.5).abs() < 1e-9);
        assert!((stats.miss_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_avg_prefix_match_length() {
        let mut cache = PrefixCache::new(default_cfg());
        cache.insert(&[1, 2], vec![0; 8]).unwrap();
        cache.insert(&[1, 2, 3, 4], vec![0; 8]).unwrap();

        // Match length 4
        let _ = cache.lookup(&[1, 2, 3, 4, 5]);
        // Match length 2 (query doesn't reach depth 4)
        let _ = cache.lookup(&[1, 2, 99]);

        let stats = cache.stats();
        // avg = (4 + 2) / 2 = 3.0
        assert!((stats.avg_prefix_match_length - 3.0).abs() < 1e-9);
    }

    // -- cache invalidation ---------------------------------------------

    #[test]
    fn test_invalidation() {
        let mut cache = PrefixCache::new(default_cfg());
        cache.insert(&[1, 2, 3], vec![0; 16]).unwrap();

        assert!(cache.lookup(&[1, 2, 3]).is_some());

        cache.invalidate(&[1, 2, 3]);
        assert!(cache.lookup(&[1, 2, 3]).is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_invalidation_preserves_siblings() {
        let mut cache = PrefixCache::new(default_cfg());
        cache.insert(&[1, 2, 3], vec![0xAA; 8]).unwrap();
        cache.insert(&[1, 2, 4], vec![0xBB; 8]).unwrap();

        cache.invalidate(&[1, 2, 3]);

        assert!(cache.lookup(&[1, 2, 3]).is_none());
        assert!(cache.lookup(&[1, 2, 4]).is_some());
    }

    // -- min prefix length ----------------------------------------------

    #[test]
    fn test_min_prefix_length_rejected() {
        let cfg = PrefixCacheConfig { min_prefix_length: 4, ..Default::default() };
        let mut cache = PrefixCache::new(cfg);
        assert!(cache.insert(&[1, 2], vec![0; 8]).is_err());
    }

    // -- eviction counter -----------------------------------------------

    #[test]
    fn test_eviction_counter() {
        let cfg = PrefixCacheConfig {
            max_entries: 1,
            eviction_policy: EvictionPolicy::LRU,
            min_prefix_length: 1,
            ..Default::default()
        };
        let mut cache = PrefixCache::new(cfg);

        cache.insert(&[1], vec![0; 8]).unwrap();
        cache.insert(&[2], vec![0; 8]).unwrap();
        cache.insert(&[3], vec![0; 8]).unwrap();

        assert_eq!(cache.stats().eviction_count, 2);
    }

    // -- clear resets everything ----------------------------------------

    #[test]
    fn test_clear() {
        let mut cache = PrefixCache::new(default_cfg());
        cache.insert(&[1, 2, 3], vec![0; 16]).unwrap();
        let _ = cache.lookup(&[1, 2, 3]);

        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.stats().memory_usage, 0);
        assert_eq!(cache.stats().eviction_count, 0);
        assert!((cache.stats().hit_rate).abs() < 1e-9);
    }

    // -- duplicate insert replaces --------------------------------------

    #[test]
    fn test_duplicate_insert_replaces() {
        let mut cache = PrefixCache::new(default_cfg());
        cache.insert(&[1, 2, 3], vec![0xAA; 16]).unwrap();
        cache.insert(&[1, 2, 3], vec![0xBB; 16]).unwrap();

        assert_eq!(cache.len(), 1);
        let (_, entry) = cache.lookup(&[1, 2, 3]).unwrap();
        assert_eq!(entry.cached_state, vec![0xBB; 16]);
    }
}
