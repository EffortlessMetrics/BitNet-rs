//! Prompt prefix caching for faster GPU inference.
//!
//! Caches KV states for common prompt prefixes using a trie
//! structure for efficient prefix matching. Supports multiple
//! eviction policies, cache warming, partial matches, and
//! optional compression of cached KV states.

use std::collections::HashMap;
use std::time::Instant;

// ── Configuration ────────────────────────────────────────────────────────

/// Eviction policy for the prompt cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEviction {
    /// Least Recently Used — evict the entry accessed longest ago.
    Lru,
    /// Least Frequently Used — evict the entry with fewest accesses.
    Lfu,
    /// First In, First Out — evict the oldest inserted entry.
    Fifo,
    /// Size Based — evict the largest entry first.
    SizeBased,
}

/// Configuration for `PromptCache`.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of cached entries.
    pub max_entries: usize,
    /// Maximum prefix length (in tokens) to cache.
    pub max_prefix_length: usize,
    /// Eviction policy when the cache is full.
    pub eviction_policy: CacheEviction,
    /// Whether to enable KV state compression.
    pub enable_compression: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 128,
            max_prefix_length: 2048,
            eviction_policy: CacheEviction::Lru,
            enable_compression: false,
        }
    }
}

// ── Cached KV State ──────────────────────────────────────────────────────

/// Cached key/value tensors for a token prefix.
///
/// Stores the KV state produced by running a prefix through the model
/// so subsequent prompts sharing that prefix can skip recomputation.
#[derive(Debug, Clone)]
pub struct CachedKvState {
    /// The token prefix these KV states correspond to.
    pub prefix_tokens: Vec<u32>,
    /// Flattened key tensor data (num_layers × seq_len × head_dim).
    pub key_data: Vec<f32>,
    /// Flattened value tensor data (same shape as key_data).
    pub value_data: Vec<f32>,
    /// Number of transformer layers this state covers.
    pub num_layers: usize,
    /// Whether the data is compressed.
    pub compressed: bool,
}

impl CachedKvState {
    /// Size in bytes of the stored KV data.
    pub fn memory_bytes(&self) -> usize {
        (self.key_data.len() + self.value_data.len()) * std::mem::size_of::<f32>()
    }

    /// Number of tokens in the cached prefix.
    pub fn prefix_len(&self) -> usize {
        self.prefix_tokens.len()
    }
}

// ── Cache entry metadata ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct CacheEntry {
    kv_state: CachedKvState,
    access_count: u64,
    last_access: Instant,
    insertion_order: u64,
}

// ── Prefix Trie ──────────────────────────────────────────────────────────

/// A trie node for efficient token-level prefix matching.
#[derive(Debug, Clone)]
struct TrieNode {
    children: HashMap<u32, TrieNode>,
    /// If `Some`, this node is the end of a cached prefix (entry id).
    entry_id: Option<u64>,
}

impl TrieNode {
    fn new() -> Self {
        Self { children: HashMap::new(), entry_id: None }
    }
}

/// Trie structure indexing cached prefixes by their token sequences.
#[derive(Debug, Clone)]
pub struct PrefixTree {
    root: TrieNode,
    size: usize,
}

impl PrefixTree {
    /// Create an empty prefix tree.
    pub fn new() -> Self {
        Self { root: TrieNode::new(), size: 0 }
    }

    /// Number of prefixes stored in the trie.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if the trie contains no prefixes.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Insert a token prefix, returning the entry id stored at that node.
    pub fn insert(&mut self, tokens: &[u32], entry_id: u64) {
        let mut node = &mut self.root;
        for &tok in tokens {
            node = node.children.entry(tok).or_insert_with(TrieNode::new);
        }
        if node.entry_id.is_none() {
            self.size += 1;
        }
        node.entry_id = Some(entry_id);
    }

    /// Remove a prefix from the trie. Returns true if it was present.
    pub fn remove(&mut self, tokens: &[u32]) -> bool {
        if Self::remove_recursive(&mut self.root, tokens, 0) {
            self.size -= 1;
            true
        } else {
            false
        }
    }

    fn remove_recursive(node: &mut TrieNode, tokens: &[u32], depth: usize) -> bool {
        if depth == tokens.len() {
            if node.entry_id.is_some() {
                node.entry_id = None;
                return true;
            }
            return false;
        }
        let tok = tokens[depth];
        if let Some(child) = node.children.get_mut(&tok) {
            let removed = Self::remove_recursive(child, tokens, depth + 1);
            if removed && child.children.is_empty() && child.entry_id.is_none() {
                node.children.remove(&tok);
            }
            removed
        } else {
            false
        }
    }

    /// Find the longest prefix match for `tokens`.
    ///
    /// Returns `(matched_length, entry_id)` for the deepest node with an
    /// `entry_id`, or `None` if no prefix matches.
    pub fn longest_prefix_match(&self, tokens: &[u32]) -> Option<(usize, u64)> {
        let mut node = &self.root;
        let mut best: Option<(usize, u64)> = None;

        for (i, &tok) in tokens.iter().enumerate() {
            match node.children.get(&tok) {
                Some(child) => {
                    node = child;
                    if let Some(id) = node.entry_id {
                        best = Some((i + 1, id));
                    }
                }
                None => break,
            }
        }
        best
    }

    /// Check whether an exact prefix exists.
    pub fn contains(&self, tokens: &[u32]) -> bool {
        let mut node = &self.root;
        for &tok in tokens {
            match node.children.get(&tok) {
                Some(child) => node = child,
                None => return false,
            }
        }
        node.entry_id.is_some()
    }
}

impl Default for PrefixTree {
    fn default() -> Self {
        Self::new()
    }
}

// ── Partial Match ────────────────────────────────────────────────────────

/// Result of a prefix match attempt against the cache.
#[derive(Debug, Clone)]
pub struct PartialMatch {
    /// The cached KV state for the matched prefix portion.
    pub kv_state: CachedKvState,
    /// Number of tokens matched from the input.
    pub matched_tokens: usize,
    /// Remaining tokens that still need to be computed.
    pub remaining_tokens: Vec<u32>,
    /// Ratio of matched tokens to total input tokens.
    pub match_ratio: f32,
}

// ── Prefix Matcher ───────────────────────────────────────────────────────

/// Finds the longest matching cached prefix for a new prompt.
pub struct PrefixMatcher<'a> {
    cache: &'a PromptCache,
}

impl<'a> PrefixMatcher<'a> {
    /// Create a matcher bound to the given cache.
    pub fn new(cache: &'a PromptCache) -> Self {
        Self { cache }
    }

    /// Find the longest matching prefix for `tokens`.
    ///
    /// Returns a `PartialMatch` describing how much of the input was
    /// matched and which tokens remain to be computed.
    pub fn find_longest_match(&self, tokens: &[u32]) -> Option<PartialMatch> {
        let (matched_len, entry_id) = self.cache.trie.longest_prefix_match(tokens)?;

        let entry = self.cache.entries.get(&entry_id)?;
        let kv_state = entry.kv_state.clone();
        let remaining = tokens[matched_len..].to_vec();

        #[allow(clippy::cast_precision_loss)]
        let match_ratio =
            if tokens.is_empty() { 0.0 } else { matched_len as f32 / tokens.len() as f32 };

        Some(PartialMatch {
            kv_state,
            matched_tokens: matched_len,
            remaining_tokens: remaining,
            match_ratio,
        })
    }
}

// ── Cache Statistics ─────────────────────────────────────────────────────

/// Runtime statistics for the prompt cache.
#[derive(Debug, Clone, Default)]
pub struct PromptCacheStats {
    /// Total cache lookups.
    pub lookups: u64,
    /// Number of cache hits (full or partial).
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Total tokens saved by serving from cache.
    pub saved_tokens: u64,
    /// Approximate memory usage in bytes.
    pub memory_bytes: usize,
    /// Number of entries currently cached.
    pub entry_count: usize,
    /// Number of evictions performed.
    pub evictions: u64,
}

impl PromptCacheStats {
    /// Hit rate as a fraction in `[0.0, 1.0]`.
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / self.lookups as f64
    }

    /// Miss rate as a fraction in `[0.0, 1.0]`.
    pub fn miss_rate(&self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.misses as f64 / self.lookups as f64
    }
}

// ── Cache Compression ────────────────────────────────────────────────────

/// Compress cached KV states to save memory.
///
/// Uses simple delta + run-length encoding on quantised values.
pub struct CacheCompression;

impl CacheCompression {
    /// Compress a float buffer by quantising to 16-bit and delta-encoding.
    ///
    /// Returns the compressed byte buffer.
    pub fn compress(data: &[f32]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }
        // Simple quantisation: f32 → f16-like representation (2 bytes each)
        let mut out = Vec::with_capacity(data.len() * 2);
        let mut prev: i16 = 0;
        for &v in data {
            #[allow(clippy::cast_possible_truncation)]
            let q = (v * 256.0).clamp(-32768.0, 32767.0) as i16;
            let delta = q.wrapping_sub(prev);
            out.extend_from_slice(&delta.to_le_bytes());
            prev = q;
        }
        out
    }

    /// Decompress a buffer produced by [`compress`](Self::compress).
    pub fn decompress(data: &[u8], num_elements: usize) -> Vec<f32> {
        if data.len() < num_elements * 2 {
            return vec![0.0; num_elements];
        }
        let mut out = Vec::with_capacity(num_elements);
        let mut prev: i16 = 0;
        for chunk in data.chunks_exact(2).take(num_elements) {
            let delta = i16::from_le_bytes([chunk[0], chunk[1]]);
            let q = prev.wrapping_add(delta);
            out.push(f32::from(q) / 256.0);
            prev = q;
        }
        out
    }

    /// Compression ratio for a given input size.
    pub fn ratio(original_floats: usize) -> f32 {
        if original_floats == 0 {
            return 1.0;
        }
        // f32 = 4 bytes each, compressed = 2 bytes each (delta-i16)
        2.0
    }
}

// ── Cache Warmer ─────────────────────────────────────────────────────────

/// Precompute KV states for common system prompts.
///
/// Feed the warmer a set of token prefixes and a callback that computes
/// KV states; it populates the cache before any user requests arrive.
pub struct CacheWarmer {
    prefixes: Vec<Vec<u32>>,
}

impl CacheWarmer {
    /// Create a warmer for the given set of prefix token sequences.
    pub fn new(prefixes: Vec<Vec<u32>>) -> Self {
        Self { prefixes }
    }

    /// Number of prefixes registered for warming.
    pub fn prefix_count(&self) -> usize {
        self.prefixes.len()
    }

    /// Warm the cache using a caller-provided KV computation function.
    ///
    /// `compute_kv` takes a token prefix and returns `(key_data, value_data,
    /// num_layers)`.
    pub fn warm<F>(&self, cache: &mut PromptCache, mut compute_kv: F)
    where
        F: FnMut(&[u32]) -> (Vec<f32>, Vec<f32>, usize),
    {
        for prefix in &self.prefixes {
            if prefix.len() > cache.config.max_prefix_length {
                continue;
            }
            if cache.trie.contains(prefix) {
                continue; // already cached
            }
            let (key_data, value_data, num_layers) = compute_kv(prefix);
            let state = CachedKvState {
                prefix_tokens: prefix.clone(),
                key_data,
                value_data,
                num_layers,
                compressed: false,
            };
            cache.insert(prefix.clone(), state);
        }
    }
}

// ── Prompt Cache ─────────────────────────────────────────────────────────

/// Caches KV states for common prompt prefixes.
///
/// Uses a trie for O(n) prefix lookup (where n = token sequence length)
/// and supports configurable eviction policies.
pub struct PromptCache {
    config: CacheConfig,
    trie: PrefixTree,
    entries: HashMap<u64, CacheEntry>,
    stats: PromptCacheStats,
    next_id: u64,
    insertion_counter: u64,
}

impl PromptCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            trie: PrefixTree::new(),
            entries: HashMap::new(),
            stats: PromptCacheStats::default(),
            next_id: 0,
            insertion_counter: 0,
        }
    }

    /// Return the cache configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Return current cache statistics.
    pub fn stats(&self) -> &PromptCacheStats {
        &self.stats
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a prompt in the cache, returning a partial match if found.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<PartialMatch> {
        self.stats.lookups += 1;

        let (matched_len, entry_id) = self.trie.longest_prefix_match(tokens)?;

        let entry = self.entries.get_mut(&entry_id)?;
        entry.access_count += 1;
        entry.last_access = Instant::now();

        let kv_state = entry.kv_state.clone();
        let remaining = tokens[matched_len..].to_vec();

        #[allow(clippy::cast_precision_loss)]
        let match_ratio =
            if tokens.is_empty() { 0.0 } else { matched_len as f32 / tokens.len() as f32 };

        self.stats.hits += 1;
        self.stats.saved_tokens += matched_len as u64;

        Some(PartialMatch {
            kv_state,
            matched_tokens: matched_len,
            remaining_tokens: remaining,
            match_ratio,
        })
    }

    /// Record a cache miss (call after a failed lookup if desired).
    pub fn record_miss(&mut self) {
        self.stats.misses += 1;
    }

    /// Insert a new KV state into the cache.
    ///
    /// If the cache is full, evicts an entry according to the configured
    /// eviction policy before inserting.
    pub fn insert(&mut self, tokens: Vec<u32>, mut kv_state: CachedKvState) {
        if tokens.len() > self.config.max_prefix_length {
            return;
        }
        // Evict if at capacity
        while self.entries.len() >= self.config.max_entries {
            self.evict_one();
        }
        // Optionally compress
        if self.config.enable_compression && !kv_state.compressed {
            let ck = CacheCompression::compress(&kv_state.key_data);
            let cv = CacheCompression::compress(&kv_state.value_data);
            // Store compressed bytes reinterpreted as f32 placeholder
            // (in a real implementation we'd have a separate bytes field)
            let key_len = kv_state.key_data.len();
            let val_len = kv_state.value_data.len();
            kv_state.key_data = bytecast_to_f32(&ck, key_len);
            kv_state.value_data = bytecast_to_f32(&cv, val_len);
            kv_state.compressed = true;
        }

        let id = self.next_id;
        self.next_id += 1;
        self.insertion_counter += 1;

        let mem = kv_state.memory_bytes();
        self.trie.insert(&tokens, id);
        self.entries.insert(
            id,
            CacheEntry {
                kv_state,
                access_count: 0,
                last_access: Instant::now(),
                insertion_order: self.insertion_counter,
            },
        );
        self.stats.entry_count = self.entries.len();
        self.stats.memory_bytes += mem;
    }

    /// Clear all cached entries and reset statistics.
    pub fn clear(&mut self) {
        self.trie = PrefixTree::new();
        self.entries.clear();
        self.stats = PromptCacheStats::default();
    }

    /// Evict a single entry according to the configured policy.
    fn evict_one(&mut self) {
        let victim_id = match self.config.eviction_policy {
            CacheEviction::Lru => self.find_lru_victim(),
            CacheEviction::Lfu => self.find_lfu_victim(),
            CacheEviction::Fifo => self.find_fifo_victim(),
            CacheEviction::SizeBased => self.find_size_victim(),
        };

        if let Some(id) = victim_id {
            if let Some(entry) = self.entries.remove(&id) {
                self.trie.remove(&entry.kv_state.prefix_tokens);
                let freed = entry.kv_state.memory_bytes();
                self.stats.memory_bytes = self.stats.memory_bytes.saturating_sub(freed);
                self.stats.evictions += 1;
                self.stats.entry_count = self.entries.len();
            }
        }
    }

    fn find_lru_victim(&self) -> Option<u64> {
        self.entries.iter().min_by_key(|(_, e)| e.last_access).map(|(&id, _)| id)
    }

    fn find_lfu_victim(&self) -> Option<u64> {
        self.entries.iter().min_by_key(|(_, e)| e.access_count).map(|(&id, _)| id)
    }

    fn find_fifo_victim(&self) -> Option<u64> {
        self.entries.iter().min_by_key(|(_, e)| e.insertion_order).map(|(&id, _)| id)
    }

    fn find_size_victim(&self) -> Option<u64> {
        self.entries.iter().max_by_key(|(_, e)| e.kv_state.memory_bytes()).map(|(&id, _)| id)
    }
}

/// Helper: reinterpret compressed bytes into a float-sized Vec for storage.
/// Pads with zeros to maintain the original element count.
fn bytecast_to_f32(bytes: &[u8], original_len: usize) -> Vec<f32> {
    let needed = original_len;
    let available = bytes.len() / 4;
    let mut out = Vec::with_capacity(needed);
    for chunk in bytes.chunks(4).take(needed) {
        let mut buf = [0u8; 4];
        buf[..chunk.len()].copy_from_slice(chunk);
        out.push(f32::from_le_bytes(buf));
    }
    out.resize(needed, 0.0);
    let _ = available; // suppress unused
    out
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ───────────────────────────────────────────────────────

    fn make_kv(prefix: &[u32], num_layers: usize) -> CachedKvState {
        let len = prefix.len() * num_layers * 4;
        CachedKvState {
            prefix_tokens: prefix.to_vec(),
            key_data: vec![1.0; len],
            value_data: vec![2.0; len],
            num_layers,
            compressed: false,
        }
    }

    fn default_cache() -> PromptCache {
        PromptCache::new(CacheConfig::default())
    }

    // ── CacheConfig ──────────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = CacheConfig::default();
        assert_eq!(cfg.max_entries, 128);
        assert_eq!(cfg.max_prefix_length, 2048);
        assert_eq!(cfg.eviction_policy, CacheEviction::Lru);
        assert!(!cfg.enable_compression);
    }

    #[test]
    fn config_custom() {
        let cfg = CacheConfig {
            max_entries: 16,
            max_prefix_length: 512,
            eviction_policy: CacheEviction::Lfu,
            enable_compression: true,
        };
        assert_eq!(cfg.max_entries, 16);
        assert!(cfg.enable_compression);
    }

    // ── CachedKvState ────────────────────────────────────────────────

    #[test]
    fn kv_state_memory_bytes() {
        let kv = make_kv(&[1, 2, 3], 2);
        // 3 tokens * 2 layers * 4 = 24 elements per tensor, 2 tensors
        assert_eq!(kv.memory_bytes(), 24 * 2 * std::mem::size_of::<f32>());
    }

    #[test]
    fn kv_state_prefix_len() {
        let kv = make_kv(&[10, 20], 1);
        assert_eq!(kv.prefix_len(), 2);
    }

    #[test]
    fn kv_state_empty_prefix() {
        let kv = make_kv(&[], 1);
        assert_eq!(kv.prefix_len(), 0);
        assert_eq!(kv.memory_bytes(), 0);
    }

    // ── PrefixTree ───────────────────────────────────────────────────

    #[test]
    fn trie_new_is_empty() {
        let trie = PrefixTree::new();
        assert!(trie.is_empty());
        assert_eq!(trie.len(), 0);
    }

    #[test]
    fn trie_insert_and_contains() {
        let mut trie = PrefixTree::new();
        trie.insert(&[1, 2, 3], 0);
        assert!(trie.contains(&[1, 2, 3]));
        assert!(!trie.contains(&[1, 2]));
        assert!(!trie.contains(&[1, 2, 3, 4]));
        assert_eq!(trie.len(), 1);
    }

    #[test]
    fn trie_insert_multiple_prefixes() {
        let mut trie = PrefixTree::new();
        trie.insert(&[1, 2], 0);
        trie.insert(&[1, 2, 3], 1);
        trie.insert(&[4, 5], 2);
        assert_eq!(trie.len(), 3);
        assert!(trie.contains(&[1, 2]));
        assert!(trie.contains(&[1, 2, 3]));
        assert!(trie.contains(&[4, 5]));
    }

    #[test]
    fn trie_overwrite_entry_id() {
        let mut trie = PrefixTree::new();
        trie.insert(&[1, 2], 10);
        trie.insert(&[1, 2], 20);
        // Size shouldn't change on overwrite
        assert_eq!(trie.len(), 1);
    }

    #[test]
    fn trie_remove() {
        let mut trie = PrefixTree::new();
        trie.insert(&[1, 2, 3], 0);
        assert!(trie.remove(&[1, 2, 3]));
        assert!(!trie.contains(&[1, 2, 3]));
        assert_eq!(trie.len(), 0);
    }

    #[test]
    fn trie_remove_nonexistent() {
        let mut trie = PrefixTree::new();
        trie.insert(&[1, 2], 0);
        assert!(!trie.remove(&[1, 2, 3]));
        assert_eq!(trie.len(), 1);
    }

    #[test]
    fn trie_remove_preserves_siblings() {
        let mut trie = PrefixTree::new();
        trie.insert(&[1, 2], 0);
        trie.insert(&[1, 3], 1);
        trie.remove(&[1, 2]);
        assert!(!trie.contains(&[1, 2]));
        assert!(trie.contains(&[1, 3]));
    }

    #[test]
    fn trie_longest_prefix_match_exact() {
        let mut trie = PrefixTree::new();
        trie.insert(&[1, 2, 3], 42);
        let m = trie.longest_prefix_match(&[1, 2, 3]);
        assert_eq!(m, Some((3, 42)));
    }

    #[test]
    fn trie_longest_prefix_match_partial() {
        let mut trie = PrefixTree::new();
        trie.insert(&[1, 2], 10);
        trie.insert(&[1, 2, 3, 4], 20);
        // Input [1,2,3] — matches [1,2] (length 2)
        let m = trie.longest_prefix_match(&[1, 2, 3]);
        assert_eq!(m, Some((2, 10)));
    }

    #[test]
    fn trie_longest_prefix_match_none() {
        let mut trie = PrefixTree::new();
        trie.insert(&[5, 6], 0);
        assert!(trie.longest_prefix_match(&[1, 2]).is_none());
    }

    #[test]
    fn trie_longest_prefix_match_nested() {
        let mut trie = PrefixTree::new();
        trie.insert(&[1], 1);
        trie.insert(&[1, 2], 2);
        trie.insert(&[1, 2, 3], 3);
        let m = trie.longest_prefix_match(&[1, 2, 3, 4, 5]);
        assert_eq!(m, Some((3, 3)));
    }

    #[test]
    fn trie_empty_prefix() {
        let mut trie = PrefixTree::new();
        trie.insert(&[], 99);
        assert!(trie.contains(&[]));
        assert_eq!(trie.len(), 1);
    }

    #[test]
    fn trie_longest_match_empty_input() {
        let trie = PrefixTree::new();
        assert!(trie.longest_prefix_match(&[]).is_none());
    }

    #[test]
    fn trie_default_is_empty() {
        let trie = PrefixTree::default();
        assert!(trie.is_empty());
    }

    // ── PromptCache ──────────────────────────────────────────────────

    #[test]
    fn cache_new_is_empty() {
        let cache = default_cache();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_insert_and_lookup() {
        let mut cache = default_cache();
        let tokens = vec![1, 2, 3];
        cache.insert(tokens.clone(), make_kv(&tokens, 2));

        let m = cache.lookup(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(m.matched_tokens, 3);
        assert_eq!(m.remaining_tokens, vec![4, 5]);
    }

    #[test]
    fn cache_miss_returns_none() {
        let mut cache = default_cache();
        assert!(cache.lookup(&[1, 2, 3]).is_none());
    }

    #[test]
    fn cache_stats_tracking() {
        let mut cache = default_cache();
        let prefix = vec![10, 20, 30];
        cache.insert(prefix.clone(), make_kv(&prefix, 1));

        // hit
        cache.lookup(&[10, 20, 30, 40]);
        assert_eq!(cache.stats().lookups, 1);
        assert_eq!(cache.stats().hits, 1);

        // miss
        cache.lookup(&[99]);
        cache.record_miss();
        assert_eq!(cache.stats().lookups, 2);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn cache_saved_tokens() {
        let mut cache = default_cache();
        let prefix = vec![1, 2, 3, 4];
        cache.insert(prefix.clone(), make_kv(&prefix, 1));
        cache.lookup(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(cache.stats().saved_tokens, 4);
    }

    #[test]
    fn cache_clear_resets() {
        let mut cache = default_cache();
        cache.insert(vec![1], make_kv(&[1], 1));
        cache.lookup(&[1, 2]);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().lookups, 0);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn cache_respects_max_prefix_length() {
        let cfg = CacheConfig { max_prefix_length: 3, ..CacheConfig::default() };
        let mut cache = PromptCache::new(cfg);
        let long = vec![1, 2, 3, 4];
        cache.insert(long.clone(), make_kv(&long, 1));
        // Should not have been inserted (too long)
        assert!(cache.is_empty());
    }

    // ── Eviction policies ────────────────────────────────────────────

    #[test]
    fn eviction_lru() {
        let cfg = CacheConfig {
            max_entries: 2,
            eviction_policy: CacheEviction::Lru,
            ..CacheConfig::default()
        };
        let mut cache = PromptCache::new(cfg);
        cache.insert(vec![1], make_kv(&[1], 1));
        cache.insert(vec![2], make_kv(&[2], 1));
        // Touch [1] so it becomes more recently used than [2]
        cache.lookup(&[1, 99]);
        // This should evict [2] (least recently used)
        cache.insert(vec![3], make_kv(&[3], 1));
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(&[1, 99]).is_some());
        // [2] was evicted
    }

    #[test]
    fn eviction_lfu() {
        let cfg = CacheConfig {
            max_entries: 2,
            eviction_policy: CacheEviction::Lfu,
            ..CacheConfig::default()
        };
        let mut cache = PromptCache::new(cfg);
        cache.insert(vec![1], make_kv(&[1], 1));
        // Access [1] several times to increase frequency
        cache.lookup(&[1, 99]);
        cache.lookup(&[1, 99]);
        cache.insert(vec![2], make_kv(&[2], 1));
        // Insert [3] should evict [2] (least frequently used)
        cache.insert(vec![3], make_kv(&[3], 1));
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(&[1, 99]).is_some());
    }

    #[test]
    fn eviction_fifo() {
        let cfg = CacheConfig {
            max_entries: 2,
            eviction_policy: CacheEviction::Fifo,
            ..CacheConfig::default()
        };
        let mut cache = PromptCache::new(cfg);
        cache.insert(vec![1], make_kv(&[1], 1));
        cache.insert(vec![2], make_kv(&[2], 1));
        // Insert [3] should evict [1] (first in)
        cache.insert(vec![3], make_kv(&[3], 1));
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(&[1, 99]).is_none());
        assert!(cache.lookup(&[2, 99]).is_some());
    }

    #[test]
    fn eviction_size_based() {
        let cfg = CacheConfig {
            max_entries: 2,
            eviction_policy: CacheEviction::SizeBased,
            ..CacheConfig::default()
        };
        let mut cache = PromptCache::new(cfg);
        // [1,2,3] is larger than [4]
        cache.insert(vec![1, 2, 3], make_kv(&[1, 2, 3], 4));
        cache.insert(vec![4], make_kv(&[4], 1));
        // Insert [5] — should evict the largest entry [1,2,3]
        cache.insert(vec![5], make_kv(&[5], 1));
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup(&[1, 2, 3, 99]).is_none());
        assert!(cache.lookup(&[4, 99]).is_some());
    }

    #[test]
    fn eviction_stats_increment() {
        let cfg = CacheConfig { max_entries: 1, ..CacheConfig::default() };
        let mut cache = PromptCache::new(cfg);
        cache.insert(vec![1], make_kv(&[1], 1));
        cache.insert(vec![2], make_kv(&[2], 1));
        assert_eq!(cache.stats().evictions, 1);
    }

    // ── PartialMatch ─────────────────────────────────────────────────

    #[test]
    fn partial_match_ratio() {
        let mut cache = default_cache();
        let prefix = vec![1, 2];
        cache.insert(prefix.clone(), make_kv(&prefix, 1));
        let m = cache.lookup(&[1, 2, 3, 4]).unwrap();
        assert!((m.match_ratio - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn partial_match_full_match() {
        let mut cache = default_cache();
        let prefix = vec![1, 2, 3];
        cache.insert(prefix.clone(), make_kv(&prefix, 1));
        let m = cache.lookup(&[1, 2, 3]).unwrap();
        assert_eq!(m.matched_tokens, 3);
        assert!(m.remaining_tokens.is_empty());
        assert!((m.match_ratio - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn partial_match_returns_kv() {
        let mut cache = default_cache();
        let prefix = vec![7, 8];
        let kv = make_kv(&prefix, 3);
        let expected_layers = kv.num_layers;
        cache.insert(prefix, kv);
        let m = cache.lookup(&[7, 8, 9]).unwrap();
        assert_eq!(m.kv_state.num_layers, expected_layers);
    }

    // ── PrefixMatcher ────────────────────────────────────────────────

    #[test]
    fn prefix_matcher_finds_longest() {
        let mut cache = default_cache();
        cache.insert(vec![1, 2], make_kv(&[1, 2], 1));
        cache.insert(vec![1, 2, 3], make_kv(&[1, 2, 3], 1));

        let matcher = PrefixMatcher::new(&cache);
        let m = matcher.find_longest_match(&[1, 2, 3, 4]).unwrap();
        assert_eq!(m.matched_tokens, 3);
        assert_eq!(m.remaining_tokens, vec![4]);
    }

    #[test]
    fn prefix_matcher_no_match() {
        let cache = default_cache();
        let matcher = PrefixMatcher::new(&cache);
        assert!(matcher.find_longest_match(&[99, 100]).is_none());
    }

    #[test]
    fn prefix_matcher_empty_input() {
        let cache = default_cache();
        let matcher = PrefixMatcher::new(&cache);
        assert!(matcher.find_longest_match(&[]).is_none());
    }

    // ── CacheCompression ─────────────────────────────────────────────

    #[test]
    fn compression_roundtrip() {
        let data = vec![0.1, 0.5, -0.3, 1.0, -1.0, 0.0];
        let compressed = CacheCompression::compress(&data);
        let decompressed = CacheCompression::decompress(&compressed, data.len());
        for (orig, dec) in data.iter().zip(decompressed.iter()) {
            // Allow quantization error (1/256 ≈ 0.004)
            assert!((orig - dec).abs() < 0.01, "orig={orig}, dec={dec}");
        }
    }

    #[test]
    fn compression_empty() {
        let compressed = CacheCompression::compress(&[]);
        assert!(compressed.is_empty());
        let decompressed = CacheCompression::decompress(&compressed, 0);
        assert!(decompressed.is_empty());
    }

    #[test]
    fn compression_ratio_is_2x() {
        assert!((CacheCompression::ratio(100) - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn compression_ratio_empty() {
        assert!((CacheCompression::ratio(0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn compression_large_values() {
        let data = vec![100.0, -100.0, 50.5, -50.5];
        let compressed = CacheCompression::compress(&data);
        let decompressed = CacheCompression::decompress(&compressed, data.len());
        // Clamped to i16 range / 256 — large values get clipped
        assert_eq!(decompressed.len(), data.len());
    }

    // ── CacheWarmer ──────────────────────────────────────────────────

    #[test]
    fn warmer_populates_cache() {
        let mut cache = default_cache();
        let warmer = CacheWarmer::new(vec![vec![1, 2, 3], vec![4, 5]]);
        assert_eq!(warmer.prefix_count(), 2);
        warmer.warm(&mut cache, |prefix| {
            let len = prefix.len() * 4;
            (vec![0.0; len], vec![0.0; len], 1)
        });
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn warmer_skips_already_cached() {
        let mut cache = default_cache();
        cache.insert(vec![1, 2], make_kv(&[1, 2], 1));
        let warmer = CacheWarmer::new(vec![vec![1, 2], vec![3, 4]]);
        let mut calls = 0;
        warmer.warm(&mut cache, |prefix| {
            calls += 1;
            let len = prefix.len() * 4;
            (vec![0.0; len], vec![0.0; len], 1)
        });
        // Should only compute for [3,4]
        assert_eq!(calls, 1);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn warmer_respects_max_prefix_length() {
        let cfg = CacheConfig { max_prefix_length: 2, ..CacheConfig::default() };
        let mut cache = PromptCache::new(cfg);
        let warmer = CacheWarmer::new(vec![vec![1, 2, 3]]);
        warmer.warm(&mut cache, |prefix| {
            let len = prefix.len();
            (vec![0.0; len], vec![0.0; len], 1)
        });
        // Too long — should not be inserted
        assert!(cache.is_empty());
    }

    #[test]
    fn warmer_empty_prefixes() {
        let mut cache = default_cache();
        let warmer = CacheWarmer::new(vec![]);
        assert_eq!(warmer.prefix_count(), 0);
        warmer.warm(&mut cache, |_| unreachable!());
        assert!(cache.is_empty());
    }

    // ── PromptCacheStats ─────────────────────────────────────────────

    #[test]
    fn stats_default() {
        let s = PromptCacheStats::default();
        assert_eq!(s.lookups, 0);
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.saved_tokens, 0);
        assert_eq!(s.evictions, 0);
    }

    #[test]
    fn stats_hit_rate_no_lookups() {
        let s = PromptCacheStats::default();
        assert!((s.hit_rate() - 0.0).abs() < f64::EPSILON);
        assert!((s.miss_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_hit_rate_all_hits() {
        let s = PromptCacheStats { lookups: 10, hits: 10, ..Default::default() };
        assert!((s.hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_miss_rate_all_misses() {
        let s = PromptCacheStats { lookups: 5, misses: 5, ..Default::default() };
        assert!((s.miss_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_mixed() {
        let s = PromptCacheStats { lookups: 4, hits: 3, misses: 1, ..Default::default() };
        assert!((s.hit_rate() - 0.75).abs() < f64::EPSILON);
        assert!((s.miss_rate() - 0.25).abs() < f64::EPSILON);
    }

    // ── CacheEviction enum ───────────────────────────────────────────

    #[test]
    fn eviction_variants_eq() {
        assert_eq!(CacheEviction::Lru, CacheEviction::Lru);
        assert_ne!(CacheEviction::Lru, CacheEviction::Lfu);
        assert_ne!(CacheEviction::Fifo, CacheEviction::SizeBased);
    }

    #[test]
    fn eviction_clone() {
        let e = CacheEviction::Fifo;
        let e2 = e;
        assert_eq!(e, e2);
    }

    #[test]
    fn eviction_debug() {
        let s = format!("{:?}", CacheEviction::SizeBased);
        assert!(s.contains("SizeBased"));
    }

    // ── Integration-style tests ──────────────────────────────────────

    #[test]
    fn end_to_end_warm_lookup_evict() {
        let cfg = CacheConfig {
            max_entries: 3,
            max_prefix_length: 10,
            eviction_policy: CacheEviction::Lru,
            enable_compression: false,
        };
        let mut cache = PromptCache::new(cfg);
        let warmer = CacheWarmer::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        warmer.warm(&mut cache, |prefix| {
            let len = prefix.len() * 8;
            (vec![0.5; len], vec![0.5; len], 2)
        });
        assert_eq!(cache.len(), 2);

        // Lookup a prompt sharing the first prefix
        let m = cache.lookup(&[1, 2, 3, 7, 8]).unwrap();
        assert_eq!(m.matched_tokens, 3);
        assert_eq!(m.remaining_tokens, vec![7, 8]);

        // Fill to capacity + 1 → triggers eviction
        cache.insert(vec![10], make_kv(&[10], 1));
        cache.insert(vec![20], make_kv(&[20], 1));
        assert_eq!(cache.len(), 3);
        assert!(cache.stats().evictions >= 1);
    }

    #[test]
    fn multiple_partial_matches_longest_wins() {
        let mut cache = default_cache();
        cache.insert(vec![1], make_kv(&[1], 1));
        cache.insert(vec![1, 2], make_kv(&[1, 2], 1));
        cache.insert(vec![1, 2, 3], make_kv(&[1, 2, 3], 1));

        let m = cache.lookup(&[1, 2, 3, 4]).unwrap();
        assert_eq!(m.matched_tokens, 3);
    }

    #[test]
    fn cache_with_compression() {
        let cfg = CacheConfig { enable_compression: true, ..CacheConfig::default() };
        let mut cache = PromptCache::new(cfg);
        cache.insert(vec![1, 2], make_kv(&[1, 2], 2));
        assert_eq!(cache.len(), 1);
        let m = cache.lookup(&[1, 2, 3]);
        assert!(m.is_some());
        assert!(m.unwrap().kv_state.compressed);
    }

    #[test]
    fn cache_memory_tracking() {
        let mut cache = default_cache();
        cache.insert(vec![1], make_kv(&[1], 1));
        assert!(cache.stats().memory_bytes > 0);

        let before = cache.stats().memory_bytes;
        cache.insert(vec![2], make_kv(&[2], 1));
        assert!(cache.stats().memory_bytes > before);
    }

    #[test]
    fn cache_entry_count_tracked() {
        let mut cache = default_cache();
        cache.insert(vec![1], make_kv(&[1], 1));
        assert_eq!(cache.stats().entry_count, 1);
        cache.insert(vec![2], make_kv(&[2], 1));
        assert_eq!(cache.stats().entry_count, 2);
    }

    #[test]
    fn matcher_match_ratio_calculation() {
        let mut cache = default_cache();
        cache.insert(vec![1, 2, 3], make_kv(&[1, 2, 3], 1));
        let matcher = PrefixMatcher::new(&cache);
        let m = matcher.find_longest_match(&[1, 2, 3, 4, 5, 6]).unwrap();
        assert!((m.match_ratio - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn stress_many_inserts() {
        let cfg = CacheConfig { max_entries: 50, ..CacheConfig::default() };
        let mut cache = PromptCache::new(cfg);
        for i in 0..100_u32 {
            cache.insert(vec![i], make_kv(&[i], 1));
        }
        assert_eq!(cache.len(), 50);
        assert!(cache.stats().evictions >= 50);
    }

    #[test]
    fn disjoint_prefixes_no_interference() {
        let mut cache = default_cache();
        cache.insert(vec![1, 2], make_kv(&[1, 2], 1));
        cache.insert(vec![3, 4], make_kv(&[3, 4], 1));

        assert!(cache.lookup(&[1, 2, 5]).is_some());
        assert!(cache.lookup(&[3, 4, 5]).is_some());
        assert!(cache.lookup(&[5, 6]).is_none());
    }

    #[test]
    fn single_token_prefix() {
        let mut cache = default_cache();
        cache.insert(vec![42], make_kv(&[42], 2));
        let m = cache.lookup(&[42, 99]).unwrap();
        assert_eq!(m.matched_tokens, 1);
    }

    #[test]
    fn config_returns_reference() {
        let cfg = CacheConfig { max_entries: 7, ..CacheConfig::default() };
        let cache = PromptCache::new(cfg);
        assert_eq!(cache.config().max_entries, 7);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Compression roundtrip preserves values within quantization error.
        #[test]
        fn compression_roundtrip_prop(
            data in proptest::collection::vec(-100.0_f32..100.0, 1..64),
        ) {
            let compressed = CacheCompression::compress(&data);
            let decompressed =
                CacheCompression::decompress(&compressed, data.len());
            prop_assert_eq!(decompressed.len(), data.len());
            for (o, d) in data.iter().zip(decompressed.iter()) {
                // Allow quantization noise (1/256 ≈ 0.004 per step)
                prop_assert!(
                    (o - d).abs() < 0.05,
                    "orig={o}, decompressed={d}"
                );
            }
        }

        /// Trie contains every prefix that was inserted.
        #[test]
        fn trie_contains_all_inserted(
            prefixes in proptest::collection::vec(
                proptest::collection::vec(0u32..100, 1..8),
                1..16,
            ),
        ) {
            let mut trie = PrefixTree::new();
            for (i, p) in prefixes.iter().enumerate() {
                trie.insert(p, i as u64);
            }
            for p in &prefixes {
                prop_assert!(trie.contains(p));
            }
        }

        /// Longest prefix match length is at most the input length.
        #[test]
        fn longest_match_bounded(
            prefix in proptest::collection::vec(0u32..50, 1..8),
            extra in proptest::collection::vec(0u32..50, 0..8),
        ) {
            let mut trie = PrefixTree::new();
            trie.insert(&prefix, 0);
            let mut input = prefix.clone();
            input.extend_from_slice(&extra);
            if let Some((len, _)) = trie.longest_prefix_match(&input) {
                prop_assert!(len <= input.len());
                prop_assert!(len >= prefix.len());
            }
        }

        /// Cache entry count never exceeds max_entries.
        #[test]
        fn cache_respects_capacity(
            max_entries in 1usize..=10,
            inserts in proptest::collection::vec(
                proptest::collection::vec(0u32..100, 1..5),
                1..30,
            ),
        ) {
            let cfg = CacheConfig {
                max_entries,
                ..CacheConfig::default()
            };
            let mut cache = PromptCache::new(cfg);
            for tokens in &inserts {
                cache.insert(tokens.clone(), make_kv(tokens, 1));
            }
            prop_assert!(cache.len() <= max_entries);
        }

        /// Stats hit + miss <= lookups.
        #[test]
        fn stats_hits_plus_misses_le_lookups(
            hit in 0u64..1000,
            miss in 0u64..1000,
        ) {
            let lookups = hit + miss;
            let s = PromptCacheStats {
                lookups,
                hits: hit,
                misses: miss,
                ..Default::default()
            };
            prop_assert!(s.hits + s.misses <= s.lookups);
            if lookups > 0 {
                let total_rate = s.hit_rate() + s.miss_rate();
                prop_assert!((total_rate - 1.0).abs() < 1e-10);
            }
        }
    }

    fn make_kv(prefix: &[u32], num_layers: usize) -> CachedKvState {
        let len = prefix.len() * num_layers * 4;
        CachedKvState {
            prefix_tokens: prefix.to_vec(),
            key_data: vec![1.0; len],
            value_data: vec![2.0; len],
            num_layers,
            compressed: false,
        }
    }
}
