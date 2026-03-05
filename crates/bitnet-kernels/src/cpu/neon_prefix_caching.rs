//! ARM NEON optimized prefix caching for Apple Silicon.
//!
//! Provides NEON-accelerated prefix caching primitives:
//! - Hash computation for token ID sequences
//! - Longest prefix match detection
//! - SIMD bulk KV cache copy
//! - Radix tree for O(n) token-sequence matching
//! - LRU eviction with NEON-assisted priority comparison
//! - Multi-tenant namespace isolation
//! - Prefix hit/miss statistics

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1. Prefix hash computation
// ---------------------------------------------------------------------------

/// FNV-1a style seed for mixing token IDs.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Compute a 64-bit hash of a token ID slice using NEON-accelerated
/// accumulation.  Falls back to scalar for tail elements.
#[cfg(target_arch = "aarch64")]
pub fn neon_prefix_hash(tokens: &[u32]) -> u64 {
    if tokens.is_empty() {
        return FNV_OFFSET;
    }

    let mut hash = FNV_OFFSET;
    let chunks = tokens.len() / 4;
    let remainder = tokens.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        // Load 4 × u32 via NEON then fold each lane with position-
        // dependent mixing to preserve ordering.
        unsafe {
            let v = vld1q_u32(tokens.as_ptr().add(base));
            let a = vgetq_lane_u32::<0>(v) as u64;
            let b = vgetq_lane_u32::<1>(v) as u64;
            let c = vgetq_lane_u32::<2>(v) as u64;
            let d = vgetq_lane_u32::<3>(v) as u64;
            hash ^= a;
            hash = hash.wrapping_mul(FNV_PRIME);
            hash ^= b;
            hash = hash.wrapping_mul(FNV_PRIME);
            hash ^= c;
            hash = hash.wrapping_mul(FNV_PRIME);
            hash ^= d;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    // Scalar tail.
    let tail_start = chunks * 4;
    for &t in &tokens[tail_start..tail_start + remainder] {
        hash ^= t as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}

/// Scalar fallback (non-aarch64 builds or testing reference).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_prefix_hash(tokens: &[u32]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &t in tokens {
        hash ^= t as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// 2. Prefix match detection
// ---------------------------------------------------------------------------

/// Find the length of the longest common prefix between `cached` and `new`
/// token ID slices using NEON 4-wide comparison.
#[cfg(target_arch = "aarch64")]
pub fn neon_prefix_match_len(cached: &[u32], new: &[u32]) -> usize {
    let common = cached.len().min(new.len());
    if common == 0 {
        return 0;
    }

    let chunks = common / 4;
    let mut matched: usize = 0;

    for i in 0..chunks {
        let base = i * 4;
        let all_eq = unsafe {
            let a = vld1q_u32(cached.as_ptr().add(base));
            let b = vld1q_u32(new.as_ptr().add(base));
            let cmp = vceqq_u32(a, b);
            // All lanes 0xFFFF_FFFF → min lane is 0xFFFF_FFFF.
            vminvq_u32(cmp)
        };
        if all_eq == 0xFFFF_FFFF {
            matched += 4;
        } else {
            // At least one lane differs — find the first.
            for j in 0..4 {
                if cached[base + j] != new[base + j] {
                    return matched + j;
                }
            }
        }
    }

    // Scalar tail.
    let tail_start = chunks * 4;
    for i in tail_start..common {
        if cached[i] != new[i] {
            return matched + (i - tail_start);
        }
    }

    matched + (common - tail_start)
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_prefix_match_len(cached: &[u32], new: &[u32]) -> usize {
    cached.iter().zip(new.iter()).take_while(|(a, b)| a == b).count()
}

// ---------------------------------------------------------------------------
// 3. KV cache copy (NEON SIMD memcpy for f32 states)
// ---------------------------------------------------------------------------

/// Bulk-copy `count` f32 values from `src` to `dst` using NEON 128-bit
/// loads/stores.  Handles unaligned tails via scalar copy.
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_prefix_copy(src: &[f32], dst: &mut [f32], count: usize) {
    assert!(count <= src.len(), "source too small: count={count} src.len={}", src.len());
    assert!(count <= dst.len(), "destination too small: count={count} dst.len={}", dst.len());

    let chunks = count / 4;
    let remainder = count % 4;

    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            let v = vld1q_f32(src.as_ptr().add(base));
            vst1q_f32(dst.as_mut_ptr().add(base), v);
        }
    }

    let tail_start = chunks * 4;
    dst[tail_start..tail_start + remainder]
        .copy_from_slice(&src[tail_start..tail_start + remainder]);
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_kv_prefix_copy(src: &[f32], dst: &mut [f32], count: usize) {
    dst[..count].copy_from_slice(&src[..count]);
}

// ---------------------------------------------------------------------------
// 4. Radix tree lookup
// ---------------------------------------------------------------------------

/// A node in the prefix radix tree.
#[derive(Debug, Clone)]
struct RadixNode {
    /// Token IDs stored on this edge.
    tokens: Vec<u32>,
    /// Child nodes keyed by their first token.
    children: HashMap<u32, RadixNode>,
    /// True when this node represents a complete cached prefix.
    is_terminal: bool,
    /// Opaque cache slot index (valid when `is_terminal`).
    cache_slot: Option<usize>,
}

impl RadixNode {
    fn new(tokens: Vec<u32>) -> Self {
        Self { tokens, children: HashMap::new(), is_terminal: false, cache_slot: None }
    }
}

/// Radix tree mapping token-ID sequences to cache slots.
#[derive(Debug, Clone)]
pub struct PrefixRadixTree {
    root: RadixNode,
    num_entries: usize,
}

impl Default for PrefixRadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefixRadixTree {
    pub fn new() -> Self {
        Self { root: RadixNode::new(vec![]), num_entries: 0 }
    }

    /// Number of terminal entries.
    pub fn len(&self) -> usize {
        self.num_entries
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.num_entries == 0
    }

    /// Insert a token sequence with an associated cache slot.
    pub fn insert(&mut self, tokens: &[u32], cache_slot: usize) {
        Self::insert_at(&mut self.root, tokens, cache_slot);
        self.num_entries += 1;
    }

    fn insert_at(node: &mut RadixNode, tokens: &[u32], slot: usize) {
        if tokens.is_empty() {
            node.is_terminal = true;
            node.cache_slot = Some(slot);
            return;
        }

        let first = tokens[0];
        if let Some(child) = node.children.get_mut(&first) {
            let common = neon_prefix_match_len(&child.tokens, tokens);
            if common == child.tokens.len() {
                // Full edge match — recurse into child.
                Self::insert_at(child, &tokens[common..], slot);
            } else {
                // Partial match — split the edge.
                let mut new_mid = RadixNode::new(child.tokens[..common].to_vec());
                let mut old_tail = child.clone();
                old_tail.tokens = child.tokens[common..].to_vec();
                let old_key = old_tail.tokens[0];
                new_mid.children.insert(old_key, old_tail);

                if common < tokens.len() {
                    let new_key = tokens[common];
                    let mut new_leaf = RadixNode::new(tokens[common..].to_vec());
                    new_leaf.is_terminal = true;
                    new_leaf.cache_slot = Some(slot);
                    new_mid.children.insert(new_key, new_leaf);
                } else {
                    new_mid.is_terminal = true;
                    new_mid.cache_slot = Some(slot);
                }
                node.children.insert(first, new_mid);
            }
        } else {
            let mut leaf = RadixNode::new(tokens.to_vec());
            leaf.is_terminal = true;
            leaf.cache_slot = Some(slot);
            node.children.insert(first, leaf);
        }
    }

    /// Find the longest matching prefix. Returns `(match_length, cache_slot)`.
    pub fn longest_prefix(&self, tokens: &[u32]) -> Option<(usize, usize)> {
        Self::longest_at(&self.root, tokens, 0)
    }

    fn longest_at(node: &RadixNode, tokens: &[u32], depth: usize) -> Option<(usize, usize)> {
        if tokens.is_empty() {
            return if node.is_terminal {
                Some((depth, node.cache_slot.unwrap_or(0)))
            } else {
                None
            };
        }

        let first = tokens[0];
        if let Some(child) = node.children.get(&first) {
            let common = neon_prefix_match_len(&child.tokens, tokens);
            if common == 0 {
                return if node.is_terminal {
                    Some((depth, node.cache_slot.unwrap_or(0)))
                } else {
                    None
                };
            }
            if common < child.tokens.len() {
                // Partial edge match — no deeper terminal possible.
                return if node.is_terminal {
                    Some((depth, node.cache_slot.unwrap_or(0)))
                } else {
                    None
                };
            }
            // Full edge match — try to go deeper.
            let deeper = Self::longest_at(child, &tokens[common..], depth + common);
            deeper.or(if node.is_terminal {
                Some((depth, node.cache_slot.unwrap_or(0)))
            } else {
                None
            })
        } else if node.is_terminal {
            Some((depth, node.cache_slot.unwrap_or(0)))
        } else {
            None
        }
    }

    /// Remove the entry whose cache slot matches. Returns true on removal.
    pub fn remove_slot(&mut self, slot: usize) -> bool {
        if Self::remove_at(&mut self.root, slot) {
            self.num_entries = self.num_entries.saturating_sub(1);
            true
        } else {
            false
        }
    }

    fn remove_at(node: &mut RadixNode, slot: usize) -> bool {
        if node.is_terminal && node.cache_slot == Some(slot) {
            node.is_terminal = false;
            node.cache_slot = None;
            return true;
        }
        for child in node.children.values_mut() {
            if Self::remove_at(child, slot) {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// 5. Cache eviction — LRU with NEON-assisted priority comparison
// ---------------------------------------------------------------------------

/// Entry in the LRU eviction tracker.
#[derive(Debug, Clone, Copy)]
pub struct EvictionEntry {
    /// Cache slot index.
    pub slot: usize,
    /// Logical clock tick of last access.
    pub last_access: u32,
    /// Prefix length (longer = higher value).
    pub prefix_len: u32,
}

/// Find the index of the entry with the smallest `last_access` (LRU victim)
/// using NEON 4-wide comparison.
#[cfg(target_arch = "aarch64")]
pub fn neon_find_lru_victim(entries: &[EvictionEntry]) -> Option<usize> {
    if entries.is_empty() {
        return None;
    }

    let mut min_val = entries[0].last_access;
    let mut min_idx: usize = 0;

    // Build an array of last_access values for NEON scanning.
    let access: Vec<u32> = entries.iter().map(|e| e.last_access).collect();
    let chunks = access.len() / 4;

    for i in 0..chunks {
        let base = i * 4;
        let lane_min = unsafe {
            let v = vld1q_u32(access.as_ptr().add(base));
            vminvq_u32(v)
        };
        if lane_min < min_val {
            // Identify exact lane.
            for j in 0..4 {
                if access[base + j] < min_val {
                    min_val = access[base + j];
                    min_idx = base + j;
                }
            }
        }
    }

    // Scalar tail.
    let tail_start = chunks * 4;
    for (i, &acc) in access.iter().enumerate().skip(tail_start) {
        if acc < min_val {
            min_val = acc;
            min_idx = i;
        }
    }

    Some(min_idx)
}

#[cfg(not(target_arch = "aarch64"))]
pub fn neon_find_lru_victim(entries: &[EvictionEntry]) -> Option<usize> {
    entries.iter().enumerate().min_by_key(|(_, e)| e.last_access).map(|(i, _)| i)
}

// ---------------------------------------------------------------------------
// 6. Multi-tenant isolation
// ---------------------------------------------------------------------------

/// Namespace-qualified prefix hash providing per-request isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NamespacedHash {
    pub namespace_id: u64,
    pub prefix_hash: u64,
}

/// Compute a namespace-isolated prefix hash.
pub fn namespaced_prefix_hash(namespace_id: u64, tokens: &[u32]) -> NamespacedHash {
    let prefix_hash = neon_prefix_hash(tokens);
    NamespacedHash {
        namespace_id,
        // Mix namespace into the hash so collisions across tenants are
        // virtually impossible.
        prefix_hash: prefix_hash ^ namespace_id.wrapping_mul(FNV_PRIME),
    }
}

/// Multi-tenant prefix cache mapping `NamespacedHash → cache_slot`.
#[derive(Debug, Default)]
pub struct MultiTenantPrefixCache {
    map: HashMap<NamespacedHash, usize>,
}

impl MultiTenantPrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: NamespacedHash, slot: usize) {
        self.map.insert(key, slot);
    }

    pub fn get(&self, key: &NamespacedHash) -> Option<usize> {
        self.map.get(key).copied()
    }

    pub fn remove(&mut self, key: &NamespacedHash) -> bool {
        self.map.remove(key).is_some()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// ---------------------------------------------------------------------------
// 7. Prefix statistics
// ---------------------------------------------------------------------------

/// Tracks prefix caching hit/miss statistics.
#[derive(Debug, Clone, Default)]
pub struct PrefixCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub total_prefix_tokens_saved: u64,
    pub total_requests: u64,
    pub evictions: u64,
}

impl PrefixCacheStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_hit(&mut self, prefix_len: usize) {
        self.hits += 1;
        self.total_requests += 1;
        self.total_prefix_tokens_saved += prefix_len as u64;
    }

    pub fn record_miss(&mut self) {
        self.misses += 1;
        self.total_requests += 1;
    }

    pub fn record_eviction(&mut self) {
        self.evictions += 1;
    }

    /// Hit rate in [0.0, 1.0].  Returns 0.0 when no requests.
    pub fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.hits as f64 / self.total_requests as f64
    }

    /// Average prefix length across hits.  Returns 0.0 when no hits.
    pub fn avg_prefix_length(&self) -> f64 {
        if self.hits == 0 {
            return 0.0;
        }
        self.total_prefix_tokens_saved as f64 / self.hits as f64
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- prefix hash tests -----

    #[test]
    fn test_prefix_hash_empty() {
        assert_eq!(neon_prefix_hash(&[]), FNV_OFFSET);
    }

    #[test]
    fn test_prefix_hash_deterministic() {
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let h1 = neon_prefix_hash(&tokens);
        let h2 = neon_prefix_hash(&tokens);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_prefix_hash_differs_for_different_inputs() {
        let a = neon_prefix_hash(&[1, 2, 3, 4]);
        let b = neon_prefix_hash(&[4, 3, 2, 1]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_prefix_hash_single_token() {
        let h = neon_prefix_hash(&[42]);
        assert_ne!(h, FNV_OFFSET);
    }

    // ----- prefix match detection -----

    #[test]
    fn test_prefix_match_identical() {
        let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(neon_prefix_match_len(&tokens, &tokens), 8);
    }

    #[test]
    fn test_prefix_match_partial() {
        let cached = [1u32, 2, 3, 4, 5, 6];
        let new = [1, 2, 3, 99, 100, 101];
        assert_eq!(neon_prefix_match_len(&cached, &new), 3);
    }

    #[test]
    fn test_prefix_match_empty() {
        let empty: &[u32] = &[];
        assert_eq!(neon_prefix_match_len(empty, &[1, 2]), 0);
        assert_eq!(neon_prefix_match_len(&[1, 2], empty), 0);
    }

    #[test]
    fn test_prefix_match_no_common() {
        assert_eq!(neon_prefix_match_len(&[10], &[20]), 0);
    }

    // ----- KV cache copy -----

    #[test]
    fn test_kv_prefix_copy_basic() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let mut dst = vec![0.0f32; 16];
        neon_kv_prefix_copy(&src, &mut dst, 16);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_kv_prefix_copy_partial() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut dst = vec![0.0f32; 10];
        neon_kv_prefix_copy(&src, &mut dst, 5);
        assert_eq!(&dst[..5], &src[..5]);
        assert_eq!(dst[5], 0.0); // untouched
    }

    #[test]
    fn test_kv_prefix_copy_unaligned_tail() {
        let src: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let mut dst = vec![-1.0f32; 7];
        neon_kv_prefix_copy(&src, &mut dst, 7);
        assert_eq!(src, dst);
    }

    // ----- radix tree -----

    #[test]
    fn test_radix_insert_and_lookup() {
        let mut tree = PrefixRadixTree::new();
        tree.insert(&[1, 2, 3, 4], 0);
        tree.insert(&[1, 2, 5, 6], 1);

        let r = tree.longest_prefix(&[1, 2, 3, 4, 99]);
        assert_eq!(r, Some((4, 0)));

        let r2 = tree.longest_prefix(&[1, 2, 5, 6, 7]);
        assert_eq!(r2, Some((4, 1)));

        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_radix_no_match() {
        let tree = PrefixRadixTree::new();
        assert_eq!(tree.longest_prefix(&[1, 2, 3]), None);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_radix_remove_slot() {
        let mut tree = PrefixRadixTree::new();
        tree.insert(&[10, 20, 30], 5);
        assert!(tree.remove_slot(5));
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.longest_prefix(&[10, 20, 30]), None);
    }

    // ----- eviction -----

    #[test]
    fn test_find_lru_victim_basic() {
        let entries = vec![
            EvictionEntry { slot: 0, last_access: 10, prefix_len: 4 },
            EvictionEntry { slot: 1, last_access: 3, prefix_len: 8 },
            EvictionEntry { slot: 2, last_access: 7, prefix_len: 2 },
            EvictionEntry { slot: 3, last_access: 15, prefix_len: 6 },
            EvictionEntry { slot: 4, last_access: 1, prefix_len: 3 },
        ];
        let victim = neon_find_lru_victim(&entries).unwrap();
        assert_eq!(victim, 4);
        assert_eq!(entries[victim].last_access, 1);
    }

    #[test]
    fn test_find_lru_victim_empty() {
        assert_eq!(neon_find_lru_victim(&[]), None);
    }

    // ----- multi-tenant -----

    #[test]
    fn test_namespace_isolation() {
        let tokens = [1u32, 2, 3];
        let h1 = namespaced_prefix_hash(100, &tokens);
        let h2 = namespaced_prefix_hash(200, &tokens);
        // Same tokens, different namespaces → different hashes.
        assert_ne!(h1, h2);
        assert_ne!(h1.prefix_hash, h2.prefix_hash);
    }

    #[test]
    fn test_multi_tenant_cache() {
        let mut cache = MultiTenantPrefixCache::new();
        let k1 = namespaced_prefix_hash(1, &[10, 20]);
        let k2 = namespaced_prefix_hash(2, &[10, 20]);
        cache.insert(k1, 0);
        cache.insert(k2, 1);
        assert_eq!(cache.get(&k1), Some(0));
        assert_eq!(cache.get(&k2), Some(1));
        assert_eq!(cache.len(), 2);
        assert!(cache.remove(&k1));
        assert_eq!(cache.len(), 1);
    }

    // ----- statistics -----

    #[test]
    fn test_stats_hit_rate() {
        let mut stats = PrefixCacheStats::new();
        stats.record_hit(10);
        stats.record_hit(20);
        stats.record_miss();
        assert!((stats.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
        assert!((stats.avg_prefix_length() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_empty() {
        let stats = PrefixCacheStats::new();
        assert_eq!(stats.hit_rate(), 0.0);
        assert_eq!(stats.avg_prefix_length(), 0.0);
    }

    #[test]
    fn test_stats_eviction_tracking() {
        let mut stats = PrefixCacheStats::new();
        stats.record_eviction();
        stats.record_eviction();
        assert_eq!(stats.evictions, 2);
    }
}
