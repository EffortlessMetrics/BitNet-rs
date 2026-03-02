//! Inference result cache.
//!
//! Caches prompt prefixes and their computed states for faster repeated inference.

use std::collections::HashMap;

/// Cache key derived from prompt prefix.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CacheKey {
    pub token_ids: Vec<u32>,
}

impl CacheKey {
    pub fn new(token_ids: Vec<u32>) -> Self {
        Self { token_ids }
    }
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// Create key from a prefix of the tokens.
    pub fn prefix(&self, n: usize) -> Self {
        Self { token_ids: self.token_ids[..n.min(self.token_ids.len())].to_vec() }
    }
}

/// Cached computation state.
#[derive(Debug, Clone)]
pub struct CachedState {
    pub key: CacheKey,
    pub logits: Vec<f32>,
    pub layer_states: usize,
    pub hits: u64,
    pub byte_size: usize,
}

/// Eviction policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    Lru,
    Lfu,
    Fifo,
}

/// Inference cache.
#[derive(Debug)]
pub struct InferenceCache {
    entries: HashMap<Vec<u32>, CachedState>,
    order: Vec<Vec<u32>>, // insertion/access order for eviction
    max_entries: usize,
    max_bytes: usize,
    current_bytes: usize,
    policy: EvictionPolicy,
    total_hits: u64,
    total_misses: u64,
}

impl Default for InferenceCache {
    fn default() -> Self {
        Self::new(256, 512 * 1024 * 1024, EvictionPolicy::Lru)
    }
}

impl InferenceCache {
    pub fn new(max_entries: usize, max_bytes: usize, policy: EvictionPolicy) -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::new(),
            max_entries,
            max_bytes,
            current_bytes: 0,
            policy,
            total_hits: 0,
            total_misses: 0,
        }
    }

    pub fn get(&mut self, key: &CacheKey) -> Option<&CachedState> {
        if self.entries.contains_key(&key.token_ids) {
            self.total_hits += 1;
            if let Some(entry) = self.entries.get_mut(&key.token_ids) {
                entry.hits += 1;
            }
            // Move to end for LRU
            if self.policy == EvictionPolicy::Lru {
                self.order.retain(|k| k != &key.token_ids);
                self.order.push(key.token_ids.clone());
            }
            self.entries.get(&key.token_ids)
        } else {
            self.total_misses += 1;
            None
        }
    }

    pub fn insert(&mut self, state: CachedState) {
        let byte_size = state.byte_size;
        let key = state.key.token_ids.clone();

        // Evict if needed
        while (self.entries.len() >= self.max_entries
            || self.current_bytes + byte_size > self.max_bytes)
            && !self.order.is_empty()
        {
            self.evict_one();
        }

        self.current_bytes += byte_size;
        self.order.push(key.clone());
        self.entries.insert(key, state);
    }

    fn evict_one(&mut self) {
        let victim = match self.policy {
            EvictionPolicy::Lru | EvictionPolicy::Fifo => self.order.first().cloned(),
            EvictionPolicy::Lfu => {
                self.entries.iter().min_by_key(|(_, v)| v.hits).map(|(k, _)| k.clone())
            }
        };

        if let Some(key) = victim {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_bytes = self.current_bytes.saturating_sub(entry.byte_size);
            }
            self.order.retain(|k| k != &key);
        }
    }

    pub fn contains(&self, key: &CacheKey) -> bool {
        self.entries.contains_key(&key.token_ids)
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }
    pub fn byte_usage(&self) -> usize {
        self.current_bytes
    }
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 { 0.0 } else { self.total_hits as f64 / total as f64 }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.current_bytes = 0;
    }

    /// Find the longest cached prefix for a token sequence.
    pub fn longest_prefix(&self, tokens: &[u32]) -> Option<usize> {
        let mut best = None;
        for len in (1..=tokens.len()).rev() {
            let prefix = &tokens[..len];
            if self.entries.contains_key(prefix) {
                best = Some(len);
                break;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(tokens: Vec<u32>, bytes: usize) -> CachedState {
        CachedState {
            key: CacheKey::new(tokens),
            logits: vec![0.0; 10],
            layer_states: 1,
            hits: 0,
            byte_size: bytes,
        }
    }

    #[test]
    fn test_new_cache() {
        let cache = InferenceCache::default();
        assert_eq!(cache.count(), 0);
    }

    #[test]
    fn test_insert_get() {
        let mut cache = InferenceCache::new(10, 1_000_000, EvictionPolicy::Lru);
        cache.insert(make_state(vec![1, 2, 3], 100));
        let key = CacheKey::new(vec![1, 2, 3]);
        assert!(cache.get(&key).is_some());
    }

    #[test]
    fn test_miss() {
        let mut cache = InferenceCache::new(10, 1_000_000, EvictionPolicy::Lru);
        let key = CacheKey::new(vec![1, 2, 3]);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_eviction_by_count() {
        let mut cache = InferenceCache::new(2, 1_000_000, EvictionPolicy::Fifo);
        cache.insert(make_state(vec![1], 10));
        cache.insert(make_state(vec![2], 10));
        cache.insert(make_state(vec![3], 10));
        assert_eq!(cache.count(), 2);
    }

    #[test]
    fn test_eviction_by_bytes() {
        let mut cache = InferenceCache::new(100, 200, EvictionPolicy::Fifo);
        cache.insert(make_state(vec![1], 100));
        cache.insert(make_state(vec![2], 100));
        cache.insert(make_state(vec![3], 100));
        assert!(cache.byte_usage() <= 200);
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = InferenceCache::new(10, 1_000_000, EvictionPolicy::Lru);
        cache.insert(make_state(vec![1], 10));
        let key = CacheKey::new(vec![1]);
        cache.get(&key); // hit
        let miss_key = CacheKey::new(vec![2]);
        cache.get(&miss_key); // miss
        assert!((cache.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_clear() {
        let mut cache = InferenceCache::new(10, 1_000_000, EvictionPolicy::Lru);
        cache.insert(make_state(vec![1], 100));
        cache.clear();
        assert_eq!(cache.count(), 0);
        assert_eq!(cache.byte_usage(), 0);
    }

    #[test]
    fn test_longest_prefix() {
        let mut cache = InferenceCache::new(10, 1_000_000, EvictionPolicy::Lru);
        cache.insert(make_state(vec![1, 2], 10));
        cache.insert(make_state(vec![1, 2, 3], 10));
        assert_eq!(cache.longest_prefix(&[1, 2, 3, 4]), Some(3));
    }

    #[test]
    fn test_cache_key_prefix() {
        let key = CacheKey::new(vec![1, 2, 3, 4]);
        let prefix = key.prefix(2);
        assert_eq!(prefix.len(), 2);
    }

    #[test]
    fn test_lfu_eviction() {
        let mut cache = InferenceCache::new(2, 1_000_000, EvictionPolicy::Lfu);
        cache.insert(make_state(vec![1], 10));
        cache.insert(make_state(vec![2], 10));
        // Hit [1] to increase its frequency
        let key1 = CacheKey::new(vec![1]);
        cache.get(&key1);
        cache.get(&key1);
        // Insert [3], should evict [2] (lowest frequency)
        cache.insert(make_state(vec![3], 10));
        assert!(cache.contains(&CacheKey::new(vec![1])));
    }

    #[test]
    fn test_contains() {
        let mut cache = InferenceCache::new(10, 1_000_000, EvictionPolicy::Lru);
        cache.insert(make_state(vec![1, 2], 10));
        assert!(cache.contains(&CacheKey::new(vec![1, 2])));
        assert!(!cache.contains(&CacheKey::new(vec![3, 4])));
    }
}
