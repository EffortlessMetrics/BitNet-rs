//! # KV Cache Implementation
//!
//! Efficient key-value cache for transformer models with memory pooling,
//! compression, and eviction policies.

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use tracing::{debug, warn};

/// Configuration for KV cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,
    /// Maximum sequence length to cache
    pub max_sequence_length: usize,
    /// Enable cache compression for older entries
    pub enable_compression: bool,
    /// Eviction policy when cache is full
    pub eviction_policy: EvictionPolicy,
    /// Block size for memory allocation
    pub block_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 1024 * 1024 * 1024, // 1GB
            max_sequence_length: 2048,
            enable_compression: false,
            eviction_policy: EvictionPolicy::LRU,
            block_size: 64,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// First In, First Out
    FIFO,
    /// Least Frequently Used
    LFU,
}

/// Key-Value cache entry
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CacheEntry {
    /// Key tensor data
    key: Vec<f32>,
    /// Value tensor data
    value: Vec<f32>,
    /// Sequence position
    position: usize,
    /// Last access timestamp
    last_accessed: std::time::Instant,
    /// Access count for LFU
    access_count: usize,
    /// Whether entry is compressed
    compressed: bool,
}

impl CacheEntry {
    fn new(key: Vec<f32>, value: Vec<f32>, position: usize) -> Self {
        Self {
            key,
            value,
            position,
            last_accessed: std::time::Instant::now(),
            access_count: 1,
            compressed: false,
        }
    }

    fn size_bytes(&self) -> usize {
        (self.key.len() + self.value.len()) * std::mem::size_of::<f32>()
    }

    fn access(&mut self) {
        self.last_accessed = std::time::Instant::now();
        self.access_count += 1;
    }
}

/// KV Cache implementation
pub struct KVCache {
    config: CacheConfig,
    /// Cache entries organized by layer and sequence position
    cache: HashMap<(usize, usize), CacheEntry>,
    /// Access order for LRU eviction
    access_order: VecDeque<(usize, usize)>,
    /// Current cache size in bytes
    current_size: usize,
    /// Memory pool for efficient allocation
    memory_pool: MemoryPool,
    /// Number of tokens that were prefilled in batch
    tokens_prefilled: usize,
    /// Total number of tokens processed (prefill + incremental)
    tokens_total: usize,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(config: CacheConfig) -> Result<Self> {
        let memory_pool = MemoryPool::new(config.block_size, config.max_size_bytes / 4)?;

        Ok(Self {
            config,
            cache: HashMap::new(),
            access_order: VecDeque::new(),
            current_size: 0,
            memory_pool,
            tokens_prefilled: 0,
            tokens_total: 0,
        })
    }

    /// Store key-value pair in cache
    pub fn store(
        &mut self,
        layer: usize,
        position: usize,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<()> {
        let entry_key = (layer, position);
        let entry_size = (key.len() + value.len()) * std::mem::size_of::<f32>();

        // Check if we need to evict entries
        while self.current_size + entry_size > self.config.max_size_bytes {
            self.evict_entry()?;
        }

        // Create new entry
        let entry = CacheEntry::new(key, value, position);

        // Remove old entry if it exists
        if let Some(old_entry) = self.cache.remove(&entry_key) {
            self.current_size -= old_entry.size_bytes();
            self.access_order.retain(|&x| x != entry_key);
        }

        // Add new entry
        self.current_size += entry.size_bytes();
        self.cache.insert(entry_key, entry);
        self.access_order.push_back(entry_key);

        debug!("Stored cache entry for layer {} position {}", layer, position);
        Ok(())
    }

    /// Retrieve key-value pair from cache
    pub fn get(&mut self, layer: usize, position: usize) -> Option<(&Vec<f32>, &Vec<f32>)> {
        let entry_key = (layer, position);

        if let Some(entry) = self.cache.get_mut(&entry_key) {
            entry.access();

            // Update access order for LRU
            if matches!(self.config.eviction_policy, EvictionPolicy::LRU) {
                self.access_order.retain(|&x| x != entry_key);
                self.access_order.push_back(entry_key);
            }

            debug!("Cache hit for layer {} position {}", layer, position);
            Some((&entry.key, &entry.value))
        } else {
            debug!("Cache miss for layer {} position {}", layer, position);
            None
        }
    }

    /// Check if cache contains entry
    pub fn contains(&self, layer: usize, position: usize) -> bool {
        self.cache.contains_key(&(layer, position))
    }

    /// Clear all cache entries
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
        self.current_size = 0;
        self.memory_pool.reset();
        self.tokens_prefilled = 0;
        self.tokens_total = 0;
        debug!("Cache cleared");
    }

    /// Clear cache entries for specific layer
    pub fn clear_layer(&mut self, layer: usize) {
        let keys_to_remove: Vec<_> =
            self.cache.keys().filter(|(l, _)| *l == layer).cloned().collect();

        for key in keys_to_remove {
            if let Some(entry) = self.cache.remove(&key) {
                self.current_size -= entry.size_bytes();
                self.access_order.retain(|&x| x != key);
            }
        }

        debug!("Cleared cache for layer {}", layer);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total_entries = self.cache.len();
        let compressed_entries = self.cache.values().filter(|entry| entry.compressed).count();

        CacheStats {
            total_entries,
            compressed_entries,
            current_size_bytes: self.current_size,
            max_size_bytes: self.config.max_size_bytes,
            hit_rate: 0.0, // Would need to track hits/misses
            memory_efficiency: self.current_size as f64 / self.config.max_size_bytes as f64,
            cache_size: self.current_size, // Alias for compatibility
        }
    }

    /// Get current cache size
    pub fn size(&self) -> usize {
        self.current_size
    }

    /// Get cache usage percentage
    pub fn usage_percent(&self) -> f64 {
        let percent = (self.current_size as f64 / self.config.max_size_bytes as f64) * 100.0;
        percent.clamp(0.0, 100.0)
    }

    /// Get number of tokens that were prefilled in the initial batch
    pub fn num_tokens_prefilled(&self) -> usize {
        self.tokens_prefilled
    }

    /// Get total number of tokens processed (prefill + incremental)
    pub fn num_tokens_total(&self) -> usize {
        self.tokens_total
    }

    /// Record that a prefill operation processed N tokens
    pub fn record_prefill(&mut self, token_count: usize) {
        self.tokens_prefilled = token_count;
        self.tokens_total = token_count;
        debug!("Cache: recorded prefill of {} tokens", token_count);
    }

    /// Record that an incremental step processed N tokens (usually 1)
    pub fn record_incremental(&mut self, token_count: usize) {
        self.tokens_total += token_count;
        debug!("Cache: recorded incremental {} tokens (total: {})", token_count, self.tokens_total);
    }

    /// Evict an entry based on the configured policy
    fn evict_entry(&mut self) -> Result<()> {
        let entry_to_evict = match self.config.eviction_policy {
            EvictionPolicy::LRU => self.access_order.front().cloned(),
            EvictionPolicy::FIFO => self.access_order.front().cloned(),
            EvictionPolicy::LFU => {
                self.cache.iter().min_by_key(|(_, entry)| entry.access_count).map(|(key, _)| *key)
            }
        };

        if let Some(key) = entry_to_evict {
            if let Some(entry) = self.cache.remove(&key) {
                self.current_size -= entry.size_bytes();
                self.access_order.retain(|&x| x != key);
                debug!("Evicted cache entry {:?}", key);
            }
        } else {
            warn!("No entries to evict from cache");
        }

        Ok(())
    }

    /// Compress old cache entries to save memory
    pub fn compress_old_entries(&mut self, age_threshold: std::time::Duration) -> Result<()> {
        if !self.config.enable_compression {
            return Ok(());
        }

        let now = std::time::Instant::now();
        let mut compressed_count = 0;

        for entry in self.cache.values_mut() {
            if !entry.compressed && now.duration_since(entry.last_accessed) > age_threshold {
                // Simple compression: reduce precision (this is a mock implementation)
                // In practice, you'd use a proper compression algorithm
                entry.compressed = true;
                compressed_count += 1;
            }
        }

        if compressed_count > 0 {
            debug!("Compressed {} cache entries", compressed_count);
        }

        Ok(())
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub compressed_entries: usize,
    pub current_size_bytes: usize,
    pub max_size_bytes: usize,
    pub hit_rate: f64,
    pub memory_efficiency: f64,
    pub cache_size: usize, // Alias for current_size_bytes for compatibility
}

/// Memory pool for efficient allocation
#[allow(dead_code)]
struct MemoryPool {
    block_size: usize,
    blocks: Vec<Vec<f32>>,
    free_blocks: Vec<usize>,
}

#[allow(dead_code)]
impl MemoryPool {
    fn new(block_size: usize, max_size: usize) -> Result<Self> {
        let num_blocks = max_size / (block_size * std::mem::size_of::<f32>());
        let blocks = Vec::with_capacity(num_blocks);
        let free_blocks = (0..num_blocks).collect();

        Ok(Self { block_size, blocks, free_blocks })
    }

    fn allocate(&mut self) -> Option<usize> {
        self.free_blocks.pop()
    }

    fn deallocate(&mut self, block_id: usize) {
        if block_id < self.blocks.len() {
            self.free_blocks.push(block_id);
        }
    }

    fn reset(&mut self) {
        self.free_blocks = (0..self.blocks.capacity()).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let config = CacheConfig::default();
        let cache = KVCache::new(config);
        assert!(cache.is_ok());
    }

    #[test]
    fn test_cache_store_and_get() {
        let config = CacheConfig::default();
        let mut cache = KVCache::new(config).unwrap();

        let key = vec![1.0, 2.0, 3.0];
        let value = vec![4.0, 5.0, 6.0];

        cache.store(0, 0, key.clone(), value.clone()).unwrap();

        let retrieved = cache.get(0, 0);
        assert!(retrieved.is_some());

        let (ret_key, ret_value) = retrieved.unwrap();
        assert_eq!(*ret_key, key);
        assert_eq!(*ret_value, value);
    }

    #[test]
    fn test_cache_miss() {
        let config = CacheConfig::default();
        let mut cache = KVCache::new(config).unwrap();

        let retrieved = cache.get(0, 0);
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_cache_clear() {
        let config = CacheConfig::default();
        let mut cache = KVCache::new(config).unwrap();

        let key = vec![1.0, 2.0, 3.0];
        let value = vec![4.0, 5.0, 6.0];

        cache.store(0, 0, key, value).unwrap();
        assert!(cache.contains(0, 0));

        cache.clear();
        assert!(!cache.contains(0, 0));
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_layer_clear() {
        let config = CacheConfig::default();
        let mut cache = KVCache::new(config).unwrap();

        let key = vec![1.0, 2.0, 3.0];
        let value = vec![4.0, 5.0, 6.0];

        cache.store(0, 0, key.clone(), value.clone()).unwrap();
        cache.store(1, 0, key, value).unwrap();

        assert!(cache.contains(0, 0));
        assert!(cache.contains(1, 0));

        cache.clear_layer(0);

        assert!(!cache.contains(0, 0));
        assert!(cache.contains(1, 0));
    }

    #[test]
    fn test_cache_stats() {
        let config = CacheConfig::default();
        let mut cache = KVCache::new(config).unwrap();

        let key = vec![1.0, 2.0, 3.0];
        let value = vec![4.0, 5.0, 6.0];

        cache.store(0, 0, key, value).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 1);
        assert!(stats.current_size_bytes > 0);
    }

    #[test]
    fn test_eviction_policy() {
        let config = CacheConfig {
            max_size_bytes: 100, // Very small to force eviction
            eviction_policy: EvictionPolicy::LRU,
            ..Default::default()
        };

        let mut cache = KVCache::new(config).unwrap();

        // Add entries that exceed cache size
        let key1 = vec![1.0; 10];
        let value1 = vec![1.0; 10];
        let key2 = vec![2.0; 10];
        let value2 = vec![2.0; 10];

        cache.store(0, 0, key1, value1).unwrap();
        cache.store(0, 1, key2, value2).unwrap();

        // First entry should be evicted due to size constraints
        assert!(!cache.contains(0, 0) || !cache.contains(0, 1));
    }
}
