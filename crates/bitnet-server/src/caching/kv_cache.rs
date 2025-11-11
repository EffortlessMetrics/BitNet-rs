//! KV cache optimization with memory pooling
#![cfg_attr(doc, allow(dead_code, unused_imports, unused_variables))]

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::CachingConfig;

/// KV cache entry
#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    /// Session identifier
    pub session_id: String,
    /// Key cache data
    pub key_cache: Vec<f32>,
    /// Value cache data
    pub value_cache: Vec<f32>,
    /// Cache size in bytes
    pub size_bytes: usize,
    /// Last access time
    pub last_accessed: Instant,
    /// Creation time
    pub created_at: Instant,
    /// Number of tokens cached
    pub token_count: usize,
    /// Maximum context length
    pub max_context_length: usize,
}

/// Memory pool for KV cache allocation
#[derive(Debug)]
pub struct MemoryPool {
    /// Backing memory buffer (the actual arena)
    memory: Vec<u8>,
    /// Available memory blocks
    available_blocks: Vec<MemoryBlock>,
    /// Total pool size in bytes
    total_size: usize,
    /// Used memory in bytes
    used_memory: usize,
}

/// Memory block in the pool
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Block offset in the pool
    pub offset: usize,
    /// Block size in bytes
    pub size: usize,
    /// Whether the block is in use
    pub in_use: bool,
}

/// KV cache manager with memory pooling
pub struct KVCacheManager {
    config: CachingConfig,
    cache: Arc<RwLock<HashMap<String, KVCacheEntry>>>,
    memory_pool: Arc<RwLock<MemoryPool>>,
    statistics: Arc<RwLock<KVCacheStatistics>>,
}

/// KV cache statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct KVCacheStatistics {
    pub total_sessions: usize,
    pub total_memory_mb: f64,
    pub used_memory_mb: f64,
    pub memory_utilization: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_rate: f64,
    pub average_session_length: f64,
    pub memory_pool_efficiency: f64,
    pub evictions: u64,
}

impl Default for KVCacheStatistics {
    fn default() -> Self {
        Self {
            total_sessions: 0,
            total_memory_mb: 0.0,
            used_memory_mb: 0.0,
            memory_utilization: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            hit_rate: 0.0,
            average_session_length: 0.0,
            memory_pool_efficiency: 0.0,
            evictions: 0,
        }
    }
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(size_bytes: usize) -> Self {
        let initial_block = MemoryBlock { offset: 0, size: size_bytes, in_use: false };

        Self {
            memory: vec![0u8; size_bytes],
            available_blocks: vec![initial_block],
            total_size: size_bytes,
            used_memory: 0,
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(&mut self, size: usize) -> Option<MemoryBlock> {
        // Find a suitable block
        for (i, block) in self.available_blocks.iter().enumerate() {
            if !block.in_use && block.size >= size {
                let allocated_block = MemoryBlock { offset: block.offset, size, in_use: true };

                // If the block is larger than needed, split it
                if block.size > size {
                    let remaining_block = MemoryBlock {
                        offset: block.offset + size,
                        size: block.size - size,
                        in_use: false,
                    };
                    self.available_blocks[i] = remaining_block;
                } else {
                    // Use the entire block
                    self.available_blocks.remove(i);
                }

                self.used_memory += size;
                return Some(allocated_block);
            }
        }

        None
    }

    /// Deallocate memory back to the pool
    pub fn deallocate(&mut self, block: MemoryBlock) {
        self.used_memory = self.used_memory.saturating_sub(block.size);

        let free_block = MemoryBlock { offset: block.offset, size: block.size, in_use: false };

        // Insert the block back and try to merge with adjacent blocks
        self.available_blocks.push(free_block);
        self.merge_adjacent_blocks();
    }

    /// Merge adjacent free blocks
    fn merge_adjacent_blocks(&mut self) {
        self.available_blocks.sort_by_key(|block| block.offset);

        let mut i = 0;
        while i < self.available_blocks.len().saturating_sub(1) {
            let current = &self.available_blocks[i];
            let next = &self.available_blocks[i + 1];

            if !current.in_use && !next.in_use && current.offset + current.size == next.offset {
                // Merge the blocks
                let merged_block = MemoryBlock {
                    offset: current.offset,
                    size: current.size + next.size,
                    in_use: false,
                };

                self.available_blocks[i] = merged_block;
                self.available_blocks.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    /// Get memory utilization
    pub fn utilization(&self) -> f64 {
        if self.total_size == 0 { 0.0 } else { self.used_memory as f64 / self.total_size as f64 }
    }

    /// Get fragmentation ratio
    pub fn fragmentation(&self) -> f64 {
        let free_blocks = self.available_blocks.iter().filter(|block| !block.in_use).count();

        if free_blocks <= 1 { 0.0 } else { (free_blocks - 1) as f64 / free_blocks as f64 }
    }

    /// Zero-initialize a range of memory (no panics in release).
    pub fn zero_range(&mut self, offset: usize, len: usize) -> anyhow::Result<()> {
        if offset.checked_add(len).map_or(false, |end| end <= self.memory.len()) {
            self.memory[offset..offset + len].fill(0);
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "zero_range out of bounds: offset={} len={} total={}",
                offset,
                len,
                self.memory.len()
            ))
        }
    }

    /// Get a raw pointer to the memory at the given offset
    ///
    /// # Safety
    /// Caller must ensure that the offset is valid and the returned pointer
    /// is used with appropriate bounds checking.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn as_ptr_at(&self, offset: usize) -> *const u8 {
        debug_assert!(offset <= self.memory.len(), "offset {} > len {}", offset, self.memory.len());
        unsafe { self.memory.as_ptr().add(offset) }
    }

    /// Get a mutable raw pointer to the memory at the given offset
    ///
    /// # Safety
    /// Caller must ensure that the offset is valid and the returned pointer
    /// is used with appropriate bounds checking.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn as_mut_ptr_at(&mut self, offset: usize) -> *mut u8 {
        debug_assert!(offset <= self.memory.len(), "offset {} > len {}", offset, self.memory.len());
        unsafe { self.memory.as_mut_ptr().add(offset) }
    }
}

#[cfg(any(test, doc, rustdoc))]
const F32_BYTES: usize = core::mem::size_of::<f32>();

/// Align a size up to the nearest multiple of `align`.
#[cfg(any(test, doc, rustdoc))]
#[inline]
fn align_up(size: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two(), "align must be power of two");
    (size + align - 1) & !(align - 1)
}

impl KVCacheManager {
    /// Create a new KV cache manager
    pub fn new(config: &CachingConfig) -> Result<Self> {
        let pool_size_bytes = config.kv_cache_size_mb * 1024 * 1024;
        let memory_pool = MemoryPool::new(pool_size_bytes);

        Ok(Self {
            config: config.clone(),
            cache: Arc::new(RwLock::new(HashMap::new())),
            memory_pool: Arc::new(RwLock::new(memory_pool)),
            statistics: Arc::new(RwLock::new(KVCacheStatistics::default())),
        })
    }

    /// Get or create a KV cache for a session
    pub async fn get_or_create_cache(
        &self,
        session_id: &str,
        context_length: usize,
    ) -> Result<Option<KVCacheEntry>> {
        let mut stats = self.statistics.write().await;

        // Check if cache exists
        {
            let cache = self.cache.read().await;
            if let Some(entry) = cache.get(session_id) {
                stats.cache_hits += 1;
                stats.hit_rate =
                    stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64;
                return Ok(Some(entry.clone()));
            }
        }

        // Cache miss - create new cache
        stats.cache_misses += 1;
        stats.hit_rate = stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64;

        self.create_cache_entry(session_id, context_length).await
    }

    /// Create a new cache entry
    async fn create_cache_entry(
        &self,
        session_id: &str,
        context_length: usize,
    ) -> Result<Option<KVCacheEntry>> {
        // Calculate required memory (simplified calculation)
        let key_size = context_length * 64 * 4; // 64 dimensions, 4 bytes per float
        let value_size = context_length * 64 * 4;
        let total_size = key_size + value_size;

        // Try to allocate memory from the pool
        let memory_block = {
            let mut pool = self.memory_pool.write().await;
            pool.allocate(total_size)
        };

        if let Some(_block) = memory_block {
            // Create the cache entry
            let entry = KVCacheEntry {
                session_id: session_id.to_string(),
                key_cache: vec![0.0; key_size / 4],
                value_cache: vec![0.0; value_size / 4],
                size_bytes: total_size,
                last_accessed: Instant::now(),
                created_at: Instant::now(),
                token_count: 0,
                max_context_length: context_length,
            };

            // Insert into cache
            {
                let mut cache = self.cache.write().await;
                cache.insert(session_id.to_string(), entry.clone());
            }

            // Update statistics
            {
                let mut stats = self.statistics.write().await;
                stats.total_sessions += 1;
                let pool = self.memory_pool.read().await;
                stats.used_memory_mb = pool.used_memory as f64 / (1024.0 * 1024.0);
                stats.total_memory_mb = pool.total_size as f64 / (1024.0 * 1024.0);
                stats.memory_utilization = pool.utilization();
                stats.memory_pool_efficiency = 1.0 - pool.fragmentation();
            }

            Ok(Some(entry))
        } else {
            // No memory available - try to evict some entries
            self.evict_lru_entries(total_size).await?;

            // Try allocation again
            let memory_block = {
                let mut pool = self.memory_pool.write().await;
                pool.allocate(total_size)
            };

            if memory_block.is_some() {
                // Retry creating the entry with Box::pin to avoid recursion
                Box::pin(self.create_cache_entry(session_id, context_length)).await
            } else {
                Ok(None)
            }
        }
    }

    /// Update cache with new tokens
    pub async fn update_cache(
        &self,
        session_id: &str,
        _key_data: &[f32],
        _value_data: &[f32],
    ) -> Result<()> {
        let mut cache = self.cache.write().await;

        if let Some(entry) = cache.get_mut(session_id) {
            // Update the cache data (simplified - in reality would append new tokens)
            entry.last_accessed = Instant::now();
            entry.token_count += 1;

            // In a real implementation, we would append the new key/value data
            // For now, we'll just simulate the update
        }

        Ok(())
    }

    /// Evict least recently used entries to free memory
    async fn evict_lru_entries(&self, required_size: usize) -> Result<()> {
        let mut entries_to_evict = Vec::new();

        // Find LRU entries
        {
            let cache = self.cache.read().await;
            let mut entries: Vec<_> = cache.values().collect();
            entries.sort_by_key(|entry| entry.last_accessed);

            let mut freed_size = 0;
            for entry in entries {
                entries_to_evict.push(entry.session_id.clone());
                freed_size += entry.size_bytes;

                if freed_size >= required_size {
                    break;
                }
            }
        }

        // Evict the entries
        for session_id in entries_to_evict {
            self.remove_cache_entry(&session_id).await?;
        }

        Ok(())
    }

    /// Remove a cache entry
    async fn remove_cache_entry(&self, session_id: &str) -> Result<()> {
        let entry = {
            let mut cache = self.cache.write().await;
            cache.remove(session_id)
        };

        if let Some(entry) = entry {
            // Return memory to the pool
            let memory_block = MemoryBlock {
                offset: 0, // In a real implementation, we'd track the actual offset
                size: entry.size_bytes,
                in_use: true,
            };

            {
                let mut pool = self.memory_pool.write().await;
                pool.deallocate(memory_block);
            }

            // Update statistics
            {
                let mut stats = self.statistics.write().await;
                stats.total_sessions = stats.total_sessions.saturating_sub(1);
                stats.evictions += 1;

                let pool = self.memory_pool.read().await;
                stats.used_memory_mb = pool.used_memory as f64 / (1024.0 * 1024.0);
                stats.memory_utilization = pool.utilization();
                stats.memory_pool_efficiency = 1.0 - pool.fragmentation();
            }
        }

        Ok(())
    }

    /// Start optimization task
    pub async fn start_optimization_task(&self) {
        let cache = self.cache.clone();
        let memory_pool = self.memory_pool.clone();
        let statistics = self.statistics.clone();

        let mut interval = tokio::time::interval(Duration::from_secs(60)); // Optimize every minute

        loop {
            interval.tick().await;

            // Clean up expired sessions
            let now = Instant::now();
            let expired_sessions: Vec<String> = {
                let cache_read = cache.read().await;
                cache_read.iter()
                    .filter(|(_, entry)| now.duration_since(entry.last_accessed) > Duration::from_secs(3600)) // 1 hour timeout
                    .map(|(session_id, _)| session_id.clone())
                    .collect()
            };

            for session_id in expired_sessions {
                if let Some(entry) = cache.write().await.remove(&session_id) {
                    // Return memory to pool
                    let memory_block =
                        MemoryBlock { offset: 0, size: entry.size_bytes, in_use: true };

                    memory_pool.write().await.deallocate(memory_block);

                    // Update statistics
                    let mut stats = statistics.write().await;
                    stats.total_sessions = stats.total_sessions.saturating_sub(1);
                    stats.evictions += 1;
                }
            }

            // Update statistics
            {
                let mut stats = statistics.write().await;
                let pool = memory_pool.read().await;
                stats.used_memory_mb = pool.used_memory as f64 / (1024.0 * 1024.0);
                stats.memory_utilization = pool.utilization();
                stats.memory_pool_efficiency = 1.0 - pool.fragmentation();

                // Calculate average session length
                let cache_read = cache.read().await;
                if !cache_read.is_empty() {
                    let total_tokens: usize =
                        cache_read.values().map(|entry| entry.token_count).sum();
                    stats.average_session_length = total_tokens as f64 / cache_read.len() as f64;
                }
            }
        }
    }

    /// Get cache statistics
    pub async fn get_statistics(&self) -> KVCacheStatistics {
        self.statistics.read().await.clone()
    }

    /// Shutdown the KV cache manager
    pub async fn shutdown(&self) -> Result<()> {
        println!("Shutting down KV cache manager");

        // Clear all cache entries
        {
            let mut cache = self.cache.write().await;
            cache.clear();
        }

        // Reset memory pool
        {
            let mut pool = self.memory_pool.write().await;
            *pool = MemoryPool::new(pool.total_size);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Alignment constant for cache-line alignment (64 bytes)
    const CACHE_LINE_ALIGN: usize = 64;

    #[test]
    fn test_pool_creation() {
        let pool = MemoryPool::new(1024 * 1024);
        assert_eq!(pool.total_size, 1024 * 1024);
        assert_eq!(pool.used_memory, 0);
        assert_eq!(pool.memory.len(), 1024 * 1024);
        assert_eq!(pool.utilization(), 0.0);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(63, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(align_up(128, 64), 128);
        assert_eq!(align_up(200, 256), 256);
    }

    #[test]
    fn test_zero_range() {
        let mut pool = MemoryPool::new(1024);

        // Fill with non-zero data
        for i in 0..1024 {
            pool.memory[i] = (i % 256) as u8;
        }

        // Zero a range
        pool.zero_range(100, 200).expect("in-bounds");

        // Check zeros in range
        for i in 100..300 {
            assert_eq!(pool.memory[i], 0, "Index {} should be zero", i);
        }

        // Check non-zeros outside range
        assert_ne!(pool.memory[99], 0);
        assert_ne!(pool.memory[300], 0);
    }

    #[test]
    fn test_zero_range_out_of_bounds() {
        let mut pool = MemoryPool::new(1024);
        let err = pool.zero_range(1000, 100).expect_err("should be OOB");
        assert!(err.to_string().contains("zero_range out of bounds"));
    }

    #[test]
    fn test_allocate_basic() {
        let mut pool = MemoryPool::new(1024);

        let block = pool.allocate(256).expect("Should allocate");
        assert_eq!(block.offset, 0);
        assert_eq!(block.size, 256);
        assert!(block.in_use);
        assert_eq!(pool.used_memory, 256);
        assert_eq!(pool.utilization(), 256.0 / 1024.0);
    }

    #[test]
    fn test_allocate_multiple() {
        let mut pool = MemoryPool::new(1024);

        let block1 = pool.allocate(256).expect("Should allocate block1");
        assert_eq!(block1.offset, 0);
        assert_eq!(block1.size, 256);

        let block2 = pool.allocate(256).expect("Should allocate block2");
        assert_eq!(block2.offset, 256);
        assert_eq!(block2.size, 256);

        let block3 = pool.allocate(256).expect("Should allocate block3");
        assert_eq!(block3.offset, 512);
        assert_eq!(block3.size, 256);

        assert_eq!(pool.used_memory, 768);
    }

    #[test]
    fn test_allocate_split_block() {
        let mut pool = MemoryPool::new(1024);

        // Allocate less than total, should split
        let block = pool.allocate(256).expect("Should allocate");
        assert_eq!(block.size, 256);

        // Check that remaining space is tracked
        assert_eq!(pool.available_blocks.len(), 1);
        assert_eq!(pool.available_blocks[0].offset, 256);
        assert_eq!(pool.available_blocks[0].size, 768);
        assert!(!pool.available_blocks[0].in_use);
    }

    #[test]
    fn test_allocate_exact_fit() {
        let mut pool = MemoryPool::new(1024);

        let block = pool.allocate(1024).expect("Should allocate entire pool");
        assert_eq!(block.size, 1024);
        assert_eq!(pool.used_memory, 1024);
        assert_eq!(pool.utilization(), 1.0);

        // No free blocks should remain
        assert_eq!(pool.available_blocks.len(), 0);
    }

    #[test]
    fn test_allocate_too_large() {
        let mut pool = MemoryPool::new(1024);

        let block = pool.allocate(2048);
        assert!(block.is_none(), "Should fail to allocate more than available");
    }

    #[test]
    fn test_deallocate_basic() {
        let mut pool = MemoryPool::new(1024);

        let block = pool.allocate(256).expect("Should allocate");
        assert_eq!(pool.used_memory, 256);

        pool.deallocate(block);
        assert_eq!(pool.used_memory, 0);
    }

    #[test]
    fn test_deallocate_merge_adjacent() {
        let mut pool = MemoryPool::new(1024);

        // Allocate three blocks
        let block1 = pool.allocate(256).expect("Should allocate block1");
        let block2 = pool.allocate(256).expect("Should allocate block2");
        let block3 = pool.allocate(256).expect("Should allocate block3");

        assert_eq!(pool.used_memory, 768);

        // Deallocate middle block
        pool.deallocate(block2.clone());

        // Deallocate first block - should merge with middle
        pool.deallocate(block1.clone());

        // Check that blocks merged
        let free_blocks: Vec<_> = pool.available_blocks.iter().filter(|b| !b.in_use).collect();

        // Should have merged into one contiguous block
        assert!(free_blocks.iter().any(|b| b.offset == 0 && b.size == 512));

        // Deallocate third block - should merge all
        pool.deallocate(block3.clone());

        // Should end with one large free block
        assert_eq!(pool.available_blocks.len(), 1);
        assert_eq!(pool.available_blocks[0].offset, 0);
        assert_eq!(pool.available_blocks[0].size, 1024);
        assert!(!pool.available_blocks[0].in_use);
        assert_eq!(pool.used_memory, 0);
    }

    #[test]
    fn test_allocate_deallocate_reuse() {
        let mut pool = MemoryPool::new(1024);

        // Allocate and deallocate
        let block1 = pool.allocate(256).expect("Should allocate");
        pool.deallocate(block1);

        // Allocate again - should reuse the freed space
        let block2 = pool.allocate(256).expect("Should reuse freed space");
        assert_eq!(block2.offset, 0);
        assert_eq!(block2.size, 256);
    }

    #[test]
    fn test_fragmentation_none() {
        let pool = MemoryPool::new(1024);
        assert_eq!(pool.fragmentation(), 0.0);
    }

    #[test]
    fn test_fragmentation_with_gaps() {
        let mut pool = MemoryPool::new(1024);

        // Allocate multiple blocks
        let block1 = pool.allocate(256).expect("Should allocate");
        let block2 = pool.allocate(256).expect("Should allocate");
        let _block3 = pool.allocate(256).expect("Should allocate");

        // Deallocate non-adjacent blocks to create fragmentation
        pool.deallocate(block1);
        pool.deallocate(block2);

        // Should have some fragmentation
        let frag = pool.fragmentation();
        assert!(frag > 0.0, "Should have fragmentation");
    }

    #[test]
    fn test_many_small_then_large() {
        let mut pool = MemoryPool::new(1024);

        // Allocate many small blocks
        let mut blocks = Vec::new();
        for _ in 0..10 {
            if let Some(block) = pool.allocate(64) {
                blocks.push(block);
            }
        }

        assert_eq!(blocks.len(), 10);
        assert_eq!(pool.used_memory, 640);

        // Deallocate all
        for block in blocks {
            pool.deallocate(block);
        }

        assert_eq!(pool.used_memory, 0);

        // Should be able to allocate one large block
        let large = pool.allocate(1024).expect("Should allocate after defrag");
        assert_eq!(large.size, 1024);
    }

    #[test]
    fn test_aligned_allocation() {
        let mut pool = MemoryPool::new(1024);

        // Allocate with cache-line alignment
        let size = 100;
        let aligned_size = align_up(size, CACHE_LINE_ALIGN);

        let block = pool.allocate(aligned_size).expect("Should allocate");
        assert_eq!(block.size, aligned_size);
        assert_eq!(aligned_size, 128); // 100 aligned up to 64-byte boundary
    }

    #[test]
    fn test_typed_view_round_trip() {
        let mut pool = MemoryPool::new(1024);

        // Allocate space for f32 values
        let num_floats = 10;
        let size_bytes = num_floats * std::mem::size_of::<f32>();
        let aligned_size = align_up(size_bytes, CACHE_LINE_ALIGN);

        let block = pool.allocate(aligned_size).expect("Should allocate");

        // Write sentinel values as f32
        {
            let ptr = pool.as_mut_ptr_at(block.offset) as *mut f32;
            unsafe {
                for i in 0..num_floats {
                    *ptr.add(i) = (i as f32) * 1.5;
                }
            }
        }

        // Read back as raw bytes and verify
        {
            let ptr = pool.as_ptr_at(block.offset) as *const f32;
            unsafe {
                for i in 0..num_floats {
                    let value = *ptr.add(i);
                    assert_eq!(value, (i as f32) * 1.5);
                }
            }
        }
    }

    #[test]
    fn test_utilization() {
        let mut pool = MemoryPool::new(1000);

        assert_eq!(pool.utilization(), 0.0);

        pool.allocate(250).expect("Should allocate");
        assert_eq!(pool.utilization(), 0.25);

        pool.allocate(250).expect("Should allocate");
        assert_eq!(pool.utilization(), 0.5);

        pool.allocate(500).expect("Should allocate");
        assert_eq!(pool.utilization(), 1.0);
    }

    #[test]
    fn test_zero_after_allocation() {
        let mut pool = MemoryPool::new(1024);

        // Allocate a block
        let block = pool.allocate(256).expect("Should allocate");

        // Zero-initialize it
        pool.zero_range(block.offset, block.size).expect("in-bounds");

        // Verify all zeros
        for i in block.offset..block.offset + block.size {
            assert_eq!(pool.memory[i], 0);
        }
    }
}
