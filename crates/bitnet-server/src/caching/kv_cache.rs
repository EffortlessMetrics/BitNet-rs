//! KV cache optimization with memory pooling
#![allow(dead_code, unused_imports, unused_variables)]

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::CachingConfig;

#[cfg(all(feature = "receipts", any(test, feature = "tuning")))]
use super::performance_tuning::PerformanceReport;
#[cfg(any(test, feature = "receipts"))]
use super::receipts::{KvEventSink, KvEvictionReport, TracingSink};

/// KV cache entry backed by pool slices (no owned Vec<f32>).
#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    /// Session identifier
    pub session_id: String,
    /// Key cache offset in pool (byte offset)
    key_off: usize,
    /// Key cache length (as f32 elements)
    key_len_f32: usize,
    /// Value cache offset in pool (byte offset)
    val_off: usize,
    /// Value cache length (as f32 elements)
    val_len_f32: usize,
    /// Original allocation block (returned to pool on evict)
    block: MemoryBlock,
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

impl KVCacheEntry {
    /// Get mutable key slice from pool
    #[inline]
    pub fn key_mut<'a>(&self, pool: &'a mut MemoryPool) -> &'a mut [f32] {
        pool.f32_slice_mut(self.key_off, self.key_len_f32)
    }

    /// Get mutable value slice from pool
    #[inline]
    pub fn value_mut<'a>(&self, pool: &'a mut MemoryPool) -> &'a mut [f32] {
        pool.f32_slice_mut(self.val_off, self.val_len_f32)
    }

    /// Get read-only key slice from pool
    #[inline]
    pub fn key<'a>(&self, pool: &'a MemoryPool) -> &'a [f32] {
        pool.f32_slice(self.key_off, self.key_len_f32)
    }

    /// Get read-only value slice from pool
    #[inline]
    pub fn value<'a>(&self, pool: &'a MemoryPool) -> &'a [f32] {
        pool.f32_slice(self.val_off, self.val_len_f32)
    }
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
    #[cfg(feature = "receipts")]
    receipt_sink: Option<Arc<dyn KvEventSink>>,
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
        if offset.checked_add(len).is_some_and(|end| end <= self.memory.len()) {
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

    /// Create a mutable f32 slice view into the arena.
    /// Bounds and alignment are checked; panic on misuse (transition period).
    #[inline]
    pub fn f32_slice_mut(&mut self, offset: usize, len_f32: usize) -> &mut [f32] {
        let bytes = len_f32.checked_mul(core::mem::size_of::<f32>()).expect("f32 len overflow");
        assert!(offset.is_multiple_of(core::mem::align_of::<f32>()), "unaligned f32 slice");
        assert!(
            offset.checked_add(bytes).is_some_and(|end| end <= self.memory.len()),
            "OOB f32 slice"
        );
        unsafe {
            let ptr = self.memory.as_mut_ptr().add(offset) as *mut f32;
            core::slice::from_raw_parts_mut(ptr, len_f32)
        }
    }

    /// Read-only f32 view.
    #[inline]
    pub fn f32_slice(&self, offset: usize, len_f32: usize) -> &[f32] {
        let bytes = len_f32.checked_mul(core::mem::size_of::<f32>()).expect("f32 len overflow");
        assert!(offset.is_multiple_of(core::mem::align_of::<f32>()), "unaligned f32 slice");
        assert!(
            offset.checked_add(bytes).is_some_and(|end| end <= self.memory.len()),
            "OOB f32 slice"
        );
        unsafe {
            let ptr = self.memory.as_ptr().add(offset) as *const f32;
            core::slice::from_raw_parts(ptr, len_f32)
        }
    }
}

const F32_BYTES: usize = core::mem::size_of::<f32>();

/// Align a size up to the nearest multiple of alignment
///
/// # Arguments
/// * `size` - Size to align
/// * `align` - Alignment (must be power of 2)
const fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

impl KVCacheManager {
    /// Create a new KV cache manager
    pub fn new(config: &CachingConfig) -> Result<Self> {
        let pool_size_bytes = config.kv_cache_size_mb * 1024 * 1024;
        let memory_pool = MemoryPool::new(pool_size_bytes);

        #[cfg(feature = "receipts")]
        let receipt_sink: Option<Arc<dyn KvEventSink>> =
            if config.enable_receipts { Some(Arc::new(TracingSink)) } else { None };

        Ok(Self {
            config: config.clone(),
            cache: Arc::new(RwLock::new(HashMap::new())),
            memory_pool: Arc::new(RwLock::new(memory_pool)),
            statistics: Arc::new(RwLock::new(KVCacheStatistics::default())),
            #[cfg(feature = "receipts")]
            receipt_sink,
        })
    }

    /// Test-only constructor with injected receipt sink.
    #[cfg(any(test, feature = "receipts"))]
    #[allow(dead_code)]
    pub(crate) fn with_receipt_sink(
        config: CachingConfig,
        sink: Option<Arc<dyn KvEventSink>>,
    ) -> Result<Self> {
        let pool_size_bytes = config.kv_cache_size_mb * 1024 * 1024;
        let memory_pool = MemoryPool::new(pool_size_bytes);

        Ok(Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            memory_pool: Arc::new(RwLock::new(memory_pool)),
            statistics: Arc::new(RwLock::new(KVCacheStatistics::default())),
            #[cfg(feature = "receipts")]
            receipt_sink: sink,
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
        let total_size = align_up(key_size, 64) + align_up(value_size, 64);

        // Try to allocate memory from the pool
        let memory_block = {
            let mut pool = self.memory_pool.write().await;
            pool.allocate(total_size)
        };

        if let Some(block) = memory_block {
            let key_off = block.offset;
            let val_off = align_up(block.offset + key_size, 64);
            debug_assert!(key_off >= block.offset);
            debug_assert!(val_off >= block.offset);
            debug_assert!(
                val_off + value_size <= block.offset + block.size,
                "KV split exceeds block"
            );

            // Zero-initialize the allocated memory
            {
                let mut pool = self.memory_pool.write().await;
                pool.zero_range(key_off, key_size).context("zero_range failed for key region")?;
                pool.zero_range(val_off, value_size)
                    .context("zero_range failed for value region")?;
            }

            // Create the cache entry
            let entry = KVCacheEntry {
                session_id: session_id.to_string(),
                key_off,
                key_len_f32: key_size / F32_BYTES,
                val_off,
                val_len_f32: value_size / F32_BYTES,
                block: block.clone(),
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

            if let Some(block) = memory_block {
                let key_off = block.offset;
                let val_off = align_up(block.offset + key_size, 64);
                debug_assert!(key_off >= block.offset);
                debug_assert!(val_off >= block.offset);
                debug_assert!(
                    val_off + value_size <= block.offset + block.size,
                    "KV split exceeds block"
                );

                // Zero-initialize the allocated memory
                {
                    let mut pool = self.memory_pool.write().await;
                    pool.zero_range(key_off, key_size)
                        .context("zero_range failed for key region (post-evict)")?;
                    pool.zero_range(val_off, value_size)
                        .context("zero_range failed for value region (post-evict)")?;
                }

                let entry = KVCacheEntry {
                    session_id: session_id.to_string(),
                    key_off,
                    key_len_f32: key_size / F32_BYTES,
                    val_off,
                    val_len_f32: value_size / F32_BYTES,
                    block,
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
                Ok(None)
            }
        }
    }

    /// Update cache with new tokens
    /// TODO(PR-4): Wire append semantics; copy into pool-backed slices.
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

    /// Record a KV eviction event.
    #[cfg(feature = "receipts")]
    fn record_kv_eviction(
        &self,
        session_id: &str,
        block: &MemoryBlock,
        before: &KVCacheStatistics,
        after: &KVCacheStatistics,
        #[cfg(any(test, feature = "tuning"))] perf: Option<PerformanceReport>,
    ) {
        use std::time::SystemTime;

        if let Some(sink) = &self.receipt_sink {
            let event = KvEvictionReport {
                session_id: session_id.to_owned(),
                block_offset: block.offset,
                block_size_bytes: block.size,
                before: before.clone(),
                after: after.clone(),
                #[cfg(any(test, feature = "tuning"))]
                performance: perf,
                timestamp: SystemTime::now(),
            };

            sink.on_eviction(event);
        }
    }

    /// Remove a cache entry
    async fn remove_cache_entry(&self, session_id: &str) -> Result<()> {
        // Capture before stats if receipts enabled
        #[cfg(feature = "receipts")]
        let maybe_before =
            if self.config.enable_receipts { Some(self.get_statistics().await) } else { None };

        let entry = {
            let mut cache = self.cache.write().await;
            cache.remove(session_id)
        };

        if let Some(entry) = entry {
            // Return memory to the pool using the real MemoryBlock captured in entry
            {
                let mut pool = self.memory_pool.write().await;
                pool.deallocate(entry.block.clone());
            }

            // Update statistics
            {
                let mut stats = self.statistics.write().await;
                stats.total_sessions = stats.total_sessions.saturating_sub(1);
                stats.evictions += 1;

                let pool = self.memory_pool.read().await;
                stats.used_memory_mb = pool.used_memory as f64 / (1024.0 * 1024.0);
                stats.total_memory_mb = pool.total_size as f64 / (1024.0 * 1024.0);
                stats.memory_utilization = pool.utilization();
                stats.memory_pool_efficiency = 1.0 - pool.fragmentation();
            }

            // Emit receipt if enabled
            #[cfg(feature = "receipts")]
            if self.config.enable_receipts {
                let after = self.get_statistics().await;

                if let Some(ref before) = maybe_before {
                    #[cfg(any(test, feature = "tuning"))]
                    {
                        let perf = Some(PerformanceReport::from_stats(&after, &self.config));
                        self.record_kv_eviction(session_id, &entry.block, before, &after, perf);
                    }
                    #[cfg(not(any(test, feature = "tuning")))]
                    {
                        self.record_kv_eviction(session_id, &entry.block, before, &after);
                    }
                }
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
        assert_eq!(aligned_size, 128); // align_up(100, 64) = 128 (next 64B boundary)
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

    #[test]
    fn entry_views_round_trip() {
        let mut pool = MemoryPool::new(1024);

        // Mimic create path
        let key_bytes = 10 * core::mem::size_of::<f32>();
        let val_bytes = 12 * core::mem::size_of::<f32>();
        let total = align_up(key_bytes, 64) + align_up(val_bytes, 64);

        let block = pool.allocate(total).expect("allocate");
        let key_off = block.offset;
        let val_off = align_up(block.offset + key_bytes, 64);

        pool.zero_range(key_off, key_bytes).expect("zero_range key");
        pool.zero_range(val_off, val_bytes).expect("zero_range val");

        let entry = KVCacheEntry {
            session_id: "s".into(),
            key_off,
            key_len_f32: key_bytes / 4,
            val_off,
            val_len_f32: val_bytes / 4,
            block,
            size_bytes: total,
            last_accessed: Instant::now(),
            created_at: Instant::now(),
            token_count: 0,
            max_context_length: 0,
        };

        // Write via typed views
        for (i, v) in entry.key_mut(&mut pool).iter_mut().enumerate() {
            *v = (i as f32) + 1.0;
        }
        for (i, v) in entry.value_mut(&mut pool).iter_mut().enumerate() {
            *v = (i as f32) + 2.0;
        }

        // Read back
        assert_eq!(entry.key(&pool)[0], 1.0);
        assert_eq!(entry.value(&pool)[1], 3.0);
    }

    #[test]
    fn test_eviction_returns_block_to_pool() {
        let mut pool = MemoryPool::new(1024);

        // Allocate a block
        let block = pool.allocate(256).expect("Should allocate");
        let start_used = pool.used_memory;
        assert_eq!(start_used, 256);

        // Simulate eviction by deallocating
        pool.deallocate(block.clone());

        // Verify memory was returned
        assert_eq!(pool.used_memory, start_used - block.size);
        assert_eq!(pool.used_memory, 0);

        // Verify fragmentation is valid
        assert!(pool.fragmentation() >= 0.0);
        assert!(pool.fragmentation() <= 1.0);

        // Verify we can reallocate the same block
        let block2 = pool.allocate(256).expect("Should reallocate");
        assert_eq!(block2.offset, 0); // Should reuse the freed region
        assert_eq!(block2.size, 256);
    }

    #[tokio::test]
    async fn test_eviction_updates_stats() -> Result<()> {
        use crate::caching::CachingConfig;

        // Create small cache to force eviction
        let config = CachingConfig {
            kv_cache_size_mb: 1, // 1 MB
            ..Default::default()
        };

        let manager = KVCacheManager::new(&config)?;

        // Create a session (triggers allocation)
        let session_id = "test_session";
        manager.create_cache_entry(session_id, 64).await?;

        // Snapshot stats before eviction
        let before = manager.get_statistics().await;
        assert!(before.used_memory_mb > 0.0, "Should have used memory");
        assert_eq!(before.total_sessions, 1);
        assert_eq!(before.evictions, 0);

        // Evict the entry
        manager.remove_cache_entry(session_id).await?;

        // Stats should reflect freed memory
        let after = manager.get_statistics().await;
        assert_eq!(after.total_sessions, 0, "Should have 0 sessions after removal");
        assert_eq!(after.evictions, 1, "Should have 1 eviction");
        assert!(after.used_memory_mb < before.used_memory_mb, "Used memory should decrease");
        assert!(after.total_memory_mb > 0.0, "Total memory should be set");
        assert!(
            after.memory_utilization >= 0.0 && after.memory_utilization <= 1.0,
            "Utilization should be in [0,1]"
        );
        assert!(
            after.memory_pool_efficiency >= 0.0 && after.memory_pool_efficiency <= 1.0,
            "Efficiency should be in [0,1]"
        );

        Ok(())
    }

    #[cfg(feature = "receipts")]
    #[tokio::test]
    async fn emits_eviction_receipt_with_correct_payload() -> Result<()> {
        use crate::caching::receipts::test_helpers::ChannelSink;

        let (sink, mut rx) = ChannelSink::channel(4);
        let mut cfg = CachingConfig::default();
        cfg.enable_receipts = true;

        let manager = KVCacheManager::with_receipt_sink(cfg, Some(Arc::new(sink)))?;

        // Create a small entry
        let session_id = "sess-evict";
        let entry = manager.create_cache_entry(session_id, 16).await?.expect("entry created");

        let before = manager.get_statistics().await;
        manager.remove_cache_entry(session_id).await?;
        let after = manager.get_statistics().await;

        // There should be exactly one event
        let report = rx.recv().await.expect("expected eviction report");
        assert_eq!(report.session_id, session_id);
        assert_eq!(report.block_offset, entry.block.offset);
        assert_eq!(report.block_size_bytes, entry.block.size);
        assert_eq!(report.before.total_sessions, before.total_sessions);
        assert_eq!(report.after.total_sessions, after.total_sessions);
        // Performance report may or may not be present depending on tuning feature
        // assert!(report.performance.is_none()); // not asserting here

        Ok(())
    }

    #[cfg(feature = "receipts")]
    #[tokio::test]
    async fn does_not_emit_receipt_when_disabled() -> Result<()> {
        use crate::caching::receipts::test_helpers::ChannelSink;
        use tokio::sync::mpsc::error::TryRecvError;

        let (sink, mut rx) = ChannelSink::channel(1);
        let mut cfg = CachingConfig::default();
        cfg.enable_receipts = false; // explicitly off

        let manager = KVCacheManager::with_receipt_sink(cfg, Some(Arc::new(sink)))?;

        let session_id = "sess-no-receipt";
        let _entry = manager.create_cache_entry(session_id, 16).await?.expect("entry");
        manager.remove_cache_entry(session_id).await?;

        // Channel should remain empty
        match rx.try_recv() {
            Err(TryRecvError::Empty) => {}
            other => panic!("expected no event, got {:?}", other),
        }

        Ok(())
    }
}
