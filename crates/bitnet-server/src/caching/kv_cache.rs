//! KV cache optimization with memory pooling

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
        let initial_block = MemoryBlock {
            offset: 0,
            size: size_bytes,
            in_use: false,
        };

        Self {
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
                let allocated_block = MemoryBlock {
                    offset: block.offset,
                    size,
                    in_use: true,
                };

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
        
        let free_block = MemoryBlock {
            offset: block.offset,
            size: block.size,
            in_use: false,
        };

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
            
            if !current.in_use && !next.in_use && 
               current.offset + current.size == next.offset {
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
        if self.total_size == 0 {
            0.0
        } else {
            self.used_memory as f64 / self.total_size as f64
        }
    }

    /// Get fragmentation ratio
    pub fn fragmentation(&self) -> f64 {
        let free_blocks = self.available_blocks.iter()
            .filter(|block| !block.in_use)
            .count();
        
        if free_blocks <= 1 {
            0.0
        } else {
            (free_blocks - 1) as f64 / free_blocks as f64
        }
    }
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
    pub async fn get_or_create_cache(&self, session_id: &str, context_length: usize) -> Result<Option<KVCacheEntry>> {
        let mut stats = self.statistics.write().await;
        
        // Check if cache exists
        {
            let cache = self.cache.read().await;
            if let Some(entry) = cache.get(session_id) {
                stats.cache_hits += 1;
                stats.hit_rate = stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64;
                return Ok(Some(entry.clone()));
            }
        }

        // Cache miss - create new cache
        stats.cache_misses += 1;
        stats.hit_rate = stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64;

        self.create_cache_entry(session_id, context_length).await
    }

    /// Create a new cache entry
    async fn create_cache_entry(&self, session_id: &str, context_length: usize) -> Result<Option<KVCacheEntry>> {
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
                // Retry creating the entry
                self.create_cache_entry(session_id, context_length).await
            } else {
                Ok(None)
            }
        }
    }

    /// Update cache with new tokens
    pub async fn update_cache(&self, session_id: &str, key_data: &[f32], value_data: &[f32]) -> Result<()> {
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
                    let memory_block = MemoryBlock {
                        offset: 0,
                        size: entry.size_bytes,
                        in_use: true,
                    };
                    
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
                    let total_tokens: usize = cache_read.values().map(|entry| entry.token_count).sum();
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