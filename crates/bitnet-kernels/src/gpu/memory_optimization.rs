//! GPU memory optimization and monitoring

use bitnet_common::{KernelError, Result};
use cudarc::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Maximum pool size in bytes
    pub max_pool_size: usize,
    /// Minimum allocation size (for alignment)
    pub min_allocation_size: usize,
    /// Maximum number of cached allocations per size
    pub max_cached_per_size: usize,
    /// Enable memory leak detection
    pub enable_leak_detection: bool,
    /// Memory usage warning threshold (percentage)
    pub warning_threshold: f32,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 2 * 1024 * 1024 * 1024, // 2GB
            min_allocation_size: 256,
            max_cached_per_size: 16,
            enable_leak_detection: true,
            warning_threshold: 0.8, // 80%
        }
    }
}

/// Memory allocation statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub total_freed: usize,
    pub current_usage: usize,
    pub peak_usage: usize,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub fragmentation_ratio: f32,
}

/// Memory allocation tracking for leak detection
#[derive(Debug, Clone)]
struct AllocationInfo {
    size: usize,
    timestamp: Instant,
    #[cfg(debug_assertions)]
    stack_trace: String,
}

/// Advanced GPU memory pool with optimization features
pub struct OptimizedMemoryPool {
    device: Arc<CudaDevice>,
    config: MemoryPoolConfig,
    free_buffers: HashMap<usize, Vec<CudaSlice<u8>>>,
    allocated_buffers: HashMap<*const u8, AllocationInfo>,
    stats: MemoryStats,
    last_cleanup: Instant,
}

impl OptimizedMemoryPool {
    /// Create a new optimized memory pool
    pub fn new(device: Arc<CudaDevice>, config: MemoryPoolConfig) -> Self {
        log::info!("Creating optimized GPU memory pool with max size: {} MB", 
                  config.max_pool_size / (1024 * 1024));

        Self {
            device,
            config,
            free_buffers: HashMap::new(),
            allocated_buffers: HashMap::new(),
            stats: MemoryStats::default(),
            last_cleanup: Instant::now(),
        }
    }

    /// Allocate memory with optimization
    pub fn allocate(&mut self, size: usize) -> Result<CudaSlice<u8>> {
        // Round up to minimum allocation size for better reuse
        let aligned_size = self.align_size(size);
        
        // Try to reuse existing buffer
        if let Some(buffer) = self.try_reuse_buffer(aligned_size) {
            self.stats.cache_hits += 1;
            self.track_allocation(&buffer, size)?;
            return Ok(buffer);
        }

        self.stats.cache_misses += 1;

        // Check if we need to cleanup before allocating
        if self.should_cleanup() {
            self.cleanup_expired_buffers();
        }

        // Check memory limits
        if self.stats.current_usage + aligned_size > self.config.max_pool_size {
            self.force_cleanup();
            
            if self.stats.current_usage + aligned_size > self.config.max_pool_size {
                return Err(KernelError::GpuError { 
                    reason: format!("Memory pool exhausted: requested {}, available {}", 
                                  aligned_size, self.config.max_pool_size - self.stats.current_usage) 
                }.into());
            }
        }

        // Allocate new buffer
        log::debug!("Allocating new GPU buffer of size {}", aligned_size);
        let buffer = self.device.alloc_zeros::<u8>(aligned_size)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to allocate GPU memory: {}", e) 
            })?;

        self.track_allocation(&buffer, size)?;
        Ok(buffer)
    }

    /// Deallocate memory back to pool
    pub fn deallocate(&mut self, buffer: CudaSlice<u8>) {
        let ptr = buffer.as_ptr();
        let size = buffer.len();

        // Remove from tracking
        if let Some(info) = self.allocated_buffers.remove(&ptr) {
            self.stats.current_usage -= info.size;
            self.stats.deallocation_count += 1;
            self.stats.total_freed += info.size;
        }

        // Add to free buffers if we have space
        let free_list = self.free_buffers.entry(size).or_default();
        if free_list.len() < self.config.max_cached_per_size {
            free_list.push(buffer);
            log::debug!("Cached GPU buffer of size {} (total cached: {})", size, free_list.len());
        } else {
            log::debug!("Dropping GPU buffer of size {} (cache full)", size);
            // Buffer will be dropped and freed automatically
        }
    }

    /// Try to reuse an existing buffer
    fn try_reuse_buffer(&mut self, size: usize) -> Option<CudaSlice<u8>> {
        // Try exact size match first
        if let Some(buffers) = self.free_buffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                log::debug!("Reusing exact size GPU buffer: {}", size);
                return Some(buffer);
            }
        }

        // Try to find a larger buffer (within reasonable limits)
        let max_size = size * 2; // Don't waste too much memory
        for (&buf_size, buffers) in self.free_buffers.iter_mut() {
            if buf_size > size && buf_size <= max_size && !buffers.is_empty() {
                let buffer = buffers.pop().unwrap();
                log::debug!("Reusing larger GPU buffer: {} for request {}", buf_size, size);
                return Some(buffer);
            }
        }

        None
    }

    /// Track allocation for leak detection and statistics
    fn track_allocation(&mut self, buffer: &CudaSlice<u8>, size: usize) -> Result<()> {
        let ptr = buffer.as_ptr();
        let info = AllocationInfo {
            size,
            timestamp: Instant::now(),
            #[cfg(debug_assertions)]
            stack_trace: self.capture_stack_trace(),
        };

        self.allocated_buffers.insert(ptr, info);
        self.stats.current_usage += size;
        self.stats.peak_usage = self.stats.peak_usage.max(self.stats.current_usage);
        self.stats.allocation_count += 1;
        self.stats.total_allocated += size;

        // Check for memory usage warnings
        let usage_ratio = self.stats.current_usage as f32 / self.config.max_pool_size as f32;
        if usage_ratio > self.config.warning_threshold {
            log::warn!("High GPU memory usage: {:.1}% ({} MB / {} MB)", 
                      usage_ratio * 100.0,
                      self.stats.current_usage / (1024 * 1024),
                      self.config.max_pool_size / (1024 * 1024));
        }

        Ok(())
    }

    /// Align size to minimum allocation size
    fn align_size(&self, size: usize) -> usize {
        let min_size = self.config.min_allocation_size;
        ((size + min_size - 1) / min_size) * min_size
    }

    /// Check if cleanup is needed
    fn should_cleanup(&self) -> bool {
        self.last_cleanup.elapsed() > Duration::from_secs(30) || // Periodic cleanup
        self.stats.current_usage as f32 / self.config.max_pool_size as f32 > 0.7 // High usage
    }

    /// Cleanup expired buffers
    fn cleanup_expired_buffers(&mut self) {
        let expire_time = Duration::from_secs(300); // 5 minutes
        let now = Instant::now();
        
        let mut total_freed = 0;
        for buffers in self.free_buffers.values_mut() {
            let original_len = buffers.len();
            buffers.retain(|_| {
                // For simplicity, we'll remove half of the cached buffers
                // In a real implementation, we'd track buffer timestamps
                buffers.len() <= original_len / 2
            });
            total_freed += original_len - buffers.len();
        }

        if total_freed > 0 {
            log::debug!("Cleaned up {} expired GPU buffers", total_freed);
        }

        self.last_cleanup = now;
    }

    /// Force cleanup to free memory
    fn force_cleanup(&mut self) {
        log::warn!("Force cleaning GPU memory pool");
        
        // Remove all cached buffers
        let total_cached: usize = self.free_buffers.values().map(|v| v.len()).sum();
        self.free_buffers.clear();
        
        log::info!("Force cleanup freed {} cached GPU buffers", total_cached);
    }

    /// Get current memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let mut stats = self.stats.clone();
        
        // Calculate fragmentation ratio
        let total_cached: usize = self.free_buffers.values().map(|v| v.len()).sum();
        stats.fragmentation_ratio = if stats.allocation_count > 0 {
            total_cached as f32 / stats.allocation_count as f32
        } else {
            0.0
        };

        stats
    }

    /// Print memory statistics
    pub fn print_stats(&self) {
        let stats = self.get_stats();
        
        println!("\n=== GPU Memory Pool Statistics ===");
        println!("Current Usage: {} MB / {} MB ({:.1}%)", 
                 stats.current_usage / (1024 * 1024),
                 self.config.max_pool_size / (1024 * 1024),
                 stats.current_usage as f32 / self.config.max_pool_size as f32 * 100.0);
        println!("Peak Usage: {} MB", stats.peak_usage / (1024 * 1024));
        println!("Total Allocated: {} MB", stats.total_allocated / (1024 * 1024));
        println!("Total Freed: {} MB", stats.total_freed / (1024 * 1024));
        println!("Allocations: {}", stats.allocation_count);
        println!("Deallocations: {}", stats.deallocation_count);
        println!("Cache Hit Rate: {:.1}%", 
                 if stats.cache_hits + stats.cache_misses > 0 {
                     stats.cache_hits as f32 / (stats.cache_hits + stats.cache_misses) as f32 * 100.0
                 } else {
                     0.0
                 });
        println!("Fragmentation Ratio: {:.3}", stats.fragmentation_ratio);
        
        // Show cached buffers by size
        println!("\nCached Buffers by Size:");
        for (&size, buffers) in &self.free_buffers {
            if !buffers.is_empty() {
                println!("  {} bytes: {} buffers", size, buffers.len());
            }
        }
    }

    /// Detect memory leaks
    pub fn detect_leaks(&self) -> Vec<(usize, Duration)> {
        if !self.config.enable_leak_detection {
            return Vec::new();
        }

        let now = Instant::now();
        let leak_threshold = Duration::from_secs(600); // 10 minutes
        
        let mut leaks = Vec::new();
        for (_, info) in &self.allocated_buffers {
            let age = now.duration_since(info.timestamp);
            if age > leak_threshold {
                leaks.push((info.size, age));
            }
        }

        if !leaks.is_empty() {
            log::warn!("Detected {} potential memory leaks", leaks.len());
            for (size, age) in &leaks {
                log::warn!("  Leak: {} bytes, age: {:.1}s", size, age.as_secs_f32());
            }
        }

        leaks
    }

    /// Capture stack trace for debugging (debug builds only)
    #[cfg(debug_assertions)]
    fn capture_stack_trace(&self) -> String {
        // In a real implementation, we'd use a backtrace crate
        format!("Stack trace not implemented")
    }

    #[cfg(not(debug_assertions))]
    fn capture_stack_trace(&self) -> String {
        String::new()
    }
}

/// Memory optimization utilities
pub struct MemoryOptimizer;

impl MemoryOptimizer {
    /// Optimize memory layout for better cache performance
    pub fn optimize_layout<T>(data: &mut [T], access_pattern: AccessPattern) {
        match access_pattern {
            AccessPattern::Sequential => {
                // Data is already optimally laid out for sequential access
            }
            AccessPattern::Random => {
                // For random access, we might want to reorganize data
                // This is a placeholder for more sophisticated optimization
            }
            AccessPattern::Blocked => {
                // Optimize for blocked access patterns (e.g., matrix tiles)
                // This would involve data reorganization
            }
        }
    }

    /// Calculate optimal batch size based on memory constraints
    pub fn calculate_optimal_batch_size(
        element_size: usize,
        available_memory: usize,
        overhead_factor: f32,
    ) -> usize {
        let usable_memory = (available_memory as f32 * (1.0 - overhead_factor)) as usize;
        let max_elements = usable_memory / element_size;
        
        // Round down to a power of 2 for better memory alignment
        let mut batch_size = 1;
        while batch_size * 2 <= max_elements {
            batch_size *= 2;
        }
        
        batch_size.max(1)
    }

    /// Prefetch data for better memory performance
    pub fn prefetch_data<T>(data: &[T], prefetch_distance: usize) {
        // This would use platform-specific prefetch instructions
        // For now, it's a no-op placeholder
        let _ = (data, prefetch_distance);
    }
}

/// Memory access patterns for optimization
#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    Sequential,
    Random,
    Blocked,
}

/// Batch processing optimization
pub struct BatchProcessor {
    optimal_batch_size: usize,
    memory_pool: Arc<Mutex<OptimizedMemoryPool>>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(memory_pool: Arc<Mutex<OptimizedMemoryPool>>, element_size: usize) -> Self {
        let pool = memory_pool.lock().unwrap();
        let stats = pool.get_stats();
        let available_memory = pool.config.max_pool_size - stats.current_usage;
        drop(pool);

        let optimal_batch_size = MemoryOptimizer::calculate_optimal_batch_size(
            element_size,
            available_memory,
            0.2, // 20% overhead
        );

        log::info!("Optimal batch size calculated: {}", optimal_batch_size);

        Self {
            optimal_batch_size,
            memory_pool,
        }
    }

    /// Process data in optimal batches
    pub fn process_batches<T, F>(&self, data: &[T], mut processor: F) -> Result<()>
    where
        F: FnMut(&[T]) -> Result<()>,
    {
        for chunk in data.chunks(self.optimal_batch_size) {
            processor(chunk)?;
        }
        Ok(())
    }

    /// Get optimal batch size
    pub fn optimal_batch_size(&self) -> usize {
        self.optimal_batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_optimizer() {
        let batch_size = MemoryOptimizer::calculate_optimal_batch_size(
            4, // 4 bytes per element
            1024 * 1024, // 1MB available
            0.2, // 20% overhead
        );
        
        assert!(batch_size > 0);
        assert!(batch_size <= 1024 * 1024 / 4);
        
        // Should be a power of 2
        assert_eq!(batch_size & (batch_size - 1), 0);
    }

    #[test]
    fn test_memory_pool_config() {
        let config = MemoryPoolConfig::default();
        assert!(config.max_pool_size > 0);
        assert!(config.min_allocation_size > 0);
        assert!(config.warning_threshold > 0.0 && config.warning_threshold < 1.0);
    }
}