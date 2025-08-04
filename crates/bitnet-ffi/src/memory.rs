//! Memory management utilities for the C API
//!
//! This module provides automatic cleanup, leak prevention, and memory
//! monitoring capabilities for the C API.

use crate::BitNetCError;
use std::sync::{Arc, Mutex, RwLock};
use std::collections::HashMap;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Memory statistics tracking
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Total bytes deallocated
    pub total_deallocated: usize,
    /// Current bytes in use
    pub current_usage: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
}

impl MemoryStats {
    /// Get current memory usage in MB
    pub fn current_usage_mb(&self) -> f64 {
        self.current_usage as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory usage in MB
    pub fn peak_usage_mb(&self) -> f64 {
        self.peak_usage as f64 / (1024.0 * 1024.0)
    }

    /// Check if there are memory leaks
    pub fn has_leaks(&self) -> bool {
        self.allocation_count != self.deallocation_count
    }

    /// Get number of leaked allocations
    pub fn leaked_allocations(&self) -> usize {
        if self.allocation_count > self.deallocation_count {
            self.allocation_count - self.deallocation_count
        } else {
            0
        }
    }
}

/// Memory tracking allocator wrapper
pub struct TrackingAllocator {
    inner: System,
    stats: Arc<Mutex<MemoryStats>>,
}

impl TrackingAllocator {
    pub fn new() -> Self {
        Self {
            inner: System,
            stats: Arc::new(Mutex::new(MemoryStats::default())),
        }
    }

    pub fn get_stats(&self) -> Result<MemoryStats, BitNetCError> {
        self.stats.lock()
            .map(|stats| stats.clone())
            .map_err(|_| BitNetCError::ThreadSafety("Failed to acquire memory stats lock".to_string()))
    }

    fn record_allocation(&self, size: usize) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_allocated += size;
            stats.current_usage += size;
            stats.allocation_count += 1;
            
            if stats.current_usage > stats.peak_usage {
                stats.peak_usage = stats.current_usage;
            }
        }
    }

    fn record_deallocation(&self, size: usize) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_deallocated += size;
            stats.current_usage = stats.current_usage.saturating_sub(size);
            stats.deallocation_count += 1;
        }
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        if !ptr.is_null() {
            self.record_allocation(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.record_deallocation(layout.size());
        self.inner.dealloc(ptr, layout);
    }
}

/// Memory pool for efficient allocation of common sizes
pub struct MemoryPool {
    pools: RwLock<HashMap<usize, Vec<*mut u8>>>,
    max_pool_size: usize,
    stats: Arc<Mutex<MemoryStats>>,
}

impl MemoryPool {
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: RwLock::new(HashMap::new()),
            max_pool_size,
            stats: Arc::new(Mutex::new(MemoryStats::default())),
        }
    }

    /// Allocate memory from pool or system
    pub fn allocate(&self, size: usize) -> Result<*mut u8, BitNetCError> {
        // Try to get from pool first
        if let Ok(mut pools) = self.pools.write() {
            if let Some(pool) = pools.get_mut(&size) {
                if let Some(ptr) = pool.pop() {
                    return Ok(ptr);
                }
            }
        }

        // Allocate from system
        let layout = Layout::from_size_align(size, std::mem::align_of::<u8>())
            .map_err(|_| BitNetCError::OutOfMemory("Invalid memory layout".to_string()))?;

        let ptr = unsafe { System.alloc(layout) };
        if ptr.is_null() {
            return Err(BitNetCError::OutOfMemory(format!("Failed to allocate {} bytes", size)));
        }

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_allocated += size;
            stats.current_usage += size;
            stats.allocation_count += 1;
            
            if stats.current_usage > stats.peak_usage {
                stats.peak_usage = stats.current_usage;
            }
        }

        Ok(ptr)
    }

    /// Deallocate memory back to pool or system
    pub fn deallocate(&self, ptr: *mut u8, size: usize) -> Result<(), BitNetCError> {
        if ptr.is_null() {
            return Ok(());
        }

        // Try to return to pool
        if let Ok(mut pools) = self.pools.write() {
            let pool = pools.entry(size).or_insert_with(Vec::new);
            if pool.len() < self.max_pool_size {
                pool.push(ptr);
                return Ok(());
            }
        }

        // Deallocate from system
        let layout = Layout::from_size_align(size, std::mem::align_of::<u8>())
            .map_err(|_| BitNetCError::Internal("Invalid memory layout for deallocation".to_string()))?;

        unsafe { System.dealloc(ptr, layout) };

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_deallocated += size;
            stats.current_usage = stats.current_usage.saturating_sub(size);
            stats.deallocation_count += 1;
        }

        Ok(())
    }

    /// Clear all pools and free memory
    pub fn clear_pools(&self) -> Result<(), BitNetCError> {
        let mut pools = self.pools.write()
            .map_err(|_| BitNetCError::ThreadSafety("Failed to acquire pools write lock".to_string()))?;

        for (size, pool) in pools.iter() {
            let layout = Layout::from_size_align(*size, std::mem::align_of::<u8>())
                .map_err(|_| BitNetCError::Internal("Invalid memory layout for pool clearing".to_string()))?;

            for ptr in pool {
                unsafe { System.dealloc(*ptr, layout) };
            }
        }

        pools.clear();
        Ok(())
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> Result<MemoryStats, BitNetCError> {
        self.stats.lock()
            .map(|stats| stats.clone())
            .map_err(|_| BitNetCError::ThreadSafety("Failed to acquire memory stats lock".to_string()))
    }
}

/// Memory manager for the C API
pub struct MemoryManager {
    pool: MemoryPool,
    tracking_enabled: RwLock<bool>,
    memory_limit: RwLock<Option<usize>>,
}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            pool: MemoryPool::new(1000), // Max 1000 items per pool
            tracking_enabled: RwLock::new(true),
            memory_limit: RwLock::new(None),
        }
    }

    /// Set memory limit in bytes
    pub fn set_memory_limit(&self, limit: Option<usize>) -> Result<(), BitNetCError> {
        let mut memory_limit = self.memory_limit.write()
            .map_err(|_| BitNetCError::ThreadSafety("Failed to acquire memory limit write lock".to_string()))?;
        *memory_limit = limit;
        Ok(())
    }

    /// Get current memory limit
    pub fn get_memory_limit(&self) -> Result<Option<usize>, BitNetCError> {
        let memory_limit = self.memory_limit.read()
            .map_err(|_| BitNetCError::ThreadSafety("Failed to acquire memory limit read lock".to_string()))?;
        Ok(*memory_limit)
    }

    /// Check if allocation would exceed memory limit
    pub fn check_memory_limit(&self, additional_bytes: usize) -> Result<bool, BitNetCError> {
        let memory_limit = self.memory_limit.read()
            .map_err(|_| BitNetCError::ThreadSafety("Failed to acquire memory limit read lock".to_string()))?;

        if let Some(limit) = *memory_limit {
            let stats = self.pool.get_stats()?;
            Ok(stats.current_usage + additional_bytes <= limit)
        } else {
            Ok(true) // No limit set
        }
    }

    /// Allocate memory with limit checking
    pub fn allocate(&self, size: usize) -> Result<*mut u8, BitNetCError> {
        // Check memory limit
        if !self.check_memory_limit(size)? {
            return Err(BitNetCError::OutOfMemory(
                format!("Allocation of {} bytes would exceed memory limit", size)
            ));
        }

        self.pool.allocate(size)
    }

    /// Deallocate memory
    pub fn deallocate(&self, ptr: *mut u8, size: usize) -> Result<(), BitNetCError> {
        self.pool.deallocate(ptr, size)
    }

    /// Enable or disable memory tracking
    pub fn set_tracking_enabled(&self, enabled: bool) -> Result<(), BitNetCError> {
        let mut tracking_enabled = self.tracking_enabled.write()
            .map_err(|_| BitNetCError::ThreadSafety("Failed to acquire tracking enabled write lock".to_string()))?;
        *tracking_enabled = enabled;
        Ok(())
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> Result<MemoryStats, BitNetCError> {
        self.pool.get_stats()
    }

    /// Perform garbage collection
    pub fn garbage_collect(&self) -> Result<(), BitNetCError> {
        self.pool.clear_pools()
    }

    /// Check for memory leaks
    pub fn check_leaks(&self) -> Result<Vec<String>, BitNetCError> {
        let stats = self.get_stats()?;
        let mut leaks = Vec::new();

        if stats.has_leaks() {
            leaks.push(format!(
                "Memory leak detected: {} allocations not freed",
                stats.leaked_allocations()
            ));
        }

        if stats.current_usage > 0 && stats.allocation_count == stats.deallocation_count {
            leaks.push(format!(
                "Potential memory leak: {} bytes still in use after all deallocations",
                stats.current_usage
            ));
        }

        Ok(leaks)
    }
}

// Global memory manager instance
static MEMORY_MANAGER: std::sync::OnceLock<MemoryManager> = std::sync::OnceLock::new();

/// Initialize the memory manager
pub fn initialize_memory_manager() -> Result<(), BitNetCError> {
    MEMORY_MANAGER.get_or_init(|| MemoryManager::new());
    Ok(())
}

/// Get the global memory manager instance
pub fn get_memory_manager() -> &'static MemoryManager {
    MEMORY_MANAGER.get_or_init(|| MemoryManager::new())
}

/// Cleanup the memory manager
pub fn cleanup_memory_manager() -> Result<(), BitNetCError> {
    if let Some(manager) = MEMORY_MANAGER.get() {
        manager.garbage_collect()?;
        
        // Check for leaks
        let leaks = manager.check_leaks()?;
        if !leaks.is_empty() {
            eprintln!("Memory leaks detected during cleanup:");
            for leak in leaks {
                eprintln!("  {}", leak);
            }
        }
    }
    Ok(())
}

/// RAII wrapper for automatic memory cleanup
pub struct AutoCleanup<T> {
    data: Option<T>,
    cleanup_fn: Box<dyn FnOnce(T) + Send>,
}

impl<T> AutoCleanup<T> {
    pub fn new<F>(data: T, cleanup_fn: F) -> Self
    where
        F: FnOnce(T) + Send + 'static,
    {
        Self {
            data: Some(data),
            cleanup_fn: Box::new(cleanup_fn),
        }
    }

    pub fn get(&self) -> Option<&T> {
        self.data.as_ref()
    }

    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.data.as_mut()
    }

    pub fn take(mut self) -> Option<T> {
        self.data.take()
    }
}

impl<T> Drop for AutoCleanup<T> {
    fn drop(&mut self) {
        if let Some(data) = self.data.take() {
            (self.cleanup_fn)(data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::default();
        stats.total_allocated = 1024 * 1024; // 1 MB
        stats.current_usage = 512 * 1024; // 512 KB
        stats.peak_usage = 2 * 1024 * 1024; // 2 MB

        assert_eq!(stats.current_usage_mb(), 0.5);
        assert_eq!(stats.peak_usage_mb(), 2.0);
        assert!(!stats.has_leaks());
    }

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(10);
        
        // Test allocation
        let ptr = pool.allocate(1024).unwrap();
        assert!(!ptr.is_null());
        
        // Test deallocation
        assert!(pool.deallocate(ptr, 1024).is_ok());
        
        // Test stats
        let stats = pool.get_stats().unwrap();
        assert!(stats.allocation_count > 0);
    }

    #[test]
    fn test_memory_manager() {
        let manager = MemoryManager::new();
        
        // Test memory limit
        assert!(manager.set_memory_limit(Some(1024 * 1024)).is_ok());
        assert_eq!(manager.get_memory_limit().unwrap(), Some(1024 * 1024));
        
        // Test allocation within limit
        assert!(manager.check_memory_limit(512).unwrap());
        
        // Test stats
        let stats = manager.get_stats().unwrap();
        assert_eq!(stats.allocation_count, stats.deallocation_count);
    }

    #[test]
    fn test_auto_cleanup() {
        let cleanup_called = Arc::new(Mutex::new(false));
        let cleanup_called_clone = Arc::clone(&cleanup_called);
        
        {
            let _auto_cleanup = AutoCleanup::new(42, move |_value| {
                *cleanup_called_clone.lock().unwrap() = true;
            });
        }
        
        assert!(*cleanup_called.lock().unwrap());
    }

    #[test]
    fn test_leak_detection() {
        let mut stats = MemoryStats::default();
        stats.allocation_count = 5;
        stats.deallocation_count = 3;
        
        assert!(stats.has_leaks());
        assert_eq!(stats.leaked_allocations(), 2);
    }
}