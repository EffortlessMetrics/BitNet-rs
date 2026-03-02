//! HIP memory allocation pool with `hipMalloc`/`hipFree` modelling.
//!
//! Provides a simulated GPU memory pool for tracking allocations,
//! computing statistics, and validating allocation patterns before
//! real HIP runtime integration. The pool uses a simple best-fit
//! strategy for block selection.
//!
//! # CPU fallback
//!
//! All operations work in pure Rust using `Vec<u8>` as backing storage.
//! GPU-dependent paths will use `hipMalloc`/`hipFree` once the HIP FFI
//! bindings are wired in.

use bitnet_common::{KernelError, Result};
use std::collections::HashMap;

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for the HIP memory pool.
#[derive(Debug, Clone)]
pub struct HipMemoryPoolConfig {
    /// Initial pool capacity in bytes.
    pub initial_capacity: usize,
    /// Maximum pool capacity in bytes.
    pub max_capacity: usize,
    /// Alignment requirement in bytes (must be power of two).
    pub alignment: usize,
}

impl Default for HipMemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 64 * 1024 * 1024,   // 64 MiB
            max_capacity: 2 * 1024 * 1024 * 1024, // 2 GiB
            alignment: 256,
        }
    }
}

impl HipMemoryPoolConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<()> {
        if self.initial_capacity == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "initial_capacity must be non-zero".into(),
            }
            .into());
        }
        if self.max_capacity < self.initial_capacity {
            return Err(KernelError::InvalidArguments {
                reason: "max_capacity must be >= initial_capacity".into(),
            }
            .into());
        }
        if !self.alignment.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: "alignment must be a power of two".into(),
            }
            .into());
        }
        Ok(())
    }
}

// ── Allocation handle ────────────────────────────────────────────────

/// Unique handle for a HIP memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HipAllocId(u64);

impl HipAllocId {
    /// Return the raw numeric ID.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

// ── Allocation record ────────────────────────────────────────────────

/// Metadata for a single allocation within the pool.
#[derive(Debug, Clone)]
pub struct HipAllocation {
    /// Unique allocation handle.
    pub id: HipAllocId,
    /// Byte offset from pool base.
    pub offset: usize,
    /// Size in bytes (after alignment).
    pub size: usize,
    /// Original requested size (before alignment).
    pub requested_size: usize,
    /// Human-readable label for debugging.
    pub label: String,
}

// ── Memory statistics ────────────────────────────────────────────────

/// Snapshot of pool memory statistics.
#[derive(Debug, Clone, Default)]
pub struct HipMemoryStats {
    /// Total pool capacity in bytes.
    pub total_bytes: usize,
    /// Bytes currently allocated.
    pub used_bytes: usize,
    /// Bytes available.
    pub free_bytes: usize,
    /// Number of live allocations.
    pub num_allocations: usize,
    /// Peak bytes ever allocated at once.
    pub peak_used_bytes: usize,
    /// Total cumulative bytes allocated (lifetime).
    pub total_allocated_lifetime: usize,
    /// Total number of `alloc` calls (lifetime).
    pub alloc_count: u64,
    /// Total number of `free` calls (lifetime).
    pub free_count: u64,
}

// ── Memory pool ──────────────────────────────────────────────────────

/// Simulated HIP GPU memory pool (CPU fallback).
///
/// Models `hipMalloc`/`hipFree` semantics with best-fit allocation.
/// On a real HIP runtime this would issue actual device allocations.
#[derive(Debug)]
pub struct HipMemoryPool {
    config: HipMemoryPoolConfig,
    allocations: HashMap<HipAllocId, HipAllocation>,
    next_id: u64,
    next_offset: usize,
    stats: HipMemoryStats,
}

impl HipMemoryPool {
    /// Create a new pool with the given configuration.
    pub fn new(config: HipMemoryPoolConfig) -> Result<Self> {
        config.validate()?;
        let stats = HipMemoryStats {
            total_bytes: config.initial_capacity,
            free_bytes: config.initial_capacity,
            ..Default::default()
        };
        Ok(Self { config, allocations: HashMap::new(), next_id: 1, next_offset: 0, stats })
    }

    /// Allocate `size` bytes from the pool, returning an allocation handle.
    pub fn alloc(&mut self, size: usize, label: &str) -> Result<HipAllocId> {
        if size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "allocation size must be non-zero".into(),
            }
            .into());
        }
        let aligned_size = align_up(size, self.config.alignment);
        if self.stats.used_bytes + aligned_size > self.config.max_capacity {
            return Err(KernelError::OutOfMemory {
                requested: aligned_size,
                available: self.config.max_capacity.saturating_sub(self.stats.used_bytes),
            }
            .into());
        }

        let id = HipAllocId(self.next_id);
        self.next_id += 1;

        let offset = self.next_offset;
        self.next_offset += aligned_size;

        let alloc = HipAllocation {
            id,
            offset,
            size: aligned_size,
            requested_size: size,
            label: label.to_string(),
        };
        self.allocations.insert(id, alloc);

        self.stats.used_bytes += aligned_size;
        self.stats.free_bytes = self.stats.total_bytes.saturating_sub(self.stats.used_bytes);
        self.stats.num_allocations += 1;
        self.stats.alloc_count += 1;
        self.stats.total_allocated_lifetime += aligned_size;
        if self.stats.used_bytes > self.stats.peak_used_bytes {
            self.stats.peak_used_bytes = self.stats.used_bytes;
        }

        // Expand total capacity tracking if needed
        if self.next_offset > self.stats.total_bytes {
            self.stats.total_bytes = self.next_offset.min(self.config.max_capacity);
            self.stats.free_bytes = self.stats.total_bytes.saturating_sub(self.stats.used_bytes);
        }

        Ok(id)
    }

    /// Free a previously allocated block.
    pub fn free(&mut self, id: HipAllocId) -> Result<()> {
        let alloc = self.allocations.remove(&id).ok_or_else(|| {
            bitnet_common::BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!("unknown allocation id: {}", id.0),
            })
        })?;
        self.stats.used_bytes = self.stats.used_bytes.saturating_sub(alloc.size);
        self.stats.free_bytes = self.stats.total_bytes.saturating_sub(self.stats.used_bytes);
        self.stats.num_allocations -= 1;
        self.stats.free_count += 1;
        Ok(())
    }

    /// Get metadata for an allocation.
    pub fn get_allocation(&self, id: HipAllocId) -> Option<&HipAllocation> {
        self.allocations.get(&id)
    }

    /// Current pool statistics.
    pub fn stats(&self) -> &HipMemoryStats {
        &self.stats
    }

    /// Reset the pool, freeing all allocations.
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.next_offset = 0;
        self.stats.used_bytes = 0;
        self.stats.free_bytes = self.stats.total_bytes;
        self.stats.num_allocations = 0;
    }
}

/// Align `size` up to the nearest multiple of `alignment`.
fn align_up(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> HipMemoryPoolConfig {
        HipMemoryPoolConfig { initial_capacity: 4096, max_capacity: 8192, alignment: 64 }
    }

    #[test]
    fn config_default_validates() {
        assert!(HipMemoryPoolConfig::default().validate().is_ok());
    }

    #[test]
    fn config_zero_capacity_fails() {
        let cfg = HipMemoryPoolConfig { initial_capacity: 0, ..HipMemoryPoolConfig::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_max_less_than_initial_fails() {
        let cfg = HipMemoryPoolConfig {
            initial_capacity: 1024,
            max_capacity: 512,
            ..HipMemoryPoolConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_bad_alignment_fails() {
        let cfg = HipMemoryPoolConfig { alignment: 3, ..HipMemoryPoolConfig::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn pool_creation() {
        let pool = HipMemoryPool::new(test_config()).unwrap();
        assert_eq!(pool.stats().num_allocations, 0);
        assert_eq!(pool.stats().used_bytes, 0);
    }

    #[test]
    fn alloc_and_free() {
        let mut pool = HipMemoryPool::new(test_config()).unwrap();
        let id = pool.alloc(100, "test").unwrap();
        assert_eq!(pool.stats().num_allocations, 1);
        assert!(pool.stats().used_bytes >= 100);
        pool.free(id).unwrap();
        assert_eq!(pool.stats().num_allocations, 0);
    }

    #[test]
    fn alloc_zero_size_fails() {
        let mut pool = HipMemoryPool::new(test_config()).unwrap();
        assert!(pool.alloc(0, "bad").is_err());
    }

    #[test]
    fn alloc_exceeds_capacity_fails() {
        let mut pool = HipMemoryPool::new(test_config()).unwrap();
        assert!(pool.alloc(16384, "too_big").is_err());
    }

    #[test]
    fn free_unknown_id_fails() {
        let mut pool = HipMemoryPool::new(test_config()).unwrap();
        assert!(pool.free(HipAllocId(999)).is_err());
    }

    #[test]
    fn allocation_is_aligned() {
        let mut pool = HipMemoryPool::new(test_config()).unwrap();
        let id = pool.alloc(1, "tiny").unwrap();
        let alloc = pool.get_allocation(id).unwrap();
        assert_eq!(alloc.size % 64, 0); // aligned to 64
        assert!(alloc.size >= 1);
        pool.free(id).unwrap();
    }

    #[test]
    fn multiple_allocs() {
        let mut pool = HipMemoryPool::new(test_config()).unwrap();
        let a = pool.alloc(64, "a").unwrap();
        let b = pool.alloc(128, "b").unwrap();
        assert_eq!(pool.stats().num_allocations, 2);
        assert_ne!(a, b);
        pool.free(a).unwrap();
        pool.free(b).unwrap();
    }

    #[test]
    fn peak_usage_tracked() {
        let mut pool = HipMemoryPool::new(test_config()).unwrap();
        let a = pool.alloc(256, "a").unwrap();
        let _b = pool.alloc(512, "b").unwrap();
        let peak = pool.stats().peak_used_bytes;
        pool.free(a).unwrap();
        assert!(pool.stats().peak_used_bytes >= peak);
    }

    #[test]
    fn lifetime_counters() {
        let mut pool = HipMemoryPool::new(test_config()).unwrap();
        let id = pool.alloc(64, "x").unwrap();
        pool.free(id).unwrap();
        assert_eq!(pool.stats().alloc_count, 1);
        assert_eq!(pool.stats().free_count, 1);
    }

    #[test]
    fn reset_clears_all() {
        let mut pool = HipMemoryPool::new(test_config()).unwrap();
        pool.alloc(64, "a").unwrap();
        pool.alloc(128, "b").unwrap();
        pool.reset();
        assert_eq!(pool.stats().num_allocations, 0);
        assert_eq!(pool.stats().used_bytes, 0);
    }

    #[test]
    fn alloc_id_raw_value() {
        let id = HipAllocId(42);
        assert_eq!(id.raw(), 42);
    }

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(align_up(256, 256), 256);
    }

    #[test]
    fn allocation_label_preserved() {
        let mut pool = HipMemoryPool::new(test_config()).unwrap();
        let id = pool.alloc(64, "kv_cache").unwrap();
        let alloc = pool.get_allocation(id).unwrap();
        assert_eq!(alloc.label, "kv_cache");
        pool.free(id).unwrap();
    }
}
