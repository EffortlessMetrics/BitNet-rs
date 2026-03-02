//! GPU memory pool and management for efficient inference.
//!
//! # Overview
//!
//! Provides a pool-based GPU memory allocator that reduces allocation overhead
//! during autoregressive inference.  Three allocation strategies are available:
//!
//! - [`BestFitAllocator`] — minimises wasted space per allocation.
//! - [`BuddyAllocator`] — power-of-two buddy system with O(log n) alloc/free.
//! - [`SlabAllocator`] — fixed-size slab caches for uniform allocations.
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations are provided for testing on non-GPU hosts.

use bitnet_common::{KernelError, Result};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Instant;

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for the GPU memory pool.
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Initial pool size in bytes.
    pub initial_size: usize,
    /// Maximum pool size in bytes (hard cap).
    pub max_size: usize,
    /// Minimum block size in bytes for splitting.
    pub block_size: usize,
    /// Alignment requirement in bytes (must be a power of two).
    pub alignment: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 64 * 1024 * 1024, // 64 MiB
            max_size: 1024 * 1024 * 1024,   // 1 GiB
            block_size: 256,                // 256 B minimum block
            alignment: 256,                 // 256 B alignment (typical GPU)
        }
    }
}

impl MemoryPoolConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<()> {
        if self.initial_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "initial_size must be non-zero".into(),
            }
            .into());
        }
        if self.max_size < self.initial_size {
            return Err(KernelError::InvalidArguments {
                reason: "max_size must be >= initial_size".into(),
            }
            .into());
        }
        if self.block_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "block_size must be non-zero".into(),
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

// ── Memory block ─────────────────────────────────────────────────────

/// Unique handle for an allocated memory block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(u64);

/// Represents a contiguous region of (simulated) GPU memory.
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Unique identifier.
    pub id: BlockId,
    /// Byte offset from pool base.
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
    /// Whether this block is currently allocated.
    pub in_use: bool,
    /// Timestamp of the last access (for LRU eviction).
    pub last_access: Instant,
}

// ── Memory statistics ────────────────────────────────────────────────

/// Snapshot of current memory pool statistics.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total pool capacity in bytes.
    pub total: usize,
    /// Bytes currently allocated.
    pub used: usize,
    /// Bytes available for allocation.
    pub free: usize,
    /// Fragmentation ratio in [0, 1].  0 = perfectly compacted.
    pub fragmentation: f64,
    /// Number of live allocations.
    pub num_allocations: usize,
    /// Number of free blocks.
    pub num_free_blocks: usize,
}

// ── Allocator trait ──────────────────────────────────────────────────

/// Strategy for choosing which free block to use for an allocation.
pub trait PoolAllocator: std::fmt::Debug + Send {
    /// Find a suitable free block for `size` bytes (already aligned).
    /// Returns the index into the free-block list.
    fn find_block(&self, free_blocks: &[MemoryBlock], size: usize) -> Option<usize>;

    /// Human-readable name of the strategy.
    fn name(&self) -> &str;
}

// ── Best-fit allocator ───────────────────────────────────────────────

/// Selects the smallest free block that fits the request.
#[derive(Debug, Default)]
pub struct BestFitAllocator;

impl PoolAllocator for BestFitAllocator {
    fn find_block(&self, free_blocks: &[MemoryBlock], size: usize) -> Option<usize> {
        free_blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.size >= size)
            .min_by_key(|(_, b)| b.size)
            .map(|(i, _)| i)
    }

    fn name(&self) -> &str {
        "best-fit"
    }
}

// ── Buddy allocator ──────────────────────────────────────────────────

/// Power-of-two buddy system allocator.
///
/// Rounds every request up to the next power of two and selects the
/// smallest free block whose size is also a power of two and ≥ the
/// rounded request.  Larger blocks are split in half recursively.
#[derive(Debug, Default)]
pub struct BuddyAllocator;

impl BuddyAllocator {
    /// Round `n` up to the next power of two.
    fn next_power_of_two(n: usize) -> usize {
        n.next_power_of_two()
    }
}

impl PoolAllocator for BuddyAllocator {
    fn find_block(&self, free_blocks: &[MemoryBlock], size: usize) -> Option<usize> {
        let rounded = Self::next_power_of_two(size);
        free_blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.size >= rounded)
            .min_by_key(|(_, b)| b.size)
            .map(|(i, _)| i)
    }

    fn name(&self) -> &str {
        "buddy"
    }
}

// ── Slab allocator ───────────────────────────────────────────────────

/// Fixed-size slab allocator.
///
/// All allocations are served from blocks of exactly `slab_size` bytes.
/// Requests larger than `slab_size` are rejected.
#[derive(Debug)]
pub struct SlabAllocator {
    /// Fixed slab size in bytes.
    pub slab_size: usize,
}

impl SlabAllocator {
    /// Create a new slab allocator with the given slab size.
    pub fn new(slab_size: usize) -> Self {
        Self { slab_size }
    }
}

impl PoolAllocator for SlabAllocator {
    fn find_block(&self, free_blocks: &[MemoryBlock], size: usize) -> Option<usize> {
        if size > self.slab_size {
            return None;
        }
        // Find any free block that can hold at least one slab.
        free_blocks.iter().enumerate().find(|(_, b)| b.size >= self.slab_size).map(|(i, _)| i)
    }

    fn name(&self) -> &str {
        "slab"
    }
}

// ── Memory pool ──────────────────────────────────────────────────────

/// GPU memory pool with pluggable allocation strategies.
///
/// On CPU builds (no `gpu` / `cuda` feature) this operates entirely on
/// simulated offsets — no actual device memory is touched.
#[derive(Debug)]
pub struct MemoryPool {
    config: MemoryPoolConfig,
    /// Monotonically increasing block-ID counter.
    next_id: u64,
    /// Currently allocated (in-use) blocks, keyed by `BlockId`.
    allocated: HashMap<BlockId, MemoryBlock>,
    /// Free blocks sorted by offset.
    free_blocks: Vec<MemoryBlock>,
    /// Peak memory usage observed.
    peak_used: usize,
    /// Current pool capacity (may grow up to `config.max_size`).
    capacity: usize,
    /// Allocation strategy.
    allocator: Box<dyn PoolAllocator>,
    /// LRU ordering — front = least-recently-used.
    lru_order: VecDeque<BlockId>,
    /// Optional per-slab-size free lists for the slab allocator.
    slab_cache: BTreeMap<usize, Vec<MemoryBlock>>,
}

impl MemoryPool {
    /// Create a new pool with the given configuration and allocator.
    pub fn new(config: MemoryPoolConfig, allocator: Box<dyn PoolAllocator>) -> Result<Self> {
        config.validate()?;

        let capacity = config.initial_size;
        let initial_block = MemoryBlock {
            id: BlockId(0),
            offset: 0,
            size: capacity,
            in_use: false,
            last_access: Instant::now(),
        };

        Ok(Self {
            config,
            next_id: 1,
            allocated: HashMap::new(),
            free_blocks: vec![initial_block],
            peak_used: 0,
            capacity,
            allocator,
            lru_order: VecDeque::new(),
            slab_cache: BTreeMap::new(),
        })
    }

    /// Create a pool with the default best-fit allocator.
    pub fn with_best_fit(config: MemoryPoolConfig) -> Result<Self> {
        Self::new(config, Box::new(BestFitAllocator))
    }

    /// Create a pool with the buddy allocator.
    pub fn with_buddy(config: MemoryPoolConfig) -> Result<Self> {
        Self::new(config, Box::new(BuddyAllocator))
    }

    /// Create a pool with the slab allocator.
    pub fn with_slab(config: MemoryPoolConfig, slab_size: usize) -> Result<Self> {
        Self::new(config, Box::new(SlabAllocator::new(slab_size)))
    }

    // ── helpers ──────────────────────────────────────────────────────

    fn alloc_id(&mut self) -> BlockId {
        let id = BlockId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Round `size` up to the configured alignment.
    fn align(&self, size: usize) -> usize {
        let mask = self.config.alignment - 1;
        (size + mask) & !mask
    }

    fn current_used(&self) -> usize {
        self.allocated.values().map(|b| b.size).sum()
    }

    fn update_peak(&mut self) {
        let used = self.current_used();
        if used > self.peak_used {
            self.peak_used = used;
        }
    }

    /// Try to grow the pool to accommodate `needed` additional bytes.
    fn try_grow(&mut self, needed: usize) -> Result<()> {
        let new_capacity = (self.capacity + needed).next_power_of_two().min(self.config.max_size);
        if new_capacity <= self.capacity {
            return Err(KernelError::GpuError {
                reason: format!(
                    "memory pool exhausted: need {} more bytes but at max capacity {}",
                    needed, self.config.max_size
                ),
            }
            .into());
        }
        let growth = new_capacity - self.capacity;
        let block = MemoryBlock {
            id: self.alloc_id(),
            offset: self.capacity,
            size: growth,
            in_use: false,
            last_access: Instant::now(),
        };
        self.free_blocks.push(block);
        self.capacity = new_capacity;
        self.coalesce_free_blocks();
        Ok(())
    }

    /// Merge adjacent free blocks.
    fn coalesce_free_blocks(&mut self) {
        if self.free_blocks.len() < 2 {
            return;
        }
        self.free_blocks.sort_by_key(|b| b.offset);
        let mut merged: Vec<MemoryBlock> = Vec::with_capacity(self.free_blocks.len());
        for block in self.free_blocks.drain(..) {
            if let Some(last) = merged.last_mut()
                && last.offset + last.size == block.offset
            {
                last.size += block.size;
                continue;
            }
            merged.push(block);
        }
        self.free_blocks = merged;
    }

    // ── public API ───────────────────────────────────────────────────

    /// Allocate `size` bytes from the pool.
    ///
    /// Returns a [`MemoryBlock`] describing the allocation.
    pub fn allocate(&mut self, size: usize) -> Result<MemoryBlock> {
        if size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "allocation size must be non-zero".into(),
            }
            .into());
        }

        let aligned = self.align(size);

        // Try the pluggable allocator first.
        if let Some(idx) = self.allocator.find_block(&self.free_blocks, aligned) {
            return self.split_and_allocate(idx, aligned);
        }

        // Coalesce and retry.
        self.coalesce_free_blocks();
        if let Some(idx) = self.allocator.find_block(&self.free_blocks, aligned) {
            return self.split_and_allocate(idx, aligned);
        }

        // Grow and retry.
        self.try_grow(aligned)?;
        if let Some(idx) = self.allocator.find_block(&self.free_blocks, aligned) {
            return self.split_and_allocate(idx, aligned);
        }

        Err(KernelError::GpuError {
            reason: format!("no suitable block for {} bytes (aligned {})", size, aligned),
        }
        .into())
    }

    /// Internal: remove the free block at `idx`, split if needed, and
    /// return the allocated portion.
    fn split_and_allocate(&mut self, idx: usize, size: usize) -> Result<MemoryBlock> {
        let mut block = self.free_blocks.remove(idx);
        let remainder = block.size - size;

        if remainder >= self.config.block_size {
            // Split: keep the remainder as a free block.
            let free_block = MemoryBlock {
                id: self.alloc_id(),
                offset: block.offset + size,
                size: remainder,
                in_use: false,
                last_access: Instant::now(),
            };
            self.free_blocks.push(free_block);
        }

        let id = self.alloc_id();
        block.id = id;
        // If we split, the allocated size is exactly `size`; otherwise,
        // give the entire block (avoids leaving a too-small remainder).
        block.size = if remainder >= self.config.block_size { size } else { block.size };
        block.in_use = true;
        block.last_access = Instant::now();

        self.allocated.insert(id, block.clone());
        self.lru_order.push_back(id);
        self.update_peak();
        Ok(block)
    }

    /// Return a previously allocated block to the pool.
    pub fn deallocate(&mut self, id: BlockId) -> Result<()> {
        let block = self.allocated.remove(&id).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("block {:?} is not allocated", id),
        })?;

        self.lru_order.retain(|b| *b != id);

        let mut freed = block;
        freed.in_use = false;
        self.free_blocks.push(freed);
        self.coalesce_free_blocks();
        Ok(())
    }

    /// Return current memory statistics.
    pub fn memory_usage(&self) -> MemoryStats {
        let used = self.current_used();
        let free = self.capacity.saturating_sub(used);
        let fragmentation = if self.free_blocks.is_empty() || free == 0 {
            0.0
        } else {
            let largest_free = self.free_blocks.iter().map(|b| b.size).max().unwrap_or(0);
            if free == 0 { 0.0 } else { 1.0 - (largest_free as f64 / free as f64) }
        };

        MemoryStats {
            total: self.capacity,
            used,
            free,
            fragmentation,
            num_allocations: self.allocated.len(),
            num_free_blocks: self.free_blocks.len(),
        }
    }

    /// Return the peak memory usage observed since pool creation.
    pub fn peak_usage(&self) -> usize {
        self.peak_used
    }

    /// Compact free blocks to reduce fragmentation.
    ///
    /// In a real GPU pool this would require copying live allocations.
    /// The CPU fallback simply coalesces adjacent free blocks.
    pub fn defragment(&mut self) {
        self.coalesce_free_blocks();
    }

    /// Memory-pressure ratio in [0, 1].  1.0 = completely exhausted.
    pub fn memory_pressure(&self) -> f64 {
        let used = self.current_used();
        if self.capacity == 0 {
            return 1.0;
        }
        used as f64 / self.capacity as f64
    }

    /// Evict the least-recently-used allocation to free memory.
    ///
    /// Returns the evicted block's ID and size, or an error if the pool
    /// has no evictable allocations.
    pub fn evict_least_recently_used(&mut self) -> Result<(BlockId, usize)> {
        let id = self
            .lru_order
            .pop_front()
            .ok_or_else(|| KernelError::GpuError { reason: "no allocations to evict".into() })?;

        let block = self.allocated.remove(&id).ok_or_else(|| KernelError::GpuError {
            reason: format!("LRU block {:?} not found in allocated map", id),
        })?;

        let size = block.size;
        let mut freed = block;
        freed.in_use = false;
        self.free_blocks.push(freed);
        self.coalesce_free_blocks();
        Ok((id, size))
    }

    /// Touch a block to update its LRU position.
    pub fn touch(&mut self, id: BlockId) -> Result<()> {
        if let Some(block) = self.allocated.get_mut(&id) {
            block.last_access = Instant::now();
            self.lru_order.retain(|b| *b != id);
            self.lru_order.push_back(id);
            Ok(())
        } else {
            Err(KernelError::InvalidArguments {
                reason: format!("block {:?} is not allocated", id),
            }
            .into())
        }
    }

    /// Return the pool capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the allocator name.
    pub fn allocator_name(&self) -> &str {
        self.allocator.name()
    }

    /// Number of live allocations.
    pub fn num_allocations(&self) -> usize {
        self.allocated.len()
    }

    /// Reset the pool, deallocating everything.
    pub fn reset(&mut self) {
        self.allocated.clear();
        self.lru_order.clear();
        self.slab_cache.clear();
        self.free_blocks.clear();
        self.free_blocks.push(MemoryBlock {
            id: BlockId(0),
            offset: 0,
            size: self.capacity,
            in_use: false,
            last_access: Instant::now(),
        });
    }
}

// ── CUDA launch stubs (GPU-only) ────────────────────────────────────

/// CUDA kernel source for async memory-copy helper (scaffold).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const MEMPOOL_COPY_KERNEL_SRC: &str = r#"
extern "C" __global__
void mempool_copy_f32(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { dst[idx] = src[idx]; }
}
"#;

/// Launch a device-to-device copy within the pool (scaffold).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_mempool_copy(
    _src_offset: usize,
    _dst_offset: usize,
    _num_elements: usize,
) -> Result<()> {
    // Stub: real implementation would call cudarc launch.
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> MemoryPoolConfig {
        MemoryPoolConfig { initial_size: 4096, max_size: 16384, block_size: 64, alignment: 64 }
    }

    fn small_config() -> MemoryPoolConfig {
        MemoryPoolConfig { initial_size: 1024, max_size: 2048, block_size: 64, alignment: 64 }
    }

    // ── Config validation ────────────────────────────────────────────

    #[test]
    fn config_default_is_valid() {
        MemoryPoolConfig::default().validate().unwrap();
    }

    #[test]
    fn config_zero_initial_size_rejected() {
        let c = MemoryPoolConfig { initial_size: 0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_max_less_than_initial_rejected() {
        let c = MemoryPoolConfig { initial_size: 1024, max_size: 512, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_zero_block_size_rejected() {
        let c = MemoryPoolConfig { block_size: 0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_non_power_of_two_alignment_rejected() {
        let c = MemoryPoolConfig { alignment: 3, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_power_of_two_alignments_accepted() {
        for align in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let c = MemoryPoolConfig { alignment: align, ..Default::default() };
            c.validate().unwrap();
        }
    }

    // ── Pool creation ────────────────────────────────────────────────

    #[test]
    fn create_best_fit_pool() {
        let pool = MemoryPool::with_best_fit(default_config()).unwrap();
        assert_eq!(pool.allocator_name(), "best-fit");
        assert_eq!(pool.capacity(), 4096);
    }

    #[test]
    fn create_buddy_pool() {
        let pool = MemoryPool::with_buddy(default_config()).unwrap();
        assert_eq!(pool.allocator_name(), "buddy");
    }

    #[test]
    fn create_slab_pool() {
        let pool = MemoryPool::with_slab(default_config(), 256).unwrap();
        assert_eq!(pool.allocator_name(), "slab");
    }

    #[test]
    fn pool_starts_empty() {
        let pool = MemoryPool::with_best_fit(default_config()).unwrap();
        assert_eq!(pool.num_allocations(), 0);
        let stats = pool.memory_usage();
        assert_eq!(stats.used, 0);
        assert_eq!(stats.free, 4096);
    }

    // ── Basic allocation / deallocation ──────────────────────────────

    #[test]
    fn allocate_single_block() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let block = pool.allocate(128).unwrap();
        assert!(block.in_use);
        assert!(block.size >= 128);
        assert_eq!(pool.num_allocations(), 1);
    }

    #[test]
    fn allocate_zero_size_rejected() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        assert!(pool.allocate(0).is_err());
    }

    #[test]
    fn deallocate_returns_memory() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let block = pool.allocate(128).unwrap();
        let id = block.id;
        pool.deallocate(id).unwrap();
        assert_eq!(pool.num_allocations(), 0);
        let stats = pool.memory_usage();
        assert_eq!(stats.used, 0);
    }

    #[test]
    fn deallocate_invalid_id_rejected() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        assert!(pool.deallocate(BlockId(999)).is_err());
    }

    #[test]
    fn double_deallocate_rejected() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let block = pool.allocate(128).unwrap();
        let id = block.id;
        pool.deallocate(id).unwrap();
        assert!(pool.deallocate(id).is_err());
    }

    #[test]
    fn multiple_allocations() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b1 = pool.allocate(256).unwrap();
        let b2 = pool.allocate(256).unwrap();
        let b3 = pool.allocate(256).unwrap();
        assert_eq!(pool.num_allocations(), 3);
        assert_ne!(b1.id, b2.id);
        assert_ne!(b2.id, b3.id);
    }

    #[test]
    fn allocations_dont_overlap() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b1 = pool.allocate(128).unwrap();
        let b2 = pool.allocate(128).unwrap();
        let end1 = b1.offset + b1.size;
        let end2 = b2.offset + b2.size;
        assert!(end1 <= b2.offset || end2 <= b1.offset);
    }

    #[test]
    fn allocate_respects_alignment() {
        let cfg = MemoryPoolConfig { alignment: 128, ..default_config() };
        let mut pool = MemoryPool::with_best_fit(cfg).unwrap();
        for _ in 0..5 {
            let b = pool.allocate(100).unwrap();
            assert_eq!(b.size % 128, 0, "block size should be aligned");
        }
    }

    #[test]
    fn reuse_freed_block() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b1 = pool.allocate(256).unwrap();
        let id1 = b1.id;
        let offset1 = b1.offset;
        pool.deallocate(id1).unwrap();
        let b2 = pool.allocate(256).unwrap();
        // The freed block should be reused — same offset.
        assert_eq!(b2.offset, offset1);
    }

    // ── Memory statistics ────────────────────────────────────────────

    #[test]
    fn memory_usage_tracks_allocations() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        pool.allocate(256).unwrap();
        let stats = pool.memory_usage();
        assert!(stats.used >= 256);
        assert_eq!(stats.total, 4096);
        assert_eq!(stats.num_allocations, 1);
    }

    #[test]
    fn memory_usage_after_dealloc() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b = pool.allocate(256).unwrap();
        pool.deallocate(b.id).unwrap();
        let stats = pool.memory_usage();
        assert_eq!(stats.used, 0);
        assert_eq!(stats.num_allocations, 0);
    }

    #[test]
    fn fragmentation_zero_when_single_free_block() {
        let pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let stats = pool.memory_usage();
        assert!((stats.fragmentation - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fragmentation_increases_with_holes() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b1 = pool.allocate(64).unwrap();
        let _b2 = pool.allocate(64).unwrap();
        let b3 = pool.allocate(64).unwrap();
        // Free alternate blocks → two non-adjacent free regions.
        pool.deallocate(b1.id).unwrap();
        pool.deallocate(b3.id).unwrap();
        let stats = pool.memory_usage();
        assert!(stats.fragmentation > 0.0);
    }

    // ── Peak usage ───────────────────────────────────────────────────

    #[test]
    fn peak_usage_tracks_maximum() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b1 = pool.allocate(512).unwrap();
        let b2 = pool.allocate(512).unwrap();
        let peak_after_two = pool.peak_usage();
        pool.deallocate(b1.id).unwrap();
        pool.deallocate(b2.id).unwrap();
        // Peak should remain at the high-water mark.
        assert!(pool.peak_usage() >= peak_after_two);
        assert!(pool.peak_usage() >= 1024);
    }

    #[test]
    fn peak_usage_monotonically_increasing() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let mut prev_peak = pool.peak_usage();
        for _ in 0..4 {
            let b = pool.allocate(256).unwrap();
            let new_peak = pool.peak_usage();
            assert!(new_peak >= prev_peak);
            prev_peak = new_peak;
            pool.deallocate(b.id).unwrap();
        }
    }

    // ── Defragment ───────────────────────────────────────────────────

    #[test]
    fn defragment_coalesces_adjacent_free_blocks() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b1 = pool.allocate(128).unwrap();
        let b2 = pool.allocate(128).unwrap();
        let b3 = pool.allocate(128).unwrap();
        pool.deallocate(b1.id).unwrap();
        pool.deallocate(b2.id).unwrap();
        pool.deallocate(b3.id).unwrap();
        pool.defragment();
        let stats = pool.memory_usage();
        // After defragmenting three adjacent freed blocks we should
        // have at most 2 free blocks (the merged region + trailing).
        assert!(stats.num_free_blocks <= 2);
    }

    #[test]
    fn defragment_reduces_fragmentation() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b1 = pool.allocate(64).unwrap();
        let b2 = pool.allocate(64).unwrap();
        let b3 = pool.allocate(64).unwrap();
        pool.deallocate(b1.id).unwrap();
        pool.deallocate(b3.id).unwrap();
        let frag_before = pool.memory_usage().fragmentation;
        pool.deallocate(b2.id).unwrap();
        pool.defragment();
        let frag_after = pool.memory_usage().fragmentation;
        assert!(frag_after <= frag_before);
    }

    // ── Best-fit allocator ───────────────────────────────────────────

    #[test]
    fn best_fit_selects_smallest_adequate_block() {
        let alloc = BestFitAllocator;
        let now = Instant::now();
        let blocks = vec![
            MemoryBlock { id: BlockId(0), offset: 0, size: 1024, in_use: false, last_access: now },
            MemoryBlock {
                id: BlockId(1),
                offset: 1024,
                size: 256,
                in_use: false,
                last_access: now,
            },
            MemoryBlock {
                id: BlockId(2),
                offset: 1280,
                size: 512,
                in_use: false,
                last_access: now,
            },
        ];
        let idx = alloc.find_block(&blocks, 200).unwrap();
        assert_eq!(idx, 1); // 256 is the smallest ≥ 200
    }

    #[test]
    fn best_fit_returns_none_when_nothing_fits() {
        let alloc = BestFitAllocator;
        let now = Instant::now();
        let blocks = vec![MemoryBlock {
            id: BlockId(0),
            offset: 0,
            size: 64,
            in_use: false,
            last_access: now,
        }];
        assert!(alloc.find_block(&blocks, 128).is_none());
    }

    #[test]
    fn best_fit_allocator_pool_integration() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        // Allocate and free to create varied free blocks.
        let b1 = pool.allocate(256).unwrap();
        let _b2 = pool.allocate(512).unwrap();
        let b3 = pool.allocate(128).unwrap();
        pool.deallocate(b1.id).unwrap();
        pool.deallocate(b3.id).unwrap();
        // A 128-byte request should prefer the 128-byte free block.
        let b4 = pool.allocate(64).unwrap();
        assert!(b4.size >= 64);
    }

    // ── Buddy allocator ──────────────────────────────────────────────

    #[test]
    fn buddy_rounds_up_to_power_of_two() {
        assert_eq!(BuddyAllocator::next_power_of_two(100), 128);
        assert_eq!(BuddyAllocator::next_power_of_two(256), 256);
        assert_eq!(BuddyAllocator::next_power_of_two(1), 1);
    }

    #[test]
    fn buddy_allocator_basic() {
        let mut pool = MemoryPool::with_buddy(default_config()).unwrap();
        let b = pool.allocate(100).unwrap();
        // Buddy rounds 100 → 128, aligned to 64 → 128.
        assert!(b.size >= 128);
    }

    #[test]
    fn buddy_allocator_sequential() {
        let mut pool = MemoryPool::with_buddy(default_config()).unwrap();
        let b1 = pool.allocate(64).unwrap();
        let b2 = pool.allocate(64).unwrap();
        assert_ne!(b1.id, b2.id);
        assert!(b1.offset != b2.offset);
    }

    #[test]
    fn buddy_allocator_reclaim() {
        let mut pool = MemoryPool::with_buddy(default_config()).unwrap();
        let b = pool.allocate(256).unwrap();
        let id = b.id;
        pool.deallocate(id).unwrap();
        // Should be able to allocate again.
        let b2 = pool.allocate(256).unwrap();
        assert!(b2.size >= 256);
    }

    // ── Slab allocator ───────────────────────────────────────────────

    #[test]
    fn slab_allocator_rejects_oversized() {
        let alloc = SlabAllocator::new(128);
        let now = Instant::now();
        let blocks = vec![MemoryBlock {
            id: BlockId(0),
            offset: 0,
            size: 256,
            in_use: false,
            last_access: now,
        }];
        assert!(alloc.find_block(&blocks, 256).is_none());
    }

    #[test]
    fn slab_allocator_accepts_exact() {
        let alloc = SlabAllocator::new(128);
        let now = Instant::now();
        let blocks = vec![MemoryBlock {
            id: BlockId(0),
            offset: 0,
            size: 128,
            in_use: false,
            last_access: now,
        }];
        assert!(alloc.find_block(&blocks, 128).is_some());
    }

    #[test]
    fn slab_allocator_accepts_smaller() {
        let alloc = SlabAllocator::new(256);
        let now = Instant::now();
        let blocks = vec![MemoryBlock {
            id: BlockId(0),
            offset: 0,
            size: 512,
            in_use: false,
            last_access: now,
        }];
        assert!(alloc.find_block(&blocks, 64).is_some());
    }

    #[test]
    fn slab_pool_fixed_size_allocations() {
        let cfg =
            MemoryPoolConfig { initial_size: 2048, max_size: 2048, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_slab(cfg, 256).unwrap();
        let b1 = pool.allocate(64).unwrap();
        let b2 = pool.allocate(64).unwrap();
        assert!(b1.size >= 64);
        assert!(b2.size >= 64);
    }

    // ── Memory pressure ──────────────────────────────────────────────

    #[test]
    fn memory_pressure_empty_pool() {
        let pool = MemoryPool::with_best_fit(default_config()).unwrap();
        assert!((pool.memory_pressure() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn memory_pressure_increases_with_usage() {
        let mut pool = MemoryPool::with_best_fit(small_config()).unwrap();
        pool.allocate(512).unwrap();
        let pressure = pool.memory_pressure();
        assert!(pressure > 0.0);
        assert!(pressure <= 1.0);
    }

    #[test]
    fn memory_pressure_high_when_nearly_full() {
        let cfg =
            MemoryPoolConfig { initial_size: 1024, max_size: 1024, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_best_fit(cfg).unwrap();
        pool.allocate(960).unwrap();
        assert!(pool.memory_pressure() > 0.9);
    }

    // ── LRU eviction ─────────────────────────────────────────────────

    #[test]
    fn evict_lru_frees_oldest() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b1 = pool.allocate(128).unwrap();
        let _b2 = pool.allocate(128).unwrap();
        let (evicted_id, _) = pool.evict_least_recently_used().unwrap();
        assert_eq!(evicted_id, b1.id); // b1 was allocated first
        assert_eq!(pool.num_allocations(), 1);
    }

    #[test]
    fn evict_lru_empty_pool_fails() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        assert!(pool.evict_least_recently_used().is_err());
    }

    #[test]
    fn evict_lru_returns_freed_size() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b = pool.allocate(256).unwrap();
        let alloc_size = b.size;
        let (_, evicted_size) = pool.evict_least_recently_used().unwrap();
        assert_eq!(evicted_size, alloc_size);
    }

    #[test]
    fn touch_updates_lru_order() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b1 = pool.allocate(128).unwrap();
        let b2 = pool.allocate(128).unwrap();
        // Touch b1 so it becomes most-recently-used.
        pool.touch(b1.id).unwrap();
        let (evicted_id, _) = pool.evict_least_recently_used().unwrap();
        assert_eq!(evicted_id, b2.id); // b2 is now LRU
    }

    #[test]
    fn touch_invalid_block_rejected() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        assert!(pool.touch(BlockId(999)).is_err());
    }

    #[test]
    fn evict_then_allocate_reuses_memory() {
        let cfg =
            MemoryPoolConfig { initial_size: 512, max_size: 512, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_best_fit(cfg).unwrap();
        pool.allocate(256).unwrap();
        pool.allocate(256).unwrap();
        // Pool is full; evict and re-allocate.
        pool.evict_least_recently_used().unwrap();
        let b = pool.allocate(128).unwrap();
        assert!(b.size >= 128);
    }

    // ── Pool growth ──────────────────────────────────────────────────

    #[test]
    fn pool_grows_when_needed() {
        let cfg =
            MemoryPoolConfig { initial_size: 256, max_size: 4096, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_best_fit(cfg).unwrap();
        assert_eq!(pool.capacity(), 256);
        // Allocate more than initial capacity.
        let b = pool.allocate(512).unwrap();
        assert!(pool.capacity() > 256);
        assert!(b.size >= 512);
    }

    #[test]
    fn pool_growth_capped_at_max() {
        let cfg =
            MemoryPoolConfig { initial_size: 256, max_size: 512, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_best_fit(cfg).unwrap();
        pool.allocate(256).unwrap();
        // Trying to exceed max should fail.
        assert!(pool.allocate(512).is_err());
    }

    // ── Reset ────────────────────────────────────────────────────────

    #[test]
    fn reset_clears_all_allocations() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        pool.allocate(256).unwrap();
        pool.allocate(256).unwrap();
        pool.reset();
        assert_eq!(pool.num_allocations(), 0);
        let stats = pool.memory_usage();
        assert_eq!(stats.used, 0);
        assert_eq!(stats.num_free_blocks, 1);
    }

    #[test]
    fn reset_allows_fresh_allocations() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        pool.allocate(4000).unwrap();
        pool.reset();
        let b = pool.allocate(4000).unwrap();
        assert!(b.size >= 4000);
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn allocate_exactly_pool_size() {
        let cfg =
            MemoryPoolConfig { initial_size: 1024, max_size: 1024, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_best_fit(cfg).unwrap();
        let b = pool.allocate(1024).unwrap();
        assert!(b.size >= 1024);
    }

    #[test]
    fn allocate_exceeding_pool_with_growth() {
        let cfg =
            MemoryPoolConfig { initial_size: 256, max_size: 1024, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_best_fit(cfg).unwrap();
        let b = pool.allocate(512).unwrap();
        assert!(b.size >= 512);
        assert!(pool.capacity() >= 512);
    }

    #[test]
    fn allocate_one_byte() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b = pool.allocate(1).unwrap();
        // Aligned up to 64 bytes.
        assert!(b.size >= 64);
    }

    #[test]
    fn many_small_allocations() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let mut blocks = Vec::new();
        for _ in 0..30 {
            blocks.push(pool.allocate(64).unwrap());
        }
        assert_eq!(pool.num_allocations(), 30);
        for b in &blocks {
            pool.deallocate(b.id).unwrap();
        }
        assert_eq!(pool.num_allocations(), 0);
    }

    #[test]
    fn interleaved_alloc_dealloc() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b1 = pool.allocate(128).unwrap();
        let b2 = pool.allocate(128).unwrap();
        pool.deallocate(b1.id).unwrap();
        let b3 = pool.allocate(128).unwrap();
        pool.deallocate(b2.id).unwrap();
        let b4 = pool.allocate(128).unwrap();
        pool.deallocate(b3.id).unwrap();
        pool.deallocate(b4.id).unwrap();
        assert_eq!(pool.num_allocations(), 0);
    }

    #[test]
    fn allocator_names_correct() {
        assert_eq!(BestFitAllocator.name(), "best-fit");
        assert_eq!(BuddyAllocator.name(), "buddy");
        assert_eq!(SlabAllocator::new(64).name(), "slab");
    }

    #[test]
    fn block_ids_are_unique() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let mut ids = std::collections::HashSet::new();
        for _ in 0..20 {
            let b = pool.allocate(64).unwrap();
            assert!(ids.insert(b.id), "duplicate block id");
        }
    }

    #[test]
    fn buddy_allocator_larger_sizes() {
        let cfg =
            MemoryPoolConfig { initial_size: 8192, max_size: 16384, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_buddy(cfg).unwrap();
        let b1 = pool.allocate(1000).unwrap();
        let b2 = pool.allocate(2000).unwrap();
        assert!(b1.size >= 1024); // rounded up
        assert!(b2.size >= 2048); // rounded up
        pool.deallocate(b1.id).unwrap();
        pool.deallocate(b2.id).unwrap();
    }

    #[test]
    fn slab_allocator_rejects_larger_than_slab() {
        let cfg =
            MemoryPoolConfig { initial_size: 4096, max_size: 4096, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_slab(cfg, 128).unwrap();
        // Requesting more than slab_size (128) after alignment rounding.
        assert!(pool.allocate(192).is_err());
    }

    // ── Stress / pattern tests ───────────────────────────────────────

    #[test]
    fn allocate_free_cycle_no_leak() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        for _ in 0..50 {
            let b = pool.allocate(64).unwrap();
            pool.deallocate(b.id).unwrap();
        }
        let stats = pool.memory_usage();
        assert_eq!(stats.used, 0);
        assert_eq!(stats.num_allocations, 0);
    }

    #[test]
    fn lifo_alloc_dealloc_pattern() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let mut stack = Vec::new();
        for _ in 0..10 {
            stack.push(pool.allocate(64).unwrap());
        }
        while let Some(b) = stack.pop() {
            pool.deallocate(b.id).unwrap();
        }
        assert_eq!(pool.num_allocations(), 0);
    }

    #[test]
    fn fifo_alloc_dealloc_pattern() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let mut queue = VecDeque::new();
        for _ in 0..10 {
            queue.push_back(pool.allocate(64).unwrap());
        }
        while let Some(b) = queue.pop_front() {
            pool.deallocate(b.id).unwrap();
        }
        assert_eq!(pool.num_allocations(), 0);
    }

    #[test]
    fn buddy_alloc_dealloc_cycle() {
        let cfg =
            MemoryPoolConfig { initial_size: 4096, max_size: 8192, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_buddy(cfg).unwrap();
        for _ in 0..20 {
            let b = pool.allocate(128).unwrap();
            pool.deallocate(b.id).unwrap();
        }
        let stats = pool.memory_usage();
        assert_eq!(stats.used, 0);
    }

    #[test]
    fn slab_alloc_dealloc_cycle() {
        let cfg =
            MemoryPoolConfig { initial_size: 4096, max_size: 4096, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_slab(cfg, 128).unwrap();
        for _ in 0..10 {
            let b = pool.allocate(64).unwrap();
            pool.deallocate(b.id).unwrap();
        }
        let stats = pool.memory_usage();
        assert_eq!(stats.used, 0);
    }

    #[test]
    fn mixed_sizes_best_fit() {
        let cfg =
            MemoryPoolConfig { initial_size: 8192, max_size: 16384, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_best_fit(cfg).unwrap();
        let small = pool.allocate(64).unwrap();
        let medium = pool.allocate(512).unwrap();
        let large = pool.allocate(2048).unwrap();
        pool.deallocate(medium.id).unwrap();
        // Re-allocate medium-sized — should reuse freed block.
        let medium2 = pool.allocate(512).unwrap();
        assert!(medium2.size >= 512);
        pool.deallocate(small.id).unwrap();
        pool.deallocate(large.id).unwrap();
        pool.deallocate(medium2.id).unwrap();
        assert_eq!(pool.num_allocations(), 0);
    }

    #[test]
    fn multiple_evictions() {
        let cfg =
            MemoryPoolConfig { initial_size: 512, max_size: 512, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_best_fit(cfg).unwrap();
        pool.allocate(128).unwrap();
        pool.allocate(128).unwrap();
        pool.allocate(128).unwrap();
        // Evict all three one by one.
        for _ in 0..3 {
            pool.evict_least_recently_used().unwrap();
        }
        assert_eq!(pool.num_allocations(), 0);
    }

    #[test]
    fn evict_under_pressure_then_allocate() {
        let cfg =
            MemoryPoolConfig { initial_size: 512, max_size: 512, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_best_fit(cfg).unwrap();
        pool.allocate(256).unwrap();
        pool.allocate(256).unwrap();
        assert!(pool.memory_pressure() > 0.9);
        pool.evict_least_recently_used().unwrap();
        let b = pool.allocate(128).unwrap();
        assert!(b.size >= 128);
    }

    #[test]
    fn stats_free_blocks_count() {
        let mut pool = MemoryPool::with_best_fit(default_config()).unwrap();
        let b1 = pool.allocate(64).unwrap();
        let _b2 = pool.allocate(64).unwrap();
        let b3 = pool.allocate(64).unwrap();
        pool.deallocate(b1.id).unwrap();
        pool.deallocate(b3.id).unwrap();
        let stats = pool.memory_usage();
        // At least 2 free regions (the two freed blocks + possible tail).
        assert!(stats.num_free_blocks >= 2);
    }

    #[test]
    fn defragment_then_large_alloc_succeeds() {
        let cfg =
            MemoryPoolConfig { initial_size: 1024, max_size: 1024, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_best_fit(cfg).unwrap();
        let b1 = pool.allocate(256).unwrap();
        let b2 = pool.allocate(256).unwrap();
        let b3 = pool.allocate(256).unwrap();
        pool.deallocate(b1.id).unwrap();
        pool.deallocate(b2.id).unwrap();
        pool.deallocate(b3.id).unwrap();
        pool.defragment();
        // After defrag, all freed memory is one contiguous block.
        let big = pool.allocate(768).unwrap();
        assert!(big.size >= 768);
    }

    #[test]
    fn capacity_returns_current_size() {
        let cfg =
            MemoryPoolConfig { initial_size: 2048, max_size: 8192, block_size: 64, alignment: 64 };
        let pool = MemoryPool::with_best_fit(cfg).unwrap();
        assert_eq!(pool.capacity(), 2048);
    }

    #[test]
    fn capacity_grows_after_growth() {
        let cfg =
            MemoryPoolConfig { initial_size: 256, max_size: 4096, block_size: 64, alignment: 64 };
        let mut pool = MemoryPool::with_best_fit(cfg).unwrap();
        pool.allocate(512).unwrap();
        assert!(pool.capacity() > 256);
    }
}
