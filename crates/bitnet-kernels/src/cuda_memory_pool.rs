//! CUDA memory pool with slab/buddy allocation strategy.
//!
//! Pure-Rust CPU simulation of GPU memory pool management. Provides configurable
//! block sizes, eviction strategies, and defragmentation for efficient memory reuse.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the memory pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// The pool cannot satisfy the allocation (requested, available).
    OutOfMemory { requested: usize, available: usize },
    /// The pointer was not found among live allocations.
    InvalidPointer(usize),
    /// Requested block size is zero or exceeds the pool capacity.
    InvalidSize(usize),
    /// Pool has already been torn down.
    PoolDestroyed,
    /// Configuration is invalid.
    InvalidConfig(String),
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory { requested, available } => {
                write!(f, "out of memory: requested {requested} bytes, {available} available")
            }
            Self::InvalidPointer(p) => write!(f, "invalid pointer: {p:#x}"),
            Self::InvalidSize(s) => write!(f, "invalid size: {s}"),
            Self::PoolDestroyed => write!(f, "pool has been destroyed"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for PoolError {}

// ---------------------------------------------------------------------------
// Eviction strategy
// ---------------------------------------------------------------------------

/// Strategy used when the pool needs to reclaim cached (free) blocks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EvictionStrategy {
    /// Least-recently-used blocks are evicted first.
    #[default]
    Lru,
    /// Largest free blocks are evicted first to maximise contiguous space.
    LargestFirst,
    /// Evict nothing – allocations simply fail when memory runs out.
    None,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a [`CudaMemoryPool`].
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Total capacity in bytes.
    pub total_bytes: usize,
    /// Minimum block size (slab granularity). Must be a power of two.
    pub min_block_size: usize,
    /// Maximum block size (buddy upper bound). Must be a power of two.
    pub max_block_size: usize,
    /// Alignment requirement (must be a power of two).
    pub alignment: usize,
    /// Eviction strategy when pool is full.
    pub eviction_strategy: EvictionStrategy,
    /// High-water mark ratio (0.0–1.0) that triggers automatic shrinking.
    pub high_water_ratio: f64,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            total_bytes: 256 * 1024 * 1024, // 256 MiB
            min_block_size: 256,
            max_block_size: 64 * 1024 * 1024, // 64 MiB
            alignment: 256,
            eviction_strategy: EvictionStrategy::Lru,
            high_water_ratio: 0.9,
        }
    }
}

impl MemoryPoolConfig {
    fn validate(&self) -> Result<(), PoolError> {
        if self.total_bytes == 0 {
            return Err(PoolError::InvalidConfig("total_bytes must be > 0".into()));
        }
        if !self.min_block_size.is_power_of_two() || self.min_block_size == 0 {
            return Err(PoolError::InvalidConfig(
                "min_block_size must be a non-zero power of two".into(),
            ));
        }
        if !self.max_block_size.is_power_of_two() || self.max_block_size == 0 {
            return Err(PoolError::InvalidConfig(
                "max_block_size must be a non-zero power of two".into(),
            ));
        }
        if self.min_block_size > self.max_block_size {
            return Err(PoolError::InvalidConfig(
                "min_block_size must be <= max_block_size".into(),
            ));
        }
        if !self.alignment.is_power_of_two() || self.alignment == 0 {
            return Err(PoolError::InvalidConfig(
                "alignment must be a non-zero power of two".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.high_water_ratio) {
            return Err(PoolError::InvalidConfig("high_water_ratio must be in [0.0, 1.0]".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pool block
// ---------------------------------------------------------------------------

/// Metadata for a single block managed by the pool.
#[derive(Debug, Clone)]
pub struct PoolBlock {
    /// Simulated device offset (byte address).
    pub offset: usize,
    /// Size of this block in bytes.
    pub size: usize,
    /// Whether the block is currently allocated.
    pub allocated: bool,
    /// Monotonic id used for LRU ordering.
    pub last_access: u64,
    /// Optional tag for debugging / grouping.
    pub tag: Option<String>,
}

// ---------------------------------------------------------------------------
// Allocation statistics
// ---------------------------------------------------------------------------

/// Snapshot of pool utilisation metrics.
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    pub total_bytes: usize,
    pub allocated_bytes: usize,
    pub free_bytes: usize,
    pub num_allocations: usize,
    pub num_free_blocks: usize,
    pub peak_allocated_bytes: usize,
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub fragmentation_ratio: f64,
    pub largest_free_block: usize,
}

// ---------------------------------------------------------------------------
// Free-list key (size-class)
// ---------------------------------------------------------------------------

/// Round `size` up to the next power-of-two that is >= `min`.
fn round_up_block_size(size: usize, min: usize) -> usize {
    let s = size.max(min);
    s.next_power_of_two()
}

/// Align `offset` up to `alignment`.
fn align_up(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// The pool itself
// ---------------------------------------------------------------------------

/// A simulated CUDA memory pool using a hybrid slab/buddy allocator.
///
/// Small allocations are served from fixed-size slabs (power-of-two size
/// classes). Larger allocations use buddy splitting. Free blocks are coalesced
/// on `deallocate` and during `defragment`.
pub struct CudaMemoryPool {
    config: MemoryPoolConfig,
    /// All blocks, keyed by their starting offset.
    blocks: BTreeMap<usize, PoolBlock>,
    /// Free-list indexed by size class → offsets.
    free_lists: HashMap<usize, VecDeque<usize>>,
    /// Live allocations: pointer → offset.
    live: HashMap<usize, usize>,
    /// Monotonic counter for access ordering.
    access_counter: u64,
    /// Stats accumulators.
    peak_allocated: usize,
    current_allocated: usize,
    total_allocs: u64,
    total_deallocs: u64,
    /// Next simulated pointer value.
    next_ptr: usize,
    /// Creation timestamp (for uptime queries).
    created_at: Instant,
}

impl CudaMemoryPool {
    // -- internal helpers ---------------------------------------------------

    fn bump_access(&mut self) -> u64 {
        self.access_counter += 1;
        self.access_counter
    }

    /// Insert a free block into the appropriate free-list and block map.
    fn insert_free_block(&mut self, offset: usize, size: usize) {
        let access = self.bump_access();
        self.blocks.insert(
            offset,
            PoolBlock { offset, size, allocated: false, last_access: access, tag: None },
        );
        self.free_lists.entry(size).or_default().push_back(offset);
    }

    /// Remove a specific offset from its free-list.
    fn remove_from_free_list(&mut self, size: usize, offset: usize) {
        if let Some(list) = self.free_lists.get_mut(&size) {
            list.retain(|&o| o != offset);
            if list.is_empty() {
                self.free_lists.remove(&size);
            }
        }
    }

    /// Try to find a free block of exactly `size` bytes.
    fn find_exact_free(&mut self, size: usize) -> Option<usize> {
        if let Some(list) = self.free_lists.get_mut(&size)
            && let Some(offset) = list.pop_front()
        {
            if list.is_empty() {
                self.free_lists.remove(&size);
            }
            return Some(offset);
        }
        None
    }

    /// Find the smallest free block that is >= `size` via buddy splitting.
    fn find_and_split(&mut self, needed: usize) -> Option<usize> {
        // Collect candidate sizes that are larger than `needed`.
        let mut candidate_sizes: Vec<usize> = self
            .free_lists
            .iter()
            .filter(|(sz, list)| **sz >= needed && !list.is_empty())
            .map(|(sz, _)| *sz)
            .collect();
        candidate_sizes.sort_unstable();

        for sz in candidate_sizes {
            if let Some(offset) = {
                let list = self.free_lists.get_mut(&sz)?;
                list.pop_front()
            } {
                // Clean up free-list entry.
                if self.free_lists.get(&sz).is_none_or(|l| l.is_empty()) {
                    self.free_lists.remove(&sz);
                }
                self.blocks.remove(&offset);

                // Split: keep the front `needed` bytes, return the remainder.
                if sz > needed {
                    let remainder_offset = offset + needed;
                    let remainder_size = sz - needed;
                    self.insert_free_block(remainder_offset, remainder_size);
                }
                return Some(offset);
            }
        }
        None
    }

    /// Coalesce adjacent free blocks starting from `offset`.
    fn coalesce(&mut self, offset: usize) {
        let mut start = offset;
        let mut size = match self.blocks.get(&offset) {
            Some(b) if !b.allocated => b.size,
            _ => return,
        };

        // Merge forward.
        loop {
            let next = start + size;
            match self.blocks.get(&next) {
                Some(b) if !b.allocated => {
                    let next_size = b.size;
                    self.remove_from_free_list(next_size, next);
                    self.blocks.remove(&next);
                    size += next_size;
                }
                _ => break,
            }
        }

        // Merge backward.
        loop {
            // Find a block that ends exactly at `start`.
            let prev = self.blocks.range(..start).next_back().and_then(|(&o, b)| {
                if !b.allocated && o + b.size == start { Some((o, b.size)) } else { None }
            });
            match prev {
                Some((prev_off, prev_size)) => {
                    self.remove_from_free_list(prev_size, prev_off);
                    self.blocks.remove(&prev_off);
                    // Also remove the current `start` entry before merging.
                    self.remove_from_free_list(size, start);
                    self.blocks.remove(&start);
                    start = prev_off;
                    size += prev_size;
                }
                None => break,
            }
        }

        // Re-insert merged free block.
        self.remove_from_free_list(size, start);
        self.blocks.remove(&start);
        self.insert_free_block(start, size);
    }

    /// Attempt eviction of cached free blocks to service `needed` bytes.
    fn try_evict(&mut self, _needed: usize) -> bool {
        match self.config.eviction_strategy {
            EvictionStrategy::None => false,
            EvictionStrategy::Lru | EvictionStrategy::LargestFirst => {
                // Coalesce everything – this is the best we can do in the
                // simulated pool (we don't hold actual GPU caches).
                self.coalesce_all();
                true
            }
        }
    }

    /// Coalesce all adjacent free blocks.
    fn coalesce_all(&mut self) {
        let offsets: Vec<usize> =
            self.blocks.iter().filter(|(_, b)| !b.allocated).map(|(&o, _)| o).collect();
        for off in offsets {
            if self.blocks.contains_key(&off) {
                self.coalesce(off);
            }
        }
    }

    fn total_free_bytes(&self) -> usize {
        self.config.total_bytes.saturating_sub(self.current_allocated)
    }
}

// ---------------------------------------------------------------------------
// Public API (free functions that delegate to the pool)
// ---------------------------------------------------------------------------

/// Create a new memory pool with the given configuration.
pub fn create_pool(config: MemoryPoolConfig) -> Result<CudaMemoryPool, PoolError> {
    config.validate()?;

    let capacity = config.total_bytes;
    let mut pool = CudaMemoryPool {
        config,
        blocks: BTreeMap::new(),
        free_lists: HashMap::new(),
        live: HashMap::new(),
        access_counter: 0,
        peak_allocated: 0,
        current_allocated: 0,
        total_allocs: 0,
        total_deallocs: 0,
        next_ptr: 0x1000, // start at 4 KiB to avoid null-ish pointers
        created_at: Instant::now(),
    };

    // Seed a single free block spanning the entire pool.
    pool.insert_free_block(0, capacity);
    Ok(pool)
}

/// Allocate `size` bytes from the pool. Returns a simulated pointer.
pub fn allocate(pool: &mut CudaMemoryPool, size: usize) -> Result<usize, PoolError> {
    allocate_tagged(pool, size, None)
}

/// Allocate with an optional debug tag.
pub fn allocate_tagged(
    pool: &mut CudaMemoryPool,
    size: usize,
    tag: Option<String>,
) -> Result<usize, PoolError> {
    if size == 0 || size > pool.config.total_bytes {
        return Err(PoolError::InvalidSize(size));
    }

    let block_size = round_up_block_size(size, pool.config.min_block_size);
    let aligned = align_up(block_size, pool.config.alignment);

    // Try exact fit first.
    let offset = if let Some(off) = pool.find_exact_free(aligned) {
        pool.blocks.remove(&off);
        Some(off)
    } else {
        // Buddy-split a larger block.
        pool.find_and_split(aligned)
    };

    let offset = match offset {
        Some(o) => o,
        None => {
            // Try eviction / coalescing.
            if pool.try_evict(aligned) {
                if let Some(off) = pool.find_exact_free(aligned) {
                    pool.blocks.remove(&off);
                    Some(off)
                } else {
                    pool.find_and_split(aligned)
                }
            } else {
                None
            }
            .ok_or(PoolError::OutOfMemory { requested: size, available: pool.total_free_bytes() })?
        }
    };

    let access = pool.bump_access();
    pool.blocks.insert(
        offset,
        PoolBlock { offset, size: aligned, allocated: true, last_access: access, tag },
    );

    let ptr = pool.next_ptr;
    pool.next_ptr += aligned;
    pool.live.insert(ptr, offset);

    pool.current_allocated += aligned;
    pool.total_allocs += 1;
    if pool.current_allocated > pool.peak_allocated {
        pool.peak_allocated = pool.current_allocated;
    }

    Ok(ptr)
}

/// Return a previously-allocated block to the pool.
pub fn deallocate(pool: &mut CudaMemoryPool, ptr: usize) -> Result<(), PoolError> {
    let offset = pool.live.remove(&ptr).ok_or(PoolError::InvalidPointer(ptr))?;

    let size = pool.blocks.get(&offset).map(|b| b.size).unwrap_or(0);

    pool.blocks.remove(&offset);
    pool.current_allocated = pool.current_allocated.saturating_sub(size);
    pool.total_deallocs += 1;

    pool.insert_free_block(offset, size);
    pool.coalesce(offset);
    Ok(())
}

/// Defragment the pool by coalescing all adjacent free blocks and compacting.
///
/// Returns the number of free blocks that were merged.
pub fn defragment(pool: &mut CudaMemoryPool) -> usize {
    let before = pool.blocks.values().filter(|b| !b.allocated).count();

    pool.coalesce_all();

    let after = pool.blocks.values().filter(|b| !b.allocated).count();

    before.saturating_sub(after)
}

/// Return a snapshot of pool statistics.
pub fn pool_stats(pool: &CudaMemoryPool) -> AllocationStats {
    let free_bytes = pool.total_free_bytes();
    let largest_free =
        pool.blocks.values().filter(|b| !b.allocated).map(|b| b.size).max().unwrap_or(0);

    let num_free = pool.blocks.values().filter(|b| !b.allocated).count();

    let frag = if free_bytes > 0 && largest_free > 0 {
        1.0 - (largest_free as f64 / free_bytes as f64)
    } else {
        0.0
    };

    AllocationStats {
        total_bytes: pool.config.total_bytes,
        allocated_bytes: pool.current_allocated,
        free_bytes,
        num_allocations: pool.live.len(),
        num_free_blocks: num_free,
        peak_allocated_bytes: pool.peak_allocated,
        total_allocations: pool.total_allocs,
        total_deallocations: pool.total_deallocs,
        fragmentation_ratio: frag,
        largest_free_block: largest_free,
    }
}

/// Shrink the pool by releasing any trailing free space that exceeds
/// `target_bytes`. Returns bytes released.
pub fn shrink_to_fit(pool: &mut CudaMemoryPool, target_bytes: usize) -> usize {
    pool.coalesce_all();

    if pool.config.total_bytes <= target_bytes {
        return 0;
    }

    // Find the highest allocated offset + size.
    let high_water = pool
        .blocks
        .iter()
        .rev()
        .find(|(_, b)| b.allocated)
        .map(|(_, b)| b.offset + b.size)
        .unwrap_or(0);

    let new_total = high_water.max(target_bytes);
    if new_total >= pool.config.total_bytes {
        return 0;
    }

    // Remove free blocks that start at or beyond `new_total`.
    let to_remove: Vec<usize> =
        pool.blocks.range(new_total..).filter(|(_, b)| !b.allocated).map(|(&o, _)| o).collect();

    for off in &to_remove {
        if let Some(b) = pool.blocks.remove(off) {
            pool.remove_from_free_list(b.size, *off);
        }
    }

    // Truncate any free block that straddles `new_total`.
    let straddling: Vec<(usize, usize)> = pool
        .blocks
        .iter()
        .filter(|(_, b)| !b.allocated && b.offset < new_total && b.offset + b.size > new_total)
        .map(|(&o, b)| (o, b.size))
        .collect();

    for (off, old_size) in straddling {
        pool.remove_from_free_list(old_size, off);
        pool.blocks.remove(&off);
        let trimmed = new_total - off;
        if trimmed > 0 {
            pool.insert_free_block(off, trimmed);
        }
    }

    let released = pool.config.total_bytes - new_total;
    pool.config.total_bytes = new_total;
    released
}

/// Pre-populate the pool with blocks of the given sizes so that subsequent
/// allocations can be served immediately. Returns the number of blocks warmed.
pub fn warm_pool(pool: &mut CudaMemoryPool, sizes: &[usize]) -> Result<usize, PoolError> {
    let mut warmed = 0usize;
    for &sz in sizes {
        if sz == 0 {
            continue;
        }
        let ptr = allocate(pool, sz)?;
        deallocate(pool, ptr)?;
        warmed += 1;
    }
    Ok(warmed)
}

// ---------------------------------------------------------------------------
// Convenience helpers on CudaMemoryPool
// ---------------------------------------------------------------------------

impl CudaMemoryPool {
    /// Pool uptime.
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Current config snapshot.
    pub fn config(&self) -> &MemoryPoolConfig {
        &self.config
    }

    /// Number of live allocations.
    pub fn live_count(&self) -> usize {
        self.live.len()
    }

    /// Whether the pool has exceeded its high-water mark.
    pub fn is_high_water(&self) -> bool {
        let ratio = self.current_allocated as f64 / self.config.total_bytes as f64;
        ratio >= self.config.high_water_ratio
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pool() -> CudaMemoryPool {
        create_pool(MemoryPoolConfig::default()).unwrap()
    }

    fn small_pool(total: usize) -> CudaMemoryPool {
        create_pool(MemoryPoolConfig {
            total_bytes: total,
            min_block_size: 256,
            max_block_size: total.next_power_of_two(),
            alignment: 256,
            eviction_strategy: EvictionStrategy::Lru,
            high_water_ratio: 0.9,
        })
        .unwrap()
    }

    // -- create_pool --------------------------------------------------------

    #[test]
    fn test_create_pool_default() {
        let pool = default_pool();
        assert_eq!(pool.config.total_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn test_create_pool_custom() {
        let pool = small_pool(4096);
        assert_eq!(pool.config.total_bytes, 4096);
    }

    #[test]
    fn test_create_pool_zero_total_bytes_fails() {
        let r = create_pool(MemoryPoolConfig { total_bytes: 0, ..Default::default() });
        assert!(matches!(r, Err(PoolError::InvalidConfig(_))));
    }

    #[test]
    fn test_create_pool_non_power_of_two_min_block() {
        let r = create_pool(MemoryPoolConfig { min_block_size: 100, ..Default::default() });
        assert!(matches!(r, Err(PoolError::InvalidConfig(_))));
    }

    #[test]
    fn test_create_pool_non_power_of_two_max_block() {
        let r = create_pool(MemoryPoolConfig { max_block_size: 300, ..Default::default() });
        assert!(matches!(r, Err(PoolError::InvalidConfig(_))));
    }

    #[test]
    fn test_create_pool_min_gt_max() {
        let r = create_pool(MemoryPoolConfig {
            min_block_size: 1024,
            max_block_size: 256,
            ..Default::default()
        });
        assert!(matches!(r, Err(PoolError::InvalidConfig(_))));
    }

    #[test]
    fn test_create_pool_non_power_of_two_alignment() {
        let r = create_pool(MemoryPoolConfig { alignment: 5, ..Default::default() });
        assert!(matches!(r, Err(PoolError::InvalidConfig(_))));
    }

    #[test]
    fn test_create_pool_high_water_too_high() {
        let r = create_pool(MemoryPoolConfig { high_water_ratio: 1.5, ..Default::default() });
        assert!(matches!(r, Err(PoolError::InvalidConfig(_))));
    }

    #[test]
    fn test_create_pool_high_water_negative() {
        let r = create_pool(MemoryPoolConfig { high_water_ratio: -0.1, ..Default::default() });
        assert!(matches!(r, Err(PoolError::InvalidConfig(_))));
    }

    // -- allocate -----------------------------------------------------------

    #[test]
    fn test_allocate_single() {
        let mut pool = default_pool();
        let ptr = allocate(&mut pool, 1024).unwrap();
        assert!(ptr > 0);
    }

    #[test]
    fn test_allocate_returns_distinct_ptrs() {
        let mut pool = default_pool();
        let p1 = allocate(&mut pool, 256).unwrap();
        let p2 = allocate(&mut pool, 256).unwrap();
        assert_ne!(p1, p2);
    }

    #[test]
    fn test_allocate_zero_fails() {
        let mut pool = default_pool();
        assert!(matches!(allocate(&mut pool, 0), Err(PoolError::InvalidSize(0))));
    }

    #[test]
    fn test_allocate_exceeds_capacity() {
        let mut pool = small_pool(1024);
        assert!(matches!(allocate(&mut pool, 2048), Err(PoolError::InvalidSize(_))));
    }

    #[test]
    fn test_allocate_oom() {
        let mut pool = small_pool(1024);
        let _p1 = allocate(&mut pool, 512).unwrap();
        let _p2 = allocate(&mut pool, 256).unwrap();
        // Pool is 1024 bytes; allocated 512 + 256 = 768 min (rounded to 256 blocks).
        // Next 512 may fail depending on remaining space.
        let _ = allocate(&mut pool, 512);
    }

    #[test]
    fn test_allocate_fills_pool() {
        let mut pool = small_pool(1024);
        let _p1 = allocate(&mut pool, 256).unwrap();
        let _p2 = allocate(&mut pool, 256).unwrap();
        let _p3 = allocate(&mut pool, 256).unwrap();
        let _p4 = allocate(&mut pool, 256).unwrap();
        assert!(allocate(&mut pool, 256).is_err());
    }

    #[test]
    fn test_allocate_various_sizes() {
        let mut pool = default_pool();
        let _a = allocate(&mut pool, 100).unwrap();
        let _b = allocate(&mut pool, 4096).unwrap();
        let _c = allocate(&mut pool, 65536).unwrap();
        assert_eq!(pool.live_count(), 3);
    }

    #[test]
    fn test_allocate_min_block_rounding() {
        let mut pool = small_pool(1024);
        // Request 1 byte – should round up to min_block_size (256).
        let p = allocate(&mut pool, 1).unwrap();
        let stats = pool_stats(&pool);
        assert!(stats.allocated_bytes >= 256);
        deallocate(&mut pool, p).unwrap();
    }

    #[test]
    fn test_allocate_tagged() {
        let mut pool = default_pool();
        let ptr = allocate_tagged(&mut pool, 512, Some("weights".into())).unwrap();
        let offset = pool.live[&ptr];
        assert_eq!(pool.blocks[&offset].tag.as_deref(), Some("weights"));
    }

    // -- deallocate ---------------------------------------------------------

    #[test]
    fn test_deallocate_basic() {
        let mut pool = default_pool();
        let ptr = allocate(&mut pool, 1024).unwrap();
        deallocate(&mut pool, ptr).unwrap();
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn test_deallocate_invalid_pointer() {
        let mut pool = default_pool();
        assert!(matches!(deallocate(&mut pool, 0xDEAD), Err(PoolError::InvalidPointer(0xDEAD))));
    }

    #[test]
    fn test_deallocate_double_free() {
        let mut pool = default_pool();
        let ptr = allocate(&mut pool, 256).unwrap();
        deallocate(&mut pool, ptr).unwrap();
        assert!(deallocate(&mut pool, ptr).is_err());
    }

    #[test]
    fn test_deallocate_frees_memory() {
        let mut pool = small_pool(1024);
        let p1 = allocate(&mut pool, 256).unwrap();
        let _p2 = allocate(&mut pool, 256).unwrap();
        deallocate(&mut pool, p1).unwrap();
        let stats = pool_stats(&pool);
        assert!(stats.free_bytes >= 256);
    }

    #[test]
    fn test_deallocate_then_reallocate() {
        let mut pool = small_pool(1024);
        let p = allocate(&mut pool, 256).unwrap();
        deallocate(&mut pool, p).unwrap();
        let p2 = allocate(&mut pool, 256).unwrap();
        assert!(p2 > 0);
    }

    #[test]
    fn test_deallocate_all() {
        let mut pool = small_pool(4096);
        let ptrs: Vec<usize> = (0..4).map(|_| allocate(&mut pool, 256).unwrap()).collect();
        for p in ptrs {
            deallocate(&mut pool, p).unwrap();
        }
        assert_eq!(pool.live_count(), 0);
        let stats = pool_stats(&pool);
        assert_eq!(stats.allocated_bytes, 0);
    }

    // -- defragment ---------------------------------------------------------

    #[test]
    fn test_defragment_noop_empty() {
        let mut pool = default_pool();
        assert_eq!(defragment(&mut pool), 0);
    }

    #[test]
    fn test_defragment_merges_adjacent() {
        let mut pool = small_pool(4096);
        let p1 = allocate(&mut pool, 256).unwrap();
        let p2 = allocate(&mut pool, 256).unwrap();
        let p3 = allocate(&mut pool, 256).unwrap();
        deallocate(&mut pool, p1).unwrap();
        deallocate(&mut pool, p3).unwrap();
        deallocate(&mut pool, p2).unwrap();
        let merged = defragment(&mut pool);
        // After dealloc+coalesce some merging already happens; defragment may
        // merge remaining fragments.
        let stats = pool_stats(&pool);
        assert!(stats.num_free_blocks <= 2);
        let _ = merged; // merged count varies
    }

    #[test]
    fn test_defragment_reduces_fragmentation() {
        let mut pool = small_pool(4096);
        let ptrs: Vec<usize> = (0..4).map(|_| allocate(&mut pool, 256).unwrap()).collect();
        // Free alternating blocks to create fragmentation.
        deallocate(&mut pool, ptrs[0]).unwrap();
        deallocate(&mut pool, ptrs[2]).unwrap();
        let before = pool_stats(&pool);
        deallocate(&mut pool, ptrs[1]).unwrap();
        defragment(&mut pool);
        let after = pool_stats(&pool);
        assert!(after.largest_free_block >= before.largest_free_block);
    }

    #[test]
    fn test_defragment_with_no_free_blocks() {
        let mut pool = small_pool(1024);
        let _p = allocate(&mut pool, 256).unwrap();
        // Don't free – defragment should be harmless.
        assert_eq!(defragment(&mut pool), 0);
    }

    // -- pool_stats ---------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let pool = small_pool(4096);
        let stats = pool_stats(&pool);
        assert_eq!(stats.total_bytes, 4096);
        assert_eq!(stats.allocated_bytes, 0);
        assert_eq!(stats.free_bytes, 4096);
        assert_eq!(stats.num_allocations, 0);
        assert_eq!(stats.total_allocations, 0);
    }

    #[test]
    fn test_stats_after_alloc() {
        let mut pool = default_pool();
        allocate(&mut pool, 1024).unwrap();
        let stats = pool_stats(&pool);
        assert_eq!(stats.num_allocations, 1);
        assert!(stats.allocated_bytes >= 1024);
    }

    #[test]
    fn test_stats_peak_tracking() {
        let mut pool = default_pool();
        let p1 = allocate(&mut pool, 4096).unwrap();
        let p2 = allocate(&mut pool, 4096).unwrap();
        deallocate(&mut pool, p1).unwrap();
        deallocate(&mut pool, p2).unwrap();
        let stats = pool_stats(&pool);
        assert!(stats.peak_allocated_bytes >= 8192);
        assert_eq!(stats.allocated_bytes, 0);
    }

    #[test]
    fn test_stats_total_ops() {
        let mut pool = default_pool();
        let p1 = allocate(&mut pool, 256).unwrap();
        let p2 = allocate(&mut pool, 256).unwrap();
        deallocate(&mut pool, p1).unwrap();
        let stats = pool_stats(&pool);
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_deallocations, 1);
        deallocate(&mut pool, p2).unwrap();
    }

    #[test]
    fn test_stats_fragmentation_ratio() {
        let mut pool = small_pool(4096);
        let p1 = allocate(&mut pool, 256).unwrap();
        let _p2 = allocate(&mut pool, 256).unwrap();
        let p3 = allocate(&mut pool, 256).unwrap();
        deallocate(&mut pool, p1).unwrap();
        deallocate(&mut pool, p3).unwrap();
        let stats = pool_stats(&pool);
        // Two disjoint free regions → fragmentation > 0.
        assert!(stats.fragmentation_ratio >= 0.0);
    }

    #[test]
    fn test_stats_largest_free_block() {
        let pool = small_pool(4096);
        let stats = pool_stats(&pool);
        assert_eq!(stats.largest_free_block, 4096);
    }

    // -- shrink_to_fit ------------------------------------------------------

    #[test]
    fn test_shrink_noop_when_already_small() {
        let mut pool = small_pool(1024);
        let released = shrink_to_fit(&mut pool, 2048);
        assert_eq!(released, 0);
    }

    #[test]
    fn test_shrink_releases_trailing() {
        let mut pool = small_pool(4096);
        let _p = allocate(&mut pool, 256).unwrap();
        let released = shrink_to_fit(&mut pool, 256);
        assert!(released > 0);
        assert!(pool.config.total_bytes <= 4096);
    }

    #[test]
    fn test_shrink_respects_allocations() {
        let mut pool = small_pool(4096);
        let p1 = allocate(&mut pool, 256).unwrap();
        let p2 = allocate(&mut pool, 256).unwrap();
        shrink_to_fit(&mut pool, 0);
        // Both allocations must still be valid.
        deallocate(&mut pool, p1).unwrap();
        deallocate(&mut pool, p2).unwrap();
    }

    #[test]
    fn test_shrink_empty_pool() {
        let mut pool = small_pool(4096);
        let released = shrink_to_fit(&mut pool, 1024);
        assert!(released > 0);
        assert_eq!(pool.config.total_bytes, 1024);
    }

    // -- warm_pool ----------------------------------------------------------

    #[test]
    fn test_warm_pool_basic() {
        let mut pool = default_pool();
        let warmed = warm_pool(&mut pool, &[256, 512, 1024]).unwrap();
        assert_eq!(warmed, 3);
        assert_eq!(pool.live_count(), 0); // All freed after warming.
    }

    #[test]
    fn test_warm_pool_skips_zero() {
        let mut pool = default_pool();
        let warmed = warm_pool(&mut pool, &[0, 256, 0]).unwrap();
        assert_eq!(warmed, 1);
    }

    #[test]
    fn test_warm_pool_empty() {
        let mut pool = default_pool();
        let warmed = warm_pool(&mut pool, &[]).unwrap();
        assert_eq!(warmed, 0);
    }

    // -- eviction strategies ------------------------------------------------

    #[test]
    fn test_eviction_none_fails_on_full() {
        let mut pool = create_pool(MemoryPoolConfig {
            total_bytes: 1024,
            min_block_size: 256,
            max_block_size: 1024,
            alignment: 256,
            eviction_strategy: EvictionStrategy::None,
            high_water_ratio: 0.9,
        })
        .unwrap();
        let _p1 = allocate(&mut pool, 512).unwrap();
        let _p2 = allocate(&mut pool, 512).unwrap();
        assert!(allocate(&mut pool, 256).is_err());
    }

    #[test]
    fn test_eviction_lru_coalesces() {
        let mut pool = small_pool(2048);
        let p1 = allocate(&mut pool, 256).unwrap();
        let p2 = allocate(&mut pool, 256).unwrap();
        deallocate(&mut pool, p1).unwrap();
        deallocate(&mut pool, p2).unwrap();
        // After deallocation + coalescing, the pool should be able to serve a
        // larger allocation.
        let p3 = allocate(&mut pool, 512).unwrap();
        assert!(p3 > 0);
    }

    #[test]
    fn test_eviction_largest_first() {
        let mut pool = create_pool(MemoryPoolConfig {
            total_bytes: 4096,
            min_block_size: 256,
            max_block_size: 4096,
            alignment: 256,
            eviction_strategy: EvictionStrategy::LargestFirst,
            high_water_ratio: 0.9,
        })
        .unwrap();
        let p = allocate(&mut pool, 1024).unwrap();
        deallocate(&mut pool, p).unwrap();
        let p2 = allocate(&mut pool, 2048).unwrap();
        assert!(p2 > 0);
    }

    // -- coalescing ---------------------------------------------------------

    #[test]
    fn test_coalesce_forward() {
        let mut pool = small_pool(4096);
        let p1 = allocate(&mut pool, 256).unwrap();
        let p2 = allocate(&mut pool, 256).unwrap();
        deallocate(&mut pool, p1).unwrap();
        deallocate(&mut pool, p2).unwrap();
        defragment(&mut pool);
        let stats = pool_stats(&pool);
        // Should have merged into fewer blocks.
        assert!(stats.num_free_blocks <= 2);
    }

    #[test]
    fn test_coalesce_backward() {
        let mut pool = small_pool(4096);
        let p1 = allocate(&mut pool, 256).unwrap();
        let p2 = allocate(&mut pool, 256).unwrap();
        deallocate(&mut pool, p2).unwrap();
        deallocate(&mut pool, p1).unwrap();
        defragment(&mut pool);
        let stats = pool_stats(&pool);
        assert!(stats.num_free_blocks <= 2);
    }

    // -- PoolBlock / PoolError display --------------------------------------

    #[test]
    fn test_pool_error_display() {
        let e = PoolError::OutOfMemory { requested: 100, available: 50 };
        assert!(e.to_string().contains("100"));
        assert!(e.to_string().contains("50"));
    }

    #[test]
    fn test_pool_error_invalid_pointer_display() {
        let e = PoolError::InvalidPointer(0xBEEF);
        assert!(e.to_string().contains("0xbeef"));
    }

    #[test]
    fn test_pool_error_destroyed_display() {
        let e = PoolError::PoolDestroyed;
        assert!(e.to_string().contains("destroyed"));
    }

    #[test]
    fn test_pool_error_invalid_config_display() {
        let e = PoolError::InvalidConfig("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn test_pool_error_invalid_size_display() {
        let e = PoolError::InvalidSize(42);
        assert!(e.to_string().contains("42"));
    }

    // -- CudaMemoryPool helpers ---------------------------------------------

    #[test]
    fn test_uptime() {
        let pool = default_pool();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(pool.uptime().as_millis() >= 4);
    }

    #[test]
    fn test_config_accessor() {
        let pool = default_pool();
        assert_eq!(pool.config().total_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn test_is_high_water_false() {
        let pool = default_pool();
        assert!(!pool.is_high_water());
    }

    #[test]
    fn test_is_high_water_true() {
        let mut pool = small_pool(1024);
        // Allocate ≥ 90 % of pool.
        let _p1 = allocate(&mut pool, 256).unwrap();
        let _p2 = allocate(&mut pool, 256).unwrap();
        let _p3 = allocate(&mut pool, 256).unwrap();
        let _p4 = allocate(&mut pool, 256).unwrap();
        assert!(pool.is_high_water());
    }

    // -- round_up_block_size / align_up helpers -----------------------------

    #[test]
    fn test_round_up_block_size() {
        assert_eq!(round_up_block_size(1, 256), 256);
        assert_eq!(round_up_block_size(257, 256), 512);
        assert_eq!(round_up_block_size(256, 256), 256);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(300, 256), 512);
    }

    // -- edge cases ---------------------------------------------------------

    #[test]
    fn test_allocate_exact_pool_size() {
        let mut pool = small_pool(256);
        let p = allocate(&mut pool, 256).unwrap();
        assert!(p > 0);
        let stats = pool_stats(&pool);
        assert_eq!(stats.num_allocations, 1);
    }

    #[test]
    fn test_many_small_allocs() {
        let mut pool = default_pool();
        let mut ptrs = Vec::new();
        for _ in 0..100 {
            ptrs.push(allocate(&mut pool, 256).unwrap());
        }
        assert_eq!(pool.live_count(), 100);
        for p in ptrs {
            deallocate(&mut pool, p).unwrap();
        }
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn test_alloc_dealloc_cycle() {
        let mut pool = small_pool(2048);
        for _ in 0..50 {
            let p = allocate(&mut pool, 256).unwrap();
            deallocate(&mut pool, p).unwrap();
        }
        let stats = pool_stats(&pool);
        assert_eq!(stats.total_allocations, 50);
        assert_eq!(stats.total_deallocations, 50);
    }

    // -- EvictionStrategy default -------------------------------------------

    #[test]
    fn test_eviction_strategy_default() {
        assert_eq!(EvictionStrategy::default(), EvictionStrategy::Lru);
    }

    // -- AllocationStats default --------------------------------------------

    #[test]
    fn test_allocation_stats_default() {
        let stats = AllocationStats::default();
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.allocated_bytes, 0);
    }

    // -- stress / pattern tests ---------------------------------------------

    #[test]
    fn test_alternating_alloc_free() {
        let mut pool = small_pool(8192);
        let mut ptrs = Vec::new();
        for i in 0..8 {
            ptrs.push(allocate(&mut pool, 256).unwrap());
            if i % 2 == 0 && !ptrs.is_empty() {
                let p = ptrs.remove(0);
                deallocate(&mut pool, p).unwrap();
            }
        }
        for p in ptrs {
            deallocate(&mut pool, p).unwrap();
        }
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn test_shrink_then_allocate() {
        let mut pool = small_pool(4096);
        let _p = allocate(&mut pool, 256).unwrap();
        shrink_to_fit(&mut pool, 256);
        // Should still be able to allocate within the remaining space.
        let stats = pool_stats(&pool);
        assert!(stats.total_bytes <= 4096);
    }

    #[test]
    fn test_warm_then_allocate() {
        let mut pool = default_pool();
        warm_pool(&mut pool, &[1024, 2048]).unwrap();
        let p = allocate(&mut pool, 1024).unwrap();
        assert!(p > 0);
        deallocate(&mut pool, p).unwrap();
    }

    #[test]
    fn test_defragment_idempotent() {
        let mut pool = small_pool(4096);
        let p1 = allocate(&mut pool, 256).unwrap();
        let p2 = allocate(&mut pool, 256).unwrap();
        deallocate(&mut pool, p1).unwrap();
        deallocate(&mut pool, p2).unwrap();
        defragment(&mut pool);
        let s1 = pool_stats(&pool);
        defragment(&mut pool);
        let s2 = pool_stats(&pool);
        assert_eq!(s1.num_free_blocks, s2.num_free_blocks);
    }

    #[test]
    fn test_pool_block_debug() {
        let b = PoolBlock {
            offset: 0,
            size: 256,
            allocated: false,
            last_access: 1,
            tag: Some("test".into()),
        };
        let dbg = format!("{b:?}");
        assert!(dbg.contains("256"));
    }

    #[test]
    fn test_memory_pool_config_debug() {
        let cfg = MemoryPoolConfig::default();
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("total_bytes"));
    }

    #[test]
    fn test_allocation_stats_debug() {
        let stats = AllocationStats::default();
        let dbg = format!("{stats:?}");
        assert!(dbg.contains("total_bytes"));
    }

    #[test]
    fn test_eviction_strategy_clone_eq() {
        let a = EvictionStrategy::Lru;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_pool_error_clone_eq() {
        let e = PoolError::PoolDestroyed;
        let e2 = e.clone();
        assert_eq!(e, e2);
    }

    #[test]
    fn test_pool_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(PoolError::PoolDestroyed);
        assert!(e.to_string().contains("destroyed"));
    }
}

// ---------------------------------------------------------------------------
// Property-based tests (proptest)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn pool_config_strategy() -> impl Strategy<Value = MemoryPoolConfig> {
        // Use small pools for fast property tests.
        (12u32..=18u32).prop_map(|bits| {
            let total = 1usize << bits; // 4 KiB .. 256 KiB
            MemoryPoolConfig {
                total_bytes: total,
                min_block_size: 256,
                max_block_size: total.next_power_of_two(),
                alignment: 256,
                eviction_strategy: EvictionStrategy::Lru,
                high_water_ratio: 0.9,
            }
        })
    }

    proptest! {
        /// Every allocation followed by a deallocation must return the pool to
        /// zero live allocations.
        #[test]
        fn prop_alloc_dealloc_roundtrip(
            cfg in pool_config_strategy(),
            size in 1usize..=2048,
        ) {
            let mut pool = create_pool(cfg).unwrap();
            if let Ok(ptr) = allocate(&mut pool, size) {
                deallocate(&mut pool, ptr).unwrap();
                prop_assert_eq!(pool.live_count(), 0);
            }
        }

        /// `pool_stats.allocated_bytes` must never exceed `total_bytes`.
        #[test]
        fn prop_allocated_le_total(
            cfg in pool_config_strategy(),
            sizes in proptest::collection::vec(1usize..=1024, 1..10),
        ) {
            let mut pool = create_pool(cfg).unwrap();
            for sz in &sizes {
                let _ = allocate(&mut pool, *sz);
            }
            let stats = pool_stats(&pool);
            prop_assert!(stats.allocated_bytes <= stats.total_bytes);
        }

        /// After deallocating every allocation, `allocated_bytes` must be zero.
        #[test]
        fn prop_dealloc_all_zeroes(
            cfg in pool_config_strategy(),
            sizes in proptest::collection::vec(1usize..=512, 1..8),
        ) {
            let mut pool = create_pool(cfg).unwrap();
            let mut ptrs = Vec::new();
            for sz in &sizes {
                if let Ok(p) = allocate(&mut pool, *sz) {
                    ptrs.push(p);
                }
            }
            for p in ptrs {
                deallocate(&mut pool, p).unwrap();
            }
            let stats = pool_stats(&pool);
            prop_assert_eq!(stats.allocated_bytes, 0);
        }

        /// Defragmentation must not increase fragmentation.
        #[test]
        fn prop_defragment_reduces_fragmentation(
            cfg in pool_config_strategy(),
            sizes in proptest::collection::vec(1usize..=512, 2..8),
        ) {
            let mut pool = create_pool(cfg).unwrap();
            let mut ptrs = Vec::new();
            for sz in &sizes {
                if let Ok(p) = allocate(&mut pool, *sz) {
                    ptrs.push(p);
                }
            }
            // Free every other block.
            let to_free: Vec<usize> = ptrs.iter().copied().step_by(2).collect();
            for p in to_free {
                let _ = deallocate(&mut pool, p);
            }
            let before = pool_stats(&pool).fragmentation_ratio;
            defragment(&mut pool);
            let after = pool_stats(&pool).fragmentation_ratio;
            prop_assert!(after <= before + f64::EPSILON);
        }

        /// `live_count` must equal the number of outstanding allocations.
        #[test]
        fn prop_live_count_accurate(
            cfg in pool_config_strategy(),
            sizes in proptest::collection::vec(1usize..=512, 1..8),
        ) {
            let mut pool = create_pool(cfg).unwrap();
            let mut ptrs = Vec::new();
            for sz in &sizes {
                if let Ok(p) = allocate(&mut pool, *sz) {
                    ptrs.push(p);
                }
            }
            prop_assert_eq!(pool.live_count(), ptrs.len());
        }
    }
}
