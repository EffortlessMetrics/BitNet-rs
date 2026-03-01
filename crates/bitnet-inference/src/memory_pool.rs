//! # Memory Pool Management
//!
//! Efficient tensor allocation via arena-style and slab-based memory pools.
//!
//! - **Arena allocation**: fast bump-pointer allocation with bulk deallocation
//!   for short-lived tensor scratch buffers.
//! - **Slab allocation**: fixed-size block pools for uniform objects like KV
//!   cache entries.
//! - **Thread-safe**: all operations go through `Arc<Mutex<…>>`.
//! - **Configurable**: [`PoolConfig`] builder sets initial size, growth
//!   strategy, and hard limits.

use std::sync::{Arc, Mutex};

// ── Configuration ───────────────────────────────────────────────────

/// Strategy used when an arena chunk runs out of space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthStrategy {
    /// Each new chunk is the same size as the initial chunk.
    Fixed,
    /// Each new chunk is double the previous one (capped at `max_pool_size`).
    Double,
}

/// Builder for [`MemoryPool`] configuration.
///
/// # Example
/// ```
/// # use bitnet_inference::memory_pool::{PoolConfig, GrowthStrategy};
/// let cfg = PoolConfig::builder()
///     .initial_size(1 << 20)
///     .growth_strategy(GrowthStrategy::Double)
///     .max_pool_size(64 << 20)
///     .max_allocation_size(4 << 20)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Size in bytes of the first arena chunk.
    pub initial_size: usize,
    /// How subsequent chunks grow.
    pub growth_strategy: GrowthStrategy,
    /// Hard cap on total memory the pool may hold.
    pub max_pool_size: usize,
    /// Largest single allocation the pool will accept.
    pub max_allocation_size: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 1 << 20, // 1 MiB
            growth_strategy: GrowthStrategy::Double,
            max_pool_size: 256 << 20,      // 256 MiB
            max_allocation_size: 64 << 20, // 64 MiB
        }
    }
}

impl PoolConfig {
    /// Start building a configuration.
    pub fn builder() -> PoolConfigBuilder {
        PoolConfigBuilder(Self::default())
    }
}

/// Fluent builder for [`PoolConfig`].
pub struct PoolConfigBuilder(PoolConfig);

impl PoolConfigBuilder {
    pub fn initial_size(mut self, bytes: usize) -> Self {
        self.0.initial_size = bytes;
        self
    }

    pub fn growth_strategy(mut self, strategy: GrowthStrategy) -> Self {
        self.0.growth_strategy = strategy;
        self
    }

    pub fn max_pool_size(mut self, bytes: usize) -> Self {
        self.0.max_pool_size = bytes;
        self
    }

    pub fn max_allocation_size(mut self, bytes: usize) -> Self {
        self.0.max_allocation_size = bytes;
        self
    }

    pub fn build(self) -> PoolConfig {
        self.0
    }
}

// ── Statistics ──────────────────────────────────────────────────────

/// Snapshot of pool memory usage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PoolStatistics {
    /// Total bytes handed out across all arenas.
    pub bytes_allocated: usize,
    /// Bytes reclaimed via arena resets.
    pub bytes_freed: usize,
    /// Peak `bytes_allocated` observed.
    pub peak_usage: usize,
    /// Number of individual allocation requests served.
    pub allocation_count: u64,
    /// Number of arena reset operations.
    pub reset_count: u64,
    /// Number of slab block checkouts.
    pub slab_checkouts: u64,
    /// Number of slab block returns.
    pub slab_returns: u64,
    /// Current total memory owned by the pool (chunks + slabs).
    pub total_pool_bytes: usize,
}

// ── Arena internals ─────────────────────────────────────────────────

/// A single contiguous chunk of arena memory.
struct ArenaChunk {
    storage: Vec<u8>,
    /// Byte offset of next free position.
    cursor: usize,
}

impl ArenaChunk {
    fn new(capacity: usize) -> Self {
        Self { storage: vec![0u8; capacity], cursor: 0 }
    }

    /// Try to bump-allocate `size` bytes (8-byte aligned).
    /// Returns the byte offset into `storage` on success.
    fn try_alloc(&mut self, size: usize) -> Option<usize> {
        let aligned = (self.cursor + 7) & !7;
        let end = aligned + size;
        if end <= self.storage.len() {
            self.cursor = end;
            // Zero the region before handing it out.
            self.storage[aligned..end].fill(0);
            Some(aligned)
        } else {
            None
        }
    }

    /// Reset the cursor without freeing the backing memory.
    fn reset(&mut self) {
        self.cursor = 0;
    }

    fn used(&self) -> usize {
        self.cursor
    }
}

// ── Slab internals ──────────────────────────────────────────────────

/// Fixed-size block pool for uniform allocations.
struct Slab {
    block_size: usize,
    /// Blocks currently available for checkout.
    free_blocks: Vec<Vec<u8>>,
    /// Number of blocks currently checked out.
    outstanding: usize,
}

impl Slab {
    fn new(block_size: usize, initial_count: usize) -> Self {
        let free_blocks = (0..initial_count).map(|_| vec![0u8; block_size]).collect();
        Self { block_size, free_blocks, outstanding: 0 }
    }

    fn checkout(&mut self) -> Vec<u8> {
        self.outstanding += 1;
        self.free_blocks.pop().unwrap_or_else(|| vec![0u8; self.block_size])
    }

    fn checkin(&mut self, mut block: Vec<u8>) {
        debug_assert_eq!(block.len(), self.block_size);
        block.fill(0);
        self.outstanding = self.outstanding.saturating_sub(1);
        self.free_blocks.push(block);
    }

    fn total_bytes(&self) -> usize {
        (self.free_blocks.len() + self.outstanding) * self.block_size
    }
}

// ── Pool inner state ────────────────────────────────────────────────

struct PoolInner {
    config: PoolConfig,
    chunks: Vec<ArenaChunk>,
    /// Next chunk capacity (used by `GrowthStrategy`).
    next_chunk_size: usize,
    slabs: Vec<Slab>,
    stats: PoolStatistics,
}

impl PoolInner {
    fn new(config: PoolConfig) -> Self {
        let initial = config.initial_size.max(64);
        let chunk = ArenaChunk::new(initial);
        let total = initial;
        Self {
            next_chunk_size: initial,
            config,
            chunks: vec![chunk],
            slabs: Vec::new(),
            stats: PoolStatistics { total_pool_bytes: total, ..Default::default() },
        }
    }

    /// Grow the arena by one chunk, respecting `max_pool_size`.
    fn grow(&mut self) -> bool {
        let new_size = match self.config.growth_strategy {
            GrowthStrategy::Fixed => self.config.initial_size.max(64),
            GrowthStrategy::Double => {
                (self.next_chunk_size.saturating_mul(2)).min(self.config.max_pool_size)
            }
        };
        if self.stats.total_pool_bytes.saturating_add(new_size) > self.config.max_pool_size {
            return false;
        }
        self.chunks.push(ArenaChunk::new(new_size));
        self.stats.total_pool_bytes += new_size;
        self.next_chunk_size = new_size;
        true
    }

    fn update_peak(&mut self) {
        if self.stats.bytes_allocated > self.stats.peak_usage {
            self.stats.peak_usage = self.stats.bytes_allocated;
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────

/// Thread-safe memory pool with arena and slab allocators.
///
/// Create via [`MemoryPool::new`] or [`MemoryPool::with_config`].
#[derive(Clone)]
pub struct MemoryPool {
    inner: Arc<Mutex<PoolInner>>,
}

/// Handle to an arena allocation. The bytes are valid until the next
/// [`MemoryPool::arena_reset`].
///
/// This is a lightweight index into pool-owned memory; it does **not** own
/// the backing storage.
#[derive(Debug, Clone, Copy)]
pub struct ArenaAlloc {
    /// Index of the chunk within the pool.
    chunk_idx: usize,
    /// Byte offset inside that chunk.
    offset: usize,
    /// Length in bytes.
    len: usize,
}

impl ArenaAlloc {
    /// Number of bytes in this allocation.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the allocation has zero length.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Errors returned by pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// The requested allocation exceeds `max_allocation_size`.
    AllocationTooLarge { requested: usize, limit: usize },
    /// The pool has reached `max_pool_size` and cannot grow.
    PoolExhausted { requested: usize, pool_max: usize },
    /// No slab registered for the given block size.
    SlabNotFound { block_size: usize },
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllocationTooLarge { requested, limit } => {
                write!(f, "allocation of {requested} bytes exceeds limit of {limit}")
            }
            Self::PoolExhausted { requested, pool_max } => {
                write!(f, "pool exhausted: cannot allocate {requested} bytes (max {pool_max})")
            }
            Self::SlabNotFound { block_size } => {
                write!(f, "no slab registered for block size {block_size}")
            }
        }
    }
}

impl std::error::Error for PoolError {}

impl MemoryPool {
    /// Create a pool with default configuration.
    pub fn new() -> Self {
        Self::with_config(PoolConfig::default())
    }

    /// Create a pool with explicit configuration.
    pub fn with_config(config: PoolConfig) -> Self {
        Self { inner: Arc::new(Mutex::new(PoolInner::new(config))) }
    }

    // ── Arena operations ────────────────────────────────────────────

    /// Allocate `size` bytes from the arena (8-byte aligned, zeroed).
    ///
    /// Returns an [`ArenaAlloc`] handle that can be used with
    /// [`read_arena`](Self::read_arena) / [`write_arena`](Self::write_arena).
    pub fn arena_alloc(&self, size: usize) -> Result<ArenaAlloc, PoolError> {
        if size == 0 {
            return Ok(ArenaAlloc { chunk_idx: 0, offset: 0, len: 0 });
        }
        let mut inner = self.inner.lock().expect("pool lock poisoned");
        if size > inner.config.max_allocation_size {
            return Err(PoolError::AllocationTooLarge {
                requested: size,
                limit: inner.config.max_allocation_size,
            });
        }

        // Try existing chunks (most-recent first for locality).
        for (idx, chunk) in inner.chunks.iter_mut().enumerate().rev() {
            if let Some(offset) = chunk.try_alloc(size) {
                inner.stats.bytes_allocated += size;
                inner.stats.allocation_count += 1;
                inner.update_peak();
                return Ok(ArenaAlloc { chunk_idx: idx, offset, len: size });
            }
        }

        // Need a new chunk.
        if !inner.grow() {
            return Err(PoolError::PoolExhausted {
                requested: size,
                pool_max: inner.config.max_pool_size,
            });
        }
        let idx = inner.chunks.len() - 1;
        let offset = inner.chunks[idx]
            .try_alloc(size)
            .expect("freshly allocated chunk too small for request");
        inner.stats.bytes_allocated += size;
        inner.stats.allocation_count += 1;
        inner.update_peak();
        Ok(ArenaAlloc { chunk_idx: idx, offset, len: size })
    }

    /// Read bytes from an arena allocation.
    pub fn read_arena(&self, alloc: &ArenaAlloc) -> Vec<u8> {
        if alloc.len == 0 {
            return Vec::new();
        }
        let inner = self.inner.lock().expect("pool lock poisoned");
        let chunk = &inner.chunks[alloc.chunk_idx];
        chunk.storage[alloc.offset..alloc.offset + alloc.len].to_vec()
    }

    /// Write bytes into an arena allocation.
    ///
    /// # Panics
    /// Panics if `data.len() != alloc.len()`.
    pub fn write_arena(&self, alloc: &ArenaAlloc, data: &[u8]) {
        assert_eq!(data.len(), alloc.len, "data length must match allocation length");
        if alloc.len == 0 {
            return;
        }
        let mut inner = self.inner.lock().expect("pool lock poisoned");
        let chunk = &mut inner.chunks[alloc.chunk_idx];
        chunk.storage[alloc.offset..alloc.offset + alloc.len].copy_from_slice(data);
    }

    /// Reset all arena chunks (bulk deallocation). Existing [`ArenaAlloc`]
    /// handles become logically invalid.
    pub fn arena_reset(&self) {
        let mut inner = self.inner.lock().expect("pool lock poisoned");
        let freed: usize = inner.chunks.iter().map(|c| c.used()).sum();
        for chunk in &mut inner.chunks {
            chunk.reset();
        }
        inner.stats.bytes_freed += freed;
        inner.stats.bytes_allocated = inner.stats.bytes_allocated.saturating_sub(freed);
        inner.stats.reset_count += 1;
    }

    // ── Slab operations ─────────────────────────────────────────────

    /// Register a slab for blocks of `block_size` bytes, pre-allocating
    /// `initial_count` blocks.
    pub fn register_slab(&self, block_size: usize, initial_count: usize) {
        let mut inner = self.inner.lock().expect("pool lock poisoned");
        // Avoid duplicates.
        if inner.slabs.iter().any(|s| s.block_size == block_size) {
            return;
        }
        let slab = Slab::new(block_size, initial_count);
        inner.stats.total_pool_bytes += slab.total_bytes();
        inner.slabs.push(slab);
    }

    /// Check out a zeroed block from the slab of `block_size`.
    pub fn slab_checkout(&self, block_size: usize) -> Result<Vec<u8>, PoolError> {
        let mut inner = self.inner.lock().expect("pool lock poisoned");
        let slab = inner
            .slabs
            .iter_mut()
            .find(|s| s.block_size == block_size)
            .ok_or(PoolError::SlabNotFound { block_size })?;
        let old_total = slab.total_bytes();
        let block = slab.checkout();
        let new_total = slab.total_bytes();
        // A new block may have been allocated.
        inner.stats.total_pool_bytes = inner.stats.total_pool_bytes + new_total - old_total;
        inner.stats.bytes_allocated += block_size;
        inner.stats.slab_checkouts += 1;
        inner.update_peak();
        Ok(block)
    }

    /// Return a block to the slab of `block_size`.
    pub fn slab_checkin(&self, block_size: usize, block: Vec<u8>) -> Result<(), PoolError> {
        let mut inner = self.inner.lock().expect("pool lock poisoned");
        let slab = inner
            .slabs
            .iter_mut()
            .find(|s| s.block_size == block_size)
            .ok_or(PoolError::SlabNotFound { block_size })?;
        slab.checkin(block);
        inner.stats.bytes_allocated = inner.stats.bytes_allocated.saturating_sub(block_size);
        inner.stats.bytes_freed += block_size;
        inner.stats.slab_returns += 1;
        Ok(())
    }

    // ── Introspection ───────────────────────────────────────────────

    /// Snapshot of current pool statistics.
    pub fn statistics(&self) -> PoolStatistics {
        self.inner.lock().expect("pool lock poisoned").stats.clone()
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    // -- Arena basics -------------------------------------------------

    #[test]
    fn arena_alloc_and_readback() {
        let pool = MemoryPool::new();
        let alloc = pool.arena_alloc(128).unwrap();
        assert_eq!(alloc.len(), 128);

        pool.write_arena(&alloc, &vec![0xAB; 128]);
        let data = pool.read_arena(&alloc);
        assert!(data.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn arena_alloc_is_zeroed() {
        let pool = MemoryPool::new();
        let alloc = pool.arena_alloc(256).unwrap();
        let data = pool.read_arena(&alloc);
        assert!(data.iter().all(|&b| b == 0));
    }

    #[test]
    fn arena_bulk_reset() {
        let pool = MemoryPool::new();
        let _a = pool.arena_alloc(512).unwrap();
        let _b = pool.arena_alloc(512).unwrap();
        let stats = pool.statistics();
        assert_eq!(stats.bytes_allocated, 1024);

        pool.arena_reset();
        let stats = pool.statistics();
        assert_eq!(stats.bytes_allocated, 0);
        assert!(stats.bytes_freed >= 1024);
        assert_eq!(stats.reset_count, 1);

        // Can allocate again from the same chunks.
        let _c = pool.arena_alloc(512).unwrap();
        assert_eq!(pool.statistics().allocation_count, 3);
    }

    #[test]
    fn arena_grows_when_full() {
        let cfg = PoolConfig::builder()
            .initial_size(128)
            .growth_strategy(GrowthStrategy::Double)
            .max_pool_size(4096)
            .max_allocation_size(4096)
            .build();
        let pool = MemoryPool::with_config(cfg);

        // First chunk is 128 bytes; allocate more than that.
        let a = pool.arena_alloc(120).unwrap();
        assert_eq!(a.len(), 120);
        // This should trigger a new chunk.
        let b = pool.arena_alloc(120).unwrap();
        assert_eq!(b.len(), 120);
        assert_eq!(pool.statistics().allocation_count, 2);
    }

    // -- Slab basics --------------------------------------------------

    #[test]
    fn slab_checkout_and_checkin() {
        let pool = MemoryPool::new();
        pool.register_slab(256, 4);

        let block = pool.slab_checkout(256).unwrap();
        assert_eq!(block.len(), 256);
        assert!(block.iter().all(|&b| b == 0));

        let stats = pool.statistics();
        assert_eq!(stats.slab_checkouts, 1);

        pool.slab_checkin(256, block).unwrap();
        let stats = pool.statistics();
        assert_eq!(stats.slab_returns, 1);
    }

    #[test]
    fn slab_grows_beyond_initial() {
        let pool = MemoryPool::new();
        pool.register_slab(64, 2);

        // Check out all pre-allocated + one extra.
        let b1 = pool.slab_checkout(64).unwrap();
        let b2 = pool.slab_checkout(64).unwrap();
        let b3 = pool.slab_checkout(64).unwrap();
        assert_eq!(pool.statistics().slab_checkouts, 3);

        pool.slab_checkin(64, b1).unwrap();
        pool.slab_checkin(64, b2).unwrap();
        pool.slab_checkin(64, b3).unwrap();
    }

    #[test]
    fn slab_not_found() {
        let pool = MemoryPool::new();
        let err = pool.slab_checkout(999).unwrap_err();
        assert_eq!(err, PoolError::SlabNotFound { block_size: 999 });
    }

    // -- Limit enforcement --------------------------------------------

    #[test]
    fn allocation_too_large() {
        let cfg = PoolConfig::builder().max_allocation_size(1024).build();
        let pool = MemoryPool::with_config(cfg);

        let err = pool.arena_alloc(2048).unwrap_err();
        assert_eq!(err, PoolError::AllocationTooLarge { requested: 2048, limit: 1024 });
    }

    #[test]
    fn pool_exhaustion() {
        let cfg = PoolConfig::builder()
            .initial_size(128)
            .growth_strategy(GrowthStrategy::Fixed)
            .max_pool_size(256)
            .max_allocation_size(256)
            .build();
        let pool = MemoryPool::with_config(cfg);

        // Fill both possible chunks.
        let _a = pool.arena_alloc(120).unwrap();
        let _b = pool.arena_alloc(120).unwrap();
        // Third should fail — no room for another chunk.
        let err = pool.arena_alloc(120).unwrap_err();
        assert!(matches!(err, PoolError::PoolExhausted { .. }));
    }

    // -- Statistics ---------------------------------------------------

    #[test]
    fn statistics_tracking() {
        let pool = MemoryPool::new();
        pool.register_slab(64, 0);

        let _a = pool.arena_alloc(100).unwrap();
        let _b = pool.arena_alloc(200).unwrap();
        let stats = pool.statistics();
        assert_eq!(stats.bytes_allocated, 300);
        assert_eq!(stats.allocation_count, 2);
        assert_eq!(stats.peak_usage, 300);

        pool.arena_reset();
        let stats = pool.statistics();
        assert_eq!(stats.bytes_allocated, 0);
        assert!(stats.bytes_freed >= 300);

        // Slab stats.
        let block = pool.slab_checkout(64).unwrap();
        assert_eq!(pool.statistics().slab_checkouts, 1);
        pool.slab_checkin(64, block).unwrap();
        assert_eq!(pool.statistics().slab_returns, 1);
    }

    #[test]
    fn peak_usage_persists_after_reset() {
        let pool = MemoryPool::new();
        let _a = pool.arena_alloc(500).unwrap();
        pool.arena_reset();
        let _b = pool.arena_alloc(100).unwrap();
        assert_eq!(pool.statistics().peak_usage, 500);
    }

    // -- Edge cases ---------------------------------------------------

    #[test]
    fn zero_size_allocation() {
        let pool = MemoryPool::new();
        let alloc = pool.arena_alloc(0).unwrap();
        assert!(alloc.is_empty());
        let data = pool.read_arena(&alloc);
        assert!(data.is_empty());
    }

    #[test]
    fn builder_defaults() {
        let cfg = PoolConfig::default();
        assert_eq!(cfg.initial_size, 1 << 20);
        assert_eq!(cfg.growth_strategy, GrowthStrategy::Double);
        assert_eq!(cfg.max_pool_size, 256 << 20);
        assert_eq!(cfg.max_allocation_size, 64 << 20);
    }

    #[test]
    fn default_pool_trait() {
        let pool = MemoryPool::default();
        let _a = pool.arena_alloc(64).unwrap();
        assert_eq!(pool.statistics().allocation_count, 1);
    }

    // -- Thread safety ------------------------------------------------

    #[test]
    fn concurrent_arena_allocations() {
        let pool = MemoryPool::with_config(
            PoolConfig::builder()
                .initial_size(1 << 20)
                .max_pool_size(64 << 20)
                .max_allocation_size(4096)
                .build(),
        );

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let p = pool.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let alloc = p.arena_alloc(64).unwrap();
                        assert_eq!(alloc.len(), 64);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }
        assert_eq!(pool.statistics().allocation_count, 800);
    }

    #[test]
    fn concurrent_slab_operations() {
        let pool = MemoryPool::new();
        pool.register_slab(128, 16);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let p = pool.clone();
                thread::spawn(move || {
                    for _ in 0..50 {
                        let block = p.slab_checkout(128).unwrap();
                        p.slab_checkin(128, block).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        let stats = pool.statistics();
        assert_eq!(stats.slab_checkouts, 200);
        assert_eq!(stats.slab_returns, 200);
    }

    // -- Property tests -----------------------------------------------

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn allocated_within_limits(sizes in proptest::collection::vec(1_usize..4096, 1..20)) {
                let cfg = PoolConfig::builder()
                    .initial_size(1 << 20)
                    .max_pool_size(64 << 20)
                    .max_allocation_size(4096)
                    .build();
                let pool = MemoryPool::with_config(cfg);

                for size in &sizes {
                    let alloc = pool.arena_alloc(*size).unwrap();
                    prop_assert!(alloc.len() <= 4096);
                    prop_assert_eq!(alloc.len(), *size);
                }

                let stats = pool.statistics();
                let expected: usize = sizes.iter().sum();
                prop_assert_eq!(stats.bytes_allocated, expected);
                prop_assert!(stats.peak_usage <= 64 << 20);
            }

            #[test]
            fn slab_round_trip_preserves_size(block_size in 1_usize..8192, count in 1_usize..10) {
                let pool = MemoryPool::new();
                pool.register_slab(block_size, 0);

                let mut blocks = Vec::new();
                for _ in 0..count {
                    let block = pool.slab_checkout(block_size).unwrap();
                    prop_assert_eq!(block.len(), block_size);
                    blocks.push(block);
                }

                for block in blocks {
                    pool.slab_checkin(block_size, block).unwrap();
                }

                let stats = pool.statistics();
                prop_assert_eq!(stats.slab_checkouts as usize, count);
                prop_assert_eq!(stats.slab_returns as usize, count);
            }

            #[test]
            fn arena_reset_frees_all(sizes in proptest::collection::vec(1_usize..1024, 1..30)) {
                let pool = MemoryPool::new();
                for size in &sizes {
                    pool.arena_alloc(*size).unwrap();
                }
                pool.arena_reset();
                prop_assert_eq!(pool.statistics().bytes_allocated, 0);
            }
        }
    }
}
