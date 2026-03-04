//! GPU memory pool with arena allocation for reducing allocation overhead.
//!
//! Provides several allocation strategies — arena (bump-pointer), slab
//! (fixed-size), and best-fit (variable-size) — unified behind a single
//! [`MemoryPool`] façade.  All implementations are CPU-only reference
//! implementations that simulate GPU memory management semantics without
//! requiring an OpenCL runtime.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ──────────────────────────────────────────────────────────────────────────────
// PoolConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`MemoryPool`].
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Initial pool capacity in bytes.
    pub initial_size: usize,
    /// Multiplicative growth factor when the pool must grow (≥ 1.0).
    pub grow_factor: f64,
    /// Hard upper bound on total pool capacity in bytes.
    pub max_size: usize,
    /// Required alignment in bytes (must be a power of two).
    pub alignment: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 64 * 1024 * 1024, // 64 MiB
            grow_factor: 2.0,
            max_size: 1024 * 1024 * 1024, // 1 GiB
            alignment: 256,               // GPU-friendly alignment
        }
    }
}

impl PoolConfig {
    /// Create a new configuration with the given initial size.
    pub fn with_initial_size(mut self, size: usize) -> Self {
        self.initial_size = size;
        self
    }

    /// Set the growth factor.
    pub fn with_grow_factor(mut self, factor: f64) -> Self {
        self.grow_factor = factor;
        self
    }

    /// Set the maximum pool size.
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }

    /// Set the required alignment.
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }

    /// Validate the configuration, returning an error message on failure.
    fn validate(&self) -> Result<(), PoolError> {
        if self.initial_size == 0 {
            return Err(PoolError::InvalidConfig("initial_size must be > 0".into()));
        }
        if self.max_size < self.initial_size {
            return Err(PoolError::InvalidConfig("max_size must be >= initial_size".into()));
        }
        if self.grow_factor < 1.0 {
            return Err(PoolError::InvalidConfig("grow_factor must be >= 1.0".into()));
        }
        if self.alignment == 0 || !self.alignment.is_power_of_two() {
            return Err(PoolError::InvalidConfig(
                "alignment must be a non-zero power of two".into(),
            ));
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Error types
// ──────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// Requested allocation exceeds pool limits.
    OutOfMemory { requested: usize, available: usize },
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Attempted to free an invalid or already-freed handle.
    InvalidHandle(u64),
    /// Zero-size allocation requested.
    ZeroSizeAllocation,
    /// Pool has reached its maximum size and cannot grow.
    PoolExhausted,
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory { requested, available } => {
                write!(f, "out of memory: requested {requested} bytes, {available} available")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid pool config: {msg}"),
            Self::InvalidHandle(id) => write!(f, "invalid allocation handle: {id}"),
            Self::ZeroSizeAllocation => write!(f, "zero-size allocation not permitted"),
            Self::PoolExhausted => write!(f, "pool exhausted: max size reached"),
        }
    }
}

impl std::error::Error for PoolError {}

// ──────────────────────────────────────────────────────────────────────────────
// OomHandler — out-of-memory strategies
// ──────────────────────────────────────────────────────────────────────────────

/// Strategy to apply when an allocation cannot be satisfied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OomStrategy {
    /// Try to grow the pool.
    Grow,
    /// Evict the least-recently-used allocation.
    EvictLru,
    /// Fail immediately with an error.
    Fail,
}

/// Configurable handler that decides what to do on OOM.
#[derive(Debug, Clone)]
pub struct OomHandler {
    strategy: OomStrategy,
}

impl OomHandler {
    pub fn new(strategy: OomStrategy) -> Self {
        Self { strategy }
    }

    pub fn strategy(&self) -> OomStrategy {
        self.strategy
    }
}

impl Default for OomHandler {
    fn default() -> Self {
        Self { strategy: OomStrategy::Grow }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PoolStats
// ──────────────────────────────────────────────────────────────────────────────

/// Live statistics for a memory pool.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolStats {
    /// Total pool capacity in bytes.
    pub total_capacity: usize,
    /// Bytes currently allocated.
    pub allocated_bytes: usize,
    /// Bytes currently free.
    pub free_bytes: usize,
    /// Number of live allocations.
    pub allocation_count: usize,
    /// Fragmentation ratio (0.0 = none, 1.0 = maximally fragmented).
    pub fragmentation: f64,
    /// Peak number of bytes ever allocated simultaneously.
    pub peak_usage: usize,
    /// Number of times the pool has been grown.
    pub grow_count: usize,
}

impl PoolStats {
    /// Verify the invariant: allocated + free = total.
    pub fn is_consistent(&self) -> bool {
        self.allocated_bytes + self.free_bytes == self.total_capacity
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AllocationHandle — RAII guard that returns memory on drop
// ──────────────────────────────────────────────────────────────────────────────

/// Globally unique allocation ID.
static NEXT_ALLOC_ID: AtomicU64 = AtomicU64::new(1);

fn next_alloc_id() -> u64 {
    NEXT_ALLOC_ID.fetch_add(1, Ordering::Relaxed)
}

/// Metadata for a single allocation.
#[derive(Debug, Clone)]
pub struct AllocMeta {
    pub id: u64,
    pub offset: usize,
    pub size: usize,
    /// Aligned size (may be larger than `size` due to alignment padding).
    pub aligned_size: usize,
    /// Monotonic timestamp used for LRU eviction.
    pub last_access: u64,
}

/// RAII handle that automatically returns memory to the pool on drop.
pub struct AllocationHandle {
    meta: AllocMeta,
    pool: Arc<Mutex<PoolInner>>,
}

impl AllocationHandle {
    /// Unique ID for this allocation.
    pub fn id(&self) -> u64 {
        self.meta.id
    }

    /// Byte offset within the pool.
    pub fn offset(&self) -> usize {
        self.meta.offset
    }

    /// Requested size in bytes.
    pub fn size(&self) -> usize {
        self.meta.size
    }

    /// Aligned (actual) size consumed in the pool.
    pub fn aligned_size(&self) -> usize {
        self.meta.aligned_size
    }
}

impl fmt::Debug for AllocationHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AllocationHandle")
            .field("id", &self.meta.id)
            .field("offset", &self.meta.offset)
            .field("size", &self.meta.size)
            .finish()
    }
}

impl Drop for AllocationHandle {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.pool.lock() {
            inner.free(self.meta.id);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// FreeBlock — represents a contiguous free region
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
struct FreeBlock {
    offset: usize,
    size: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// PoolInner — shared mutable state behind Arc<Mutex<>>
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct PoolInner {
    /// Simulated GPU memory backing (CPU Vec).
    storage: Vec<u8>,
    /// Allocation metadata keyed by ID.
    allocations: HashMap<u64, AllocMeta>,
    /// Sorted list of free blocks (by offset).
    free_blocks: Vec<FreeBlock>,
    config: PoolConfig,
    peak_usage: usize,
    grow_count: usize,
    access_clock: u64,
}

impl PoolInner {
    fn new(config: &PoolConfig) -> Self {
        let cap = config.initial_size;
        Self {
            storage: vec![0u8; cap],
            allocations: HashMap::new(),
            free_blocks: vec![FreeBlock { offset: 0, size: cap }],
            config: config.clone(),
            peak_usage: 0,
            grow_count: 0,
            access_clock: 0,
        }
    }

    /// Total pool capacity.
    fn capacity(&self) -> usize {
        self.storage.len()
    }

    /// Sum of all live allocation aligned sizes.
    fn allocated_bytes(&self) -> usize {
        self.allocations.values().map(|a| a.aligned_size).sum()
    }

    /// Align `size` up to the configured alignment.
    fn align_up(&self, size: usize) -> usize {
        let mask = self.config.alignment - 1;
        (size + mask) & !mask
    }

    /// Tick the internal clock and return the new value.
    fn tick(&mut self) -> u64 {
        self.access_clock += 1;
        self.access_clock
    }

    /// Try to grow the pool to accommodate at least `needed` extra bytes.
    fn try_grow(&mut self, needed: usize) -> Result<(), PoolError> {
        let current = self.capacity();
        let mut new_cap = ((current as f64) * self.config.grow_factor) as usize;
        // Ensure we grow enough for the requested allocation.
        if new_cap < current + needed {
            new_cap = current + needed;
        }
        if new_cap > self.config.max_size {
            new_cap = self.config.max_size;
        }
        if new_cap <= current {
            return Err(PoolError::PoolExhausted);
        }
        let added = new_cap - current;
        self.storage.resize(new_cap, 0);
        // Merge new space with the last free block if it abuts, else push new.
        if let Some(last) = self.free_blocks.last_mut() {
            if last.offset + last.size == current {
                last.size += added;
            } else {
                self.free_blocks.push(FreeBlock { offset: current, size: added });
            }
        } else {
            self.free_blocks.push(FreeBlock { offset: current, size: added });
        }
        self.grow_count += 1;
        Ok(())
    }

    /// Best-fit allocation from the free list.
    fn alloc_best_fit(&mut self, size: usize) -> Option<AllocMeta> {
        let aligned = self.align_up(size);
        // Find the smallest free block that fits.
        let mut best_idx = None;
        let mut best_waste = usize::MAX;
        for (i, blk) in self.free_blocks.iter().enumerate() {
            if blk.size >= aligned && blk.size - aligned < best_waste {
                best_waste = blk.size - aligned;
                best_idx = Some(i);
            }
        }
        let idx = best_idx?;
        let blk = &self.free_blocks[idx];
        let offset = blk.offset;
        let remaining = blk.size - aligned;
        if remaining == 0 {
            self.free_blocks.remove(idx);
        } else {
            self.free_blocks[idx] = FreeBlock { offset: offset + aligned, size: remaining };
        }
        let ts = self.tick();
        let id = next_alloc_id();
        let meta = AllocMeta { id, offset, size, aligned_size: aligned, last_access: ts };
        self.allocations.insert(id, meta.clone());
        let used = self.allocated_bytes();
        if used > self.peak_usage {
            self.peak_usage = used;
        }
        Some(meta)
    }

    /// Free an allocation by ID, returning its block to the free list.
    fn free(&mut self, id: u64) -> bool {
        let Some(meta) = self.allocations.remove(&id) else {
            return false;
        };
        let new_block = FreeBlock { offset: meta.offset, size: meta.aligned_size };
        // Insert in sorted order.
        let pos = self.free_blocks.partition_point(|b| b.offset < new_block.offset);
        self.free_blocks.insert(pos, new_block);
        // Merge with neighbours.
        self.coalesce_at(pos);
        true
    }

    /// Merge the block at `idx` with its left and right neighbours if adjacent.
    fn coalesce_at(&mut self, idx: usize) {
        // Merge right first so index stays valid.
        if idx + 1 < self.free_blocks.len() {
            let (cur_end, next_off, next_sz) = {
                let cur = &self.free_blocks[idx];
                let next = &self.free_blocks[idx + 1];
                (cur.offset + cur.size, next.offset, next.size)
            };
            if cur_end == next_off {
                self.free_blocks[idx].size += next_sz;
                self.free_blocks.remove(idx + 1);
            }
        }
        // Merge left.
        if idx > 0 {
            let (prev_end, cur_off, cur_sz) = {
                let prev = &self.free_blocks[idx - 1];
                let cur = &self.free_blocks[idx];
                (prev.offset + prev.size, cur.offset, cur.size)
            };
            if prev_end == cur_off {
                self.free_blocks[idx - 1].size += cur_sz;
                self.free_blocks.remove(idx);
            }
        }
    }

    /// Evict the least-recently-used allocation.
    fn evict_lru(&mut self) -> Option<AllocMeta> {
        let lru_id = self.allocations.values().min_by_key(|a| a.last_access).map(|a| a.id)?;
        let meta = self.allocations.get(&lru_id).cloned()?;
        self.free(lru_id);
        Some(meta)
    }

    /// Compute fragmentation ratio.
    fn fragmentation(&self) -> f64 {
        if self.free_blocks.is_empty() {
            return 0.0;
        }
        if self.free_blocks.len() == 1 {
            return 0.0;
        }
        let total_free: usize = self.free_blocks.iter().map(|b| b.size).sum();
        if total_free == 0 {
            return 0.0;
        }
        let largest_free = self.free_blocks.iter().map(|b| b.size).max().unwrap_or(0);
        // fragmentation = 1 - (largest / total_free)
        1.0 - (largest_free as f64 / total_free as f64)
    }

    /// Collect stats.
    fn stats(&self) -> PoolStats {
        let allocated = self.allocated_bytes();
        let total = self.capacity();
        PoolStats {
            total_capacity: total,
            allocated_bytes: allocated,
            free_bytes: total - allocated,
            allocation_count: self.allocations.len(),
            fragmentation: self.fragmentation(),
            peak_usage: self.peak_usage,
            grow_count: self.grow_count,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MemoryPool — the public façade
// ──────────────────────────────────────────────────────────────────────────────

/// Pre-allocated memory pool with configurable block sizes.
///
/// Wraps an internal free-list allocator and provides several higher-level
/// allocation strategies ([`ArenaAllocator`], [`SlabAllocator`],
/// [`BestFitAllocator`]) on top.
pub struct MemoryPool {
    inner: Arc<Mutex<PoolInner>>,
    oom_handler: OomHandler,
}

impl fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryPool")
            .field("stats", &self.stats())
            .field("oom_handler", &self.oom_handler)
            .finish()
    }
}

impl MemoryPool {
    /// Create a new pool with the given configuration.
    pub fn new(config: PoolConfig) -> Result<Self, PoolError> {
        config.validate()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(PoolInner::new(&config))),
            oom_handler: OomHandler::default(),
        })
    }

    /// Create a pool with a custom OOM handler.
    pub fn with_oom_handler(
        config: PoolConfig,
        oom_handler: OomHandler,
    ) -> Result<Self, PoolError> {
        config.validate()?;
        Ok(Self { inner: Arc::new(Mutex::new(PoolInner::new(&config))), oom_handler })
    }

    /// Allocate `size` bytes from the pool.
    pub fn allocate(&self, size: usize) -> Result<AllocationHandle, PoolError> {
        if size == 0 {
            return Err(PoolError::ZeroSizeAllocation);
        }
        let mut inner = self.inner.lock().unwrap();
        // First attempt.
        if let Some(meta) = inner.alloc_best_fit(size) {
            return Ok(AllocationHandle { meta, pool: Arc::clone(&self.inner) });
        }
        // Handle OOM.
        match self.oom_handler.strategy() {
            OomStrategy::Grow => {
                let aligned = inner.align_up(size);
                inner.try_grow(aligned)?;
                inner
                    .alloc_best_fit(size)
                    .map(|meta| AllocationHandle { meta, pool: Arc::clone(&self.inner) })
                    .ok_or(PoolError::OutOfMemory {
                        requested: size,
                        available: inner.capacity() - inner.allocated_bytes(),
                    })
            }
            OomStrategy::EvictLru => {
                // Keep evicting until enough space is available or nothing left.
                let aligned = inner.align_up(size);
                loop {
                    if let Some(meta) = inner.alloc_best_fit(size) {
                        return Ok(AllocationHandle { meta, pool: Arc::clone(&self.inner) });
                    }
                    if inner.evict_lru().is_none() {
                        break;
                    }
                    // Safety: we evicted something — retry allocation.
                    if inner.allocated_bytes() + aligned <= inner.capacity() {
                        continue;
                    }
                }
                inner
                    .alloc_best_fit(size)
                    .map(|meta| AllocationHandle { meta, pool: Arc::clone(&self.inner) })
                    .ok_or(PoolError::OutOfMemory {
                        requested: size,
                        available: inner.capacity() - inner.allocated_bytes(),
                    })
            }
            OomStrategy::Fail => Err(PoolError::OutOfMemory {
                requested: size,
                available: inner.capacity() - inner.allocated_bytes(),
            }),
        }
    }

    /// Explicitly free an allocation by handle ID (normally handled by RAII).
    pub fn free(&self, id: u64) -> Result<(), PoolError> {
        let mut inner = self.inner.lock().unwrap();
        if inner.free(id) { Ok(()) } else { Err(PoolError::InvalidHandle(id)) }
    }

    /// Get a snapshot of the current pool statistics.
    pub fn stats(&self) -> PoolStats {
        self.inner.lock().unwrap().stats()
    }

    /// Get an [`ArenaAllocator`] view over this pool.
    pub fn arena(&self) -> ArenaAllocator {
        ArenaAllocator { inner: Arc::clone(&self.inner), handles: Vec::new() }
    }

    /// Get a [`SlabAllocator`] for a fixed element size.
    pub fn slab(&self, element_size: usize) -> Result<SlabAllocator, PoolError> {
        if element_size == 0 {
            return Err(PoolError::ZeroSizeAllocation);
        }
        Ok(SlabAllocator {
            inner: Arc::clone(&self.inner),
            element_size,
            free_slots: VecDeque::new(),
            handles: Vec::new(),
        })
    }

    /// Get a [`BestFitAllocator`] view over this pool.
    pub fn best_fit(&self) -> BestFitAllocator {
        BestFitAllocator { inner: Arc::clone(&self.inner) }
    }

    /// Get a [`MemoryDefragmenter`] for this pool.
    pub fn defragmenter(&self) -> MemoryDefragmenter {
        MemoryDefragmenter { inner: Arc::clone(&self.inner) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ArenaAllocator — bump-pointer style
// ──────────────────────────────────────────────────────────────────────────────

/// Bump-pointer arena that allocates from the pool and frees all at once on
/// [`reset`](Self::reset).  Ideal for per-inference temporary buffers.
pub struct ArenaAllocator {
    inner: Arc<Mutex<PoolInner>>,
    handles: Vec<u64>,
}

impl fmt::Debug for ArenaAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArenaAllocator").field("live_allocations", &self.handles.len()).finish()
    }
}

impl ArenaAllocator {
    /// Allocate `size` bytes from the arena (backed by the pool).
    pub fn alloc(&mut self, size: usize) -> Result<AllocMeta, PoolError> {
        if size == 0 {
            return Err(PoolError::ZeroSizeAllocation);
        }
        let mut inner = self.inner.lock().unwrap();
        let meta = inner.alloc_best_fit(size).ok_or_else(|| PoolError::OutOfMemory {
            requested: size,
            available: inner.capacity() - inner.allocated_bytes(),
        })?;
        self.handles.push(meta.id);
        Ok(meta)
    }

    /// Number of live allocations in this arena.
    pub fn allocation_count(&self) -> usize {
        self.handles.len()
    }

    /// Total aligned bytes currently owned by this arena.
    pub fn allocated_bytes(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        self.handles.iter().filter_map(|id| inner.allocations.get(id)).map(|m| m.aligned_size).sum()
    }

    /// Free all allocations made through this arena, making the memory
    /// available for reuse.
    pub fn reset(&mut self) {
        let mut inner = self.inner.lock().unwrap();
        for id in self.handles.drain(..) {
            inner.free(id);
        }
    }
}

impl Drop for ArenaAllocator {
    fn drop(&mut self) {
        self.reset();
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SlabAllocator — fixed-size slots
// ──────────────────────────────────────────────────────────────────────────────

/// Fixed-size slab allocator: every allocation is exactly `element_size` bytes.
/// Freed slots are recycled through an internal free list before asking the
/// pool for more memory.
pub struct SlabAllocator {
    inner: Arc<Mutex<PoolInner>>,
    element_size: usize,
    /// Recycled offsets+ids available for re-issue.
    free_slots: VecDeque<AllocMeta>,
    /// All live allocation IDs (for cleanup on drop).
    handles: Vec<u64>,
}

impl fmt::Debug for SlabAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlabAllocator")
            .field("element_size", &self.element_size)
            .field("live", &self.handles.len())
            .field("free_slots", &self.free_slots.len())
            .finish()
    }
}

impl SlabAllocator {
    /// Allocate one slab-sized slot.
    pub fn alloc(&mut self) -> Result<AllocMeta, PoolError> {
        // Prefer a recycled slot.
        if let Some(mut meta) = self.free_slots.pop_front() {
            // Re-register in the pool's allocation table.
            let mut inner = self.inner.lock().unwrap();
            meta.id = next_alloc_id();
            meta.last_access = {
                inner.access_clock += 1;
                inner.access_clock
            };
            inner.allocations.insert(meta.id, meta.clone());
            let used = inner.allocated_bytes();
            if used > inner.peak_usage {
                inner.peak_usage = used;
            }
            self.handles.push(meta.id);
            return Ok(meta);
        }
        let mut inner = self.inner.lock().unwrap();
        let meta =
            inner.alloc_best_fit(self.element_size).ok_or_else(|| PoolError::OutOfMemory {
                requested: self.element_size,
                available: inner.capacity() - inner.allocated_bytes(),
            })?;
        self.handles.push(meta.id);
        Ok(meta)
    }

    /// Return a slot for reuse.  The slot is **not** returned to the main pool
    /// free list — it is kept in the slab's internal free list.
    pub fn free(&mut self, id: u64) -> Result<(), PoolError> {
        let pos = self.handles.iter().position(|h| *h == id).ok_or(PoolError::InvalidHandle(id))?;
        self.handles.remove(pos);
        let mut inner = self.inner.lock().unwrap();
        if let Some(meta) = inner.allocations.remove(&id) {
            self.free_slots.push_back(meta);
            Ok(())
        } else {
            Err(PoolError::InvalidHandle(id))
        }
    }

    /// Number of live allocations.
    pub fn live_count(&self) -> usize {
        self.handles.len()
    }

    /// Number of recycled slots available.
    pub fn free_slot_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Element size for this slab.
    pub fn element_size(&self) -> usize {
        self.element_size
    }
}

impl Drop for SlabAllocator {
    fn drop(&mut self) {
        let mut inner = self.inner.lock().unwrap();
        for id in self.handles.drain(..) {
            inner.free(id);
        }
        // Return recycled slots to the pool.
        for meta in self.free_slots.drain(..) {
            let blk = FreeBlock { offset: meta.offset, size: meta.aligned_size };
            let pos = inner.free_blocks.partition_point(|b| b.offset < blk.offset);
            inner.free_blocks.insert(pos, blk);
            inner.coalesce_at(pos);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BestFitAllocator
// ──────────────────────────────────────────────────────────────────────────────

/// Best-fit allocator view: picks the smallest free block that satisfies each
/// request, minimising fragmentation.
pub struct BestFitAllocator {
    inner: Arc<Mutex<PoolInner>>,
}

impl fmt::Debug for BestFitAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BestFitAllocator").finish()
    }
}

impl BestFitAllocator {
    /// Allocate `size` bytes using best-fit strategy.
    pub fn alloc(&self, size: usize) -> Result<AllocationHandle, PoolError> {
        if size == 0 {
            return Err(PoolError::ZeroSizeAllocation);
        }
        let mut inner = self.inner.lock().unwrap();
        inner
            .alloc_best_fit(size)
            .map(|meta| AllocationHandle { meta, pool: Arc::clone(&self.inner) })
            .ok_or_else(|| PoolError::OutOfMemory {
                requested: size,
                available: inner.capacity() - inner.allocated_bytes(),
            })
    }

    /// Free by handle ID.
    pub fn free(&self, id: u64) -> Result<(), PoolError> {
        let mut inner = self.inner.lock().unwrap();
        if inner.free(id) { Ok(()) } else { Err(PoolError::InvalidHandle(id)) }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MemoryDefragmenter
// ──────────────────────────────────────────────────────────────────────────────

/// Defragmenter that compacts live allocations toward the start of the pool,
/// coalescing free space at the end.
///
/// Returns a relocation map so callers can update any external pointers.
pub struct MemoryDefragmenter {
    inner: Arc<Mutex<PoolInner>>,
}

impl fmt::Debug for MemoryDefragmenter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryDefragmenter").finish()
    }
}

/// Result of a defragmentation pass.
#[derive(Debug, Clone)]
pub struct DefragResult {
    /// Map from allocation ID to (old_offset, new_offset).
    pub relocations: HashMap<u64, (usize, usize)>,
    /// Number of allocations that were moved.
    pub moved_count: usize,
    /// Fragmentation ratio before defrag.
    pub fragmentation_before: f64,
    /// Fragmentation ratio after defrag.
    pub fragmentation_after: f64,
}

impl MemoryDefragmenter {
    /// Run a compaction pass.
    pub fn defragment(&self) -> DefragResult {
        let mut inner = self.inner.lock().unwrap();
        let frag_before = inner.fragmentation();

        // Collect live allocations sorted by offset.
        let mut allocs: Vec<AllocMeta> = inner.allocations.values().cloned().collect();
        allocs.sort_by_key(|a| a.offset);

        let mut relocations = HashMap::new();
        let mut cursor: usize = 0;

        for alloc in &allocs {
            if alloc.offset != cursor {
                let old_offset = alloc.offset;
                // Move data in the backing storage.
                let src = old_offset;
                let dst = cursor;
                let len = alloc.aligned_size;
                // Safe: src and dst ranges don't overlap when cursor < src.
                if dst < src {
                    inner.storage.copy_within(src..src + len, dst);
                }
                // Update the allocation metadata.
                if let Some(meta) = inner.allocations.get_mut(&alloc.id) {
                    meta.offset = cursor;
                }
                relocations.insert(alloc.id, (old_offset, cursor));
            }
            cursor += alloc.aligned_size;
        }

        // Rebuild free list: single block from cursor to end.
        inner.free_blocks.clear();
        let cap = inner.capacity();
        if cursor < cap {
            inner.free_blocks.push(FreeBlock { offset: cursor, size: cap - cursor });
        }

        let frag_after = inner.fragmentation();
        let moved = relocations.len();

        DefragResult {
            relocations,
            moved_count: moved,
            fragmentation_before: frag_before,
            fragmentation_after: frag_after,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: small pool for testing.
    fn small_pool(size: usize) -> MemoryPool {
        MemoryPool::new(
            PoolConfig::default()
                .with_initial_size(size)
                .with_max_size(size * 4)
                .with_alignment(16),
        )
        .unwrap()
    }

    fn small_pool_fail(size: usize) -> MemoryPool {
        MemoryPool::with_oom_handler(
            PoolConfig::default().with_initial_size(size).with_max_size(size).with_alignment(16),
            OomHandler::new(OomStrategy::Fail),
        )
        .unwrap()
    }

    // ── PoolConfig validation ─────────────────────────────────────────────

    #[test]
    fn test_config_default_is_valid() {
        PoolConfig::default().validate().unwrap();
    }

    #[test]
    fn test_config_zero_initial_size() {
        let err = PoolConfig::default().with_initial_size(0).validate();
        assert!(matches!(err, Err(PoolError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_max_less_than_initial() {
        let err = PoolConfig::default().with_initial_size(1024).with_max_size(512).validate();
        assert!(matches!(err, Err(PoolError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_grow_factor_below_one() {
        let err = PoolConfig::default().with_grow_factor(0.5).validate();
        assert!(matches!(err, Err(PoolError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_alignment_zero() {
        let err = PoolConfig::default().with_alignment(0).validate();
        assert!(matches!(err, Err(PoolError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_alignment_not_power_of_two() {
        let err = PoolConfig::default().with_alignment(7).validate();
        assert!(matches!(err, Err(PoolError::InvalidConfig(_))));
    }

    // ── Basic allocate / free ─────────────────────────────────────────────

    #[test]
    fn test_allocate_single() {
        let pool = small_pool(4096);
        let h = pool.allocate(128).unwrap();
        assert_eq!(h.size(), 128);
        assert!(h.aligned_size() >= 128);
    }

    #[test]
    fn test_allocate_and_free() {
        let pool = small_pool(4096);
        let h = pool.allocate(256).unwrap();
        let id = h.id();
        drop(h);
        // After free, stats should show 0 allocated.
        let stats = pool.stats();
        assert_eq!(stats.allocated_bytes, 0);
        // Double-free should error.
        assert!(matches!(pool.free(id), Err(PoolError::InvalidHandle(_))));
    }

    #[test]
    fn test_allocate_multiple() {
        let pool = small_pool(4096);
        let a = pool.allocate(128).unwrap();
        let b = pool.allocate(256).unwrap();
        let c = pool.allocate(64).unwrap();
        assert_ne!(a.id(), b.id());
        assert_ne!(b.id(), c.id());
        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 3);
        assert!(stats.allocated_bytes >= 128 + 256 + 64);
    }

    #[test]
    fn test_allocate_fill_pool() {
        let pool = small_pool_fail(1024);
        let _a = pool.allocate(512).unwrap();
        let _b = pool.allocate(512).unwrap();
        // Pool is full — next alloc should fail.
        let err = pool.allocate(16);
        assert!(matches!(err, Err(PoolError::OutOfMemory { .. })));
    }

    #[test]
    fn test_allocate_after_free_reuses_space() {
        let pool = small_pool_fail(1024);
        let a = pool.allocate(512).unwrap();
        let offset_a = a.offset();
        drop(a);
        let b = pool.allocate(512).unwrap();
        // Should reuse the same offset.
        assert_eq!(b.offset(), offset_a);
    }

    #[test]
    fn test_zero_size_alloc_rejected() {
        let pool = small_pool(4096);
        assert!(matches!(pool.allocate(0), Err(PoolError::ZeroSizeAllocation)));
    }

    // ── Alignment ─────────────────────────────────────────────────────────

    #[test]
    fn test_allocation_alignment() {
        let pool = MemoryPool::new(
            PoolConfig::default().with_initial_size(4096).with_max_size(4096).with_alignment(64),
        )
        .unwrap();
        for size in [1, 13, 33, 63, 65, 100] {
            let h = pool.allocate(size).unwrap();
            assert_eq!(h.offset() % 64, 0, "offset {0} not aligned to 64", h.offset());
            assert!(h.aligned_size() % 64 == 0);
        }
    }

    // ── RAII handle auto-return ───────────────────────────────────────────

    #[test]
    fn test_raii_auto_return() {
        let pool = small_pool(4096);
        {
            let _a = pool.allocate(1024).unwrap();
            let _b = pool.allocate(1024).unwrap();
            assert_eq!(pool.stats().allocation_count, 2);
        }
        // Both handles dropped — pool should be fully free.
        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 0);
        assert_eq!(stats.allocated_bytes, 0);
        assert_eq!(stats.free_bytes, stats.total_capacity);
    }

    #[test]
    fn test_raii_partial_drop() {
        let pool = small_pool(4096);
        let a = pool.allocate(256).unwrap();
        {
            let _b = pool.allocate(256).unwrap();
        }
        assert_eq!(pool.stats().allocation_count, 1);
        drop(a);
        assert_eq!(pool.stats().allocation_count, 0);
    }

    // ── Arena allocator ───────────────────────────────────────────────────

    #[test]
    fn test_arena_alloc_and_reset() {
        let pool = small_pool(4096);
        let mut arena = pool.arena();
        arena.alloc(128).unwrap();
        arena.alloc(256).unwrap();
        assert_eq!(arena.allocation_count(), 2);
        assert!(arena.allocated_bytes() >= 384);
        arena.reset();
        assert_eq!(arena.allocation_count(), 0);
        assert_eq!(arena.allocated_bytes(), 0);
        // Pool should be fully free.
        assert_eq!(pool.stats().allocated_bytes, 0);
    }

    #[test]
    fn test_arena_reset_and_reuse() {
        let pool = small_pool_fail(1024);
        let mut arena = pool.arena();
        arena.alloc(512).unwrap();
        arena.alloc(512).unwrap();
        // Pool is full.
        assert!(arena.alloc(16).is_err());
        arena.reset();
        // After reset we should be able to allocate again.
        arena.alloc(1024).unwrap();
        assert_eq!(arena.allocation_count(), 1);
    }

    #[test]
    fn test_arena_drop_frees_all() {
        let pool = small_pool(4096);
        {
            let mut arena = pool.arena();
            arena.alloc(128).unwrap();
            arena.alloc(256).unwrap();
        }
        assert_eq!(pool.stats().allocated_bytes, 0);
    }

    #[test]
    fn test_arena_zero_size_rejected() {
        let pool = small_pool(4096);
        let mut arena = pool.arena();
        assert!(matches!(arena.alloc(0), Err(PoolError::ZeroSizeAllocation)));
    }

    #[test]
    fn test_arena_many_small_allocs() {
        let pool = small_pool(65536);
        let mut arena = pool.arena();
        for _ in 0..100 {
            arena.alloc(64).unwrap();
        }
        assert_eq!(arena.allocation_count(), 100);
        arena.reset();
        assert_eq!(arena.allocation_count(), 0);
    }

    // ── Slab allocator ────────────────────────────────────────────────────

    #[test]
    fn test_slab_basic() {
        let pool = small_pool(4096);
        let mut slab = pool.slab(128).unwrap();
        let a = slab.alloc().unwrap();
        let b = slab.alloc().unwrap();
        assert_eq!(a.size, 128);
        assert_eq!(b.size, 128);
        assert_eq!(slab.live_count(), 2);
    }

    #[test]
    fn test_slab_recycle() {
        let pool = small_pool_fail(1024);
        let mut slab = pool.slab(512).unwrap();
        let a = slab.alloc().unwrap();
        let b = slab.alloc().unwrap();
        let id_a = a.id;
        let offset_a = a.offset;
        // Free a.
        slab.free(id_a).unwrap();
        assert_eq!(slab.live_count(), 1);
        assert_eq!(slab.free_slot_count(), 1);
        // Alloc again — should recycle the freed slot.
        let c = slab.alloc().unwrap();
        assert_eq!(c.offset, offset_a);
        assert_eq!(slab.free_slot_count(), 0);
        drop(b);
    }

    #[test]
    fn test_slab_uniform_size() {
        let pool = small_pool(65536);
        let mut slab = pool.slab(256).unwrap();
        let mut metas = Vec::new();
        for _ in 0..10 {
            metas.push(slab.alloc().unwrap());
        }
        for m in &metas {
            assert_eq!(m.size, 256);
            assert!(m.aligned_size >= 256);
        }
    }

    #[test]
    fn test_slab_zero_element_rejected() {
        let pool = small_pool(4096);
        assert!(matches!(pool.slab(0), Err(PoolError::ZeroSizeAllocation)));
    }

    #[test]
    fn test_slab_drop_returns_all() {
        let pool = small_pool(4096);
        {
            let mut slab = pool.slab(128).unwrap();
            slab.alloc().unwrap();
            slab.alloc().unwrap();
            let a = slab.alloc().unwrap();
            slab.free(a.id).unwrap();
        }
        assert_eq!(pool.stats().allocated_bytes, 0);
    }

    #[test]
    fn test_slab_invalid_free() {
        let pool = small_pool(4096);
        let mut slab = pool.slab(128).unwrap();
        assert!(matches!(slab.free(9999), Err(PoolError::InvalidHandle(9999))));
    }

    // ── Best-fit allocator ────────────────────────────────────────────────

    #[test]
    fn test_best_fit_basic() {
        let pool = small_pool(4096);
        let bf = pool.best_fit();
        let h = bf.alloc(512).unwrap();
        assert!(h.size() == 512);
    }

    #[test]
    fn test_best_fit_reduces_fragmentation() {
        // Create a pool, allocate A B C, free B, then allocate D that fits B.
        let pool = small_pool_fail(4096);
        let bf = pool.best_fit();
        let a = bf.alloc(256).unwrap();
        let b = bf.alloc(512).unwrap();
        let c = bf.alloc(256).unwrap();
        let b_offset = b.offset();
        drop(b);
        // Allocate something that fits exactly in B's slot (or smaller).
        let d = bf.alloc(512).unwrap();
        assert_eq!(d.offset(), b_offset, "best-fit should reuse freed slot");
        drop(a);
        drop(c);
        drop(d);
    }

    #[test]
    fn test_best_fit_picks_smallest_fitting() {
        let pool = small_pool_fail(8192);
        let bf = pool.best_fit();
        // Create holes of different sizes.
        let a = bf.alloc(256).unwrap(); // [0..256)
        let b = bf.alloc(1024).unwrap(); // [256..1280)
        let c = bf.alloc(256).unwrap(); // [1280..1536)
        let d = bf.alloc(512).unwrap(); // [1536..2048)
        let _e = bf.alloc(256).unwrap(); // [2048..2304) — keeps things pinned
        // Free b (1024) and d (512) to create two holes.
        let _b_off = b.offset();
        let d_off = d.offset();
        drop(b); // hole of 1024
        drop(d); // hole of 512
        // Allocate 512 — best-fit should pick the 512 hole, not the 1024.
        let f = bf.alloc(512).unwrap();
        assert_eq!(f.offset(), d_off, "should pick 512-byte hole over 1024-byte hole");
        drop(a);
        drop(c);
        drop(f);
    }

    #[test]
    fn test_best_fit_zero_rejected() {
        let pool = small_pool(4096);
        let bf = pool.best_fit();
        assert!(matches!(bf.alloc(0), Err(PoolError::ZeroSizeAllocation)));
    }

    #[test]
    fn test_best_fit_free_invalid() {
        let pool = small_pool(4096);
        let bf = pool.best_fit();
        assert!(matches!(bf.free(42), Err(PoolError::InvalidHandle(42))));
    }

    // ── Pool stats & invariants ───────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let pool = small_pool(4096);
        let stats = pool.stats();
        assert_eq!(stats.total_capacity, 4096);
        assert_eq!(stats.allocated_bytes, 0);
        assert_eq!(stats.free_bytes, 4096);
        assert_eq!(stats.allocation_count, 0);
        assert!(stats.is_consistent());
    }

    #[test]
    fn test_stats_after_alloc() {
        let pool = small_pool(4096);
        let _h = pool.allocate(128).unwrap();
        let stats = pool.stats();
        assert!(stats.allocated_bytes >= 128);
        assert_eq!(stats.allocation_count, 1);
        assert!(stats.is_consistent());
    }

    #[test]
    fn test_stats_after_free() {
        let pool = small_pool(4096);
        let h = pool.allocate(1024).unwrap();
        drop(h);
        let stats = pool.stats();
        assert_eq!(stats.allocated_bytes, 0);
        assert_eq!(stats.free_bytes, 4096);
        assert!(stats.is_consistent());
    }

    #[test]
    fn test_stats_consistency_many_ops() {
        let pool = small_pool(65536);
        let mut handles = Vec::new();
        for i in 0..20 {
            handles.push(pool.allocate(64 * (i + 1)).unwrap());
            assert!(pool.stats().is_consistent());
        }
        for _ in 0..10 {
            handles.pop();
            assert!(pool.stats().is_consistent());
        }
    }

    #[test]
    fn test_peak_usage_tracked() {
        let pool = small_pool(8192);
        let a = pool.allocate(1024).unwrap();
        let b = pool.allocate(2048).unwrap();
        let peak_with_both = pool.stats().peak_usage;
        drop(b);
        let _c = pool.allocate(512).unwrap();
        let stats = pool.stats();
        assert!(stats.peak_usage >= peak_with_both);
        drop(a);
    }

    // ── Pool growth ───────────────────────────────────────────────────────

    #[test]
    fn test_pool_grows_on_demand() {
        let pool = MemoryPool::new(
            PoolConfig::default()
                .with_initial_size(256)
                .with_max_size(4096)
                .with_alignment(16)
                .with_grow_factor(2.0),
        )
        .unwrap();
        // Fill initial pool.
        let _a = pool.allocate(256).unwrap();
        // This should trigger growth.
        let _b = pool.allocate(128).unwrap();
        let stats = pool.stats();
        assert!(stats.total_capacity > 256);
        assert!(stats.grow_count >= 1);
    }

    #[test]
    fn test_pool_growth_respects_max() {
        let pool = MemoryPool::new(
            PoolConfig::default()
                .with_initial_size(256)
                .with_max_size(512)
                .with_alignment(16)
                .with_grow_factor(2.0),
        )
        .unwrap();
        let _a = pool.allocate(256).unwrap();
        let _b = pool.allocate(256).unwrap();
        // Now at max — should fail.
        let err = pool.allocate(128);
        assert!(err.is_err());
    }

    #[test]
    fn test_pool_growth_count() {
        let pool = MemoryPool::new(
            PoolConfig::default()
                .with_initial_size(128)
                .with_max_size(2048)
                .with_alignment(16)
                .with_grow_factor(1.5),
        )
        .unwrap();
        let _a = pool.allocate(128).unwrap();
        let _b = pool.allocate(128).unwrap();
        assert!(pool.stats().grow_count >= 1);
    }

    // ── OOM handler strategies ────────────────────────────────────────────

    #[test]
    fn test_oom_fail_strategy() {
        let pool = small_pool_fail(256);
        let _a = pool.allocate(256).unwrap();
        assert!(matches!(pool.allocate(16), Err(PoolError::OutOfMemory { .. })));
    }

    #[test]
    fn test_oom_grow_strategy() {
        let pool = MemoryPool::with_oom_handler(
            PoolConfig::default().with_initial_size(256).with_max_size(4096).with_alignment(16),
            OomHandler::new(OomStrategy::Grow),
        )
        .unwrap();
        let _a = pool.allocate(256).unwrap();
        let b = pool.allocate(128).unwrap();
        assert!(b.size() == 128);
    }

    #[test]
    fn test_oom_evict_lru_strategy() {
        let pool = MemoryPool::with_oom_handler(
            PoolConfig::default().with_initial_size(512).with_max_size(512).with_alignment(16),
            OomHandler::new(OomStrategy::EvictLru),
        )
        .unwrap();
        // Allocate two blocks that fill the pool.
        let a = pool.allocate(256).unwrap();
        let b = pool.allocate(256).unwrap();
        // The next allocate will evict the oldest (a).
        std::mem::forget(a); // prevent RAII free
        std::mem::forget(b);
        let c = pool.allocate(256).unwrap();
        assert!(c.size() == 256);
        assert_eq!(pool.stats().allocation_count, 2); // b + c (a was evicted)
        // Clean up remaining via explicit free.
        std::mem::forget(c);
    }

    #[test]
    fn test_oom_handler_default_is_grow() {
        let handler = OomHandler::default();
        assert_eq!(handler.strategy(), OomStrategy::Grow);
    }

    // ── Defragmentation ───────────────────────────────────────────────────

    #[test]
    fn test_defrag_no_fragmentation() {
        let pool = small_pool(4096);
        let _a = pool.allocate(128).unwrap();
        let _b = pool.allocate(256).unwrap();
        let defrag = pool.defragmenter();
        let result = defrag.defragment();
        assert_eq!(result.moved_count, 0);
    }

    #[test]
    fn test_defrag_reduces_fragmentation() {
        let pool = small_pool_fail(4096);
        let a = pool.allocate(256).unwrap();
        let b = pool.allocate(256).unwrap();
        let c = pool.allocate(256).unwrap();
        // Free b to create a hole in the middle.
        let b_id = b.id();
        pool.free(b_id).unwrap();
        std::mem::forget(b);
        let _frag_before = pool.stats().fragmentation;
        let defrag = pool.defragmenter();
        let result = defrag.defragment();
        assert!(result.fragmentation_after <= result.fragmentation_before);
        // After defrag, fragmentation should be 0 (single free block at end).
        assert_eq!(pool.stats().fragmentation, 0.0);
        drop(a);
        drop(c);
    }

    #[test]
    fn test_defrag_reports_relocations() {
        let pool = small_pool_fail(4096);
        let a = pool.allocate(256).unwrap();
        let b = pool.allocate(256).unwrap();
        let c = pool.allocate(256).unwrap();
        let a_id = a.id();
        let _c_id = c.id();
        // Free a to create hole at the beginning.
        pool.free(a_id).unwrap();
        std::mem::forget(a);
        let defrag = pool.defragmenter();
        let result = defrag.defragment();
        // b and c should have been relocated.
        assert!(result.moved_count >= 1);
        drop(b);
        drop(c);
    }

    #[test]
    fn test_defrag_empty_pool() {
        let pool = small_pool(4096);
        let defrag = pool.defragmenter();
        let result = defrag.defragment();
        assert_eq!(result.moved_count, 0);
        assert_eq!(result.fragmentation_after, 0.0);
    }

    #[test]
    fn test_defrag_preserves_data_size() {
        let pool = small_pool_fail(4096);
        let a = pool.allocate(128).unwrap();
        let b = pool.allocate(256).unwrap();
        let c = pool.allocate(128).unwrap();
        let _allocated_before = pool.stats().allocated_bytes;
        pool.free(b.id()).unwrap();
        std::mem::forget(b);
        let allocated_after_free = pool.stats().allocated_bytes;
        let defrag = pool.defragmenter();
        defrag.defragment();
        // Total allocated should not change from defrag.
        assert_eq!(pool.stats().allocated_bytes, allocated_after_free);
        drop(a);
        drop(c);
    }

    // ── Coalescing ────────────────────────────────────────────────────────

    #[test]
    fn test_adjacent_frees_coalesce() {
        let pool = small_pool_fail(4096);
        let a = pool.allocate(256).unwrap();
        let b = pool.allocate(256).unwrap();
        let c = pool.allocate(256).unwrap();
        drop(b);
        drop(a);
        // a and b should have coalesced — allocating 512 should succeed.
        let d = pool.allocate(512).unwrap();
        assert_eq!(d.offset(), 0);
        drop(c);
    }

    #[test]
    fn test_full_coalesce_after_all_freed() {
        let pool = small_pool_fail(4096);
        let a = pool.allocate(1024).unwrap();
        let b = pool.allocate(1024).unwrap();
        let c = pool.allocate(1024).unwrap();
        drop(a);
        drop(c);
        drop(b);
        // Should have fully coalesced.
        let stats = pool.stats();
        assert_eq!(stats.fragmentation, 0.0);
        // Should be able to allocate the full pool.
        let _d = pool.allocate(4096).unwrap();
    }

    // ── Edge cases ────────────────────────────────────────────────────────

    #[test]
    fn test_max_size_allocation() {
        let pool = small_pool_fail(4096);
        let h = pool.allocate(4096).unwrap();
        assert!(h.size() == 4096);
        assert_eq!(pool.stats().allocation_count, 1);
    }

    #[test]
    fn test_allocation_larger_than_pool_fails() {
        let pool = small_pool_fail(256);
        assert!(pool.allocate(512).is_err());
    }

    #[test]
    fn test_many_alloc_free_cycles() {
        let pool = small_pool(4096);
        for _ in 0..100 {
            let h = pool.allocate(64).unwrap();
            drop(h);
        }
        let stats = pool.stats();
        assert_eq!(stats.allocated_bytes, 0);
        assert!(stats.is_consistent());
    }

    #[test]
    fn test_alternating_sizes() {
        let pool = small_pool(65536);
        let mut handles = Vec::new();
        for i in 0..20 {
            let size = if i % 2 == 0 { 128 } else { 1024 };
            handles.push(pool.allocate(size).unwrap());
        }
        assert_eq!(pool.stats().allocation_count, 20);
        assert!(pool.stats().is_consistent());
    }

    #[test]
    fn test_single_byte_alloc() {
        let pool = small_pool(4096);
        let h = pool.allocate(1).unwrap();
        assert_eq!(h.size(), 1);
        assert!(h.aligned_size() >= 1);
    }

    // ── Fragmentation metrics ─────────────────────────────────────────────

    #[test]
    fn test_no_fragmentation_initially() {
        let pool = small_pool(4096);
        assert_eq!(pool.stats().fragmentation, 0.0);
    }

    #[test]
    fn test_fragmentation_after_hole() {
        let pool = small_pool_fail(4096);
        let a = pool.allocate(256).unwrap();
        let b = pool.allocate(256).unwrap();
        let c = pool.allocate(256).unwrap();
        drop(b); // creates hole
        let frag = pool.stats().fragmentation;
        assert!(frag > 0.0, "expected fragmentation > 0, got {frag}");
        drop(a);
        drop(c);
    }

    #[test]
    fn test_fragmentation_goes_to_zero_after_coalesce() {
        let pool = small_pool_fail(4096);
        let a = pool.allocate(256).unwrap();
        let b = pool.allocate(256).unwrap();
        drop(a);
        drop(b);
        assert_eq!(pool.stats().fragmentation, 0.0);
    }

    // ── Property-like tests ───────────────────────────────────────────────

    #[test]
    fn test_allocated_plus_free_equals_total() {
        let pool = small_pool(8192);
        let mut handles = Vec::new();
        for _ in 0..10 {
            handles.push(pool.allocate(64).unwrap());
        }
        let stats = pool.stats();
        assert!(stats.is_consistent());
        assert_eq!(stats.allocated_bytes + stats.free_bytes, stats.total_capacity);
    }

    #[test]
    fn test_consistency_through_lifecycle() {
        let pool = small_pool(16384);
        let mut handles = Vec::new();
        // Allocate
        for _ in 0..30 {
            handles.push(pool.allocate(128).unwrap());
            assert!(pool.stats().is_consistent());
        }
        // Free every other one
        let mut i = 0;
        handles.retain(|_| {
            i += 1;
            i % 2 == 0
        });
        assert!(pool.stats().is_consistent());
        // Allocate more
        for _ in 0..10 {
            handles.push(pool.allocate(64).unwrap());
            assert!(pool.stats().is_consistent());
        }
        // Free all
        handles.clear();
        let stats = pool.stats();
        assert!(stats.is_consistent());
        assert_eq!(stats.allocated_bytes, 0);
    }

    #[test]
    fn test_peak_never_decreases() {
        let pool = small_pool(65536);
        let mut max_peak = 0;
        let mut handles = Vec::new();
        for _ in 0..20 {
            handles.push(pool.allocate(256).unwrap());
            let peak = pool.stats().peak_usage;
            assert!(peak >= max_peak);
            max_peak = peak;
        }
        handles.clear();
        assert!(pool.stats().peak_usage >= max_peak);
    }

    // ── PoolError display ─────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e = PoolError::OutOfMemory { requested: 100, available: 50 };
        assert!(e.to_string().contains("100"));
        assert!(e.to_string().contains("50"));

        let e = PoolError::ZeroSizeAllocation;
        assert!(e.to_string().contains("zero"));
    }

    #[test]
    fn test_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(PoolError::InvalidHandle(7));
        assert!(e.to_string().contains("7"));
    }

    // ── Debug impls ───────────────────────────────────────────────────────

    #[test]
    fn test_debug_impls() {
        let pool = small_pool(4096);
        let _ = format!("{pool:?}");
        let arena = pool.arena();
        let _ = format!("{arena:?}");
        let slab = pool.slab(64).unwrap();
        let _ = format!("{slab:?}");
        let bf = pool.best_fit();
        let _ = format!("{bf:?}");
        let defrag = pool.defragmenter();
        let _ = format!("{defrag:?}");
    }

    #[test]
    fn test_handle_debug() {
        let pool = small_pool(4096);
        let h = pool.allocate(128).unwrap();
        let dbg = format!("{h:?}");
        assert!(dbg.contains("AllocationHandle"));
    }

    // ── Config builder ────────────────────────────────────────────────────

    #[test]
    fn test_config_builder_chain() {
        let cfg = PoolConfig::default()
            .with_initial_size(1024)
            .with_grow_factor(1.5)
            .with_max_size(8192)
            .with_alignment(32);
        assert_eq!(cfg.initial_size, 1024);
        assert!((cfg.grow_factor - 1.5).abs() < f64::EPSILON);
        assert_eq!(cfg.max_size, 8192);
        assert_eq!(cfg.alignment, 32);
    }

    // ── Stress / interleaved operations ───────────────────────────────────

    #[test]
    fn test_interleaved_alloc_free() {
        let pool = small_pool(65536);
        let mut handles = Vec::new();
        for i in 0..50 {
            handles.push(pool.allocate(128).unwrap());
            if i % 3 == 0 && !handles.is_empty() {
                handles.remove(0);
            }
            assert!(pool.stats().is_consistent());
        }
    }

    #[test]
    fn test_slab_many_recycles() {
        let pool = small_pool(4096);
        let mut slab = pool.slab(64).unwrap();
        for _ in 0..50 {
            let m = slab.alloc().unwrap();
            slab.free(m.id).unwrap();
        }
        assert_eq!(slab.live_count(), 0);
        assert!(slab.free_slot_count() > 0);
    }

    #[test]
    fn test_arena_rapid_reset_cycles() {
        let pool = small_pool(4096);
        let mut arena = pool.arena();
        for _ in 0..20 {
            for _ in 0..5 {
                arena.alloc(64).unwrap();
            }
            arena.reset();
        }
        assert_eq!(pool.stats().allocated_bytes, 0);
    }

    #[test]
    fn test_mixed_allocators_share_pool() {
        let pool = small_pool(65536);
        let mut arena = pool.arena();
        let bf = pool.best_fit();
        let mut slab = pool.slab(128).unwrap();
        arena.alloc(256).unwrap();
        let _h = bf.alloc(512).unwrap();
        slab.alloc().unwrap();
        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 3);
        assert!(stats.is_consistent());
    }

    // ── OomHandler struct ─────────────────────────────────────────────────

    #[test]
    fn test_oom_handler_strategies() {
        assert_eq!(OomHandler::new(OomStrategy::Grow).strategy(), OomStrategy::Grow);
        assert_eq!(OomHandler::new(OomStrategy::Fail).strategy(), OomStrategy::Fail);
        assert_eq!(OomHandler::new(OomStrategy::EvictLru).strategy(), OomStrategy::EvictLru);
    }

    // ── Allocation metadata ───────────────────────────────────────────────

    #[test]
    fn test_alloc_meta_fields() {
        let pool = small_pool(4096);
        let h = pool.allocate(100).unwrap();
        assert!(h.id() > 0);
        assert_eq!(h.size(), 100);
        assert!(h.aligned_size() >= 100);
        assert!(h.offset() + h.aligned_size() <= pool.stats().total_capacity);
    }

    #[test]
    fn test_alloc_ids_unique() {
        let pool = small_pool(65536);
        let mut ids = Vec::new();
        for _ in 0..50 {
            let h = pool.allocate(64).unwrap();
            ids.push(h.id());
            std::mem::forget(h); // prevent drop
        }
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 50);
    }

    // ── PoolStats ─────────────────────────────────────────────────────────

    #[test]
    fn test_pool_stats_is_consistent_method() {
        let good = PoolStats {
            total_capacity: 1000,
            allocated_bytes: 400,
            free_bytes: 600,
            allocation_count: 2,
            fragmentation: 0.1,
            peak_usage: 500,
            grow_count: 0,
        };
        assert!(good.is_consistent());

        let bad = PoolStats {
            total_capacity: 1000,
            allocated_bytes: 400,
            free_bytes: 500, // 400 + 500 != 1000
            allocation_count: 2,
            fragmentation: 0.1,
            peak_usage: 500,
            grow_count: 0,
        };
        assert!(!bad.is_consistent());
    }

    // ── Growth edge cases ─────────────────────────────────────────────────

    #[test]
    fn test_grow_factor_one() {
        // grow_factor = 1.0 means we only grow by exactly the needed amount.
        let pool = MemoryPool::new(
            PoolConfig::default()
                .with_initial_size(256)
                .with_max_size(4096)
                .with_alignment(16)
                .with_grow_factor(1.0),
        )
        .unwrap();
        let _a = pool.allocate(256).unwrap();
        let _b = pool.allocate(128).unwrap();
        let stats = pool.stats();
        assert!(stats.total_capacity >= 256 + 128);
        assert!(stats.grow_count >= 1);
    }

    #[test]
    fn test_pool_exhausted_error() {
        let pool = MemoryPool::with_oom_handler(
            PoolConfig::default().with_initial_size(256).with_max_size(256).with_alignment(16),
            OomHandler::new(OomStrategy::Grow),
        )
        .unwrap();
        let _a = pool.allocate(256).unwrap();
        let err = pool.allocate(16);
        assert!(
            matches!(err, Err(PoolError::PoolExhausted)),
            "expected PoolExhausted, got {err:?}"
        );
    }

    #[test]
    fn test_explicit_free_vs_raii() {
        let pool = small_pool(4096);
        let a = pool.allocate(128).unwrap();
        let a_id = a.id();
        // Explicit free.
        pool.free(a_id).unwrap();
        // Prevent RAII double-free (drop after explicit free is no-op because
        // PoolInner::free returns false for already-freed ids).
        std::mem::forget(a);
        assert_eq!(pool.stats().allocation_count, 0);
    }
}
