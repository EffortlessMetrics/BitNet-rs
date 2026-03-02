//! GPU memory allocator with buddy system, sub-allocation, and memory pooling.
//!
//! Designed for Intel Arc A770 (16 GB VRAM) OpenCL workloads. Provides:
//! - Power-of-2 buddy allocation with split/coalesce
//! - Sub-allocation from large GPU buffers (slab pattern)
//! - Typed memory pools (weights, KV cache, activations, scratch)
//! - Fragmentation analysis and OOM recovery

use std::collections::{BTreeMap, HashMap};
use std::fmt;

// ---------------------------------------------------------------------------
// Constants — A770-specific
// ---------------------------------------------------------------------------

/// Total VRAM budget for A770 (16 GB).
pub const A770_TOTAL_VRAM: u64 = 16 * 1024 * 1024 * 1024;

/// Large-allocation page alignment (64 KB).
pub const LARGE_PAGE_ALIGN: usize = 64 * 1024;

/// Small-allocation alignment (256 B).
pub const SMALL_ALIGN: usize = 256;

/// Minimum buddy block size (256 B).
pub const MIN_BLOCK_SIZE: u64 = 256;

/// Threshold for "large" allocation (anything ≥ 64 KB uses large alignment).
pub const LARGE_ALLOC_THRESHOLD: u64 = 64 * 1024;

// ---------------------------------------------------------------------------
// MemoryPool kind
// ---------------------------------------------------------------------------

/// Logical pool a GPU allocation belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolKind {
    Weights,
    KvCache,
    Activations,
    Scratch,
}

impl fmt::Display for PoolKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weights => write!(f, "Weights"),
            Self::KvCache => write!(f, "KvCache"),
            Self::Activations => write!(f, "Activations"),
            Self::Scratch => write!(f, "Scratch"),
        }
    }
}

// ---------------------------------------------------------------------------
// Urgency
// ---------------------------------------------------------------------------

/// How urgently the allocation is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Urgency {
    Low,
    Normal,
    High,
    Critical,
}

// ---------------------------------------------------------------------------
// MemoryBlock
// ---------------------------------------------------------------------------

/// A contiguous region inside a buddy / sub-allocator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryBlock {
    pub offset: u64,
    pub size: u64,
    pub alignment: usize,
    pub is_free: bool,
    pub pool_id: Option<PoolKind>,
}

impl MemoryBlock {
    pub fn new(offset: u64, size: u64, alignment: usize) -> Self {
        Self { offset, size, alignment, is_free: true, pool_id: None }
    }

    /// End offset (exclusive).
    pub fn end(&self) -> u64 {
        self.offset + self.size
    }
}

// ---------------------------------------------------------------------------
// AllocationRequest
// ---------------------------------------------------------------------------

/// Describes a single allocation request.
#[derive(Debug, Clone)]
pub struct AllocationRequest {
    pub size: u64,
    pub alignment: usize,
    pub pool: PoolKind,
    pub urgency: Urgency,
    pub label: String,
}

impl AllocationRequest {
    pub fn new(size: u64, pool: PoolKind, label: impl Into<String>) -> Self {
        let alignment = if size >= LARGE_ALLOC_THRESHOLD { LARGE_PAGE_ALIGN } else { SMALL_ALIGN };
        Self { size, alignment, pool, urgency: Urgency::Normal, label: label.into() }
    }

    pub fn with_urgency(mut self, urgency: Urgency) -> Self {
        self.urgency = urgency;
        self
    }

    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }
}

// ---------------------------------------------------------------------------
// PoolConfig
// ---------------------------------------------------------------------------

/// Configuration for a single memory pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub initial_size: u64,
    pub max_size: u64,
    pub growth_factor: f64,
    pub alignment: usize,
}

impl PoolConfig {
    pub fn new(initial_size: u64, max_size: u64) -> Self {
        Self { initial_size, max_size, growth_factor: 2.0, alignment: LARGE_PAGE_ALIGN }
    }

    pub fn with_growth_factor(mut self, factor: f64) -> Self {
        self.growth_factor = factor;
        self
    }

    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }

    /// A770 default pool configs.
    pub fn a770_defaults() -> HashMap<PoolKind, PoolConfig> {
        let mut m = HashMap::new();
        let gb = 1024 * 1024 * 1024u64;
        m.insert(PoolKind::Weights, PoolConfig::new(4 * gb, 6 * gb));
        m.insert(PoolKind::KvCache, PoolConfig::new(4 * gb, 6 * gb));
        m.insert(PoolKind::Activations, PoolConfig::new(2 * gb, 3 * gb).with_growth_factor(1.5));
        m.insert(PoolKind::Scratch, PoolConfig::new(gb, 2 * gb).with_growth_factor(1.5));
        m
    }
}

// ---------------------------------------------------------------------------
// BuddyAllocator
// ---------------------------------------------------------------------------

/// Power-of-2 buddy allocator over a virtual address range.
///
/// Free lists are keyed by order (log2 of block size relative to
/// `MIN_BLOCK_SIZE`). Splitting halves a block; coalescing merges buddies.
#[derive(Debug)]
pub struct BuddyAllocator {
    /// Total capacity — must be a power of two.
    capacity: u64,
    /// Maximum order (log2(capacity / MIN_BLOCK_SIZE)).
    max_order: u32,
    /// Free lists per order. Each entry is a set of offsets.
    free_lists: BTreeMap<u32, Vec<u64>>,
    /// Active allocations: offset → (size, order).
    allocations: HashMap<u64, (u64, u32)>,
}

impl BuddyAllocator {
    /// Create a new buddy allocator with the given capacity.
    ///
    /// `capacity` is rounded up to the next power of two if needed.
    pub fn new(capacity: u64) -> Self {
        let capacity = capacity.next_power_of_two().max(MIN_BLOCK_SIZE);
        let max_order = Self::size_to_order(capacity);
        let mut free_lists = BTreeMap::new();
        free_lists.entry(max_order).or_insert_with(Vec::new).push(0);
        Self { capacity, max_order, free_lists, allocations: HashMap::new() }
    }

    /// Capacity of the allocator.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Total bytes currently allocated.
    pub fn used(&self) -> u64 {
        self.allocations.values().map(|(sz, _)| *sz).sum()
    }

    /// Free bytes available (may be fragmented).
    pub fn free_bytes(&self) -> u64 {
        self.capacity - self.used()
    }

    /// Allocate `size` bytes, returning the offset on success.
    pub fn allocate(&mut self, size: u64) -> Option<u64> {
        if size == 0 {
            return None;
        }
        let actual = size.next_power_of_two().max(MIN_BLOCK_SIZE);
        let order = Self::size_to_order(actual);
        if order > self.max_order {
            return None;
        }
        self.find_and_split(order).inspect(|&offset| {
            self.allocations.insert(offset, (actual, order));
        })
    }

    /// Free a previously allocated block.
    pub fn free(&mut self, offset: u64) -> bool {
        let Some((size, order)) = self.allocations.remove(&offset) else {
            return false;
        };
        self.return_and_coalesce(offset, size, order);
        true
    }

    // -- internal helpers ---------------------------------------------------

    fn size_to_order(size: u64) -> u32 {
        assert!(size >= MIN_BLOCK_SIZE);
        (size / MIN_BLOCK_SIZE).trailing_zeros()
    }

    fn order_to_size(order: u32) -> u64 {
        MIN_BLOCK_SIZE << order
    }

    /// Walk up from `order` to find a free block, then split down.
    fn find_and_split(&mut self, target_order: u32) -> Option<u64> {
        // Find the smallest order ≥ target_order that has a free block.
        let found_order = (target_order..=self.max_order)
            .find(|&o| self.free_lists.get(&o).is_some_and(|v| !v.is_empty()))?;

        let offset = self.free_lists.get_mut(&found_order).unwrap().pop().unwrap();

        // Split down to target order.
        for o in (target_order..found_order).rev() {
            let buddy_offset = offset + Self::order_to_size(o);
            self.free_lists.entry(o).or_default().push(buddy_offset);
        }
        Some(offset)
    }

    /// Return a block and merge with its buddy if the buddy is also free.
    fn return_and_coalesce(&mut self, mut offset: u64, _size: u64, mut order: u32) {
        while order < self.max_order {
            let block_size = Self::order_to_size(order);
            let buddy = offset ^ block_size;

            // Check if buddy is in the free list for this order.
            let buddy_free = self.free_lists.get(&order).is_some_and(|list| list.contains(&buddy));

            if !buddy_free {
                break;
            }
            // Remove buddy from free list.
            if let Some(list) = self.free_lists.get_mut(&order) {
                list.retain(|&o| o != buddy);
            }
            // Merge: take the lower offset.
            offset = offset.min(buddy);
            order += 1;
        }
        self.free_lists.entry(order).or_default().push(offset);
    }

    /// Number of active allocations.
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }
}

// ---------------------------------------------------------------------------
// SubAllocator
// ---------------------------------------------------------------------------

/// Sub-allocates from a single large buffer using a simple bump / free-list
/// slab approach.
#[derive(Debug)]
pub struct SubAllocator {
    buffer_size: u64,
    /// Sorted list of free blocks.
    free_blocks: Vec<MemoryBlock>,
    /// Active allocations keyed by offset.
    allocations: HashMap<u64, MemoryBlock>,
}

impl SubAllocator {
    pub fn new(buffer_size: u64) -> Self {
        let initial = MemoryBlock::new(0, buffer_size, SMALL_ALIGN);
        Self { buffer_size, free_blocks: vec![initial], allocations: HashMap::new() }
    }

    pub fn buffer_size(&self) -> u64 {
        self.buffer_size
    }

    pub fn used(&self) -> u64 {
        self.allocations.values().map(|b| b.size).sum()
    }

    pub fn free_bytes(&self) -> u64 {
        self.buffer_size - self.used()
    }

    /// Allocate `size` bytes with `alignment`. Returns the offset.
    pub fn allocate(&mut self, size: u64, alignment: usize) -> Option<u64> {
        if size == 0 {
            return None;
        }
        let alignment = alignment.max(1) as u64;

        // First-fit among free blocks.
        let idx = self.free_blocks.iter().position(|blk| {
            let aligned_off = align_up(blk.offset, alignment);
            let padding = aligned_off - blk.offset;
            blk.size >= size + padding
        })?;

        let blk = self.free_blocks.remove(idx);
        let aligned_off = align_up(blk.offset, alignment);
        let padding = aligned_off - blk.offset;

        // If there is leading padding, return it as a free block.
        if padding > 0 {
            self.free_blocks.push(MemoryBlock::new(blk.offset, padding, blk.alignment));
        }
        // If there is trailing space, return it as a free block.
        let remainder = blk.size - padding - size;
        if remainder > 0 {
            self.free_blocks.push(MemoryBlock::new(aligned_off + size, remainder, blk.alignment));
        }
        self.sort_free();

        let alloc = MemoryBlock {
            offset: aligned_off,
            size,
            alignment: alignment as usize,
            is_free: false,
            pool_id: None,
        };
        self.allocations.insert(aligned_off, alloc);
        Some(aligned_off)
    }

    /// Free the allocation at `offset`.
    pub fn free(&mut self, offset: u64) -> bool {
        let Some(mut blk) = self.allocations.remove(&offset) else {
            return false;
        };
        blk.is_free = true;
        self.free_blocks.push(blk);
        self.coalesce();
        true
    }

    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    // -- helpers ------------------------------------------------------------

    fn sort_free(&mut self) {
        self.free_blocks.sort_by_key(|b| b.offset);
    }

    fn coalesce(&mut self) {
        self.sort_free();
        let mut i = 0;
        while i + 1 < self.free_blocks.len() {
            if self.free_blocks[i].end() == self.free_blocks[i + 1].offset {
                let next = self.free_blocks.remove(i + 1);
                self.free_blocks[i].size += next.size;
            } else {
                i += 1;
            }
        }
    }
}

/// Align `value` up to the next multiple of `alignment`.
#[inline]
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let mask = alignment - 1;
    (value + mask) & !mask
}

// ---------------------------------------------------------------------------
// MemoryBudget
// ---------------------------------------------------------------------------

/// Tracks per-pool and global memory usage with watermarks.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    pub total: u64,
    entries: HashMap<PoolKind, BudgetEntry>,
}

#[derive(Debug, Clone)]
pub struct BudgetEntry {
    pub allocated: u64,
    pub limit: u64,
    pub high_watermark: u64,
}

impl MemoryBudget {
    pub fn new(total: u64) -> Self {
        Self { total, entries: HashMap::new() }
    }

    pub fn register_pool(&mut self, kind: PoolKind, limit: u64) {
        self.entries.insert(kind, BudgetEntry { allocated: 0, limit, high_watermark: 0 });
    }

    pub fn record_alloc(&mut self, kind: PoolKind, bytes: u64) -> bool {
        let (allocated, limit) = match self.entries.get(&kind) {
            Some(e) => (e.allocated, e.limit),
            None => return false,
        };
        if allocated + bytes > limit {
            return false;
        }
        let total_used: u64 = self.entries.values().map(|e| e.allocated).sum();
        if total_used + bytes > self.total {
            return false;
        }
        let entry = self.entries.get_mut(&kind).unwrap();
        entry.allocated += bytes;
        if entry.allocated > entry.high_watermark {
            entry.high_watermark = entry.allocated;
        }
        true
    }

    pub fn record_free(&mut self, kind: PoolKind, bytes: u64) {
        if let Some(entry) = self.entries.get_mut(&kind) {
            entry.allocated = entry.allocated.saturating_sub(bytes);
        }
    }

    pub fn used(&self) -> u64 {
        self.entries.values().map(|e| e.allocated).sum()
    }

    pub fn free(&self) -> u64 {
        self.total.saturating_sub(self.used())
    }

    pub fn pool_used(&self, kind: PoolKind) -> u64 {
        self.entries.get(&kind).map_or(0, |e| e.allocated)
    }

    pub fn pool_limit(&self, kind: PoolKind) -> u64 {
        self.entries.get(&kind).map_or(0, |e| e.limit)
    }

    pub fn pool_watermark(&self, kind: PoolKind) -> u64 {
        self.entries.get(&kind).map_or(0, |e| e.high_watermark)
    }
}

// ---------------------------------------------------------------------------
// FragmentationAnalyzer
// ---------------------------------------------------------------------------

/// Measures internal and external fragmentation of a sub-allocator.
#[derive(Debug, Clone)]
pub struct FragmentationReport {
    /// Fraction of free space that is not in the largest contiguous block.
    pub external_fragmentation: f64,
    /// Number of disjoint free regions.
    pub free_region_count: usize,
    /// Largest contiguous free block.
    pub largest_free_block: u64,
    /// Total free bytes.
    pub total_free: u64,
}

pub struct FragmentationAnalyzer;

impl FragmentationAnalyzer {
    /// Analyze a sub-allocator's free list.
    pub fn analyze(sub: &SubAllocator) -> FragmentationReport {
        let total_free: u64 = sub.free_blocks.iter().map(|b| b.size).sum();
        let largest = sub.free_blocks.iter().map(|b| b.size).max().unwrap_or(0);
        let external =
            if total_free == 0 { 0.0 } else { 1.0 - (largest as f64 / total_free as f64) };
        FragmentationReport {
            external_fragmentation: external,
            free_region_count: sub.free_blocks.len(),
            largest_free_block: largest,
            total_free,
        }
    }

    /// Analyze a buddy allocator.
    pub fn analyze_buddy(buddy: &BuddyAllocator) -> FragmentationReport {
        let total_free = buddy.free_bytes();
        let largest = buddy
            .free_lists
            .iter()
            .rev()
            .find(|(_, v)| !v.is_empty())
            .map(|(order, _)| BuddyAllocator::order_to_size(*order))
            .unwrap_or(0);
        let free_count: usize = buddy.free_lists.values().map(|v| v.len()).sum();
        let external =
            if total_free == 0 { 0.0 } else { 1.0 - (largest as f64 / total_free as f64) };
        FragmentationReport {
            external_fragmentation: external,
            free_region_count: free_count,
            largest_free_block: largest,
            total_free,
        }
    }
}

// ---------------------------------------------------------------------------
// OomAction / OomHandler
// ---------------------------------------------------------------------------

/// Actions the OOM handler can take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OomAction {
    /// Evict the lowest-priority allocation from the given pool.
    Evict(PoolKind),
    /// Attempt defragmentation.
    Defragment,
    /// Fail the allocation gracefully.
    Fail(String),
}

/// Decides what to do when an allocation cannot be satisfied.
pub struct OomHandler {
    /// Maximum number of eviction attempts before giving up.
    pub max_evictions: usize,
}

impl Default for OomHandler {
    fn default() -> Self {
        Self { max_evictions: 3 }
    }
}

impl OomHandler {
    pub fn new(max_evictions: usize) -> Self {
        Self { max_evictions }
    }

    /// Decide on recovery actions for a failed allocation.
    pub fn handle(&self, request: &AllocationRequest, budget: &MemoryBudget) -> Vec<OomAction> {
        let mut actions = Vec::new();

        // If the pool is over 80% full, try eviction from that pool.
        let pool_used = budget.pool_used(request.pool);
        let pool_limit = budget.pool_limit(request.pool);
        if pool_limit > 0 && pool_used as f64 / pool_limit as f64 > 0.8 {
            for _ in 0..self.max_evictions.min(3) {
                actions.push(OomAction::Evict(request.pool));
            }
        }

        // Try defragmenting.
        actions.push(OomAction::Defragment);

        // If still critical, try evicting from scratch (lowest priority).
        if request.urgency >= Urgency::High && request.pool != PoolKind::Scratch {
            actions.push(OomAction::Evict(PoolKind::Scratch));
        }

        // Final fallback: fail gracefully.
        actions.push(OomAction::Fail(format!(
            "OOM: cannot allocate {} bytes for '{}' in {} pool",
            request.size, request.label, request.pool
        )));

        actions
    }
}

// ---------------------------------------------------------------------------
// MemoryPool — typed pool backed by a BuddyAllocator
// ---------------------------------------------------------------------------

/// A typed memory pool for one [`PoolKind`], backed by a [`BuddyAllocator`].
#[derive(Debug)]
pub struct MemoryPool {
    pub kind: PoolKind,
    pub config: PoolConfig,
    allocator: BuddyAllocator,
    current_size: u64,
    /// Maps label → offset for named allocations.
    labels: HashMap<String, u64>,
}

impl MemoryPool {
    pub fn new(kind: PoolKind, config: PoolConfig) -> Self {
        let allocator = BuddyAllocator::new(config.initial_size);
        let current_size = allocator.capacity();
        Self { kind, config, allocator, current_size, labels: HashMap::new() }
    }

    pub fn allocate(&mut self, request: &AllocationRequest) -> Option<u64> {
        if request.pool != self.kind {
            return None;
        }
        let offset = self.allocator.allocate(request.size);
        if let Some(off) = offset
            && !request.label.is_empty()
        {
            self.labels.insert(request.label.clone(), off);
        }
        offset
    }

    pub fn free(&mut self, offset: u64) -> bool {
        self.labels.retain(|_, &mut v| v != offset);
        self.allocator.free(offset)
    }

    pub fn free_by_label(&mut self, label: &str) -> bool {
        if let Some(offset) = self.labels.remove(label) {
            self.allocator.free(offset)
        } else {
            false
        }
    }

    /// Try to grow the pool by `growth_factor`.
    pub fn try_grow(&mut self) -> bool {
        let new_size = (self.current_size as f64 * self.config.growth_factor) as u64;
        if new_size > self.config.max_size {
            return false;
        }
        // Replace allocator — only valid when pool is empty.
        if self.allocator.allocation_count() != 0 {
            return false;
        }
        self.allocator = BuddyAllocator::new(new_size);
        self.current_size = self.allocator.capacity();
        true
    }

    pub fn used(&self) -> u64 {
        self.allocator.used()
    }

    pub fn capacity(&self) -> u64 {
        self.allocator.capacity()
    }

    pub fn allocation_count(&self) -> usize {
        self.allocator.allocation_count()
    }
}

// ---------------------------------------------------------------------------
// GpuMemoryAllocator — top-level façade
// ---------------------------------------------------------------------------

/// Top-level GPU memory allocator managing typed pools and a global budget.
pub struct GpuMemoryAllocator {
    pools: HashMap<PoolKind, MemoryPool>,
    budget: MemoryBudget,
    oom_handler: OomHandler,
}

impl GpuMemoryAllocator {
    /// Create an allocator with A770 defaults.
    pub fn new_a770() -> Self {
        let configs = PoolConfig::a770_defaults();
        Self::with_configs(A770_TOTAL_VRAM, configs)
    }

    /// Create with custom pool configurations.
    pub fn with_configs(total_vram: u64, configs: HashMap<PoolKind, PoolConfig>) -> Self {
        let mut budget = MemoryBudget::new(total_vram);
        let mut pools = HashMap::new();
        for (kind, cfg) in &configs {
            budget.register_pool(*kind, cfg.max_size);
            pools.insert(*kind, MemoryPool::new(*kind, cfg.clone()));
        }
        Self { pools, budget, oom_handler: OomHandler::default() }
    }

    /// Attempt to allocate memory.
    pub fn allocate(&mut self, request: &AllocationRequest) -> Result<u64, Vec<OomAction>> {
        // Budget gate.
        if !self.budget.record_alloc(request.pool, request.size) {
            return Err(self.oom_handler.handle(request, &self.budget));
        }

        let pool = match self.pools.get_mut(&request.pool) {
            Some(p) => p,
            None => {
                self.budget.record_free(request.pool, request.size);
                return Err(vec![OomAction::Fail(format!("no pool for {:?}", request.pool))]);
            }
        };

        match pool.allocate(request) {
            Some(offset) => Ok(offset),
            None => {
                // Roll back budget.
                self.budget.record_free(request.pool, request.size);
                Err(self.oom_handler.handle(request, &self.budget))
            }
        }
    }

    /// Free a previously allocated block.
    pub fn free(&mut self, pool_kind: PoolKind, offset: u64, size: u64) -> bool {
        if let Some(pool) = self.pools.get_mut(&pool_kind)
            && pool.free(offset)
        {
            self.budget.record_free(pool_kind, size);
            return true;
        }
        false
    }

    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    pub fn pool(&self, kind: PoolKind) -> Option<&MemoryPool> {
        self.pools.get(&kind)
    }

    pub fn pool_mut(&mut self, kind: PoolKind) -> Option<&mut MemoryPool> {
        self.pools.get_mut(&kind)
    }

    pub fn set_oom_handler(&mut self, handler: OomHandler) {
        self.oom_handler = handler;
    }
}

impl fmt::Debug for GpuMemoryAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuMemoryAllocator")
            .field("budget_used", &self.budget.used())
            .field("budget_free", &self.budget.free())
            .finish()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- MemoryBlock --------------------------------------------------------

    #[test]
    fn memory_block_new() {
        let b = MemoryBlock::new(0, 1024, 256);
        assert!(b.is_free);
        assert_eq!(b.offset, 0);
        assert_eq!(b.size, 1024);
        assert_eq!(b.end(), 1024);
        assert!(b.pool_id.is_none());
    }

    #[test]
    fn memory_block_end() {
        let b = MemoryBlock::new(100, 200, 64);
        assert_eq!(b.end(), 300);
    }

    // -- BuddyAllocator basic -----------------------------------------------

    #[test]
    fn buddy_create() {
        let b = BuddyAllocator::new(4096);
        assert_eq!(b.capacity(), 4096);
        assert_eq!(b.used(), 0);
        assert_eq!(b.allocation_count(), 0);
    }

    #[test]
    fn buddy_allocate_basic() {
        let mut b = BuddyAllocator::new(4096);
        let off = b.allocate(256).unwrap();
        assert_eq!(off, 0);
        assert_eq!(b.used(), 256);
        assert_eq!(b.allocation_count(), 1);
    }

    #[test]
    fn buddy_free_basic() {
        let mut b = BuddyAllocator::new(4096);
        let off = b.allocate(256).unwrap();
        assert!(b.free(off));
        assert_eq!(b.used(), 0);
        assert_eq!(b.allocation_count(), 0);
    }

    #[test]
    fn buddy_free_invalid() {
        let mut b = BuddyAllocator::new(4096);
        assert!(!b.free(9999));
    }

    #[test]
    fn buddy_allocate_zero_returns_none() {
        let mut b = BuddyAllocator::new(4096);
        assert!(b.allocate(0).is_none());
    }

    #[test]
    fn buddy_allocate_too_large() {
        let mut b = BuddyAllocator::new(4096);
        assert!(b.allocate(8192).is_none());
    }

    // -- Buddy split and coalesce -------------------------------------------

    #[test]
    fn buddy_split() {
        let mut b = BuddyAllocator::new(4096);
        // Allocating 256 from a 4096-capacity allocator causes multiple splits.
        let off = b.allocate(256).unwrap();
        assert_eq!(off, 0);
        // Capacity minus one 256-byte block = 3840 free.
        assert_eq!(b.free_bytes(), 4096 - 256);
    }

    #[test]
    fn buddy_coalesce_full() {
        let mut b = BuddyAllocator::new(4096);
        let a = b.allocate(256).unwrap();
        let c = b.allocate(256).unwrap();
        b.free(a);
        b.free(c);
        // After freeing both, everything should coalesce back to one block.
        assert_eq!(b.free_bytes(), 4096);
        assert_eq!(b.allocation_count(), 0);
    }

    #[test]
    fn buddy_coalesce_partial() {
        let mut b = BuddyAllocator::new(4096);
        let a = b.allocate(512).unwrap();
        let _c = b.allocate(512).unwrap();
        b.free(a);
        // Only one block freed; not fully coalesced.
        assert_eq!(b.free_bytes(), 4096 - 512);
    }

    #[test]
    fn buddy_multiple_allocations() {
        let mut b = BuddyAllocator::new(4096);
        let a = b.allocate(256).unwrap();
        let c = b.allocate(512).unwrap();
        let d = b.allocate(1024).unwrap();
        assert_eq!(b.allocation_count(), 3);
        // Each rounds up to power of 2 (already are).
        assert_eq!(b.used(), 256 + 512 + 1024);
        b.free(c);
        assert_eq!(b.allocation_count(), 2);
        b.free(a);
        b.free(d);
        assert_eq!(b.allocation_count(), 0);
    }

    #[test]
    fn buddy_rounds_up_non_pow2() {
        let mut b = BuddyAllocator::new(4096);
        let off = b.allocate(300).unwrap(); // rounds up to 512
        assert_eq!(b.used(), 512);
        b.free(off);
    }

    #[test]
    fn buddy_exhaust_capacity() {
        let mut b = BuddyAllocator::new(1024);
        let a = b.allocate(512).unwrap();
        let c = b.allocate(512).unwrap();
        assert!(b.allocate(256).is_none()); // full
        b.free(a);
        b.free(c);
    }

    // -- SubAllocator -------------------------------------------------------

    #[test]
    fn sub_allocator_basic() {
        let mut s = SubAllocator::new(4096);
        assert_eq!(s.buffer_size(), 4096);
        let off = s.allocate(100, SMALL_ALIGN).unwrap();
        assert_eq!(off, 0);
        assert_eq!(s.allocation_count(), 1);
    }

    #[test]
    fn sub_allocator_free() {
        let mut s = SubAllocator::new(4096);
        let off = s.allocate(100, SMALL_ALIGN).unwrap();
        assert!(s.free(off));
        assert_eq!(s.allocation_count(), 0);
        assert_eq!(s.free_bytes(), 4096);
    }

    #[test]
    fn sub_allocator_free_invalid() {
        let mut s = SubAllocator::new(4096);
        assert!(!s.free(9999));
    }

    #[test]
    fn sub_allocator_zero_returns_none() {
        let mut s = SubAllocator::new(4096);
        assert!(s.allocate(0, SMALL_ALIGN).is_none());
    }

    #[test]
    fn sub_allocator_coalesce() {
        let mut s = SubAllocator::new(4096);
        let a = s.allocate(1000, 1).unwrap();
        let b = s.allocate(1000, 1).unwrap();
        s.free(a);
        s.free(b);
        // After coalescing, should be able to allocate the full buffer.
        let big = s.allocate(4096, 1);
        assert!(big.is_some());
    }

    #[test]
    fn sub_allocator_alignment() {
        let mut s = SubAllocator::new(1_000_000);
        let off = s.allocate(100, 4096).unwrap();
        assert_eq!(off % 4096, 0, "allocation must be aligned to 4096");
    }

    #[test]
    fn sub_allocator_multiple() {
        let mut s = SubAllocator::new(10_000);
        let a = s.allocate(1000, 1).unwrap();
        let b = s.allocate(2000, 1).unwrap();
        let c = s.allocate(3000, 1).unwrap();
        assert_eq!(s.allocation_count(), 3);
        assert_eq!(s.used(), 6000);
        s.free(b);
        assert_eq!(s.allocation_count(), 2);
        assert_eq!(s.used(), 4000);
        s.free(a);
        s.free(c);
        assert_eq!(s.free_bytes(), 10_000);
    }

    #[test]
    fn sub_allocator_too_large() {
        let mut s = SubAllocator::new(1024);
        assert!(s.allocate(2048, 1).is_none());
    }

    // -- PoolConfig / MemoryPool -------------------------------------------

    #[test]
    fn pool_config_defaults() {
        let defaults = PoolConfig::a770_defaults();
        let gb = 1024 * 1024 * 1024u64;
        assert_eq!(defaults[&PoolKind::Weights].initial_size, 4 * gb);
        assert_eq!(defaults[&PoolKind::KvCache].initial_size, 4 * gb);
        assert_eq!(defaults[&PoolKind::Activations].initial_size, 2 * gb);
        assert_eq!(defaults[&PoolKind::Scratch].initial_size, 1 * gb);
    }

    #[test]
    fn pool_config_builder() {
        let cfg = PoolConfig::new(1024, 4096).with_growth_factor(1.5).with_alignment(512);
        assert_eq!(cfg.initial_size, 1024);
        assert_eq!(cfg.max_size, 4096);
        assert!((cfg.growth_factor - 1.5).abs() < f64::EPSILON);
        assert_eq!(cfg.alignment, 512);
    }

    #[test]
    fn memory_pool_allocate() {
        let cfg = PoolConfig::new(4096, 8192);
        let mut pool = MemoryPool::new(PoolKind::Scratch, cfg);
        let req = AllocationRequest::new(256, PoolKind::Scratch, "test");
        let off = pool.allocate(&req).unwrap();
        assert_eq!(off, 0);
        assert_eq!(pool.allocation_count(), 1);
    }

    #[test]
    fn memory_pool_wrong_kind() {
        let cfg = PoolConfig::new(4096, 8192);
        let mut pool = MemoryPool::new(PoolKind::Scratch, cfg);
        let req = AllocationRequest::new(256, PoolKind::Weights, "test");
        assert!(pool.allocate(&req).is_none());
    }

    #[test]
    fn memory_pool_free() {
        let cfg = PoolConfig::new(4096, 8192);
        let mut pool = MemoryPool::new(PoolKind::Scratch, cfg);
        let req = AllocationRequest::new(256, PoolKind::Scratch, "test");
        let off = pool.allocate(&req).unwrap();
        assert!(pool.free(off));
        assert_eq!(pool.allocation_count(), 0);
    }

    #[test]
    fn memory_pool_free_by_label() {
        let cfg = PoolConfig::new(4096, 8192);
        let mut pool = MemoryPool::new(PoolKind::Scratch, cfg);
        let req = AllocationRequest::new(256, PoolKind::Scratch, "my_buf");
        pool.allocate(&req).unwrap();
        assert!(pool.free_by_label("my_buf"));
        assert_eq!(pool.allocation_count(), 0);
    }

    #[test]
    fn memory_pool_isolation() {
        let cfg_w = PoolConfig::new(4096, 8192);
        let cfg_s = PoolConfig::new(4096, 8192);
        let mut weights = MemoryPool::new(PoolKind::Weights, cfg_w);
        let mut scratch = MemoryPool::new(PoolKind::Scratch, cfg_s);

        let req_w = AllocationRequest::new(256, PoolKind::Weights, "w1");
        let req_s = AllocationRequest::new(256, PoolKind::Scratch, "s1");

        // Can only allocate in the matching pool.
        assert!(weights.allocate(&req_w).is_some());
        assert!(scratch.allocate(&req_s).is_some());
        assert!(weights.allocate(&req_s).is_none());
        assert!(scratch.allocate(&req_w).is_none());
    }

    #[test]
    fn memory_pool_growth() {
        let cfg = PoolConfig::new(1024, 8192).with_growth_factor(2.0);
        let mut pool = MemoryPool::new(PoolKind::Scratch, cfg);
        let old_cap = pool.capacity();
        assert!(pool.try_grow());
        assert!(pool.capacity() > old_cap);
    }

    #[test]
    fn memory_pool_growth_denied_when_at_max() {
        let cfg = PoolConfig::new(4096, 4096).with_growth_factor(2.0);
        let mut pool = MemoryPool::new(PoolKind::Scratch, cfg);
        assert!(!pool.try_grow());
    }

    #[test]
    fn memory_pool_growth_denied_when_nonempty() {
        let cfg = PoolConfig::new(4096, 65536).with_growth_factor(2.0);
        let mut pool = MemoryPool::new(PoolKind::Scratch, cfg);
        let req = AllocationRequest::new(256, PoolKind::Scratch, "x");
        pool.allocate(&req).unwrap();
        assert!(!pool.try_grow());
    }

    // -- MemoryBudget -------------------------------------------------------

    #[test]
    fn budget_basic() {
        let mut b = MemoryBudget::new(10000);
        b.register_pool(PoolKind::Scratch, 5000);
        assert!(b.record_alloc(PoolKind::Scratch, 3000));
        assert_eq!(b.pool_used(PoolKind::Scratch), 3000);
        assert_eq!(b.used(), 3000);
        assert_eq!(b.free(), 7000);
    }

    #[test]
    fn budget_exceeds_pool_limit() {
        let mut b = MemoryBudget::new(100_000);
        b.register_pool(PoolKind::Scratch, 1000);
        assert!(!b.record_alloc(PoolKind::Scratch, 2000));
        assert_eq!(b.pool_used(PoolKind::Scratch), 0);
    }

    #[test]
    fn budget_exceeds_total() {
        let mut b = MemoryBudget::new(1000);
        b.register_pool(PoolKind::Scratch, 2000);
        assert!(!b.record_alloc(PoolKind::Scratch, 1500));
    }

    #[test]
    fn budget_free() {
        let mut b = MemoryBudget::new(10000);
        b.register_pool(PoolKind::Scratch, 5000);
        b.record_alloc(PoolKind::Scratch, 3000);
        b.record_free(PoolKind::Scratch, 3000);
        assert_eq!(b.pool_used(PoolKind::Scratch), 0);
    }

    #[test]
    fn budget_watermark() {
        let mut b = MemoryBudget::new(10000);
        b.register_pool(PoolKind::Scratch, 5000);
        b.record_alloc(PoolKind::Scratch, 4000);
        b.record_free(PoolKind::Scratch, 2000);
        assert_eq!(b.pool_watermark(PoolKind::Scratch), 4000);
    }

    #[test]
    fn budget_unregistered_pool() {
        let mut b = MemoryBudget::new(10000);
        assert!(!b.record_alloc(PoolKind::Weights, 100));
    }

    #[test]
    fn budget_multiple_pools() {
        let mut b = MemoryBudget::new(10000);
        b.register_pool(PoolKind::Weights, 5000);
        b.register_pool(PoolKind::Scratch, 3000);
        b.record_alloc(PoolKind::Weights, 2000);
        b.record_alloc(PoolKind::Scratch, 1000);
        assert_eq!(b.used(), 3000);
        assert_eq!(b.pool_used(PoolKind::Weights), 2000);
        assert_eq!(b.pool_used(PoolKind::Scratch), 1000);
    }

    // -- FragmentationAnalyzer ----------------------------------------------

    #[test]
    fn fragmentation_empty() {
        let s = SubAllocator::new(4096);
        let report = FragmentationAnalyzer::analyze(&s);
        assert!((report.external_fragmentation - 0.0).abs() < f64::EPSILON);
        assert_eq!(report.free_region_count, 1);
        assert_eq!(report.largest_free_block, 4096);
    }

    #[test]
    fn fragmentation_after_alloc_free() {
        let mut s = SubAllocator::new(4096);
        let a = s.allocate(1000, 1).unwrap();
        let _b = s.allocate(1000, 1).unwrap();
        let _c = s.allocate(1000, 1).unwrap();
        // Free middle block to create fragmentation.
        s.free(a);
        let report = FragmentationAnalyzer::analyze(&s);
        assert!(report.free_region_count >= 1);
        assert!(report.total_free > 0);
    }

    #[test]
    fn fragmentation_buddy() {
        let mut b = BuddyAllocator::new(4096);
        b.allocate(256).unwrap();
        let report = FragmentationAnalyzer::analyze_buddy(&b);
        assert!(report.total_free > 0);
        assert!(report.largest_free_block > 0);
    }

    #[test]
    fn fragmentation_external_increases_with_holes() {
        let mut s = SubAllocator::new(10_000);
        let a = s.allocate(1000, 1).unwrap();
        let b = s.allocate(1000, 1).unwrap();
        let c = s.allocate(1000, 1).unwrap();
        // Free a and c to create two non-adjacent holes.
        s.free(a);
        s.free(c);
        let report = FragmentationAnalyzer::analyze(&s);
        // At least two free regions.
        assert!(report.free_region_count >= 2);
        assert!(report.external_fragmentation > 0.0);
        // Clean up.
        s.free(b);
    }

    // -- OomHandler ---------------------------------------------------------

    #[test]
    fn oom_handler_default() {
        let handler = OomHandler::default();
        assert_eq!(handler.max_evictions, 3);
    }

    #[test]
    fn oom_handler_high_urgency() {
        let handler = OomHandler::new(2);
        let mut budget = MemoryBudget::new(10000);
        budget.register_pool(PoolKind::Scratch, 5000);
        budget.record_alloc(PoolKind::Scratch, 4500);

        let req =
            AllocationRequest::new(1000, PoolKind::Scratch, "big_buf").with_urgency(Urgency::High);
        let actions = handler.handle(&req, &budget);

        assert!(actions.iter().any(|a| matches!(a, OomAction::Defragment)));
        assert!(actions.iter().any(|a| matches!(a, OomAction::Fail(_))));
    }

    #[test]
    fn oom_handler_eviction() {
        let handler = OomHandler::new(2);
        let mut budget = MemoryBudget::new(10000);
        budget.register_pool(PoolKind::Activations, 5000);
        budget.record_alloc(PoolKind::Activations, 4500);

        let req = AllocationRequest::new(1000, PoolKind::Activations, "act");
        let actions = handler.handle(&req, &budget);

        assert!(actions.iter().any(|a| matches!(a, OomAction::Evict(PoolKind::Activations))));
    }

    #[test]
    fn oom_handler_scratch_eviction_on_high_urgency() {
        let handler = OomHandler::new(1);
        let mut budget = MemoryBudget::new(10000);
        budget.register_pool(PoolKind::Weights, 5000);
        budget.record_alloc(PoolKind::Weights, 4500);

        let req =
            AllocationRequest::new(1000, PoolKind::Weights, "w").with_urgency(Urgency::Critical);
        let actions = handler.handle(&req, &budget);

        assert!(actions.iter().any(|a| matches!(a, OomAction::Evict(PoolKind::Scratch))));
    }

    // -- AllocationRequest --------------------------------------------------

    #[test]
    fn alloc_request_auto_alignment_large() {
        let req = AllocationRequest::new(128 * 1024, PoolKind::Weights, "big");
        assert_eq!(req.alignment, LARGE_PAGE_ALIGN);
    }

    #[test]
    fn alloc_request_auto_alignment_small() {
        let req = AllocationRequest::new(100, PoolKind::Scratch, "tiny");
        assert_eq!(req.alignment, SMALL_ALIGN);
    }

    #[test]
    fn alloc_request_builder() {
        let req = AllocationRequest::new(1024, PoolKind::Scratch, "buf")
            .with_urgency(Urgency::High)
            .with_alignment(4096);
        assert_eq!(req.urgency, Urgency::High);
        assert_eq!(req.alignment, 4096);
    }

    // -- GpuMemoryAllocator -------------------------------------------------

    #[test]
    fn gpu_alloc_basic() {
        let cfg = small_test_configs();
        let mut alloc = GpuMemoryAllocator::with_configs(65536, cfg);
        let req = AllocationRequest::new(256, PoolKind::Scratch, "test");
        let off = alloc.allocate(&req).unwrap();
        assert!(alloc.free(PoolKind::Scratch, off, 256));
    }

    #[test]
    fn gpu_alloc_budget_tracking() {
        let cfg = small_test_configs();
        let mut alloc = GpuMemoryAllocator::with_configs(65536, cfg);
        let req = AllocationRequest::new(256, PoolKind::Scratch, "test");
        alloc.allocate(&req).unwrap();
        assert_eq!(alloc.budget().pool_used(PoolKind::Scratch), 256);
        assert!(alloc.budget().used() > 0);
    }

    #[test]
    fn gpu_alloc_oom() {
        let mut cfgs = HashMap::new();
        cfgs.insert(PoolKind::Scratch, PoolConfig::new(512, 512));
        let mut alloc = GpuMemoryAllocator::with_configs(1024, cfgs);
        // Fill the pool.
        let _ = alloc.allocate(&AllocationRequest::new(256, PoolKind::Scratch, "a"));
        let _ = alloc.allocate(&AllocationRequest::new(256, PoolKind::Scratch, "b"));
        // Next should OOM.
        let result = alloc.allocate(&AllocationRequest::new(256, PoolKind::Scratch, "c"));
        assert!(result.is_err());
    }

    #[test]
    fn gpu_alloc_pool_isolation() {
        let cfg = small_test_configs();
        let mut alloc = GpuMemoryAllocator::with_configs(65536, cfg);
        let req_w = AllocationRequest::new(256, PoolKind::Weights, "w1");
        let req_s = AllocationRequest::new(256, PoolKind::Scratch, "s1");
        alloc.allocate(&req_w).unwrap();
        alloc.allocate(&req_s).unwrap();
        assert_eq!(alloc.budget().pool_used(PoolKind::Weights), 256);
        assert_eq!(alloc.budget().pool_used(PoolKind::Scratch), 256);
    }

    #[test]
    fn gpu_alloc_debug_fmt() {
        let cfg = small_test_configs();
        let alloc = GpuMemoryAllocator::with_configs(65536, cfg);
        let dbg = format!("{:?}", alloc);
        assert!(dbg.contains("GpuMemoryAllocator"));
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn buddy_capacity_rounds_up() {
        let b = BuddyAllocator::new(3000);
        assert_eq!(b.capacity(), 4096);
    }

    #[test]
    fn buddy_min_capacity() {
        let b = BuddyAllocator::new(1);
        assert_eq!(b.capacity(), MIN_BLOCK_SIZE);
    }

    #[test]
    fn sub_allocator_rapid_alloc_free() {
        let mut s = SubAllocator::new(4096);
        for _ in 0..100 {
            let off = s.allocate(64, 1).unwrap();
            s.free(off);
        }
        assert_eq!(s.allocation_count(), 0);
        assert_eq!(s.free_bytes(), 4096);
    }

    #[test]
    fn buddy_rapid_alloc_free() {
        let mut b = BuddyAllocator::new(65536);
        for _ in 0..100 {
            let off = b.allocate(256).unwrap();
            b.free(off);
        }
        assert_eq!(b.allocation_count(), 0);
        assert_eq!(b.free_bytes(), 65536);
    }

    #[test]
    fn align_up_helper() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(100, 0), 100);
    }

    #[test]
    fn pool_kind_display() {
        assert_eq!(format!("{}", PoolKind::Weights), "Weights");
        assert_eq!(format!("{}", PoolKind::KvCache), "KvCache");
        assert_eq!(format!("{}", PoolKind::Activations), "Activations");
        assert_eq!(format!("{}", PoolKind::Scratch), "Scratch");
    }

    #[test]
    fn urgency_ordering() {
        assert!(Urgency::Low < Urgency::Normal);
        assert!(Urgency::Normal < Urgency::High);
        assert!(Urgency::High < Urgency::Critical);
    }

    #[test]
    fn budget_free_underflow_saturates() {
        let mut b = MemoryBudget::new(10000);
        b.register_pool(PoolKind::Scratch, 5000);
        b.record_alloc(PoolKind::Scratch, 100);
        b.record_free(PoolKind::Scratch, 500); // more than allocated
        assert_eq!(b.pool_used(PoolKind::Scratch), 0);
    }

    /// Property: total allocated never exceeds capacity.
    #[test]
    fn property_alloc_le_capacity() {
        let mut b = BuddyAllocator::new(4096);
        let mut offsets = Vec::new();
        // Keep allocating 256-byte blocks.
        while let Some(off) = b.allocate(256) {
            offsets.push(off);
        }
        assert!(b.used() <= b.capacity());
        // Verify count: 4096 / 256 = 16 blocks.
        assert_eq!(offsets.len(), 16);
        for off in offsets {
            b.free(off);
        }
        assert_eq!(b.used(), 0);
    }

    /// Property: sub-allocator used + free = buffer_size.
    #[test]
    fn property_sub_alloc_conservation() {
        let mut s = SubAllocator::new(10_000);
        s.allocate(1000, 1).unwrap();
        s.allocate(2000, 1).unwrap();
        s.allocate(3000, 1).unwrap();
        assert_eq!(s.used() + s.free_bytes(), s.buffer_size());
    }

    /// Multiple aligned sub-allocations all satisfy alignment.
    #[test]
    fn sub_alloc_multiple_aligned() {
        let mut s = SubAllocator::new(1_000_000);
        let mut offsets = Vec::new();
        for _ in 0..10 {
            let off = s.allocate(1000, 4096).unwrap();
            assert_eq!(off % 4096, 0);
            offsets.push(off);
        }
        for off in offsets {
            s.free(off);
        }
    }

    // -- Helper -------------------------------------------------------------

    fn small_test_configs() -> HashMap<PoolKind, PoolConfig> {
        let mut m = HashMap::new();
        m.insert(PoolKind::Weights, PoolConfig::new(8192, 16384));
        m.insert(PoolKind::KvCache, PoolConfig::new(8192, 16384));
        m.insert(PoolKind::Activations, PoolConfig::new(4096, 8192));
        m.insert(PoolKind::Scratch, PoolConfig::new(4096, 8192));
        m
    }
}
