//! OpenCL KV-cache paging / PagedAttention for A770 GPU inference.
//!
//! # Overview
//!
//! This module implements paged KV-cache management for efficient memory use
//! during long-sequence inference, inspired by vLLM's PagedAttention. Instead
//! of pre-allocating a contiguous buffer for the full sequence length, key and
//! value tensors are stored in fixed-size **pages** managed by a slab allocator.
//!
//! Key components:
//!
//! - **[`KvPage`]** — fixed-size page (default: 16 tokens × head_dim floats).
//! - **[`PageTable`]** — maps (layer, head, block_index) → `PageId`.
//! - **[`PageAllocator`]** — slab allocator with free-list.
//! - **[`PagedKvCache`]** — manages K and V caches with append / lookup / evict.
//! - **[`SequenceGroup`]** — groups sequences that share prefix pages (COW).
//! - **[`PageEvictionPolicy`]** — LRU, FIFO, or frequency-based eviction.
//! - **[`CopyOnWrite`]** — fork pages for shared prefixes, copy only on write.
//! - **[`PagingStats`]** — page utilization, fragmentation, eviction count.
//!
//! # A770 Tuning
//!
//! Page size is tuned for 64 KB SLM on Intel Arc A770:
//! 16 tokens × 64 head_dim × 4 bytes = 4 KB per page.
//! With ~3.5 GB budget for KV cache, that yields ~900 K pages.

use std::collections::{HashMap, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Constants — A770-tuned page geometry
// ---------------------------------------------------------------------------

/// Number of token positions stored in each page.
pub const TOKENS_PER_PAGE: usize = 16;

/// Default head dimension (matches common 2B-param models).
pub const DEFAULT_HEAD_DIM: usize = 64;

/// Bytes per element (f32).
pub const DTYPE_BYTES: usize = 4;

/// Page size in bytes: `TOKENS_PER_PAGE * DEFAULT_HEAD_DIM * DTYPE_BYTES` = 4 KB.
pub const PAGE_SIZE_BYTES: usize = TOKENS_PER_PAGE * DEFAULT_HEAD_DIM * DTYPE_BYTES;

/// A770 KV budget (~3.5 GB) expressed in pages.
pub const A770_MAX_PAGES: usize = (3_500_000_000_usize) / PAGE_SIZE_BYTES;

// ---------------------------------------------------------------------------
// PageId
// ---------------------------------------------------------------------------

/// Opaque identifier for a physical page in the allocator slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId(pub u32);

impl fmt::Display for PageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Page({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by paged KV-cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagingError {
    /// Allocator has no free pages.
    OutOfPages,
    /// Requested page id does not exist.
    InvalidPageId(PageId),
    /// Page table entry not found.
    PageTableMiss { layer: usize, head: usize, block_idx: usize },
    /// Sequence id not found.
    UnknownSequence(u64),
    /// Dimension mismatch between page and supplied data.
    DimensionMismatch { expected: usize, got: usize },
    /// Attempted to write into a read-only (shared) page without COW fork.
    SharedPageWrite(PageId),
}

impl fmt::Display for PagingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfPages => write!(f, "page allocator exhausted"),
            Self::InvalidPageId(id) => write!(f, "invalid page id: {id}"),
            Self::PageTableMiss { layer, head, block_idx } => {
                write!(f, "page table miss: layer={layer} head={head} block={block_idx}")
            }
            Self::UnknownSequence(id) => write!(f, "unknown sequence id: {id}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::SharedPageWrite(id) => {
                write!(f, "cannot write to shared page {id} without COW fork")
            }
        }
    }
}

impl std::error::Error for PagingError {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, PagingError>;

// ---------------------------------------------------------------------------
// KvPage
// ---------------------------------------------------------------------------

/// A fixed-size page holding `tokens_per_page × head_dim` f32 values.
///
/// Each page stores either key or value vectors for a contiguous block of
/// token positions within a single (layer, head) slice.
#[derive(Clone)]
pub struct KvPage {
    /// Backing storage: `[tokens_per_page * head_dim]` in row-major order.
    pub data: Vec<f32>,
    /// Number of tokens currently written (0..=tokens_per_page).
    pub len: usize,
    /// Tokens per page for this page.
    pub tokens_per_page: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl fmt::Debug for KvPage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvPage")
            .field("len", &self.len)
            .field("tokens_per_page", &self.tokens_per_page)
            .field("head_dim", &self.head_dim)
            .field("capacity_bytes", &(self.data.len() * DTYPE_BYTES))
            .finish()
    }
}

impl KvPage {
    /// Create a zero-initialised page.
    pub fn new(tokens_per_page: usize, head_dim: usize) -> Self {
        Self { data: vec![0.0; tokens_per_page * head_dim], len: 0, tokens_per_page, head_dim }
    }

    /// Number of remaining token slots.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.tokens_per_page - self.len
    }

    /// Whether the page is completely full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == self.tokens_per_page
    }

    /// Append a single token vector. Returns `Err` on dimension mismatch or
    /// if the page is full.
    pub fn append_token(&mut self, token_vec: &[f32]) -> Result<()> {
        if token_vec.len() != self.head_dim {
            return Err(PagingError::DimensionMismatch {
                expected: self.head_dim,
                got: token_vec.len(),
            });
        }
        if self.is_full() {
            return Err(PagingError::OutOfPages);
        }
        let offset = self.len * self.head_dim;
        self.data[offset..offset + self.head_dim].copy_from_slice(token_vec);
        self.len += 1;
        Ok(())
    }

    /// Read the vector at a given token position within this page.
    pub fn read_token(&self, pos: usize) -> Result<&[f32]> {
        if pos >= self.len {
            return Err(PagingError::DimensionMismatch { expected: self.len, got: pos + 1 });
        }
        let offset = pos * self.head_dim;
        Ok(&self.data[offset..offset + self.head_dim])
    }

    /// Clear all data and reset the length to zero.
    pub fn clear(&mut self) {
        self.data.fill(0.0);
        self.len = 0;
    }

    /// Size in bytes of the backing buffer.
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.data.len() * DTYPE_BYTES
    }
}

// ---------------------------------------------------------------------------
// PageAllocator
// ---------------------------------------------------------------------------

/// Slab allocator managing a pool of [`KvPage`]s with a free list.
pub struct PageAllocator {
    pages: Vec<KvPage>,
    free_list: VecDeque<PageId>,
    tokens_per_page: usize,
    head_dim: usize,
}

impl fmt::Debug for PageAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PageAllocator")
            .field("total_pages", &self.pages.len())
            .field("free_pages", &self.free_list.len())
            .finish()
    }
}

impl PageAllocator {
    /// Create an allocator with `num_pages` pre-allocated pages.
    pub fn new(num_pages: usize, tokens_per_page: usize, head_dim: usize) -> Self {
        let pages: Vec<KvPage> =
            (0..num_pages).map(|_| KvPage::new(tokens_per_page, head_dim)).collect();
        let free_list: VecDeque<PageId> = (0..num_pages as u32).map(PageId).collect();
        Self { pages, free_list, tokens_per_page, head_dim }
    }

    /// Allocate a page, returning its [`PageId`].
    pub fn allocate(&mut self) -> Result<PageId> {
        self.free_list.pop_front().ok_or(PagingError::OutOfPages)
    }

    /// Return a page to the free list and clear its data.
    pub fn deallocate(&mut self, id: PageId) -> Result<()> {
        let idx = id.0 as usize;
        if idx >= self.pages.len() {
            return Err(PagingError::InvalidPageId(id));
        }
        self.pages[idx].clear();
        self.free_list.push_back(id);
        Ok(())
    }

    /// Get a shared reference to a page.
    pub fn get(&self, id: PageId) -> Result<&KvPage> {
        self.pages.get(id.0 as usize).ok_or(PagingError::InvalidPageId(id))
    }

    /// Get a mutable reference to a page.
    pub fn get_mut(&mut self, id: PageId) -> Result<&mut KvPage> {
        let idx = id.0 as usize;
        self.pages.get_mut(idx).ok_or(PagingError::InvalidPageId(id))
    }

    /// Number of free pages.
    #[inline]
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of pages in the slab.
    #[inline]
    pub fn total_count(&self) -> usize {
        self.pages.len()
    }

    /// Number of currently allocated (in-use) pages.
    #[inline]
    pub fn allocated_count(&self) -> usize {
        self.pages.len() - self.free_list.len()
    }

    /// Tokens per page.
    #[inline]
    pub fn tokens_per_page(&self) -> usize {
        self.tokens_per_page
    }

    /// Head dimension.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Clone a page's data into a newly allocated page.
    pub fn clone_page(&mut self, src: PageId) -> Result<PageId> {
        let src_idx = src.0 as usize;
        if src_idx >= self.pages.len() {
            return Err(PagingError::InvalidPageId(src));
        }
        let cloned_data = self.pages[src_idx].data.clone();
        let cloned_len = self.pages[src_idx].len;
        let dst = self.allocate()?;
        let dst_page = &mut self.pages[dst.0 as usize];
        dst_page.data = cloned_data;
        dst_page.len = cloned_len;
        Ok(dst)
    }
}

// ---------------------------------------------------------------------------
// PageTable
// ---------------------------------------------------------------------------

/// Maps (layer, head, block_index) → [`PageId`].
///
/// `block_index` is the index of the page within the sequence for a given
/// (layer, head) pair: block 0 covers tokens 0..tokens_per_page, block 1
/// covers the next range, etc.
#[derive(Debug, Clone)]
pub struct PageTable {
    /// `table[layer][head]` is a `Vec<PageId>` of blocks in sequence order.
    table: Vec<Vec<Vec<PageId>>>,
    num_layers: usize,
    num_heads: usize,
}

impl PageTable {
    /// Create an empty page table for the given geometry.
    pub fn new(num_layers: usize, num_heads: usize) -> Self {
        let table = vec![vec![Vec::new(); num_heads]; num_layers];
        Self { table, num_layers, num_heads }
    }

    /// Append a page to the block list for `(layer, head)`.
    pub fn push(&mut self, layer: usize, head: usize, page_id: PageId) {
        self.table[layer][head].push(page_id);
    }

    /// Look up the page for a given block index.
    pub fn lookup(&self, layer: usize, head: usize, block_idx: usize) -> Result<PageId> {
        self.table
            .get(layer)
            .and_then(|heads| heads.get(head))
            .and_then(|blocks| blocks.get(block_idx))
            .copied()
            .ok_or(PagingError::PageTableMiss { layer, head, block_idx })
    }

    /// Number of blocks currently mapped for `(layer, head)`.
    pub fn block_count(&self, layer: usize, head: usize) -> usize {
        self.table.get(layer).and_then(|h| h.get(head)).map_or(0, Vec::len)
    }

    /// Collect all mapped page ids (for deallocation).
    pub fn all_page_ids(&self) -> Vec<PageId> {
        self.table
            .iter()
            .flat_map(|heads| heads.iter().flat_map(|blocks| blocks.iter()))
            .copied()
            .collect()
    }

    /// Remove and return the last block for `(layer, head)`.
    pub fn pop(&mut self, layer: usize, head: usize) -> Option<PageId> {
        self.table.get_mut(layer).and_then(|h| h.get_mut(head)).and_then(Vec::pop)
    }

    /// Replace the page id at a specific block index.
    pub fn replace(
        &mut self,
        layer: usize,
        head: usize,
        block_idx: usize,
        new_id: PageId,
    ) -> Result<PageId> {
        let slot = self
            .table
            .get_mut(layer)
            .and_then(|h| h.get_mut(head))
            .and_then(|blocks| blocks.get_mut(block_idx))
            .ok_or(PagingError::PageTableMiss { layer, head, block_idx })?;
        let old = *slot;
        *slot = new_id;
        Ok(old)
    }

    /// Number of layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of heads.
    #[inline]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Clear all mappings.
    pub fn clear(&mut self) {
        for heads in &mut self.table {
            for blocks in heads {
                blocks.clear();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PageEvictionPolicy
// ---------------------------------------------------------------------------

/// Eviction strategy for reclaiming pages when the allocator is exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionStrategy {
    /// Least-recently-used: evict the page accessed longest ago.
    Lru,
    /// First-in, first-out: evict the oldest allocated page.
    Fifo,
    /// Least-frequently-used: evict the page with fewest accesses.
    Frequency,
}

/// Tracks page access metadata for eviction decisions.
#[derive(Debug)]
pub struct PageEvictionPolicy {
    strategy: EvictionStrategy,
    /// Monotonically increasing access counter (serves as logical clock).
    clock: u64,
    /// Last access time per page.
    last_access: HashMap<PageId, u64>,
    /// Allocation order per page.
    alloc_order: HashMap<PageId, u64>,
    /// Access frequency per page.
    frequency: HashMap<PageId, u64>,
    /// Total number of evictions performed.
    eviction_count: u64,
}

impl PageEvictionPolicy {
    /// Create a new eviction policy with the given strategy.
    pub fn new(strategy: EvictionStrategy) -> Self {
        Self {
            strategy,
            clock: 0,
            last_access: HashMap::new(),
            alloc_order: HashMap::new(),
            frequency: HashMap::new(),
            eviction_count: 0,
        }
    }

    /// Record that a page was allocated.
    pub fn on_allocate(&mut self, id: PageId) {
        self.clock += 1;
        self.last_access.insert(id, self.clock);
        self.alloc_order.insert(id, self.clock);
        self.frequency.insert(id, 0);
    }

    /// Record that a page was accessed (read or write).
    pub fn on_access(&mut self, id: PageId) {
        self.clock += 1;
        self.last_access.insert(id, self.clock);
        *self.frequency.entry(id).or_insert(0) += 1;
    }

    /// Record that a page was deallocated.
    pub fn on_deallocate(&mut self, id: PageId) {
        self.last_access.remove(&id);
        self.alloc_order.remove(&id);
        self.frequency.remove(&id);
    }

    /// Select the best page to evict from the given candidate set.
    ///
    /// Returns `None` if `candidates` is empty.
    pub fn select_victim(&self, candidates: &[PageId]) -> Option<PageId> {
        if candidates.is_empty() {
            return None;
        }
        match self.strategy {
            EvictionStrategy::Lru => candidates
                .iter()
                .min_by_key(|id| self.last_access.get(id).copied().unwrap_or(0))
                .copied(),
            EvictionStrategy::Fifo => candidates
                .iter()
                .min_by_key(|id| self.alloc_order.get(id).copied().unwrap_or(0))
                .copied(),
            EvictionStrategy::Frequency => candidates
                .iter()
                .min_by_key(|id| self.frequency.get(id).copied().unwrap_or(0))
                .copied(),
        }
    }

    /// Increment eviction counter. Called after a successful eviction.
    pub fn record_eviction(&mut self) {
        self.eviction_count += 1;
    }

    /// Total evictions performed.
    #[inline]
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    /// Current strategy.
    #[inline]
    pub fn strategy(&self) -> EvictionStrategy {
        self.strategy
    }
}

// ---------------------------------------------------------------------------
// CopyOnWrite
// ---------------------------------------------------------------------------

/// Copy-on-write manager for shared prefix pages.
///
/// Tracks reference counts on pages. When a page has refcount > 1 and a write
/// is requested, the page is duplicated (forked) so that the writer gets an
/// exclusive copy.
#[derive(Debug)]
pub struct CopyOnWrite {
    /// Reference count per page id.
    refcounts: HashMap<PageId, u32>,
    /// Total COW forks performed (for stats).
    fork_count: u64,
    /// Total bytes saved by sharing (approximate).
    bytes_saved: u64,
    /// Page size in bytes (for stats calculation).
    page_size_bytes: usize,
}

impl CopyOnWrite {
    /// Create a new COW manager.
    pub fn new(page_size_bytes: usize) -> Self {
        Self { refcounts: HashMap::new(), fork_count: 0, bytes_saved: 0, page_size_bytes }
    }

    /// Increment the reference count for a page (sharing).
    pub fn share(&mut self, id: PageId) {
        let rc = self.refcounts.entry(id).or_insert(1);
        *rc += 1;
        self.bytes_saved += self.page_size_bytes as u64;
    }

    /// Decrement the reference count. Returns `true` if the page is now
    /// unreferenced (refcount == 0) and can be freed.
    pub fn release(&mut self, id: PageId) -> bool {
        if let Some(rc) = self.refcounts.get_mut(&id) {
            *rc = rc.saturating_sub(1);
            if *rc == 0 {
                self.refcounts.remove(&id);
                return true;
            }
        }
        false
    }

    /// Check whether a page is shared (refcount > 1).
    #[inline]
    pub fn is_shared(&self, id: PageId) -> bool {
        self.refcounts.get(&id).copied().unwrap_or(1) > 1
    }

    /// Reference count for a page (1 if not tracked).
    #[inline]
    pub fn refcount(&self, id: PageId) -> u32 {
        self.refcounts.get(&id).copied().unwrap_or(1)
    }

    /// Record a COW fork event.
    pub fn record_fork(&mut self) {
        self.fork_count += 1;
    }

    /// Total COW forks performed.
    #[inline]
    pub fn fork_count(&self) -> u64 {
        self.fork_count
    }

    /// Approximate bytes saved by sharing pages.
    #[inline]
    pub fn bytes_saved(&self) -> u64 {
        self.bytes_saved
    }
}

// ---------------------------------------------------------------------------
// PagingStats
// ---------------------------------------------------------------------------

/// Statistics for page utilization, fragmentation, and eviction.
#[derive(Debug, Clone, PartialEq)]
pub struct PagingStats {
    /// Total pages in the allocator.
    pub total_pages: usize,
    /// Currently allocated pages.
    pub allocated_pages: usize,
    /// Free pages.
    pub free_pages: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented).
    pub fragmentation: f64,
    /// Total evictions since creation.
    pub eviction_count: u64,
    /// Total COW forks.
    pub cow_forks: u64,
    /// Bytes saved by COW sharing.
    pub cow_bytes_saved: u64,
    /// Page utilization ratio (0.0 – 1.0): fraction of allocated page slots
    /// that actually hold token data.
    pub utilization: f64,
}

impl fmt::Display for PagingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "pages={}/{} free={} frag={:.2}% util={:.1}% evictions={} cow_forks={} cow_saved={}B",
            self.allocated_pages,
            self.total_pages,
            self.free_pages,
            self.fragmentation * 100.0,
            self.utilization * 100.0,
            self.eviction_count,
            self.cow_forks,
            self.cow_bytes_saved,
        )
    }
}

// ---------------------------------------------------------------------------
// SequenceGroup
// ---------------------------------------------------------------------------

/// A group of sequences that may share prefix pages via copy-on-write.
///
/// Each sequence has its own K and V [`PageTable`], but prefix pages can be
/// shared across sequences within the group until one of them diverges.
#[derive(Debug)]
pub struct SequenceGroup {
    /// Unique id for this group.
    pub group_id: u64,
    /// Member sequence ids.
    pub sequence_ids: Vec<u64>,
    /// Number of shared prefix tokens across all members.
    pub shared_prefix_len: usize,
}

impl SequenceGroup {
    /// Create a new group with a single founding sequence.
    pub fn new(group_id: u64, founder_seq_id: u64) -> Self {
        Self { group_id, sequence_ids: vec![founder_seq_id], shared_prefix_len: 0 }
    }

    /// Add a sequence that shares the current prefix.
    pub fn add_sequence(&mut self, seq_id: u64) {
        if !self.sequence_ids.contains(&seq_id) {
            self.sequence_ids.push(seq_id);
        }
    }

    /// Remove a sequence from the group.
    pub fn remove_sequence(&mut self, seq_id: u64) {
        self.sequence_ids.retain(|&id| id != seq_id);
    }

    /// Number of member sequences.
    #[inline]
    pub fn len(&self) -> usize {
        self.sequence_ids.len()
    }

    /// Whether the group is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sequence_ids.is_empty()
    }
}

// ---------------------------------------------------------------------------
// SequenceState (internal per-sequence bookkeeping)
// ---------------------------------------------------------------------------

/// Internal state for a single sequence inside [`PagedKvCache`].
#[derive(Debug)]
struct SequenceState {
    /// K-cache page table.
    k_table: PageTable,
    /// V-cache page table.
    v_table: PageTable,
    /// Total tokens appended so far.
    token_count: usize,
    /// Optional group membership.
    group_id: Option<u64>,
}

// ---------------------------------------------------------------------------
// PagedKvCache
// ---------------------------------------------------------------------------

/// Paged KV-cache manager.
///
/// Manages key and value caches for multiple sequences using paged memory,
/// supporting append, lookup, eviction, and copy-on-write sharing.
pub struct PagedKvCache {
    allocator: PageAllocator,
    sequences: HashMap<u64, SequenceState>,
    groups: HashMap<u64, SequenceGroup>,
    eviction: PageEvictionPolicy,
    cow: CopyOnWrite,
    num_layers: usize,
    num_heads: usize,
}

impl fmt::Debug for PagedKvCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PagedKvCache")
            .field("num_layers", &self.num_layers)
            .field("num_heads", &self.num_heads)
            .field("sequences", &self.sequences.len())
            .field("groups", &self.groups.len())
            .field("allocator", &self.allocator)
            .finish()
    }
}

impl PagedKvCache {
    /// Create a new paged KV cache.
    pub fn new(
        num_pages: usize,
        tokens_per_page: usize,
        head_dim: usize,
        num_layers: usize,
        num_heads: usize,
        eviction_strategy: EvictionStrategy,
    ) -> Self {
        let page_bytes = tokens_per_page * head_dim * DTYPE_BYTES;
        Self {
            allocator: PageAllocator::new(num_pages, tokens_per_page, head_dim),
            sequences: HashMap::new(),
            groups: HashMap::new(),
            eviction: PageEvictionPolicy::new(eviction_strategy),
            cow: CopyOnWrite::new(page_bytes),
            num_layers,
            num_heads,
        }
    }

    /// Register a new sequence.
    pub fn add_sequence(&mut self, seq_id: u64) {
        self.sequences.entry(seq_id).or_insert_with(|| SequenceState {
            k_table: PageTable::new(self.num_layers, self.num_heads),
            v_table: PageTable::new(self.num_layers, self.num_heads),
            token_count: 0,
            group_id: None,
        });
    }

    /// Remove a sequence, deallocating its non-shared pages.
    pub fn remove_sequence(&mut self, seq_id: u64) -> Result<()> {
        let state = self.sequences.remove(&seq_id).ok_or(PagingError::UnknownSequence(seq_id))?;

        // Deallocate K pages
        for page_id in state.k_table.all_page_ids() {
            if self.cow.release(page_id) {
                self.eviction.on_deallocate(page_id);
                let _ = self.allocator.deallocate(page_id);
            }
        }
        // Deallocate V pages
        for page_id in state.v_table.all_page_ids() {
            if self.cow.release(page_id) {
                self.eviction.on_deallocate(page_id);
                let _ = self.allocator.deallocate(page_id);
            }
        }

        // Remove from group if applicable
        if let Some(gid) = state.group_id
            && let Some(group) = self.groups.get_mut(&gid)
        {
            group.remove_sequence(seq_id);
            if group.is_empty() {
                self.groups.remove(&gid);
            }
        }

        Ok(())
    }

    /// Append a token's K and V vectors for all layers and heads.
    ///
    /// `k_data` and `v_data` are shaped `[num_layers][num_heads][head_dim]`
    /// stored as flat slices in row-major order.
    pub fn append_token(&mut self, seq_id: u64, k_data: &[f32], v_data: &[f32]) -> Result<()> {
        let expected = self.num_layers * self.num_heads * self.allocator.head_dim();
        if k_data.len() != expected {
            return Err(PagingError::DimensionMismatch { expected, got: k_data.len() });
        }
        if v_data.len() != expected {
            return Err(PagingError::DimensionMismatch { expected, got: v_data.len() });
        }

        let head_dim = self.allocator.head_dim();
        let tpp = self.allocator.tokens_per_page();

        // Determine which block this token falls into and its offset within.
        let seq = self.sequences.get(&seq_id).ok_or(PagingError::UnknownSequence(seq_id))?;
        let token_idx = seq.token_count;
        let block_idx = token_idx / tpp;

        for layer in 0..self.num_layers {
            for head in 0..self.num_heads {
                let flat_offset = (layer * self.num_heads + head) * head_dim;
                let k_vec = &k_data[flat_offset..flat_offset + head_dim];
                let v_vec = &v_data[flat_offset..flat_offset + head_dim];

                // Allocate new page if we need a new block.
                let seq = self.sequences.get(&seq_id).unwrap();
                let need_new_block = seq.k_table.block_count(layer, head) <= block_idx;

                if need_new_block {
                    let k_page_id = self.allocator.allocate()?;
                    let v_page_id = self.allocator.allocate()?;
                    self.eviction.on_allocate(k_page_id);
                    self.eviction.on_allocate(v_page_id);
                    let seq = self.sequences.get_mut(&seq_id).unwrap();
                    seq.k_table.push(layer, head, k_page_id);
                    seq.v_table.push(layer, head, v_page_id);
                }

                // Write K
                let seq = self.sequences.get(&seq_id).unwrap();
                let k_page_id = seq.k_table.lookup(layer, head, block_idx)?;
                self.eviction.on_access(k_page_id);
                // COW check
                if self.cow.is_shared(k_page_id) {
                    let new_id = self.allocator.clone_page(k_page_id)?;
                    self.cow.record_fork();
                    self.eviction.on_allocate(new_id);
                    let seq = self.sequences.get_mut(&seq_id).unwrap();
                    seq.k_table.replace(layer, head, block_idx, new_id)?;
                    self.allocator.get_mut(new_id)?.append_token(k_vec)?;
                } else {
                    self.allocator.get_mut(k_page_id)?.append_token(k_vec)?;
                }

                // Write V
                let seq = self.sequences.get(&seq_id).unwrap();
                let v_page_id = seq.v_table.lookup(layer, head, block_idx)?;
                self.eviction.on_access(v_page_id);
                if self.cow.is_shared(v_page_id) {
                    let new_id = self.allocator.clone_page(v_page_id)?;
                    self.cow.record_fork();
                    self.eviction.on_allocate(new_id);
                    let seq = self.sequences.get_mut(&seq_id).unwrap();
                    seq.v_table.replace(layer, head, block_idx, new_id)?;
                    self.allocator.get_mut(new_id)?.append_token(v_vec)?;
                } else {
                    self.allocator.get_mut(v_page_id)?.append_token(v_vec)?;
                }
            }
        }

        let seq = self.sequences.get_mut(&seq_id).unwrap();
        seq.token_count += 1;
        Ok(())
    }

    /// Look up the K vector for a specific (seq, layer, head, token_pos).
    pub fn lookup_k(
        &mut self,
        seq_id: u64,
        layer: usize,
        head: usize,
        token_pos: usize,
    ) -> Result<Vec<f32>> {
        let tpp = self.allocator.tokens_per_page();
        let block_idx = token_pos / tpp;
        let offset_in_block = token_pos % tpp;

        let seq = self.sequences.get(&seq_id).ok_or(PagingError::UnknownSequence(seq_id))?;
        let page_id = seq.k_table.lookup(layer, head, block_idx)?;
        self.eviction.on_access(page_id);
        let page = self.allocator.get(page_id)?;
        Ok(page.read_token(offset_in_block)?.to_vec())
    }

    /// Look up the V vector for a specific (seq, layer, head, token_pos).
    pub fn lookup_v(
        &mut self,
        seq_id: u64,
        layer: usize,
        head: usize,
        token_pos: usize,
    ) -> Result<Vec<f32>> {
        let tpp = self.allocator.tokens_per_page();
        let block_idx = token_pos / tpp;
        let offset_in_block = token_pos % tpp;

        let seq = self.sequences.get(&seq_id).ok_or(PagingError::UnknownSequence(seq_id))?;
        let page_id = seq.v_table.lookup(layer, head, block_idx)?;
        self.eviction.on_access(page_id);
        let page = self.allocator.get(page_id)?;
        Ok(page.read_token(offset_in_block)?.to_vec())
    }

    /// Evict pages from a sequence to free `count` pages.
    ///
    /// Evicts from the tail (most recent blocks) of the given sequence.
    pub fn evict_pages(&mut self, seq_id: u64, count: usize) -> Result<usize> {
        let seq = self.sequences.get_mut(&seq_id).ok_or(PagingError::UnknownSequence(seq_id))?;
        let mut freed = 0;
        for _ in 0..count {
            let mut evicted_any = false;
            for layer in (0..self.num_layers).rev() {
                for head in (0..self.num_heads).rev() {
                    if let Some(k_id) = seq.k_table.pop(layer, head) {
                        self.eviction.on_deallocate(k_id);
                        self.eviction.record_eviction();
                        let _ = self.allocator.deallocate(k_id);
                        freed += 1;
                        evicted_any = true;
                    }
                    if let Some(v_id) = seq.v_table.pop(layer, head) {
                        self.eviction.on_deallocate(v_id);
                        self.eviction.record_eviction();
                        let _ = self.allocator.deallocate(v_id);
                        freed += 1;
                        evicted_any = true;
                    }
                }
            }
            if !evicted_any {
                break;
            }
        }
        Ok(freed)
    }

    /// Evict a single victim page chosen by the eviction policy from all
    /// allocated pages across all sequences. Returns the freed [`PageId`].
    pub fn evict_one_by_policy(&mut self) -> Result<PageId> {
        // Gather all allocated page ids across all sequences
        let candidates: Vec<PageId> = self
            .sequences
            .values()
            .flat_map(|s| s.k_table.all_page_ids().into_iter().chain(s.v_table.all_page_ids()))
            .collect();

        let victim = self.eviction.select_victim(&candidates).ok_or(PagingError::OutOfPages)?;

        // Remove from whichever sequence/table holds it
        for seq in self.sequences.values_mut() {
            for layer in 0..self.num_layers {
                for head in 0..self.num_heads {
                    let k_ids: Vec<_> = (0..seq.k_table.block_count(layer, head))
                        .filter_map(|b| seq.k_table.lookup(layer, head, b).ok())
                        .collect();
                    for (b, &kid) in k_ids.iter().enumerate() {
                        if kid == victim {
                            seq.k_table.pop(layer, head);
                            // Shift might be needed, but pop removes last.
                            // For simplicity we only evict tail pages.
                            let _ = b; // suppress unused
                        }
                    }
                }
            }
        }

        self.eviction.on_deallocate(victim);
        self.eviction.record_eviction();
        self.allocator.deallocate(victim)?;
        Ok(victim)
    }

    /// Create a group and register the founding sequence.
    pub fn create_group(&mut self, group_id: u64, founder_seq_id: u64) -> Result<()> {
        if !self.sequences.contains_key(&founder_seq_id) {
            return Err(PagingError::UnknownSequence(founder_seq_id));
        }
        self.groups.insert(group_id, SequenceGroup::new(group_id, founder_seq_id));
        self.sequences.get_mut(&founder_seq_id).unwrap().group_id = Some(group_id);
        Ok(())
    }

    /// Fork a sequence: the new sequence shares prefix pages via COW.
    pub fn fork_sequence(&mut self, source_seq_id: u64, new_seq_id: u64) -> Result<()> {
        let source = self
            .sequences
            .get(&source_seq_id)
            .ok_or(PagingError::UnknownSequence(source_seq_id))?;

        let k_table = source.k_table.clone();
        let v_table = source.v_table.clone();
        let token_count = source.token_count;
        let group_id = source.group_id;

        // Mark all pages as shared
        for page_id in k_table.all_page_ids() {
            self.cow.share(page_id);
        }
        for page_id in v_table.all_page_ids() {
            self.cow.share(page_id);
        }

        self.sequences
            .insert(new_seq_id, SequenceState { k_table, v_table, token_count, group_id });

        if let Some(gid) = group_id
            && let Some(group) = self.groups.get_mut(&gid)
        {
            group.add_sequence(new_seq_id);
        }

        Ok(())
    }

    /// Collect paging statistics.
    pub fn stats(&self) -> PagingStats {
        let total = self.allocator.total_count();
        let allocated = self.allocator.allocated_count();
        let free = self.allocator.free_count();

        // Utilization: fraction of allocated page token-slots that are used.
        let tpp = self.allocator.tokens_per_page();
        let mut total_slots = 0usize;
        let mut used_slots = 0usize;
        for seq in self.sequences.values() {
            for page_id in seq.k_table.all_page_ids() {
                if let Ok(page) = self.allocator.get(page_id) {
                    total_slots += tpp;
                    used_slots += page.len;
                }
            }
            for page_id in seq.v_table.all_page_ids() {
                if let Ok(page) = self.allocator.get(page_id) {
                    total_slots += tpp;
                    used_slots += page.len;
                }
            }
        }
        let utilization =
            if total_slots > 0 { used_slots as f64 / total_slots as f64 } else { 0.0 };

        // Fragmentation: 1 - (free_pages / total_pages) when pages are
        // allocated but partially filled.
        let frag = if allocated > 0 { 1.0 - utilization } else { 0.0 };

        PagingStats {
            total_pages: total,
            allocated_pages: allocated,
            free_pages: free,
            fragmentation: frag,
            eviction_count: self.eviction.eviction_count(),
            cow_forks: self.cow.fork_count(),
            cow_bytes_saved: self.cow.bytes_saved(),
            utilization,
        }
    }

    /// Number of tokens stored for a sequence.
    pub fn sequence_token_count(&self, seq_id: u64) -> Result<usize> {
        self.sequences
            .get(&seq_id)
            .map(|s| s.token_count)
            .ok_or(PagingError::UnknownSequence(seq_id))
    }

    /// Reference to the allocator (for advanced queries).
    pub fn allocator(&self) -> &PageAllocator {
        &self.allocator
    }

    /// Reference to the COW manager.
    pub fn cow(&self) -> &CopyOnWrite {
        &self.cow
    }

    /// Reference to the eviction policy.
    pub fn eviction_policy(&self) -> &PageEvictionPolicy {
        &self.eviction
    }
}

// ---------------------------------------------------------------------------
// CPU reference: paged attention lookup
// ---------------------------------------------------------------------------

/// CPU reference implementation of paged attention for verification.
///
/// Computes `softmax(Q @ K^T / sqrt(head_dim)) @ V` using paged K/V data.
///
/// - `query`: `[head_dim]` — single query vector for one head.
/// - `k_pages` / `v_pages`: list of `(page_data, token_count)` pairs in
///   sequence order.
/// - `head_dim`: dimension per head.
///
/// Returns the attention output vector `[head_dim]`.
pub fn cpu_paged_attention(
    query: &[f32],
    k_pages: &[(&[f32], usize)],
    v_pages: &[(&[f32], usize)],
    head_dim: usize,
) -> Result<Vec<f32>> {
    if query.len() != head_dim {
        return Err(PagingError::DimensionMismatch { expected: head_dim, got: query.len() });
    }

    // Gather total token count
    let total_tokens: usize = k_pages.iter().map(|(_, n)| *n).sum();
    if total_tokens == 0 {
        return Ok(vec![0.0; head_dim]);
    }

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Compute Q @ K^T scores
    let mut scores = Vec::with_capacity(total_tokens);
    for &(k_data, count) in k_pages {
        for t in 0..count {
            let k_offset = t * head_dim;
            let k_vec = &k_data[k_offset..k_offset + head_dim];
            let dot: f32 = query.iter().zip(k_vec.iter()).map(|(q, k)| q * k).sum();
            scores.push(dot * scale);
        }
    }

    // Softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp: f32 = exp_scores.iter().sum();
    let weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

    // Weighted sum of V
    let mut output = vec![0.0; head_dim];
    let mut token_idx = 0;
    for &(v_data, count) in v_pages {
        for t in 0..count {
            let v_offset = t * head_dim;
            let v_vec = &v_data[v_offset..v_offset + head_dim];
            let w = weights[token_idx];
            for (o, &v) in output.iter_mut().zip(v_vec.iter()) {
                *o += w * v;
            }
            token_idx += 1;
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// OpenCL kernel source for paged attention lookup
// ---------------------------------------------------------------------------

/// Embedded OpenCL C kernel for paged attention on Intel Arc A770.
///
/// This kernel reads K and V data from paged buffers, computes scaled
/// dot-product attention for a single query head, and writes the output.
/// Designed for 64 KB SLM with 4 KB pages (16 tokens × 64 head_dim × f32).
pub const PAGED_ATTENTION_CL: &str = r#"
// Paged attention kernel for Intel Arc A770 (OpenCL 3.0).
// Each work-group handles one query head.
// Pages are 16 tokens × head_dim floats laid out contiguously.

__kernel void paged_attention(
    __global const float* query,        // [num_heads, head_dim]
    __global const float* k_pages,      // [num_pages, tokens_per_page, head_dim]
    __global const float* v_pages,      // [num_pages, tokens_per_page, head_dim]
    __global const int*   page_table,   // [num_heads, max_blocks] page indices
    __global const int*   page_lengths, // [num_heads, max_blocks] valid tokens per page
    __global float*       output,       // [num_heads, head_dim]
    const int head_dim,
    const int tokens_per_page,
    const int max_blocks,
    const int num_heads
) {
    int head = get_global_id(0);
    if (head >= num_heads) return;

    float scale = 1.0f / sqrt((float)head_dim);

    // --- Phase 1: compute Q @ K^T scores ---
    float max_score = -1e30f;
    int total_tokens = 0;

    // First pass: find max score for numerical stability
    for (int b = 0; b < max_blocks; b++) {
        int page_idx = page_table[head * max_blocks + b];
        if (page_idx < 0) break;
        int count = page_lengths[head * max_blocks + b];
        for (int t = 0; t < count; t++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float q = query[head * head_dim + d];
                float k = k_pages[page_idx * tokens_per_page * head_dim + t * head_dim + d];
                dot += q * k;
            }
            dot *= scale;
            if (dot > max_score) max_score = dot;
            total_tokens++;
        }
    }

    if (total_tokens == 0) {
        for (int d = 0; d < head_dim; d++) {
            output[head * head_dim + d] = 0.0f;
        }
        return;
    }

    // --- Phase 2: softmax weights and weighted V sum ---
    float sum_exp = 0.0f;
    // Accumulate output in registers
    float acc[256]; // max head_dim assumed <= 256
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;

    for (int b = 0; b < max_blocks; b++) {
        int page_idx = page_table[head * max_blocks + b];
        if (page_idx < 0) break;
        int count = page_lengths[head * max_blocks + b];
        for (int t = 0; t < count; t++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float q = query[head * head_dim + d];
                float k = k_pages[page_idx * tokens_per_page * head_dim + t * head_dim + d];
                dot += q * k;
            }
            dot *= scale;
            float w = exp(dot - max_score);
            sum_exp += w;
            for (int d = 0; d < head_dim; d++) {
                float v = v_pages[page_idx * tokens_per_page * head_dim + t * head_dim + d];
                acc[d] += w * v;
            }
        }
    }

    for (int d = 0; d < head_dim; d++) {
        output[head * head_dim + d] = acc[d] / sum_exp;
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a small cache for testing.
    fn test_cache(num_pages: usize, tokens_per_page: usize, head_dim: usize) -> PagedKvCache {
        PagedKvCache::new(num_pages, tokens_per_page, head_dim, 1, 1, EvictionStrategy::Lru)
    }

    // -----------------------------------------------------------------------
    // KvPage tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_kv_page_new() {
        let page = KvPage::new(16, 64);
        assert_eq!(page.len, 0);
        assert_eq!(page.remaining(), 16);
        assert!(!page.is_full());
        assert_eq!(page.size_bytes(), 16 * 64 * 4);
    }

    #[test]
    fn test_kv_page_append_and_read() {
        let mut page = KvPage::new(4, 8);
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        page.append_token(&v).unwrap();
        assert_eq!(page.len, 1);
        assert_eq!(page.read_token(0).unwrap(), &v[..]);
    }

    #[test]
    fn test_kv_page_fill_to_capacity() {
        let mut page = KvPage::new(4, 2);
        for i in 0..4 {
            page.append_token(&[i as f32, i as f32 + 0.5]).unwrap();
        }
        assert!(page.is_full());
        assert_eq!(page.remaining(), 0);
        assert!(page.append_token(&[99.0, 99.0]).is_err());
    }

    #[test]
    fn test_kv_page_dimension_mismatch() {
        let mut page = KvPage::new(4, 8);
        let err = page.append_token(&[1.0, 2.0]).unwrap_err();
        assert_eq!(err, PagingError::DimensionMismatch { expected: 8, got: 2 });
    }

    #[test]
    fn test_kv_page_read_out_of_bounds() {
        let page = KvPage::new(4, 2);
        assert!(page.read_token(0).is_err());
    }

    #[test]
    fn test_kv_page_clear() {
        let mut page = KvPage::new(4, 2);
        page.append_token(&[1.0, 2.0]).unwrap();
        page.clear();
        assert_eq!(page.len, 0);
        assert_eq!(page.data.iter().sum::<f32>(), 0.0);
    }

    // -----------------------------------------------------------------------
    // PageAllocator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_allocator_new() {
        let alloc = PageAllocator::new(10, 16, 64);
        assert_eq!(alloc.total_count(), 10);
        assert_eq!(alloc.free_count(), 10);
        assert_eq!(alloc.allocated_count(), 0);
    }

    #[test]
    fn test_allocator_allocate_and_free() {
        let mut alloc = PageAllocator::new(4, 4, 2);
        let p0 = alloc.allocate().unwrap();
        let p1 = alloc.allocate().unwrap();
        assert_eq!(alloc.free_count(), 2);
        assert_eq!(alloc.allocated_count(), 2);

        alloc.deallocate(p0).unwrap();
        assert_eq!(alloc.free_count(), 3);

        alloc.deallocate(p1).unwrap();
        assert_eq!(alloc.free_count(), 4);
    }

    #[test]
    fn test_allocator_exhaust() {
        let mut alloc = PageAllocator::new(2, 4, 2);
        alloc.allocate().unwrap();
        alloc.allocate().unwrap();
        assert_eq!(alloc.allocate().unwrap_err(), PagingError::OutOfPages);
    }

    #[test]
    fn test_allocator_deallocate_invalid() {
        let mut alloc = PageAllocator::new(2, 4, 2);
        let err = alloc.deallocate(PageId(99)).unwrap_err();
        assert_eq!(err, PagingError::InvalidPageId(PageId(99)));
    }

    #[test]
    fn test_allocator_clone_page() {
        let mut alloc = PageAllocator::new(4, 2, 2);
        let src = alloc.allocate().unwrap();
        alloc.get_mut(src).unwrap().append_token(&[1.0, 2.0]).unwrap();
        let dst = alloc.clone_page(src).unwrap();
        assert_ne!(src, dst);
        assert_eq!(alloc.get(dst).unwrap().read_token(0).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn test_allocator_allocate_free_all_available() {
        let n = 32;
        let mut alloc = PageAllocator::new(n, 4, 2);
        let mut ids = Vec::new();
        for _ in 0..n {
            ids.push(alloc.allocate().unwrap());
        }
        assert_eq!(alloc.free_count(), 0);
        for id in ids {
            alloc.deallocate(id).unwrap();
        }
        assert_eq!(alloc.free_count(), n);
    }

    // -----------------------------------------------------------------------
    // PageTable tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_page_table_push_and_lookup() {
        let mut pt = PageTable::new(2, 2);
        pt.push(0, 0, PageId(10));
        pt.push(0, 0, PageId(20));
        assert_eq!(pt.lookup(0, 0, 0).unwrap(), PageId(10));
        assert_eq!(pt.lookup(0, 0, 1).unwrap(), PageId(20));
    }

    #[test]
    fn test_page_table_miss() {
        let pt = PageTable::new(1, 1);
        assert_eq!(
            pt.lookup(0, 0, 0).unwrap_err(),
            PagingError::PageTableMiss { layer: 0, head: 0, block_idx: 0 }
        );
    }

    #[test]
    fn test_page_table_block_count() {
        let mut pt = PageTable::new(1, 1);
        assert_eq!(pt.block_count(0, 0), 0);
        pt.push(0, 0, PageId(1));
        pt.push(0, 0, PageId(2));
        assert_eq!(pt.block_count(0, 0), 2);
    }

    #[test]
    fn test_page_table_all_page_ids() {
        let mut pt = PageTable::new(2, 2);
        pt.push(0, 0, PageId(1));
        pt.push(0, 1, PageId(2));
        pt.push(1, 0, PageId(3));
        let mut ids = pt.all_page_ids();
        ids.sort();
        assert_eq!(ids, vec![PageId(1), PageId(2), PageId(3)]);
    }

    #[test]
    fn test_page_table_pop() {
        let mut pt = PageTable::new(1, 1);
        pt.push(0, 0, PageId(5));
        pt.push(0, 0, PageId(6));
        assert_eq!(pt.pop(0, 0), Some(PageId(6)));
        assert_eq!(pt.pop(0, 0), Some(PageId(5)));
        assert_eq!(pt.pop(0, 0), None);
    }

    #[test]
    fn test_page_table_replace() {
        let mut pt = PageTable::new(1, 1);
        pt.push(0, 0, PageId(10));
        let old = pt.replace(0, 0, 0, PageId(20)).unwrap();
        assert_eq!(old, PageId(10));
        assert_eq!(pt.lookup(0, 0, 0).unwrap(), PageId(20));
    }

    #[test]
    fn test_page_table_clear() {
        let mut pt = PageTable::new(1, 1);
        pt.push(0, 0, PageId(1));
        pt.clear();
        assert_eq!(pt.block_count(0, 0), 0);
    }

    // -----------------------------------------------------------------------
    // PageEvictionPolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_eviction_lru() {
        let mut policy = PageEvictionPolicy::new(EvictionStrategy::Lru);
        policy.on_allocate(PageId(0));
        policy.on_allocate(PageId(1));
        policy.on_allocate(PageId(2));
        // Access page 0 again so it's most recently used
        policy.on_access(PageId(0));
        let victim = policy.select_victim(&[PageId(0), PageId(1), PageId(2)]).unwrap();
        // Page 1 was allocated second but never re-accessed after alloc, and
        // page 2 was allocated third. Page 1 has the smallest last_access.
        assert_eq!(victim, PageId(1));
    }

    #[test]
    fn test_eviction_fifo() {
        let mut policy = PageEvictionPolicy::new(EvictionStrategy::Fifo);
        policy.on_allocate(PageId(0));
        policy.on_allocate(PageId(1));
        policy.on_allocate(PageId(2));
        let victim = policy.select_victim(&[PageId(0), PageId(1), PageId(2)]).unwrap();
        assert_eq!(victim, PageId(0));
    }

    #[test]
    fn test_eviction_frequency() {
        let mut policy = PageEvictionPolicy::new(EvictionStrategy::Frequency);
        policy.on_allocate(PageId(0));
        policy.on_allocate(PageId(1));
        policy.on_allocate(PageId(2));
        policy.on_access(PageId(0));
        policy.on_access(PageId(0));
        policy.on_access(PageId(2));
        // Page 1 has frequency 0
        let victim = policy.select_victim(&[PageId(0), PageId(1), PageId(2)]).unwrap();
        assert_eq!(victim, PageId(1));
    }

    #[test]
    fn test_eviction_empty_candidates() {
        let policy = PageEvictionPolicy::new(EvictionStrategy::Lru);
        assert!(policy.select_victim(&[]).is_none());
    }

    #[test]
    fn test_eviction_counter() {
        let mut policy = PageEvictionPolicy::new(EvictionStrategy::Lru);
        assert_eq!(policy.eviction_count(), 0);
        policy.record_eviction();
        policy.record_eviction();
        assert_eq!(policy.eviction_count(), 2);
    }

    // -----------------------------------------------------------------------
    // CopyOnWrite tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cow_share_and_refcount() {
        let mut cow = CopyOnWrite::new(4096);
        assert_eq!(cow.refcount(PageId(0)), 1);
        cow.share(PageId(0));
        assert_eq!(cow.refcount(PageId(0)), 2);
        assert!(cow.is_shared(PageId(0)));
    }

    #[test]
    fn test_cow_release() {
        let mut cow = CopyOnWrite::new(4096);
        cow.share(PageId(0)); // refcount 2
        assert!(!cow.release(PageId(0))); // refcount 1
        assert!(!cow.is_shared(PageId(0)));
        assert!(cow.release(PageId(0))); // refcount 0 → can free
    }

    #[test]
    fn test_cow_fork_count() {
        let mut cow = CopyOnWrite::new(4096);
        assert_eq!(cow.fork_count(), 0);
        cow.record_fork();
        cow.record_fork();
        assert_eq!(cow.fork_count(), 2);
    }

    #[test]
    fn test_cow_bytes_saved() {
        let mut cow = CopyOnWrite::new(4096);
        cow.share(PageId(0));
        cow.share(PageId(1));
        assert_eq!(cow.bytes_saved(), 4096 * 2);
    }

    // -----------------------------------------------------------------------
    // SequenceGroup tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sequence_group_create() {
        let sg = SequenceGroup::new(1, 100);
        assert_eq!(sg.group_id, 1);
        assert_eq!(sg.len(), 1);
        assert!(!sg.is_empty());
        assert_eq!(sg.sequence_ids, vec![100]);
    }

    #[test]
    fn test_sequence_group_add_remove() {
        let mut sg = SequenceGroup::new(1, 100);
        sg.add_sequence(200);
        sg.add_sequence(300);
        assert_eq!(sg.len(), 3);
        sg.remove_sequence(200);
        assert_eq!(sg.len(), 2);
        assert_eq!(sg.sequence_ids, vec![100, 300]);
    }

    #[test]
    fn test_sequence_group_dedup_add() {
        let mut sg = SequenceGroup::new(1, 100);
        sg.add_sequence(100); // duplicate
        assert_eq!(sg.len(), 1);
    }

    #[test]
    fn test_sequence_group_empty_after_remove_all() {
        let mut sg = SequenceGroup::new(1, 100);
        sg.remove_sequence(100);
        assert!(sg.is_empty());
    }

    // -----------------------------------------------------------------------
    // PagedKvCache tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_add_and_remove_sequence() {
        let mut cache = test_cache(16, 4, 2);
        cache.add_sequence(1);
        assert!(cache.remove_sequence(1).is_ok());
        assert_eq!(cache.remove_sequence(1).unwrap_err(), PagingError::UnknownSequence(1));
    }

    #[test]
    fn test_cache_append_single_token() {
        let mut cache = test_cache(16, 4, 2);
        cache.add_sequence(1);
        let k = vec![1.0, 2.0];
        let v = vec![3.0, 4.0];
        cache.append_token(1, &k, &v).unwrap();
        assert_eq!(cache.sequence_token_count(1).unwrap(), 1);
    }

    #[test]
    fn test_cache_append_and_lookup() {
        let mut cache = test_cache(16, 4, 2);
        cache.add_sequence(1);
        let k = vec![1.0, 2.0];
        let v = vec![3.0, 4.0];
        cache.append_token(1, &k, &v).unwrap();
        assert_eq!(cache.lookup_k(1, 0, 0, 0).unwrap(), vec![1.0, 2.0]);
        assert_eq!(cache.lookup_v(1, 0, 0, 0).unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_cache_append_multiple_tokens() {
        let mut cache = test_cache(16, 4, 2);
        cache.add_sequence(1);
        for i in 0..8 {
            let val = i as f32;
            cache.append_token(1, &[val, val + 0.1], &[val + 0.5, val + 0.6]).unwrap();
        }
        assert_eq!(cache.sequence_token_count(1).unwrap(), 8);
        // Verify first and last
        assert_eq!(cache.lookup_k(1, 0, 0, 0).unwrap(), vec![0.0, 0.1]);
        assert_eq!(cache.lookup_k(1, 0, 0, 7).unwrap(), vec![7.0, 7.1]);
    }

    #[test]
    fn test_cache_cross_page_boundary() {
        // tokens_per_page = 2, so token 2 should go to a second page.
        let mut cache = test_cache(16, 2, 2);
        cache.add_sequence(1);
        cache.append_token(1, &[1.0, 1.0], &[10.0, 10.0]).unwrap();
        cache.append_token(1, &[2.0, 2.0], &[20.0, 20.0]).unwrap();
        cache.append_token(1, &[3.0, 3.0], &[30.0, 30.0]).unwrap();
        assert_eq!(cache.lookup_k(1, 0, 0, 2).unwrap(), vec![3.0, 3.0]);
    }

    #[test]
    fn test_cache_dimension_mismatch_k() {
        let mut cache = test_cache(16, 4, 2);
        cache.add_sequence(1);
        let err = cache.append_token(1, &[1.0], &[3.0, 4.0]).unwrap_err();
        assert!(matches!(err, PagingError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_cache_dimension_mismatch_v() {
        let mut cache = test_cache(16, 4, 2);
        cache.add_sequence(1);
        let err = cache.append_token(1, &[1.0, 2.0], &[3.0]).unwrap_err();
        assert!(matches!(err, PagingError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_cache_unknown_sequence_append() {
        let mut cache = test_cache(16, 4, 2);
        assert_eq!(
            cache.append_token(999, &[1.0, 2.0], &[3.0, 4.0]).unwrap_err(),
            PagingError::UnknownSequence(999)
        );
    }

    #[test]
    fn test_cache_unknown_sequence_lookup() {
        let mut cache = test_cache(16, 4, 2);
        assert_eq!(cache.lookup_k(999, 0, 0, 0).unwrap_err(), PagingError::UnknownSequence(999));
    }

    #[test]
    fn test_cache_evict_pages() {
        let mut cache = test_cache(16, 4, 2);
        cache.add_sequence(1);
        for i in 0..8 {
            let v = i as f32;
            cache.append_token(1, &[v, v], &[v, v]).unwrap();
        }
        let freed = cache.evict_pages(1, 1).unwrap();
        assert!(freed > 0);
    }

    #[test]
    fn test_cache_full_evict_reuse() {
        // tokens_per_page=1 means each token needs its own block (= 2 pages for K+V).
        // 4 pages total → 2 tokens capacity.
        let mut cache = test_cache(4, 1, 2);
        cache.add_sequence(1);
        cache.append_token(1, &[1.0, 1.0], &[2.0, 2.0]).unwrap();
        cache.append_token(1, &[3.0, 3.0], &[4.0, 4.0]).unwrap();
        // Cache is now full (4 pages used).
        assert_eq!(cache.allocator().free_count(), 0);

        // Evict 1 round of pages → should free some
        let freed = cache.evict_pages(1, 1).unwrap();
        assert!(freed > 0);
        assert!(cache.allocator().free_count() > 0);

        // Now we can reuse: add a new sequence
        cache.add_sequence(2);
        cache.append_token(2, &[5.0, 5.0], &[6.0, 6.0]).unwrap();
        assert_eq!(cache.sequence_token_count(2).unwrap(), 1);
    }

    #[test]
    fn test_cache_stats_initial() {
        let cache = test_cache(16, 4, 2);
        let stats = cache.stats();
        assert_eq!(stats.total_pages, 16);
        assert_eq!(stats.free_pages, 16);
        assert_eq!(stats.allocated_pages, 0);
        assert_eq!(stats.eviction_count, 0);
        assert_eq!(stats.cow_forks, 0);
    }

    #[test]
    fn test_cache_stats_after_append() {
        let mut cache = test_cache(16, 4, 2);
        cache.add_sequence(1);
        cache.append_token(1, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        let stats = cache.stats();
        assert_eq!(stats.allocated_pages, 2); // 1 K page + 1 V page
        assert!(stats.utilization > 0.0);
        assert!(stats.utilization < 1.0);
    }

    #[test]
    fn test_cache_stats_display() {
        let cache = test_cache(16, 4, 2);
        let s = format!("{}", cache.stats());
        assert!(s.contains("pages="));
        assert!(s.contains("free="));
    }

    // -----------------------------------------------------------------------
    // Fork / COW integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fork_shares_pages() {
        let mut cache = test_cache(32, 4, 2);
        cache.add_sequence(1);
        cache.append_token(1, &[1.0, 2.0], &[3.0, 4.0]).unwrap();

        cache.fork_sequence(1, 2).unwrap();

        // Both sequences see the same data.
        assert_eq!(cache.lookup_k(1, 0, 0, 0).unwrap(), vec![1.0, 2.0]);
        assert_eq!(cache.lookup_k(2, 0, 0, 0).unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_fork_cow_diverge() {
        let mut cache = test_cache(32, 4, 2);
        cache.add_sequence(1);
        cache.append_token(1, &[1.0, 2.0], &[3.0, 4.0]).unwrap();

        cache.fork_sequence(1, 2).unwrap();

        // Append to sequence 2 — should trigger COW fork on the shared page.
        cache.append_token(2, &[5.0, 6.0], &[7.0, 8.0]).unwrap();

        // Sequence 1 still sees original.
        assert_eq!(cache.lookup_k(1, 0, 0, 0).unwrap(), vec![1.0, 2.0]);
        // Sequence 2 sees its new data at position 1.
        assert_eq!(cache.lookup_k(2, 0, 0, 1).unwrap(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_fork_cow_stats() {
        let mut cache = test_cache(32, 4, 2);
        cache.add_sequence(1);
        cache.append_token(1, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.fork_sequence(1, 2).unwrap();
        cache.append_token(2, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
        let stats = cache.stats();
        assert!(stats.cow_bytes_saved > 0);
        assert!(stats.cow_forks > 0);
    }

    #[test]
    fn test_fork_unknown_source() {
        let mut cache = test_cache(16, 4, 2);
        assert_eq!(cache.fork_sequence(999, 2).unwrap_err(), PagingError::UnknownSequence(999));
    }

    // -----------------------------------------------------------------------
    // Group tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_group() {
        let mut cache = test_cache(16, 4, 2);
        cache.add_sequence(1);
        cache.create_group(100, 1).unwrap();
    }

    #[test]
    fn test_create_group_unknown_seq() {
        let mut cache = test_cache(16, 4, 2);
        assert_eq!(cache.create_group(100, 999).unwrap_err(), PagingError::UnknownSequence(999));
    }

    // -----------------------------------------------------------------------
    // Edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_page_cache() {
        // Only 2 pages (K+V), 1 token capacity.
        let mut cache = test_cache(2, 1, 2);
        cache.add_sequence(1);
        cache.append_token(1, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert_eq!(cache.allocator().free_count(), 0);
    }

    #[test]
    fn test_max_pages_allocate() {
        let n = 1024;
        let mut alloc = PageAllocator::new(n, 4, 2);
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            ids.push(alloc.allocate().unwrap());
        }
        assert_eq!(alloc.free_count(), 0);
        for id in &ids {
            alloc.deallocate(*id).unwrap();
        }
        assert_eq!(alloc.free_count(), n);
    }

    #[test]
    fn test_zero_length_sequence() {
        let mut cache = test_cache(16, 4, 2);
        cache.add_sequence(1);
        assert_eq!(cache.sequence_token_count(1).unwrap(), 0);
        assert!(cache.lookup_k(1, 0, 0, 0).is_err());
    }

    // -----------------------------------------------------------------------
    // CPU reference paged attention tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpu_paged_attention_single_token() {
        let head_dim = 4;
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let k_data = vec![1.0, 0.0, 0.0, 0.0]; // single token K
        let v_data = vec![0.0, 1.0, 0.0, 0.0]; // single token V
        let result =
            cpu_paged_attention(&query, &[(&k_data, 1)], &[(&v_data, 1)], head_dim).unwrap();
        // With single token, output = V (softmax is trivially 1.0).
        assert_eq!(result, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cpu_paged_attention_two_tokens() {
        let head_dim = 2;
        let query = vec![1.0, 0.0];
        let k_data = vec![1.0, 0.0, 0.0, 1.0]; // 2 tokens
        let v_data = vec![1.0, 0.0, 0.0, 1.0]; // 2 tokens
        let result =
            cpu_paged_attention(&query, &[(&k_data, 2)], &[(&v_data, 2)], head_dim).unwrap();
        // Score for token 0: dot([1,0],[1,0]) * scale = 1/sqrt(2)
        // Score for token 1: dot([1,0],[0,1]) * scale = 0
        // Softmax: w0 = exp(1/sqrt(2)) / (exp(1/sqrt(2)) + exp(0))
        // Output should weight token 0 more heavily.
        assert!(result[0] > result[1]);
    }

    #[test]
    fn test_cpu_paged_attention_multi_page() {
        let head_dim = 2;
        let query = vec![1.0, 0.0];
        let k_page0 = vec![1.0, 0.0, 0.5, 0.5]; // 2 tokens
        let k_page1 = vec![0.0, 1.0]; // 1 token
        let v_page0 = vec![1.0, 0.0, 0.5, 0.5];
        let v_page1 = vec![0.0, 1.0];
        let result = cpu_paged_attention(
            &query,
            &[(&k_page0, 2), (&k_page1, 1)],
            &[(&v_page0, 2), (&v_page1, 1)],
            head_dim,
        )
        .unwrap();
        assert_eq!(result.len(), head_dim);
        // Output should be a valid weighted sum.
        assert!(result[0] >= 0.0 && result[0] <= 1.0);
    }

    #[test]
    fn test_cpu_paged_attention_empty() {
        let head_dim = 4;
        let query = vec![1.0; head_dim];
        let result: Vec<f32> = cpu_paged_attention(&query, &[], &[], head_dim).unwrap();
        assert_eq!(result, vec![0.0; head_dim]);
    }

    #[test]
    fn test_cpu_paged_attention_query_dim_mismatch() {
        let err = cpu_paged_attention(&[1.0, 2.0], &[], &[], 4).unwrap_err();
        assert!(matches!(err, PagingError::DimensionMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // OpenCL kernel source test
    // -----------------------------------------------------------------------

    #[test]
    fn test_opencl_kernel_source_present() {
        assert!(!PAGED_ATTENTION_CL.is_empty());
        assert!(PAGED_ATTENTION_CL.contains("paged_attention"));
        assert!(PAGED_ATTENTION_CL.contains("__kernel"));
        assert!(PAGED_ATTENTION_CL.contains("page_table"));
    }

    // -----------------------------------------------------------------------
    // Constants tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_page_size_bytes_constant() {
        assert_eq!(PAGE_SIZE_BYTES, 16 * 64 * 4);
        assert_eq!(PAGE_SIZE_BYTES, 4096);
    }

    #[test]
    fn test_a770_max_pages() {
        // ~3.5 GB / 4 KB ≈ 854K pages
        assert!(A770_MAX_PAGES > 800_000);
        assert!(A770_MAX_PAGES < 1_000_000);
    }

    // -----------------------------------------------------------------------
    // PagingStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_paging_stats_fragmentation() {
        let mut cache = test_cache(16, 4, 2);
        cache.add_sequence(1);
        // Append 1 token into a 4-slot page → 75% fragmentation
        cache.append_token(1, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        let stats = cache.stats();
        assert!(stats.fragmentation > 0.5);
        assert!(stats.utilization > 0.0);
        assert!(stats.utilization < 0.5);
    }

    #[test]
    fn test_paging_stats_full_utilization() {
        let mut cache = test_cache(16, 2, 2);
        cache.add_sequence(1);
        cache.append_token(1, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.append_token(1, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
        // 2 tokens in a 2-slot page → 100% utilization per page
        let stats = cache.stats();
        assert!((stats.utilization - 1.0).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Eviction policy integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_lru_evicts_oldest_accessed() {
        let mut cache = test_cache(8, 2, 2);
        cache.add_sequence(1);
        cache.add_sequence(2);
        // Fill both sequences
        cache.append_token(1, &[1.0, 1.0], &[1.0, 1.0]).unwrap();
        cache.append_token(2, &[2.0, 2.0], &[2.0, 2.0]).unwrap();
        // Access seq 1 again
        let _ = cache.lookup_k(1, 0, 0, 0);
        // Evict from seq 2 (its pages were accessed less recently)
        let freed = cache.evict_pages(2, 1).unwrap();
        assert!(freed > 0);
    }

    // -----------------------------------------------------------------------
    // Multi-layer / multi-head tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_layer_multi_head() {
        let num_layers = 2;
        let num_heads = 2;
        let head_dim = 2;
        let tpp = 4;
        let num_pages = 64;
        let mut cache = PagedKvCache::new(
            num_pages,
            tpp,
            head_dim,
            num_layers,
            num_heads,
            EvictionStrategy::Lru,
        );
        cache.add_sequence(1);
        // K/V data: [num_layers * num_heads * head_dim] = 2 * 2 * 2 = 8 floats
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        cache.append_token(1, &k, &v).unwrap();

        // Verify each (layer, head)
        assert_eq!(cache.lookup_k(1, 0, 0, 0).unwrap(), vec![1.0, 2.0]);
        assert_eq!(cache.lookup_k(1, 0, 1, 0).unwrap(), vec![3.0, 4.0]);
        assert_eq!(cache.lookup_k(1, 1, 0, 0).unwrap(), vec![5.0, 6.0]);
        assert_eq!(cache.lookup_k(1, 1, 1, 0).unwrap(), vec![7.0, 8.0]);
        assert_eq!(cache.lookup_v(1, 0, 0, 0).unwrap(), vec![10.0, 20.0]);
    }

    // -----------------------------------------------------------------------
    // Error display tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let e = PagingError::OutOfPages;
        assert_eq!(format!("{e}"), "page allocator exhausted");

        let e = PagingError::InvalidPageId(PageId(42));
        assert!(format!("{e}").contains("42"));

        let e = PagingError::PageTableMiss { layer: 1, head: 2, block_idx: 3 };
        let s = format!("{e}");
        assert!(s.contains("layer=1"));
        assert!(s.contains("head=2"));

        let e = PagingError::UnknownSequence(7);
        assert!(format!("{e}").contains("7"));

        let e = PagingError::DimensionMismatch { expected: 64, got: 32 };
        assert!(format!("{e}").contains("64"));

        let e = PagingError::SharedPageWrite(PageId(5));
        assert!(format!("{e}").contains("Page(5)"));
    }

    #[test]
    fn test_page_id_display() {
        assert_eq!(format!("{}", PageId(42)), "Page(42)");
    }

    #[test]
    fn test_kv_page_debug() {
        let page = KvPage::new(16, 64);
        let dbg = format!("{:?}", page);
        assert!(dbg.contains("KvPage"));
        assert!(dbg.contains("capacity_bytes"));
    }
}
