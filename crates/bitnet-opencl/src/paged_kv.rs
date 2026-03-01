//! PagedAttention KV cache for GPU inference.
//!
//! Implements a paged virtual-memory style KV cache that maps logical token
//! positions to fixed-size physical pages. This avoids contiguous allocation
//! for long sequences, reduces memory fragmentation, and enables page sharing
//! across sequences in batched inference.

use std::collections::HashMap;

/// Default number of tokens stored per physical page.
pub const DEFAULT_PAGE_SIZE: usize = 16;

/// Unique identifier for a physical page in the GPU buffer pool.
pub type PageId = u32;

/// Unique identifier for a logical sequence (request / beam).
pub type SeqId = u64;

/// A single fixed-size page that holds KV data for `page_size` tokens.
#[derive(Debug, Clone)]
pub struct KvPage {
    /// Unique page identifier.
    pub id: PageId,
    /// Number of tokens currently written into this page (0..=page_size).
    pub num_filled: usize,
    /// Reference count for copy-on-write sharing.
    pub ref_count: u32,
}

/// Per-sequence metadata tracking the page table and current length.
#[derive(Debug, Clone)]
pub struct SequenceEntry {
    /// Ordered list of page IDs mapping logical blocks to physical pages.
    pub page_table: Vec<PageId>,
    /// Total number of tokens cached for this sequence.
    pub token_count: usize,
}

/// PagedAttention KV cache manager.
///
/// Manages a pool of fixed-size pages and maps logical (sequence, position)
/// pairs to physical pages. Supports:
/// - Configurable page size (default 16 tokens)
/// - Free-list based allocation / deallocation
/// - Variable-length sequences with dynamic page growth
/// - Reference-counted pages for copy-on-write sharing
#[derive(Debug)]
pub struct PagedKvCache {
    /// Tokens per page.
    page_size: usize,
    /// Total number of pages in the pool.
    pool_capacity: usize,
    /// All physical pages, indexed by PageId.
    pages: Vec<KvPage>,
    /// Stack-based free list of available page IDs.
    free_list: Vec<PageId>,
    /// Sequence ID → page table mapping.
    sequences: HashMap<SeqId, SequenceEntry>,
    /// Monotonic counter for generating page IDs.
    next_page_id: PageId,
}

impl PagedKvCache {
    /// Create a new paged KV cache.
    ///
    /// # Arguments
    /// * `pool_capacity` – maximum number of physical pages in the pool
    /// * `page_size` – number of tokens per page (default: [`DEFAULT_PAGE_SIZE`])
    pub fn new(pool_capacity: usize, page_size: usize) -> Self {
        let mut pages = Vec::with_capacity(pool_capacity);
        let mut free_list = Vec::with_capacity(pool_capacity);

        for id in (0..pool_capacity as PageId).rev() {
            pages.push(KvPage {
                id,
                num_filled: 0,
                ref_count: 0,
            });
            free_list.push(id);
        }

        Self {
            page_size,
            pool_capacity,
            pages,
            free_list,
            sequences: HashMap::new(),
            next_page_id: pool_capacity as PageId,
        }
    }

    /// Create with the default page size.
    pub fn with_capacity(pool_capacity: usize) -> Self {
        Self::new(pool_capacity, DEFAULT_PAGE_SIZE)
    }

    /// Number of free pages remaining in the pool.
    pub fn free_pages(&self) -> usize {
        self.free_list.len()
    }

    /// Total pool capacity (in pages).
    pub fn pool_capacity(&self) -> usize {
        self.pool_capacity
    }

    /// Configured tokens-per-page.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Number of active sequences.
    pub fn active_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Allocate a single page from the free list.
    ///
    /// Returns `None` if the pool is exhausted.
    fn alloc_page(&mut self) -> Option<PageId> {
        let page_id = self.free_list.pop()?;
        let page = &mut self.pages[page_id as usize];
        page.num_filled = 0;
        page.ref_count = 1;
        Some(page_id)
    }

    /// Return a page to the free list (decrements ref count first).
    fn free_page(&mut self, page_id: PageId) {
        let page = &mut self.pages[page_id as usize];
        if page.ref_count > 0 {
            page.ref_count -= 1;
        }
        if page.ref_count == 0 {
            page.num_filled = 0;
            self.free_list.push(page_id);
        }
    }

    /// Register a new empty sequence. Returns `false` if the ID is already taken.
    pub fn register_sequence(&mut self, seq_id: SeqId) -> bool {
        if self.sequences.contains_key(&seq_id) {
            return false;
        }
        self.sequences.insert(
            seq_id,
            SequenceEntry {
                page_table: Vec::new(),
                token_count: 0,
            },
        );
        true
    }

    /// Append `count` tokens to a sequence, allocating new pages as needed.
    ///
    /// Returns `Ok(())` on success, or `Err(needed)` with the number of
    /// additional pages required if the pool is exhausted.
    pub fn append_tokens(&mut self, seq_id: SeqId, count: usize) -> Result<(), usize> {
        let entry = self
            .sequences
            .get(&seq_id)
            .ok_or(0usize)?;

        // How many slots remain in the last page?
        let slots_in_last = if entry.page_table.is_empty() {
            0
        } else {
            let last_id = *entry.page_table.last().unwrap();
            self.page_size - self.pages[last_id as usize].num_filled
        };

        let overflow = count.saturating_sub(slots_in_last);
        let new_pages_needed = (overflow + self.page_size - 1) / self.page_size;

        if new_pages_needed > self.free_list.len() {
            return Err(new_pages_needed - self.free_list.len());
        }

        // Fill remaining slots in current last page.
        let entry = self.sequences.get_mut(&seq_id).unwrap();
        let mut remaining = count;

        if !entry.page_table.is_empty() && slots_in_last > 0 {
            let last_id = *entry.page_table.last().unwrap();
            let fill = remaining.min(slots_in_last);
            self.pages[last_id as usize].num_filled += fill;
            remaining -= fill;
        }

        // Allocate new pages for the rest.
        while remaining > 0 {
            let page_id = self.alloc_page().unwrap();
            let fill = remaining.min(self.page_size);
            self.pages[page_id as usize].num_filled = fill;
            let entry = self.sequences.get_mut(&seq_id).unwrap();
            entry.page_table.push(page_id);
            remaining -= fill;
        }

        let entry = self.sequences.get_mut(&seq_id).unwrap();
        entry.token_count += count;
        Ok(())
    }

    /// Remove a sequence and free all its pages.
    pub fn remove_sequence(&mut self, seq_id: SeqId) -> bool {
        let entry = match self.sequences.remove(&seq_id) {
            Some(e) => e,
            None => return false,
        };
        for &page_id in &entry.page_table {
            self.free_page(page_id);
        }
        true
    }

    /// Look up the physical page and intra-page offset for a logical position.
    ///
    /// Returns `(page_id, offset_within_page)` or `None` if out of range.
    pub fn translate(&self, seq_id: SeqId, token_pos: usize) -> Option<(PageId, usize)> {
        let entry = self.sequences.get(&seq_id)?;
        if token_pos >= entry.token_count {
            return None;
        }
        let page_idx = token_pos / self.page_size;
        let offset = token_pos % self.page_size;
        let page_id = *entry.page_table.get(page_idx)?;
        Some((page_id, offset))
    }

    /// Share pages from `src_seq` into a new sequence `dst_seq` (copy-on-write).
    ///
    /// Both sequences will reference the same physical pages with incremented
    /// ref counts. Returns `false` if `src_seq` doesn't exist or `dst_seq`
    /// already exists.
    pub fn fork_sequence(&mut self, src_seq: SeqId, dst_seq: SeqId) -> bool {
        if self.sequences.contains_key(&dst_seq) {
            return false;
        }
        let src_entry = match self.sequences.get(&src_seq) {
            Some(e) => e.clone(),
            None => return false,
        };

        // Bump ref counts on all shared pages.
        for &page_id in &src_entry.page_table {
            self.pages[page_id as usize].ref_count += 1;
        }

        self.sequences.insert(dst_seq, src_entry);
        true
    }

    /// Get the page table for a sequence (for debugging / inspection).
    pub fn page_table(&self, seq_id: SeqId) -> Option<&[PageId]> {
        self.sequences.get(&seq_id).map(|e| e.page_table.as_slice())
    }

    /// Total tokens cached for a sequence.
    pub fn sequence_length(&self, seq_id: SeqId) -> Option<usize> {
        self.sequences.get(&seq_id).map(|e| e.token_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_cache_defaults() {
        let cache = PagedKvCache::with_capacity(64);
        assert_eq!(cache.pool_capacity(), 64);
        assert_eq!(cache.page_size(), DEFAULT_PAGE_SIZE);
        assert_eq!(cache.free_pages(), 64);
        assert_eq!(cache.active_sequences(), 0);
    }

    #[test]
    fn test_custom_page_size() {
        let cache = PagedKvCache::new(32, 8);
        assert_eq!(cache.page_size(), 8);
        assert_eq!(cache.free_pages(), 32);
    }

    #[test]
    fn test_register_and_remove_sequence() {
        let mut cache = PagedKvCache::with_capacity(16);
        assert!(cache.register_sequence(1));
        assert_eq!(cache.active_sequences(), 1);
        // Duplicate registration fails.
        assert!(!cache.register_sequence(1));
        assert!(cache.remove_sequence(1));
        assert_eq!(cache.active_sequences(), 0);
        // Removing non-existent fails.
        assert!(!cache.remove_sequence(999));
    }

    #[test]
    fn test_append_tokens_single_page() {
        let mut cache = PagedKvCache::new(8, 4);
        cache.register_sequence(1);
        assert!(cache.append_tokens(1, 3).is_ok());
        assert_eq!(cache.sequence_length(1), Some(3));
        // Used 1 page, 7 free.
        assert_eq!(cache.free_pages(), 7);
    }

    #[test]
    fn test_append_tokens_spans_pages() {
        let mut cache = PagedKvCache::new(8, 4);
        cache.register_sequence(1);
        // 10 tokens across page_size=4 → 3 pages (4+4+2).
        assert!(cache.append_tokens(1, 10).is_ok());
        assert_eq!(cache.sequence_length(1), Some(10));
        assert_eq!(cache.free_pages(), 5); // 8 - 3
        assert_eq!(cache.page_table(1).unwrap().len(), 3);
    }

    #[test]
    fn test_append_fills_last_page_first() {
        let mut cache = PagedKvCache::new(8, 4);
        cache.register_sequence(1);
        // First append: 2 tokens → 1 page, 2 filled.
        cache.append_tokens(1, 2).unwrap();
        assert_eq!(cache.free_pages(), 7);
        // Second append: 2 more → still fits in the same page.
        cache.append_tokens(1, 2).unwrap();
        assert_eq!(cache.free_pages(), 7);
        assert_eq!(cache.sequence_length(1), Some(4));
        // Third append: 1 more → spills to a new page.
        cache.append_tokens(1, 1).unwrap();
        assert_eq!(cache.free_pages(), 6);
        assert_eq!(cache.sequence_length(1), Some(5));
    }

    #[test]
    fn test_pool_exhaustion_returns_err() {
        let mut cache = PagedKvCache::new(2, 4);
        cache.register_sequence(1);
        // 8 tokens fit in 2 pages.
        assert!(cache.append_tokens(1, 8).is_ok());
        // 1 more token needs a third page but pool is full.
        let err = cache.append_tokens(1, 1).unwrap_err();
        assert_eq!(err, 1); // needs 1 more page
    }

    #[test]
    fn test_translate_logical_to_physical() {
        let mut cache = PagedKvCache::new(8, 4);
        cache.register_sequence(1);
        cache.append_tokens(1, 10).unwrap(); // 3 pages

        // Position 0 → first page, offset 0.
        let (pid0, off0) = cache.translate(1, 0).unwrap();
        assert_eq!(off0, 0);

        // Position 5 → second page (idx=1), offset 1.
        let (pid1, off1) = cache.translate(1, 5).unwrap();
        assert_eq!(off1, 1);
        assert_ne!(pid0, pid1);

        // Position 9 → third page, offset 1.
        let (_pid2, off2) = cache.translate(1, 9).unwrap();
        assert_eq!(off2, 1);

        // Out of range.
        assert!(cache.translate(1, 10).is_none());
        assert!(cache.translate(999, 0).is_none());
    }

    #[test]
    fn test_remove_frees_pages() {
        let mut cache = PagedKvCache::new(8, 4);
        cache.register_sequence(1);
        cache.append_tokens(1, 10).unwrap(); // 3 pages used
        assert_eq!(cache.free_pages(), 5);
        cache.remove_sequence(1);
        assert_eq!(cache.free_pages(), 8);
    }

    #[test]
    fn test_fork_sequence_shares_pages() {
        let mut cache = PagedKvCache::new(8, 4);
        cache.register_sequence(1);
        cache.append_tokens(1, 6).unwrap(); // 2 pages

        // Fork seq 1 → seq 2 (no new pages allocated).
        assert!(cache.fork_sequence(1, 2));
        assert_eq!(cache.active_sequences(), 2);
        assert_eq!(cache.sequence_length(2), Some(6));
        // Free pages unchanged (shared).
        assert_eq!(cache.free_pages(), 6);

        // Remove original — pages still held by fork.
        cache.remove_sequence(1);
        assert_eq!(cache.free_pages(), 6); // ref count still > 0

        // Remove fork — pages finally freed.
        cache.remove_sequence(2);
        assert_eq!(cache.free_pages(), 8);
    }

    #[test]
    fn test_fork_fails_on_duplicate_dst() {
        let mut cache = PagedKvCache::new(8, 4);
        cache.register_sequence(1);
        cache.register_sequence(2);
        assert!(!cache.fork_sequence(1, 2)); // dst exists
        assert!(!cache.fork_sequence(999, 3)); // src missing
    }
}
