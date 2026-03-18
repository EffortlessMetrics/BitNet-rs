//! ARM NEON paged KV cache kernel for Apple Silicon.
//!
//! Implements a paged virtual memory system for KV caches used during
//! autoregressive inference. Fixed-size blocks (default 16 tokens) are
//! managed through a free-list allocator with support for:
//!
//! - NEON-accelerated gather (read) and scatter (write) over paged blocks
//! - LRU and sliding-window eviction policies
//! - Copy-on-write semantics for beam search / speculative decoding
//! - Sequence-aware page table management
//! - NEON-accelerated attention score computation over paged layouts

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Number of f32 lanes in a NEON `float32x4_t` register.
const NEON_F32_LANES: usize = 4;

/// Default number of tokens stored per cache block.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

// ── Eviction policy ────────────────────────────────────────────────

/// Cache eviction strategy used when the block pool is exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least-recently-used: evict the block whose last access is oldest.
    Lru,
    /// Sliding window: keep only the most recent `window_size` tokens,
    /// evicting blocks that fall outside the window.
    SlidingWindow {
        /// Maximum number of tokens to retain.
        window_size: usize,
    },
}

// ── Cache block ────────────────────────────────────────────────────

/// A fixed-size block holding KV vectors for up to `block_size` tokens.
///
/// Each token occupies `head_dim` f32 elements; both keys and values are
/// stored contiguously inside the block.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used by aarch64 methods gated with cfg.
pub struct PagedCacheBlock {
    /// Key vectors: `[tokens_stored, head_dim]` flattened.
    pub keys: Vec<f32>,
    /// Value vectors: `[tokens_stored, head_dim]` flattened.
    pub values: Vec<f32>,
    /// Number of tokens currently stored (0..=block_size).
    pub tokens_stored: usize,
    /// Maximum tokens this block can hold.
    pub block_size: usize,
    /// Elements per token (i.e. `head_dim`).
    pub head_dim: usize,
    /// Reference count for copy-on-write.
    ref_count: usize,
    /// Monotonic access counter for LRU eviction.
    last_access: u64,
}

impl PagedCacheBlock {
    /// Allocate a new empty block.
    pub fn new(block_size: usize, head_dim: usize) -> Self {
        let cap = block_size * head_dim;
        Self {
            keys: vec![0.0; cap],
            values: vec![0.0; cap],
            tokens_stored: 0,
            block_size,
            head_dim,
            ref_count: 1,
            last_access: 0,
        }
    }

    /// Remaining token capacity in this block.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.block_size - self.tokens_stored
    }

    /// Whether the block is completely full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.tokens_stored == self.block_size
    }

    /// Reset the block without de-allocating backing storage.
    pub fn clear(&mut self) {
        self.tokens_stored = 0;
        self.ref_count = 1;
    }

    /// Memory occupied by this block in bytes.
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * size_of::<f32>()
    }
}

// ── Page table entry ───────────────────────────────────────────────

/// Maps a sequence's logical block index to a physical block in the pool.
#[derive(Debug, Clone, Copy)]
struct PageTableEntry {
    /// Index into `PagedKvCachePool::blocks`.
    physical_block: usize,
    /// True when this page is shared (CoW pending).
    cow_pending: bool,
}

// ── Sequence descriptor ────────────────────────────────────────────

/// Per-sequence metadata including its page table.
#[derive(Debug, Clone)]
pub struct SequenceState {
    /// Sequence identifier.
    pub seq_id: u64,
    /// Ordered list of page table entries for this sequence.
    page_table: Vec<PageTableEntry>,
    /// Total tokens appended to this sequence.
    pub total_tokens: usize,
}

impl SequenceState {
    fn new(seq_id: u64) -> Self {
        Self { seq_id, page_table: Vec::new(), total_tokens: 0 }
    }

    /// Number of physical blocks mapped by this sequence.
    pub fn num_blocks(&self) -> usize {
        self.page_table.len()
    }
}

// ── Paged KV cache pool ───────────────────────────────────────────

/// Central pool managing physical blocks, free-list, sequences, and
/// eviction.
#[derive(Debug)]
#[allow(dead_code)] // block_size stored for diagnostics and future resize support.
pub struct PagedKvCachePool {
    /// All physical blocks (both in-use and free).
    blocks: Vec<PagedCacheBlock>,
    /// Indices of available blocks.
    free_list: Vec<usize>,
    /// Per-sequence state.
    sequences: Vec<SequenceState>,
    /// Tokens per block.
    block_size: usize,
    /// Elements per token.
    head_dim: usize,
    /// Eviction strategy.
    eviction_policy: EvictionPolicy,
    /// Monotonic clock for LRU tracking.
    access_counter: u64,
}

impl PagedKvCachePool {
    /// Create a new pool with `num_blocks` pre-allocated blocks.
    ///
    /// # Panics
    ///
    /// Panics if `num_blocks`, `block_size`, or `head_dim` is zero.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        head_dim: usize,
        eviction_policy: EvictionPolicy,
    ) -> Self {
        assert!(num_blocks > 0, "num_blocks must be > 0");
        assert!(block_size > 0, "block_size must be > 0");
        assert!(head_dim > 0, "head_dim must be > 0");

        let blocks: Vec<PagedCacheBlock> =
            (0..num_blocks).map(|_| PagedCacheBlock::new(block_size, head_dim)).collect();
        let free_list: Vec<usize> = (0..num_blocks).rev().collect();

        Self {
            blocks,
            free_list,
            sequences: Vec::new(),
            block_size,
            head_dim,
            eviction_policy,
            access_counter: 0,
        }
    }

    // ── Allocation helpers ─────────────────────────────────────────

    /// Allocate a single block from the free list, applying eviction
    /// when the pool is exhausted. Returns the physical index.
    fn alloc_block(&mut self) -> Option<usize> {
        if let Some(idx) = self.free_list.pop() {
            self.blocks[idx].clear();
            return Some(idx);
        }
        // Try eviction.
        self.evict_one().and_then(|_| self.free_list.pop()).inspect(|&idx| {
            self.blocks[idx].clear();
        })
    }

    /// Return a block to the free list.
    fn free_block(&mut self, idx: usize) {
        assert!(idx < self.blocks.len(), "block index out of range");
        self.blocks[idx].ref_count = self.blocks[idx].ref_count.saturating_sub(1);
        if self.blocks[idx].ref_count == 0 {
            self.blocks[idx].clear();
            self.free_list.push(idx);
        }
    }

    /// Number of free blocks.
    #[inline]
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of blocks in the pool.
    #[inline]
    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Total memory used by all blocks in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.memory_bytes()).sum()
    }

    // ── Sequence management ────────────────────────────────────────

    /// Register a new sequence and return its index in `self.sequences`.
    pub fn add_sequence(&mut self, seq_id: u64) -> usize {
        let idx = self.sequences.len();
        self.sequences.push(SequenceState::new(seq_id));
        idx
    }

    /// Remove a sequence, freeing all of its physical blocks.
    pub fn remove_sequence(&mut self, seq_idx: usize) {
        assert!(seq_idx < self.sequences.len(), "seq_idx out of range");
        let entries: Vec<PageTableEntry> = self.sequences[seq_idx].page_table.drain(..).collect();
        for e in entries {
            self.free_block(e.physical_block);
        }
        self.sequences[seq_idx].total_tokens = 0;
    }

    /// Look up a sequence by its public `seq_id`.
    pub fn find_sequence(&self, seq_id: u64) -> Option<usize> {
        self.sequences.iter().position(|s| s.seq_id == seq_id)
    }

    // ── Eviction ───────────────────────────────────────────────────

    /// Attempt to evict one block according to the configured policy.
    /// Returns `Some(())` on success, `None` if nothing can be evicted.
    fn evict_one(&mut self) -> Option<()> {
        match self.eviction_policy {
            EvictionPolicy::Lru => self.evict_lru(),
            EvictionPolicy::SlidingWindow { window_size } => self.evict_sliding_window(window_size),
        }
    }

    /// LRU eviction: find the allocated block with the smallest
    /// `last_access` and free it.
    fn evict_lru(&mut self) -> Option<()> {
        let free_set: std::collections::HashSet<usize> = self.free_list.iter().copied().collect();

        let victim = (0..self.blocks.len())
            .filter(|i| !free_set.contains(i))
            .min_by_key(|&i| self.blocks[i].last_access)?;

        // Remove the victim from every sequence that references it.
        for seq in &mut self.sequences {
            seq.page_table.retain(|e| e.physical_block != victim);
        }
        self.blocks[victim].clear();
        self.free_list.push(victim);
        Some(())
    }

    /// Sliding-window eviction: for each sequence, drop the oldest
    /// blocks that exceed `window_size` tokens.
    fn evict_sliding_window(&mut self, window_size: usize) -> Option<()> {
        let mut freed_any = false;
        for seq in &mut self.sequences {
            while seq.total_tokens > window_size && !seq.page_table.is_empty() {
                let entry = seq.page_table.remove(0);
                let blk = &mut self.blocks[entry.physical_block];
                let tokens_in_block = blk.tokens_stored;
                blk.ref_count = blk.ref_count.saturating_sub(1);
                if blk.ref_count == 0 {
                    blk.clear();
                    // Note: we can't push to self.free_list here due to
                    // the mutable borrow on self.sequences, so we record
                    // separately.
                }
                seq.total_tokens = seq.total_tokens.saturating_sub(tokens_in_block);
                freed_any = true;
            }
        }
        // Re-derive free list from ref_counts.
        if freed_any {
            self.rebuild_free_list();
            Some(())
        } else {
            None
        }
    }

    /// Rebuild free_list based on ref_count == 0.
    fn rebuild_free_list(&mut self) {
        let in_use: std::collections::HashSet<usize> = self
            .sequences
            .iter()
            .flat_map(|s| s.page_table.iter().map(|e| e.physical_block))
            .collect();
        self.free_list.clear();
        for (i, blk) in self.blocks.iter().enumerate() {
            if !in_use.contains(&i) && blk.ref_count == 0 {
                self.free_list.push(i);
            }
        }
    }

    // ── Touch (LRU bookkeeping) ────────────────────────────────────

    /// Update LRU access timestamp for the given physical block.
    #[inline]
    fn touch(&mut self, physical: usize) {
        self.access_counter += 1;
        self.blocks[physical].last_access = self.access_counter;
    }

    // ── Write (scatter to paged blocks) ────────────────────────────

    /// Append key/value vectors for new tokens to a sequence.
    ///
    /// `new_keys` and `new_values` must each have length
    /// `num_new_tokens * head_dim`.
    ///
    /// Uses NEON `vld1q_f32` / `vst1q_f32` for bulk copies on aarch64.
    #[cfg(target_arch = "aarch64")]
    pub fn write_cache(
        &mut self,
        seq_idx: usize,
        new_keys: &[f32],
        new_values: &[f32],
    ) -> Result<(), &'static str> {
        let hd = self.head_dim;
        if !new_keys.len().is_multiple_of(hd) {
            return Err("new_keys length not a multiple of head_dim");
        }
        if new_keys.len() != new_values.len() {
            return Err("new_keys and new_values length mismatch");
        }
        let num_tokens = new_keys.len() / hd;
        if seq_idx >= self.sequences.len() {
            return Err("seq_idx out of range");
        }

        let mut written = 0usize;

        while written < num_tokens {
            // Get or allocate the tail block for this sequence.
            let need_new_block = {
                let seq = &self.sequences[seq_idx];
                seq.page_table.is_empty()
                    || self.blocks[seq.page_table.last().unwrap().physical_block].is_full()
            };

            if need_new_block {
                let phys = self.alloc_block().ok_or("no free blocks (eviction failed)")?;
                self.sequences[seq_idx]
                    .page_table
                    .push(PageTableEntry { physical_block: phys, cow_pending: false });
            }

            // Handle copy-on-write before reading physical block index.
            if self.sequences[seq_idx].page_table.last().unwrap().cow_pending {
                self.cow_duplicate(seq_idx)?;
            }

            // Read phys AFTER CoW so we get the duplicated block.
            let phys = self.sequences[seq_idx].page_table.last().unwrap().physical_block;
            let blk = &mut self.blocks[phys];
            let space = blk.remaining().min(num_tokens - written);
            let src_off = written * hd;
            let dst_off = blk.tokens_stored * hd;
            let count = space * hd;

            neon_copy_f32(
                &new_keys[src_off..src_off + count],
                &mut blk.keys[dst_off..dst_off + count],
            );
            neon_copy_f32(
                &new_values[src_off..src_off + count],
                &mut blk.values[dst_off..dst_off + count],
            );

            blk.tokens_stored += space;
            written += space;
            self.sequences[seq_idx].total_tokens += space;
            self.touch(phys);
        }

        Ok(())
    }

    // ── Read (gather from paged blocks) ────────────────────────────

    /// Gather contiguous key and value vectors for `seq_idx` into
    /// caller-supplied buffers.
    ///
    /// Buffers must each have length ≥ `total_tokens * head_dim`.
    #[cfg(target_arch = "aarch64")]
    pub fn read_cache(
        &mut self,
        seq_idx: usize,
        keys_out: &mut [f32],
        values_out: &mut [f32],
    ) -> Result<usize, &'static str> {
        if seq_idx >= self.sequences.len() {
            return Err("seq_idx out of range");
        }
        let hd = self.head_dim;
        let total = self.sequences[seq_idx].total_tokens;
        if keys_out.len() < total * hd || values_out.len() < total * hd {
            return Err("output buffer too small");
        }

        let mut offset = 0usize;
        let num_pages = self.sequences[seq_idx].page_table.len();
        for pi in 0..num_pages {
            let phys = self.sequences[seq_idx].page_table[pi].physical_block;
            let blk = &self.blocks[phys];
            let count = blk.tokens_stored * hd;
            neon_copy_f32(&blk.keys[..count], &mut keys_out[offset..offset + count]);
            neon_copy_f32(&blk.values[..count], &mut values_out[offset..offset + count]);
            offset += count;
            self.touch(phys);
        }

        Ok(total)
    }

    // ── Copy-on-write ──────────────────────────────────────────────

    /// Fork `src_seq_idx` into a new sequence for beam search /
    /// speculative decoding. Pages are shared with CoW semantics.
    pub fn fork_sequence(
        &mut self,
        src_seq_idx: usize,
        new_seq_id: u64,
    ) -> Result<usize, &'static str> {
        if src_seq_idx >= self.sequences.len() {
            return Err("src_seq_idx out of range");
        }
        let src = &self.sequences[src_seq_idx];
        let new_table: Vec<PageTableEntry> = src
            .page_table
            .iter()
            .map(|e| PageTableEntry { physical_block: e.physical_block, cow_pending: true })
            .collect();
        let total_tokens = src.total_tokens;

        // Bump ref counts on shared physical blocks.
        for entry in &new_table {
            self.blocks[entry.physical_block].ref_count += 1;
        }
        // Mark source pages as CoW too.
        for entry in &mut self.sequences[src_seq_idx].page_table {
            entry.cow_pending = true;
        }

        let idx = self.sequences.len();
        self.sequences.push(SequenceState {
            seq_id: new_seq_id,
            page_table: new_table,
            total_tokens,
        });
        Ok(idx)
    }

    /// Materialise a CoW page: duplicate the tail block of `seq_idx`
    /// so it is exclusively owned.
    #[cfg(target_arch = "aarch64")]
    fn cow_duplicate(&mut self, seq_idx: usize) -> Result<(), &'static str> {
        let last = match self.sequences[seq_idx].page_table.last() {
            Some(e) => *e,
            None => return Ok(()),
        };
        if !last.cow_pending {
            return Ok(());
        }

        let new_phys = self.alloc_block().ok_or("no free blocks for CoW")?;
        let old_phys = last.physical_block;

        let count = self.blocks[old_phys].tokens_stored * self.head_dim;
        // Copy key and value data.
        // Split borrows: copy to a temp then into new block.
        let k_tmp: Vec<f32> = self.blocks[old_phys].keys[..count].to_vec();
        let v_tmp: Vec<f32> = self.blocks[old_phys].values[..count].to_vec();
        self.blocks[new_phys].keys[..count].copy_from_slice(&k_tmp);
        self.blocks[new_phys].values[..count].copy_from_slice(&v_tmp);
        self.blocks[new_phys].tokens_stored = self.blocks[old_phys].tokens_stored;

        // Decrement old ref.
        self.blocks[old_phys].ref_count = self.blocks[old_phys].ref_count.saturating_sub(1);

        let entry = self.sequences[seq_idx].page_table.last_mut().unwrap();
        entry.physical_block = new_phys;
        entry.cow_pending = false;

        Ok(())
    }

    // ── Paged attention ────────────────────────────────────────────

    /// Compute scaled dot-product attention scores `Q·K^T / √d` over
    /// the paged KV cache for a single query head.
    ///
    /// - `query`: `[head_dim]`
    /// - `scores_out`: must have length ≥ `total_tokens` for `seq_idx`
    ///
    /// Returns the number of scores written.
    #[cfg(target_arch = "aarch64")]
    pub fn attention_scores(
        &mut self,
        seq_idx: usize,
        query: &[f32],
        scores_out: &mut [f32],
    ) -> Result<usize, &'static str> {
        if seq_idx >= self.sequences.len() {
            return Err("seq_idx out of range");
        }
        let hd = self.head_dim;
        if query.len() < hd {
            return Err("query shorter than head_dim");
        }
        let total = self.sequences[seq_idx].total_tokens;
        if scores_out.len() < total {
            return Err("scores_out buffer too small");
        }

        let inv_sqrt = 1.0 / (hd as f32).sqrt();
        let mut pos = 0usize;

        let num_pages = self.sequences[seq_idx].page_table.len();
        for pi in 0..num_pages {
            let phys = self.sequences[seq_idx].page_table[pi].physical_block;
            let blk = &self.blocks[phys];
            for t in 0..blk.tokens_stored {
                let key_off = t * hd;
                let key = &blk.keys[key_off..key_off + hd];
                let dot = neon_dot_f32(query, key, hd);
                scores_out[pos] = dot * inv_sqrt;
                pos += 1;
            }
            self.touch(phys);
        }

        Ok(pos)
    }
}

// ── NEON helper: vectorised f32 copy ───────────────────────────────

/// Copy `src` into `dst` using NEON 128-bit loads/stores with a scalar
/// tail for elements not aligned to 4.
#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_copy_f32(src: &[f32], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    let chunks = len / NEON_F32_LANES;
    let remainder = len % NEON_F32_LANES;

    for i in 0..chunks {
        let base = i * NEON_F32_LANES;
        unsafe {
            let v = vld1q_f32(src.as_ptr().add(base));
            vst1q_f32(dst.as_mut_ptr().add(base), v);
        }
    }

    let tail = chunks * NEON_F32_LANES;
    dst[tail..tail + remainder].copy_from_slice(&src[tail..tail + remainder]);
}

// ── NEON helper: vectorised dot product ────────────────────────────

/// Compute the dot product of two `f32` slices using NEON FMA with a
/// scalar tail.
#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    let chunks = len / NEON_F32_LANES;
    let remainder = len % NEON_F32_LANES;

    let mut acc = unsafe { vdupq_n_f32(0.0) };

    for c in 0..chunks {
        let base = c * NEON_F32_LANES;
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(base));
            let vb = vld1q_f32(b.as_ptr().add(base));
            acc = vfmaq_f32(acc, va, vb);
        }
    }

    let mut dot: f32 = unsafe { vaddvq_f32(acc) };

    let tail = chunks * NEON_F32_LANES;
    for i in 0..remainder {
        dot += a[tail + i] * b[tail + i];
    }
    dot
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    fn make_pool(num_blocks: usize, block_size: usize, head_dim: usize) -> PagedKvCachePool {
        PagedKvCachePool::new(num_blocks, block_size, head_dim, EvictionPolicy::Lru)
    }

    // -- Construction ------------------------------------------------

    #[test]
    fn test_pool_creation() {
        let pool = make_pool(8, 16, 64);
        assert_eq!(pool.total_blocks(), 8);
        assert_eq!(pool.free_count(), 8);
    }

    #[test]
    #[should_panic(expected = "num_blocks must be > 0")]
    fn test_pool_zero_blocks_panics() {
        let _ = make_pool(0, 16, 64);
    }

    #[test]
    #[should_panic(expected = "block_size must be > 0")]
    fn test_pool_zero_block_size_panics() {
        let _ = make_pool(8, 0, 64);
    }

    #[test]
    #[should_panic(expected = "head_dim must be > 0")]
    fn test_pool_zero_head_dim_panics() {
        let _ = make_pool(8, 16, 0);
    }

    // -- Block basics ------------------------------------------------

    #[test]
    fn test_block_new() {
        let blk = PagedCacheBlock::new(16, 64);
        assert_eq!(blk.block_size, 16);
        assert_eq!(blk.head_dim, 64);
        assert_eq!(blk.tokens_stored, 0);
        assert_eq!(blk.remaining(), 16);
        assert!(!blk.is_full());
    }

    #[test]
    fn test_block_memory_bytes() {
        let blk = PagedCacheBlock::new(16, 64);
        // keys + values = 2 * 16 * 64 * 4 bytes
        assert_eq!(blk.memory_bytes(), 2 * 16 * 64 * 4);
    }

    // -- Sequence management -----------------------------------------

    #[test]
    fn test_add_sequence() {
        let mut pool = make_pool(8, 4, 4);
        let idx = pool.add_sequence(100);
        assert_eq!(idx, 0);
        assert_eq!(pool.find_sequence(100), Some(0));
        assert_eq!(pool.find_sequence(999), None);
    }

    #[test]
    fn test_remove_sequence_frees_blocks() {
        let mut pool = make_pool(8, 4, 4);
        let seq = pool.add_sequence(1);
        let keys = vec![1.0f32; 4 * 4]; // 4 tokens
        let vals = vec![2.0f32; 4 * 4];
        pool.write_cache(seq, &keys, &vals).unwrap();
        let free_before = pool.free_count();
        pool.remove_sequence(seq);
        assert!(pool.free_count() > free_before);
    }

    // -- Write / read ------------------------------------------------

    #[test]
    fn test_write_and_read_single_token() {
        let hd = 8;
        let mut pool = make_pool(4, 4, hd);
        let seq = pool.add_sequence(1);

        let keys: Vec<f32> = (0..hd).map(|i| i as f32).collect();
        let vals: Vec<f32> = (0..hd).map(|i| (i as f32) + 100.0).collect();
        pool.write_cache(seq, &keys, &vals).unwrap();

        let mut k_out = vec![0.0f32; hd];
        let mut v_out = vec![0.0f32; hd];
        let n = pool.read_cache(seq, &mut k_out, &mut v_out).unwrap();
        assert_eq!(n, 1);
        assert_eq!(k_out, keys);
        assert_eq!(v_out, vals);
    }

    #[test]
    fn test_write_spanning_multiple_blocks() {
        let hd = 4;
        let block_size = 2;
        let mut pool = make_pool(8, block_size, hd);
        let seq = pool.add_sequence(1);

        // Write 5 tokens → should span 3 blocks (2+2+1).
        let num_tokens = 5;
        let keys: Vec<f32> = (0..num_tokens * hd).map(|i| i as f32).collect();
        let vals: Vec<f32> = (0..num_tokens * hd).map(|i| (i as f32) + 1000.0).collect();
        pool.write_cache(seq, &keys, &vals).unwrap();

        assert_eq!(pool.sequences[seq].total_tokens, 5);
        assert_eq!(pool.sequences[seq].num_blocks(), 3);

        let mut k_out = vec![0.0f32; num_tokens * hd];
        let mut v_out = vec![0.0f32; num_tokens * hd];
        let n = pool.read_cache(seq, &mut k_out, &mut v_out).unwrap();
        assert_eq!(n, 5);
        assert_eq!(k_out, keys);
        assert_eq!(v_out, vals);
    }

    #[test]
    fn test_write_invalid_seq() {
        let mut pool = make_pool(4, 4, 4);
        let res = pool.write_cache(99, &[1.0; 4], &[2.0; 4]);
        assert!(res.is_err());
    }

    #[test]
    fn test_write_mismatched_lengths() {
        let mut pool = make_pool(4, 4, 4);
        let seq = pool.add_sequence(1);
        let res = pool.write_cache(seq, &[1.0; 4], &[2.0; 8]);
        assert!(res.is_err());
    }

    #[test]
    fn test_write_bad_alignment() {
        let mut pool = make_pool(4, 4, 4);
        let seq = pool.add_sequence(1);
        let res = pool.write_cache(seq, &[1.0; 3], &[2.0; 3]);
        assert!(res.is_err());
    }

    #[test]
    fn test_read_invalid_seq() {
        let mut pool = make_pool(4, 4, 4);
        let mut k = [0.0; 4];
        let mut v = [0.0; 4];
        let res = pool.read_cache(99, &mut k, &mut v);
        assert!(res.is_err());
    }

    #[test]
    fn test_read_buffer_too_small() {
        let hd = 4;
        let mut pool = make_pool(4, 4, hd);
        let seq = pool.add_sequence(1);
        pool.write_cache(seq, &[1.0; 4], &[2.0; 4]).unwrap();
        let mut k = [0.0; 2]; // too small
        let mut v = [0.0; 2];
        let res = pool.read_cache(seq, &mut k, &mut v);
        assert!(res.is_err());
    }

    // -- Attention scores --------------------------------------------

    #[test]
    fn test_attention_scores_basic() {
        let hd = 4;
        let mut pool = make_pool(4, 4, hd);
        let seq = pool.add_sequence(1);

        // Two key vectors.
        let keys = vec![
            1.0, 0.0, 0.0, 0.0, // k0: dot with q = 1.0
            0.0, 1.0, 0.0, 0.0, // k1: dot with q = 0.0
        ];
        let vals = vec![0.0f32; 2 * hd];
        pool.write_cache(seq, &keys, &vals).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let mut scores = [0.0f32; 2];
        let n = pool.attention_scores(seq, &query, &mut scores).unwrap();
        assert_eq!(n, 2);

        let inv_sqrt = 1.0 / (hd as f32).sqrt();
        assert!((scores[0] - 1.0 * inv_sqrt).abs() < 1e-5);
        assert!((scores[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_attention_scores_across_blocks() {
        let hd = 4;
        let block_size = 1; // 1 token per block → forces paging
        let mut pool = PagedKvCachePool::new(8, block_size, hd, EvictionPolicy::Lru);
        let seq = pool.add_sequence(1);

        // 3 keys, each in its own block.
        let keys = vec![
            1.0, 0.0, 0.0, 0.0, // dot = 1
            0.0, 0.0, 1.0, 0.0, // dot = 0
            0.5, 0.5, 0.0, 0.0, // dot = 0.5
        ];
        let vals = vec![0.0f32; 3 * hd];
        pool.write_cache(seq, &keys, &vals).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let mut scores = [0.0f32; 3];
        let n = pool.attention_scores(seq, &query, &mut scores).unwrap();
        assert_eq!(n, 3);

        let inv = 1.0 / (hd as f32).sqrt();
        assert!((scores[0] - inv).abs() < 1e-5);
        assert!((scores[1]).abs() < 1e-5);
        assert!((scores[2] - 0.5 * inv).abs() < 1e-5);
    }

    #[test]
    fn test_attention_invalid_seq() {
        let mut pool = make_pool(4, 4, 4);
        let q = [1.0; 4];
        let mut s = [0.0; 1];
        assert!(pool.attention_scores(99, &q, &mut s).is_err());
    }

    #[test]
    fn test_attention_query_too_short() {
        let mut pool = make_pool(4, 4, 8);
        let seq = pool.add_sequence(1);
        pool.write_cache(seq, &vec![0.0; 8], &vec![0.0; 8]).unwrap();
        let q = [1.0; 4]; // head_dim is 8
        let mut s = [0.0; 1];
        assert!(pool.attention_scores(seq, &q, &mut s).is_err());
    }

    // -- Copy-on-write -----------------------------------------------

    #[test]
    fn test_fork_sequence() {
        let hd = 4;
        let mut pool = make_pool(8, 4, hd);
        let seq0 = pool.add_sequence(1);
        let keys = vec![1.0f32; 2 * hd];
        let vals = vec![2.0f32; 2 * hd];
        pool.write_cache(seq0, &keys, &vals).unwrap();

        let seq1 = pool.fork_sequence(seq0, 2).unwrap();

        // Both sequences should read back the same data.
        let mut k0 = vec![0.0f32; 2 * hd];
        let mut v0 = vec![0.0f32; 2 * hd];
        let mut k1 = vec![0.0f32; 2 * hd];
        let mut v1 = vec![0.0f32; 2 * hd];
        pool.read_cache(seq0, &mut k0, &mut v0).unwrap();
        pool.read_cache(seq1, &mut k1, &mut v1).unwrap();
        assert_eq!(k0, k1);
        assert_eq!(v0, v1);
    }

    #[test]
    fn test_cow_write_diverges() {
        let hd = 4;
        let mut pool = make_pool(16, 4, hd);
        let seq0 = pool.add_sequence(1);
        let keys = vec![1.0f32; hd];
        let vals = vec![2.0f32; hd];
        pool.write_cache(seq0, &keys, &vals).unwrap();

        let seq1 = pool.fork_sequence(seq0, 2).unwrap();

        // Write different data to the fork.
        let new_keys = vec![9.0f32; hd];
        let new_vals = vec![8.0f32; hd];
        pool.write_cache(seq1, &new_keys, &new_vals).unwrap();

        // Original should be unaffected.
        let mut k0 = vec![0.0f32; hd];
        let mut v0 = vec![0.0f32; hd];
        pool.read_cache(seq0, &mut k0, &mut v0).unwrap();
        assert_eq!(k0, keys);
        assert_eq!(v0, vals);
    }

    #[test]
    fn test_fork_invalid_seq() {
        let mut pool = make_pool(4, 4, 4);
        assert!(pool.fork_sequence(99, 2).is_err());
    }

    // -- Eviction: LRU -----------------------------------------------

    #[test]
    fn test_lru_eviction_frees_oldest() {
        let hd = 4;
        // 2 blocks total, each holds 1 token.
        let mut pool = PagedKvCachePool::new(2, 1, hd, EvictionPolicy::Lru);
        let seq0 = pool.add_sequence(1);
        let seq1 = pool.add_sequence(2);

        // Fill both blocks.
        pool.write_cache(seq0, &vec![1.0; hd], &vec![1.0; hd]).unwrap();
        pool.write_cache(seq1, &vec![2.0; hd], &vec![2.0; hd]).unwrap();
        assert_eq!(pool.free_count(), 0);

        // Touch seq1 to make seq0 the LRU victim.
        let mut k = vec![0.0f32; hd];
        let mut v = vec![0.0f32; hd];
        pool.read_cache(seq1, &mut k, &mut v).unwrap();

        // Writing a third sequence should evict seq0's block.
        let seq2 = pool.add_sequence(3);
        let res = pool.write_cache(seq2, &vec![3.0; hd], &vec![3.0; hd]);
        assert!(res.is_ok());
    }

    // -- Eviction: sliding window ------------------------------------

    #[test]
    fn test_sliding_window_eviction() {
        let hd = 4;
        // 4 blocks, each 1 token; window = 2 tokens.
        let mut pool =
            PagedKvCachePool::new(4, 1, hd, EvictionPolicy::SlidingWindow { window_size: 2 });
        let seq = pool.add_sequence(1);

        // Write 4 tokens (fills all blocks).
        for i in 0..4 {
            pool.write_cache(seq, &vec![(i + 1) as f32; hd], &vec![(i + 1) as f32; hd]).unwrap();
        }

        // With window=2 and 4 tokens, eviction should have trimmed
        // the oldest blocks.
        assert!(pool.sequences[seq].total_tokens <= 4);
    }

    // -- NEON helpers ------------------------------------------------

    #[test]
    fn test_neon_copy_aligned() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = [0.0f32; 16];
        neon_copy_f32(&src, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_neon_copy_unaligned_tail() {
        let src: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let mut dst = [0.0f32; 7];
        neon_copy_f32(&src, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_neon_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        // 2 + 6 + 12 + 20 + 30 = 70
        let dot = neon_dot_f32(&a, &b, 5);
        assert!((dot - 70.0).abs() < 1e-5);
    }

    #[test]
    fn test_neon_dot_single_element() {
        let a = [3.0];
        let b = [4.0];
        let dot = neon_dot_f32(&a, &b, 1);
        assert!((dot - 12.0).abs() < 1e-5);
    }

    // -- Memory accounting -------------------------------------------

    #[test]
    fn test_pool_memory_bytes() {
        let pool = make_pool(4, 16, 64);
        // 4 blocks × (16 × 64 × 4 bytes × 2 for k+v)
        let expected = 4 * 2 * 16 * 64 * 4;
        assert_eq!(pool.memory_bytes(), expected);
    }

    // -- Incremental append ------------------------------------------

    #[test]
    fn test_incremental_append_then_read() {
        let hd = 4;
        let mut pool = make_pool(8, 4, hd);
        let seq = pool.add_sequence(1);

        for i in 0..6 {
            let k = vec![(i + 1) as f32; hd];
            let v = vec![((i + 1) * 10) as f32; hd];
            pool.write_cache(seq, &k, &v).unwrap();
        }
        assert_eq!(pool.sequences[seq].total_tokens, 6);

        let mut k_out = vec![0.0f32; 6 * hd];
        let mut v_out = vec![0.0f32; 6 * hd];
        let n = pool.read_cache(seq, &mut k_out, &mut v_out).unwrap();
        assert_eq!(n, 6);

        // First token's key should be [1,1,1,1].
        assert!((k_out[0] - 1.0).abs() < 1e-5);
        // Last token's key should be [6,6,6,6].
        assert!((k_out[5 * hd] - 6.0).abs() < 1e-5);
    }

    // -- Scaffolding tests -------------------------------------------

    #[test]
    #[ignore = "TDD scaffold: requires fp16 NEON storage path"]
    fn test_fp16_cache_blocks() {
        unimplemented!();
    }

    #[test]
    #[ignore = "TDD scaffold: requires multi-head paged attention"]
    fn test_multi_head_paged_attention() {
        unimplemented!();
    }

    #[test]
    #[ignore = "TDD scaffold: requires prefetch hint integration"]
    fn test_neon_prefetch_hint() {
        unimplemented!();
    }
}
