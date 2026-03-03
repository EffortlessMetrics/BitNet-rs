//! NEON-optimized memory allocator/manager for Apple Silicon inference.
//!
//! Provides a tensor memory arena with aligned allocation, deallocation, and
//! a memory planner that computes optimal tensor placement with lifetime-based
//! reuse. 256-byte alignment ensures Metal buffer compatibility.
//!
//! Pure Rust implementation — no NEON intrinsics, no unsafe blocks.

/// Default alignment for Metal buffer compatibility (256 bytes).
pub const METAL_ALIGNMENT: usize = 256;

// ── Helpers ─────────────────────────────────────────────────────────────

/// Align `size` up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two; returns `size` unchanged when it is
/// already aligned. Returns 0 for zero-size inputs.
#[inline]
pub fn align_to(size: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two(), "alignment must be power of two");
    if alignment == 0 {
        return size;
    }
    (size + alignment - 1) & !(alignment - 1)
}

/// Compute total tensor memory in bytes, including alignment padding to
/// `METAL_ALIGNMENT`.
pub fn compute_tensor_size(shape: &[usize], dtype_bytes: usize) -> usize {
    if shape.is_empty() {
        return 0;
    }
    let elements: usize = shape.iter().copied().product();
    let raw = elements * dtype_bytes;
    align_to(raw, METAL_ALIGNMENT)
}

// ── MemoryBlock ─────────────────────────────────────────────────────────

/// A handle to an allocated region inside a [`MemoryArena`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBlock {
    /// Byte offset from the arena start (aligned).
    pub offset: usize,
    /// Usable size in bytes (as requested, before padding).
    pub size: usize,
    /// Actual size consumed in the arena (aligned).
    pub aligned_size: usize,
}

// ── Free block tracking ─────────────────────────────────────────────────

/// Internal free-list entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeBlock {
    offset: usize,
    size: usize,
}

// ── MemoryArena ─────────────────────────────────────────────────────────

/// Pre-allocated memory pool for tensor allocations.
///
/// The arena hands out [`MemoryBlock`] handles without actually backing them
/// with storage — callers use the `offset` to index into their own buffer.
/// This keeps the planner zero-copy and allocation-free on the hot path.
pub struct MemoryArena {
    capacity: usize,
    used: usize,
    high_water: usize,
    free_list: Vec<FreeBlock>,
    /// Bump pointer for fast sequential allocation when the free-list is empty.
    bump: usize,
    alloc_count: usize,
    dealloc_count: usize,
}

impl MemoryArena {
    /// Create a new arena with the given byte capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            used: 0,
            high_water: 0,
            free_list: Vec::new(),
            bump: 0,
            alloc_count: 0,
            dealloc_count: 0,
        }
    }

    /// Try to allocate `size` bytes with the given `alignment`.
    ///
    /// Returns `None` when the arena cannot satisfy the request.
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Option<MemoryBlock> {
        if size == 0 {
            return Some(MemoryBlock { offset: 0, size: 0, aligned_size: 0 });
        }
        let aligned_size = align_to(size, alignment);

        // First-fit search on the free list.
        if let Some(idx) = self.find_free_block(aligned_size, alignment) {
            let free = self.free_list[idx];
            let aligned_offset = align_to(free.offset, alignment);
            let padding = aligned_offset - free.offset;
            let total_needed = padding + aligned_size;

            if total_needed <= free.size {
                // Carve out the allocation; keep any remainder.
                self.free_list.remove(idx);

                // Leading fragment (before alignment padding).
                if padding > 0 {
                    self.free_list.push(FreeBlock { offset: free.offset, size: padding });
                }
                // Trailing fragment.
                let remainder = free.size - total_needed;
                if remainder > 0 {
                    self.free_list
                        .push(FreeBlock { offset: aligned_offset + aligned_size, size: remainder });
                }

                self.used += aligned_size;
                if self.used > self.high_water {
                    self.high_water = self.used;
                }
                self.alloc_count += 1;
                return Some(MemoryBlock { offset: aligned_offset, size, aligned_size });
            }
        }

        // Bump allocation from the tail of the arena.
        let aligned_offset = align_to(self.bump, alignment);
        if aligned_offset + aligned_size > self.capacity {
            return None; // out of capacity
        }

        // Gap between old bump and aligned start becomes a free fragment.
        let gap = aligned_offset - self.bump;
        if gap > 0 {
            self.free_list.push(FreeBlock { offset: self.bump, size: gap });
        }

        self.bump = aligned_offset + aligned_size;
        self.used += aligned_size;
        if self.used > self.high_water {
            self.high_water = self.used;
        }
        self.alloc_count += 1;
        Some(MemoryBlock { offset: aligned_offset, size, aligned_size })
    }

    /// Return a block to the arena.
    pub fn deallocate(&mut self, block: MemoryBlock) {
        if block.aligned_size == 0 {
            return;
        }
        self.used = self.used.saturating_sub(block.aligned_size);
        self.dealloc_count += 1;

        // Insert into free list and coalesce neighbours.
        self.free_list.push(FreeBlock { offset: block.offset, size: block.aligned_size });
        self.coalesce();
    }

    /// Reset the arena — O(1) bulk deallocation.
    pub fn reset(&mut self) {
        self.used = 0;
        self.bump = 0;
        self.free_list.clear();
        self.alloc_count = 0;
        self.dealloc_count = 0;
    }

    /// Bytes currently allocated.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Total arena capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// High-water mark — maximum `used` seen since last `reset`.
    pub fn high_water_mark(&self) -> usize {
        self.high_water
    }

    /// Fragmentation ratio in `[0.0, 1.0]`.
    ///
    /// 0.0 means no fragmentation (all free space is contiguous at the tail).
    /// Higher values indicate scattered free blocks.
    pub fn fragmentation(&self) -> f32 {
        if self.used == 0 && self.bump == 0 {
            return 0.0;
        }
        let total_free: usize = self.free_list.iter().map(|b| b.size).sum();
        let committed = self.bump;
        if committed == 0 {
            return 0.0;
        }
        let ideal_free = committed.saturating_sub(self.used);
        if ideal_free == 0 {
            return 0.0;
        }
        // Fragmentation = 1 - (largest_contiguous_free / total_free).
        let largest = self.free_list.iter().map(|b| b.size).max().unwrap_or(0);
        if total_free == 0 {
            return 0.0;
        }
        1.0 - (largest as f32 / total_free as f32)
    }

    // ── Internal helpers ────────────────────────────────────────────────

    /// First-fit search that respects alignment.
    fn find_free_block(&self, needed: usize, alignment: usize) -> Option<usize> {
        for (i, blk) in self.free_list.iter().enumerate() {
            let aligned_offset = align_to(blk.offset, alignment);
            let padding = aligned_offset - blk.offset;
            if padding + needed <= blk.size {
                return Some(i);
            }
        }
        None
    }

    /// Sort the free list by offset and merge adjacent blocks.
    fn coalesce(&mut self) {
        if self.free_list.len() < 2 {
            return;
        }
        self.free_list.sort_by_key(|b| b.offset);
        let mut merged: Vec<FreeBlock> = Vec::with_capacity(self.free_list.len());
        let mut current = self.free_list[0];
        for blk in self.free_list.iter().skip(1) {
            if current.offset + current.size == blk.offset {
                current.size += blk.size;
            } else {
                merged.push(current);
                current = *blk;
            }
        }
        merged.push(current);
        self.free_list = merged;
    }
}

// ── MemoryAssignment ────────────────────────────────────────────────────

/// Planned placement of one tensor inside a shared memory buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryAssignment {
    /// Tensor index (matches input order).
    pub tensor_id: usize,
    /// Byte offset in the shared buffer.
    pub offset: usize,
    /// Size in bytes (aligned).
    pub size: usize,
}

// ── MemoryPlanner ───────────────────────────────────────────────────────

/// Plans memory reuse across inference layers using interval colouring.
pub struct MemoryPlanner;

impl MemoryPlanner {
    /// Compute an optimal memory layout for tensors with known shapes and
    /// lifetimes.
    ///
    /// `shapes` — `(rows, cols)` per tensor (element count = rows * cols,
    ///   each element is `f32` = 4 bytes).
    /// `lifetimes` — `(start_layer, end_layer)` per tensor (inclusive).
    ///
    /// Tensors whose lifetimes do not overlap may share the same memory
    /// region.
    pub fn plan_allocations(
        shapes: &[(usize, usize)],
        lifetimes: &[(usize, usize)],
    ) -> Vec<MemoryAssignment> {
        assert_eq!(shapes.len(), lifetimes.len(), "shapes and lifetimes must match");
        let n = shapes.len();
        if n == 0 {
            return Vec::new();
        }

        // Compute aligned sizes.
        let sizes: Vec<usize> =
            shapes.iter().map(|&(r, c)| align_to(r * c * 4, METAL_ALIGNMENT)).collect();

        // Sort tensors by start time, breaking ties by descending size for
        // tighter packing.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            lifetimes[a].0.cmp(&lifetimes[b].0).then_with(|| sizes[b].cmp(&sizes[a]))
        });

        // Greedy interval-colouring with first-fit offset assignment.
        let mut assignments: Vec<MemoryAssignment> =
            vec![MemoryAssignment { tensor_id: 0, offset: 0, size: 0 }; n];
        // Track placed intervals: (offset, end_of_region, lifetime_end).
        let mut placed: Vec<(usize, usize, usize)> = Vec::new();

        for &idx in &order {
            let size = sizes[idx];
            let (lt_start, lt_end) = lifetimes[idx];

            // Collect regions still alive (overlapping lifetime).
            let mut occupied: Vec<(usize, usize)> = placed
                .iter()
                .filter(|&&(_, _, end)| end >= lt_start)
                .map(|&(off, region_end, _)| (off, region_end))
                .collect();
            occupied.sort_by_key(|&(off, _)| off);

            // First-fit in gaps.
            let mut offset = 0usize;
            for &(occ_start, occ_end) in &occupied {
                if offset + size <= occ_start {
                    break;
                }
                if occ_end > offset {
                    offset = occ_end;
                }
            }

            assignments[idx] = MemoryAssignment { tensor_id: idx, offset, size };
            placed.push((offset, offset + size, lt_end));
        }

        assignments
    }

    /// Compute peak (maximum) memory needed for the given assignments.
    pub fn peak_memory(assignments: &[MemoryAssignment]) -> usize {
        assignments.iter().map(|a| a.offset + a.size).max().unwrap_or(0)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── align_to ────────────────────────────────────────────────────────

    #[test]
    fn test_align_to_already_aligned() {
        assert_eq!(align_to(256, 256), 256);
    }

    #[test]
    fn test_align_to_needs_padding() {
        assert_eq!(align_to(100, 256), 256);
    }

    #[test]
    fn test_align_to_16() {
        assert_eq!(align_to(17, 16), 32);
        assert_eq!(align_to(16, 16), 16);
    }

    #[test]
    fn test_align_to_64() {
        assert_eq!(align_to(65, 64), 128);
        assert_eq!(align_to(64, 64), 64);
    }

    #[test]
    fn test_align_to_256() {
        assert_eq!(align_to(1, 256), 256);
        assert_eq!(align_to(257, 256), 512);
    }

    #[test]
    fn test_align_to_zero_size() {
        assert_eq!(align_to(0, 256), 0);
    }

    #[test]
    fn test_align_to_one() {
        assert_eq!(align_to(7, 1), 7);
    }

    // ── compute_tensor_size ─────────────────────────────────────────────

    #[test]
    fn test_compute_tensor_size_basic() {
        // 10 * 10 * 4 = 400 → aligned to 512
        assert_eq!(compute_tensor_size(&[10, 10], 4), 512);
    }

    #[test]
    fn test_compute_tensor_size_exact_alignment() {
        // 8 * 8 * 4 = 256 → already aligned
        assert_eq!(compute_tensor_size(&[8, 8], 4), 256);
    }

    #[test]
    fn test_compute_tensor_size_3d() {
        // 2 * 3 * 4 * 4 = 96 → 256
        assert_eq!(compute_tensor_size(&[2, 3, 4], 4), 256);
    }

    #[test]
    fn test_compute_tensor_size_empty_shape() {
        assert_eq!(compute_tensor_size(&[], 4), 0);
    }

    #[test]
    fn test_compute_tensor_size_single_element() {
        assert_eq!(compute_tensor_size(&[1], 4), 256);
    }

    #[test]
    fn test_compute_tensor_size_large() {
        // 1024 * 1024 * 4 = 4_194_304 — already a multiple of 256
        assert_eq!(compute_tensor_size(&[1024, 1024], 4), 4_194_304);
    }

    #[test]
    fn test_compute_tensor_size_f16() {
        // 16 * 16 * 2 = 512 — aligned
        assert_eq!(compute_tensor_size(&[16, 16], 2), 512);
    }

    // ── MemoryArena — basic allocation ──────────────────────────────────

    #[test]
    fn test_arena_new() {
        let arena = MemoryArena::new(4096);
        assert_eq!(arena.capacity(), 4096);
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_arena_single_alloc() {
        let mut arena = MemoryArena::new(4096);
        let blk = arena.allocate(100, 16).unwrap();
        assert_eq!(blk.offset % 16, 0);
        assert!(blk.aligned_size >= 100);
        assert!(arena.used() > 0);
    }

    #[test]
    fn test_arena_multiple_allocs() {
        let mut arena = MemoryArena::new(8192);
        let a = arena.allocate(256, 256).unwrap();
        let b = arena.allocate(256, 256).unwrap();
        assert_ne!(a.offset, b.offset);
        assert!(b.offset >= a.offset + a.aligned_size);
    }

    #[test]
    fn test_arena_alignment_16() {
        let mut arena = MemoryArena::new(4096);
        for _ in 0..10 {
            let blk = arena.allocate(33, 16).unwrap();
            assert_eq!(blk.offset % 16, 0);
        }
    }

    #[test]
    fn test_arena_alignment_64() {
        let mut arena = MemoryArena::new(8192);
        for _ in 0..10 {
            let blk = arena.allocate(50, 64).unwrap();
            assert_eq!(blk.offset % 64, 0);
        }
    }

    #[test]
    fn test_arena_alignment_256() {
        let mut arena = MemoryArena::new(65536);
        for _ in 0..10 {
            let blk = arena.allocate(100, 256).unwrap();
            assert_eq!(blk.offset % 256, 0);
        }
    }

    #[test]
    fn test_arena_over_capacity() {
        let mut arena = MemoryArena::new(256);
        assert!(arena.allocate(512, 16).is_none());
    }

    #[test]
    fn test_arena_exact_capacity() {
        let mut arena = MemoryArena::new(256);
        let blk = arena.allocate(256, 1);
        assert!(blk.is_some());
    }

    #[test]
    fn test_arena_zero_size_allocation() {
        let mut arena = MemoryArena::new(1024);
        let blk = arena.allocate(0, 16).unwrap();
        assert_eq!(blk.size, 0);
        assert_eq!(blk.aligned_size, 0);
        assert_eq!(arena.used(), 0);
    }

    // ── MemoryArena — deallocation ──────────────────────────────────────

    #[test]
    fn test_arena_dealloc_restores_used() {
        let mut arena = MemoryArena::new(4096);
        let blk = arena.allocate(256, 16).unwrap();
        let used_after_alloc = arena.used();
        arena.deallocate(blk);
        assert!(arena.used() < used_after_alloc);
    }

    #[test]
    fn test_arena_dealloc_then_realloc() {
        let mut arena = MemoryArena::new(512);
        let blk = arena.allocate(256, 256).unwrap();
        arena.deallocate(blk);
        // Should be able to reuse the freed space.
        let blk2 = arena.allocate(256, 256);
        assert!(blk2.is_some());
    }

    #[test]
    fn test_arena_dealloc_zero_block() {
        let mut arena = MemoryArena::new(1024);
        let blk = MemoryBlock { offset: 0, size: 0, aligned_size: 0 };
        arena.deallocate(blk); // should not panic
        assert_eq!(arena.used(), 0);
    }

    // ── MemoryArena — reset ─────────────────────────────────────────────

    #[test]
    fn test_arena_reset() {
        let mut arena = MemoryArena::new(4096);
        arena.allocate(256, 16).unwrap();
        arena.allocate(512, 64).unwrap();
        arena.reset();
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_arena_reset_allows_full_reuse() {
        let mut arena = MemoryArena::new(512);
        arena.allocate(512, 1).unwrap();
        assert!(arena.allocate(1, 1).is_none());
        arena.reset();
        let blk = arena.allocate(512, 1);
        assert!(blk.is_some());
    }

    // ── MemoryArena — capacity tracking ─────────────────────────────────

    #[test]
    fn test_arena_used_never_exceeds_capacity() {
        let mut arena = MemoryArena::new(4096);
        for _ in 0..100 {
            if arena.allocate(64, 16).is_none() {
                break;
            }
            assert!(arena.used() <= arena.capacity());
        }
    }

    #[test]
    fn test_arena_high_water_mark() {
        let mut arena = MemoryArena::new(8192);
        arena.allocate(1024, 16).unwrap();
        arena.allocate(2048, 16).unwrap();
        let hw = arena.high_water_mark();
        assert!(hw >= 1024 + 2048);
    }

    #[test]
    fn test_arena_used_after_mixed_ops() {
        let mut arena = MemoryArena::new(8192);
        let a = arena.allocate(256, 16).unwrap();
        let b = arena.allocate(256, 16).unwrap();
        arena.deallocate(a);
        let _c = arena.allocate(128, 16).unwrap();
        assert!(arena.used() <= arena.capacity());
        arena.deallocate(b);
        assert!(arena.used() <= arena.capacity());
    }

    // ── MemoryArena — fragmentation ─────────────────────────────────────

    #[test]
    fn test_fragmentation_empty() {
        let arena = MemoryArena::new(4096);
        assert_eq!(arena.fragmentation(), 0.0);
    }

    #[test]
    fn test_fragmentation_no_dealloc() {
        let mut arena = MemoryArena::new(4096);
        arena.allocate(256, 16).unwrap();
        assert_eq!(arena.fragmentation(), 0.0);
    }

    #[test]
    fn test_fragmentation_after_alternating_dealloc() {
        let mut arena = MemoryArena::new(65536);
        let mut blocks = Vec::new();
        for _ in 0..8 {
            blocks.push(arena.allocate(256, 16).unwrap());
        }
        // Deallocate every other block → scattered free space.
        for i in (0..8).step_by(2) {
            arena.deallocate(blocks[i]);
        }
        let frag = arena.fragmentation();
        assert!(frag > 0.0, "expected fragmentation after scattered deallocs, got {frag}");
    }

    #[test]
    fn test_fragmentation_after_full_dealloc() {
        let mut arena = MemoryArena::new(4096);
        let a = arena.allocate(256, 16).unwrap();
        let b = arena.allocate(256, 16).unwrap();
        arena.deallocate(a);
        arena.deallocate(b);
        // All free space should coalesce into one block → no fragmentation.
        assert_eq!(arena.fragmentation(), 0.0);
    }

    #[test]
    fn test_fragmentation_bounded_01() {
        let mut arena = MemoryArena::new(65536);
        let mut blocks = Vec::new();
        for _ in 0..16 {
            blocks.push(arena.allocate(128, 16).unwrap());
        }
        for i in (0..16).step_by(2) {
            arena.deallocate(blocks[i]);
        }
        let frag = arena.fragmentation();
        assert!((0.0..=1.0).contains(&frag), "fragmentation out of range: {frag}");
    }

    // ── MemoryArena — stress / property ─────────────────────────────────

    #[test]
    fn test_arena_many_small_allocs() {
        let mut arena = MemoryArena::new(1_048_576);
        for _ in 0..1000 {
            let blk = arena.allocate(64, 16);
            assert!(blk.is_some() || arena.used() >= arena.capacity() - 64);
        }
        assert!(arena.used() <= arena.capacity());
    }

    #[test]
    fn test_arena_alloc_dealloc_cycle() {
        let mut arena = MemoryArena::new(4096);
        for _ in 0..100 {
            let blk = arena.allocate(64, 16).unwrap();
            arena.deallocate(blk);
        }
        // After all cycles the used should be 0.
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_arena_alignment_larger_than_alloc() {
        let mut arena = MemoryArena::new(4096);
        let blk = arena.allocate(8, 256).unwrap();
        assert_eq!(blk.offset % 256, 0);
        assert!(blk.aligned_size >= 256);
    }

    #[test]
    fn test_arena_mixed_alignments() {
        let mut arena = MemoryArena::new(65536);
        let a = arena.allocate(100, 16).unwrap();
        let b = arena.allocate(200, 64).unwrap();
        let c = arena.allocate(300, 256).unwrap();
        assert_eq!(a.offset % 16, 0);
        assert_eq!(b.offset % 64, 0);
        assert_eq!(c.offset % 256, 0);
    }

    #[test]
    fn test_arena_no_overlap() {
        let mut arena = MemoryArena::new(65536);
        let mut blocks = Vec::new();
        for i in 0..20 {
            let size = 64 * (i + 1);
            blocks.push(arena.allocate(size, 16).unwrap());
        }
        // Verify no two blocks overlap.
        for i in 0..blocks.len() {
            for j in (i + 1)..blocks.len() {
                let a = &blocks[i];
                let b = &blocks[j];
                let a_end = a.offset + a.aligned_size;
                let b_end = b.offset + b.aligned_size;
                assert!(a_end <= b.offset || b_end <= a.offset, "blocks {i} and {j} overlap");
            }
        }
    }

    // ── MemoryPlanner — basic ───────────────────────────────────────────

    #[test]
    fn test_planner_empty() {
        let assignments = MemoryPlanner::plan_allocations(&[], &[]);
        assert!(assignments.is_empty());
    }

    #[test]
    fn test_planner_single_tensor() {
        let shapes = [(64, 64)];
        let lifetimes = [(0, 0)];
        let a = MemoryPlanner::plan_allocations(&shapes, &lifetimes);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].tensor_id, 0);
    }

    #[test]
    fn test_planner_non_overlapping_reuse() {
        // Two tensors with non-overlapping lifetimes should share memory.
        let shapes = [(64, 64); 2];
        let lifetimes = [(0, 1), (2, 3)];
        let a = MemoryPlanner::plan_allocations(&shapes, &lifetimes);
        assert_eq!(a[0].offset, a[1].offset, "non-overlapping tensors should reuse");
    }

    #[test]
    fn test_planner_overlapping_no_reuse() {
        // Two tensors alive at the same time must not share memory.
        let shapes = [(64, 64); 2];
        let lifetimes = [(0, 2), (1, 3)];
        let a = MemoryPlanner::plan_allocations(&shapes, &lifetimes);
        let end0 = a[0].offset + a[0].size;
        let end1 = a[1].offset + a[1].size;
        assert!(
            end0 <= a[1].offset || end1 <= a[0].offset,
            "overlapping tensors must not share memory"
        );
    }

    #[test]
    fn test_planner_linear_pipeline() {
        // A → B → C → D, each alive for one layer.
        let shapes = [(128, 128); 4];
        let lifetimes = [(0, 0), (1, 1), (2, 2), (3, 3)];
        let a = MemoryPlanner::plan_allocations(&shapes, &lifetimes);
        let peak = MemoryPlanner::peak_memory(&a);
        // All tensors can reuse the same slot.
        let single = a[0].size;
        assert_eq!(peak, single, "linear pipeline should reuse all slots");
    }

    #[test]
    fn test_planner_branching_dag() {
        // Layer 0: A produced
        // Layer 1: B produced, A still alive
        // Layer 2: C consumes A and B
        let shapes = [(32, 32); 3];
        let lifetimes = [(0, 2), (1, 2), (2, 2)];
        let a = MemoryPlanner::plan_allocations(&shapes, &lifetimes);
        let peak = MemoryPlanner::peak_memory(&a);
        let single = a[0].size;
        // At layer 2, all three are alive → need 3 slots.
        assert!(peak >= single * 3, "DAG branch needs ≥3 slots, peak={peak}");
    }

    #[test]
    fn test_planner_inplace_same_lifetime() {
        // Two tensors with identical lifetime and same size could theoretically
        // be in-place, but our planner separates them for correctness.
        let shapes = [(16, 16); 2];
        let lifetimes = [(0, 0), (0, 0)];
        let a = MemoryPlanner::plan_allocations(&shapes, &lifetimes);
        let end0 = a[0].offset + a[0].size;
        let end1 = a[1].offset + a[1].size;
        assert!(end0 <= a[1].offset || end1 <= a[0].offset);
    }

    #[test]
    fn test_planner_mixed_sizes() {
        let shapes = [(256, 256), (16, 16), (128, 128)];
        let lifetimes = [(0, 0), (0, 1), (1, 1)];
        let a = MemoryPlanner::plan_allocations(&shapes, &lifetimes);
        assert_eq!(a.len(), 3);
        // Tensor 0 and 2 don't overlap, so tensor 2 can reuse tensor 0's space.
        let peak = MemoryPlanner::peak_memory(&a);
        assert!(peak > 0);
    }

    // ── Peak memory ─────────────────────────────────────────────────────

    #[test]
    fn test_peak_memory_empty() {
        assert_eq!(MemoryPlanner::peak_memory(&[]), 0);
    }

    #[test]
    fn test_peak_memory_single() {
        let a = [MemoryAssignment { tensor_id: 0, offset: 0, size: 1024 }];
        assert_eq!(MemoryPlanner::peak_memory(&a), 1024);
    }

    #[test]
    fn test_peak_memory_stacked() {
        let a = [
            MemoryAssignment { tensor_id: 0, offset: 0, size: 512 },
            MemoryAssignment { tensor_id: 1, offset: 512, size: 512 },
        ];
        assert_eq!(MemoryPlanner::peak_memory(&a), 1024);
    }

    #[test]
    fn test_peak_memory_reused() {
        let a = [
            MemoryAssignment { tensor_id: 0, offset: 0, size: 1024 },
            MemoryAssignment { tensor_id: 1, offset: 0, size: 1024 },
        ];
        assert_eq!(MemoryPlanner::peak_memory(&a), 1024);
    }

    // ── Property tests ──────────────────────────────────────────────────

    #[test]
    fn test_property_used_le_capacity() {
        let mut arena = MemoryArena::new(8192);
        let mut live = Vec::new();
        for i in 0..200 {
            if i % 3 == 0 && !live.is_empty() {
                let blk = live.remove(0);
                arena.deallocate(blk);
            } else {
                if let Some(blk) = arena.allocate(32 + (i * 7) % 256, 16) {
                    live.push(blk);
                }
            }
            assert!(
                arena.used() <= arena.capacity(),
                "used {} exceeded capacity {}",
                arena.used(),
                arena.capacity()
            );
        }
    }

    #[test]
    fn test_property_alignment_always_correct() {
        let mut arena = MemoryArena::new(1_048_576);
        let alignments = [16, 64, 256];
        for (i, &align) in alignments.iter().cycle().take(100).enumerate() {
            let size = 16 + (i * 13) % 512;
            if let Some(blk) = arena.allocate(size, align) {
                assert_eq!(
                    blk.offset % align,
                    0,
                    "block {i}: offset {} not aligned to {align}",
                    blk.offset
                );
            }
        }
    }

    #[test]
    fn test_property_reset_full_reuse() {
        let mut arena = MemoryArena::new(4096);
        for _ in 0..5 {
            while arena.allocate(64, 16).is_some() {}
            arena.reset();
            assert_eq!(arena.used(), 0);
        }
    }

    #[test]
    fn test_property_planner_no_overlap() {
        // Random-ish lifetimes; verify no two concurrently-alive tensors overlap.
        let shapes = [(32, 32), (64, 16), (16, 64), (32, 32), (64, 64)];
        let lifetimes = [(0, 2), (1, 3), (3, 5), (0, 5), (4, 6)];
        let assignments = MemoryPlanner::plan_allocations(&shapes, &lifetimes);

        for i in 0..assignments.len() {
            for j in (i + 1)..assignments.len() {
                let (li_s, li_e) = lifetimes[assignments[i].tensor_id];
                let (lj_s, lj_e) = lifetimes[assignments[j].tensor_id];
                let overlaps_time = li_s <= lj_e && lj_s <= li_e;
                if overlaps_time {
                    let ai_end = assignments[i].offset + assignments[i].size;
                    let aj_end = assignments[j].offset + assignments[j].size;
                    assert!(
                        ai_end <= assignments[j].offset || aj_end <= assignments[i].offset,
                        "tensors {i} and {j} overlap in both time and memory"
                    );
                }
            }
        }
    }

    #[test]
    fn test_property_planner_peak_ge_max_single() {
        let shapes = [(128, 128), (64, 64), (256, 256)];
        let lifetimes = [(0, 0), (0, 0), (0, 0)];
        let a = MemoryPlanner::plan_allocations(&shapes, &lifetimes);
        let peak = MemoryPlanner::peak_memory(&a);
        let max_single = a.iter().map(|x| x.size).max().unwrap();
        assert!(peak >= max_single, "peak {peak} < largest tensor {max_single}");
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_arena_capacity_zero() {
        let mut arena = MemoryArena::new(0);
        assert!(arena.allocate(1, 1).is_none());
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_arena_repeated_reset() {
        let mut arena = MemoryArena::new(1024);
        for _ in 0..10 {
            arena.reset();
        }
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_arena_alloc_after_full_dealloc_reuse() {
        let mut arena = MemoryArena::new(1024);
        let a = arena.allocate(512, 16).unwrap();
        let b = arena.allocate(512, 16).unwrap();
        arena.deallocate(a);
        arena.deallocate(b);
        // Should be able to allocate the full capacity again.
        let c = arena.allocate(1024, 1);
        assert!(c.is_some(), "should reuse all freed space");
    }

    #[test]
    fn test_planner_many_tensors() {
        let n = 50;
        let shapes: Vec<(usize, usize)> = (0..n).map(|i| (16 + i * 4, 16)).collect();
        let lifetimes: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        let a = MemoryPlanner::plan_allocations(&shapes, &lifetimes);
        assert_eq!(a.len(), n);
        let peak = MemoryPlanner::peak_memory(&a);
        // All are sequential → should reuse, peak = max single tensor.
        let max_single = a.iter().map(|x| x.size).max().unwrap();
        assert_eq!(peak, max_single);
    }

    #[test]
    fn test_planner_all_alive_simultaneously() {
        let n = 5;
        let shapes: Vec<(usize, usize)> = vec![(32, 32); n];
        let lifetimes: Vec<(usize, usize)> = vec![(0, 10); n];
        let a = MemoryPlanner::plan_allocations(&shapes, &lifetimes);
        let peak = MemoryPlanner::peak_memory(&a);
        let single = a[0].size;
        assert_eq!(peak, single * n, "all simultaneous → need {n} slots");
    }

    #[test]
    fn test_planner_assignments_correct_tensor_ids() {
        let shapes = [(16, 16), (32, 32), (64, 64)];
        let lifetimes = [(0, 0), (1, 1), (2, 2)];
        let a = MemoryPlanner::plan_allocations(&shapes, &lifetimes);
        for (i, assignment) in a.iter().enumerate() {
            assert_eq!(assignment.tensor_id, i);
        }
    }
}
