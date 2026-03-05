#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_contains)]
#![allow(clippy::needless_return)]
#![allow(clippy::manual_div_ceil)]
//! CUDA memory pool with buddy allocation and block coalescing.
//!
//! Provides pre-allocated device memory management to reduce
//! `cudaMalloc`/`cudaFree` overhead during inference. Supports
//! best-fit, first-fit, and buddy-system allocation strategies.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Allocation strategy used by [`CudaMemoryPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Pick the smallest free block that satisfies the request.
    BestFit,
    /// Pick the first free block that satisfies the request.
    FirstFit,
    /// Power-of-two buddy system with automatic splitting/merging.
    BuddySystem,
}

/// A single memory block inside a pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryBlock {
    /// Byte offset from the start of the pool.
    pub offset: usize,
    /// Size of the block in bytes.
    pub size: usize,
    /// Required alignment (always a power of two).
    pub alignment: usize,
    /// Whether the block is currently allocated.
    pub in_use: bool,
    /// CUDA device ordinal.
    pub device_id: u32,
}

/// Run-time statistics for a [`CudaMemoryPool`].
#[derive(Debug, Clone, PartialEq)]
pub struct PoolStats {
    /// Total pool capacity in bytes.
    pub total_bytes: usize,
    /// Bytes currently allocated.
    pub allocated_bytes: usize,
    /// Bytes currently free.
    pub free_bytes: usize,
    /// Fragmentation ratio in `[0.0, 1.0]`.
    ///
    /// Defined as `1 − (largest_free_block / total_free)`.
    /// A ratio of 0.0 means all free memory is contiguous.
    pub fragmentation_ratio: f64,
    /// Number of outstanding allocations.
    pub allocation_count: usize,
    /// Number of free blocks.
    pub free_block_count: usize,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by pool operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PoolError {
    #[error("out of memory: requested {requested} bytes, largest free block is {available} bytes")]
    OutOfMemory { requested: usize, available: usize },
    #[error("invalid block handle (offset {offset})")]
    InvalidHandle { offset: usize },
    #[error("double free of block at offset {offset}")]
    DoubleFree { offset: usize },
    #[error("zero-size allocation is not allowed")]
    ZeroSize,
    #[error("alignment {0} is not a power of two")]
    BadAlignment(usize),
    #[error("pool capacity must be > 0")]
    ZeroCapacity,
}

// ---------------------------------------------------------------------------
// Pool implementation
// ---------------------------------------------------------------------------

/// Pre-allocated CUDA memory pool.
///
/// Memory is tracked as a list of contiguous [`MemoryBlock`]s. Allocation
/// picks a free block according to the chosen [`AllocationStrategy`], and
/// deallocation coalesces adjacent free blocks to reduce fragmentation.
pub struct CudaMemoryPool {
    capacity: usize,
    device_id: u32,
    strategy: AllocationStrategy,
    /// Blocks keyed by offset for O(log n) lookup.
    blocks: BTreeMap<usize, MemoryBlock>,
    allocated_bytes: usize,
    allocation_count: usize,
}

impl CudaMemoryPool {
    /// Create a new pool of `capacity` bytes on `device_id`.
    pub fn new(
        capacity: usize,
        device_id: u32,
        strategy: AllocationStrategy,
    ) -> Result<Self, PoolError> {
        if capacity == 0 {
            return Err(PoolError::ZeroCapacity);
        }

        let effective_capacity = if strategy == AllocationStrategy::BuddySystem {
            capacity.next_power_of_two()
        } else {
            capacity
        };

        let mut blocks = BTreeMap::new();
        blocks.insert(
            0,
            MemoryBlock {
                offset: 0,
                size: effective_capacity,
                alignment: 1,
                in_use: false,
                device_id,
            },
        );

        Ok(Self {
            capacity: effective_capacity,
            device_id,
            strategy,
            blocks,
            allocated_bytes: 0,
            allocation_count: 0,
        })
    }

    // -- public API ---------------------------------------------------------

    /// Allocate `size` bytes with default alignment (256 bytes, typical for CUDA).
    pub fn allocate(&mut self, size: usize) -> Result<MemoryBlock, PoolError> {
        self.allocate_aligned(size, 256)
    }

    /// Allocate `size` bytes with a specific `alignment`.
    pub fn allocate_aligned(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<MemoryBlock, PoolError> {
        if size == 0 {
            return Err(PoolError::ZeroSize);
        }
        if !alignment.is_power_of_two() {
            return Err(PoolError::BadAlignment(alignment));
        }

        match self.strategy {
            AllocationStrategy::BestFit => self.alloc_best_fit(size, alignment),
            AllocationStrategy::FirstFit => self.alloc_first_fit(size, alignment),
            AllocationStrategy::BuddySystem => self.alloc_buddy(size, alignment),
        }
    }

    /// Free a previously allocated block identified by its `offset`.
    pub fn deallocate(&mut self, offset: usize) -> Result<(), PoolError> {
        let block = self.blocks.get_mut(&offset).ok_or(PoolError::InvalidHandle { offset })?;

        if !block.in_use {
            return Err(PoolError::DoubleFree { offset });
        }

        block.in_use = false;
        let freed = block.size;
        self.allocated_bytes -= freed;
        self.allocation_count -= 1;

        if self.strategy == AllocationStrategy::BuddySystem {
            self.buddy_merge(offset);
        } else {
            self.coalesce(offset);
        }

        Ok(())
    }

    /// Return current pool statistics.
    pub fn stats(&self) -> PoolStats {
        let free_bytes = self.capacity - self.allocated_bytes;
        let (largest_free, free_count) = self.free_block_summary();

        let fragmentation_ratio =
            if free_bytes == 0 { 0.0 } else { 1.0 - (largest_free as f64 / free_bytes as f64) };

        PoolStats {
            total_bytes: self.capacity,
            allocated_bytes: self.allocated_bytes,
            free_bytes,
            fragmentation_ratio,
            allocation_count: self.allocation_count,
            free_block_count: free_count,
        }
    }

    /// Pool capacity in bytes (may be rounded up for buddy system).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The device this pool manages.
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    /// Active allocation strategy.
    pub fn strategy(&self) -> AllocationStrategy {
        self.strategy
    }

    /// Reset the pool, freeing all allocations.
    pub fn reset(&mut self) {
        self.blocks.clear();
        self.blocks.insert(
            0,
            MemoryBlock {
                offset: 0,
                size: self.capacity,
                alignment: 1,
                in_use: false,
                device_id: self.device_id,
            },
        );
        self.allocated_bytes = 0;
        self.allocation_count = 0;
    }

    // -- first-fit / best-fit -----------------------------------------------

    fn alloc_first_fit(&mut self, size: usize, alignment: usize) -> Result<MemoryBlock, PoolError> {
        let candidate = self
            .blocks
            .values()
            .find(|b| !b.in_use && Self::aligned_fit(b, size, alignment).is_some())
            .map(|b| b.offset);

        match candidate {
            Some(off) => self.split_and_alloc(off, size, alignment),
            None => Err(self.oom_error(size)),
        }
    }

    fn alloc_best_fit(&mut self, size: usize, alignment: usize) -> Result<MemoryBlock, PoolError> {
        let candidate = self
            .blocks
            .values()
            .filter(|b| !b.in_use && Self::aligned_fit(b, size, alignment).is_some())
            .min_by_key(|b| b.size)
            .map(|b| b.offset);

        match candidate {
            Some(off) => self.split_and_alloc(off, size, alignment),
            None => Err(self.oom_error(size)),
        }
    }

    // -- buddy system -------------------------------------------------------

    fn alloc_buddy(&mut self, size: usize, alignment: usize) -> Result<MemoryBlock, PoolError> {
        let required = size.max(alignment).next_power_of_two();

        // Find smallest power-of-two free block >= required.
        let candidate = self
            .blocks
            .values()
            .filter(|b| !b.in_use && b.size >= required)
            .min_by_key(|b| b.size)
            .map(|b| b.offset);

        let offset = match candidate {
            Some(off) => off,
            None => return Err(self.oom_error(size)),
        };

        // Repeatedly split until block is the right size.
        while self.blocks[&offset].size > required {
            let block = self.blocks[&offset].clone();
            let half = block.size / 2;

            // Shrink current block.
            self.blocks.get_mut(&offset).unwrap().size = half;

            // Insert buddy.
            self.blocks.insert(
                offset + half,
                MemoryBlock {
                    offset: offset + half,
                    size: half,
                    alignment: 1,
                    in_use: false,
                    device_id: self.device_id,
                },
            );
        }

        let block = self.blocks.get_mut(&offset).unwrap();
        block.in_use = true;
        block.alignment = alignment;
        self.allocated_bytes += block.size;
        self.allocation_count += 1;

        Ok(block.clone())
    }

    /// Merge buddy pairs bottom-up after a free.
    fn buddy_merge(&mut self, offset: usize) {
        let mut current = offset;

        loop {
            let block_size = match self.blocks.get(&current) {
                Some(b) if !b.in_use => b.size,
                _ => break,
            };

            if block_size >= self.capacity {
                break;
            }

            let buddy_offset = current ^ block_size;

            let buddy_free =
                self.blocks.get(&buddy_offset).is_some_and(|b| !b.in_use && b.size == block_size);

            if !buddy_free {
                break;
            }

            // Merge: keep the lower offset.
            let lower = current.min(buddy_offset);
            let upper = current.max(buddy_offset);

            self.blocks.remove(&upper);
            self.blocks.get_mut(&lower).unwrap().size = block_size * 2;

            current = lower;
        }
    }

    // -- shared helpers -----------------------------------------------------

    /// Check if `block` can satisfy `size` with `alignment` and return the
    /// aligned offset if so.
    fn aligned_fit(block: &MemoryBlock, size: usize, alignment: usize) -> Option<usize> {
        let aligned_start = (block.offset + alignment - 1) & !(alignment - 1);
        let end = aligned_start.checked_add(size)?;
        if end <= block.offset + block.size { Some(aligned_start) } else { None }
    }

    /// Split a free block, carving out `size` bytes at `alignment` and leaving
    /// a remainder block (if any).
    fn split_and_alloc(
        &mut self,
        offset: usize,
        size: usize,
        alignment: usize,
    ) -> Result<MemoryBlock, PoolError> {
        let block = self.blocks.remove(&offset).unwrap();
        let aligned_start = (block.offset + alignment - 1) & !(alignment - 1);

        // Padding block before the allocation (alignment waste).
        if aligned_start > block.offset {
            self.blocks.insert(
                block.offset,
                MemoryBlock {
                    offset: block.offset,
                    size: aligned_start - block.offset,
                    alignment: 1,
                    in_use: false,
                    device_id: self.device_id,
                },
            );
        }

        // Remainder after the allocation.
        let alloc_end = aligned_start + size;
        let block_end = block.offset + block.size;
        if alloc_end < block_end {
            self.blocks.insert(
                alloc_end,
                MemoryBlock {
                    offset: alloc_end,
                    size: block_end - alloc_end,
                    alignment: 1,
                    in_use: false,
                    device_id: self.device_id,
                },
            );
        }

        let alloc_block = MemoryBlock {
            offset: aligned_start,
            size,
            alignment,
            in_use: true,
            device_id: self.device_id,
        };
        self.blocks.insert(aligned_start, alloc_block.clone());

        self.allocated_bytes += size;
        self.allocation_count += 1;

        Ok(alloc_block)
    }

    /// Coalesce the block at `offset` with adjacent free neighbours.
    fn coalesce(&mut self, offset: usize) {
        // Merge with the *next* free block.
        let block = self.blocks[&offset].clone();
        let next_offset = block.offset + block.size;
        if let Some(next) = self.blocks.get(&next_offset) {
            if !next.in_use {
                let merged_size = block.size + next.size;
                self.blocks.remove(&next_offset);
                self.blocks.get_mut(&offset).unwrap().size = merged_size;
            }
        }

        // Merge with the *previous* free block.
        let prev = self.blocks.range(..offset).next_back().map(|(&o, b)| (o, b.clone()));
        if let Some((prev_off, prev_block)) = prev {
            if !prev_block.in_use && prev_off + prev_block.size == offset {
                let cur = &self.blocks[&offset];
                let merged_size = prev_block.size + cur.size;
                self.blocks.remove(&offset);
                self.blocks.get_mut(&prev_off).unwrap().size = merged_size;
            }
        }
    }

    fn free_block_summary(&self) -> (usize, usize) {
        let mut largest = 0usize;
        let mut count = 0usize;
        for b in self.blocks.values() {
            if !b.in_use {
                largest = largest.max(b.size);
                count += 1;
            }
        }
        (largest, count)
    }

    fn oom_error(&self, requested: usize) -> PoolError {
        let (largest, _) = self.free_block_summary();
        PoolError::OutOfMemory { requested, available: largest }
    }
}

impl std::fmt::Debug for CudaMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaMemoryPool")
            .field("capacity", &self.capacity)
            .field("device_id", &self.device_id)
            .field("strategy", &self.strategy)
            .field("allocated_bytes", &self.allocated_bytes)
            .field("allocation_count", &self.allocation_count)
            .field("block_count", &self.blocks.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ------------------------------------------------------------

    fn pool(cap: usize, strategy: AllocationStrategy) -> CudaMemoryPool {
        CudaMemoryPool::new(cap, 0, strategy).unwrap()
    }

    // == basic allocation ===================================================

    #[test]
    fn test_single_allocation_best_fit() {
        let mut p = pool(4096, AllocationStrategy::BestFit);
        let b = p.allocate_aligned(512, 1).unwrap();
        assert!(b.in_use);
        assert_eq!(b.size, 512);
        assert_eq!(b.device_id, 0);
    }

    #[test]
    fn test_single_allocation_first_fit() {
        let mut p = pool(4096, AllocationStrategy::FirstFit);
        let b = p.allocate_aligned(256, 1).unwrap();
        assert!(b.in_use);
        assert_eq!(b.size, 256);
    }

    #[test]
    fn test_single_allocation_buddy() {
        let mut p = pool(4096, AllocationStrategy::BuddySystem);
        let b = p.allocate_aligned(100, 1).unwrap();
        assert!(b.in_use);
        // Buddy rounds up to next power of two.
        assert!(b.size.is_power_of_two());
        assert!(b.size >= 100);
    }

    // == multiple allocations ===============================================

    #[test]
    fn test_multiple_allocations_track_stats() {
        let mut p = pool(4096, AllocationStrategy::BestFit);
        p.allocate_aligned(1024, 1).unwrap();
        p.allocate_aligned(512, 1).unwrap();
        let s = p.stats();
        assert_eq!(s.allocated_bytes, 1536);
        assert_eq!(s.allocation_count, 2);
        assert_eq!(s.free_bytes, 4096 - 1536);
    }

    #[test]
    fn test_fill_entire_pool() {
        let mut p = pool(1024, AllocationStrategy::FirstFit);
        p.allocate_aligned(1024, 1).unwrap();
        let s = p.stats();
        assert_eq!(s.allocated_bytes, 1024);
        assert_eq!(s.free_bytes, 0);
    }

    // == deallocation and coalescing ========================================

    #[test]
    fn test_deallocate_single() {
        let mut p = pool(4096, AllocationStrategy::BestFit);
        let b = p.allocate_aligned(512, 1).unwrap();
        p.deallocate(b.offset).unwrap();
        let s = p.stats();
        assert_eq!(s.allocated_bytes, 0);
        assert_eq!(s.allocation_count, 0);
    }

    #[test]
    fn test_coalesce_adjacent_blocks() {
        let mut p = pool(4096, AllocationStrategy::FirstFit);
        let a = p.allocate_aligned(1024, 1).unwrap();
        let b = p.allocate_aligned(1024, 1).unwrap();
        p.deallocate(a.offset).unwrap();
        p.deallocate(b.offset).unwrap();

        let s = p.stats();
        // Both freed blocks should merge into one.
        assert_eq!(s.free_block_count, 1);
        assert_eq!(s.free_bytes, 4096);
        assert!((s.fragmentation_ratio).abs() < f64::EPSILON);
    }

    #[test]
    fn test_coalesce_with_previous_block() {
        let mut p = pool(4096, AllocationStrategy::FirstFit);
        let a = p.allocate_aligned(1024, 1).unwrap();
        let b = p.allocate_aligned(1024, 1).unwrap();
        let _c = p.allocate_aligned(1024, 1).unwrap();

        // Free b then a — a should coalesce backward into b's freed block.
        p.deallocate(b.offset).unwrap();
        p.deallocate(a.offset).unwrap();

        // Two free chunks: merged a+b, and the tail after c.
        let s = p.stats();
        assert_eq!(s.allocated_bytes, 1024);
    }

    // == fragmentation tracking =============================================

    #[test]
    fn test_fragmentation_zero_when_contiguous() {
        let p = pool(4096, AllocationStrategy::BestFit);
        let s = p.stats();
        assert!((s.fragmentation_ratio).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fragmentation_increases_with_holes() {
        let mut p = pool(4096, AllocationStrategy::FirstFit);
        let a = p.allocate_aligned(1024, 1).unwrap();
        let _b = p.allocate_aligned(1024, 1).unwrap();
        let c = p.allocate_aligned(1024, 1).unwrap();

        // Free a and c to create two non-adjacent holes.
        p.deallocate(a.offset).unwrap();
        p.deallocate(c.offset).unwrap();

        let s = p.stats();
        assert!(s.fragmentation_ratio > 0.0);
        assert!(s.free_block_count >= 2);
    }

    #[test]
    fn test_fragmentation_after_full_free() {
        let mut p = pool(4096, AllocationStrategy::BestFit);
        let blocks: Vec<_> = (0..4).map(|_| p.allocate_aligned(1024, 1).unwrap()).collect();
        for b in &blocks {
            p.deallocate(b.offset).unwrap();
        }
        let s = p.stats();
        assert!((s.fragmentation_ratio).abs() < f64::EPSILON);
    }

    // == buddy system specifics =============================================

    #[test]
    fn test_buddy_split_produces_power_of_two() {
        let mut p = pool(4096, AllocationStrategy::BuddySystem);
        let b = p.allocate_aligned(300, 1).unwrap();
        assert_eq!(b.size, 512); // next power of two >= 300
    }

    #[test]
    fn test_buddy_merge_after_free() {
        let mut p = pool(4096, AllocationStrategy::BuddySystem);
        let a = p.allocate_aligned(1024, 1).unwrap();
        let b = p.allocate_aligned(1024, 1).unwrap();

        p.deallocate(a.offset).unwrap();
        p.deallocate(b.offset).unwrap();

        let s = p.stats();
        // Both halves freed → merged back into full pool.
        assert_eq!(s.free_block_count, 1);
        assert_eq!(s.free_bytes, 4096);
    }

    #[test]
    fn test_buddy_no_merge_with_occupied_buddy() {
        let mut p = pool(4096, AllocationStrategy::BuddySystem);
        let a = p.allocate_aligned(1024, 1).unwrap();
        let _b = p.allocate_aligned(1024, 1).unwrap();

        p.deallocate(a.offset).unwrap();
        // b is still allocated — no merge should occur.
        let s = p.stats();
        assert!(s.free_block_count >= 1);
        assert_eq!(s.allocated_bytes, 1024);
    }

    #[test]
    fn test_buddy_cascading_merge() {
        let mut p = pool(4096, AllocationStrategy::BuddySystem);
        let a = p.allocate_aligned(512, 1).unwrap();
        let b = p.allocate_aligned(512, 1).unwrap();
        let c = p.allocate_aligned(512, 1).unwrap();
        let d = p.allocate_aligned(512, 1).unwrap();

        // Free all in reverse – should cascade-merge.
        p.deallocate(d.offset).unwrap();
        p.deallocate(c.offset).unwrap();
        p.deallocate(b.offset).unwrap();
        p.deallocate(a.offset).unwrap();

        let s = p.stats();
        assert_eq!(s.free_block_count, 1);
        assert_eq!(s.free_bytes, 4096);
    }

    // == OOM handling =======================================================

    #[test]
    fn test_oom_when_pool_full() {
        let mut p = pool(1024, AllocationStrategy::BestFit);
        p.allocate_aligned(1024, 1).unwrap();
        let err = p.allocate_aligned(1, 1).unwrap_err();
        assert!(matches!(err, PoolError::OutOfMemory { .. }));
    }

    #[test]
    fn test_oom_returns_largest_available() {
        let mut p = pool(4096, AllocationStrategy::FirstFit);
        p.allocate_aligned(3000, 1).unwrap();
        let err = p.allocate_aligned(2000, 1).unwrap_err();
        match err {
            PoolError::OutOfMemory { available, .. } => {
                assert!(available > 0);
                assert!(available < 2000);
            }
            _ => panic!("expected OutOfMemory"),
        }
    }

    #[test]
    fn test_oom_buddy_system() {
        let mut p = pool(1024, AllocationStrategy::BuddySystem);
        p.allocate_aligned(512, 1).unwrap();
        p.allocate_aligned(512, 1).unwrap();
        let err = p.allocate_aligned(1, 1).unwrap_err();
        assert!(matches!(err, PoolError::OutOfMemory { .. }));
    }

    // == error conditions ===================================================

    #[test]
    fn test_zero_size_allocation() {
        let mut p = pool(4096, AllocationStrategy::BestFit);
        assert!(matches!(p.allocate_aligned(0, 1), Err(PoolError::ZeroSize)));
    }

    #[test]
    fn test_bad_alignment() {
        let mut p = pool(4096, AllocationStrategy::BestFit);
        assert!(matches!(p.allocate_aligned(64, 3), Err(PoolError::BadAlignment(3))));
    }

    #[test]
    fn test_double_free() {
        let mut p = pool(4096, AllocationStrategy::BestFit);
        let b = p.allocate_aligned(512, 1).unwrap();
        p.deallocate(b.offset).unwrap();
        assert!(matches!(p.deallocate(b.offset), Err(PoolError::DoubleFree { .. })));
    }

    #[test]
    fn test_invalid_handle() {
        let mut p = pool(4096, AllocationStrategy::BestFit);
        assert!(matches!(p.deallocate(9999), Err(PoolError::InvalidHandle { .. })));
    }

    #[test]
    fn test_zero_capacity() {
        assert!(matches!(
            CudaMemoryPool::new(0, 0, AllocationStrategy::BestFit),
            Err(PoolError::ZeroCapacity)
        ));
    }

    // == multi-device pools =================================================

    #[test]
    fn test_multi_device_independent_pools() {
        let mut p0 = CudaMemoryPool::new(2048, 0, AllocationStrategy::BestFit).unwrap();
        let mut p1 = CudaMemoryPool::new(2048, 1, AllocationStrategy::BestFit).unwrap();

        let b0 = p0.allocate_aligned(1024, 1).unwrap();
        let b1 = p1.allocate_aligned(512, 1).unwrap();

        assert_eq!(b0.device_id, 0);
        assert_eq!(b1.device_id, 1);
        assert_eq!(p0.stats().allocated_bytes, 1024);
        assert_eq!(p1.stats().allocated_bytes, 512);
    }

    #[test]
    fn test_device_id_propagated_to_blocks() {
        let mut p = CudaMemoryPool::new(4096, 7, AllocationStrategy::FirstFit).unwrap();
        let b = p.allocate_aligned(256, 1).unwrap();
        assert_eq!(b.device_id, 7);
    }

    // == alignment ==========================================================

    #[test]
    fn test_aligned_allocation() {
        let mut p = pool(8192, AllocationStrategy::BestFit);
        // Consume some unaligned space first.
        p.allocate_aligned(100, 1).unwrap();
        let b = p.allocate_aligned(256, 512).unwrap();
        assert_eq!(b.offset % 512, 0, "allocation must be aligned to 512");
    }

    // == reset ==============================================================

    #[test]
    fn test_reset_clears_allocations() {
        let mut p = pool(4096, AllocationStrategy::BestFit);
        p.allocate_aligned(1024, 1).unwrap();
        p.allocate_aligned(1024, 1).unwrap();
        p.reset();
        let s = p.stats();
        assert_eq!(s.allocated_bytes, 0);
        assert_eq!(s.allocation_count, 0);
        assert_eq!(s.free_block_count, 1);
    }

    // == best-fit picks smallest block ======================================

    #[test]
    fn test_best_fit_picks_smallest_free() {
        let mut p = pool(4096, AllocationStrategy::BestFit);
        // Create a layout: [alloc 1024][free 1024][alloc 1024][free 1024]
        let a = p.allocate_aligned(1024, 1).unwrap();
        let b = p.allocate_aligned(1024, 1).unwrap();
        let _c = p.allocate_aligned(1024, 1).unwrap();

        p.deallocate(a.offset).unwrap(); // free first 1024
        p.deallocate(b.offset).unwrap(); // free second 1024 → coalesces to 2048

        // Now we have [free 2048][alloc 1024][free 1024].
        // A 512-byte best-fit should land in the 1024-byte tail block.
        let d = p.allocate_aligned(512, 1).unwrap();
        // It should pick the 1024-byte block (tail) over the 2048-byte block.
        assert!(d.offset >= 3072, "best-fit should prefer the smaller free block");
    }

    // == debug impl =========================================================

    #[test]
    fn test_debug_impl() {
        let p = pool(4096, AllocationStrategy::BuddySystem);
        let dbg = format!("{p:?}");
        assert!(dbg.contains("CudaMemoryPool"));
        assert!(dbg.contains("4096"));
    }

    // == proptest property tests ============================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_strategy() -> impl Strategy<Value = AllocationStrategy> {
            prop_oneof![
                Just(AllocationStrategy::BestFit),
                Just(AllocationStrategy::FirstFit),
                Just(AllocationStrategy::BuddySystem),
            ]
        }

        proptest! {
            #[test]
            fn prop_alloc_dealloc_preserves_capacity(
                strategy in arb_strategy(),
                sizes in proptest::collection::vec(1usize..512, 1..8),
            ) {
                let cap = 8192usize;
                let mut p = CudaMemoryPool::new(cap, 0, strategy).unwrap();
                let mut offsets = Vec::new();

                for &sz in &sizes {
                    if let Ok(b) = p.allocate_aligned(sz, 1) {
                        offsets.push(b.offset);
                    }
                }

                for off in offsets {
                    p.deallocate(off).unwrap();
                }

                let s = p.stats();
                prop_assert_eq!(s.allocated_bytes, 0);
                prop_assert_eq!(s.allocation_count, 0);
            }

            #[test]
            fn prop_allocated_never_exceeds_capacity(
                strategy in arb_strategy(),
                sizes in proptest::collection::vec(1usize..1024, 1..16),
            ) {
                let cap = 4096usize;
                let mut p = CudaMemoryPool::new(cap, 0, strategy).unwrap();

                for &sz in &sizes {
                    let _ = p.allocate_aligned(sz, 1);
                    let s = p.stats();
                    prop_assert!(s.allocated_bytes <= p.capacity());
                }
            }

            #[test]
            fn prop_fragmentation_in_range(
                strategy in arb_strategy(),
                sizes in proptest::collection::vec(1usize..256, 1..8),
            ) {
                let cap = 4096usize;
                let mut p = CudaMemoryPool::new(cap, 0, strategy).unwrap();

                for &sz in &sizes {
                    let _ = p.allocate_aligned(sz, 1);
                }

                let s = p.stats();
                prop_assert!(s.fragmentation_ratio >= 0.0);
                prop_assert!(s.fragmentation_ratio <= 1.0);
            }

            #[test]
            fn prop_stats_consistency(
                strategy in arb_strategy(),
                sizes in proptest::collection::vec(1usize..512, 1..10),
            ) {
                let cap = 8192usize;
                let mut p = CudaMemoryPool::new(cap, 0, strategy).unwrap();

                for &sz in &sizes {
                    let _ = p.allocate_aligned(sz, 1);
                }

                let s = p.stats();
                prop_assert_eq!(s.total_bytes, p.capacity());
                prop_assert_eq!(
                    s.allocated_bytes + s.free_bytes,
                    s.total_bytes,
                    "allocated + free must equal total"
                );
            }

            #[test]
            fn prop_buddy_blocks_are_power_of_two(
                sizes in proptest::collection::vec(1usize..512, 1..8),
            ) {
                let cap = 8192usize;
                let mut p = CudaMemoryPool::new(cap, 0, AllocationStrategy::BuddySystem).unwrap();

                for &sz in &sizes {
                    if let Ok(b) = p.allocate_aligned(sz, 1) {
                        prop_assert!(
                            b.size.is_power_of_two(),
                            "buddy block size {} is not power of two",
                            b.size
                        );
                    }
                }
            }
        }
    }
}
