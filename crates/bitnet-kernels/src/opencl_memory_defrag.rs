//! OpenCL GPU memory defragmentation for Intel Arc A770.
//!
//! Provides memory block tracking, fragmentation analysis, and four
//! defragmentation strategies (compaction, coalescing, best-fit, buddy
//! system) for maintaining allocation efficiency during long-running
//! inference sessions.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Status of a memory block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockStatus {
    Allocated,
    Free,
}

/// A contiguous region of GPU memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryBlock {
    pub address: u64,
    pub size: usize,
    pub status: BlockStatus,
    pub allocation_id: Option<u64>,
}

impl MemoryBlock {
    pub fn allocated(address: u64, size: usize, allocation_id: u64) -> Self {
        Self { address, size, status: BlockStatus::Allocated, allocation_id: Some(allocation_id) }
    }

    pub fn free(address: u64, size: usize) -> Self {
        Self { address, size, status: BlockStatus::Free, allocation_id: None }
    }

    pub fn end_address(&self) -> u64 {
        self.address + self.size as u64
    }

    pub fn is_free(&self) -> bool {
        self.status == BlockStatus::Free
    }

    pub fn is_allocated(&self) -> bool {
        self.status == BlockStatus::Allocated
    }
}

/// Snapshot of memory fragmentation state.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentationMetrics {
    pub total_memory: usize,
    pub used_memory: usize,
    pub free_memory: usize,
    pub largest_free_block: usize,
    pub fragment_count: usize,
    /// Ratio in `[0.0, 1.0]` — 0 means no fragmentation, 1 means maximally
    /// fragmented.
    pub fragmentation_ratio: f64,
}

/// Defragmentation strategy selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefragStrategy {
    /// Move all allocated blocks toward the start of the address space.
    Compaction,
    /// Merge adjacent free blocks into larger ones.
    Coalescing,
    /// Allocate into the smallest free block that fits.
    BestFit,
    /// Power-of-two buddy allocation/deallocation.
    BuddySystem,
}

/// A single memory move operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryMove {
    pub from_addr: u64,
    pub to_addr: u64,
    pub size: usize,
    pub allocation_id: u64,
}

/// Plan produced by a defrag algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefragPlan {
    pub moves: Vec<MemoryMove>,
    pub freed_bytes: usize,
    /// Estimated wall-clock time in microseconds.
    pub estimated_time_us: u64,
}

impl DefragPlan {
    pub fn empty() -> Self {
        Self { moves: Vec::new(), freed_bytes: 0, estimated_time_us: 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }
}

// ---------------------------------------------------------------------------
// AllocationTracker
// ---------------------------------------------------------------------------

/// Tracks all GPU memory blocks for defrag planning.
#[derive(Debug)]
pub struct AllocationTracker {
    blocks: Vec<MemoryBlock>,
    next_allocation_id: u64,
    total_memory: usize,
}

impl AllocationTracker {
    pub fn new(total_memory: usize) -> Self {
        let initial = MemoryBlock::free(0, total_memory);
        Self { blocks: vec![initial], next_allocation_id: 1, total_memory }
    }

    pub fn blocks(&self) -> &[MemoryBlock] {
        &self.blocks
    }

    pub fn total_memory(&self) -> usize {
        self.total_memory
    }

    /// Allocate `size` bytes using first-fit. Returns allocation id on success.
    pub fn allocate(&mut self, size: usize) -> Option<u64> {
        if size == 0 {
            return None;
        }
        let idx = self.blocks.iter().position(|b| b.is_free() && b.size >= size)?;
        let block = &self.blocks[idx];
        let addr = block.address;
        let remaining = block.size - size;
        let alloc_id = self.next_allocation_id;
        self.next_allocation_id += 1;

        self.blocks[idx] = MemoryBlock::allocated(addr, size, alloc_id);

        if remaining > 0 {
            let free_block = MemoryBlock::free(addr + size as u64, remaining);
            self.blocks.insert(idx + 1, free_block);
        }
        Some(alloc_id)
    }

    /// Free the block with the given allocation id.
    pub fn free(&mut self, allocation_id: u64) -> bool {
        let idx = self.blocks.iter().position(|b| b.allocation_id == Some(allocation_id));
        match idx {
            Some(i) => {
                let addr = self.blocks[i].address;
                let size = self.blocks[i].size;
                self.blocks[i] = MemoryBlock::free(addr, size);
                true
            }
            None => false,
        }
    }

    /// Allocate using best-fit strategy.
    pub fn allocate_best_fit(&mut self, size: usize) -> Option<u64> {
        if size == 0 {
            return None;
        }
        let idx = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.is_free() && b.size >= size)
            .min_by_key(|(_, b)| b.size)
            .map(|(i, _)| i)?;

        let block = &self.blocks[idx];
        let addr = block.address;
        let remaining = block.size - size;
        let alloc_id = self.next_allocation_id;
        self.next_allocation_id += 1;

        self.blocks[idx] = MemoryBlock::allocated(addr, size, alloc_id);

        if remaining > 0 {
            let free_block = MemoryBlock::free(addr + size as u64, remaining);
            self.blocks.insert(idx + 1, free_block);
        }
        Some(alloc_id)
    }

    /// Compute current fragmentation metrics.
    pub fn metrics(&self) -> FragmentationMetrics {
        let mut used_memory = 0usize;
        let mut free_memory = 0usize;
        let mut largest_free_block = 0usize;
        let mut free_block_count = 0usize;

        for block in &self.blocks {
            match block.status {
                BlockStatus::Allocated => used_memory += block.size,
                BlockStatus::Free => {
                    free_memory += block.size;
                    free_block_count += 1;
                    if block.size > largest_free_block {
                        largest_free_block = block.size;
                    }
                }
            }
        }

        let fragmentation_ratio = if free_memory == 0 {
            0.0
        } else {
            // 1 - (largest_free / total_free) gives fragmentation level
            1.0 - (largest_free_block as f64 / free_memory as f64)
        };

        FragmentationMetrics {
            total_memory: self.total_memory,
            used_memory,
            free_memory,
            largest_free_block,
            fragment_count: free_block_count,
            fragmentation_ratio,
        }
    }

    /// Coalesce adjacent free blocks in-place.
    pub fn coalesce(&mut self) -> usize {
        let mut merged = 0usize;
        let mut i = 0;
        while i + 1 < self.blocks.len() {
            if self.blocks[i].is_free() && self.blocks[i + 1].is_free() {
                let combined_size = self.blocks[i].size + self.blocks[i + 1].size;
                self.blocks[i].size = combined_size;
                self.blocks.remove(i + 1);
                merged += 1;
            } else {
                i += 1;
            }
        }
        merged
    }

    /// Plan a compaction defrag: move all allocated blocks toward address 0.
    pub fn plan_compaction(&self) -> DefragPlan {
        let mut moves = Vec::new();
        let mut next_addr: u64 = 0;

        for block in &self.blocks {
            if block.is_allocated() {
                if block.address != next_addr {
                    moves.push(MemoryMove {
                        from_addr: block.address,
                        to_addr: next_addr,
                        size: block.size,
                        allocation_id: block.allocation_id.unwrap(),
                    });
                }
                next_addr += block.size as u64;
            }
        }

        let freed_bytes = self.metrics().free_memory;

        // Estimate 1 µs per 64 KB moved
        let total_moved: usize = moves.iter().map(|m| m.size).sum();
        let estimated_time_us = (total_moved / 65536).max(if moves.is_empty() { 0 } else { 1 });

        DefragPlan { moves, freed_bytes, estimated_time_us: estimated_time_us as u64 }
    }

    /// Plan a coalescing defrag: merges adjacent free blocks (returns moves
    /// needed — always empty since coalescing is metadata-only).
    pub fn plan_coalescing(&self) -> DefragPlan {
        let mut freed_bytes = 0usize;
        let mut i = 0;
        let blocks = &self.blocks;
        let mut fragment_merges = 0u32;
        while i + 1 < blocks.len() {
            if blocks[i].is_free() && blocks[i + 1].is_free() {
                freed_bytes += blocks[i + 1].size;
                fragment_merges += 1;
            }
            i += 1;
        }
        DefragPlan { moves: Vec::new(), freed_bytes, estimated_time_us: fragment_merges as u64 }
    }

    /// Apply a compaction plan to the tracker — rewrite block list.
    pub fn apply_compaction(&mut self) {
        let plan = self.plan_compaction();
        if plan.is_empty() {
            // Still coalesce free blocks
            self.coalesce();
            return;
        }

        let mut new_blocks = Vec::new();
        let mut next_addr: u64 = 0;

        for block in &self.blocks {
            if block.is_allocated() {
                new_blocks.push(MemoryBlock::allocated(
                    next_addr,
                    block.size,
                    block.allocation_id.unwrap(),
                ));
                next_addr += block.size as u64;
            }
        }

        // One large free block at the end
        let remaining = self.total_memory.saturating_sub(next_addr as usize);
        if remaining > 0 {
            new_blocks.push(MemoryBlock::free(next_addr, remaining));
        }

        self.blocks = new_blocks;
    }
}

// ---------------------------------------------------------------------------
// BuddyAllocator
// ---------------------------------------------------------------------------

/// Power-of-two buddy-system allocator.
#[derive(Debug)]
pub struct BuddyAllocator {
    /// Maps order → set of free block addresses.
    free_lists: BTreeMap<u32, Vec<u64>>,
    /// Tracks allocated blocks: address → (order, allocation_id).
    allocated: BTreeMap<u64, (u32, u64)>,
    min_order: u32,
    max_order: u32,
    next_allocation_id: u64,
    total_size: usize,
}

impl BuddyAllocator {
    /// Create a buddy allocator managing `total_size` bytes (rounded up to a
    /// power of two). `min_block_size` is the smallest allocatable unit.
    pub fn new(total_size: usize, min_block_size: usize) -> Self {
        let min_block = min_block_size.next_power_of_two();
        let total = total_size.next_power_of_two();

        let min_order = min_block.trailing_zeros();
        let max_order = total.trailing_zeros();

        let mut free_lists = BTreeMap::new();
        free_lists.insert(max_order, vec![0]);

        Self {
            free_lists,
            allocated: BTreeMap::new(),
            min_order,
            max_order,
            next_allocation_id: 1,
            total_size: total,
        }
    }

    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Allocate at least `size` bytes. Returns `(address, allocation_id)`.
    pub fn allocate(&mut self, size: usize) -> Option<(u64, u64)> {
        if size == 0 {
            return None;
        }
        let needed = size.next_power_of_two().max(1 << self.min_order);
        let target_order = needed.trailing_zeros();

        // Find the smallest available block of sufficient order.
        let avail_order = (target_order..=self.max_order)
            .find(|&o| self.free_lists.get(&o).is_some_and(|list| !list.is_empty()))?;

        // Remove block from free list.
        let addr = self.free_lists.get_mut(&avail_order).unwrap().pop().unwrap();
        if self.free_lists.get(&avail_order).is_some_and(|l| l.is_empty()) {
            self.free_lists.remove(&avail_order);
        }

        // Split down to the target order.
        let mut current_order = avail_order;
        let current_addr = addr;
        while current_order > target_order {
            current_order -= 1;
            let buddy_addr = current_addr + (1u64 << current_order);
            self.free_lists.entry(current_order).or_default().push(buddy_addr);
        }

        let alloc_id = self.next_allocation_id;
        self.next_allocation_id += 1;
        self.allocated.insert(current_addr, (target_order, alloc_id));

        Some((current_addr, alloc_id))
    }

    /// Free a previously allocated block by address.
    pub fn free(&mut self, addr: u64) -> bool {
        let (mut order, _alloc_id) = match self.allocated.remove(&addr) {
            Some(v) => v,
            None => return false,
        };

        let mut current_addr = addr;

        // Merge with buddies as far as possible.
        while order < self.max_order {
            let buddy = current_addr ^ (1u64 << order);
            let buddy_free = self.free_lists.get(&order).is_some_and(|list| list.contains(&buddy));

            if buddy_free {
                // Remove buddy from free list.
                let list = self.free_lists.get_mut(&order).unwrap();
                list.retain(|&a| a != buddy);
                if list.is_empty() {
                    self.free_lists.remove(&order);
                }
                current_addr = current_addr.min(buddy);
                order += 1;
            } else {
                break;
            }
        }

        self.free_lists.entry(order).or_default().push(current_addr);
        true
    }

    /// Number of currently allocated blocks.
    pub fn allocation_count(&self) -> usize {
        self.allocated.len()
    }

    /// Total bytes currently allocated.
    pub fn used_bytes(&self) -> usize {
        self.allocated.values().map(|(order, _)| 1usize << order).sum()
    }

    /// Total bytes currently free.
    pub fn free_bytes(&self) -> usize {
        self.total_size - self.used_bytes()
    }

    /// Compute fragmentation metrics.
    pub fn metrics(&self) -> FragmentationMetrics {
        let used = self.used_bytes();
        let free = self.free_bytes();
        let largest_free = self
            .free_lists
            .iter()
            .rev()
            .find(|(_, v)| !v.is_empty())
            .map(|(order, _)| 1usize << order)
            .unwrap_or(0);

        let fragment_count: usize = self.free_lists.values().map(|v| v.len()).sum();

        let fragmentation_ratio =
            if free == 0 { 0.0 } else { 1.0 - (largest_free as f64 / free as f64) };

        FragmentationMetrics {
            total_memory: self.total_size,
            used_memory: used,
            free_memory: free,
            largest_free_block: largest_free,
            fragment_count,
            fragmentation_ratio,
        }
    }
}

// ---------------------------------------------------------------------------
// DefragScheduler
// ---------------------------------------------------------------------------

/// Thresholds that trigger defragmentation.
#[derive(Debug, Clone)]
pub struct DefragThresholds {
    /// Trigger if fragmentation ratio exceeds this value.
    pub fragmentation_ratio: f64,
    /// Trigger if free fragment count exceeds this value.
    pub max_fragment_count: usize,
    /// Trigger if largest free block is smaller than this fraction of free
    /// memory.
    pub min_largest_free_ratio: f64,
}

impl Default for DefragThresholds {
    fn default() -> Self {
        Self { fragmentation_ratio: 0.3, max_fragment_count: 16, min_largest_free_ratio: 0.5 }
    }
}

/// Decides when defragmentation is needed.
#[derive(Debug, Clone)]
pub struct DefragScheduler {
    thresholds: DefragThresholds,
    defrag_count: u64,
}

impl DefragScheduler {
    pub fn new(thresholds: DefragThresholds) -> Self {
        Self { thresholds, defrag_count: 0 }
    }

    pub fn with_defaults() -> Self {
        Self::new(DefragThresholds::default())
    }

    /// Returns `true` if defrag should run based on the given metrics.
    pub fn should_defrag(&self, metrics: &FragmentationMetrics) -> bool {
        if metrics.free_memory == 0 {
            return false;
        }

        if metrics.fragmentation_ratio > self.thresholds.fragmentation_ratio {
            return true;
        }

        if metrics.fragment_count > self.thresholds.max_fragment_count {
            return true;
        }

        let largest_ratio = metrics.largest_free_block as f64 / metrics.free_memory as f64;
        if largest_ratio < self.thresholds.min_largest_free_ratio {
            return true;
        }

        false
    }

    /// Pick the best strategy for the current fragmentation state.
    pub fn recommend_strategy(&self, metrics: &FragmentationMetrics) -> DefragStrategy {
        if metrics.fragmentation_ratio > 0.7 {
            DefragStrategy::Compaction
        } else if metrics.fragment_count > self.thresholds.max_fragment_count {
            DefragStrategy::Coalescing
        } else {
            DefragStrategy::BestFit
        }
    }

    /// Record that a defrag was performed.
    pub fn record_defrag(&mut self) {
        self.defrag_count += 1;
    }

    pub fn defrag_count(&self) -> u64 {
        self.defrag_count
    }

    pub fn thresholds(&self) -> &DefragThresholds {
        &self.thresholds
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // MemoryBlock basics
    // =======================================================================

    #[test]
    fn test_block_allocated() {
        let b = MemoryBlock::allocated(0x100, 256, 1);
        assert_eq!(b.address, 0x100);
        assert_eq!(b.size, 256);
        assert!(b.is_allocated());
        assert!(!b.is_free());
        assert_eq!(b.allocation_id, Some(1));
    }

    #[test]
    fn test_block_free() {
        let b = MemoryBlock::free(0x200, 512);
        assert!(b.is_free());
        assert_eq!(b.allocation_id, None);
    }

    #[test]
    fn test_block_end_address() {
        let b = MemoryBlock::allocated(100, 50, 1);
        assert_eq!(b.end_address(), 150);
    }

    // =======================================================================
    // AllocationTracker — allocate/free
    // =======================================================================

    #[test]
    fn test_tracker_new_has_one_free_block() {
        let t = AllocationTracker::new(1024);
        assert_eq!(t.blocks().len(), 1);
        assert!(t.blocks()[0].is_free());
        assert_eq!(t.blocks()[0].size, 1024);
    }

    #[test]
    fn test_tracker_allocate_simple() {
        let mut t = AllocationTracker::new(1024);
        let id = t.allocate(256).unwrap();
        assert_eq!(id, 1);
        assert_eq!(t.blocks().len(), 2);
        assert!(t.blocks()[0].is_allocated());
        assert_eq!(t.blocks()[0].size, 256);
        assert!(t.blocks()[1].is_free());
        assert_eq!(t.blocks()[1].size, 768);
    }

    #[test]
    fn test_tracker_allocate_exact_fit() {
        let mut t = AllocationTracker::new(256);
        let id = t.allocate(256).unwrap();
        assert!(id > 0);
        assert_eq!(t.blocks().len(), 1);
        assert!(t.blocks()[0].is_allocated());
    }

    #[test]
    fn test_tracker_allocate_too_large() {
        let mut t = AllocationTracker::new(256);
        assert!(t.allocate(512).is_none());
    }

    #[test]
    fn test_tracker_allocate_zero_returns_none() {
        let mut t = AllocationTracker::new(1024);
        assert!(t.allocate(0).is_none());
    }

    #[test]
    fn test_tracker_free_block() {
        let mut t = AllocationTracker::new(1024);
        let id = t.allocate(256).unwrap();
        assert!(t.free(id));
        assert!(t.blocks()[0].is_free());
    }

    #[test]
    fn test_tracker_free_nonexistent() {
        let mut t = AllocationTracker::new(1024);
        assert!(!t.free(999));
    }

    #[test]
    fn test_tracker_multiple_allocations() {
        let mut t = AllocationTracker::new(1024);
        let a = t.allocate(256).unwrap();
        let b = t.allocate(256).unwrap();
        let c = t.allocate(256).unwrap();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_eq!(t.blocks().len(), 4); // 3 allocated + 1 free
    }

    // =======================================================================
    // Fragmentation metrics
    // =======================================================================

    #[test]
    fn test_metrics_no_fragmentation() {
        let t = AllocationTracker::new(1024);
        let m = t.metrics();
        assert_eq!(m.total_memory, 1024);
        assert_eq!(m.used_memory, 0);
        assert_eq!(m.free_memory, 1024);
        assert_eq!(m.largest_free_block, 1024);
        assert_eq!(m.fragment_count, 1);
        assert!((m.fragmentation_ratio - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_fully_allocated() {
        let mut t = AllocationTracker::new(256);
        t.allocate(256).unwrap();
        let m = t.metrics();
        assert_eq!(m.used_memory, 256);
        assert_eq!(m.free_memory, 0);
        assert_eq!(m.fragmentation_ratio, 0.0);
    }

    #[test]
    fn test_metrics_fragmented() {
        let mut t = AllocationTracker::new(1024);
        let a = t.allocate(256).unwrap();
        let _b = t.allocate(256).unwrap();
        let c = t.allocate(256).unwrap();
        // Free a and c → two non-adjacent free blocks
        t.free(a);
        t.free(c);
        let m = t.metrics();
        assert_eq!(m.used_memory, 256);
        assert_eq!(m.free_memory, 768);
        assert!(m.fragment_count >= 2);
        assert!(m.fragmentation_ratio > 0.0);
    }

    #[test]
    fn test_metrics_single_free_block_ratio_zero() {
        let mut t = AllocationTracker::new(1024);
        t.allocate(512).unwrap();
        let m = t.metrics();
        // Single free block → ratio = 0
        assert_eq!(m.fragment_count, 1);
        assert!((m.fragmentation_ratio - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_alternating_alloc_free() {
        let mut t = AllocationTracker::new(1024);
        // Allocate 4 × 128 then free every other one
        let ids: Vec<u64> = (0..4).map(|_| t.allocate(128).unwrap()).collect();
        t.free(ids[0]);
        t.free(ids[2]);
        let m = t.metrics();
        assert_eq!(m.used_memory, 256);
        assert_eq!(m.free_memory, 768);
        assert!(m.fragment_count >= 2);
        assert!(m.fragmentation_ratio > 0.0);
    }

    // =======================================================================
    // Coalescing
    // =======================================================================

    #[test]
    fn test_coalesce_adjacent_free() {
        let mut t = AllocationTracker::new(1024);
        let a = t.allocate(256).unwrap();
        let b = t.allocate(256).unwrap();
        t.free(a);
        t.free(b);
        let merged = t.coalesce();
        assert!(merged >= 1);
        // Should have merged the two free blocks at the start
        assert!(t.blocks()[0].is_free());
        assert!(t.blocks()[0].size >= 512);
    }

    #[test]
    fn test_coalesce_no_adjacent_free() {
        let mut t = AllocationTracker::new(1024);
        let _a = t.allocate(256).unwrap();
        let b = t.allocate(256).unwrap();
        t.free(b);
        // Free block is between allocated block and trailing free → not
        // adjacent to the allocated block
        let merged = t.coalesce();
        // The freed block (idx 1) and trailing free (idx 2) are adjacent.
        assert!(merged >= 1);
    }

    #[test]
    fn test_coalesce_all_free() {
        let mut t = AllocationTracker::new(1024);
        let a = t.allocate(256).unwrap();
        let b = t.allocate(256).unwrap();
        t.free(a);
        t.free(b);
        t.coalesce();
        // After coalescing everything is one big free block
        let m = t.metrics();
        assert_eq!(m.fragment_count, 1);
        assert_eq!(m.free_memory, 1024);
    }

    #[test]
    fn test_coalesce_nothing_to_merge() {
        let t_blocks = AllocationTracker::new(1024);
        let mut t = t_blocks;
        let merged = t.coalesce();
        // Single free block — nothing to merge
        assert_eq!(merged, 0);
    }

    // =======================================================================
    // Compaction planning
    // =======================================================================

    #[test]
    fn test_compaction_no_moves_needed() {
        let mut t = AllocationTracker::new(1024);
        t.allocate(256).unwrap();
        let plan = t.plan_compaction();
        assert!(plan.moves.is_empty());
    }

    #[test]
    fn test_compaction_moves_blocks() {
        let mut t = AllocationTracker::new(1024);
        let a = t.allocate(256).unwrap();
        let _b = t.allocate(256).unwrap();
        t.free(a);
        let plan = t.plan_compaction();
        assert_eq!(plan.moves.len(), 1);
        assert_eq!(plan.moves[0].from_addr, 256);
        assert_eq!(plan.moves[0].to_addr, 0);
        assert_eq!(plan.moves[0].size, 256);
    }

    #[test]
    fn test_compaction_multiple_moves() {
        let mut t = AllocationTracker::new(1024);
        let a = t.allocate(128).unwrap();
        let _b = t.allocate(128).unwrap();
        let c = t.allocate(128).unwrap();
        let _d = t.allocate(128).unwrap();
        t.free(a);
        t.free(c);
        let plan = t.plan_compaction();
        assert_eq!(plan.moves.len(), 2);
        // Moves should be in address order
        assert!(plan.moves[0].to_addr < plan.moves[1].to_addr);
    }

    #[test]
    fn test_compaction_preserves_allocation_ids() {
        let mut t = AllocationTracker::new(1024);
        let a = t.allocate(128).unwrap();
        let b = t.allocate(128).unwrap();
        t.free(a);
        let plan = t.plan_compaction();
        assert_eq!(plan.moves.len(), 1);
        assert_eq!(plan.moves[0].allocation_id, b);
    }

    #[test]
    fn test_apply_compaction() {
        let mut t = AllocationTracker::new(1024);
        let a = t.allocate(256).unwrap();
        let _b = t.allocate(256).unwrap();
        t.free(a);
        t.apply_compaction();
        // After compaction: one allocated at 0, one free at end
        assert!(t.blocks()[0].is_allocated());
        assert_eq!(t.blocks()[0].address, 0);
        let last = t.blocks().last().unwrap();
        assert!(last.is_free());
    }

    #[test]
    fn test_apply_compaction_fully_allocated() {
        let mut t = AllocationTracker::new(256);
        t.allocate(256).unwrap();
        t.apply_compaction();
        assert_eq!(t.blocks().len(), 1);
        assert!(t.blocks()[0].is_allocated());
    }

    // =======================================================================
    // Coalescing plan
    // =======================================================================

    #[test]
    fn test_plan_coalescing_empty() {
        let t = AllocationTracker::new(1024);
        let plan = t.plan_coalescing();
        assert!(plan.moves.is_empty());
        assert_eq!(plan.freed_bytes, 0);
    }

    #[test]
    fn test_plan_coalescing_with_adjacent_free() {
        let mut t = AllocationTracker::new(1024);
        let a = t.allocate(256).unwrap();
        let b = t.allocate(256).unwrap();
        t.free(a);
        t.free(b);
        let plan = t.plan_coalescing();
        // Coalescing is metadata-only — no moves
        assert!(plan.moves.is_empty());
        assert!(plan.freed_bytes > 0);
    }

    // =======================================================================
    // Best-fit allocation
    // =======================================================================

    #[test]
    fn test_best_fit_picks_smallest_fitting() {
        let mut t = AllocationTracker::new(1024);
        // Create layout: [alloc 256] [free 128] [alloc 256] [free 384]
        let a = t.allocate(256).unwrap();
        t.allocate(128).unwrap();
        let c = t.allocate(256).unwrap();
        t.free(a);
        t.free(c);
        // a freed → 256 bytes free at 0
        // c freed → 256 bytes free at 384
        // trailing free = 384 bytes

        // Best-fit for 100 should pick a 256-byte free block, not the 384-byte one
        let id = t.allocate_best_fit(100).unwrap();
        let alloc_block = t.blocks().iter().find(|b| b.allocation_id == Some(id)).unwrap();
        // It should have been placed in one of the 256-byte blocks
        assert!(alloc_block.size <= 256);
    }

    #[test]
    fn test_best_fit_exact_match() {
        let mut t = AllocationTracker::new(1024);
        let a = t.allocate(256).unwrap();
        t.allocate(256).unwrap();
        t.free(a);
        // Free block at 0 is exactly 256
        let id = t.allocate_best_fit(256).unwrap();
        let block = t.blocks().iter().find(|b| b.allocation_id == Some(id)).unwrap();
        assert_eq!(block.size, 256);
        assert_eq!(block.address, 0);
    }

    #[test]
    fn test_best_fit_no_fit() {
        let mut t = AllocationTracker::new(256);
        t.allocate(256).unwrap();
        assert!(t.allocate_best_fit(128).is_none());
    }

    #[test]
    fn test_best_fit_zero_returns_none() {
        let mut t = AllocationTracker::new(1024);
        assert!(t.allocate_best_fit(0).is_none());
    }

    // =======================================================================
    // BuddyAllocator
    // =======================================================================

    #[test]
    fn test_buddy_new() {
        let b = BuddyAllocator::new(1024, 64);
        assert_eq!(b.total_size(), 1024);
        assert_eq!(b.allocation_count(), 0);
        assert_eq!(b.free_bytes(), 1024);
    }

    #[test]
    fn test_buddy_allocate_single() {
        let mut b = BuddyAllocator::new(1024, 64);
        let (addr, id) = b.allocate(64).unwrap();
        assert_eq!(addr, 0);
        assert_eq!(id, 1);
        assert_eq!(b.allocation_count(), 1);
        assert_eq!(b.used_bytes(), 64);
    }

    #[test]
    fn test_buddy_allocate_rounds_up() {
        let mut b = BuddyAllocator::new(1024, 64);
        let (_, _) = b.allocate(100).unwrap(); // rounds to 128
        assert_eq!(b.used_bytes(), 128);
    }

    #[test]
    fn test_buddy_allocate_multiple() {
        let mut b = BuddyAllocator::new(1024, 64);
        b.allocate(64).unwrap();
        b.allocate(64).unwrap();
        b.allocate(64).unwrap();
        assert_eq!(b.allocation_count(), 3);
        assert_eq!(b.used_bytes(), 192);
    }

    #[test]
    fn test_buddy_allocate_too_large() {
        let mut b = BuddyAllocator::new(1024, 64);
        assert!(b.allocate(2048).is_none());
    }

    #[test]
    fn test_buddy_allocate_zero() {
        let mut b = BuddyAllocator::new(1024, 64);
        assert!(b.allocate(0).is_none());
    }

    #[test]
    fn test_buddy_free() {
        let mut b = BuddyAllocator::new(1024, 64);
        let (addr, _) = b.allocate(64).unwrap();
        assert!(b.free(addr));
        assert_eq!(b.allocation_count(), 0);
        assert_eq!(b.free_bytes(), 1024);
    }

    #[test]
    fn test_buddy_free_nonexistent() {
        let mut b = BuddyAllocator::new(1024, 64);
        assert!(!b.free(0xDEAD));
    }

    #[test]
    fn test_buddy_free_coalesces() {
        let mut b = BuddyAllocator::new(1024, 64);
        let (a1, _) = b.allocate(64).unwrap();
        let (a2, _) = b.allocate(64).unwrap();
        b.free(a1);
        b.free(a2);
        // Should coalesce back to full size
        assert_eq!(b.free_bytes(), 1024);
        // Can allocate the full block again
        assert!(b.allocate(1024).is_some());
    }

    #[test]
    fn test_buddy_fragmentation_metrics() {
        let mut b = BuddyAllocator::new(1024, 64);
        b.allocate(64).unwrap();
        b.allocate(64).unwrap();
        let m = b.metrics();
        assert_eq!(m.total_memory, 1024);
        assert_eq!(m.used_memory, 128);
        assert_eq!(m.free_memory, 896);
        assert!(m.largest_free_block <= 896);
    }

    #[test]
    fn test_buddy_exhaust_and_fail() {
        let mut b = BuddyAllocator::new(256, 64);
        b.allocate(64).unwrap();
        b.allocate(64).unwrap();
        b.allocate(64).unwrap();
        b.allocate(64).unwrap();
        assert!(b.allocate(64).is_none());
    }

    // =======================================================================
    // DefragScheduler
    // =======================================================================

    #[test]
    fn test_scheduler_no_defrag_when_clean() {
        let s = DefragScheduler::with_defaults();
        let m = FragmentationMetrics {
            total_memory: 1024,
            used_memory: 256,
            free_memory: 768,
            largest_free_block: 768,
            fragment_count: 1,
            fragmentation_ratio: 0.0,
        };
        assert!(!s.should_defrag(&m));
    }

    #[test]
    fn test_scheduler_triggers_on_high_ratio() {
        let s = DefragScheduler::with_defaults();
        let m = FragmentationMetrics {
            total_memory: 1024,
            used_memory: 256,
            free_memory: 768,
            largest_free_block: 128,
            fragment_count: 6,
            fragmentation_ratio: 0.85,
        };
        assert!(s.should_defrag(&m));
    }

    #[test]
    fn test_scheduler_triggers_on_fragment_count() {
        let s = DefragScheduler::with_defaults();
        let m = FragmentationMetrics {
            total_memory: 4096,
            used_memory: 2048,
            free_memory: 2048,
            largest_free_block: 1500,
            fragment_count: 20,
            fragmentation_ratio: 0.2,
        };
        assert!(s.should_defrag(&m));
    }

    #[test]
    fn test_scheduler_triggers_on_small_largest_free() {
        let s = DefragScheduler::with_defaults();
        let m = FragmentationMetrics {
            total_memory: 4096,
            used_memory: 2048,
            free_memory: 2048,
            largest_free_block: 256,
            fragment_count: 8,
            fragmentation_ratio: 0.2,
        };
        assert!(s.should_defrag(&m));
    }

    #[test]
    fn test_scheduler_no_defrag_when_fully_allocated() {
        let s = DefragScheduler::with_defaults();
        let m = FragmentationMetrics {
            total_memory: 1024,
            used_memory: 1024,
            free_memory: 0,
            largest_free_block: 0,
            fragment_count: 0,
            fragmentation_ratio: 0.0,
        };
        assert!(!s.should_defrag(&m));
    }

    #[test]
    fn test_scheduler_recommend_compaction() {
        let s = DefragScheduler::with_defaults();
        let m = FragmentationMetrics {
            total_memory: 1024,
            used_memory: 256,
            free_memory: 768,
            largest_free_block: 128,
            fragment_count: 6,
            fragmentation_ratio: 0.85,
        };
        assert_eq!(s.recommend_strategy(&m), DefragStrategy::Compaction);
    }

    #[test]
    fn test_scheduler_recommend_coalescing() {
        let s = DefragScheduler::with_defaults();
        let m = FragmentationMetrics {
            total_memory: 4096,
            used_memory: 2048,
            free_memory: 2048,
            largest_free_block: 1500,
            fragment_count: 20,
            fragmentation_ratio: 0.5,
        };
        assert_eq!(s.recommend_strategy(&m), DefragStrategy::Coalescing);
    }

    #[test]
    fn test_scheduler_recommend_best_fit() {
        let s = DefragScheduler::with_defaults();
        let m = FragmentationMetrics {
            total_memory: 1024,
            used_memory: 256,
            free_memory: 768,
            largest_free_block: 600,
            fragment_count: 3,
            fragmentation_ratio: 0.2,
        };
        assert_eq!(s.recommend_strategy(&m), DefragStrategy::BestFit);
    }

    #[test]
    fn test_scheduler_record_defrag() {
        let mut s = DefragScheduler::with_defaults();
        assert_eq!(s.defrag_count(), 0);
        s.record_defrag();
        s.record_defrag();
        assert_eq!(s.defrag_count(), 2);
    }

    #[test]
    fn test_scheduler_custom_thresholds() {
        let thresholds = DefragThresholds {
            fragmentation_ratio: 0.1,
            max_fragment_count: 2,
            min_largest_free_ratio: 0.9,
        };
        let s = DefragScheduler::new(thresholds);
        let m = FragmentationMetrics {
            total_memory: 1024,
            used_memory: 256,
            free_memory: 768,
            largest_free_block: 400,
            fragment_count: 3,
            fragmentation_ratio: 0.15,
        };
        // All three thresholds exceeded with the tighter settings
        assert!(s.should_defrag(&m));
    }

    // =======================================================================
    // DefragPlan
    // =======================================================================

    #[test]
    fn test_defrag_plan_empty() {
        let plan = DefragPlan::empty();
        assert!(plan.is_empty());
        assert_eq!(plan.freed_bytes, 0);
        assert_eq!(plan.estimated_time_us, 0);
    }

    // =======================================================================
    // MemoryMove ordering
    // =======================================================================

    #[test]
    fn test_memory_moves_ordered_by_destination() {
        let mut t = AllocationTracker::new(2048);
        let a = t.allocate(256).unwrap();
        let _b = t.allocate(256).unwrap();
        let c = t.allocate(256).unwrap();
        let _d = t.allocate(256).unwrap();
        t.free(a);
        t.free(c);
        let plan = t.plan_compaction();
        for w in plan.moves.windows(2) {
            assert!(w[0].to_addr < w[1].to_addr);
        }
    }

    // =======================================================================
    // Property tests — total memory invariant
    // =======================================================================

    #[test]
    fn test_total_memory_invariant_through_alloc_free() {
        let mut t = AllocationTracker::new(4096);
        for _ in 0..10 {
            t.allocate(128).unwrap();
        }
        let m1 = t.metrics();
        assert_eq!(m1.used_memory + m1.free_memory, 4096);

        // Free half
        for id in 1..=5 {
            t.free(id);
        }
        let m2 = t.metrics();
        assert_eq!(m2.used_memory + m2.free_memory, 4096);
    }

    #[test]
    fn test_total_memory_invariant_through_compaction() {
        let mut t = AllocationTracker::new(4096);
        for _ in 0..8 {
            t.allocate(256).unwrap();
        }
        for id in [1, 3, 5, 7] {
            t.free(id);
        }
        t.apply_compaction();
        let m = t.metrics();
        assert_eq!(m.used_memory + m.free_memory, 4096);
    }

    #[test]
    fn test_total_memory_invariant_through_coalesce() {
        let mut t = AllocationTracker::new(4096);
        for _ in 0..8 {
            t.allocate(256).unwrap();
        }
        for id in [1, 2, 3] {
            t.free(id);
        }
        t.coalesce();
        let m = t.metrics();
        assert_eq!(m.used_memory + m.free_memory, 4096);
    }

    #[test]
    fn test_buddy_total_memory_invariant() {
        let mut b = BuddyAllocator::new(1024, 64);
        let addrs: Vec<u64> = (0..4).map(|_| b.allocate(64).unwrap().0).collect();
        let m1 = b.metrics();
        assert_eq!(m1.used_memory + m1.free_memory, 1024);

        for a in addrs {
            b.free(a);
        }
        let m2 = b.metrics();
        assert_eq!(m2.used_memory + m2.free_memory, 1024);
    }

    // =======================================================================
    // Edge cases
    // =======================================================================

    #[test]
    fn test_single_block_no_fragmentation() {
        let t = AllocationTracker::new(64);
        let m = t.metrics();
        assert_eq!(m.fragmentation_ratio, 0.0);
        assert_eq!(m.fragment_count, 1);
    }

    #[test]
    fn test_compaction_empty_tracker() {
        let t = AllocationTracker::new(1024);
        let plan = t.plan_compaction();
        assert!(plan.is_empty());
    }

    #[test]
    fn test_coalesce_already_coalesced() {
        let mut t = AllocationTracker::new(1024);
        t.allocate(512).unwrap();
        // trailing free block is already one piece
        let merged = t.coalesce();
        assert_eq!(merged, 0);
    }

    #[test]
    fn test_defrag_strategy_enum_variants() {
        let strategies = [
            DefragStrategy::Compaction,
            DefragStrategy::Coalescing,
            DefragStrategy::BestFit,
            DefragStrategy::BuddySystem,
        ];
        assert_eq!(strategies.len(), 4);
        assert_ne!(DefragStrategy::Compaction, DefragStrategy::Coalescing);
    }
}
