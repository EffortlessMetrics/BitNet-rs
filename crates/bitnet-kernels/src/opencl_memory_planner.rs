//! OpenCL memory allocation planner for GPU buffer reuse.
//!
//! Minimizes peak GPU memory usage through tensor lifetime analysis and
//! buffer pool management. Targets Intel Arc A770 with 64-byte cache-line
//! alignment.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Lifetime span of a tensor in the execution schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLifetime {
    pub tensor_id: u64,
    pub size_bytes: usize,
    pub first_use: u64,
    pub last_use: u64,
    pub alignment: usize,
}

/// A contiguous block inside the planned memory region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryBlock {
    pub offset: usize,
    pub size: usize,
    pub alignment: usize,
    pub tensor_id: Option<u64>,
    pub in_use: bool,
}

/// Strategy used when searching for a free block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    BuddySystem,
    PoolBased { pool_sizes: Vec<usize> },
}

/// Result of [`cpu_compute_plan`].
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    pub total_size: usize,
    pub blocks: Vec<MemoryBlock>,
    /// Maps `tensor_id` → offset in the unified buffer.
    pub assignments: HashMap<u64, usize>,
    pub peak_usage: usize,
    pub reuse_savings: usize,
}

/// Fragmentation metrics for a block list.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentationInfo {
    pub total_free: usize,
    pub largest_free_block: usize,
    pub num_free_blocks: usize,
    pub fragmentation_ratio: f32,
}

/// Accumulates planning statistics.
#[derive(Debug, Clone, Default)]
pub struct PlannerStats {
    pub total_planned: u64,
    pub total_allocations: u64,
    pub total_reuses: u64,
    pub peak_reduction_pct: f32,
}

/// Errors that can arise during planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlannerError {
    LifetimeConflict { tensor_a: u64, tensor_b: u64 },
    InsufficientMemory { required: usize, available: usize },
    InvalidLifetime { tensor_id: u64 },
    AlignmentError,
}

impl fmt::Display for PlannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LifetimeConflict { tensor_a, tensor_b } => {
                write!(f, "lifetime conflict between tensors {tensor_a} and {tensor_b}")
            }
            Self::InsufficientMemory { required, available } => {
                write!(f, "insufficient memory: need {required}, have {available}")
            }
            Self::InvalidLifetime { tensor_id } => {
                write!(f, "invalid lifetime for tensor {tensor_id}")
            }
            Self::AlignmentError => write!(f, "alignment error"),
        }
    }
}

impl std::error::Error for PlannerError {}

/// Top-level planner that owns lifetimes and produces a [`MemoryPlan`].
#[derive(Debug)]
pub struct MemoryPlanner {
    pub strategy: AllocationStrategy,
    pub lifetimes: Vec<TensorLifetime>,
    pub plan: Option<MemoryPlan>,
    pub stats: PlannerStats,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Create a fresh planner with the given strategy.
pub fn create_memory_planner(strategy: AllocationStrategy) -> MemoryPlanner {
    MemoryPlanner { strategy, lifetimes: Vec::new(), plan: None, stats: PlannerStats::default() }
}

/// Register a tensor lifetime with the planner.
pub fn cpu_add_lifetime(
    planner: &mut MemoryPlanner,
    tensor_id: u64,
    size: usize,
    first_use: u64,
    last_use: u64,
    alignment: usize,
) {
    planner.lifetimes.push(TensorLifetime {
        tensor_id,
        size_bytes: size,
        first_use,
        last_use,
        alignment,
    });
}

/// Compute an allocation plan using the planner's strategy.
pub fn cpu_compute_plan(planner: &mut MemoryPlanner) -> Result<MemoryPlan, PlannerError> {
    // Validate lifetimes.
    for lt in &planner.lifetimes {
        if lt.first_use > lt.last_use {
            return Err(PlannerError::InvalidLifetime { tensor_id: lt.tensor_id });
        }
        if lt.alignment == 0 || !lt.alignment.is_power_of_two() {
            return Err(PlannerError::AlignmentError);
        }
    }

    // Sort by first_use (schedule order), then by descending size for
    // better packing.
    let mut sorted: Vec<TensorLifetime> = planner.lifetimes.clone();
    sorted.sort_by(|a, b| a.first_use.cmp(&b.first_use).then(b.size_bytes.cmp(&a.size_bytes)));

    let mut blocks: Vec<MemoryBlock> = Vec::new();
    let mut assignments: HashMap<u64, usize> = HashMap::new();
    let mut total_size: usize = 0;
    let mut reuses: u64 = 0;

    for lt in &sorted {
        // Free blocks whose owning tensor has already expired.
        for blk in &mut blocks {
            if let Some(owner) = blk.tensor_id
                && let Some(owner_lt) = sorted.iter().find(|t| t.tensor_id == owner)
                && owner_lt.last_use < lt.first_use
            {
                blk.in_use = false;
                blk.tensor_id = None;
            }
        }

        let aligned_size = cpu_align_up(lt.size_bytes, lt.alignment);

        // Try to reuse an existing block.
        let offset = match &planner.strategy {
            AllocationStrategy::BestFit => cpu_best_fit_allocate(&mut blocks, aligned_size, lt.alignment),
            _ => cpu_first_fit_allocate(&mut blocks, aligned_size, lt.alignment),
        };

        if let Some(off) = offset {
            // Mark the block as reused.
            if let Some(blk) = blocks.iter_mut().find(|b| b.offset == off && !b.in_use) {
                blk.in_use = true;
                blk.tensor_id = Some(lt.tensor_id);
                reuses += 1;
            }
            assignments.insert(lt.tensor_id, off);
        } else {
            // Allocate at end.
            let aligned_offset = cpu_align_up(total_size, lt.alignment);
            blocks.push(MemoryBlock {
                offset: aligned_offset,
                size: aligned_size,
                alignment: lt.alignment,
                tensor_id: Some(lt.tensor_id),
                in_use: true,
            });
            assignments.insert(lt.tensor_id, aligned_offset);
            total_size = aligned_offset + aligned_size;
        }
    }

    let naive_total: usize =
        sorted.iter().map(|lt| cpu_align_up(lt.size_bytes, lt.alignment)).sum();
    let peak_usage = cpu_compute_peak_from_assignments(&sorted, &assignments);
    let reuse_savings = naive_total.saturating_sub(total_size);

    let plan = MemoryPlan { total_size, blocks, assignments, peak_usage, reuse_savings };

    // Update stats.
    planner.stats.total_planned += 1;
    planner.stats.total_allocations += sorted.len() as u64;
    planner.stats.total_reuses += reuses;
    planner.stats.peak_reduction_pct = if naive_total > 0 {
        (reuse_savings as f32 / naive_total as f32) * 100.0
    } else {
        0.0
    };

    planner.plan = Some(plan.clone());
    Ok(plan)
}

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

/// First-fit: return offset of the first free block that is large enough.
pub fn cpu_first_fit_allocate(
    blocks: &mut [MemoryBlock],
    size: usize,
    alignment: usize,
) -> Option<usize> {
    for blk in blocks.iter() {
        if !blk.in_use && blk.size >= size {
            let aligned = cpu_align_up(blk.offset, alignment);
            if aligned + size <= blk.offset + blk.size {
                return Some(aligned);
            }
        }
    }
    None
}

/// Best-fit: return offset of the smallest free block that fits.
pub fn cpu_best_fit_allocate(
    blocks: &mut [MemoryBlock],
    size: usize,
    alignment: usize,
) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None; // (offset, block_size)
    for blk in blocks.iter() {
        if !blk.in_use && blk.size >= size {
            let aligned = cpu_align_up(blk.offset, alignment);
            if aligned + size <= blk.offset + blk.size {
                match best {
                    None => best = Some((aligned, blk.size)),
                    Some((_, best_sz)) if blk.size < best_sz => {
                        best = Some((aligned, blk.size));
                    }
                    _ => {}
                }
            }
        }
    }
    best.map(|(off, _)| off)
}

/// Two lifetimes can share a buffer when their live ranges do not overlap.
pub fn cpu_can_reuse(a: &TensorLifetime, b: &TensorLifetime) -> bool {
    a.last_use < b.first_use || b.last_use < a.first_use
}

/// Return all pairs `(earlier_id, later_id)` that could share a buffer.
pub fn cpu_find_reuse_opportunities(lifetimes: &[TensorLifetime]) -> Vec<(u64, u64)> {
    let mut pairs = Vec::new();
    for (i, a) in lifetimes.iter().enumerate() {
        for b in &lifetimes[i + 1..] {
            if cpu_can_reuse(a, b) {
                let (first, second) =
                    if a.first_use <= b.first_use { (a.tensor_id, b.tensor_id) } else { (b.tensor_id, a.tensor_id) };
                pairs.push((first, second));
            }
        }
    }
    pairs
}

// ---------------------------------------------------------------------------
// Peak / fragmentation analysis
// ---------------------------------------------------------------------------

/// Compute peak memory if every tensor were separately allocated (no reuse).
pub fn cpu_compute_peak_without_reuse(lifetimes: &[TensorLifetime]) -> usize {
    if lifetimes.is_empty() {
        return 0;
    }
    let max_step = lifetimes.iter().map(|l| l.last_use).max().unwrap_or(0);
    let mut peak: usize = 0;
    for step in 0..=max_step {
        let usage: usize = lifetimes
            .iter()
            .filter(|l| l.first_use <= step && step <= l.last_use)
            .map(|l| cpu_align_up(l.size_bytes, l.alignment))
            .sum();
        peak = peak.max(usage);
    }
    peak
}

/// Peak memory from an already-computed plan.
pub fn cpu_compute_peak_with_reuse(plan: &MemoryPlan) -> usize {
    plan.peak_usage
}

/// Fragmentation metrics for the current block list.
pub fn cpu_compute_fragmentation(blocks: &[MemoryBlock]) -> FragmentationInfo {
    let free_blocks: Vec<&MemoryBlock> = blocks.iter().filter(|b| !b.in_use).collect();
    let total_free: usize = free_blocks.iter().map(|b| b.size).sum();
    let largest = free_blocks.iter().map(|b| b.size).max().unwrap_or(0);
    let num_free = free_blocks.len();
    let ratio = if total_free > 0 && num_free > 0 {
        1.0 - (largest as f32 / total_free as f32)
    } else {
        0.0
    };
    FragmentationInfo {
        total_free,
        largest_free_block: largest,
        num_free_blocks: num_free,
        fragmentation_ratio: ratio,
    }
}

/// Compact free blocks and return bytes reclaimed through merging.
pub fn cpu_defragment(blocks: &mut Vec<MemoryBlock>) -> usize {
    let before: usize = blocks.iter().filter(|b| !b.in_use).map(|b| b.size).sum();

    // Keep only in-use blocks, sorted by offset.
    blocks.retain(|b| b.in_use);
    blocks.sort_by_key(|b| b.offset);

    // Re-pack: shift blocks to remove gaps.
    let mut cursor: usize = 0;
    for blk in blocks.iter_mut() {
        let aligned = cpu_align_up(cursor, blk.alignment);
        blk.offset = aligned;
        cursor = aligned + blk.size;
    }

    let after_total: usize = blocks.iter().map(|b| b.offset + b.size).max().unwrap_or(0);
    let original_total = before + blocks.iter().map(|b| b.size).sum::<usize>();
    original_total.saturating_sub(after_total)
}

// ---------------------------------------------------------------------------
// Alignment utility
// ---------------------------------------------------------------------------

/// Round `size` up to the next multiple of `alignment`.
///
/// `alignment` **must** be a power of two.
pub fn cpu_align_up(size: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two(), "alignment must be power of two");
    (size + alignment - 1) & !(alignment - 1)
}

/// Human-readable summary of a [`MemoryPlan`].
pub fn format_memory_plan(plan: &MemoryPlan) -> String {
    let mut out = String::new();
    out.push_str(&format!("MemoryPlan: total_size={}", plan.total_size));
    out.push_str(&format!(", peak_usage={}", plan.peak_usage));
    out.push_str(&format!(", reuse_savings={}", plan.reuse_savings));
    out.push_str(&format!(", blocks={}", plan.blocks.len()));
    out.push_str(&format!(", assignments={}", plan.assignments.len()));
    out
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute peak usage from per-step analysis of assigned offsets.
fn cpu_compute_peak_from_assignments(
    lifetimes: &[TensorLifetime],
    assignments: &HashMap<u64, usize>,
) -> usize {
    if lifetimes.is_empty() {
        return 0;
    }
    let max_step = lifetimes.iter().map(|l| l.last_use).max().unwrap_or(0);
    let mut peak: usize = 0;
    for step in 0..=max_step {
        let mut max_end: usize = 0;
        for lt in lifetimes {
            if lt.first_use <= step
                && step <= lt.last_use
                && let Some(&off) = assignments.get(&lt.tensor_id)
            {
                let end = off + cpu_align_up(lt.size_bytes, lt.alignment);
                max_end = max_end.max(end);
            }
        }
        peak = peak.max(max_end);
    }
    peak
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers -----------------------------------------------------------

    fn lt(id: u64, size: usize, first: u64, last: u64) -> TensorLifetime {
        TensorLifetime { tensor_id: id, size_bytes: size, first_use: first, last_use: last, alignment: 64 }
    }

    fn lt_align(id: u64, size: usize, first: u64, last: u64, align: usize) -> TensorLifetime {
        TensorLifetime { tensor_id: id, size_bytes: size, first_use: first, last_use: last, alignment: align }
    }

    // 1. Create planner: empty
    #[test]
    fn test_create_planner_empty() {
        let p = create_memory_planner(AllocationStrategy::FirstFit);
        assert!(p.lifetimes.is_empty());
        assert!(p.plan.is_none());
        assert_eq!(p.stats.total_planned, 0);
    }

    // 2. Add lifetime: stored correctly
    #[test]
    fn test_add_lifetime_stored() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 1024, 0, 5, 64);
        assert_eq!(p.lifetimes.len(), 1);
        assert_eq!(p.lifetimes[0].tensor_id, 1);
        assert_eq!(p.lifetimes[0].size_bytes, 1024);
    }

    // 3. First fit: simple case
    #[test]
    fn test_first_fit_simple() {
        let mut blocks = vec![MemoryBlock {
            offset: 0,
            size: 256,
            alignment: 64,
            tensor_id: None,
            in_use: false,
        }];
        let off = cpu_first_fit_allocate(&mut blocks, 128, 64);
        assert_eq!(off, Some(0));
    }

    // 4. Best fit: picks smallest sufficient block
    #[test]
    fn test_best_fit_picks_smallest() {
        let mut blocks = vec![
            MemoryBlock { offset: 0, size: 512, alignment: 64, tensor_id: None, in_use: false },
            MemoryBlock { offset: 512, size: 128, alignment: 64, tensor_id: None, in_use: false },
            MemoryBlock { offset: 640, size: 256, alignment: 64, tensor_id: None, in_use: false },
        ];
        let off = cpu_best_fit_allocate(&mut blocks, 128, 64);
        assert_eq!(off, Some(512));
    }

    // 5. Can reuse: non-overlapping returns true
    #[test]
    fn test_can_reuse_non_overlapping() {
        let a = lt(1, 100, 0, 3);
        let b = lt(2, 100, 4, 7);
        assert!(cpu_can_reuse(&a, &b));
    }

    // 6. Can reuse: overlapping returns false
    #[test]
    fn test_can_reuse_overlapping() {
        let a = lt(1, 100, 0, 5);
        let b = lt(2, 100, 3, 7);
        assert!(!cpu_can_reuse(&a, &b));
    }

    // 7. Find reuse: detects compatible pairs
    #[test]
    fn test_find_reuse_pairs() {
        let lifetimes = vec![lt(1, 100, 0, 2), lt(2, 100, 3, 5), lt(3, 100, 1, 4)];
        let pairs = cpu_find_reuse_opportunities(&lifetimes);
        // 1 and 2 can reuse (0..2 vs 3..5), 1&3 overlap, 2&3 overlap
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (1, 2));
    }

    // 8. Compute plan: linear chain
    #[test]
    fn test_compute_plan_linear_chain() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        // A → B → C, sequential tensors
        cpu_add_lifetime(&mut p, 1, 256, 0, 1, 64);
        cpu_add_lifetime(&mut p, 2, 256, 2, 3, 64);
        cpu_add_lifetime(&mut p, 3, 256, 4, 5, 64);
        let plan = cpu_compute_plan(&mut p).unwrap();
        // All three can reuse the same slot — saves 512 bytes.
        assert!(plan.reuse_savings > 0);
        assert!(plan.total_size <= 256);
    }

    // 9. Compute plan: diamond DAG
    #[test]
    fn test_compute_plan_diamond_dag() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        // A(0-0) → B(1-2), C(1-2) → D(3-3)
        cpu_add_lifetime(&mut p, 1, 128, 0, 0, 64);
        cpu_add_lifetime(&mut p, 2, 128, 1, 2, 64);
        cpu_add_lifetime(&mut p, 3, 128, 1, 2, 64);
        cpu_add_lifetime(&mut p, 4, 128, 3, 3, 64);
        let plan = cpu_compute_plan(&mut p).unwrap();
        // B and C are simultaneous → need 2 slots at step 1-2.
        assert!(plan.peak_usage <= 256);
    }

    // 10. Peak without reuse: sum of all simultaneous
    #[test]
    fn test_peak_without_reuse() {
        let lifetimes = vec![lt(1, 100, 0, 2), lt(2, 200, 1, 3)];
        let peak = cpu_compute_peak_without_reuse(&lifetimes);
        // At steps 1–2, both are live: align(100,64) + align(200,64) = 128+256 = 384
        assert_eq!(peak, 384);
    }

    // 11. Peak with reuse: less than without
    #[test]
    fn test_peak_with_reuse_less() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 512, 0, 1, 64);
        cpu_add_lifetime(&mut p, 2, 512, 2, 3, 64);
        let plan = cpu_compute_plan(&mut p).unwrap();
        let naive = cpu_compute_peak_without_reuse(&p.lifetimes);
        assert!(cpu_compute_peak_with_reuse(&plan) <= naive);
    }

    // 12. Fragmentation: zero for compact layout
    #[test]
    fn test_fragmentation_zero_compact() {
        let blocks = vec![MemoryBlock {
            offset: 0,
            size: 256,
            alignment: 64,
            tensor_id: Some(1),
            in_use: true,
        }];
        let info = cpu_compute_fragmentation(&blocks);
        assert_eq!(info.total_free, 0);
        assert_eq!(info.fragmentation_ratio, 0.0);
    }

    // 13. Defragment: reclaims space
    #[test]
    fn test_defragment_reclaims() {
        let mut blocks = vec![
            MemoryBlock { offset: 0, size: 64, alignment: 64, tensor_id: Some(1), in_use: true },
            MemoryBlock { offset: 64, size: 128, alignment: 64, tensor_id: None, in_use: false },
            MemoryBlock { offset: 192, size: 64, alignment: 64, tensor_id: Some(2), in_use: true },
        ];
        let reclaimed = cpu_defragment(&mut blocks);
        assert!(reclaimed > 0);
        // After defrag the two in-use blocks are adjacent.
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[1].offset, 64);
    }

    // 14. Alignment: correct rounding
    #[test]
    fn test_align_up() {
        assert_eq!(cpu_align_up(0, 64), 0);
        assert_eq!(cpu_align_up(1, 64), 64);
        assert_eq!(cpu_align_up(63, 64), 64);
        assert_eq!(cpu_align_up(64, 64), 64);
        assert_eq!(cpu_align_up(65, 64), 128);
    }

    // 15. Edge: single tensor
    #[test]
    fn test_single_tensor() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 1024, 0, 5, 64);
        let plan = cpu_compute_plan(&mut p).unwrap();
        assert_eq!(plan.assignments.len(), 1);
        assert_eq!(plan.total_size, 1024);
    }

    // 16. Edge: all tensors simultaneous (no reuse)
    #[test]
    fn test_all_simultaneous() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 128, 0, 5, 64);
        cpu_add_lifetime(&mut p, 2, 128, 0, 5, 64);
        cpu_add_lifetime(&mut p, 3, 128, 0, 5, 64);
        let plan = cpu_compute_plan(&mut p).unwrap();
        assert_eq!(plan.reuse_savings, 0);
        assert_eq!(plan.total_size, 384);
    }

    // 17. Edge: all tensors sequential (max reuse)
    #[test]
    fn test_all_sequential() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        for i in 0..5 {
            cpu_add_lifetime(&mut p, i, 256, i * 2, i * 2 + 1, 64);
        }
        let plan = cpu_compute_plan(&mut p).unwrap();
        assert_eq!(plan.total_size, 256);
    }

    // 18. Property: plan peak <= naive peak
    #[test]
    fn test_plan_peak_le_naive() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 256, 0, 2, 64);
        cpu_add_lifetime(&mut p, 2, 512, 1, 3, 64);
        cpu_add_lifetime(&mut p, 3, 128, 3, 5, 64);
        let naive = cpu_compute_peak_without_reuse(&p.lifetimes);
        let plan = cpu_compute_plan(&mut p).unwrap();
        assert!(plan.peak_usage <= naive);
    }

    // 19. Property: aligned addresses are multiples of alignment
    #[test]
    fn test_aligned_addresses() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 100, 0, 1, 64);
        cpu_add_lifetime(&mut p, 2, 200, 0, 1, 128);
        let plan = cpu_compute_plan(&mut p).unwrap();
        for lt in &p.lifetimes {
            let off = plan.assignments[&lt.tensor_id];
            assert_eq!(off % lt.alignment, 0, "tensor {} offset {} not aligned to {}", lt.tensor_id, off, lt.alignment);
        }
    }

    // 20. A770 specific: 64-byte alignment for cache lines
    #[test]
    fn test_a770_64byte_alignment() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 50, 0, 1, 64);
        let plan = cpu_compute_plan(&mut p).unwrap();
        let off = plan.assignments[&1];
        assert_eq!(off % 64, 0);
        assert!(plan.total_size >= 64); // rounded up
    }

    // 21. Best fit strategy end-to-end
    #[test]
    fn test_best_fit_end_to_end() {
        let mut p = create_memory_planner(AllocationStrategy::BestFit);
        cpu_add_lifetime(&mut p, 1, 256, 0, 1, 64);
        cpu_add_lifetime(&mut p, 2, 128, 0, 1, 64);
        cpu_add_lifetime(&mut p, 3, 128, 2, 3, 64);
        let plan = cpu_compute_plan(&mut p).unwrap();
        assert!(plan.assignments.contains_key(&3));
    }

    // 22. Multiple add lifetimes accumulate
    #[test]
    fn test_multiple_add_lifetimes() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        for i in 0..10 {
            cpu_add_lifetime(&mut p, i, 64, i, i + 1, 64);
        }
        assert_eq!(p.lifetimes.len(), 10);
    }

    // 23. Empty planner produces empty plan
    #[test]
    fn test_empty_plan() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        let plan = cpu_compute_plan(&mut p).unwrap();
        assert_eq!(plan.total_size, 0);
        assert!(plan.assignments.is_empty());
    }

    // 24. Invalid lifetime (first > last) is rejected
    #[test]
    fn test_invalid_lifetime_rejected() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 100, 5, 2, 64);
        let res = cpu_compute_plan(&mut p);
        assert!(matches!(res, Err(PlannerError::InvalidLifetime { tensor_id: 1 })));
    }

    // 25. Zero alignment is rejected
    #[test]
    fn test_zero_alignment_rejected() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        p.lifetimes.push(TensorLifetime {
            tensor_id: 1,
            size_bytes: 100,
            first_use: 0,
            last_use: 1,
            alignment: 0,
        });
        let res = cpu_compute_plan(&mut p);
        assert!(matches!(res, Err(PlannerError::AlignmentError)));
    }

    // 26. Non-power-of-two alignment is rejected
    #[test]
    fn test_non_pow2_alignment_rejected() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        p.lifetimes.push(lt_align(1, 100, 0, 1, 48));
        let res = cpu_compute_plan(&mut p);
        assert!(matches!(res, Err(PlannerError::AlignmentError)));
    }

    // 27. format_memory_plan produces non-empty output
    #[test]
    fn test_format_memory_plan() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 256, 0, 1, 64);
        let plan = cpu_compute_plan(&mut p).unwrap();
        let s = format_memory_plan(&plan);
        assert!(s.contains("MemoryPlan"));
        assert!(s.contains("total_size="));
    }

    // 28. Stats updated after plan
    #[test]
    fn test_stats_updated() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 128, 0, 1, 64);
        cpu_add_lifetime(&mut p, 2, 128, 2, 3, 64);
        let _plan = cpu_compute_plan(&mut p).unwrap();
        assert_eq!(p.stats.total_planned, 1);
        assert_eq!(p.stats.total_allocations, 2);
    }

    // 29. Reuse count recorded
    #[test]
    fn test_reuse_count() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 128, 0, 1, 64);
        cpu_add_lifetime(&mut p, 2, 128, 2, 3, 64);
        let _plan = cpu_compute_plan(&mut p).unwrap();
        assert!(p.stats.total_reuses > 0);
    }

    // 30. Can reuse: adjacent lifetimes
    #[test]
    fn test_can_reuse_adjacent() {
        let a = lt(1, 100, 0, 3);
        let b = lt(2, 100, 4, 7);
        assert!(cpu_can_reuse(&a, &b));
    }

    // 31. Can reuse: touching boundary (last_use == first_use-1)
    #[test]
    fn test_can_reuse_touching() {
        let a = lt(1, 100, 0, 2);
        let b = lt(2, 100, 3, 5);
        assert!(cpu_can_reuse(&a, &b));
    }

    // 32. Cannot reuse: identical range
    #[test]
    fn test_cannot_reuse_identical_range() {
        let a = lt(1, 100, 0, 5);
        let b = lt(2, 100, 0, 5);
        assert!(!cpu_can_reuse(&a, &b));
    }

    // 33. First fit: no block large enough returns None
    #[test]
    fn test_first_fit_none() {
        let mut blocks = vec![MemoryBlock {
            offset: 0,
            size: 32,
            alignment: 64,
            tensor_id: None,
            in_use: false,
        }];
        assert_eq!(cpu_first_fit_allocate(&mut blocks, 128, 64), None);
    }

    // 34. Best fit: no block large enough returns None
    #[test]
    fn test_best_fit_none() {
        let mut blocks = vec![MemoryBlock {
            offset: 0,
            size: 32,
            alignment: 64,
            tensor_id: None,
            in_use: false,
        }];
        assert_eq!(cpu_best_fit_allocate(&mut blocks, 128, 64), None);
    }

    // 35. Fragmentation with mixed blocks
    #[test]
    fn test_fragmentation_mixed() {
        let blocks = vec![
            MemoryBlock { offset: 0, size: 64, alignment: 64, tensor_id: Some(1), in_use: true },
            MemoryBlock { offset: 64, size: 128, alignment: 64, tensor_id: None, in_use: false },
            MemoryBlock { offset: 192, size: 64, alignment: 64, tensor_id: Some(2), in_use: true },
            MemoryBlock { offset: 256, size: 64, alignment: 64, tensor_id: None, in_use: false },
        ];
        let info = cpu_compute_fragmentation(&blocks);
        assert_eq!(info.num_free_blocks, 2);
        assert_eq!(info.total_free, 192);
        assert_eq!(info.largest_free_block, 128);
        assert!(info.fragmentation_ratio > 0.0);
    }

    // 36. Large tensor alignment (256 bytes)
    #[test]
    fn test_large_alignment() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 100, 0, 1, 256);
        let plan = cpu_compute_plan(&mut p).unwrap();
        let off = plan.assignments[&1];
        assert_eq!(off % 256, 0);
    }

    // 37. Peak without reuse: single tensor
    #[test]
    fn test_peak_without_reuse_single() {
        let lifetimes = vec![lt(1, 512, 0, 5)];
        assert_eq!(cpu_compute_peak_without_reuse(&lifetimes), 512);
    }

    // 38. Peak without reuse: empty
    #[test]
    fn test_peak_without_reuse_empty() {
        assert_eq!(cpu_compute_peak_without_reuse(&[]), 0);
    }

    // 39. Find reuse: empty input
    #[test]
    fn test_find_reuse_empty() {
        assert!(cpu_find_reuse_opportunities(&[]).is_empty());
    }

    // 40. Planner can be reused for multiple plans
    #[test]
    fn test_planner_reuse() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 128, 0, 1, 64);
        let _ = cpu_compute_plan(&mut p).unwrap();
        assert_eq!(p.stats.total_planned, 1);

        // Add more lifetimes and plan again.
        cpu_add_lifetime(&mut p, 2, 128, 2, 3, 64);
        let _ = cpu_compute_plan(&mut p).unwrap();
        assert_eq!(p.stats.total_planned, 2);
    }

    // 41. Defragment: empty blocks
    #[test]
    fn test_defragment_empty() {
        let mut blocks: Vec<MemoryBlock> = Vec::new();
        let reclaimed = cpu_defragment(&mut blocks);
        assert_eq!(reclaimed, 0);
    }

    // 42. Strategy: WorstFit uses first-fit fallback
    #[test]
    fn test_worst_fit_fallback() {
        let mut p = create_memory_planner(AllocationStrategy::WorstFit);
        cpu_add_lifetime(&mut p, 1, 128, 0, 1, 64);
        cpu_add_lifetime(&mut p, 2, 128, 2, 3, 64);
        let plan = cpu_compute_plan(&mut p).unwrap();
        assert_eq!(plan.assignments.len(), 2);
    }

    // 43. Strategy: BuddySystem uses first-fit fallback
    #[test]
    fn test_buddy_system_fallback() {
        let mut p = create_memory_planner(AllocationStrategy::BuddySystem);
        cpu_add_lifetime(&mut p, 1, 256, 0, 1, 64);
        let plan = cpu_compute_plan(&mut p).unwrap();
        assert_eq!(plan.total_size, 256);
    }

    // 44. Strategy: PoolBased uses first-fit fallback
    #[test]
    fn test_pool_based_fallback() {
        let mut p = create_memory_planner(AllocationStrategy::PoolBased {
            pool_sizes: vec![64, 128, 256],
        });
        cpu_add_lifetime(&mut p, 1, 64, 0, 2, 64);
        let plan = cpu_compute_plan(&mut p).unwrap();
        assert_eq!(plan.total_size, 64);
    }

    // 45. PlannerError display
    #[test]
    fn test_planner_error_display() {
        let e = PlannerError::InsufficientMemory { required: 1000, available: 500 };
        let s = format!("{e}");
        assert!(s.contains("1000"));
        assert!(s.contains("500"));
    }

    // 46. Can reuse: reversed order
    #[test]
    fn test_can_reuse_reversed() {
        let a = lt(1, 100, 5, 8);
        let b = lt(2, 100, 0, 3);
        assert!(cpu_can_reuse(&a, &b));
    }

    // 47. Large-scale reuse (many sequential tensors)
    #[test]
    fn test_large_scale_sequential() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        for i in 0u64..100 {
            cpu_add_lifetime(&mut p, i, 1024, i * 2, i * 2 + 1, 64);
        }
        let plan = cpu_compute_plan(&mut p).unwrap();
        assert_eq!(plan.total_size, 1024);
    }

    // 48. Mixed sizes with reuse
    #[test]
    fn test_mixed_sizes_reuse() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        cpu_add_lifetime(&mut p, 1, 512, 0, 1, 64);
        cpu_add_lifetime(&mut p, 2, 128, 2, 3, 64);
        let plan = cpu_compute_plan(&mut p).unwrap();
        // tensor 2 can reuse tensor 1's block
        assert!(plan.total_size <= 512);
    }

    // 49. Peak reduction percentage
    #[test]
    fn test_peak_reduction_pct() {
        let mut p = create_memory_planner(AllocationStrategy::FirstFit);
        // Sequential tensors allow reuse: 256*3 = 768 naive total, 256 actual.
        cpu_add_lifetime(&mut p, 1, 256, 0, 1, 64);
        cpu_add_lifetime(&mut p, 2, 256, 2, 3, 64);
        cpu_add_lifetime(&mut p, 3, 256, 4, 5, 64);
        let _ = cpu_compute_plan(&mut p).unwrap();
        assert!(p.stats.peak_reduction_pct > 0.0);
    }

    // 50. Fragmentation ratio for single free block is 0
    #[test]
    fn test_frag_single_free_block() {
        let blocks = vec![MemoryBlock {
            offset: 0,
            size: 256,
            alignment: 64,
            tensor_id: None,
            in_use: false,
        }];
        let info = cpu_compute_fragmentation(&blocks);
        assert_eq!(info.fragmentation_ratio, 0.0);
    }
}
