//! GPU memory planning for efficient buffer reuse.
//!
//! Analyses the sequence of [`PipelineStage`]s and assigns buffer IDs so
//! that non-overlapping lifetimes share the same allocation. The result
//! is a set of [`BufferAllocation`]s plus the estimated peak memory.

use crate::pipeline::PipelineStage;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single planned GPU buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferAllocation {
    /// Unique allocation ID.
    pub id: usize,
    /// Size in bytes.
    pub size: u64,
    /// Index of the first stage that uses this buffer.
    pub first_use: usize,
    /// Index of the last stage that uses this buffer.
    pub last_use: usize,
    /// If this buffer reuses a previously freed allocation, the ID of
    /// that allocation. `None` for fresh allocations.
    pub reuse_of: Option<usize>,
}

/// Plans GPU memory allocation to minimise peak usage.
#[derive(Debug)]
pub struct MemoryPlanner {
    allocations: Vec<BufferAllocation>,
    peak_memory: u64,
}

impl MemoryPlanner {
    /// Analyse `stages` and produce a memory plan.
    ///
    /// Each stage produces one output buffer that is consumed by the
    /// *next* stage. When a buffer's last consumer has completed, its
    /// slot is available for reuse by a later stage of equal or smaller
    /// size.
    pub fn plan(stages: &[PipelineStage]) -> Self {
        if stages.is_empty() {
            return Self { allocations: Vec::new(), peak_memory: 0 };
        }

        let mut allocations = Vec::new();
        // Pool of freed (id, size) pairs available for reuse.
        let mut free_pool: Vec<(usize, u64)> = Vec::new();
        let mut peak: u64 = 0;
        let mut live_bytes: u64 = 0;

        for (idx, stage) in stages.iter().enumerate() {
            let needed = stage.output_bytes();

            // Try to reuse the smallest freed buffer that fits.
            let reuse = free_pool
                .iter()
                .enumerate()
                .filter(|(_, (_, sz))| *sz >= needed)
                .min_by_key(|(_, (_, sz))| *sz)
                .map(|(pool_idx, (alloc_id, _))| (pool_idx, *alloc_id));

            let alloc = if let Some((pool_idx, reuse_id)) = reuse {
                let (_, reuse_sz) = free_pool.remove(pool_idx);
                let id = allocations.len();
                BufferAllocation {
                    id,
                    size: reuse_sz, // keep original size
                    first_use: idx,
                    last_use: idx,
                    reuse_of: Some(reuse_id),
                }
            } else {
                live_bytes += needed;
                if live_bytes > peak {
                    peak = live_bytes;
                }
                let id = allocations.len();
                BufferAllocation { id, size: needed, first_use: idx, last_use: idx, reuse_of: None }
            };

            allocations.push(alloc);

            // Free buffers whose last_use ≤ current index (all prior
            // single-use buffers).
            // In this simple model every buffer is consumed by the
            // immediately following stage, so we free the buffer from
            // the *previous* stage.
            if idx > 0 {
                let prev_id = idx - 1;
                if let Some(prev) = allocations.get(prev_id) {
                    if prev.reuse_of.is_none() {
                        // Only return genuinely-allocated buffers.
                        live_bytes = live_bytes.saturating_sub(prev.size);
                    }
                    free_pool.push((prev.id, prev.size));
                }
            }
        }

        Self { allocations, peak_memory: peak }
    }

    /// Estimated peak GPU memory usage in bytes.
    #[must_use]
    pub const fn peak_memory(&self) -> u64 {
        self.peak_memory
    }

    /// All planned buffer allocations.
    #[must_use]
    pub fn allocations(&self) -> &[BufferAllocation] {
        &self.allocations
    }

    /// Number of buffers that reuse a previous allocation.
    #[must_use]
    pub fn reuse_count(&self) -> usize {
        self.allocations.iter().filter(|a| a.reuse_of.is_some()).count()
    }
}
