//! Work partitioning strategies for distributing elements across GPU blocks.

use crate::grid_blocks_1d;

/// Strategy for distributing work across GPU thread blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PartitionStrategy {
    /// Each block processes a contiguous chunk.  Simplest and best for
    /// coalesced-memory workloads.
    RoundRobin,
    /// Chunks are sized proportionally to estimated per-element cost.
    /// The weight slice (one per partition) is normalised internally.
    LoadBalanced,
    /// Blocks pull work from a shared atomic counter.  Modelled here as
    /// uniform initial partitions with a "steal" margin.
    Dynamic {
        /// Fraction of a block's chunk that may be stolen (`0.0..=1.0`).
        steal_ratio: f64,
    },
}

/// A single partition (`start..end`) in element space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkPartition {
    /// First element index (inclusive).
    pub start: u64,
    /// Past-the-end element index (exclusive).
    pub end: u64,
    /// Block index this partition maps to.
    pub block_id: u32,
}

impl WorkPartition {
    /// Number of elements in this partition.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.end - self.start
    }

    /// Whether this partition is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// Compute partitions for `n` elements across grid blocks.
///
/// Returns one [`WorkPartition`] per block.
#[must_use]
pub fn partition(
    n: u64,
    block_size: u32,
    strategy: PartitionStrategy,
    weights: &[f64],
) -> Vec<WorkPartition> {
    if n == 0 {
        return Vec::new();
    }
    let num_blocks = u64::from(grid_blocks_1d(n, block_size));
    match strategy {
        PartitionStrategy::RoundRobin => partition_round_robin(n, num_blocks),
        PartitionStrategy::LoadBalanced => partition_load_balanced(n, num_blocks, weights),
        PartitionStrategy::Dynamic { steal_ratio } => partition_dynamic(n, num_blocks, steal_ratio),
    }
}

fn partition_round_robin(n: u64, num_blocks: u64) -> Vec<WorkPartition> {
    let chunk = n / num_blocks;
    let remainder = n % num_blocks;
    let mut parts = Vec::with_capacity(usize::try_from(num_blocks).unwrap_or(usize::MAX));
    let mut offset = 0u64;
    for i in 0..num_blocks {
        let extra = u64::from(i < remainder);
        let len = chunk + extra;
        parts.push(WorkPartition {
            start: offset,
            end: offset + len,
            block_id: u32::try_from(i).unwrap_or(u32::MAX),
        });
        offset += len;
    }
    parts
}

fn partition_load_balanced(n: u64, num_blocks: u64, weights: &[f64]) -> Vec<WorkPartition> {
    let expected_len = usize::try_from(num_blocks).unwrap_or(usize::MAX);
    // If no weights or wrong length, fall back to round-robin.
    if weights.is_empty() || weights.len() != expected_len {
        return partition_round_robin(n, num_blocks);
    }
    let total_weight: f64 = weights.iter().copied().sum();
    if total_weight <= 0.0 {
        return partition_round_robin(n, num_blocks);
    }

    let mut parts = Vec::with_capacity(expected_len);
    let mut offset = 0u64;
    let mut assigned = 0u64;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    for (i, &w) in weights.iter().enumerate() {
        let is_last = i + 1 == weights.len();
        let share = if is_last {
            n - assigned
        } else {
            // Precision loss is acceptable for work-distribution heuristic.
            let raw = (n as f64 * (w / total_weight)).round() as u64;
            raw.min(n - assigned)
        };
        parts.push(WorkPartition {
            start: offset,
            end: offset + share,
            block_id: u32::try_from(i).unwrap_or(u32::MAX),
        });
        offset += share;
        assigned += share;
    }
    parts
}

fn partition_dynamic(n: u64, num_blocks: u64, steal_ratio: f64) -> Vec<WorkPartition> {
    let ratio = steal_ratio.clamp(0.0, 1.0);
    let base = partition_round_robin(n, num_blocks);
    base.iter()
        .map(|p| {
            let full_len = p.len();
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let keep = ((full_len as f64) * (1.0 - ratio)).ceil() as u64;
            let keep = keep.max(1).min(full_len);
            WorkPartition { start: p.start, end: p.start + keep, block_id: p.block_id }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_BLOCK_SIZE;

    // ── Round-robin ────────────────────────────────────────────────────

    #[test]
    fn rr_exact_division() {
        let parts = partition(1024, DEFAULT_BLOCK_SIZE, PartitionStrategy::RoundRobin, &[]);
        assert_eq!(parts.len(), 4);
        assert!(parts.iter().all(|p| p.len() == 256));
    }

    #[test]
    fn rr_remainder() {
        let parts = partition(1000, DEFAULT_BLOCK_SIZE, PartitionStrategy::RoundRobin, &[]);
        assert_eq!(parts.len(), 4);
        let total: u64 = parts.iter().map(WorkPartition::len).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn rr_single_element() {
        let parts = partition(1, DEFAULT_BLOCK_SIZE, PartitionStrategy::RoundRobin, &[]);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].len(), 1);
    }

    #[test]
    fn rr_zero_elements() {
        let parts = partition(0, DEFAULT_BLOCK_SIZE, PartitionStrategy::RoundRobin, &[]);
        assert!(parts.is_empty());
    }

    #[test]
    fn rr_contiguous() {
        let parts = partition(1000, DEFAULT_BLOCK_SIZE, PartitionStrategy::RoundRobin, &[]);
        for w in parts.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
    }

    // ── Load-balanced ──────────────────────────────────────────────────

    #[test]
    fn lb_equal_weights() {
        let parts = partition(
            1024,
            DEFAULT_BLOCK_SIZE,
            PartitionStrategy::LoadBalanced,
            &[1.0, 1.0, 1.0, 1.0],
        );
        assert_eq!(parts.len(), 4);
        let total: u64 = parts.iter().map(WorkPartition::len).sum();
        assert_eq!(total, 1024);
    }

    #[test]
    fn lb_unequal_weights() {
        let parts = partition(1000, 500, PartitionStrategy::LoadBalanced, &[3.0, 1.0]);
        assert_eq!(parts.len(), 2);
        assert!(parts[0].len() > parts[1].len());
        let total: u64 = parts.iter().map(WorkPartition::len).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn lb_wrong_weight_count_falls_back() {
        let parts = partition(
            256,
            DEFAULT_BLOCK_SIZE,
            PartitionStrategy::LoadBalanced,
            &[1.0], // wrong count
        );
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].len(), 256);
    }

    #[test]
    fn lb_zero_weights_falls_back() {
        let parts = partition(256, DEFAULT_BLOCK_SIZE, PartitionStrategy::LoadBalanced, &[0.0]);
        assert_eq!(parts.len(), 1);
    }

    // ── Dynamic ────────────────────────────────────────────────────────

    #[test]
    fn dynamic_zero_steal() {
        let parts = partition(
            1024,
            DEFAULT_BLOCK_SIZE,
            PartitionStrategy::Dynamic { steal_ratio: 0.0 },
            &[],
        );
        assert_eq!(parts.len(), 4);
        assert!(parts.iter().all(|p| p.len() == 256));
    }

    #[test]
    fn dynamic_partial_steal() {
        let parts = partition(
            1024,
            DEFAULT_BLOCK_SIZE,
            PartitionStrategy::Dynamic { steal_ratio: 0.5 },
            &[],
        );
        assert_eq!(parts.len(), 4);
        for p in &parts {
            assert!(p.len() <= 256);
            assert!(!p.is_empty());
        }
    }

    #[test]
    fn dynamic_full_steal() {
        let parts = partition(
            1024,
            DEFAULT_BLOCK_SIZE,
            PartitionStrategy::Dynamic { steal_ratio: 1.0 },
            &[],
        );
        assert!(parts.iter().all(|p| !p.is_empty()));
    }

    // ── WorkPartition ──────────────────────────────────────────────────

    #[test]
    fn partition_is_empty() {
        let wp = WorkPartition { start: 5, end: 5, block_id: 0 };
        assert!(wp.is_empty());
        assert_eq!(wp.len(), 0);
    }

    #[test]
    fn partition_non_empty() {
        let wp = WorkPartition { start: 0, end: 10, block_id: 0 };
        assert!(!wp.is_empty());
        assert_eq!(wp.len(), 10);
    }
}
