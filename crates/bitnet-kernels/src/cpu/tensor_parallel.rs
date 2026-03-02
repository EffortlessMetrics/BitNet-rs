//! CPU tensor parallelism kernel with sharding, all-reduce, and scatter/gather.
//!
//! Provides simulated multi-rank tensor parallelism for CPU inference,
//! enabling column-parallel and row-parallel sharding strategies with
//! configurable communication backends.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Communication backend used for inter-rank data exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommBackend {
    /// In-process memcpy (single-node simulation).
    InProcess,
    /// Shared-memory transport (placeholder for future IPC).
    SharedMemory,
}

/// Top-level configuration for tensor parallelism.
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// Total number of ranks participating in the parallel group.
    pub num_ranks: usize,
    /// Rank id of the *current* participant (0-based).
    pub rank_id: usize,
    /// Communication backend.
    pub comm_backend: CommBackend,
    /// When `true`, computation and communication may overlap.
    pub overlap_compute_comm: bool,
}

impl TensorParallelConfig {
    /// Validate that the configuration is internally consistent.
    pub fn validate(&self) -> Result<(), TensorParallelError> {
        if self.num_ranks == 0 {
            return Err(TensorParallelError::InvalidConfig("num_ranks must be > 0".into()));
        }
        if self.rank_id >= self.num_ranks {
            return Err(TensorParallelError::InvalidConfig(format!(
                "rank_id {} >= num_ranks {}",
                self.rank_id, self.num_ranks
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the tensor-parallel subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorParallelError {
    /// The configuration is invalid.
    InvalidConfig(String),
    /// Tensor cannot be evenly sharded with the given parameters.
    UnevenSharding { tensor_len: usize, num_shards: usize },
    /// A shard index or rank is out of bounds.
    ShardIndexOutOfBounds { index: usize, total: usize },
    /// Shards provided to `gather` are inconsistent.
    InconsistentShards(String),
    /// Communication failure (placeholder).
    CommFailure(String),
}

impl fmt::Display for TensorParallelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid tensor-parallel config: {msg}"),
            Self::UnevenSharding { tensor_len, num_shards } => {
                write!(f, "tensor length {tensor_len} not evenly divisible by {num_shards} shards")
            }
            Self::ShardIndexOutOfBounds { index, total } => {
                write!(f, "shard index {index} out of bounds (total {total})")
            }
            Self::InconsistentShards(msg) => write!(f, "inconsistent shards: {msg}"),
            Self::CommFailure(msg) => write!(f, "communication failure: {msg}"),
        }
    }
}

impl std::error::Error for TensorParallelError {}

// ---------------------------------------------------------------------------
// Sharding strategy
// ---------------------------------------------------------------------------

/// Strategy for partitioning a tensor across ranks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardingStrategy {
    /// Split columns (second dimension) evenly across ranks.
    ColumnParallel,
    /// Split rows (first dimension) evenly across ranks.
    RowParallel,
    /// Caller-specified split sizes per rank.
    Custom { splits: Vec<usize> },
}

// ---------------------------------------------------------------------------
// Shard
// ---------------------------------------------------------------------------

/// A single shard of a tensor, produced by a sharding operation.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorShard {
    /// The data owned by this shard.
    pub data: Vec<f32>,
    /// Rank that owns this shard.
    pub rank_id: usize,
    /// Zero-based shard index.
    pub shard_index: usize,
    /// Total number of shards the tensor was split into.
    pub total_shards: usize,
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Performance counters for a single tensor-parallel operation.
#[derive(Debug, Clone, Default)]
pub struct TensorParallelMetrics {
    /// Communication wall-clock time in milliseconds.
    pub comm_time_ms: f64,
    /// Compute wall-clock time in milliseconds.
    pub compute_time_ms: f64,
    /// Overlap efficiency in `[0.0, 1.0]`. `1.0` means perfect overlap.
    pub overlap_efficiency: f64,
}

// ---------------------------------------------------------------------------
// Helper: shard range computation
// ---------------------------------------------------------------------------

/// Compute `(start, end)` index ranges for each rank when splitting `len`
/// elements across `num_ranks`.
///
/// If `len` is not evenly divisible, the first `len % num_ranks` ranks receive
/// one extra element.
pub fn compute_shard_ranges(
    len: usize,
    num_ranks: usize,
) -> Result<Vec<(usize, usize)>, TensorParallelError> {
    if num_ranks == 0 {
        return Err(TensorParallelError::InvalidConfig("num_ranks must be > 0".into()));
    }
    let base = len / num_ranks;
    let remainder = len % num_ranks;
    let mut ranges = Vec::with_capacity(num_ranks);
    let mut offset = 0;
    for i in 0..num_ranks {
        let size = base + if i < remainder { 1 } else { 0 };
        ranges.push((offset, offset + size));
        offset += size;
    }
    Ok(ranges)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Check that `tensor_len` can be evenly divided into `num_shards`.
pub fn validate_sharding(tensor_len: usize, num_shards: usize) -> Result<(), TensorParallelError> {
    if num_shards == 0 {
        return Err(TensorParallelError::InvalidConfig("num_shards must be > 0".into()));
    }
    if !tensor_len.is_multiple_of(num_shards) {
        return Err(TensorParallelError::UnevenSharding { tensor_len, num_shards });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Core operations
// ---------------------------------------------------------------------------

/// Split `tensor` into shards according to `strategy` across `config.num_ranks`.
///
/// For `ColumnParallel` / `RowParallel` the tensor is treated as a flat 1-D
/// buffer and split into `num_ranks` equal-sized pieces.  For `Custom`, the
/// caller-provided `splits` list determines the size of each piece.
pub fn shard_tensor(
    tensor: &[f32],
    config: &TensorParallelConfig,
    strategy: &ShardingStrategy,
) -> Result<(Vec<TensorShard>, TensorParallelMetrics), TensorParallelError> {
    config.validate()?;

    let start = Instant::now();

    let splits: Vec<usize> = match strategy {
        ShardingStrategy::ColumnParallel | ShardingStrategy::RowParallel => {
            validate_sharding(tensor.len(), config.num_ranks)?;
            let chunk = tensor.len() / config.num_ranks;
            vec![chunk; config.num_ranks]
        }
        ShardingStrategy::Custom { splits } => {
            if splits.len() != config.num_ranks {
                return Err(TensorParallelError::InvalidConfig(format!(
                    "custom splits length {} != num_ranks {}",
                    splits.len(),
                    config.num_ranks
                )));
            }
            let total: usize = splits.iter().sum();
            if total != tensor.len() {
                return Err(TensorParallelError::InvalidConfig(format!(
                    "custom splits sum {total} != tensor length {}",
                    tensor.len()
                )));
            }
            splits.clone()
        }
    };

    let compute_elapsed = start.elapsed();

    let comm_start = Instant::now();
    let mut shards = Vec::with_capacity(config.num_ranks);
    let mut offset = 0;
    for (i, &size) in splits.iter().enumerate() {
        shards.push(TensorShard {
            data: tensor[offset..offset + size].to_vec(),
            rank_id: i,
            shard_index: i,
            total_shards: config.num_ranks,
        });
        offset += size;
    }
    let comm_elapsed = comm_start.elapsed();

    let total = compute_elapsed + comm_elapsed;
    let overlap_efficiency = if config.overlap_compute_comm && !total.is_zero() {
        let max_component = compute_elapsed.max(comm_elapsed);
        max_component.as_secs_f64() / total.as_secs_f64()
    } else {
        0.0
    };

    let metrics = TensorParallelMetrics {
        comm_time_ms: comm_elapsed.as_secs_f64() * 1000.0,
        compute_time_ms: compute_elapsed.as_secs_f64() * 1000.0,
        overlap_efficiency,
    };

    Ok((shards, metrics))
}

/// Combine shards back into a single tensor.
///
/// Shards are concatenated in shard-index order. All shards must agree on
/// `total_shards` and indices must be a permutation of `0..total_shards`.
pub fn gather_shards(
    shards: &[TensorShard],
) -> Result<(Vec<f32>, TensorParallelMetrics), TensorParallelError> {
    if shards.is_empty() {
        return Err(TensorParallelError::InconsistentShards("no shards provided".into()));
    }

    let total = shards[0].total_shards;
    if shards.len() != total {
        return Err(TensorParallelError::InconsistentShards(format!(
            "expected {} shards, got {}",
            total,
            shards.len()
        )));
    }

    // Sort by shard_index.
    let mut sorted: Vec<&TensorShard> = shards.iter().collect();
    sorted.sort_by_key(|s| s.shard_index);

    for (i, s) in sorted.iter().enumerate() {
        if s.shard_index != i {
            return Err(TensorParallelError::InconsistentShards(format!(
                "missing shard index {i}"
            )));
        }
        if s.total_shards != total {
            return Err(TensorParallelError::InconsistentShards(
                "total_shards mismatch across shards".into(),
            ));
        }
    }

    let comm_start = Instant::now();
    let mut result: Vec<f32> = Vec::with_capacity(sorted.iter().map(|s| s.data.len()).sum());
    for s in &sorted {
        result.extend_from_slice(&s.data);
    }
    let comm_elapsed = comm_start.elapsed();

    Ok((
        result,
        TensorParallelMetrics {
            comm_time_ms: comm_elapsed.as_secs_f64() * 1000.0,
            compute_time_ms: 0.0,
            overlap_efficiency: 0.0,
        },
    ))
}

/// Element-wise sum across all shards (simulated all-reduce for CPU).
///
/// Every shard must have the same length. Returns the reduced buffer.
pub fn all_reduce_sum(
    shards: &[TensorShard],
) -> Result<(Vec<f32>, TensorParallelMetrics), TensorParallelError> {
    if shards.is_empty() {
        return Err(TensorParallelError::InconsistentShards("no shards provided".into()));
    }

    let len = shards[0].data.len();
    for s in shards.iter().skip(1) {
        if s.data.len() != len {
            return Err(TensorParallelError::InconsistentShards(format!(
                "shard lengths differ: expected {len}, got {}",
                s.data.len()
            )));
        }
    }

    let start = Instant::now();
    let mut acc = vec![0.0f32; len];
    for s in shards {
        for (a, &v) in acc.iter_mut().zip(s.data.iter()) {
            *a += v;
        }
    }
    let elapsed = start.elapsed();

    Ok((
        acc,
        TensorParallelMetrics {
            comm_time_ms: elapsed.as_secs_f64() * 1000.0,
            compute_time_ms: 0.0,
            overlap_efficiency: 0.0,
        },
    ))
}

/// Send one shard per rank according to `assignments`.
///
/// `assignments[i]` is the rank that receives shard `i`.
/// Returns one `TensorShard` per rank (in rank order).
pub fn scatter_tensor(
    tensor: &[f32],
    config: &TensorParallelConfig,
    assignments: &[usize],
) -> Result<(Vec<TensorShard>, TensorParallelMetrics), TensorParallelError> {
    config.validate()?;

    if assignments.len() != config.num_ranks {
        return Err(TensorParallelError::InvalidConfig(format!(
            "assignments length {} != num_ranks {}",
            assignments.len(),
            config.num_ranks
        )));
    }

    validate_sharding(tensor.len(), config.num_ranks)?;
    let chunk = tensor.len() / config.num_ranks;

    let start = Instant::now();
    let mut shards: Vec<Option<TensorShard>> = (0..config.num_ranks).map(|_| None).collect();

    for (i, &target_rank) in assignments.iter().enumerate() {
        if target_rank >= config.num_ranks {
            return Err(TensorParallelError::ShardIndexOutOfBounds {
                index: target_rank,
                total: config.num_ranks,
            });
        }
        shards[target_rank] = Some(TensorShard {
            data: tensor[i * chunk..(i + 1) * chunk].to_vec(),
            rank_id: target_rank,
            shard_index: i,
            total_shards: config.num_ranks,
        });
    }
    let elapsed = start.elapsed();

    let result: Vec<TensorShard> = shards.into_iter().flatten().collect();
    if result.len() != config.num_ranks {
        return Err(TensorParallelError::InvalidConfig(
            "duplicate target rank in assignments".into(),
        ));
    }

    Ok((
        result,
        TensorParallelMetrics {
            comm_time_ms: elapsed.as_secs_f64() * 1000.0,
            compute_time_ms: 0.0,
            overlap_efficiency: 0.0,
        },
    ))
}

/// Reduce (element-wise sum) then scatter the result across ranks.
///
/// Each rank receives an equal-sized chunk of the reduced tensor.
pub fn reduce_scatter(
    shards: &[TensorShard],
    config: &TensorParallelConfig,
) -> Result<(Vec<TensorShard>, TensorParallelMetrics), TensorParallelError> {
    config.validate()?;

    let compute_start = Instant::now();
    let (reduced, _) = all_reduce_sum(shards)?;
    let compute_elapsed = compute_start.elapsed();

    validate_sharding(reduced.len(), config.num_ranks)?;
    let chunk = reduced.len() / config.num_ranks;

    let comm_start = Instant::now();
    let mut result = Vec::with_capacity(config.num_ranks);
    for i in 0..config.num_ranks {
        let offset = i * chunk;
        result.push(TensorShard {
            data: reduced[offset..offset + chunk].to_vec(),
            rank_id: i,
            shard_index: i,
            total_shards: config.num_ranks,
        });
    }
    let comm_elapsed = comm_start.elapsed();

    let total = compute_elapsed + comm_elapsed;
    let overlap_efficiency = if config.overlap_compute_comm && !total.is_zero() {
        let max_component = compute_elapsed.max(comm_elapsed);
        max_component.as_secs_f64() / total.as_secs_f64()
    } else {
        0.0
    };

    Ok((
        result,
        TensorParallelMetrics {
            comm_time_ms: comm_elapsed.as_secs_f64() * 1000.0,
            compute_time_ms: compute_elapsed.as_secs_f64() * 1000.0,
            overlap_efficiency,
        },
    ))
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- helpers -------------------------------------------------------

    fn default_config(num_ranks: usize) -> TensorParallelConfig {
        TensorParallelConfig {
            num_ranks,
            rank_id: 0,
            comm_backend: CommBackend::InProcess,
            overlap_compute_comm: false,
        }
    }

    fn make_tensor(len: usize) -> Vec<f32> {
        (0..len).map(|i| i as f32).collect()
    }

    // ----- config validation ---------------------------------------------

    #[test]
    fn config_valid() {
        let cfg = default_config(4);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_zero_ranks() {
        let cfg = TensorParallelConfig {
            num_ranks: 0,
            rank_id: 0,
            comm_backend: CommBackend::InProcess,
            overlap_compute_comm: false,
        };
        assert!(matches!(cfg.validate(), Err(TensorParallelError::InvalidConfig(_))));
    }

    #[test]
    fn config_rank_out_of_range() {
        let cfg = TensorParallelConfig {
            num_ranks: 2,
            rank_id: 2,
            comm_backend: CommBackend::InProcess,
            overlap_compute_comm: false,
        };
        assert!(matches!(cfg.validate(), Err(TensorParallelError::InvalidConfig(_))));
    }

    #[test]
    fn config_rank_at_boundary() {
        let cfg = TensorParallelConfig {
            num_ranks: 4,
            rank_id: 3,
            comm_backend: CommBackend::InProcess,
            overlap_compute_comm: false,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_shared_memory_backend() {
        let cfg = TensorParallelConfig {
            num_ranks: 2,
            rank_id: 0,
            comm_backend: CommBackend::SharedMemory,
            overlap_compute_comm: true,
        };
        assert!(cfg.validate().is_ok());
    }

    // ----- validate_sharding --------------------------------------------

    #[test]
    fn validate_sharding_even() {
        assert!(validate_sharding(12, 4).is_ok());
    }

    #[test]
    fn validate_sharding_uneven() {
        assert!(matches!(
            validate_sharding(13, 4),
            Err(TensorParallelError::UnevenSharding { .. })
        ));
    }

    #[test]
    fn validate_sharding_zero_shards() {
        assert!(matches!(validate_sharding(8, 0), Err(TensorParallelError::InvalidConfig(_))));
    }

    #[test]
    fn validate_sharding_single_shard() {
        assert!(validate_sharding(7, 1).is_ok());
    }

    #[test]
    fn validate_sharding_zero_length() {
        assert!(validate_sharding(0, 3).is_ok());
    }

    // ----- compute_shard_ranges ------------------------------------------

    #[test]
    fn shard_ranges_even() {
        let ranges = compute_shard_ranges(12, 4).unwrap();
        assert_eq!(ranges, vec![(0, 3), (3, 6), (6, 9), (9, 12)]);
    }

    #[test]
    fn shard_ranges_uneven() {
        let ranges = compute_shard_ranges(10, 3).unwrap();
        assert_eq!(ranges, vec![(0, 4), (4, 7), (7, 10)]);
    }

    #[test]
    fn shard_ranges_single_rank() {
        let ranges = compute_shard_ranges(5, 1).unwrap();
        assert_eq!(ranges, vec![(0, 5)]);
    }

    #[test]
    fn shard_ranges_zero_length() {
        let ranges = compute_shard_ranges(0, 3).unwrap();
        assert_eq!(ranges, vec![(0, 0), (0, 0), (0, 0)]);
    }

    #[test]
    fn shard_ranges_zero_ranks() {
        assert!(compute_shard_ranges(10, 0).is_err());
    }

    #[test]
    fn shard_ranges_more_ranks_than_elements() {
        let ranges = compute_shard_ranges(2, 5).unwrap();
        assert_eq!(ranges, vec![(0, 1), (1, 2), (2, 2), (2, 2), (2, 2)]);
    }

    // ----- shard_tensor --------------------------------------------------

    #[test]
    fn shard_column_parallel() {
        let t = make_tensor(8);
        let cfg = default_config(4);
        let (shards, _) = shard_tensor(&t, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0].data, vec![0.0, 1.0]);
        assert_eq!(shards[3].data, vec![6.0, 7.0]);
    }

    #[test]
    fn shard_row_parallel() {
        let t = make_tensor(6);
        let cfg = default_config(3);
        let (shards, _) = shard_tensor(&t, &cfg, &ShardingStrategy::RowParallel).unwrap();
        assert_eq!(shards.len(), 3);
        assert_eq!(shards[1].data, vec![2.0, 3.0]);
    }

    #[test]
    fn shard_custom_splits() {
        let t = make_tensor(10);
        let cfg = default_config(3);
        let strategy = ShardingStrategy::Custom { splits: vec![2, 5, 3] };
        let (shards, _) = shard_tensor(&t, &cfg, &strategy).unwrap();
        assert_eq!(shards[0].data.len(), 2);
        assert_eq!(shards[1].data.len(), 5);
        assert_eq!(shards[2].data.len(), 3);
    }

    #[test]
    fn shard_custom_splits_wrong_count() {
        let t = make_tensor(10);
        let cfg = default_config(2);
        let strategy = ShardingStrategy::Custom { splits: vec![2, 5, 3] };
        assert!(shard_tensor(&t, &cfg, &strategy).is_err());
    }

    #[test]
    fn shard_custom_splits_wrong_sum() {
        let t = make_tensor(10);
        let cfg = default_config(2);
        let strategy = ShardingStrategy::Custom { splits: vec![3, 3] };
        assert!(shard_tensor(&t, &cfg, &strategy).is_err());
    }

    #[test]
    fn shard_uneven_column_parallel() {
        let t = make_tensor(7);
        let cfg = default_config(3);
        assert!(matches!(
            shard_tensor(&t, &cfg, &ShardingStrategy::ColumnParallel),
            Err(TensorParallelError::UnevenSharding { .. })
        ));
    }

    #[test]
    fn shard_metadata_correct() {
        let t = make_tensor(6);
        let cfg = default_config(3);
        let (shards, _) = shard_tensor(&t, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        for (i, s) in shards.iter().enumerate() {
            assert_eq!(s.rank_id, i);
            assert_eq!(s.shard_index, i);
            assert_eq!(s.total_shards, 3);
        }
    }

    // ----- gather_shards -------------------------------------------------

    #[test]
    fn gather_basic() {
        let t = make_tensor(8);
        let cfg = default_config(4);
        let (shards, _) = shard_tensor(&t, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        let (gathered, _) = gather_shards(&shards).unwrap();
        assert_eq!(gathered, t);
    }

    #[test]
    fn gather_empty() {
        assert!(gather_shards(&[]).is_err());
    }

    #[test]
    fn gather_wrong_count() {
        let s = TensorShard { data: vec![1.0], rank_id: 0, shard_index: 0, total_shards: 2 };
        assert!(gather_shards(&[s]).is_err());
    }

    #[test]
    fn gather_missing_index() {
        let s0 = TensorShard { data: vec![1.0], rank_id: 0, shard_index: 0, total_shards: 2 };
        let s1 = TensorShard {
            data: vec![2.0],
            rank_id: 1,
            shard_index: 0, // duplicate index
            total_shards: 2,
        };
        assert!(gather_shards(&[s0, s1]).is_err());
    }

    #[test]
    fn gather_out_of_order() {
        let s0 = TensorShard { data: vec![3.0, 4.0], rank_id: 1, shard_index: 1, total_shards: 2 };
        let s1 = TensorShard { data: vec![1.0, 2.0], rank_id: 0, shard_index: 0, total_shards: 2 };
        let (gathered, _) = gather_shards(&[s0, s1]).unwrap();
        assert_eq!(gathered, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ----- shard/gather round-trip ----------------------------------------

    #[test]
    fn round_trip_column_parallel() {
        let t = make_tensor(16);
        let cfg = default_config(4);
        let (shards, _) = shard_tensor(&t, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        let (gathered, _) = gather_shards(&shards).unwrap();
        assert_eq!(gathered, t);
    }

    #[test]
    fn round_trip_row_parallel() {
        let t = make_tensor(12);
        let cfg = default_config(3);
        let (shards, _) = shard_tensor(&t, &cfg, &ShardingStrategy::RowParallel).unwrap();
        let (gathered, _) = gather_shards(&shards).unwrap();
        assert_eq!(gathered, t);
    }

    #[test]
    fn round_trip_custom() {
        let t = make_tensor(10);
        let cfg = default_config(3);
        let strategy = ShardingStrategy::Custom { splits: vec![2, 5, 3] };
        let (shards, _) = shard_tensor(&t, &cfg, &strategy).unwrap();
        let (gathered, _) = gather_shards(&shards).unwrap();
        assert_eq!(gathered, t);
    }

    #[test]
    fn round_trip_single_rank() {
        let t = make_tensor(5);
        let cfg = default_config(1);
        let (shards, _) = shard_tensor(&t, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        let (gathered, _) = gather_shards(&shards).unwrap();
        assert_eq!(gathered, t);
    }

    // ----- all_reduce_sum ------------------------------------------------

    #[test]
    fn all_reduce_basic() {
        let shards = vec![
            TensorShard { data: vec![1.0, 2.0, 3.0], rank_id: 0, shard_index: 0, total_shards: 2 },
            TensorShard { data: vec![4.0, 5.0, 6.0], rank_id: 1, shard_index: 1, total_shards: 2 },
        ];
        let (result, _) = all_reduce_sum(&shards).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn all_reduce_single_shard() {
        let shards =
            vec![TensorShard { data: vec![1.0, 2.0], rank_id: 0, shard_index: 0, total_shards: 1 }];
        let (result, _) = all_reduce_sum(&shards).unwrap();
        assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn all_reduce_empty() {
        assert!(all_reduce_sum(&[]).is_err());
    }

    #[test]
    fn all_reduce_mismatched_lengths() {
        let shards = vec![
            TensorShard { data: vec![1.0], rank_id: 0, shard_index: 0, total_shards: 2 },
            TensorShard { data: vec![1.0, 2.0], rank_id: 1, shard_index: 1, total_shards: 2 },
        ];
        assert!(all_reduce_sum(&shards).is_err());
    }

    #[test]
    fn all_reduce_four_ranks() {
        let shards: Vec<TensorShard> = (0..4)
            .map(|i| TensorShard {
                data: vec![1.0; 3],
                rank_id: i,
                shard_index: i,
                total_shards: 4,
            })
            .collect();
        let (result, _) = all_reduce_sum(&shards).unwrap();
        assert_eq!(result, vec![4.0, 4.0, 4.0]);
    }

    #[test]
    fn all_reduce_zeros() {
        let shards: Vec<TensorShard> = (0..3)
            .map(|i| TensorShard {
                data: vec![0.0; 4],
                rank_id: i,
                shard_index: i,
                total_shards: 3,
            })
            .collect();
        let (result, _) = all_reduce_sum(&shards).unwrap();
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn all_reduce_negative_values() {
        let shards = vec![
            TensorShard { data: vec![-1.0, 2.0], rank_id: 0, shard_index: 0, total_shards: 2 },
            TensorShard { data: vec![1.0, -2.0], rank_id: 1, shard_index: 1, total_shards: 2 },
        ];
        let (result, _) = all_reduce_sum(&shards).unwrap();
        assert_eq!(result, vec![0.0, 0.0]);
    }

    // ----- scatter_tensor ------------------------------------------------

    #[test]
    fn scatter_basic() {
        let t = make_tensor(6);
        let cfg = default_config(3);
        let (shards, _) = scatter_tensor(&t, &cfg, &[0, 1, 2]).unwrap();
        assert_eq!(shards.len(), 3);
        assert_eq!(shards[0].data, vec![0.0, 1.0]);
        assert_eq!(shards[1].data, vec![2.0, 3.0]);
        assert_eq!(shards[2].data, vec![4.0, 5.0]);
    }

    #[test]
    fn scatter_reversed_assignments() {
        let t = make_tensor(4);
        let cfg = default_config(2);
        let (shards, _) = scatter_tensor(&t, &cfg, &[1, 0]).unwrap();
        assert_eq!(shards[0].data, vec![2.0, 3.0]); // rank 0 ← shard 1
        assert_eq!(shards[1].data, vec![0.0, 1.0]); // rank 1 ← shard 0
    }

    #[test]
    fn scatter_wrong_assignment_length() {
        let t = make_tensor(4);
        let cfg = default_config(2);
        assert!(scatter_tensor(&t, &cfg, &[0]).is_err());
    }

    #[test]
    fn scatter_out_of_bounds_rank() {
        let t = make_tensor(4);
        let cfg = default_config(2);
        assert!(scatter_tensor(&t, &cfg, &[0, 5]).is_err());
    }

    #[test]
    fn scatter_duplicate_rank() {
        let t = make_tensor(4);
        let cfg = default_config(2);
        assert!(scatter_tensor(&t, &cfg, &[0, 0]).is_err());
    }

    // ----- scatter/gather symmetry ----------------------------------------

    #[test]
    fn scatter_gather_round_trip() {
        let t = make_tensor(8);
        let cfg = default_config(4);
        let (shards, _) = scatter_tensor(&t, &cfg, &[0, 1, 2, 3]).unwrap();
        let (gathered, _) = gather_shards(&shards).unwrap();
        assert_eq!(gathered, t);
    }

    // ----- reduce_scatter ------------------------------------------------

    #[test]
    fn reduce_scatter_basic() {
        let shards = vec![
            TensorShard {
                data: vec![1.0, 2.0, 3.0, 4.0],
                rank_id: 0,
                shard_index: 0,
                total_shards: 2,
            },
            TensorShard {
                data: vec![10.0, 20.0, 30.0, 40.0],
                rank_id: 1,
                shard_index: 1,
                total_shards: 2,
            },
        ];
        let cfg = default_config(2);
        let (result, _) = reduce_scatter(&shards, &cfg).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].data, vec![11.0, 22.0]);
        assert_eq!(result[1].data, vec![33.0, 44.0]);
    }

    #[test]
    fn reduce_scatter_single_rank() {
        let shards =
            vec![TensorShard { data: vec![5.0, 6.0], rank_id: 0, shard_index: 0, total_shards: 1 }];
        let cfg = default_config(1);
        let (result, _) = reduce_scatter(&shards, &cfg).unwrap();
        assert_eq!(result[0].data, vec![5.0, 6.0]);
    }

    #[test]
    fn reduce_scatter_uneven_fails() {
        let shards = vec![
            TensorShard { data: vec![1.0, 2.0, 3.0], rank_id: 0, shard_index: 0, total_shards: 2 },
            TensorShard { data: vec![4.0, 5.0, 6.0], rank_id: 1, shard_index: 1, total_shards: 2 },
        ];
        let cfg = default_config(2);
        assert!(matches!(
            reduce_scatter(&shards, &cfg),
            Err(TensorParallelError::UnevenSharding { .. })
        ));
    }

    // ----- metrics tracking -----------------------------------------------

    #[test]
    fn shard_returns_metrics() {
        let t = make_tensor(8);
        let cfg = default_config(2);
        let (_, m) = shard_tensor(&t, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        assert!(m.comm_time_ms >= 0.0);
        assert!(m.compute_time_ms >= 0.0);
    }

    #[test]
    fn gather_returns_metrics() {
        let t = make_tensor(4);
        let cfg = default_config(2);
        let (shards, _) = shard_tensor(&t, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        let (_, m) = gather_shards(&shards).unwrap();
        assert!(m.comm_time_ms >= 0.0);
    }

    #[test]
    fn all_reduce_returns_metrics() {
        let shards =
            vec![TensorShard { data: vec![1.0], rank_id: 0, shard_index: 0, total_shards: 1 }];
        let (_, m) = all_reduce_sum(&shards).unwrap();
        assert!(m.comm_time_ms >= 0.0);
    }

    #[test]
    fn overlap_efficiency_with_flag() {
        let t = make_tensor(8);
        let cfg = TensorParallelConfig {
            num_ranks: 2,
            rank_id: 0,
            comm_backend: CommBackend::InProcess,
            overlap_compute_comm: true,
        };
        let (_, m) = shard_tensor(&t, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        assert!(m.overlap_efficiency >= 0.0 && m.overlap_efficiency <= 1.0);
    }

    #[test]
    fn no_overlap_efficiency_without_flag() {
        let t = make_tensor(8);
        let cfg = default_config(2);
        let (_, m) = shard_tensor(&t, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        assert_eq!(m.overlap_efficiency, 0.0);
    }

    // ----- error display --------------------------------------------------

    #[test]
    fn error_display_invalid_config() {
        let e = TensorParallelError::InvalidConfig("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn error_display_uneven_sharding() {
        let e = TensorParallelError::UnevenSharding { tensor_len: 7, num_shards: 3 };
        let s = e.to_string();
        assert!(s.contains("7") && s.contains("3"));
    }

    #[test]
    fn error_display_shard_oob() {
        let e = TensorParallelError::ShardIndexOutOfBounds { index: 5, total: 3 };
        let s = e.to_string();
        assert!(s.contains("5") && s.contains("3"));
    }

    #[test]
    fn error_display_inconsistent() {
        let e = TensorParallelError::InconsistentShards("mismatch".into());
        assert!(e.to_string().contains("mismatch"));
    }

    #[test]
    fn error_display_comm_failure() {
        let e = TensorParallelError::CommFailure("timeout".into());
        assert!(e.to_string().contains("timeout"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(TensorParallelError::CommFailure("test".into()));
        assert!(e.to_string().contains("test"));
    }

    // ----- large tensor ---------------------------------------------------

    #[test]
    fn shard_gather_large_tensor() {
        let t: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let cfg = default_config(8);
        let (shards, _) = shard_tensor(&t, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
        assert_eq!(shards.len(), 8);
        assert_eq!(shards[0].data.len(), 128);
        let (gathered, _) = gather_shards(&shards).unwrap();
        assert_eq!(gathered, t);
    }

    // ----- property tests with proptest ----------------------------------

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Generate `num_ranks` in [1, 16] and tensor whose length is a
        /// multiple of num_ranks.
        fn rankable_tensor() -> impl Strategy<Value = (usize, Vec<f32>)> {
            (1usize..=16).prop_flat_map(|nr| {
                let max_chunks = 64usize;
                (1usize..=max_chunks).prop_map(move |chunks| {
                    let len = nr * chunks;
                    let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
                    (nr, data)
                })
            })
        }

        proptest! {
            #[test]
            fn prop_shard_gather_roundtrip((num_ranks, tensor) in rankable_tensor()) {
                let cfg = default_config(num_ranks);
                let (shards, _) =
                    shard_tensor(&tensor, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
                let (gathered, _) = gather_shards(&shards).unwrap();
                prop_assert_eq!(gathered, tensor);
            }

            #[test]
            fn prop_shard_preserves_total_length((num_ranks, tensor) in rankable_tensor()) {
                let cfg = default_config(num_ranks);
                let (shards, _) =
                    shard_tensor(&tensor, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
                let total_len: usize = shards.iter().map(|s| s.data.len()).sum();
                prop_assert_eq!(total_len, tensor.len());
            }

            #[test]
            fn prop_shard_indices_unique((num_ranks, tensor) in rankable_tensor()) {
                let cfg = default_config(num_ranks);
                let (shards, _) =
                    shard_tensor(&tensor, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
                let mut indices: Vec<usize> = shards.iter().map(|s| s.shard_index).collect();
                indices.sort();
                indices.dedup();
                prop_assert_eq!(indices.len(), num_ranks);
            }

            #[test]
            fn prop_all_reduce_sum_correct(
                len in 1usize..64,
                n_ranks in 2usize..8,
            ) {
                let shards: Vec<TensorShard> = (0..n_ranks)
                    .map(|r| TensorShard {
                        data: vec![1.0; len],
                        rank_id: r,
                        shard_index: r,
                        total_shards: n_ranks,
                    })
                    .collect();
                let (result, _) = all_reduce_sum(&shards).unwrap();
                for &v in &result {
                    prop_assert!((v - n_ranks as f32).abs() < 1e-6);
                }
            }

            #[test]
            fn prop_scatter_gather_identity((num_ranks, tensor) in rankable_tensor()) {
                let cfg = default_config(num_ranks);
                let assignments: Vec<usize> = (0..num_ranks).collect();
                let (shards, _) = scatter_tensor(&tensor, &cfg, &assignments).unwrap();
                let (gathered, _) = gather_shards(&shards).unwrap();
                prop_assert_eq!(gathered, tensor);
            }

            #[test]
            fn prop_compute_shard_ranges_covers_all(
                len in 0usize..256,
                num_ranks in 1usize..16,
            ) {
                let ranges = compute_shard_ranges(len, num_ranks).unwrap();
                prop_assert_eq!(ranges[0].0, 0);
                prop_assert_eq!(ranges.last().unwrap().1, len);
                for w in ranges.windows(2) {
                    prop_assert_eq!(w[0].1, w[1].0);
                }
            }

            #[test]
            fn prop_metrics_non_negative((num_ranks, tensor) in rankable_tensor()) {
                let cfg = default_config(num_ranks);
                let (_, m) = shard_tensor(&tensor, &cfg, &ShardingStrategy::ColumnParallel).unwrap();
                prop_assert!(m.comm_time_ms >= 0.0);
                prop_assert!(m.compute_time_ms >= 0.0);
                prop_assert!(m.overlap_efficiency >= 0.0);
            }
        }
    }
}
