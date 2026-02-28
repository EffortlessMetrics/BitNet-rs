//! # Tensor Parallelism
//!
//! Distributes tensor computations across multiple devices for parallel inference.
//! Supports row-parallel, column-parallel, and hybrid partitioning strategies
//! with simulated collective operations (all-reduce, scatter, gather).

use std::fmt;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Strategy for partitioning tensors across devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Split along the row (first) dimension.
    RowParallel,
    /// Split along the column (last) dimension.
    ColumnParallel,
    /// Split rows for weight matrices, columns for activations.
    Hybrid,
}

/// Communication backend used for collective operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationBackend {
    /// In-process CPU simulation (default).
    CpuSimulated,
    /// Placeholder for future NCCL support.
    Nccl,
}

/// Reduction operation applied during an all-reduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllReduceOp {
    Sum,
    Mean,
    Max,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for tensor-parallel inference.
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// Number of devices to distribute across (must be ≥ 1).
    pub num_devices: usize,
    /// How tensors are split across devices.
    pub partition_strategy: PartitionStrategy,
    /// Backend used for collective communication.
    pub communication_backend: CommunicationBackend,
}

impl TensorParallelConfig {
    /// Create a new config, returning an error when `num_devices` is zero.
    pub fn new(
        num_devices: usize,
        partition_strategy: PartitionStrategy,
        communication_backend: CommunicationBackend,
    ) -> Result<Self, TensorParallelError> {
        if num_devices == 0 {
            return Err(TensorParallelError::InvalidConfig(
                "num_devices must be at least 1".into(),
            ));
        }
        Ok(Self { num_devices, partition_strategy, communication_backend })
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by tensor-parallel operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorParallelError {
    InvalidConfig(String),
    ShapeError(String),
    ReduceError(String),
}

impl fmt::Display for TensorParallelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::ShapeError(msg) => write!(f, "shape error: {msg}"),
            Self::ReduceError(msg) => write!(f, "reduce error: {msg}"),
        }
    }
}

impl std::error::Error for TensorParallelError {}

// ---------------------------------------------------------------------------
// TensorPartition
// ---------------------------------------------------------------------------

/// A slice of a tensor assigned to a single device.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorPartition {
    /// Zero-based device index.
    pub device_id: usize,
    /// Starting offset along the partitioned dimension.
    pub offset: usize,
    /// Number of elements along the partitioned dimension.
    pub size: usize,
    /// Full shape of this partition (all dimensions).
    pub shape: Vec<usize>,
    /// Flat data buffer (row-major).
    pub data: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Partitioning
// ---------------------------------------------------------------------------

/// Partition a tensor shape across `num_devices` using the given `strategy`.
///
/// Returns one [`TensorPartition`] per device. The partitioned dimension must
/// be evenly divisible by `num_devices`.
pub fn partition_tensor(
    tensor_shape: &[usize],
    num_devices: usize,
    strategy: PartitionStrategy,
) -> Result<Vec<TensorPartition>, TensorParallelError> {
    if tensor_shape.is_empty() {
        return Err(TensorParallelError::ShapeError("tensor shape must not be empty".into()));
    }
    if num_devices == 0 {
        return Err(TensorParallelError::InvalidConfig("num_devices must be at least 1".into()));
    }

    let (split_dim, split_dim_size) = match strategy {
        PartitionStrategy::RowParallel | PartitionStrategy::Hybrid => (0, tensor_shape[0]),
        PartitionStrategy::ColumnParallel => {
            let last = tensor_shape.len() - 1;
            (last, tensor_shape[last])
        }
    };

    if !split_dim_size.is_multiple_of(num_devices) {
        return Err(TensorParallelError::ShapeError(format!(
            "dimension {split_dim} (size {split_dim_size}) is not divisible by {num_devices}"
        )));
    }

    let chunk = split_dim_size / num_devices;

    let partitions = (0..num_devices)
        .map(|i| {
            let mut shape = tensor_shape.to_vec();
            shape[split_dim] = chunk;
            TensorPartition {
                device_id: i,
                offset: i * chunk,
                size: chunk,
                shape,
                data: Vec::new(), // shape-only partition
            }
        })
        .collect();

    Ok(partitions)
}

// ---------------------------------------------------------------------------
// Collective operations
// ---------------------------------------------------------------------------

/// Element-wise all-reduce over partitions.
///
/// Every partition must contain the same number of elements. Returns the
/// reduced buffer.
pub fn all_reduce(
    partitions: &[TensorPartition],
    op: AllReduceOp,
) -> Result<Vec<f32>, TensorParallelError> {
    if partitions.is_empty() {
        return Err(TensorParallelError::ReduceError("no partitions to reduce".into()));
    }
    let len = partitions[0].data.len();
    if len == 0 {
        return Err(TensorParallelError::ReduceError("partition data is empty".into()));
    }
    for (i, p) in partitions.iter().enumerate() {
        if p.data.len() != len {
            return Err(TensorParallelError::ReduceError(format!(
                "partition {i} length {} differs from expected {len}",
                p.data.len()
            )));
        }
    }

    let n = partitions.len() as f32;
    let mut result = vec![0.0_f32; len];

    match op {
        AllReduceOp::Sum => {
            for p in partitions {
                for (r, &v) in result.iter_mut().zip(&p.data) {
                    *r += v;
                }
            }
        }
        AllReduceOp::Mean => {
            for p in partitions {
                for (r, &v) in result.iter_mut().zip(&p.data) {
                    *r += v;
                }
            }
            for r in &mut result {
                *r /= n;
            }
        }
        AllReduceOp::Max => {
            result.copy_from_slice(&partitions[0].data);
            for p in &partitions[1..] {
                for (r, &v) in result.iter_mut().zip(&p.data) {
                    if v > *r {
                        *r = v;
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Scatter `data` into `num_partitions` equal chunks along the first axis.
///
/// `data.len()` must be divisible by `num_partitions`.
pub fn scatter(
    data: &[f32],
    num_partitions: usize,
) -> Result<Vec<TensorPartition>, TensorParallelError> {
    if num_partitions == 0 {
        return Err(TensorParallelError::InvalidConfig("num_partitions must be at least 1".into()));
    }
    if !data.len().is_multiple_of(num_partitions) {
        return Err(TensorParallelError::ShapeError(format!(
            "data length {} is not divisible by {num_partitions}",
            data.len()
        )));
    }

    let chunk_size = data.len() / num_partitions;
    let partitions = data
        .chunks_exact(chunk_size)
        .enumerate()
        .map(|(i, chunk)| TensorPartition {
            device_id: i,
            offset: i * chunk_size,
            size: chunk_size,
            shape: vec![chunk_size],
            data: chunk.to_vec(),
        })
        .collect();

    Ok(partitions)
}

/// Gather partitions back into a single contiguous buffer (ordered by
/// `device_id`).
pub fn gather(partitions: &[TensorPartition]) -> Result<Vec<f32>, TensorParallelError> {
    if partitions.is_empty() {
        return Err(TensorParallelError::ReduceError("no partitions to gather".into()));
    }

    let mut sorted: Vec<&TensorPartition> = partitions.iter().collect();
    sorted.sort_by_key(|p| p.device_id);

    let total_len: usize = sorted.iter().map(|p| p.data.len()).sum();
    let mut result = Vec::with_capacity(total_len);
    for p in sorted {
        result.extend_from_slice(&p.data);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Compute a load-balance score in `[0.0, 1.0]` where 1.0 is perfectly even.
///
/// Defined as `min_size / max_size` across all partitions.
pub fn compute_load_balance(partitions: &[TensorPartition]) -> f64 {
    if partitions.is_empty() {
        return 0.0;
    }
    let sizes: Vec<usize> = partitions.iter().map(|p| p.size).collect();
    let min = *sizes.iter().min().unwrap_or(&0);
    let max = *sizes.iter().max().unwrap_or(&1);
    if max == 0 {
        return 0.0;
    }
    min as f64 / max as f64
}

/// Estimate the communication volume (in elements) for an all-reduce across
/// the given partitions using the ring algorithm: `2 * (N-1)/N * partition_size`.
pub fn communication_volume(partitions: &[TensorPartition]) -> f64 {
    if partitions.len() <= 1 {
        return 0.0;
    }
    let n = partitions.len() as f64;
    let elems = partitions[0].data.len().max(partitions[0].size) as f64;
    2.0 * ((n - 1.0) / n) * elems
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- partitioning -------------------------------------------------------

    #[test]
    fn test_row_parallel_partition() {
        let parts = partition_tensor(&[8, 4], 4, PartitionStrategy::RowParallel).unwrap();
        assert_eq!(parts.len(), 4);
        for (i, p) in parts.iter().enumerate() {
            assert_eq!(p.device_id, i);
            assert_eq!(p.size, 2);
            assert_eq!(p.offset, i * 2);
            assert_eq!(p.shape, vec![2, 4]);
        }
    }

    #[test]
    fn test_column_parallel_partition() {
        let parts = partition_tensor(&[4, 8], 2, PartitionStrategy::ColumnParallel).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape, vec![4, 4]);
        assert_eq!(parts[1].shape, vec![4, 4]);
        assert_eq!(parts[0].offset, 0);
        assert_eq!(parts[1].offset, 4);
    }

    #[test]
    fn test_hybrid_partitions_along_rows() {
        let parts = partition_tensor(&[6, 3], 3, PartitionStrategy::Hybrid).unwrap();
        assert_eq!(parts.len(), 3);
        for p in &parts {
            assert_eq!(p.shape, vec![2, 3]);
        }
    }

    #[test]
    fn test_single_device_is_no_op() {
        let parts = partition_tensor(&[5, 5], 1, PartitionStrategy::RowParallel).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].shape, vec![5, 5]);
        assert_eq!(parts[0].offset, 0);
        assert_eq!(parts[0].size, 5);
    }

    #[test]
    fn test_indivisible_dimension_returns_error() {
        let err = partition_tensor(&[7, 4], 3, PartitionStrategy::RowParallel).unwrap_err();
        assert!(matches!(err, TensorParallelError::ShapeError(_)));
    }

    #[test]
    fn test_zero_devices_returns_error() {
        let err = partition_tensor(&[4, 4], 0, PartitionStrategy::RowParallel).unwrap_err();
        assert!(matches!(err, TensorParallelError::InvalidConfig(_)));
    }

    #[test]
    fn test_empty_shape_returns_error() {
        let err = partition_tensor(&[], 2, PartitionStrategy::RowParallel).unwrap_err();
        assert!(matches!(err, TensorParallelError::ShapeError(_)));
    }

    #[test]
    fn test_1d_tensor_row_partition() {
        let parts = partition_tensor(&[10], 5, PartitionStrategy::RowParallel).unwrap();
        assert_eq!(parts.len(), 5);
        for p in &parts {
            assert_eq!(p.shape, vec![2]);
        }
    }

    #[test]
    fn test_1d_tensor_column_partition() {
        // For a 1-D tensor, last dim == first dim, so Column == Row.
        let parts = partition_tensor(&[10], 5, PartitionStrategy::ColumnParallel).unwrap();
        assert_eq!(parts.len(), 5);
        for p in &parts {
            assert_eq!(p.shape, vec![2]);
        }
    }

    // -- all-reduce ---------------------------------------------------------

    fn make_partition(device_id: usize, data: Vec<f32>) -> TensorPartition {
        let size = data.len();
        TensorPartition { device_id, offset: 0, size, shape: vec![size], data }
    }

    #[test]
    fn test_all_reduce_sum() {
        let parts =
            vec![make_partition(0, vec![1.0, 2.0, 3.0]), make_partition(1, vec![4.0, 5.0, 6.0])];
        let result = all_reduce(&parts, AllReduceOp::Sum).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_all_reduce_mean() {
        let parts = vec![make_partition(0, vec![2.0, 4.0]), make_partition(1, vec![6.0, 8.0])];
        let result = all_reduce(&parts, AllReduceOp::Mean).unwrap();
        assert_eq!(result, vec![4.0, 6.0]);
    }

    #[test]
    fn test_all_reduce_max() {
        let parts =
            vec![make_partition(0, vec![1.0, 9.0, 3.0]), make_partition(1, vec![7.0, 2.0, 8.0])];
        let result = all_reduce(&parts, AllReduceOp::Max).unwrap();
        assert_eq!(result, vec![7.0, 9.0, 8.0]);
    }

    #[test]
    fn test_all_reduce_empty_partitions() {
        let err = all_reduce(&[], AllReduceOp::Sum).unwrap_err();
        assert!(matches!(err, TensorParallelError::ReduceError(_)));
    }

    #[test]
    fn test_all_reduce_empty_data() {
        let parts = vec![make_partition(0, vec![])];
        // data len is 0 but we set it explicitly
        let err = all_reduce(&parts, AllReduceOp::Sum).unwrap_err();
        assert!(matches!(err, TensorParallelError::ReduceError(_)));
    }

    #[test]
    fn test_all_reduce_mismatched_lengths() {
        let parts = vec![make_partition(0, vec![1.0, 2.0]), make_partition(1, vec![3.0])];
        let err = all_reduce(&parts, AllReduceOp::Sum).unwrap_err();
        assert!(matches!(err, TensorParallelError::ReduceError(_)));
    }

    // -- scatter / gather ---------------------------------------------------

    #[test]
    fn test_scatter_gather_roundtrip() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let parts = scatter(&data, 3).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].data, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(parts[1].data, vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(parts[2].data, vec![8.0, 9.0, 10.0, 11.0]);

        let combined = gather(&parts).unwrap();
        assert_eq!(combined, data);
    }

    #[test]
    fn test_scatter_single_partition() {
        let data = vec![1.0, 2.0, 3.0];
        let parts = scatter(&data, 1).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].data, data);
    }

    #[test]
    fn test_scatter_indivisible_returns_error() {
        let err = scatter(&[1.0, 2.0, 3.0], 2).unwrap_err();
        assert!(matches!(err, TensorParallelError::ShapeError(_)));
    }

    #[test]
    fn test_scatter_zero_partitions_returns_error() {
        let err = scatter(&[1.0], 0).unwrap_err();
        assert!(matches!(err, TensorParallelError::InvalidConfig(_)));
    }

    #[test]
    fn test_gather_empty_returns_error() {
        let err = gather(&[]).unwrap_err();
        assert!(matches!(err, TensorParallelError::ReduceError(_)));
    }

    #[test]
    fn test_gather_preserves_device_order() {
        // Supply out-of-order; gather should sort by device_id.
        let parts = vec![
            make_partition(2, vec![5.0, 6.0]),
            make_partition(0, vec![1.0, 2.0]),
            make_partition(1, vec![3.0, 4.0]),
        ];
        let combined = gather(&parts).unwrap();
        assert_eq!(combined, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // -- statistics ---------------------------------------------------------

    #[test]
    fn test_load_balance_even() {
        let parts = partition_tensor(&[8, 4], 4, PartitionStrategy::RowParallel).unwrap();
        let balance = compute_load_balance(&parts);
        assert!((balance - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_load_balance_empty() {
        assert!((compute_load_balance(&[]) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_communication_volume_single_device() {
        let parts = partition_tensor(&[4, 4], 1, PartitionStrategy::RowParallel).unwrap();
        assert!((communication_volume(&parts) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_communication_volume_two_devices() {
        let mut parts = scatter(&[0.0; 8], 2).unwrap();
        // Each partition has 4 elements; ring algo: 2*(1/2)*4 = 4.0
        for p in &mut parts {
            assert_eq!(p.data.len(), 4);
        }
        let vol = communication_volume(&parts);
        assert!((vol - 4.0).abs() < 1e-9);
    }

    // -- config -------------------------------------------------------------

    #[test]
    fn test_config_creation() {
        let cfg = TensorParallelConfig::new(
            4,
            PartitionStrategy::RowParallel,
            CommunicationBackend::CpuSimulated,
        )
        .unwrap();
        assert_eq!(cfg.num_devices, 4);
    }

    #[test]
    fn test_config_zero_devices_error() {
        let err = TensorParallelConfig::new(
            0,
            PartitionStrategy::RowParallel,
            CommunicationBackend::CpuSimulated,
        )
        .unwrap_err();
        assert!(matches!(err, TensorParallelError::InvalidConfig(_)));
    }

    // -- Display for error --------------------------------------------------

    #[test]
    fn test_error_display() {
        let e = TensorParallelError::ShapeError("bad".into());
        assert_eq!(format!("{e}"), "shape error: bad");
    }
}
