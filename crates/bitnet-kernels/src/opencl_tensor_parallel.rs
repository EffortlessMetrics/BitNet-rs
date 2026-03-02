//! Tensor parallelism support for OpenCL (Intel Arc A770).
//!
//! Partitions tensors across compute units for multi-GPU or simulated parallel
//! execution, manages all-reduce / all-gather communication, and supports
//! common model-parallelism patterns (data, tensor, pipeline, expert).
//! All partitioning and communication algorithms have CPU reference
//! implementations so the module compiles and tests unconditionally.

use std::fmt;

// ---------------------------------------------------------------------------
// ParallelStrategy
// ---------------------------------------------------------------------------

/// High-level parallelism strategy applied to a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParallelStrategy {
    /// Replicate the full model; split batches across partitions.
    DataParallel,
    /// Split individual weight tensors across partitions (column / row).
    TensorParallel,
    /// Assign consecutive layers (stages) to consecutive partitions.
    PipelineParallel,
    /// Assign MoE experts to partitions.
    ExpertParallel,
}

impl fmt::Display for ParallelStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataParallel => write!(f, "data_parallel"),
            Self::TensorParallel => write!(f, "tensor_parallel"),
            Self::PipelineParallel => write!(f, "pipeline_parallel"),
            Self::ExpertParallel => write!(f, "expert_parallel"),
        }
    }
}

// ---------------------------------------------------------------------------
// TensorPartition
// ---------------------------------------------------------------------------

/// Describes how a single partition maps onto a contiguous slice of a tensor
/// along a given dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPartition {
    /// Zero-based identifier of this partition.
    pub partition_id: usize,
    /// Total number of partitions across the dimension.
    pub total_partitions: usize,
    /// Dimension along which the tensor is split.
    pub dim: usize,
    /// Start index (inclusive) within the dimension.
    pub start_idx: usize,
    /// End index (exclusive) within the dimension.
    pub end_idx: usize,
}

impl TensorPartition {
    /// Number of elements this partition covers along the split dimension.
    pub fn len(&self) -> usize {
        self.end_idx - self.start_idx
    }

    /// Whether the partition is empty.
    pub fn is_empty(&self) -> bool {
        self.start_idx >= self.end_idx
    }
}

impl fmt::Display for TensorPartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "partition {}/{} dim={} [{}..{})",
            self.partition_id, self.total_partitions, self.dim, self.start_idx, self.end_idx,
        )
    }
}

/// Compute partitions for splitting `dim_size` elements across `num_partitions`.
///
/// When `dim_size` is not evenly divisible the first `dim_size % num_partitions`
/// partitions get one extra element.
pub fn compute_partitions(
    dim_size: usize,
    num_partitions: usize,
    dim: usize,
) -> Vec<TensorPartition> {
    assert!(num_partitions > 0, "num_partitions must be > 0");
    let base = dim_size / num_partitions;
    let remainder = dim_size % num_partitions;
    let mut partitions = Vec::with_capacity(num_partitions);
    let mut offset = 0;
    for i in 0..num_partitions {
        let size = base + if i < remainder { 1 } else { 0 };
        partitions.push(TensorPartition {
            partition_id: i,
            total_partitions: num_partitions,
            dim,
            start_idx: offset,
            end_idx: offset + size,
        });
        offset += size;
    }
    partitions
}

/// Extract a partition's slice from a flat row-major tensor.
///
/// `shape` is the full tensor shape, the partition splits along `partition.dim`.
pub fn extract_partition(data: &[f32], shape: &[usize], partition: &TensorPartition) -> Vec<f32> {
    assert!(!shape.is_empty());
    assert!(partition.dim < shape.len());
    let total: usize = shape.iter().product();
    assert_eq!(data.len(), total, "data length must match shape product");

    // Compute strides (row-major)
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let d = partition.dim;
    let mut out = Vec::new();
    for (flat, &val) in data.iter().enumerate() {
        let coord_d = (flat / strides[d]) % shape[d];
        if coord_d >= partition.start_idx && coord_d < partition.end_idx {
            out.push(val);
        }
    }
    out
}

/// Gather partitions back into a full tensor (inverse of extract).
///
/// `partitions_data` must be ordered by `partition_id` and cover the whole dim.
pub fn gather_partitions(
    partitions_data: &[(&TensorPartition, &[f32])],
    shape: &[usize],
) -> Vec<f32> {
    assert!(!partitions_data.is_empty());
    let total: usize = shape.iter().product();
    let mut out = vec![0.0f32; total];

    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    for &(part, pdata) in partitions_data {
        let d = part.dim;
        let mut write_idx = 0;
        for (flat, slot) in out.iter_mut().enumerate() {
            let coord_d = (flat / strides[d]) % shape[d];
            if coord_d >= part.start_idx && coord_d < part.end_idx {
                *slot = pdata[write_idx];
                write_idx += 1;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CommOperation
// ---------------------------------------------------------------------------

/// Communication primitive for inter-partition data exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommOperation {
    /// Sum (or other reduction) across all partitions.
    AllReduce,
    /// Gather variable-length chunks so every partition has the full tensor.
    AllGather,
    /// Reduce then scatter distinct chunks to each partition.
    ReduceScatter,
    /// One partition broadcasts its data to all others.
    Broadcast,
    /// Point-to-point transfer between two partitions.
    P2P,
}

impl fmt::Display for CommOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllReduce => write!(f, "all_reduce"),
            Self::AllGather => write!(f, "all_gather"),
            Self::ReduceScatter => write!(f, "reduce_scatter"),
            Self::Broadcast => write!(f, "broadcast"),
            Self::P2P => write!(f, "p2p"),
        }
    }
}

// ---------------------------------------------------------------------------
// AllReduceAlgorithm
// ---------------------------------------------------------------------------

/// Algorithm used to implement the all-reduce collective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllReduceAlgorithm {
    /// Ring all-reduce – each partition sends to the next in a ring.
    Ring,
    /// Tree-based reduction followed by broadcast.
    Tree,
    /// Butterfly (recursive halving / doubling).
    Butterfly,
    /// Direct all-to-all – every partition exchanges with every other.
    Direct,
}

impl fmt::Display for AllReduceAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ring => write!(f, "ring"),
            Self::Tree => write!(f, "tree"),
            Self::Butterfly => write!(f, "butterfly"),
            Self::Direct => write!(f, "direct"),
        }
    }
}

// ---------------------------------------------------------------------------
// CommPlan
// ---------------------------------------------------------------------------

/// An ordered plan of communication operations with cost estimates.
#[derive(Debug, Clone)]
pub struct CommPlan {
    /// Ordered sequence of operations.
    pub operations: Vec<CommOperation>,
    /// Estimated bytes transferred across all operations.
    pub estimated_bytes: usize,
    /// Estimated wall-clock latency in microseconds.
    pub estimated_latency_us: f64,
}

impl CommPlan {
    pub fn new() -> Self {
        Self { operations: Vec::new(), estimated_bytes: 0, estimated_latency_us: 0.0 }
    }

    pub fn push(&mut self, op: CommOperation, bytes: usize, latency_us: f64) {
        self.operations.push(op);
        self.estimated_bytes += bytes;
        self.estimated_latency_us += latency_us;
    }

    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

impl Default for CommPlan {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ParallelConfig
// ---------------------------------------------------------------------------

/// Full configuration for a parallel execution session.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Parallelism strategy.
    pub strategy: ParallelStrategy,
    /// Number of partitions (devices / simulated partitions).
    pub num_partitions: usize,
    /// Algorithm used for all-reduce collectives.
    pub allreduce_algorithm: AllReduceAlgorithm,
    /// Whether to overlap communication with computation.
    pub overlap_compute: bool,
}

impl ParallelConfig {
    pub fn new(strategy: ParallelStrategy, num_partitions: usize) -> Self {
        Self {
            strategy,
            num_partitions,
            allreduce_algorithm: AllReduceAlgorithm::Ring,
            overlap_compute: true,
        }
    }
}

// ---------------------------------------------------------------------------
// PartitionPlanner
// ---------------------------------------------------------------------------

/// Decides how to partition model layers across the configured partitions.
#[derive(Debug)]
pub struct PartitionPlanner {
    config: ParallelConfig,
}

impl PartitionPlanner {
    pub fn new(config: ParallelConfig) -> Self {
        Self { config }
    }

    /// Return the `ParallelConfig`.
    pub fn config(&self) -> &ParallelConfig {
        &self.config
    }

    /// Partition a weight tensor of the given `shape` along `dim`.
    pub fn partition_tensor(&self, shape: &[usize], dim: usize) -> Vec<TensorPartition> {
        assert!(dim < shape.len(), "dim out of range");
        compute_partitions(shape[dim], self.config.num_partitions, dim)
    }

    /// Assign transformer layers to pipeline stages.
    ///
    /// Returns a `Vec` of `(stage_id, start_layer, end_layer)`.
    pub fn pipeline_stages(&self, num_layers: usize) -> Vec<(usize, usize, usize)> {
        let n = self.config.num_partitions;
        let base = num_layers / n;
        let rem = num_layers % n;
        let mut stages = Vec::with_capacity(n);
        let mut offset = 0;
        for i in 0..n {
            let count = base + if i < rem { 1 } else { 0 };
            stages.push((i, offset, offset + count));
            offset += count;
        }
        stages
    }

    /// Assign MoE experts to partitions.
    ///
    /// Returns a `Vec` of `(partition_id, Vec<expert_id>)`.
    pub fn expert_assignment(&self, num_experts: usize) -> Vec<(usize, Vec<usize>)> {
        let n = self.config.num_partitions;
        let base = num_experts / n;
        let rem = num_experts % n;
        let mut assignments = Vec::with_capacity(n);
        let mut offset = 0;
        for i in 0..n {
            let count = base + if i < rem { 1 } else { 0 };
            let experts: Vec<usize> = (offset..offset + count).collect();
            assignments.push((i, experts));
            offset += count;
        }
        assignments
    }

    /// Build a communication plan for the current strategy and a given tensor
    /// of `num_elements` f32 values.
    pub fn comm_plan(&self, num_elements: usize) -> CommPlan {
        let bytes = num_elements * std::mem::size_of::<f32>();
        let n = self.config.num_partitions;
        let mut plan = CommPlan::new();

        match self.config.strategy {
            ParallelStrategy::DataParallel => {
                // Gradient all-reduce after backward pass.
                let ar_bytes = bytes * 2 * (n - 1) / n;
                let latency =
                    allreduce_latency_estimate(ar_bytes, n, self.config.allreduce_algorithm);
                plan.push(CommOperation::AllReduce, ar_bytes, latency);
            }
            ParallelStrategy::TensorParallel => {
                // All-reduce after each partitioned matmul.
                let ar_bytes = bytes;
                let latency =
                    allreduce_latency_estimate(ar_bytes, n, self.config.allreduce_algorithm);
                plan.push(CommOperation::AllReduce, ar_bytes, latency);
            }
            ParallelStrategy::PipelineParallel => {
                // P2P between consecutive stages.
                for _ in 0..n.saturating_sub(1) {
                    let stage_bytes = bytes / n;
                    let latency = p2p_latency_estimate(stage_bytes);
                    plan.push(CommOperation::P2P, stage_bytes, latency);
                }
            }
            ParallelStrategy::ExpertParallel => {
                // All-to-all for routing then gather results.
                let scatter_bytes = bytes;
                plan.push(
                    CommOperation::ReduceScatter,
                    scatter_bytes,
                    scatter_bytes as f64 / BW_BYTES_PER_US,
                );
                plan.push(CommOperation::AllGather, bytes, bytes as f64 / BW_BYTES_PER_US);
            }
        }
        plan
    }
}

// ---------------------------------------------------------------------------
// CPU reference communication primitives
// ---------------------------------------------------------------------------

/// CPU reference all-reduce (sum) using the specified algorithm.
///
/// `buffers` is a mutable slice of per-partition buffers, all of equal length.
/// After the call every buffer contains the element-wise sum.
pub fn cpu_allreduce(buffers: &mut [Vec<f32>], algorithm: AllReduceAlgorithm) {
    if buffers.len() <= 1 {
        return;
    }
    let len = buffers[0].len();
    assert!(buffers.iter().all(|b| b.len() == len), "buffers must have equal length");

    match algorithm {
        AllReduceAlgorithm::Ring => cpu_allreduce_ring(buffers),
        AllReduceAlgorithm::Tree => cpu_allreduce_tree(buffers),
        AllReduceAlgorithm::Butterfly => cpu_allreduce_butterfly(buffers),
        AllReduceAlgorithm::Direct => cpu_allreduce_direct(buffers),
    }
}

fn cpu_allreduce_ring(buffers: &mut [Vec<f32>]) {
    // Save originals before any mutation.
    let originals: Vec<Vec<f32>> = buffers.to_vec();
    let len = originals[0].len();
    let mut total = vec![0.0f32; len];
    for buf in &originals {
        for (i, &v) in buf.iter().enumerate() {
            total[i] += v;
        }
    }
    for buf in buffers.iter_mut() {
        buf.copy_from_slice(&total);
    }
}

fn cpu_allreduce_tree(buffers: &mut [Vec<f32>]) {
    // Binary tree reduction to rank 0, then broadcast.
    let len = buffers[0].len();
    let mut total = vec![0.0f32; len];
    cpu_allreduce_direct_inner(buffers, &mut total);

    // Tree structure: at each level, pairs reduce; result stored in lower rank.
    // For CPU reference the final result is the same as direct sum.
    for buf in buffers.iter_mut() {
        buf.copy_from_slice(&total);
    }
}

fn cpu_allreduce_butterfly(buffers: &mut [Vec<f32>]) {
    let n = buffers.len();
    let len = buffers[0].len();

    if n.is_power_of_two() {
        // Recursive halving-doubling: at each step partners exchange and sum.
        let levels = (n as f64).log2().ceil() as u32;
        for level in 0..levels {
            let mask = 1 << level;
            let snapshot: Vec<Vec<f32>> = buffers.to_vec();
            for rank in 0..n {
                let partner = rank ^ mask;
                if partner < n {
                    for j in 0..len {
                        buffers[rank][j] = snapshot[rank][j] + snapshot[partner][j];
                    }
                }
            }
        }
    } else {
        // For non-power-of-two, compute the true sum from originals directly.
        let originals: Vec<Vec<f32>> = buffers.to_vec();
        let mut total = vec![0.0f32; len];
        for buf in &originals {
            for (i, &v) in buf.iter().enumerate() {
                total[i] += v;
            }
        }
        for buf in buffers.iter_mut() {
            buf.copy_from_slice(&total);
        }
    }
}

fn cpu_allreduce_direct(buffers: &mut [Vec<f32>]) {
    let len = buffers[0].len();
    let mut total = vec![0.0f32; len];
    cpu_allreduce_direct_inner(buffers, &mut total);
    for buf in buffers.iter_mut() {
        buf.copy_from_slice(&total);
    }
}

/// Compute element-wise sum of original inputs.
///
/// Computes the element-wise sum of all buffers into `out`.
fn cpu_allreduce_direct_inner(buffers: &[Vec<f32>], out: &mut [f32]) {
    out.fill(0.0);
    for buf in buffers {
        for (i, &v) in buf.iter().enumerate() {
            out[i] += v;
        }
    }
}

/// CPU reference all-gather: concatenate chunks from each partition.
pub fn cpu_allgather(chunks: &[Vec<f32>]) -> Vec<f32> {
    chunks.iter().flat_map(|c| c.iter().copied()).collect()
}

/// CPU reference reduce-scatter (sum then scatter).
///
/// Each partition receives a distinct chunk of the element-wise sum.
pub fn cpu_reduce_scatter(buffers: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = buffers.len();
    if n == 0 {
        return Vec::new();
    }
    let len = buffers[0].len();
    assert!(buffers.iter().all(|b| b.len() == len));

    // Element-wise sum.
    let mut total = vec![0.0f32; len];
    for buf in buffers {
        for (i, &v) in buf.iter().enumerate() {
            total[i] += v;
        }
    }

    // Scatter equal-sized chunks.
    let chunk_size = len.div_ceil(n);
    (0..n)
        .map(|i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(len);
            total[start..end].to_vec()
        })
        .collect()
}

/// CPU reference broadcast: clone `source` to every partition.
pub fn cpu_broadcast(source: &[f32], num_partitions: usize) -> Vec<Vec<f32>> {
    (0..num_partitions).map(|_| source.to_vec()).collect()
}

/// CPU reference point-to-point: just copies data.
pub fn cpu_p2p_transfer(data: &[f32]) -> Vec<f32> {
    data.to_vec()
}

// ---------------------------------------------------------------------------
// Cost model helpers
// ---------------------------------------------------------------------------

/// Assumed bandwidth in bytes per microsecond (≈12.8 GB/s ≈ PCIe 4.0 x16).
const BW_BYTES_PER_US: f64 = 12800.0;
/// Fixed overhead per message in microseconds.
const MSG_OVERHEAD_US: f64 = 5.0;

fn allreduce_latency_estimate(bytes: usize, n: usize, algo: AllReduceAlgorithm) -> f64 {
    let transfer = bytes as f64 / BW_BYTES_PER_US;
    match algo {
        AllReduceAlgorithm::Ring => {
            2.0 * (n as f64 - 1.0) * MSG_OVERHEAD_US + 2.0 * transfer * (n as f64 - 1.0) / n as f64
        }
        AllReduceAlgorithm::Tree => {
            let depth = (n as f64).log2().ceil();
            2.0 * depth * MSG_OVERHEAD_US + 2.0 * transfer
        }
        AllReduceAlgorithm::Butterfly => {
            let depth = (n as f64).log2().ceil();
            depth * MSG_OVERHEAD_US + depth * transfer
        }
        AllReduceAlgorithm::Direct => {
            (n as f64 - 1.0) * MSG_OVERHEAD_US + (n as f64 - 1.0) * transfer
        }
    }
}

fn p2p_latency_estimate(bytes: usize) -> f64 {
    MSG_OVERHEAD_US + bytes as f64 / BW_BYTES_PER_US
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // == ParallelStrategy ==================================================

    #[test]
    fn strategy_display() {
        assert_eq!(ParallelStrategy::DataParallel.to_string(), "data_parallel");
        assert_eq!(ParallelStrategy::TensorParallel.to_string(), "tensor_parallel");
        assert_eq!(ParallelStrategy::PipelineParallel.to_string(), "pipeline_parallel");
        assert_eq!(ParallelStrategy::ExpertParallel.to_string(), "expert_parallel");
    }

    #[test]
    fn strategy_equality() {
        assert_eq!(ParallelStrategy::DataParallel, ParallelStrategy::DataParallel);
        assert_ne!(ParallelStrategy::DataParallel, ParallelStrategy::TensorParallel);
    }

    // == TensorPartition ====================================================

    #[test]
    fn partition_even_split() {
        let parts = compute_partitions(12, 3, 0);
        assert_eq!(parts.len(), 3);
        for (i, p) in parts.iter().enumerate() {
            assert_eq!(p.partition_id, i);
            assert_eq!(p.total_partitions, 3);
            assert_eq!(p.len(), 4);
        }
        assert_eq!(parts[0].start_idx, 0);
        assert_eq!(parts[0].end_idx, 4);
        assert_eq!(parts[2].start_idx, 8);
        assert_eq!(parts[2].end_idx, 12);
    }

    #[test]
    fn partition_uneven_split() {
        let parts = compute_partitions(10, 3, 1);
        // 10 / 3 = 3 rem 1 → sizes: 4, 3, 3
        assert_eq!(parts[0].len(), 4);
        assert_eq!(parts[1].len(), 3);
        assert_eq!(parts[2].len(), 3);
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn partition_single() {
        let parts = compute_partitions(7, 1, 0);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].start_idx, 0);
        assert_eq!(parts[0].end_idx, 7);
    }

    #[test]
    fn partition_more_partitions_than_elements() {
        let parts = compute_partitions(3, 5, 0);
        assert_eq!(parts.len(), 5);
        let non_empty: Vec<_> = parts.iter().filter(|p| !p.is_empty()).collect();
        assert_eq!(non_empty.len(), 3);
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn partition_prime_size() {
        let parts = compute_partitions(13, 4, 0);
        // 13 / 4 = 3 rem 1 → sizes: 4, 3, 3, 3
        assert_eq!(parts[0].len(), 4);
        for p in &parts[1..] {
            assert_eq!(p.len(), 3);
        }
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 13);
    }

    #[test]
    fn partition_display() {
        let p = TensorPartition {
            partition_id: 1,
            total_partitions: 4,
            dim: 0,
            start_idx: 3,
            end_idx: 6,
        };
        assert!(p.to_string().contains("1/4"));
    }

    #[test]
    fn partition_dim0_extract_1d() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = [12];
        let parts = compute_partitions(12, 3, 0);

        let chunk0 = extract_partition(&data, &shape, &parts[0]);
        assert_eq!(chunk0, vec![0.0, 1.0, 2.0, 3.0]);
        let chunk2 = extract_partition(&data, &shape, &parts[2]);
        assert_eq!(chunk2, vec![8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn partition_dim0_extract_2d() {
        // 4×3 matrix split along rows (dim 0)
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = [4, 3];
        let parts = compute_partitions(4, 2, 0);

        let top = extract_partition(&data, &shape, &parts[0]);
        assert_eq!(top, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]); // rows 0,1
        let bot = extract_partition(&data, &shape, &parts[1]);
        assert_eq!(bot, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]); // rows 2,3
    }

    #[test]
    fn partition_dim1_extract_2d() {
        // 3×4 matrix split along columns (dim 1)
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = [3, 4];
        let parts = compute_partitions(4, 2, 1);

        let left = extract_partition(&data, &shape, &parts[0]);
        // columns 0,1: [0,1], [4,5], [8,9]
        assert_eq!(left, vec![0.0, 1.0, 4.0, 5.0, 8.0, 9.0]);
    }

    #[test]
    fn partition_gather_roundtrip_1d() {
        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let shape = [10];
        let parts = compute_partitions(10, 3, 0);
        let chunks: Vec<Vec<f32>> =
            parts.iter().map(|p| extract_partition(&data, &shape, p)).collect();
        let refs: Vec<(&TensorPartition, &[f32])> =
            parts.iter().zip(chunks.iter().map(|c| c.as_slice())).collect();
        let restored = gather_partitions(&refs, &shape);
        assert_eq!(restored, data);
    }

    #[test]
    fn partition_gather_roundtrip_2d() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shape = [3, 4];
        let parts = compute_partitions(4, 2, 1);
        let chunks: Vec<Vec<f32>> =
            parts.iter().map(|p| extract_partition(&data, &shape, p)).collect();
        let refs: Vec<(&TensorPartition, &[f32])> =
            parts.iter().zip(chunks.iter().map(|c| c.as_slice())).collect();
        let restored = gather_partitions(&refs, &shape);
        assert_eq!(restored, data);
    }

    #[test]
    fn partition_gather_roundtrip_prime() {
        let data: Vec<f32> = (0..13).map(|x| x as f32).collect();
        let shape = [13];
        let parts = compute_partitions(13, 4, 0);
        let chunks: Vec<Vec<f32>> =
            parts.iter().map(|p| extract_partition(&data, &shape, p)).collect();
        let refs: Vec<(&TensorPartition, &[f32])> =
            parts.iter().zip(chunks.iter().map(|c| c.as_slice())).collect();
        let restored = gather_partitions(&refs, &shape);
        assert_eq!(restored, data);
    }

    // == CommOperation ======================================================

    #[test]
    fn comm_op_display() {
        assert_eq!(CommOperation::AllReduce.to_string(), "all_reduce");
        assert_eq!(CommOperation::AllGather.to_string(), "all_gather");
        assert_eq!(CommOperation::ReduceScatter.to_string(), "reduce_scatter");
        assert_eq!(CommOperation::Broadcast.to_string(), "broadcast");
        assert_eq!(CommOperation::P2P.to_string(), "p2p");
    }

    // == AllReduceAlgorithm =================================================

    #[test]
    fn allreduce_algo_display() {
        assert_eq!(AllReduceAlgorithm::Ring.to_string(), "ring");
        assert_eq!(AllReduceAlgorithm::Tree.to_string(), "tree");
        assert_eq!(AllReduceAlgorithm::Butterfly.to_string(), "butterfly");
        assert_eq!(AllReduceAlgorithm::Direct.to_string(), "direct");
    }

    // == All-reduce correctness =============================================

    fn make_buffers(n: usize, len: usize) -> Vec<Vec<f32>> {
        (0..n).map(|i| (0..len).map(|j| (i * len + j) as f32).collect()).collect()
    }

    fn expected_sum(buffers: &[Vec<f32>]) -> Vec<f32> {
        let len = buffers[0].len();
        let mut sum = vec![0.0f32; len];
        for buf in buffers {
            for (i, &v) in buf.iter().enumerate() {
                sum[i] += v;
            }
        }
        sum
    }

    #[test]
    fn allreduce_direct_correctness() {
        let mut bufs = make_buffers(4, 8);
        let expected = expected_sum(&bufs);
        cpu_allreduce(&mut bufs, AllReduceAlgorithm::Direct);
        for buf in &bufs {
            assert_eq!(buf, &expected);
        }
    }

    #[test]
    fn allreduce_ring_correctness() {
        let orig = make_buffers(4, 8);
        let expected = expected_sum(&orig);
        let mut bufs = orig;
        cpu_allreduce(&mut bufs, AllReduceAlgorithm::Ring);
        for buf in &bufs {
            assert_eq!(buf, &expected);
        }
    }

    #[test]
    fn allreduce_tree_correctness() {
        let orig = make_buffers(4, 8);
        let expected = expected_sum(&orig);
        let mut bufs = orig;
        cpu_allreduce(&mut bufs, AllReduceAlgorithm::Tree);
        for buf in &bufs {
            assert_eq!(buf, &expected);
        }
    }

    #[test]
    fn allreduce_butterfly_correctness() {
        let orig = make_buffers(4, 8);
        let expected = expected_sum(&orig);
        let mut bufs = orig;
        cpu_allreduce(&mut bufs, AllReduceAlgorithm::Butterfly);
        for buf in &bufs {
            assert_eq!(buf, &expected);
        }
    }

    #[test]
    fn allreduce_butterfly_non_power_of_two() {
        let orig = make_buffers(3, 6);
        let expected = expected_sum(&orig);
        let mut bufs = orig;
        cpu_allreduce(&mut bufs, AllReduceAlgorithm::Butterfly);
        for buf in &bufs {
            assert_eq!(buf, &expected);
        }
    }

    #[test]
    fn allreduce_single_partition() {
        let mut bufs = vec![vec![1.0, 2.0, 3.0]];
        cpu_allreduce(&mut bufs, AllReduceAlgorithm::Direct);
        assert_eq!(bufs[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn allreduce_two_partitions() {
        let mut bufs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let expected = vec![4.0, 6.0];
        cpu_allreduce(&mut bufs, AllReduceAlgorithm::Ring);
        assert_eq!(bufs[0], expected);
        assert_eq!(bufs[1], expected);
    }

    // == All-gather =========================================================

    #[test]
    fn allgather_basic() {
        let chunks = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = cpu_allgather(&chunks);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn allgather_single() {
        let chunks = vec![vec![10.0]];
        assert_eq!(cpu_allgather(&chunks), vec![10.0]);
    }

    #[test]
    fn allgather_empty_chunks() {
        let chunks: Vec<Vec<f32>> = vec![vec![], vec![]];
        assert!(cpu_allgather(&chunks).is_empty());
    }

    // == Reduce-scatter =====================================================

    #[test]
    fn reduce_scatter_basic() {
        let bufs = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let result = cpu_reduce_scatter(&bufs);
        // Sum = [6, 8, 10, 12], chunk_size = 2
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![6.0, 8.0]);
        assert_eq!(result[1], vec![10.0, 12.0]);
    }

    #[test]
    fn reduce_scatter_uneven() {
        let bufs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
        let result = cpu_reduce_scatter(&bufs);
        // Sum = [12, 15, 18], chunk_size = 1
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![12.0]);
        assert_eq!(result[1], vec![15.0]);
        assert_eq!(result[2], vec![18.0]);
    }

    #[test]
    fn reduce_scatter_empty() {
        let bufs: Vec<Vec<f32>> = Vec::new();
        assert!(cpu_reduce_scatter(&bufs).is_empty());
    }

    // == Broadcast ==========================================================

    #[test]
    fn broadcast_basic() {
        let src = vec![1.0, 2.0, 3.0];
        let result = cpu_broadcast(&src, 3);
        assert_eq!(result.len(), 3);
        for chunk in &result {
            assert_eq!(chunk, &src);
        }
    }

    #[test]
    fn broadcast_single() {
        let src = vec![42.0];
        let result = cpu_broadcast(&src, 1);
        assert_eq!(result, vec![vec![42.0]]);
    }

    // == P2P transfer =======================================================

    #[test]
    fn p2p_transfer_copies() {
        let data = vec![1.0, 2.0, 3.0];
        let result = cpu_p2p_transfer(&data);
        assert_eq!(result, data);
    }

    // == CommPlan ===========================================================

    #[test]
    fn comm_plan_empty() {
        let plan = CommPlan::new();
        assert!(plan.is_empty());
        assert_eq!(plan.estimated_bytes, 0);
    }

    #[test]
    fn comm_plan_accumulates() {
        let mut plan = CommPlan::new();
        plan.push(CommOperation::AllReduce, 1024, 10.0);
        plan.push(CommOperation::Broadcast, 512, 5.0);
        assert_eq!(plan.operations.len(), 2);
        assert_eq!(plan.estimated_bytes, 1536);
        assert!((plan.estimated_latency_us - 15.0).abs() < 1e-6);
    }

    // == ParallelConfig =====================================================

    #[test]
    fn config_defaults() {
        let cfg = ParallelConfig::new(ParallelStrategy::DataParallel, 4);
        assert_eq!(cfg.strategy, ParallelStrategy::DataParallel);
        assert_eq!(cfg.num_partitions, 4);
        assert!(cfg.overlap_compute);
        assert_eq!(cfg.allreduce_algorithm, AllReduceAlgorithm::Ring);
    }

    // == PartitionPlanner ===================================================

    #[test]
    fn planner_partition_tensor() {
        let cfg = ParallelConfig::new(ParallelStrategy::TensorParallel, 4);
        let planner = PartitionPlanner::new(cfg);
        let parts = planner.partition_tensor(&[32, 64], 1);
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0].len(), 16);
    }

    #[test]
    fn planner_pipeline_stages() {
        let cfg = ParallelConfig::new(ParallelStrategy::PipelineParallel, 3);
        let planner = PartitionPlanner::new(cfg);
        let stages = planner.pipeline_stages(10);
        assert_eq!(stages.len(), 3);
        // 10/3 = 3 rem 1 → 4, 3, 3
        assert_eq!(stages[0], (0, 0, 4));
        assert_eq!(stages[1], (1, 4, 7));
        assert_eq!(stages[2], (2, 7, 10));
    }

    #[test]
    fn planner_pipeline_stages_even() {
        let cfg = ParallelConfig::new(ParallelStrategy::PipelineParallel, 4);
        let planner = PartitionPlanner::new(cfg);
        let stages = planner.pipeline_stages(12);
        for (i, &(id, s, e)) in stages.iter().enumerate() {
            assert_eq!(id, i);
            assert_eq!(e - s, 3);
        }
    }

    #[test]
    fn planner_expert_assignment() {
        let cfg = ParallelConfig::new(ParallelStrategy::ExpertParallel, 3);
        let planner = PartitionPlanner::new(cfg);
        let assignments = planner.expert_assignment(8);
        assert_eq!(assignments.len(), 3);
        // 8/3 = 2 rem 2 → 3, 3, 2
        assert_eq!(assignments[0].1.len(), 3);
        assert_eq!(assignments[1].1.len(), 3);
        assert_eq!(assignments[2].1.len(), 2);
        let all_experts: Vec<usize> =
            assignments.iter().flat_map(|(_, e)| e.iter().copied()).collect();
        assert_eq!(all_experts, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn planner_expert_assignment_single() {
        let cfg = ParallelConfig::new(ParallelStrategy::ExpertParallel, 1);
        let planner = PartitionPlanner::new(cfg);
        let assignments = planner.expert_assignment(4);
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].1, vec![0, 1, 2, 3]);
    }

    #[test]
    fn planner_comm_plan_data_parallel() {
        let cfg = ParallelConfig::new(ParallelStrategy::DataParallel, 4);
        let planner = PartitionPlanner::new(cfg);
        let plan = planner.comm_plan(1024);
        assert_eq!(plan.operations.len(), 1);
        assert_eq!(plan.operations[0], CommOperation::AllReduce);
        assert!(plan.estimated_bytes > 0);
        assert!(plan.estimated_latency_us > 0.0);
    }

    #[test]
    fn planner_comm_plan_tensor_parallel() {
        let cfg = ParallelConfig::new(ParallelStrategy::TensorParallel, 2);
        let planner = PartitionPlanner::new(cfg);
        let plan = planner.comm_plan(512);
        assert_eq!(plan.operations, vec![CommOperation::AllReduce]);
    }

    #[test]
    fn planner_comm_plan_pipeline_parallel() {
        let cfg = ParallelConfig::new(ParallelStrategy::PipelineParallel, 4);
        let planner = PartitionPlanner::new(cfg);
        let plan = planner.comm_plan(256);
        assert_eq!(plan.operations.len(), 3); // n-1 P2P
        assert!(plan.operations.iter().all(|op| *op == CommOperation::P2P));
    }

    #[test]
    fn planner_comm_plan_expert_parallel() {
        let cfg = ParallelConfig::new(ParallelStrategy::ExpertParallel, 2);
        let planner = PartitionPlanner::new(cfg);
        let plan = planner.comm_plan(128);
        assert_eq!(plan.operations.len(), 2);
        assert_eq!(plan.operations[0], CommOperation::ReduceScatter);
        assert_eq!(plan.operations[1], CommOperation::AllGather);
    }

    // == Data-parallel gradient aggregation =================================

    #[test]
    fn data_parallel_gradient_aggregation() {
        // Simulate 4 workers each with local gradients.
        let grads: Vec<Vec<f32>> =
            (0..4).map(|w| (0..8).map(|j| (w * 8 + j) as f32 * 0.01).collect()).collect();
        let expected = expected_sum(&grads);
        let mut bufs = grads;
        cpu_allreduce(&mut bufs, AllReduceAlgorithm::Direct);
        for buf in &bufs {
            for (a, b) in buf.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    // == Tensor-parallel weight splitting ===================================

    #[test]
    fn tensor_parallel_weight_split_and_gather() {
        // 4×8 weight matrix split along columns (dim=1) across 2 partitions.
        let weight: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let shape = [4, 8];
        let parts = compute_partitions(8, 2, 1);
        let halves: Vec<Vec<f32>> =
            parts.iter().map(|p| extract_partition(&weight, &shape, p)).collect();
        assert_eq!(halves[0].len(), 16);
        assert_eq!(halves[1].len(), 16);

        let refs: Vec<(&TensorPartition, &[f32])> =
            parts.iter().zip(halves.iter().map(|h| h.as_slice())).collect();
        let restored = gather_partitions(&refs, &shape);
        assert_eq!(restored, weight);
    }

    // == Edge cases =========================================================

    #[test]
    fn partition_zero_size_dim() {
        let parts = compute_partitions(0, 3, 0);
        assert_eq!(parts.len(), 3);
        for p in &parts {
            assert!(p.is_empty());
        }
    }

    #[test]
    fn planner_config_accessor() {
        let cfg = ParallelConfig::new(ParallelStrategy::DataParallel, 8);
        let planner = PartitionPlanner::new(cfg);
        assert_eq!(planner.config().num_partitions, 8);
    }

    // == Cost model sanity ==================================================

    #[test]
    fn latency_increases_with_partitions() {
        let l2 = allreduce_latency_estimate(4096, 2, AllReduceAlgorithm::Ring);
        let l8 = allreduce_latency_estimate(4096, 8, AllReduceAlgorithm::Ring);
        assert!(l8 > l2);
    }

    #[test]
    fn latency_increases_with_bytes() {
        let small = allreduce_latency_estimate(1024, 4, AllReduceAlgorithm::Direct);
        let large = allreduce_latency_estimate(1_048_576, 4, AllReduceAlgorithm::Direct);
        assert!(large > small);
    }

    #[test]
    fn p2p_latency_positive() {
        assert!(p2p_latency_estimate(1024) > 0.0);
    }
}
