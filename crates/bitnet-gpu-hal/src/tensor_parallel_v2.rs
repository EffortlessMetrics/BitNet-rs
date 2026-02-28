//! Advanced tensor parallelism with all-reduce and scatter-gather for multi-GPU inference.
//!
//! Provides column-parallel and row-parallel linear layers, vocabulary-partitioned
//! embeddings, collective communication primitives (all-reduce, all-gather,
//! reduce-scatter, broadcast), and communication profiling.

use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Communication backend for collective operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommBackend {
    /// NVIDIA NCCL – optimised GPU-to-GPU.
    Nccl,
    /// Gloo – CPU-based collectives.
    Gloo,
    /// MPI – general-purpose.
    Mpi,
    /// Shared-memory (single-node).
    SharedMemory,
    /// In-process mock for testing.
    Mock,
}

impl fmt::Display for CommBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nccl => write!(f, "NCCL"),
            Self::Gloo => write!(f, "Gloo"),
            Self::Mpi => write!(f, "MPI"),
            Self::SharedMemory => write!(f, "SharedMemory"),
            Self::Mock => write!(f, "Mock"),
        }
    }
}

/// Tensor-parallelism configuration.
#[derive(Debug, Clone)]
pub struct TPConfig {
    /// Total number of ranks (devices).
    pub world_size: usize,
    /// This rank's index (0-based).
    pub rank: usize,
    /// Dimension along which tensors are partitioned.
    pub partition_dim: usize,
    /// Communication backend.
    pub communication_backend: CommBackend,
    /// Whether to overlap computation with communication.
    pub overlap_compute_comm: bool,
}

impl TPConfig {
    /// Create a new configuration.
    ///
    /// # Errors
    /// Returns `Err` if `world_size` is 0 or `rank >= world_size`.
    pub fn new(
        world_size: usize,
        rank: usize,
        partition_dim: usize,
        communication_backend: CommBackend,
        overlap_compute_comm: bool,
    ) -> Result<Self, TPError> {
        if world_size == 0 {
            return Err(TPError::InvalidConfig("world_size must be > 0".into()));
        }
        if rank >= world_size {
            return Err(TPError::InvalidConfig(format!(
                "rank {rank} must be < world_size {world_size}"
            )));
        }
        Ok(Self { world_size, rank, partition_dim, communication_backend, overlap_compute_comm })
    }

    /// Single-rank convenience constructor (no-op parallelism).
    pub fn single() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            partition_dim: 0,
            communication_backend: CommBackend::Mock,
            overlap_compute_comm: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by tensor-parallelism operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TPError {
    /// Invalid configuration.
    InvalidConfig(String),
    /// Dimension mismatch during partitioning.
    DimensionMismatch { expected: usize, got: usize },
    /// Unsupported collective operation.
    UnsupportedOp(String),
    /// Communication failure.
    CommFailure(String),
}

impl fmt::Display for TPError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid TP config: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::UnsupportedOp(msg) => write!(f, "unsupported op: {msg}"),
            Self::CommFailure(msg) => write!(f, "communication failure: {msg}"),
        }
    }
}

impl std::error::Error for TPError {}

// ---------------------------------------------------------------------------
// Partition strategy
// ---------------------------------------------------------------------------

/// How a tensor is distributed across ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionStrategy {
    /// Split along the output (column) dimension – each rank computes a slice of the output.
    ColumnParallel,
    /// Split along the input (row) dimension – each rank holds a slice of the weight rows.
    RowParallel,
    /// Expert parallelism (MoE) – each rank owns a subset of experts.
    ExpertParallel,
    /// Fully replicated – every rank holds the full tensor.
    ReplicatedParallel,
}

/// Metadata describing how a single tensor is partitioned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPartition {
    /// Strategy used.
    pub strategy: PartitionStrategy,
    /// Total number of partitions (== world_size for the group).
    pub num_partitions: usize,
    /// Size of each partition along the split dimension.
    pub partition_sizes: Vec<usize>,
    /// Index of the local partition on this rank.
    pub local_partition_idx: usize,
}

impl TensorPartition {
    /// Create an even partition of `total_size` across `num_partitions`.
    ///
    /// When `total_size` is not evenly divisible the last partition absorbs the remainder.
    pub fn even(
        strategy: PartitionStrategy,
        total_size: usize,
        num_partitions: usize,
        local_idx: usize,
    ) -> Result<Self, TPError> {
        if num_partitions == 0 {
            return Err(TPError::InvalidConfig("num_partitions must be > 0".into()));
        }
        if local_idx >= num_partitions {
            return Err(TPError::InvalidConfig(format!(
                "local_idx {local_idx} >= num_partitions {num_partitions}"
            )));
        }
        let base = total_size / num_partitions;
        let remainder = total_size % num_partitions;
        let mut sizes = vec![base; num_partitions];
        if remainder > 0 {
            // distribute remainder across the first `remainder` partitions
            for s in sizes.iter_mut().take(remainder) {
                *s += 1;
            }
        }
        Ok(Self {
            strategy,
            num_partitions,
            partition_sizes: sizes,
            local_partition_idx: local_idx,
        })
    }

    /// Total size across all partitions.
    pub fn total_size(&self) -> usize {
        self.partition_sizes.iter().sum()
    }

    /// Size of the local partition.
    pub fn local_size(&self) -> usize {
        self.partition_sizes[self.local_partition_idx]
    }

    /// Byte offset of the local partition (assuming contiguous layout).
    pub fn local_offset(&self) -> usize {
        self.partition_sizes[..self.local_partition_idx].iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Collective operations
// ---------------------------------------------------------------------------

/// Reduction operator for all-reduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

/// Mock collective-operations executor.
///
/// In a real deployment these would call into NCCL / MPI / etc.
/// This mock version operates on in-process `Vec<f32>` buffers to
/// enable comprehensive testing without hardware dependencies.
#[derive(Debug)]
pub struct CollectiveOps {
    config: TPConfig,
}

impl CollectiveOps {
    pub fn new(config: TPConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &TPConfig {
        &self.config
    }

    /// Simulated all-reduce across `per_rank_buffers`.
    ///
    /// Every rank ends up with the same reduced result.
    pub fn all_reduce(
        &self,
        per_rank_buffers: &mut [Vec<f32>],
        op: AllReduceOp,
    ) -> Result<(), TPError> {
        if per_rank_buffers.len() != self.config.world_size {
            return Err(TPError::DimensionMismatch {
                expected: self.config.world_size,
                got: per_rank_buffers.len(),
            });
        }
        let len = per_rank_buffers[0].len();
        for buf in per_rank_buffers.iter() {
            if buf.len() != len {
                return Err(TPError::DimensionMismatch { expected: len, got: buf.len() });
            }
        }

        let reduced: Vec<f32> = (0..len)
            .map(|i| {
                let vals: Vec<f32> = per_rank_buffers.iter().map(|b| b[i]).collect();
                match op {
                    AllReduceOp::Sum => vals.iter().sum(),
                    AllReduceOp::Mean => vals.iter().sum::<f32>() / vals.len() as f32,
                    AllReduceOp::Max => vals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    AllReduceOp::Min => vals.iter().copied().fold(f32::INFINITY, f32::min),
                }
            })
            .collect();

        for buf in per_rank_buffers.iter_mut() {
            buf.clone_from(&reduced);
        }
        Ok(())
    }

    /// Simulated all-gather: concatenate each rank's chunk into every rank's buffer.
    pub fn all_gather(&self, per_rank_chunks: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, TPError> {
        if per_rank_chunks.len() != self.config.world_size {
            return Err(TPError::DimensionMismatch {
                expected: self.config.world_size,
                got: per_rank_chunks.len(),
            });
        }
        let gathered: Vec<f32> = per_rank_chunks.iter().flat_map(|c| c.iter().copied()).collect();
        Ok(vec![gathered; self.config.world_size])
    }

    /// Simulated reduce-scatter: reduce element-wise, then scatter chunks.
    pub fn reduce_scatter(
        &self,
        per_rank_buffers: &[Vec<f32>],
        op: AllReduceOp,
    ) -> Result<Vec<Vec<f32>>, TPError> {
        if per_rank_buffers.len() != self.config.world_size {
            return Err(TPError::DimensionMismatch {
                expected: self.config.world_size,
                got: per_rank_buffers.len(),
            });
        }
        let len = per_rank_buffers[0].len();
        for buf in per_rank_buffers.iter() {
            if buf.len() != len {
                return Err(TPError::DimensionMismatch { expected: len, got: buf.len() });
            }
        }
        if len % self.config.world_size != 0 {
            return Err(TPError::DimensionMismatch {
                expected: len,
                got: len % self.config.world_size,
            });
        }

        // reduce
        let reduced: Vec<f32> = (0..len)
            .map(|i| {
                let vals: Vec<f32> = per_rank_buffers.iter().map(|b| b[i]).collect();
                match op {
                    AllReduceOp::Sum => vals.iter().sum(),
                    AllReduceOp::Mean => vals.iter().sum::<f32>() / vals.len() as f32,
                    AllReduceOp::Max => vals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    AllReduceOp::Min => vals.iter().copied().fold(f32::INFINITY, f32::min),
                }
            })
            .collect();

        // scatter
        let chunk_size = len / self.config.world_size;
        let scattered: Vec<Vec<f32>> = (0..self.config.world_size)
            .map(|r| reduced[r * chunk_size..(r + 1) * chunk_size].to_vec())
            .collect();
        Ok(scattered)
    }

    /// Simulated broadcast from `root` to all ranks.
    pub fn broadcast(&self, per_rank_buffers: &mut [Vec<f32>], root: usize) -> Result<(), TPError> {
        if root >= self.config.world_size {
            return Err(TPError::InvalidConfig(format!(
                "broadcast root {root} >= world_size {}",
                self.config.world_size
            )));
        }
        if per_rank_buffers.len() != self.config.world_size {
            return Err(TPError::DimensionMismatch {
                expected: self.config.world_size,
                got: per_rank_buffers.len(),
            });
        }
        let src = per_rank_buffers[root].clone();
        for buf in per_rank_buffers.iter_mut() {
            buf.clone_from(&src);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Scatter / Gather manager
// ---------------------------------------------------------------------------

/// Manages scatter (split) and gather (concat) of tensor data across ranks.
#[derive(Debug)]
pub struct ScatterGatherManager {
    world_size: usize,
}

impl ScatterGatherManager {
    pub fn new(world_size: usize) -> Result<Self, TPError> {
        if world_size == 0 {
            return Err(TPError::InvalidConfig("world_size must be > 0".into()));
        }
        Ok(Self { world_size })
    }

    /// Scatter `data` into `world_size` chunks. Last chunk absorbs remainder.
    pub fn scatter(&self, data: &[f32]) -> Vec<Vec<f32>> {
        if self.world_size == 1 {
            return vec![data.to_vec()];
        }
        let base = data.len() / self.world_size;
        let remainder = data.len() % self.world_size;
        let mut chunks = Vec::with_capacity(self.world_size);
        let mut offset = 0;
        for r in 0..self.world_size {
            let size = if r < remainder { base + 1 } else { base };
            chunks.push(data[offset..offset + size].to_vec());
            offset += size;
        }
        chunks
    }

    /// Gather chunks back into a single contiguous buffer.
    pub fn gather(&self, chunks: &[Vec<f32>]) -> Result<Vec<f32>, TPError> {
        if chunks.len() != self.world_size {
            return Err(TPError::DimensionMismatch {
                expected: self.world_size,
                got: chunks.len(),
            });
        }
        Ok(chunks.iter().flat_map(|c| c.iter().copied()).collect())
    }

    /// Round-trip: scatter then gather. Useful for verifying lossless partitioning.
    pub fn scatter_gather_round_trip(&self, data: &[f32]) -> Result<Vec<f32>, TPError> {
        let chunks = self.scatter(data);
        self.gather(&chunks)
    }
}

// ---------------------------------------------------------------------------
// Tensor-parallel linear layer
// ---------------------------------------------------------------------------

/// A linear layer that is partitioned across ranks.
///
/// * **Column-parallel**: weight is split along the output dimension.
///   Each rank computes `y_local = x @ W_local` (a slice of the full output).
///   An all-gather reconstructs the full output.
///
/// * **Row-parallel**: weight is split along the input dimension.
///   Each rank computes a partial sum. An all-reduce produces the full output.
#[derive(Debug, Clone)]
pub struct TPLinear {
    /// Local weight shard – stored as a flat row-major matrix.
    pub weight: Vec<f32>,
    /// Optional bias (only on rank 0 for row-parallel, full on column-parallel).
    pub bias: Option<Vec<f32>>,
    /// Input features of the *full* layer.
    pub in_features: usize,
    /// Output features of the *full* layer.
    pub out_features: usize,
    /// Partition metadata.
    pub partition: TensorPartition,
}

impl TPLinear {
    /// Create a column-parallel linear layer for a single rank.
    pub fn column_parallel(
        full_weight: &[f32],
        full_bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        world_size: usize,
        rank: usize,
    ) -> Result<Self, TPError> {
        let partition = TensorPartition::even(
            PartitionStrategy::ColumnParallel,
            out_features,
            world_size,
            rank,
        )?;
        let local_out = partition.local_size();
        let col_offset = partition.local_offset();

        // Slice columns [col_offset .. col_offset + local_out] from each row.
        let mut weight = Vec::with_capacity(in_features * local_out);
        for row in 0..in_features {
            let row_start = row * out_features;
            weight.extend_from_slice(
                &full_weight[row_start + col_offset..row_start + col_offset + local_out],
            );
        }

        let bias = full_bias.map(|b| b[col_offset..col_offset + local_out].to_vec());

        Ok(Self { weight, bias, in_features, out_features, partition })
    }

    /// Create a row-parallel linear layer for a single rank.
    pub fn row_parallel(
        full_weight: &[f32],
        full_bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        world_size: usize,
        rank: usize,
    ) -> Result<Self, TPError> {
        let partition =
            TensorPartition::even(PartitionStrategy::RowParallel, in_features, world_size, rank)?;
        let local_in = partition.local_size();
        let row_offset = partition.local_offset();

        // Slice rows [row_offset .. row_offset + local_in].
        let weight =
            full_weight[row_offset * out_features..(row_offset + local_in) * out_features].to_vec();

        // Bias is only applied once after the all-reduce (rank 0 holds it).
        let bias = if rank == 0 { full_bias.map(|b| b.to_vec()) } else { None };

        Ok(Self { weight, bias, in_features, out_features, partition })
    }

    /// Local forward: `y_local = x_local @ W_local + bias_local`.
    ///
    /// `input` has shape `[batch, local_in]` (row-parallel) or `[batch, in_features]` (column-parallel).
    /// Returns `[batch, local_out]` (column-parallel) or `[batch, out_features]` (row-parallel).
    pub fn forward_local(&self, input: &[f32], batch_size: usize) -> Result<Vec<f32>, TPError> {
        let (rows_in, cols_out) = match self.partition.strategy {
            PartitionStrategy::ColumnParallel => (self.in_features, self.partition.local_size()),
            PartitionStrategy::RowParallel => (self.partition.local_size(), self.out_features),
            other => {
                return Err(TPError::UnsupportedOp(format!(
                    "{other:?} not supported for TPLinear"
                )));
            }
        };

        let expected_input_len = batch_size * rows_in;
        if input.len() != expected_input_len {
            return Err(TPError::DimensionMismatch {
                expected: expected_input_len,
                got: input.len(),
            });
        }

        let mut output = vec![0.0f32; batch_size * cols_out];
        for b in 0..batch_size {
            for j in 0..cols_out {
                let mut sum = 0.0f32;
                for k in 0..rows_in {
                    sum += input[b * rows_in + k] * self.weight[k * cols_out + j];
                }
                output[b * cols_out + j] = sum;
            }
        }

        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for j in 0..cols_out {
                    output[b * cols_out + j] += bias[j];
                }
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Tensor-parallel embedding
// ---------------------------------------------------------------------------

/// Embedding table partitioned across ranks along the vocabulary dimension.
///
/// Each rank holds rows `[vocab_offset .. vocab_offset + local_vocab_size]`.
#[derive(Debug, Clone)]
pub struct TPEmbedding {
    /// Local embedding table – shape `[local_vocab_size, embed_dim]`.
    pub table: Vec<f32>,
    /// Full vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Partition metadata.
    pub partition: TensorPartition,
}

impl TPEmbedding {
    /// Create a partition of the full embedding table for a single rank.
    pub fn new(
        full_table: &[f32],
        vocab_size: usize,
        embed_dim: usize,
        world_size: usize,
        rank: usize,
    ) -> Result<Self, TPError> {
        if full_table.len() != vocab_size * embed_dim {
            return Err(TPError::DimensionMismatch {
                expected: vocab_size * embed_dim,
                got: full_table.len(),
            });
        }
        let partition =
            TensorPartition::even(PartitionStrategy::ColumnParallel, vocab_size, world_size, rank)?;
        let offset = partition.local_offset();
        let local_vocab = partition.local_size();
        let table = full_table[offset * embed_dim..(offset + local_vocab) * embed_dim].to_vec();
        Ok(Self { table, vocab_size, embed_dim, partition })
    }

    /// Look up a token id, returning the embedding vector if the id falls on this rank.
    ///
    /// Returns `None` if the token is owned by another rank (caller should
    /// query the correct rank or rely on all-reduce to combine).
    pub fn lookup(&self, token_id: usize) -> Option<Vec<f32>> {
        let offset = self.partition.local_offset();
        let local_vocab = self.partition.local_size();
        if token_id >= offset && token_id < offset + local_vocab {
            let local_id = token_id - offset;
            Some(self.table[local_id * self.embed_dim..(local_id + 1) * self.embed_dim].to_vec())
        } else {
            None
        }
    }

    /// Local vocabulary size on this rank.
    pub fn local_vocab_size(&self) -> usize {
        self.partition.local_size()
    }
}

// ---------------------------------------------------------------------------
// Communication profiler
// ---------------------------------------------------------------------------

/// A single profiled communication event.
#[derive(Debug, Clone)]
pub struct CommEvent {
    pub op_name: String,
    pub bytes_transferred: usize,
    pub duration: Duration,
}

/// Profiles communication overhead.
#[derive(Debug, Default)]
pub struct TPCommunicationProfiler {
    events: Vec<CommEvent>,
}

impl TPCommunicationProfiler {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Record a communication event.
    pub fn record(&mut self, op_name: impl Into<String>, bytes: usize, duration: Duration) {
        self.events.push(CommEvent { op_name: op_name.into(), bytes_transferred: bytes, duration });
    }

    /// Start a timer. Returns an opaque [`Instant`].
    pub fn start_timer(&self) -> Instant {
        Instant::now()
    }

    /// Finish timing and record.
    pub fn finish_timer(&mut self, start: Instant, op_name: impl Into<String>, bytes: usize) {
        let duration = start.elapsed();
        self.record(op_name, bytes, duration);
    }

    /// Total bytes transferred across all recorded events.
    pub fn total_bytes(&self) -> usize {
        self.events.iter().map(|e| e.bytes_transferred).sum()
    }

    /// Total communication time.
    pub fn total_duration(&self) -> Duration {
        self.events.iter().map(|e| e.duration).sum()
    }

    /// Number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Average bandwidth in bytes/sec. Returns `None` if no events recorded.
    pub fn average_bandwidth(&self) -> Option<f64> {
        let secs = self.total_duration().as_secs_f64();
        if secs == 0.0 || self.events.is_empty() {
            return None;
        }
        Some(self.total_bytes() as f64 / secs)
    }

    /// Immutable access to all events.
    pub fn events(&self) -> &[CommEvent] {
        &self.events
    }

    /// Clear all recorded events.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

// ---------------------------------------------------------------------------
// Tensor-parallel engine
// ---------------------------------------------------------------------------

/// Top-level engine that partitions a model and orchestrates parallel execution.
#[derive(Debug)]
pub struct TensorParallelEngine {
    config: TPConfig,
    collective: CollectiveOps,
    scatter_gather: ScatterGatherManager,
    profiler: TPCommunicationProfiler,
}

impl TensorParallelEngine {
    /// Create a new engine from a configuration.
    pub fn new(config: TPConfig) -> Result<Self, TPError> {
        let scatter_gather = ScatterGatherManager::new(config.world_size)?;
        let collective = CollectiveOps::new(config.clone());
        Ok(Self { config, collective, scatter_gather, profiler: TPCommunicationProfiler::new() })
    }

    pub fn config(&self) -> &TPConfig {
        &self.config
    }

    pub fn collective(&self) -> &CollectiveOps {
        &self.collective
    }

    pub fn scatter_gather(&self) -> &ScatterGatherManager {
        &self.scatter_gather
    }

    pub fn profiler(&self) -> &TPCommunicationProfiler {
        &self.profiler
    }

    pub fn profiler_mut(&mut self) -> &mut TPCommunicationProfiler {
        &mut self.profiler
    }

    /// Partition a 1-D tensor (e.g. bias) across ranks.
    pub fn partition_tensor(&self, data: &[f32]) -> Vec<Vec<f32>> {
        self.scatter_gather.scatter(data)
    }

    /// Execute an all-reduce across simulated per-rank buffers and profile it.
    pub fn execute_all_reduce(
        &mut self,
        buffers: &mut [Vec<f32>],
        op: AllReduceOp,
    ) -> Result<(), TPError> {
        let bytes = buffers.iter().map(|b| b.len() * 4).sum::<usize>();
        let start = self.profiler.start_timer();
        self.collective.all_reduce(buffers, op)?;
        self.profiler.finish_timer(start, "all_reduce", bytes);
        Ok(())
    }

    /// Execute an all-gather and profile it.
    pub fn execute_all_gather(&mut self, chunks: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, TPError> {
        let bytes = chunks.iter().map(|c| c.len() * 4).sum::<usize>();
        let start = self.profiler.start_timer();
        let result = self.collective.all_gather(chunks)?;
        self.profiler.finish_timer(start, "all_gather", bytes);
        Ok(result)
    }

    /// Execute a reduce-scatter and profile it.
    pub fn execute_reduce_scatter(
        &mut self,
        buffers: &[Vec<f32>],
        op: AllReduceOp,
    ) -> Result<Vec<Vec<f32>>, TPError> {
        let bytes = buffers.iter().map(|b| b.len() * 4).sum::<usize>();
        let start = self.profiler.start_timer();
        let result = self.collective.reduce_scatter(buffers, op)?;
        self.profiler.finish_timer(start, "reduce_scatter", bytes);
        Ok(result)
    }

    /// Execute a broadcast from `root` and profile it.
    pub fn execute_broadcast(
        &mut self,
        buffers: &mut [Vec<f32>],
        root: usize,
    ) -> Result<(), TPError> {
        let bytes = buffers.iter().map(|b| b.len() * 4).sum::<usize>();
        let start = self.profiler.start_timer();
        self.collective.broadcast(buffers, root)?;
        self.profiler.finish_timer(start, "broadcast", bytes);
        Ok(())
    }

    /// Column-parallel forward: each rank computes a slice, then all-gather.
    pub fn column_parallel_forward(
        &mut self,
        layers: &[TPLinear],
        input: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, TPError> {
        if layers.len() != self.config.world_size {
            return Err(TPError::DimensionMismatch {
                expected: self.config.world_size,
                got: layers.len(),
            });
        }
        let local_outputs: Vec<Vec<f32>> = layers
            .iter()
            .map(|l| l.forward_local(input, batch_size))
            .collect::<Result<Vec<_>, _>>()?;
        let gathered = self.execute_all_gather(&local_outputs)?;
        // All ranks get the same gathered result.
        Ok(gathered.into_iter().next().unwrap_or_default())
    }

    /// Row-parallel forward: each rank computes a partial sum, then all-reduce.
    pub fn row_parallel_forward(
        &mut self,
        layers: &[TPLinear],
        per_rank_inputs: &[Vec<f32>],
        batch_size: usize,
    ) -> Result<Vec<f32>, TPError> {
        if layers.len() != self.config.world_size {
            return Err(TPError::DimensionMismatch {
                expected: self.config.world_size,
                got: layers.len(),
            });
        }
        if per_rank_inputs.len() != self.config.world_size {
            return Err(TPError::DimensionMismatch {
                expected: self.config.world_size,
                got: per_rank_inputs.len(),
            });
        }

        let mut partial_outputs: Vec<Vec<f32>> = layers
            .iter()
            .zip(per_rank_inputs.iter())
            .map(|(l, inp)| l.forward_local(inp, batch_size))
            .collect::<Result<Vec<_>, _>>()?;

        self.execute_all_reduce(&mut partial_outputs, AllReduceOp::Sum)?;
        Ok(partial_outputs.into_iter().next().unwrap_or_default())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // TPConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_valid_construction() {
        let cfg = TPConfig::new(4, 2, 0, CommBackend::Mock, false).unwrap();
        assert_eq!(cfg.world_size, 4);
        assert_eq!(cfg.rank, 2);
    }

    #[test]
    fn config_single_rank() {
        let cfg = TPConfig::single();
        assert_eq!(cfg.world_size, 1);
        assert_eq!(cfg.rank, 0);
    }

    #[test]
    fn config_world_size_zero_rejected() {
        assert!(TPConfig::new(0, 0, 0, CommBackend::Mock, false).is_err());
    }

    #[test]
    fn config_rank_out_of_range_rejected() {
        assert!(TPConfig::new(2, 2, 0, CommBackend::Mock, false).is_err());
    }

    #[test]
    fn config_rank_equals_world_size_rejected() {
        let err = TPConfig::new(4, 4, 0, CommBackend::Nccl, false).unwrap_err();
        assert!(matches!(err, TPError::InvalidConfig(_)));
    }

    #[test]
    fn config_rank_zero_is_always_valid() {
        assert!(TPConfig::new(1, 0, 0, CommBackend::Mock, false).is_ok());
    }

    #[test]
    fn config_overlap_flag_preserved() {
        let cfg = TPConfig::new(2, 0, 1, CommBackend::Gloo, true).unwrap();
        assert!(cfg.overlap_compute_comm);
        assert_eq!(cfg.partition_dim, 1);
    }

    #[test]
    fn config_all_backends() {
        for backend in [
            CommBackend::Nccl,
            CommBackend::Gloo,
            CommBackend::Mpi,
            CommBackend::SharedMemory,
            CommBackend::Mock,
        ] {
            let cfg = TPConfig::new(2, 0, 0, backend, false).unwrap();
            assert_eq!(cfg.communication_backend, backend);
        }
    }

    #[test]
    fn comm_backend_display() {
        assert_eq!(format!("{}", CommBackend::Nccl), "NCCL");
        assert_eq!(format!("{}", CommBackend::Mock), "Mock");
    }

    #[test]
    fn tp_error_display() {
        let e = TPError::InvalidConfig("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    // -----------------------------------------------------------------------
    // TensorPartition tests
    // -----------------------------------------------------------------------

    #[test]
    fn partition_even_split() {
        let p = TensorPartition::even(PartitionStrategy::ColumnParallel, 16, 4, 0).unwrap();
        assert_eq!(p.partition_sizes, vec![4, 4, 4, 4]);
        assert_eq!(p.total_size(), 16);
    }

    #[test]
    fn partition_uneven_split_remainder_distributed() {
        let p = TensorPartition::even(PartitionStrategy::ColumnParallel, 10, 3, 0).unwrap();
        // 10 / 3 = 3 remainder 1 → sizes [4, 3, 3]
        assert_eq!(p.partition_sizes, vec![4, 3, 3]);
        assert_eq!(p.total_size(), 10);
    }

    #[test]
    fn partition_local_size_and_offset() {
        let p = TensorPartition::even(PartitionStrategy::RowParallel, 12, 3, 1).unwrap();
        assert_eq!(p.local_size(), 4);
        assert_eq!(p.local_offset(), 4);
    }

    #[test]
    fn partition_local_offset_rank_zero() {
        let p = TensorPartition::even(PartitionStrategy::ColumnParallel, 8, 2, 0).unwrap();
        assert_eq!(p.local_offset(), 0);
    }

    #[test]
    fn partition_local_offset_last_rank() {
        let p = TensorPartition::even(PartitionStrategy::ColumnParallel, 8, 2, 1).unwrap();
        assert_eq!(p.local_offset(), 4);
        assert_eq!(p.local_size(), 4);
    }

    #[test]
    fn partition_zero_partitions_rejected() {
        assert!(TensorPartition::even(PartitionStrategy::ColumnParallel, 8, 0, 0).is_err());
    }

    #[test]
    fn partition_idx_out_of_range_rejected() {
        assert!(TensorPartition::even(PartitionStrategy::ColumnParallel, 8, 2, 2).is_err());
    }

    #[test]
    fn partition_single_partition_is_whole() {
        let p = TensorPartition::even(PartitionStrategy::ColumnParallel, 100, 1, 0).unwrap();
        assert_eq!(p.partition_sizes, vec![100]);
        assert_eq!(p.local_size(), 100);
    }

    #[test]
    fn partition_total_equals_original_for_all_world_sizes() {
        for ws in 1..=8 {
            for total in [1, 7, 16, 33, 100, 255] {
                let p =
                    TensorPartition::even(PartitionStrategy::ColumnParallel, total, ws, 0).unwrap();
                assert_eq!(p.total_size(), total, "world_size={ws}, total={total}");
            }
        }
    }

    #[test]
    fn partition_sizes_sum_invariant() {
        for ws in 1..=6 {
            for total in [0, 1, 5, 13, 64] {
                let p =
                    TensorPartition::even(PartitionStrategy::RowParallel, total, ws, 0).unwrap();
                let sum: usize = p.partition_sizes.iter().sum();
                assert_eq!(sum, total);
            }
        }
    }

    #[test]
    fn partition_each_rank_covers_disjoint_range() {
        let total = 17;
        let ws = 4;
        let mut covered = vec![false; total];
        for r in 0..ws {
            let p = TensorPartition::even(PartitionStrategy::ColumnParallel, total, ws, r).unwrap();
            let off = p.local_offset();
            for i in off..off + p.local_size() {
                assert!(!covered[i], "overlap at index {i}");
                covered[i] = true;
            }
        }
        assert!(covered.iter().all(|&c| c));
    }

    #[test]
    fn partition_strategy_variants_exist() {
        let _c = PartitionStrategy::ColumnParallel;
        let _r = PartitionStrategy::RowParallel;
        let _e = PartitionStrategy::ExpertParallel;
        let _p = PartitionStrategy::ReplicatedParallel;
    }

    // -----------------------------------------------------------------------
    // CollectiveOps – all-reduce
    // -----------------------------------------------------------------------

    fn make_collective(world_size: usize) -> CollectiveOps {
        let cfg = TPConfig::new(world_size, 0, 0, CommBackend::Mock, false).unwrap();
        CollectiveOps::new(cfg)
    }

    #[test]
    fn all_reduce_sum_two_ranks() {
        let coll = make_collective(2);
        let mut bufs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        coll.all_reduce(&mut bufs, AllReduceOp::Sum).unwrap();
        assert_eq!(bufs[0], vec![4.0, 6.0]);
        assert_eq!(bufs[1], vec![4.0, 6.0]);
    }

    #[test]
    fn all_reduce_mean() {
        let coll = make_collective(2);
        let mut bufs = vec![vec![2.0, 6.0], vec![4.0, 8.0]];
        coll.all_reduce(&mut bufs, AllReduceOp::Mean).unwrap();
        assert_eq!(bufs[0], vec![3.0, 7.0]);
    }

    #[test]
    fn all_reduce_max() {
        let coll = make_collective(3);
        let mut bufs = vec![vec![1.0, 5.0], vec![3.0, 2.0], vec![2.0, 9.0]];
        coll.all_reduce(&mut bufs, AllReduceOp::Max).unwrap();
        assert_eq!(bufs[0], vec![3.0, 9.0]);
    }

    #[test]
    fn all_reduce_min() {
        let coll = make_collective(3);
        let mut bufs = vec![vec![1.0, 5.0], vec![3.0, 2.0], vec![2.0, 9.0]];
        coll.all_reduce(&mut bufs, AllReduceOp::Min).unwrap();
        assert_eq!(bufs[0], vec![1.0, 2.0]);
    }

    #[test]
    fn all_reduce_single_rank_is_identity() {
        let coll = make_collective(1);
        let mut bufs = vec![vec![42.0, 7.0]];
        coll.all_reduce(&mut bufs, AllReduceOp::Sum).unwrap();
        assert_eq!(bufs[0], vec![42.0, 7.0]);
    }

    #[test]
    fn all_reduce_wrong_buffer_count() {
        let coll = make_collective(2);
        let mut bufs = vec![vec![1.0]];
        assert!(coll.all_reduce(&mut bufs, AllReduceOp::Sum).is_err());
    }

    #[test]
    fn all_reduce_mismatched_lengths() {
        let coll = make_collective(2);
        let mut bufs = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(coll.all_reduce(&mut bufs, AllReduceOp::Sum).is_err());
    }

    #[test]
    fn all_reduce_sum_four_ranks() {
        let coll = make_collective(4);
        let mut bufs = vec![vec![1.0]; 4];
        coll.all_reduce(&mut bufs, AllReduceOp::Sum).unwrap();
        assert_eq!(bufs[0], vec![4.0]);
    }

    #[test]
    fn all_reduce_empty_buffers() {
        let coll = make_collective(2);
        let mut bufs = vec![vec![], vec![]];
        coll.all_reduce(&mut bufs, AllReduceOp::Sum).unwrap();
        assert!(bufs[0].is_empty());
    }

    // -----------------------------------------------------------------------
    // CollectiveOps – all-gather
    // -----------------------------------------------------------------------

    #[test]
    fn all_gather_two_ranks() {
        let coll = make_collective(2);
        let chunks = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let gathered = coll.all_gather(&chunks).unwrap();
        assert_eq!(gathered[0], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(gathered[1], vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn all_gather_single_rank() {
        let coll = make_collective(1);
        let chunks = vec![vec![5.0]];
        let gathered = coll.all_gather(&chunks).unwrap();
        assert_eq!(gathered[0], vec![5.0]);
    }

    #[test]
    fn all_gather_wrong_count() {
        let coll = make_collective(2);
        let chunks = vec![vec![1.0]];
        assert!(coll.all_gather(&chunks).is_err());
    }

    // -----------------------------------------------------------------------
    // CollectiveOps – reduce-scatter
    // -----------------------------------------------------------------------

    #[test]
    fn reduce_scatter_sum_two_ranks() {
        let coll = make_collective(2);
        let bufs = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let scattered = coll.reduce_scatter(&bufs, AllReduceOp::Sum).unwrap();
        // reduced = [6, 8, 10, 12], chunk0 = [6, 8], chunk1 = [10, 12]
        assert_eq!(scattered[0], vec![6.0, 8.0]);
        assert_eq!(scattered[1], vec![10.0, 12.0]);
    }

    #[test]
    fn reduce_scatter_single_rank() {
        let coll = make_collective(1);
        let bufs = vec![vec![10.0, 20.0]];
        let scattered = coll.reduce_scatter(&bufs, AllReduceOp::Sum).unwrap();
        assert_eq!(scattered[0], vec![10.0, 20.0]);
    }

    #[test]
    fn reduce_scatter_indivisible_length_errors() {
        let coll = make_collective(2);
        let bufs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        assert!(coll.reduce_scatter(&bufs, AllReduceOp::Sum).is_err());
    }

    // -----------------------------------------------------------------------
    // CollectiveOps – broadcast
    // -----------------------------------------------------------------------

    #[test]
    fn broadcast_from_root_zero() {
        let coll = make_collective(3);
        let mut bufs = vec![vec![1.0, 2.0], vec![0.0, 0.0], vec![0.0, 0.0]];
        coll.broadcast(&mut bufs, 0).unwrap();
        assert_eq!(bufs[1], vec![1.0, 2.0]);
        assert_eq!(bufs[2], vec![1.0, 2.0]);
    }

    #[test]
    fn broadcast_from_non_zero_root() {
        let coll = make_collective(2);
        let mut bufs = vec![vec![0.0], vec![99.0]];
        coll.broadcast(&mut bufs, 1).unwrap();
        assert_eq!(bufs[0], vec![99.0]);
    }

    #[test]
    fn broadcast_invalid_root() {
        let coll = make_collective(2);
        let mut bufs = vec![vec![1.0], vec![2.0]];
        assert!(coll.broadcast(&mut bufs, 5).is_err());
    }

    // -----------------------------------------------------------------------
    // ScatterGatherManager
    // -----------------------------------------------------------------------

    #[test]
    fn scatter_even() {
        let sg = ScatterGatherManager::new(2).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let chunks = sg.scatter(&data);
        assert_eq!(chunks, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn scatter_uneven() {
        let sg = ScatterGatherManager::new(3).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let chunks = sg.scatter(&data);
        // 5/3 = 1 rem 2 → sizes [2, 2, 1]
        assert_eq!(chunks[0], vec![1.0, 2.0]);
        assert_eq!(chunks[1], vec![3.0, 4.0]);
        assert_eq!(chunks[2], vec![5.0]);
    }

    #[test]
    fn scatter_single_rank() {
        let sg = ScatterGatherManager::new(1).unwrap();
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(sg.scatter(&data), vec![vec![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn scatter_empty_data() {
        let sg = ScatterGatherManager::new(3).unwrap();
        let chunks = sg.scatter(&[]);
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|c| c.is_empty()));
    }

    #[test]
    fn gather_round_trip() {
        let sg = ScatterGatherManager::new(4).unwrap();
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let result = sg.scatter_gather_round_trip(&data).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn gather_round_trip_uneven() {
        let sg = ScatterGatherManager::new(3).unwrap();
        let data: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let result = sg.scatter_gather_round_trip(&data).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn gather_wrong_chunk_count() {
        let sg = ScatterGatherManager::new(2).unwrap();
        assert!(sg.gather(&[vec![1.0]]).is_err());
    }

    #[test]
    fn scatter_gather_manager_zero_world_size() {
        assert!(ScatterGatherManager::new(0).is_err());
    }

    // -----------------------------------------------------------------------
    // TPLinear – column parallel
    // -----------------------------------------------------------------------

    fn identity_weight(n: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        w
    }

    #[test]
    fn column_parallel_identity_forward() {
        let n = 4;
        let w = identity_weight(n);
        let layers: Vec<TPLinear> =
            (0..2).map(|r| TPLinear::column_parallel(&w, None, n, n, 2, r).unwrap()).collect();

        let input = vec![1.0, 2.0, 3.0, 4.0];
        // Each rank produces half of the output columns.
        let y0 = layers[0].forward_local(&input, 1).unwrap();
        let y1 = layers[1].forward_local(&input, 1).unwrap();
        // Concatenated should equal the input (identity).
        let mut full: Vec<f32> = y0;
        full.extend(y1);
        assert_eq!(full, input);
    }

    #[test]
    fn column_parallel_with_bias() {
        let w = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        let bias = vec![10.0, 20.0];
        let layer = TPLinear::column_parallel(&w, Some(&bias), 2, 2, 1, 0).unwrap();
        let y = layer.forward_local(&[1.0, 2.0], 1).unwrap();
        assert_eq!(y, vec![11.0, 22.0]);
    }

    #[test]
    fn column_parallel_batch_size_two() {
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let layer = TPLinear::column_parallel(&w, None, 2, 2, 1, 0).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0]; // batch=2
        let y = layer.forward_local(&input, 2).unwrap();
        assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn column_parallel_wrong_input_length() {
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let layer = TPLinear::column_parallel(&w, None, 2, 2, 1, 0).unwrap();
        assert!(layer.forward_local(&[1.0], 1).is_err());
    }

    #[test]
    fn column_parallel_partition_metadata() {
        let w = vec![0.0; 8 * 4];
        let layer = TPLinear::column_parallel(&w, None, 8, 4, 2, 1).unwrap();
        assert_eq!(layer.partition.strategy, PartitionStrategy::ColumnParallel);
        assert_eq!(layer.partition.num_partitions, 2);
        assert_eq!(layer.partition.local_partition_idx, 1);
    }

    // -----------------------------------------------------------------------
    // TPLinear – row parallel
    // -----------------------------------------------------------------------

    #[test]
    fn row_parallel_identity_forward() {
        let n = 4;
        let w = identity_weight(n);
        let layers: Vec<TPLinear> =
            (0..2).map(|r| TPLinear::row_parallel(&w, None, n, n, 2, r).unwrap()).collect();

        // Each rank gets half of the input rows.
        let input_r0 = vec![1.0, 2.0]; // first 2 elements
        let input_r1 = vec![3.0, 4.0]; // last 2 elements

        let y0 = layers[0].forward_local(&input_r0, 1).unwrap();
        let y1 = layers[1].forward_local(&input_r1, 1).unwrap();

        // Sum of partial results should give the full output.
        let sum: Vec<f32> = y0.iter().zip(y1.iter()).map(|(a, b)| a + b).collect();
        assert_eq!(sum, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn row_parallel_bias_only_on_rank_zero() {
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![5.0, 5.0];
        let l0 = TPLinear::row_parallel(&w, Some(&bias), 2, 2, 2, 0).unwrap();
        let l1 = TPLinear::row_parallel(&w, Some(&bias), 2, 2, 2, 1).unwrap();
        assert!(l0.bias.is_some());
        assert!(l1.bias.is_none());
    }

    #[test]
    fn row_parallel_partition_metadata() {
        let w = vec![0.0; 4 * 8];
        let layer = TPLinear::row_parallel(&w, None, 4, 8, 2, 0).unwrap();
        assert_eq!(layer.partition.strategy, PartitionStrategy::RowParallel);
    }

    // -----------------------------------------------------------------------
    // TPEmbedding
    // -----------------------------------------------------------------------

    #[test]
    fn embedding_partition_correct_sizes() {
        let vocab = 10;
        let dim = 4;
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let e0 = TPEmbedding::new(&table, vocab, dim, 2, 0).unwrap();
        let e1 = TPEmbedding::new(&table, vocab, dim, 2, 1).unwrap();
        assert_eq!(e0.local_vocab_size(), 5);
        assert_eq!(e1.local_vocab_size(), 5);
    }

    #[test]
    fn embedding_lookup_on_correct_rank() {
        let vocab = 6;
        let dim = 2;
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let e0 = TPEmbedding::new(&table, vocab, dim, 2, 0).unwrap();
        let e1 = TPEmbedding::new(&table, vocab, dim, 2, 1).unwrap();

        // Token 1 should be on rank 0 (owns tokens 0-2).
        assert!(e0.lookup(1).is_some());
        assert!(e1.lookup(1).is_none());

        // Token 4 should be on rank 1 (owns tokens 3-5).
        assert!(e0.lookup(4).is_none());
        assert!(e1.lookup(4).is_some());
    }

    #[test]
    fn embedding_lookup_value_correct() {
        let vocab = 4;
        let dim = 3;
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let e = TPEmbedding::new(&table, vocab, dim, 1, 0).unwrap();
        assert_eq!(e.lookup(2).unwrap(), vec![6.0, 7.0, 8.0]);
    }

    #[test]
    fn embedding_out_of_range_returns_none() {
        let table = vec![0.0; 12];
        let e = TPEmbedding::new(&table, 4, 3, 1, 0).unwrap();
        assert!(e.lookup(4).is_none());
    }

    #[test]
    fn embedding_wrong_table_size_rejected() {
        assert!(TPEmbedding::new(&[0.0; 5], 4, 2, 1, 0).is_err());
    }

    #[test]
    fn embedding_single_rank_owns_all() {
        let vocab = 8;
        let dim = 2;
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let e = TPEmbedding::new(&table, vocab, dim, 1, 0).unwrap();
        for t in 0..vocab {
            assert!(e.lookup(t).is_some());
        }
    }

    #[test]
    fn embedding_every_token_owned_by_exactly_one_rank() {
        let vocab = 11;
        let dim = 2;
        let ws = 3;
        let table: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let embeddings: Vec<TPEmbedding> =
            (0..ws).map(|r| TPEmbedding::new(&table, vocab, dim, ws, r).unwrap()).collect();

        for t in 0..vocab {
            let owners: Vec<usize> = embeddings
                .iter()
                .enumerate()
                .filter(|(_, e)| e.lookup(t).is_some())
                .map(|(r, _)| r)
                .collect();
            assert_eq!(owners.len(), 1, "token {t} owned by {owners:?}");
        }
    }

    // -----------------------------------------------------------------------
    // TPCommunicationProfiler
    // -----------------------------------------------------------------------

    #[test]
    fn profiler_initially_empty() {
        let p = TPCommunicationProfiler::new();
        assert_eq!(p.event_count(), 0);
        assert_eq!(p.total_bytes(), 0);
    }

    #[test]
    fn profiler_records_events() {
        let mut p = TPCommunicationProfiler::new();
        p.record("all_reduce", 1024, Duration::from_millis(5));
        p.record("broadcast", 512, Duration::from_millis(3));
        assert_eq!(p.event_count(), 2);
        assert_eq!(p.total_bytes(), 1536);
    }

    #[test]
    fn profiler_total_duration() {
        let mut p = TPCommunicationProfiler::new();
        p.record("op1", 100, Duration::from_millis(10));
        p.record("op2", 200, Duration::from_millis(20));
        assert_eq!(p.total_duration(), Duration::from_millis(30));
    }

    #[test]
    fn profiler_average_bandwidth() {
        let mut p = TPCommunicationProfiler::new();
        p.record("op", 1_000_000, Duration::from_secs(1));
        let bw = p.average_bandwidth().unwrap();
        assert!((bw - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn profiler_no_events_bandwidth_none() {
        let p = TPCommunicationProfiler::new();
        assert!(p.average_bandwidth().is_none());
    }

    #[test]
    fn profiler_clear() {
        let mut p = TPCommunicationProfiler::new();
        p.record("x", 100, Duration::from_millis(1));
        p.clear();
        assert_eq!(p.event_count(), 0);
    }

    #[test]
    fn profiler_events_accessor() {
        let mut p = TPCommunicationProfiler::new();
        p.record("a", 10, Duration::from_millis(1));
        assert_eq!(p.events()[0].op_name, "a");
    }

    #[test]
    fn profiler_timer_round_trip() {
        let mut p = TPCommunicationProfiler::new();
        let start = p.start_timer();
        // Minimal work to ensure non-zero duration possibility.
        std::hint::black_box(42);
        p.finish_timer(start, "timed_op", 256);
        assert_eq!(p.event_count(), 1);
        assert_eq!(p.total_bytes(), 256);
    }

    // -----------------------------------------------------------------------
    // TensorParallelEngine
    // -----------------------------------------------------------------------

    fn make_engine(world_size: usize) -> TensorParallelEngine {
        let cfg = TPConfig::new(world_size, 0, 0, CommBackend::Mock, false).unwrap();
        TensorParallelEngine::new(cfg).unwrap()
    }

    #[test]
    fn engine_creation() {
        let engine = make_engine(4);
        assert_eq!(engine.config().world_size, 4);
    }

    #[test]
    fn engine_partition_tensor() {
        let engine = make_engine(2);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let parts = engine.partition_tensor(&data);
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn engine_all_reduce_profiled() {
        let mut engine = make_engine(2);
        let mut bufs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        engine.execute_all_reduce(&mut bufs, AllReduceOp::Sum).unwrap();
        assert_eq!(bufs[0], vec![4.0, 6.0]);
        assert_eq!(engine.profiler().event_count(), 1);
    }

    #[test]
    fn engine_all_gather_profiled() {
        let mut engine = make_engine(2);
        let chunks = vec![vec![1.0], vec![2.0]];
        let gathered = engine.execute_all_gather(&chunks).unwrap();
        assert_eq!(gathered[0], vec![1.0, 2.0]);
        assert_eq!(engine.profiler().event_count(), 1);
    }

    #[test]
    fn engine_reduce_scatter_profiled() {
        let mut engine = make_engine(2);
        let bufs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let scattered = engine.execute_reduce_scatter(&bufs, AllReduceOp::Sum).unwrap();
        assert_eq!(scattered[0], vec![4.0]);
        assert_eq!(scattered[1], vec![6.0]);
        assert_eq!(engine.profiler().event_count(), 1);
    }

    #[test]
    fn engine_broadcast_profiled() {
        let mut engine = make_engine(2);
        let mut bufs = vec![vec![7.0], vec![0.0]];
        engine.execute_broadcast(&mut bufs, 0).unwrap();
        assert_eq!(bufs[1], vec![7.0]);
        assert_eq!(engine.profiler().event_count(), 1);
    }

    #[test]
    fn engine_column_parallel_forward_identity() {
        let mut engine = make_engine(2);
        let n = 4;
        let w = identity_weight(n);
        let layers: Vec<TPLinear> =
            (0..2).map(|r| TPLinear::column_parallel(&w, None, n, n, 2, r).unwrap()).collect();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = engine.column_parallel_forward(&layers, &input, 1).unwrap();
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn engine_row_parallel_forward_identity() {
        let mut engine = make_engine(2);
        let n = 4;
        let w = identity_weight(n);
        let layers: Vec<TPLinear> =
            (0..2).map(|r| TPLinear::row_parallel(&w, None, n, n, 2, r).unwrap()).collect();
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let output = engine.row_parallel_forward(&layers, &inputs, 1).unwrap();
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn engine_column_forward_wrong_layer_count() {
        let mut engine = make_engine(2);
        let layers = vec![];
        assert!(engine.column_parallel_forward(&layers, &[1.0], 1).is_err());
    }

    #[test]
    fn engine_row_forward_wrong_input_count() {
        let mut engine = make_engine(2);
        let w = identity_weight(2);
        let layers: Vec<TPLinear> =
            (0..2).map(|r| TPLinear::row_parallel(&w, None, 2, 2, 2, r).unwrap()).collect();
        assert!(engine.row_parallel_forward(&layers, &[vec![1.0]], 1).is_err());
    }

    #[test]
    fn engine_profiler_accumulates() {
        let mut engine = make_engine(2);
        let mut bufs = vec![vec![1.0], vec![2.0]];
        engine.execute_all_reduce(&mut bufs, AllReduceOp::Sum).unwrap();
        engine.execute_broadcast(&mut bufs, 0).unwrap();
        assert_eq!(engine.profiler().event_count(), 2);
    }

    #[test]
    fn engine_profiler_clear() {
        let mut engine = make_engine(2);
        let mut bufs = vec![vec![1.0], vec![2.0]];
        engine.execute_all_reduce(&mut bufs, AllReduceOp::Sum).unwrap();
        engine.profiler_mut().clear();
        assert_eq!(engine.profiler().event_count(), 0);
    }
}
