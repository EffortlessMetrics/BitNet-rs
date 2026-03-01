//! Distributed inference engine for multi-node deployments.
//!
//! Provides tensor-parallel, pipeline-parallel, and data-parallel
//! strategies with fault detection, load balancing, and
//! checkpoint/restore recovery.

use std::collections::HashMap;
use std::fmt;

// ── Configuration ─────────────────────────────────────────────────────

/// Communication transport used between nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommBackendKind {
    /// In-process mock for testing.
    Mock,
    /// NCCL-style GPU-direct transport.
    Nccl,
    /// TCP/IP sockets.
    Tcp,
}

/// Configuration for distributed inference.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Total number of participating nodes.
    pub world_size: usize,
    /// Rank of this node (0-based).
    pub rank: usize,
    /// Parallelism approach.
    pub strategy: ParallelismStrategy,
    /// Transport backend.
    pub backend: CommBackendKind,
    /// Heartbeat interval in milliseconds.
    pub heartbeat_interval_ms: u64,
    /// Number of missed heartbeats before declaring a node dead.
    pub heartbeat_timeout_count: u32,
}

impl DistributedConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), DistributedError> {
        if self.world_size == 0 {
            return Err(DistributedError::InvalidConfig(
                "world_size must be >= 1".into(),
            ));
        }
        if self.rank >= self.world_size {
            return Err(DistributedError::InvalidConfig(
                format!(
                    "rank {} >= world_size {}",
                    self.rank, self.world_size
                ),
            ));
        }
        if self.heartbeat_interval_ms == 0 {
            return Err(DistributedError::InvalidConfig(
                "heartbeat_interval_ms must be > 0".into(),
            ));
        }
        if let ParallelismStrategy::Hybrid {
            tensor_parallel_size,
            pipeline_parallel_size,
        } = &self.strategy
        {
            let product =
                tensor_parallel_size * pipeline_parallel_size;
            if product != self.world_size {
                return Err(DistributedError::InvalidConfig(
                    format!(
                        "TP({}) * PP({}) = {} != world_size({})",
                        tensor_parallel_size,
                        pipeline_parallel_size,
                        product,
                        self.world_size,
                    ),
                ));
            }
        }
        Ok(())
    }
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            strategy: ParallelismStrategy::TensorParallel,
            backend: CommBackendKind::Mock,
            heartbeat_interval_ms: 1000,
            heartbeat_timeout_count: 3,
        }
    }
}

// ── Parallelism strategies ────────────────────────────────────────────

/// How to distribute computation across nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelismStrategy {
    /// Split individual tensor operations across nodes.
    TensorParallel,
    /// Assign different layers to different nodes.
    PipelineParallel,
    /// Replicate the model; each node handles separate requests.
    DataParallel,
    /// Combined tensor + pipeline parallelism.
    Hybrid {
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
    },
}

impl fmt::Display for ParallelismStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TensorParallel => write!(f, "TensorParallel"),
            Self::PipelineParallel => write!(f, "PipelineParallel"),
            Self::DataParallel => write!(f, "DataParallel"),
            Self::Hybrid {
                tensor_parallel_size: tp,
                pipeline_parallel_size: pp,
            } => write!(f, "Hybrid(TP={tp}, PP={pp})"),
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────

/// Errors produced by distributed operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedError {
    /// Configuration is invalid.
    InvalidConfig(String),
    /// A communication operation failed.
    CommunicationFailure(String),
    /// A node has been detected as failed.
    NodeFailure { rank: usize },
    /// Checkpoint/restore error.
    CheckpointError(String),
    /// Data length mismatch in a collective operation.
    LengthMismatch { expected: usize, actual: usize },
}

impl fmt::Display for DistributedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => {
                write!(f, "invalid distributed config: {msg}")
            }
            Self::CommunicationFailure(msg) => {
                write!(f, "communication failure: {msg}")
            }
            Self::NodeFailure { rank } => {
                write!(f, "node {rank} failed")
            }
            Self::CheckpointError(msg) => {
                write!(f, "checkpoint error: {msg}")
            }
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "length mismatch: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for DistributedError {}

// ── Communication backend trait ───────────────────────────────────────

/// Trait for inter-node communication primitives.
pub trait CommunicationBackend: fmt::Debug {
    /// Sum-reduce across all nodes (in-place).
    fn all_reduce_sum(
        &self,
        data: &mut [f32],
    ) -> Result<(), DistributedError>;

    /// Broadcast `data` from `root` to all nodes.
    fn broadcast(
        &self,
        data: &mut [f32],
        root: usize,
    ) -> Result<(), DistributedError>;

    /// Scatter equal-sized chunks from `root` to each node.
    fn scatter(
        &self,
        send_buf: &[f32],
        recv_buf: &mut [f32],
        root: usize,
    ) -> Result<(), DistributedError>;

    /// Gather equal-sized chunks from each node to `root`.
    fn gather(
        &self,
        send_buf: &[f32],
        recv_buf: &mut [f32],
        root: usize,
    ) -> Result<(), DistributedError>;

    /// Gather from all nodes to all nodes.
    fn all_gather(
        &self,
        send_buf: &[f32],
        recv_buf: &mut [f32],
    ) -> Result<(), DistributedError>;

    /// Reduce-scatter: reduce then scatter the result.
    fn reduce_scatter(
        &self,
        send_buf: &[f32],
        recv_buf: &mut [f32],
    ) -> Result<(), DistributedError>;

    /// World size known to this backend.
    fn world_size(&self) -> usize;

    /// Local rank known to this backend.
    fn rank(&self) -> usize;
}

// ── Mock communication backend ────────────────────────────────────────

/// In-process mock backend that simulates multi-node collectives.
///
/// All operations behave as if a single node executes the collective
/// with `world_size` identical copies of itself.
#[derive(Debug, Clone)]
pub struct MockCommunicationBackend {
    world_size: usize,
    rank: usize,
}

impl MockCommunicationBackend {
    /// Create a new mock backend.
    pub const fn new(world_size: usize, rank: usize) -> Self {
        Self { world_size, rank }
    }
}

impl CommunicationBackend for MockCommunicationBackend {
    fn all_reduce_sum(
        &self,
        data: &mut [f32],
    ) -> Result<(), DistributedError> {
        // Simulate: each of `world_size` nodes contributes the same
        // values, so the sum is value * world_size.
        #[allow(clippy::cast_precision_loss)]
        let factor = self.world_size as f32;
        for v in data.iter_mut() {
            *v *= factor;
        }
        Ok(())
    }

    fn broadcast(
        &self,
        _data: &mut [f32],
        root: usize,
    ) -> Result<(), DistributedError> {
        if root >= self.world_size {
            return Err(DistributedError::CommunicationFailure(
                format!("root {root} >= world_size {}", self.world_size),
            ));
        }
        // Mock: data already on this node.
        Ok(())
    }

    fn scatter(
        &self,
        send_buf: &[f32],
        recv_buf: &mut [f32],
        root: usize,
    ) -> Result<(), DistributedError> {
        if root >= self.world_size {
            return Err(DistributedError::CommunicationFailure(
                format!("root {root} >= world_size {}", self.world_size),
            ));
        }
        let chunk = send_buf.len() / self.world_size;
        if recv_buf.len() < chunk {
            return Err(DistributedError::LengthMismatch {
                expected: chunk,
                actual: recv_buf.len(),
            });
        }
        let start = self.rank * chunk;
        recv_buf[..chunk].copy_from_slice(&send_buf[start..start + chunk]);
        Ok(())
    }

    fn gather(
        &self,
        send_buf: &[f32],
        recv_buf: &mut [f32],
        root: usize,
    ) -> Result<(), DistributedError> {
        if root >= self.world_size {
            return Err(DistributedError::CommunicationFailure(
                format!("root {root} >= world_size {}", self.world_size),
            ));
        }
        let chunk = send_buf.len();
        let needed = chunk * self.world_size;
        if recv_buf.len() < needed {
            return Err(DistributedError::LengthMismatch {
                expected: needed,
                actual: recv_buf.len(),
            });
        }
        // Mock: place our chunk at our rank offset, fill others
        // with copies (simulates all-same scenario).
        for n in 0..self.world_size {
            let off = n * chunk;
            recv_buf[off..off + chunk].copy_from_slice(send_buf);
        }
        Ok(())
    }

    fn all_gather(
        &self,
        send_buf: &[f32],
        recv_buf: &mut [f32],
    ) -> Result<(), DistributedError> {
        let chunk = send_buf.len();
        let needed = chunk * self.world_size;
        if recv_buf.len() < needed {
            return Err(DistributedError::LengthMismatch {
                expected: needed,
                actual: recv_buf.len(),
            });
        }
        for n in 0..self.world_size {
            let off = n * chunk;
            recv_buf[off..off + chunk].copy_from_slice(send_buf);
        }
        Ok(())
    }

    fn reduce_scatter(
        &self,
        send_buf: &[f32],
        recv_buf: &mut [f32],
    ) -> Result<(), DistributedError> {
        let chunk = send_buf.len() / self.world_size;
        if recv_buf.len() < chunk {
            return Err(DistributedError::LengthMismatch {
                expected: chunk,
                actual: recv_buf.len(),
            });
        }
        let start = self.rank * chunk;
        // Simulate reduce (sum) then scatter: each element is
        // summed across world_size identical copies.
        #[allow(clippy::cast_precision_loss)]
        let factor = self.world_size as f32;
        for (i, v) in recv_buf[..chunk].iter_mut().enumerate() {
            *v = send_buf[start + i] * factor;
        }
        Ok(())
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }
}

// ── Shard placement ───────────────────────────────────────────────────

/// Describes where a tensor shard lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardPlacement {
    /// Name of the tensor being sharded.
    pub tensor_name: String,
    /// The dimension along which to shard.
    pub shard_dim: usize,
    /// Total number of shards.
    pub num_shards: usize,
    /// Which shard this placement represents.
    pub shard_index: usize,
    /// The rank that owns this shard.
    pub owner_rank: usize,
}

impl ShardPlacement {
    /// Compute the range of elements this shard covers.
    pub fn element_range(
        &self,
        total_elements: usize,
    ) -> (usize, usize) {
        let chunk = total_elements / self.num_shards;
        let remainder = total_elements % self.num_shards;
        let start = chunk * self.shard_index
            + self.shard_index.min(remainder);
        let end = start
            + chunk
            + usize::from(self.shard_index < remainder);
        (start, end)
    }
}

/// Build shard placements for a tensor distributed across ranks.
pub fn plan_tensor_shards(
    tensor_name: &str,
    shard_dim: usize,
    world_size: usize,
) -> Vec<ShardPlacement> {
    (0..world_size)
        .map(|i| ShardPlacement {
            tensor_name: tensor_name.to_string(),
            shard_dim,
            num_shards: world_size,
            shard_index: i,
            owner_rank: i,
        })
        .collect()
}

// ── Pipeline stage ────────────────────────────────────────────────────

/// A stage in pipeline-parallel execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStage {
    /// Stage index (0-based).
    pub stage_id: usize,
    /// Rank that owns this stage.
    pub owner_rank: usize,
    /// Inclusive start layer index.
    pub layer_start: usize,
    /// Exclusive end layer index.
    pub layer_end: usize,
}

impl PipelineStage {
    /// Number of layers in this stage.
    pub const fn num_layers(&self) -> usize {
        self.layer_end - self.layer_start
    }
}

/// Partition `num_layers` into `num_stages` pipeline stages.
pub fn partition_layers(
    num_layers: usize,
    num_stages: usize,
) -> Vec<PipelineStage> {
    if num_stages == 0 {
        return Vec::new();
    }
    let base = num_layers / num_stages;
    let remainder = num_layers % num_stages;
    let mut stages = Vec::with_capacity(num_stages);
    let mut offset = 0;
    for i in 0..num_stages {
        let count = base + usize::from(i < remainder);
        stages.push(PipelineStage {
            stage_id: i,
            owner_rank: i,
            layer_start: offset,
            layer_end: offset + count,
        });
        offset += count;
    }
    stages
}

// ── Fault detector ────────────────────────────────────────────────────

/// Health status of a single node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeHealth {
    Healthy,
    Suspected,
    Dead,
}

/// Heartbeat-based fault detector.
#[derive(Debug)]
pub struct FaultDetector {
    world_size: usize,
    /// Timestamps (ms) of the last heartbeat from each rank.
    last_heartbeat: Vec<u64>,
    /// Number of consecutive missed heartbeats per rank.
    missed_count: Vec<u32>,
    /// Threshold before declaring dead.
    timeout_count: u32,
}

impl FaultDetector {
    /// Create a new fault detector initialised at `now_ms`.
    pub fn new(
        world_size: usize,
        timeout_count: u32,
        now_ms: u64,
    ) -> Self {
        Self {
            world_size,
            last_heartbeat: vec![now_ms; world_size],
            missed_count: vec![0; world_size],
            timeout_count,
        }
    }

    /// Record a heartbeat from `rank` at time `now_ms`.
    pub fn heartbeat(
        &mut self,
        rank: usize,
        now_ms: u64,
    ) -> Result<(), DistributedError> {
        if rank >= self.world_size {
            return Err(DistributedError::InvalidConfig(
                format!(
                    "rank {rank} >= world_size {}",
                    self.world_size
                ),
            ));
        }
        self.last_heartbeat[rank] = now_ms;
        self.missed_count[rank] = 0;
        Ok(())
    }

    /// Tick the detector: increment missed count for nodes that
    /// have not sent a heartbeat since `last_check_ms`.
    pub fn tick(&mut self, now_ms: u64, interval_ms: u64) {
        for rank in 0..self.world_size {
            if now_ms.saturating_sub(self.last_heartbeat[rank])
                >= interval_ms
            {
                self.missed_count[rank] += 1;
            }
        }
    }

    /// Return the health of `rank`.
    pub fn health(&self, rank: usize) -> NodeHealth {
        if rank >= self.world_size {
            return NodeHealth::Dead;
        }
        if self.missed_count[rank] >= self.timeout_count {
            NodeHealth::Dead
        } else if self.missed_count[rank] > 0 {
            NodeHealth::Suspected
        } else {
            NodeHealth::Healthy
        }
    }

    /// Collect all dead ranks.
    pub fn dead_nodes(&self) -> Vec<usize> {
        (0..self.world_size)
            .filter(|&r| self.health(r) == NodeHealth::Dead)
            .collect()
    }
}

// ── Load balancer ─────────────────────────────────────────────────────

/// Capacity snapshot of a single node.
#[derive(Debug, Clone)]
pub struct NodeCapacity {
    pub rank: usize,
    /// Available memory in bytes.
    pub free_memory: usize,
    /// Current queue depth (lower is better).
    pub queue_depth: u32,
    /// Whether the node is alive.
    pub alive: bool,
}

/// Distributes work across nodes based on capacity.
#[derive(Debug)]
pub struct LoadBalancer {
    capacities: Vec<NodeCapacity>,
}

impl LoadBalancer {
    /// Create a load balancer for the given node capacities.
    pub const fn new(capacities: Vec<NodeCapacity>) -> Self {
        Self { capacities }
    }

    /// Update capacity for a single node.
    pub fn update(
        &mut self,
        rank: usize,
        free_memory: usize,
        queue_depth: u32,
        alive: bool,
    ) {
        if let Some(cap) = self.capacities.iter_mut().find(|c| c.rank == rank)
        {
            cap.free_memory = free_memory;
            cap.queue_depth = queue_depth;
            cap.alive = alive;
        }
    }

    /// Mark a node as dead.
    pub fn mark_dead(&mut self, rank: usize) {
        if let Some(cap) = self.capacities.iter_mut().find(|c| c.rank == rank)
        {
            cap.alive = false;
        }
    }

    /// Select the best node to handle the next request.
    ///
    /// Prefers alive nodes with the lowest queue depth, breaking
    /// ties by highest free memory.
    pub fn select_node(&self) -> Option<usize> {
        self.capacities
            .iter()
            .filter(|c| c.alive)
            .min_by(|a, b| {
                a.queue_depth
                    .cmp(&b.queue_depth)
                    .then(b.free_memory.cmp(&a.free_memory))
            })
            .map(|c| c.rank)
    }

    /// Return alive nodes sorted by descending capacity (memory).
    pub fn ranked_nodes(&self) -> Vec<usize> {
        let mut alive: Vec<&NodeCapacity> =
            self.capacities.iter().filter(|c| c.alive).collect();
        alive.sort_by(|a, b| b.free_memory.cmp(&a.free_memory));
        alive.iter().map(|c| c.rank).collect()
    }

    /// Number of alive nodes.
    pub fn alive_count(&self) -> usize {
        self.capacities.iter().filter(|c| c.alive).count()
    }
}

// ── Checkpoint / restore ──────────────────────────────────────────────

/// Serialisable checkpoint of engine state.
#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    /// Monotonic checkpoint id.
    pub id: u64,
    /// Which ranks participated.
    pub ranks: Vec<usize>,
    /// KV-cache state per rank (opaque blobs in production;
    /// here we store f32 slices for testing).
    pub kv_states: HashMap<usize, Vec<f32>>,
    /// Token position at checkpoint time.
    pub token_position: usize,
}

// ── Distributed engine ────────────────────────────────────────────────

/// Manages multi-node inference orchestration.
#[derive(Debug)]
pub struct DistributedEngine<B: CommunicationBackend> {
    config: DistributedConfig,
    backend: B,
    fault_detector: FaultDetector,
    load_balancer: LoadBalancer,
    pipeline_stages: Vec<PipelineStage>,
    shard_map: Vec<ShardPlacement>,
    checkpoints: Vec<Checkpoint>,
    token_position: usize,
}

impl<B: CommunicationBackend> DistributedEngine<B> {
    /// Build a new engine from config and backend.
    pub fn new(
        config: DistributedConfig,
        backend: B,
    ) -> Result<Self, DistributedError> {
        config.validate()?;

        let fault_detector = FaultDetector::new(
            config.world_size,
            config.heartbeat_timeout_count,
            0,
        );

        let capacities: Vec<NodeCapacity> = (0..config.world_size)
            .map(|r| NodeCapacity {
                rank: r,
                free_memory: 1 << 30, // 1 GiB default
                queue_depth: 0,
                alive: true,
            })
            .collect();
        let load_balancer = LoadBalancer::new(capacities);

        let pipeline_stages = match &config.strategy {
            ParallelismStrategy::PipelineParallel => {
                partition_layers(32, config.world_size)
            }
            ParallelismStrategy::Hybrid {
                pipeline_parallel_size,
                ..
            } => partition_layers(32, *pipeline_parallel_size),
            _ => Vec::new(),
        };

        let shard_map = match &config.strategy {
            ParallelismStrategy::TensorParallel => {
                plan_tensor_shards("weight", 0, config.world_size)
            }
            ParallelismStrategy::Hybrid {
                tensor_parallel_size,
                ..
            } => {
                plan_tensor_shards("weight", 0, *tensor_parallel_size)
            }
            _ => Vec::new(),
        };

        Ok(Self {
            config,
            backend,
            fault_detector,
            load_balancer,
            pipeline_stages,
            shard_map,
            checkpoints: Vec::new(),
            token_position: 0,
        })
    }

    /// Reference to the configuration.
    pub const fn config(&self) -> &DistributedConfig {
        &self.config
    }

    /// Reference to the backend.
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Current pipeline stages.
    pub fn pipeline_stages(&self) -> &[PipelineStage] {
        &self.pipeline_stages
    }

    /// Current shard map.
    pub fn shard_map(&self) -> &[ShardPlacement] {
        &self.shard_map
    }

    /// Reference to the load balancer.
    pub const fn load_balancer(&self) -> &LoadBalancer {
        &self.load_balancer
    }

    /// Mutable reference to the load balancer.
    pub const fn load_balancer_mut(&mut self) -> &mut LoadBalancer {
        &mut self.load_balancer
    }

    /// Process a heartbeat from `rank`.
    pub fn heartbeat(
        &mut self,
        rank: usize,
        now_ms: u64,
    ) -> Result<(), DistributedError> {
        self.fault_detector.heartbeat(rank, now_ms)
    }

    /// Advance the fault detector by one tick.
    pub fn tick_fault_detector(
        &mut self,
        now_ms: u64,
    ) {
        self.fault_detector
            .tick(now_ms, self.config.heartbeat_interval_ms);
    }

    /// Check health of a rank.
    pub fn node_health(&self, rank: usize) -> NodeHealth {
        self.fault_detector.health(rank)
    }

    /// Get all dead node ranks.
    pub fn dead_nodes(&self) -> Vec<usize> {
        self.fault_detector.dead_nodes()
    }

    /// Re-shard after a node failure: redistribute shards among
    /// surviving nodes. Returns the new shard placements.
    pub fn reshard_on_failure(
        &mut self,
        failed_rank: usize,
    ) -> Result<Vec<ShardPlacement>, DistributedError> {
        self.load_balancer.mark_dead(failed_rank);
        let alive = self.load_balancer.alive_count();
        if alive == 0 {
            return Err(DistributedError::NodeFailure {
                rank: failed_rank,
            });
        }
        let alive_ranks = self.load_balancer.ranked_nodes();
        let new_shards: Vec<ShardPlacement> = alive_ranks
            .iter()
            .enumerate()
            .map(|(i, &owner)| ShardPlacement {
                tensor_name: "weight".to_string(),
                shard_dim: 0,
                num_shards: alive,
                shard_index: i,
                owner_rank: owner,
            })
            .collect();
        self.shard_map.clone_from(&new_shards);
        Ok(new_shards)
    }

    /// Re-partition pipeline stages after a node failure.
    pub fn repartition_pipeline(
        &mut self,
        failed_rank: usize,
        num_layers: usize,
    ) -> Result<Vec<PipelineStage>, DistributedError> {
        self.load_balancer.mark_dead(failed_rank);
        let alive_ranks = self.load_balancer.ranked_nodes();
        if alive_ranks.is_empty() {
            return Err(DistributedError::NodeFailure {
                rank: failed_rank,
            });
        }
        let base = num_layers / alive_ranks.len();
        let remainder = num_layers % alive_ranks.len();
        let mut stages = Vec::with_capacity(alive_ranks.len());
        let mut offset = 0;
        for (i, &owner) in alive_ranks.iter().enumerate() {
            let count = base + usize::from(i < remainder);
            stages.push(PipelineStage {
                stage_id: i,
                owner_rank: owner,
                layer_start: offset,
                layer_end: offset + count,
            });
            offset += count;
        }
        self.pipeline_stages.clone_from(&stages);
        Ok(stages)
    }

    /// Create a checkpoint of the current state.
    pub fn create_checkpoint(
        &mut self,
        kv_states: HashMap<usize, Vec<f32>>,
    ) -> Checkpoint {
        let ckpt = Checkpoint {
            id: self.checkpoints.len() as u64,
            ranks: (0..self.config.world_size).collect(),
            kv_states,
            token_position: self.token_position,
        };
        self.checkpoints.push(ckpt.clone());
        ckpt
    }

    /// Restore from a checkpoint.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: &Checkpoint,
    ) -> Result<(), DistributedError> {
        #[allow(clippy::cast_possible_truncation)]
        if checkpoint.id as usize >= self.checkpoints.len() {
            return Err(DistributedError::CheckpointError(
                format!(
                    "checkpoint {} not found (have {})",
                    checkpoint.id,
                    self.checkpoints.len()
                ),
            ));
        }
        self.token_position = checkpoint.token_position;
        Ok(())
    }

    /// Current token position.
    pub const fn token_position(&self) -> usize {
        self.token_position
    }

    /// Advance token position.
    pub const fn advance(&mut self, count: usize) {
        self.token_position += count;
    }

    /// Execute an all-reduce sum on the given buffer.
    pub fn all_reduce_sum(
        &self,
        data: &mut [f32],
    ) -> Result<(), DistributedError> {
        self.backend.all_reduce_sum(data)
    }

    /// Broadcast data from `root`.
    pub fn broadcast(
        &self,
        data: &mut [f32],
        root: usize,
    ) -> Result<(), DistributedError> {
        self.backend.broadcast(data, root)
    }

    /// Select the best node via the load balancer.
    pub fn select_node(&self) -> Option<usize> {
        self.load_balancer.select_node()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // — helpers —

    fn default_config(world_size: usize) -> DistributedConfig {
        DistributedConfig {
            world_size,
            rank: 0,
            strategy: ParallelismStrategy::TensorParallel,
            backend: CommBackendKind::Mock,
            heartbeat_interval_ms: 100,
            heartbeat_timeout_count: 3,
        }
    }

    fn mock_engine(
        world_size: usize,
    ) -> DistributedEngine<MockCommunicationBackend> {
        let cfg = default_config(world_size);
        let backend = MockCommunicationBackend::new(world_size, 0);
        DistributedEngine::new(cfg, backend).unwrap()
    }

    fn pipeline_engine(
        world_size: usize,
    ) -> DistributedEngine<MockCommunicationBackend> {
        let cfg = DistributedConfig {
            world_size,
            rank: 0,
            strategy: ParallelismStrategy::PipelineParallel,
            backend: CommBackendKind::Mock,
            heartbeat_interval_ms: 100,
            heartbeat_timeout_count: 3,
        };
        let backend = MockCommunicationBackend::new(world_size, 0);
        DistributedEngine::new(cfg, backend).unwrap()
    }

    // ── Config validation ──────────────────────────────────────────

    #[test]
    fn config_default_is_valid() {
        DistributedConfig::default().validate().unwrap();
    }

    #[test]
    fn config_zero_world_size_rejected() {
        let cfg = DistributedConfig { world_size: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rank_gte_world_size_rejected() {
        let cfg = DistributedConfig {
            world_size: 2,
            rank: 2,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_zero_heartbeat_rejected() {
        let cfg = DistributedConfig {
            heartbeat_interval_ms: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_hybrid_product_must_match_world() {
        let cfg = DistributedConfig {
            world_size: 8,
            rank: 0,
            strategy: ParallelismStrategy::Hybrid {
                tensor_parallel_size: 2,
                pipeline_parallel_size: 4,
            },
            ..Default::default()
        };
        cfg.validate().unwrap();
    }

    #[test]
    fn config_hybrid_mismatch_rejected() {
        let cfg = DistributedConfig {
            world_size: 8,
            rank: 0,
            strategy: ParallelismStrategy::Hybrid {
                tensor_parallel_size: 3,
                pipeline_parallel_size: 4,
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ── Strategy display ───────────────────────────────────────────

    #[test]
    fn strategy_display_tensor_parallel() {
        assert_eq!(
            ParallelismStrategy::TensorParallel.to_string(),
            "TensorParallel"
        );
    }

    #[test]
    fn strategy_display_pipeline_parallel() {
        assert_eq!(
            ParallelismStrategy::PipelineParallel.to_string(),
            "PipelineParallel"
        );
    }

    #[test]
    fn strategy_display_data_parallel() {
        assert_eq!(
            ParallelismStrategy::DataParallel.to_string(),
            "DataParallel"
        );
    }

    #[test]
    fn strategy_display_hybrid() {
        let s = ParallelismStrategy::Hybrid {
            tensor_parallel_size: 2,
            pipeline_parallel_size: 4,
        };
        assert_eq!(s.to_string(), "Hybrid(TP=2, PP=4)");
    }

    // ── Error display ──────────────────────────────────────────────

    #[test]
    fn error_display_coverage() {
        let cases: Vec<(DistributedError, &str)> = vec![
            (
                DistributedError::InvalidConfig("bad".into()),
                "invalid distributed config: bad",
            ),
            (
                DistributedError::CommunicationFailure("oops".into()),
                "communication failure: oops",
            ),
            (
                DistributedError::NodeFailure { rank: 3 },
                "node 3 failed",
            ),
            (
                DistributedError::CheckpointError("gone".into()),
                "checkpoint error: gone",
            ),
            (
                DistributedError::LengthMismatch {
                    expected: 10,
                    actual: 5,
                },
                "length mismatch: expected 10, got 5",
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }

    // ── Mock backend: all_reduce_sum ───────────────────────────────

    #[test]
    fn all_reduce_sum_scales_by_world_size() {
        let backend = MockCommunicationBackend::new(4, 0);
        let mut data = vec![1.0, 2.0, 3.0];
        backend.all_reduce_sum(&mut data).unwrap();
        assert_eq!(data, vec![4.0, 8.0, 12.0]);
    }

    #[test]
    fn all_reduce_sum_single_node_identity() {
        let backend = MockCommunicationBackend::new(1, 0);
        let mut data = vec![5.0, 10.0];
        backend.all_reduce_sum(&mut data).unwrap();
        assert_eq!(data, vec![5.0, 10.0]);
    }

    #[test]
    fn all_reduce_sum_empty() {
        let backend = MockCommunicationBackend::new(4, 0);
        let mut data: Vec<f32> = vec![];
        backend.all_reduce_sum(&mut data).unwrap();
        assert!(data.is_empty());
    }

    // ── Mock backend: broadcast ────────────────────────────────────

    #[test]
    fn broadcast_valid_root() {
        let backend = MockCommunicationBackend::new(4, 1);
        let mut data = vec![1.0, 2.0];
        backend.broadcast(&mut data, 0).unwrap();
        assert_eq!(data, vec![1.0, 2.0]);
    }

    #[test]
    fn broadcast_invalid_root() {
        let backend = MockCommunicationBackend::new(2, 0);
        let mut data = vec![1.0];
        assert!(backend.broadcast(&mut data, 5).is_err());
    }

    // ── Mock backend: scatter ──────────────────────────────────────

    #[test]
    fn scatter_selects_rank_chunk() {
        let backend = MockCommunicationBackend::new(4, 2);
        let send = vec![10.0, 20.0, 30.0, 40.0];
        let mut recv = vec![0.0];
        backend.scatter(&send, &mut recv, 0).unwrap();
        assert_eq!(recv, vec![30.0]);
    }

    #[test]
    fn scatter_recv_too_small() {
        let backend = MockCommunicationBackend::new(2, 0);
        let send = vec![1.0, 2.0, 3.0, 4.0];
        let mut recv = vec![0.0]; // need 2
        // chunk = 4/2 = 2, recv.len()=1 < 2 → error
        assert!(backend.scatter(&send, &mut recv, 0).is_err());
    }

    #[test]
    fn scatter_invalid_root() {
        let backend = MockCommunicationBackend::new(2, 0);
        let send = vec![1.0, 2.0];
        let mut recv = vec![0.0];
        assert!(backend.scatter(&send, &mut recv, 9).is_err());
    }

    // ── Mock backend: gather ───────────────────────────────────────

    #[test]
    fn gather_collects_chunks() {
        let backend = MockCommunicationBackend::new(3, 1);
        let send = vec![7.0, 8.0];
        let mut recv = vec![0.0; 6];
        backend.gather(&send, &mut recv, 0).unwrap();
        // mock fills all slots with our send_buf
        assert_eq!(recv, vec![7.0, 8.0, 7.0, 8.0, 7.0, 8.0]);
    }

    #[test]
    fn gather_recv_too_small() {
        let backend = MockCommunicationBackend::new(2, 0);
        let send = vec![1.0, 2.0];
        let mut recv = vec![0.0; 2]; // need 4
        assert!(backend.gather(&send, &mut recv, 0).is_err());
    }

    #[test]
    fn gather_invalid_root() {
        let backend = MockCommunicationBackend::new(2, 0);
        let send = vec![1.0];
        let mut recv = vec![0.0; 2];
        assert!(backend.gather(&send, &mut recv, 5).is_err());
    }

    // ── Mock backend: all_gather ───────────────────────────────────

    #[test]
    fn all_gather_replicates() {
        let backend = MockCommunicationBackend::new(3, 0);
        let send = vec![1.0, 2.0];
        let mut recv = vec![0.0; 6];
        backend.all_gather(&send, &mut recv).unwrap();
        assert_eq!(recv, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn all_gather_recv_too_small() {
        let backend = MockCommunicationBackend::new(2, 0);
        let send = vec![1.0];
        let mut recv = vec![0.0]; // need 2
        assert!(backend.all_gather(&send, &mut recv).is_err());
    }

    // ── Mock backend: reduce_scatter ───────────────────────────────

    #[test]
    fn reduce_scatter_produces_scaled_chunk() {
        let backend = MockCommunicationBackend::new(2, 1);
        let send = vec![1.0, 2.0, 3.0, 4.0];
        let mut recv = vec![0.0; 2];
        backend.reduce_scatter(&send, &mut recv).unwrap();
        // rank 1 chunk: [3.0, 4.0] × 2 = [6.0, 8.0]
        assert_eq!(recv, vec![6.0, 8.0]);
    }

    #[test]
    fn reduce_scatter_recv_too_small() {
        let backend = MockCommunicationBackend::new(2, 0);
        let send = vec![1.0, 2.0, 3.0, 4.0];
        let mut recv = vec![0.0]; // need 2
        assert!(backend.reduce_scatter(&send, &mut recv).is_err());
    }

    // ── Mock backend: accessors ────────────────────────────────────

    #[test]
    fn mock_backend_world_size_and_rank() {
        let backend = MockCommunicationBackend::new(8, 3);
        assert_eq!(backend.world_size(), 8);
        assert_eq!(backend.rank(), 3);
    }

    // ── ShardPlacement ─────────────────────────────────────────────

    #[test]
    fn shard_element_range_even_split() {
        let shard = ShardPlacement {
            tensor_name: "w".into(),
            shard_dim: 0,
            num_shards: 4,
            shard_index: 2,
            owner_rank: 2,
        };
        assert_eq!(shard.element_range(100), (50, 75));
    }

    #[test]
    fn shard_element_range_uneven_split() {
        let shard = ShardPlacement {
            tensor_name: "w".into(),
            shard_dim: 0,
            num_shards: 3,
            shard_index: 0,
            owner_rank: 0,
        };
        // 10/3 = 3 rem 1 → shard 0 gets 4 elements [0..4)
        assert_eq!(shard.element_range(10), (0, 4));
    }

    #[test]
    fn plan_tensor_shards_count() {
        let shards = plan_tensor_shards("attn.q", 1, 4);
        assert_eq!(shards.len(), 4);
        for (i, s) in shards.iter().enumerate() {
            assert_eq!(s.shard_index, i);
            assert_eq!(s.owner_rank, i);
            assert_eq!(s.tensor_name, "attn.q");
        }
    }

    // ── Pipeline stages ────────────────────────────────────────────

    #[test]
    fn partition_layers_even() {
        let stages = partition_layers(32, 4);
        assert_eq!(stages.len(), 4);
        for s in &stages {
            assert_eq!(s.num_layers(), 8);
        }
        assert_eq!(stages[0].layer_start, 0);
        assert_eq!(stages[3].layer_end, 32);
    }

    #[test]
    fn partition_layers_uneven() {
        let stages = partition_layers(10, 3);
        assert_eq!(stages.len(), 3);
        // 10/3 = 3 rem 1 → first gets 4, rest get 3
        assert_eq!(stages[0].num_layers(), 4);
        assert_eq!(stages[1].num_layers(), 3);
        assert_eq!(stages[2].num_layers(), 3);
        assert_eq!(stages[2].layer_end, 10);
    }

    #[test]
    fn partition_layers_zero_stages() {
        assert!(partition_layers(10, 0).is_empty());
    }

    #[test]
    fn partition_layers_more_stages_than_layers() {
        let stages = partition_layers(2, 5);
        assert_eq!(stages.len(), 5);
        let total: usize = stages.iter().map(PipelineStage::num_layers).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn pipeline_stage_owner_matches_id() {
        let stages = partition_layers(16, 4);
        for (i, s) in stages.iter().enumerate() {
            assert_eq!(s.stage_id, i);
            assert_eq!(s.owner_rank, i);
        }
    }

    // ── Fault detector ─────────────────────────────────────────────

    #[test]
    fn fault_detector_all_healthy_initially() {
        let fd = FaultDetector::new(4, 3, 0);
        for r in 0..4 {
            assert_eq!(fd.health(r), NodeHealth::Healthy);
        }
    }

    #[test]
    fn fault_detector_suspected_after_one_miss() {
        let mut fd = FaultDetector::new(4, 3, 0);
        fd.tick(200, 100);
        assert_eq!(fd.health(0), NodeHealth::Suspected);
    }

    #[test]
    fn fault_detector_dead_after_timeout() {
        let mut fd = FaultDetector::new(2, 2, 0);
        fd.tick(200, 100);
        fd.tick(300, 100);
        assert_eq!(fd.health(0), NodeHealth::Dead);
        assert_eq!(fd.health(1), NodeHealth::Dead);
    }

    #[test]
    fn fault_detector_heartbeat_resets() {
        let mut fd = FaultDetector::new(2, 3, 0);
        fd.tick(200, 100);
        assert_eq!(fd.health(0), NodeHealth::Suspected);
        fd.heartbeat(0, 200).unwrap();
        assert_eq!(fd.health(0), NodeHealth::Healthy);
    }

    #[test]
    fn fault_detector_heartbeat_invalid_rank() {
        let mut fd = FaultDetector::new(2, 3, 0);
        assert!(fd.heartbeat(5, 100).is_err());
    }

    #[test]
    fn fault_detector_dead_nodes() {
        let mut fd = FaultDetector::new(4, 1, 0);
        fd.tick(200, 100);
        // All missed once with timeout=1 → all dead
        assert_eq!(fd.dead_nodes().len(), 4);
    }

    #[test]
    fn fault_detector_partial_failure() {
        let mut fd = FaultDetector::new(3, 2, 0);
        fd.heartbeat(0, 150).unwrap();
        fd.tick(200, 100);
        fd.tick(300, 100);
        // rank 0: heartbeat at 150, tick at 200 (diff=50<100) → missed=0
        //         tick at 300 (diff=150>=100) → missed=1 → Suspected
        // rank 1,2: missed twice → Dead
        assert_eq!(fd.health(0), NodeHealth::Suspected);
        assert_eq!(fd.health(1), NodeHealth::Dead);
        assert_eq!(fd.health(2), NodeHealth::Dead);
    }

    #[test]
    fn fault_detector_out_of_range_rank_is_dead() {
        let fd = FaultDetector::new(2, 3, 0);
        assert_eq!(fd.health(99), NodeHealth::Dead);
    }

    // ── Load balancer ──────────────────────────────────────────────

    #[test]
    fn load_balancer_selects_lowest_queue_depth() {
        let caps = vec![
            NodeCapacity {
                rank: 0, free_memory: 1000, queue_depth: 5, alive: true,
            },
            NodeCapacity {
                rank: 1, free_memory: 1000, queue_depth: 2, alive: true,
            },
            NodeCapacity {
                rank: 2, free_memory: 1000, queue_depth: 8, alive: true,
            },
        ];
        let lb = LoadBalancer::new(caps);
        assert_eq!(lb.select_node(), Some(1));
    }

    #[test]
    fn load_balancer_breaks_ties_by_memory() {
        let caps = vec![
            NodeCapacity {
                rank: 0, free_memory: 500, queue_depth: 1, alive: true,
            },
            NodeCapacity {
                rank: 1, free_memory: 900, queue_depth: 1, alive: true,
            },
        ];
        let lb = LoadBalancer::new(caps);
        assert_eq!(lb.select_node(), Some(1));
    }

    #[test]
    fn load_balancer_skips_dead_nodes() {
        let caps = vec![
            NodeCapacity {
                rank: 0, free_memory: 1000, queue_depth: 0, alive: false,
            },
            NodeCapacity {
                rank: 1, free_memory: 100, queue_depth: 5, alive: true,
            },
        ];
        let lb = LoadBalancer::new(caps);
        assert_eq!(lb.select_node(), Some(1));
    }

    #[test]
    fn load_balancer_no_alive_returns_none() {
        let caps = vec![NodeCapacity {
            rank: 0, free_memory: 1000, queue_depth: 0, alive: false,
        }];
        let lb = LoadBalancer::new(caps);
        assert_eq!(lb.select_node(), None);
    }

    #[test]
    fn load_balancer_mark_dead() {
        let caps = vec![
            NodeCapacity {
                rank: 0, free_memory: 1000, queue_depth: 0, alive: true,
            },
            NodeCapacity {
                rank: 1, free_memory: 1000, queue_depth: 0, alive: true,
            },
        ];
        let mut lb = LoadBalancer::new(caps);
        assert_eq!(lb.alive_count(), 2);
        lb.mark_dead(0);
        assert_eq!(lb.alive_count(), 1);
        assert_eq!(lb.select_node(), Some(1));
    }

    #[test]
    fn load_balancer_update_capacity() {
        let caps = vec![
            NodeCapacity {
                rank: 0, free_memory: 500, queue_depth: 3, alive: true,
            },
            NodeCapacity {
                rank: 1, free_memory: 500, queue_depth: 3, alive: true,
            },
        ];
        let mut lb = LoadBalancer::new(caps);
        lb.update(0, 2000, 0, true);
        assert_eq!(lb.select_node(), Some(0));
    }

    #[test]
    fn load_balancer_ranked_nodes_by_memory() {
        let caps = vec![
            NodeCapacity {
                rank: 0, free_memory: 100, queue_depth: 0, alive: true,
            },
            NodeCapacity {
                rank: 1, free_memory: 300, queue_depth: 0, alive: true,
            },
            NodeCapacity {
                rank: 2, free_memory: 200, queue_depth: 0, alive: true,
            },
        ];
        let lb = LoadBalancer::new(caps);
        assert_eq!(lb.ranked_nodes(), vec![1, 2, 0]);
    }

    // ── Engine construction ────────────────────────────────────────

    #[test]
    fn engine_tensor_parallel_has_shards() {
        let e = mock_engine(4);
        assert_eq!(e.shard_map().len(), 4);
        assert!(e.pipeline_stages().is_empty());
    }

    #[test]
    fn engine_pipeline_parallel_has_stages() {
        let e = pipeline_engine(4);
        assert_eq!(e.pipeline_stages().len(), 4);
        assert!(e.shard_map().is_empty());
    }

    #[test]
    fn engine_data_parallel_has_neither() {
        let cfg = DistributedConfig {
            world_size: 4,
            rank: 0,
            strategy: ParallelismStrategy::DataParallel,
            ..Default::default()
        };
        let backend = MockCommunicationBackend::new(4, 0);
        let e = DistributedEngine::new(cfg, backend).unwrap();
        assert!(e.shard_map().is_empty());
        assert!(e.pipeline_stages().is_empty());
    }

    #[test]
    fn engine_hybrid_has_both() {
        let cfg = DistributedConfig {
            world_size: 8,
            rank: 0,
            strategy: ParallelismStrategy::Hybrid {
                tensor_parallel_size: 2,
                pipeline_parallel_size: 4,
            },
            ..Default::default()
        };
        let backend = MockCommunicationBackend::new(8, 0);
        let e = DistributedEngine::new(cfg, backend).unwrap();
        assert_eq!(e.shard_map().len(), 2);
        assert_eq!(e.pipeline_stages().len(), 4);
    }

    #[test]
    fn engine_invalid_config_rejected() {
        let cfg = DistributedConfig {
            world_size: 0,
            ..Default::default()
        };
        let backend = MockCommunicationBackend::new(0, 0);
        assert!(DistributedEngine::new(cfg, backend).is_err());
    }

    // ── Engine: communication ──────────────────────────────────────

    #[test]
    fn engine_all_reduce_sum() {
        let e = mock_engine(4);
        let mut data = vec![1.0, 2.0, 3.0];
        e.all_reduce_sum(&mut data).unwrap();
        assert_eq!(data, vec![4.0, 8.0, 12.0]);
    }

    #[test]
    fn engine_broadcast() {
        let e = mock_engine(4);
        let mut data = vec![42.0];
        e.broadcast(&mut data, 0).unwrap();
        assert_eq!(data, vec![42.0]);
    }

    // ── Engine: fault detection ────────────────────────────────────

    #[test]
    fn engine_heartbeat_and_health() {
        let mut e = mock_engine(3);
        e.heartbeat(1, 50).unwrap();
        assert_eq!(e.node_health(1), NodeHealth::Healthy);
    }

    #[test]
    fn engine_tick_detects_failure() {
        let mut e = mock_engine(3);
        e.tick_fault_detector(200);
        e.tick_fault_detector(300);
        e.tick_fault_detector(400);
        assert_eq!(e.node_health(0), NodeHealth::Dead);
        assert!(!e.dead_nodes().is_empty());
    }

    // ── Engine: resharding ─────────────────────────────────────────

    #[test]
    fn reshard_on_failure_removes_dead_node() {
        let mut e = mock_engine(4);
        let new_shards = e.reshard_on_failure(2).unwrap();
        assert_eq!(new_shards.len(), 3);
        assert!(new_shards.iter().all(|s| s.owner_rank != 2));
    }

    #[test]
    fn reshard_on_failure_all_dead_errors() {
        let mut e = mock_engine(1);
        assert!(e.reshard_on_failure(0).is_err());
    }

    // ── Engine: pipeline repartition ───────────────────────────────

    #[test]
    fn repartition_pipeline_after_failure() {
        let mut e = pipeline_engine(4);
        let stages = e.repartition_pipeline(1, 32).unwrap();
        assert_eq!(stages.len(), 3);
        let total: usize =
            stages.iter().map(PipelineStage::num_layers).sum();
        assert_eq!(total, 32);
        assert!(stages.iter().all(|s| s.owner_rank != 1));
    }

    #[test]
    fn repartition_all_dead_errors() {
        let mut e = pipeline_engine(1);
        assert!(e.repartition_pipeline(0, 32).is_err());
    }

    // ── Engine: checkpoint/restore ─────────────────────────────────

    #[test]
    fn checkpoint_round_trip() {
        let mut e = mock_engine(2);
        e.advance(10);
        let kv = HashMap::from([
            (0, vec![1.0, 2.0]),
            (1, vec![3.0, 4.0]),
        ]);
        let ckpt = e.create_checkpoint(kv.clone());
        assert_eq!(ckpt.id, 0);
        assert_eq!(ckpt.token_position, 10);
        assert_eq!(ckpt.kv_states, kv);

        e.advance(5);
        assert_eq!(e.token_position(), 15);
        e.restore_checkpoint(&ckpt).unwrap();
        assert_eq!(e.token_position(), 10);
    }

    #[test]
    fn restore_invalid_checkpoint_errors() {
        let mut e = mock_engine(2);
        let bad = Checkpoint {
            id: 99,
            ranks: vec![0, 1],
            kv_states: HashMap::new(),
            token_position: 0,
        };
        assert!(e.restore_checkpoint(&bad).is_err());
    }

    #[test]
    fn multiple_checkpoints() {
        let mut e = mock_engine(2);
        e.advance(5);
        let c1 = e.create_checkpoint(HashMap::new());
        e.advance(10);
        let c2 = e.create_checkpoint(HashMap::new());
        assert_eq!(c1.id, 0);
        assert_eq!(c2.id, 1);
        assert_eq!(c2.token_position, 15);
        e.restore_checkpoint(&c1).unwrap();
        assert_eq!(e.token_position(), 5);
    }

    // ── Engine: load balancing ─────────────────────────────────────

    #[test]
    fn engine_select_node() {
        let mut e = mock_engine(3);
        // Give rank 1 more memory and bump others' queue depth
        e.load_balancer_mut().update(0, 500, 3, true);
        e.load_balancer_mut().update(1, 2_000_000, 0, true);
        e.load_balancer_mut().update(2, 500, 3, true);
        assert_eq!(e.select_node(), Some(1));
    }

    #[test]
    fn engine_select_node_after_failure() {
        let mut e = mock_engine(3);
        e.load_balancer_mut().mark_dead(0);
        let node = e.select_node().unwrap();
        assert_ne!(node, 0);
    }

    // ── Engine: token position ─────────────────────────────────────

    #[test]
    fn token_position_advances() {
        let mut e = mock_engine(2);
        assert_eq!(e.token_position(), 0);
        e.advance(7);
        assert_eq!(e.token_position(), 7);
        e.advance(3);
        assert_eq!(e.token_position(), 10);
    }

    // ── Engine: accessors ──────────────────────────────────────────

    #[test]
    fn engine_accessors() {
        let e = mock_engine(4);
        assert_eq!(e.config().world_size, 4);
        assert_eq!(e.backend().world_size(), 4);
        assert_eq!(e.load_balancer().alive_count(), 4);
    }

    // ── CommBackendKind ────────────────────────────────────────────

    #[test]
    fn comm_backend_kind_debug() {
        assert_eq!(format!("{:?}", CommBackendKind::Mock), "Mock");
        assert_eq!(format!("{:?}", CommBackendKind::Nccl), "Nccl");
        assert_eq!(format!("{:?}", CommBackendKind::Tcp), "Tcp");
    }

    // ── Shard element ranges cover full tensor ─────────────────────

    #[test]
    fn shard_ranges_cover_all_elements() {
        let shards = plan_tensor_shards("w", 0, 4);
        let total = 100;
        let mut covered = vec![false; total];
        for s in &shards {
            let (start, end) = s.element_range(total);
            for c in &mut covered[start..end] {
                *c = true;
            }
        }
        assert!(covered.iter().all(|&c| c));
    }

    #[test]
    fn shard_ranges_no_overlap() {
        let shards = plan_tensor_shards("w", 0, 3);
        let total = 10;
        let mut counts = vec![0u32; total];
        for s in &shards {
            let (start, end) = s.element_range(total);
            for c in &mut counts[start..end] {
                *c += 1;
            }
        }
        assert!(counts.iter().all(|&c| c == 1));
    }
}
