//! Module stub - implementation pending merge from feature branch
//! Distributed inference with pipeline/tensor parallelism across nodes.
//!
//! Provides multi-node execution primitives: process groups, collective
//! operations (allreduce, broadcast, scatter, gather), tensor partitioning,
//! pipeline scheduling (`GPipe` / 1F1B), communication–compute overlap, fault
//! detection, and an orchestrating [`DistributedInferenceEngine`].

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Communication backend used by the process group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationBackend {
    /// TCP/IP sockets (portable fallback).
    Tcp,
    /// NCCL for NVIDIA GPUs.
    Nccl,
    /// Gloo for CPU-based collectives.
    Gloo,
    /// Shared-memory transport (single-node multi-GPU).
    SharedMemory,
}

/// Configuration for a distributed inference run.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Total number of participating ranks.
    pub world_size: usize,
    /// Rank of the local process (0-based).
    pub rank: usize,
    /// Address of the rendezvous master.
    pub master_addr: String,
    /// Port of the rendezvous master.
    pub master_port: u16,
    /// Communication backend to use.
    pub backend: CommunicationBackend,
    /// Timeout for collective operations.
    pub timeout: Duration,
    /// Number of pipeline stages (≥1).
    pub pipeline_stages: usize,
    /// Degree of tensor parallelism per stage.
    pub tensor_parallel_degree: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            master_addr: "127.0.0.1".into(),
            master_port: 29500,
            backend: CommunicationBackend::Tcp,
            timeout: Duration::from_secs(30),
            pipeline_stages: 1,
            tensor_parallel_degree: 1,
        }
    }
}

impl DistributedConfig {
    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<(), String> {
        if self.world_size == 0 {
            return Err("world_size must be > 0".into());
        }
        if self.rank >= self.world_size {
            return Err(format!("rank {} must be < world_size {}", self.rank, self.world_size));
        }
        if self.master_addr.is_empty() {
            return Err("master_addr must not be empty".into());
        }
        if self.master_port == 0 {
            return Err("master_port must be > 0".into());
        }
        if self.pipeline_stages == 0 {
            return Err("pipeline_stages must be > 0".into());
        }
        if self.tensor_parallel_degree == 0 {
            return Err("tensor_parallel_degree must be > 0".into());
        }
        let required = self.pipeline_stages * self.tensor_parallel_degree;
        if self.world_size < required {
            return Err(format!(
                "world_size {} < pipeline_stages({}) * tensor_parallel_degree({})",
                self.world_size, self.pipeline_stages, self.tensor_parallel_degree
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AllReduceOp
// ---------------------------------------------------------------------------

/// Reduction operation for collective allreduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllReduceOp {
    Sum,
    Avg,
    Max,
    Min,
}

impl AllReduceOp {
    /// Apply the reduction to two scalars.
    #[allow(clippy::manual_midpoint)]
    pub fn reduce(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Sum => a + b,
            Self::Avg => (a + b) / 2.0,
            Self::Max => a.max(b),
            Self::Min => a.min(b),
        }
    }

    /// Fold a slice using this operation.
    pub fn reduce_slice(self, values: &[f32]) -> Option<f32> {
        if values.is_empty() {
            return None;
        }
        let init = values[0];
        Some(values[1..].iter().fold(init, |acc, &v| self.reduce(acc, v)))
    }
}

// ---------------------------------------------------------------------------
// ProcessGroup
// ---------------------------------------------------------------------------

/// Status of a single rank inside the group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankStatus {
    Initializing,
    Ready,
    Busy,
    Failed,
    Disconnected,
}

/// A communication group of distributed ranks.
#[derive(Debug)]
pub struct ProcessGroup {
    config: DistributedConfig,
    rank_statuses: Vec<RankStatus>,
    created_at: Instant,
    collective_count: u64,
}

impl ProcessGroup {
    /// Create a new process group from the given configuration.
    pub fn new(config: DistributedConfig) -> Result<Self, String> {
        config.validate()?;
        let rank_statuses = vec![RankStatus::Initializing; config.world_size];
        Ok(Self { config, rank_statuses, created_at: Instant::now(), collective_count: 0 })
    }

    pub const fn world_size(&self) -> usize {
        self.config.world_size
    }

    pub const fn rank(&self) -> usize {
        self.config.rank
    }

    pub const fn backend(&self) -> CommunicationBackend {
        self.config.backend
    }

    pub const fn config(&self) -> &DistributedConfig {
        &self.config
    }

    pub const fn collective_count(&self) -> u64 {
        self.collective_count
    }

    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Mark a rank as ready.
    pub fn mark_ready(&mut self, rank: usize) -> Result<(), String> {
        self.set_rank_status(rank, RankStatus::Ready)
    }

    /// Mark a rank as failed.
    pub fn mark_failed(&mut self, rank: usize) -> Result<(), String> {
        self.set_rank_status(rank, RankStatus::Failed)
    }

    fn set_rank_status(&mut self, rank: usize, status: RankStatus) -> Result<(), String> {
        if rank >= self.config.world_size {
            return Err(format!("rank {rank} out of range"));
        }
        self.rank_statuses[rank] = status;
        Ok(())
    }

    pub fn rank_status(&self, rank: usize) -> Option<RankStatus> {
        self.rank_statuses.get(rank).copied()
    }

    /// Returns `true` when every rank is `Ready`.
    pub fn all_ready(&self) -> bool {
        self.rank_statuses.iter().all(|s| *s == RankStatus::Ready)
    }

    /// Count how many ranks have the given status.
    pub fn count_status(&self, status: RankStatus) -> usize {
        self.rank_statuses.iter().filter(|s| **s == status).count()
    }

    // -- collective operations (simulate locally) --

    /// Allreduce across all ranks.
    pub fn allreduce(&mut self, local: &[f32], _op: AllReduceOp) -> Result<Vec<f32>, String> {
        if local.is_empty() {
            return Err("allreduce: empty input".into());
        }
        self.collective_count += 1;
        // Simulate: in a real implementation each rank contributes its buffer.
        Ok(local.to_vec())
    }

    /// Broadcast from `root` to all ranks.
    pub fn broadcast(&mut self, data: &[f32], root: usize) -> Result<Vec<f32>, String> {
        if root >= self.config.world_size {
            return Err(format!("broadcast root {root} out of range"));
        }
        self.collective_count += 1;
        Ok(data.to_vec())
    }

    /// Scatter: split `data` into `world_size` equal chunks and return the
    /// chunk belonging to the local rank.
    pub fn scatter(&mut self, data: &[f32], root: usize) -> Result<Vec<f32>, String> {
        if root >= self.config.world_size {
            return Err(format!("scatter root {root} out of range"));
        }
        if !data.len().is_multiple_of(self.config.world_size) {
            return Err("scatter: data length not divisible by world_size".into());
        }
        self.collective_count += 1;
        let chunk = data.len() / self.config.world_size;
        let start = self.config.rank * chunk;
        Ok(data[start..start + chunk].to_vec())
    }

    /// Gather: collect local chunks from every rank into one buffer.
    pub fn gather(&mut self, local: &[f32], root: usize) -> Result<Vec<f32>, String> {
        if root >= self.config.world_size {
            return Err(format!("gather root {root} out of range"));
        }
        self.collective_count += 1;
        // Simulate: replicate local chunk `world_size` times.
        let mut out = Vec::with_capacity(local.len() * self.config.world_size);
        for _ in 0..self.config.world_size {
            out.extend_from_slice(local);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// TensorPartitioner
// ---------------------------------------------------------------------------

/// Axis along which a tensor is split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionAxis {
    Row,
    Column,
}

/// Describes one partition of a distributed tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorPartition {
    pub rank: usize,
    pub axis: PartitionAxis,
    pub offset: usize,
    pub length: usize,
    pub data: Vec<f32>,
}

/// Splits tensors across ranks for distributed inference.
#[derive(Debug)]
pub struct TensorPartitioner {
    world_size: usize,
    axis: PartitionAxis,
}

impl TensorPartitioner {
    pub fn new(world_size: usize, axis: PartitionAxis) -> Result<Self, String> {
        if world_size == 0 {
            return Err("world_size must be > 0".into());
        }
        Ok(Self { world_size, axis })
    }

    pub const fn world_size(&self) -> usize {
        self.world_size
    }

    pub const fn axis(&self) -> PartitionAxis {
        self.axis
    }

    /// Partition a 1-D tensor into `world_size` pieces.
    pub fn partition(&self, data: &[f32]) -> Result<Vec<TensorPartition>, String> {
        if data.len() < self.world_size {
            return Err(format!("tensor length {} < world_size {}", data.len(), self.world_size));
        }
        let base = data.len() / self.world_size;
        let remainder = data.len() % self.world_size;
        let mut parts = Vec::with_capacity(self.world_size);
        let mut offset = 0;
        for rank in 0..self.world_size {
            let len = base + usize::from(rank < remainder);
            parts.push(TensorPartition {
                rank,
                axis: self.axis,
                offset,
                length: len,
                data: data[offset..offset + len].to_vec(),
            });
            offset += len;
        }
        Ok(parts)
    }

    /// Reconstruct a tensor from its partitions.
    pub fn reconstruct(&self, parts: &[TensorPartition]) -> Result<Vec<f32>, String> {
        if parts.len() != self.world_size {
            return Err(format!("expected {} partitions, got {}", self.world_size, parts.len()));
        }
        let total: usize = parts.iter().map(|p| p.length).sum();
        let mut out = vec![0.0f32; total];
        for p in parts {
            out[p.offset..p.offset + p.length].copy_from_slice(&p.data);
        }
        Ok(out)
    }

    /// Return the partition assigned to a given rank.
    pub fn partition_for_rank(&self, data: &[f32], rank: usize) -> Result<TensorPartition, String> {
        if rank >= self.world_size {
            return Err(format!("rank {rank} out of range"));
        }
        let parts = self.partition(data)?;
        Ok(parts.into_iter().nth(rank).unwrap())
    }
}

// ---------------------------------------------------------------------------
// PipelineStage
// ---------------------------------------------------------------------------

/// Execution status of a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageStatus {
    Idle,
    Computing,
    WaitingForInput,
    SendingOutput,
    Completed,
    Failed,
}

/// A single pipeline stage assigned to a rank.
#[derive(Debug)]
pub struct PipelineStage {
    pub stage_id: usize,
    pub rank: usize,
    pub layer_start: usize,
    pub layer_end: usize,
    status: StageStatus,
    microbatches_processed: u64,
    activations: Vec<f32>,
}

impl PipelineStage {
    pub const fn new(stage_id: usize, rank: usize, layer_start: usize, layer_end: usize) -> Self {
        Self {
            stage_id,
            rank,
            layer_start,
            layer_end,
            status: StageStatus::Idle,
            microbatches_processed: 0,
            activations: Vec::new(),
        }
    }

    pub const fn status(&self) -> StageStatus {
        self.status
    }

    pub const fn microbatches_processed(&self) -> u64 {
        self.microbatches_processed
    }

    pub const fn layer_count(&self) -> usize {
        self.layer_end.saturating_sub(self.layer_start)
    }

    pub fn activations(&self) -> &[f32] {
        &self.activations
    }

    /// Execute a forward pass on the given input activations.
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.is_empty() {
            return Err("empty input activations".into());
        }
        self.status = StageStatus::Computing;
        // Simulate: scale each element by (stage_id + 1) so stages are distinguishable.
        #[allow(clippy::cast_precision_loss)]
        let scale = (self.stage_id + 1) as f32;
        let output: Vec<f32> = input.iter().map(|v| v * scale).collect();
        self.activations.clone_from(&output);
        self.microbatches_processed += 1;
        self.status = StageStatus::Completed;
        Ok(output)
    }

    /// Reset stage to idle.
    pub fn reset(&mut self) {
        self.status = StageStatus::Idle;
        self.activations.clear();
    }
}

// ---------------------------------------------------------------------------
// PipelineScheduler
// ---------------------------------------------------------------------------

/// Scheduling strategy for pipeline execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleStrategy {
    /// `GPipe`: all-forward then all-backward.
    GPipe,
    /// 1F1B: interleaved forward/backward for reduced memory.
    OneFOnB,
}

/// Schedules microbatches across pipeline stages.
#[derive(Debug)]
pub struct PipelineScheduler {
    strategy: ScheduleStrategy,
    num_microbatches: usize,
    stages: Vec<PipelineStage>,
    schedule_log: Vec<ScheduleEntry>,
}

/// An entry in the execution schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleEntry {
    pub stage_id: usize,
    pub microbatch_id: usize,
    pub step: usize,
    pub direction: Direction,
}

/// Direction of a pipeline step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

impl PipelineScheduler {
    pub const fn new(strategy: ScheduleStrategy, num_microbatches: usize) -> Self {
        Self { strategy, num_microbatches, stages: Vec::new(), schedule_log: Vec::new() }
    }

    pub const fn strategy(&self) -> ScheduleStrategy {
        self.strategy
    }

    pub const fn num_microbatches(&self) -> usize {
        self.num_microbatches
    }

    pub const fn num_stages(&self) -> usize {
        self.stages.len()
    }

    pub fn schedule_log(&self) -> &[ScheduleEntry] {
        &self.schedule_log
    }

    /// Add a pipeline stage.
    pub fn add_stage(&mut self, stage: PipelineStage) {
        self.stages.push(stage);
    }

    /// Build the execution schedule based on the selected strategy.
    pub fn build_schedule(&mut self) -> Result<Vec<ScheduleEntry>, String> {
        if self.stages.is_empty() {
            return Err("no stages in scheduler".into());
        }
        if self.num_microbatches == 0 {
            return Err("num_microbatches must be > 0".into());
        }

        self.schedule_log.clear();
        let mut step = 0;

        match self.strategy {
            ScheduleStrategy::GPipe => {
                // All forwards, then all backwards.
                for mb in 0..self.num_microbatches {
                    for sid in 0..self.stages.len() {
                        self.schedule_log.push(ScheduleEntry {
                            stage_id: sid,
                            microbatch_id: mb,
                            step,
                            direction: Direction::Forward,
                        });
                        step += 1;
                    }
                }
                for mb in (0..self.num_microbatches).rev() {
                    for sid in (0..self.stages.len()).rev() {
                        self.schedule_log.push(ScheduleEntry {
                            stage_id: sid,
                            microbatch_id: mb,
                            step,
                            direction: Direction::Backward,
                        });
                        step += 1;
                    }
                }
            }
            ScheduleStrategy::OneFOnB => {
                // Interleaved: forward mb, backward mb-1 alternating.
                for mb in 0..self.num_microbatches {
                    for sid in 0..self.stages.len() {
                        self.schedule_log.push(ScheduleEntry {
                            stage_id: sid,
                            microbatch_id: mb,
                            step,
                            direction: Direction::Forward,
                        });
                        step += 1;
                    }
                    if mb > 0 {
                        for sid in (0..self.stages.len()).rev() {
                            self.schedule_log.push(ScheduleEntry {
                                stage_id: sid,
                                microbatch_id: mb - 1,
                                step,
                                direction: Direction::Backward,
                            });
                            step += 1;
                        }
                    }
                }
                // Final backward for last microbatch.
                for sid in (0..self.stages.len()).rev() {
                    self.schedule_log.push(ScheduleEntry {
                        stage_id: sid,
                        microbatch_id: self.num_microbatches - 1,
                        step,
                        direction: Direction::Backward,
                    });
                    step += 1;
                }
            }
        }

        Ok(self.schedule_log.clone())
    }

    /// Execute the schedule on concrete input data.
    pub fn execute(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let schedule = self.build_schedule()?;
        let mut current = input.to_vec();
        for entry in &schedule {
            if entry.direction == Direction::Forward
                && let Some(stage) = self.stages.get_mut(entry.stage_id)
            {
                current = stage.forward(&current)?;
            }
        }
        Ok(current)
    }
}

// ---------------------------------------------------------------------------
// CommunicationOverlap
// ---------------------------------------------------------------------------

/// Phase of an overlapped operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapPhase {
    Idle,
    ComputeOnly,
    CommOnly,
    Overlapped,
}

/// Overlaps compute with communication for efficiency.
#[derive(Debug)]
pub struct CommunicationOverlap {
    phase: OverlapPhase,
    compute_ops: u64,
    comm_ops: u64,
    overlap_ops: u64,
    enabled: bool,
}

impl CommunicationOverlap {
    pub const fn new(enabled: bool) -> Self {
        Self { phase: OverlapPhase::Idle, compute_ops: 0, comm_ops: 0, overlap_ops: 0, enabled }
    }

    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub const fn phase(&self) -> OverlapPhase {
        self.phase
    }

    pub const fn compute_ops(&self) -> u64 {
        self.compute_ops
    }

    pub const fn comm_ops(&self) -> u64 {
        self.comm_ops
    }

    pub const fn overlap_ops(&self) -> u64 {
        self.overlap_ops
    }

    /// Begin a compute-only step.
    pub const fn begin_compute(&mut self) {
        self.phase = OverlapPhase::ComputeOnly;
        self.compute_ops += 1;
    }

    /// Begin a communication-only step.
    pub const fn begin_comm(&mut self) {
        self.phase = OverlapPhase::CommOnly;
        self.comm_ops += 1;
    }

    /// Begin an overlapped compute + communication step.
    pub fn begin_overlap(&mut self) -> Result<(), String> {
        if !self.enabled {
            return Err("overlap not enabled".into());
        }
        self.phase = OverlapPhase::Overlapped;
        self.overlap_ops += 1;
        Ok(())
    }

    /// Complete the current phase and return to idle.
    pub const fn complete(&mut self) {
        self.phase = OverlapPhase::Idle;
    }

    /// Ratio of overlapped ops to total ops.
    #[allow(clippy::cast_precision_loss)]
    pub fn overlap_ratio(&self) -> f64 {
        let total = self.compute_ops + self.comm_ops + self.overlap_ops;
        if total == 0 {
            return 0.0;
        }
        self.overlap_ops as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// DistributedSynchronizer
// ---------------------------------------------------------------------------

/// Provides barrier and allgather synchronization primitives.
#[derive(Debug)]
pub struct DistributedSynchronizer {
    world_size: usize,
    barrier_count: u64,
    allgather_count: u64,
    ranks_at_barrier: Vec<bool>,
}

impl DistributedSynchronizer {
    pub fn new(world_size: usize) -> Result<Self, String> {
        if world_size == 0 {
            return Err("world_size must be > 0".into());
        }
        Ok(Self {
            world_size,
            barrier_count: 0,
            allgather_count: 0,
            ranks_at_barrier: vec![false; world_size],
        })
    }

    pub const fn world_size(&self) -> usize {
        self.world_size
    }

    pub const fn barrier_count(&self) -> u64 {
        self.barrier_count
    }

    pub const fn allgather_count(&self) -> u64 {
        self.allgather_count
    }

    /// Signal that `rank` has reached the barrier.
    pub fn arrive(&mut self, rank: usize) -> Result<bool, String> {
        if rank >= self.world_size {
            return Err(format!("rank {rank} out of range"));
        }
        self.ranks_at_barrier[rank] = true;
        if self.ranks_at_barrier.iter().all(|&r| r) {
            self.barrier_count += 1;
            self.ranks_at_barrier.fill(false);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// How many ranks have arrived at the current barrier.
    pub fn arrived_count(&self) -> usize {
        self.ranks_at_barrier.iter().filter(|&&r| r).count()
    }

    /// All-gather: combine per-rank buffers into a single buffer.
    pub fn allgather(&mut self, buffers: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        if buffers.len() != self.world_size {
            return Err(format!("expected {} buffers, got {}", self.world_size, buffers.len()));
        }
        self.allgather_count += 1;
        Ok(buffers.iter().flat_map(|b| b.iter().copied()).collect())
    }
}

// ---------------------------------------------------------------------------
// FaultDetector
// ---------------------------------------------------------------------------

/// Health status of a rank as observed by the fault detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeHealth {
    Healthy,
    Suspect,
    Dead,
}

/// Detects node failures and triggers replication/recovery.
#[derive(Debug)]
pub struct FaultDetector {
    world_size: usize,
    health: Vec<NodeHealth>,
    heartbeat_deadline: Duration,
    last_heartbeat: Vec<Option<Instant>>,
    failure_count: u64,
    recovery_count: u64,
}

impl FaultDetector {
    pub fn new(world_size: usize, heartbeat_deadline: Duration) -> Result<Self, String> {
        if world_size == 0 {
            return Err("world_size must be > 0".into());
        }
        Ok(Self {
            world_size,
            health: vec![NodeHealth::Healthy; world_size],
            heartbeat_deadline,
            last_heartbeat: vec![None; world_size],
            failure_count: 0,
            recovery_count: 0,
        })
    }

    pub const fn world_size(&self) -> usize {
        self.world_size
    }

    pub const fn failure_count(&self) -> u64 {
        self.failure_count
    }

    pub const fn recovery_count(&self) -> u64 {
        self.recovery_count
    }

    pub fn node_health(&self, rank: usize) -> Option<NodeHealth> {
        self.health.get(rank).copied()
    }

    /// Count of healthy ranks.
    pub fn healthy_count(&self) -> usize {
        self.health.iter().filter(|h| **h == NodeHealth::Healthy).count()
    }

    /// Record a heartbeat from `rank`.
    pub fn record_heartbeat(&mut self, rank: usize) -> Result<(), String> {
        if rank >= self.world_size {
            return Err(format!("rank {rank} out of range"));
        }
        self.last_heartbeat[rank] = Some(Instant::now());
        if self.health[rank] != NodeHealth::Healthy {
            self.health[rank] = NodeHealth::Healthy;
            self.recovery_count += 1;
        }
        Ok(())
    }

    /// Mark a rank as dead.
    pub fn mark_dead(&mut self, rank: usize) -> Result<(), String> {
        if rank >= self.world_size {
            return Err(format!("rank {rank} out of range"));
        }
        if self.health[rank] != NodeHealth::Dead {
            self.health[rank] = NodeHealth::Dead;
            self.failure_count += 1;
        }
        Ok(())
    }

    /// Mark a rank as suspect.
    pub fn mark_suspect(&mut self, rank: usize) -> Result<(), String> {
        if rank >= self.world_size {
            return Err(format!("rank {rank} out of range"));
        }
        self.health[rank] = NodeHealth::Suspect;
        Ok(())
    }

    /// Check all ranks against the heartbeat deadline.
    pub fn check_all(&mut self) -> Vec<usize> {
        let now = Instant::now();
        let mut failed = Vec::new();
        for rank in 0..self.world_size {
            if let Some(last) = self.last_heartbeat[rank]
                && now.duration_since(last) > self.heartbeat_deadline
                && self.health[rank] != NodeHealth::Dead
            {
                self.health[rank] = NodeHealth::Suspect;
                failed.push(rank);
            }
        }
        failed
    }

    /// Returns `true` if all ranks are healthy.
    pub fn all_healthy(&self) -> bool {
        self.health.iter().all(|h| *h == NodeHealth::Healthy)
    }

    /// Return indices of dead ranks.
    pub fn dead_ranks(&self) -> Vec<usize> {
        self.health
            .iter()
            .enumerate()
            .filter(|(_, h)| **h == NodeHealth::Dead)
            .map(|(i, _)| i)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DistributedInferenceEngine
// ---------------------------------------------------------------------------

/// Status of the distributed engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineStatus {
    Uninitialized,
    Initializing,
    Ready,
    Running,
    ShuttingDown,
    Failed,
}

/// Metrics collected during distributed inference.
#[derive(Debug, Clone, Default)]
pub struct DistributedMetrics {
    pub total_tokens: u64,
    pub total_microbatches: u64,
    pub allreduce_count: u64,
    pub barrier_count: u64,
    pub failures_detected: u64,
    pub recoveries: u64,
}

/// Orchestrates distributed inference: init group → partition → schedule →
/// execute → aggregate.
#[derive(Debug)]
pub struct DistributedInferenceEngine {
    config: DistributedConfig,
    group: ProcessGroup,
    partitioner: TensorPartitioner,
    scheduler: PipelineScheduler,
    overlap: CommunicationOverlap,
    synchronizer: DistributedSynchronizer,
    fault_detector: FaultDetector,
    status: EngineStatus,
    metrics: DistributedMetrics,
}

impl DistributedInferenceEngine {
    /// Create a new distributed inference engine.
    pub fn new(config: DistributedConfig) -> Result<Self, String> {
        config.validate()?;
        let group = ProcessGroup::new(config.clone())?;
        let partitioner =
            TensorPartitioner::new(config.tensor_parallel_degree, PartitionAxis::Column)?;
        let scheduler = PipelineScheduler::new(ScheduleStrategy::GPipe, 1);
        let overlap = CommunicationOverlap::new(true);
        let synchronizer = DistributedSynchronizer::new(config.world_size)?;
        let fault_detector = FaultDetector::new(config.world_size, config.timeout)?;
        Ok(Self {
            config,
            group,
            partitioner,
            scheduler,
            overlap,
            synchronizer,
            fault_detector,
            status: EngineStatus::Uninitialized,
            metrics: DistributedMetrics::default(),
        })
    }

    pub const fn status(&self) -> EngineStatus {
        self.status
    }

    pub const fn metrics(&self) -> &DistributedMetrics {
        &self.metrics
    }

    pub const fn config(&self) -> &DistributedConfig {
        &self.config
    }

    pub const fn world_size(&self) -> usize {
        self.config.world_size
    }

    pub const fn rank(&self) -> usize {
        self.config.rank
    }

    /// Initialize the engine: set up pipeline stages and mark ranks ready.
    pub fn initialize(&mut self) -> Result<(), String> {
        self.status = EngineStatus::Initializing;

        let total_layers = 32; // simulated model depth
        let layers_per_stage = total_layers / self.config.pipeline_stages;
        for i in 0..self.config.pipeline_stages {
            let start = i * layers_per_stage;
            let end = if i == self.config.pipeline_stages - 1 {
                total_layers
            } else {
                (i + 1) * layers_per_stage
            };
            self.scheduler.add_stage(PipelineStage::new(i, i, start, end));
        }

        for r in 0..self.config.world_size {
            self.group.mark_ready(r)?;
            self.fault_detector.record_heartbeat(r)?;
        }

        self.status = EngineStatus::Ready;
        Ok(())
    }

    /// Run inference on the given input tokens (represented as f32 embeddings).
    pub fn run(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        if self.status != EngineStatus::Ready {
            return Err(format!("engine not ready (status={:?})", self.status));
        }
        self.status = EngineStatus::Running;

        // 1. Partition input across tensor-parallel ranks.
        let partitions = self.partitioner.partition(input)?;
        let local_part = &partitions[self.config.rank % partitions.len()];

        // 2. Overlap: begin compute.
        self.overlap.begin_compute();

        // 3. Run pipeline schedule on local partition.
        let output = self.scheduler.execute(&local_part.data)?;

        // 4. Allreduce aggregation.
        let result = self.group.allreduce(&output, AllReduceOp::Sum)?;
        self.metrics.allreduce_count += 1;

        // 5. Synchronize.
        self.synchronizer.arrive(self.config.rank)?;
        self.metrics.barrier_count += 1;

        self.overlap.complete();
        self.metrics.total_microbatches += 1;
        self.metrics.total_tokens += result.len() as u64;

        self.status = EngineStatus::Ready;
        Ok(result)
    }

    /// Graceful shutdown.
    pub const fn shutdown(&mut self) -> Result<(), String> {
        self.status = EngineStatus::ShuttingDown;
        // In a real implementation: drain queues, disconnect peers.
        self.status = EngineStatus::Uninitialized;
        Ok(())
    }

    /// Access the fault detector.
    pub const fn fault_detector(&self) -> &FaultDetector {
        &self.fault_detector
    }

    /// Access the fault detector mutably.
    pub const fn fault_detector_mut(&mut self) -> &mut FaultDetector {
        &mut self.fault_detector
    }

    /// Access the synchronizer.
    pub const fn synchronizer(&self) -> &DistributedSynchronizer {
        &self.synchronizer
    }

    /// Access the process group.
    pub const fn group(&self) -> &ProcessGroup {
        &self.group
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // DistributedConfig
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_is_valid() {
        let cfg = DistributedConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_zero_world_size() {
        let mut cfg = DistributedConfig::default();
        cfg.world_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rank_out_of_range() {
        let mut cfg = DistributedConfig::default();
        cfg.rank = 5;
        cfg.world_size = 4;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_empty_master_addr() {
        let mut cfg = DistributedConfig::default();
        cfg.master_addr = String::new();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_zero_master_port() {
        let mut cfg = DistributedConfig::default();
        cfg.master_port = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_zero_pipeline_stages() {
        let mut cfg = DistributedConfig::default();
        cfg.pipeline_stages = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_zero_tensor_parallel() {
        let mut cfg = DistributedConfig::default();
        cfg.tensor_parallel_degree = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_world_size_less_than_required() {
        let mut cfg = DistributedConfig::default();
        cfg.world_size = 3;
        cfg.pipeline_stages = 2;
        cfg.tensor_parallel_degree = 2;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_exact_world_size() {
        let mut cfg = DistributedConfig::default();
        cfg.world_size = 4;
        cfg.pipeline_stages = 2;
        cfg.tensor_parallel_degree = 2;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_backend_variants() {
        let cfg = DistributedConfig { backend: CommunicationBackend::Nccl, ..Default::default() };
        assert_eq!(cfg.backend, CommunicationBackend::Nccl);
    }

    // -----------------------------------------------------------------------
    // AllReduceOp
    // -----------------------------------------------------------------------

    #[test]
    fn allreduce_sum() {
        assert_eq!(AllReduceOp::Sum.reduce(3.0, 4.0), 7.0);
    }

    #[test]
    fn allreduce_avg() {
        assert_eq!(AllReduceOp::Avg.reduce(3.0, 5.0), 4.0);
    }

    #[test]
    fn allreduce_max() {
        assert_eq!(AllReduceOp::Max.reduce(3.0, 5.0), 5.0);
    }

    #[test]
    fn allreduce_min() {
        assert_eq!(AllReduceOp::Min.reduce(5.0, 3.0), 3.0);
    }

    #[test]
    fn allreduce_slice_sum() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(AllReduceOp::Sum.reduce_slice(&v), Some(10.0));
    }

    #[test]
    fn allreduce_slice_max() {
        let v = vec![1.0, 5.0, 3.0];
        assert_eq!(AllReduceOp::Max.reduce_slice(&v), Some(5.0));
    }

    #[test]
    fn allreduce_slice_min() {
        let v = vec![4.0, 2.0, 7.0];
        assert_eq!(AllReduceOp::Min.reduce_slice(&v), Some(2.0));
    }

    #[test]
    fn allreduce_slice_empty() {
        assert_eq!(AllReduceOp::Sum.reduce_slice(&[]), None);
    }

    #[test]
    fn allreduce_slice_single() {
        assert_eq!(AllReduceOp::Avg.reduce_slice(&[42.0]), Some(42.0));
    }

    // -----------------------------------------------------------------------
    // ProcessGroup
    // -----------------------------------------------------------------------

    fn make_group(world_size: usize, rank: usize) -> ProcessGroup {
        let cfg = DistributedConfig { world_size, rank, ..Default::default() };
        ProcessGroup::new(cfg).unwrap()
    }

    #[test]
    fn group_creation() {
        let g = make_group(4, 0);
        assert_eq!(g.world_size(), 4);
        assert_eq!(g.rank(), 0);
    }

    #[test]
    fn group_invalid_config() {
        let cfg = DistributedConfig { world_size: 0, ..Default::default() };
        assert!(ProcessGroup::new(cfg).is_err());
    }

    #[test]
    fn group_initial_status_is_initializing() {
        let g = make_group(2, 0);
        assert_eq!(g.rank_status(0), Some(RankStatus::Initializing));
    }

    #[test]
    fn group_mark_ready() {
        let mut g = make_group(2, 0);
        g.mark_ready(0).unwrap();
        assert_eq!(g.rank_status(0), Some(RankStatus::Ready));
    }

    #[test]
    fn group_mark_failed() {
        let mut g = make_group(2, 0);
        g.mark_failed(1).unwrap();
        assert_eq!(g.rank_status(1), Some(RankStatus::Failed));
    }

    #[test]
    fn group_mark_out_of_range() {
        let mut g = make_group(2, 0);
        assert!(g.mark_ready(5).is_err());
    }

    #[test]
    fn group_all_ready() {
        let mut g = make_group(3, 0);
        assert!(!g.all_ready());
        for r in 0..3 {
            g.mark_ready(r).unwrap();
        }
        assert!(g.all_ready());
    }

    #[test]
    fn group_count_status() {
        let mut g = make_group(4, 0);
        g.mark_ready(0).unwrap();
        g.mark_ready(1).unwrap();
        assert_eq!(g.count_status(RankStatus::Ready), 2);
        assert_eq!(g.count_status(RankStatus::Initializing), 2);
    }

    #[test]
    fn group_allreduce() {
        let mut g = make_group(2, 0);
        let result = g.allreduce(&[1.0, 2.0], AllReduceOp::Sum).unwrap();
        assert_eq!(result, vec![1.0, 2.0]);
        assert_eq!(g.collective_count(), 1);
    }

    #[test]
    fn group_allreduce_empty() {
        let mut g = make_group(2, 0);
        assert!(g.allreduce(&[], AllReduceOp::Sum).is_err());
    }

    #[test]
    fn group_broadcast() {
        let mut g = make_group(2, 0);
        let r = g.broadcast(&[3.0, 4.0], 0).unwrap();
        assert_eq!(r, vec![3.0, 4.0]);
    }

    #[test]
    fn group_broadcast_invalid_root() {
        let mut g = make_group(2, 0);
        assert!(g.broadcast(&[1.0], 5).is_err());
    }

    #[test]
    fn group_scatter() {
        let mut g = make_group(2, 0);
        let r = g.scatter(&[1.0, 2.0, 3.0, 4.0], 0).unwrap();
        assert_eq!(r, vec![1.0, 2.0]);
    }

    #[test]
    fn group_scatter_indivisible() {
        let mut g = make_group(3, 0);
        assert!(g.scatter(&[1.0, 2.0], 0).is_err());
    }

    #[test]
    fn group_scatter_invalid_root() {
        let mut g = make_group(2, 0);
        assert!(g.scatter(&[1.0, 2.0], 9).is_err());
    }

    #[test]
    fn group_gather() {
        let mut g = make_group(2, 0);
        let r = g.gather(&[5.0], 0).unwrap();
        assert_eq!(r, vec![5.0, 5.0]);
    }

    #[test]
    fn group_gather_invalid_root() {
        let mut g = make_group(2, 0);
        assert!(g.gather(&[1.0], 9).is_err());
    }

    #[test]
    fn group_collective_count_increments() {
        let mut g = make_group(2, 0);
        g.allreduce(&[1.0], AllReduceOp::Sum).unwrap();
        g.broadcast(&[1.0], 0).unwrap();
        g.scatter(&[1.0, 2.0], 0).unwrap();
        g.gather(&[1.0], 0).unwrap();
        assert_eq!(g.collective_count(), 4);
    }

    #[test]
    fn group_backend() {
        let g = make_group(1, 0);
        assert_eq!(g.backend(), CommunicationBackend::Tcp);
    }

    // -----------------------------------------------------------------------
    // TensorPartitioner
    // -----------------------------------------------------------------------

    #[test]
    fn partitioner_creation() {
        let p = TensorPartitioner::new(4, PartitionAxis::Row).unwrap();
        assert_eq!(p.world_size(), 4);
        assert_eq!(p.axis(), PartitionAxis::Row);
    }

    #[test]
    fn partitioner_zero_world_size() {
        assert!(TensorPartitioner::new(0, PartitionAxis::Row).is_err());
    }

    #[test]
    fn partitioner_even_split() {
        let p = TensorPartitioner::new(2, PartitionAxis::Row).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let parts = p.partition(&data).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].data, vec![1.0, 2.0]);
        assert_eq!(parts[1].data, vec![3.0, 4.0]);
    }

    #[test]
    fn partitioner_uneven_split() {
        let p = TensorPartitioner::new(3, PartitionAxis::Column).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let parts = p.partition(&data).unwrap();
        assert_eq!(parts[0].data, vec![1.0, 2.0]);
        assert_eq!(parts[1].data, vec![3.0, 4.0]);
        assert_eq!(parts[2].data, vec![5.0]);
    }

    #[test]
    fn partitioner_too_small() {
        let p = TensorPartitioner::new(5, PartitionAxis::Row).unwrap();
        assert!(p.partition(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn partitioner_reconstruct() {
        let p = TensorPartitioner::new(2, PartitionAxis::Row).unwrap();
        let data = vec![10.0, 20.0, 30.0, 40.0];
        let parts = p.partition(&data).unwrap();
        let reconstructed = p.reconstruct(&parts).unwrap();
        assert_eq!(reconstructed, data);
    }

    #[test]
    fn partitioner_reconstruct_wrong_count() {
        let p = TensorPartitioner::new(3, PartitionAxis::Row).unwrap();
        let parts = vec![TensorPartition {
            rank: 0,
            axis: PartitionAxis::Row,
            offset: 0,
            length: 2,
            data: vec![1.0, 2.0],
        }];
        assert!(p.reconstruct(&parts).is_err());
    }

    #[test]
    fn partitioner_for_rank() {
        let p = TensorPartitioner::new(2, PartitionAxis::Row).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let part = p.partition_for_rank(&data, 1).unwrap();
        assert_eq!(part.rank, 1);
        assert_eq!(part.data, vec![3.0, 4.0]);
    }

    #[test]
    fn partitioner_for_rank_out_of_range() {
        let p = TensorPartitioner::new(2, PartitionAxis::Row).unwrap();
        assert!(p.partition_for_rank(&[1.0, 2.0], 5).is_err());
    }

    #[test]
    fn partitioner_single_rank() {
        let p = TensorPartitioner::new(1, PartitionAxis::Row).unwrap();
        let data = vec![1.0, 2.0, 3.0];
        let parts = p.partition(&data).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].data, data);
    }

    #[test]
    fn partitioner_offsets_correct() {
        let p = TensorPartitioner::new(3, PartitionAxis::Row).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let parts = p.partition(&data).unwrap();
        assert_eq!(parts[0].offset, 0);
        assert_eq!(parts[1].offset, 2);
        assert_eq!(parts[2].offset, 4);
    }

    // -----------------------------------------------------------------------
    // PipelineStage
    // -----------------------------------------------------------------------

    #[test]
    fn stage_creation() {
        let s = PipelineStage::new(0, 0, 0, 8);
        assert_eq!(s.status(), StageStatus::Idle);
        assert_eq!(s.layer_count(), 8);
    }

    #[test]
    fn stage_forward() {
        let mut s = PipelineStage::new(1, 0, 0, 8);
        let out = s.forward(&[1.0, 2.0]).unwrap();
        // stage_id=1 → scale=2.0
        assert_eq!(out, vec![2.0, 4.0]);
        assert_eq!(s.status(), StageStatus::Completed);
    }

    #[test]
    fn stage_forward_empty() {
        let mut s = PipelineStage::new(0, 0, 0, 4);
        assert!(s.forward(&[]).is_err());
    }

    #[test]
    fn stage_microbatch_count() {
        let mut s = PipelineStage::new(0, 0, 0, 4);
        s.forward(&[1.0]).unwrap();
        s.forward(&[2.0]).unwrap();
        assert_eq!(s.microbatches_processed(), 2);
    }

    #[test]
    fn stage_reset() {
        let mut s = PipelineStage::new(0, 0, 0, 4);
        s.forward(&[1.0]).unwrap();
        s.reset();
        assert_eq!(s.status(), StageStatus::Idle);
        assert!(s.activations().is_empty());
    }

    #[test]
    fn stage_activations_stored() {
        let mut s = PipelineStage::new(0, 0, 0, 4);
        s.forward(&[3.0, 4.0]).unwrap();
        assert_eq!(s.activations(), &[3.0, 4.0]);
    }

    #[test]
    fn stage_layer_count_saturating() {
        let s = PipelineStage::new(0, 0, 10, 5);
        assert_eq!(s.layer_count(), 0);
    }

    // -----------------------------------------------------------------------
    // PipelineScheduler
    // -----------------------------------------------------------------------

    #[test]
    fn scheduler_gpipe_schedule() {
        let mut sched = PipelineScheduler::new(ScheduleStrategy::GPipe, 2);
        sched.add_stage(PipelineStage::new(0, 0, 0, 4));
        sched.add_stage(PipelineStage::new(1, 1, 4, 8));
        let log = sched.build_schedule().unwrap();
        // GPipe: 2 mbs × 2 stages forward + 2 mbs × 2 stages backward = 8
        assert_eq!(log.len(), 8);
    }

    #[test]
    fn scheduler_gpipe_forward_first() {
        let mut sched = PipelineScheduler::new(ScheduleStrategy::GPipe, 1);
        sched.add_stage(PipelineStage::new(0, 0, 0, 4));
        let log = sched.build_schedule().unwrap();
        assert_eq!(log[0].direction, Direction::Forward);
        assert_eq!(log[1].direction, Direction::Backward);
    }

    #[test]
    fn scheduler_1f1b_schedule() {
        let mut sched = PipelineScheduler::new(ScheduleStrategy::OneFOnB, 2);
        sched.add_stage(PipelineStage::new(0, 0, 0, 4));
        let log = sched.build_schedule().unwrap();
        // mb0 fwd(1) + mb1 fwd(1) + mb0 bwd(1) + mb1 bwd(1) = 4
        assert_eq!(log.len(), 4);
    }

    #[test]
    fn scheduler_no_stages() {
        let mut sched = PipelineScheduler::new(ScheduleStrategy::GPipe, 1);
        assert!(sched.build_schedule().is_err());
    }

    #[test]
    fn scheduler_zero_microbatches() {
        let mut sched = PipelineScheduler::new(ScheduleStrategy::GPipe, 0);
        sched.add_stage(PipelineStage::new(0, 0, 0, 4));
        assert!(sched.build_schedule().is_err());
    }

    #[test]
    fn scheduler_execute() {
        let mut sched = PipelineScheduler::new(ScheduleStrategy::GPipe, 1);
        sched.add_stage(PipelineStage::new(0, 0, 0, 4));
        sched.add_stage(PipelineStage::new(1, 1, 4, 8));
        let out = sched.execute(&[1.0, 2.0]).unwrap();
        // stage0 scale=1, stage1 scale=2 → [1*1*2, 2*1*2] = [2.0, 4.0]
        assert_eq!(out, vec![2.0, 4.0]);
    }

    #[test]
    fn scheduler_strategy_accessor() {
        let sched = PipelineScheduler::new(ScheduleStrategy::OneFOnB, 3);
        assert_eq!(sched.strategy(), ScheduleStrategy::OneFOnB);
        assert_eq!(sched.num_microbatches(), 3);
    }

    #[test]
    fn scheduler_num_stages() {
        let mut sched = PipelineScheduler::new(ScheduleStrategy::GPipe, 1);
        assert_eq!(sched.num_stages(), 0);
        sched.add_stage(PipelineStage::new(0, 0, 0, 4));
        assert_eq!(sched.num_stages(), 1);
    }

    #[test]
    fn scheduler_schedule_log_persists() {
        let mut sched = PipelineScheduler::new(ScheduleStrategy::GPipe, 1);
        sched.add_stage(PipelineStage::new(0, 0, 0, 4));
        sched.build_schedule().unwrap();
        assert!(!sched.schedule_log().is_empty());
    }

    // -----------------------------------------------------------------------
    // CommunicationOverlap
    // -----------------------------------------------------------------------

    #[test]
    fn overlap_default_idle() {
        let o = CommunicationOverlap::new(true);
        assert_eq!(o.phase(), OverlapPhase::Idle);
    }

    #[test]
    fn overlap_compute() {
        let mut o = CommunicationOverlap::new(true);
        o.begin_compute();
        assert_eq!(o.phase(), OverlapPhase::ComputeOnly);
        assert_eq!(o.compute_ops(), 1);
    }

    #[test]
    fn overlap_comm() {
        let mut o = CommunicationOverlap::new(true);
        o.begin_comm();
        assert_eq!(o.phase(), OverlapPhase::CommOnly);
        assert_eq!(o.comm_ops(), 1);
    }

    #[test]
    fn overlap_overlapped() {
        let mut o = CommunicationOverlap::new(true);
        o.begin_overlap().unwrap();
        assert_eq!(o.phase(), OverlapPhase::Overlapped);
    }

    #[test]
    fn overlap_disabled() {
        let mut o = CommunicationOverlap::new(false);
        assert!(!o.is_enabled());
        assert!(o.begin_overlap().is_err());
    }

    #[test]
    fn overlap_complete() {
        let mut o = CommunicationOverlap::new(true);
        o.begin_compute();
        o.complete();
        assert_eq!(o.phase(), OverlapPhase::Idle);
    }

    #[test]
    fn overlap_ratio_empty() {
        let o = CommunicationOverlap::new(true);
        assert_eq!(o.overlap_ratio(), 0.0);
    }

    #[test]
    fn overlap_ratio_all_overlapped() {
        let mut o = CommunicationOverlap::new(true);
        o.begin_overlap().unwrap();
        o.complete();
        assert!((o.overlap_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn overlap_ratio_mixed() {
        let mut o = CommunicationOverlap::new(true);
        o.begin_compute();
        o.complete();
        o.begin_overlap().unwrap();
        o.complete();
        // 1 compute + 1 overlap = ratio 0.5
        assert!((o.overlap_ratio() - 0.5).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // DistributedSynchronizer
    // -----------------------------------------------------------------------

    #[test]
    fn sync_creation() {
        let s = DistributedSynchronizer::new(4).unwrap();
        assert_eq!(s.world_size(), 4);
    }

    #[test]
    fn sync_zero_world_size() {
        assert!(DistributedSynchronizer::new(0).is_err());
    }

    #[test]
    fn sync_arrive_partial() {
        let mut s = DistributedSynchronizer::new(3).unwrap();
        assert!(!s.arrive(0).unwrap());
        assert_eq!(s.arrived_count(), 1);
    }

    #[test]
    fn sync_arrive_all() {
        let mut s = DistributedSynchronizer::new(2).unwrap();
        assert!(!s.arrive(0).unwrap());
        assert!(s.arrive(1).unwrap());
        assert_eq!(s.barrier_count(), 1);
    }

    #[test]
    fn sync_arrive_out_of_range() {
        let mut s = DistributedSynchronizer::new(2).unwrap();
        assert!(s.arrive(5).is_err());
    }

    #[test]
    fn sync_barrier_resets() {
        let mut s = DistributedSynchronizer::new(2).unwrap();
        s.arrive(0).unwrap();
        s.arrive(1).unwrap();
        assert_eq!(s.arrived_count(), 0);
    }

    #[test]
    fn sync_allgather() {
        let mut s = DistributedSynchronizer::new(2).unwrap();
        let result = s.allgather(&[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.allgather_count(), 1);
    }

    #[test]
    fn sync_allgather_wrong_count() {
        let mut s = DistributedSynchronizer::new(3).unwrap();
        assert!(s.allgather(&[vec![1.0]]).is_err());
    }

    #[test]
    fn sync_multiple_barriers() {
        let mut s = DistributedSynchronizer::new(2).unwrap();
        for _ in 0..3 {
            s.arrive(0).unwrap();
            s.arrive(1).unwrap();
        }
        assert_eq!(s.barrier_count(), 3);
    }

    // -----------------------------------------------------------------------
    // FaultDetector
    // -----------------------------------------------------------------------

    #[test]
    fn fault_creation() {
        let f = FaultDetector::new(4, Duration::from_secs(5)).unwrap();
        assert_eq!(f.world_size(), 4);
        assert!(f.all_healthy());
    }

    #[test]
    fn fault_zero_world_size() {
        assert!(FaultDetector::new(0, Duration::from_secs(1)).is_err());
    }

    #[test]
    fn fault_mark_dead() {
        let mut f = FaultDetector::new(2, Duration::from_secs(5)).unwrap();
        f.mark_dead(1).unwrap();
        assert_eq!(f.node_health(1), Some(NodeHealth::Dead));
        assert!(!f.all_healthy());
        assert_eq!(f.failure_count(), 1);
    }

    #[test]
    fn fault_mark_dead_idempotent() {
        let mut f = FaultDetector::new(2, Duration::from_secs(5)).unwrap();
        f.mark_dead(0).unwrap();
        f.mark_dead(0).unwrap();
        assert_eq!(f.failure_count(), 1);
    }

    #[test]
    fn fault_mark_suspect() {
        let mut f = FaultDetector::new(2, Duration::from_secs(5)).unwrap();
        f.mark_suspect(0).unwrap();
        assert_eq!(f.node_health(0), Some(NodeHealth::Suspect));
    }

    #[test]
    fn fault_mark_out_of_range() {
        let mut f = FaultDetector::new(2, Duration::from_secs(5)).unwrap();
        assert!(f.mark_dead(5).is_err());
        assert!(f.mark_suspect(5).is_err());
        assert!(f.record_heartbeat(5).is_err());
    }

    #[test]
    fn fault_heartbeat_recovery() {
        let mut f = FaultDetector::new(2, Duration::from_secs(5)).unwrap();
        f.mark_dead(0).unwrap();
        f.record_heartbeat(0).unwrap();
        assert_eq!(f.node_health(0), Some(NodeHealth::Healthy));
        assert_eq!(f.recovery_count(), 1);
    }

    #[test]
    fn fault_dead_ranks() {
        let mut f = FaultDetector::new(4, Duration::from_secs(5)).unwrap();
        f.mark_dead(1).unwrap();
        f.mark_dead(3).unwrap();
        assert_eq!(f.dead_ranks(), vec![1, 3]);
    }

    #[test]
    fn fault_healthy_count() {
        let mut f = FaultDetector::new(3, Duration::from_secs(5)).unwrap();
        f.mark_dead(2).unwrap();
        assert_eq!(f.healthy_count(), 2);
    }

    #[test]
    fn fault_check_all_no_heartbeats() {
        let mut f = FaultDetector::new(2, Duration::from_secs(5)).unwrap();
        // No heartbeats recorded → check_all returns empty (no last_heartbeat set).
        let suspects = f.check_all();
        assert!(suspects.is_empty());
    }

    // -----------------------------------------------------------------------
    // DistributedInferenceEngine
    // -----------------------------------------------------------------------

    fn make_engine(world_size: usize, rank: usize) -> DistributedInferenceEngine {
        let cfg = DistributedConfig { world_size, rank, ..Default::default() };
        DistributedInferenceEngine::new(cfg).unwrap()
    }

    #[test]
    fn engine_creation() {
        let e = make_engine(1, 0);
        assert_eq!(e.status(), EngineStatus::Uninitialized);
        assert_eq!(e.world_size(), 1);
    }

    #[test]
    fn engine_invalid_config() {
        let cfg = DistributedConfig { world_size: 0, ..Default::default() };
        assert!(DistributedInferenceEngine::new(cfg).is_err());
    }

    #[test]
    fn engine_initialize() {
        let mut e = make_engine(1, 0);
        e.initialize().unwrap();
        assert_eq!(e.status(), EngineStatus::Ready);
    }

    #[test]
    fn engine_run_before_init() {
        let mut e = make_engine(1, 0);
        assert!(e.run(&[1.0]).is_err());
    }

    #[test]
    fn engine_run() {
        let mut e = make_engine(1, 0);
        e.initialize().unwrap();
        let out = e.run(&[1.0, 2.0]).unwrap();
        assert!(!out.is_empty());
        assert_eq!(e.metrics().total_microbatches, 1);
    }

    #[test]
    fn engine_multiple_runs() {
        let mut e = make_engine(1, 0);
        e.initialize().unwrap();
        e.run(&[1.0]).unwrap();
        e.run(&[2.0]).unwrap();
        assert_eq!(e.metrics().total_microbatches, 2);
    }

    #[test]
    fn engine_shutdown() {
        let mut e = make_engine(1, 0);
        e.initialize().unwrap();
        e.shutdown().unwrap();
        assert_eq!(e.status(), EngineStatus::Uninitialized);
    }

    #[test]
    fn engine_run_after_shutdown() {
        let mut e = make_engine(1, 0);
        e.initialize().unwrap();
        e.shutdown().unwrap();
        assert!(e.run(&[1.0]).is_err());
    }

    #[test]
    fn engine_fault_detector_access() {
        let e = make_engine(2, 0);
        assert!(e.fault_detector().all_healthy());
    }

    #[test]
    fn engine_fault_detector_mut() {
        let mut e = make_engine(2, 0);
        e.fault_detector_mut().mark_dead(1).unwrap();
        assert_eq!(e.fault_detector().failure_count(), 1);
    }

    #[test]
    fn engine_synchronizer_access() {
        let e = make_engine(2, 0);
        assert_eq!(e.synchronizer().world_size(), 2);
    }

    #[test]
    fn engine_group_access() {
        let e = make_engine(2, 0);
        assert_eq!(e.group().world_size(), 2);
    }

    #[test]
    fn engine_metrics_initial() {
        let e = make_engine(1, 0);
        let m = e.metrics();
        assert_eq!(m.total_tokens, 0);
        assert_eq!(m.allreduce_count, 0);
    }

    #[test]
    fn engine_metrics_after_run() {
        let mut e = make_engine(1, 0);
        e.initialize().unwrap();
        e.run(&[1.0, 2.0]).unwrap();
        assert!(e.metrics().total_tokens > 0);
        assert_eq!(e.metrics().allreduce_count, 1);
        assert_eq!(e.metrics().barrier_count, 1);
    }

    #[test]
    fn engine_config_access() {
        let e = make_engine(4, 2);
        assert_eq!(e.config().world_size, 4);
        assert_eq!(e.rank(), 2);
    }
}
