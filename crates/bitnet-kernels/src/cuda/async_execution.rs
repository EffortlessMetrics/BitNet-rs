//! CUDA asynchronous execution engine for overlapping compute and data transfer.
//!
//! # Overview
//!
//! Provides an async execution framework that manages multiple CUDA streams to
//! overlap kernel execution with host↔device memory transfers.  Key components:
//!
//! - [`AsyncExecutionEngine`] — manages scheduling, execution, and completion
//!   tracking across multiple CUDA streams.
//! - [`StreamPool`] — pool of CUDA streams with round-robin and priority-based
//!   allocation strategies.
//! - [`AsyncOperation`] — trait for operations that can be scheduled for
//!   asynchronous execution on a CUDA stream.
//! - [`EventSync`] — CUDA event-based synchronization primitives for
//!   inter-stream ordering and host↔device coordination.
//! - [`PipelinedExecution`] — overlaps kernel execution with data transfer
//!   across dedicated streams for maximum throughput.
//!
//! All GPU dispatch is feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations execute operations sequentially so that
//! the full API surface is testable on non-GPU hosts.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bitnet_common::{KernelError, Result};

// ── Identifiers ──────────────────────────────────────────────────────

static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_EVENT_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_OP_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_ENGINE_ID: AtomicU64 = AtomicU64::new(1);

fn next_stream_id() -> u64 {
    NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed)
}

fn next_event_id() -> u64 {
    NEXT_EVENT_ID.fetch_add(1, Ordering::Relaxed)
}

fn next_op_id() -> u64 {
    NEXT_OP_ID.fetch_add(1, Ordering::Relaxed)
}

fn next_engine_id() -> u64 {
    NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed)
}

// ── StreamPriority ───────────────────────────────────────────────────

/// Priority level for a CUDA stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StreamPriority {
    /// Low priority — background or best-effort work.
    Low,
    /// Normal (default) priority.
    #[default]
    Normal,
    /// High priority — latency-sensitive work.
    High,
    /// Critical — real-time inference path.
    Critical,
}

impl StreamPriority {
    /// Map to a CUDA-compatible numeric priority (lower = higher priority).
    pub fn as_cuda_priority(self) -> i32 {
        match self {
            Self::Low => 0,
            Self::Normal => -1,
            Self::High => -2,
            Self::Critical => -3,
        }
    }
}

impl fmt::Display for StreamPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Normal => write!(f, "normal"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

// ── StreamAllocationStrategy ─────────────────────────────────────────

/// Strategy for allocating streams from the pool.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StreamAllocationStrategy {
    /// Cycle through streams in order.
    #[default]
    RoundRobin,
    /// Pick the stream with the fewest pending operations.
    LeastLoaded,
    /// Select based on operation priority.
    PriorityBased,
    /// Dedicate streams by transfer direction (H2D / compute / D2H).
    DedicatedByKind,
}

// ── StreamInfo ───────────────────────────────────────────────────────

/// Metadata for a single logical CUDA stream.
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Unique stream identifier.
    pub id: u64,
    /// Priority of this stream.
    pub priority: StreamPriority,
    /// Whether all enqueued work has completed.
    pub synchronized: bool,
    /// Cumulative number of operations dispatched.
    pub ops_dispatched: u64,
    /// Number of operations currently pending.
    pub pending_ops: u64,
    /// Creation timestamp.
    pub created_at: Instant,
    /// Optional label for debugging.
    pub label: Option<String>,
}

impl StreamInfo {
    /// Create a new stream with the given priority.
    pub fn new(priority: StreamPriority) -> Self {
        Self {
            id: next_stream_id(),
            priority,
            synchronized: true,
            ops_dispatched: 0,
            pending_ops: 0,
            created_at: Instant::now(),
            label: None,
        }
    }

    /// Create a labelled stream.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Mark the stream as having pending work.
    pub fn mark_dirty(&mut self) {
        self.synchronized = false;
        self.ops_dispatched += 1;
        self.pending_ops += 1;
    }

    /// Mark the stream as fully synchronised.
    pub fn mark_synchronized(&mut self) {
        self.synchronized = true;
        self.pending_ops = 0;
    }

    /// Complete one pending operation.
    pub fn complete_one(&mut self) {
        self.pending_ops = self.pending_ops.saturating_sub(1);
        if self.pending_ops == 0 {
            self.synchronized = true;
        }
    }
}

// ── StreamPool ───────────────────────────────────────────────────────

/// Configuration for the stream pool.
#[derive(Debug, Clone)]
pub struct StreamPoolConfig {
    /// Number of streams.
    pub num_streams: usize,
    /// Default priority for new streams.
    pub default_priority: StreamPriority,
    /// Allocation strategy.
    pub strategy: StreamAllocationStrategy,
    /// Enable per-stream profiling.
    pub enable_profiling: bool,
}

impl Default for StreamPoolConfig {
    fn default() -> Self {
        Self {
            num_streams: 4,
            default_priority: StreamPriority::Normal,
            strategy: StreamAllocationStrategy::RoundRobin,
            enable_profiling: false,
        }
    }
}

impl StreamPoolConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.num_streams == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_streams must be at least 1".into(),
            }
            .into());
        }
        if self.num_streams > 256 {
            return Err(KernelError::InvalidArguments {
                reason: "num_streams must not exceed 256".into(),
            }
            .into());
        }
        Ok(())
    }
}

/// Pool of CUDA streams with configurable allocation strategies.
///
/// On CPU this is a logical wrapper that executes all work sequentially
/// but maintains the same API for device-agnostic callers.
#[derive(Debug)]
pub struct StreamPool {
    config: StreamPoolConfig,
    streams: Vec<StreamInfo>,
    /// Round-robin counter.
    rr_index: usize,
}

impl StreamPool {
    /// Create a new stream pool from the given config.
    pub fn new(config: StreamPoolConfig) -> Result<Self> {
        config.validate()?;
        let streams =
            (0..config.num_streams).map(|_| StreamInfo::new(config.default_priority)).collect();
        Ok(Self { config, streams, rr_index: 0 })
    }

    /// Create a pool with default settings.
    pub fn with_defaults() -> Result<Self> {
        Self::new(StreamPoolConfig::default())
    }

    /// Number of streams in the pool.
    pub fn num_streams(&self) -> usize {
        self.streams.len()
    }

    /// Get a reference to a stream by index.
    pub fn stream(&self, index: usize) -> Result<&StreamInfo> {
        self.streams.get(index).ok_or_else(|| {
            KernelError::InvalidArguments {
                reason: format!(
                    "stream index {} out of range (pool has {})",
                    index,
                    self.streams.len()
                ),
            }
            .into()
        })
    }

    /// Get a mutable reference to a stream by index.
    pub fn stream_mut(&mut self, index: usize) -> Result<&mut StreamInfo> {
        let len = self.streams.len();
        self.streams.get_mut(index).ok_or_else(|| {
            KernelError::InvalidArguments {
                reason: format!("stream index {index} out of range (pool has {len})"),
            }
            .into()
        })
    }

    /// Acquire the next stream using the configured strategy.
    pub fn acquire(&mut self) -> usize {
        match self.config.strategy {
            StreamAllocationStrategy::RoundRobin => self.acquire_round_robin(),
            StreamAllocationStrategy::LeastLoaded => self.acquire_least_loaded(),
            StreamAllocationStrategy::PriorityBased => self.acquire_highest_priority(),
            StreamAllocationStrategy::DedicatedByKind => {
                // Fallback to round-robin; callers use acquire_for_kind instead.
                self.acquire_round_robin()
            }
        }
    }

    /// Round-robin stream selection.
    pub fn acquire_round_robin(&mut self) -> usize {
        let idx = self.rr_index % self.streams.len();
        self.rr_index = self.rr_index.wrapping_add(1);
        idx
    }

    /// Select stream with fewest pending operations.
    pub fn acquire_least_loaded(&self) -> usize {
        self.streams
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.pending_ops)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Select stream with highest priority.
    pub fn acquire_highest_priority(&self) -> usize {
        self.streams.iter().enumerate().max_by_key(|(_, s)| s.priority).map(|(i, _)| i).unwrap_or(0)
    }

    /// Select a stream dedicated to a specific pipeline stage kind.
    ///
    /// Convention: stream 0 = H2D, stream 1 = compute, stream 2+ = D2H.
    pub fn acquire_for_kind(&self, kind: PipelineStageKind) -> usize {
        let n = self.streams.len();
        match kind {
            PipelineStageKind::HostToDevice => 0,
            PipelineStageKind::Compute => 1.min(n - 1),
            PipelineStageKind::DeviceToHost => {
                if n > 2 {
                    2
                } else {
                    0
                }
            }
        }
    }

    /// Synchronise a single stream.
    pub fn sync_stream(&mut self, index: usize) -> Result<()> {
        self.stream_mut(index)?.mark_synchronized();
        Ok(())
    }

    /// Synchronise all streams.
    pub fn sync_all(&mut self) -> Result<()> {
        for s in &mut self.streams {
            s.mark_synchronized();
        }
        Ok(())
    }

    /// Return the pool configuration.
    pub fn config(&self) -> &StreamPoolConfig {
        &self.config
    }

    /// All stream handles.
    pub fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    /// Total operations dispatched across all streams.
    pub fn total_ops_dispatched(&self) -> u64 {
        self.streams.iter().map(|s| s.ops_dispatched).sum()
    }

    /// Total pending operations across all streams.
    pub fn total_pending_ops(&self) -> u64 {
        self.streams.iter().map(|s| s.pending_ops).sum()
    }

    /// Whether every stream is synchronised.
    pub fn all_synchronized(&self) -> bool {
        self.streams.iter().all(|s| s.synchronized)
    }

    /// Reset all stream counters (for benchmarking epochs).
    pub fn reset_counters(&mut self) {
        for s in &mut self.streams {
            s.ops_dispatched = 0;
            s.pending_ops = 0;
            s.synchronized = true;
        }
        self.rr_index = 0;
    }
}

// ── EventSync ────────────────────────────────────────────────────────

/// State of a synchronisation event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventState {
    /// Event created but not yet recorded on any stream.
    Created,
    /// Event recorded on a stream; work may still be pending.
    Recorded,
    /// All work preceding the event has completed.
    Signalled,
}

/// CUDA event-based synchronisation primitive.
#[derive(Debug, Clone)]
pub struct CudaEvent {
    /// Unique event identifier.
    pub id: u64,
    /// Current state.
    pub state: EventState,
    /// Stream on which the event was recorded.
    pub stream_id: Option<u64>,
    /// Timestamp when the event was recorded.
    pub recorded_at: Option<Instant>,
    /// Timestamp when the event was signalled.
    pub signalled_at: Option<Instant>,
}

impl CudaEvent {
    /// Create a new event in the `Created` state.
    pub fn new() -> Self {
        Self {
            id: next_event_id(),
            state: EventState::Created,
            stream_id: None,
            recorded_at: None,
            signalled_at: None,
        }
    }

    /// Whether the event has been signalled.
    pub fn is_signalled(&self) -> bool {
        self.state == EventState::Signalled
    }

    /// Elapsed time between recording and signalling (if both occurred).
    pub fn elapsed(&self) -> Option<Duration> {
        match (self.recorded_at, self.signalled_at) {
            (Some(r), Some(s)) => Some(s.duration_since(r)),
            _ => None,
        }
    }
}

impl Default for CudaEvent {
    fn default() -> Self {
        Self::new()
    }
}

/// Manager for CUDA events used for inter-stream synchronisation.
#[derive(Debug)]
pub struct EventSync {
    events: HashMap<u64, CudaEvent>,
}

impl EventSync {
    /// Create a new event manager.
    pub fn new() -> Self {
        Self { events: HashMap::new() }
    }

    /// Create and register a new event.
    pub fn create_event(&mut self) -> &CudaEvent {
        let event = CudaEvent::new();
        let id = event.id;
        self.events.insert(id, event);
        self.events.get(&id).unwrap()
    }

    /// Get an event by id.
    pub fn get(&self, id: u64) -> Result<&CudaEvent> {
        self.events.get(&id).ok_or_else(|| {
            KernelError::InvalidArguments { reason: format!("event {id} not found") }.into()
        })
    }

    /// Record an event on a stream (CPU fallback: immediately signals).
    pub fn record(&mut self, event_id: u64, stream: &StreamInfo) -> Result<()> {
        let event = self.events.get_mut(&event_id).ok_or_else(|| {
            KernelError::InvalidArguments { reason: format!("event {event_id} not found") }
        })?;
        event.stream_id = Some(stream.id);
        event.recorded_at = Some(Instant::now());
        event.state = EventState::Recorded;
        // CPU fallback: work is sequential so immediately signalled.
        event.signalled_at = Some(Instant::now());
        event.state = EventState::Signalled;
        Ok(())
    }

    /// Wait for an event to be signalled (CPU fallback: check state).
    pub fn wait(&self, event_id: u64) -> Result<()> {
        let event = self.get(event_id)?;
        if !event.is_signalled() {
            return Err(KernelError::GpuError {
                reason: format!("event {event_id} not yet signalled"),
            }
            .into());
        }
        Ok(())
    }

    /// Wait for multiple events.
    pub fn wait_all(&self, event_ids: &[u64]) -> Result<()> {
        for &id in event_ids {
            self.wait(id)?;
        }
        Ok(())
    }

    /// Destroy an event.
    pub fn destroy(&mut self, event_id: u64) -> Result<()> {
        self.events.remove(&event_id).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("event {event_id} not found"),
        })?;
        Ok(())
    }

    /// Number of live events.
    pub fn num_events(&self) -> usize {
        self.events.len()
    }

    /// Number of signalled events.
    pub fn num_signalled(&self) -> usize {
        self.events.values().filter(|e| e.is_signalled()).count()
    }

    /// Destroy all signalled events (garbage collection).
    pub fn gc_signalled(&mut self) -> usize {
        let before = self.events.len();
        self.events.retain(|_, e| !e.is_signalled());
        before - self.events.len()
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

impl Default for EventSync {
    fn default() -> Self {
        Self::new()
    }
}

// ── AsyncOperation ───────────────────────────────────────────────────

/// Category of an async operation for scheduling purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationKind {
    /// Host-to-device data transfer.
    HostToDevice,
    /// Device-to-host data transfer.
    DeviceToHost,
    /// Device-to-device data transfer.
    DeviceToDevice,
    /// Compute kernel execution.
    Compute,
    /// Synchronisation barrier.
    Barrier,
}

impl fmt::Display for OperationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "H2D"),
            Self::DeviceToHost => write!(f, "D2H"),
            Self::DeviceToDevice => write!(f, "D2D"),
            Self::Compute => write!(f, "compute"),
            Self::Barrier => write!(f, "barrier"),
        }
    }
}

/// Trait for operations that can be scheduled for async execution.
///
/// Implementors describe *what* to execute; the engine decides *when* and
/// *where* (which stream).
pub trait AsyncOperation: fmt::Debug {
    /// Unique operation identifier.
    fn op_id(&self) -> u64;

    /// Human-readable label.
    fn label(&self) -> &str;

    /// The kind of operation (for stream assignment).
    fn kind(&self) -> OperationKind;

    /// Priority of this operation.
    fn priority(&self) -> StreamPriority {
        StreamPriority::Normal
    }

    /// Estimated cost in arbitrary units (for load balancing).
    fn estimated_cost(&self) -> u64 {
        1
    }

    /// Event ids that must be signalled before this operation starts.
    fn dependencies(&self) -> &[u64] {
        &[]
    }

    /// Execute the operation (CPU fallback).
    ///
    /// Returns the number of bytes processed (for bandwidth tracking).
    fn execute_cpu(&self) -> Result<u64>;
}

/// Concrete async operation descriptor.
#[derive(Debug, Clone)]
pub struct AsyncOpDescriptor {
    id: u64,
    label: String,
    kind: OperationKind,
    priority: StreamPriority,
    cost: u64,
    bytes: u64,
    dependencies: Vec<u64>,
}

impl AsyncOpDescriptor {
    /// Create a new operation descriptor.
    pub fn new(label: impl Into<String>, kind: OperationKind, bytes: u64) -> Self {
        Self {
            id: next_op_id(),
            label: label.into(),
            kind,
            priority: StreamPriority::Normal,
            cost: 1,
            bytes,
            dependencies: Vec::new(),
        }
    }

    /// Set the priority.
    pub fn with_priority(mut self, p: StreamPriority) -> Self {
        self.priority = p;
        self
    }

    /// Set the estimated cost.
    pub fn with_cost(mut self, cost: u64) -> Self {
        self.cost = cost;
        self
    }

    /// Add a dependency on an event.
    pub fn with_dependency(mut self, event_id: u64) -> Self {
        self.dependencies.push(event_id);
        self
    }

    /// Add multiple dependencies.
    pub fn with_dependencies(mut self, ids: &[u64]) -> Self {
        self.dependencies.extend_from_slice(ids);
        self
    }

    /// Get the operation id.
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl AsyncOperation for AsyncOpDescriptor {
    fn op_id(&self) -> u64 {
        self.id
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn kind(&self) -> OperationKind {
        self.kind
    }

    fn priority(&self) -> StreamPriority {
        self.priority
    }

    fn estimated_cost(&self) -> u64 {
        self.cost
    }

    fn dependencies(&self) -> &[u64] {
        &self.dependencies
    }

    fn execute_cpu(&self) -> Result<u64> {
        Ok(self.bytes)
    }
}

// ── Execution result ─────────────────────────────────────────────────

/// Result of scheduling an operation.
#[derive(Debug, Clone)]
pub struct ScheduleResult {
    /// Operation id.
    pub op_id: u64,
    /// Stream index the operation was assigned to.
    pub stream_index: usize,
    /// Completion event id.
    pub completion_event_id: u64,
}

/// Accumulated statistics for an execution session.
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Total operations submitted.
    pub total_ops: u64,
    /// Total bytes transferred (H2D + D2H + D2D).
    pub total_bytes_transferred: u64,
    /// Total compute operations executed.
    pub total_compute_ops: u64,
    /// Total transfer operations executed.
    pub total_transfer_ops: u64,
    /// Total barrier operations executed.
    pub total_barrier_ops: u64,
    /// Wall-clock duration of the execution session.
    pub wall_time: Duration,
    /// Per-stream operation counts.
    pub per_stream_ops: Vec<u64>,
}

// ── PipelineStageKind ────────────────────────────────────────────────

/// Stage kind for pipeline scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStageKind {
    /// Host-to-device data transfer.
    HostToDevice,
    /// Compute kernel.
    Compute,
    /// Device-to-host data transfer.
    DeviceToHost,
}

impl From<PipelineStageKind> for OperationKind {
    fn from(kind: PipelineStageKind) -> Self {
        match kind {
            PipelineStageKind::HostToDevice => OperationKind::HostToDevice,
            PipelineStageKind::Compute => OperationKind::Compute,
            PipelineStageKind::DeviceToHost => OperationKind::DeviceToHost,
        }
    }
}

// ── PipelinedExecution ───────────────────────────────────────────────

/// A single stage in a pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage kind.
    pub kind: PipelineStageKind,
    /// Label for profiling.
    pub label: String,
    /// Estimated cost.
    pub cost: u64,
    /// Bytes involved in this stage.
    pub bytes: u64,
}

impl PipelineStage {
    /// Create a new pipeline stage.
    pub fn new(kind: PipelineStageKind, label: impl Into<String>, cost: u64, bytes: u64) -> Self {
        Self { kind, label: label.into(), cost, bytes }
    }
}

/// Batch of pipeline stages representing one iteration (e.g. one token step).
#[derive(Debug, Clone)]
pub struct PipelineBatch {
    /// Label for this batch.
    pub label: String,
    /// Ordered stages within the batch.
    pub stages: Vec<PipelineStage>,
}

impl PipelineBatch {
    pub fn new(label: impl Into<String>, stages: Vec<PipelineStage>) -> Self {
        Self { label: label.into(), stages }
    }
}

/// Result of executing a pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Per-stage assignments: `(stage_index, stream_index)`.
    pub assignments: Vec<(usize, usize)>,
    /// Completion events for each stage.
    pub stage_events: Vec<u64>,
    /// Total bytes processed.
    pub total_bytes: u64,
    /// Wall-clock time.
    pub wall_time: Duration,
}

/// Pipelined execution engine that overlaps compute and transfer.
///
/// Uses dedicated streams for each stage kind (H2D, compute, D2H) and
/// inserts inter-stream events so that dependencies are honoured while
/// independent stages overlap.
#[derive(Debug)]
pub struct PipelinedExecution {
    pool: StreamPool,
    events: EventSync,
    results: Vec<PipelineResult>,
}

impl PipelinedExecution {
    /// Create a new pipelined execution context.
    ///
    /// Requires at least 3 streams for full overlap (H2D, compute, D2H).
    pub fn new(num_streams: usize) -> Result<Self> {
        if num_streams < 2 {
            return Err(KernelError::InvalidArguments {
                reason: "PipelinedExecution requires at least 2 streams".into(),
            }
            .into());
        }
        let config = StreamPoolConfig {
            num_streams,
            default_priority: StreamPriority::Normal,
            strategy: StreamAllocationStrategy::DedicatedByKind,
            enable_profiling: false,
        };
        Ok(Self { pool: StreamPool::new(config)?, events: EventSync::new(), results: Vec::new() })
    }

    /// Execute a batch of pipeline stages.
    pub fn execute_batch(&mut self, batch: &PipelineBatch) -> Result<PipelineResult> {
        let start = Instant::now();
        let mut assignments = Vec::with_capacity(batch.stages.len());
        let mut stage_events = Vec::with_capacity(batch.stages.len());
        let mut total_bytes: u64 = 0;
        let mut prev_event_id: Option<u64> = None;

        for (i, stage) in batch.stages.iter().enumerate() {
            let stream_idx = self.pool.acquire_for_kind(stage.kind);

            // Wait for predecessor.
            if let Some(dep) = prev_event_id {
                self.events.wait(dep)?;
            }

            // Execute (CPU fallback).
            self.pool.stream_mut(stream_idx)?.mark_dirty();
            total_bytes += stage.bytes;

            // Record completion event.
            let event = self.events.create_event();
            let eid = event.id;
            {
                let stream = self.pool.stream(stream_idx)?;
                self.events.record(eid, stream)?;
            }
            self.pool.stream_mut(stream_idx)?.complete_one();

            assignments.push((i, stream_idx));
            stage_events.push(eid);
            prev_event_id = Some(eid);
        }

        let result =
            PipelineResult { assignments, stage_events, total_bytes, wall_time: start.elapsed() };
        self.results.push(result.clone());
        Ok(result)
    }

    /// Execute multiple batches sequentially (double-buffered overlap on GPU).
    pub fn execute_batches(&mut self, batches: &[PipelineBatch]) -> Result<Vec<PipelineResult>> {
        let mut results = Vec::with_capacity(batches.len());
        for batch in batches {
            results.push(self.execute_batch(batch)?);
        }
        Ok(results)
    }

    /// Number of batches executed so far.
    pub fn batches_executed(&self) -> usize {
        self.results.len()
    }

    /// Access the underlying stream pool.
    pub fn pool(&self) -> &StreamPool {
        &self.pool
    }

    /// Access the event manager.
    pub fn events(&self) -> &EventSync {
        &self.events
    }

    /// Synchronise all streams and return accumulated results.
    pub fn finish(&mut self) -> Result<Vec<PipelineResult>> {
        self.pool.sync_all()?;
        Ok(self.results.clone())
    }

    /// Reset for a new execution session.
    pub fn reset(&mut self) {
        self.pool.reset_counters();
        self.events.clear();
        self.results.clear();
    }
}

// ── AsyncExecutionEngine ─────────────────────────────────────────────

/// Configuration for the async execution engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Number of streams for the pool.
    pub num_streams: usize,
    /// Stream allocation strategy.
    pub strategy: StreamAllocationStrategy,
    /// Maximum number of queued operations before back-pressure.
    pub max_queue_depth: usize,
    /// Enable profiling.
    pub enable_profiling: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            num_streams: 4,
            strategy: StreamAllocationStrategy::RoundRobin,
            max_queue_depth: 1024,
            enable_profiling: false,
        }
    }
}

impl EngineConfig {
    pub fn validate(&self) -> Result<()> {
        if self.num_streams == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_streams must be at least 1".into(),
            }
            .into());
        }
        if self.max_queue_depth == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "max_queue_depth must be at least 1".into(),
            }
            .into());
        }
        Ok(())
    }
}

/// Entry in the completed-ops log.
#[derive(Debug, Clone)]
pub struct CompletedOp {
    /// Operation id.
    pub op_id: u64,
    /// Stream index it ran on.
    pub stream_index: usize,
    /// Completion event id.
    pub event_id: u64,
    /// Bytes processed.
    pub bytes: u64,
    /// Kind of operation.
    pub kind: OperationKind,
}

/// The async execution engine — top-level coordinator.
///
/// Manages a [`StreamPool`] and [`EventSync`], accepts
/// [`AsyncOperation`] submissions, schedules them across streams,
/// and tracks execution statistics.
#[derive(Debug)]
pub struct AsyncExecutionEngine {
    /// Unique engine id.
    id: u64,
    /// Configuration.
    config: EngineConfig,
    /// Stream pool.
    pool: StreamPool,
    /// Event synchronisation manager.
    events: EventSync,
    /// Queue of pending operations.
    pending: VecDeque<AsyncOpDescriptor>,
    /// Completed operations log.
    completed: Vec<CompletedOp>,
    /// Session start time.
    start_time: Instant,
    /// Whether the engine has been shut down.
    stopped: bool,
}

impl AsyncExecutionEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: EngineConfig) -> Result<Self> {
        config.validate()?;
        let pool_config = StreamPoolConfig {
            num_streams: config.num_streams,
            default_priority: StreamPriority::Normal,
            strategy: config.strategy,
            enable_profiling: config.enable_profiling,
        };
        Ok(Self {
            id: next_engine_id(),
            config,
            pool: StreamPool::new(pool_config)?,
            events: EventSync::new(),
            pending: VecDeque::new(),
            completed: Vec::new(),
            start_time: Instant::now(),
            stopped: false,
        })
    }

    /// Create an engine with default settings.
    pub fn with_defaults() -> Result<Self> {
        Self::new(EngineConfig::default())
    }

    /// Engine identifier.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Whether the engine is stopped.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Submit an operation for execution.
    pub fn submit(&mut self, op: AsyncOpDescriptor) -> Result<()> {
        if self.stopped {
            return Err(KernelError::GpuError { reason: "engine is stopped".into() }.into());
        }
        if self.pending.len() >= self.config.max_queue_depth {
            return Err(KernelError::GpuError {
                reason: format!(
                    "queue full ({} pending, max {})",
                    self.pending.len(),
                    self.config.max_queue_depth
                ),
            }
            .into());
        }
        self.pending.push_back(op);
        Ok(())
    }

    /// Number of pending operations.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Number of completed operations.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Schedule and execute the next pending operation.
    ///
    /// Returns `None` if the queue is empty.
    pub fn execute_next(&mut self) -> Result<Option<ScheduleResult>> {
        let op = match self.pending.pop_front() {
            Some(op) => op,
            None => return Ok(None),
        };

        // Check dependencies.
        for &dep in op.dependencies() {
            self.events.wait(dep)?;
        }

        // Select stream.
        let stream_idx = self.select_stream(&op);

        // Execute (CPU fallback).
        self.pool.stream_mut(stream_idx)?.mark_dirty();
        let bytes = op.execute_cpu()?;

        // Record completion event.
        let event = self.events.create_event();
        let eid = event.id;
        {
            let stream = self.pool.stream(stream_idx)?;
            self.events.record(eid, stream)?;
        }
        self.pool.stream_mut(stream_idx)?.complete_one();

        self.completed.push(CompletedOp {
            op_id: op.op_id(),
            stream_index: stream_idx,
            event_id: eid,
            bytes,
            kind: op.kind(),
        });

        Ok(Some(ScheduleResult {
            op_id: op.op_id(),
            stream_index: stream_idx,
            completion_event_id: eid,
        }))
    }

    /// Drain and execute all pending operations.
    pub fn execute_all(&mut self) -> Result<Vec<ScheduleResult>> {
        let mut results = Vec::new();
        while let Some(r) = self.execute_next()? {
            results.push(r);
        }
        Ok(results)
    }

    /// Submit and immediately execute an operation.
    pub fn submit_and_execute(&mut self, op: AsyncOpDescriptor) -> Result<ScheduleResult> {
        self.submit(op)?;
        self.execute_next()?.ok_or_else(|| {
            KernelError::GpuError { reason: "submitted op was not dequeued".into() }.into()
        })
    }

    /// Synchronise all streams and collect statistics.
    pub fn sync_all(&mut self) -> Result<ExecutionStats> {
        self.pool.sync_all()?;
        Ok(self.stats())
    }

    /// Compute execution statistics.
    pub fn stats(&self) -> ExecutionStats {
        let mut stats = ExecutionStats {
            total_ops: self.completed.len() as u64,
            wall_time: self.start_time.elapsed(),
            per_stream_ops: vec![0u64; self.pool.num_streams()],
            ..Default::default()
        };
        for c in &self.completed {
            stats.total_bytes_transferred += c.bytes;
            if c.stream_index < stats.per_stream_ops.len() {
                stats.per_stream_ops[c.stream_index] += 1;
            }
            match c.kind {
                OperationKind::Compute => stats.total_compute_ops += 1,
                OperationKind::Barrier => stats.total_barrier_ops += 1,
                _ => stats.total_transfer_ops += 1,
            }
        }
        stats
    }

    /// Access the underlying stream pool.
    pub fn pool(&self) -> &StreamPool {
        &self.pool
    }

    /// Access the event manager.
    pub fn events(&self) -> &EventSync {
        &self.events
    }

    /// Access the completed operations log.
    pub fn completed_ops(&self) -> &[CompletedOp] {
        &self.completed
    }

    /// Stop the engine. Further submissions will be rejected.
    pub fn stop(&mut self) -> Result<ExecutionStats> {
        let stats = self.sync_all()?;
        self.stopped = true;
        Ok(stats)
    }

    /// Reset the engine for a new session.
    pub fn reset(&mut self) {
        self.pool.reset_counters();
        self.events.clear();
        self.pending.clear();
        self.completed.clear();
        self.start_time = Instant::now();
        self.stopped = false;
    }

    /// Configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    // ── internal ─────────────────────────────────────────────────────

    fn select_stream(&mut self, op: &AsyncOpDescriptor) -> usize {
        match self.config.strategy {
            StreamAllocationStrategy::DedicatedByKind => {
                let kind = match op.kind() {
                    OperationKind::HostToDevice => PipelineStageKind::HostToDevice,
                    OperationKind::DeviceToHost => PipelineStageKind::DeviceToHost,
                    OperationKind::Compute => PipelineStageKind::Compute,
                    _ => PipelineStageKind::Compute,
                };
                self.pool.acquire_for_kind(kind)
            }
            _ => self.pool.acquire(),
        }
    }
}

// ── Convenience constructors ─────────────────────────────────────────

/// Create a compute operation descriptor.
pub fn compute_op(label: impl Into<String>, cost: u64) -> AsyncOpDescriptor {
    AsyncOpDescriptor::new(label, OperationKind::Compute, 0).with_cost(cost)
}

/// Create a host-to-device transfer descriptor.
pub fn h2d_transfer(label: impl Into<String>, bytes: u64) -> AsyncOpDescriptor {
    AsyncOpDescriptor::new(label, OperationKind::HostToDevice, bytes)
}

/// Create a device-to-host transfer descriptor.
pub fn d2h_transfer(label: impl Into<String>, bytes: u64) -> AsyncOpDescriptor {
    AsyncOpDescriptor::new(label, OperationKind::DeviceToHost, bytes)
}

/// Create a device-to-device transfer descriptor.
pub fn d2d_transfer(label: impl Into<String>, bytes: u64) -> AsyncOpDescriptor {
    AsyncOpDescriptor::new(label, OperationKind::DeviceToDevice, bytes)
}

/// Create a barrier operation descriptor.
pub fn barrier_op(label: impl Into<String>) -> AsyncOpDescriptor {
    AsyncOpDescriptor::new(label, OperationKind::Barrier, 0)
}

// ═════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── StreamPriority ───────────────────────────────────────────────

    #[test]
    fn test_stream_priority_ordering() {
        assert!(StreamPriority::Low < StreamPriority::Normal);
        assert!(StreamPriority::Normal < StreamPriority::High);
        assert!(StreamPriority::High < StreamPriority::Critical);
    }

    #[test]
    fn test_stream_priority_cuda_mapping() {
        assert_eq!(StreamPriority::Low.as_cuda_priority(), 0);
        assert_eq!(StreamPriority::Normal.as_cuda_priority(), -1);
        assert_eq!(StreamPriority::High.as_cuda_priority(), -2);
        assert_eq!(StreamPriority::Critical.as_cuda_priority(), -3);
    }

    #[test]
    fn test_stream_priority_display() {
        assert_eq!(format!("{}", StreamPriority::Low), "low");
        assert_eq!(format!("{}", StreamPriority::High), "high");
        assert_eq!(format!("{}", StreamPriority::Critical), "critical");
    }

    #[test]
    fn test_stream_priority_default() {
        assert_eq!(StreamPriority::default(), StreamPriority::Normal);
    }

    // ── StreamInfo ───────────────────────────────────────────────────

    #[test]
    fn test_stream_info_new() {
        let s = StreamInfo::new(StreamPriority::High);
        assert_eq!(s.priority, StreamPriority::High);
        assert!(s.synchronized);
        assert_eq!(s.ops_dispatched, 0);
        assert_eq!(s.pending_ops, 0);
        assert!(s.label.is_none());
    }

    #[test]
    fn test_stream_info_with_label() {
        let s = StreamInfo::new(StreamPriority::Normal).with_label("compute-0");
        assert_eq!(s.label.as_deref(), Some("compute-0"));
    }

    #[test]
    fn test_stream_info_mark_dirty() {
        let mut s = StreamInfo::new(StreamPriority::Normal);
        s.mark_dirty();
        assert!(!s.synchronized);
        assert_eq!(s.ops_dispatched, 1);
        assert_eq!(s.pending_ops, 1);
    }

    #[test]
    fn test_stream_info_mark_synchronized() {
        let mut s = StreamInfo::new(StreamPriority::Normal);
        s.mark_dirty();
        s.mark_dirty();
        s.mark_synchronized();
        assert!(s.synchronized);
        assert_eq!(s.pending_ops, 0);
        assert_eq!(s.ops_dispatched, 2);
    }

    #[test]
    fn test_stream_info_complete_one() {
        let mut s = StreamInfo::new(StreamPriority::Normal);
        s.mark_dirty();
        s.mark_dirty();
        s.mark_dirty();
        assert_eq!(s.pending_ops, 3);
        s.complete_one();
        assert_eq!(s.pending_ops, 2);
        assert!(!s.synchronized);
        s.complete_one();
        s.complete_one();
        assert_eq!(s.pending_ops, 0);
        assert!(s.synchronized);
    }

    #[test]
    fn test_stream_info_complete_one_saturates_at_zero() {
        let mut s = StreamInfo::new(StreamPriority::Normal);
        s.complete_one();
        assert_eq!(s.pending_ops, 0);
        assert!(s.synchronized);
    }

    #[test]
    fn test_stream_info_unique_ids() {
        let s1 = StreamInfo::new(StreamPriority::Normal);
        let s2 = StreamInfo::new(StreamPriority::Normal);
        assert_ne!(s1.id, s2.id);
    }

    // ── StreamPoolConfig ─────────────────────────────────────────────

    #[test]
    fn test_pool_config_default() {
        let cfg = StreamPoolConfig::default();
        assert_eq!(cfg.num_streams, 4);
        assert_eq!(cfg.default_priority, StreamPriority::Normal);
        assert_eq!(cfg.strategy, StreamAllocationStrategy::RoundRobin);
        assert!(!cfg.enable_profiling);
    }

    #[test]
    fn test_pool_config_validate_zero_streams() {
        let cfg = StreamPoolConfig { num_streams: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_pool_config_validate_too_many_streams() {
        let cfg = StreamPoolConfig { num_streams: 300, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_pool_config_validate_ok() {
        let cfg = StreamPoolConfig { num_streams: 8, ..Default::default() };
        assert!(cfg.validate().is_ok());
    }

    // ── StreamPool ───────────────────────────────────────────────────

    #[test]
    fn test_pool_new() {
        let pool = StreamPool::with_defaults().unwrap();
        assert_eq!(pool.num_streams(), 4);
        assert!(pool.all_synchronized());
    }

    #[test]
    fn test_pool_stream_access() {
        let pool = StreamPool::with_defaults().unwrap();
        assert!(pool.stream(0).is_ok());
        assert!(pool.stream(3).is_ok());
        assert!(pool.stream(4).is_err());
    }

    #[test]
    fn test_pool_round_robin() {
        let mut pool = StreamPool::with_defaults().unwrap();
        assert_eq!(pool.acquire_round_robin(), 0);
        assert_eq!(pool.acquire_round_robin(), 1);
        assert_eq!(pool.acquire_round_robin(), 2);
        assert_eq!(pool.acquire_round_robin(), 3);
        assert_eq!(pool.acquire_round_robin(), 0);
    }

    #[test]
    fn test_pool_least_loaded() {
        let mut pool = StreamPool::with_defaults().unwrap();
        pool.stream_mut(0).unwrap().mark_dirty();
        pool.stream_mut(0).unwrap().mark_dirty();
        pool.stream_mut(1).unwrap().mark_dirty();
        let idx = pool.acquire_least_loaded();
        assert!(idx == 2 || idx == 3);
    }

    #[test]
    fn test_pool_highest_priority() {
        let config = StreamPoolConfig { num_streams: 3, ..Default::default() };
        let pool = StreamPool::new(config).unwrap();
        // All same priority — any stream is valid.
        let idx = pool.acquire_highest_priority();
        assert!(idx < 3);
    }

    #[test]
    fn test_pool_acquire_for_kind() {
        let config = StreamPoolConfig { num_streams: 4, ..Default::default() };
        let pool = StreamPool::new(config).unwrap();
        assert_eq!(pool.acquire_for_kind(PipelineStageKind::HostToDevice), 0);
        assert_eq!(pool.acquire_for_kind(PipelineStageKind::Compute), 1);
        assert_eq!(pool.acquire_for_kind(PipelineStageKind::DeviceToHost), 2);
    }

    #[test]
    fn test_pool_acquire_for_kind_two_streams() {
        let config = StreamPoolConfig { num_streams: 2, ..Default::default() };
        let pool = StreamPool::new(config).unwrap();
        assert_eq!(pool.acquire_for_kind(PipelineStageKind::HostToDevice), 0);
        assert_eq!(pool.acquire_for_kind(PipelineStageKind::Compute), 1);
        assert_eq!(pool.acquire_for_kind(PipelineStageKind::DeviceToHost), 0);
    }

    #[test]
    fn test_pool_sync_stream() {
        let mut pool = StreamPool::with_defaults().unwrap();
        pool.stream_mut(1).unwrap().mark_dirty();
        assert!(!pool.stream(1).unwrap().synchronized);
        pool.sync_stream(1).unwrap();
        assert!(pool.stream(1).unwrap().synchronized);
    }

    #[test]
    fn test_pool_sync_all() {
        let mut pool = StreamPool::with_defaults().unwrap();
        for i in 0..4 {
            pool.stream_mut(i).unwrap().mark_dirty();
        }
        assert!(!pool.all_synchronized());
        pool.sync_all().unwrap();
        assert!(pool.all_synchronized());
    }

    #[test]
    fn test_pool_total_ops_dispatched() {
        let mut pool = StreamPool::with_defaults().unwrap();
        pool.stream_mut(0).unwrap().mark_dirty();
        pool.stream_mut(0).unwrap().mark_dirty();
        pool.stream_mut(2).unwrap().mark_dirty();
        assert_eq!(pool.total_ops_dispatched(), 3);
    }

    #[test]
    fn test_pool_total_pending_ops() {
        let mut pool = StreamPool::with_defaults().unwrap();
        pool.stream_mut(0).unwrap().mark_dirty();
        pool.stream_mut(1).unwrap().mark_dirty();
        assert_eq!(pool.total_pending_ops(), 2);
        pool.stream_mut(0).unwrap().complete_one();
        assert_eq!(pool.total_pending_ops(), 1);
    }

    #[test]
    fn test_pool_reset_counters() {
        let mut pool = StreamPool::with_defaults().unwrap();
        pool.stream_mut(0).unwrap().mark_dirty();
        pool.acquire_round_robin();
        pool.reset_counters();
        assert_eq!(pool.total_ops_dispatched(), 0);
        assert!(pool.all_synchronized());
        assert_eq!(pool.acquire_round_robin(), 0);
    }

    #[test]
    fn test_pool_acquire_strategy_round_robin() {
        let config = StreamPoolConfig {
            num_streams: 3,
            strategy: StreamAllocationStrategy::RoundRobin,
            ..Default::default()
        };
        let mut pool = StreamPool::new(config).unwrap();
        assert_eq!(pool.acquire(), 0);
        assert_eq!(pool.acquire(), 1);
        assert_eq!(pool.acquire(), 2);
        assert_eq!(pool.acquire(), 0);
    }

    #[test]
    fn test_pool_acquire_strategy_least_loaded() {
        let config = StreamPoolConfig {
            num_streams: 3,
            strategy: StreamAllocationStrategy::LeastLoaded,
            ..Default::default()
        };
        let mut pool = StreamPool::new(config).unwrap();
        pool.stream_mut(0).unwrap().pending_ops = 5;
        pool.stream_mut(1).unwrap().pending_ops = 2;
        pool.stream_mut(2).unwrap().pending_ops = 3;
        assert_eq!(pool.acquire(), 1);
    }

    // ── CudaEvent ────────────────────────────────────────────────────

    #[test]
    fn test_cuda_event_new() {
        let event = CudaEvent::new();
        assert_eq!(event.state, EventState::Created);
        assert!(!event.is_signalled());
        assert!(event.stream_id.is_none());
        assert!(event.elapsed().is_none());
    }

    #[test]
    fn test_cuda_event_default() {
        let event = CudaEvent::default();
        assert_eq!(event.state, EventState::Created);
    }

    #[test]
    fn test_cuda_event_unique_ids() {
        let e1 = CudaEvent::new();
        let e2 = CudaEvent::new();
        assert_ne!(e1.id, e2.id);
    }

    // ── EventSync ────────────────────────────────────────────────────

    #[test]
    fn test_event_sync_create() {
        let mut es = EventSync::new();
        let e = es.create_event();
        assert_eq!(e.state, EventState::Created);
        assert_eq!(es.num_events(), 1);
    }

    #[test]
    fn test_event_sync_record_and_signal() {
        let mut es = EventSync::new();
        let eid = es.create_event().id;
        let stream = StreamInfo::new(StreamPriority::Normal);
        es.record(eid, &stream).unwrap();
        assert!(es.get(eid).unwrap().is_signalled());
    }

    #[test]
    fn test_event_sync_wait_signalled() {
        let mut es = EventSync::new();
        let eid = es.create_event().id;
        let stream = StreamInfo::new(StreamPriority::Normal);
        es.record(eid, &stream).unwrap();
        assert!(es.wait(eid).is_ok());
    }

    #[test]
    fn test_event_sync_wait_not_signalled() {
        let mut es = EventSync::new();
        let eid = es.create_event().id;
        // Not recorded yet.
        assert!(es.wait(eid).is_err());
    }

    #[test]
    fn test_event_sync_wait_all() {
        let mut es = EventSync::new();
        let stream = StreamInfo::new(StreamPriority::Normal);
        let e1 = es.create_event().id;
        let e2 = es.create_event().id;
        es.record(e1, &stream).unwrap();
        es.record(e2, &stream).unwrap();
        assert!(es.wait_all(&[e1, e2]).is_ok());
    }

    #[test]
    fn test_event_sync_wait_all_fails_if_any_unsignalled() {
        let mut es = EventSync::new();
        let stream = StreamInfo::new(StreamPriority::Normal);
        let e1 = es.create_event().id;
        let e2 = es.create_event().id;
        es.record(e1, &stream).unwrap();
        // e2 not recorded.
        assert!(es.wait_all(&[e1, e2]).is_err());
    }

    #[test]
    fn test_event_sync_destroy() {
        let mut es = EventSync::new();
        let eid = es.create_event().id;
        assert_eq!(es.num_events(), 1);
        es.destroy(eid).unwrap();
        assert_eq!(es.num_events(), 0);
    }

    #[test]
    fn test_event_sync_destroy_nonexistent() {
        let mut es = EventSync::new();
        assert!(es.destroy(99999).is_err());
    }

    #[test]
    fn test_event_sync_gc_signalled() {
        let mut es = EventSync::new();
        let stream = StreamInfo::new(StreamPriority::Normal);
        let e1 = es.create_event().id;
        let _e2 = es.create_event().id;
        es.record(e1, &stream).unwrap();
        let removed = es.gc_signalled();
        assert_eq!(removed, 1);
        assert_eq!(es.num_events(), 1);
    }

    #[test]
    fn test_event_sync_num_signalled() {
        let mut es = EventSync::new();
        let stream = StreamInfo::new(StreamPriority::Normal);
        let e1 = es.create_event().id;
        let _e2 = es.create_event().id;
        es.record(e1, &stream).unwrap();
        assert_eq!(es.num_signalled(), 1);
    }

    #[test]
    fn test_event_sync_clear() {
        let mut es = EventSync::new();
        es.create_event();
        es.create_event();
        es.clear();
        assert_eq!(es.num_events(), 0);
    }

    #[test]
    fn test_event_sync_default_trait() {
        let es = EventSync::default();
        assert_eq!(es.num_events(), 0);
    }

    #[test]
    fn test_event_sync_get_nonexistent() {
        let es = EventSync::new();
        assert!(es.get(42).is_err());
    }

    #[test]
    fn test_event_sync_record_nonexistent() {
        let mut es = EventSync::new();
        let stream = StreamInfo::new(StreamPriority::Normal);
        assert!(es.record(42, &stream).is_err());
    }

    #[test]
    fn test_event_elapsed_after_record() {
        let mut es = EventSync::new();
        let eid = es.create_event().id;
        let stream = StreamInfo::new(StreamPriority::Normal);
        es.record(eid, &stream).unwrap();
        let elapsed = es.get(eid).unwrap().elapsed();
        assert!(elapsed.is_some());
    }

    // ── OperationKind ────────────────────────────────────────────────

    #[test]
    fn test_operation_kind_display() {
        assert_eq!(format!("{}", OperationKind::HostToDevice), "H2D");
        assert_eq!(format!("{}", OperationKind::DeviceToHost), "D2H");
        assert_eq!(format!("{}", OperationKind::DeviceToDevice), "D2D");
        assert_eq!(format!("{}", OperationKind::Compute), "compute");
        assert_eq!(format!("{}", OperationKind::Barrier), "barrier");
    }

    // ── AsyncOpDescriptor ────────────────────────────────────────────

    #[test]
    fn test_op_descriptor_new() {
        let op = AsyncOpDescriptor::new("matmul", OperationKind::Compute, 1024);
        assert_eq!(op.label(), "matmul");
        assert_eq!(op.kind(), OperationKind::Compute);
        assert_eq!(op.priority(), StreamPriority::Normal);
        assert_eq!(op.estimated_cost(), 1);
        assert!(op.dependencies().is_empty());
    }

    #[test]
    fn test_op_descriptor_with_priority() {
        let op = AsyncOpDescriptor::new("h2d", OperationKind::HostToDevice, 0)
            .with_priority(StreamPriority::High);
        assert_eq!(op.priority(), StreamPriority::High);
    }

    #[test]
    fn test_op_descriptor_with_cost() {
        let op = compute_op("gemm", 100).with_cost(42);
        assert_eq!(op.estimated_cost(), 42);
    }

    #[test]
    fn test_op_descriptor_with_dependency() {
        let op = compute_op("add", 1).with_dependency(7);
        assert_eq!(op.dependencies(), &[7]);
    }

    #[test]
    fn test_op_descriptor_with_dependencies() {
        let op = compute_op("mul", 1).with_dependencies(&[1, 2, 3]);
        assert_eq!(op.dependencies(), &[1, 2, 3]);
    }

    #[test]
    fn test_op_descriptor_execute_cpu() {
        let op = h2d_transfer("upload", 4096);
        assert_eq!(op.execute_cpu().unwrap(), 4096);
    }

    #[test]
    fn test_op_descriptor_unique_ids() {
        let a = compute_op("a", 1);
        let b = compute_op("b", 1);
        assert_ne!(a.op_id(), b.op_id());
    }

    // ── Convenience constructors ─────────────────────────────────────

    #[test]
    fn test_compute_op_constructor() {
        let op = compute_op("kernel", 10);
        assert_eq!(op.kind(), OperationKind::Compute);
        assert_eq!(op.execute_cpu().unwrap(), 0);
    }

    #[test]
    fn test_h2d_transfer_constructor() {
        let op = h2d_transfer("upload", 2048);
        assert_eq!(op.kind(), OperationKind::HostToDevice);
        assert_eq!(op.execute_cpu().unwrap(), 2048);
    }

    #[test]
    fn test_d2h_transfer_constructor() {
        let op = d2h_transfer("download", 512);
        assert_eq!(op.kind(), OperationKind::DeviceToHost);
        assert_eq!(op.execute_cpu().unwrap(), 512);
    }

    #[test]
    fn test_d2d_transfer_constructor() {
        let op = d2d_transfer("copy", 1024);
        assert_eq!(op.kind(), OperationKind::DeviceToDevice);
    }

    #[test]
    fn test_barrier_op_constructor() {
        let op = barrier_op("sync");
        assert_eq!(op.kind(), OperationKind::Barrier);
        assert_eq!(op.execute_cpu().unwrap(), 0);
    }

    // ── PipelineStageKind conversion ─────────────────────────────────

    #[test]
    fn test_pipeline_stage_kind_to_operation_kind() {
        assert_eq!(
            OperationKind::from(PipelineStageKind::HostToDevice),
            OperationKind::HostToDevice
        );
        assert_eq!(OperationKind::from(PipelineStageKind::Compute), OperationKind::Compute);
        assert_eq!(
            OperationKind::from(PipelineStageKind::DeviceToHost),
            OperationKind::DeviceToHost
        );
    }

    // ── PipelinedExecution ───────────────────────────────────────────

    #[test]
    fn test_pipeline_new_min_streams() {
        assert!(PipelinedExecution::new(1).is_err());
        assert!(PipelinedExecution::new(2).is_ok());
    }

    #[test]
    fn test_pipeline_execute_empty_batch() {
        let mut pipe = PipelinedExecution::new(3).unwrap();
        let batch = PipelineBatch::new("empty", vec![]);
        let result = pipe.execute_batch(&batch).unwrap();
        assert!(result.assignments.is_empty());
        assert_eq!(result.total_bytes, 0);
    }

    #[test]
    fn test_pipeline_execute_single_stage() {
        let mut pipe = PipelinedExecution::new(3).unwrap();
        let stages = vec![PipelineStage::new(PipelineStageKind::Compute, "matmul", 10, 0)];
        let batch = PipelineBatch::new("step-1", stages);
        let result = pipe.execute_batch(&batch).unwrap();
        assert_eq!(result.assignments.len(), 1);
        assert_eq!(result.stage_events.len(), 1);
    }

    #[test]
    fn test_pipeline_execute_h2d_compute_d2h() {
        let mut pipe = PipelinedExecution::new(3).unwrap();
        let stages = vec![
            PipelineStage::new(PipelineStageKind::HostToDevice, "upload", 1, 4096),
            PipelineStage::new(PipelineStageKind::Compute, "kernel", 10, 0),
            PipelineStage::new(PipelineStageKind::DeviceToHost, "download", 1, 2048),
        ];
        let batch = PipelineBatch::new("inference", stages);
        let result = pipe.execute_batch(&batch).unwrap();
        assert_eq!(result.assignments.len(), 3);
        assert_eq!(result.total_bytes, 4096 + 2048);
        // Each stage on a different stream.
        let streams: Vec<_> = result.assignments.iter().map(|(_, s)| *s).collect();
        assert_eq!(streams[0], 0); // H2D
        assert_eq!(streams[1], 1); // Compute
        assert_eq!(streams[2], 2); // D2H
    }

    #[test]
    fn test_pipeline_execute_multiple_batches() {
        let mut pipe = PipelinedExecution::new(3).unwrap();
        let stages = vec![
            PipelineStage::new(PipelineStageKind::HostToDevice, "upload", 1, 1024),
            PipelineStage::new(PipelineStageKind::Compute, "kernel", 5, 0),
        ];
        let batch = PipelineBatch::new("step", stages);
        let batches = vec![batch.clone(), batch.clone(), batch];
        let results = pipe.execute_batches(&batches).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(pipe.batches_executed(), 3);
    }

    #[test]
    fn test_pipeline_finish() {
        let mut pipe = PipelinedExecution::new(3).unwrap();
        let stages = vec![PipelineStage::new(PipelineStageKind::Compute, "k", 1, 0)];
        let batch = PipelineBatch::new("b", stages);
        pipe.execute_batch(&batch).unwrap();
        let all = pipe.finish().unwrap();
        assert_eq!(all.len(), 1);
        assert!(pipe.pool().all_synchronized());
    }

    #[test]
    fn test_pipeline_reset() {
        let mut pipe = PipelinedExecution::new(3).unwrap();
        let stages = vec![PipelineStage::new(PipelineStageKind::Compute, "k", 1, 0)];
        let batch = PipelineBatch::new("b", stages);
        pipe.execute_batch(&batch).unwrap();
        pipe.reset();
        assert_eq!(pipe.batches_executed(), 0);
        assert_eq!(pipe.events().num_events(), 0);
    }

    // ── EngineConfig ─────────────────────────────────────────────────

    #[test]
    fn test_engine_config_default() {
        let cfg = EngineConfig::default();
        assert_eq!(cfg.num_streams, 4);
        assert_eq!(cfg.max_queue_depth, 1024);
    }

    #[test]
    fn test_engine_config_validate_zero_streams() {
        let cfg = EngineConfig { num_streams: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_engine_config_validate_zero_queue() {
        let cfg = EngineConfig { max_queue_depth: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    // ── AsyncExecutionEngine ─────────────────────────────────────────

    #[test]
    fn test_engine_new() {
        let engine = AsyncExecutionEngine::with_defaults().unwrap();
        assert!(!engine.is_stopped());
        assert_eq!(engine.pending_count(), 0);
        assert_eq!(engine.completed_count(), 0);
    }

    #[test]
    fn test_engine_unique_ids() {
        let e1 = AsyncExecutionEngine::with_defaults().unwrap();
        let e2 = AsyncExecutionEngine::with_defaults().unwrap();
        assert_ne!(e1.id(), e2.id());
    }

    #[test]
    fn test_engine_submit() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        engine.submit(compute_op("k1", 1)).unwrap();
        assert_eq!(engine.pending_count(), 1);
    }

    #[test]
    fn test_engine_submit_when_stopped() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        engine.stop().unwrap();
        assert!(engine.submit(compute_op("k", 1)).is_err());
    }

    #[test]
    fn test_engine_submit_queue_full() {
        let cfg = EngineConfig { max_queue_depth: 2, ..Default::default() };
        let mut engine = AsyncExecutionEngine::new(cfg).unwrap();
        engine.submit(compute_op("a", 1)).unwrap();
        engine.submit(compute_op("b", 1)).unwrap();
        assert!(engine.submit(compute_op("c", 1)).is_err());
    }

    #[test]
    fn test_engine_execute_next_empty() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        assert!(engine.execute_next().unwrap().is_none());
    }

    #[test]
    fn test_engine_execute_next() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        engine.submit(compute_op("k", 5)).unwrap();
        let result = engine.execute_next().unwrap().unwrap();
        assert_eq!(engine.completed_count(), 1);
        assert_eq!(engine.pending_count(), 0);
        assert!(result.stream_index < 4);
    }

    #[test]
    fn test_engine_execute_all() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        for i in 0..8 {
            engine.submit(compute_op(format!("k{i}"), 1)).unwrap();
        }
        let results = engine.execute_all().unwrap();
        assert_eq!(results.len(), 8);
        assert_eq!(engine.completed_count(), 8);
        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn test_engine_submit_and_execute() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        let r = engine.submit_and_execute(compute_op("k", 1)).unwrap();
        assert_eq!(engine.completed_count(), 1);
        assert!(r.completion_event_id > 0);
    }

    #[test]
    fn test_engine_mixed_operations() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        engine.submit(h2d_transfer("upload", 4096)).unwrap();
        engine.submit(compute_op("matmul", 10)).unwrap();
        engine.submit(d2h_transfer("download", 2048)).unwrap();
        let results = engine.execute_all().unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_engine_stats_basic() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        engine.submit(h2d_transfer("up", 1024)).unwrap();
        engine.submit(compute_op("k", 1)).unwrap();
        engine.submit(d2h_transfer("down", 512)).unwrap();
        engine.execute_all().unwrap();
        let stats = engine.stats();
        assert_eq!(stats.total_ops, 3);
        assert_eq!(stats.total_bytes_transferred, 1024 + 512);
        assert_eq!(stats.total_compute_ops, 1);
        assert_eq!(stats.total_transfer_ops, 2);
    }

    #[test]
    fn test_engine_stats_per_stream() {
        let cfg = EngineConfig {
            num_streams: 2,
            strategy: StreamAllocationStrategy::RoundRobin,
            ..Default::default()
        };
        let mut engine = AsyncExecutionEngine::new(cfg).unwrap();
        for i in 0..4 {
            engine.submit(compute_op(format!("k{i}"), 1)).unwrap();
        }
        engine.execute_all().unwrap();
        let stats = engine.stats();
        assert_eq!(stats.per_stream_ops.len(), 2);
        assert_eq!(stats.per_stream_ops[0] + stats.per_stream_ops[1], 4);
    }

    #[test]
    fn test_engine_stats_barrier_ops() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        engine.submit(barrier_op("sync")).unwrap();
        engine.execute_all().unwrap();
        let stats = engine.stats();
        assert_eq!(stats.total_barrier_ops, 1);
    }

    #[test]
    fn test_engine_sync_all() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        engine.submit(compute_op("k", 1)).unwrap();
        engine.execute_all().unwrap();
        let stats = engine.sync_all().unwrap();
        assert_eq!(stats.total_ops, 1);
        assert!(engine.pool().all_synchronized());
    }

    #[test]
    fn test_engine_stop() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        engine.submit(compute_op("k", 1)).unwrap();
        engine.execute_all().unwrap();
        let stats = engine.stop().unwrap();
        assert!(engine.is_stopped());
        assert_eq!(stats.total_ops, 1);
    }

    #[test]
    fn test_engine_reset() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        engine.submit(compute_op("k", 1)).unwrap();
        engine.execute_all().unwrap();
        engine.stop().unwrap();
        engine.reset();
        assert!(!engine.is_stopped());
        assert_eq!(engine.pending_count(), 0);
        assert_eq!(engine.completed_count(), 0);
    }

    #[test]
    fn test_engine_completed_ops_log() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        engine.submit_and_execute(h2d_transfer("up", 2048)).unwrap();
        let ops = engine.completed_ops();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].bytes, 2048);
        assert_eq!(ops[0].kind, OperationKind::HostToDevice);
    }

    #[test]
    fn test_engine_dependency_chain() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        // Step 1: upload.
        let r1 = engine.submit_and_execute(h2d_transfer("upload", 4096)).unwrap();
        // Step 2: compute depends on upload.
        let op2 = compute_op("matmul", 10).with_dependency(r1.completion_event_id);
        let r2 = engine.submit_and_execute(op2).unwrap();
        // Step 3: download depends on compute.
        let op3 = d2h_transfer("download", 2048).with_dependency(r2.completion_event_id);
        engine.submit_and_execute(op3).unwrap();
        assert_eq!(engine.completed_count(), 3);
    }

    #[test]
    fn test_engine_dedicated_stream_strategy() {
        let cfg = EngineConfig {
            num_streams: 4,
            strategy: StreamAllocationStrategy::DedicatedByKind,
            ..Default::default()
        };
        let mut engine = AsyncExecutionEngine::new(cfg).unwrap();
        let r1 = engine.submit_and_execute(h2d_transfer("up", 1024)).unwrap();
        let r2 = engine.submit_and_execute(compute_op("k", 1)).unwrap();
        let r3 = engine.submit_and_execute(d2h_transfer("down", 512)).unwrap();
        assert_eq!(r1.stream_index, 0); // H2D → stream 0
        assert_eq!(r2.stream_index, 1); // Compute → stream 1
        assert_eq!(r3.stream_index, 2); // D2H → stream 2
    }

    #[test]
    fn test_engine_config_accessor() {
        let cfg = EngineConfig { num_streams: 8, ..Default::default() };
        let engine = AsyncExecutionEngine::new(cfg).unwrap();
        assert_eq!(engine.config().num_streams, 8);
    }

    #[test]
    fn test_engine_wall_time_increases() {
        let mut engine = AsyncExecutionEngine::with_defaults().unwrap();
        let s1 = engine.stats();
        // Do some work.
        engine.submit(compute_op("k", 1)).unwrap();
        engine.execute_all().unwrap();
        let s2 = engine.stats();
        assert!(s2.wall_time >= s1.wall_time);
    }

    // ── StreamAllocationStrategy ─────────────────────────────────────

    #[test]
    fn test_allocation_strategy_default() {
        assert_eq!(StreamAllocationStrategy::default(), StreamAllocationStrategy::RoundRobin);
    }

    #[test]
    fn test_allocation_strategy_debug() {
        let s = format!("{:?}", StreamAllocationStrategy::PriorityBased);
        assert!(s.contains("PriorityBased"));
    }

    // ── ScheduleResult ───────────────────────────────────────────────

    #[test]
    fn test_schedule_result_debug() {
        let r = ScheduleResult { op_id: 1, stream_index: 0, completion_event_id: 2 };
        let s = format!("{:?}", r);
        assert!(s.contains("op_id"));
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_single_stream_pool() {
        let config = StreamPoolConfig { num_streams: 1, ..Default::default() };
        let mut pool = StreamPool::new(config).unwrap();
        assert_eq!(pool.acquire_round_robin(), 0);
        assert_eq!(pool.acquire_round_robin(), 0);
        assert_eq!(pool.acquire_least_loaded(), 0);
    }

    #[test]
    fn test_single_stream_engine() {
        let cfg = EngineConfig { num_streams: 1, ..Default::default() };
        let mut engine = AsyncExecutionEngine::new(cfg).unwrap();
        for i in 0..3 {
            engine.submit(compute_op(format!("k{i}"), 1)).unwrap();
        }
        let results = engine.execute_all().unwrap();
        assert_eq!(results.len(), 3);
        // All on stream 0.
        for r in &results {
            assert_eq!(r.stream_index, 0);
        }
    }

    #[test]
    fn test_large_batch_pipeline() {
        let mut pipe = PipelinedExecution::new(3).unwrap();
        let stages: Vec<_> = (0..20)
            .map(|i| {
                let kind = match i % 3 {
                    0 => PipelineStageKind::HostToDevice,
                    1 => PipelineStageKind::Compute,
                    _ => PipelineStageKind::DeviceToHost,
                };
                PipelineStage::new(kind, format!("stage-{i}"), 1, 64)
            })
            .collect();
        let batch = PipelineBatch::new("big", stages);
        let result = pipe.execute_batch(&batch).unwrap();
        assert_eq!(result.assignments.len(), 20);
        assert_eq!(result.total_bytes, 20 * 64);
    }

    #[test]
    fn test_pool_streams_accessor() {
        let pool = StreamPool::with_defaults().unwrap();
        assert_eq!(pool.streams().len(), 4);
    }

    #[test]
    fn test_pipeline_pool_accessor() {
        let pipe = PipelinedExecution::new(3).unwrap();
        assert_eq!(pipe.pool().num_streams(), 3);
    }

    #[test]
    fn test_pipeline_events_accessor() {
        let pipe = PipelinedExecution::new(3).unwrap();
        assert_eq!(pipe.events().num_events(), 0);
    }
}
