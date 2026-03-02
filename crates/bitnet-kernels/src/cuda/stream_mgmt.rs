//! CUDA stream management for concurrent kernel execution.
//!
//! # Overview
//!
//! Provides a pool of CUDA streams for overlapping kernel execution, memory
//! transfers, and synchronization.  Key components:
//!
//! - [`StreamPool`] — pool of streams with round-robin or priority-based dispatch.
//! - [`StreamConfig`] — configuration (count, priority, default behaviour).
//! - [`StreamHandle`] — lightweight handle to a single stream.
//! - [`StreamEvent`] — synchronization event between streams.
//! - [`StreamScheduler`] — schedule operations across streams for max utilisation.
//! - [`StreamProfiler`] — per-stream utilisation profiling.
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations execute operations sequentially.

use bitnet_common::{KernelError, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ── Identifiers ──────────────────────────────────────────────────────

static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_EVENT_ID: AtomicU64 = AtomicU64::new(1);

fn next_stream_id() -> u64 {
    NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed)
}

fn next_event_id() -> u64 {
    NEXT_EVENT_ID.fetch_add(1, Ordering::Relaxed)
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
}

impl StreamPriority {
    /// Map to a CUDA-compatible numeric priority (lower = higher priority).
    pub fn as_cuda_priority(self) -> i32 {
        match self {
            Self::Low => 0,
            Self::Normal => -1,
            Self::High => -2,
        }
    }
}

// ── DefaultStreamBehavior ────────────────────────────────────────────

/// How the default (NULL) stream interacts with created streams.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DefaultStreamBehavior {
    /// Streams synchronize with the default stream implicitly.
    Legacy,
    /// Streams do **not** synchronize with the default stream.
    #[default]
    PerThread,
}

// ── StreamConfig ─────────────────────────────────────────────────────

/// Configuration for the stream pool.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Number of streams in the pool.
    pub num_streams: usize,
    /// Default priority for newly created streams.
    pub priority: StreamPriority,
    /// Default-stream interaction behaviour.
    pub default_stream_behavior: DefaultStreamBehavior,
    /// Enable profiling on all streams.
    pub enable_profiling: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            num_streams: 4,
            priority: StreamPriority::Normal,
            default_stream_behavior: DefaultStreamBehavior::PerThread,
            enable_profiling: false,
        }
    }
}

impl StreamConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.num_streams == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_streams must be at least 1".into(),
            }
            .into());
        }
        if self.num_streams > 128 {
            return Err(KernelError::InvalidArguments {
                reason: "num_streams must not exceed 128".into(),
            }
            .into());
        }
        Ok(())
    }
}

// ── StreamHandle ─────────────────────────────────────────────────────

/// Handle to a single CUDA stream (or CPU fallback).
#[derive(Debug, Clone)]
pub struct StreamHandle {
    /// Unique stream identifier.
    pub id: u64,
    /// Priority of this stream.
    pub priority: StreamPriority,
    /// Whether this stream has been synchronised.
    pub synchronized: bool,
    /// Cumulative number of operations dispatched.
    pub ops_dispatched: u64,
    /// Creation timestamp.
    pub created_at: Instant,
}

impl StreamHandle {
    /// Create a new stream handle.
    pub fn new(priority: StreamPriority) -> Self {
        Self {
            id: next_stream_id(),
            priority,
            synchronized: true,
            ops_dispatched: 0,
            created_at: Instant::now(),
        }
    }

    /// Mark the stream as having pending work (not yet synchronised).
    pub fn mark_dirty(&mut self) {
        self.synchronized = false;
        self.ops_dispatched += 1;
    }

    /// Mark the stream as synchronised (all prior work complete).
    pub fn mark_synchronized(&mut self) {
        self.synchronized = true;
    }
}

// ── StreamEvent ──────────────────────────────────────────────────────

/// Synchronization event for inter-stream ordering.
#[derive(Debug, Clone)]
pub struct StreamEvent {
    /// Unique event identifier.
    pub id: u64,
    /// Stream on which the event was recorded (`None` if not yet recorded).
    pub recorded_on: Option<u64>,
    /// Whether the event has been signalled (work completed).
    pub signalled: bool,
    /// Timestamp of recording.
    pub recorded_at: Option<Instant>,
}

impl StreamEvent {
    /// Create a new unsignalled event.
    pub fn new() -> Self {
        Self { id: next_event_id(), recorded_on: None, signalled: false, recorded_at: None }
    }
}

impl Default for StreamEvent {
    fn default() -> Self {
        Self::new()
    }
}

// ── StreamPool ───────────────────────────────────────────────────────

/// Pool of CUDA streams for concurrent kernel execution.
///
/// On CPU this is a thin wrapper that executes all work sequentially,
/// but maintains the same API so callers are device-agnostic.
#[derive(Debug)]
pub struct StreamPool {
    config: StreamConfig,
    streams: Vec<StreamHandle>,
    events: HashMap<u64, StreamEvent>,
    /// Round-robin counter for next-stream selection.
    next_index: usize,
}

impl StreamPool {
    /// Create a new stream pool from the given configuration.
    pub fn new(config: StreamConfig) -> Result<Self> {
        config.validate()?;
        let streams = (0..config.num_streams).map(|_| StreamHandle::new(config.priority)).collect();
        Ok(Self { config, streams, events: HashMap::new(), next_index: 0 })
    }

    /// Create a pool with default settings.
    pub fn with_defaults() -> Result<Self> {
        Self::new(StreamConfig::default())
    }

    /// Number of streams in the pool.
    pub fn num_streams(&self) -> usize {
        self.streams.len()
    }

    /// Get a reference to a stream by index.
    pub fn stream(&self, index: usize) -> Result<&StreamHandle> {
        self.streams.get(index).ok_or_else(|| {
            KernelError::InvalidArguments {
                reason: format!(
                    "stream index {index} out of range (pool has {})",
                    self.streams.len()
                ),
            }
            .into()
        })
    }

    /// Get a mutable reference to a stream by index.
    pub fn stream_mut(&mut self, index: usize) -> Result<&mut StreamHandle> {
        let len = self.streams.len();
        self.streams.get_mut(index).ok_or_else(|| {
            KernelError::InvalidArguments {
                reason: format!("stream index {index} out of range (pool has {len})"),
            }
            .into()
        })
    }

    /// Acquire the next stream using round-robin selection.
    pub fn acquire_next(&mut self) -> usize {
        let idx = self.next_index % self.streams.len();
        self.next_index = self.next_index.wrapping_add(1);
        idx
    }

    /// Acquire the stream with the fewest dispatched operations.
    pub fn acquire_least_loaded(&self) -> usize {
        self.streams
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.ops_dispatched)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Create a synchronization event.
    pub fn create_event(&mut self) -> StreamEvent {
        let event = StreamEvent::new();
        self.events.insert(event.id, event.clone());
        event
    }

    /// Get an event by id.
    pub fn event(&self, id: u64) -> Result<&StreamEvent> {
        self.events.get(&id).ok_or_else(|| {
            KernelError::InvalidArguments { reason: format!("event {id} not found") }.into()
        })
    }

    /// Record an event on a stream (CPU fallback: immediately signals).
    pub fn record_event(&mut self, event_id: u64, stream_index: usize) -> Result<()> {
        let stream_id = self.stream(stream_index)?.id;
        let event = self.events.get_mut(&event_id).ok_or_else(|| {
            KernelError::InvalidArguments { reason: format!("event {event_id} not found") }
        })?;
        event.recorded_on = Some(stream_id);
        event.recorded_at = Some(Instant::now());
        // CPU fallback: work is sequential so the event is immediately signalled.
        event.signalled = true;
        Ok(())
    }

    /// Wait for an event on a different stream (CPU fallback: no-op — already done).
    pub fn wait_event(&self, event_id: u64, _waiting_stream_index: usize) -> Result<()> {
        let event = self.event(event_id)?;
        if !event.signalled {
            return Err(KernelError::GpuError {
                reason: format!("event {event_id} not yet signalled"),
            }
            .into());
        }
        Ok(())
    }

    /// Synchronize a single stream (CPU fallback: mark as synchronised).
    pub fn sync_stream(&mut self, index: usize) -> Result<()> {
        self.stream_mut(index)?.mark_synchronized();
        Ok(())
    }

    /// Synchronize all streams.
    pub fn sync_all(&mut self) -> Result<()> {
        for s in &mut self.streams {
            s.mark_synchronized();
        }
        Ok(())
    }

    /// Return current configuration.
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// Return all stream handles.
    pub fn streams(&self) -> &[StreamHandle] {
        &self.streams
    }

    /// Number of live events.
    pub fn num_events(&self) -> usize {
        self.events.len()
    }

    /// Destroy an event.
    pub fn destroy_event(&mut self, event_id: u64) -> Result<()> {
        self.events.remove(&event_id).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("event {event_id} not found"),
        })?;
        Ok(())
    }
}

// ── Free functions ───────────────────────────────────────────────────

/// Synchronize a stream by index within a pool (convenience wrapper).
pub fn stream_sync(pool: &mut StreamPool, stream_index: usize) -> Result<()> {
    pool.sync_stream(stream_index)
}

/// Record an event on a stream (convenience wrapper).
pub fn event_record(pool: &mut StreamPool, event_id: u64, stream_index: usize) -> Result<()> {
    pool.record_event(event_id, stream_index)
}

/// Wait for an event on a stream (convenience wrapper).
pub fn event_wait(pool: &StreamPool, event_id: u64, stream_index: usize) -> Result<()> {
    pool.wait_event(event_id, stream_index)
}

// ── Operation ────────────────────────────────────────────────────────

/// A dispatchable operation that can be assigned to a stream.
#[derive(Debug, Clone)]
pub struct StreamOp {
    /// Human-readable label for profiling / debugging.
    pub label: String,
    /// Estimated cost (arbitrary units) for load balancing.
    pub cost: u64,
    /// Optional dependency: event id that must be signalled first.
    pub depends_on: Option<u64>,
}

impl StreamOp {
    pub fn new(label: impl Into<String>, cost: u64) -> Self {
        Self { label: label.into(), cost, depends_on: None }
    }

    pub fn with_dependency(mut self, event_id: u64) -> Self {
        self.depends_on = Some(event_id);
        self
    }
}

// ── multi_stream_dispatch ────────────────────────────────────────────

/// Result of dispatching an operation on a stream.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    /// Stream index the operation was dispatched on.
    pub stream_index: usize,
    /// Event recorded after the operation.
    pub completion_event: u64,
}

/// Dispatch a list of operations across pool streams (round-robin).
///
/// On CPU the operations execute sequentially but are assigned to
/// different logical stream indices for correctness testing.
pub fn multi_stream_dispatch(
    pool: &mut StreamPool,
    ops: &[StreamOp],
) -> Result<Vec<DispatchResult>> {
    let mut results = Vec::with_capacity(ops.len());
    for op in ops {
        // Honour dependency.
        if let Some(dep_event) = op.depends_on {
            let ev = pool.event(dep_event)?;
            if !ev.signalled {
                return Err(KernelError::GpuError {
                    reason: format!(
                        "dependency event {} for op '{}' not signalled",
                        dep_event, op.label
                    ),
                }
                .into());
            }
        }

        let idx = pool.acquire_next();
        pool.stream_mut(idx)?.mark_dirty();

        // Record a completion event.
        let event = pool.create_event();
        let eid = event.id;
        pool.record_event(eid, idx)?;

        results.push(DispatchResult { stream_index: idx, completion_event: eid });
    }
    Ok(results)
}

// ── pipeline_stages ──────────────────────────────────────────────────

/// Stage kind for pipelining.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStageKind {
    /// Host-to-device memory transfer.
    HostToDevice,
    /// Compute kernel.
    Compute,
    /// Device-to-host memory transfer.
    DeviceToHost,
}

/// A single pipeline stage.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    pub kind: PipelineStageKind,
    pub label: String,
    pub cost: u64,
}

impl PipelineStage {
    pub fn new(kind: PipelineStageKind, label: impl Into<String>, cost: u64) -> Self {
        Self { kind, label: label.into(), cost }
    }
}

/// Result of pipeline scheduling.
#[derive(Debug, Clone)]
pub struct PipelineSchedule {
    /// (stage_index, stream_index) assignments.
    pub assignments: Vec<(usize, usize)>,
    /// Events connecting stages.
    pub stage_events: Vec<u64>,
}

/// Pipeline compute and memory transfer across streams.
///
/// Assigns each stage to a stream based on its kind, inserting events
/// so that stages execute in order while allowing different kinds to
/// overlap on separate streams.
pub fn pipeline_stages(
    pool: &mut StreamPool,
    stages: &[PipelineStage],
) -> Result<PipelineSchedule> {
    if stages.is_empty() {
        return Ok(PipelineSchedule { assignments: Vec::new(), stage_events: Vec::new() });
    }
    if pool.num_streams() < 2 {
        return Err(KernelError::InvalidArguments {
            reason: "pipeline_stages requires at least 2 streams".into(),
        }
        .into());
    }

    let mut assignments = Vec::with_capacity(stages.len());
    let mut stage_events = Vec::with_capacity(stages.len());
    let mut prev_event: Option<u64> = None;

    for (i, stage) in stages.iter().enumerate() {
        // Assign stream by kind: H2D→0, Compute→1, D2H→stream 0 or 2 if available.
        let stream_idx = match stage.kind {
            PipelineStageKind::HostToDevice => 0,
            PipelineStageKind::Compute => 1 % pool.num_streams(),
            PipelineStageKind::DeviceToHost => {
                if pool.num_streams() > 2 {
                    2
                } else {
                    0
                }
            }
        };

        // Wait for previous stage to complete.
        if let Some(dep) = prev_event {
            pool.wait_event(dep, stream_idx)?;
        }

        pool.stream_mut(stream_idx)?.mark_dirty();

        let event = pool.create_event();
        let eid = event.id;
        pool.record_event(eid, stream_idx)?;
        stage_events.push(eid);
        prev_event = Some(eid);

        assignments.push((i, stream_idx));
    }

    Ok(PipelineSchedule { assignments, stage_events })
}

// ── stream_priority_manager ──────────────────────────────────────────

/// Manage per-stream priorities within a pool.
pub struct StreamPriorityManager<'a> {
    pool: &'a mut StreamPool,
}

impl<'a> StreamPriorityManager<'a> {
    pub fn new(pool: &'a mut StreamPool) -> Self {
        Self { pool }
    }

    /// Set the priority of a specific stream.
    pub fn set_priority(&mut self, index: usize, priority: StreamPriority) -> Result<()> {
        self.pool.stream_mut(index)?.priority = priority;
        Ok(())
    }

    /// Get the priority of a specific stream.
    pub fn get_priority(&self, index: usize) -> Result<StreamPriority> {
        Ok(self.pool.stream(index)?.priority)
    }

    /// Set all streams to the same priority.
    pub fn set_all(&mut self, priority: StreamPriority) {
        for s in &mut self.pool.streams {
            s.priority = priority;
        }
    }

    /// Return streams sorted by priority (highest first).
    pub fn by_priority(&self) -> Vec<(usize, StreamPriority)> {
        let mut v: Vec<_> =
            self.pool.streams.iter().enumerate().map(|(i, s)| (i, s.priority)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }
}

// ── dependency_graph_to_streams ──────────────────────────────────────

/// A node in a dependency graph.
#[derive(Debug, Clone)]
pub struct DepNode {
    /// Unique node identifier.
    pub id: usize,
    /// Human-readable label.
    pub label: String,
    /// Ids of nodes that must complete before this one.
    pub depends_on: Vec<usize>,
    /// Estimated cost.
    pub cost: u64,
}

/// Mapping from graph node to stream index.
#[derive(Debug, Clone)]
pub struct StreamAssignment {
    pub node_id: usize,
    pub stream_index: usize,
    pub event_id: u64,
}

/// Map a dependency graph to stream assignments.
///
/// Uses a simple topological-order, round-robin assignment: nodes are
/// processed in dependency order and assigned to the next available
/// stream.  Events enforce cross-stream ordering.
pub fn dependency_graph_to_streams(
    pool: &mut StreamPool,
    nodes: &[DepNode],
) -> Result<Vec<StreamAssignment>> {
    if nodes.is_empty() {
        return Ok(Vec::new());
    }

    // Build adjacency / in-degree.
    let mut in_degree: HashMap<usize, usize> = HashMap::new();
    let mut dependents: HashMap<usize, Vec<usize>> = HashMap::new();
    for node in nodes {
        in_degree.entry(node.id).or_insert(0);
        for &dep in &node.depends_on {
            *in_degree.entry(node.id).or_insert(0) += 1;
            dependents.entry(dep).or_default().push(node.id);
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for (&id, &deg) in &in_degree {
        if deg == 0 {
            queue.push_back(id);
        }
    }

    let node_map: HashMap<usize, &DepNode> = nodes.iter().map(|n| (n.id, n)).collect();
    let mut node_event: HashMap<usize, u64> = HashMap::new();
    let mut assignments = Vec::with_capacity(nodes.len());
    let mut processed = 0usize;

    while let Some(nid) = queue.pop_front() {
        let _node = node_map.get(&nid).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("node {nid} not found in graph"),
        })?;

        let idx = pool.acquire_next();

        // Wait for all dependency events.
        if let Some(n) = node_map.get(&nid) {
            for &dep in &n.depends_on {
                if let Some(&eid) = node_event.get(&dep) {
                    pool.wait_event(eid, idx)?;
                }
            }
        }

        pool.stream_mut(idx)?.mark_dirty();

        let event = pool.create_event();
        let eid = event.id;
        pool.record_event(eid, idx)?;
        node_event.insert(nid, eid);

        assignments.push(StreamAssignment { node_id: nid, stream_index: idx, event_id: eid });

        processed += 1;

        if let Some(deps) = dependents.get(&nid) {
            for &d in deps {
                if let Some(deg) = in_degree.get_mut(&d) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(d);
                    }
                }
            }
        }
    }

    if processed != nodes.len() {
        return Err(KernelError::InvalidArguments {
            reason: "dependency graph contains a cycle".into(),
        }
        .into());
    }

    Ok(assignments)
}

// ── StreamScheduler ──────────────────────────────────────────────────

/// Strategy for how the scheduler selects streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleStrategy {
    /// Round-robin across all streams.
    RoundRobin,
    /// Pick the stream with fewest dispatched ops.
    LeastLoaded,
    /// Assign by priority tier.
    PriorityBased,
}

/// Scheduled task produced by the scheduler.
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    pub op_index: usize,
    pub stream_index: usize,
    pub event_id: u64,
}

/// Schedule operations across streams for maximum utilisation.
#[derive(Debug)]
pub struct StreamScheduler {
    strategy: ScheduleStrategy,
}

impl StreamScheduler {
    pub fn new(strategy: ScheduleStrategy) -> Self {
        Self { strategy }
    }

    /// Schedule a batch of operations onto the pool.
    pub fn schedule(&self, pool: &mut StreamPool, ops: &[StreamOp]) -> Result<Vec<ScheduledTask>> {
        let mut tasks = Vec::with_capacity(ops.len());

        for (i, op) in ops.iter().enumerate() {
            // Honour dependency.
            if let Some(dep) = op.depends_on {
                let ev = pool.event(dep)?;
                if !ev.signalled {
                    return Err(KernelError::GpuError {
                        reason: format!(
                            "dependency event {} for op '{}' not signalled",
                            dep, op.label
                        ),
                    }
                    .into());
                }
            }

            let idx = match self.strategy {
                ScheduleStrategy::RoundRobin => pool.acquire_next(),
                ScheduleStrategy::LeastLoaded => pool.acquire_least_loaded(),
                ScheduleStrategy::PriorityBased => {
                    // Pick the highest-priority stream with fewest ops.
                    let mut best = 0;
                    let mut best_score = (StreamPriority::Low, u64::MAX);
                    for (i, s) in pool.streams().iter().enumerate() {
                        let score = (s.priority, u64::MAX - s.ops_dispatched);
                        if score > best_score {
                            best_score = score;
                            best = i;
                        }
                    }
                    best
                }
            };

            pool.stream_mut(idx)?.mark_dirty();

            let event = pool.create_event();
            let eid = event.id;
            pool.record_event(eid, idx)?;

            tasks.push(ScheduledTask { op_index: i, stream_index: idx, event_id: eid });
        }

        Ok(tasks)
    }

    /// Return the strategy.
    pub fn strategy(&self) -> ScheduleStrategy {
        self.strategy
    }
}

// ── StreamProfiler ───────────────────────────────────────────────────

/// Profiling record for a single operation.
#[derive(Debug, Clone)]
pub struct ProfileRecord {
    pub label: String,
    pub stream_index: usize,
    pub start: Instant,
    pub end: Instant,
}

impl ProfileRecord {
    pub fn duration(&self) -> Duration {
        self.end.duration_since(self.start)
    }
}

/// Per-stream utilisation statistics.
#[derive(Debug, Clone)]
pub struct StreamUtilization {
    pub stream_index: usize,
    pub total_ops: usize,
    pub total_busy: Duration,
    pub idle_time: Duration,
    pub utilization_pct: f64,
}

/// Profile per-stream utilisation.
#[derive(Debug)]
pub struct StreamProfiler {
    records: Vec<ProfileRecord>,
    num_streams: usize,
    wall_start: Instant,
}

impl StreamProfiler {
    /// Create a new profiler for a pool with `num_streams` streams.
    pub fn new(num_streams: usize) -> Self {
        Self { records: Vec::new(), num_streams, wall_start: Instant::now() }
    }

    /// Record a profiling entry.
    pub fn record(&mut self, label: impl Into<String>, stream_index: usize, duration: Duration) {
        let end = Instant::now();
        let start = end - duration;
        self.records.push(ProfileRecord { label: label.into(), stream_index, start, end });
    }

    /// Add a pre-built record.
    pub fn add_record(&mut self, record: ProfileRecord) {
        self.records.push(record);
    }

    /// Total number of records.
    pub fn num_records(&self) -> usize {
        self.records.len()
    }

    /// Return all records.
    pub fn records(&self) -> &[ProfileRecord] {
        &self.records
    }

    /// Compute per-stream utilisation from wall-clock start to now.
    pub fn utilization(&self) -> Vec<StreamUtilization> {
        let wall_elapsed = self.wall_start.elapsed();
        let wall_ns = wall_elapsed.as_nanos().max(1) as f64;

        let mut per_stream: HashMap<usize, (usize, Duration)> = HashMap::new();
        for r in &self.records {
            let entry = per_stream.entry(r.stream_index).or_insert((0, Duration::ZERO));
            entry.0 += 1;
            entry.1 += r.duration();
        }

        (0..self.num_streams)
            .map(|idx| {
                let (total_ops, total_busy) =
                    per_stream.get(&idx).copied().unwrap_or((0, Duration::ZERO));
                let busy_ns = total_busy.as_nanos() as f64;
                let utilization_pct = (busy_ns / wall_ns * 100.0).min(100.0);
                let idle_time = wall_elapsed.saturating_sub(total_busy);
                StreamUtilization {
                    stream_index: idx,
                    total_ops,
                    total_busy,
                    idle_time,
                    utilization_pct,
                }
            })
            .collect()
    }

    /// Aggregate utilisation across all streams.
    pub fn aggregate_utilization(&self) -> f64 {
        let utils = self.utilization();
        if utils.is_empty() {
            return 0.0;
        }
        utils.iter().map(|u| u.utilization_pct).sum::<f64>() / utils.len() as f64
    }

    /// Reset all records.
    pub fn reset(&mut self) {
        self.records.clear();
        self.wall_start = Instant::now();
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── StreamConfig tests ───────────────────────────────────────

    #[test]
    fn config_default_is_valid() {
        StreamConfig::default().validate().unwrap();
    }

    #[test]
    fn config_zero_streams_rejected() {
        let mut cfg = StreamConfig::default();
        cfg.num_streams = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_too_many_streams_rejected() {
        let mut cfg = StreamConfig::default();
        cfg.num_streams = 200;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_boundary_128_accepted() {
        let mut cfg = StreamConfig::default();
        cfg.num_streams = 128;
        cfg.validate().unwrap();
    }

    #[test]
    fn config_boundary_1_accepted() {
        let mut cfg = StreamConfig::default();
        cfg.num_streams = 1;
        cfg.validate().unwrap();
    }

    // ── StreamPriority tests ─────────────────────────────────────

    #[test]
    fn priority_ordering() {
        assert!(StreamPriority::High > StreamPriority::Normal);
        assert!(StreamPriority::Normal > StreamPriority::Low);
    }

    #[test]
    fn priority_cuda_values() {
        assert_eq!(StreamPriority::Low.as_cuda_priority(), 0);
        assert_eq!(StreamPriority::Normal.as_cuda_priority(), -1);
        assert_eq!(StreamPriority::High.as_cuda_priority(), -2);
    }

    #[test]
    fn priority_default_is_normal() {
        assert_eq!(StreamPriority::default(), StreamPriority::Normal);
    }

    // ── StreamHandle tests ───────────────────────────────────────

    #[test]
    fn handle_new_is_synchronized() {
        let h = StreamHandle::new(StreamPriority::Normal);
        assert!(h.synchronized);
        assert_eq!(h.ops_dispatched, 0);
    }

    #[test]
    fn handle_mark_dirty() {
        let mut h = StreamHandle::new(StreamPriority::Normal);
        h.mark_dirty();
        assert!(!h.synchronized);
        assert_eq!(h.ops_dispatched, 1);
    }

    #[test]
    fn handle_mark_synchronized() {
        let mut h = StreamHandle::new(StreamPriority::Normal);
        h.mark_dirty();
        h.mark_synchronized();
        assert!(h.synchronized);
    }

    #[test]
    fn handle_ids_unique() {
        let a = StreamHandle::new(StreamPriority::Normal);
        let b = StreamHandle::new(StreamPriority::Normal);
        assert_ne!(a.id, b.id);
    }

    // ── StreamEvent tests ────────────────────────────────────────

    #[test]
    fn event_new_unsignalled() {
        let e = StreamEvent::new();
        assert!(!e.signalled);
        assert!(e.recorded_on.is_none());
    }

    #[test]
    fn event_default_same_as_new() {
        let e = StreamEvent::default();
        assert!(!e.signalled);
    }

    #[test]
    fn event_ids_unique() {
        let a = StreamEvent::new();
        let b = StreamEvent::new();
        assert_ne!(a.id, b.id);
    }

    // ── StreamPool basic tests ───────────────────────────────────

    #[test]
    fn pool_creation_default() {
        let pool = StreamPool::with_defaults().unwrap();
        assert_eq!(pool.num_streams(), 4);
    }

    #[test]
    fn pool_creation_custom() {
        let cfg = StreamConfig { num_streams: 8, ..Default::default() };
        let pool = StreamPool::new(cfg).unwrap();
        assert_eq!(pool.num_streams(), 8);
    }

    #[test]
    fn pool_stream_access_valid() {
        let pool = StreamPool::with_defaults().unwrap();
        assert!(pool.stream(0).is_ok());
        assert!(pool.stream(3).is_ok());
    }

    #[test]
    fn pool_stream_access_out_of_range() {
        let pool = StreamPool::with_defaults().unwrap();
        assert!(pool.stream(10).is_err());
    }

    #[test]
    fn pool_acquire_next_round_robin() {
        let mut pool = StreamPool::with_defaults().unwrap();
        assert_eq!(pool.acquire_next(), 0);
        assert_eq!(pool.acquire_next(), 1);
        assert_eq!(pool.acquire_next(), 2);
        assert_eq!(pool.acquire_next(), 3);
        assert_eq!(pool.acquire_next(), 0); // wraps
    }

    #[test]
    fn pool_acquire_least_loaded() {
        let mut pool = StreamPool::with_defaults().unwrap();
        pool.stream_mut(0).unwrap().mark_dirty();
        pool.stream_mut(0).unwrap().mark_dirty();
        pool.stream_mut(1).unwrap().mark_dirty();
        let idx = pool.acquire_least_loaded();
        // Streams 2 and 3 have 0 ops; expect one of them.
        assert!(idx == 2 || idx == 3);
    }

    // ── Event management ─────────────────────────────────────────

    #[test]
    fn pool_create_and_lookup_event() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let e = pool.create_event();
        assert!(pool.event(e.id).is_ok());
    }

    #[test]
    fn pool_destroy_event() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let e = pool.create_event();
        pool.destroy_event(e.id).unwrap();
        assert!(pool.event(e.id).is_err());
    }

    #[test]
    fn pool_destroy_nonexistent_event_errors() {
        let mut pool = StreamPool::with_defaults().unwrap();
        assert!(pool.destroy_event(999_999).is_err());
    }

    #[test]
    fn pool_num_events() {
        let mut pool = StreamPool::with_defaults().unwrap();
        assert_eq!(pool.num_events(), 0);
        let e1 = pool.create_event();
        let _e2 = pool.create_event();
        assert_eq!(pool.num_events(), 2);
        pool.destroy_event(e1.id).unwrap();
        assert_eq!(pool.num_events(), 1);
    }

    // ── Record / wait event ──────────────────────────────────────

    #[test]
    fn record_event_signals_immediately() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let e = pool.create_event();
        pool.record_event(e.id, 0).unwrap();
        let ev = pool.event(e.id).unwrap();
        assert!(ev.signalled);
        assert!(ev.recorded_on.is_some());
    }

    #[test]
    fn wait_signalled_event_ok() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let e = pool.create_event();
        pool.record_event(e.id, 0).unwrap();
        pool.wait_event(e.id, 1).unwrap();
    }

    #[test]
    fn wait_unsignalled_event_errors() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let e = pool.create_event();
        assert!(pool.wait_event(e.id, 0).is_err());
    }

    #[test]
    fn record_event_nonexistent_errors() {
        let mut pool = StreamPool::with_defaults().unwrap();
        assert!(pool.record_event(999_999, 0).is_err());
    }

    // ── Sync ─────────────────────────────────────────────────────

    #[test]
    fn sync_stream_marks_synchronized() {
        let mut pool = StreamPool::with_defaults().unwrap();
        pool.stream_mut(0).unwrap().mark_dirty();
        assert!(!pool.stream(0).unwrap().synchronized);
        pool.sync_stream(0).unwrap();
        assert!(pool.stream(0).unwrap().synchronized);
    }

    #[test]
    fn sync_all_marks_all_synchronized() {
        let mut pool = StreamPool::with_defaults().unwrap();
        for i in 0..pool.num_streams() {
            pool.stream_mut(i).unwrap().mark_dirty();
        }
        pool.sync_all().unwrap();
        for s in pool.streams() {
            assert!(s.synchronized);
        }
    }

    // ── Free functions ───────────────────────────────────────────

    #[test]
    fn stream_sync_fn() {
        let mut pool = StreamPool::with_defaults().unwrap();
        pool.stream_mut(1).unwrap().mark_dirty();
        stream_sync(&mut pool, 1).unwrap();
        assert!(pool.stream(1).unwrap().synchronized);
    }

    #[test]
    fn event_record_fn() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let e = pool.create_event();
        event_record(&mut pool, e.id, 2).unwrap();
        assert!(pool.event(e.id).unwrap().signalled);
    }

    #[test]
    fn event_wait_fn() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let e = pool.create_event();
        pool.record_event(e.id, 0).unwrap();
        event_wait(&pool, e.id, 1).unwrap();
    }

    // ── multi_stream_dispatch ────────────────────────────────────

    #[test]
    fn dispatch_empty() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let res = multi_stream_dispatch(&mut pool, &[]).unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn dispatch_single_op() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let ops = vec![StreamOp::new("op0", 1)];
        let res = multi_stream_dispatch(&mut pool, &ops).unwrap();
        assert_eq!(res.len(), 1);
    }

    #[test]
    fn dispatch_round_robin_assignment() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let ops: Vec<_> = (0..8).map(|i| StreamOp::new(format!("op{i}"), 1)).collect();
        let res = multi_stream_dispatch(&mut pool, &ops).unwrap();
        // First 4 ops → streams 0,1,2,3; next 4 → 0,1,2,3 again.
        for (i, r) in res.iter().enumerate() {
            assert_eq!(r.stream_index, i % 4);
        }
    }

    #[test]
    fn dispatch_with_dependency() {
        let mut pool = StreamPool::with_defaults().unwrap();
        // Dispatch first op.
        let ops1 = vec![StreamOp::new("first", 1)];
        let res1 = multi_stream_dispatch(&mut pool, &ops1).unwrap();
        let dep_event = res1[0].completion_event;
        // Second op depends on first.
        let ops2 = vec![StreamOp::new("second", 1).with_dependency(dep_event)];
        let res2 = multi_stream_dispatch(&mut pool, &ops2).unwrap();
        assert_eq!(res2.len(), 1);
    }

    #[test]
    fn dispatch_unsatisfied_dependency_errors() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let e = pool.create_event(); // not recorded → not signalled
        let ops = vec![StreamOp::new("fail", 1).with_dependency(e.id)];
        assert!(multi_stream_dispatch(&mut pool, &ops).is_err());
    }

    #[test]
    fn dispatch_creates_completion_events() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let ops: Vec<_> = (0..3).map(|i| StreamOp::new(format!("op{i}"), 1)).collect();
        let res = multi_stream_dispatch(&mut pool, &ops).unwrap();
        for r in &res {
            assert!(pool.event(r.completion_event).unwrap().signalled);
        }
    }

    // ── pipeline_stages ──────────────────────────────────────────

    #[test]
    fn pipeline_empty() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let sched = pipeline_stages(&mut pool, &[]).unwrap();
        assert!(sched.assignments.is_empty());
    }

    #[test]
    fn pipeline_requires_two_streams() {
        let cfg = StreamConfig { num_streams: 1, ..Default::default() };
        let mut pool = StreamPool::new(cfg).unwrap();
        let stages = vec![PipelineStage::new(PipelineStageKind::Compute, "c", 1)];
        assert!(pipeline_stages(&mut pool, &stages).is_err());
    }

    #[test]
    fn pipeline_assigns_different_kinds_to_different_streams() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let stages = vec![
            PipelineStage::new(PipelineStageKind::HostToDevice, "h2d", 1),
            PipelineStage::new(PipelineStageKind::Compute, "compute", 5),
            PipelineStage::new(PipelineStageKind::DeviceToHost, "d2h", 1),
        ];
        let sched = pipeline_stages(&mut pool, &stages).unwrap();
        assert_eq!(sched.assignments.len(), 3);
        // H2D → stream 0, Compute → stream 1, D2H → stream 2.
        assert_eq!(sched.assignments[0].1, 0);
        assert_eq!(sched.assignments[1].1, 1);
        assert_eq!(sched.assignments[2].1, 2);
    }

    #[test]
    fn pipeline_creates_stage_events() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let stages = vec![
            PipelineStage::new(PipelineStageKind::HostToDevice, "h2d", 1),
            PipelineStage::new(PipelineStageKind::Compute, "compute", 5),
        ];
        let sched = pipeline_stages(&mut pool, &stages).unwrap();
        assert_eq!(sched.stage_events.len(), 2);
    }

    #[test]
    fn pipeline_with_two_streams_maps_d2h_to_stream0() {
        let cfg = StreamConfig { num_streams: 2, ..Default::default() };
        let mut pool = StreamPool::new(cfg).unwrap();
        let stages = vec![
            PipelineStage::new(PipelineStageKind::HostToDevice, "h2d", 1),
            PipelineStage::new(PipelineStageKind::Compute, "compute", 5),
            PipelineStage::new(PipelineStageKind::DeviceToHost, "d2h", 1),
        ];
        let sched = pipeline_stages(&mut pool, &stages).unwrap();
        // D2H falls back to stream 0 when only 2 streams.
        assert_eq!(sched.assignments[2].1, 0);
    }

    // ── StreamPriorityManager ────────────────────────────────────

    #[test]
    fn priority_manager_set_get() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let mut mgr = StreamPriorityManager::new(&mut pool);
        mgr.set_priority(0, StreamPriority::High).unwrap();
        assert_eq!(mgr.get_priority(0).unwrap(), StreamPriority::High);
    }

    #[test]
    fn priority_manager_set_all() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let mut mgr = StreamPriorityManager::new(&mut pool);
        mgr.set_all(StreamPriority::Low);
        for i in 0..4 {
            assert_eq!(mgr.get_priority(i).unwrap(), StreamPriority::Low);
        }
    }

    #[test]
    fn priority_manager_by_priority_sorted() {
        let mut pool = StreamPool::with_defaults().unwrap();
        {
            let mut mgr = StreamPriorityManager::new(&mut pool);
            mgr.set_priority(0, StreamPriority::Low).unwrap();
            mgr.set_priority(1, StreamPriority::High).unwrap();
            mgr.set_priority(2, StreamPriority::Normal).unwrap();
            mgr.set_priority(3, StreamPriority::High).unwrap();
        }
        let mgr = StreamPriorityManager::new(&mut pool);
        let sorted = mgr.by_priority();
        assert_eq!(sorted[0].1, StreamPriority::High);
        assert_eq!(sorted[sorted.len() - 1].1, StreamPriority::Low);
    }

    #[test]
    fn priority_manager_out_of_range_errors() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let mut mgr = StreamPriorityManager::new(&mut pool);
        assert!(mgr.set_priority(99, StreamPriority::High).is_err());
        assert!(mgr.get_priority(99).is_err());
    }

    // ── dependency_graph_to_streams ──────────────────────────────

    #[test]
    fn depgraph_empty() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let a = dependency_graph_to_streams(&mut pool, &[]).unwrap();
        assert!(a.is_empty());
    }

    #[test]
    fn depgraph_single_node() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let nodes = vec![DepNode { id: 0, label: "a".into(), depends_on: vec![], cost: 1 }];
        let a = dependency_graph_to_streams(&mut pool, &nodes).unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].node_id, 0);
    }

    #[test]
    fn depgraph_linear_chain() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let nodes = vec![
            DepNode { id: 0, label: "a".into(), depends_on: vec![], cost: 1 },
            DepNode { id: 1, label: "b".into(), depends_on: vec![0], cost: 1 },
            DepNode { id: 2, label: "c".into(), depends_on: vec![1], cost: 1 },
        ];
        let a = dependency_graph_to_streams(&mut pool, &nodes).unwrap();
        assert_eq!(a.len(), 3);
        // Ordering must be topological.
        let ids: Vec<_> = a.iter().map(|x| x.node_id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn depgraph_diamond() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let nodes = vec![
            DepNode { id: 0, label: "root".into(), depends_on: vec![], cost: 1 },
            DepNode { id: 1, label: "left".into(), depends_on: vec![0], cost: 1 },
            DepNode { id: 2, label: "right".into(), depends_on: vec![0], cost: 1 },
            DepNode { id: 3, label: "join".into(), depends_on: vec![1, 2], cost: 1 },
        ];
        let a = dependency_graph_to_streams(&mut pool, &nodes).unwrap();
        assert_eq!(a.len(), 4);
        // Root must come first, join must come last.
        assert_eq!(a[0].node_id, 0);
        assert_eq!(a[3].node_id, 3);
    }

    #[test]
    fn depgraph_cycle_detected() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let nodes = vec![
            DepNode { id: 0, label: "a".into(), depends_on: vec![1], cost: 1 },
            DepNode { id: 1, label: "b".into(), depends_on: vec![0], cost: 1 },
        ];
        assert!(dependency_graph_to_streams(&mut pool, &nodes).is_err());
    }

    #[test]
    fn depgraph_events_recorded() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let nodes = vec![
            DepNode { id: 0, label: "a".into(), depends_on: vec![], cost: 1 },
            DepNode { id: 1, label: "b".into(), depends_on: vec![0], cost: 1 },
        ];
        let a = dependency_graph_to_streams(&mut pool, &nodes).unwrap();
        for sa in &a {
            assert!(pool.event(sa.event_id).unwrap().signalled);
        }
    }

    // ── StreamScheduler ──────────────────────────────────────────

    #[test]
    fn scheduler_round_robin() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let sched = StreamScheduler::new(ScheduleStrategy::RoundRobin);
        let ops: Vec<_> = (0..6).map(|i| StreamOp::new(format!("op{i}"), 1)).collect();
        let tasks = sched.schedule(&mut pool, &ops).unwrap();
        assert_eq!(tasks.len(), 6);
        for (i, t) in tasks.iter().enumerate() {
            assert_eq!(t.stream_index, i % 4);
        }
    }

    #[test]
    fn scheduler_least_loaded() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let sched = StreamScheduler::new(ScheduleStrategy::LeastLoaded);
        // Pre-load stream 0.
        pool.stream_mut(0).unwrap().mark_dirty();
        pool.stream_mut(0).unwrap().mark_dirty();
        let ops = vec![StreamOp::new("x", 1)];
        let tasks = sched.schedule(&mut pool, &ops).unwrap();
        // Should NOT pick stream 0.
        assert_ne!(tasks[0].stream_index, 0);
    }

    #[test]
    fn scheduler_priority_based() {
        let mut pool = StreamPool::with_defaults().unwrap();
        pool.streams[0].priority = StreamPriority::Low;
        pool.streams[1].priority = StreamPriority::High;
        pool.streams[2].priority = StreamPriority::Normal;
        pool.streams[3].priority = StreamPriority::Low;
        let sched = StreamScheduler::new(ScheduleStrategy::PriorityBased);
        let ops = vec![StreamOp::new("p", 1)];
        let tasks = sched.schedule(&mut pool, &ops).unwrap();
        assert_eq!(tasks[0].stream_index, 1); // highest priority
    }

    #[test]
    fn scheduler_empty_ops() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let sched = StreamScheduler::new(ScheduleStrategy::RoundRobin);
        let tasks = sched.schedule(&mut pool, &[]).unwrap();
        assert!(tasks.is_empty());
    }

    #[test]
    fn scheduler_with_dependency() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let sched = StreamScheduler::new(ScheduleStrategy::RoundRobin);
        // First batch.
        let ops1 = vec![StreamOp::new("a", 1)];
        let t1 = sched.schedule(&mut pool, &ops1).unwrap();
        // Second batch depends on first.
        let ops2 = vec![StreamOp::new("b", 1).with_dependency(t1[0].event_id)];
        let t2 = sched.schedule(&mut pool, &ops2).unwrap();
        assert_eq!(t2.len(), 1);
    }

    #[test]
    fn scheduler_unsatisfied_dep_errors() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let sched = StreamScheduler::new(ScheduleStrategy::RoundRobin);
        let e = pool.create_event();
        let ops = vec![StreamOp::new("fail", 1).with_dependency(e.id)];
        assert!(sched.schedule(&mut pool, &ops).is_err());
    }

    #[test]
    fn scheduler_strategy_accessor() {
        let s = StreamScheduler::new(ScheduleStrategy::LeastLoaded);
        assert_eq!(s.strategy(), ScheduleStrategy::LeastLoaded);
    }

    // ── StreamProfiler ───────────────────────────────────────────

    #[test]
    fn profiler_new_empty() {
        let p = StreamProfiler::new(4);
        assert_eq!(p.num_records(), 0);
    }

    #[test]
    fn profiler_record_and_count() {
        let mut p = StreamProfiler::new(4);
        p.record("op1", 0, Duration::from_millis(10));
        p.record("op2", 1, Duration::from_millis(20));
        assert_eq!(p.num_records(), 2);
    }

    #[test]
    fn profiler_add_record() {
        let mut p = StreamProfiler::new(2);
        let now = Instant::now();
        p.add_record(ProfileRecord {
            label: "x".into(),
            stream_index: 0,
            start: now,
            end: now + Duration::from_millis(5),
        });
        assert_eq!(p.num_records(), 1);
    }

    #[test]
    fn profiler_utilization_all_idle() {
        let p = StreamProfiler::new(4);
        let utils = p.utilization();
        assert_eq!(utils.len(), 4);
        for u in &utils {
            assert_eq!(u.total_ops, 0);
            assert!(u.utilization_pct <= 100.0);
        }
    }

    #[test]
    fn profiler_utilization_with_work() {
        let mut p = StreamProfiler::new(2);
        p.record("work", 0, Duration::from_millis(50));
        let utils = p.utilization();
        assert_eq!(utils[0].total_ops, 1);
        assert_eq!(utils[1].total_ops, 0);
    }

    #[test]
    fn profiler_aggregate_utilization() {
        let p = StreamProfiler::new(2);
        let agg = p.aggregate_utilization();
        assert!(agg >= 0.0 && agg <= 100.0);
    }

    #[test]
    fn profiler_reset() {
        let mut p = StreamProfiler::new(2);
        p.record("a", 0, Duration::from_millis(1));
        p.record("b", 1, Duration::from_millis(1));
        p.reset();
        assert_eq!(p.num_records(), 0);
    }

    #[test]
    fn profiler_records_accessor() {
        let mut p = StreamProfiler::new(2);
        p.record("x", 0, Duration::from_millis(5));
        let recs = p.records();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].label, "x");
    }

    #[test]
    fn profile_record_duration() {
        let now = Instant::now();
        let r = ProfileRecord {
            label: "op".into(),
            stream_index: 0,
            start: now,
            end: now + Duration::from_millis(42),
        };
        assert_eq!(r.duration().as_millis(), 42);
    }

    // ── DefaultStreamBehavior ────────────────────────────────────

    #[test]
    fn default_stream_behavior_default() {
        assert_eq!(DefaultStreamBehavior::default(), DefaultStreamBehavior::PerThread);
    }

    // ── Integration-style tests ──────────────────────────────────

    #[test]
    fn end_to_end_dispatch_and_sync() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let ops: Vec<_> = (0..12).map(|i| StreamOp::new(format!("op{i}"), 1)).collect();
        let res = multi_stream_dispatch(&mut pool, &ops).unwrap();
        assert_eq!(res.len(), 12);
        pool.sync_all().unwrap();
        for s in pool.streams() {
            assert!(s.synchronized);
        }
    }

    #[test]
    fn end_to_end_pipeline_and_profile() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let mut profiler = StreamProfiler::new(pool.num_streams());

        let stages = vec![
            PipelineStage::new(PipelineStageKind::HostToDevice, "upload", 1),
            PipelineStage::new(PipelineStageKind::Compute, "matmul", 10),
            PipelineStage::new(PipelineStageKind::DeviceToHost, "download", 1),
        ];
        let sched = pipeline_stages(&mut pool, &stages).unwrap();
        assert_eq!(sched.assignments.len(), 3);

        // Simulate profiling.
        for (i, (_, _stream_idx)) in sched.assignments.iter().enumerate() {
            profiler.record(&stages[i].label, *_stream_idx, Duration::from_millis(5));
        }
        assert_eq!(profiler.num_records(), 3);
        pool.sync_all().unwrap();
    }

    #[test]
    fn end_to_end_scheduler_and_depgraph() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let nodes = vec![
            DepNode { id: 0, label: "load_weights".into(), depends_on: vec![], cost: 5 },
            DepNode { id: 1, label: "load_input".into(), depends_on: vec![], cost: 2 },
            DepNode { id: 2, label: "matmul".into(), depends_on: vec![0, 1], cost: 10 },
            DepNode { id: 3, label: "softmax".into(), depends_on: vec![2], cost: 3 },
        ];
        let assignments = dependency_graph_to_streams(&mut pool, &nodes).unwrap();
        assert_eq!(assignments.len(), 4);

        // Verify topological order: matmul after both loads, softmax last.
        let id_order: Vec<_> = assignments.iter().map(|a| a.node_id).collect();
        let matmul_pos = id_order.iter().position(|&x| x == 2).unwrap();
        let load_w_pos = id_order.iter().position(|&x| x == 0).unwrap();
        let load_i_pos = id_order.iter().position(|&x| x == 1).unwrap();
        let softmax_pos = id_order.iter().position(|&x| x == 3).unwrap();
        assert!(matmul_pos > load_w_pos);
        assert!(matmul_pos > load_i_pos);
        assert!(softmax_pos > matmul_pos);
    }

    #[test]
    fn end_to_end_priority_manager_and_scheduler() {
        let mut pool = StreamPool::with_defaults().unwrap();
        {
            let mut mgr = StreamPriorityManager::new(&mut pool);
            mgr.set_priority(0, StreamPriority::High).unwrap();
            mgr.set_priority(1, StreamPriority::Low).unwrap();
        }
        let sched = StreamScheduler::new(ScheduleStrategy::PriorityBased);
        let ops = vec![StreamOp::new("critical", 10)];
        let tasks = sched.schedule(&mut pool, &ops).unwrap();
        assert_eq!(tasks[0].stream_index, 0); // highest prio
    }

    #[test]
    fn large_batch_dispatch() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let ops: Vec<_> = (0..256).map(|i| StreamOp::new(format!("op{i}"), 1)).collect();
        let res = multi_stream_dispatch(&mut pool, &ops).unwrap();
        assert_eq!(res.len(), 256);
        // Each stream got 64 ops.
        for i in 0..4 {
            let count = pool.stream(i).unwrap().ops_dispatched;
            assert_eq!(count, 64);
        }
    }

    #[test]
    fn scheduler_all_strategies_work() {
        for strategy in [
            ScheduleStrategy::RoundRobin,
            ScheduleStrategy::LeastLoaded,
            ScheduleStrategy::PriorityBased,
        ] {
            let mut pool = StreamPool::with_defaults().unwrap();
            let sched = StreamScheduler::new(strategy);
            let ops: Vec<_> = (0..4).map(|i| StreamOp::new(format!("op{i}"), 1)).collect();
            let tasks = sched.schedule(&mut pool, &ops).unwrap();
            assert_eq!(tasks.len(), 4);
        }
    }

    #[test]
    fn stream_config_with_profiling_enabled() {
        let cfg = StreamConfig { enable_profiling: true, ..Default::default() };
        cfg.validate().unwrap();
        assert!(cfg.enable_profiling);
    }

    #[test]
    fn handle_ops_counter_increments() {
        let mut h = StreamHandle::new(StreamPriority::Normal);
        for i in 1..=5 {
            h.mark_dirty();
            assert_eq!(h.ops_dispatched, i);
        }
    }

    #[test]
    fn profiler_multiple_streams_utilization() {
        let mut p = StreamProfiler::new(4);
        p.record("a", 0, Duration::from_millis(10));
        p.record("b", 0, Duration::from_millis(10));
        p.record("c", 1, Duration::from_millis(5));
        p.record("d", 2, Duration::from_millis(15));
        let utils = p.utilization();
        assert_eq!(utils[0].total_ops, 2);
        assert_eq!(utils[1].total_ops, 1);
        assert_eq!(utils[2].total_ops, 1);
        assert_eq!(utils[3].total_ops, 0);
    }
}
