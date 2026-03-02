//! Async command queue and pipeline executor for overlapping compute with
//! memory transfers on Intel Arc A770 GPUs.
//!
//! Provides double-buffered transfer/compute overlap, dependency tracking,
//! event-based profiling, and a CPU reference simulation for deterministic
//! testing without hardware.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
// Buffer Descriptor
// ═══════════════════════════════════════════════════════════════════

/// Describes a GPU buffer by name and byte size.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferDescriptor {
    pub name: String,
    pub size_bytes: usize,
}

impl BufferDescriptor {
    pub fn new(name: impl Into<String>, size_bytes: usize) -> Self {
        Self { name: name.into(), size_bytes }
    }
}

impl fmt::Display for BufferDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}B)", self.name, self.size_bytes)
    }
}

// ═══════════════════════════════════════════════════════════════════
// PipelineStage
// ═══════════════════════════════════════════════════════════════════

/// Named stage in an async pipeline with input/output buffer descriptors.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    pub name: String,
    pub inputs: Vec<BufferDescriptor>,
    pub outputs: Vec<BufferDescriptor>,
    /// Estimated compute cost in microseconds (used by the CPU simulator).
    pub estimated_compute_us: u64,
    /// Estimated transfer cost in microseconds (upload + download).
    pub estimated_transfer_us: u64,
}

impl PipelineStage {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            estimated_compute_us: 100,
            estimated_transfer_us: 50,
        }
    }

    pub fn with_input(mut self, buf: BufferDescriptor) -> Self {
        self.inputs.push(buf);
        self
    }

    pub fn with_output(mut self, buf: BufferDescriptor) -> Self {
        self.outputs.push(buf);
        self
    }

    pub fn with_compute_us(mut self, us: u64) -> Self {
        self.estimated_compute_us = us;
        self
    }

    pub fn with_transfer_us(mut self, us: u64) -> Self {
        self.estimated_transfer_us = us;
        self
    }

    /// Total estimated cost (compute + transfer) in microseconds.
    pub fn total_cost_us(&self) -> u64 {
        self.estimated_compute_us + self.estimated_transfer_us
    }
}

impl fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Stage({}  in={} out={} compute={}us xfer={}us)",
            self.name,
            self.inputs.len(),
            self.outputs.len(),
            self.estimated_compute_us,
            self.estimated_transfer_us,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// CommandKind / CommandBatch
// ═══════════════════════════════════════════════════════════════════

/// A single GPU command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandKind {
    KernelLaunch { kernel_name: String, global_work: usize },
    UploadBuffer { buffer: String, size_bytes: usize },
    DownloadBuffer { buffer: String, size_bytes: usize },
    Barrier,
}

impl fmt::Display for CommandKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelLaunch { kernel_name, global_work } => {
                write!(f, "Launch({kernel_name}, gw={global_work})")
            }
            Self::UploadBuffer { buffer, size_bytes } => {
                write!(f, "Upload({buffer}, {size_bytes}B)")
            }
            Self::DownloadBuffer { buffer, size_bytes } => {
                write!(f, "Download({buffer}, {size_bytes}B)")
            }
            Self::Barrier => write!(f, "Barrier"),
        }
    }
}

/// A batch of GPU commands that belong to the same logical unit.
#[derive(Debug, Clone)]
pub struct CommandBatch {
    pub label: String,
    pub commands: Vec<CommandKind>,
}

impl CommandBatch {
    pub fn new(label: impl Into<String>) -> Self {
        Self { label: label.into(), commands: Vec::new() }
    }

    pub fn push(&mut self, cmd: CommandKind) {
        self.commands.push(cmd);
    }

    pub fn kernel(mut self, name: impl Into<String>, global_work: usize) -> Self {
        self.commands.push(CommandKind::KernelLaunch { kernel_name: name.into(), global_work });
        self
    }

    pub fn upload(mut self, buffer: impl Into<String>, size_bytes: usize) -> Self {
        self.commands.push(CommandKind::UploadBuffer { buffer: buffer.into(), size_bytes });
        self
    }

    pub fn download(mut self, buffer: impl Into<String>, size_bytes: usize) -> Self {
        self.commands.push(CommandKind::DownloadBuffer { buffer: buffer.into(), size_bytes });
        self
    }

    pub fn barrier(mut self) -> Self {
        self.commands.push(CommandKind::Barrier);
        self
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════
// QueueKind / AsyncQueue
// ═══════════════════════════════════════════════════════════════════

/// Which hardware queue a command targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueKind {
    Compute,
    Transfer,
}

impl fmt::Display for QueueKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compute => write!(f, "compute"),
            Self::Transfer => write!(f, "transfer"),
        }
    }
}

/// Manages two overlapping command queues (compute + transfer).
///
/// On a real GPU the queues map to separate hardware command processors.
/// Here we simulate execution order and track submitted batches.
#[derive(Debug)]
pub struct AsyncQueue {
    compute_batches: Vec<CommandBatch>,
    transfer_batches: Vec<CommandBatch>,
    pending_barriers: usize,
}

impl AsyncQueue {
    pub fn new() -> Self {
        Self { compute_batches: Vec::new(), transfer_batches: Vec::new(), pending_barriers: 0 }
    }

    /// Submit a batch to the given queue.
    pub fn submit(&mut self, queue: QueueKind, batch: CommandBatch) {
        match queue {
            QueueKind::Compute => self.compute_batches.push(batch),
            QueueKind::Transfer => self.transfer_batches.push(batch),
        }
    }

    /// Insert a cross-queue barrier (forces all prior work to complete).
    pub fn barrier(&mut self) {
        self.pending_barriers += 1;
    }

    pub fn compute_batch_count(&self) -> usize {
        self.compute_batches.len()
    }

    pub fn transfer_batch_count(&self) -> usize {
        self.transfer_batches.len()
    }

    pub fn total_batch_count(&self) -> usize {
        self.compute_batches.len() + self.transfer_batches.len()
    }

    pub fn barrier_count(&self) -> usize {
        self.pending_barriers
    }

    /// Drain all submitted batches and return them in (compute, transfer) order.
    pub fn drain(&mut self) -> (Vec<CommandBatch>, Vec<CommandBatch>) {
        self.pending_barriers = 0;
        (std::mem::take(&mut self.compute_batches), std::mem::take(&mut self.transfer_batches))
    }
}

impl Default for AsyncQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// DoubleBuffer
// ═══════════════════════════════════════════════════════════════════

/// Slot identifier for double-buffering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Slot {
    A,
    B,
}

impl Slot {
    pub fn other(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }
}

impl fmt::Display for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::B => write!(f, "B"),
        }
    }
}

/// Double-buffered transfer manager: uploads into one slot while the other
/// slot is used for compute.
#[derive(Debug)]
pub struct DoubleBuffer {
    pub name: String,
    pub slot_size: usize,
    active_slot: Slot,
    /// Number of completed swap cycles.
    swap_count: u64,
    /// Records which slots are currently "dirty" (contain in-flight data).
    dirty: HashSet<Slot>,
}

impl DoubleBuffer {
    pub fn new(name: impl Into<String>, slot_size: usize) -> Self {
        Self {
            name: name.into(),
            slot_size,
            active_slot: Slot::A,
            swap_count: 0,
            dirty: HashSet::new(),
        }
    }

    /// The slot currently used for compute reads.
    pub fn compute_slot(&self) -> Slot {
        self.active_slot
    }

    /// The slot currently used for transfer writes (the *other* slot).
    pub fn transfer_slot(&self) -> Slot {
        self.active_slot.other()
    }

    /// Mark the transfer slot as dirty (data in flight).
    pub fn begin_transfer(&mut self) {
        self.dirty.insert(self.transfer_slot());
    }

    /// Mark transfer complete and clean the slot.
    pub fn end_transfer(&mut self) {
        self.dirty.remove(&self.transfer_slot());
    }

    /// Swap active and transfer slots. Returns the new compute slot.
    pub fn swap(&mut self) -> Slot {
        self.active_slot = self.active_slot.other();
        self.swap_count += 1;
        self.active_slot
    }

    pub fn swap_count(&self) -> u64 {
        self.swap_count
    }

    /// Returns `true` if the given slot has in-flight data.
    pub fn is_dirty(&self, slot: Slot) -> bool {
        self.dirty.contains(&slot)
    }

    /// Returns `true` when a swap would be safe (transfer slot is clean).
    pub fn can_swap(&self) -> bool {
        !self.is_dirty(self.transfer_slot())
    }

    /// Total memory footprint (both slots).
    pub fn total_bytes(&self) -> usize {
        self.slot_size * 2
    }
}

// ═══════════════════════════════════════════════════════════════════
// DependencyGraph
// ═══════════════════════════════════════════════════════════════════

/// Tracks data-flow dependencies between pipeline stages.
///
/// Each stage is identified by its index in the pipeline's stage list.
/// An edge `(a, b)` means stage `b` depends on stage `a`.
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Number of stages.
    stage_count: usize,
    /// Adjacency list: `edges[a]` contains the set of stages that depend on `a`.
    edges: HashMap<usize, HashSet<usize>>,
}

/// Errors raised by [`DependencyGraph`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyError {
    InvalidStage(usize),
    CyclicDependency,
    UnmetDependency { stage: usize, depends_on: usize },
}

impl fmt::Display for DependencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStage(id) => write!(f, "invalid stage index: {id}"),
            Self::CyclicDependency => write!(f, "dependency graph contains a cycle"),
            Self::UnmetDependency { stage, depends_on } => {
                write!(f, "stage {stage} depends on {depends_on} which has not completed")
            }
        }
    }
}

impl std::error::Error for DependencyError {}

impl DependencyGraph {
    pub fn new(stage_count: usize) -> Self {
        Self { stage_count, edges: HashMap::new() }
    }

    /// Add a dependency: `dependent` must wait for `dependency`.
    pub fn add_dependency(
        &mut self,
        dependency: usize,
        dependent: usize,
    ) -> Result<(), DependencyError> {
        if dependency >= self.stage_count {
            return Err(DependencyError::InvalidStage(dependency));
        }
        if dependent >= self.stage_count {
            return Err(DependencyError::InvalidStage(dependent));
        }
        self.edges.entry(dependency).or_default().insert(dependent);
        Ok(())
    }

    /// Returns the set of stages that `stage_idx` depends on (its predecessors).
    pub fn dependencies_of(&self, stage_idx: usize) -> HashSet<usize> {
        let mut deps = HashSet::new();
        for (&src, dsts) in &self.edges {
            if dsts.contains(&stage_idx) {
                deps.insert(src);
            }
        }
        deps
    }

    /// Returns the set of stages that depend on `stage_idx`.
    pub fn dependents_of(&self, stage_idx: usize) -> HashSet<usize> {
        self.edges.get(&stage_idx).cloned().unwrap_or_default()
    }

    /// Compute a topological ordering, or return `CyclicDependency`.
    pub fn topological_order(&self) -> Result<Vec<usize>, DependencyError> {
        let mut in_degree = vec![0usize; self.stage_count];
        for dsts in self.edges.values() {
            for &d in dsts {
                in_degree[d] += 1;
            }
        }

        let mut queue: VecDeque<usize> =
            in_degree.iter().enumerate().filter(|&(_, d)| *d == 0).map(|(i, _)| i).collect();

        let mut order = Vec::with_capacity(self.stage_count);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            if let Some(dsts) = self.edges.get(&node) {
                for &d in dsts {
                    in_degree[d] -= 1;
                    if in_degree[d] == 0 {
                        queue.push_back(d);
                    }
                }
            }
        }

        if order.len() == self.stage_count {
            Ok(order)
        } else {
            Err(DependencyError::CyclicDependency)
        }
    }

    /// Validate that the given execution order satisfies all dependencies.
    pub fn validate_order(&self, order: &[usize]) -> Result<(), DependencyError> {
        let mut completed: HashSet<usize> = HashSet::new();
        for &stage in order {
            let deps = self.dependencies_of(stage);
            for dep in &deps {
                if !completed.contains(dep) {
                    return Err(DependencyError::UnmetDependency { stage, depends_on: *dep });
                }
            }
            completed.insert(stage);
        }
        Ok(())
    }

    pub fn stage_count(&self) -> usize {
        self.stage_count
    }

    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|s| s.len()).sum()
    }

    /// Returns `true` if there are no dependency edges.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty() || self.edge_count() == 0
    }
}

// ═══════════════════════════════════════════════════════════════════
// EventTimeline
// ═══════════════════════════════════════════════════════════════════

/// Kind of event recorded in the timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    TransferStart,
    TransferEnd,
    ComputeStart,
    ComputeEnd,
    Barrier,
    SwapBuffers,
}

impl fmt::Display for EventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::TransferStart => "xfer_start",
            Self::TransferEnd => "xfer_end",
            Self::ComputeStart => "compute_start",
            Self::ComputeEnd => "compute_end",
            Self::Barrier => "barrier",
            Self::SwapBuffers => "swap",
        };
        write!(f, "{s}")
    }
}

/// A single event in the profiling timeline.
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    pub stage_index: usize,
    pub kind: EventKind,
    /// Simulated timestamp in microseconds from pipeline start.
    pub timestamp_us: u64,
    pub label: String,
}

/// Records event timestamps for post-execution profiling.
#[derive(Debug, Clone)]
pub struct EventTimeline {
    events: Vec<TimelineEvent>,
    start: Option<Instant>,
}

impl EventTimeline {
    pub fn new() -> Self {
        Self { events: Vec::new(), start: None }
    }

    /// Mark the wall-clock start of the pipeline.
    pub fn mark_start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Record a simulated event with an explicit timestamp.
    pub fn record(
        &mut self,
        stage_index: usize,
        kind: EventKind,
        timestamp_us: u64,
        label: impl Into<String>,
    ) {
        self.events.push(TimelineEvent { stage_index, kind, timestamp_us, label: label.into() });
    }

    pub fn events(&self) -> &[TimelineEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Total simulated duration (max timestamp − min timestamp).
    pub fn total_duration_us(&self) -> u64 {
        if self.events.is_empty() {
            return 0;
        }
        let min = self.events.iter().map(|e| e.timestamp_us).min().unwrap_or(0);
        let max = self.events.iter().map(|e| e.timestamp_us).max().unwrap_or(0);
        max - min
    }

    /// Filter events by kind.
    pub fn filter_by_kind(&self, kind: EventKind) -> Vec<&TimelineEvent> {
        self.events.iter().filter(|e| e.kind == kind).collect()
    }

    /// All events for a given stage index.
    pub fn events_for_stage(&self, stage_index: usize) -> Vec<&TimelineEvent> {
        self.events.iter().filter(|e| e.stage_index == stage_index).collect()
    }

    /// Returns `true` if events are in non-decreasing timestamp order.
    pub fn is_ordered(&self) -> bool {
        self.events.windows(2).all(|w| w[0].timestamp_us <= w[1].timestamp_us)
    }

    /// Wall-clock elapsed since `mark_start` (if called).
    pub fn wall_elapsed_us(&self) -> Option<u64> {
        self.start.map(|s| s.elapsed().as_micros() as u64)
    }
}

impl Default for EventTimeline {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// PipelineStats
// ═══════════════════════════════════════════════════════════════════

/// Aggregate statistics produced after a pipeline execution.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Fraction of time where compute and transfer overlapped (0.0–1.0).
    pub overlap_ratio: f64,
    /// Fraction of total wall time spent computing (0.0–1.0).
    pub compute_util: f64,
    /// Fraction of total wall time spent transferring (0.0–1.0).
    pub transfer_util: f64,
    /// Total time (µs) where neither compute nor transfer was active.
    pub stall_time_us: u64,
    /// Total simulated wall time (µs).
    pub total_time_us: u64,
    /// Number of stages executed.
    pub stages_executed: usize,
}

impl PipelineStats {
    /// Compute stats from an event timeline.
    pub fn from_timeline(timeline: &EventTimeline, stages_executed: usize) -> Self {
        let events = timeline.events();
        if events.is_empty() {
            return Self {
                overlap_ratio: 0.0,
                compute_util: 0.0,
                transfer_util: 0.0,
                stall_time_us: 0,
                total_time_us: 0,
                stages_executed,
            };
        }

        let total_time_us = timeline.total_duration_us();
        if total_time_us == 0 {
            return Self {
                overlap_ratio: 0.0,
                compute_util: 0.0,
                transfer_util: 0.0,
                stall_time_us: 0,
                total_time_us: 0,
                stages_executed,
            };
        }

        // Collect compute and transfer intervals.
        let compute_intervals =
            Self::collect_intervals(events, EventKind::ComputeStart, EventKind::ComputeEnd);
        let transfer_intervals =
            Self::collect_intervals(events, EventKind::TransferStart, EventKind::TransferEnd);

        let compute_total = Self::sum_intervals(&compute_intervals);
        let transfer_total = Self::sum_intervals(&transfer_intervals);
        let overlap_total = Self::overlap_between(&compute_intervals, &transfer_intervals);

        let total_f = total_time_us as f64;
        let overlap_ratio = if compute_total + transfer_total > 0 {
            overlap_total as f64 / (compute_total + transfer_total) as f64
        } else {
            0.0
        };

        let compute_util = compute_total as f64 / total_f;
        let transfer_util = transfer_total as f64 / total_f;

        // Stall = wall time not covered by any compute or transfer.
        let union = Self::union_intervals(&compute_intervals, &transfer_intervals);
        let active_time = Self::sum_intervals(&union);
        let stall_time_us = total_time_us.saturating_sub(active_time);

        Self {
            overlap_ratio,
            compute_util: compute_util.min(1.0),
            transfer_util: transfer_util.min(1.0),
            stall_time_us,
            total_time_us,
            stages_executed,
        }
    }

    // -- interval helpers --

    fn collect_intervals(
        events: &[TimelineEvent],
        start_kind: EventKind,
        end_kind: EventKind,
    ) -> Vec<(u64, u64)> {
        let starts: Vec<u64> =
            events.iter().filter(|e| e.kind == start_kind).map(|e| e.timestamp_us).collect();
        let ends: Vec<u64> =
            events.iter().filter(|e| e.kind == end_kind).map(|e| e.timestamp_us).collect();
        starts.into_iter().zip(ends).collect()
    }

    fn sum_intervals(intervals: &[(u64, u64)]) -> u64 {
        intervals.iter().map(|(s, e)| e.saturating_sub(*s)).sum()
    }

    fn overlap_between(a: &[(u64, u64)], b: &[(u64, u64)]) -> u64 {
        let mut total = 0u64;
        for &(a_start, a_end) in a {
            for &(b_start, b_end) in b {
                let start = a_start.max(b_start);
                let end = a_end.min(b_end);
                if start < end {
                    total += end - start;
                }
            }
        }
        total
    }

    fn union_intervals(a: &[(u64, u64)], b: &[(u64, u64)]) -> Vec<(u64, u64)> {
        let mut all: Vec<(u64, u64)> = a.iter().chain(b.iter()).copied().collect();
        all.sort_by_key(|&(s, _)| s);

        let mut merged: Vec<(u64, u64)> = Vec::new();
        for interval in all {
            if let Some(last) = merged.last_mut() {
                if interval.0 <= last.1 {
                    last.1 = last.1.max(interval.1);
                    continue;
                }
            }
            merged.push(interval);
        }
        merged
    }
}

impl fmt::Display for PipelineStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PipelineStats(overlap={:.1}% compute={:.1}% xfer={:.1}% stall={}us total={}us stages={})",
            self.overlap_ratio * 100.0,
            self.compute_util * 100.0,
            self.transfer_util * 100.0,
            self.stall_time_us,
            self.total_time_us,
            self.stages_executed,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// ExecutorError
// ═══════════════════════════════════════════════════════════════════

/// Errors produced by [`PipelineExecutor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutorError {
    EmptyPipeline,
    DependencyError(DependencyError),
    StageIndexOutOfRange(usize),
    DoubleBufferConflict(String),
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPipeline => write!(f, "pipeline has no stages"),
            Self::DependencyError(e) => write!(f, "dependency error: {e}"),
            Self::StageIndexOutOfRange(i) => write!(f, "stage index {i} out of range"),
            Self::DoubleBufferConflict(msg) => write!(f, "double buffer conflict: {msg}"),
        }
    }
}

impl std::error::Error for ExecutorError {}

impl From<DependencyError> for ExecutorError {
    fn from(e: DependencyError) -> Self {
        Self::DependencyError(e)
    }
}

// ═══════════════════════════════════════════════════════════════════
// PipelineExecutor
// ═══════════════════════════════════════════════════════════════════

/// Result of executing the pipeline.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub stats: PipelineStats,
    pub timeline: EventTimeline,
    pub execution_order: Vec<usize>,
    /// Per-stage simulated output (sum of input values × stage index, for verification).
    pub stage_outputs: Vec<f64>,
}

/// Executes a staged pipeline with automatic compute/transfer overlap.
///
/// CPU reference implementation: simulates GPU timing to allow deterministic
/// testing. Stages are executed in topological order; where possible, transfers
/// for stage N+1 are overlapped with compute for stage N.
#[derive(Debug)]
pub struct PipelineExecutor {
    stages: Vec<PipelineStage>,
    deps: DependencyGraph,
    double_buffers: Vec<DoubleBuffer>,
    enable_overlap: bool,
}

impl PipelineExecutor {
    /// Create an executor for the given stages.
    ///
    /// Dependencies are auto-inferred from matching output→input buffer names
    /// unless `manual_deps` is `true`.
    pub fn new(stages: Vec<PipelineStage>, manual_deps: bool) -> Result<Self, ExecutorError> {
        if stages.is_empty() {
            return Err(ExecutorError::EmptyPipeline);
        }

        let n = stages.len();
        let mut deps = DependencyGraph::new(n);

        if !manual_deps {
            // Auto-infer: stage j depends on stage i if any output of i matches
            // an input of j (by buffer name).
            for i in 0..n {
                for j in (i + 1)..n {
                    let has_dep = stages[i]
                        .outputs
                        .iter()
                        .any(|o| stages[j].inputs.iter().any(|inp| inp.name == o.name));
                    if has_dep {
                        deps.add_dependency(i, j)?;
                    }
                }
            }
        }

        // Validate DAG.
        deps.topological_order()?;

        Ok(Self { stages, deps, double_buffers: Vec::new(), enable_overlap: true })
    }

    /// Manually add a dependency edge.
    pub fn add_dependency(
        &mut self,
        dependency: usize,
        dependent: usize,
    ) -> Result<(), ExecutorError> {
        self.deps.add_dependency(dependency, dependent)?;
        // Re-validate after mutation.
        self.deps.topological_order()?;
        Ok(())
    }

    /// Register a double buffer for overlapped transfers.
    pub fn add_double_buffer(&mut self, db: DoubleBuffer) {
        self.double_buffers.push(db);
    }

    /// Enable or disable compute/transfer overlap.
    pub fn set_overlap(&mut self, enabled: bool) {
        self.enable_overlap = enabled;
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    pub fn dependency_graph(&self) -> &DependencyGraph {
        &self.deps
    }

    /// Execute the pipeline using CPU reference simulation.
    ///
    /// `inputs` provides a `f64` value per stage (used to compute a trivial
    /// stage output for verification). If empty, defaults to `1.0` per stage.
    pub fn execute(&mut self, inputs: &[f64]) -> Result<ExecutionResult, ExecutorError> {
        let order = self.deps.topological_order()?;
        let n = self.stages.len();

        let default_inputs: Vec<f64> = vec![1.0; n];
        let effective = if inputs.len() >= n { inputs } else { &default_inputs };

        let mut timeline = EventTimeline::new();
        timeline.mark_start();

        let mut stage_outputs: Vec<f64> = vec![0.0; n];
        let mut clock_us: u64 = 0;

        // Track per-stage completion time for overlap simulation.
        let mut compute_end: Vec<u64> = vec![0; n];

        for (exec_idx, &stage_idx) in order.iter().enumerate() {
            let stage = &self.stages[stage_idx];

            // Earliest start: all dependencies must have finished.
            let dep_ready: u64 = self
                .deps
                .dependencies_of(stage_idx)
                .iter()
                .map(|&d| compute_end[d])
                .max()
                .unwrap_or(0);
            let start = clock_us.max(dep_ready);

            // ── Transfer phase ──
            let xfer_start = start;
            let xfer_end = xfer_start + stage.estimated_transfer_us;
            timeline.record(stage_idx, EventKind::TransferStart, xfer_start, &stage.name);
            timeline.record(stage_idx, EventKind::TransferEnd, xfer_end, &stage.name);

            // ── Compute phase ──
            let comp_start = if self.enable_overlap && exec_idx > 0 {
                // Overlap: compute can start as soon as this stage's transfer ends
                // but also not before the previous compute ends.
                xfer_end.max(if exec_idx > 0 { compute_end[order[exec_idx - 1]] } else { 0 })
            } else {
                xfer_end
            };
            let comp_end = comp_start + stage.estimated_compute_us;
            timeline.record(stage_idx, EventKind::ComputeStart, comp_start, &stage.name);
            timeline.record(stage_idx, EventKind::ComputeEnd, comp_end, &stage.name);

            compute_end[stage_idx] = comp_end;
            clock_us = comp_end;

            // Swap double buffers.
            for db in &mut self.double_buffers {
                db.swap();
                timeline.record(stage_idx, EventKind::SwapBuffers, comp_end, &db.name);
            }

            // Trivial CPU reference output: input × (stage_index + 1).
            stage_outputs[stage_idx] = effective[stage_idx] * (stage_idx as f64 + 1.0);
        }

        let stats = PipelineStats::from_timeline(&timeline, n);

        Ok(ExecutionResult { stats, timeline, execution_order: order, stage_outputs })
    }

    /// Execute without any overlap (purely sequential, for comparison).
    pub fn execute_sequential(&mut self, inputs: &[f64]) -> Result<ExecutionResult, ExecutorError> {
        let was = self.enable_overlap;
        self.enable_overlap = false;
        let result = self.execute(inputs);
        self.enable_overlap = was;
        result
    }
}

// ═══════════════════════════════════════════════════════════════════
// Convenience builder
// ═══════════════════════════════════════════════════════════════════

/// Build a simple linear N-stage pipeline where each stage's output feeds the
/// next stage's input.
pub fn build_linear_pipeline(
    stage_count: usize,
    compute_us: u64,
    transfer_us: u64,
) -> Vec<PipelineStage> {
    (0..stage_count)
        .map(|i| {
            let mut s = PipelineStage::new(format!("stage_{i}"))
                .with_compute_us(compute_us)
                .with_transfer_us(transfer_us);
            if i > 0 {
                s = s.with_input(BufferDescriptor::new(format!("buf_{}", i - 1), 1024));
            }
            s = s.with_output(BufferDescriptor::new(format!("buf_{i}"), 1024));
            s
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── BufferDescriptor ───────────────────────────────────────────

    #[test]
    fn buffer_descriptor_new() {
        let bd = BufferDescriptor::new("weights", 4096);
        assert_eq!(bd.name, "weights");
        assert_eq!(bd.size_bytes, 4096);
    }

    #[test]
    fn buffer_descriptor_display() {
        let bd = BufferDescriptor::new("act", 256);
        assert_eq!(format!("{bd}"), "act(256B)");
    }

    #[test]
    fn buffer_descriptor_equality() {
        let a = BufferDescriptor::new("x", 100);
        let b = BufferDescriptor::new("x", 100);
        let c = BufferDescriptor::new("y", 100);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── PipelineStage ──────────────────────────────────────────────

    #[test]
    fn stage_builder_defaults() {
        let s = PipelineStage::new("test");
        assert_eq!(s.name, "test");
        assert!(s.inputs.is_empty());
        assert!(s.outputs.is_empty());
        assert_eq!(s.estimated_compute_us, 100);
        assert_eq!(s.estimated_transfer_us, 50);
    }

    #[test]
    fn stage_builder_chain() {
        let s = PipelineStage::new("matmul")
            .with_input(BufferDescriptor::new("a", 1024))
            .with_output(BufferDescriptor::new("c", 512))
            .with_compute_us(200)
            .with_transfer_us(80);
        assert_eq!(s.inputs.len(), 1);
        assert_eq!(s.outputs.len(), 1);
        assert_eq!(s.estimated_compute_us, 200);
        assert_eq!(s.estimated_transfer_us, 80);
    }

    #[test]
    fn stage_total_cost() {
        let s = PipelineStage::new("s").with_compute_us(300).with_transfer_us(100);
        assert_eq!(s.total_cost_us(), 400);
    }

    #[test]
    fn stage_display() {
        let s = PipelineStage::new("embed")
            .with_input(BufferDescriptor::new("tok", 64))
            .with_output(BufferDescriptor::new("emb", 256))
            .with_compute_us(50)
            .with_transfer_us(20);
        let d = format!("{s}");
        assert!(d.contains("embed"));
        assert!(d.contains("50us"));
    }

    // ── CommandKind / CommandBatch ──────────────────────────────────

    #[test]
    fn command_kind_display() {
        let k = CommandKind::KernelLaunch { kernel_name: "gemm".into(), global_work: 1024 };
        assert!(format!("{k}").contains("gemm"));
    }

    #[test]
    fn command_batch_builder() {
        let b = CommandBatch::new("fwd")
            .upload("w", 4096)
            .kernel("gemm", 1024)
            .download("out", 2048)
            .barrier();
        assert_eq!(b.len(), 4);
        assert!(!b.is_empty());
    }

    #[test]
    fn command_batch_push() {
        let mut b = CommandBatch::new("b");
        assert!(b.is_empty());
        b.push(CommandKind::Barrier);
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn command_batch_empty() {
        let b = CommandBatch::new("empty");
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
    }

    // ── AsyncQueue ─────────────────────────────────────────────────

    #[test]
    fn async_queue_submit_compute() {
        let mut q = AsyncQueue::new();
        q.submit(QueueKind::Compute, CommandBatch::new("c1").kernel("k", 64));
        assert_eq!(q.compute_batch_count(), 1);
        assert_eq!(q.transfer_batch_count(), 0);
    }

    #[test]
    fn async_queue_submit_transfer() {
        let mut q = AsyncQueue::new();
        q.submit(QueueKind::Transfer, CommandBatch::new("t1").upload("buf", 512));
        assert_eq!(q.transfer_batch_count(), 1);
        assert_eq!(q.compute_batch_count(), 0);
    }

    #[test]
    fn async_queue_total() {
        let mut q = AsyncQueue::new();
        q.submit(QueueKind::Compute, CommandBatch::new("c1"));
        q.submit(QueueKind::Transfer, CommandBatch::new("t1"));
        q.submit(QueueKind::Compute, CommandBatch::new("c2"));
        assert_eq!(q.total_batch_count(), 3);
    }

    #[test]
    fn async_queue_barrier() {
        let mut q = AsyncQueue::new();
        q.barrier();
        q.barrier();
        assert_eq!(q.barrier_count(), 2);
    }

    #[test]
    fn async_queue_drain() {
        let mut q = AsyncQueue::new();
        q.submit(QueueKind::Compute, CommandBatch::new("c"));
        q.submit(QueueKind::Transfer, CommandBatch::new("t"));
        q.barrier();
        let (c, t) = q.drain();
        assert_eq!(c.len(), 1);
        assert_eq!(t.len(), 1);
        assert_eq!(q.total_batch_count(), 0);
        assert_eq!(q.barrier_count(), 0);
    }

    #[test]
    fn async_queue_default() {
        let q = AsyncQueue::default();
        assert_eq!(q.total_batch_count(), 0);
    }

    // ── Slot / DoubleBuffer ────────────────────────────────────────

    #[test]
    fn slot_other() {
        assert_eq!(Slot::A.other(), Slot::B);
        assert_eq!(Slot::B.other(), Slot::A);
    }

    #[test]
    fn slot_display() {
        assert_eq!(format!("{}", Slot::A), "A");
        assert_eq!(format!("{}", Slot::B), "B");
    }

    #[test]
    fn double_buffer_initial_slots() {
        let db = DoubleBuffer::new("act", 4096);
        assert_eq!(db.compute_slot(), Slot::A);
        assert_eq!(db.transfer_slot(), Slot::B);
    }

    #[test]
    fn double_buffer_swap() {
        let mut db = DoubleBuffer::new("act", 4096);
        let new = db.swap();
        assert_eq!(new, Slot::B);
        assert_eq!(db.compute_slot(), Slot::B);
        assert_eq!(db.transfer_slot(), Slot::A);
        assert_eq!(db.swap_count(), 1);
    }

    #[test]
    fn double_buffer_multiple_swaps() {
        let mut db = DoubleBuffer::new("x", 256);
        for _ in 0..10 {
            db.swap();
        }
        assert_eq!(db.swap_count(), 10);
        // Even number of swaps → back to initial.
        assert_eq!(db.compute_slot(), Slot::A);
    }

    #[test]
    fn double_buffer_dirty_tracking() {
        let mut db = DoubleBuffer::new("w", 1024);
        assert!(!db.is_dirty(Slot::A));
        assert!(!db.is_dirty(Slot::B));

        db.begin_transfer();
        assert!(db.is_dirty(Slot::B)); // transfer slot
        assert!(!db.is_dirty(Slot::A));

        db.end_transfer();
        assert!(!db.is_dirty(Slot::B));
    }

    #[test]
    fn double_buffer_can_swap() {
        let mut db = DoubleBuffer::new("x", 64);
        assert!(db.can_swap());
        db.begin_transfer();
        // Transfer slot is dirty → cannot safely swap into it.
        assert!(!db.can_swap());
        db.end_transfer();
        assert!(db.can_swap());
    }

    #[test]
    fn double_buffer_total_bytes() {
        let db = DoubleBuffer::new("x", 512);
        assert_eq!(db.total_bytes(), 1024);
    }

    // ── DependencyGraph ────────────────────────────────────────────

    #[test]
    fn dep_graph_empty() {
        let g = DependencyGraph::new(4);
        assert_eq!(g.stage_count(), 4);
        assert_eq!(g.edge_count(), 0);
        assert!(g.is_empty());
    }

    #[test]
    fn dep_graph_add_valid() {
        let mut g = DependencyGraph::new(3);
        assert!(g.add_dependency(0, 1).is_ok());
        assert!(g.add_dependency(1, 2).is_ok());
        assert_eq!(g.edge_count(), 2);
        assert!(!g.is_empty());
    }

    #[test]
    fn dep_graph_invalid_stage() {
        let mut g = DependencyGraph::new(2);
        assert_eq!(g.add_dependency(0, 5), Err(DependencyError::InvalidStage(5)));
        assert_eq!(g.add_dependency(9, 0), Err(DependencyError::InvalidStage(9)));
    }

    #[test]
    fn dep_graph_dependencies_of() {
        let mut g = DependencyGraph::new(3);
        g.add_dependency(0, 2).unwrap();
        g.add_dependency(1, 2).unwrap();
        let deps = g.dependencies_of(2);
        assert!(deps.contains(&0));
        assert!(deps.contains(&1));
        assert_eq!(deps.len(), 2);
    }

    #[test]
    fn dep_graph_dependents_of() {
        let mut g = DependencyGraph::new(3);
        g.add_dependency(0, 1).unwrap();
        g.add_dependency(0, 2).unwrap();
        let d = g.dependents_of(0);
        assert!(d.contains(&1));
        assert!(d.contains(&2));
    }

    #[test]
    fn dep_graph_topological_linear() {
        let mut g = DependencyGraph::new(4);
        g.add_dependency(0, 1).unwrap();
        g.add_dependency(1, 2).unwrap();
        g.add_dependency(2, 3).unwrap();
        let order = g.topological_order().unwrap();
        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn dep_graph_topological_diamond() {
        //   0
        //  / \
        // 1   2
        //  \ /
        //   3
        let mut g = DependencyGraph::new(4);
        g.add_dependency(0, 1).unwrap();
        g.add_dependency(0, 2).unwrap();
        g.add_dependency(1, 3).unwrap();
        g.add_dependency(2, 3).unwrap();
        let order = g.topological_order().unwrap();
        // 0 must come first, 3 must come last, 1 and 2 in between.
        assert_eq!(order[0], 0);
        assert_eq!(order[3], 3);
        assert!(order.contains(&1));
        assert!(order.contains(&2));
    }

    #[test]
    fn dep_graph_cycle_detection() {
        let mut g = DependencyGraph::new(3);
        g.add_dependency(0, 1).unwrap();
        g.add_dependency(1, 2).unwrap();
        g.add_dependency(2, 0).unwrap();
        assert_eq!(g.topological_order(), Err(DependencyError::CyclicDependency));
    }

    #[test]
    fn dep_graph_validate_order_ok() {
        let mut g = DependencyGraph::new(3);
        g.add_dependency(0, 1).unwrap();
        g.add_dependency(1, 2).unwrap();
        assert!(g.validate_order(&[0, 1, 2]).is_ok());
    }

    #[test]
    fn dep_graph_validate_order_bad() {
        let mut g = DependencyGraph::new(3);
        g.add_dependency(0, 1).unwrap();
        g.add_dependency(1, 2).unwrap();
        assert!(g.validate_order(&[2, 1, 0]).is_err());
    }

    #[test]
    fn dep_graph_no_edges_all_valid() {
        let g = DependencyGraph::new(3);
        // Any order is valid when there are no dependencies.
        assert!(g.validate_order(&[2, 0, 1]).is_ok());
    }

    // ── EventTimeline ──────────────────────────────────────────────

    #[test]
    fn timeline_empty() {
        let tl = EventTimeline::new();
        assert!(tl.is_empty());
        assert_eq!(tl.len(), 0);
        assert_eq!(tl.total_duration_us(), 0);
    }

    #[test]
    fn timeline_record_and_len() {
        let mut tl = EventTimeline::new();
        tl.record(0, EventKind::ComputeStart, 0, "s0");
        tl.record(0, EventKind::ComputeEnd, 100, "s0");
        assert_eq!(tl.len(), 2);
    }

    #[test]
    fn timeline_duration() {
        let mut tl = EventTimeline::new();
        tl.record(0, EventKind::TransferStart, 10, "s0");
        tl.record(0, EventKind::ComputeEnd, 500, "s0");
        assert_eq!(tl.total_duration_us(), 490);
    }

    #[test]
    fn timeline_filter_by_kind() {
        let mut tl = EventTimeline::new();
        tl.record(0, EventKind::ComputeStart, 0, "a");
        tl.record(0, EventKind::ComputeEnd, 100, "a");
        tl.record(1, EventKind::TransferStart, 50, "b");
        tl.record(1, EventKind::TransferEnd, 120, "b");
        assert_eq!(tl.filter_by_kind(EventKind::ComputeStart).len(), 1);
        assert_eq!(tl.filter_by_kind(EventKind::TransferEnd).len(), 1);
    }

    #[test]
    fn timeline_events_for_stage() {
        let mut tl = EventTimeline::new();
        tl.record(0, EventKind::ComputeStart, 0, "s0");
        tl.record(1, EventKind::ComputeStart, 10, "s1");
        tl.record(0, EventKind::ComputeEnd, 20, "s0");
        assert_eq!(tl.events_for_stage(0).len(), 2);
        assert_eq!(tl.events_for_stage(1).len(), 1);
    }

    #[test]
    fn timeline_is_ordered() {
        let mut tl = EventTimeline::new();
        tl.record(0, EventKind::ComputeStart, 0, "x");
        tl.record(0, EventKind::ComputeEnd, 100, "x");
        tl.record(1, EventKind::ComputeStart, 100, "y");
        assert!(tl.is_ordered());
    }

    #[test]
    fn timeline_not_ordered() {
        let mut tl = EventTimeline::new();
        tl.record(0, EventKind::ComputeEnd, 100, "x");
        tl.record(0, EventKind::ComputeStart, 0, "x");
        assert!(!tl.is_ordered());
    }

    #[test]
    fn timeline_default() {
        let tl = EventTimeline::default();
        assert!(tl.is_empty());
    }

    #[test]
    fn timeline_wall_elapsed_none_before_start() {
        let tl = EventTimeline::new();
        assert!(tl.wall_elapsed_us().is_none());
    }

    #[test]
    fn timeline_wall_elapsed_some_after_start() {
        let mut tl = EventTimeline::new();
        tl.mark_start();
        // Elapsed is non-deterministic but should be Some.
        assert!(tl.wall_elapsed_us().is_some());
    }

    // ── PipelineStats ──────────────────────────────────────────────

    #[test]
    fn stats_from_empty_timeline() {
        let tl = EventTimeline::new();
        let s = PipelineStats::from_timeline(&tl, 0);
        assert_eq!(s.overlap_ratio, 0.0);
        assert_eq!(s.total_time_us, 0);
    }

    #[test]
    fn stats_no_overlap() {
        let mut tl = EventTimeline::new();
        // Compute [0, 100], Transfer [100, 200] — no overlap.
        tl.record(0, EventKind::ComputeStart, 0, "s0");
        tl.record(0, EventKind::ComputeEnd, 100, "s0");
        tl.record(0, EventKind::TransferStart, 100, "s0");
        tl.record(0, EventKind::TransferEnd, 200, "s0");
        let s = PipelineStats::from_timeline(&tl, 1);
        assert_eq!(s.overlap_ratio, 0.0);
        assert_eq!(s.stall_time_us, 0);
        assert_eq!(s.total_time_us, 200);
    }

    #[test]
    fn stats_full_overlap() {
        let mut tl = EventTimeline::new();
        // Compute [0, 100], Transfer [0, 100] — 100% overlap.
        tl.record(0, EventKind::ComputeStart, 0, "s0");
        tl.record(0, EventKind::ComputeEnd, 100, "s0");
        tl.record(0, EventKind::TransferStart, 0, "s0");
        tl.record(0, EventKind::TransferEnd, 100, "s0");
        let s = PipelineStats::from_timeline(&tl, 1);
        assert!((s.overlap_ratio - 0.5).abs() < 0.01);
        assert_eq!(s.stall_time_us, 0);
    }

    #[test]
    fn stats_partial_overlap() {
        let mut tl = EventTimeline::new();
        // Compute [0, 100], Transfer [50, 150] — 50µs overlap.
        tl.record(0, EventKind::ComputeStart, 0, "s0");
        tl.record(0, EventKind::ComputeEnd, 100, "s0");
        tl.record(0, EventKind::TransferStart, 50, "s0");
        tl.record(0, EventKind::TransferEnd, 150, "s0");
        let s = PipelineStats::from_timeline(&tl, 1);
        // overlap=50, total compute+xfer = 200, ratio = 50/200 = 0.25
        assert!((s.overlap_ratio - 0.25).abs() < 0.01);
    }

    #[test]
    fn stats_display() {
        let mut tl = EventTimeline::new();
        tl.record(0, EventKind::ComputeStart, 0, "s");
        tl.record(0, EventKind::ComputeEnd, 100, "s");
        let s = PipelineStats::from_timeline(&tl, 1);
        let d = format!("{s}");
        assert!(d.contains("PipelineStats"));
        assert!(d.contains("overlap="));
    }

    #[test]
    fn stats_stall_time() {
        let mut tl = EventTimeline::new();
        // Compute [0, 50], gap, Transfer [100, 150] → stall = 50.
        tl.record(0, EventKind::ComputeStart, 0, "s");
        tl.record(0, EventKind::ComputeEnd, 50, "s");
        tl.record(0, EventKind::TransferStart, 100, "s");
        tl.record(0, EventKind::TransferEnd, 150, "s");
        let s = PipelineStats::from_timeline(&tl, 1);
        assert_eq!(s.stall_time_us, 50);
    }

    // ── ExecutorError ──────────────────────────────────────────────

    #[test]
    fn executor_error_display() {
        let e = ExecutorError::EmptyPipeline;
        assert_eq!(format!("{e}"), "pipeline has no stages");
    }

    #[test]
    fn executor_error_from_dep_error() {
        let de = DependencyError::CyclicDependency;
        let ee: ExecutorError = de.into();
        assert!(matches!(ee, ExecutorError::DependencyError(DependencyError::CyclicDependency)));
    }

    // ── PipelineExecutor: construction ─────────────────────────────

    #[test]
    fn executor_empty_pipeline_rejected() {
        let r = PipelineExecutor::new(vec![], false);
        assert!(matches!(r, Err(ExecutorError::EmptyPipeline)));
    }

    #[test]
    fn executor_single_stage() {
        let stages = vec![PipelineStage::new("only")];
        let ex = PipelineExecutor::new(stages, false).unwrap();
        assert_eq!(ex.stage_count(), 1);
    }

    #[test]
    fn executor_auto_infers_deps() {
        let stages = vec![
            PipelineStage::new("a").with_output(BufferDescriptor::new("buf0", 64)),
            PipelineStage::new("b")
                .with_input(BufferDescriptor::new("buf0", 64))
                .with_output(BufferDescriptor::new("buf1", 64)),
        ];
        let ex = PipelineExecutor::new(stages, false).unwrap();
        let deps = ex.dependency_graph().dependencies_of(1);
        assert!(deps.contains(&0));
    }

    #[test]
    fn executor_manual_deps() {
        let stages = build_linear_pipeline(3, 100, 50);
        let mut ex = PipelineExecutor::new(stages, true).unwrap();
        // No auto-deps → graph is empty.
        assert!(ex.dependency_graph().is_empty());
        // Manually add.
        ex.add_dependency(0, 1).unwrap();
        ex.add_dependency(1, 2).unwrap();
        assert_eq!(ex.dependency_graph().edge_count(), 2);
    }

    // ── PipelineExecutor: execution ────────────────────────────────

    #[test]
    fn executor_sequential_produces_output() {
        let stages = build_linear_pipeline(3, 100, 50);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        let r = ex.execute_sequential(&[]).unwrap();
        assert_eq!(r.stage_outputs.len(), 3);
        // stage 0 → 1.0 * 1 = 1.0, stage 1 → 1.0 * 2 = 2.0, stage 2 → 1.0 * 3 = 3.0
        assert!((r.stage_outputs[0] - 1.0).abs() < f64::EPSILON);
        assert!((r.stage_outputs[1] - 2.0).abs() < f64::EPSILON);
        assert!((r.stage_outputs[2] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn executor_sequential_custom_inputs() {
        let stages = build_linear_pipeline(2, 100, 50);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        let r = ex.execute_sequential(&[10.0, 20.0]).unwrap();
        assert!((r.stage_outputs[0] - 10.0).abs() < f64::EPSILON);
        assert!((r.stage_outputs[1] - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn executor_overlapped_faster_than_sequential() {
        let stages = build_linear_pipeline(4, 100, 50);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        let seq = ex.execute_sequential(&[]).unwrap();
        let ovl = ex.execute(&[]).unwrap();
        // With overlap, total time should be <= sequential.
        assert!(ovl.stats.total_time_us <= seq.stats.total_time_us);
    }

    #[test]
    fn executor_overlapped_same_outputs_as_sequential() {
        let stages = build_linear_pipeline(4, 100, 50);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        let seq = ex.execute_sequential(&[]).unwrap();
        let ovl = ex.execute(&[]).unwrap();
        for (a, b) in seq.stage_outputs.iter().zip(&ovl.stage_outputs) {
            assert!((a - b).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn executor_execution_order_respects_deps() {
        let stages = build_linear_pipeline(4, 100, 50);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        let r = ex.execute(&[]).unwrap();
        // Linear pipeline → order must be 0,1,2,3.
        assert_eq!(r.execution_order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn executor_timeline_records_events() {
        let stages = build_linear_pipeline(2, 100, 50);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        let r = ex.execute(&[]).unwrap();
        // Each stage produces ≥4 events (xfer start/end, compute start/end).
        assert!(r.timeline.len() >= 8);
    }

    #[test]
    fn executor_timeline_ordering() {
        let stages = build_linear_pipeline(3, 100, 50);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        let r = ex.execute_sequential(&[]).unwrap();
        // In purely sequential mode, all events should be ordered.
        assert!(r.timeline.is_ordered());
    }

    #[test]
    fn executor_with_double_buffer() {
        let stages = build_linear_pipeline(4, 100, 50);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        let db = DoubleBuffer::new("activation", 2048);
        ex.add_double_buffer(db);
        let r = ex.execute(&[]).unwrap();
        assert_eq!(r.stats.stages_executed, 4);
    }

    #[test]
    fn executor_depth_2() {
        let stages = build_linear_pipeline(2, 200, 100);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        let r = ex.execute(&[]).unwrap();
        assert_eq!(r.stats.stages_executed, 2);
        assert!(r.stats.total_time_us > 0);
    }

    #[test]
    fn executor_depth_8() {
        let stages = build_linear_pipeline(8, 50, 30);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        let r = ex.execute(&[]).unwrap();
        assert_eq!(r.stats.stages_executed, 8);
        assert_eq!(r.execution_order.len(), 8);
    }

    #[test]
    fn executor_single_stage_no_overlap() {
        let stages = build_linear_pipeline(1, 100, 50);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        let r = ex.execute(&[]).unwrap();
        assert_eq!(r.stats.stages_executed, 1);
        // Single stage → overlap ratio is 0.
        assert_eq!(r.stats.overlap_ratio, 0.0);
    }

    #[test]
    fn executor_set_overlap_off() {
        let stages = build_linear_pipeline(3, 100, 50);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        ex.set_overlap(false);
        let r = ex.execute(&[]).unwrap();
        // With overlap disabled, total time = sum of all stages' (compute+xfer).
        let expected: u64 = (0..3).map(|_| 100 + 50).sum();
        assert_eq!(r.stats.total_time_us, expected);
    }

    // ── Property-like tests ────────────────────────────────────────

    #[test]
    fn property_all_deps_satisfied_before_execution() {
        for depth in [2, 3, 4, 5, 6, 7, 8] {
            let stages = build_linear_pipeline(depth, 100, 50);
            let mut ex = PipelineExecutor::new(stages, false).unwrap();
            let r = ex.execute(&[]).unwrap();
            // The execution order must satisfy all dependencies.
            assert!(
                ex.dependency_graph().validate_order(&r.execution_order).is_ok(),
                "deps violated for depth={depth}"
            );
        }
    }

    #[test]
    fn property_stage_outputs_deterministic() {
        let stages = build_linear_pipeline(5, 80, 40);
        let mut ex = PipelineExecutor::new(stages, false).unwrap();
        let r1 = ex.execute(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let r2 = ex.execute(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(r1.stage_outputs, r2.stage_outputs);
    }

    #[test]
    fn property_overlapped_time_le_sequential() {
        for depth in [2, 4, 8] {
            let stages = build_linear_pipeline(depth, 100, 50);
            let mut ex = PipelineExecutor::new(stages, false).unwrap();
            let seq = ex.execute_sequential(&[]).unwrap();
            let ovl = ex.execute(&[]).unwrap();
            assert!(
                ovl.stats.total_time_us <= seq.stats.total_time_us,
                "overlap was slower at depth={depth}"
            );
        }
    }

    #[test]
    fn property_no_deps_any_order_valid() {
        // Independent stages (no shared buffers) → any order is valid.
        let stages: Vec<PipelineStage> = (0..4)
            .map(|i| {
                PipelineStage::new(format!("ind_{i}"))
                    .with_output(BufferDescriptor::new(format!("unique_{i}"), 64))
                    .with_compute_us(100)
                    .with_transfer_us(50)
            })
            .collect();
        let ex = PipelineExecutor::new(stages, false).unwrap();
        let g = ex.dependency_graph();
        // With no matching buffer names, all orders satisfy deps.
        assert!(g.validate_order(&[3, 1, 0, 2]).is_ok());
    }

    // ── build_linear_pipeline helper ───────────────────────────────

    #[test]
    fn build_linear_pipeline_stage_count() {
        let p = build_linear_pipeline(5, 100, 50);
        assert_eq!(p.len(), 5);
    }

    #[test]
    fn build_linear_pipeline_naming() {
        let p = build_linear_pipeline(3, 10, 5);
        assert_eq!(p[0].name, "stage_0");
        assert_eq!(p[2].name, "stage_2");
    }

    #[test]
    fn build_linear_pipeline_buffer_chaining() {
        let p = build_linear_pipeline(3, 10, 5);
        // Stage 0 has no input (first stage), but has output buf_0.
        assert!(p[0].inputs.is_empty());
        assert_eq!(p[0].outputs[0].name, "buf_0");
        // Stage 1 reads buf_0, writes buf_1.
        assert_eq!(p[1].inputs[0].name, "buf_0");
        assert_eq!(p[1].outputs[0].name, "buf_1");
    }

    // ── Additional edge-case tests ─────────────────────────────────

    #[test]
    fn dep_error_display_unmet() {
        let e = DependencyError::UnmetDependency { stage: 3, depends_on: 1 };
        let s = format!("{e}");
        assert!(s.contains("3"));
        assert!(s.contains("1"));
    }

    #[test]
    fn executor_error_stage_out_of_range_display() {
        let e = ExecutorError::StageIndexOutOfRange(99);
        assert!(format!("{e}").contains("99"));
    }

    #[test]
    fn executor_error_double_buffer_conflict_display() {
        let e = ExecutorError::DoubleBufferConflict("slot A busy".into());
        assert!(format!("{e}").contains("slot A busy"));
    }

    #[test]
    fn queue_kind_display() {
        assert_eq!(format!("{}", QueueKind::Compute), "compute");
        assert_eq!(format!("{}", QueueKind::Transfer), "transfer");
    }

    #[test]
    fn event_kind_display_variants() {
        assert_eq!(format!("{}", EventKind::TransferStart), "xfer_start");
        assert_eq!(format!("{}", EventKind::ComputeEnd), "compute_end");
        assert_eq!(format!("{}", EventKind::Barrier), "barrier");
        assert_eq!(format!("{}", EventKind::SwapBuffers), "swap");
    }

    #[test]
    fn stats_compute_util_capped() {
        let mut tl = EventTimeline::new();
        tl.record(0, EventKind::ComputeStart, 0, "s");
        tl.record(0, EventKind::ComputeEnd, 100, "s");
        let s = PipelineStats::from_timeline(&tl, 1);
        assert!(s.compute_util <= 1.0);
    }

    #[test]
    fn stats_transfer_util_capped() {
        let mut tl = EventTimeline::new();
        tl.record(0, EventKind::TransferStart, 0, "s");
        tl.record(0, EventKind::TransferEnd, 100, "s");
        let s = PipelineStats::from_timeline(&tl, 1);
        assert!(s.transfer_util <= 1.0);
    }

    #[test]
    fn double_buffer_begin_end_cycle() {
        let mut db = DoubleBuffer::new("cyc", 128);
        for _ in 0..5 {
            db.begin_transfer();
            assert!(db.is_dirty(db.transfer_slot()));
            db.end_transfer();
            assert!(!db.is_dirty(db.transfer_slot()));
            db.swap();
        }
        assert_eq!(db.swap_count(), 5);
    }
}
