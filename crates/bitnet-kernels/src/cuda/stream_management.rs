#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_contains)]
#![allow(clippy::needless_return)]
#![allow(clippy::manual_div_ceil)]
//! CUDA stream and event management for async GPU execution.
//!
//! # Overview
//!
//! Provides low-level wrappers around CUDA streams and events for
//! asynchronous kernel execution, synchronization, and pipelining:
//!
//! - [`CudaStream`] — RAII wrapper for a CUDA stream with create/destroy
//!   lifecycle and priority control.
//! - [`CudaEvent`] — RAII wrapper for a CUDA event used for inter-stream
//!   synchronization and elapsed-time measurement.
//! - [`StreamConfig`] — per-stream configuration (priority, non-blocking flag,
//!   custom flags).
//! - [`StreamPool`] — manages a collection of [`CudaStream`]s with
//!   round-robin dispatch and synchronization helpers.
//! - Free functions for stream/event operations: [`create_stream`],
//!   [`destroy_stream`], [`stream_synchronize`], [`create_event`],
//!   [`record_event`], [`wait_event`], [`elapsed_time`], [`stream_callback`],
//!   [`multi_stream_execute`], [`stream_ordered_alloc`],
//!   [`stream_ordered_free`], [`pipeline_stages_across_streams`].
//!
//! All GPU dispatch is feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations simulate stream behaviour with
//! single-threaded sequential execution.

use bitnet_common::{KernelError, Result};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ── Identifiers ──────────────────────────────────────────────────────

static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_EVENT_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_ALLOC_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a [`CudaStream`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(u64);

impl StreamId {
    fn next() -> Self {
        Self(NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Returns the raw numeric id.
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for StreamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stream-{}", self.0)
    }
}

/// Unique identifier for a [`CudaEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId(u64);

impl EventId {
    fn next() -> Self {
        Self(NEXT_EVENT_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Returns the raw numeric id.
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "event-{}", self.0)
    }
}

/// Unique identifier for a stream-ordered allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocId(u64);

impl AllocId {
    fn next() -> Self {
        Self(NEXT_ALLOC_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Returns the raw numeric id.
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "alloc-{}", self.0)
    }
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
    /// Critical — real-time / lowest-latency.
    Critical,
}

impl StreamPriority {
    /// Map to CUDA-compatible numeric priority (lower numeric = higher priority).
    pub fn as_cuda_priority(self) -> i32 {
        match self {
            Self::Low => 0,
            Self::Normal => -1,
            Self::High => -2,
            Self::Critical => -3,
        }
    }
}

// ── StreamFlags ──────────────────────────────────────────────────────

/// Flags controlling CUDA stream creation behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamFlags(u32);

impl StreamFlags {
    /// Default stream semantics (no flags set).
    pub const DEFAULT: Self = Self(0);
    /// Non-blocking: stream does not synchronize with the NULL stream.
    pub const NON_BLOCKING: Self = Self(1 << 0);
    /// Enable timing on events recorded to this stream.
    pub const ENABLE_TIMING: Self = Self(1 << 1);
    /// Disable timing on events (allows driver optimisations).
    pub const DISABLE_TIMING: Self = Self(1 << 2);
    /// Interprocess: the stream may be shared across processes.
    pub const INTERPROCESS: Self = Self(1 << 3);

    /// Returns the raw bits.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether `self` contains all bits in `other`.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no bits are set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Bitwise OR of two flag sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOr for StreamFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl Default for StreamFlags {
    fn default() -> Self {
        Self::NON_BLOCKING
    }
}

// ── StreamConfig ─────────────────────────────────────────────────────

/// Per-stream configuration.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Priority level.
    pub priority: StreamPriority,
    /// Whether the stream is non-blocking w.r.t. the NULL stream.
    pub non_blocking: bool,
    /// Additional creation flags.
    pub flags: StreamFlags,
    /// Optional human-readable label for debugging.
    pub label: Option<String>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            priority: StreamPriority::Normal,
            non_blocking: true,
            flags: StreamFlags::NON_BLOCKING,
            label: None,
        }
    }
}

impl StreamConfig {
    /// Create a new config with the given priority.
    pub fn with_priority(priority: StreamPriority) -> Self {
        Self { priority, ..Default::default() }
    }

    /// Create a blocking (legacy NULL-stream synchronised) stream config.
    pub fn blocking() -> Self {
        Self { non_blocking: false, flags: StreamFlags::DEFAULT, ..Default::default() }
    }

    /// Validate configuration consistency.
    pub fn validate(&self) -> Result<()> {
        if self.flags.contains(StreamFlags::ENABLE_TIMING)
            && self.flags.contains(StreamFlags::DISABLE_TIMING)
        {
            return Err(KernelError::InvalidArguments {
                reason: "ENABLE_TIMING and DISABLE_TIMING are mutually exclusive".into(),
            }
            .into());
        }
        Ok(())
    }

    /// Effective CUDA stream creation flags.
    pub fn effective_flags(&self) -> u32 {
        let mut f = self.flags.bits();
        if self.non_blocking {
            f |= StreamFlags::NON_BLOCKING.bits();
        }
        f
    }
}

// ── StreamState ──────────────────────────────────────────────────────

/// Runtime state of a [`CudaStream`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// Stream is idle.
    Idle,
    /// Stream has pending work.
    Active,
    /// Stream has been synchronised and is now idle.
    Synchronized,
    /// Stream has been destroyed.
    Destroyed,
}

// ── CudaStream ───────────────────────────────────────────────────────

/// RAII wrapper around a CUDA stream.
///
/// On CPU builds the stream is a logical handle that tracks submitted
/// operations sequentially for testing and simulation.
#[derive(Debug)]
pub struct CudaStream {
    /// Unique identifier.
    pub id: StreamId,
    /// Configuration used at creation.
    pub config: StreamConfig,
    /// Current state.
    state: StreamState,
    /// Number of operations submitted.
    ops_submitted: u64,
    /// Creation timestamp.
    created_at: Instant,
    /// Last synchronisation timestamp.
    last_sync: Option<Instant>,
    /// Recorded events on this stream (event_id → record time).
    recorded_events: HashMap<EventId, Instant>,
    /// Pending work duration (simulated).
    pending_work: Duration,
}

impl CudaStream {
    /// Create a new stream with the given configuration.
    fn new(config: StreamConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            id: StreamId::next(),
            config,
            state: StreamState::Idle,
            ops_submitted: 0,
            created_at: Instant::now(),
            last_sync: None,
            recorded_events: HashMap::new(),
            pending_work: Duration::ZERO,
        })
    }

    /// Returns the stream's unique identifier.
    pub fn id(&self) -> StreamId {
        self.id
    }

    /// Returns the current state.
    pub fn state(&self) -> StreamState {
        self.state
    }

    /// Returns the number of operations submitted.
    pub fn ops_submitted(&self) -> u64 {
        self.ops_submitted
    }

    /// Returns `true` if the stream has been destroyed.
    pub fn is_destroyed(&self) -> bool {
        self.state == StreamState::Destroyed
    }

    /// Submit simulated work to the stream.
    pub fn submit_work(&mut self, duration: Duration) -> Result<()> {
        if self.state == StreamState::Destroyed {
            return Err(KernelError::InvalidArguments {
                reason: "cannot submit work to destroyed stream".into(),
            }
            .into());
        }
        self.ops_submitted += 1;
        self.pending_work += duration;
        self.state = StreamState::Active;
        Ok(())
    }

    /// Synchronise: wait for all pending work to complete (CPU simulation).
    pub fn synchronize(&mut self) -> Result<()> {
        if self.state == StreamState::Destroyed {
            return Err(KernelError::InvalidArguments {
                reason: "cannot synchronize destroyed stream".into(),
            }
            .into());
        }
        self.pending_work = Duration::ZERO;
        self.state = StreamState::Synchronized;
        self.last_sync = Some(Instant::now());
        Ok(())
    }

    /// Record an event on this stream.
    pub fn record_event(&mut self, event_id: EventId) -> Result<()> {
        if self.state == StreamState::Destroyed {
            return Err(KernelError::InvalidArguments {
                reason: "cannot record event on destroyed stream".into(),
            }
            .into());
        }
        self.recorded_events.insert(event_id, Instant::now());
        Ok(())
    }

    /// Mark the stream as destroyed.
    fn mark_destroyed(&mut self) {
        self.state = StreamState::Destroyed;
        self.pending_work = Duration::ZERO;
    }

    /// Duration since stream creation.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Duration since last synchronisation, if any.
    pub fn time_since_sync(&self) -> Option<Duration> {
        self.last_sync.map(|t| t.elapsed())
    }

    /// Pending work duration (simulated).
    pub fn pending_work(&self) -> Duration {
        self.pending_work
    }
}

impl fmt::Display for CudaStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CudaStream({}, {:?}, ops={}, state={:?})",
            self.id, self.config.priority, self.ops_submitted, self.state
        )
    }
}

// ── EventState ───────────────────────────────────────────────────────

/// Runtime state of a [`CudaEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventState {
    /// Created but not yet recorded.
    Created,
    /// Recorded on a stream — captures a point in the stream's work queue.
    Recorded,
    /// Completed (all work prior to the record point has finished).
    Completed,
    /// Destroyed.
    Destroyed,
}

// ── CudaEvent ────────────────────────────────────────────────────────

/// RAII wrapper around a CUDA event for synchronisation and timing.
///
/// On CPU builds the event is a logical marker with wall-clock timestamps.
#[derive(Debug)]
pub struct CudaEvent {
    /// Unique identifier.
    pub id: EventId,
    /// Whether timing is enabled.
    pub timing_enabled: bool,
    /// Current state.
    state: EventState,
    /// Timestamp when the event was recorded.
    record_time: Option<Instant>,
    /// Stream on which the event was recorded.
    recorded_on: Option<StreamId>,
    /// Creation timestamp.
    created_at: Instant,
}

impl CudaEvent {
    /// Create a new event.
    fn new(timing_enabled: bool) -> Self {
        Self {
            id: EventId::next(),
            timing_enabled,
            state: EventState::Created,
            record_time: None,
            recorded_on: None,
            created_at: Instant::now(),
        }
    }

    /// Returns the event's unique identifier.
    pub fn id(&self) -> EventId {
        self.id
    }

    /// Returns the current state.
    pub fn state(&self) -> EventState {
        self.state
    }

    /// Returns `true` if the event has been recorded.
    pub fn is_recorded(&self) -> bool {
        matches!(self.state, EventState::Recorded | EventState::Completed)
    }

    /// Returns the stream on which the event was last recorded.
    pub fn recorded_on(&self) -> Option<StreamId> {
        self.recorded_on
    }

    /// Record this event on a stream (CPU simulation: capture wall-clock).
    pub fn record(&mut self, stream_id: StreamId) -> Result<()> {
        if self.state == EventState::Destroyed {
            return Err(KernelError::InvalidArguments {
                reason: "cannot record destroyed event".into(),
            }
            .into());
        }
        self.record_time = Some(Instant::now());
        self.recorded_on = Some(stream_id);
        self.state = EventState::Recorded;
        Ok(())
    }

    /// Mark the event as completed.
    pub fn complete(&mut self) {
        if self.state == EventState::Recorded {
            self.state = EventState::Completed;
        }
    }

    /// Mark the event as destroyed.
    fn mark_destroyed(&mut self) {
        self.state = EventState::Destroyed;
    }

    /// Age of this event since creation.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Elapsed time between two events (requires both to be recorded with
    /// timing enabled).
    pub fn elapsed_since(&self, earlier: &CudaEvent) -> Result<Duration> {
        if !self.timing_enabled || !earlier.timing_enabled {
            return Err(KernelError::InvalidArguments {
                reason: "both events must have timing enabled".into(),
            }
            .into());
        }
        let t_start = earlier.record_time.ok_or_else(|| KernelError::InvalidArguments {
            reason: "start event not yet recorded".into(),
        })?;
        let t_end = self.record_time.ok_or_else(|| KernelError::InvalidArguments {
            reason: "end event not yet recorded".into(),
        })?;
        if t_end < t_start {
            return Err(KernelError::InvalidArguments {
                reason: "end event recorded before start event".into(),
            }
            .into());
        }
        Ok(t_end.duration_since(t_start))
    }
}

impl fmt::Display for CudaEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CudaEvent({}, state={:?}, timing={})", self.id, self.state, self.timing_enabled)
    }
}

// ── StreamCallback ───────────────────────────────────────────────────

/// A host-side callback that can be enqueued on a stream.
pub struct StreamCallback {
    /// Callback function.
    func: Box<dyn FnOnce() + Send + 'static>,
    /// Human-readable label.
    pub label: String,
}

impl StreamCallback {
    /// Create a new callback with a label and closure.
    pub fn new(label: impl Into<String>, func: impl FnOnce() + Send + 'static) -> Self {
        Self { func: Box::new(func), label: label.into() }
    }

    /// Execute the callback, consuming it.
    pub fn execute(self) {
        (self.func)();
    }
}

impl fmt::Debug for StreamCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamCallback").field("label", &self.label).finish()
    }
}

// ── StreamOrderedAlloc ───────────────────────────────────────────────

/// Represents a stream-ordered memory allocation.
#[derive(Debug, Clone)]
pub struct StreamOrderedAlloc {
    /// Allocation identifier.
    pub id: AllocId,
    /// Size in bytes.
    pub size: usize,
    /// Stream on which the allocation is ordered.
    pub stream_id: StreamId,
    /// Simulated device pointer (offset in a virtual address space).
    pub device_ptr: u64,
    /// Whether the allocation has been freed.
    pub freed: bool,
    /// Timestamp of allocation.
    pub allocated_at: Instant,
}

// ── PipelineStage ────────────────────────────────────────────────────

/// Kind of work in a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStageKind {
    /// Host-to-device memory transfer.
    HostToDevice,
    /// Compute kernel execution.
    Compute,
    /// Device-to-host memory transfer.
    DeviceToHost,
    /// Device-to-device copy.
    DeviceToDevice,
    /// Custom user-defined stage.
    Custom,
}

/// A single stage in a multi-stream pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage label.
    pub label: String,
    /// Kind of work.
    pub kind: PipelineStageKind,
    /// Estimated duration (used in CPU simulation).
    pub estimated_duration: Duration,
    /// Stream index (assigned during scheduling).
    pub stream_index: Option<usize>,
}

impl PipelineStage {
    /// Create a new pipeline stage.
    pub fn new(
        label: impl Into<String>,
        kind: PipelineStageKind,
        estimated_duration: Duration,
    ) -> Self {
        Self { label: label.into(), kind, estimated_duration, stream_index: None }
    }
}

// ── PipelineResult ───────────────────────────────────────────────────

/// Result of executing a multi-stream pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Per-stage completion times.
    pub stage_times: Vec<Duration>,
    /// Total wall-clock pipeline time.
    pub total_time: Duration,
    /// Number of streams used.
    pub streams_used: usize,
    /// Per-stream operation count.
    pub ops_per_stream: Vec<u64>,
}

// ── MultiStreamTask ──────────────────────────────────────────────────

/// A task to be executed on a specific stream.
#[derive(Debug)]
pub struct MultiStreamTask {
    /// Human-readable label.
    pub label: String,
    /// Estimated work duration (CPU simulation).
    pub duration: Duration,
    /// Target stream index (None = auto-assign).
    pub target_stream: Option<usize>,
}

impl MultiStreamTask {
    /// Create a new task.
    pub fn new(label: impl Into<String>, duration: Duration) -> Self {
        Self { label: label.into(), duration, target_stream: None }
    }

    /// Create a new task pinned to a specific stream.
    pub fn on_stream(label: impl Into<String>, duration: Duration, stream_index: usize) -> Self {
        Self { label: label.into(), duration, target_stream: Some(stream_index) }
    }
}

// ── MultiStreamResult ────────────────────────────────────────────────

/// Result of a multi-stream execution.
#[derive(Debug, Clone)]
pub struct MultiStreamResult {
    /// Number of tasks executed.
    pub tasks_executed: usize,
    /// Per-stream task counts.
    pub per_stream_tasks: Vec<u64>,
    /// Total simulated time.
    pub total_time: Duration,
}

// ── StreamPool ───────────────────────────────────────────────────────

/// Manages a collection of [`CudaStream`]s for concurrent GPU execution.
///
/// On CPU builds all operations execute sequentially — the pool exists
/// for API compatibility and testing.
#[derive(Debug)]
pub struct StreamPool {
    /// Owned streams.
    streams: Vec<CudaStream>,
    /// Events created through this pool.
    events: HashMap<EventId, CudaEvent>,
    /// Stream-ordered allocations.
    allocations: HashMap<AllocId, StreamOrderedAlloc>,
    /// Round-robin counter.
    rr_counter: usize,
    /// Next simulated device pointer.
    next_device_ptr: u64,
    /// Pool creation time.
    created_at: Instant,
}

impl StreamPool {
    /// Create a pool with `n` streams using the given config.
    pub fn new(n: usize, config: &StreamConfig) -> Result<Self> {
        if n == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "stream pool must have at least one stream".into(),
            }
            .into());
        }
        config.validate()?;
        let mut streams = Vec::with_capacity(n);
        for _ in 0..n {
            streams.push(CudaStream::new(config.clone())?);
        }
        Ok(Self {
            streams,
            events: HashMap::new(),
            allocations: HashMap::new(),
            rr_counter: 0,
            next_device_ptr: 0x1000,
            created_at: Instant::now(),
        })
    }

    /// Create a pool with 4 default-priority non-blocking streams.
    pub fn with_defaults() -> Result<Self> {
        Self::new(4, &StreamConfig::default())
    }

    /// Create a pool where each stream has a different priority.
    pub fn with_priorities(priorities: &[StreamPriority]) -> Result<Self> {
        if priorities.is_empty() {
            return Err(KernelError::InvalidArguments {
                reason: "must provide at least one priority".into(),
            }
            .into());
        }
        let mut streams = Vec::with_capacity(priorities.len());
        for &p in priorities {
            let cfg = StreamConfig::with_priority(p);
            streams.push(CudaStream::new(cfg)?);
        }
        Ok(Self {
            streams,
            events: HashMap::new(),
            allocations: HashMap::new(),
            rr_counter: 0,
            next_device_ptr: 0x1000,
            created_at: Instant::now(),
        })
    }

    /// Number of streams.
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Get a reference to a stream by index.
    pub fn stream(&self, index: usize) -> Result<&CudaStream> {
        self.streams.get(index).ok_or_else(|| {
            KernelError::InvalidArguments {
                reason: format!("stream index {} out of range ({})", index, self.streams.len()),
            }
            .into()
        })
    }

    /// Get a mutable reference to a stream by index.
    pub fn stream_mut(&mut self, index: usize) -> Result<&mut CudaStream> {
        let len = self.streams.len();
        self.streams.get_mut(index).ok_or_else(|| {
            KernelError::InvalidArguments {
                reason: format!("stream index {} out of range ({})", index, len),
            }
            .into()
        })
    }

    /// Round-robin: get the next stream index.
    pub fn next_stream_index(&mut self) -> usize {
        let idx = self.rr_counter % self.streams.len();
        self.rr_counter += 1;
        idx
    }

    /// Synchronise a specific stream.
    pub fn sync_stream(&mut self, index: usize) -> Result<()> {
        self.stream_mut(index)?.synchronize()
    }

    /// Synchronise all streams.
    pub fn sync_all(&mut self) -> Result<()> {
        for s in &mut self.streams {
            s.synchronize()?;
        }
        Ok(())
    }

    /// Create an event in the pool.
    pub fn create_event(&mut self, timing_enabled: bool) -> EventId {
        let event = CudaEvent::new(timing_enabled);
        let id = event.id;
        self.events.insert(id, event);
        id
    }

    /// Record an event on a stream.
    pub fn record_event_on_stream(&mut self, event_id: EventId, stream_index: usize) -> Result<()> {
        let stream = self.stream_mut(stream_index)?;
        let stream_id = stream.id();
        stream.record_event(event_id)?;
        let event = self.events.get_mut(&event_id).ok_or_else(|| {
            KernelError::InvalidArguments { reason: format!("event {} not found", event_id) }
        })?;
        event.record(stream_id)?;
        Ok(())
    }

    /// Make a stream wait on an event.
    pub fn wait_event_on_stream(&self, event_id: EventId, _stream_index: usize) -> Result<()> {
        let event = self.events.get(&event_id).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("event {} not found", event_id),
        })?;
        if !event.is_recorded() {
            return Err(KernelError::InvalidArguments {
                reason: "cannot wait on event that has not been recorded".into(),
            }
            .into());
        }
        // CPU simulation: nothing to wait for.
        Ok(())
    }

    /// Elapsed time between two events.
    pub fn elapsed_time(&self, start_event: EventId, end_event: EventId) -> Result<Duration> {
        let start = self.events.get(&start_event).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("start event {} not found", start_event),
        })?;
        let end = self.events.get(&end_event).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("end event {} not found", end_event),
        })?;
        end.elapsed_since(start)
    }

    /// Get an event by id.
    pub fn event(&self, id: EventId) -> Option<&CudaEvent> {
        self.events.get(&id)
    }

    /// Number of events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Destroy all streams and events.
    pub fn destroy_all(&mut self) {
        for s in &mut self.streams {
            s.mark_destroyed();
        }
        for e in self.events.values_mut() {
            e.mark_destroyed();
        }
    }

    /// Returns pool age.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Number of live (non-destroyed) allocations.
    pub fn live_alloc_count(&self) -> usize {
        self.allocations.values().filter(|a| !a.freed).count()
    }

    /// Total bytes currently allocated.
    pub fn total_allocated_bytes(&self) -> usize {
        self.allocations.values().filter(|a| !a.freed).map(|a| a.size).sum()
    }

    /// Get a reference to an allocation.
    pub fn allocation(&self, id: AllocId) -> Option<&StreamOrderedAlloc> {
        self.allocations.get(&id)
    }

    /// Iterate over all streams.
    pub fn iter(&self) -> impl Iterator<Item = &CudaStream> {
        self.streams.iter()
    }

    /// Get the stream with the least pending work.
    pub fn least_loaded_index(&self) -> usize {
        self.streams
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.pending_work)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ── Free functions ───────────────────────────────────────────────────

/// Create a new CUDA stream with the given configuration.
pub fn create_stream(config: StreamConfig) -> Result<CudaStream> {
    CudaStream::new(config)
}

/// Destroy a CUDA stream, releasing its resources.
pub fn destroy_stream(stream: &mut CudaStream) -> Result<()> {
    if stream.is_destroyed() {
        return Err(
            KernelError::InvalidArguments { reason: "stream already destroyed".into() }.into()
        );
    }
    stream.mark_destroyed();
    Ok(())
}

/// Synchronise a CUDA stream — block until all queued work completes.
pub fn stream_synchronize(stream: &mut CudaStream) -> Result<()> {
    stream.synchronize()
}

/// Create a new CUDA event.
pub fn create_event(timing_enabled: bool) -> CudaEvent {
    CudaEvent::new(timing_enabled)
}

/// Record an event on a stream.
pub fn record_event(event: &mut CudaEvent, stream: &mut CudaStream) -> Result<()> {
    let stream_id = stream.id();
    stream.record_event(event.id)?;
    event.record(stream_id)
}

/// Make the calling context wait on an event.
pub fn wait_event(event: &CudaEvent) -> Result<()> {
    if !event.is_recorded() {
        return Err(KernelError::InvalidArguments {
            reason: "cannot wait on event that has not been recorded".into(),
        }
        .into());
    }
    // CPU simulation: event already completed.
    Ok(())
}

/// Compute elapsed time between a start and end event.
pub fn elapsed_time(start: &CudaEvent, end: &CudaEvent) -> Result<Duration> {
    end.elapsed_since(start)
}

/// Enqueue a host-side callback on a stream.
///
/// In CPU simulation the callback executes immediately.
pub fn stream_callback(stream: &mut CudaStream, callback: StreamCallback) -> Result<()> {
    if stream.is_destroyed() {
        return Err(KernelError::InvalidArguments {
            reason: "cannot enqueue callback on destroyed stream".into(),
        }
        .into());
    }
    stream.ops_submitted += 1;
    // CPU simulation: execute immediately.
    callback.execute();
    Ok(())
}

/// Execute tasks across multiple streams in a pool.
///
/// Tasks with a `target_stream` are pinned; others are distributed
/// round-robin.
pub fn multi_stream_execute(
    pool: &mut StreamPool,
    tasks: &[MultiStreamTask],
) -> Result<MultiStreamResult> {
    if tasks.is_empty() {
        return Ok(MultiStreamResult {
            tasks_executed: 0,
            per_stream_tasks: vec![0; pool.len()],
            total_time: Duration::ZERO,
        });
    }
    let mut per_stream = vec![0u64; pool.len()];
    let start = Instant::now();
    for task in tasks {
        let idx = match task.target_stream {
            Some(i) if i < pool.len() => i,
            Some(i) => {
                return Err(KernelError::InvalidArguments {
                    reason: format!("target stream {} out of range ({})", i, pool.len()),
                }
                .into());
            }
            None => pool.next_stream_index(),
        };
        pool.stream_mut(idx)?.submit_work(task.duration)?;
        per_stream[idx] += 1;
    }
    let total = start.elapsed();
    Ok(MultiStreamResult {
        tasks_executed: tasks.len(),
        per_stream_tasks: per_stream,
        total_time: total,
    })
}

/// Allocate memory ordered on a stream.
///
/// In CPU simulation this returns a simulated device pointer.
pub fn stream_ordered_alloc(
    pool: &mut StreamPool,
    stream_index: usize,
    size: usize,
) -> Result<AllocId> {
    if size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "allocation size must be non-zero".into(),
        }
        .into());
    }
    let stream_id = pool.stream(stream_index)?.id();
    let id = AllocId::next();
    let ptr = pool.next_device_ptr;
    // Align to 256 bytes.
    let aligned_size = (size + 255) & !255;
    pool.next_device_ptr += aligned_size as u64;
    pool.allocations.insert(
        id,
        StreamOrderedAlloc {
            id,
            size,
            stream_id,
            device_ptr: ptr,
            freed: false,
            allocated_at: Instant::now(),
        },
    );
    Ok(id)
}

/// Free a stream-ordered allocation.
pub fn stream_ordered_free(pool: &mut StreamPool, alloc_id: AllocId) -> Result<()> {
    let alloc = pool.allocations.get_mut(&alloc_id).ok_or_else(|| {
        KernelError::InvalidArguments { reason: format!("allocation {} not found", alloc_id) }
    })?;
    if alloc.freed {
        return Err(KernelError::InvalidArguments {
            reason: format!("allocation {} already freed", alloc_id),
        }
        .into());
    }
    alloc.freed = true;
    Ok(())
}

/// Execute a multi-stage pipeline across streams.
///
/// Stages are assigned to streams round-robin by kind: transfers go to
/// dedicated streams when available, and compute to the remaining ones.
/// Events are inserted between stages for inter-stream synchronisation.
pub fn pipeline_stages_across_streams(
    pool: &mut StreamPool,
    stages: &[PipelineStage],
) -> Result<PipelineResult> {
    if stages.is_empty() {
        return Ok(PipelineResult {
            stage_times: vec![],
            total_time: Duration::ZERO,
            streams_used: 0,
            ops_per_stream: vec![0; pool.len()],
        });
    }

    let num_streams = pool.len();
    let mut stage_times = Vec::with_capacity(stages.len());
    let mut ops_per_stream = vec![0u64; num_streams];
    let mut prev_event: Option<EventId> = None;
    let start = Instant::now();

    for (i, stage) in stages.iter().enumerate() {
        // Assign stream: transfers → stream 0, compute → round-robin rest.
        let stream_idx = match stage.kind {
            PipelineStageKind::HostToDevice | PipelineStageKind::DeviceToHost => 0,
            PipelineStageKind::DeviceToDevice => 0.min(num_streams - 1),
            _ => {
                if num_streams > 1 {
                    1 + (i % (num_streams - 1))
                } else {
                    0
                }
            }
        };

        // Wait on previous stage's event if on a different stream.
        if let Some(evt) = prev_event {
            pool.wait_event_on_stream(evt, stream_idx)?;
        }

        pool.stream_mut(stream_idx)?.submit_work(stage.estimated_duration)?;
        ops_per_stream[stream_idx] += 1;
        stage_times.push(stage.estimated_duration);

        // Record event for the next stage.
        let evt = pool.create_event(true);
        pool.record_event_on_stream(evt, stream_idx)?;
        prev_event = Some(evt);
    }

    let total = start.elapsed();
    let streams_used = ops_per_stream.iter().filter(|&&c| c > 0).count();

    Ok(PipelineResult { stage_times, total_time: total, streams_used, ops_per_stream })
}

// ── GPU kernel source (placeholder for cudarc integration) ───────────

/// CUDA kernel source for stream-ordered memory operations.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const STREAM_MANAGEMENT_KERNEL_SRC: &str = r#"
extern "C" __global__
void stream_fill_f32(float* dst, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = value;
    }
}

extern "C" __global__
void stream_copy_f32(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

extern "C" __global__
void stream_scale_f32(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}
"#;

/// Launch configuration for stream-ordered kernels.
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[derive(Debug, Clone)]
pub struct StreamKernelLaunchConfig {
    /// Grid dimensions.
    pub grid: [u32; 3],
    /// Block dimensions.
    pub block: [u32; 3],
    /// Shared memory size in bytes.
    pub shared_mem: u32,
    /// Stream index to launch on.
    pub stream_index: usize,
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
impl StreamKernelLaunchConfig {
    /// Create a 1-D launch config for `n` elements.
    pub fn for_1d(n: usize, block_size: u32, stream_index: usize) -> Self {
        let grid_x = ((n as u32) + block_size - 1) / block_size;
        Self { grid: [grid_x, 1, 1], block: [block_size, 1, 1], shared_mem: 0, stream_index }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── StreamConfig tests ───────────────────────────────────────

    #[test]
    fn config_default_is_non_blocking() {
        let cfg = StreamConfig::default();
        assert!(cfg.non_blocking);
        assert_eq!(cfg.priority, StreamPriority::Normal);
        assert!(cfg.flags.contains(StreamFlags::NON_BLOCKING));
    }

    #[test]
    fn config_with_priority() {
        let cfg = StreamConfig::with_priority(StreamPriority::High);
        assert_eq!(cfg.priority, StreamPriority::High);
        assert!(cfg.non_blocking);
    }

    #[test]
    fn config_blocking() {
        let cfg = StreamConfig::blocking();
        assert!(!cfg.non_blocking);
        assert!(!cfg.flags.contains(StreamFlags::NON_BLOCKING));
    }

    #[test]
    fn config_validate_ok() {
        StreamConfig::default().validate().unwrap();
    }

    #[test]
    fn config_validate_conflicting_timing_flags() {
        let cfg = StreamConfig {
            flags: StreamFlags::ENABLE_TIMING | StreamFlags::DISABLE_TIMING,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_effective_flags_includes_non_blocking() {
        let cfg = StreamConfig::default();
        assert_ne!(cfg.effective_flags() & StreamFlags::NON_BLOCKING.bits(), 0);
    }

    #[test]
    fn config_label() {
        let cfg = StreamConfig { label: Some("compute".into()), ..Default::default() };
        assert_eq!(cfg.label.as_deref(), Some("compute"));
    }

    // ── StreamPriority tests ─────────────────────────────────────

    #[test]
    fn priority_ordering() {
        assert!(StreamPriority::Low < StreamPriority::Normal);
        assert!(StreamPriority::Normal < StreamPriority::High);
        assert!(StreamPriority::High < StreamPriority::Critical);
    }

    #[test]
    fn priority_cuda_mapping() {
        assert_eq!(StreamPriority::Low.as_cuda_priority(), 0);
        assert_eq!(StreamPriority::Normal.as_cuda_priority(), -1);
        assert_eq!(StreamPriority::High.as_cuda_priority(), -2);
        assert_eq!(StreamPriority::Critical.as_cuda_priority(), -3);
    }

    #[test]
    fn priority_default_is_normal() {
        assert_eq!(StreamPriority::default(), StreamPriority::Normal);
    }

    // ── StreamFlags tests ────────────────────────────────────────

    #[test]
    fn flags_default_is_non_blocking() {
        assert_eq!(StreamFlags::default(), StreamFlags::NON_BLOCKING);
    }

    #[test]
    fn flags_combinations() {
        let f = StreamFlags::NON_BLOCKING | StreamFlags::ENABLE_TIMING;
        assert!(f.contains(StreamFlags::NON_BLOCKING));
        assert!(f.contains(StreamFlags::ENABLE_TIMING));
        assert!(!f.contains(StreamFlags::INTERPROCESS));
    }

    #[test]
    fn flags_empty() {
        let f = StreamFlags::DEFAULT;
        assert!(f.is_empty());
    }

    // ── CudaStream tests ─────────────────────────────────────────

    #[test]
    fn stream_create_default() {
        let s = create_stream(StreamConfig::default()).unwrap();
        assert_eq!(s.state(), StreamState::Idle);
        assert_eq!(s.ops_submitted(), 0);
        assert!(!s.is_destroyed());
    }

    #[test]
    fn stream_create_high_priority() {
        let s = create_stream(StreamConfig::with_priority(StreamPriority::High)).unwrap();
        assert_eq!(s.config.priority, StreamPriority::High);
    }

    #[test]
    fn stream_submit_work() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        s.submit_work(Duration::from_millis(10)).unwrap();
        assert_eq!(s.state(), StreamState::Active);
        assert_eq!(s.ops_submitted(), 1);
        assert_eq!(s.pending_work(), Duration::from_millis(10));
    }

    #[test]
    fn stream_synchronize_clears_pending() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        s.submit_work(Duration::from_millis(50)).unwrap();
        stream_synchronize(&mut s).unwrap();
        assert_eq!(s.state(), StreamState::Synchronized);
        assert_eq!(s.pending_work(), Duration::ZERO);
    }

    #[test]
    fn stream_destroy() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        destroy_stream(&mut s).unwrap();
        assert!(s.is_destroyed());
        assert_eq!(s.state(), StreamState::Destroyed);
    }

    #[test]
    fn stream_double_destroy_errors() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        destroy_stream(&mut s).unwrap();
        assert!(destroy_stream(&mut s).is_err());
    }

    #[test]
    fn stream_submit_after_destroy_errors() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        destroy_stream(&mut s).unwrap();
        assert!(s.submit_work(Duration::from_millis(1)).is_err());
    }

    #[test]
    fn stream_sync_after_destroy_errors() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        destroy_stream(&mut s).unwrap();
        assert!(stream_synchronize(&mut s).is_err());
    }

    #[test]
    fn stream_unique_ids() {
        let a = create_stream(StreamConfig::default()).unwrap();
        let b = create_stream(StreamConfig::default()).unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn stream_display() {
        let s = create_stream(StreamConfig::default()).unwrap();
        let disp = format!("{}", s);
        assert!(disp.contains("CudaStream"));
    }

    #[test]
    fn stream_age_is_nonnegative() {
        let s = create_stream(StreamConfig::default()).unwrap();
        assert!(s.age() >= Duration::ZERO);
    }

    #[test]
    fn stream_time_since_sync_none_before_sync() {
        let s = create_stream(StreamConfig::default()).unwrap();
        assert!(s.time_since_sync().is_none());
    }

    #[test]
    fn stream_time_since_sync_some_after_sync() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        s.synchronize().unwrap();
        assert!(s.time_since_sync().is_some());
    }

    #[test]
    fn stream_multiple_submits_accumulate() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        s.submit_work(Duration::from_millis(10)).unwrap();
        s.submit_work(Duration::from_millis(20)).unwrap();
        assert_eq!(s.ops_submitted(), 2);
        assert_eq!(s.pending_work(), Duration::from_millis(30));
    }

    // ── CudaEvent tests ──────────────────────────────────────────

    #[test]
    fn event_create_with_timing() {
        let e = create_event(true);
        assert!(e.timing_enabled);
        assert_eq!(e.state(), EventState::Created);
        assert!(!e.is_recorded());
    }

    #[test]
    fn event_create_without_timing() {
        let e = create_event(false);
        assert!(!e.timing_enabled);
    }

    #[test]
    fn event_record_on_stream() {
        let mut e = create_event(true);
        let mut s = create_stream(StreamConfig::default()).unwrap();
        record_event(&mut e, &mut s).unwrap();
        assert!(e.is_recorded());
        assert_eq!(e.state(), EventState::Recorded);
        assert_eq!(e.recorded_on(), Some(s.id()));
    }

    #[test]
    fn event_wait_unrecorded_errors() {
        let e = create_event(true);
        assert!(wait_event(&e).is_err());
    }

    #[test]
    fn event_wait_recorded_ok() {
        let mut e = create_event(true);
        let mut s = create_stream(StreamConfig::default()).unwrap();
        record_event(&mut e, &mut s).unwrap();
        wait_event(&e).unwrap();
    }

    #[test]
    fn event_complete() {
        let mut e = create_event(true);
        let mut s = create_stream(StreamConfig::default()).unwrap();
        record_event(&mut e, &mut s).unwrap();
        e.complete();
        assert_eq!(e.state(), EventState::Completed);
    }

    #[test]
    fn event_display() {
        let e = create_event(true);
        let disp = format!("{}", e);
        assert!(disp.contains("CudaEvent"));
    }

    #[test]
    fn event_unique_ids() {
        let a = create_event(true);
        let b = create_event(true);
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn event_record_destroyed_errors() {
        let mut e = create_event(true);
        e.mark_destroyed();
        let mut s = create_stream(StreamConfig::default()).unwrap();
        assert!(record_event(&mut e, &mut s).is_err());
    }

    #[test]
    fn event_age() {
        let e = create_event(true);
        assert!(e.age() >= Duration::ZERO);
    }

    // ── elapsed_time tests ───────────────────────────────────────

    #[test]
    fn elapsed_time_between_events() {
        let mut e1 = create_event(true);
        let mut e2 = create_event(true);
        let mut s = create_stream(StreamConfig::default()).unwrap();
        record_event(&mut e1, &mut s).unwrap();
        std::thread::sleep(Duration::from_millis(1));
        record_event(&mut e2, &mut s).unwrap();
        let dt = elapsed_time(&e1, &e2).unwrap();
        assert!(dt >= Duration::from_millis(1));
    }

    #[test]
    fn elapsed_time_no_timing_errors() {
        let mut e1 = create_event(false);
        let mut e2 = create_event(false);
        let mut s = create_stream(StreamConfig::default()).unwrap();
        record_event(&mut e1, &mut s).unwrap();
        record_event(&mut e2, &mut s).unwrap();
        assert!(elapsed_time(&e1, &e2).is_err());
    }

    #[test]
    fn elapsed_time_unrecorded_start_errors() {
        let e1 = create_event(true);
        let mut e2 = create_event(true);
        let mut s = create_stream(StreamConfig::default()).unwrap();
        record_event(&mut e2, &mut s).unwrap();
        assert!(elapsed_time(&e1, &e2).is_err());
    }

    #[test]
    fn elapsed_time_unrecorded_end_errors() {
        let mut e1 = create_event(true);
        let e2 = create_event(true);
        let mut s = create_stream(StreamConfig::default()).unwrap();
        record_event(&mut e1, &mut s).unwrap();
        assert!(elapsed_time(&e1, &e2).is_err());
    }

    // ── stream_callback tests ────────────────────────────────────

    #[test]
    fn callback_executes() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f2 = flag.clone();
        let cb = StreamCallback::new("test", move || {
            f2.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        stream_callback(&mut s, cb).unwrap();
        assert!(flag.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn callback_on_destroyed_stream_errors() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        destroy_stream(&mut s).unwrap();
        let cb = StreamCallback::new("noop", || {});
        assert!(stream_callback(&mut s, cb).is_err());
    }

    #[test]
    fn callback_increments_ops() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        stream_callback(&mut s, StreamCallback::new("a", || {})).unwrap();
        stream_callback(&mut s, StreamCallback::new("b", || {})).unwrap();
        assert_eq!(s.ops_submitted(), 2);
    }

    #[test]
    fn callback_debug_format() {
        let cb = StreamCallback::new("test_cb", || {});
        let dbg = format!("{:?}", cb);
        assert!(dbg.contains("test_cb"));
    }

    // ── StreamPool tests ─────────────────────────────────────────

    #[test]
    fn pool_with_defaults() {
        let pool = StreamPool::with_defaults().unwrap();
        assert_eq!(pool.len(), 4);
        assert!(!pool.is_empty());
    }

    #[test]
    fn pool_custom_size() {
        let pool = StreamPool::new(8, &StreamConfig::default()).unwrap();
        assert_eq!(pool.len(), 8);
    }

    #[test]
    fn pool_zero_streams_errors() {
        assert!(StreamPool::new(0, &StreamConfig::default()).is_err());
    }

    #[test]
    fn pool_with_priorities() {
        let pool = StreamPool::with_priorities(&[
            StreamPriority::Low,
            StreamPriority::Normal,
            StreamPriority::High,
        ])
        .unwrap();
        assert_eq!(pool.len(), 3);
        assert_eq!(pool.stream(0).unwrap().config.priority, StreamPriority::Low);
        assert_eq!(pool.stream(2).unwrap().config.priority, StreamPriority::High);
    }

    #[test]
    fn pool_empty_priorities_errors() {
        assert!(StreamPool::with_priorities(&[]).is_err());
    }

    #[test]
    fn pool_stream_out_of_range() {
        let pool = StreamPool::with_defaults().unwrap();
        assert!(pool.stream(100).is_err());
    }

    #[test]
    fn pool_round_robin() {
        let mut pool = StreamPool::new(3, &StreamConfig::default()).unwrap();
        assert_eq!(pool.next_stream_index(), 0);
        assert_eq!(pool.next_stream_index(), 1);
        assert_eq!(pool.next_stream_index(), 2);
        assert_eq!(pool.next_stream_index(), 0); // wraps
    }

    #[test]
    fn pool_sync_stream() {
        let mut pool = StreamPool::with_defaults().unwrap();
        pool.stream_mut(0).unwrap().submit_work(Duration::from_millis(5)).unwrap();
        pool.sync_stream(0).unwrap();
        assert_eq!(pool.stream(0).unwrap().pending_work(), Duration::ZERO);
    }

    #[test]
    fn pool_sync_all() {
        let mut pool = StreamPool::with_defaults().unwrap();
        for i in 0..4 {
            pool.stream_mut(i).unwrap().submit_work(Duration::from_millis(1)).unwrap();
        }
        pool.sync_all().unwrap();
        for i in 0..4 {
            assert_eq!(pool.stream(i).unwrap().state(), StreamState::Synchronized);
        }
    }

    #[test]
    fn pool_create_and_record_event() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let eid = pool.create_event(true);
        pool.record_event_on_stream(eid, 0).unwrap();
        let e = pool.event(eid).unwrap();
        assert!(e.is_recorded());
    }

    #[test]
    fn pool_wait_event() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let eid = pool.create_event(true);
        pool.record_event_on_stream(eid, 0).unwrap();
        pool.wait_event_on_stream(eid, 1).unwrap();
    }

    #[test]
    fn pool_wait_unrecorded_event_errors() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let eid = pool.create_event(true);
        assert!(pool.wait_event_on_stream(eid, 0).is_err());
    }

    #[test]
    fn pool_elapsed_time() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let e1 = pool.create_event(true);
        pool.record_event_on_stream(e1, 0).unwrap();
        std::thread::sleep(Duration::from_millis(1));
        let e2 = pool.create_event(true);
        pool.record_event_on_stream(e2, 0).unwrap();
        let dt = pool.elapsed_time(e1, e2).unwrap();
        assert!(dt >= Duration::from_millis(1));
    }

    #[test]
    fn pool_event_count() {
        let mut pool = StreamPool::with_defaults().unwrap();
        assert_eq!(pool.event_count(), 0);
        pool.create_event(true);
        pool.create_event(false);
        assert_eq!(pool.event_count(), 2);
    }

    #[test]
    fn pool_destroy_all() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let eid = pool.create_event(true);
        pool.destroy_all();
        for i in 0..4 {
            assert!(pool.stream(i).unwrap().is_destroyed());
        }
        assert_eq!(pool.event(eid).unwrap().state(), EventState::Destroyed);
    }

    #[test]
    fn pool_age() {
        let pool = StreamPool::with_defaults().unwrap();
        assert!(pool.age() >= Duration::ZERO);
    }

    #[test]
    fn pool_least_loaded_index() {
        let mut pool = StreamPool::with_defaults().unwrap();
        pool.stream_mut(0).unwrap().submit_work(Duration::from_millis(100)).unwrap();
        pool.stream_mut(1).unwrap().submit_work(Duration::from_millis(50)).unwrap();
        let idx = pool.least_loaded_index();
        assert!(idx == 2 || idx == 3);
    }

    #[test]
    fn pool_iter_count() {
        let pool = StreamPool::with_defaults().unwrap();
        assert_eq!(pool.iter().count(), 4);
    }

    // ── multi_stream_execute tests ───────────────────────────────

    #[test]
    fn multi_execute_empty() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let res = multi_stream_execute(&mut pool, &[]).unwrap();
        assert_eq!(res.tasks_executed, 0);
    }

    #[test]
    fn multi_execute_round_robin() {
        let mut pool = StreamPool::new(2, &StreamConfig::default()).unwrap();
        let tasks: Vec<_> = (0..4)
            .map(|i| MultiStreamTask::new(format!("t{i}"), Duration::from_millis(1)))
            .collect();
        let res = multi_stream_execute(&mut pool, &tasks).unwrap();
        assert_eq!(res.tasks_executed, 4);
        assert_eq!(res.per_stream_tasks, vec![2, 2]);
    }

    #[test]
    fn multi_execute_pinned() {
        let mut pool = StreamPool::new(3, &StreamConfig::default()).unwrap();
        let tasks = vec![
            MultiStreamTask::on_stream("a", Duration::from_millis(1), 2),
            MultiStreamTask::on_stream("b", Duration::from_millis(1), 2),
        ];
        let res = multi_stream_execute(&mut pool, &tasks).unwrap();
        assert_eq!(res.per_stream_tasks[2], 2);
    }

    #[test]
    fn multi_execute_out_of_range_errors() {
        let mut pool = StreamPool::new(2, &StreamConfig::default()).unwrap();
        let tasks = vec![MultiStreamTask::on_stream("x", Duration::from_millis(1), 99)];
        assert!(multi_stream_execute(&mut pool, &tasks).is_err());
    }

    #[test]
    fn multi_execute_large_batch() {
        let mut pool = StreamPool::new(4, &StreamConfig::default()).unwrap();
        let tasks: Vec<_> = (0..200)
            .map(|i| MultiStreamTask::new(format!("t{i}"), Duration::from_millis(1)))
            .collect();
        let res = multi_stream_execute(&mut pool, &tasks).unwrap();
        assert_eq!(res.tasks_executed, 200);
        assert_eq!(res.per_stream_tasks.iter().sum::<u64>(), 200);
    }

    // ── stream_ordered_alloc / free tests ────────────────────────

    #[test]
    fn alloc_basic() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let id = stream_ordered_alloc(&mut pool, 0, 1024).unwrap();
        assert_eq!(pool.live_alloc_count(), 1);
        assert_eq!(pool.total_allocated_bytes(), 1024);
        let a = pool.allocation(id).unwrap();
        assert!(!a.freed);
        assert_eq!(a.size, 1024);
    }

    #[test]
    fn alloc_zero_size_errors() {
        let mut pool = StreamPool::with_defaults().unwrap();
        assert!(stream_ordered_alloc(&mut pool, 0, 0).is_err());
    }

    #[test]
    fn alloc_and_free() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let id = stream_ordered_alloc(&mut pool, 0, 512).unwrap();
        stream_ordered_free(&mut pool, id).unwrap();
        assert_eq!(pool.live_alloc_count(), 0);
        assert_eq!(pool.total_allocated_bytes(), 0);
    }

    #[test]
    fn alloc_double_free_errors() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let id = stream_ordered_alloc(&mut pool, 0, 256).unwrap();
        stream_ordered_free(&mut pool, id).unwrap();
        assert!(stream_ordered_free(&mut pool, id).is_err());
    }

    #[test]
    fn alloc_multiple() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let a = stream_ordered_alloc(&mut pool, 0, 100).unwrap();
        let b = stream_ordered_alloc(&mut pool, 1, 200).unwrap();
        let c = stream_ordered_alloc(&mut pool, 2, 300).unwrap();
        assert_eq!(pool.live_alloc_count(), 3);
        assert_eq!(pool.total_allocated_bytes(), 600);
        stream_ordered_free(&mut pool, b).unwrap();
        assert_eq!(pool.live_alloc_count(), 2);
        assert_eq!(pool.total_allocated_bytes(), 400);
        stream_ordered_free(&mut pool, a).unwrap();
        stream_ordered_free(&mut pool, c).unwrap();
        assert_eq!(pool.live_alloc_count(), 0);
    }

    #[test]
    fn alloc_nonexistent_errors() {
        let mut pool = StreamPool::with_defaults().unwrap();
        assert!(stream_ordered_free(&mut pool, AllocId(99999)).is_err());
    }

    #[test]
    fn alloc_device_ptrs_are_aligned() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let a = stream_ordered_alloc(&mut pool, 0, 1).unwrap();
        let b = stream_ordered_alloc(&mut pool, 0, 1).unwrap();
        let pa = pool.allocation(a).unwrap().device_ptr;
        let pb = pool.allocation(b).unwrap().device_ptr;
        assert_eq!(pb - pa, 256);
    }

    #[test]
    fn alloc_out_of_range_stream_errors() {
        let mut pool = StreamPool::with_defaults().unwrap();
        assert!(stream_ordered_alloc(&mut pool, 99, 1024).is_err());
    }

    // ── pipeline_stages_across_streams tests ─────────────────────

    #[test]
    fn pipeline_empty() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let res = pipeline_stages_across_streams(&mut pool, &[]).unwrap();
        assert_eq!(res.streams_used, 0);
        assert!(res.stage_times.is_empty());
    }

    #[test]
    fn pipeline_single_stage() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let stages = vec![PipelineStage::new(
            "compute",
            PipelineStageKind::Compute,
            Duration::from_millis(10),
        )];
        let res = pipeline_stages_across_streams(&mut pool, &stages).unwrap();
        assert_eq!(res.stage_times.len(), 1);
        assert_eq!(res.streams_used, 1);
    }

    #[test]
    fn pipeline_h2d_compute_d2h() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let stages = vec![
            PipelineStage::new("h2d", PipelineStageKind::HostToDevice, Duration::from_millis(5)),
            PipelineStage::new("compute", PipelineStageKind::Compute, Duration::from_millis(10)),
            PipelineStage::new("d2h", PipelineStageKind::DeviceToHost, Duration::from_millis(5)),
        ];
        let res = pipeline_stages_across_streams(&mut pool, &stages).unwrap();
        assert_eq!(res.stage_times.len(), 3);
        assert!(res.streams_used >= 1);
    }

    #[test]
    fn pipeline_transfers_use_stream_zero() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let stages = vec![
            PipelineStage::new("h2d", PipelineStageKind::HostToDevice, Duration::from_millis(1)),
            PipelineStage::new("d2h", PipelineStageKind::DeviceToHost, Duration::from_millis(1)),
        ];
        let res = pipeline_stages_across_streams(&mut pool, &stages).unwrap();
        assert_eq!(res.ops_per_stream[0], 2);
    }

    #[test]
    fn pipeline_compute_spreads_across_streams() {
        let mut pool = StreamPool::new(4, &StreamConfig::default()).unwrap();
        let stages: Vec<_> = (0..6)
            .map(|i| {
                PipelineStage::new(
                    format!("c{i}"),
                    PipelineStageKind::Compute,
                    Duration::from_millis(1),
                )
            })
            .collect();
        let res = pipeline_stages_across_streams(&mut pool, &stages).unwrap();
        assert!(res.ops_per_stream[0] == 0 || res.streams_used > 1);
    }

    #[test]
    fn pipeline_single_stream_pool() {
        let mut pool = StreamPool::new(1, &StreamConfig::default()).unwrap();
        let stages = vec![
            PipelineStage::new("h2d", PipelineStageKind::HostToDevice, Duration::from_millis(1)),
            PipelineStage::new("compute", PipelineStageKind::Compute, Duration::from_millis(1)),
        ];
        let res = pipeline_stages_across_streams(&mut pool, &stages).unwrap();
        assert_eq!(res.ops_per_stream[0], 2);
    }

    #[test]
    fn pipeline_d2d_stage() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let stages = vec![PipelineStage::new(
            "d2d",
            PipelineStageKind::DeviceToDevice,
            Duration::from_millis(2),
        )];
        let res = pipeline_stages_across_streams(&mut pool, &stages).unwrap();
        assert_eq!(res.stage_times.len(), 1);
    }

    #[test]
    fn pipeline_custom_stage() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let stages =
            vec![PipelineStage::new("custom", PipelineStageKind::Custom, Duration::from_millis(3))];
        let res = pipeline_stages_across_streams(&mut pool, &stages).unwrap();
        assert_eq!(res.stage_times.len(), 1);
    }

    #[test]
    fn pipeline_many_stages() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let stages: Vec<_> = (0..100)
            .map(|i| {
                PipelineStage::new(
                    format!("s{i}"),
                    if i % 3 == 0 {
                        PipelineStageKind::HostToDevice
                    } else {
                        PipelineStageKind::Compute
                    },
                    Duration::from_millis(1),
                )
            })
            .collect();
        let res = pipeline_stages_across_streams(&mut pool, &stages).unwrap();
        assert_eq!(res.stage_times.len(), 100);
    }

    // ── Integration / end-to-end tests ───────────────────────────

    #[test]
    fn end_to_end_create_submit_sync_destroy() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        s.submit_work(Duration::from_millis(5)).unwrap();
        stream_synchronize(&mut s).unwrap();
        destroy_stream(&mut s).unwrap();
        assert!(s.is_destroyed());
    }

    #[test]
    fn end_to_end_event_timing() {
        let mut s = create_stream(StreamConfig::default()).unwrap();
        let mut e1 = create_event(true);
        record_event(&mut e1, &mut s).unwrap();
        s.submit_work(Duration::from_millis(10)).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        let mut e2 = create_event(true);
        record_event(&mut e2, &mut s).unwrap();
        let dt = elapsed_time(&e1, &e2).unwrap();
        assert!(dt >= Duration::from_millis(1));
    }

    #[test]
    fn end_to_end_pool_alloc_pipeline() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let aid = stream_ordered_alloc(&mut pool, 0, 4096).unwrap();
        let stages = vec![
            PipelineStage::new("h2d", PipelineStageKind::HostToDevice, Duration::from_millis(1)),
            PipelineStage::new("compute", PipelineStageKind::Compute, Duration::from_millis(5)),
            PipelineStage::new("d2h", PipelineStageKind::DeviceToHost, Duration::from_millis(1)),
        ];
        let res = pipeline_stages_across_streams(&mut pool, &stages).unwrap();
        assert_eq!(res.stage_times.len(), 3);
        stream_ordered_free(&mut pool, aid).unwrap();
        assert_eq!(pool.live_alloc_count(), 0);
    }

    #[test]
    fn end_to_end_pool_event_sync_between_streams() {
        let mut pool = StreamPool::with_defaults().unwrap();
        pool.stream_mut(0).unwrap().submit_work(Duration::from_millis(5)).unwrap();
        let eid = pool.create_event(true);
        pool.record_event_on_stream(eid, 0).unwrap();
        pool.wait_event_on_stream(eid, 1).unwrap();
        pool.sync_all().unwrap();
    }

    #[test]
    fn end_to_end_multi_execute_then_sync() {
        let mut pool = StreamPool::new(3, &StreamConfig::default()).unwrap();
        let tasks: Vec<_> = (0..9)
            .map(|i| MultiStreamTask::new(format!("task{i}"), Duration::from_millis(1)))
            .collect();
        let res = multi_stream_execute(&mut pool, &tasks).unwrap();
        assert_eq!(res.tasks_executed, 9);
        pool.sync_all().unwrap();
        for i in 0..3 {
            assert_eq!(pool.stream(i).unwrap().state(), StreamState::Synchronized);
        }
    }

    #[test]
    fn end_to_end_callback_in_pipeline() {
        let mut pool = StreamPool::with_defaults().unwrap();
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f2 = flag.clone();
        let s = pool.stream_mut(0).unwrap();
        stream_callback(
            s,
            StreamCallback::new("marker", move || {
                f2.store(true, std::sync::atomic::Ordering::SeqCst);
            }),
        )
        .unwrap();
        assert!(flag.load(std::sync::atomic::Ordering::SeqCst));
    }

    // ── Identifier tests ─────────────────────────────────────────

    #[test]
    fn stream_id_display() {
        let id = StreamId(42);
        assert_eq!(format!("{}", id), "stream-42");
    }

    #[test]
    fn event_id_display() {
        let id = EventId(7);
        assert_eq!(format!("{}", id), "event-7");
    }

    #[test]
    fn alloc_id_display() {
        let id = AllocId(99);
        assert_eq!(format!("{}", id), "alloc-99");
    }

    #[test]
    fn stream_id_raw() {
        let id = StreamId(5);
        assert_eq!(id.raw(), 5);
    }

    #[test]
    fn event_id_raw() {
        let id = EventId(10);
        assert_eq!(id.raw(), 10);
    }

    #[test]
    fn alloc_id_raw() {
        let id = AllocId(3);
        assert_eq!(id.raw(), 3);
    }

    // ── GPU-gated tests ──────────────────────────────────────────

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn kernel_src_not_empty() {
        assert!(!STREAM_MANAGEMENT_KERNEL_SRC.is_empty());
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn launch_config_1d() {
        let cfg = StreamKernelLaunchConfig::for_1d(1024, 256, 0);
        assert_eq!(cfg.grid, [4, 1, 1]);
        assert_eq!(cfg.block, [256, 1, 1]);
        assert_eq!(cfg.stream_index, 0);
    }
}
