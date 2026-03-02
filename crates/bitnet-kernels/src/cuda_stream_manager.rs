//! High-level CUDA stream manager with priority scheduling, event synchronization,
//! stream recycling, and workload-aware stream selection.
//!
//! # Overview
//!
//! This module provides a production-grade stream management layer that sits above
//! the low-level CUDA stream pool. Key abstractions:
//!
//! - [`StreamManager`] — manages a pool of [`CudaStream`]s with priority tiers,
//!   automatic workload-based selection, and stream recycling.
//! - [`CudaStream`] — wraps a logical stream with priority, device affinity,
//!   callback support, and timeline tracking.
//! - [`StreamEvent`] — inter-stream synchronization primitive for ordering work
//!   across streams.
//!
//! All public types are feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! The CPU fallback executes operations sequentially while preserving the same API.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ── Global ID generators ─────────────────────────────────────────────

static STREAM_ID_GEN: AtomicU64 = AtomicU64::new(1);
static EVENT_ID_GEN: AtomicU64 = AtomicU64::new(1);

fn next_stream_id() -> u64 {
    STREAM_ID_GEN.fetch_add(1, Ordering::Relaxed)
}

fn next_event_id() -> u64 {
    EVENT_ID_GEN.fetch_add(1, Ordering::Relaxed)
}

// ── Priority ─────────────────────────────────────────────────────────

/// Priority level for a CUDA stream.
///
/// CUDA maps lower numeric values to higher priority.  [`Priority::High`] maps
/// to CUDA priority −2 while [`Priority::Low`] maps to 0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Background / best-effort work.
    Low,
    /// Default priority.
    #[default]
    Normal,
    /// Latency-sensitive work.
    High,
}

impl Priority {
    /// Convert to a CUDA-compatible integer priority (lower = higher priority).
    pub fn as_cuda_priority(self) -> i32 {
        match self {
            Self::Low => 0,
            Self::Normal => -1,
            Self::High => -2,
        }
    }

    /// Return all priority levels ordered from lowest to highest.
    pub fn all() -> [Priority; 3] {
        [Priority::Low, Priority::Normal, Priority::High]
    }
}

// ── WorkloadType ─────────────────────────────────────────────────────

/// Describes the kind of work to be submitted, used by the stream manager for
/// automatic stream selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    /// Compute-heavy kernel (e.g. matmul, attention).
    Compute,
    /// Host-to-device memory transfer.
    HostToDevice,
    /// Device-to-host memory transfer.
    DeviceToHost,
    /// Device-to-device copy or peer transfer.
    DeviceToDevice,
}

impl WorkloadType {
    /// Suggest a default priority for this workload type.
    pub fn default_priority(self) -> Priority {
        match self {
            Self::Compute => Priority::Normal,
            Self::HostToDevice => Priority::High,
            Self::DeviceToHost => Priority::Low,
            Self::DeviceToDevice => Priority::Normal,
        }
    }
}

// ── Callback ─────────────────────────────────────────────────────────

/// A completion callback that can be attached to a [`CudaStream`].
///
/// The callback fires (in the CPU fallback) immediately when work on the stream
/// is marked complete.  On real CUDA hardware this would be invoked from a
/// host-side callback registered via `cuLaunchHostFunc`.
pub type CompletionCallback = Box<dyn FnOnce() + Send>;

// ── TimelineEntry ────────────────────────────────────────────────────

/// A single entry in a stream's timeline, recording when work was submitted
/// and (optionally) completed.
#[derive(Debug, Clone)]
pub struct TimelineEntry {
    /// Human-readable label.
    pub label: String,
    /// When the operation was enqueued.
    pub enqueued_at: Instant,
    /// When the operation completed (`None` while still in-flight).
    pub completed_at: Option<Instant>,
}

// ── CudaStream ───────────────────────────────────────────────────────

/// Abstraction over a single CUDA stream (or CPU sequential fallback).
pub struct CudaStream {
    /// Globally unique identifier.
    pub id: u64,
    /// Priority level of this stream.
    pub priority: Priority,
    /// Logical device ordinal this stream is associated with.
    pub device_id: usize,
    /// Number of operations dispatched (lifetime counter).
    pub ops_dispatched: u64,
    /// Whether the stream is currently idle (all prior work complete).
    pub idle: bool,
    /// Whether this stream is currently checked out from the pool.
    in_use: bool,
    /// Creation timestamp.
    pub created_at: Instant,
    /// Timeline of operations (bounded by `StreamManager::max_timeline_entries`).
    timeline: VecDeque<TimelineEntry>,
    /// Pending completion callbacks.
    callbacks: Vec<CompletionCallback>,
}

impl std::fmt::Debug for CudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStream")
            .field("id", &self.id)
            .field("priority", &self.priority)
            .field("device_id", &self.device_id)
            .field("ops_dispatched", &self.ops_dispatched)
            .field("idle", &self.idle)
            .field("in_use", &self.in_use)
            .field("callbacks_pending", &self.callbacks.len())
            .finish()
    }
}

impl CudaStream {
    /// Create a new stream with the given priority and device.
    fn new(priority: Priority, device_id: usize) -> Self {
        Self {
            id: next_stream_id(),
            priority,
            device_id,
            ops_dispatched: 0,
            idle: true,
            in_use: false,
            created_at: Instant::now(),
            timeline: VecDeque::new(),
            callbacks: Vec::new(),
        }
    }

    /// Submit a labelled operation to this stream.
    pub fn submit(&mut self, label: impl Into<String>) {
        self.ops_dispatched += 1;
        self.idle = false;
        self.timeline.push_back(TimelineEntry {
            label: label.into(),
            enqueued_at: Instant::now(),
            completed_at: None,
        });
    }

    /// Mark the most recent in-flight operation as complete and fire callbacks.
    pub fn complete_last(&mut self) {
        // Complete the oldest unfinished entry.
        for entry in &mut self.timeline {
            if entry.completed_at.is_none() {
                entry.completed_at = Some(Instant::now());
                break;
            }
        }
        // If nothing is still in-flight, mark idle and fire callbacks.
        let all_done = self.timeline.iter().all(|e| e.completed_at.is_some());
        if all_done {
            self.idle = true;
            let cbs: Vec<_> = self.callbacks.drain(..).collect();
            for cb in cbs {
                cb();
            }
        }
    }

    /// Synchronize (complete all pending work).  CPU fallback: completes everything.
    pub fn synchronize(&mut self) {
        for entry in &mut self.timeline {
            if entry.completed_at.is_none() {
                entry.completed_at = Some(Instant::now());
            }
        }
        self.idle = true;
        let cbs: Vec<_> = self.callbacks.drain(..).collect();
        for cb in cbs {
            cb();
        }
    }

    /// Register a completion callback.
    pub fn on_complete(&mut self, cb: CompletionCallback) {
        if self.idle {
            cb();
        } else {
            self.callbacks.push(cb);
        }
    }

    /// Return the timeline entries.
    pub fn timeline(&self) -> &VecDeque<TimelineEntry> {
        &self.timeline
    }

    /// Number of pending (incomplete) operations.
    pub fn pending_ops(&self) -> usize {
        self.timeline.iter().filter(|e| e.completed_at.is_none()).count()
    }

    /// Trim old timeline entries, keeping at most `max` entries.
    fn trim_timeline(&mut self, max: usize) {
        while self.timeline.len() > max {
            self.timeline.pop_front();
        }
    }
}

// ── StreamEvent ──────────────────────────────────────────────────────

/// An inter-stream synchronization event.
///
/// An event is *recorded* on one stream and *waited* on another, establishing
/// a happens-before relationship between the two streams.
#[derive(Debug, Clone)]
pub struct StreamEvent {
    /// Globally unique identifier.
    pub id: u64,
    /// Stream on which the event was recorded.
    pub recorded_on_stream: Option<u64>,
    /// Whether the event has been signalled (the recorded work completed).
    pub signalled: bool,
    /// Timestamp of recording.
    pub recorded_at: Option<Instant>,
}

impl StreamEvent {
    /// Create a new unsignalled event.
    pub fn new() -> Self {
        Self { id: next_event_id(), recorded_on_stream: None, signalled: false, recorded_at: None }
    }

    /// Record this event on the given stream (CPU fallback: immediately signals).
    pub fn record(&mut self, stream_id: u64) {
        self.recorded_on_stream = Some(stream_id);
        self.recorded_at = Some(Instant::now());
        // CPU fallback — work is sequential so the event fires immediately.
        self.signalled = true;
    }

    /// Check whether the event has been signalled.
    pub fn is_signalled(&self) -> bool {
        self.signalled
    }

    /// Wait for the event (CPU fallback: returns immediately if signalled).
    pub fn wait(&self) -> Result<(), StreamError> {
        if self.signalled {
            Ok(())
        } else {
            Err(StreamError::EventNotSignalled { event_id: self.id })
        }
    }
}

impl Default for StreamEvent {
    fn default() -> Self {
        Self::new()
    }
}

// ── StreamError ──────────────────────────────────────────────────────

/// Errors produced by the stream manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    /// No streams available for the requested priority / workload.
    NoStreamAvailable,
    /// Stream index out of bounds.
    InvalidStreamIndex { index: usize, pool_size: usize },
    /// Event has not been signalled yet.
    EventNotSignalled { event_id: u64 },
    /// The stream pool is empty (misconfiguration).
    EmptyPool,
    /// Configuration validation failure.
    InvalidConfig { reason: String },
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoStreamAvailable => write!(f, "no CUDA stream available"),
            Self::InvalidStreamIndex { index, pool_size } => {
                write!(f, "stream index {index} out of range (pool size {pool_size})")
            }
            Self::EventNotSignalled { event_id } => {
                write!(f, "event {event_id} not signalled")
            }
            Self::EmptyPool => write!(f, "stream pool is empty"),
            Self::InvalidConfig { reason } => write!(f, "invalid config: {reason}"),
        }
    }
}

impl std::error::Error for StreamError {}

// ── StreamManagerConfig ──────────────────────────────────────────────

/// Configuration for [`StreamManager`].
#[derive(Debug, Clone)]
pub struct StreamManagerConfig {
    /// Number of streams per priority level.
    pub streams_per_priority: usize,
    /// Logical CUDA device ordinal.
    pub device_id: usize,
    /// Maximum number of timeline entries kept per stream.
    pub max_timeline_entries: usize,
    /// Whether to enable automatic recycling of idle streams.
    pub enable_recycling: bool,
}

impl Default for StreamManagerConfig {
    fn default() -> Self {
        Self {
            streams_per_priority: 2,
            device_id: 0,
            max_timeline_entries: 256,
            enable_recycling: true,
        }
    }
}

impl StreamManagerConfig {
    pub fn validate(&self) -> Result<(), StreamError> {
        if self.streams_per_priority == 0 {
            return Err(StreamError::InvalidConfig {
                reason: "streams_per_priority must be ≥ 1".into(),
            });
        }
        if self.streams_per_priority > 64 {
            return Err(StreamError::InvalidConfig {
                reason: "streams_per_priority must be ≤ 64".into(),
            });
        }
        if self.max_timeline_entries == 0 {
            return Err(StreamError::InvalidConfig {
                reason: "max_timeline_entries must be ≥ 1".into(),
            });
        }
        Ok(())
    }
}

// ── StreamManager ────────────────────────────────────────────────────

/// High-level CUDA stream manager with priority-tiered pools, automatic
/// workload-based selection, event synchronization, and stream recycling.
pub struct StreamManager {
    config: StreamManagerConfig,
    /// All streams, partitioned by priority.  Index 0 = Low, 1 = Normal, 2 = High.
    pools: [Vec<CudaStream>; 3],
    /// Recycled (idle) stream IDs ready for reuse, per priority tier.
    recycle_bins: [VecDeque<usize>; 3],
    /// Live events.
    events: Vec<StreamEvent>,
    /// Total streams ever created (including recycled).
    total_created: u64,
    /// Total streams recycled.
    total_recycled: u64,
}

impl std::fmt::Debug for StreamManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamManager")
            .field("config", &self.config)
            .field("total_streams", &self.total_streams())
            .field("num_events", &self.events.len())
            .field("total_created", &self.total_created)
            .field("total_recycled", &self.total_recycled)
            .finish()
    }
}

impl StreamManager {
    /// Create a new stream manager from the given configuration.
    pub fn new(config: StreamManagerConfig) -> Result<Self, StreamError> {
        config.validate()?;

        let mut pools: [Vec<CudaStream>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut recycle_bins: [VecDeque<usize>; 3] =
            [VecDeque::new(), VecDeque::new(), VecDeque::new()];

        for (tier, priority) in Priority::all().iter().enumerate() {
            for _ in 0..config.streams_per_priority {
                let stream = CudaStream::new(*priority, config.device_id);
                let idx = pools[tier].len();
                pools[tier].push(stream);
                recycle_bins[tier].push_back(idx);
            }
        }

        let total_created = (config.streams_per_priority * 3) as u64;

        Ok(Self {
            config,
            pools,
            recycle_bins,
            events: Vec::new(),
            total_created,
            total_recycled: 0,
        })
    }

    /// Create a manager with default configuration.
    pub fn with_defaults() -> Result<Self, StreamError> {
        Self::new(StreamManagerConfig::default())
    }

    // ── Pool inspection ──────────────────────────────────────────

    /// Total number of streams across all priority tiers.
    pub fn total_streams(&self) -> usize {
        self.pools.iter().map(|p| p.len()).sum()
    }

    /// Number of streams at a given priority level.
    pub fn streams_at_priority(&self, priority: Priority) -> usize {
        self.pools[Self::tier(priority)].len()
    }

    /// Number of idle (recyclable) streams at a given priority level.
    pub fn idle_streams_at_priority(&self, priority: Priority) -> usize {
        let tier = Self::tier(priority);
        self.pools[tier].iter().filter(|s| s.idle && !s.in_use).count()
    }

    /// Return the configuration.
    pub fn config(&self) -> &StreamManagerConfig {
        &self.config
    }

    /// Total streams ever created.
    pub fn total_created(&self) -> u64 {
        self.total_created
    }

    /// Total streams recycled.
    pub fn total_recycled(&self) -> u64 {
        self.total_recycled
    }

    // ── Stream acquisition ───────────────────────────────────────

    /// Acquire a stream with explicit priority.
    ///
    /// If recycling is enabled and an idle stream is available it will be
    /// reused; otherwise the least-loaded stream in the tier is returned.
    pub fn acquire(&mut self, priority: Priority) -> Result<u64, StreamError> {
        let tier = Self::tier(priority);
        if self.pools[tier].is_empty() {
            return Err(StreamError::EmptyPool);
        }

        // Try recycled stream first.
        if self.config.enable_recycling
            && let Some(idx) = self.recycle_bins[tier].pop_front()
            && let Some(s) = self.pools[tier].get_mut(idx)
            && s.idle
        {
            s.in_use = true;
            self.total_recycled += 1;
            return Ok(s.id);
        }

        // Fallback: pick least-loaded.
        let idx = self.pools[tier]
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.in_use)
            .min_by_key(|(_, s)| s.ops_dispatched)
            .map(|(i, _)| i);

        if let Some(idx) = idx {
            self.pools[tier][idx].in_use = true;
            Ok(self.pools[tier][idx].id)
        } else {
            // All in-use — pick the one with fewest pending ops regardless.
            let idx = self.pools[tier]
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.pending_ops())
                .map(|(i, _)| i)
                .ok_or(StreamError::NoStreamAvailable)?;
            Ok(self.pools[tier][idx].id)
        }
    }

    /// Acquire a stream based on the workload type's default priority.
    pub fn acquire_for_workload(&mut self, workload: WorkloadType) -> Result<u64, StreamError> {
        self.acquire(workload.default_priority())
    }

    /// Release a stream back to the recycling pool.
    pub fn release(&mut self, stream_id: u64) {
        for (tier, pool) in self.pools.iter_mut().enumerate() {
            if let Some((idx, stream)) =
                pool.iter_mut().enumerate().find(|(_, s)| s.id == stream_id)
            {
                stream.in_use = false;
                stream.trim_timeline(self.config.max_timeline_entries);
                if stream.idle {
                    self.recycle_bins[tier].push_back(idx);
                }
                return;
            }
        }
    }

    // ── Stream operations ────────────────────────────────────────

    /// Submit work on a stream identified by its id.
    pub fn submit(&mut self, stream_id: u64, label: impl Into<String>) -> Result<(), StreamError> {
        let stream = self.find_stream_mut(stream_id)?;
        stream.submit(label);
        Ok(())
    }

    /// Mark the last operation on the stream as complete.
    pub fn complete(&mut self, stream_id: u64) -> Result<(), StreamError> {
        let stream = self.find_stream_mut(stream_id)?;
        stream.complete_last();
        Ok(())
    }

    /// Synchronize a single stream.
    pub fn synchronize_stream(&mut self, stream_id: u64) -> Result<(), StreamError> {
        let stream = self.find_stream_mut(stream_id)?;
        stream.synchronize();
        Ok(())
    }

    /// Synchronize all streams across all priority tiers.
    pub fn synchronize_all(&mut self) {
        for pool in &mut self.pools {
            for stream in pool {
                stream.synchronize();
            }
        }
    }

    /// Register a completion callback on a stream.
    pub fn on_stream_complete(
        &mut self,
        stream_id: u64,
        cb: CompletionCallback,
    ) -> Result<(), StreamError> {
        let stream = self.find_stream_mut(stream_id)?;
        stream.on_complete(cb);
        Ok(())
    }

    /// Return a read-only view of a stream's timeline.
    pub fn stream_timeline(&self, stream_id: u64) -> Result<&VecDeque<TimelineEntry>, StreamError> {
        let stream = self.find_stream(stream_id)?;
        Ok(stream.timeline())
    }

    /// Get a stream's priority.
    pub fn stream_priority(&self, stream_id: u64) -> Result<Priority, StreamError> {
        let stream = self.find_stream(stream_id)?;
        Ok(stream.priority)
    }

    /// Get a stream's device id.
    pub fn stream_device(&self, stream_id: u64) -> Result<usize, StreamError> {
        let stream = self.find_stream(stream_id)?;
        Ok(stream.device_id)
    }

    // ── Events ───────────────────────────────────────────────────

    /// Create a new synchronization event.
    pub fn create_event(&mut self) -> StreamEvent {
        let event = StreamEvent::new();
        self.events.push(event.clone());
        event
    }

    /// Record an event on a stream.
    pub fn record_event(&mut self, event_id: u64, stream_id: u64) -> Result<(), StreamError> {
        // Validate stream exists.
        let _ = self.find_stream(stream_id)?;
        let event = self
            .events
            .iter_mut()
            .find(|e| e.id == event_id)
            .ok_or(StreamError::EventNotSignalled { event_id })?;
        event.record(stream_id);
        Ok(())
    }

    /// Wait on an event from a given stream.
    pub fn wait_event(&self, event_id: u64) -> Result<(), StreamError> {
        let event = self
            .events
            .iter()
            .find(|e| e.id == event_id)
            .ok_or(StreamError::EventNotSignalled { event_id })?;
        event.wait()
    }

    /// Return the number of live events.
    pub fn num_events(&self) -> usize {
        self.events.len()
    }

    /// Create a barrier: record an event on `src_stream` and wait for it from
    /// the caller's perspective.  Returns the event for further chaining.
    pub fn barrier(&mut self, src_stream_id: u64) -> Result<StreamEvent, StreamError> {
        let mut event = self.create_event();
        let _ = self.find_stream(src_stream_id)?;
        event.record(src_stream_id);
        // Update our stored copy.
        if let Some(stored) = self.events.iter_mut().find(|e| e.id == event.id) {
            stored.record(src_stream_id);
        }
        Ok(event)
    }

    // ── Internals ────────────────────────────────────────────────

    fn tier(priority: Priority) -> usize {
        match priority {
            Priority::Low => 0,
            Priority::Normal => 1,
            Priority::High => 2,
        }
    }

    fn find_stream(&self, stream_id: u64) -> Result<&CudaStream, StreamError> {
        for pool in &self.pools {
            if let Some(s) = pool.iter().find(|s| s.id == stream_id) {
                return Ok(s);
            }
        }
        Err(StreamError::NoStreamAvailable)
    }

    fn find_stream_mut(&mut self, stream_id: u64) -> Result<&mut CudaStream, StreamError> {
        for pool in &mut self.pools {
            if let Some(s) = pool.iter_mut().find(|s| s.id == stream_id) {
                return Ok(s);
            }
        }
        Err(StreamError::NoStreamAvailable)
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // ── Priority tests ───────────────────────────────────────────

    #[test]
    fn priority_ordering_low_normal_high() {
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
        assert!(Priority::High > Priority::Low);
    }

    #[test]
    fn priority_default_is_normal() {
        assert_eq!(Priority::default(), Priority::Normal);
    }

    #[test]
    fn priority_cuda_values() {
        assert_eq!(Priority::Low.as_cuda_priority(), 0);
        assert_eq!(Priority::Normal.as_cuda_priority(), -1);
        assert_eq!(Priority::High.as_cuda_priority(), -2);
    }

    #[test]
    fn priority_all_returns_three_levels() {
        let all = Priority::all();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], Priority::Low);
        assert_eq!(all[2], Priority::High);
    }

    // ── WorkloadType tests ───────────────────────────────────────

    #[test]
    fn workload_default_priorities() {
        assert_eq!(WorkloadType::Compute.default_priority(), Priority::Normal);
        assert_eq!(WorkloadType::HostToDevice.default_priority(), Priority::High);
        assert_eq!(WorkloadType::DeviceToHost.default_priority(), Priority::Low);
        assert_eq!(WorkloadType::DeviceToDevice.default_priority(), Priority::Normal);
    }

    // ── CudaStream tests ─────────────────────────────────────────

    #[test]
    fn stream_new_is_idle() {
        let s = CudaStream::new(Priority::Normal, 0);
        assert!(s.idle);
        assert_eq!(s.ops_dispatched, 0);
        assert_eq!(s.device_id, 0);
        assert_eq!(s.priority, Priority::Normal);
    }

    #[test]
    fn stream_ids_are_unique() {
        let a = CudaStream::new(Priority::Normal, 0);
        let b = CudaStream::new(Priority::Normal, 0);
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn stream_submit_marks_not_idle() {
        let mut s = CudaStream::new(Priority::Normal, 0);
        s.submit("matmul");
        assert!(!s.idle);
        assert_eq!(s.ops_dispatched, 1);
        assert_eq!(s.timeline().len(), 1);
    }

    #[test]
    fn stream_complete_last_marks_idle() {
        let mut s = CudaStream::new(Priority::Normal, 0);
        s.submit("op1");
        s.complete_last();
        assert!(s.idle);
        assert!(s.timeline()[0].completed_at.is_some());
    }

    #[test]
    fn stream_synchronize_completes_all() {
        let mut s = CudaStream::new(Priority::Normal, 0);
        s.submit("op1");
        s.submit("op2");
        s.submit("op3");
        assert_eq!(s.pending_ops(), 3);
        s.synchronize();
        assert!(s.idle);
        assert_eq!(s.pending_ops(), 0);
    }

    #[test]
    fn stream_callback_fires_on_complete() {
        let fired = Arc::new(Mutex::new(false));
        let fired_clone = Arc::clone(&fired);
        let mut s = CudaStream::new(Priority::Normal, 0);
        s.submit("op");
        s.on_complete(Box::new(move || {
            *fired_clone.lock().unwrap() = true;
        }));
        assert!(!*fired.lock().unwrap());
        s.synchronize();
        assert!(*fired.lock().unwrap());
    }

    #[test]
    fn stream_callback_fires_immediately_when_idle() {
        let fired = Arc::new(Mutex::new(false));
        let fired_clone = Arc::clone(&fired);
        let mut s = CudaStream::new(Priority::Normal, 0);
        // Stream is idle, callback should fire immediately.
        s.on_complete(Box::new(move || {
            *fired_clone.lock().unwrap() = true;
        }));
        assert!(*fired.lock().unwrap());
    }

    #[test]
    fn stream_timeline_trimming() {
        let mut s = CudaStream::new(Priority::Normal, 0);
        for i in 0..20 {
            s.submit(format!("op{i}"));
        }
        s.trim_timeline(5);
        assert_eq!(s.timeline().len(), 5);
    }

    // ── StreamEvent tests ────────────────────────────────────────

    #[test]
    fn event_new_is_unsignalled() {
        let e = StreamEvent::new();
        assert!(!e.is_signalled());
        assert!(e.recorded_on_stream.is_none());
    }

    #[test]
    fn event_default_same_as_new() {
        let e = StreamEvent::default();
        assert!(!e.is_signalled());
    }

    #[test]
    fn event_record_signals_immediately() {
        let mut e = StreamEvent::new();
        e.record(42);
        assert!(e.is_signalled());
        assert_eq!(e.recorded_on_stream, Some(42));
        assert!(e.recorded_at.is_some());
    }

    #[test]
    fn event_wait_signalled_ok() {
        let mut e = StreamEvent::new();
        e.record(1);
        assert!(e.wait().is_ok());
    }

    #[test]
    fn event_wait_unsignalled_err() {
        let e = StreamEvent::new();
        assert!(e.wait().is_err());
    }

    #[test]
    fn event_ids_are_unique() {
        let a = StreamEvent::new();
        let b = StreamEvent::new();
        assert_ne!(a.id, b.id);
    }

    // ── StreamManagerConfig tests ────────────────────────────────

    #[test]
    fn config_default_is_valid() {
        StreamManagerConfig::default().validate().unwrap();
    }

    #[test]
    fn config_zero_streams_rejected() {
        let mut cfg = StreamManagerConfig::default();
        cfg.streams_per_priority = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_too_many_streams_rejected() {
        let mut cfg = StreamManagerConfig::default();
        cfg.streams_per_priority = 100;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_zero_timeline_rejected() {
        let mut cfg = StreamManagerConfig::default();
        cfg.max_timeline_entries = 0;
        assert!(cfg.validate().is_err());
    }

    // ── StreamManager creation ───────────────────────────────────

    #[test]
    fn manager_creation_default() {
        let mgr = StreamManager::with_defaults().unwrap();
        // 2 streams × 3 priorities = 6 total.
        assert_eq!(mgr.total_streams(), 6);
    }

    #[test]
    fn manager_streams_per_priority() {
        let mgr = StreamManager::with_defaults().unwrap();
        assert_eq!(mgr.streams_at_priority(Priority::Low), 2);
        assert_eq!(mgr.streams_at_priority(Priority::Normal), 2);
        assert_eq!(mgr.streams_at_priority(Priority::High), 2);
    }

    // ── Stream acquisition and recycling ─────────────────────────

    #[test]
    fn acquire_returns_valid_stream() {
        let mut mgr = StreamManager::with_defaults().unwrap();
        let id = mgr.acquire(Priority::Normal).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn acquire_for_workload_uses_correct_priority() {
        let mut mgr = StreamManager::with_defaults().unwrap();
        let id = mgr.acquire_for_workload(WorkloadType::HostToDevice).unwrap();
        let prio = mgr.stream_priority(id).unwrap();
        assert_eq!(prio, Priority::High);
    }

    #[test]
    fn release_and_recycle() {
        let mut mgr = StreamManager::with_defaults().unwrap();
        let id = mgr.acquire(Priority::Normal).unwrap();
        mgr.release(id);
        let recycled_before = mgr.total_recycled();
        let id2 = mgr.acquire(Priority::Normal).unwrap();
        assert!(mgr.total_recycled() > recycled_before || id2 == id);
    }

    #[test]
    fn acquire_all_streams_then_still_get_one() {
        let cfg = StreamManagerConfig { streams_per_priority: 1, ..Default::default() };
        let mut mgr = StreamManager::new(cfg).unwrap();
        let _id1 = mgr.acquire(Priority::Normal).unwrap();
        // Even with all in-use, should still return a stream.
        let id2 = mgr.acquire(Priority::Normal).unwrap();
        assert!(id2 > 0);
    }

    // ── Submit / complete / synchronize ──────────────────────────

    #[test]
    fn submit_and_complete_cycle() {
        let mut mgr = StreamManager::with_defaults().unwrap();
        let id = mgr.acquire(Priority::Normal).unwrap();
        mgr.submit(id, "gemm").unwrap();
        mgr.complete(id).unwrap();
        let timeline = mgr.stream_timeline(id).unwrap();
        assert_eq!(timeline.len(), 1);
        assert!(timeline[0].completed_at.is_some());
    }

    #[test]
    fn synchronize_all_completes_everything() {
        let mut mgr = StreamManager::with_defaults().unwrap();
        let id1 = mgr.acquire(Priority::Low).unwrap();
        let id2 = mgr.acquire(Priority::High).unwrap();
        mgr.submit(id1, "a").unwrap();
        mgr.submit(id2, "b").unwrap();
        mgr.synchronize_all();
        let t1 = mgr.stream_timeline(id1).unwrap();
        let t2 = mgr.stream_timeline(id2).unwrap();
        assert!(t1[0].completed_at.is_some());
        assert!(t2[0].completed_at.is_some());
    }

    // ── Event and barrier tests ──────────────────────────────────

    #[test]
    fn create_and_record_event() {
        let mut mgr = StreamManager::with_defaults().unwrap();
        let id = mgr.acquire(Priority::Normal).unwrap();
        let event = mgr.create_event();
        mgr.record_event(event.id, id).unwrap();
        mgr.wait_event(event.id).unwrap();
    }

    #[test]
    fn barrier_creates_signalled_event() {
        let mut mgr = StreamManager::with_defaults().unwrap();
        let id = mgr.acquire(Priority::Normal).unwrap();
        let event = mgr.barrier(id).unwrap();
        assert!(event.is_signalled());
    }

    #[test]
    fn wait_unsignalled_event_errors() {
        let mut mgr = StreamManager::with_defaults().unwrap();
        let event = mgr.create_event();
        assert!(mgr.wait_event(event.id).is_err());
    }

    #[test]
    fn num_events_tracks_creation() {
        let mut mgr = StreamManager::with_defaults().unwrap();
        assert_eq!(mgr.num_events(), 0);
        let _e1 = mgr.create_event();
        let _e2 = mgr.create_event();
        assert_eq!(mgr.num_events(), 2);
    }

    // ── Multi-stream concurrent operation tests ──────────────────

    #[test]
    fn multi_stream_independent_work() {
        let mut mgr = StreamManager::with_defaults().unwrap();
        let s_low = mgr.acquire(Priority::Low).unwrap();
        let s_norm = mgr.acquire(Priority::Normal).unwrap();
        let s_high = mgr.acquire(Priority::High).unwrap();

        mgr.submit(s_low, "bg_copy").unwrap();
        mgr.submit(s_norm, "matmul").unwrap();
        mgr.submit(s_high, "attention").unwrap();

        mgr.synchronize_all();

        // All timelines have 1 completed entry.
        for id in [s_low, s_norm, s_high] {
            let tl = mgr.stream_timeline(id).unwrap();
            assert_eq!(tl.len(), 1);
            assert!(tl[0].completed_at.is_some());
        }
    }

    #[test]
    fn event_synchronizes_two_streams() {
        let mut mgr = StreamManager::with_defaults().unwrap();
        let producer = mgr.acquire(Priority::High).unwrap();
        let consumer = mgr.acquire(Priority::Normal).unwrap();

        mgr.submit(producer, "produce").unwrap();
        let event = mgr.barrier(producer).unwrap();

        // Consumer waits on producer's event before proceeding.
        mgr.wait_event(event.id).unwrap();
        mgr.submit(consumer, "consume").unwrap();
        mgr.synchronize_all();

        let tl = mgr.stream_timeline(consumer).unwrap();
        assert!(tl[0].completed_at.is_some());
    }

    // ── StreamError Display ──────────────────────────────────────

    #[test]
    fn stream_error_display() {
        let err = StreamError::NoStreamAvailable;
        assert_eq!(format!("{err}"), "no CUDA stream available");

        let err = StreamError::InvalidStreamIndex { index: 5, pool_size: 4 };
        assert!(format!("{err}").contains("5"));

        let err = StreamError::EventNotSignalled { event_id: 42 };
        assert!(format!("{err}").contains("42"));

        let err = StreamError::EmptyPool;
        assert!(format!("{err}").contains("empty"));

        let err = StreamError::InvalidConfig { reason: "bad".into() };
        assert!(format!("{err}").contains("bad"));
    }

    // ── Callback through manager ─────────────────────────────────

    #[test]
    fn manager_callback_fires_on_sync() {
        let fired = Arc::new(Mutex::new(false));
        let fired_clone = Arc::clone(&fired);
        let mut mgr = StreamManager::with_defaults().unwrap();
        let id = mgr.acquire(Priority::Normal).unwrap();
        mgr.submit(id, "work").unwrap();
        mgr.on_stream_complete(
            id,
            Box::new(move || {
                *fired_clone.lock().unwrap() = true;
            }),
        )
        .unwrap();
        assert!(!*fired.lock().unwrap());
        mgr.synchronize_stream(id).unwrap();
        assert!(*fired.lock().unwrap());
    }

    // ── Property tests (proptest) ────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_priority() -> impl Strategy<Value = Priority> {
            prop_oneof![Just(Priority::Low), Just(Priority::Normal), Just(Priority::High),]
        }

        fn arb_workload() -> impl Strategy<Value = WorkloadType> {
            prop_oneof![
                Just(WorkloadType::Compute),
                Just(WorkloadType::HostToDevice),
                Just(WorkloadType::DeviceToHost),
                Just(WorkloadType::DeviceToDevice),
            ]
        }

        proptest! {
            #[test]
            fn acquire_never_panics(p in arb_priority()) {
                let mut mgr = StreamManager::with_defaults().unwrap();
                let _ = mgr.acquire(p);
            }

            #[test]
            fn workload_acquire_never_panics(w in arb_workload()) {
                let mut mgr = StreamManager::with_defaults().unwrap();
                let _ = mgr.acquire_for_workload(w);
            }

            #[test]
            fn submit_complete_round_trip(p in arb_priority(), n in 1usize..20) {
                let mut mgr = StreamManager::with_defaults().unwrap();
                let id = mgr.acquire(p).unwrap();
                for i in 0..n {
                    mgr.submit(id, format!("op{i}")).unwrap();
                }
                mgr.synchronize_stream(id).unwrap();
                let tl = mgr.stream_timeline(id).unwrap();
                prop_assert_eq!(tl.len(), n);
                for entry in tl {
                    prop_assert!(entry.completed_at.is_some());
                }
            }

            #[test]
            fn total_streams_invariant(spp in 1usize..8) {
                let cfg = StreamManagerConfig {
                    streams_per_priority: spp,
                    ..Default::default()
                };
                let mgr = StreamManager::new(cfg).unwrap();
                prop_assert_eq!(mgr.total_streams(), spp * 3);
            }

            #[test]
            fn event_record_always_signals(p in arb_priority()) {
                let mut mgr = StreamManager::with_defaults().unwrap();
                let id = mgr.acquire(p).unwrap();
                let event = mgr.create_event();
                mgr.record_event(event.id, id).unwrap();
                prop_assert!(mgr.wait_event(event.id).is_ok());
            }

            #[test]
            fn priority_cuda_roundtrip(p in arb_priority()) {
                let val = p.as_cuda_priority();
                // CUDA priorities are non-positive for our mapping.
                prop_assert!(val <= 0);
            }
        }
    }
}
