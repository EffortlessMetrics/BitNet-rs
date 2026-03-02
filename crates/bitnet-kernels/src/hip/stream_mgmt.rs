//! HIP stream and event management for concurrent kernel execution.
//!
//! Provides stream pooling, event synchronization, and round-robin
//! dispatch modelling the HIP runtime. Mirrors the CUDA stream management
//! in [`crate::cuda::stream_mgmt`] but uses HIP terminology
//! (work-groups, wavefronts, hipStream_t).
//!
//! # CPU fallback
//!
//! All operations execute sequentially in pure Rust.
//! GPU-dependent dispatch will use `hipStreamCreate`/`hipStreamSynchronize`
//! once the HIP FFI bindings are wired in.

use bitnet_common::{KernelError, Result};
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

/// Priority level for a HIP stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HipStreamPriority {
    /// Low priority — background work.
    Low,
    /// Normal (default) priority.
    #[default]
    Normal,
    /// High priority — latency-sensitive work.
    High,
}

impl HipStreamPriority {
    /// Map to a HIP-compatible numeric priority (lower = higher priority).
    pub fn as_hip_priority(self) -> i32 {
        match self {
            Self::Low => 0,
            Self::Normal => -1,
            Self::High => -2,
        }
    }
}

// ── StreamConfig ─────────────────────────────────────────────────────

/// Configuration for the HIP stream pool.
#[derive(Debug, Clone)]
pub struct HipStreamConfig {
    /// Number of streams in the pool.
    pub num_streams: usize,
    /// Default priority for new streams.
    pub priority: HipStreamPriority,
    /// Enable per-stream profiling.
    pub enable_profiling: bool,
}

impl Default for HipStreamConfig {
    fn default() -> Self {
        Self { num_streams: 4, priority: HipStreamPriority::Normal, enable_profiling: false }
    }
}

impl HipStreamConfig {
    /// Validate configuration.
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

/// Handle to a single HIP stream.
#[derive(Debug, Clone)]
pub struct HipStreamHandle {
    /// Unique identifier.
    pub id: u64,
    /// Priority of this stream.
    pub priority: HipStreamPriority,
    /// Whether this stream has been synchronized.
    pub synchronized: bool,
    /// Number of operations dispatched on this stream.
    pub ops_dispatched: u64,
    /// Creation timestamp.
    pub created_at: Instant,
}

impl HipStreamHandle {
    /// Create a new stream handle.
    pub fn new(priority: HipStreamPriority) -> Self {
        Self {
            id: next_stream_id(),
            priority,
            synchronized: true,
            ops_dispatched: 0,
            created_at: Instant::now(),
        }
    }

    /// Record that an operation was dispatched on this stream.
    pub fn dispatch_op(&mut self) {
        self.ops_dispatched += 1;
        self.synchronized = false;
    }

    /// Mark the stream as synchronized.
    pub fn synchronize(&mut self) {
        self.synchronized = true;
    }
}

// ── StreamEvent ──────────────────────────────────────────────────────

/// Synchronization event between HIP streams.
#[derive(Debug, Clone)]
pub struct HipStreamEvent {
    /// Unique event identifier.
    pub id: u64,
    /// Stream that recorded this event.
    pub stream_id: u64,
    /// Whether the event has been recorded.
    pub recorded: bool,
    /// Whether the event has been completed.
    pub completed: bool,
    /// Timestamp when the event was recorded.
    pub recorded_at: Option<Instant>,
}

impl HipStreamEvent {
    /// Create a new unrecorded event.
    pub fn new() -> Self {
        Self {
            id: next_event_id(),
            stream_id: 0,
            recorded: false,
            completed: false,
            recorded_at: None,
        }
    }

    /// Record the event on a given stream.
    pub fn record(&mut self, stream_id: u64) {
        self.stream_id = stream_id;
        self.recorded = true;
        self.recorded_at = Some(Instant::now());
    }

    /// Mark the event as completed (simulates device completion).
    pub fn complete(&mut self) {
        self.completed = true;
    }

    /// Elapsed time since recording (CPU-side simulation).
    pub fn elapsed(&self) -> Option<Duration> {
        self.recorded_at.map(|t| t.elapsed())
    }
}

impl Default for HipStreamEvent {
    fn default() -> Self {
        Self::new()
    }
}

// ── StreamPool ───────────────────────────────────────────────────────

/// Pool of HIP streams for round-robin dispatch.
#[derive(Debug)]
pub struct HipStreamPool {
    streams: Vec<HipStreamHandle>,
    next_index: usize,
    config: HipStreamConfig,
}

impl HipStreamPool {
    /// Create a new stream pool.
    pub fn new(config: HipStreamConfig) -> Result<Self> {
        config.validate()?;
        let streams =
            (0..config.num_streams).map(|_| HipStreamHandle::new(config.priority)).collect();
        Ok(Self { streams, next_index: 0, config })
    }

    /// Get the next stream in round-robin order.
    pub fn next_stream(&mut self) -> &mut HipStreamHandle {
        let idx = self.next_index % self.streams.len();
        self.next_index += 1;
        &mut self.streams[idx]
    }

    /// Get a stream by index.
    pub fn get_stream(&self, index: usize) -> Option<&HipStreamHandle> {
        self.streams.get(index)
    }

    /// Number of streams in the pool.
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Synchronize all streams.
    pub fn synchronize_all(&mut self) {
        for stream in &mut self.streams {
            stream.synchronize();
        }
    }

    /// Total operations dispatched across all streams.
    pub fn total_ops(&self) -> u64 {
        self.streams.iter().map(|s| s.ops_dispatched).sum()
    }

    /// Configuration reference.
    pub fn config(&self) -> &HipStreamConfig {
        &self.config
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_validates() {
        assert!(HipStreamConfig::default().validate().is_ok());
    }

    #[test]
    fn config_zero_streams_fails() {
        let cfg = HipStreamConfig { num_streams: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_too_many_streams_fails() {
        let cfg = HipStreamConfig { num_streams: 200, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn pool_creation() {
        let pool = HipStreamPool::new(HipStreamConfig::default()).unwrap();
        assert_eq!(pool.len(), 4);
        assert!(!pool.is_empty());
    }

    #[test]
    fn round_robin_dispatch() {
        let mut pool =
            HipStreamPool::new(HipStreamConfig { num_streams: 2, ..Default::default() }).unwrap();
        let id0 = pool.next_stream().id;
        let id1 = pool.next_stream().id;
        let id2 = pool.next_stream().id;
        assert_ne!(id0, id1);
        assert_eq!(id0, id2); // wraps around
    }

    #[test]
    fn stream_dispatch_marks_unsynchronized() {
        let mut stream = HipStreamHandle::new(HipStreamPriority::Normal);
        assert!(stream.synchronized);
        stream.dispatch_op();
        assert!(!stream.synchronized);
        assert_eq!(stream.ops_dispatched, 1);
    }

    #[test]
    fn stream_synchronize() {
        let mut stream = HipStreamHandle::new(HipStreamPriority::Normal);
        stream.dispatch_op();
        stream.synchronize();
        assert!(stream.synchronized);
    }

    #[test]
    fn event_lifecycle() {
        let mut event = HipStreamEvent::new();
        assert!(!event.recorded);
        assert!(!event.completed);
        event.record(42);
        assert!(event.recorded);
        assert_eq!(event.stream_id, 42);
        event.complete();
        assert!(event.completed);
    }

    #[test]
    fn event_elapsed_none_before_record() {
        let event = HipStreamEvent::new();
        assert!(event.elapsed().is_none());
    }

    #[test]
    fn event_elapsed_some_after_record() {
        let mut event = HipStreamEvent::new();
        event.record(1);
        assert!(event.elapsed().is_some());
    }

    #[test]
    fn synchronize_all_streams() {
        let mut pool =
            HipStreamPool::new(HipStreamConfig { num_streams: 3, ..Default::default() }).unwrap();
        pool.next_stream().dispatch_op();
        pool.next_stream().dispatch_op();
        pool.next_stream().dispatch_op();
        pool.synchronize_all();
        for i in 0..3 {
            assert!(pool.get_stream(i).unwrap().synchronized);
        }
    }

    #[test]
    fn total_ops_aggregation() {
        let mut pool =
            HipStreamPool::new(HipStreamConfig { num_streams: 2, ..Default::default() }).unwrap();
        pool.next_stream().dispatch_op();
        pool.next_stream().dispatch_op();
        pool.next_stream().dispatch_op();
        assert_eq!(pool.total_ops(), 3);
    }

    #[test]
    fn priority_as_hip_value() {
        assert_eq!(HipStreamPriority::Low.as_hip_priority(), 0);
        assert_eq!(HipStreamPriority::Normal.as_hip_priority(), -1);
        assert_eq!(HipStreamPriority::High.as_hip_priority(), -2);
    }

    #[test]
    fn get_stream_out_of_bounds() {
        let pool =
            HipStreamPool::new(HipStreamConfig { num_streams: 1, ..Default::default() }).unwrap();
        assert!(pool.get_stream(0).is_some());
        assert!(pool.get_stream(1).is_none());
    }

    #[test]
    fn event_default_trait() {
        let event = HipStreamEvent::default();
        assert!(!event.recorded);
    }
}
