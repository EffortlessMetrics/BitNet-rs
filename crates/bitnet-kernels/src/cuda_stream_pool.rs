//! CUDA stream **pool** manager with RAII leases and event synchronization.
//!
//! This module is complementary to [`crate::cuda::stream_mgmt`] which handles
//! individual stream operations, scheduling, and profiling.  This module adds
//! **pool-level** lifecycle management:
//!
//! - [`StreamPool`] — bounded pool of virtual CUDA streams with automatic recycling.
//! - [`StreamLease`] — RAII guard that returns a stream to the pool on drop.
//! - [`EventPool`] — companion pool for CUDA events used for inter-stream sync.
//! - [`EventLease`] — RAII guard for borrowed events.
//! - [`StreamGraph`] — lightweight dependency graph (stream A waits on event from B).
//! - [`PoolStats`] — runtime statistics (active/idle/total, peak, wait histogram).
//!
//! All types have CPU reference implementations so the pool logic can be tested
//! without a GPU.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bitnet_common::Result;

// ── Identifiers ──────────────────────────────────────────────────────

static NEXT_POOL_STREAM_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_POOL_EVENT_ID: AtomicU64 = AtomicU64::new(1);

fn next_stream_id() -> u64 {
    NEXT_POOL_STREAM_ID.fetch_add(1, Ordering::Relaxed)
}

fn next_event_id() -> u64 {
    NEXT_POOL_EVENT_ID.fetch_add(1, Ordering::Relaxed)
}

// ── StreamPriority ───────────────────────────────────────────────────

/// Priority level for a pooled CUDA stream.
///
/// Maps to CUDA numeric priorities (lower numeric = higher priority).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StreamPriority {
    /// Background / best-effort work.
    Low,
    /// Default priority.
    #[default]
    Normal,
    /// Latency-sensitive work.
    High,
    /// Real-time / preemptive — highest priority.
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

// ── StreamPoolConfig ─────────────────────────────────────────────────

/// Configuration for [`StreamPool`].
#[derive(Debug, Clone)]
pub struct StreamPoolConfig {
    /// Maximum number of streams the pool may hold.
    pub max_streams: usize,
    /// Default priority assigned to newly created streams.
    pub default_priority: StreamPriority,
    /// Whether to attach CUDA timing events for profiling.
    pub enable_profiling: bool,
}

impl Default for StreamPoolConfig {
    fn default() -> Self {
        Self { max_streams: 8, default_priority: StreamPriority::Normal, enable_profiling: false }
    }
}

impl StreamPoolConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.max_streams == 0 {
            return Err(bitnet_common::KernelError::InvalidArguments {
                reason: "max_streams must be > 0".into(),
            }
            .into());
        }
        Ok(())
    }
}

// ── PooledStream (internal) ──────────────────────────────────────────

/// Internal representation of a single pooled stream.
#[derive(Debug)]
struct PooledStream {
    id: u64,
    priority: StreamPriority,
    /// Monotonically-increasing use counter.
    use_count: u64,
    /// Whether profiling events are attached.
    profiling: bool,
    created_at: Instant,
}

impl PooledStream {
    fn new(priority: StreamPriority, profiling: bool) -> Self {
        Self { id: next_stream_id(), priority, use_count: 0, profiling, created_at: Instant::now() }
    }
}

// ── WaitHistogram ────────────────────────────────────────────────────

/// Simple histogram of wait times bucketed by duration.
#[derive(Debug, Clone, Default)]
pub struct WaitHistogram {
    /// Bucket boundaries (exclusive upper bound) and counts.
    buckets: Vec<(Duration, u64)>,
    /// Count of waits that exceeded the largest bucket.
    overflow: u64,
}

impl WaitHistogram {
    fn new() -> Self {
        Self {
            buckets: vec![
                (Duration::from_micros(10), 0),
                (Duration::from_micros(100), 0),
                (Duration::from_millis(1), 0),
                (Duration::from_millis(10), 0),
                (Duration::from_millis(100), 0),
                (Duration::from_secs(1), 0),
            ],
            overflow: 0,
        }
    }

    fn record(&mut self, dur: Duration) {
        for bucket in &mut self.buckets {
            if dur < bucket.0 {
                bucket.1 += 1;
                return;
            }
        }
        self.overflow += 1;
    }

    /// Total number of recorded waits.
    pub fn total(&self) -> u64 {
        self.buckets.iter().map(|b| b.1).sum::<u64>() + self.overflow
    }

    /// Bucket boundaries and their counts.
    pub fn buckets(&self) -> &[(Duration, u64)] {
        &self.buckets
    }

    /// Count of waits exceeding the largest bucket.
    pub fn overflow(&self) -> u64 {
        self.overflow
    }
}

// ── PoolStats ────────────────────────────────────────────────────────

/// Runtime statistics for [`StreamPool`].
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Number of streams currently leased.
    pub active: usize,
    /// Number of streams idle in the pool.
    pub idle: usize,
    /// Total streams ever created by this pool.
    pub total_created: u64,
    /// Peak number of simultaneously active streams.
    pub peak_active: usize,
    /// Total acquire calls.
    pub acquire_count: u64,
    /// Total release (drop) calls.
    pub release_count: u64,
    /// Wait-time histogram for acquire calls that blocked.
    pub wait_histogram: WaitHistogram,
}

// ── StreamPool ───────────────────────────────────────────────────────

/// Inner mutable state of the pool, protected by a [`Mutex`].
#[derive(Debug)]
struct PoolInner {
    config: StreamPoolConfig,
    idle: VecDeque<PooledStream>,
    active_count: usize,
    total_created: u64,
    peak_active: usize,
    acquire_count: u64,
    release_count: u64,
    wait_histogram: WaitHistogram,
    drained: bool,
}

/// A bounded pool of virtual CUDA streams.
///
/// Streams are created lazily up to [`StreamPoolConfig::max_streams`].
/// Once the cap is reached, [`acquire`](StreamPool::acquire) blocks (in a real
/// CUDA implementation) or returns an error, while
/// [`try_acquire`](StreamPool::try_acquire) returns `None`.
///
/// Returned [`StreamLease`] guards automatically recycle the stream on drop.
#[derive(Debug, Clone)]
pub struct StreamPool {
    inner: Arc<Mutex<PoolInner>>,
}

impl StreamPool {
    /// Create a new stream pool with the given config.
    pub fn new(config: StreamPoolConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(PoolInner {
                config,
                idle: VecDeque::new(),
                active_count: 0,
                total_created: 0,
                peak_active: 0,
                acquire_count: 0,
                release_count: 0,
                wait_histogram: WaitHistogram::new(),
                drained: false,
            })),
        })
    }

    /// Create a pool with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(StreamPoolConfig::default())
    }

    /// Acquire a stream with the pool's default priority.
    ///
    /// Returns `Err` if the pool has been drained or the pool is exhausted.
    pub fn acquire(&self) -> Result<StreamLease> {
        let inner = &self.inner;
        let mut guard = inner.lock().unwrap();
        guard.acquire_count += 1;

        if guard.drained {
            return Err(bitnet_common::KernelError::InvalidArguments {
                reason: "stream pool has been drained".into(),
            }
            .into());
        }

        let priority = guard.config.default_priority;
        Self::acquire_inner(&mut guard, priority, Arc::clone(inner))
    }

    /// Acquire a stream with an explicit priority.
    pub fn acquire_with_priority(&self, priority: StreamPriority) -> Result<StreamLease> {
        let inner = &self.inner;
        let mut guard = inner.lock().unwrap();
        guard.acquire_count += 1;

        if guard.drained {
            return Err(bitnet_common::KernelError::InvalidArguments {
                reason: "stream pool has been drained".into(),
            }
            .into());
        }

        Self::acquire_inner(&mut guard, priority, Arc::clone(inner))
    }

    /// Non-blocking acquire — returns `None` when no stream is available.
    pub fn try_acquire(&self) -> Option<StreamLease> {
        let inner = &self.inner;
        let mut guard = inner.lock().unwrap();

        if guard.drained {
            return None;
        }

        guard.acquire_count += 1;

        // Try to reuse an idle stream.
        if let Some(mut stream) = guard.idle.pop_front() {
            stream.use_count += 1;
            guard.active_count += 1;
            if guard.active_count > guard.peak_active {
                guard.peak_active = guard.active_count;
            }
            return Some(StreamLease {
                stream_id: stream.id,
                priority: stream.priority,
                profiling: stream.profiling,
                pool: Arc::clone(inner),
                returned: false,
                _stream: stream,
            });
        }

        // Try to create a new stream.
        if guard.total_created < guard.config.max_streams as u64 {
            let profiling = guard.config.enable_profiling;
            let mut stream = PooledStream::new(guard.config.default_priority, profiling);
            stream.use_count = 1;
            guard.total_created += 1;
            guard.active_count += 1;
            if guard.active_count > guard.peak_active {
                guard.peak_active = guard.active_count;
            }
            return Some(StreamLease {
                stream_id: stream.id,
                priority: stream.priority,
                profiling: stream.profiling,
                pool: Arc::clone(inner),
                returned: false,
                _stream: stream,
            });
        }

        None
    }

    /// Synchronize all idle streams (CPU ref: no-op).
    pub fn sync_all(&self) -> Result<()> {
        // CPU reference: nothing to synchronize.
        Ok(())
    }

    /// Drain the pool — mark it closed, clear idle streams, prevent new acquires.
    ///
    /// Outstanding leases will still return their streams, but they will be
    /// discarded rather than recycled.
    pub fn drain(&self) {
        let mut guard = self.inner.lock().unwrap();
        guard.drained = true;
        guard.idle.clear();
    }

    /// Return current pool statistics.
    pub fn stats(&self) -> PoolStats {
        let guard = self.inner.lock().unwrap();
        PoolStats {
            active: guard.active_count,
            idle: guard.idle.len(),
            total_created: guard.total_created,
            peak_active: guard.peak_active,
            acquire_count: guard.acquire_count,
            release_count: guard.release_count,
            wait_histogram: guard.wait_histogram.clone(),
        }
    }

    /// Number of streams currently idle.
    pub fn idle_count(&self) -> usize {
        self.inner.lock().unwrap().idle.len()
    }

    /// Number of streams currently active (leased).
    pub fn active_count(&self) -> usize {
        self.inner.lock().unwrap().active_count
    }

    // ── private ──

    fn acquire_inner(
        guard: &mut PoolInner,
        priority: StreamPriority,
        pool: Arc<Mutex<PoolInner>>,
    ) -> Result<StreamLease> {
        let start = Instant::now();

        // Prefer an idle stream with matching priority.
        let idx = guard
            .idle
            .iter()
            .position(|s| s.priority == priority)
            .or_else(|| if !guard.idle.is_empty() { Some(0) } else { None });

        if let Some(i) = idx {
            let mut stream = guard.idle.remove(i).unwrap();
            stream.use_count += 1;
            guard.active_count += 1;
            if guard.active_count > guard.peak_active {
                guard.peak_active = guard.active_count;
            }
            guard.wait_histogram.record(start.elapsed());
            return Ok(StreamLease {
                stream_id: stream.id,
                priority: stream.priority,
                profiling: stream.profiling,
                pool,
                returned: false,
                _stream: stream,
            });
        }

        // Create a new stream if under the cap.
        if guard.total_created < guard.config.max_streams as u64 {
            let profiling = guard.config.enable_profiling;
            let mut stream = PooledStream::new(priority, profiling);
            stream.use_count = 1;
            guard.total_created += 1;
            guard.active_count += 1;
            if guard.active_count > guard.peak_active {
                guard.peak_active = guard.active_count;
            }
            guard.wait_histogram.record(start.elapsed());
            return Ok(StreamLease {
                stream_id: stream.id,
                priority: stream.priority,
                profiling: stream.profiling,
                pool,
                returned: false,
                _stream: stream,
            });
        }

        guard.wait_histogram.record(start.elapsed());
        Err(bitnet_common::KernelError::InvalidArguments { reason: "stream pool exhausted".into() }
            .into())
    }
}

// ── StreamLease ──────────────────────────────────────────────────────

/// RAII guard for a borrowed stream.  Automatically returns the stream to the
/// pool on drop.
pub struct StreamLease {
    stream_id: u64,
    priority: StreamPriority,
    profiling: bool,
    pool: Arc<Mutex<PoolInner>>,
    returned: bool,
    _stream: PooledStream,
}

impl StreamLease {
    /// Unique identifier for this stream.
    pub fn id(&self) -> u64 {
        self.stream_id
    }

    /// Priority of this stream.
    pub fn priority(&self) -> StreamPriority {
        self.priority
    }

    /// Whether profiling is enabled on this stream.
    pub fn profiling(&self) -> bool {
        self.profiling
    }

    /// Synchronize this stream (CPU ref: no-op).
    pub fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    /// Record an event on this stream for dependency tracking.
    pub fn record_event(&self, event: &EventLease) -> Result<()> {
        let mut guard = event.inner.lock().unwrap();
        guard.recorded_on_stream = Some(self.stream_id);
        guard.signalled = true;
        Ok(())
    }

    /// Wait for an event (recorded on another stream) before proceeding.
    pub fn wait_event(&self, event: &EventLease) -> Result<()> {
        let guard = event.inner.lock().unwrap();
        if !guard.signalled {
            return Err(bitnet_common::KernelError::InvalidArguments {
                reason: "event has not been recorded yet".into(),
            }
            .into());
        }
        // CPU ref: already "done".
        Ok(())
    }
}

impl fmt::Debug for StreamLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamLease")
            .field("stream_id", &self.stream_id)
            .field("priority", &self.priority)
            .field("profiling", &self.profiling)
            .finish()
    }
}

impl Drop for StreamLease {
    fn drop(&mut self) {
        if self.returned {
            return;
        }
        self.returned = true;

        let mut guard = self.pool.lock().unwrap();
        guard.active_count = guard.active_count.saturating_sub(1);
        guard.release_count += 1;

        if guard.drained {
            // Pool is shutting down — discard.
            return;
        }

        // Recycle: create a fresh PooledStream carrying forward the identity.
        let recycled = PooledStream {
            id: self.stream_id,
            priority: self.priority,
            use_count: self._stream.use_count,
            profiling: self.profiling,
            created_at: self._stream.created_at,
        };
        guard.idle.push_back(recycled);
    }
}

// ── PooledEvent (internal) ───────────────────────────────────────────

/// Mutable state of a pooled event.
#[derive(Debug)]
struct PooledEventInner {
    #[allow(dead_code)]
    id: u64,
    recorded_on_stream: Option<u64>,
    signalled: bool,
}

// ── EventPool ────────────────────────────────────────────────────────

/// Inner mutable state of the event pool.
#[derive(Debug)]
struct EventPoolInner {
    max_events: usize,
    idle: VecDeque<u64>,
    active_count: usize,
    total_created: u64,
}

/// A companion pool for CUDA events used for inter-stream synchronization.
#[derive(Debug, Clone)]
pub struct EventPool {
    inner: Arc<Mutex<EventPoolInner>>,
}

impl EventPool {
    /// Create an event pool with the given capacity.
    pub fn new(max_events: usize) -> Result<Self> {
        if max_events == 0 {
            return Err(bitnet_common::KernelError::InvalidArguments {
                reason: "max_events must be > 0".into(),
            }
            .into());
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(EventPoolInner {
                max_events,
                idle: VecDeque::new(),
                active_count: 0,
                total_created: 0,
            })),
        })
    }

    /// Acquire an event from the pool.
    pub fn acquire(&self) -> Result<EventLease> {
        let mut guard = self.inner.lock().unwrap();

        let id = if let Some(recycled_id) = guard.idle.pop_front() {
            recycled_id
        } else if guard.total_created < guard.max_events as u64 {
            let id = next_event_id();
            guard.total_created += 1;
            id
        } else {
            return Err(bitnet_common::KernelError::InvalidArguments {
                reason: "event pool exhausted".into(),
            }
            .into());
        };

        guard.active_count += 1;

        Ok(EventLease {
            event_id: id,
            inner: Arc::new(Mutex::new(PooledEventInner {
                id,
                recorded_on_stream: None,
                signalled: false,
            })),
            pool: Arc::clone(&self.inner),
            returned: false,
        })
    }

    /// Non-blocking acquire.
    pub fn try_acquire(&self) -> Option<EventLease> {
        self.acquire().ok()
    }

    /// Number of events currently idle.
    pub fn idle_count(&self) -> usize {
        self.inner.lock().unwrap().idle.len()
    }

    /// Number of events currently active.
    pub fn active_count(&self) -> usize {
        self.inner.lock().unwrap().active_count
    }
}

// ── EventLease ───────────────────────────────────────────────────────

/// RAII guard for a borrowed CUDA event.
pub struct EventLease {
    event_id: u64,
    inner: Arc<Mutex<PooledEventInner>>,
    pool: Arc<Mutex<EventPoolInner>>,
    returned: bool,
}

impl EventLease {
    /// Unique identifier for this event.
    pub fn id(&self) -> u64 {
        self.event_id
    }

    /// Whether this event has been signalled (recorded on a stream).
    pub fn is_signalled(&self) -> bool {
        self.inner.lock().unwrap().signalled
    }

    /// The stream this event was recorded on, if any.
    pub fn recorded_on(&self) -> Option<u64> {
        self.inner.lock().unwrap().recorded_on_stream
    }

    /// Reset the event (clear recorded/signalled state).
    pub fn reset(&self) {
        let mut guard = self.inner.lock().unwrap();
        guard.signalled = false;
        guard.recorded_on_stream = None;
    }
}

impl fmt::Debug for EventLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventLease").field("event_id", &self.event_id).finish()
    }
}

impl Drop for EventLease {
    fn drop(&mut self) {
        if self.returned {
            return;
        }
        self.returned = true;

        let mut guard = self.pool.lock().unwrap();
        guard.active_count = guard.active_count.saturating_sub(1);
        guard.idle.push_back(self.event_id);
    }
}

// ── StreamGraph ──────────────────────────────────────────────────────

/// An edge in the dependency graph: stream `from` signals event, stream `to` waits.
#[derive(Debug, Clone)]
pub struct Dependency {
    /// Stream that produces the event.
    pub from_stream: u64,
    /// Stream that waits on the event.
    pub to_stream: u64,
    /// The event connecting them.
    pub event_id: u64,
}

/// Lightweight dependency graph between pooled streams.
///
/// Records which stream must wait on which event produced by another stream.
#[derive(Debug, Clone, Default)]
pub struct StreamGraph {
    edges: Vec<Dependency>,
}

impl StreamGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a dependency: `to_stream` must wait for `event` produced by `from_stream`.
    pub fn add_dependency(&mut self, from_stream: u64, to_stream: u64, event_id: u64) {
        self.edges.push(Dependency { from_stream, to_stream, event_id });
    }

    /// Return all dependencies.
    pub fn dependencies(&self) -> &[Dependency] {
        &self.edges
    }

    /// Return dependencies where `stream_id` is the consumer (waiter).
    pub fn dependencies_of(&self, stream_id: u64) -> Vec<&Dependency> {
        self.edges.iter().filter(|d| d.to_stream == stream_id).collect()
    }

    /// Return dependencies where `stream_id` is the producer.
    pub fn produced_by(&self, stream_id: u64) -> Vec<&Dependency> {
        self.edges.iter().filter(|d| d.from_stream == stream_id).collect()
    }

    /// Whether the graph contains a cycle (using simple DFS).
    pub fn has_cycle(&self) -> bool {
        use std::collections::{HashMap, HashSet};

        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut nodes = HashSet::new();
        for dep in &self.edges {
            adj.entry(dep.from_stream).or_default().push(dep.to_stream);
            nodes.insert(dep.from_stream);
            nodes.insert(dep.to_stream);
        }

        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();

        fn dfs(
            node: u64,
            adj: &HashMap<u64, Vec<u64>>,
            visited: &mut HashSet<u64>,
            in_stack: &mut HashSet<u64>,
        ) -> bool {
            visited.insert(node);
            in_stack.insert(node);
            if let Some(neighbors) = adj.get(&node) {
                for &next in neighbors {
                    if !visited.contains(&next) {
                        if dfs(next, adj, visited, in_stack) {
                            return true;
                        }
                    } else if in_stack.contains(&next) {
                        return true;
                    }
                }
            }
            in_stack.remove(&node);
            false
        }

        for &node in &nodes {
            if !visited.contains(&node) && dfs(node, &adj, &mut visited, &mut in_stack) {
                return true;
            }
        }
        false
    }

    /// Number of dependency edges.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether the graph has no edges.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Clear all edges.
    pub fn clear(&mut self) {
        self.edges.clear();
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── StreamPoolConfig ─────────────────────────────────────────

    #[test]
    fn config_default_is_valid() {
        let cfg = StreamPoolConfig::default();
        cfg.validate().unwrap();
        assert_eq!(cfg.max_streams, 8);
        assert_eq!(cfg.default_priority, StreamPriority::Normal);
        assert!(!cfg.enable_profiling);
    }

    #[test]
    fn config_zero_streams_is_invalid() {
        let cfg = StreamPoolConfig { max_streams: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_custom_values() {
        let cfg = StreamPoolConfig {
            max_streams: 16,
            default_priority: StreamPriority::High,
            enable_profiling: true,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.max_streams, 16);
        assert_eq!(cfg.default_priority, StreamPriority::High);
        assert!(cfg.enable_profiling);
    }

    // ── StreamPriority ───────────────────────────────────────────

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
    fn priority_display() {
        assert_eq!(format!("{}", StreamPriority::Low), "low");
        assert_eq!(format!("{}", StreamPriority::Normal), "normal");
        assert_eq!(format!("{}", StreamPriority::High), "high");
        assert_eq!(format!("{}", StreamPriority::Critical), "critical");
    }

    #[test]
    fn priority_default_is_normal() {
        assert_eq!(StreamPriority::default(), StreamPriority::Normal);
    }

    // ── StreamPool creation ──────────────────────────────────────

    #[test]
    fn pool_with_defaults() {
        let pool = StreamPool::with_defaults().unwrap();
        let stats = pool.stats();
        assert_eq!(stats.active, 0);
        assert_eq!(stats.idle, 0);
        assert_eq!(stats.total_created, 0);
    }

    #[test]
    fn pool_with_custom_config() {
        let cfg = StreamPoolConfig {
            max_streams: 4,
            default_priority: StreamPriority::High,
            enable_profiling: true,
        };
        let pool = StreamPool::new(cfg).unwrap();
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn pool_zero_streams_rejected() {
        let cfg = StreamPoolConfig { max_streams: 0, ..Default::default() };
        assert!(StreamPool::new(cfg).is_err());
    }

    // ── Acquire / release cycle ──────────────────────────────────

    #[test]
    fn acquire_creates_stream_lazily() {
        let pool = StreamPool::with_defaults().unwrap();
        assert_eq!(pool.stats().total_created, 0);
        let lease = pool.acquire().unwrap();
        assert_eq!(pool.stats().total_created, 1);
        assert_eq!(pool.active_count(), 1);
        drop(lease);
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.idle_count(), 1);
    }

    #[test]
    fn release_recycles_stream() {
        let pool = StreamPool::with_defaults().unwrap();
        let lease = pool.acquire().unwrap();
        let id = lease.id();
        drop(lease);

        // Re-acquire should reuse the same stream.
        let lease2 = pool.acquire().unwrap();
        assert_eq!(lease2.id(), id);
        assert_eq!(pool.stats().total_created, 1);
    }

    #[test]
    fn multiple_acquire_release_cycles() {
        let pool = StreamPool::with_defaults().unwrap();
        for _ in 0..20 {
            let lease = pool.acquire().unwrap();
            assert_eq!(pool.active_count(), 1);
            drop(lease);
            assert_eq!(pool.active_count(), 0);
        }
        // Only one stream ever created because we release before re-acquiring.
        assert_eq!(pool.stats().total_created, 1);
    }

    // ── Priority-based acquisition ───────────────────────────────

    #[test]
    fn acquire_with_priority_creates_matching_stream() {
        let pool = StreamPool::with_defaults().unwrap();
        let lease = pool.acquire_with_priority(StreamPriority::Critical).unwrap();
        assert_eq!(lease.priority(), StreamPriority::Critical);
    }

    #[test]
    fn acquire_prefers_matching_priority_idle_stream() {
        let cfg = StreamPoolConfig { max_streams: 4, ..Default::default() };
        let pool = StreamPool::new(cfg).unwrap();

        // Create a Low and a High stream, then release both.
        let low = pool.acquire_with_priority(StreamPriority::Low).unwrap();
        let high = pool.acquire_with_priority(StreamPriority::High).unwrap();
        let low_id = low.id();
        let high_id = high.id();
        drop(low);
        drop(high);

        // Acquire with High — should get the High stream back.
        let lease = pool.acquire_with_priority(StreamPriority::High).unwrap();
        assert_eq!(lease.id(), high_id);
        drop(lease);

        // Acquire with Low — should get the Low stream back.
        let lease = pool.acquire_with_priority(StreamPriority::Low).unwrap();
        assert_eq!(lease.id(), low_id);
    }

    #[test]
    fn acquire_falls_back_to_any_idle_stream() {
        let cfg = StreamPoolConfig { max_streams: 1, ..Default::default() };
        let pool = StreamPool::new(cfg).unwrap();
        let lease = pool.acquire_with_priority(StreamPriority::Low).unwrap();
        drop(lease);

        // Request Critical, but only a Low stream is idle — should still get it.
        let lease = pool.acquire_with_priority(StreamPriority::Critical).unwrap();
        assert_eq!(lease.priority(), StreamPriority::Low); // inherited from original
    }

    // ── Pool exhaustion ──────────────────────────────────────────

    #[test]
    fn pool_exhaustion_acquire_returns_err() {
        let cfg = StreamPoolConfig { max_streams: 2, ..Default::default() };
        let pool = StreamPool::new(cfg).unwrap();
        let _l1 = pool.acquire().unwrap();
        let _l2 = pool.acquire().unwrap();
        assert!(pool.acquire().is_err());
    }

    #[test]
    fn try_acquire_returns_none_when_exhausted() {
        let cfg = StreamPoolConfig { max_streams: 1, ..Default::default() };
        let pool = StreamPool::new(cfg).unwrap();
        let _l1 = pool.try_acquire().unwrap();
        assert!(pool.try_acquire().is_none());
    }

    #[test]
    fn try_acquire_succeeds_after_release() {
        let cfg = StreamPoolConfig { max_streams: 1, ..Default::default() };
        let pool = StreamPool::new(cfg).unwrap();
        let l = pool.try_acquire().unwrap();
        drop(l);
        assert!(pool.try_acquire().is_some());
    }

    // ── Pool stats ───────────────────────────────────────────────

    #[test]
    fn stats_track_acquire_release() {
        let pool = StreamPool::with_defaults().unwrap();
        let l1 = pool.acquire().unwrap();
        let l2 = pool.acquire().unwrap();
        assert_eq!(pool.stats().acquire_count, 2);
        assert_eq!(pool.stats().active, 2);
        drop(l1);
        assert_eq!(pool.stats().release_count, 1);
        assert_eq!(pool.stats().active, 1);
        drop(l2);
        assert_eq!(pool.stats().release_count, 2);
        assert_eq!(pool.stats().active, 0);
    }

    #[test]
    fn stats_peak_active() {
        let cfg = StreamPoolConfig { max_streams: 4, ..Default::default() };
        let pool = StreamPool::new(cfg).unwrap();
        let l1 = pool.acquire().unwrap();
        let l2 = pool.acquire().unwrap();
        let l3 = pool.acquire().unwrap();
        assert_eq!(pool.stats().peak_active, 3);
        drop(l2);
        assert_eq!(pool.stats().peak_active, 3); // peak doesn't decrease
        drop(l1);
        drop(l3);
        assert_eq!(pool.stats().peak_active, 3);
    }

    #[test]
    fn stats_total_created() {
        let cfg = StreamPoolConfig { max_streams: 4, ..Default::default() };
        let pool = StreamPool::new(cfg).unwrap();
        let l1 = pool.acquire().unwrap();
        let l2 = pool.acquire().unwrap();
        assert_eq!(pool.stats().total_created, 2);
        drop(l1);
        drop(l2);
        // Reuse — no new creation.
        let _l3 = pool.acquire().unwrap();
        assert_eq!(pool.stats().total_created, 2);
    }

    // ── RAII drop behavior ───────────────────────────────────────

    #[test]
    fn lease_drop_returns_stream() {
        let pool = StreamPool::with_defaults().unwrap();
        {
            let _l = pool.acquire().unwrap();
            assert_eq!(pool.active_count(), 1);
        }
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.idle_count(), 1);
    }

    #[test]
    fn lease_debug_format() {
        let pool = StreamPool::with_defaults().unwrap();
        let lease = pool.acquire().unwrap();
        let dbg = format!("{lease:?}");
        assert!(dbg.contains("StreamLease"));
        assert!(dbg.contains("stream_id"));
    }

    #[test]
    fn lease_synchronize_cpu_noop() {
        let pool = StreamPool::with_defaults().unwrap();
        let lease = pool.acquire().unwrap();
        lease.synchronize().unwrap();
    }

    #[test]
    fn lease_profiling_reflects_config() {
        let cfg = StreamPoolConfig { enable_profiling: true, ..Default::default() };
        let pool = StreamPool::new(cfg).unwrap();
        let lease = pool.acquire().unwrap();
        assert!(lease.profiling());
    }

    // ── Drain and sync_all ───────────────────────────────────────

    #[test]
    fn sync_all_succeeds() {
        let pool = StreamPool::with_defaults().unwrap();
        let _l = pool.acquire().unwrap();
        pool.sync_all().unwrap();
    }

    #[test]
    fn drain_prevents_new_acquires() {
        let pool = StreamPool::with_defaults().unwrap();
        pool.drain();
        assert!(pool.acquire().is_err());
        assert!(pool.try_acquire().is_none());
    }

    #[test]
    fn drain_clears_idle_streams() {
        let pool = StreamPool::with_defaults().unwrap();
        let l = pool.acquire().unwrap();
        drop(l);
        assert_eq!(pool.idle_count(), 1);
        pool.drain();
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn drain_outstanding_leases_do_not_recycle() {
        let cfg = StreamPoolConfig { max_streams: 2, ..Default::default() };
        let pool = StreamPool::new(cfg).unwrap();
        let l = pool.acquire().unwrap();
        pool.drain();
        drop(l); // should discard, not recycle
        assert_eq!(pool.idle_count(), 0);
    }

    // ── Concurrent simulation ────────────────────────────────────

    #[test]
    fn multiple_leases_active_simultaneously() {
        let cfg = StreamPoolConfig { max_streams: 4, ..Default::default() };
        let pool = StreamPool::new(cfg).unwrap();
        let l1 = pool.acquire().unwrap();
        let l2 = pool.acquire().unwrap();
        let l2_id = l2.id();
        let l3 = pool.acquire().unwrap();
        let l4 = pool.acquire().unwrap();
        assert_eq!(pool.active_count(), 4);
        assert!(pool.try_acquire().is_none());
        drop(l2);
        assert_eq!(pool.active_count(), 3);
        let l5 = pool.acquire().unwrap();
        assert_eq!(l5.id(), l2_id); // recycled
        drop(l1);
        drop(l3);
        drop(l4);
        drop(l5);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn unique_stream_ids() {
        let cfg = StreamPoolConfig { max_streams: 8, ..Default::default() };
        let pool = StreamPool::new(cfg).unwrap();
        let leases: Vec<_> = (0..8).map(|_| pool.acquire().unwrap()).collect();
        let mut ids: Vec<_> = leases.iter().map(|l| l.id()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 8);
    }

    // ── EventPool ────────────────────────────────────────────────

    #[test]
    fn event_pool_creation() {
        let pool = EventPool::new(4).unwrap();
        assert_eq!(pool.idle_count(), 0);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn event_pool_zero_rejected() {
        assert!(EventPool::new(0).is_err());
    }

    #[test]
    fn event_acquire_and_release() {
        let pool = EventPool::new(2).unwrap();
        let e1 = pool.acquire().unwrap();
        assert_eq!(pool.active_count(), 1);
        drop(e1);
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.idle_count(), 1);
    }

    #[test]
    fn event_pool_exhaustion() {
        let pool = EventPool::new(1).unwrap();
        let _e1 = pool.acquire().unwrap();
        assert!(pool.try_acquire().is_none());
    }

    #[test]
    fn event_recycle() {
        let pool = EventPool::new(1).unwrap();
        let e = pool.acquire().unwrap();
        let id = e.id();
        drop(e);
        let e2 = pool.acquire().unwrap();
        assert_eq!(e2.id(), id);
    }

    #[test]
    fn event_reset() {
        let epool = EventPool::new(2).unwrap();
        let spool = StreamPool::with_defaults().unwrap();
        let stream = spool.acquire().unwrap();
        let event = epool.acquire().unwrap();
        stream.record_event(&event).unwrap();
        assert!(event.is_signalled());
        event.reset();
        assert!(!event.is_signalled());
        assert!(event.recorded_on().is_none());
    }

    // ── Event-based stream dependencies ──────────────────────────

    #[test]
    fn record_and_wait_event() {
        let spool = StreamPool::with_defaults().unwrap();
        let epool = EventPool::new(4).unwrap();

        let producer = spool.acquire().unwrap();
        let consumer = spool.acquire().unwrap();
        let event = epool.acquire().unwrap();

        producer.record_event(&event).unwrap();
        assert!(event.is_signalled());
        assert_eq!(event.recorded_on(), Some(producer.id()));

        consumer.wait_event(&event).unwrap();
    }

    #[test]
    fn wait_unsignalled_event_fails() {
        let spool = StreamPool::with_defaults().unwrap();
        let epool = EventPool::new(4).unwrap();

        let stream = spool.acquire().unwrap();
        let event = epool.acquire().unwrap();
        assert!(stream.wait_event(&event).is_err());
    }

    #[test]
    fn event_debug_format() {
        let epool = EventPool::new(1).unwrap();
        let event = epool.acquire().unwrap();
        let dbg = format!("{event:?}");
        assert!(dbg.contains("EventLease"));
    }

    // ── StreamGraph ──────────────────────────────────────────────

    #[test]
    fn graph_empty() {
        let g = StreamGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn graph_add_dependency() {
        let mut g = StreamGraph::new();
        g.add_dependency(1, 2, 100);
        assert_eq!(g.len(), 1);
        assert!(!g.is_empty());
    }

    #[test]
    fn graph_dependencies_of() {
        let mut g = StreamGraph::new();
        g.add_dependency(1, 2, 100);
        g.add_dependency(3, 2, 101);
        g.add_dependency(1, 4, 102);

        let deps = g.dependencies_of(2);
        assert_eq!(deps.len(), 2);

        let deps = g.dependencies_of(4);
        assert_eq!(deps.len(), 1);
    }

    #[test]
    fn graph_produced_by() {
        let mut g = StreamGraph::new();
        g.add_dependency(1, 2, 100);
        g.add_dependency(1, 3, 101);
        assert_eq!(g.produced_by(1).len(), 2);
        assert_eq!(g.produced_by(2).len(), 0);
    }

    #[test]
    fn graph_no_cycle() {
        let mut g = StreamGraph::new();
        g.add_dependency(1, 2, 100);
        g.add_dependency(2, 3, 101);
        g.add_dependency(1, 3, 102);
        assert!(!g.has_cycle());
    }

    #[test]
    fn graph_with_cycle() {
        let mut g = StreamGraph::new();
        g.add_dependency(1, 2, 100);
        g.add_dependency(2, 3, 101);
        g.add_dependency(3, 1, 102); // cycle!
        assert!(g.has_cycle());
    }

    #[test]
    fn graph_clear() {
        let mut g = StreamGraph::new();
        g.add_dependency(1, 2, 100);
        g.clear();
        assert!(g.is_empty());
    }

    #[test]
    fn graph_self_loop_is_cycle() {
        let mut g = StreamGraph::new();
        g.add_dependency(1, 1, 100);
        assert!(g.has_cycle());
    }

    // ── WaitHistogram ────────────────────────────────────────────

    #[test]
    fn histogram_empty() {
        let h = WaitHistogram::new();
        assert_eq!(h.total(), 0);
        assert_eq!(h.overflow(), 0);
    }

    #[test]
    fn histogram_records_in_buckets() {
        let mut h = WaitHistogram::new();
        h.record(Duration::from_nanos(500)); // < 10µs
        h.record(Duration::from_micros(50)); // < 100µs
        h.record(Duration::from_secs(5)); // overflow
        assert_eq!(h.total(), 3);
        assert_eq!(h.overflow(), 1);
    }

    // ── Integration: pool + events + graph ───────────────────────

    #[test]
    fn integration_producer_consumer_graph() {
        let spool =
            StreamPool::new(StreamPoolConfig { max_streams: 4, ..Default::default() }).unwrap();
        let epool = EventPool::new(4).unwrap();
        let mut graph = StreamGraph::new();

        let producer = spool.acquire().unwrap();
        let consumer = spool.acquire().unwrap();
        let event = epool.acquire().unwrap();

        producer.record_event(&event).unwrap();
        graph.add_dependency(producer.id(), consumer.id(), event.id());
        consumer.wait_event(&event).unwrap();

        assert!(!graph.has_cycle());
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn integration_diamond_dependency() {
        let spool =
            StreamPool::new(StreamPoolConfig { max_streams: 4, ..Default::default() }).unwrap();
        let epool = EventPool::new(4).unwrap();
        let mut graph = StreamGraph::new();

        let s0 = spool.acquire().unwrap();
        let s1 = spool.acquire().unwrap();
        let s2 = spool.acquire().unwrap();
        let s3 = spool.acquire().unwrap();

        let e01 = epool.acquire().unwrap();
        let e02 = epool.acquire().unwrap();
        let e13 = epool.acquire().unwrap();
        let e23 = epool.acquire().unwrap();

        // s0 -> s1, s0 -> s2, s1 -> s3, s2 -> s3  (diamond)
        s0.record_event(&e01).unwrap();
        graph.add_dependency(s0.id(), s1.id(), e01.id());
        s1.wait_event(&e01).unwrap();

        s0.record_event(&e02).unwrap();
        graph.add_dependency(s0.id(), s2.id(), e02.id());
        s2.wait_event(&e02).unwrap();

        s1.record_event(&e13).unwrap();
        graph.add_dependency(s1.id(), s3.id(), e13.id());

        s2.record_event(&e23).unwrap();
        graph.add_dependency(s2.id(), s3.id(), e23.id());
        s3.wait_event(&e13).unwrap();
        s3.wait_event(&e23).unwrap();

        assert!(!graph.has_cycle());
        assert_eq!(graph.dependencies_of(s3.id()).len(), 2);
    }

    // ── Property-style tests ─────────────────────────────────────

    #[test]
    fn property_acquire_release_preserves_count() {
        for max in 1..=8 {
            let pool = StreamPool::new(StreamPoolConfig { max_streams: max, ..Default::default() })
                .unwrap();
            let mut leases = Vec::new();
            for _ in 0..max {
                leases.push(pool.acquire().unwrap());
            }
            assert_eq!(pool.active_count(), max);
            assert!(pool.try_acquire().is_none());
            leases.clear();
            assert_eq!(pool.active_count(), 0);
            assert_eq!(pool.idle_count(), max);
        }
    }

    #[test]
    fn property_peak_never_exceeds_max() {
        for max in 1..=6 {
            let pool = StreamPool::new(StreamPoolConfig { max_streams: max, ..Default::default() })
                .unwrap();
            let leases: Vec<_> = (0..max).map(|_| pool.acquire().unwrap()).collect();
            assert!(pool.stats().peak_active <= max);
            drop(leases);
        }
    }

    #[test]
    fn property_total_created_monotonic() {
        let pool =
            StreamPool::new(StreamPoolConfig { max_streams: 4, ..Default::default() }).unwrap();
        let mut prev = 0u64;
        for _ in 0..10 {
            let l = pool.acquire().unwrap();
            let created = pool.stats().total_created;
            assert!(created >= prev);
            prev = created;
            drop(l);
        }
    }

    #[test]
    fn property_idle_plus_active_le_total_created() {
        let pool =
            StreamPool::new(StreamPoolConfig { max_streams: 4, ..Default::default() }).unwrap();
        let l1 = pool.acquire().unwrap();
        let l2 = pool.acquire().unwrap();
        drop(l1);
        let stats = pool.stats();
        assert!(stats.active + stats.idle <= stats.total_created as usize);
        drop(l2);
    }

    #[test]
    fn property_release_count_matches_drops() {
        let pool = StreamPool::with_defaults().unwrap();
        let n = 15;
        for _ in 0..n {
            let l = pool.acquire().unwrap();
            drop(l);
        }
        assert_eq!(pool.stats().release_count, n);
    }

    #[test]
    fn property_acquire_count_matches_calls() {
        let pool = StreamPool::with_defaults().unwrap();
        for _ in 0..12 {
            let _ = pool.acquire();
        }
        assert_eq!(pool.stats().acquire_count, 12);
    }
}
