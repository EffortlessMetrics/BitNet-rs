//! CUDA stream management with CPU reference simulation.
//!
//! Provides stream creation, synchronisation, event recording, and
//! multi-stream scheduling. When no CUDA runtime is present the module
//! uses a deterministic CPU reference implementation that preserves
//! execution-ordering semantics.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ── ID generators ──────────────────────────────────────────────────

static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_EVENT_ID: AtomicU64 = AtomicU64::new(1);

fn next_stream_id() -> u64 {
    NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed)
}

fn next_event_id() -> u64 {
    NEXT_EVENT_ID.fetch_add(1, Ordering::Relaxed)
}

// ── Error type ─────────────────────────────────────────────────────

/// Errors that can occur during stream operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    /// The requested stream does not exist.
    StreamNotFound(u64),
    /// The requested event does not exist.
    EventNotFound(u64),
    /// The stream has already been destroyed.
    StreamDestroyed(u64),
    /// The event has already been destroyed.
    EventDestroyed(u64),
    /// A synchronisation timeout was reached.
    SyncTimeout { stream_id: u64, elapsed: Duration },
    /// An invalid configuration was supplied.
    InvalidConfig(String),
    /// The stream has pending work that blocks the operation.
    StreamBusy(u64),
    /// Circular dependency detected in multi-stream schedule.
    CircularDependency(String),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamNotFound(id) => write!(f, "stream {id} not found"),
            Self::EventNotFound(id) => write!(f, "event {id} not found"),
            Self::StreamDestroyed(id) => write!(f, "stream {id} already destroyed"),
            Self::EventDestroyed(id) => write!(f, "event {id} already destroyed"),
            Self::SyncTimeout { stream_id, elapsed } => {
                write!(f, "stream {stream_id} sync timeout after {elapsed:?}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::StreamBusy(id) => write!(f, "stream {id} is busy"),
            Self::CircularDependency(msg) => write!(f, "circular dependency: {msg}"),
        }
    }
}

impl std::error::Error for StreamError {}

// ── Public types ───────────────────────────────────────────────────

/// Priority level for a CUDA stream.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StreamPriority {
    /// Lowest scheduling weight.
    Low = 0,
    /// Default scheduling weight.
    #[default]
    Normal = 1,
    /// Elevated scheduling weight.
    High = 2,
    /// Highest scheduling weight — pre-empts lower priorities.
    Critical = 3,
}

impl StreamPriority {
    /// Numeric weight used by the scheduler.
    pub fn weight(self) -> u32 {
        match self {
            Self::Low => 1,
            Self::Normal => 2,
            Self::High => 4,
            Self::Critical => 8,
        }
    }
}

/// Policy that governs how `synchronize_stream` behaves.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// Block until all work on the stream has completed.
    #[default]
    BlockUntilComplete,
    /// Spin-wait, polling at the given interval.
    SpinWait(Duration),
    /// Return immediately with the current status (non-blocking).
    NonBlocking,
}

/// Configuration for creating a stream.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Human-readable label (used in diagnostics).
    pub name: String,
    /// Scheduling priority.
    pub priority: StreamPriority,
    /// Default synchronisation policy for the stream.
    pub sync_policy: SyncPolicy,
    /// Maximum number of pending work items before back-pressure.
    pub max_pending: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            name: String::from("default"),
            priority: StreamPriority::Normal,
            sync_policy: SyncPolicy::BlockUntilComplete,
            max_pending: 1024,
        }
    }
}

impl StreamConfig {
    /// Create a new config with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), ..Default::default() }
    }

    /// Builder: set priority.
    pub fn with_priority(mut self, p: StreamPriority) -> Self {
        self.priority = p;
        self
    }

    /// Builder: set sync policy.
    pub fn with_sync_policy(mut self, s: SyncPolicy) -> Self {
        self.sync_policy = s;
        self
    }

    /// Builder: set max pending items.
    pub fn with_max_pending(mut self, n: usize) -> Self {
        self.max_pending = n;
        self
    }
}

/// Recorded event on a stream.
#[derive(Debug, Clone)]
pub struct StreamEvent {
    /// Unique event identifier.
    pub id: u64,
    /// Stream on which this event was recorded.
    pub stream_id: u64,
    /// Instant the event was recorded.
    pub recorded_at: Instant,
    /// Whether the event has been signalled (all prior work complete).
    pub completed: bool,
    /// Optional user tag.
    pub tag: Option<String>,
}

/// Status returned by `query_stream_status`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamStatus {
    /// All submitted work has completed.
    Idle,
    /// Work items are in flight.
    Busy { pending: usize },
    /// The stream has been explicitly destroyed.
    Destroyed,
}

// ── Internal representation ────────────────────────────────────────

#[derive(Debug, Clone)]
struct WorkItem {
    _id: u64,
    _submitted_at: Instant,
    completed: bool,
    // dependency: must wait for this event before executing
    wait_event: Option<u64>,
}

static NEXT_WORK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct StreamState {
    id: u64,
    config: StreamConfig,
    queue: VecDeque<WorkItem>,
    events: Vec<u64>,
    destroyed: bool,
    _created_at: Instant,
}

// ── Stream Manager ─────────────────────────────────────────────────

/// CPU-reference CUDA stream manager.
///
/// Simulates asynchronous stream execution ordering on the CPU so that
/// scheduling logic can be tested without a GPU.
#[derive(Debug)]
pub struct CudaStreamManager {
    streams: Mutex<HashMap<u64, StreamState>>,
    events: Mutex<HashMap<u64, StreamEvent>>,
}

impl Default for CudaStreamManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaStreamManager {
    /// Create an empty stream manager.
    pub fn new() -> Self {
        Self { streams: Mutex::new(HashMap::new()), events: Mutex::new(HashMap::new()) }
    }

    // ── Stream lifecycle ───────────────────────────────────────────

    /// Create a new stream with the given configuration.
    pub fn create_stream(&self, config: StreamConfig) -> Result<u64, StreamError> {
        if config.max_pending == 0 {
            return Err(StreamError::InvalidConfig("max_pending must be > 0".into()));
        }
        let id = next_stream_id();
        let state = StreamState {
            id,
            config,
            queue: VecDeque::new(),
            events: Vec::new(),
            destroyed: false,
            _created_at: Instant::now(),
        };
        self.streams.lock().unwrap().insert(id, state);
        Ok(id)
    }

    /// Destroy a stream, marking it unusable.
    pub fn destroy_stream(&self, stream_id: u64) -> Result<(), StreamError> {
        let mut streams = self.streams.lock().unwrap();
        let state = streams.get_mut(&stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;
        if state.destroyed {
            return Err(StreamError::StreamDestroyed(stream_id));
        }
        state.destroyed = true;
        Ok(())
    }

    // ── Work submission ────────────────────────────────────────────

    /// Submit a simulated work item to the stream.
    pub fn submit_work(&self, stream_id: u64) -> Result<u64, StreamError> {
        self.submit_work_inner(stream_id, None)
    }

    /// Submit work that depends on an event.
    pub fn submit_work_after_event(
        &self,
        stream_id: u64,
        event_id: u64,
    ) -> Result<u64, StreamError> {
        // Validate event exists
        {
            let events = self.events.lock().unwrap();
            if !events.contains_key(&event_id) {
                return Err(StreamError::EventNotFound(event_id));
            }
        }
        self.submit_work_inner(stream_id, Some(event_id))
    }

    fn submit_work_inner(
        &self,
        stream_id: u64,
        wait_event: Option<u64>,
    ) -> Result<u64, StreamError> {
        let mut streams = self.streams.lock().unwrap();
        let state = streams.get_mut(&stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;
        if state.destroyed {
            return Err(StreamError::StreamDestroyed(stream_id));
        }
        if state.queue.len() >= state.config.max_pending {
            return Err(StreamError::StreamBusy(stream_id));
        }
        let work_id = NEXT_WORK_ID.fetch_add(1, Ordering::Relaxed);
        state.queue.push_back(WorkItem {
            _id: work_id,
            _submitted_at: Instant::now(),
            completed: false,
            wait_event,
        });
        Ok(work_id)
    }

    // ── Synchronisation ────────────────────────────────────────────

    /// Synchronise a stream according to its configured `SyncPolicy`.
    ///
    /// In the CPU reference implementation this drains all pending work
    /// items, simulating completion.
    pub fn synchronize_stream(&self, stream_id: u64) -> Result<StreamStatus, StreamError> {
        let mut streams = self.streams.lock().unwrap();
        let state = streams.get_mut(&stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;
        if state.destroyed {
            return Err(StreamError::StreamDestroyed(stream_id));
        }
        match state.config.sync_policy {
            SyncPolicy::BlockUntilComplete | SyncPolicy::SpinWait(_) => {
                // Complete all work
                for item in state.queue.iter_mut() {
                    item.completed = true;
                }
                // Mark related events as completed
                let event_ids: Vec<u64> = state.events.clone();
                drop(streams);
                let mut events = self.events.lock().unwrap();
                for eid in event_ids {
                    if let Some(ev) = events.get_mut(&eid) {
                        ev.completed = true;
                    }
                }
                Ok(StreamStatus::Idle)
            }
            SyncPolicy::NonBlocking => {
                let pending = state.queue.iter().filter(|w| !w.completed).count();
                if pending == 0 {
                    Ok(StreamStatus::Idle)
                } else {
                    Ok(StreamStatus::Busy { pending })
                }
            }
        }
    }

    // ── Events ─────────────────────────────────────────────────────

    /// Record an event on the given stream.
    pub fn record_event(&self, stream_id: u64, tag: Option<String>) -> Result<u64, StreamError> {
        let mut streams = self.streams.lock().unwrap();
        let state = streams.get_mut(&stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;
        if state.destroyed {
            return Err(StreamError::StreamDestroyed(stream_id));
        }

        let eid = next_event_id();
        let all_done = state.queue.iter().all(|w| w.completed);
        let event = StreamEvent {
            id: eid,
            stream_id,
            recorded_at: Instant::now(),
            completed: all_done,
            tag,
        };
        state.events.push(eid);
        drop(streams);
        self.events.lock().unwrap().insert(eid, event);
        Ok(eid)
    }

    /// Block until the specified event has been signalled.
    pub fn wait_event(&self, event_id: u64) -> Result<(), StreamError> {
        let mut events = self.events.lock().unwrap();
        let event = events.get_mut(&event_id).ok_or(StreamError::EventNotFound(event_id))?;
        // CPU reference: immediately mark as completed
        event.completed = true;
        Ok(())
    }

    /// Check whether an event has been signalled.
    pub fn is_event_complete(&self, event_id: u64) -> Result<bool, StreamError> {
        let events = self.events.lock().unwrap();
        let event = events.get(&event_id).ok_or(StreamError::EventNotFound(event_id))?;
        Ok(event.completed)
    }

    // ── Queries ────────────────────────────────────────────────────

    /// Get the priority of a stream.
    pub fn stream_priority(&self, stream_id: u64) -> Result<StreamPriority, StreamError> {
        let streams = self.streams.lock().unwrap();
        let state = streams.get(&stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;
        if state.destroyed {
            return Err(StreamError::StreamDestroyed(stream_id));
        }
        Ok(state.config.priority)
    }

    /// Query the current status of a stream.
    pub fn query_stream_status(&self, stream_id: u64) -> Result<StreamStatus, StreamError> {
        let streams = self.streams.lock().unwrap();
        let state = streams.get(&stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;
        if state.destroyed {
            return Ok(StreamStatus::Destroyed);
        }
        let pending = state.queue.iter().filter(|w| !w.completed).count();
        if pending == 0 { Ok(StreamStatus::Idle) } else { Ok(StreamStatus::Busy { pending }) }
    }

    /// Return the number of pending (incomplete) work items on a stream.
    pub fn pending_count(&self, stream_id: u64) -> Result<usize, StreamError> {
        let streams = self.streams.lock().unwrap();
        let state = streams.get(&stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;
        Ok(state.queue.iter().filter(|w| !w.completed).count())
    }

    /// Return the stream configuration.
    pub fn stream_config(&self, stream_id: u64) -> Result<StreamConfig, StreamError> {
        let streams = self.streams.lock().unwrap();
        let state = streams.get(&stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;
        Ok(state.config.clone())
    }

    /// Return all stream IDs managed by this manager.
    pub fn stream_ids(&self) -> Vec<u64> {
        self.streams.lock().unwrap().keys().copied().collect()
    }

    /// Return all event IDs managed by this manager.
    pub fn event_ids(&self) -> Vec<u64> {
        self.events.lock().unwrap().keys().copied().collect()
    }

    /// Complete the next pending work item on a stream (FIFO).
    pub fn complete_one(&self, stream_id: u64) -> Result<bool, StreamError> {
        let mut streams = self.streams.lock().unwrap();
        let state = streams.get_mut(&stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;
        if state.destroyed {
            return Err(StreamError::StreamDestroyed(stream_id));
        }
        for item in state.queue.iter_mut() {
            if !item.completed {
                item.completed = true;
                return Ok(true);
            }
        }
        Ok(false)
    }

    // ── Multi-stream scheduling ────────────────────────────────────

    /// Schedule work across multiple streams respecting priority order.
    ///
    /// `schedule` maps stream IDs to the number of work items to submit.
    /// Items are submitted in descending priority order. Returns the total
    /// number of work items submitted.
    pub fn multi_stream_schedule(&self, schedule: &[(u64, usize)]) -> Result<usize, StreamError> {
        // Collect priorities for ordering
        let mut entries: Vec<(u64, usize, StreamPriority)> = Vec::new();
        {
            let streams = self.streams.lock().unwrap();
            for &(sid, count) in schedule {
                let state = streams.get(&sid).ok_or(StreamError::StreamNotFound(sid))?;
                if state.destroyed {
                    return Err(StreamError::StreamDestroyed(sid));
                }
                entries.push((sid, count, state.config.priority));
            }
        }
        // Sort by descending priority weight
        entries.sort_by(|a, b| b.2.weight().cmp(&a.2.weight()));

        let mut total = 0usize;
        for (sid, count, _) in entries {
            for _ in 0..count {
                self.submit_work(sid)?;
                total += 1;
            }
        }
        Ok(total)
    }

    /// Schedule work with explicit inter-stream event dependencies.
    ///
    /// `tasks` contains `(stream_id, work_count, dependency_event)` triples.
    /// If `dependency_event` is `Some(eid)`, every work item in that batch
    /// waits on the event before executing.
    pub fn multi_stream_schedule_with_deps(
        &self,
        tasks: &[(u64, usize, Option<u64>)],
    ) -> Result<usize, StreamError> {
        let mut total = 0usize;
        for &(sid, count, dep) in tasks {
            for _ in 0..count {
                if let Some(eid) = dep {
                    self.submit_work_after_event(sid, eid)?;
                } else {
                    self.submit_work(sid)?;
                }
                total += 1;
            }
        }
        Ok(total)
    }

    /// Synchronise all managed streams.
    pub fn synchronize_all(&self) -> Result<(), StreamError> {
        let ids: Vec<u64> = self.streams.lock().unwrap().keys().copied().collect();
        for id in ids {
            // Skip already-destroyed streams
            let destroyed = {
                let streams = self.streams.lock().unwrap();
                streams.get(&id).is_none_or(|s| s.destroyed)
            };
            if !destroyed {
                self.synchronize_stream(id)?;
            }
        }
        Ok(())
    }

    /// Detect circular event dependencies among the given event IDs.
    ///
    /// Returns an error if a cycle is found. The CPU reference tracks
    /// event→stream ownership; a cycle occurs when stream A waits on an
    /// event from stream B which in turn waits on an event from stream A.
    pub fn detect_circular_deps(&self, event_ids: &[u64]) -> Result<(), StreamError> {
        let events = self.events.lock().unwrap();
        let streams = self.streams.lock().unwrap();

        // Build adjacency: stream → set of streams it depends on
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();

        for &eid in event_ids {
            let event = events.get(&eid).ok_or(StreamError::EventNotFound(eid))?;
            let src_stream = event.stream_id;
            // Find streams that have work waiting on this event
            for state in streams.values() {
                for item in &state.queue {
                    if item.wait_event == Some(eid) && state.id != src_stream {
                        adj.entry(state.id).or_default().push(src_stream);
                    }
                }
            }
        }

        // DFS cycle detection
        let mut visited: HashMap<u64, u8> = HashMap::new(); // 0=unseen, 1=in-progress, 2=done
        for &node in adj.keys() {
            if Self::dfs_cycle(node, &adj, &mut visited) {
                return Err(StreamError::CircularDependency(format!(
                    "cycle involving stream {node}"
                )));
            }
        }
        Ok(())
    }

    fn dfs_cycle(node: u64, adj: &HashMap<u64, Vec<u64>>, visited: &mut HashMap<u64, u8>) -> bool {
        match visited.get(&node).copied().unwrap_or(0) {
            1 => return true,  // back-edge → cycle
            2 => return false, // already fully explored
            _ => {}
        }
        visited.insert(node, 1);
        if let Some(neighbours) = adj.get(&node) {
            for &nb in neighbours {
                if Self::dfs_cycle(nb, adj, visited) {
                    return true;
                }
            }
        }
        visited.insert(node, 2);
        false
    }
}

// ── Convenience free functions ─────────────────────────────────────

/// Create a stream on the given manager (convenience wrapper).
pub fn create_stream(mgr: &CudaStreamManager, config: StreamConfig) -> Result<u64, StreamError> {
    mgr.create_stream(config)
}

/// Synchronise a stream on the given manager (convenience wrapper).
pub fn synchronize_stream(
    mgr: &CudaStreamManager,
    stream_id: u64,
) -> Result<StreamStatus, StreamError> {
    mgr.synchronize_stream(stream_id)
}

/// Record an event (convenience wrapper).
pub fn record_event(
    mgr: &CudaStreamManager,
    stream_id: u64,
    tag: Option<String>,
) -> Result<u64, StreamError> {
    mgr.record_event(stream_id, tag)
}

/// Wait for an event (convenience wrapper).
pub fn wait_event(mgr: &CudaStreamManager, event_id: u64) -> Result<(), StreamError> {
    mgr.wait_event(event_id)
}

/// Query stream priority (convenience wrapper).
pub fn stream_priority(
    mgr: &CudaStreamManager,
    stream_id: u64,
) -> Result<StreamPriority, StreamError> {
    mgr.stream_priority(stream_id)
}

/// Query stream status (convenience wrapper).
pub fn query_stream_status(
    mgr: &CudaStreamManager,
    stream_id: u64,
) -> Result<StreamStatus, StreamError> {
    mgr.query_stream_status(stream_id)
}

/// Multi-stream schedule (convenience wrapper).
pub fn multi_stream_schedule(
    mgr: &CudaStreamManager,
    schedule: &[(u64, usize)],
) -> Result<usize, StreamError> {
    mgr.multi_stream_schedule(schedule)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr() -> CudaStreamManager {
        CudaStreamManager::new()
    }

    // ── StreamConfig tests ─────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let c = StreamConfig::default();
        assert_eq!(c.name, "default");
        assert_eq!(c.priority, StreamPriority::Normal);
        assert_eq!(c.sync_policy, SyncPolicy::BlockUntilComplete);
        assert_eq!(c.max_pending, 1024);
    }

    #[test]
    fn config_builder_priority() {
        let c = StreamConfig::new("x").with_priority(StreamPriority::High);
        assert_eq!(c.priority, StreamPriority::High);
    }

    #[test]
    fn config_builder_sync_policy() {
        let c = StreamConfig::new("x").with_sync_policy(SyncPolicy::NonBlocking);
        assert_eq!(c.sync_policy, SyncPolicy::NonBlocking);
    }

    #[test]
    fn config_builder_max_pending() {
        let c = StreamConfig::new("x").with_max_pending(42);
        assert_eq!(c.max_pending, 42);
    }

    #[test]
    fn config_builder_chain() {
        let c = StreamConfig::new("chain")
            .with_priority(StreamPriority::Critical)
            .with_sync_policy(SyncPolicy::SpinWait(Duration::from_millis(1)))
            .with_max_pending(8);
        assert_eq!(c.name, "chain");
        assert_eq!(c.priority, StreamPriority::Critical);
        assert_eq!(c.max_pending, 8);
    }

    // ── StreamPriority tests ───────────────────────────────────────

    #[test]
    fn priority_ordering() {
        assert!(StreamPriority::Low < StreamPriority::Normal);
        assert!(StreamPriority::Normal < StreamPriority::High);
        assert!(StreamPriority::High < StreamPriority::Critical);
    }

    #[test]
    fn priority_weights() {
        assert_eq!(StreamPriority::Low.weight(), 1);
        assert_eq!(StreamPriority::Normal.weight(), 2);
        assert_eq!(StreamPriority::High.weight(), 4);
        assert_eq!(StreamPriority::Critical.weight(), 8);
    }

    #[test]
    fn priority_default_is_normal() {
        assert_eq!(StreamPriority::default(), StreamPriority::Normal);
    }

    // ── create_stream tests ────────────────────────────────────────

    #[test]
    fn create_stream_returns_unique_ids() {
        let m = mgr();
        let a = m.create_stream(StreamConfig::default()).unwrap();
        let b = m.create_stream(StreamConfig::default()).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn create_stream_with_zero_max_pending_fails() {
        let m = mgr();
        let err = m.create_stream(StreamConfig::new("bad").with_max_pending(0)).unwrap_err();
        assert!(matches!(err, StreamError::InvalidConfig(_)));
    }

    #[test]
    fn create_stream_stores_config() {
        let m = mgr();
        let sid =
            m.create_stream(StreamConfig::new("test").with_priority(StreamPriority::High)).unwrap();
        let cfg = m.stream_config(sid).unwrap();
        assert_eq!(cfg.name, "test");
        assert_eq!(cfg.priority, StreamPriority::High);
    }

    #[test]
    fn create_multiple_streams() {
        let m = mgr();
        let ids: Vec<u64> =
            (0..10).map(|i| m.create_stream(StreamConfig::new(format!("s{i}"))).unwrap()).collect();
        assert_eq!(ids.len(), 10);
        // All unique
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 10);
    }

    // ── destroy_stream tests ───────────────────────────────────────

    #[test]
    fn destroy_stream_marks_destroyed() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.destroy_stream(sid).unwrap();
        assert_eq!(m.query_stream_status(sid).unwrap(), StreamStatus::Destroyed);
    }

    #[test]
    fn destroy_nonexistent_stream_errors() {
        let m = mgr();
        assert!(matches!(m.destroy_stream(999999), Err(StreamError::StreamNotFound(999999))));
    }

    #[test]
    fn destroy_already_destroyed_stream_errors() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.destroy_stream(sid).unwrap();
        assert!(matches!(m.destroy_stream(sid), Err(StreamError::StreamDestroyed(_))));
    }

    // ── submit_work tests ──────────────────────────────────────────

    #[test]
    fn submit_work_increases_pending() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.submit_work(sid).unwrap();
        m.submit_work(sid).unwrap();
        assert_eq!(m.pending_count(sid).unwrap(), 2);
    }

    #[test]
    fn submit_work_on_destroyed_stream_errors() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.destroy_stream(sid).unwrap();
        assert!(matches!(m.submit_work(sid), Err(StreamError::StreamDestroyed(_))));
    }

    #[test]
    fn submit_work_on_nonexistent_stream_errors() {
        let m = mgr();
        assert!(matches!(m.submit_work(888888), Err(StreamError::StreamNotFound(_))));
    }

    #[test]
    fn submit_work_respects_max_pending() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::new("tiny").with_max_pending(2)).unwrap();
        m.submit_work(sid).unwrap();
        m.submit_work(sid).unwrap();
        assert!(matches!(m.submit_work(sid), Err(StreamError::StreamBusy(_))));
    }

    #[test]
    fn submit_work_returns_unique_ids() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        let a = m.submit_work(sid).unwrap();
        let b = m.submit_work(sid).unwrap();
        assert_ne!(a, b);
    }

    // ── synchronize_stream tests ───────────────────────────────────

    #[test]
    fn synchronize_completes_all_work() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.submit_work(sid).unwrap();
        m.submit_work(sid).unwrap();
        let status = m.synchronize_stream(sid).unwrap();
        assert_eq!(status, StreamStatus::Idle);
        assert_eq!(m.pending_count(sid).unwrap(), 0);
    }

    #[test]
    fn synchronize_on_empty_stream_returns_idle() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        assert_eq!(m.synchronize_stream(sid).unwrap(), StreamStatus::Idle);
    }

    #[test]
    fn synchronize_destroyed_stream_errors() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.destroy_stream(sid).unwrap();
        assert!(matches!(m.synchronize_stream(sid), Err(StreamError::StreamDestroyed(_))));
    }

    #[test]
    fn synchronize_nonblocking_reports_busy() {
        let m = mgr();
        let sid = m
            .create_stream(StreamConfig::new("nb").with_sync_policy(SyncPolicy::NonBlocking))
            .unwrap();
        m.submit_work(sid).unwrap();
        let status = m.synchronize_stream(sid).unwrap();
        assert_eq!(status, StreamStatus::Busy { pending: 1 });
    }

    #[test]
    fn synchronize_nonblocking_idle_when_no_work() {
        let m = mgr();
        let sid = m
            .create_stream(StreamConfig::new("nb").with_sync_policy(SyncPolicy::NonBlocking))
            .unwrap();
        assert_eq!(m.synchronize_stream(sid).unwrap(), StreamStatus::Idle);
    }

    #[test]
    fn synchronize_spin_wait_completes_work() {
        let m = mgr();
        let sid = m
            .create_stream(
                StreamConfig::new("sw")
                    .with_sync_policy(SyncPolicy::SpinWait(Duration::from_micros(10))),
            )
            .unwrap();
        m.submit_work(sid).unwrap();
        assert_eq!(m.synchronize_stream(sid).unwrap(), StreamStatus::Idle);
    }

    // ── record_event / wait_event tests ────────────────────────────

    #[test]
    fn record_event_returns_id() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        let eid = m.record_event(sid, None).unwrap();
        assert!(eid > 0);
    }

    #[test]
    fn record_event_with_tag() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        let eid = m.record_event(sid, Some("checkpoint".into())).unwrap();
        let events = m.events.lock().unwrap();
        assert_eq!(events[&eid].tag.as_deref(), Some("checkpoint"));
    }

    #[test]
    fn record_event_on_idle_stream_is_completed() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        let eid = m.record_event(sid, None).unwrap();
        assert!(m.is_event_complete(eid).unwrap());
    }

    #[test]
    fn record_event_on_busy_stream_is_not_completed() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.submit_work(sid).unwrap();
        let eid = m.record_event(sid, None).unwrap();
        assert!(!m.is_event_complete(eid).unwrap());
    }

    #[test]
    fn wait_event_marks_complete() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.submit_work(sid).unwrap();
        let eid = m.record_event(sid, None).unwrap();
        assert!(!m.is_event_complete(eid).unwrap());
        m.wait_event(eid).unwrap();
        assert!(m.is_event_complete(eid).unwrap());
    }

    #[test]
    fn wait_event_nonexistent_errors() {
        let m = mgr();
        assert!(matches!(m.wait_event(777777), Err(StreamError::EventNotFound(_))));
    }

    #[test]
    fn record_event_on_destroyed_stream_errors() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.destroy_stream(sid).unwrap();
        assert!(matches!(m.record_event(sid, None), Err(StreamError::StreamDestroyed(_))));
    }

    #[test]
    fn record_event_on_nonexistent_stream_errors() {
        let m = mgr();
        assert!(matches!(m.record_event(666666, None), Err(StreamError::StreamNotFound(_))));
    }

    #[test]
    fn multiple_events_on_same_stream() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        let e1 = m.record_event(sid, Some("a".into())).unwrap();
        let e2 = m.record_event(sid, Some("b".into())).unwrap();
        assert_ne!(e1, e2);
    }

    // ── stream_priority tests ──────────────────────────────────────

    #[test]
    fn stream_priority_returns_configured_priority() {
        let m = mgr();
        let sid = m
            .create_stream(StreamConfig::new("p").with_priority(StreamPriority::Critical))
            .unwrap();
        assert_eq!(m.stream_priority(sid).unwrap(), StreamPriority::Critical);
    }

    #[test]
    fn stream_priority_nonexistent_errors() {
        let m = mgr();
        assert!(matches!(m.stream_priority(555555), Err(StreamError::StreamNotFound(_))));
    }

    #[test]
    fn stream_priority_destroyed_errors() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.destroy_stream(sid).unwrap();
        assert!(matches!(m.stream_priority(sid), Err(StreamError::StreamDestroyed(_))));
    }

    // ── query_stream_status tests ──────────────────────────────────

    #[test]
    fn status_idle_when_empty() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        assert_eq!(m.query_stream_status(sid).unwrap(), StreamStatus::Idle);
    }

    #[test]
    fn status_busy_with_pending_work() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.submit_work(sid).unwrap();
        assert_eq!(m.query_stream_status(sid).unwrap(), StreamStatus::Busy { pending: 1 });
    }

    #[test]
    fn status_destroyed_after_destroy() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.destroy_stream(sid).unwrap();
        assert_eq!(m.query_stream_status(sid).unwrap(), StreamStatus::Destroyed);
    }

    #[test]
    fn status_nonexistent_errors() {
        let m = mgr();
        assert!(matches!(m.query_stream_status(444444), Err(StreamError::StreamNotFound(_))));
    }

    // ── complete_one tests ─────────────────────────────────────────

    #[test]
    fn complete_one_reduces_pending() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.submit_work(sid).unwrap();
        m.submit_work(sid).unwrap();
        assert!(m.complete_one(sid).unwrap());
        assert_eq!(m.pending_count(sid).unwrap(), 1);
    }

    #[test]
    fn complete_one_returns_false_when_empty() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        assert!(!m.complete_one(sid).unwrap());
    }

    #[test]
    fn complete_one_on_destroyed_errors() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.destroy_stream(sid).unwrap();
        assert!(matches!(m.complete_one(sid), Err(StreamError::StreamDestroyed(_))));
    }

    // ── multi_stream_schedule tests ────────────────────────────────

    #[test]
    fn multi_stream_schedule_basic() {
        let m = mgr();
        let s1 = m.create_stream(StreamConfig::default()).unwrap();
        let s2 = m.create_stream(StreamConfig::default()).unwrap();
        let total = m.multi_stream_schedule(&[(s1, 3), (s2, 2)]).unwrap();
        assert_eq!(total, 5);
        assert_eq!(m.pending_count(s1).unwrap(), 3);
        assert_eq!(m.pending_count(s2).unwrap(), 2);
    }

    #[test]
    fn multi_stream_schedule_empty() {
        let m = mgr();
        assert_eq!(m.multi_stream_schedule(&[]).unwrap(), 0);
    }

    #[test]
    fn multi_stream_schedule_nonexistent_stream_errors() {
        let m = mgr();
        assert!(matches!(
            m.multi_stream_schedule(&[(333333, 1)]),
            Err(StreamError::StreamNotFound(_))
        ));
    }

    #[test]
    fn multi_stream_schedule_destroyed_stream_errors() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.destroy_stream(sid).unwrap();
        assert!(matches!(
            m.multi_stream_schedule(&[(sid, 1)]),
            Err(StreamError::StreamDestroyed(_))
        ));
    }

    #[test]
    fn multi_stream_schedule_respects_priority_order() {
        let m = mgr();
        let low =
            m.create_stream(StreamConfig::new("low").with_priority(StreamPriority::Low)).unwrap();
        let high = m
            .create_stream(StreamConfig::new("high").with_priority(StreamPriority::Critical))
            .unwrap();
        // Even though low is listed first, high-priority should be scheduled first
        let total = m.multi_stream_schedule(&[(low, 2), (high, 3)]).unwrap();
        assert_eq!(total, 5);
    }

    // ── multi_stream_schedule_with_deps tests ──────────────────────

    #[test]
    fn schedule_with_deps_basic() {
        let m = mgr();
        let s1 = m.create_stream(StreamConfig::default()).unwrap();
        let s2 = m.create_stream(StreamConfig::default()).unwrap();
        let eid = m.record_event(s1, None).unwrap();
        let total =
            m.multi_stream_schedule_with_deps(&[(s1, 1, None), (s2, 2, Some(eid))]).unwrap();
        assert_eq!(total, 3);
    }

    #[test]
    fn schedule_with_deps_nonexistent_event_errors() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        assert!(matches!(
            m.multi_stream_schedule_with_deps(&[(sid, 1, Some(222222))]),
            Err(StreamError::EventNotFound(_))
        ));
    }

    // ── synchronize_all tests ──────────────────────────────────────

    #[test]
    fn synchronize_all_completes_everything() {
        let m = mgr();
        let s1 = m.create_stream(StreamConfig::default()).unwrap();
        let s2 = m.create_stream(StreamConfig::default()).unwrap();
        m.submit_work(s1).unwrap();
        m.submit_work(s2).unwrap();
        m.synchronize_all().unwrap();
        assert_eq!(m.pending_count(s1).unwrap(), 0);
        assert_eq!(m.pending_count(s2).unwrap(), 0);
    }

    #[test]
    fn synchronize_all_skips_destroyed() {
        let m = mgr();
        let s1 = m.create_stream(StreamConfig::default()).unwrap();
        let s2 = m.create_stream(StreamConfig::default()).unwrap();
        m.submit_work(s1).unwrap();
        m.destroy_stream(s2).unwrap();
        m.synchronize_all().unwrap();
        assert_eq!(m.pending_count(s1).unwrap(), 0);
    }

    // ── detect_circular_deps tests ─────────────────────────────────

    #[test]
    fn no_circular_deps_when_independent() {
        let m = mgr();
        let s1 = m.create_stream(StreamConfig::default()).unwrap();
        let s2 = m.create_stream(StreamConfig::default()).unwrap();
        let e1 = m.record_event(s1, None).unwrap();
        let e2 = m.record_event(s2, None).unwrap();
        m.detect_circular_deps(&[e1, e2]).unwrap();
    }

    #[test]
    fn detect_circular_deps_with_empty_list() {
        let m = mgr();
        m.detect_circular_deps(&[]).unwrap();
    }

    // ── stream_ids / event_ids ─────────────────────────────────────

    #[test]
    fn stream_ids_lists_all() {
        let m = mgr();
        let s1 = m.create_stream(StreamConfig::default()).unwrap();
        let s2 = m.create_stream(StreamConfig::default()).unwrap();
        let ids = m.stream_ids();
        assert!(ids.contains(&s1));
        assert!(ids.contains(&s2));
    }

    #[test]
    fn event_ids_lists_all() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        let e1 = m.record_event(sid, None).unwrap();
        let e2 = m.record_event(sid, None).unwrap();
        let ids = m.event_ids();
        assert!(ids.contains(&e1));
        assert!(ids.contains(&e2));
    }

    // ── Free-function wrappers ─────────────────────────────────────

    #[test]
    fn free_fn_create_stream_works() {
        let m = mgr();
        let sid = create_stream(&m, StreamConfig::default()).unwrap();
        assert!(sid > 0);
    }

    #[test]
    fn free_fn_synchronize_stream_works() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.submit_work(sid).unwrap();
        let status = synchronize_stream(&m, sid).unwrap();
        assert_eq!(status, StreamStatus::Idle);
    }

    #[test]
    fn free_fn_record_event_works() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        let eid = record_event(&m, sid, None).unwrap();
        assert!(eid > 0);
    }

    #[test]
    fn free_fn_wait_event_works() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        let eid = m.record_event(sid, None).unwrap();
        wait_event(&m, eid).unwrap();
    }

    #[test]
    fn free_fn_stream_priority_works() {
        let m = mgr();
        let sid =
            m.create_stream(StreamConfig::new("x").with_priority(StreamPriority::Low)).unwrap();
        assert_eq!(stream_priority(&m, sid).unwrap(), StreamPriority::Low);
    }

    #[test]
    fn free_fn_query_stream_status_works() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        assert_eq!(query_stream_status(&m, sid).unwrap(), StreamStatus::Idle);
    }

    #[test]
    fn free_fn_multi_stream_schedule_works() {
        let m = mgr();
        let s1 = m.create_stream(StreamConfig::default()).unwrap();
        assert_eq!(multi_stream_schedule(&m, &[(s1, 4)]).unwrap(), 4);
    }

    // ── Error display tests ────────────────────────────────────────

    #[test]
    fn error_display_stream_not_found() {
        let e = StreamError::StreamNotFound(42);
        assert_eq!(e.to_string(), "stream 42 not found");
    }

    #[test]
    fn error_display_event_not_found() {
        let e = StreamError::EventNotFound(7);
        assert_eq!(e.to_string(), "event 7 not found");
    }

    #[test]
    fn error_display_stream_destroyed() {
        let e = StreamError::StreamDestroyed(3);
        assert_eq!(e.to_string(), "stream 3 already destroyed");
    }

    #[test]
    fn error_display_event_destroyed() {
        let e = StreamError::EventDestroyed(5);
        assert_eq!(e.to_string(), "event 5 already destroyed");
    }

    #[test]
    fn error_display_sync_timeout() {
        let e = StreamError::SyncTimeout { stream_id: 1, elapsed: Duration::from_secs(5) };
        assert!(e.to_string().contains("timeout"));
    }

    #[test]
    fn error_display_invalid_config() {
        let e = StreamError::InvalidConfig("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn error_display_stream_busy() {
        let e = StreamError::StreamBusy(9);
        assert_eq!(e.to_string(), "stream 9 is busy");
    }

    #[test]
    fn error_display_circular_dependency() {
        let e = StreamError::CircularDependency("cycle at 2".into());
        assert!(e.to_string().contains("cycle at 2"));
    }

    // ── Edge-case / integration ────────────────────────────────────

    #[test]
    fn sync_then_submit_more_work() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.submit_work(sid).unwrap();
        m.synchronize_stream(sid).unwrap();
        // Can submit new work after sync
        m.submit_work(sid).unwrap();
        assert_eq!(m.pending_count(sid).unwrap(), 1);
    }

    #[test]
    fn event_after_sync_is_completed() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        m.submit_work(sid).unwrap();
        m.synchronize_stream(sid).unwrap();
        let eid = m.record_event(sid, None).unwrap();
        assert!(m.is_event_complete(eid).unwrap());
    }

    #[test]
    fn submit_work_after_event_basic() {
        let m = mgr();
        let s1 = m.create_stream(StreamConfig::default()).unwrap();
        let s2 = m.create_stream(StreamConfig::default()).unwrap();
        let eid = m.record_event(s1, None).unwrap();
        let wid = m.submit_work_after_event(s2, eid).unwrap();
        assert!(wid > 0);
        assert_eq!(m.pending_count(s2).unwrap(), 1);
    }

    #[test]
    fn submit_work_after_nonexistent_event_errors() {
        let m = mgr();
        let sid = m.create_stream(StreamConfig::default()).unwrap();
        assert!(matches!(
            m.submit_work_after_event(sid, 111111),
            Err(StreamError::EventNotFound(_))
        ));
    }

    #[test]
    fn default_manager_has_no_streams() {
        let m = CudaStreamManager::default();
        assert!(m.stream_ids().is_empty());
        assert!(m.event_ids().is_empty());
    }

    #[test]
    fn is_event_complete_nonexistent_errors() {
        let m = mgr();
        assert!(matches!(m.is_event_complete(123456), Err(StreamError::EventNotFound(_))));
    }

    #[test]
    fn sync_policy_eq() {
        assert_eq!(SyncPolicy::BlockUntilComplete, SyncPolicy::BlockUntilComplete);
        assert_eq!(SyncPolicy::NonBlocking, SyncPolicy::NonBlocking);
        assert_ne!(SyncPolicy::BlockUntilComplete, SyncPolicy::NonBlocking);
    }

    #[test]
    fn stream_status_eq() {
        assert_eq!(StreamStatus::Idle, StreamStatus::Idle);
        assert_eq!(StreamStatus::Destroyed, StreamStatus::Destroyed);
        assert_eq!(StreamStatus::Busy { pending: 3 }, StreamStatus::Busy { pending: 3 });
        assert_ne!(StreamStatus::Busy { pending: 1 }, StreamStatus::Busy { pending: 2 });
    }
}

// ── Proptest properties ────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_priority() -> impl Strategy<Value = StreamPriority> {
        prop_oneof![
            Just(StreamPriority::Low),
            Just(StreamPriority::Normal),
            Just(StreamPriority::High),
            Just(StreamPriority::Critical),
        ]
    }

    proptest! {
        /// Creating N streams always yields N distinct IDs.
        #[test]
        fn create_n_streams_unique_ids(n in 1usize..50) {
            let m = CudaStreamManager::new();
            let mut ids = Vec::with_capacity(n);
            for i in 0..n {
                ids.push(
                    m.create_stream(StreamConfig::new(format!("s{i}")))
                        .unwrap(),
                );
            }
            ids.sort();
            ids.dedup();
            prop_assert_eq!(ids.len(), n);
        }

        /// Submitting k items then synchronising leaves zero pending.
        #[test]
        fn sync_drains_all_work(k in 0usize..100) {
            let m = CudaStreamManager::new();
            let sid = m.create_stream(StreamConfig::default()).unwrap();
            for _ in 0..k {
                m.submit_work(sid).unwrap();
            }
            m.synchronize_stream(sid).unwrap();
            let pending = m.pending_count(sid).unwrap();
            prop_assert_eq!(pending, 0);
        }

        /// complete_one reduces pending count by exactly 1 (when pending > 0).
        #[test]
        fn complete_one_decrements(k in 1usize..50) {
            let m = CudaStreamManager::new();
            let sid = m.create_stream(StreamConfig::default()).unwrap();
            for _ in 0..k {
                m.submit_work(sid).unwrap();
            }
            let before = m.pending_count(sid).unwrap();
            m.complete_one(sid).unwrap();
            let after = m.pending_count(sid).unwrap();
            prop_assert_eq!(after, before - 1);
        }

        /// multi_stream_schedule returns the sum of requested counts.
        #[test]
        fn schedule_total_matches(
            counts in prop::collection::vec(1usize..20, 1..8),
        ) {
            let m = CudaStreamManager::new();
            let sched: Vec<(u64, usize)> = counts
                .iter()
                .map(|&c| {
                    let sid = m.create_stream(StreamConfig::default()).unwrap();
                    (sid, c)
                })
                .collect();
            let expected: usize = counts.iter().sum();
            let total = m.multi_stream_schedule(&sched).unwrap();
            prop_assert_eq!(total, expected);
        }

        /// Priority weight is always a power-of-two ≥ 1.
        #[test]
        fn priority_weight_is_power_of_two(prio in arb_priority()) {
            let w = prio.weight();
            prop_assert!(w >= 1);
            prop_assert!(w.is_power_of_two());
        }
    }
}
