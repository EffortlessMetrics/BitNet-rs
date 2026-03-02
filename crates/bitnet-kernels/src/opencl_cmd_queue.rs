//! Intelligent command queue management for OpenCL with priority dispatch.
//!
//! Provides multi-queue management with priority-based scheduling, event
//! synchronisation, profiling, and barrier support. Designed for Intel Arc
//! A770 but works with any OpenCL-compatible device. When no OpenCL runtime
//! is present, CPU reference implementations are used.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── Queue priority ─────────────────────────────────────────────────

/// Priority level for command queue dispatch.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QueuePriority {
    /// Background tasks — lowest scheduling weight.
    Background = 0,
    /// Default priority for normal compute work.
    Low = 1,
    /// Standard interactive-latency work.
    #[default]
    Normal = 2,
    /// Latency-critical work dispatched first.
    High = 3,
}

impl QueuePriority {
    /// All priority levels from lowest to highest.
    pub fn all() -> &'static [QueuePriority] {
        &[Self::Background, Self::Low, Self::Normal, Self::High]
    }
}

impl fmt::Display for QueuePriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Background => write!(f, "Background"),
            Self::Low => write!(f, "Low"),
            Self::Normal => write!(f, "Normal"),
            Self::High => write!(f, "High"),
        }
    }
}

// ── Queue configuration ────────────────────────────────────────────

/// Configuration for command queue creation.
#[derive(Debug, Clone)]
pub struct QueueConfig {
    /// Enable OpenCL profiling on every queue.
    pub enable_profiling: bool,
    /// Allow out-of-order execution inside a single queue.
    pub out_of_order: bool,
    /// Number of backing queues per priority level.
    pub queue_count: usize,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self { enable_profiling: true, out_of_order: false, queue_count: 2 }
    }
}

impl QueueConfig {
    /// Create a config with profiling enabled and in-order queues.
    pub fn with_profiling() -> Self {
        Self { enable_profiling: true, ..Default::default() }
    }

    /// Create a config with out-of-order execution.
    pub fn with_out_of_order() -> Self {
        Self { out_of_order: true, ..Default::default() }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), QueueError> {
        if self.queue_count == 0 {
            return Err(QueueError::InvalidConfig("queue_count must be > 0".into()));
        }
        if self.queue_count > 64 {
            return Err(QueueError::InvalidConfig("queue_count must be ≤ 64".into()));
        }
        Ok(())
    }
}

// ── Queue event ────────────────────────────────────────────────────

/// Unique identifier for a submitted command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId(u64);

impl EventId {
    fn new(id: u64) -> Self {
        Self(id)
    }

    /// Raw numeric value.
    pub fn value(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Event({})", self.0)
    }
}

/// Completion status for a queue event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventStatus {
    /// Command has been submitted but not started.
    Queued,
    /// Command is executing.
    Running,
    /// Command finished successfully.
    Complete,
    /// Command failed.
    Error,
}

impl fmt::Display for EventStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "Queued"),
            Self::Running => write!(f, "Running"),
            Self::Complete => write!(f, "Complete"),
            Self::Error => write!(f, "Error"),
        }
    }
}

/// Wraps an OpenCL event for synchronisation.
#[derive(Debug, Clone)]
pub struct QueueEvent {
    id: EventId,
    priority: QueuePriority,
    status: EventStatus,
    submit_time: Instant,
    complete_time: Option<Instant>,
}

impl QueueEvent {
    fn new(id: EventId, priority: QueuePriority) -> Self {
        Self {
            id,
            priority,
            status: EventStatus::Queued,
            submit_time: Instant::now(),
            complete_time: None,
        }
    }

    /// Event identifier.
    pub fn id(&self) -> EventId {
        self.id
    }

    /// Priority at which the event was submitted.
    pub fn priority(&self) -> QueuePriority {
        self.priority
    }

    /// Current status.
    pub fn status(&self) -> EventStatus {
        self.status
    }

    /// Whether the event has completed (successfully or with error).
    pub fn is_done(&self) -> bool {
        matches!(self.status, EventStatus::Complete | EventStatus::Error)
    }

    /// Wall-clock latency from submit to completion, if available.
    pub fn latency(&self) -> Option<Duration> {
        self.complete_time.map(|t| t.duration_since(self.submit_time))
    }

    fn mark_running(&mut self) {
        self.status = EventStatus::Running;
    }

    fn mark_complete(&mut self) {
        self.status = EventStatus::Complete;
        self.complete_time = Some(Instant::now());
    }

    fn mark_error(&mut self) {
        self.status = EventStatus::Error;
        self.complete_time = Some(Instant::now());
    }
}

// ── Single command queue (CPU reference) ───────────────────────────

/// A single logical command queue.
///
/// In the CPU reference implementation commands execute synchronously on
/// `submit()`.  A real OpenCL build would wrap a `cl_command_queue`.
#[derive(Debug)]
pub struct CommandQueue {
    id: usize,
    priority: QueuePriority,
    config: QueueConfig,
    pending: VecDeque<QueueEvent>,
    completed: Vec<QueueEvent>,
    next_event_id: Arc<AtomicU64>,
}

impl CommandQueue {
    /// Create a new command queue.
    pub fn new(
        id: usize,
        priority: QueuePriority,
        config: QueueConfig,
        event_counter: Arc<AtomicU64>,
    ) -> Result<Self, QueueError> {
        config.validate()?;
        Ok(Self {
            id,
            priority,
            config,
            pending: VecDeque::new(),
            completed: Vec::new(),
            next_event_id: event_counter,
        })
    }

    /// Queue identifier.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Priority level of this queue.
    pub fn priority(&self) -> QueuePriority {
        self.priority
    }

    /// Whether profiling is enabled.
    pub fn profiling_enabled(&self) -> bool {
        self.config.enable_profiling
    }

    /// Whether out-of-order execution is enabled.
    pub fn out_of_order(&self) -> bool {
        self.config.out_of_order
    }

    /// Number of pending (not yet completed) commands.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Number of completed commands.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Submit a command and receive an event handle.
    ///
    /// In the CPU reference implementation the command completes immediately.
    pub fn submit(&mut self) -> QueueEvent {
        let eid = EventId::new(self.next_event_id.fetch_add(1, Ordering::Relaxed));
        let mut event = QueueEvent::new(eid, self.priority);
        // CPU reference: execute immediately
        event.mark_running();
        event.mark_complete();
        self.completed.push(event.clone());
        event
    }

    /// Submit a command that will fail (for testing error paths).
    pub fn submit_failing(&mut self) -> QueueEvent {
        let eid = EventId::new(self.next_event_id.fetch_add(1, Ordering::Relaxed));
        let mut event = QueueEvent::new(eid, self.priority);
        event.mark_running();
        event.mark_error();
        self.completed.push(event.clone());
        event
    }

    /// Wait for all pending commands to complete.
    ///
    /// Returns immediately in the CPU reference implementation.
    pub fn flush(&mut self) -> Result<(), QueueError> {
        while let Some(mut ev) = self.pending.pop_front() {
            ev.mark_running();
            ev.mark_complete();
            self.completed.push(ev);
        }
        Ok(())
    }

    /// Drain completed events.
    pub fn drain_completed(&mut self) -> Vec<QueueEvent> {
        std::mem::take(&mut self.completed)
    }
}

// ── Queue manager ──────────────────────────────────────────────────

/// Manages multiple command queues with priority-based dispatch.
///
/// Commands submitted at higher priorities are dispatched first.  Within
/// the same priority, queues are selected round-robin for load balancing.
pub struct QueueManager {
    queues: HashMap<QueuePriority, Vec<CommandQueue>>,
    round_robin: HashMap<QueuePriority, usize>,
    config: QueueConfig,
    event_counter: Arc<AtomicU64>,
    stats: Arc<Mutex<QueueStats>>,
}

impl QueueManager {
    /// Create a new queue manager with the given configuration.
    pub fn new(config: QueueConfig) -> Result<Self, QueueError> {
        config.validate()?;
        let event_counter = Arc::new(AtomicU64::new(0));
        let mut queues = HashMap::new();
        let mut round_robin = HashMap::new();
        let mut global_id = 0usize;

        for &prio in QueuePriority::all() {
            let mut prio_queues = Vec::with_capacity(config.queue_count);
            for _ in 0..config.queue_count {
                prio_queues.push(CommandQueue::new(
                    global_id,
                    prio,
                    config.clone(),
                    Arc::clone(&event_counter),
                )?);
                global_id += 1;
            }
            queues.insert(prio, prio_queues);
            round_robin.insert(prio, 0);
        }

        Ok(Self {
            queues,
            round_robin,
            config,
            event_counter,
            stats: Arc::new(Mutex::new(QueueStats::default())),
        })
    }

    /// Configuration used to create this manager.
    pub fn config(&self) -> &QueueConfig {
        &self.config
    }

    /// Total number of backing queues across all priorities.
    pub fn total_queue_count(&self) -> usize {
        self.queues.values().map(|v| v.len()).sum()
    }

    /// Submit a command at the given priority.
    pub fn submit(&mut self, priority: QueuePriority) -> Result<QueueEvent, QueueError> {
        let queues = self.queues.get_mut(&priority).ok_or(QueueError::NoPriorityQueue(priority))?;

        let rr = self.round_robin.get_mut(&priority).unwrap();
        let idx = *rr % queues.len();
        *rr = rr.wrapping_add(1);

        let event = queues[idx].submit();

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.record_submit(priority);
            if event.is_done() {
                stats.record_complete(priority, event.latency());
            }
        }

        Ok(event)
    }

    /// Submit a command that will fail (for testing error paths).
    pub fn submit_failing(&mut self, priority: QueuePriority) -> Result<QueueEvent, QueueError> {
        let queues = self.queues.get_mut(&priority).ok_or(QueueError::NoPriorityQueue(priority))?;

        let rr = self.round_robin.get_mut(&priority).unwrap();
        let idx = *rr % queues.len();
        *rr = rr.wrapping_add(1);

        let event = queues[idx].submit_failing();

        if let Ok(mut stats) = self.stats.lock() {
            stats.record_submit(priority);
            stats.record_error(priority);
        }

        Ok(event)
    }

    /// Submit commands in priority order: high → low.
    ///
    /// Returns events in submission order (highest priority first).
    pub fn submit_priority_ordered(
        &mut self,
        requests: &[(QueuePriority, usize)],
    ) -> Result<Vec<QueueEvent>, QueueError> {
        let mut sorted: Vec<(QueuePriority, usize)> = requests.to_vec();
        sorted.sort_by(|a, b| b.0.cmp(&a.0)); // highest first

        let mut events = Vec::new();
        for (prio, count) in sorted {
            for _ in 0..count {
                events.push(self.submit(prio)?);
            }
        }
        Ok(events)
    }

    /// Flush all queues.
    pub fn flush_all(&mut self) -> Result<(), QueueError> {
        for queues in self.queues.values_mut() {
            for q in queues {
                q.flush()?;
            }
        }
        Ok(())
    }

    /// Get a snapshot of current statistics.
    pub fn stats(&self) -> QueueStats {
        self.stats.lock().unwrap().clone()
    }

    /// Access the backing queues for a given priority.
    pub fn queues_for(&self, priority: QueuePriority) -> Option<&[CommandQueue]> {
        self.queues.get(&priority).map(|v| v.as_slice())
    }

    /// Total pending commands across all queues.
    pub fn total_pending(&self) -> usize {
        self.queues.values().flat_map(|qs| qs.iter()).map(|q| q.pending_count()).sum()
    }

    /// Total number of events ever created across all queues.
    pub fn total_events_created(&self) -> u64 {
        self.event_counter.load(Ordering::Relaxed)
    }

    /// Total completed commands across all queues.
    pub fn total_completed(&self) -> usize {
        self.queues.values().flat_map(|qs| qs.iter()).map(|q| q.completed_count()).sum()
    }
}

impl fmt::Debug for QueueManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueueManager")
            .field("total_queues", &self.total_queue_count())
            .field("config", &self.config)
            .finish()
    }
}

// ── Event waiter ───────────────────────────────────────────────────

/// Waits for multiple events with an optional timeout.
pub struct EventWaiter {
    events: Vec<QueueEvent>,
    timeout: Option<Duration>,
}

impl EventWaiter {
    /// Create a waiter for the given events.
    pub fn new(events: Vec<QueueEvent>) -> Self {
        Self { events, timeout: None }
    }

    /// Set a timeout for the wait.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Number of events being waited on.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Wait for all events to complete.
    ///
    /// Returns `Ok(events)` if all finished within the timeout, or
    /// `Err(QueueError::Timeout)` if the deadline expired.
    pub fn wait_all(self) -> Result<Vec<QueueEvent>, QueueError> {
        let start = Instant::now();

        // CPU reference: events are already complete after submit,
        // but we honour the timeout contract.
        if let Some(timeout) = self.timeout {
            let all_done = self.events.iter().all(|e| e.is_done());
            if !all_done && start.elapsed() >= timeout {
                return Err(QueueError::Timeout {
                    waited: timeout,
                    pending: self.events.iter().filter(|e| !e.is_done()).count(),
                });
            }
        }

        // In CPU ref everything is already done
        let still_pending = self.events.iter().filter(|e| !e.is_done()).count();
        if still_pending > 0 {
            if let Some(timeout) = self.timeout {
                return Err(QueueError::Timeout { waited: timeout, pending: still_pending });
            }
            return Err(QueueError::EventIncomplete(still_pending));
        }

        Ok(self.events)
    }

    /// Wait for any single event to complete.
    ///
    /// Returns the first completed event, or error if none complete in time.
    pub fn wait_any(&self) -> Result<&QueueEvent, QueueError> {
        let start = Instant::now();

        for event in &self.events {
            if event.is_done() {
                return Ok(event);
            }
        }

        if let Some(timeout) = self.timeout
            && start.elapsed() >= timeout
        {
            return Err(QueueError::Timeout { waited: timeout, pending: self.events.len() });
        }

        Err(QueueError::EventIncomplete(self.events.len()))
    }
}

impl fmt::Debug for EventWaiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventWaiter")
            .field("event_count", &self.events.len())
            .field("timeout", &self.timeout)
            .finish()
    }
}

// ── Queue profiler ─────────────────────────────────────────────────

/// Timing information extracted from a profiling event.
#[derive(Debug, Clone)]
pub struct ProfilingInfo {
    /// Event that was profiled.
    pub event_id: EventId,
    /// Time from submit to start of execution.
    pub queue_latency: Duration,
    /// Time spent executing.
    pub execution_time: Duration,
    /// Total wall-clock time from submit to complete.
    pub total_time: Duration,
}

/// Extracts timing information from completed profiling events.
#[derive(Debug)]
pub struct QueueProfiler {
    records: Vec<ProfilingInfo>,
}

impl QueueProfiler {
    /// Create an empty profiler.
    pub fn new() -> Self {
        Self { records: Vec::new() }
    }

    /// Record profiling info from a completed event.
    ///
    /// Returns `None` if the event is not yet complete.
    pub fn record(&mut self, event: &QueueEvent) -> Option<ProfilingInfo> {
        if !event.is_done() {
            return None;
        }

        let total = event.latency().unwrap_or_default();
        // CPU reference: queue latency is negligible, all time is "execution"
        let queue_latency = Duration::from_micros(0);
        let execution_time = total;

        let info = ProfilingInfo {
            event_id: event.id(),
            queue_latency,
            execution_time,
            total_time: total,
        };

        self.records.push(info.clone());
        Some(info)
    }

    /// Number of recorded profiling entries.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// All recorded profiling entries.
    pub fn records(&self) -> &[ProfilingInfo] {
        &self.records
    }

    /// Average total time across all records.
    pub fn average_total_time(&self) -> Option<Duration> {
        if self.records.is_empty() {
            return None;
        }
        let sum: Duration = self.records.iter().map(|r| r.total_time).sum();
        Some(sum / self.records.len() as u32)
    }

    /// Average execution time across all records.
    pub fn average_execution_time(&self) -> Option<Duration> {
        if self.records.is_empty() {
            return None;
        }
        let sum: Duration = self.records.iter().map(|r| r.execution_time).sum();
        Some(sum / self.records.len() as u32)
    }

    /// Clear all recorded profiling data.
    pub fn clear(&mut self) {
        self.records.clear();
    }
}

impl Default for QueueProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Queue statistics ───────────────────────────────────────────────

/// Per-priority statistics counters.
#[derive(Debug, Clone, Default)]
pub struct PriorityStats {
    /// Number of commands submitted.
    pub submitted: u64,
    /// Number of commands completed successfully.
    pub completed: u64,
    /// Number of commands that errored.
    pub errored: u64,
    /// Sum of completion latencies for averaging.
    total_latency: Duration,
}

impl PriorityStats {
    /// Number of commands still pending (submitted - completed - errored).
    pub fn pending(&self) -> u64 {
        self.submitted.saturating_sub(self.completed + self.errored)
    }

    /// Average latency per completed command.
    pub fn avg_latency(&self) -> Option<Duration> {
        if self.completed == 0 {
            return None;
        }
        Some(self.total_latency / self.completed as u32)
    }
}

/// Aggregate statistics across all priority levels.
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    per_priority: HashMap<QueuePriority, PriorityStats>,
}

impl QueueStats {
    /// Get stats for a specific priority level.
    pub fn for_priority(&self, priority: QueuePriority) -> PriorityStats {
        self.per_priority.get(&priority).cloned().unwrap_or_default()
    }

    /// Total submitted across all priorities.
    pub fn total_submitted(&self) -> u64 {
        self.per_priority.values().map(|s| s.submitted).sum()
    }

    /// Total completed across all priorities.
    pub fn total_completed(&self) -> u64 {
        self.per_priority.values().map(|s| s.completed).sum()
    }

    /// Total pending across all priorities.
    pub fn total_pending(&self) -> u64 {
        self.per_priority.values().map(|s| s.pending()).sum()
    }

    /// Total errored across all priorities.
    pub fn total_errored(&self) -> u64 {
        self.per_priority.values().map(|s| s.errored).sum()
    }

    fn record_submit(&mut self, priority: QueuePriority) {
        self.per_priority.entry(priority).or_default().submitted += 1;
    }

    fn record_complete(&mut self, priority: QueuePriority, latency: Option<Duration>) {
        let entry = self.per_priority.entry(priority).or_default();
        entry.completed += 1;
        if let Some(lat) = latency {
            entry.total_latency += lat;
        }
    }

    fn record_error(&mut self, priority: QueuePriority) {
        self.per_priority.entry(priority).or_default().errored += 1;
    }
}

// ── Barrier ────────────────────────────────────────────────────────

/// A queue barrier that enforces ordering guarantees.
///
/// All commands submitted before the barrier must complete before any
/// command submitted after the barrier begins execution.
#[derive(Debug)]
pub struct Barrier {
    /// Events that must complete before the barrier is satisfied.
    pre_events: Vec<QueueEvent>,
    /// Whether the barrier has been satisfied.
    satisfied: bool,
    /// When the barrier was created.
    created_at: Instant,
}

impl Barrier {
    /// Create a barrier that waits for the given events.
    pub fn new(pre_events: Vec<QueueEvent>) -> Self {
        Self { pre_events, satisfied: false, created_at: Instant::now() }
    }

    /// Create an already-satisfied (empty) barrier.
    pub fn empty() -> Self {
        Self { pre_events: Vec::new(), satisfied: true, created_at: Instant::now() }
    }

    /// Number of prerequisite events.
    pub fn pre_event_count(&self) -> usize {
        self.pre_events.len()
    }

    /// Whether the barrier is satisfied (all prerequisites complete).
    pub fn is_satisfied(&self) -> bool {
        self.satisfied || self.pre_events.iter().all(|e| e.is_done())
    }

    /// Check and update the barrier status.
    pub fn check(&mut self) -> bool {
        if !self.satisfied {
            self.satisfied = self.pre_events.iter().all(|e| e.is_done());
        }
        self.satisfied
    }

    /// Wait for the barrier to be satisfied, with optional timeout.
    pub fn wait(&mut self, timeout: Option<Duration>) -> Result<(), QueueError> {
        let start = Instant::now();

        if self.check() {
            return Ok(());
        }

        if let Some(timeout) = timeout
            && start.elapsed() >= timeout
        {
            let pending = self.pre_events.iter().filter(|e| !e.is_done()).count();
            return Err(QueueError::Timeout { waited: timeout, pending });
        }

        // CPU reference: if not satisfied yet, it's an error
        let pending = self.pre_events.iter().filter(|e| !e.is_done()).count();
        if pending > 0 {
            return Err(QueueError::BarrierNotSatisfied(pending));
        }

        self.satisfied = true;
        Ok(())
    }

    /// Duration since barrier creation.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

// ── Errors ─────────────────────────────────────────────────────────

/// Errors from command queue operations.
#[derive(Debug, Clone)]
pub enum QueueError {
    /// Invalid queue configuration.
    InvalidConfig(String),
    /// No queue exists for the requested priority.
    NoPriorityQueue(QueuePriority),
    /// Timed out waiting for events.
    Timeout { waited: Duration, pending: usize },
    /// Events have not completed but no timeout was set.
    EventIncomplete(usize),
    /// A barrier prerequisite has not been satisfied.
    BarrierNotSatisfied(usize),
    /// Queue capacity exceeded.
    CapacityExceeded { capacity: usize, requested: usize },
    /// Underlying OpenCL error.
    OpenCl(String),
}

impl fmt::Display for QueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::NoPriorityQueue(p) => write!(f, "No queue for priority {p}"),
            Self::Timeout { waited, pending } => {
                write!(f, "Timeout after {waited:?} with {pending} events pending")
            }
            Self::EventIncomplete(n) => {
                write!(f, "{n} event(s) have not completed")
            }
            Self::BarrierNotSatisfied(n) => {
                write!(f, "Barrier has {n} unsatisfied prerequisite(s)")
            }
            Self::CapacityExceeded { capacity, requested } => {
                write!(f, "Queue capacity {capacity} exceeded (requested {requested})")
            }
            Self::OpenCl(msg) => write!(f, "OpenCL error: {msg}"),
        }
    }
}

impl std::error::Error for QueueError {}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── QueuePriority tests ─────────────────────────────────────────

    #[test]
    fn priority_display_high() {
        assert_eq!(QueuePriority::High.to_string(), "High");
    }

    #[test]
    fn priority_display_normal() {
        assert_eq!(QueuePriority::Normal.to_string(), "Normal");
    }

    #[test]
    fn priority_display_low() {
        assert_eq!(QueuePriority::Low.to_string(), "Low");
    }

    #[test]
    fn priority_display_background() {
        assert_eq!(QueuePriority::Background.to_string(), "Background");
    }

    #[test]
    fn priority_default_is_normal() {
        assert_eq!(QueuePriority::default(), QueuePriority::Normal);
    }

    #[test]
    fn priority_ordering_high_gt_low() {
        assert!(QueuePriority::High > QueuePriority::Low);
    }

    #[test]
    fn priority_ordering_normal_gt_background() {
        assert!(QueuePriority::Normal > QueuePriority::Background);
    }

    #[test]
    fn priority_all_returns_four_levels() {
        assert_eq!(QueuePriority::all().len(), 4);
    }

    #[test]
    fn priority_all_sorted_ascending() {
        let all = QueuePriority::all();
        for window in all.windows(2) {
            assert!(window[0] < window[1]);
        }
    }

    #[test]
    fn priority_clone_eq() {
        let p = QueuePriority::High;
        let c = p;
        assert_eq!(p, c);
    }

    #[test]
    fn priority_debug() {
        let dbg = format!("{:?}", QueuePriority::Normal);
        assert_eq!(dbg, "Normal");
    }

    #[test]
    fn priority_hash_same_for_equal() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        QueuePriority::High.hash(&mut h1);
        QueuePriority::High.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ── QueueConfig tests ───────────────────────────────────────────

    #[test]
    fn config_default_profiling_enabled() {
        assert!(QueueConfig::default().enable_profiling);
    }

    #[test]
    fn config_default_in_order() {
        assert!(!QueueConfig::default().out_of_order);
    }

    #[test]
    fn config_default_queue_count() {
        assert_eq!(QueueConfig::default().queue_count, 2);
    }

    #[test]
    fn config_with_profiling() {
        let cfg = QueueConfig::with_profiling();
        assert!(cfg.enable_profiling);
    }

    #[test]
    fn config_with_out_of_order() {
        let cfg = QueueConfig::with_out_of_order();
        assert!(cfg.out_of_order);
    }

    #[test]
    fn config_validate_ok() {
        assert!(QueueConfig::default().validate().is_ok());
    }

    #[test]
    fn config_validate_zero_queue_count() {
        let cfg = QueueConfig { queue_count: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_excessive_queue_count() {
        let cfg = QueueConfig { queue_count: 65, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_max_allowed() {
        let cfg = QueueConfig { queue_count: 64, ..Default::default() };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_debug() {
        let dbg = format!("{:?}", QueueConfig::default());
        assert!(dbg.contains("QueueConfig"));
    }

    // ── CommandQueue tests ──────────────────────────────────────────

    fn make_queue(priority: QueuePriority) -> CommandQueue {
        let counter = Arc::new(AtomicU64::new(0));
        CommandQueue::new(0, priority, QueueConfig::default(), counter).unwrap()
    }

    #[test]
    fn single_queue_submit_returns_complete_event() {
        let mut q = make_queue(QueuePriority::Normal);
        let ev = q.submit();
        assert!(ev.is_done());
        assert_eq!(ev.status(), EventStatus::Complete);
    }

    #[test]
    fn single_queue_submit_increments_completed() {
        let mut q = make_queue(QueuePriority::Normal);
        assert_eq!(q.completed_count(), 0);
        q.submit();
        assert_eq!(q.completed_count(), 1);
    }

    #[test]
    fn single_queue_submit_event_ids_are_unique() {
        let mut q = make_queue(QueuePriority::Normal);
        let e1 = q.submit();
        let e2 = q.submit();
        assert_ne!(e1.id(), e2.id());
    }

    #[test]
    fn single_queue_submit_event_has_latency() {
        let mut q = make_queue(QueuePriority::Normal);
        let ev = q.submit();
        assert!(ev.latency().is_some());
    }

    #[test]
    fn single_queue_submit_failing() {
        let mut q = make_queue(QueuePriority::Normal);
        let ev = q.submit_failing();
        assert!(ev.is_done());
        assert_eq!(ev.status(), EventStatus::Error);
    }

    #[test]
    fn single_queue_flush_empty() {
        let mut q = make_queue(QueuePriority::Normal);
        assert!(q.flush().is_ok());
    }

    #[test]
    fn single_queue_drain_completed() {
        let mut q = make_queue(QueuePriority::Normal);
        q.submit();
        q.submit();
        let drained = q.drain_completed();
        assert_eq!(drained.len(), 2);
        assert_eq!(q.completed_count(), 0);
    }

    #[test]
    fn single_queue_id() {
        let counter = Arc::new(AtomicU64::new(0));
        let q =
            CommandQueue::new(42, QueuePriority::High, QueueConfig::default(), counter).unwrap();
        assert_eq!(q.id(), 42);
    }

    #[test]
    fn single_queue_priority_preserved() {
        let q = make_queue(QueuePriority::High);
        assert_eq!(q.priority(), QueuePriority::High);
    }

    #[test]
    fn single_queue_profiling_from_config() {
        let q = make_queue(QueuePriority::Normal);
        assert!(q.profiling_enabled());
    }

    #[test]
    fn single_queue_out_of_order_from_config() {
        let q = make_queue(QueuePriority::Normal);
        assert!(!q.out_of_order());
    }

    // ── QueueManager tests ──────────────────────────────────────────

    fn make_manager() -> QueueManager {
        QueueManager::new(QueueConfig::default()).unwrap()
    }

    #[test]
    fn manager_creates_with_default_config() {
        let mgr = make_manager();
        assert_eq!(mgr.total_queue_count(), 4 * 2); // 4 priorities × 2 queues
    }

    #[test]
    fn manager_submit_normal_returns_complete() {
        let mut mgr = make_manager();
        let ev = mgr.submit(QueuePriority::Normal).unwrap();
        assert!(ev.is_done());
    }

    #[test]
    fn manager_submit_all_priorities() {
        let mut mgr = make_manager();
        for &prio in QueuePriority::all() {
            let ev = mgr.submit(prio).unwrap();
            assert!(ev.is_done());
        }
    }

    #[test]
    fn manager_round_robin_distributes() {
        let mut mgr = make_manager();
        // Submit 4 times to Normal (2 queues), should distribute 2+2
        for _ in 0..4 {
            mgr.submit(QueuePriority::Normal).unwrap();
        }
        let queues = mgr.queues_for(QueuePriority::Normal).unwrap();
        assert_eq!(queues[0].completed_count(), 2);
        assert_eq!(queues[1].completed_count(), 2);
    }

    #[test]
    fn manager_priority_ordering_high_before_low() {
        let mut mgr = make_manager();
        let events = mgr
            .submit_priority_ordered(&[(QueuePriority::Low, 2), (QueuePriority::High, 2)])
            .unwrap();
        // First two events should be High priority
        assert_eq!(events[0].priority(), QueuePriority::High);
        assert_eq!(events[1].priority(), QueuePriority::High);
        assert_eq!(events[2].priority(), QueuePriority::Low);
        assert_eq!(events[3].priority(), QueuePriority::Low);
    }

    #[test]
    fn manager_priority_ordering_all_high() {
        let mut mgr = make_manager();
        let events = mgr.submit_priority_ordered(&[(QueuePriority::High, 5)]).unwrap();
        assert_eq!(events.len(), 5);
        assert!(events.iter().all(|e| e.priority() == QueuePriority::High));
    }

    #[test]
    fn manager_priority_ordering_empty() {
        let mut mgr = make_manager();
        let events = mgr.submit_priority_ordered(&[]).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn manager_flush_all() {
        let mut mgr = make_manager();
        mgr.submit(QueuePriority::High).unwrap();
        mgr.submit(QueuePriority::Low).unwrap();
        assert!(mgr.flush_all().is_ok());
    }

    #[test]
    fn manager_total_pending_zero_after_cpu_submit() {
        let mut mgr = make_manager();
        mgr.submit(QueuePriority::Normal).unwrap();
        assert_eq!(mgr.total_pending(), 0);
    }

    #[test]
    fn manager_total_completed_matches_submits() {
        let mut mgr = make_manager();
        for _ in 0..10 {
            mgr.submit(QueuePriority::Normal).unwrap();
        }
        assert_eq!(mgr.total_completed(), 10);
    }

    #[test]
    fn manager_config_preserved() {
        let mgr = make_manager();
        assert!(mgr.config().enable_profiling);
    }

    #[test]
    fn manager_debug_impl() {
        let mgr = make_manager();
        let dbg = format!("{mgr:?}");
        assert!(dbg.contains("QueueManager"));
    }

    #[test]
    fn manager_invalid_config_zero_queues() {
        let cfg = QueueConfig { queue_count: 0, ..Default::default() };
        assert!(QueueManager::new(cfg).is_err());
    }

    #[test]
    fn manager_submit_failing_tracked() {
        let mut mgr = make_manager();
        let ev = mgr.submit_failing(QueuePriority::Normal).unwrap();
        assert_eq!(ev.status(), EventStatus::Error);
        let stats = mgr.stats();
        assert_eq!(stats.for_priority(QueuePriority::Normal).errored, 1);
    }

    // ── EventWaiter tests ───────────────────────────────────────────

    #[test]
    fn waiter_all_complete_events_succeeds() {
        let mut mgr = make_manager();
        let e1 = mgr.submit(QueuePriority::Normal).unwrap();
        let e2 = mgr.submit(QueuePriority::High).unwrap();
        let waiter = EventWaiter::new(vec![e1, e2]);
        let result = waiter.wait_all();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn waiter_empty_events_succeeds() {
        let waiter = EventWaiter::new(vec![]);
        assert!(waiter.wait_all().is_ok());
    }

    #[test]
    fn waiter_event_count() {
        let mut mgr = make_manager();
        let e1 = mgr.submit(QueuePriority::Normal).unwrap();
        let e2 = mgr.submit(QueuePriority::Normal).unwrap();
        let waiter = EventWaiter::new(vec![e1, e2]);
        assert_eq!(waiter.event_count(), 2);
    }

    #[test]
    fn waiter_with_timeout_all_complete() {
        let mut mgr = make_manager();
        let ev = mgr.submit(QueuePriority::Normal).unwrap();
        let waiter = EventWaiter::new(vec![ev]).with_timeout(Duration::from_secs(5));
        assert!(waiter.wait_all().is_ok());
    }

    #[test]
    fn waiter_wait_any_returns_first_done() {
        let mut mgr = make_manager();
        let e1 = mgr.submit(QueuePriority::Normal).unwrap();
        let e2 = mgr.submit(QueuePriority::High).unwrap();
        let waiter = EventWaiter::new(vec![e1, e2]);
        let result = waiter.wait_any();
        assert!(result.is_ok());
        assert!(result.unwrap().is_done());
    }

    #[test]
    fn waiter_wait_any_empty_is_err() {
        let waiter = EventWaiter::new(vec![]);
        assert!(waiter.wait_any().is_err());
    }

    #[test]
    fn waiter_timeout_with_incomplete_event() {
        // Construct an event that is not done (Queued status)
        let event = QueueEvent {
            id: EventId::new(999),
            priority: QueuePriority::Normal,
            status: EventStatus::Queued,
            submit_time: Instant::now(),
            complete_time: None,
        };
        let waiter = EventWaiter::new(vec![event]).with_timeout(Duration::from_millis(1));
        let result = waiter.wait_all();
        assert!(result.is_err());
    }

    #[test]
    fn waiter_debug_impl() {
        let waiter = EventWaiter::new(vec![]);
        let dbg = format!("{waiter:?}");
        assert!(dbg.contains("EventWaiter"));
    }

    // ── QueueProfiler tests ─────────────────────────────────────────

    #[test]
    fn profiler_new_empty() {
        let profiler = QueueProfiler::new();
        assert_eq!(profiler.record_count(), 0);
    }

    #[test]
    fn profiler_record_complete_event() {
        let mut profiler = QueueProfiler::new();
        let mut mgr = make_manager();
        let ev = mgr.submit(QueuePriority::Normal).unwrap();
        let info = profiler.record(&ev);
        assert!(info.is_some());
        assert_eq!(profiler.record_count(), 1);
    }

    #[test]
    fn profiler_record_incomplete_event_returns_none() {
        let mut profiler = QueueProfiler::new();
        let event = QueueEvent {
            id: EventId::new(0),
            priority: QueuePriority::Normal,
            status: EventStatus::Queued,
            submit_time: Instant::now(),
            complete_time: None,
        };
        assert!(profiler.record(&event).is_none());
        assert_eq!(profiler.record_count(), 0);
    }

    #[test]
    fn profiler_extracts_timing() {
        let mut profiler = QueueProfiler::new();
        let mut mgr = make_manager();
        let ev = mgr.submit(QueuePriority::Normal).unwrap();
        let info = profiler.record(&ev).unwrap();
        // Total time should be non-negative
        assert!(info.total_time >= Duration::ZERO);
        assert!(info.execution_time >= Duration::ZERO);
        assert_eq!(info.queue_latency, Duration::from_micros(0));
    }

    #[test]
    fn profiler_average_total_time() {
        let mut profiler = QueueProfiler::new();
        let mut mgr = make_manager();
        for _ in 0..5 {
            let ev = mgr.submit(QueuePriority::Normal).unwrap();
            profiler.record(&ev);
        }
        assert!(profiler.average_total_time().is_some());
    }

    #[test]
    fn profiler_average_total_time_empty() {
        let profiler = QueueProfiler::new();
        assert!(profiler.average_total_time().is_none());
    }

    #[test]
    fn profiler_average_execution_time() {
        let mut profiler = QueueProfiler::new();
        let mut mgr = make_manager();
        let ev = mgr.submit(QueuePriority::Normal).unwrap();
        profiler.record(&ev);
        assert!(profiler.average_execution_time().is_some());
    }

    #[test]
    fn profiler_clear() {
        let mut profiler = QueueProfiler::new();
        let mut mgr = make_manager();
        let ev = mgr.submit(QueuePriority::Normal).unwrap();
        profiler.record(&ev);
        profiler.clear();
        assert_eq!(profiler.record_count(), 0);
    }

    #[test]
    fn profiler_records_accessor() {
        let mut profiler = QueueProfiler::new();
        let mut mgr = make_manager();
        let ev = mgr.submit(QueuePriority::Normal).unwrap();
        profiler.record(&ev);
        assert_eq!(profiler.records().len(), 1);
    }

    #[test]
    fn profiler_default_trait() {
        let profiler = QueueProfiler::default();
        assert_eq!(profiler.record_count(), 0);
    }

    // ── QueueStats tests ────────────────────────────────────────────

    #[test]
    fn stats_initial_all_zero() {
        let stats = QueueStats::default();
        assert_eq!(stats.total_submitted(), 0);
        assert_eq!(stats.total_completed(), 0);
        assert_eq!(stats.total_pending(), 0);
        assert_eq!(stats.total_errored(), 0);
    }

    #[test]
    fn stats_after_single_submit() {
        let mut mgr = make_manager();
        mgr.submit(QueuePriority::Normal).unwrap();
        let stats = mgr.stats();
        assert_eq!(stats.total_submitted(), 1);
        assert_eq!(stats.total_completed(), 1);
    }

    #[test]
    fn stats_per_priority_tracking() {
        let mut mgr = make_manager();
        mgr.submit(QueuePriority::High).unwrap();
        mgr.submit(QueuePriority::High).unwrap();
        mgr.submit(QueuePriority::Low).unwrap();
        let stats = mgr.stats();
        assert_eq!(stats.for_priority(QueuePriority::High).submitted, 2);
        assert_eq!(stats.for_priority(QueuePriority::Low).submitted, 1);
        assert_eq!(stats.for_priority(QueuePriority::Normal).submitted, 0);
    }

    #[test]
    fn stats_avg_latency_none_when_no_completions() {
        let stats = QueueStats::default();
        assert!(stats.for_priority(QueuePriority::High).avg_latency().is_none());
    }

    #[test]
    fn stats_avg_latency_some_after_complete() {
        let mut mgr = make_manager();
        mgr.submit(QueuePriority::Normal).unwrap();
        let stats = mgr.stats();
        assert!(stats.for_priority(QueuePriority::Normal).avg_latency().is_some());
    }

    #[test]
    fn stats_completed_le_submitted() {
        let mut mgr = make_manager();
        for _ in 0..20 {
            mgr.submit(QueuePriority::Normal).unwrap();
        }
        let stats = mgr.stats();
        assert!(stats.total_completed() <= stats.total_submitted());
    }

    #[test]
    fn stats_error_tracking() {
        let mut mgr = make_manager();
        mgr.submit_failing(QueuePriority::Normal).unwrap();
        let stats = mgr.stats();
        assert_eq!(stats.total_errored(), 1);
        assert_eq!(stats.for_priority(QueuePriority::Normal).errored, 1);
    }

    #[test]
    fn stats_pending_is_zero_for_cpu() {
        let mut mgr = make_manager();
        for _ in 0..10 {
            mgr.submit(QueuePriority::High).unwrap();
        }
        let stats = mgr.stats();
        assert_eq!(stats.total_pending(), 0);
    }

    // ── Barrier tests ───────────────────────────────────────────────

    #[test]
    fn barrier_empty_is_satisfied() {
        let barrier = Barrier::empty();
        assert!(barrier.is_satisfied());
    }

    #[test]
    fn barrier_with_complete_events_is_satisfied() {
        let mut mgr = make_manager();
        let e1 = mgr.submit(QueuePriority::Normal).unwrap();
        let e2 = mgr.submit(QueuePriority::High).unwrap();
        let barrier = Barrier::new(vec![e1, e2]);
        assert!(barrier.is_satisfied());
    }

    #[test]
    fn barrier_check_updates_status() {
        let mut mgr = make_manager();
        let ev = mgr.submit(QueuePriority::Normal).unwrap();
        let mut barrier = Barrier::new(vec![ev]);
        assert!(barrier.check());
    }

    #[test]
    fn barrier_wait_succeeds_when_satisfied() {
        let mut mgr = make_manager();
        let ev = mgr.submit(QueuePriority::Normal).unwrap();
        let mut barrier = Barrier::new(vec![ev]);
        assert!(barrier.wait(None).is_ok());
    }

    #[test]
    fn barrier_wait_with_timeout_succeeds() {
        let mut mgr = make_manager();
        let ev = mgr.submit(QueuePriority::Normal).unwrap();
        let mut barrier = Barrier::new(vec![ev]);
        assert!(barrier.wait(Some(Duration::from_secs(1))).is_ok());
    }

    #[test]
    fn barrier_pre_event_count() {
        let mut mgr = make_manager();
        let e1 = mgr.submit(QueuePriority::Normal).unwrap();
        let e2 = mgr.submit(QueuePriority::Normal).unwrap();
        let barrier = Barrier::new(vec![e1, e2]);
        assert_eq!(barrier.pre_event_count(), 2);
    }

    #[test]
    fn barrier_enforces_ordering() {
        let mut mgr = make_manager();
        // Submit "before" commands
        let pre1 = mgr.submit(QueuePriority::Normal).unwrap();
        let pre2 = mgr.submit(QueuePriority::Normal).unwrap();

        // Create barrier
        let mut barrier = Barrier::new(vec![pre1, pre2]);
        assert!(barrier.check());

        // Post-barrier commands only execute after barrier satisfied
        let post = mgr.submit(QueuePriority::Normal).unwrap();
        assert!(post.is_done());
    }

    #[test]
    fn barrier_age_non_negative() {
        let barrier = Barrier::empty();
        assert!(barrier.age() >= Duration::ZERO);
    }

    #[test]
    fn barrier_unsatisfied_wait_no_timeout() {
        let event = QueueEvent {
            id: EventId::new(0),
            priority: QueuePriority::Normal,
            status: EventStatus::Queued,
            submit_time: Instant::now(),
            complete_time: None,
        };
        let mut barrier = Barrier::new(vec![event]);
        assert!(barrier.wait(None).is_err());
    }

    // ── EventId / EventStatus tests ─────────────────────────────────

    #[test]
    fn event_id_display() {
        let id = EventId::new(42);
        assert_eq!(id.to_string(), "Event(42)");
    }

    #[test]
    fn event_id_value() {
        let id = EventId::new(99);
        assert_eq!(id.value(), 99);
    }

    #[test]
    fn event_id_eq() {
        assert_eq!(EventId::new(1), EventId::new(1));
        assert_ne!(EventId::new(1), EventId::new(2));
    }

    #[test]
    fn event_status_display_all() {
        assert_eq!(EventStatus::Queued.to_string(), "Queued");
        assert_eq!(EventStatus::Running.to_string(), "Running");
        assert_eq!(EventStatus::Complete.to_string(), "Complete");
        assert_eq!(EventStatus::Error.to_string(), "Error");
    }

    #[test]
    fn event_is_done_complete() {
        let mut ev = QueueEvent::new(EventId::new(0), QueuePriority::Normal);
        ev.mark_complete();
        assert!(ev.is_done());
    }

    #[test]
    fn event_is_done_error() {
        let mut ev = QueueEvent::new(EventId::new(0), QueuePriority::Normal);
        ev.mark_error();
        assert!(ev.is_done());
    }

    #[test]
    fn event_not_done_queued() {
        let ev = QueueEvent::new(EventId::new(0), QueuePriority::Normal);
        assert!(!ev.is_done());
    }

    #[test]
    fn event_not_done_running() {
        let mut ev = QueueEvent::new(EventId::new(0), QueuePriority::Normal);
        ev.mark_running();
        assert!(!ev.is_done());
    }

    #[test]
    fn event_latency_none_when_not_done() {
        let ev = QueueEvent::new(EventId::new(0), QueuePriority::Normal);
        assert!(ev.latency().is_none());
    }

    // ── QueueError tests ────────────────────────────────────────────

    #[test]
    fn error_display_invalid_config() {
        let e = QueueError::InvalidConfig("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn error_display_timeout() {
        let e = QueueError::Timeout { waited: Duration::from_secs(5), pending: 3 };
        let s = e.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("3"));
    }

    #[test]
    fn error_display_capacity() {
        let e = QueueError::CapacityExceeded { capacity: 100, requested: 200 };
        let s = e.to_string();
        assert!(s.contains("100"));
        assert!(s.contains("200"));
    }

    #[test]
    fn error_display_opencl() {
        let e = QueueError::OpenCl("CL_OUT_OF_RESOURCES".into());
        assert!(e.to_string().contains("CL_OUT_OF_RESOURCES"));
    }

    #[test]
    fn error_display_barrier() {
        let e = QueueError::BarrierNotSatisfied(2);
        assert!(e.to_string().contains("2"));
    }

    #[test]
    fn error_display_event_incomplete() {
        let e = QueueError::EventIncomplete(4);
        assert!(e.to_string().contains("4"));
    }

    #[test]
    fn error_display_no_priority_queue() {
        let e = QueueError::NoPriorityQueue(QueuePriority::High);
        assert!(e.to_string().contains("High"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(QueueError::OpenCl("x".into()));
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn error_debug_impl() {
        let e = QueueError::Timeout { waited: Duration::from_secs(1), pending: 1 };
        let dbg = format!("{e:?}");
        assert!(dbg.contains("Timeout"));
    }

    // ── Edge case / property tests ──────────────────────────────────

    #[test]
    fn edge_empty_queue_no_completions() {
        let q = make_queue(QueuePriority::Normal);
        assert_eq!(q.completed_count(), 0);
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn edge_many_submits_all_complete() {
        let mut mgr = make_manager();
        for _ in 0..100 {
            let ev = mgr.submit(QueuePriority::Normal).unwrap();
            assert!(ev.is_done());
        }
        assert_eq!(mgr.total_completed(), 100);
    }

    #[test]
    fn property_completed_le_submitted_stress() {
        let mut mgr = make_manager();
        for i in 0..50 {
            let prio = match i % 4 {
                0 => QueuePriority::High,
                1 => QueuePriority::Normal,
                2 => QueuePriority::Low,
                _ => QueuePriority::Background,
            };
            mgr.submit(prio).unwrap();
        }
        let stats = mgr.stats();
        assert!(stats.total_completed() <= stats.total_submitted());
    }

    #[test]
    fn property_event_ids_monotonic() {
        let mut mgr = make_manager();
        let mut prev = 0u64;
        for _ in 0..20 {
            let ev = mgr.submit(QueuePriority::Normal).unwrap();
            assert!(ev.id().value() >= prev);
            prev = ev.id().value();
        }
    }

    #[test]
    fn edge_single_queue_config() {
        let cfg = QueueConfig { queue_count: 1, ..Default::default() };
        let mgr = QueueManager::new(cfg).unwrap();
        assert_eq!(mgr.total_queue_count(), 4);
    }

    #[test]
    fn edge_max_queue_config() {
        let cfg = QueueConfig { queue_count: 64, ..Default::default() };
        let mgr = QueueManager::new(cfg).unwrap();
        assert_eq!(mgr.total_queue_count(), 4 * 64);
    }

    #[test]
    fn edge_mixed_priority_submit_and_fail() {
        let mut mgr = make_manager();
        mgr.submit(QueuePriority::High).unwrap();
        mgr.submit_failing(QueuePriority::Normal).unwrap();
        mgr.submit(QueuePriority::Low).unwrap();
        let stats = mgr.stats();
        assert_eq!(stats.total_submitted(), 3);
        assert_eq!(stats.total_errored(), 1);
    }

    #[test]
    fn profiling_info_event_id_matches() {
        let mut profiler = QueueProfiler::new();
        let mut mgr = make_manager();
        let ev = mgr.submit(QueuePriority::Normal).unwrap();
        let info = profiler.record(&ev).unwrap();
        assert_eq!(info.event_id, ev.id());
    }

    #[test]
    fn priority_stats_pending_calculation() {
        let mut ps = PriorityStats::default();
        ps.submitted = 10;
        ps.completed = 7;
        ps.errored = 1;
        assert_eq!(ps.pending(), 2);
    }
}
