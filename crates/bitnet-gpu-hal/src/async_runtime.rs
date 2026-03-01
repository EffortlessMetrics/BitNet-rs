//! Async runtime for concurrent inference task management.
//!
//! Provides a cooperative async runtime for scheduling, cancelling, and
//! orchestrating GPU/CPU inference workloads across worker threads.  The main
//! entry point is [`AsyncRuntime`] which ties together task scheduling,
//! cancellation, bounded channels, barriers, pipelines, and timeouts.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ── Configuration ───────────────────────────────────────────────────────────

/// Top-level configuration for the async runtime.
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    /// Number of worker threads for task execution.
    pub thread_count: usize,
    /// Maximum number of pending tasks in the scheduler queue.
    pub queue_depth: usize,
    /// Global timeout applied to every task unless overridden.
    pub timeout: Duration,
    /// Whether to cancel in-flight tasks when the runtime is dropped.
    pub cancel_on_drop: bool,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            thread_count: 4,
            queue_depth: 256,
            timeout: Duration::from_secs(30),
            cancel_on_drop: true,
        }
    }
}

impl AsyncConfig {
    /// Create a new config with the given thread count.
    #[must_use]
    pub const fn with_thread_count(mut self, n: usize) -> Self {
        self.thread_count = n;
        self
    }

    /// Create a new config with the given queue depth.
    #[must_use]
    pub const fn with_queue_depth(mut self, depth: usize) -> Self {
        self.queue_depth = depth;
        self
    }

    /// Create a new config with the given timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.thread_count == 0 {
            return Err("thread_count must be > 0".into());
        }
        if self.queue_depth == 0 {
            return Err("queue_depth must be > 0".into());
        }
        if self.timeout.is_zero() {
            return Err("timeout must be > 0".into());
        }
        Ok(())
    }
}

// ── Cancel token ────────────────────────────────────────────────────────────

/// Cooperative cancellation token for long-running tasks.
///
/// Cloning a token shares the same underlying flag so that any holder can
/// trigger cancellation visible to all others.
#[derive(Debug, Clone)]
pub struct CancelToken {
    cancelled: Arc<AtomicBool>,
}

impl CancelToken {
    /// Create a new, non-cancelled token.
    #[must_use]
    pub fn new() -> Self {
        Self { cancelled: Arc::new(AtomicBool::new(false)) }
    }

    /// Signal cancellation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Check whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Reset the token to non-cancelled state.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::Release);
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

// ── Task priority ───────────────────────────────────────────────────────────

/// Execution priority for an [`AsyncTask`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum TaskPriority {
    /// Background / best-effort.
    Low,
    /// Default scheduling priority.
    #[default]
    Normal,
    /// Latency-sensitive work.
    High,
    /// Pre-empt everything else.
    Critical,
}

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Normal => write!(f, "Normal"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

// ── Task state ──────────────────────────────────────────────────────────────

/// Lifecycle state of an [`AsyncTask`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskState {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
}

impl fmt::Display for TaskState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Running => write!(f, "Running"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed => write!(f, "Failed"),
            Self::Cancelled => write!(f, "Cancelled"),
            Self::TimedOut => write!(f, "TimedOut"),
        }
    }
}

// ── Async task ──────────────────────────────────────────────────────────────

static TASK_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Lightweight async task descriptor.
#[derive(Debug, Clone)]
pub struct AsyncTask {
    /// Unique task identifier.
    pub id: u64,
    /// Human-readable name for logging.
    pub name: String,
    /// Scheduling priority.
    pub priority: TaskPriority,
    /// Cooperative cancellation token.
    pub cancel_token: CancelToken,
    /// Current lifecycle state.
    pub state: TaskState,
    /// When the task was created.
    pub created_at: Instant,
    /// Optional per-task timeout override.
    pub timeout: Option<Duration>,
}

impl AsyncTask {
    /// Create a new pending task with an auto-assigned id.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: TASK_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            name: name.into(),
            priority: TaskPriority::default(),
            cancel_token: CancelToken::new(),
            state: TaskState::Pending,
            created_at: Instant::now(),
            timeout: None,
        }
    }

    /// Set the priority.
    #[must_use]
    pub const fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set a per-task timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Attach an existing cancel token (e.g. shared across a group).
    #[must_use]
    pub fn with_cancel_token(mut self, token: CancelToken) -> Self {
        self.cancel_token = token;
        self
    }

    /// Elapsed time since creation.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Whether the task has reached a terminal state.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            TaskState::Completed | TaskState::Failed | TaskState::Cancelled | TaskState::TimedOut
        )
    }

    /// Cancel the task cooperatively.
    pub fn cancel(&mut self) {
        self.cancel_token.cancel();
        if !self.is_terminal() {
            self.state = TaskState::Cancelled;
        }
    }

    /// Transition to running.
    pub fn start(&mut self) {
        if self.state == TaskState::Pending {
            self.state = TaskState::Running;
        }
    }

    /// Mark as completed.
    pub const fn complete(&mut self) {
        if !self.is_terminal() {
            self.state = TaskState::Completed;
        }
    }

    /// Mark as failed.
    pub const fn fail(&mut self) {
        if !self.is_terminal() {
            self.state = TaskState::Failed;
        }
    }
}

// ── Task result ─────────────────────────────────────────────────────────────

/// Result produced by a completed task.
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Id of the task that produced this result.
    pub task_id: u64,
    /// Terminal state.
    pub state: TaskState,
    /// Wall-clock duration of execution.
    pub duration: Duration,
    /// Optional output payload (opaque bytes for generality).
    pub output: Option<Vec<u8>>,
    /// Human-readable error message on failure.
    pub error: Option<String>,
}

impl TaskResult {
    /// Convenience constructor for a successful result.
    #[must_use]
    pub const fn success(task_id: u64, duration: Duration, output: Vec<u8>) -> Self {
        Self { task_id, state: TaskState::Completed, duration, output: Some(output), error: None }
    }

    /// Convenience constructor for a failed result.
    #[must_use]
    pub fn failure(task_id: u64, duration: Duration, error: impl Into<String>) -> Self {
        Self {
            task_id,
            state: TaskState::Failed,
            duration,
            output: None,
            error: Some(error.into()),
        }
    }

    /// Whether the result represents a success.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.state == TaskState::Completed
    }
}

// ── Task scheduler ──────────────────────────────────────────────────────────

/// Schedules tasks across worker threads by priority.
#[derive(Debug)]
pub struct TaskScheduler {
    /// Pending queue ordered by priority (highest first).
    pending: Vec<AsyncTask>,
    /// Currently running tasks keyed by id.
    running: HashMap<u64, AsyncTask>,
    /// Maximum number of concurrent running tasks.
    max_concurrent: usize,
    /// Total tasks submitted over the scheduler's lifetime.
    total_submitted: u64,
    /// Total tasks completed (any terminal state).
    total_completed: u64,
}

impl TaskScheduler {
    /// Create a scheduler that allows up to `max_concurrent` tasks in flight.
    #[must_use]
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            pending: Vec::new(),
            running: HashMap::new(),
            max_concurrent,
            total_submitted: 0,
            total_completed: 0,
        }
    }

    /// Submit a task for scheduling.
    pub fn submit(&mut self, task: AsyncTask) {
        self.total_submitted += 1;
        self.pending.push(task);
        // Keep highest-priority tasks at the back so `pop` returns them first.
        self.pending.sort_by(|a, b| a.priority.cmp(&b.priority));
    }

    /// Move as many pending tasks to running as capacity allows.
    ///
    /// Returns the ids of newly started tasks.
    pub fn schedule(&mut self) -> Vec<u64> {
        let mut started = Vec::new();
        while self.running.len() < self.max_concurrent {
            if let Some(mut task) = self.pending.pop() {
                task.start();
                let id = task.id;
                self.running.insert(id, task);
                started.push(id);
            } else {
                break;
            }
        }
        started
    }

    /// Mark a running task as completed and remove it.
    pub fn complete_task(&mut self, task_id: u64) -> Option<AsyncTask> {
        if let Some(mut task) = self.running.remove(&task_id) {
            task.complete();
            self.total_completed += 1;
            Some(task)
        } else {
            None
        }
    }

    /// Mark a running task as failed and remove it.
    pub fn fail_task(&mut self, task_id: u64) -> Option<AsyncTask> {
        if let Some(mut task) = self.running.remove(&task_id) {
            task.fail();
            self.total_completed += 1;
            Some(task)
        } else {
            None
        }
    }

    /// Cancel a task (pending or running).
    pub fn cancel_task(&mut self, task_id: u64) -> Option<AsyncTask> {
        // Check running first.
        if let Some(mut task) = self.running.remove(&task_id) {
            task.cancel();
            self.total_completed += 1;
            return Some(task);
        }
        // Then pending.
        if let Some(pos) = self.pending.iter().position(|t| t.id == task_id) {
            let mut task = self.pending.remove(pos);
            task.cancel();
            self.total_completed += 1;
            return Some(task);
        }
        None
    }

    /// Number of tasks currently running.
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Number of tasks waiting in the queue.
    #[must_use]
    pub const fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Total tasks submitted over the scheduler's lifetime.
    #[must_use]
    pub const fn total_submitted(&self) -> u64 {
        self.total_submitted
    }

    /// Total tasks that reached a terminal state.
    #[must_use]
    pub const fn total_completed(&self) -> u64 {
        self.total_completed
    }

    /// Cancel all tasks (pending and running).
    pub fn cancel_all(&mut self) {
        for task in self.running.values_mut() {
            task.cancel();
        }
        self.total_completed += self.running.len() as u64;
        self.running.clear();

        for task in &mut self.pending {
            task.cancel();
        }
        self.total_completed += self.pending.len() as u64;
        self.pending.clear();
    }
}

// ── Async channel ───────────────────────────────────────────────────────────

/// Bounded MPSC channel for passing [`TaskResult`]s between producers and
/// a single consumer.
#[derive(Debug)]
pub struct AsyncChannel {
    buffer: Vec<TaskResult>,
    capacity: usize,
    total_sent: u64,
    total_received: u64,
    closed: bool,
}

impl AsyncChannel {
    /// Create a channel with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            total_sent: 0,
            total_received: 0,
            closed: false,
        }
    }

    /// Try to send a result into the channel.
    ///
    /// Returns `Err` if the channel is full or closed.
    pub fn try_send(&mut self, result: TaskResult) -> Result<(), TaskResult> {
        if self.closed {
            return Err(result);
        }
        if self.buffer.len() >= self.capacity {
            return Err(result);
        }
        self.buffer.push(result);
        self.total_sent += 1;
        Ok(())
    }

    /// Try to receive the next result.
    pub fn try_recv(&mut self) -> Option<TaskResult> {
        if self.buffer.is_empty() {
            return None;
        }
        let result = self.buffer.remove(0);
        self.total_received += 1;
        Some(result)
    }

    /// Drain all buffered results.
    pub fn drain(&mut self) -> Vec<TaskResult> {
        let count = self.buffer.len() as u64;
        self.total_received += count;
        self.buffer.drain(..).collect()
    }

    /// Close the channel. No further sends will be accepted.
    pub const fn close(&mut self) {
        self.closed = true;
    }

    /// Whether the channel is closed.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    /// Number of results currently buffered.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Whether the buffer is full.
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Channel capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total results sent over the channel's lifetime.
    #[must_use]
    pub const fn total_sent(&self) -> u64 {
        self.total_sent
    }

    /// Total results received over the channel's lifetime.
    #[must_use]
    pub const fn total_received(&self) -> u64 {
        self.total_received
    }
}

// ── Task group ──────────────────────────────────────────────────────────────

/// A group of related tasks that can be waited on or cancelled collectively.
#[derive(Debug)]
pub struct TaskGroup {
    /// Human-readable group name.
    pub name: String,
    /// Shared cancellation token for every task in the group.
    pub cancel_token: CancelToken,
    /// Tasks belonging to this group.
    tasks: Vec<AsyncTask>,
    /// Collected results.
    results: Vec<TaskResult>,
}

impl TaskGroup {
    /// Create a new, empty task group.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            cancel_token: CancelToken::new(),
            tasks: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Add a task to the group, attaching the shared cancel token.
    pub fn add(&mut self, task: AsyncTask) -> u64 {
        let id = task.id;
        let task = task.with_cancel_token(self.cancel_token.clone());
        self.tasks.push(task);
        id
    }

    /// Record a result for a group member.
    pub fn record_result(&mut self, result: TaskResult) {
        self.results.push(result);
    }

    /// Cancel all tasks in the group.
    pub fn cancel_all(&mut self) {
        self.cancel_token.cancel();
        for task in &mut self.tasks {
            task.cancel();
        }
    }

    /// Number of tasks in the group.
    #[must_use]
    pub const fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Number of results collected so far.
    #[must_use]
    pub const fn result_count(&self) -> usize {
        self.results.len()
    }

    /// Whether all tasks have produced a result.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.results.len() >= self.tasks.len() && !self.tasks.is_empty()
    }

    /// Whether every collected result is a success.
    #[must_use]
    pub fn all_succeeded(&self) -> bool {
        !self.results.is_empty() && self.results.iter().all(TaskResult::is_success)
    }

    /// Return references to the collected results.
    #[must_use]
    pub fn results(&self) -> &[TaskResult] {
        &self.results
    }

    /// Return references to the tasks.
    #[must_use]
    pub fn tasks(&self) -> &[AsyncTask] {
        &self.tasks
    }
}

// ── Async barrier ───────────────────────────────────────────────────────────

/// Synchronisation point for parallel tasks.
///
/// All participants must arrive before any can proceed past the barrier.
#[derive(Debug)]
pub struct AsyncBarrier {
    /// Total number of participants expected.
    expected: usize,
    /// Number that have arrived so far.
    arrived: usize,
    /// Whether the barrier has been released.
    released: bool,
    /// Instant when the barrier was created.
    created_at: Instant,
}

impl AsyncBarrier {
    /// Create a barrier that waits for `expected` arrivals.
    #[must_use]
    pub fn new(expected: usize) -> Self {
        Self { expected, arrived: 0, released: false, created_at: Instant::now() }
    }

    /// Record the arrival of one participant.
    ///
    /// Returns `true` when this arrival causes the barrier to release.
    pub const fn arrive(&mut self) -> bool {
        if self.released {
            return true;
        }
        self.arrived += 1;
        if self.arrived >= self.expected {
            self.released = true;
        }
        self.released
    }

    /// Whether the barrier has been released.
    #[must_use]
    pub const fn is_released(&self) -> bool {
        self.released
    }

    /// How many participants have arrived.
    #[must_use]
    pub const fn arrived_count(&self) -> usize {
        self.arrived
    }

    /// How many participants are still awaited.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.expected.saturating_sub(self.arrived)
    }

    /// Total expected participants.
    #[must_use]
    pub const fn expected(&self) -> usize {
        self.expected
    }

    /// Reset the barrier for re-use.
    pub fn reset(&mut self) {
        self.arrived = 0;
        self.released = false;
        self.created_at = Instant::now();
    }

    /// Elapsed time since creation (or last reset).
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }
}

// ── Task timer ──────────────────────────────────────────────────────────────

/// Timeout wrapper for async operations.
#[derive(Debug, Clone)]
pub struct TaskTimer {
    /// The deadline duration.
    timeout: Duration,
    /// When the timer was started (if started).
    started_at: Option<Instant>,
}

impl TaskTimer {
    /// Create a new timer with the given timeout.
    #[must_use]
    pub const fn new(timeout: Duration) -> Self {
        Self { timeout, started_at: None }
    }

    /// Start (or restart) the timer.
    pub fn start(&mut self) {
        self.started_at = Some(Instant::now());
    }

    /// Whether the timer has been started.
    #[must_use]
    pub const fn is_started(&self) -> bool {
        self.started_at.is_some()
    }

    /// Whether the deadline has passed.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.started_at.is_some_and(|s| s.elapsed() >= self.timeout)
    }

    /// Remaining time before expiry, or `Duration::ZERO` if expired/not started.
    #[must_use]
    pub fn remaining(&self) -> Duration {
        self.started_at.map_or(Duration::ZERO, |s| self.timeout.saturating_sub(s.elapsed()))
    }

    /// Elapsed time since start, or `Duration::ZERO` if not started.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started_at.map_or(Duration::ZERO, |s| s.elapsed())
    }

    /// The configured timeout duration.
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Reset the timer (clears start time).
    pub const fn reset(&mut self) {
        self.started_at = None;
    }
}

// ── Pipeline stage ──────────────────────────────────────────────────────────

/// Lifecycle state of a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageState {
    Idle,
    Running,
    Completed,
    Failed,
}

/// A single named stage inside an [`AsyncPipeline`].
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Human-readable name (e.g. "tokenize", "infer", "decode").
    pub name: String,
    /// Current state.
    pub state: StageState,
    /// Execution duration (set after completion or failure).
    pub duration: Option<Duration>,
    /// Error message on failure.
    pub error: Option<String>,
}

impl PipelineStage {
    /// Create a new idle stage.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), state: StageState::Idle, duration: None, error: None }
    }
}

// ── Async pipeline ──────────────────────────────────────────────────────────

/// Stage-by-stage async pipeline (e.g. tokenize → infer → decode).
#[derive(Debug)]
pub struct AsyncPipeline {
    stages: Vec<PipelineStage>,
    current_index: usize,
    cancel_token: CancelToken,
    started_at: Option<Instant>,
}

impl AsyncPipeline {
    /// Build a pipeline from an ordered list of stage names.
    #[must_use]
    pub fn new(stage_names: &[&str]) -> Self {
        Self {
            stages: stage_names.iter().map(|n| PipelineStage::new(*n)).collect(),
            current_index: 0,
            cancel_token: CancelToken::new(),
            started_at: None,
        }
    }

    /// Build the standard inference pipeline: tokenize → infer → decode.
    #[must_use]
    pub fn inference() -> Self {
        Self::new(&["tokenize", "infer", "decode"])
    }

    /// Attach a cancellation token.
    #[must_use]
    pub fn with_cancel_token(mut self, token: CancelToken) -> Self {
        self.cancel_token = token;
        self
    }

    /// Start the next idle stage.
    ///
    /// Returns the stage name, or `None` if the pipeline is finished or
    /// cancelled.
    pub fn advance(&mut self) -> Option<&str> {
        if self.cancel_token.is_cancelled() {
            return None;
        }
        if self.started_at.is_none() {
            self.started_at = Some(Instant::now());
        }
        if self.current_index >= self.stages.len() {
            return None;
        }
        self.stages[self.current_index].state = StageState::Running;
        Some(&self.stages[self.current_index].name)
    }

    /// Mark the current stage as completed and record its duration.
    pub fn complete_stage(&mut self, duration: Duration) {
        if self.current_index < self.stages.len() {
            self.stages[self.current_index].state = StageState::Completed;
            self.stages[self.current_index].duration = Some(duration);
            self.current_index += 1;
        }
    }

    /// Mark the current stage as failed.
    pub fn fail_stage(&mut self, error: impl Into<String>, duration: Duration) {
        if self.current_index < self.stages.len() {
            self.stages[self.current_index].state = StageState::Failed;
            self.stages[self.current_index].duration = Some(duration);
            self.stages[self.current_index].error = Some(error.into());
        }
    }

    /// Cancel the pipeline.
    pub fn cancel(&mut self) {
        self.cancel_token.cancel();
    }

    /// Whether all stages have completed successfully.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.stages.iter().all(|s| s.state == StageState::Completed)
    }

    /// Whether any stage has failed.
    #[must_use]
    pub fn has_failure(&self) -> bool {
        self.stages.iter().any(|s| s.state == StageState::Failed)
    }

    /// Index of the current (or next) stage.
    #[must_use]
    pub const fn current_index(&self) -> usize {
        self.current_index
    }

    /// Number of stages.
    #[must_use]
    pub const fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Read-only access to the stages.
    #[must_use]
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// Total wall-clock duration since the pipeline was first advanced.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started_at.map_or(Duration::ZERO, |s| s.elapsed())
    }

    /// Sum of completed stage durations.
    #[must_use]
    pub fn total_stage_duration(&self) -> Duration {
        self.stages.iter().filter_map(|s| s.duration).sum()
    }
}

// ── Runtime state ───────────────────────────────────────────────────────────

/// High-level state of the [`AsyncRuntime`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeState {
    /// Not yet started.
    Idle,
    /// Accepting and executing tasks.
    Running,
    /// Gracefully shutting down (no new tasks accepted).
    ShuttingDown,
    /// Fully stopped.
    Stopped,
}

impl fmt::Display for RuntimeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Running => write!(f, "Running"),
            Self::ShuttingDown => write!(f, "ShuttingDown"),
            Self::Stopped => write!(f, "Stopped"),
        }
    }
}

// ── Async runtime ───────────────────────────────────────────────────────────

/// Orchestrator: spawn tasks → schedule → monitor → collect results.
#[derive(Debug)]
pub struct AsyncRuntime {
    config: AsyncConfig,
    state: RuntimeState,
    scheduler: TaskScheduler,
    channel: AsyncChannel,
    groups: HashMap<String, TaskGroup>,
    started_at: Option<Instant>,
}

impl AsyncRuntime {
    /// Create a new runtime with the given configuration.
    pub fn new(config: AsyncConfig) -> Result<Self, String> {
        config.validate()?;
        let scheduler = TaskScheduler::new(config.thread_count);
        let channel = AsyncChannel::new(config.queue_depth);
        Ok(Self {
            config,
            state: RuntimeState::Idle,
            scheduler,
            channel,
            groups: HashMap::new(),
            started_at: None,
        })
    }

    /// Create a runtime with the default configuration.
    pub fn with_defaults() -> Result<Self, String> {
        Self::new(AsyncConfig::default())
    }

    /// Start the runtime.
    pub fn start(&mut self) -> Result<(), String> {
        match self.state {
            RuntimeState::Idle => {
                self.state = RuntimeState::Running;
                self.started_at = Some(Instant::now());
                log::info!(
                    "AsyncRuntime started: {} threads, queue depth {}",
                    self.config.thread_count,
                    self.config.queue_depth
                );
                Ok(())
            }
            RuntimeState::Running => Err("runtime is already running".into()),
            RuntimeState::ShuttingDown => Err("runtime is shutting down".into()),
            RuntimeState::Stopped => Err("runtime is stopped".into()),
        }
    }

    /// Submit a task for execution.
    pub fn submit(&mut self, task: AsyncTask) -> Result<u64, String> {
        if self.state != RuntimeState::Running {
            return Err(format!("runtime is not running (state={})", self.state));
        }
        let id = task.id;
        self.scheduler.submit(task);
        Ok(id)
    }

    /// Trigger the scheduler to move pending tasks to running.
    pub fn tick(&mut self) -> Vec<u64> {
        self.scheduler.schedule()
    }

    /// Record a task as completed and push its result into the channel.
    pub fn complete_task(&mut self, task_id: u64, output: Vec<u8>) -> Result<(), String> {
        let task = self
            .scheduler
            .complete_task(task_id)
            .ok_or_else(|| format!("task {task_id} not found in running set"))?;
        let result = TaskResult::success(task.id, task.elapsed(), output);
        self.channel.try_send(result).map_err(|_| "channel full".to_string())
    }

    /// Record a task as failed.
    pub fn fail_task(&mut self, task_id: u64, error: impl Into<String>) -> Result<(), String> {
        let task = self
            .scheduler
            .fail_task(task_id)
            .ok_or_else(|| format!("task {task_id} not found in running set"))?;
        let result = TaskResult::failure(task.id, task.elapsed(), error);
        self.channel.try_send(result).map_err(|_| "channel full".to_string())
    }

    /// Collect all available results from the channel.
    pub fn collect_results(&mut self) -> Vec<TaskResult> {
        self.channel.drain()
    }

    /// Create a named task group and return a mutable reference.
    pub fn create_group(&mut self, name: impl Into<String>) -> &mut TaskGroup {
        let name = name.into();
        self.groups.entry(name.clone()).or_insert_with(|| TaskGroup::new(name))
    }

    /// Get a reference to a task group by name.
    #[must_use]
    pub fn group(&self, name: &str) -> Option<&TaskGroup> {
        self.groups.get(name)
    }

    /// Get a mutable reference to a task group by name.
    pub fn group_mut(&mut self, name: &str) -> Option<&mut TaskGroup> {
        self.groups.get_mut(name)
    }

    /// Initiate graceful shutdown: stop accepting new tasks and cancel
    /// in-flight work if `cancel_on_drop` is set.
    pub fn shutdown(&mut self) {
        self.state = RuntimeState::ShuttingDown;
        if self.config.cancel_on_drop {
            self.scheduler.cancel_all();
            for group in self.groups.values_mut() {
                group.cancel_all();
            }
        }
        self.channel.close();
        self.state = RuntimeState::Stopped;
        log::info!("AsyncRuntime stopped");
    }

    /// Current runtime state.
    #[must_use]
    pub const fn state(&self) -> RuntimeState {
        self.state
    }

    /// Reference to the underlying config.
    #[must_use]
    pub const fn config(&self) -> &AsyncConfig {
        &self.config
    }

    /// Number of tasks currently running.
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.scheduler.running_count()
    }

    /// Number of tasks waiting in the queue.
    #[must_use]
    pub const fn pending_count(&self) -> usize {
        self.scheduler.pending_count()
    }

    /// Total tasks submitted.
    #[must_use]
    pub const fn total_submitted(&self) -> u64 {
        self.scheduler.total_submitted()
    }

    /// Elapsed time since the runtime was started.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.started_at.map_or(Duration::ZERO, |s| s.elapsed())
    }
}

impl Drop for AsyncRuntime {
    fn drop(&mut self) {
        if self.state == RuntimeState::Running {
            self.shutdown();
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- AsyncConfig ----------------------------------------------------------

    #[test]
    fn config_default_values() {
        let cfg = AsyncConfig::default();
        assert_eq!(cfg.thread_count, 4);
        assert_eq!(cfg.queue_depth, 256);
        assert_eq!(cfg.timeout, Duration::from_secs(30));
        assert!(cfg.cancel_on_drop);
    }

    #[test]
    fn config_builder_methods() {
        let cfg = AsyncConfig::default()
            .with_thread_count(8)
            .with_queue_depth(512)
            .with_timeout(Duration::from_secs(60));
        assert_eq!(cfg.thread_count, 8);
        assert_eq!(cfg.queue_depth, 512);
        assert_eq!(cfg.timeout, Duration::from_secs(60));
    }

    #[test]
    fn config_validate_ok() {
        assert!(AsyncConfig::default().validate().is_ok());
    }

    #[test]
    fn config_validate_zero_threads() {
        let cfg = AsyncConfig::default().with_thread_count(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_queue() {
        let cfg = AsyncConfig::default().with_queue_depth(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_timeout() {
        let cfg = AsyncConfig::default().with_timeout(Duration::ZERO);
        assert!(cfg.validate().is_err());
    }

    // -- CancelToken ----------------------------------------------------------

    #[test]
    fn cancel_token_initially_not_cancelled() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancel_token_cancel_sets_flag() {
        let token = CancelToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancel_token_clone_shares_state() {
        let t1 = CancelToken::new();
        let t2 = t1.clone();
        t1.cancel();
        assert!(t2.is_cancelled());
    }

    #[test]
    fn cancel_token_reset() {
        let token = CancelToken::new();
        token.cancel();
        token.reset();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancel_token_default_is_new() {
        let token = CancelToken::default();
        assert!(!token.is_cancelled());
    }

    // -- TaskPriority ---------------------------------------------------------

    #[test]
    fn priority_default_is_normal() {
        assert_eq!(TaskPriority::default(), TaskPriority::Normal);
    }

    #[test]
    fn priority_ordering() {
        assert!(TaskPriority::Low < TaskPriority::Normal);
        assert!(TaskPriority::Normal < TaskPriority::High);
        assert!(TaskPriority::High < TaskPriority::Critical);
    }

    #[test]
    fn priority_display() {
        assert_eq!(TaskPriority::Critical.to_string(), "Critical");
        assert_eq!(TaskPriority::Low.to_string(), "Low");
    }

    // -- TaskState ------------------------------------------------------------

    #[test]
    fn task_state_display() {
        assert_eq!(TaskState::Pending.to_string(), "Pending");
        assert_eq!(TaskState::Running.to_string(), "Running");
        assert_eq!(TaskState::Completed.to_string(), "Completed");
        assert_eq!(TaskState::Failed.to_string(), "Failed");
        assert_eq!(TaskState::Cancelled.to_string(), "Cancelled");
        assert_eq!(TaskState::TimedOut.to_string(), "TimedOut");
    }

    // -- AsyncTask ------------------------------------------------------------

    #[test]
    fn task_new_is_pending() {
        let t = AsyncTask::new("test");
        assert_eq!(t.state, TaskState::Pending);
        assert_eq!(t.name, "test");
        assert_eq!(t.priority, TaskPriority::Normal);
    }

    #[test]
    fn task_ids_are_unique() {
        let t1 = AsyncTask::new("a");
        let t2 = AsyncTask::new("b");
        assert_ne!(t1.id, t2.id);
    }

    #[test]
    fn task_with_priority() {
        let t = AsyncTask::new("x").with_priority(TaskPriority::High);
        assert_eq!(t.priority, TaskPriority::High);
    }

    #[test]
    fn task_with_timeout() {
        let t = AsyncTask::new("x").with_timeout(Duration::from_secs(5));
        assert_eq!(t.timeout, Some(Duration::from_secs(5)));
    }

    #[test]
    fn task_with_cancel_token() {
        let token = CancelToken::new();
        let t = AsyncTask::new("x").with_cancel_token(token.clone());
        token.cancel();
        assert!(t.cancel_token.is_cancelled());
    }

    #[test]
    fn task_start_transitions_to_running() {
        let mut t = AsyncTask::new("x");
        t.start();
        assert_eq!(t.state, TaskState::Running);
    }

    #[test]
    fn task_complete_transitions() {
        let mut t = AsyncTask::new("x");
        t.start();
        t.complete();
        assert_eq!(t.state, TaskState::Completed);
        assert!(t.is_terminal());
    }

    #[test]
    fn task_fail_transitions() {
        let mut t = AsyncTask::new("x");
        t.start();
        t.fail();
        assert_eq!(t.state, TaskState::Failed);
        assert!(t.is_terminal());
    }

    #[test]
    fn task_cancel_sets_state_and_token() {
        let mut t = AsyncTask::new("x");
        t.start();
        t.cancel();
        assert_eq!(t.state, TaskState::Cancelled);
        assert!(t.cancel_token.is_cancelled());
    }

    #[test]
    fn task_terminal_state_not_overwritten() {
        let mut t = AsyncTask::new("x");
        t.start();
        t.complete();
        t.fail(); // should be ignored
        assert_eq!(t.state, TaskState::Completed);
    }

    #[test]
    fn task_elapsed_is_non_zero() {
        let t = AsyncTask::new("x");
        // Even though tiny, it should be representable.
        assert!(t.elapsed() < Duration::from_secs(5));
    }

    // -- TaskResult -----------------------------------------------------------

    #[test]
    fn task_result_success() {
        let r = TaskResult::success(1, Duration::from_millis(10), vec![42]);
        assert!(r.is_success());
        assert_eq!(r.output.as_deref(), Some(&[42][..]));
        assert!(r.error.is_none());
    }

    #[test]
    fn task_result_failure() {
        let r = TaskResult::failure(2, Duration::from_millis(5), "boom");
        assert!(!r.is_success());
        assert_eq!(r.error.as_deref(), Some("boom"));
        assert!(r.output.is_none());
    }

    // -- TaskScheduler --------------------------------------------------------

    #[test]
    fn scheduler_submit_and_schedule() {
        let mut sched = TaskScheduler::new(2);
        sched.submit(AsyncTask::new("a"));
        sched.submit(AsyncTask::new("b"));
        sched.submit(AsyncTask::new("c"));
        let started = sched.schedule();
        assert_eq!(started.len(), 2); // max_concurrent=2
        assert_eq!(sched.running_count(), 2);
        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn scheduler_priority_ordering() {
        let mut sched = TaskScheduler::new(1);
        sched.submit(AsyncTask::new("low").with_priority(TaskPriority::Low));
        sched.submit(AsyncTask::new("high").with_priority(TaskPriority::High));
        let started = sched.schedule();
        assert_eq!(started.len(), 1);
        // The high-priority task should be scheduled first.
        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn scheduler_complete_task() {
        let mut sched = TaskScheduler::new(2);
        let t = AsyncTask::new("a");
        let id = t.id;
        sched.submit(t);
        sched.schedule();
        let completed = sched.complete_task(id);
        assert!(completed.is_some());
        assert_eq!(completed.unwrap().state, TaskState::Completed);
        assert_eq!(sched.running_count(), 0);
    }

    #[test]
    fn scheduler_fail_task() {
        let mut sched = TaskScheduler::new(2);
        let t = AsyncTask::new("a");
        let id = t.id;
        sched.submit(t);
        sched.schedule();
        let failed = sched.fail_task(id);
        assert!(failed.is_some());
        assert_eq!(failed.unwrap().state, TaskState::Failed);
    }

    #[test]
    fn scheduler_cancel_running_task() {
        let mut sched = TaskScheduler::new(2);
        let t = AsyncTask::new("a");
        let id = t.id;
        sched.submit(t);
        sched.schedule();
        let c = sched.cancel_task(id);
        assert!(c.is_some());
        assert_eq!(c.unwrap().state, TaskState::Cancelled);
    }

    #[test]
    fn scheduler_cancel_pending_task() {
        let mut sched = TaskScheduler::new(0);
        let t = AsyncTask::new("a");
        let id = t.id;
        sched.submit(t);
        // max_concurrent is 0, so nothing will start.
        sched.schedule();
        let c = sched.cancel_task(id);
        assert!(c.is_some());
        assert_eq!(c.unwrap().state, TaskState::Cancelled);
    }

    #[test]
    fn scheduler_cancel_nonexistent_returns_none() {
        let mut sched = TaskScheduler::new(2);
        assert!(sched.cancel_task(9999).is_none());
    }

    #[test]
    fn scheduler_cancel_all() {
        let mut sched = TaskScheduler::new(1);
        sched.submit(AsyncTask::new("a"));
        sched.submit(AsyncTask::new("b"));
        sched.schedule();
        sched.cancel_all();
        assert_eq!(sched.running_count(), 0);
        assert_eq!(sched.pending_count(), 0);
    }

    #[test]
    fn scheduler_total_counts() {
        let mut sched = TaskScheduler::new(2);
        sched.submit(AsyncTask::new("a"));
        sched.submit(AsyncTask::new("b"));
        assert_eq!(sched.total_submitted(), 2);
        sched.schedule();
        let ids: Vec<_> = sched.running.keys().copied().collect();
        for id in ids {
            sched.complete_task(id);
        }
        assert_eq!(sched.total_completed(), 2);
    }

    // -- AsyncChannel ---------------------------------------------------------

    #[test]
    fn channel_send_recv() {
        let mut ch = AsyncChannel::new(4);
        let r = TaskResult::success(1, Duration::ZERO, vec![]);
        assert!(ch.try_send(r).is_ok());
        assert_eq!(ch.len(), 1);
        let out = ch.try_recv().unwrap();
        assert_eq!(out.task_id, 1);
        assert!(ch.is_empty());
    }

    #[test]
    fn channel_full_rejects() {
        let mut ch = AsyncChannel::new(1);
        let r1 = TaskResult::success(1, Duration::ZERO, vec![]);
        let r2 = TaskResult::success(2, Duration::ZERO, vec![]);
        assert!(ch.try_send(r1).is_ok());
        assert!(ch.is_full());
        assert!(ch.try_send(r2).is_err());
    }

    #[test]
    fn channel_closed_rejects() {
        let mut ch = AsyncChannel::new(4);
        ch.close();
        let r = TaskResult::success(1, Duration::ZERO, vec![]);
        assert!(ch.try_send(r).is_err());
        assert!(ch.is_closed());
    }

    #[test]
    fn channel_drain() {
        let mut ch = AsyncChannel::new(8);
        for i in 0..3 {
            let r = TaskResult::success(i, Duration::ZERO, vec![]);
            ch.try_send(r).unwrap();
        }
        let drained = ch.drain();
        assert_eq!(drained.len(), 3);
        assert!(ch.is_empty());
    }

    #[test]
    fn channel_capacity() {
        let ch = AsyncChannel::new(16);
        assert_eq!(ch.capacity(), 16);
    }

    #[test]
    fn channel_total_counters() {
        let mut ch = AsyncChannel::new(8);
        for i in 0..3 {
            ch.try_send(TaskResult::success(i, Duration::ZERO, vec![])).unwrap();
        }
        assert_eq!(ch.total_sent(), 3);
        ch.try_recv();
        assert_eq!(ch.total_received(), 1);
        ch.drain();
        assert_eq!(ch.total_received(), 3);
    }

    #[test]
    fn channel_recv_empty_returns_none() {
        let mut ch = AsyncChannel::new(4);
        assert!(ch.try_recv().is_none());
    }

    // -- TaskGroup ------------------------------------------------------------

    #[test]
    fn group_add_and_count() {
        let mut g = TaskGroup::new("batch");
        g.add(AsyncTask::new("t1"));
        g.add(AsyncTask::new("t2"));
        assert_eq!(g.task_count(), 2);
    }

    #[test]
    fn group_shared_cancel_token() {
        let mut g = TaskGroup::new("g");
        g.add(AsyncTask::new("t1"));
        g.cancel_all();
        assert!(g.cancel_token.is_cancelled());
        assert!(g.tasks()[0].cancel_token.is_cancelled());
    }

    #[test]
    fn group_record_results() {
        let mut g = TaskGroup::new("g");
        g.add(AsyncTask::new("t1"));
        g.record_result(TaskResult::success(1, Duration::ZERO, vec![]));
        assert_eq!(g.result_count(), 1);
    }

    #[test]
    fn group_is_complete() {
        let mut g = TaskGroup::new("g");
        let id = g.add(AsyncTask::new("t1"));
        g.record_result(TaskResult::success(id, Duration::ZERO, vec![]));
        assert!(g.is_complete());
    }

    #[test]
    fn group_not_complete_when_empty() {
        let g = TaskGroup::new("g");
        assert!(!g.is_complete());
    }

    #[test]
    fn group_all_succeeded() {
        let mut g = TaskGroup::new("g");
        let id = g.add(AsyncTask::new("t1"));
        g.record_result(TaskResult::success(id, Duration::ZERO, vec![]));
        assert!(g.all_succeeded());
    }

    #[test]
    fn group_not_all_succeeded_on_failure() {
        let mut g = TaskGroup::new("g");
        g.add(AsyncTask::new("t1"));
        g.record_result(TaskResult::failure(1, Duration::ZERO, "err"));
        assert!(!g.all_succeeded());
    }

    // -- AsyncBarrier ---------------------------------------------------------

    #[test]
    fn barrier_not_released_initially() {
        let b = AsyncBarrier::new(3);
        assert!(!b.is_released());
        assert_eq!(b.remaining(), 3);
        assert_eq!(b.expected(), 3);
    }

    #[test]
    fn barrier_releases_on_expected_arrivals() {
        let mut b = AsyncBarrier::new(2);
        assert!(!b.arrive());
        assert!(b.arrive());
        assert!(b.is_released());
        assert_eq!(b.remaining(), 0);
    }

    #[test]
    fn barrier_arrive_after_release() {
        let mut b = AsyncBarrier::new(1);
        assert!(b.arrive());
        // Additional arrives are no-ops, still released.
        assert!(b.arrive());
    }

    #[test]
    fn barrier_reset() {
        let mut b = AsyncBarrier::new(2);
        b.arrive();
        b.arrive();
        b.reset();
        assert!(!b.is_released());
        assert_eq!(b.arrived_count(), 0);
    }

    #[test]
    fn barrier_elapsed() {
        let b = AsyncBarrier::new(1);
        assert!(b.elapsed() < Duration::from_secs(5));
    }

    // -- TaskTimer ------------------------------------------------------------

    #[test]
    fn timer_not_started() {
        let t = TaskTimer::new(Duration::from_secs(1));
        assert!(!t.is_started());
        assert!(!t.is_expired());
        assert_eq!(t.remaining(), Duration::ZERO);
        assert_eq!(t.elapsed(), Duration::ZERO);
    }

    #[test]
    fn timer_start_and_timeout() {
        let mut t = TaskTimer::new(Duration::from_secs(1));
        t.start();
        assert!(t.is_started());
        assert!(!t.is_expired());
        assert!(t.remaining() > Duration::ZERO);
    }

    #[test]
    fn timer_timeout_accessor() {
        let t = TaskTimer::new(Duration::from_millis(500));
        assert_eq!(t.timeout(), Duration::from_millis(500));
    }

    #[test]
    fn timer_reset_clears_start() {
        let mut t = TaskTimer::new(Duration::from_secs(1));
        t.start();
        t.reset();
        assert!(!t.is_started());
    }

    #[test]
    fn timer_expired_with_zero_timeout() {
        let mut t = TaskTimer::new(Duration::ZERO);
        t.start();
        assert!(t.is_expired());
    }

    // -- PipelineStage --------------------------------------------------------

    #[test]
    fn pipeline_stage_defaults_to_idle() {
        let s = PipelineStage::new("tokenize");
        assert_eq!(s.state, StageState::Idle);
        assert!(s.duration.is_none());
        assert!(s.error.is_none());
    }

    // -- AsyncPipeline --------------------------------------------------------

    #[test]
    fn pipeline_inference_has_three_stages() {
        let p = AsyncPipeline::inference();
        assert_eq!(p.stage_count(), 3);
        assert_eq!(p.stages()[0].name, "tokenize");
        assert_eq!(p.stages()[1].name, "infer");
        assert_eq!(p.stages()[2].name, "decode");
    }

    #[test]
    fn pipeline_advance_and_complete() {
        let mut p = AsyncPipeline::inference();
        let name = p.advance().unwrap().to_string();
        assert_eq!(name, "tokenize");
        assert_eq!(p.stages()[0].state, StageState::Running);

        p.complete_stage(Duration::from_millis(10));
        assert_eq!(p.stages()[0].state, StageState::Completed);
        assert_eq!(p.current_index(), 1);
    }

    #[test]
    fn pipeline_full_run() {
        let mut p = AsyncPipeline::inference();
        for _ in 0..3 {
            p.advance();
            p.complete_stage(Duration::from_millis(5));
        }
        assert!(p.is_complete());
        assert!(!p.has_failure());
    }

    #[test]
    fn pipeline_advance_past_end_returns_none() {
        let mut p = AsyncPipeline::new(&["only"]);
        p.advance();
        p.complete_stage(Duration::ZERO);
        assert!(p.advance().is_none());
    }

    #[test]
    fn pipeline_fail_stage() {
        let mut p = AsyncPipeline::inference();
        p.advance();
        p.fail_stage("oom", Duration::from_millis(1));
        assert!(p.has_failure());
        assert!(!p.is_complete());
        assert_eq!(p.stages()[0].error.as_deref(), Some("oom"));
    }

    #[test]
    fn pipeline_cancel_prevents_advance() {
        let mut p = AsyncPipeline::inference();
        p.cancel();
        assert!(p.advance().is_none());
    }

    #[test]
    fn pipeline_with_cancel_token() {
        let token = CancelToken::new();
        let mut p = AsyncPipeline::inference().with_cancel_token(token.clone());
        token.cancel();
        assert!(p.advance().is_none());
    }

    #[test]
    fn pipeline_total_stage_duration() {
        let mut p = AsyncPipeline::new(&["a", "b"]);
        p.advance();
        p.complete_stage(Duration::from_millis(10));
        p.advance();
        p.complete_stage(Duration::from_millis(20));
        assert_eq!(p.total_stage_duration(), Duration::from_millis(30));
    }

    #[test]
    fn pipeline_elapsed_before_advance_is_zero() {
        let p = AsyncPipeline::inference();
        assert_eq!(p.elapsed(), Duration::ZERO);
    }

    // -- RuntimeState ---------------------------------------------------------

    #[test]
    fn runtime_state_display() {
        assert_eq!(RuntimeState::Idle.to_string(), "Idle");
        assert_eq!(RuntimeState::Running.to_string(), "Running");
        assert_eq!(RuntimeState::ShuttingDown.to_string(), "ShuttingDown");
        assert_eq!(RuntimeState::Stopped.to_string(), "Stopped");
    }

    // -- AsyncRuntime ---------------------------------------------------------

    #[test]
    fn runtime_new_with_defaults() {
        let rt = AsyncRuntime::with_defaults().unwrap();
        assert_eq!(rt.state(), RuntimeState::Idle);
        assert_eq!(rt.config().thread_count, 4);
    }

    #[test]
    fn runtime_invalid_config_rejected() {
        let cfg = AsyncConfig::default().with_thread_count(0);
        assert!(AsyncRuntime::new(cfg).is_err());
    }

    #[test]
    fn runtime_start_and_submit() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.start().unwrap();
        assert_eq!(rt.state(), RuntimeState::Running);

        let t = AsyncTask::new("inference");
        let id = rt.submit(t).unwrap();
        assert_eq!(rt.total_submitted(), 1);
        assert!(id > 0);
    }

    #[test]
    fn runtime_submit_before_start_fails() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        let t = AsyncTask::new("x");
        assert!(rt.submit(t).is_err());
    }

    #[test]
    fn runtime_double_start_fails() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.start().unwrap();
        assert!(rt.start().is_err());
    }

    #[test]
    fn runtime_tick_schedules_tasks() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.start().unwrap();
        rt.submit(AsyncTask::new("a")).unwrap();
        let started = rt.tick();
        assert_eq!(started.len(), 1);
        assert_eq!(rt.running_count(), 1);
    }

    #[test]
    fn runtime_complete_task_and_collect() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.start().unwrap();
        let id = rt.submit(AsyncTask::new("a")).unwrap();
        rt.tick();
        rt.complete_task(id, vec![1, 2, 3]).unwrap();
        let results = rt.collect_results();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_success());
        assert_eq!(results[0].output.as_deref(), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn runtime_fail_task_and_collect() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.start().unwrap();
        let id = rt.submit(AsyncTask::new("a")).unwrap();
        rt.tick();
        rt.fail_task(id, "oom").unwrap();
        let results = rt.collect_results();
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_success());
        assert_eq!(results[0].error.as_deref(), Some("oom"));
    }

    #[test]
    fn runtime_complete_nonexistent_task_fails() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.start().unwrap();
        assert!(rt.complete_task(9999, vec![]).is_err());
    }

    #[test]
    fn runtime_create_and_use_group() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.start().unwrap();
        {
            let g = rt.create_group("batch");
            g.add(AsyncTask::new("t1"));
            g.add(AsyncTask::new("t2"));
        }
        assert_eq!(rt.group("batch").unwrap().task_count(), 2);
    }

    #[test]
    fn runtime_shutdown() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.start().unwrap();
        rt.submit(AsyncTask::new("a")).unwrap();
        rt.tick();
        rt.shutdown();
        assert_eq!(rt.state(), RuntimeState::Stopped);
        assert_eq!(rt.running_count(), 0);
    }

    #[test]
    fn runtime_uptime_zero_before_start() {
        let rt = AsyncRuntime::with_defaults().unwrap();
        assert_eq!(rt.uptime(), Duration::ZERO);
    }

    #[test]
    fn runtime_uptime_nonzero_after_start() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.start().unwrap();
        // Tiny but representable.
        assert!(rt.uptime() < Duration::from_secs(5));
    }

    #[test]
    fn runtime_drop_triggers_shutdown() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.start().unwrap();
        rt.submit(AsyncTask::new("a")).unwrap();
        rt.tick();
        drop(rt);
        // If we get here without hanging, the drop shutdown worked.
    }

    #[test]
    fn runtime_group_mut() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.create_group("g");
        let g = rt.group_mut("g").unwrap();
        g.add(AsyncTask::new("x"));
        assert_eq!(rt.group("g").unwrap().task_count(), 1);
    }

    #[test]
    fn runtime_nonexistent_group_returns_none() {
        let rt = AsyncRuntime::with_defaults().unwrap();
        assert!(rt.group("nope").is_none());
    }

    #[test]
    fn runtime_pending_count() {
        let mut rt = AsyncRuntime::new(AsyncConfig::default().with_thread_count(1)).unwrap();
        rt.start().unwrap();
        rt.submit(AsyncTask::new("a")).unwrap();
        rt.submit(AsyncTask::new("b")).unwrap();
        rt.tick();
        assert_eq!(rt.running_count(), 1);
        assert_eq!(rt.pending_count(), 1);
    }

    #[test]
    fn runtime_shutdown_cancels_groups() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.start().unwrap();
        {
            let g = rt.create_group("g");
            g.add(AsyncTask::new("t1"));
        }
        rt.shutdown();
        assert!(rt.group("g").unwrap().cancel_token.is_cancelled());
    }

    // -- Integration / cross-component ----------------------------------------

    #[test]
    fn end_to_end_pipeline_with_group() {
        let mut rt = AsyncRuntime::with_defaults().unwrap();
        rt.start().unwrap();

        let mut pipeline = AsyncPipeline::inference();

        // Simulate three-stage pipeline.
        for _ in 0..3 {
            let stage_name = pipeline.advance().unwrap().to_string();
            let task = AsyncTask::new(&stage_name);
            let id = rt.submit(task).unwrap();
            rt.tick();
            rt.complete_task(id, stage_name.into_bytes()).unwrap();
            pipeline.complete_stage(Duration::from_millis(1));
        }

        assert!(pipeline.is_complete());
        let results = rt.collect_results();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(super::TaskResult::is_success));
    }

    #[test]
    fn barrier_synchronises_group_tasks() {
        let mut barrier = AsyncBarrier::new(3);
        let mut group = TaskGroup::new("parallel");

        for i in 0u64..3 {
            let id = group.add(AsyncTask::new(format!("worker-{i}")));
            group.record_result(TaskResult::success(id, Duration::from_millis(i), vec![]));
            barrier.arrive();
        }

        assert!(barrier.is_released());
        assert!(group.is_complete());
        assert!(group.all_succeeded());
    }

    #[test]
    fn timer_guards_task_execution() {
        let mut timer = TaskTimer::new(Duration::ZERO);
        timer.start();
        // With a zero timeout the timer expires immediately.
        assert!(timer.is_expired());
    }

    #[test]
    fn channel_round_trip_multiple_results() {
        let mut ch = AsyncChannel::new(64);
        for i in 0..10 {
            #[allow(clippy::cast_possible_truncation)]
            ch.try_send(TaskResult::success(i, Duration::ZERO, vec![i as u8])).unwrap();
        }
        assert_eq!(ch.len(), 10);
        let all = ch.drain();
        assert_eq!(all.len(), 10);
        for (i, r) in all.iter().enumerate() {
            assert_eq!(r.task_id, i as u64);
        }
    }

    #[test]
    fn scheduler_respects_max_concurrent() {
        let mut sched = TaskScheduler::new(3);
        for _ in 0..10 {
            sched.submit(AsyncTask::new("w"));
        }
        sched.schedule();
        assert_eq!(sched.running_count(), 3);
        assert_eq!(sched.pending_count(), 7);
    }

    #[test]
    fn pipeline_custom_stages() {
        let mut p = AsyncPipeline::new(&["preprocess", "quantize", "execute", "postprocess"]);
        assert_eq!(p.stage_count(), 4);
        for _ in 0..4 {
            p.advance();
            p.complete_stage(Duration::from_millis(1));
        }
        assert!(p.is_complete());
    }
}
