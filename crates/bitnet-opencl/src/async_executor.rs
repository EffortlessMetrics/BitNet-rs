//! Async GPU execution engine with multi-queue command management.
//!
//! Provides [`AsyncGpuExecutor`] for submitting and synchronizing GPU operations
//! across multiple command queues, with [`GpuFuture`] for async completion tracking
//! and double-buffering support for overlapped transfers.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::{GpuError, GpuResult};

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

static NEXT_OP_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_EVENT_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a submitted GPU operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(u64);

impl OpId {
    fn next() -> Self {
        Self(NEXT_OP_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for OpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Op({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// GpuEvent – synchronisation primitive
// ---------------------------------------------------------------------------

/// Wrapper around an `OpenCL` event (or mock equivalent) for synchronisation.
#[derive(Debug, Clone)]
pub struct GpuEvent {
    id: u64,
    state: Arc<Mutex<EventState>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventState {
    Pending,
    Complete,
    Error,
}

impl GpuEvent {
    /// Create a new pending event.
    pub fn new() -> Self {
        Self {
            id: NEXT_EVENT_ID.fetch_add(1, Ordering::Relaxed),
            state: Arc::new(Mutex::new(EventState::Pending)),
        }
    }

    /// Signal that the event has completed successfully.
    pub fn signal_complete(&self) {
        *self.state.lock().expect("event lock poisoned") = EventState::Complete;
    }

    /// Signal that the event has encountered an error.
    pub fn signal_error(&self) {
        *self.state.lock().expect("event lock poisoned") = EventState::Error;
    }

    /// Returns `true` if the event has completed (success **or** error).
    pub fn is_done(&self) -> bool {
        let s = *self.state.lock().expect("event lock poisoned");
        matches!(s, EventState::Complete | EventState::Error)
    }

    /// Returns `true` if the event completed successfully.
    pub fn is_complete(&self) -> bool {
        *self.state.lock().expect("event lock poisoned") == EventState::Complete
    }

    /// Returns `true` if the event ended with an error.
    pub fn is_error(&self) -> bool {
        *self.state.lock().expect("event lock poisoned") == EventState::Error
    }

    /// Unique id of this event (for diagnostics).
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }
}

impl Default for GpuEvent {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GpuFuture<T> – async completion handle
// ---------------------------------------------------------------------------

/// A future-like handle for an in-flight GPU operation.
///
/// Not a `std::future::Future` — call [`wait`](GpuFuture::wait) to block
/// until the result is available (the mock backend completes synchronously).
pub struct GpuFuture<T> {
    event: GpuEvent,
    value: Arc<Mutex<Option<GpuResult<T>>>>,
}

impl<T: Clone> GpuFuture<T> {
    /// Create a future that is already resolved with `value`.
    pub fn ready(value: T) -> Self {
        let event = GpuEvent::new();
        event.signal_complete();
        Self { event, value: Arc::new(Mutex::new(Some(Ok(value)))) }
    }

    /// Create a pending future; call [`resolve`](GpuFuture::resolve) later.
    pub fn pending() -> Self {
        Self { event: GpuEvent::new(), value: Arc::new(Mutex::new(None)) }
    }

    /// Resolve the future with a result.
    pub fn resolve(&self, result: GpuResult<T>) {
        let is_ok = result.is_ok();
        *self.value.lock().expect("future lock poisoned") = Some(result);
        if is_ok {
            self.event.signal_complete();
        } else {
            self.event.signal_error();
        }
    }

    /// Block until the result is available.
    pub fn wait(&self) -> GpuResult<T> {
        // Mock backend: result is already present.
        self.value
            .lock()
            .expect("future lock poisoned")
            .clone()
            .unwrap_or(Err(GpuError::OperationIncomplete))
    }

    /// Returns a reference to the underlying event.
    #[must_use]
    pub const fn event(&self) -> &GpuEvent {
        &self.event
    }
}

impl<T: fmt::Debug> fmt::Debug for GpuFuture<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuFuture")
            .field("event", &self.event)
            .field("done", &self.event.is_done())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// PipelineStage
// ---------------------------------------------------------------------------

/// Logical stage within the GPU execution pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    /// Host → Device data transfer.
    Transfer,
    /// GPU kernel computation.
    Compute,
    /// Device → Host read-back.
    Readback,
}

impl fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transfer => write!(f, "Transfer"),
            Self::Compute => write!(f, "Compute"),
            Self::Readback => write!(f, "Readback"),
        }
    }
}

// ---------------------------------------------------------------------------
// ExecutionPlan
// ---------------------------------------------------------------------------

/// A recorded sequence of GPU operations with dependency edges.
///
/// Build a plan, then submit it to [`AsyncGpuExecutor::execute_plan`].
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    ops: Vec<PlannedOp>,
}

#[derive(Debug, Clone)]
struct PlannedOp {
    stage: PipelineStage,
    label: String,
    depends_on: Vec<usize>,
}

impl ExecutionPlan {
    /// Create an empty execution plan.
    #[must_use]
    pub const fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Add an operation that depends on zero or more preceding operations
    /// (referenced by their indices in insertion order).
    pub fn add_op(
        &mut self,
        stage: PipelineStage,
        label: impl Into<String>,
        depends_on: &[usize],
    ) -> usize {
        let idx = self.ops.len();
        self.ops.push(PlannedOp { stage, label: label.into(), depends_on: depends_on.to_vec() });
        idx
    }

    /// Number of operations in the plan.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns `true` if the plan contains no operations.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// QueueSlot – internal per-queue state
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct QueueSlot {
    pending: VecDeque<SubmittedOp>,
}

#[derive(Debug)]
struct SubmittedOp {
    id: OpId,
    #[allow(dead_code)]
    stage: PipelineStage,
    #[allow(dead_code)]
    label: String,
    event: GpuEvent,
}

// ---------------------------------------------------------------------------
// AsyncGpuExecutor
// ---------------------------------------------------------------------------

/// Multi-queue GPU command executor with double-buffering support.
///
/// In **mock mode** (no `oneapi` feature) every operation completes
/// synchronously so that tests can run without real hardware.
pub struct AsyncGpuExecutor {
    /// One queue per pipeline stage for concurrent execution.
    queues: Vec<Mutex<QueueSlot>>,
    /// Which buffer index is active (0 or 1) for double-buffering.
    active_buffer: AtomicU64,
    /// Total number of operations submitted (lifetime counter).
    total_submitted: AtomicU64,
}

impl AsyncGpuExecutor {
    /// Create a new executor with `num_queues` command queues.
    ///
    /// Typically 3 queues are used (one per [`PipelineStage`]).
    #[must_use]
    pub fn new(num_queues: usize) -> Self {
        let num_queues = num_queues.max(1);
        let queues =
            (0..num_queues).map(|_| Mutex::new(QueueSlot { pending: VecDeque::new() })).collect();
        Self { queues, active_buffer: AtomicU64::new(0), total_submitted: AtomicU64::new(0) }
    }

    /// Number of command queues.
    #[must_use]
    pub const fn num_queues(&self) -> usize {
        self.queues.len()
    }

    /// Current active buffer index (0 or 1).
    #[must_use]
    pub fn active_buffer(&self) -> u64 {
        self.active_buffer.load(Ordering::Relaxed)
    }

    /// Swap the active double-buffer (0 ↔ 1).
    pub fn swap_buffer(&self) {
        let prev = self.active_buffer.load(Ordering::Relaxed);
        self.active_buffer.store(1 - prev, Ordering::Relaxed);
    }

    /// Total operations submitted over the executor's lifetime.
    #[must_use]
    pub fn total_submitted(&self) -> u64 {
        self.total_submitted.load(Ordering::Relaxed)
    }

    // -- submission helpers ------------------------------------------------

    /// Submit a kernel execution to the given queue.
    pub fn submit_kernel(
        &self,
        queue: usize,
        label: impl Into<String>,
    ) -> GpuResult<GpuFuture<()>> {
        self.submit(queue, PipelineStage::Compute, label)
    }

    /// Submit a host→device transfer.
    pub fn submit_transfer(
        &self,
        queue: usize,
        label: impl Into<String>,
    ) -> GpuResult<GpuFuture<()>> {
        self.submit(queue, PipelineStage::Transfer, label)
    }

    /// Submit a device→host readback.
    pub fn submit_readback(
        &self,
        queue: usize,
        label: impl Into<String>,
    ) -> GpuResult<GpuFuture<()>> {
        self.submit(queue, PipelineStage::Readback, label)
    }

    /// Generic submission to a specific queue and stage.
    pub fn submit(
        &self,
        queue: usize,
        stage: PipelineStage,
        label: impl Into<String>,
    ) -> GpuResult<GpuFuture<()>> {
        if queue >= self.queues.len() {
            return Err(GpuError::InvalidQueue { requested: queue, available: self.queues.len() });
        }

        let op_id = OpId::next();
        let label = label.into();
        let event = GpuEvent::new();

        // Mock backend: complete immediately.
        mock_execute(&event);

        let future = GpuFuture::ready(());

        {
            let mut q = self.queues[queue].lock().expect("queue lock poisoned");
            q.pending.push_back(SubmittedOp { id: op_id, stage, label, event });
        }

        self.total_submitted.fetch_add(1, Ordering::Relaxed);
        log::trace!("submitted {op_id} on queue {queue} ({stage})");

        Ok(future)
    }

    /// Execute all operations in an [`ExecutionPlan`] respecting
    /// dependency order.  Returns one [`GpuEvent`] per operation.
    pub fn execute_plan(&self, plan: &ExecutionPlan) -> GpuResult<Vec<GpuEvent>> {
        let mut events: Vec<GpuEvent> = Vec::with_capacity(plan.ops.len());

        for (i, op) in plan.ops.iter().enumerate() {
            // Validate dependency indices.
            for &dep in &op.depends_on {
                if dep >= i {
                    return Err(GpuError::InvalidDependency { op_index: i, dep_index: dep });
                }
            }

            let queue = self.queue_for_stage(op.stage);
            let future = self.submit(queue, op.stage, &op.label)?;
            events.push(future.event().clone());
        }

        Ok(events)
    }

    /// Wait for **all** pending operations across every queue to finish.
    pub fn wait_all(&self) -> GpuResult<()> {
        for q in &self.queues {
            let q = q.lock().expect("queue lock poisoned");
            for op in &q.pending {
                if op.event.is_error() {
                    return Err(GpuError::OperationFailed { op: op.id.to_string() });
                }
            }
        }
        Ok(())
    }

    /// Drain all completed operations from every queue.
    pub fn flush(&self) -> usize {
        let mut flushed = 0usize;
        for q in &self.queues {
            let mut q = q.lock().expect("queue lock poisoned");
            let before = q.pending.len();
            q.pending.retain(|op| !op.event.is_done());
            flushed += before - q.pending.len();
        }
        log::trace!("flushed {flushed} completed operations");
        flushed
    }

    /// Count of operations still pending across all queues.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queues.iter().map(|q| q.lock().expect("queue lock poisoned").pending.len()).sum()
    }

    // -- internal helpers ---------------------------------------------------

    /// Map a pipeline stage to a default queue index.
    const fn queue_for_stage(&self, stage: PipelineStage) -> usize {
        let idx = match stage {
            PipelineStage::Transfer => 0,
            PipelineStage::Compute => 1,
            PipelineStage::Readback => 2,
        };
        idx % self.queues.len()
    }
}

impl fmt::Debug for AsyncGpuExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncGpuExecutor")
            .field("num_queues", &self.queues.len())
            .field("active_buffer", &self.active_buffer())
            .field("total_submitted", &self.total_submitted.load(Ordering::Relaxed))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Mock backend (always compiled; real OpenCL behind `oneapi` feature)
// ---------------------------------------------------------------------------

/// Mock: immediately mark the event as complete.
fn mock_execute(event: &GpuEvent) {
    #[cfg(not(feature = "oneapi"))]
    {
        event.signal_complete();
    }

    #[cfg(feature = "oneapi")]
    {
        // Real OpenCL path would enqueue the command and attach
        // a callback to signal the event on completion.
        let _ = event;
        todo!("real OpenCL dispatch not yet implemented");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_id_is_unique() {
        let a = OpId::next();
        let b = OpId::next();
        assert_ne!(a, b);
    }

    #[test]
    fn event_lifecycle() {
        let e = GpuEvent::new();
        assert!(!e.is_done());
        e.signal_complete();
        assert!(e.is_done());
        assert!(e.is_complete());
    }

    #[test]
    fn future_ready() {
        let f = GpuFuture::ready(42u32);
        assert!(f.event().is_complete());
        assert_eq!(f.wait().unwrap(), 42);
    }

    #[test]
    fn future_pending_then_resolve() {
        let f = GpuFuture::<i32>::pending();
        assert!(!f.event().is_done());
        f.resolve(Ok(7));
        assert!(f.event().is_complete());
        assert_eq!(f.wait().unwrap(), 7);
    }

    #[test]
    fn executor_submit_and_wait() {
        let exec = AsyncGpuExecutor::new(3);
        let fut = exec.submit_kernel(1, "matmul").unwrap();
        assert!(fut.event().is_complete());
        exec.wait_all().unwrap();
    }

    #[test]
    fn executor_invalid_queue() {
        let exec = AsyncGpuExecutor::new(2);
        let res = exec.submit_kernel(5, "bad");
        assert!(res.is_err());
    }

    #[test]
    fn double_buffer_swap() {
        let exec = AsyncGpuExecutor::new(1);
        assert_eq!(exec.active_buffer(), 0);
        exec.swap_buffer();
        assert_eq!(exec.active_buffer(), 1);
        exec.swap_buffer();
        assert_eq!(exec.active_buffer(), 0);
    }

    #[test]
    fn flush_drains_completed() {
        let exec = AsyncGpuExecutor::new(1);
        exec.submit_kernel(0, "a").unwrap();
        exec.submit_kernel(0, "b").unwrap();
        assert_eq!(exec.pending_count(), 2);
        let flushed = exec.flush();
        assert_eq!(flushed, 2);
        assert_eq!(exec.pending_count(), 0);
    }

    #[test]
    fn plan_empty_is_noop() {
        let exec = AsyncGpuExecutor::new(3);
        let plan = ExecutionPlan::new();
        let events = exec.execute_plan(&plan).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn plan_three_stages_in_order() {
        let exec = AsyncGpuExecutor::new(3);
        let mut plan = ExecutionPlan::new();
        let t = plan.add_op(PipelineStage::Transfer, "upload", &[]);
        let c = plan.add_op(PipelineStage::Compute, "matmul", &[t]);
        let _r = plan.add_op(PipelineStage::Readback, "download", &[c]);

        let events = exec.execute_plan(&plan).unwrap();
        assert_eq!(events.len(), 3);
        for e in &events {
            assert!(e.is_complete());
        }
    }
}
