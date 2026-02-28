//! Asynchronous OpenCL kernel execution with event-based synchronization.
//!
//! This module provides [`AsyncKernelExecution`] for non-blocking kernel
//! dispatch, callback registration, double-buffered data transfer, and
//! dependency-tracked kernel pipelines via OpenCL event wait lists.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Unique identifier for an enqueued kernel.
pub type KernelId = u64;

/// Simulated OpenCL event status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventStatus {
    /// The command has been enqueued but not started.
    Queued,
    /// The command is currently executing on the device.
    Running,
    /// The command completed successfully.
    Complete,
    /// The command failed with a vendor error code.
    Error(i32),
}

/// A lightweight handle representing an OpenCL event.
///
/// In a real implementation this would wrap `cl_event`; here we use a
/// status + optional timing so the async-execution logic can be tested
/// without an actual OpenCL runtime.
#[derive(Debug, Clone)]
pub struct ClEvent {
    id: KernelId,
    status: Arc<Mutex<EventStatus>>,
    /// Wall-clock time when the event was created.
    created_at: Instant,
    /// Wall-clock time when the event completed (if any).
    completed_at: Arc<Mutex<Option<Instant>>>,
}

impl ClEvent {
    /// Create a new event in `Queued` state.
    pub fn new(id: KernelId) -> Self {
        Self {
            id,
            status: Arc::new(Mutex::new(EventStatus::Queued)),
            created_at: Instant::now(),
            completed_at: Arc::new(Mutex::new(None)),
        }
    }

    /// Return the kernel id associated with this event.
    pub fn id(&self) -> KernelId {
        self.id
    }

    /// Current status of the event.
    pub fn status(&self) -> EventStatus {
        *self.status.lock().unwrap()
    }

    /// Mark the event as running.
    pub fn set_running(&self) {
        *self.status.lock().unwrap() = EventStatus::Running;
    }

    /// Mark the event as complete.
    pub fn set_complete(&self) {
        *self.status.lock().unwrap() = EventStatus::Complete;
        *self.completed_at.lock().unwrap() = Some(Instant::now());
    }

    /// Mark the event as failed.
    pub fn set_error(&self, code: i32) {
        *self.status.lock().unwrap() = EventStatus::Error(code);
        *self.completed_at.lock().unwrap() = Some(Instant::now());
    }

    /// Returns `true` when the event has reached a terminal state.
    pub fn is_finished(&self) -> bool {
        matches!(self.status(), EventStatus::Complete | EventStatus::Error(_))
    }

    /// Elapsed time since event creation.
    pub fn elapsed(&self) -> Duration {
        self.completed_at
            .lock()
            .unwrap()
            .map(|t| t.duration_since(self.created_at))
            .unwrap_or_else(|| self.created_at.elapsed())
    }
}

/// A registered callback that fires when a kernel completes.
type CompletionCallback = Box<dyn FnOnce(KernelId, EventStatus) + Send + 'static>;

/// Async kernel execution engine built on OpenCL events.
///
/// Manages non-blocking kernel dispatches, registers completion callbacks,
/// and tracks dependencies between kernels via event wait lists.
pub struct AsyncKernelExecution {
    next_id: KernelId,
    /// All events indexed by kernel id.
    events: HashMap<KernelId, ClEvent>,
    /// Callbacks pending for each kernel id.
    callbacks: HashMap<KernelId, Vec<CompletionCallback>>,
    /// Dependency graph: `deps[k]` = set of kernel ids that `k` waits on.
    deps: HashMap<KernelId, Vec<KernelId>>,
}

impl AsyncKernelExecution {
    /// Create a new execution context.
    pub fn new() -> Self {
        Self {
            next_id: 1,
            events: HashMap::new(),
            callbacks: HashMap::new(),
            deps: HashMap::new(),
        }
    }

    /// Enqueue a kernel with no dependencies, returning its event handle.
    pub fn enqueue(&mut self, _kernel_name: &str) -> ClEvent {
        self.enqueue_after(_kernel_name, &[])
    }

    /// Enqueue a kernel that waits on the listed events before executing.
    pub fn enqueue_after(
        &mut self,
        _kernel_name: &str,
        wait_list: &[KernelId],
    ) -> ClEvent {
        let id = self.next_id;
        self.next_id += 1;

        let event = ClEvent::new(id);
        self.events.insert(id, event.clone());
        self.deps.insert(id, wait_list.to_vec());
        event
    }

    /// Register a callback that fires when `kernel_id` completes.
    pub fn on_complete<F>(&mut self, kernel_id: KernelId, cb: F)
    where
        F: FnOnce(KernelId, EventStatus) + Send + 'static,
    {
        self.callbacks
            .entry(kernel_id)
            .or_default()
            .push(Box::new(cb));
    }

    /// Check whether all dependencies for `kernel_id` are satisfied.
    pub fn deps_satisfied(&self, kernel_id: KernelId) -> bool {
        self.deps
            .get(&kernel_id)
            .map(|dep_ids| {
                dep_ids.iter().all(|dep| {
                    self.events
                        .get(dep)
                        .is_some_and(|e| e.status() == EventStatus::Complete)
                })
            })
            .unwrap_or(true)
    }

    /// Simulate completing a kernel: set status, fire callbacks, return them.
    pub fn complete_kernel(&mut self, kernel_id: KernelId) -> bool {
        let Some(event) = self.events.get(&kernel_id) else {
            return false;
        };
        event.set_running();
        event.set_complete();

        if let Some(cbs) = self.callbacks.remove(&kernel_id) {
            for cb in cbs {
                cb(kernel_id, EventStatus::Complete);
            }
        }
        true
    }

    /// Simulate a kernel failure.
    pub fn fail_kernel(&mut self, kernel_id: KernelId, code: i32) -> bool {
        let Some(event) = self.events.get(&kernel_id) else {
            return false;
        };
        event.set_error(code);

        if let Some(cbs) = self.callbacks.remove(&kernel_id) {
            for cb in cbs {
                cb(kernel_id, EventStatus::Error(code));
            }
        }
        true
    }

    /// Get an event by kernel id.
    pub fn event(&self, kernel_id: KernelId) -> Option<&ClEvent> {
        self.events.get(&kernel_id)
    }

    /// Return all kernel ids that are ready to execute (deps satisfied, still
    /// queued).
    pub fn ready_kernels(&self) -> Vec<KernelId> {
        self.events
            .iter()
            .filter(|(_, ev)| ev.status() == EventStatus::Queued)
            .filter(|(id, _)| self.deps_satisfied(**id))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Number of pending (non-finished) kernels.
    pub fn pending_count(&self) -> usize {
        self.events.values().filter(|e| !e.is_finished()).count()
    }
}

impl Default for AsyncKernelExecution {
    fn default() -> Self {
        Self::new()
    }
}

/// Double-buffer manager for overlapping compute and data transfer.
///
/// Holds two buffers (A/B). While one is being consumed by a compute
/// kernel, the other can receive the next chunk of data via async DMA.
pub struct DoubleBuffer {
    buf_a: Vec<u8>,
    buf_b: Vec<u8>,
    /// Which buffer is currently the "active" compute buffer (0 = A, 1 = B).
    active: usize,
    /// Event tracking the in-flight transfer (if any).
    transfer_event: Option<ClEvent>,
    /// Event tracking the in-flight compute (if any).
    compute_event: Option<ClEvent>,
}

impl DoubleBuffer {
    /// Create a double buffer with the given per-buffer capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buf_a: vec![0u8; capacity],
            buf_b: vec![0u8; capacity],
            active: 0,
            transfer_event: None,
            compute_event: None,
        }
    }

    /// Return a mutable reference to the buffer currently used for transfers.
    pub fn transfer_buf_mut(&mut self) -> &mut Vec<u8> {
        if self.active == 0 { &mut self.buf_b } else { &mut self.buf_a }
    }

    /// Return a reference to the buffer currently used for compute.
    pub fn compute_buf(&self) -> &Vec<u8> {
        if self.active == 0 { &self.buf_a } else { &self.buf_b }
    }

    /// Swap active / transfer roles. Call after both events are done.
    pub fn swap(&mut self) {
        self.active = 1 - self.active;
        self.transfer_event = None;
        self.compute_event = None;
    }

    /// Attach the transfer event for the current staging buffer.
    pub fn set_transfer_event(&mut self, event: ClEvent) {
        self.transfer_event = Some(event);
    }

    /// Attach the compute event for the current active buffer.
    pub fn set_compute_event(&mut self, event: ClEvent) {
        self.compute_event = Some(event);
    }

    /// Returns `true` when both the transfer and compute events have completed.
    pub fn both_complete(&self) -> bool {
        let t_done = self
            .transfer_event
            .as_ref()
            .is_none_or(|e| e.is_finished());
        let c_done = self
            .compute_event
            .as_ref()
            .is_none_or(|e| e.is_finished());
        t_done && c_done
    }

    /// Per-buffer capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.buf_a.len()
    }
}

/// A linear pipeline of kernels with automatic dependency chaining.
pub struct KernelPipeline {
    exec: AsyncKernelExecution,
    /// Ordered list of (kernel_name, event) pairs.
    stages: Vec<(String, ClEvent)>,
}

impl KernelPipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            exec: AsyncKernelExecution::new(),
            stages: Vec::new(),
        }
    }

    /// Add a stage to the pipeline. It will automatically depend on the
    /// previous stage (if any).
    pub fn add_stage(&mut self, kernel_name: &str) -> KernelId {
        let wait_list: Vec<KernelId> = self
            .stages
            .last()
            .map(|(_, ev)| vec![ev.id()])
            .unwrap_or_default();
        let event = self.exec.enqueue_after(kernel_name, &wait_list);
        let id = event.id();
        self.stages.push((kernel_name.to_string(), event));
        id
    }

    /// Number of stages in the pipeline.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Access the underlying execution context.
    pub fn exec(&self) -> &AsyncKernelExecution {
        &self.exec
    }

    /// Mutable access to the underlying execution context (for completing kernels).
    pub fn exec_mut(&mut self) -> &mut AsyncKernelExecution {
        &mut self.exec
    }

    /// Get the kernel ids in pipeline order.
    pub fn kernel_ids(&self) -> Vec<KernelId> {
        self.stages.iter().map(|(_, ev)| ev.id()).collect()
    }

    /// Complete all stages in order, returning `true` if all succeeded.
    pub fn run_all(&mut self) -> bool {
        let ids = self.kernel_ids();
        for id in ids {
            if !self.exec.complete_kernel(id) {
                return false;
            }
        }
        true
    }
}

impl Default for KernelPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn test_enqueue_returns_queued_event() {
        let mut exec = AsyncKernelExecution::new();
        let ev = exec.enqueue("matmul");
        assert_eq!(ev.status(), EventStatus::Queued);
        assert!(!ev.is_finished());
    }

    #[test]
    fn test_complete_kernel_sets_status() {
        let mut exec = AsyncKernelExecution::new();
        let ev = exec.enqueue("add");
        let id = ev.id();
        exec.complete_kernel(id);
        assert_eq!(ev.status(), EventStatus::Complete);
        assert!(ev.is_finished());
    }

    #[test]
    fn test_fail_kernel_sets_error() {
        let mut exec = AsyncKernelExecution::new();
        let ev = exec.enqueue("bad_kernel");
        let id = ev.id();
        exec.fail_kernel(id, -5);
        assert_eq!(ev.status(), EventStatus::Error(-5));
        assert!(ev.is_finished());
    }

    #[test]
    fn test_callback_fires_on_complete() {
        let mut exec = AsyncKernelExecution::new();
        let ev = exec.enqueue("relu");
        let id = ev.id();

        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = fired.clone();
        exec.on_complete(id, move |_kid, status| {
            assert_eq!(status, EventStatus::Complete);
            fired_clone.store(true, Ordering::SeqCst);
        });

        exec.complete_kernel(id);
        assert!(fired.load(Ordering::SeqCst));
    }

    #[test]
    fn test_callback_fires_on_failure() {
        let mut exec = AsyncKernelExecution::new();
        let ev = exec.enqueue("oops");
        let id = ev.id();

        let got_code = Arc::new(Mutex::new(0i32));
        let gc = got_code.clone();
        exec.on_complete(id, move |_kid, status| {
            if let EventStatus::Error(c) = status {
                *gc.lock().unwrap() = c;
            }
        });

        exec.fail_kernel(id, -42);
        assert_eq!(*got_code.lock().unwrap(), -42);
    }

    #[test]
    fn test_dependency_blocks_readiness() {
        let mut exec = AsyncKernelExecution::new();
        let ev_a = exec.enqueue("stage_1");
        let id_a = ev_a.id();
        let ev_b = exec.enqueue_after("stage_2", &[id_a]);
        let id_b = ev_b.id();

        // stage_2 should NOT be ready while stage_1 is queued.
        assert!(!exec.deps_satisfied(id_b));
        assert!(exec.deps_satisfied(id_a)); // no deps

        exec.complete_kernel(id_a);
        assert!(exec.deps_satisfied(id_b));
    }

    #[test]
    fn test_ready_kernels_respects_deps() {
        let mut exec = AsyncKernelExecution::new();
        let ev_a = exec.enqueue("a");
        let id_a = ev_a.id();
        let _ev_b = exec.enqueue_after("b", &[id_a]);

        let ready = exec.ready_kernels();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], id_a);
    }

    #[test]
    fn test_pending_count() {
        let mut exec = AsyncKernelExecution::new();
        let ev1 = exec.enqueue("k1");
        let _ev2 = exec.enqueue("k2");
        assert_eq!(exec.pending_count(), 2);

        exec.complete_kernel(ev1.id());
        assert_eq!(exec.pending_count(), 1);
    }

    #[test]
    fn test_double_buffer_swap() {
        let mut db = DoubleBuffer::new(64);
        // Initially active=0 so compute uses A, transfer uses B
        db.transfer_buf_mut()[0] = 0xAA;
        assert_eq!(db.compute_buf()[0], 0);

        db.swap();
        // After swap, active=1 so compute uses B, transfer uses A
        assert_eq!(db.compute_buf()[0], 0xAA);
    }

    #[test]
    fn test_double_buffer_both_complete() {
        let mut db = DoubleBuffer::new(32);
        assert!(db.both_complete()); // no events attached

        let t_ev = ClEvent::new(100);
        let c_ev = ClEvent::new(101);
        db.set_transfer_event(t_ev.clone());
        db.set_compute_event(c_ev.clone());
        assert!(!db.both_complete());

        t_ev.set_complete();
        assert!(!db.both_complete());

        c_ev.set_complete();
        assert!(db.both_complete());
    }

    #[test]
    fn test_pipeline_dependency_chain() {
        let mut pipe = KernelPipeline::new();
        let k1 = pipe.add_stage("load");
        let k2 = pipe.add_stage("compute");
        let k3 = pipe.add_stage("store");

        assert_eq!(pipe.stage_count(), 3);

        // k1 has no deps, k2 depends on k1, k3 depends on k2
        assert!(pipe.exec().deps_satisfied(k1));
        assert!(!pipe.exec().deps_satisfied(k2));
        assert!(!pipe.exec().deps_satisfied(k3));

        pipe.exec_mut().complete_kernel(k1);
        assert!(pipe.exec().deps_satisfied(k2));
        assert!(!pipe.exec().deps_satisfied(k3));

        pipe.exec_mut().complete_kernel(k2);
        assert!(pipe.exec().deps_satisfied(k3));
    }

    #[test]
    fn test_pipeline_run_all() {
        let mut pipe = KernelPipeline::new();
        pipe.add_stage("s1");
        pipe.add_stage("s2");
        pipe.add_stage("s3");
        assert!(pipe.run_all());
        assert_eq!(pipe.exec().pending_count(), 0);
    }

    #[test]
    fn test_event_elapsed_after_complete() {
        let ev = ClEvent::new(42);
        // Small sleep so elapsed > 0
        std::thread::sleep(Duration::from_millis(1));
        ev.set_complete();
        let d = ev.elapsed();
        assert!(d >= Duration::from_millis(1));
    }
}
