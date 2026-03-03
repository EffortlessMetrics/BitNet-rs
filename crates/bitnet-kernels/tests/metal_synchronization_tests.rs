#![cfg(feature = "cpu")]
#![allow(clippy::needless_range_loop)]
//! Metal GPU synchronization and barrier pattern tests.
//!
//! Validates Metal synchronization primitives using pure-Rust models:
//! command buffer ordering, fences, events, memory barriers,
//! resource hazard tracking, multi-queue coordination, and
//! double/triple buffering patterns.
//!
//! All tests run with `--features cpu` — no Metal runtime required.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

// ═══════════════════════════════════════════════════════════════════════
// Model types for Metal synchronization concepts
// ═══════════════════════════════════════════════════════════════════════

/// Simulated command buffer status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandBufferStatus {
    NotEnqueued,
    Enqueued,
    Committed,
    Scheduled,
    Completed,
    Error,
}

/// A modeled Metal command buffer.
#[derive(Debug)]
struct CommandBuffer {
    id: u64,
    status: CommandBufferStatus,
    label: String,
    execution_order: Option<u64>,
    dependencies: Vec<u64>,
    writes: Vec<String>,
    reads: Vec<String>,
}

impl CommandBuffer {
    fn new(id: u64, label: &str) -> Self {
        Self {
            id,
            status: CommandBufferStatus::NotEnqueued,
            label: label.to_string(),
            execution_order: None,
            dependencies: Vec::new(),
            writes: Vec::new(),
            reads: Vec::new(),
        }
    }

    fn enqueue(&mut self) {
        assert_eq!(self.status, CommandBufferStatus::NotEnqueued);
        self.status = CommandBufferStatus::Enqueued;
    }

    fn commit(&mut self) {
        assert!(
            self.status == CommandBufferStatus::Enqueued
                || self.status == CommandBufferStatus::NotEnqueued
        );
        self.status = CommandBufferStatus::Committed;
    }

    fn schedule(&mut self) {
        assert_eq!(self.status, CommandBufferStatus::Committed);
        self.status = CommandBufferStatus::Scheduled;
    }

    fn complete(&mut self, order: u64) {
        assert_eq!(self.status, CommandBufferStatus::Scheduled);
        self.status = CommandBufferStatus::Completed;
        self.execution_order = Some(order);
    }

    fn fail(&mut self) {
        self.status = CommandBufferStatus::Error;
    }

    fn add_dependency(&mut self, dep_id: u64) {
        self.dependencies.push(dep_id);
    }

    fn add_write(&mut self, resource: &str) {
        self.writes.push(resource.to_string());
    }

    fn add_read(&mut self, resource: &str) {
        self.reads.push(resource.to_string());
    }
}

/// Command queue that processes command buffers in submission order.
#[derive(Debug)]
struct CommandQueue {
    label: String,
    priority: QueuePriority,
    buffers: VecDeque<u64>,
    next_order: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum QueuePriority {
    Low,
    Normal,
    High,
}

impl CommandQueue {
    fn new(label: &str, priority: QueuePriority) -> Self {
        Self {
            label: label.to_string(),
            priority,
            buffers: VecDeque::new(),
            next_order: AtomicU64::new(0),
        }
    }

    fn submit(&mut self, buffer_id: u64) {
        self.buffers.push_back(buffer_id);
    }

    fn next_execution_order(&self) -> u64 {
        self.next_order.fetch_add(1, Ordering::SeqCst)
    }

    fn pending_count(&self) -> usize {
        self.buffers.len()
    }

    fn drain_all(&mut self) -> Vec<u64> {
        self.buffers.drain(..).collect()
    }
}

/// Fence for cross-encoder / cross-queue synchronization.
#[derive(Debug)]
struct Fence {
    id: u64,
    signaled: bool,
    signal_value: u64,
    label: String,
}

impl Fence {
    fn new(id: u64, label: &str) -> Self {
        Self { id, signaled: false, signal_value: 0, label: label.to_string() }
    }

    fn signal(&mut self, value: u64) {
        self.signaled = true;
        self.signal_value = value;
    }

    fn wait(&self, expected: u64) -> bool {
        self.signaled && self.signal_value >= expected
    }

    fn reset(&mut self) {
        self.signaled = false;
        self.signal_value = 0;
    }
}

/// GPU event for fine-grained synchronization.
#[derive(Debug)]
struct GpuEvent {
    id: u64,
    signaled_value: u64,
    label: String,
}

impl GpuEvent {
    fn new(id: u64, label: &str) -> Self {
        Self { id, signaled_value: 0, label: label.to_string() }
    }

    fn signal(&mut self, value: u64) {
        self.signaled_value = self.signaled_value.max(value);
    }

    fn wait(&self, value: u64) -> bool {
        self.signaled_value >= value
    }
}

/// Resource access type for hazard tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccessType {
    Read,
    Write,
    ReadWrite,
}

/// Memory barrier scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BarrierScope {
    Buffers,
    Textures,
    RenderTargets,
    All,
}

/// Resource state for hazard tracking.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ResourceState {
    name: String,
    last_access: AccessType,
    last_writer_cmd: Option<u64>,
    last_reader_cmds: Vec<u64>,
    barrier_pending: bool,
}

impl ResourceState {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            last_access: AccessType::Read,
            last_writer_cmd: None,
            last_reader_cmds: Vec::new(),
            barrier_pending: false,
        }
    }

    fn record_write(&mut self, cmd_id: u64) -> bool {
        let needs_barrier = !self.last_reader_cmds.is_empty() || self.last_writer_cmd.is_some();
        self.last_access = AccessType::Write;
        self.last_writer_cmd = Some(cmd_id);
        self.last_reader_cmds.clear();
        self.barrier_pending = needs_barrier;
        needs_barrier
    }

    fn record_read(&mut self, cmd_id: u64) -> bool {
        let needs_barrier = self.last_writer_cmd.is_some() && self.last_access == AccessType::Write;
        self.last_access = AccessType::Read;
        self.last_reader_cmds.push(cmd_id);
        if needs_barrier {
            self.barrier_pending = true;
        }
        needs_barrier
    }

    fn clear_barrier(&mut self) {
        self.barrier_pending = false;
    }
}

/// Hazard tracking mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HazardTrackingMode {
    Automatic,
    Manual,
    Untracked,
}

/// Memory barrier descriptor.
#[derive(Debug, Clone)]
struct MemoryBarrier {
    scope: BarrierScope,
    after_stages: &'static str,
    before_stages: &'static str,
    resources: Vec<String>,
}

impl MemoryBarrier {
    fn buffer_barrier(resources: Vec<String>) -> Self {
        Self {
            scope: BarrierScope::Buffers,
            after_stages: "compute",
            before_stages: "compute",
            resources,
        }
    }

    fn texture_barrier(resources: Vec<String>) -> Self {
        Self {
            scope: BarrierScope::Textures,
            after_stages: "fragment",
            before_stages: "compute",
            resources,
        }
    }

    fn full_barrier() -> Self {
        Self {
            scope: BarrierScope::All,
            after_stages: "all",
            before_stages: "all",
            resources: Vec::new(),
        }
    }
}

/// Ring buffer slot for double/triple buffering.
#[derive(Debug)]
struct BufferSlot {
    index: usize,
    in_flight: bool,
    frame_id: u64,
    data: Vec<f32>,
}

impl BufferSlot {
    fn new(index: usize, capacity: usize) -> Self {
        Self { index, in_flight: false, frame_id: 0, data: vec![0.0; capacity] }
    }
}

/// Ring buffer manager for N-buffering patterns.
#[derive(Debug)]
struct RingBuffer {
    slots: Vec<BufferSlot>,
    current: usize,
    frame_counter: u64,
    buffer_count: usize,
}

impl RingBuffer {
    fn new(count: usize, capacity: usize) -> Self {
        let slots = (0..count).map(|i| BufferSlot::new(i, capacity)).collect();
        Self { slots, current: 0, frame_counter: 0, buffer_count: count }
    }

    fn acquire(&mut self) -> Option<usize> {
        let idx = self.current;
        if self.slots[idx].in_flight {
            return None;
        }
        self.slots[idx].in_flight = true;
        self.slots[idx].frame_id = self.frame_counter;
        self.frame_counter += 1;
        self.current = (self.current + 1) % self.buffer_count;
        Some(idx)
    }

    fn release(&mut self, index: usize) {
        assert!(index < self.buffer_count);
        self.slots[index].in_flight = false;
    }

    fn in_flight_count(&self) -> usize {
        self.slots.iter().filter(|s| s.in_flight).count()
    }

    fn available_count(&self) -> usize {
        self.slots.iter().filter(|s| !s.in_flight).count()
    }
}

/// Completion handler record for ordering verification.
#[derive(Debug)]
struct CompletionRecord {
    buffer_id: u64,
    timestamp: u64,
}

/// Simple synchronization coordinator.
struct SyncCoordinator {
    completed: Vec<CompletionRecord>,
    fences: Vec<Fence>,
    events: Vec<GpuEvent>,
    barriers: Vec<MemoryBarrier>,
    resources: Vec<ResourceState>,
    clock: AtomicU64,
}

impl SyncCoordinator {
    fn new() -> Self {
        Self {
            completed: Vec::new(),
            fences: Vec::new(),
            events: Vec::new(),
            barriers: Vec::new(),
            resources: Vec::new(),
            clock: AtomicU64::new(0),
        }
    }

    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::SeqCst)
    }

    fn record_completion(&mut self, buffer_id: u64) {
        let ts = self.tick();
        self.completed.push(CompletionRecord { buffer_id, timestamp: ts });
    }

    fn add_fence(&mut self, id: u64, label: &str) -> usize {
        self.fences.push(Fence::new(id, label));
        self.fences.len() - 1
    }

    fn add_event(&mut self, id: u64, label: &str) -> usize {
        self.events.push(GpuEvent::new(id, label));
        self.events.len() - 1
    }

    fn add_resource(&mut self, name: &str) -> usize {
        self.resources.push(ResourceState::new(name));
        self.resources.len() - 1
    }

    fn insert_barrier(&mut self, barrier: MemoryBarrier) {
        self.barriers.push(barrier);
    }

    fn completion_order(&self) -> Vec<u64> {
        let mut sorted = self.completed.clone();
        sorted.sort_by_key(|r| r.timestamp);
        sorted.iter().map(|r| r.buffer_id).collect()
    }
}

impl Clone for CompletionRecord {
    fn clone(&self) -> Self {
        Self { buffer_id: self.buffer_id, timestamp: self.timestamp }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 1. Command buffer synchronization (20+ tests)
// ═══════════════════════════════════════════════════════════════════════

mod command_buffer_sync {
    use super::*;

    #[test]
    fn command_buffer_initial_state() {
        let cb = CommandBuffer::new(1, "init");
        assert_eq!(cb.status, CommandBufferStatus::NotEnqueued);
        assert!(cb.execution_order.is_none());
        assert!(cb.dependencies.is_empty());
    }

    #[test]
    fn command_buffer_enqueue_transition() {
        let mut cb = CommandBuffer::new(1, "enqueue");
        cb.enqueue();
        assert_eq!(cb.status, CommandBufferStatus::Enqueued);
    }

    #[test]
    fn command_buffer_commit_from_enqueued() {
        let mut cb = CommandBuffer::new(1, "commit");
        cb.enqueue();
        cb.commit();
        assert_eq!(cb.status, CommandBufferStatus::Committed);
    }

    #[test]
    fn command_buffer_commit_from_not_enqueued() {
        let mut cb = CommandBuffer::new(1, "direct-commit");
        cb.commit();
        assert_eq!(cb.status, CommandBufferStatus::Committed);
    }

    #[test]
    fn command_buffer_full_lifecycle() {
        let mut cb = CommandBuffer::new(1, "lifecycle");
        cb.enqueue();
        cb.commit();
        cb.schedule();
        cb.complete(0);
        assert_eq!(cb.status, CommandBufferStatus::Completed);
        assert_eq!(cb.execution_order, Some(0));
    }

    #[test]
    fn sequential_command_buffer_execution_order() {
        let mut queue = CommandQueue::new("serial", QueuePriority::Normal);
        let mut buffers: Vec<CommandBuffer> =
            (0..5).map(|i| CommandBuffer::new(i, &format!("cb-{i}"))).collect();

        for buf in &mut buffers {
            buf.enqueue();
            buf.commit();
            queue.submit(buf.id);
        }

        let drained = queue.drain_all();
        for (i, &buf_id) in drained.iter().enumerate() {
            buffers[buf_id as usize].schedule();
            let order = i as u64;
            buffers[buf_id as usize].complete(order);
        }

        for (i, buf) in buffers.iter().enumerate() {
            assert_eq!(buf.execution_order, Some(i as u64));
        }
    }

    #[test]
    fn completion_handler_ordering() {
        let mut coord = SyncCoordinator::new();
        for i in 0..4 {
            coord.record_completion(i);
        }
        assert_eq!(coord.completion_order(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn completion_handlers_preserve_submission_order() {
        let mut coord = SyncCoordinator::new();
        let ids = [10, 20, 30, 40, 50];
        for &id in &ids {
            coord.record_completion(id);
        }
        let order = coord.completion_order();
        assert_eq!(order, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn command_buffer_dependency_single() {
        let mut cb_a = CommandBuffer::new(0, "producer");
        let mut cb_b = CommandBuffer::new(1, "consumer");
        cb_b.add_dependency(cb_a.id);

        cb_a.enqueue();
        cb_a.commit();
        cb_a.schedule();
        cb_a.complete(0);

        assert!(cb_b.dependencies.contains(&cb_a.id));
        assert_eq!(cb_a.status, CommandBufferStatus::Completed);

        cb_b.enqueue();
        cb_b.commit();
        cb_b.schedule();
        cb_b.complete(1);
        assert_eq!(cb_b.status, CommandBufferStatus::Completed);
        assert!(cb_b.execution_order.unwrap() > cb_a.execution_order.unwrap());
    }

    #[test]
    fn command_buffer_dependency_chain() {
        let mut buffers: Vec<CommandBuffer> =
            (0..4).map(|i| CommandBuffer::new(i, &format!("chain-{i}"))).collect();

        for i in 1..4 {
            buffers[i].add_dependency((i - 1) as u64);
        }

        for (order, buf) in buffers.iter_mut().enumerate() {
            buf.enqueue();
            buf.commit();
            buf.schedule();
            buf.complete(order as u64);
        }

        for i in 1..4 {
            assert!(buffers[i].execution_order.unwrap() > buffers[i - 1].execution_order.unwrap());
        }
    }

    #[test]
    fn command_buffer_fan_out_dependencies() {
        let mut root = CommandBuffer::new(0, "root");
        let mut children: Vec<CommandBuffer> = (1..=4)
            .map(|i| {
                let mut cb = CommandBuffer::new(i, &format!("child-{i}"));
                cb.add_dependency(0);
                cb
            })
            .collect();

        root.enqueue();
        root.commit();
        root.schedule();
        root.complete(0);

        for (i, child) in children.iter_mut().enumerate() {
            assert!(child.dependencies.contains(&0));
            child.enqueue();
            child.commit();
            child.schedule();
            child.complete((i + 1) as u64);
        }

        for child in &children {
            assert!(child.execution_order.unwrap() > 0);
        }
    }

    #[test]
    fn command_buffer_fan_in_dependencies() {
        let mut producers: Vec<CommandBuffer> =
            (0..3).map(|i| CommandBuffer::new(i, &format!("prod-{i}"))).collect();
        let mut consumer = CommandBuffer::new(10, "consumer");
        for p in &producers {
            consumer.add_dependency(p.id);
        }

        for (i, p) in producers.iter_mut().enumerate() {
            p.enqueue();
            p.commit();
            p.schedule();
            p.complete(i as u64);
        }

        assert_eq!(consumer.dependencies.len(), 3);
        consumer.enqueue();
        consumer.commit();
        consumer.schedule();
        consumer.complete(10);
        assert_eq!(consumer.execution_order, Some(10));
    }

    #[test]
    fn gpu_cpu_sync_point_wait_until_completed() {
        let mut cb = CommandBuffer::new(1, "gpu-work");
        cb.enqueue();
        cb.commit();
        cb.schedule();
        cb.complete(0);
        // Simulate CPU waiting for GPU completion.
        assert_eq!(cb.status, CommandBufferStatus::Completed);
    }

    #[test]
    fn gpu_cpu_sync_multiple_buffers() {
        let mut buffers: Vec<CommandBuffer> =
            (0..3).map(|i| CommandBuffer::new(i, &format!("gpu-{i}"))).collect();
        for (order, buf) in buffers.iter_mut().enumerate() {
            buf.enqueue();
            buf.commit();
            buf.schedule();
            buf.complete(order as u64);
        }
        // CPU sync: all must be completed.
        assert!(buffers.iter().all(|b| b.status == CommandBufferStatus::Completed));
    }

    #[test]
    fn command_buffer_error_state() {
        let mut cb = CommandBuffer::new(1, "error-test");
        cb.enqueue();
        cb.commit();
        cb.fail();
        assert_eq!(cb.status, CommandBufferStatus::Error);
    }

    #[test]
    fn command_buffer_with_resource_tracking() {
        let mut cb = CommandBuffer::new(1, "resource-track");
        cb.add_write("buffer_a");
        cb.add_read("buffer_b");
        cb.add_write("buffer_c");
        assert_eq!(cb.writes, vec!["buffer_a", "buffer_c"]);
        assert_eq!(cb.reads, vec!["buffer_b"]);
    }

    #[test]
    fn command_queue_fifo_ordering() {
        let mut queue = CommandQueue::new("fifo", QueuePriority::Normal);
        queue.submit(5);
        queue.submit(3);
        queue.submit(7);
        let drained = queue.drain_all();
        assert_eq!(drained, vec![5, 3, 7]);
    }

    #[test]
    fn command_queue_pending_count() {
        let mut queue = CommandQueue::new("count", QueuePriority::Normal);
        assert_eq!(queue.pending_count(), 0);
        queue.submit(1);
        queue.submit(2);
        assert_eq!(queue.pending_count(), 2);
        let _ = queue.drain_all();
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn command_buffer_label_preserved() {
        let cb = CommandBuffer::new(42, "my-compute-pass");
        assert_eq!(cb.label, "my-compute-pass");
        assert_eq!(cb.id, 42);
    }

    #[test]
    fn sequential_execution_no_reordering() {
        let mut coord = SyncCoordinator::new();
        let mut queue = CommandQueue::new("serial", QueuePriority::Normal);

        for i in 0..8 {
            queue.submit(i);
        }

        let drained = queue.drain_all();
        for &id in &drained {
            coord.record_completion(id);
        }

        let order = coord.completion_order();
        for i in 0..7 {
            assert!(order[i] < order[i + 1] || order[i] == i as u64);
        }
    }

    #[test]
    fn command_buffer_empty_dependencies() {
        let cb = CommandBuffer::new(0, "no-deps");
        assert!(cb.dependencies.is_empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Fence and event tests (20+ tests)
// ═══════════════════════════════════════════════════════════════════════

mod fence_and_event_sync {
    use super::*;

    #[test]
    fn fence_initial_state() {
        let fence = Fence::new(0, "init");
        assert!(!fence.signaled);
        assert_eq!(fence.signal_value, 0);
    }

    #[test]
    fn fence_signal_and_wait() {
        let mut fence = Fence::new(0, "sig");
        fence.signal(1);
        assert!(fence.signaled);
        assert!(fence.wait(1));
    }

    #[test]
    fn fence_wait_before_signal_fails() {
        let fence = Fence::new(0, "early-wait");
        assert!(!fence.wait(1));
    }

    #[test]
    fn fence_signal_higher_value_satisfies_lower_wait() {
        let mut fence = Fence::new(0, "high");
        fence.signal(10);
        assert!(fence.wait(5));
        assert!(fence.wait(10));
        assert!(!fence.wait(11));
    }

    #[test]
    fn fence_reset() {
        let mut fence = Fence::new(0, "reset");
        fence.signal(5);
        assert!(fence.signaled);
        fence.reset();
        assert!(!fence.signaled);
        assert_eq!(fence.signal_value, 0);
    }

    #[test]
    fn fence_label_preserved() {
        let fence = Fence::new(42, "my-fence");
        assert_eq!(fence.label, "my-fence");
        assert_eq!(fence.id, 42);
    }

    #[test]
    fn fence_signal_monotonic_values() {
        let mut fence = Fence::new(0, "mono");
        for v in 1..=10 {
            fence.signal(v);
            assert!(fence.wait(v));
        }
    }

    #[test]
    fn event_initial_state() {
        let event = GpuEvent::new(0, "init");
        assert_eq!(event.signaled_value, 0);
    }

    #[test]
    fn event_signal_and_wait() {
        let mut event = GpuEvent::new(0, "sig");
        event.signal(1);
        assert!(event.wait(1));
    }

    #[test]
    fn event_monotonic_signal() {
        let mut event = GpuEvent::new(0, "mono");
        event.signal(5);
        event.signal(3); // lower value should not decrease
        assert_eq!(event.signaled_value, 5);
        assert!(event.wait(5));
        assert!(!event.wait(6));
    }

    #[test]
    fn event_wait_zero_always_succeeds() {
        let event = GpuEvent::new(0, "zero");
        assert!(event.wait(0));
    }

    #[test]
    fn event_label_preserved() {
        let event = GpuEvent::new(99, "sync-point");
        assert_eq!(event.label, "sync-point");
        assert_eq!(event.id, 99);
    }

    #[test]
    fn cross_queue_fence_coordination() {
        let mut fence = Fence::new(0, "cross-queue");
        let mut q1 = CommandQueue::new("producer", QueuePriority::Normal);
        let mut q2 = CommandQueue::new("consumer", QueuePriority::Normal);

        q1.submit(1);
        let drained = q1.drain_all();
        assert_eq!(drained, vec![1]);
        // Producer signals fence after completing work.
        fence.signal(1);

        // Consumer waits on fence before starting.
        assert!(fence.wait(1));
        q2.submit(2);
        let drained = q2.drain_all();
        assert_eq!(drained, vec![2]);
    }

    #[test]
    fn cross_queue_fence_blocks_until_signal() {
        let fence = Fence::new(0, "blocking");
        // Consumer checks fence before producer signals.
        assert!(!fence.wait(1));
    }

    #[test]
    fn multiple_fences_independent() {
        let mut f1 = Fence::new(0, "fence-a");
        let mut f2 = Fence::new(1, "fence-b");
        f1.signal(1);
        assert!(f1.wait(1));
        assert!(!f2.wait(1));
        f2.signal(2);
        assert!(f2.wait(2));
    }

    #[test]
    fn fence_coordinator_multiple_fences() {
        let mut coord = SyncCoordinator::new();
        let f0 = coord.add_fence(0, "a");
        let f1 = coord.add_fence(1, "b");
        let f2 = coord.add_fence(2, "c");

        coord.fences[f0].signal(1);
        coord.fences[f1].signal(2);
        coord.fences[f2].signal(3);

        assert!(coord.fences[f0].wait(1));
        assert!(coord.fences[f1].wait(2));
        assert!(coord.fences[f2].wait(3));
    }

    #[test]
    fn fence_timeout_simulation() {
        let fence = Fence::new(0, "timeout");
        let max_polls = 100;
        let mut signaled_in_time = false;
        for poll in 0..max_polls {
            if fence.wait(1) {
                signaled_in_time = true;
                break;
            }
            // Fence never signaled — simulates timeout.
            assert!(poll < max_polls);
        }
        assert!(!signaled_in_time, "fence should timeout (never signaled)");
    }

    #[test]
    fn event_progressive_signaling() {
        let mut event = GpuEvent::new(0, "progress");
        for step in 1..=5 {
            event.signal(step);
            assert!(event.wait(step));
            assert!(!event.wait(step + 1));
        }
    }

    #[test]
    fn event_coordinator_multiple_events() {
        let mut coord = SyncCoordinator::new();
        let e0 = coord.add_event(0, "pass-a");
        let e1 = coord.add_event(1, "pass-b");

        coord.events[e0].signal(10);
        coord.events[e1].signal(20);

        assert!(coord.events[e0].wait(10));
        assert!(!coord.events[e0].wait(11));
        assert!(coord.events[e1].wait(20));
    }

    #[test]
    fn cross_queue_event_synchronization() {
        let mut event = GpuEvent::new(0, "xqueue");
        // Queue A signals after dispatch.
        event.signal(1);
        // Queue B waits on the event.
        assert!(event.wait(1));
        // Queue B signals after its dispatch.
        event.signal(2);
        // Queue A can see the update.
        assert!(event.wait(2));
    }

    #[test]
    fn fence_reuse_after_reset() {
        let mut fence = Fence::new(0, "reuse");
        for round in 1..=3 {
            fence.signal(round);
            assert!(fence.wait(round));
            fence.reset();
            assert!(!fence.wait(1));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Memory barrier tests (20+ tests)
// ═══════════════════════════════════════════════════════════════════════

mod memory_barriers {
    use super::*;

    #[test]
    fn buffer_barrier_creation() {
        let barrier = MemoryBarrier::buffer_barrier(vec!["buf_a".into()]);
        assert_eq!(barrier.scope, BarrierScope::Buffers);
        assert_eq!(barrier.after_stages, "compute");
        assert_eq!(barrier.before_stages, "compute");
        assert_eq!(barrier.resources, vec!["buf_a"]);
    }

    #[test]
    fn texture_barrier_creation() {
        let barrier = MemoryBarrier::texture_barrier(vec!["tex_0".into()]);
        assert_eq!(barrier.scope, BarrierScope::Textures);
        assert_eq!(barrier.after_stages, "fragment");
        assert_eq!(barrier.before_stages, "compute");
    }

    #[test]
    fn full_barrier_creation() {
        let barrier = MemoryBarrier::full_barrier();
        assert_eq!(barrier.scope, BarrierScope::All);
        assert!(barrier.resources.is_empty());
    }

    #[test]
    fn read_after_write_hazard_detected() {
        let mut res = ResourceState::new("shared_buf");
        let needs_barrier = res.record_write(0);
        assert!(!needs_barrier); // First write needs no barrier.
        let needs_barrier = res.record_read(1);
        assert!(needs_barrier); // Read after write → barrier needed.
    }

    #[test]
    fn write_after_read_hazard_detected() {
        let mut res = ResourceState::new("shared_buf");
        res.record_read(0);
        let needs_barrier = res.record_write(1);
        assert!(needs_barrier); // Write after read → barrier needed.
    }

    #[test]
    fn write_after_write_hazard_detected() {
        let mut res = ResourceState::new("shared_buf");
        res.record_write(0);
        let needs_barrier = res.record_write(1);
        assert!(needs_barrier); // Write after write → barrier needed.
    }

    #[test]
    fn read_after_read_no_hazard() {
        let mut res = ResourceState::new("readonly_buf");
        res.record_read(0);
        let needs_barrier = res.record_read(1);
        assert!(!needs_barrier); // Read after read → no hazard.
    }

    #[test]
    fn barrier_clears_pending_state() {
        let mut res = ResourceState::new("buf");
        res.record_write(0);
        res.record_read(1);
        assert!(res.barrier_pending);
        res.clear_barrier();
        assert!(!res.barrier_pending);
    }

    #[test]
    fn multiple_readers_then_write() {
        let mut res = ResourceState::new("multi_read");
        res.record_read(0);
        res.record_read(1);
        res.record_read(2);
        assert_eq!(res.last_reader_cmds.len(), 3);
        let needs_barrier = res.record_write(3);
        assert!(needs_barrier);
        assert!(res.last_reader_cmds.is_empty());
    }

    #[test]
    fn write_clears_reader_list() {
        let mut res = ResourceState::new("clear-readers");
        res.record_read(0);
        res.record_read(1);
        assert_eq!(res.last_reader_cmds.len(), 2);
        res.record_write(2);
        assert!(res.last_reader_cmds.is_empty());
        assert_eq!(res.last_writer_cmd, Some(2));
    }

    #[test]
    fn resource_state_tracks_last_writer() {
        let mut res = ResourceState::new("writer-track");
        res.record_write(5);
        assert_eq!(res.last_writer_cmd, Some(5));
        res.record_write(10);
        assert_eq!(res.last_writer_cmd, Some(10));
    }

    #[test]
    fn barrier_with_multiple_resources() {
        let barrier =
            MemoryBarrier::buffer_barrier(vec!["buf_0".into(), "buf_1".into(), "buf_2".into()]);
        assert_eq!(barrier.resources.len(), 3);
    }

    #[test]
    fn coordinator_barrier_insertion() {
        let mut coord = SyncCoordinator::new();
        coord.insert_barrier(MemoryBarrier::full_barrier());
        coord.insert_barrier(MemoryBarrier::buffer_barrier(vec!["x".into()]));
        assert_eq!(coord.barriers.len(), 2);
    }

    #[test]
    fn raw_hazard_sequence_compute_to_compute() {
        let mut res = ResourceState::new("compute_buf");
        // Pass 1 writes.
        res.record_write(0);
        // Pass 2 reads — needs barrier.
        let raw = res.record_read(1);
        assert!(raw);
        res.clear_barrier();
        // Pass 3 reads — no new barrier (writer unchanged).
        let raw2 = res.record_read(2);
        assert!(!raw2);
    }

    #[test]
    fn war_hazard_sequence() {
        let mut res = ResourceState::new("war_buf");
        // Cmd 0 reads.
        res.record_read(0);
        // Cmd 1 writes — WAR hazard.
        let war = res.record_write(1);
        assert!(war);
    }

    #[test]
    fn waw_hazard_sequence() {
        let mut res = ResourceState::new("waw_buf");
        res.record_write(0);
        let waw = res.record_write(1);
        assert!(waw);
    }

    #[test]
    fn barrier_scope_variants() {
        assert_ne!(BarrierScope::Buffers, BarrierScope::Textures);
        assert_ne!(BarrierScope::Textures, BarrierScope::RenderTargets);
        assert_ne!(BarrierScope::RenderTargets, BarrierScope::All);
    }

    #[test]
    fn access_type_variants() {
        assert_ne!(AccessType::Read, AccessType::Write);
        assert_ne!(AccessType::Write, AccessType::ReadWrite);
        assert_ne!(AccessType::Read, AccessType::ReadWrite);
    }

    #[test]
    fn complex_hazard_pattern_pipeline() {
        let mut buf_a = ResourceState::new("a");
        let mut buf_b = ResourceState::new("b");

        // Step 1: write A, read B.
        buf_a.record_write(0);
        buf_b.record_read(0);

        // Step 2: read A (RAW on A), write B (WAR on B).
        let raw_a = buf_a.record_read(1);
        let war_b = buf_b.record_write(1);
        assert!(raw_a);
        assert!(war_b);

        buf_a.clear_barrier();
        buf_b.clear_barrier();

        // Step 3: write A (WAR on A), read B (RAW on B).
        let war_a = buf_a.record_write(2);
        let raw_b = buf_b.record_read(2);
        assert!(war_a);
        assert!(raw_b);
    }

    #[test]
    fn resource_name_preserved() {
        let res = ResourceState::new("my_tensor_buffer");
        assert_eq!(res.name, "my_tensor_buffer");
    }

    #[test]
    fn coordinator_tracks_resources() {
        let mut coord = SyncCoordinator::new();
        let r0 = coord.add_resource("weight_buf");
        let r1 = coord.add_resource("activation_buf");
        assert_eq!(coord.resources[r0].name, "weight_buf");
        assert_eq!(coord.resources[r1].name, "activation_buf");
        assert_eq!(coord.resources.len(), 2);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Resource hazard tracking (15+ tests)
// ═══════════════════════════════════════════════════════════════════════

mod resource_hazard_tracking {
    use super::*;

    #[test]
    fn hazard_tracking_mode_variants() {
        assert_ne!(HazardTrackingMode::Automatic, HazardTrackingMode::Manual);
        assert_ne!(HazardTrackingMode::Manual, HazardTrackingMode::Untracked);
    }

    #[test]
    fn automatic_tracking_inserts_barriers_on_raw() {
        let mode = HazardTrackingMode::Automatic;
        assert_eq!(mode, HazardTrackingMode::Automatic);

        let mut res = ResourceState::new("auto_buf");
        res.record_write(0);
        let needs = res.record_read(1);
        // Automatic mode would insert barrier here.
        assert!(needs);
    }

    #[test]
    fn automatic_tracking_inserts_barriers_on_war() {
        let mut res = ResourceState::new("auto_war");
        res.record_read(0);
        let needs = res.record_write(1);
        assert!(needs);
    }

    #[test]
    fn manual_tracking_requires_explicit_barriers() {
        let mode = HazardTrackingMode::Manual;
        assert_eq!(mode, HazardTrackingMode::Manual);
        // In manual mode, the app must insert barriers explicitly.
        let mut res = ResourceState::new("manual_buf");
        res.record_write(0);
        let needs = res.record_read(1);
        assert!(needs);
        // Manual: caller is responsible for calling clear_barrier.
        assert!(res.barrier_pending);
        res.clear_barrier();
        assert!(!res.barrier_pending);
    }

    #[test]
    fn untracked_mode_no_implicit_barriers() {
        let mode = HazardTrackingMode::Untracked;
        assert_eq!(mode, HazardTrackingMode::Untracked);
        // Untracked resources rely on the application for correctness.
    }

    #[test]
    fn resource_state_transition_read_to_write() {
        let mut res = ResourceState::new("transition");
        assert_eq!(res.last_access, AccessType::Read);
        res.record_write(0);
        assert_eq!(res.last_access, AccessType::Write);
    }

    #[test]
    fn resource_state_transition_write_to_read() {
        let mut res = ResourceState::new("transition");
        res.record_write(0);
        assert_eq!(res.last_access, AccessType::Write);
        res.record_read(1);
        assert_eq!(res.last_access, AccessType::Read);
    }

    #[test]
    fn resource_state_transition_write_to_write() {
        let mut res = ResourceState::new("waw");
        res.record_write(0);
        res.record_write(1);
        assert_eq!(res.last_access, AccessType::Write);
        assert_eq!(res.last_writer_cmd, Some(1));
    }

    #[test]
    fn multi_resource_independent_tracking() {
        let mut r_a = ResourceState::new("a");
        let mut r_b = ResourceState::new("b");
        let mut r_c = ResourceState::new("c");

        r_a.record_write(0);
        r_b.record_read(0);
        r_c.record_write(0);

        let raw_a = r_a.record_read(1);
        let no_hazard_b = r_b.record_read(1);
        let raw_c = r_c.record_read(1);

        assert!(raw_a);
        assert!(!no_hazard_b);
        assert!(raw_c);
    }

    #[test]
    fn hazard_tracking_with_barrier_insertion() {
        let mut coord = SyncCoordinator::new();
        let r = coord.add_resource("tracked");

        coord.resources[r].record_write(0);
        let needs = coord.resources[r].record_read(1);
        assert!(needs);

        coord.insert_barrier(MemoryBarrier::buffer_barrier(vec!["tracked".into()]));
        coord.resources[r].clear_barrier();
        assert!(!coord.resources[r].barrier_pending);
    }

    #[test]
    fn hazard_detection_across_many_commands() {
        let mut res = ResourceState::new("many_cmds");
        let mut hazard_count = 0;

        // Pattern: W, R, R, W, R, W, W, R
        let ops = [
            AccessType::Write,
            AccessType::Read,
            AccessType::Read,
            AccessType::Write,
            AccessType::Read,
            AccessType::Write,
            AccessType::Write,
            AccessType::Read,
        ];

        for (cmd_id, &op) in ops.iter().enumerate() {
            let needs = match op {
                AccessType::Write => res.record_write(cmd_id as u64),
                AccessType::Read => res.record_read(cmd_id as u64),
                AccessType::ReadWrite => {
                    let w = res.record_write(cmd_id as u64);
                    let r = res.record_read(cmd_id as u64);
                    w || r
                }
            };
            if needs {
                hazard_count += 1;
                res.clear_barrier();
            }
        }

        assert!(hazard_count >= 4, "expected at least 4 hazards");
    }

    #[test]
    fn fresh_resource_no_pending_barrier() {
        let res = ResourceState::new("fresh");
        assert!(!res.barrier_pending);
        assert!(res.last_writer_cmd.is_none());
        assert!(res.last_reader_cmds.is_empty());
    }

    #[test]
    fn automatic_tracking_pipeline_example() {
        // Simulates a compute pipeline with automatic hazard tracking.
        let mut coord = SyncCoordinator::new();
        let weights = coord.add_resource("weights");
        let activations = coord.add_resource("activations");
        let output = coord.add_resource("output");

        // Layer 1: read weights, write activations.
        coord.resources[weights].record_read(0);
        coord.resources[activations].record_write(0);

        // Layer 2: read activations (RAW), read weights, write output.
        let raw = coord.resources[activations].record_read(1);
        assert!(raw);
        coord.insert_barrier(MemoryBarrier::buffer_barrier(vec!["activations".into()]));
        coord.resources[activations].clear_barrier();

        coord.resources[weights].record_read(1);
        coord.resources[output].record_write(1);

        assert_eq!(coord.barriers.len(), 1);
    }

    #[test]
    fn manual_tracking_requires_explicit_clear() {
        let mut res = ResourceState::new("manual");
        res.record_write(0);
        res.record_read(1);
        assert!(res.barrier_pending);
        // Without clear, barrier stays pending.
        res.record_read(2);
        assert!(res.barrier_pending);
        res.clear_barrier();
        assert!(!res.barrier_pending);
    }

    #[test]
    fn resource_tracks_all_readers() {
        let mut res = ResourceState::new("multi_reader");
        res.record_read(10);
        res.record_read(20);
        res.record_read(30);
        assert_eq!(res.last_reader_cmds, vec![10, 20, 30]);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Multi-queue coordination (15+ tests)
// ═══════════════════════════════════════════════════════════════════════

mod multi_queue_coordination {
    use super::*;

    #[test]
    fn serial_queue_preserves_order() {
        let mut queue = CommandQueue::new("serial", QueuePriority::Normal);
        for i in 0..10 {
            queue.submit(i);
        }
        let drained = queue.drain_all();
        let expected: Vec<u64> = (0..10).collect();
        assert_eq!(drained, expected);
    }

    #[test]
    fn queue_priority_ordering() {
        assert!(QueuePriority::Low < QueuePriority::Normal);
        assert!(QueuePriority::Normal < QueuePriority::High);
    }

    #[test]
    fn multiple_queues_independent() {
        let mut q1 = CommandQueue::new("q1", QueuePriority::Normal);
        let mut q2 = CommandQueue::new("q2", QueuePriority::Normal);

        q1.submit(1);
        q1.submit(2);
        q2.submit(10);
        q2.submit(20);

        assert_eq!(q1.pending_count(), 2);
        assert_eq!(q2.pending_count(), 2);

        let d1 = q1.drain_all();
        let d2 = q2.drain_all();
        assert_eq!(d1, vec![1, 2]);
        assert_eq!(d2, vec![10, 20]);
    }

    #[test]
    fn high_priority_queue_label() {
        let q = CommandQueue::new("compute-high", QueuePriority::High);
        assert_eq!(q.label, "compute-high");
        assert_eq!(q.priority, QueuePriority::High);
    }

    #[test]
    fn work_submission_ordering_across_queues() {
        let mut q_compute = CommandQueue::new("compute", QueuePriority::High);
        let mut q_copy = CommandQueue::new("copy", QueuePriority::Normal);

        // Submit compute work.
        for i in 0..5 {
            q_compute.submit(i);
        }
        // Submit copy work.
        for i in 100..103 {
            q_copy.submit(i);
        }

        let compute_items = q_compute.drain_all();
        let copy_items = q_copy.drain_all();

        assert_eq!(compute_items.len(), 5);
        assert_eq!(copy_items.len(), 3);
    }

    #[test]
    fn queue_with_fence_synchronization() {
        let mut q1 = CommandQueue::new("producer", QueuePriority::Normal);
        let mut q2 = CommandQueue::new("consumer", QueuePriority::Normal);
        let mut fence = Fence::new(0, "q1->q2");

        q1.submit(1);
        let _ = q1.drain_all();
        fence.signal(1);

        assert!(fence.wait(1));
        q2.submit(2);
        let items = q2.drain_all();
        assert_eq!(items, vec![2]);
    }

    #[test]
    fn concurrent_queue_execution_model() {
        // Two concurrent queues executing independently.
        let mut q_a = CommandQueue::new("a", QueuePriority::Normal);
        let mut q_b = CommandQueue::new("b", QueuePriority::Normal);
        let mut coord = SyncCoordinator::new();

        q_a.submit(1);
        q_a.submit(2);
        q_b.submit(3);
        q_b.submit(4);

        // Interleaved execution simulating concurrency.
        let a_items = q_a.drain_all();
        let b_items = q_b.drain_all();

        for &id in &a_items {
            coord.record_completion(id);
        }
        for &id in &b_items {
            coord.record_completion(id);
        }

        assert_eq!(coord.completed.len(), 4);
    }

    #[test]
    fn priority_based_scheduling_order() {
        let mut queues = vec![
            CommandQueue::new("low", QueuePriority::Low),
            CommandQueue::new("normal", QueuePriority::Normal),
            CommandQueue::new("high", QueuePriority::High),
        ];

        queues[0].submit(1);
        queues[1].submit(2);
        queues[2].submit(3);

        // Sort by priority descending for scheduling.
        queues.sort_by(|a, b| b.priority.cmp(&a.priority));
        assert_eq!(queues[0].label, "high");
        assert_eq!(queues[1].label, "normal");
        assert_eq!(queues[2].label, "low");
    }

    #[test]
    fn queue_drain_empties_queue() {
        let mut q = CommandQueue::new("drain", QueuePriority::Normal);
        q.submit(1);
        q.submit(2);
        assert_eq!(q.pending_count(), 2);
        let _ = q.drain_all();
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn empty_queue_drain_returns_empty() {
        let mut q = CommandQueue::new("empty", QueuePriority::Normal);
        let drained = q.drain_all();
        assert!(drained.is_empty());
    }

    #[test]
    fn execution_order_counter_increments() {
        let q = CommandQueue::new("counter", QueuePriority::Normal);
        let o0 = q.next_execution_order();
        let o1 = q.next_execution_order();
        let o2 = q.next_execution_order();
        assert_eq!(o0, 0);
        assert_eq!(o1, 1);
        assert_eq!(o2, 2);
    }

    #[test]
    fn multi_queue_fence_chain() {
        let mut q1 = CommandQueue::new("stage-1", QueuePriority::High);
        let mut q2 = CommandQueue::new("stage-2", QueuePriority::Normal);
        let mut q3 = CommandQueue::new("stage-3", QueuePriority::Low);
        let mut f12 = Fence::new(0, "1->2");
        let mut f23 = Fence::new(1, "2->3");

        q1.submit(1);
        let _ = q1.drain_all();
        f12.signal(1);

        assert!(f12.wait(1));
        q2.submit(2);
        let _ = q2.drain_all();
        f23.signal(1);

        assert!(f23.wait(1));
        q3.submit(3);
        let items = q3.drain_all();
        assert_eq!(items, vec![3]);
    }

    #[test]
    fn producer_consumer_with_event() {
        let mut event = GpuEvent::new(0, "prod-cons");
        let mut q_prod = CommandQueue::new("producer", QueuePriority::Normal);
        let mut q_cons = CommandQueue::new("consumer", QueuePriority::Normal);

        // Producer enqueues and signals event.
        q_prod.submit(1);
        let _ = q_prod.drain_all();
        event.signal(1);

        // Consumer waits and then processes.
        assert!(event.wait(1));
        q_cons.submit(2);
        assert_eq!(q_cons.pending_count(), 1);
    }

    #[test]
    fn three_queue_diamond_sync() {
        // A → B and A → C, then B+C → D (modeled as fence waits).
        let mut q_a = CommandQueue::new("A", QueuePriority::High);
        let mut q_b = CommandQueue::new("B", QueuePriority::Normal);
        let mut q_c = CommandQueue::new("C", QueuePriority::Normal);
        let mut f_ab = Fence::new(0, "A->B");
        let mut f_ac = Fence::new(1, "A->C");

        q_a.submit(0);
        let _ = q_a.drain_all();
        f_ab.signal(1);
        f_ac.signal(1);

        assert!(f_ab.wait(1));
        q_b.submit(1);
        let _ = q_b.drain_all();

        assert!(f_ac.wait(1));
        q_c.submit(2);
        let _ = q_c.drain_all();

        // Both B and C complete, D can proceed.
        let mut f_bd = Fence::new(2, "B->D");
        let mut f_cd = Fence::new(3, "C->D");
        f_bd.signal(1);
        f_cd.signal(1);
        assert!(f_bd.wait(1) && f_cd.wait(1));
    }

    #[test]
    fn queue_priority_all_variants() {
        let priorities = [QueuePriority::Low, QueuePriority::Normal, QueuePriority::High];
        for (i, p) in priorities.iter().enumerate() {
            for q in priorities.iter().skip(i + 1) {
                assert!(p < q);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Double/triple buffering (15+ tests)
// ═══════════════════════════════════════════════════════════════════════

mod ring_buffer_patterns {
    use super::*;

    #[test]
    fn double_buffer_creation() {
        let rb = RingBuffer::new(2, 256);
        assert_eq!(rb.buffer_count, 2);
        assert_eq!(rb.slots.len(), 2);
        assert_eq!(rb.in_flight_count(), 0);
        assert_eq!(rb.available_count(), 2);
    }

    #[test]
    fn triple_buffer_creation() {
        let rb = RingBuffer::new(3, 512);
        assert_eq!(rb.buffer_count, 3);
        assert_eq!(rb.available_count(), 3);
    }

    #[test]
    fn acquire_returns_slot_index() {
        let mut rb = RingBuffer::new(2, 128);
        let slot = rb.acquire();
        assert_eq!(slot, Some(0));
        assert_eq!(rb.in_flight_count(), 1);
    }

    #[test]
    fn acquire_rotates_through_slots() {
        let mut rb = RingBuffer::new(3, 128);
        assert_eq!(rb.acquire(), Some(0));
        assert_eq!(rb.acquire(), Some(1));
        assert_eq!(rb.acquire(), Some(2));
        assert_eq!(rb.in_flight_count(), 3);
    }

    #[test]
    fn acquire_fails_when_all_in_flight() {
        let mut rb = RingBuffer::new(2, 128);
        assert!(rb.acquire().is_some());
        assert!(rb.acquire().is_some());
        assert_eq!(rb.acquire(), None); // All in flight.
    }

    #[test]
    fn release_makes_slot_available() {
        let mut rb = RingBuffer::new(2, 128);
        let idx = rb.acquire().unwrap();
        assert_eq!(rb.available_count(), 1);
        rb.release(idx);
        assert_eq!(rb.available_count(), 2);
    }

    #[test]
    fn double_buffer_ping_pong() {
        let mut rb = RingBuffer::new(2, 128);

        for frame in 0..10 {
            let slot = rb.acquire().expect("should have free slot");
            assert_eq!(slot, frame % 2);
            // Simulate GPU work completing on previous frame.
            if frame > 0 {
                rb.release((frame - 1) % 2);
            }
            // On last iteration, release current.
            if frame == 9 {
                rb.release(slot);
            }
        }
    }

    #[test]
    fn triple_buffer_rotation() {
        let mut rb = RingBuffer::new(3, 64);

        // Fill all three.
        let s0 = rb.acquire().unwrap();
        let s1 = rb.acquire().unwrap();
        let s2 = rb.acquire().unwrap();
        assert_eq!((s0, s1, s2), (0, 1, 2));
        assert_eq!(rb.available_count(), 0);

        // Release oldest, acquire new.
        rb.release(s0);
        let s3 = rb.acquire().unwrap();
        assert_eq!(s3, 0); // Wraps around to slot 0.
    }

    #[test]
    fn frame_id_increments() {
        let mut rb = RingBuffer::new(2, 64);
        let s0 = rb.acquire().unwrap();
        assert_eq!(rb.slots[s0].frame_id, 0);
        rb.release(s0);
        let s1 = rb.acquire().unwrap();
        assert_eq!(rb.slots[s1].frame_id, 1);
    }

    #[test]
    fn buffer_slot_data_capacity() {
        let rb = RingBuffer::new(2, 1024);
        assert_eq!(rb.slots[0].data.len(), 1024);
        assert_eq!(rb.slots[1].data.len(), 1024);
    }

    #[test]
    fn ring_buffer_with_fence_sync() {
        let mut rb = RingBuffer::new(2, 256);
        let mut fences = vec![Fence::new(0, "slot-0"), Fence::new(1, "slot-1")];

        // Frame 0: acquire slot 0, submit GPU work.
        let s0 = rb.acquire().unwrap();
        assert_eq!(s0, 0);

        // Frame 1: acquire slot 1, signal fence for slot 0.
        let s1 = rb.acquire().unwrap();
        assert_eq!(s1, 1);
        fences[s0].signal(1);

        // Frame 2: wait for slot 0 fence, release, reacquire.
        assert!(fences[s0].wait(1));
        rb.release(s0);
        let s2 = rb.acquire().unwrap();
        assert_eq!(s2, 0); // Reuses slot 0.

        // Clean up.
        fences[s1].signal(1);
        rb.release(s1);
        rb.release(s2);
    }

    #[test]
    fn frame_synchronization_with_events() {
        let mut rb = RingBuffer::new(3, 128);
        let mut frame_event = GpuEvent::new(0, "frame-done");

        for frame in 0u64..6 {
            // Wait for oldest in-flight frame if buffer full.
            if rb.available_count() == 0 {
                let oldest_frame = frame - rb.buffer_count as u64;
                assert!(frame_event.wait(oldest_frame + 1), "frame {oldest_frame} should be done");
                rb.release((oldest_frame as usize) % rb.buffer_count);
            }

            let slot = rb.acquire().unwrap();
            // Simulate GPU completion.
            frame_event.signal(frame + 1);

            if frame == 5 {
                rb.release(slot);
            }
        }
    }

    #[test]
    fn buffer_data_isolation() {
        let mut rb = RingBuffer::new(2, 4);
        rb.slots[0].data = vec![1.0, 2.0, 3.0, 4.0];
        rb.slots[1].data = vec![5.0, 6.0, 7.0, 8.0];

        assert_ne!(rb.slots[0].data, rb.slots[1].data);
        assert_eq!(rb.slots[0].data[0], 1.0);
        assert_eq!(rb.slots[1].data[0], 5.0);
    }

    #[test]
    fn single_buffer_blocks_immediately() {
        let mut rb = RingBuffer::new(1, 64);
        let s = rb.acquire().unwrap();
        assert_eq!(rb.acquire(), None);
        rb.release(s);
        assert!(rb.acquire().is_some());
    }

    #[test]
    fn quad_buffer_rotation() {
        let mut rb = RingBuffer::new(4, 32);
        for i in 0..4 {
            assert_eq!(rb.acquire(), Some(i));
        }
        assert_eq!(rb.acquire(), None);
        rb.release(0);
        assert_eq!(rb.acquire(), Some(0));
    }

    #[test]
    fn ring_buffer_frame_counter_monotonic() {
        let mut rb = RingBuffer::new(2, 16);
        let mut last_frame = 0u64;
        for _ in 0..10 {
            let slot = rb.acquire().unwrap();
            assert!(rb.slots[slot].frame_id >= last_frame);
            last_frame = rb.slots[slot].frame_id;
            rb.release(slot);
        }
    }

    #[test]
    fn buffer_recycling_preserves_capacity() {
        let mut rb = RingBuffer::new(2, 512);
        for _ in 0..20 {
            let slot = rb.acquire().unwrap();
            assert_eq!(rb.slots[slot].data.len(), 512);
            rb.release(slot);
        }
    }

    #[test]
    fn in_flight_and_available_sum_to_total() {
        let mut rb = RingBuffer::new(3, 64);
        assert_eq!(rb.in_flight_count() + rb.available_count(), rb.buffer_count);
        rb.acquire();
        assert_eq!(rb.in_flight_count() + rb.available_count(), rb.buffer_count);
        rb.acquire();
        assert_eq!(rb.in_flight_count() + rb.available_count(), rb.buffer_count);
    }
}
