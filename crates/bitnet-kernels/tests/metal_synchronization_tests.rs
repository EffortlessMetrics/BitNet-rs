//! Metal synchronization and concurrency pattern tests for Apple Silicon.
//!
//! Validates command buffer ordering, fence/event synchronization, resource
//! hazard tracking, multi-queue coordination, atomic operations, buffer
//! coherence, execution dependency graphs, and Apple Silicon–specific sync
//! primitives. All tests use mock/simulated types — no GPU hardware required.

#![cfg(feature = "cpu")]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ═══════════════════════════════════════════════════════════════════════════
// Mock / simulated Metal types
// ═══════════════════════════════════════════════════════════════════════════

/// Simulated Metal command buffer status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandBufferStatus {
    NotEnqueued,
    Enqueued,
    Committed,
    Scheduled,
    Completed,
    Error,
}

/// A mock Metal command buffer that tracks encoding order and dependencies.
#[derive(Debug, Clone)]
struct MockCommandBuffer {
    id: u64,
    label: String,
    status: CommandBufferStatus,
    encoded_at: Option<u64>,
    completed_at: Option<u64>,
    /// IDs of command buffers this one depends on.
    dependencies: Vec<u64>,
    /// Resource accesses recorded during encoding.
    resource_accesses: Vec<ResourceAccess>,
}

impl MockCommandBuffer {
    fn new(id: u64, label: &str) -> Self {
        Self {
            id,
            label: label.to_string(),
            status: CommandBufferStatus::NotEnqueued,
            encoded_at: None,
            completed_at: None,
            dependencies: Vec::new(),
            resource_accesses: Vec::new(),
        }
    }

    fn enqueue(&mut self) {
        assert_eq!(self.status, CommandBufferStatus::NotEnqueued);
        self.status = CommandBufferStatus::Enqueued;
    }

    fn commit(&mut self, tick: u64) {
        assert!(
            self.status == CommandBufferStatus::Enqueued
                || self.status == CommandBufferStatus::NotEnqueued
        );
        self.status = CommandBufferStatus::Committed;
        self.encoded_at = Some(tick);
    }

    fn complete(&mut self, tick: u64) {
        assert_eq!(self.status, CommandBufferStatus::Committed);
        self.status = CommandBufferStatus::Completed;
        self.completed_at = Some(tick);
    }
}

/// Access mode for a GPU resource inside a command buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

/// A single resource access record.
#[derive(Debug, Clone)]
struct ResourceAccess {
    resource_id: u64,
    mode: AccessMode,
    offset: usize,
    length: usize,
}

/// Simulated GPU fence.
#[derive(Debug)]
struct MockFence {
    id: u64,
    label: String,
    signaled: bool,
    signal_value: u64,
}

impl MockFence {
    fn new(id: u64, label: &str) -> Self {
        Self { id, label: label.to_string(), signaled: false, signal_value: 0 }
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

/// Simulated shared event (CPU↔GPU synchronization).
#[derive(Debug)]
struct MockSharedEvent {
    id: u64,
    value: AtomicU64,
    listeners: Vec<(u64, bool)>, // (threshold, notified)
}

impl MockSharedEvent {
    fn new(id: u64) -> Self {
        Self { id, value: AtomicU64::new(0), listeners: Vec::new() }
    }

    fn signal(&self, value: u64) {
        self.value.store(value, Ordering::SeqCst);
    }

    fn current_value(&self) -> u64 {
        self.value.load(Ordering::SeqCst)
    }

    fn add_listener(&mut self, threshold: u64) {
        self.listeners.push((threshold, false));
    }

    fn poll_listeners(&mut self) -> Vec<u64> {
        let current = self.current_value();
        let mut notified = Vec::new();
        for (threshold, fired) in &mut self.listeners {
            if !*fired && current >= *threshold {
                *fired = true;
                notified.push(*threshold);
            }
        }
        notified
    }
}

/// Simulated command queue with ordering guarantees.
struct MockCommandQueue {
    id: u64,
    label: String,
    submitted: Vec<MockCommandBuffer>,
    completion_order: Vec<u64>,
    next_tick: u64,
}

impl MockCommandQueue {
    fn new(id: u64, label: &str) -> Self {
        Self {
            id,
            label: label.to_string(),
            submitted: Vec::new(),
            completion_order: Vec::new(),
            next_tick: 0,
        }
    }

    fn submit(&mut self, mut buf: MockCommandBuffer) -> u64 {
        let tick = self.next_tick;
        self.next_tick += 1;
        buf.commit(tick);
        let id = buf.id;
        self.submitted.push(buf);
        id
    }

    fn complete_next(&mut self) -> Option<u64> {
        for buf in &mut self.submitted {
            if buf.status == CommandBufferStatus::Committed {
                let tick = self.next_tick;
                self.next_tick += 1;
                buf.complete(tick);
                self.completion_order.push(buf.id);
                return Some(buf.id);
            }
        }
        None
    }

    fn complete_all(&mut self) {
        while self.complete_next().is_some() {}
    }

    fn is_completed(&self, id: u64) -> bool {
        self.submitted.iter().any(|b| b.id == id && b.status == CommandBufferStatus::Completed)
    }
}

/// Hazard types between two resource accesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HazardType {
    ReadAfterWrite,
    WriteAfterRead,
    WriteAfterWrite,
}

/// Detects data hazards between two command buffers.
fn detect_hazards(a: &MockCommandBuffer, b: &MockCommandBuffer) -> Vec<(u64, HazardType)> {
    let mut hazards = Vec::new();
    for acc_a in &a.resource_accesses {
        for acc_b in &b.resource_accesses {
            if acc_a.resource_id != acc_b.resource_id {
                continue;
            }
            // Check range overlap.
            let a_end = acc_a.offset + acc_a.length;
            let b_end = acc_b.offset + acc_b.length;
            if acc_a.offset >= b_end || acc_b.offset >= a_end {
                continue;
            }
            match (acc_a.mode, acc_b.mode) {
                (AccessMode::Write, AccessMode::Read)
                | (AccessMode::Write, AccessMode::ReadWrite)
                | (AccessMode::ReadWrite, AccessMode::Read) => {
                    hazards.push((acc_a.resource_id, HazardType::ReadAfterWrite));
                }
                (AccessMode::Read, AccessMode::Write)
                | (AccessMode::Read, AccessMode::ReadWrite)
                | (AccessMode::ReadWrite, AccessMode::Write) => {
                    hazards.push((acc_a.resource_id, HazardType::WriteAfterRead));
                }
                (AccessMode::Write, AccessMode::Write)
                | (AccessMode::ReadWrite, AccessMode::ReadWrite) => {
                    hazards.push((acc_a.resource_id, HazardType::WriteAfterWrite));
                }
                (AccessMode::Read, AccessMode::Read) => { /* no hazard */ }
            }
        }
    }
    hazards
}

/// Dependency graph for execution ordering (topological sort + cycle detection).
struct DependencyGraph {
    /// node → set of nodes it depends on (predecessors).
    edges: HashMap<u64, HashSet<u64>>,
    nodes: HashSet<u64>,
}

impl DependencyGraph {
    fn new() -> Self {
        Self { edges: HashMap::new(), nodes: HashSet::new() }
    }

    fn add_node(&mut self, id: u64) {
        self.nodes.insert(id);
        self.edges.entry(id).or_default();
    }

    fn add_edge(&mut self, from: u64, to: u64) {
        self.nodes.insert(from);
        self.nodes.insert(to);
        self.edges.entry(to).or_default().insert(from);
        self.edges.entry(from).or_default();
    }

    /// Kahn's algorithm: returns topological order or `Err` if cycle.
    fn topological_sort(&self) -> Result<Vec<u64>, &'static str> {
        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        for &n in &self.nodes {
            in_degree.entry(n).or_insert(0);
        }
        for deps in self.edges.values() {
            for &_d in deps {
                // _d is a predecessor, so this node has in-degree contribution.
            }
        }
        // Recompute: in_degree[n] = number of predecessors.
        for &n in &self.nodes {
            in_degree.insert(n, self.edges.get(&n).map_or(0, |s| s.len()));
        }
        // Reverse: in_degree[n] = how many edges point *to* n.
        // Our edges: edges[n] = set of predecessors of n.
        // So in_degree[n] = edges[n].len() means "n waits for this many".
        // But topological sort needs: in_degree = number of incoming edges.
        // edges[n] = predecessors → these ARE the incoming edges of n.
        let mut in_deg: HashMap<u64, usize> = HashMap::new();
        for &n in &self.nodes {
            in_deg.insert(n, self.edges.get(&n).map_or(0, |s| s.len()));
        }

        let mut queue: VecDeque<u64> =
            in_deg.iter().filter(|&(_, &d)| d == 0).map(|(&n, _)| n).collect();
        // Sort the initial queue for deterministic output.
        let mut sorted_init: Vec<u64> = queue.drain(..).collect();
        sorted_init.sort();
        queue.extend(sorted_init);

        let mut order = Vec::new();
        while let Some(n) = queue.pop_front() {
            order.push(n);
            // For every node m whose predecessor set contains n, decrement.
            let mut ready: Vec<u64> = Vec::new();
            for (&m, preds) in &self.edges {
                if preds.contains(&n) {
                    let deg = in_deg.get_mut(&m).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        ready.push(m);
                    }
                }
            }
            ready.sort();
            queue.extend(ready);
        }

        if order.len() == self.nodes.len() { Ok(order) } else { Err("cycle detected") }
    }

    fn has_cycle(&self) -> bool {
        self.topological_sort().is_err()
    }
}

/// Atomic counter for simulated GPU atomics.
struct AtomicCounter {
    value: AtomicU64,
}

impl AtomicCounter {
    fn new(initial: u64) -> Self {
        Self { value: AtomicU64::new(initial) }
    }

    fn load(&self) -> u64 {
        self.value.load(Ordering::SeqCst)
    }

    fn store(&self, val: u64) {
        self.value.store(val, Ordering::SeqCst);
    }

    fn fetch_add(&self, val: u64) -> u64 {
        self.value.fetch_add(val, Ordering::SeqCst)
    }

    fn compare_exchange(&self, expected: u64, desired: u64) -> Result<u64, u64> {
        self.value.compare_exchange(expected, desired, Ordering::SeqCst, Ordering::SeqCst)
    }

    fn fetch_max(&self, val: u64) -> u64 {
        self.value.fetch_max(val, Ordering::SeqCst)
    }

    fn fetch_min(&self, val: u64) -> u64 {
        self.value.fetch_min(val, Ordering::SeqCst)
    }
}

/// Shared buffer with simulated coherence tracking.
struct CoherentBuffer {
    data: Arc<Mutex<Vec<u8>>>,
    /// Tracks whether the CPU-side copy is stale.
    cpu_dirty: Arc<Mutex<bool>>,
    /// Tracks whether the GPU-side copy is stale.
    gpu_dirty: Arc<Mutex<bool>>,
}

impl CoherentBuffer {
    fn new(size: usize) -> Self {
        Self {
            data: Arc::new(Mutex::new(vec![0u8; size])),
            cpu_dirty: Arc::new(Mutex::new(false)),
            gpu_dirty: Arc::new(Mutex::new(false)),
        }
    }

    fn gpu_write(&self, offset: usize, bytes: &[u8]) {
        let mut data = self.data.lock().unwrap();
        data[offset..offset + bytes.len()].copy_from_slice(bytes);
        *self.cpu_dirty.lock().unwrap() = true;
        *self.gpu_dirty.lock().unwrap() = false;
    }

    fn cpu_write(&self, offset: usize, bytes: &[u8]) {
        let mut data = self.data.lock().unwrap();
        data[offset..offset + bytes.len()].copy_from_slice(bytes);
        *self.gpu_dirty.lock().unwrap() = true;
        *self.cpu_dirty.lock().unwrap() = false;
    }

    fn cpu_read(&self, offset: usize, len: usize) -> Vec<u8> {
        let data = self.data.lock().unwrap();
        data[offset..offset + len].to_vec()
    }

    fn is_cpu_stale(&self) -> bool {
        *self.cpu_dirty.lock().unwrap()
    }

    fn is_gpu_stale(&self) -> bool {
        *self.gpu_dirty.lock().unwrap()
    }

    fn flush_gpu_cache(&self) {
        *self.cpu_dirty.lock().unwrap() = false;
    }

    fn invalidate_gpu_cache(&self) {
        *self.gpu_dirty.lock().unwrap() = false;
    }

    fn len(&self) -> usize {
        self.data.lock().unwrap().len()
    }
}

/// Pipeline stage for render-to-compute transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineStage {
    Vertex,
    Fragment,
    Compute,
    Blit,
}

/// Records a pipeline transition barrier.
#[derive(Debug, Clone, Copy)]
struct PipelineBarrier {
    src_stage: PipelineStage,
    dst_stage: PipelineStage,
    resource_id: u64,
}

impl PipelineBarrier {
    fn is_valid_transition(&self) -> bool {
        // Metal allows transitions between any stages, but certain ones need barriers.
        self.src_stage != self.dst_stage
    }
}

/// Simple sync-overhead tracker for regression detection.
struct SyncOverheadTracker {
    fence_count: u32,
    event_signals: u32,
    barrier_count: u32,
    max_fence_limit: u32,
    max_barrier_limit: u32,
}

impl SyncOverheadTracker {
    fn new(max_fences: u32, max_barriers: u32) -> Self {
        Self {
            fence_count: 0,
            event_signals: 0,
            barrier_count: 0,
            max_fence_limit: max_fences,
            max_barrier_limit: max_barriers,
        }
    }

    fn record_fence(&mut self) {
        self.fence_count += 1;
    }

    fn record_event_signal(&mut self) {
        self.event_signals += 1;
    }

    fn record_barrier(&mut self) {
        self.barrier_count += 1;
    }

    fn is_within_budget(&self) -> bool {
        self.fence_count <= self.max_fence_limit && self.barrier_count <= self.max_barrier_limit
    }

    fn total_sync_ops(&self) -> u32 {
        self.fence_count + self.event_signals + self.barrier_count
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Command Buffer Ordering
// ═══════════════════════════════════════════════════════════════════════════

mod command_buffer_ordering {
    use super::*;

    #[test]
    fn sequential_encoding_preserves_order() {
        let mut queue = MockCommandQueue::new(0, "main");
        for i in 0..4 {
            let buf = MockCommandBuffer::new(i, &format!("cmd_{i}"));
            queue.submit(buf);
        }
        queue.complete_all();
        assert_eq!(queue.completion_order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn parallel_buffers_all_complete() {
        let mut queue = MockCommandQueue::new(0, "main");
        let ids: Vec<u64> = (0..8).collect();
        for &id in &ids {
            queue.submit(MockCommandBuffer::new(id, &format!("par_{id}")));
        }
        queue.complete_all();
        for &id in &ids {
            assert!(queue.is_completed(id), "buffer {id} not completed");
        }
    }

    #[test]
    fn dependency_dag_respects_edges() {
        let mut graph = DependencyGraph::new();
        // A(0) → B(1) → C(2), A(0) → D(3)
        for id in 0..4 {
            graph.add_node(id);
        }
        graph.add_edge(0, 1); // B depends on A
        graph.add_edge(1, 2); // C depends on B
        graph.add_edge(0, 3); // D depends on A
        let order = graph.topological_sort().unwrap();
        let pos = |id: u64| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(1) < pos(2));
        assert!(pos(0) < pos(3));
    }

    #[test]
    fn empty_queue_completes_immediately() {
        let mut queue = MockCommandQueue::new(0, "empty");
        assert_eq!(queue.complete_next(), None);
        assert!(queue.completion_order.is_empty());
    }

    #[test]
    fn single_buffer_lifecycle() {
        let mut buf = MockCommandBuffer::new(0, "single");
        assert_eq!(buf.status, CommandBufferStatus::NotEnqueued);
        buf.enqueue();
        assert_eq!(buf.status, CommandBufferStatus::Enqueued);
        buf.commit(0);
        assert_eq!(buf.status, CommandBufferStatus::Committed);
        buf.complete(1);
        assert_eq!(buf.status, CommandBufferStatus::Completed);
    }

    #[test]
    fn encoded_at_tick_monotonic() {
        let mut queue = MockCommandQueue::new(0, "ticks");
        for i in 0..5 {
            queue.submit(MockCommandBuffer::new(i, "t"));
        }
        let ticks: Vec<u64> = queue.submitted.iter().map(|b| b.encoded_at.unwrap()).collect();
        for w in ticks.windows(2) {
            assert!(w[0] < w[1], "ticks must be strictly increasing");
        }
    }

    #[test]
    fn diamond_dependency_ordering() {
        // Diamond: A→B, A→C, B→D, C→D
        let mut graph = DependencyGraph::new();
        for id in 0..4 {
            graph.add_node(id);
        }
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 3);
        let order = graph.topological_sort().unwrap();
        let pos = |id: u64| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(0) < pos(2));
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(3));
    }

    #[test]
    fn long_chain_preserves_order() {
        let mut graph = DependencyGraph::new();
        let n = 16;
        for i in 0..n {
            graph.add_node(i);
        }
        for i in 0..n - 1 {
            graph.add_edge(i, i + 1);
        }
        let order = graph.topological_sort().unwrap();
        assert_eq!(order, (0..n).collect::<Vec<_>>());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Fence Synchronization
// ═══════════════════════════════════════════════════════════════════════════

mod fence_synchronization {
    use super::*;

    #[test]
    fn fence_creation_defaults_unsignaled() {
        let fence = MockFence::new(0, "f0");
        assert!(!fence.signaled);
        assert_eq!(fence.signal_value, 0);
    }

    #[test]
    fn fence_signal_and_wait_succeeds() {
        let mut fence = MockFence::new(0, "f0");
        fence.signal(42);
        assert!(fence.wait(42));
        assert!(fence.wait(1)); // Any value ≤ 42.
    }

    #[test]
    fn fence_wait_before_signal_fails() {
        let fence = MockFence::new(0, "f0");
        assert!(!fence.wait(1));
    }

    #[test]
    fn fence_wait_exceeds_signal_value_fails() {
        let mut fence = MockFence::new(0, "f0");
        fence.signal(10);
        assert!(!fence.wait(11));
    }

    #[test]
    fn fence_reset_clears_state() {
        let mut fence = MockFence::new(0, "f0");
        fence.signal(99);
        assert!(fence.signaled);
        fence.reset();
        assert!(!fence.signaled);
        assert_eq!(fence.signal_value, 0);
        assert!(!fence.wait(1));
    }

    #[test]
    fn multiple_fences_independent() {
        let mut f1 = MockFence::new(1, "f1");
        let mut f2 = MockFence::new(2, "f2");
        f1.signal(10);
        assert!(f1.wait(10));
        assert!(!f2.wait(1));
        f2.signal(20);
        assert!(f2.wait(20));
    }

    #[test]
    fn fence_monotonic_signal_values() {
        let mut fence = MockFence::new(0, "mono");
        for v in [1, 5, 10, 50, 100] {
            fence.signal(v);
            assert!(fence.wait(v));
        }
    }

    #[test]
    fn timeout_simulation_on_unsignaled_fence() {
        let fence = MockFence::new(0, "timeout");
        let deadline = 3u32;
        let mut timed_out = false;
        for _ in 0..deadline {
            if fence.wait(1) {
                break;
            }
            timed_out = true;
        }
        assert!(timed_out, "should time out on unsignaled fence");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Event Synchronization
// ═══════════════════════════════════════════════════════════════════════════

mod event_synchronization {
    use super::*;

    #[test]
    fn shared_event_initial_value_zero() {
        let event = MockSharedEvent::new(0);
        assert_eq!(event.current_value(), 0);
    }

    #[test]
    fn gpu_signals_cpu_observes() {
        let event = MockSharedEvent::new(0);
        event.signal(5);
        assert_eq!(event.current_value(), 5);
    }

    #[test]
    fn event_value_monotonically_increases() {
        let event = MockSharedEvent::new(0);
        for v in 1..=10 {
            event.signal(v);
            assert_eq!(event.current_value(), v);
        }
    }

    #[test]
    fn listener_fires_when_threshold_reached() {
        let mut event = MockSharedEvent::new(0);
        event.add_listener(5);
        event.add_listener(10);

        event.signal(5);
        let notified = event.poll_listeners();
        assert_eq!(notified, vec![5]);

        event.signal(10);
        let notified = event.poll_listeners();
        assert_eq!(notified, vec![10]);
    }

    #[test]
    fn listener_does_not_fire_below_threshold() {
        let mut event = MockSharedEvent::new(0);
        event.add_listener(10);
        event.signal(9);
        let notified = event.poll_listeners();
        assert!(notified.is_empty());
    }

    #[test]
    fn listener_fires_only_once() {
        let mut event = MockSharedEvent::new(0);
        event.add_listener(3);
        event.signal(3);
        let first = event.poll_listeners();
        assert_eq!(first.len(), 1);
        let second = event.poll_listeners();
        assert!(second.is_empty(), "listener must not fire twice");
    }

    #[test]
    fn multiple_listeners_same_threshold() {
        let mut event = MockSharedEvent::new(0);
        event.add_listener(7);
        event.add_listener(7);
        event.signal(7);
        let notified = event.poll_listeners();
        assert_eq!(notified.len(), 2);
    }

    #[test]
    fn event_overshoot_fires_pending_listeners() {
        let mut event = MockSharedEvent::new(0);
        event.add_listener(2);
        event.add_listener(4);
        event.add_listener(6);
        // Signal past all thresholds in one shot.
        event.signal(100);
        let notified = event.poll_listeners();
        assert_eq!(notified.len(), 3);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Resource Hazard Tracking
// ═══════════════════════════════════════════════════════════════════════════

mod resource_hazard_tracking {
    use super::*;

    fn buf_with_access(id: u64, res: u64, mode: AccessMode) -> MockCommandBuffer {
        let mut b = MockCommandBuffer::new(id, "h");
        b.resource_accesses.push(ResourceAccess {
            resource_id: res,
            mode,
            offset: 0,
            length: 1024,
        });
        b
    }

    #[test]
    fn read_read_no_hazard() {
        let a = buf_with_access(0, 1, AccessMode::Read);
        let b = buf_with_access(1, 1, AccessMode::Read);
        assert!(detect_hazards(&a, &b).is_empty());
    }

    #[test]
    fn write_then_read_is_raw() {
        let a = buf_with_access(0, 1, AccessMode::Write);
        let b = buf_with_access(1, 1, AccessMode::Read);
        let h = detect_hazards(&a, &b);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].1, HazardType::ReadAfterWrite);
    }

    #[test]
    fn read_then_write_is_war() {
        let a = buf_with_access(0, 1, AccessMode::Read);
        let b = buf_with_access(1, 1, AccessMode::Write);
        let h = detect_hazards(&a, &b);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].1, HazardType::WriteAfterRead);
    }

    #[test]
    fn write_then_write_is_waw() {
        let a = buf_with_access(0, 1, AccessMode::Write);
        let b = buf_with_access(1, 1, AccessMode::Write);
        let h = detect_hazards(&a, &b);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].1, HazardType::WriteAfterWrite);
    }

    #[test]
    fn non_overlapping_ranges_no_hazard() {
        let mut a = MockCommandBuffer::new(0, "a");
        a.resource_accesses.push(ResourceAccess {
            resource_id: 1,
            mode: AccessMode::Write,
            offset: 0,
            length: 512,
        });
        let mut b = MockCommandBuffer::new(1, "b");
        b.resource_accesses.push(ResourceAccess {
            resource_id: 1,
            mode: AccessMode::Read,
            offset: 512,
            length: 512,
        });
        assert!(detect_hazards(&a, &b).is_empty());
    }

    #[test]
    fn different_resources_no_hazard() {
        let a = buf_with_access(0, 1, AccessMode::Write);
        let b = buf_with_access(1, 2, AccessMode::Read);
        assert!(detect_hazards(&a, &b).is_empty());
    }

    #[test]
    fn readwrite_vs_read_is_raw() {
        let a = buf_with_access(0, 1, AccessMode::ReadWrite);
        let b = buf_with_access(1, 1, AccessMode::Read);
        let h = detect_hazards(&a, &b);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].1, HazardType::ReadAfterWrite);
    }

    #[test]
    fn multiple_resources_multiple_hazards() {
        let mut a = MockCommandBuffer::new(0, "a");
        a.resource_accesses.push(ResourceAccess {
            resource_id: 1,
            mode: AccessMode::Write,
            offset: 0,
            length: 256,
        });
        a.resource_accesses.push(ResourceAccess {
            resource_id: 2,
            mode: AccessMode::Read,
            offset: 0,
            length: 256,
        });
        let mut b = MockCommandBuffer::new(1, "b");
        b.resource_accesses.push(ResourceAccess {
            resource_id: 1,
            mode: AccessMode::Read,
            offset: 0,
            length: 256,
        });
        b.resource_accesses.push(ResourceAccess {
            resource_id: 2,
            mode: AccessMode::Write,
            offset: 0,
            length: 256,
        });
        let h = detect_hazards(&a, &b);
        assert_eq!(h.len(), 2);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Compute-to-Compute Sync
// ═══════════════════════════════════════════════════════════════════════════

mod compute_to_compute_sync {
    use super::*;

    #[test]
    fn back_to_back_dispatches_ordered() {
        let mut queue = MockCommandQueue::new(0, "compute");
        let ids: Vec<u64> = (0..4)
            .map(|i| queue.submit(MockCommandBuffer::new(i, &format!("dispatch_{i}"))))
            .collect();
        queue.complete_all();
        assert_eq!(queue.completion_order, ids);
    }

    #[test]
    fn shared_resource_fence_required() {
        let mut a = MockCommandBuffer::new(0, "write");
        a.resource_accesses.push(ResourceAccess {
            resource_id: 10,
            mode: AccessMode::Write,
            offset: 0,
            length: 4096,
        });
        let mut b = MockCommandBuffer::new(1, "read");
        b.resource_accesses.push(ResourceAccess {
            resource_id: 10,
            mode: AccessMode::Read,
            offset: 0,
            length: 4096,
        });
        let hazards = detect_hazards(&a, &b);
        assert!(!hazards.is_empty(), "RAW hazard requires fence");
    }

    #[test]
    fn independent_dispatches_no_sync_needed() {
        let a = buf_with_access(0, 1, AccessMode::Write);
        let b = buf_with_access(1, 2, AccessMode::Write);
        assert!(detect_hazards(&a, &b).is_empty());
    }

    fn buf_with_access(id: u64, res: u64, mode: AccessMode) -> MockCommandBuffer {
        let mut b = MockCommandBuffer::new(id, "c");
        b.resource_accesses.push(ResourceAccess {
            resource_id: res,
            mode,
            offset: 0,
            length: 1024,
        });
        b
    }

    #[test]
    fn producer_consumer_chain() {
        let mut graph = DependencyGraph::new();
        // producer(0) → transform(1) → consumer(2)
        for i in 0..3 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        let order = graph.topological_sort().unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn reduction_fan_in_pattern() {
        // 4 producers → 1 reducer
        let mut graph = DependencyGraph::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        for i in 0..4 {
            graph.add_edge(i, 4);
        }
        let order = graph.topological_sort().unwrap();
        let reducer_pos = order.iter().position(|&x| x == 4).unwrap();
        assert_eq!(reducer_pos, 4, "reducer must come last");
    }

    #[test]
    fn scatter_gather_pattern() {
        // scatter(0) → [work_0(1), work_1(2), work_2(3)] → gather(4)
        let mut graph = DependencyGraph::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        graph.add_edge(1, 4);
        graph.add_edge(2, 4);
        graph.add_edge(3, 4);
        let order = graph.topological_sort().unwrap();
        let pos = |id: u64| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(0) < pos(2));
        assert!(pos(0) < pos(3));
        assert!(pos(1) < pos(4));
        assert!(pos(2) < pos(4));
        assert!(pos(3) < pos(4));
    }

    #[test]
    fn fence_guards_compute_output() {
        let mut fence = MockFence::new(0, "compute_fence");
        // Simulate: compute writes, signals fence, then next dispatch reads.
        fence.signal(1);
        assert!(fence.wait(1), "reader must see fence after writer signals");
    }

    #[test]
    fn accumulator_pattern_waw() {
        // Two dispatches writing to the same accumulator buffer.
        let a = buf_with_access(0, 42, AccessMode::Write);
        let b = buf_with_access(1, 42, AccessMode::Write);
        let h = detect_hazards(&a, &b);
        assert!(h.iter().any(|(_, t)| *t == HazardType::WriteAfterWrite));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Render-to-Compute Sync
// ═══════════════════════════════════════════════════════════════════════════

mod render_to_compute_sync {
    use super::*;

    #[test]
    fn fragment_to_compute_barrier_valid() {
        let b = PipelineBarrier {
            src_stage: PipelineStage::Fragment,
            dst_stage: PipelineStage::Compute,
            resource_id: 1,
        };
        assert!(b.is_valid_transition());
    }

    #[test]
    fn compute_to_vertex_barrier_valid() {
        let b = PipelineBarrier {
            src_stage: PipelineStage::Compute,
            dst_stage: PipelineStage::Vertex,
            resource_id: 1,
        };
        assert!(b.is_valid_transition());
    }

    #[test]
    fn same_stage_barrier_invalid() {
        let b = PipelineBarrier {
            src_stage: PipelineStage::Compute,
            dst_stage: PipelineStage::Compute,
            resource_id: 1,
        };
        assert!(!b.is_valid_transition());
    }

    #[test]
    fn blit_to_compute_transition() {
        let b = PipelineBarrier {
            src_stage: PipelineStage::Blit,
            dst_stage: PipelineStage::Compute,
            resource_id: 5,
        };
        assert!(b.is_valid_transition());
    }

    #[test]
    fn render_output_to_compute_input_hazard() {
        // Fragment writes a texture, compute reads it → RAW.
        let mut render = MockCommandBuffer::new(0, "render");
        render.resource_accesses.push(ResourceAccess {
            resource_id: 100,
            mode: AccessMode::Write,
            offset: 0,
            length: 4096,
        });
        let mut compute = MockCommandBuffer::new(1, "compute");
        compute.resource_accesses.push(ResourceAccess {
            resource_id: 100,
            mode: AccessMode::Read,
            offset: 0,
            length: 4096,
        });
        let h = detect_hazards(&render, &compute);
        assert!(h.iter().any(|(_, t)| *t == HazardType::ReadAfterWrite));
    }

    #[test]
    fn vertex_to_fragment_to_compute_chain() {
        let barriers = vec![
            PipelineBarrier {
                src_stage: PipelineStage::Vertex,
                dst_stage: PipelineStage::Fragment,
                resource_id: 1,
            },
            PipelineBarrier {
                src_stage: PipelineStage::Fragment,
                dst_stage: PipelineStage::Compute,
                resource_id: 1,
            },
        ];
        for b in &barriers {
            assert!(b.is_valid_transition());
        }
    }

    #[test]
    fn compute_feedback_to_vertex() {
        // Compute writes vertex buffer, next frame vertex stage reads it.
        let b = PipelineBarrier {
            src_stage: PipelineStage::Compute,
            dst_stage: PipelineStage::Vertex,
            resource_id: 77,
        };
        assert!(b.is_valid_transition());
    }

    #[test]
    fn all_cross_stage_transitions_valid() {
        let stages = [
            PipelineStage::Vertex,
            PipelineStage::Fragment,
            PipelineStage::Compute,
            PipelineStage::Blit,
        ];
        let mut valid_count = 0;
        for &src in &stages {
            for &dst in &stages {
                let b = PipelineBarrier { src_stage: src, dst_stage: dst, resource_id: 0 };
                if src != dst {
                    assert!(b.is_valid_transition());
                    valid_count += 1;
                }
            }
        }
        // 4 stages × 3 other stages = 12.
        assert_eq!(valid_count, 12);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Multi-Queue Synchronization
// ═══════════════════════════════════════════════════════════════════════════

mod multi_queue_synchronization {
    use super::*;

    #[test]
    fn two_queues_independent_completion() {
        let mut q1 = MockCommandQueue::new(0, "queue_0");
        let mut q2 = MockCommandQueue::new(1, "queue_1");
        q1.submit(MockCommandBuffer::new(0, "q1_cmd"));
        q2.submit(MockCommandBuffer::new(1, "q2_cmd"));
        q1.complete_all();
        q2.complete_all();
        assert!(q1.is_completed(0));
        assert!(q2.is_completed(1));
    }

    #[test]
    fn cross_queue_event_sync() {
        let event = MockSharedEvent::new(0);
        // Queue A signals event, Queue B waits.
        event.signal(1);
        assert!(event.current_value() >= 1, "queue B can proceed");
    }

    #[test]
    fn cross_queue_fence_handoff() {
        let mut fence = MockFence::new(0, "cross_q");
        // Queue A completes and signals.
        fence.signal(1);
        // Queue B waits.
        assert!(fence.wait(1));
    }

    #[test]
    fn parallel_queues_unordered_ids() {
        let mut q1 = MockCommandQueue::new(0, "q1");
        let mut q2 = MockCommandQueue::new(1, "q2");
        // Submit in interleaved order: q1 gets even, q2 gets odd.
        for i in 0..4u64 {
            if i % 2 == 0 {
                q1.submit(MockCommandBuffer::new(i, "even"));
            } else {
                q2.submit(MockCommandBuffer::new(i, "odd"));
            }
        }
        q1.complete_all();
        q2.complete_all();
        assert_eq!(q1.completion_order.len(), 2);
        assert_eq!(q2.completion_order.len(), 2);
    }

    #[test]
    fn queue_starvation_detection() {
        let mut q1 = MockCommandQueue::new(0, "busy");
        let q2 = MockCommandQueue::new(1, "starved");
        for i in 0..10 {
            q1.submit(MockCommandBuffer::new(i, "work"));
        }
        // q2 has nothing.
        q1.complete_all();
        assert_eq!(q1.completion_order.len(), 10);
        assert!(q2.completion_order.is_empty(), "starved queue has no work");
    }

    #[test]
    fn event_coordinates_three_queues() {
        let mut event = MockSharedEvent::new(0);
        event.add_listener(1); // queue B waits for 1
        event.add_listener(2); // queue C waits for 2

        // Queue A signals 1.
        event.signal(1);
        let n1 = event.poll_listeners();
        assert_eq!(n1, vec![1]);

        // Queue A signals 2.
        event.signal(2);
        let n2 = event.poll_listeners();
        assert_eq!(n2, vec![2]);
    }

    #[test]
    fn multi_queue_dependency_graph() {
        let mut graph = DependencyGraph::new();
        // q1: A(0)→B(1), q2: C(2)→D(3), cross: B(1)→D(3)
        for i in 0..4 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1);
        graph.add_edge(2, 3);
        graph.add_edge(1, 3); // cross-queue dep
        let order = graph.topological_sort().unwrap();
        let pos = |id: u64| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(1) < pos(3), "B must complete before D");
    }

    #[test]
    fn separate_queues_no_implicit_ordering() {
        // Two independent queues with independent buffers have no hazards.
        let a = {
            let mut b = MockCommandBuffer::new(0, "q1");
            b.resource_accesses.push(ResourceAccess {
                resource_id: 1,
                mode: AccessMode::Write,
                offset: 0,
                length: 1024,
            });
            b
        };
        let b = {
            let mut b = MockCommandBuffer::new(1, "q2");
            b.resource_accesses.push(ResourceAccess {
                resource_id: 2,
                mode: AccessMode::Write,
                offset: 0,
                length: 1024,
            });
            b
        };
        assert!(detect_hazards(&a, &b).is_empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Atomic Operations
// ═══════════════════════════════════════════════════════════════════════════

mod atomic_operations {
    use super::*;

    #[test]
    fn atomic_counter_initial_value() {
        let c = AtomicCounter::new(0);
        assert_eq!(c.load(), 0);
    }

    #[test]
    fn atomic_fetch_add() {
        let c = AtomicCounter::new(0);
        let prev = c.fetch_add(5);
        assert_eq!(prev, 0);
        assert_eq!(c.load(), 5);
    }

    #[test]
    fn atomic_compare_exchange_success() {
        let c = AtomicCounter::new(10);
        let res = c.compare_exchange(10, 20);
        assert_eq!(res, Ok(10));
        assert_eq!(c.load(), 20);
    }

    #[test]
    fn atomic_compare_exchange_failure() {
        let c = AtomicCounter::new(10);
        let res = c.compare_exchange(99, 20);
        assert_eq!(res, Err(10));
        assert_eq!(c.load(), 10);
    }

    #[test]
    fn atomic_fetch_max() {
        let c = AtomicCounter::new(5);
        c.fetch_max(10);
        assert_eq!(c.load(), 10);
        c.fetch_max(3);
        assert_eq!(c.load(), 10);
    }

    #[test]
    fn atomic_fetch_min() {
        let c = AtomicCounter::new(10);
        c.fetch_min(3);
        assert_eq!(c.load(), 3);
        c.fetch_min(7);
        assert_eq!(c.load(), 3);
    }

    #[test]
    fn concurrent_fetch_add_simulated() {
        let c = AtomicCounter::new(0);
        let n = 100u64;
        for _ in 0..n {
            c.fetch_add(1);
        }
        assert_eq!(c.load(), n);
    }

    #[test]
    fn cas_spin_loop_converges() {
        let c = AtomicCounter::new(0);
        let target = 42u64;
        let mut attempts = 0u32;
        loop {
            let current = c.load();
            if current == target {
                break;
            }
            match c.compare_exchange(current, current + 1) {
                Ok(_) => {}
                Err(_) => {} // retry
            }
            attempts += 1;
            assert!(attempts < 1000, "spin loop must converge");
        }
        assert_eq!(c.load(), target);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Buffer Coherence
// ═══════════════════════════════════════════════════════════════════════════

mod buffer_coherence {
    use super::*;

    #[test]
    fn initial_buffer_not_stale() {
        let buf = CoherentBuffer::new(256);
        assert!(!buf.is_cpu_stale());
        assert!(!buf.is_gpu_stale());
    }

    #[test]
    fn gpu_write_makes_cpu_stale() {
        let buf = CoherentBuffer::new(256);
        buf.gpu_write(0, &[1, 2, 3, 4]);
        assert!(buf.is_cpu_stale());
        assert!(!buf.is_gpu_stale());
    }

    #[test]
    fn cpu_write_makes_gpu_stale() {
        let buf = CoherentBuffer::new(256);
        buf.cpu_write(0, &[5, 6, 7, 8]);
        assert!(buf.is_gpu_stale());
        assert!(!buf.is_cpu_stale());
    }

    #[test]
    fn flush_gpu_cache_clears_cpu_staleness() {
        let buf = CoherentBuffer::new(256);
        buf.gpu_write(0, &[1]);
        assert!(buf.is_cpu_stale());
        buf.flush_gpu_cache();
        assert!(!buf.is_cpu_stale());
    }

    #[test]
    fn invalidate_gpu_cache_clears_gpu_staleness() {
        let buf = CoherentBuffer::new(256);
        buf.cpu_write(0, &[1]);
        assert!(buf.is_gpu_stale());
        buf.invalidate_gpu_cache();
        assert!(!buf.is_gpu_stale());
    }

    #[test]
    fn cpu_read_returns_latest_data() {
        let buf = CoherentBuffer::new(16);
        buf.cpu_write(0, &[0xAA, 0xBB]);
        let data = buf.cpu_read(0, 2);
        assert_eq!(data, vec![0xAA, 0xBB]);
    }

    #[test]
    fn gpu_write_then_cpu_read_sees_data() {
        let buf = CoherentBuffer::new(16);
        buf.gpu_write(4, &[0xDE, 0xAD]);
        buf.flush_gpu_cache();
        let data = buf.cpu_read(4, 2);
        assert_eq!(data, vec![0xDE, 0xAD]);
    }

    #[test]
    fn coherence_round_trip() {
        let buf = CoherentBuffer::new(64);
        // CPU writes, GPU reads (need invalidate), GPU writes back, CPU reads (need flush).
        buf.cpu_write(0, &[1, 2, 3, 4]);
        assert!(buf.is_gpu_stale());
        buf.invalidate_gpu_cache();
        assert!(!buf.is_gpu_stale());

        buf.gpu_write(0, &[10, 20, 30, 40]);
        assert!(buf.is_cpu_stale());
        buf.flush_gpu_cache();
        assert!(!buf.is_cpu_stale());

        let data = buf.cpu_read(0, 4);
        assert_eq!(data, vec![10, 20, 30, 40]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Execution Dependencies
// ═══════════════════════════════════════════════════════════════════════════

mod execution_dependencies {
    use super::*;

    #[test]
    fn linear_chain_resolves() {
        let mut g = DependencyGraph::new();
        for i in 0..5 {
            g.add_node(i);
        }
        for i in 0..4 {
            g.add_edge(i, i + 1);
        }
        assert!(!g.has_cycle());
        assert_eq!(g.topological_sort().unwrap(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn simple_cycle_detected() {
        let mut g = DependencyGraph::new();
        g.add_node(0);
        g.add_node(1);
        g.add_edge(0, 1);
        g.add_edge(1, 0);
        assert!(g.has_cycle());
    }

    #[test]
    fn self_loop_detected() {
        let mut g = DependencyGraph::new();
        g.add_node(0);
        g.add_edge(0, 0);
        assert!(g.has_cycle());
    }

    #[test]
    fn triangle_cycle_detected() {
        let mut g = DependencyGraph::new();
        for i in 0..3 {
            g.add_node(i);
        }
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        assert!(g.has_cycle());
    }

    #[test]
    fn disconnected_components_resolve() {
        let mut g = DependencyGraph::new();
        // Component 1: 0→1
        g.add_node(0);
        g.add_node(1);
        g.add_edge(0, 1);
        // Component 2: 2→3
        g.add_node(2);
        g.add_node(3);
        g.add_edge(2, 3);
        let order = g.topological_sort().unwrap();
        assert_eq!(order.len(), 4);
        let pos = |id: u64| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(2) < pos(3));
    }

    #[test]
    fn wide_fan_out_no_cycle() {
        let mut g = DependencyGraph::new();
        g.add_node(0);
        for i in 1..=10 {
            g.add_node(i);
            g.add_edge(0, i);
        }
        assert!(!g.has_cycle());
        let order = g.topological_sort().unwrap();
        assert_eq!(order[0], 0);
    }

    #[test]
    fn single_node_no_cycle() {
        let mut g = DependencyGraph::new();
        g.add_node(42);
        assert!(!g.has_cycle());
        assert_eq!(g.topological_sort().unwrap(), vec![42]);
    }

    #[test]
    fn complex_dag_with_convergence() {
        // 0→1, 0→2, 1→3, 2→3, 3→4
        let mut g = DependencyGraph::new();
        for i in 0..5 {
            g.add_node(i);
        }
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 3);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        assert!(!g.has_cycle());
        let order = g.topological_sort().unwrap();
        let pos = |id: u64| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(0) < pos(1));
        assert!(pos(0) < pos(2));
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(3));
        assert!(pos(3) < pos(4));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Apple Silicon Sync
// ═══════════════════════════════════════════════════════════════════════════

mod apple_silicon_sync {
    use super::*;

    /// Apple Silicon UMA means CPU and GPU share physical memory.
    /// Simulate: after GPU write + fence, CPU sees the data without an explicit copy.
    #[test]
    fn uma_coherence_after_fence() {
        let buf = CoherentBuffer::new(128);
        let mut fence = MockFence::new(0, "uma");
        buf.gpu_write(0, &[0xCA, 0xFE]);
        fence.signal(1);
        // On UMA, flush is conceptually free but still needed for ordering.
        assert!(fence.wait(1));
        buf.flush_gpu_cache();
        assert_eq!(buf.cpu_read(0, 2), vec![0xCA, 0xFE]);
    }

    /// Tile memory is local to a tile; simulate that tile-local data must be
    /// flushed before being visible outside the tile.
    #[test]
    fn tile_memory_flush_required() {
        let tile_buf = CoherentBuffer::new(32);
        tile_buf.gpu_write(0, &[1, 2, 3, 4]);
        assert!(tile_buf.is_cpu_stale(), "tile output not yet visible");
        tile_buf.flush_gpu_cache();
        assert!(!tile_buf.is_cpu_stale());
    }

    /// simdgroup barrier: all threads in a SIMD group must reach the barrier
    /// before any proceed. Simulate with a counter that reaches group width.
    #[test]
    fn simdgroup_barrier_all_threads_reach() {
        let simd_width = 32u64;
        let counter = AtomicCounter::new(0);
        for _ in 0..simd_width {
            counter.fetch_add(1);
        }
        assert_eq!(counter.load(), simd_width, "all threads reached barrier");
    }

    /// threadgroup barrier: all threads in a threadgroup synchronize.
    #[test]
    fn threadgroup_barrier_synchronization() {
        let threadgroup_size = 256u64;
        let counter = AtomicCounter::new(0);
        for _ in 0..threadgroup_size {
            counter.fetch_add(1);
        }
        assert_eq!(counter.load(), threadgroup_size);
    }

    #[test]
    fn uma_no_explicit_copy_needed() {
        // On UMA architectures, the buffer data pointer is the same for CPU & GPU.
        let buf = CoherentBuffer::new(64);
        buf.cpu_write(0, &[42]);
        // GPU "reads" the same physical memory — invalidate makes it logically fresh.
        buf.invalidate_gpu_cache();
        assert!(!buf.is_gpu_stale());
        // Simulate GPU processing and writing result.
        buf.gpu_write(8, &[84]);
        buf.flush_gpu_cache();
        assert_eq!(buf.cpu_read(0, 1), vec![42]);
        assert_eq!(buf.cpu_read(8, 1), vec![84]);
    }

    #[test]
    fn tile_memory_size_within_limit() {
        // Apple Silicon: max 32 KB threadgroup memory.
        let max_tile_memory: usize = 32 * 1024;
        let requested: usize = 16 * 1024;
        assert!(requested <= max_tile_memory, "tile memory request within limit");
    }

    #[test]
    fn simdgroup_reduction_produces_single_result() {
        let simd_width = 32usize;
        let values: Vec<f32> = (0..simd_width).map(|i| i as f32).collect();
        let sum: f32 = values.iter().sum();
        let expected = (simd_width as f32 - 1.0) * simd_width as f32 / 2.0;
        assert!((sum - expected).abs() < 1e-6);
    }

    #[test]
    fn event_cpu_gpu_round_trip_uma() {
        let event = MockSharedEvent::new(0);
        // CPU signals "data ready".
        event.signal(1);
        assert!(event.current_value() >= 1, "GPU can see CPU signal on UMA");
        // GPU signals "done".
        event.signal(2);
        assert!(event.current_value() >= 2, "CPU can see GPU signal on UMA");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. Regression Detection
// ═══════════════════════════════════════════════════════════════════════════

mod regression_detection {
    use super::*;

    #[test]
    fn sync_overhead_within_budget() {
        let mut t = SyncOverheadTracker::new(10, 20);
        for _ in 0..10 {
            t.record_fence();
        }
        for _ in 0..20 {
            t.record_barrier();
        }
        assert!(t.is_within_budget());
    }

    #[test]
    fn fence_count_exceeds_budget() {
        let mut t = SyncOverheadTracker::new(5, 100);
        for _ in 0..6 {
            t.record_fence();
        }
        assert!(!t.is_within_budget());
    }

    #[test]
    fn barrier_count_exceeds_budget() {
        let mut t = SyncOverheadTracker::new(100, 3);
        for _ in 0..4 {
            t.record_barrier();
        }
        assert!(!t.is_within_budget());
    }

    #[test]
    fn total_sync_ops_accumulates() {
        let mut t = SyncOverheadTracker::new(100, 100);
        t.record_fence();
        t.record_fence();
        t.record_event_signal();
        t.record_barrier();
        t.record_barrier();
        t.record_barrier();
        assert_eq!(t.total_sync_ops(), 6);
    }

    #[test]
    fn stall_detection_via_fence_timeout() {
        let fence = MockFence::new(0, "stall_check");
        let max_polls = 5;
        let mut polls = 0;
        while !fence.wait(1) && polls < max_polls {
            polls += 1;
        }
        assert_eq!(polls, max_polls, "detected stall: fence never signaled");
    }

    #[test]
    fn zero_budget_rejects_any_sync() {
        let mut t = SyncOverheadTracker::new(0, 0);
        t.record_fence();
        assert!(!t.is_within_budget());
    }

    #[test]
    fn event_signal_does_not_count_against_fence_budget() {
        let mut t = SyncOverheadTracker::new(0, 100);
        t.record_event_signal();
        t.record_event_signal();
        // Fence budget is 0 but we only added events — still within budget.
        assert!(t.is_within_budget());
    }

    #[test]
    fn regression_overhead_scales_linearly() {
        // For N dispatches we expect at most N-1 fences (pairwise).
        let dispatches = 16u32;
        let max_fences = dispatches - 1;
        let mut t = SyncOverheadTracker::new(max_fences, dispatches * 2);
        for _ in 0..max_fences {
            t.record_fence();
        }
        for _ in 0..dispatches {
            t.record_barrier();
            t.record_barrier();
        }
        assert!(t.is_within_budget());
    }
}
