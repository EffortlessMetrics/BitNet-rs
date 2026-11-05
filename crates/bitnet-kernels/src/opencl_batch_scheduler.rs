//! Intelligent batch scheduling for inference serving.
//!
//! Provides scheduling policies (`FCFS`, `ShortestJobFirst`, `PriorityBased`,
//! `FairShare`) to group incoming inference requests into GPU-sized batches that
//! respect memory and compute budgets.  A `PreemptionManager` can evict low-
//! priority work when high-priority requests arrive, an `AdmissionController`
//! rejects work that would exceed queue-depth or memory limits, and a
//! `DeadlineScheduler` orders work to meet latency SLOs.
//!
//! All implementations are CPU-reference (no OpenCL runtime required) and are
//! suitable for correctness testing and non-GPU environments.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Request & slot types
// ---------------------------------------------------------------------------

/// Unique identifier for a batch request.
pub type RequestId = u64;

/// Priority level for a request (higher numeric value = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(pub u32);

impl Priority {
    pub const LOW: Self = Self(0);
    pub const NORMAL: Self = Self(1);
    pub const HIGH: Self = Self(2);
    pub const CRITICAL: Self = Self(3);
}

/// A single inference request awaiting scheduling.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub id: RequestId,
    pub priority: Priority,
    /// Upper bound on the number of tokens this request will generate.
    pub max_tokens: usize,
    /// Estimated GPU memory (bytes) required for this request's KV cache.
    pub estimated_memory_bytes: u64,
    /// When the request entered the system.
    pub arrival_time: Instant,
    /// Optional latency deadline (absolute).
    pub deadline: Option<Instant>,
}

impl BatchRequest {
    pub fn new(id: RequestId, priority: Priority, max_tokens: usize) -> Self {
        // Rough heuristic: 2 KiB per token for KV cache.
        let estimated_memory_bytes = (max_tokens as u64) * 2048;
        Self {
            id,
            priority,
            max_tokens,
            estimated_memory_bytes,
            arrival_time: Instant::now(),
            deadline: None,
        }
    }

    /// Create a request with an explicit arrival time (useful for testing).
    pub fn with_arrival(mut self, arrival: Instant) -> Self {
        self.arrival_time = arrival;
        self
    }

    /// Attach a latency deadline.
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Override estimated memory.
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.estimated_memory_bytes = bytes;
        self
    }
}

/// A slot allocated inside a scheduled batch.
#[derive(Debug, Clone)]
pub struct BatchSlot {
    pub request_id: RequestId,
    pub slot_index: usize,
    /// Memory reserved for this slot (bytes).
    pub reserved_memory_bytes: u64,
    pub max_tokens: usize,
}

// ---------------------------------------------------------------------------
// Scheduling policies
// ---------------------------------------------------------------------------

/// Scheduling policy used by [`BatchScheduler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-Come First-Served — strict arrival-order.
    Fcfs,
    /// Shortest-Job-First — smallest `max_tokens` first.
    ShortestJobFirst,
    /// Priority-Based — highest [`Priority`] first, FCFS within same priority.
    PriorityBased,
    /// Fair-Share — round-robin across priority levels, then FCFS within each.
    FairShare,
}

// ---------------------------------------------------------------------------
// Memory budget
// ---------------------------------------------------------------------------

/// Tracks available GPU memory for batch sizing.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Total GPU memory available for inference batches (bytes).
    pub total_bytes: u64,
    /// Currently allocated bytes.
    pub allocated_bytes: u64,
}

impl MemoryBudget {
    pub fn new(total_bytes: u64) -> Self {
        Self { total_bytes, allocated_bytes: 0 }
    }

    /// Remaining free memory.
    pub fn available(&self) -> u64 {
        self.total_bytes.saturating_sub(self.allocated_bytes)
    }

    /// Try to allocate `bytes`; returns `true` on success.
    pub fn try_allocate(&mut self, bytes: u64) -> bool {
        if bytes <= self.available() {
            self.allocated_bytes += bytes;
            true
        } else {
            false
        }
    }

    /// Release previously-allocated memory.
    pub fn release(&mut self, bytes: u64) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
    }

    /// Utilization ratio `[0.0, 1.0]`.
    pub fn utilization(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.allocated_bytes as f64 / self.total_bytes as f64
    }
}

// ---------------------------------------------------------------------------
// Batch statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics for the scheduler.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total batches formed.
    pub batches_formed: u64,
    /// Total requests scheduled.
    pub requests_scheduled: u64,
    /// Total requests rejected by admission control.
    pub requests_rejected: u64,
    /// Total preemptions performed.
    pub preemptions: u64,
    /// Cumulative wait time across all scheduled requests.
    pub total_wait: Duration,
    /// Total requests that met their deadline.
    pub deadlines_met: u64,
    /// Total requests that missed their deadline.
    pub deadlines_missed: u64,
}

impl BatchStats {
    /// Average wait time per scheduled request.
    pub fn avg_wait(&self) -> Duration {
        if self.requests_scheduled == 0 {
            return Duration::ZERO;
        }
        self.total_wait / self.requests_scheduled as u32
    }

    /// Batch utilization: average requests per batch.
    pub fn avg_batch_size(&self) -> f64 {
        if self.batches_formed == 0 {
            return 0.0;
        }
        self.requests_scheduled as f64 / self.batches_formed as f64
    }
}

// ---------------------------------------------------------------------------
// Admission controller
// ---------------------------------------------------------------------------

/// Rejects requests when the system is overloaded.
#[derive(Debug, Clone)]
pub struct AdmissionController {
    /// Maximum number of pending requests allowed.
    pub max_queue_depth: usize,
    /// Maximum total memory across all pending requests.
    pub max_pending_memory_bytes: u64,
}

impl AdmissionController {
    pub fn new(max_queue_depth: usize, max_pending_memory_bytes: u64) -> Self {
        Self { max_queue_depth, max_pending_memory_bytes }
    }

    /// Returns `true` if the request should be admitted.
    pub fn should_admit(
        &self,
        current_queue_len: usize,
        current_pending_memory: u64,
        request: &BatchRequest,
    ) -> bool {
        if current_queue_len >= self.max_queue_depth {
            return false;
        }
        if current_pending_memory + request.estimated_memory_bytes > self.max_pending_memory_bytes {
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Preemption manager
// ---------------------------------------------------------------------------

/// Manages preemption of low-priority requests.
#[derive(Debug)]
pub struct PreemptionManager {
    /// Minimum priority difference required to trigger preemption.
    pub priority_gap: u32,
}

impl PreemptionManager {
    pub fn new(priority_gap: u32) -> Self {
        Self { priority_gap }
    }

    /// Identify requests that should be preempted to make room for `incoming`.
    ///
    /// Returns indices (into `running`) of requests to evict, ordered from
    /// lowest priority first.  Eviction stops once enough memory is freed.
    pub fn select_victims(
        &self,
        incoming: &BatchRequest,
        running: &[BatchSlot],
        running_requests: &[BatchRequest],
        needed_bytes: u64,
    ) -> Vec<usize> {
        // Collect candidates whose priority is sufficiently below incoming.
        let mut candidates: Vec<(usize, Priority, u64)> = running
            .iter()
            .zip(running_requests.iter())
            .enumerate()
            .filter(|(_, (_, req))| incoming.priority.0 >= req.priority.0 + self.priority_gap)
            .map(|(i, (slot, req))| (i, req.priority, slot.reserved_memory_bytes))
            .collect();

        // Sort by priority ascending (lowest first), stable.
        candidates.sort_by_key(|&(_, prio, _)| prio);

        let mut freed = 0u64;
        let mut victims = Vec::new();
        for (idx, _, mem) in &candidates {
            if freed >= needed_bytes {
                break;
            }
            victims.push(*idx);
            freed += mem;
        }
        victims
    }
}

// ---------------------------------------------------------------------------
// Deadline scheduler
// ---------------------------------------------------------------------------

/// Sorts requests by their deadline (Earliest-Deadline-First) to meet SLOs.
#[derive(Debug)]
pub struct DeadlineScheduler;

impl DeadlineScheduler {
    /// Sort `requests` so that those with the earliest deadline come first.
    /// Requests without a deadline are placed after all deadline-bearing ones.
    pub fn order(requests: &mut [BatchRequest]) {
        requests.sort_by(|a, b| match (&a.deadline, &b.deadline) {
            (Some(da), Some(db)) => da.cmp(db),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.arrival_time.cmp(&b.arrival_time),
        });
    }

    /// Check which requests in a completed batch met their deadline.
    pub fn evaluate(requests: &[BatchRequest], completion_time: Instant) -> (u64, u64) {
        let mut met = 0u64;
        let mut missed = 0u64;
        for req in requests {
            if let Some(dl) = req.deadline {
                if completion_time <= dl {
                    met += 1;
                } else {
                    missed += 1;
                }
            }
        }
        (met, missed)
    }
}

// ---------------------------------------------------------------------------
// Core batch scheduler
// ---------------------------------------------------------------------------

/// Schedules inference requests into batches respecting memory and compute
/// budgets.
#[derive(Debug)]
pub struct BatchScheduler {
    policy: SchedulingPolicy,
    max_batch_size: usize,
    memory: MemoryBudget,
    admission: AdmissionController,
    preemption: PreemptionManager,
    queue: VecDeque<BatchRequest>,
    stats: BatchStats,
}

impl BatchScheduler {
    /// Create a new scheduler.
    pub fn new(
        policy: SchedulingPolicy,
        max_batch_size: usize,
        memory: MemoryBudget,
        admission: AdmissionController,
        preemption: PreemptionManager,
    ) -> Self {
        Self {
            policy,
            max_batch_size,
            memory,
            admission,
            preemption,
            queue: VecDeque::new(),
            stats: BatchStats::default(),
        }
    }

    // -- convenience builder ------------------------------------------------

    /// Quick builder with sensible defaults.
    pub fn with_defaults(policy: SchedulingPolicy, max_batch_size: usize) -> Self {
        Self::new(
            policy,
            max_batch_size,
            MemoryBudget::new(16 * 1024 * 1024 * 1024), // 16 GiB
            AdmissionController::new(1024, 16 * 1024 * 1024 * 1024),
            PreemptionManager::new(2),
        )
    }

    // -- accessors ----------------------------------------------------------

    pub fn policy(&self) -> SchedulingPolicy {
        self.policy
    }

    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }

    pub fn memory(&self) -> &MemoryBudget {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut MemoryBudget {
        &mut self.memory
    }

    // -- submission ---------------------------------------------------------

    /// Submit a request for scheduling.
    ///
    /// Returns `Ok(())` if admitted, `Err(request)` if rejected by admission
    /// control.
    pub fn submit(&mut self, request: BatchRequest) -> Result<(), BatchRequest> {
        let pending_mem: u64 = self.queue.iter().map(|r| r.estimated_memory_bytes).sum();
        if !self.admission.should_admit(self.queue.len(), pending_mem, &request) {
            self.stats.requests_rejected += 1;
            return Err(request);
        }
        self.queue.push_back(request);
        Ok(())
    }

    // -- scheduling ---------------------------------------------------------

    /// Form the next batch of up to `max_batch_size` requests that fit in the
    /// current memory budget, ordered according to the active policy.
    ///
    /// Returns `(slots, scheduled_requests)`. Callers should later call
    /// [`complete_batch`](Self::complete_batch) to release memory.
    pub fn schedule_batch(&mut self) -> (Vec<BatchSlot>, Vec<BatchRequest>) {
        if self.queue.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // 1. Sort the queue according to policy.
        let mut pending: Vec<BatchRequest> = self.queue.drain(..).collect();
        Self::sort_by_policy(&mut pending, self.policy);

        let now = Instant::now();
        let mut slots = Vec::new();
        let mut scheduled = Vec::new();
        let mut remaining = Vec::new();

        for req in pending {
            if slots.len() >= self.max_batch_size {
                remaining.push(req);
                continue;
            }
            if !self.memory.try_allocate(req.estimated_memory_bytes) {
                remaining.push(req);
                continue;
            }
            let slot = BatchSlot {
                request_id: req.id,
                slot_index: slots.len(),
                reserved_memory_bytes: req.estimated_memory_bytes,
                max_tokens: req.max_tokens,
            };
            self.stats.total_wait += now.duration_since(req.arrival_time);
            self.stats.requests_scheduled += 1;
            slots.push(slot);
            scheduled.push(req);
        }

        // Put un-scheduled requests back.
        self.queue = VecDeque::from(remaining);

        if !slots.is_empty() {
            self.stats.batches_formed += 1;
        }

        (slots, scheduled)
    }

    /// Release memory for a completed batch and record deadline statistics.
    pub fn complete_batch(
        &mut self,
        slots: &[BatchSlot],
        requests: &[BatchRequest],
        completion_time: Instant,
    ) {
        for slot in slots {
            self.memory.release(slot.reserved_memory_bytes);
        }
        let (met, missed) = DeadlineScheduler::evaluate(requests, completion_time);
        self.stats.deadlines_met += met;
        self.stats.deadlines_missed += missed;
    }

    // -- preemption ---------------------------------------------------------

    /// Attempt to preempt running work to admit a high-priority `incoming`
    /// request.
    ///
    /// Returns the indices of slots to evict.  The caller is responsible for
    /// actually stopping those slots and calling
    /// [`release_preempted`](Self::release_preempted).
    pub fn try_preempt(
        &self,
        incoming: &BatchRequest,
        running_slots: &[BatchSlot],
        running_requests: &[BatchRequest],
    ) -> Vec<usize> {
        let needed = incoming.estimated_memory_bytes.saturating_sub(self.memory.available());
        if needed == 0 {
            // Enough memory already — no preemption needed.
            return Vec::new();
        }
        self.preemption.select_victims(incoming, running_slots, running_requests, needed)
    }

    /// Release memory from preempted slots and bump the preemption counter.
    pub fn release_preempted(&mut self, slots: &[BatchSlot]) {
        for slot in slots {
            self.memory.release(slot.reserved_memory_bytes);
            self.stats.preemptions += 1;
        }
    }

    // -- internal helpers ---------------------------------------------------

    fn sort_by_policy(requests: &mut [BatchRequest], policy: SchedulingPolicy) {
        match policy {
            SchedulingPolicy::Fcfs => {
                // Stable sort by arrival time.
                requests.sort_by(|a, b| a.arrival_time.cmp(&b.arrival_time));
            }
            SchedulingPolicy::ShortestJobFirst => {
                requests.sort_by_key(|r| r.max_tokens);
            }
            SchedulingPolicy::PriorityBased => {
                // Highest priority first; within same priority, FCFS.
                requests.sort_by(|a, b| {
                    b.priority.cmp(&a.priority).then_with(|| a.arrival_time.cmp(&b.arrival_time))
                });
            }
            SchedulingPolicy::FairShare => {
                // Round-robin across priority tiers, FCFS within each.
                // Implemented as interleaved merge of per-priority queues.
                Self::fair_share_sort(requests);
            }
        }
    }

    /// Interleave requests from different priority levels in round-robin order.
    fn fair_share_sort(requests: &mut [BatchRequest]) {
        use std::collections::BTreeMap;

        // Bucket by priority (descending order via BTreeMap reversed).
        let mut buckets: BTreeMap<std::cmp::Reverse<u32>, VecDeque<usize>> = BTreeMap::new();
        for (i, req) in requests.iter().enumerate() {
            buckets.entry(std::cmp::Reverse(req.priority.0)).or_default().push_back(i);
        }

        let mut order: Vec<usize> = Vec::with_capacity(requests.len());
        loop {
            let mut progress = false;
            for bucket in buckets.values_mut() {
                if let Some(idx) = bucket.pop_front() {
                    order.push(idx);
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }

        // Apply the permutation in-place.
        let mut sorted: Vec<BatchRequest> =
            order.into_iter().map(|i| requests[i].clone()).collect();
        requests.swap_with_slice(&mut sorted);
    }
}

// ===========================================================================
// Tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn req_at(
        id: RequestId,
        prio: Priority,
        max_tokens: usize,
        base: Instant,
        offset: Duration,
    ) -> BatchRequest {
        BatchRequest::new(id, prio, max_tokens).with_arrival(base + offset)
    }

    fn default_scheduler(policy: SchedulingPolicy, max_batch: usize) -> BatchScheduler {
        BatchScheduler::with_defaults(policy, max_batch)
    }

    // ── FCFS ordering ──────────────────────────────────────────────

    #[test]
    fn fcfs_orders_by_arrival() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::Fcfs, 10);
        for i in 0..5 {
            let r = req_at(i, Priority::NORMAL, 32, base, Duration::from_millis(i * 10));
            sched.submit(r).unwrap();
        }
        let (slots, _reqs) = sched.schedule_batch();
        let ids: Vec<u64> = slots.iter().map(|s| s.request_id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn fcfs_ignores_priority() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::Fcfs, 10);
        sched.submit(req_at(1, Priority::LOW, 32, base, Duration::from_millis(0))).unwrap();
        sched.submit(req_at(2, Priority::CRITICAL, 32, base, Duration::from_millis(10))).unwrap();
        let (slots, _) = sched.schedule_batch();
        assert_eq!(slots[0].request_id, 1);
        assert_eq!(slots[1].request_id, 2);
    }

    #[test]
    fn fcfs_empty_queue_returns_empty() {
        let mut sched = default_scheduler(SchedulingPolicy::Fcfs, 10);
        let (slots, reqs) = sched.schedule_batch();
        assert!(slots.is_empty());
        assert!(reqs.is_empty());
    }

    #[test]
    fn fcfs_single_request() {
        let mut sched = default_scheduler(SchedulingPolicy::Fcfs, 10);
        sched.submit(BatchRequest::new(42, Priority::NORMAL, 16)).unwrap();
        let (slots, reqs) = sched.schedule_batch();
        assert_eq!(slots.len(), 1);
        assert_eq!(reqs.len(), 1);
        assert_eq!(slots[0].request_id, 42);
    }

    #[test]
    fn fcfs_preserves_order_for_many() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::Fcfs, 100);
        for i in 0..50 {
            sched
                .submit(req_at(i, Priority::NORMAL, 10, base, Duration::from_micros(i * 100)))
                .unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        let ids: Vec<u64> = slots.iter().map(|s| s.request_id).collect();
        let expected: Vec<u64> = (0..50).collect();
        assert_eq!(ids, expected);
    }

    #[test]
    fn fcfs_no_same_priority_reordering() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::Fcfs, 10);
        // All same priority, different arrival.
        for i in (0..5).rev() {
            sched
                .submit(req_at(i, Priority::HIGH, 10, base, Duration::from_millis(i * 5)))
                .unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        let ids: Vec<u64> = slots.iter().map(|s| s.request_id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4], "FCFS must sort by arrival_time");
    }

    // ── Shortest-job-first ─────────────────────────────────────────

    #[test]
    fn sjf_orders_by_max_tokens() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::ShortestJobFirst, 10);
        for (i, tokens) in [128, 16, 64, 8, 256].iter().enumerate() {
            sched
                .submit(req_at(
                    i as u64,
                    Priority::NORMAL,
                    *tokens,
                    base,
                    Duration::from_millis(i as u64),
                ))
                .unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        let tokens: Vec<usize> = slots.iter().map(|s| s.max_tokens).collect();
        assert_eq!(tokens, vec![8, 16, 64, 128, 256]);
    }

    #[test]
    fn sjf_ties_preserved_stably() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::ShortestJobFirst, 10);
        for i in 0..4 {
            sched.submit(req_at(i, Priority::NORMAL, 32, base, Duration::from_millis(i))).unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        // All same max_tokens: stable sort preserves original order.
        let ids: Vec<u64> = slots.iter().map(|s| s.request_id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn sjf_single_request() {
        let mut sched = default_scheduler(SchedulingPolicy::ShortestJobFirst, 10);
        sched.submit(BatchRequest::new(1, Priority::NORMAL, 100)).unwrap();
        let (slots, _) = sched.schedule_batch();
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].request_id, 1);
    }

    // ── Priority-based ─────────────────────────────────────────────

    #[test]
    fn priority_orders_highest_first() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::PriorityBased, 10);
        sched.submit(req_at(1, Priority::LOW, 32, base, Duration::ZERO)).unwrap();
        sched.submit(req_at(2, Priority::CRITICAL, 32, base, Duration::from_millis(1))).unwrap();
        sched.submit(req_at(3, Priority::NORMAL, 32, base, Duration::from_millis(2))).unwrap();
        sched.submit(req_at(4, Priority::HIGH, 32, base, Duration::from_millis(3))).unwrap();
        let (slots, _) = sched.schedule_batch();
        let ids: Vec<u64> = slots.iter().map(|s| s.request_id).collect();
        assert_eq!(ids, vec![2, 4, 3, 1]);
    }

    #[test]
    fn priority_fcfs_within_same_priority() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::PriorityBased, 10);
        for i in 0..4 {
            sched
                .submit(req_at(i, Priority::HIGH, 32, base, Duration::from_millis(i * 10)))
                .unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        let ids: Vec<u64> = slots.iter().map(|s| s.request_id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn priority_all_same_is_fcfs() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::PriorityBased, 10);
        for i in (0..5).rev() {
            sched
                .submit(req_at(i, Priority::NORMAL, 20, base, Duration::from_millis(i * 5)))
                .unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        let ids: Vec<u64> = slots.iter().map(|s| s.request_id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    // ── Fair-share ─────────────────────────────────────────────────

    #[test]
    fn fair_share_interleaves_priorities() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::FairShare, 20);
        // 2 HIGH, 2 NORMAL, 2 LOW
        sched.submit(req_at(10, Priority::HIGH, 32, base, Duration::ZERO)).unwrap();
        sched.submit(req_at(11, Priority::HIGH, 32, base, Duration::from_millis(1))).unwrap();
        sched.submit(req_at(20, Priority::NORMAL, 32, base, Duration::from_millis(2))).unwrap();
        sched.submit(req_at(21, Priority::NORMAL, 32, base, Duration::from_millis(3))).unwrap();
        sched.submit(req_at(30, Priority::LOW, 32, base, Duration::from_millis(4))).unwrap();
        sched.submit(req_at(31, Priority::LOW, 32, base, Duration::from_millis(5))).unwrap();

        let (slots, _) = sched.schedule_batch();
        let ids: Vec<u64> = slots.iter().map(|s| s.request_id).collect();
        // Round-robin: one from each tier, then repeat.
        assert_eq!(ids, vec![10, 20, 30, 11, 21, 31]);
    }

    #[test]
    fn fair_share_single_tier_is_fcfs() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::FairShare, 10);
        for i in 0..4 {
            sched
                .submit(req_at(i, Priority::NORMAL, 32, base, Duration::from_millis(i * 10)))
                .unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        let ids: Vec<u64> = slots.iter().map(|s| s.request_id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn fair_share_uneven_tiers() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::FairShare, 20);
        // 3 HIGH, 1 LOW
        sched.submit(req_at(1, Priority::HIGH, 32, base, Duration::ZERO)).unwrap();
        sched.submit(req_at(2, Priority::HIGH, 32, base, Duration::from_millis(1))).unwrap();
        sched.submit(req_at(3, Priority::HIGH, 32, base, Duration::from_millis(2))).unwrap();
        sched.submit(req_at(4, Priority::LOW, 32, base, Duration::from_millis(3))).unwrap();

        let (slots, _) = sched.schedule_batch();
        let ids: Vec<u64> = slots.iter().map(|s| s.request_id).collect();
        // Round 1: HIGH(1), LOW(4). Round 2: HIGH(2). Round 3: HIGH(3).
        assert_eq!(ids, vec![1, 4, 2, 3]);
    }

    // ── Memory budget ──────────────────────────────────────────────

    #[test]
    fn memory_budget_basic() {
        let mut mb = MemoryBudget::new(1024);
        assert_eq!(mb.available(), 1024);
        assert!(mb.try_allocate(512));
        assert_eq!(mb.available(), 512);
        assert!(!mb.try_allocate(1024));
        mb.release(512);
        assert_eq!(mb.available(), 1024);
    }

    #[test]
    fn memory_budget_utilization() {
        let mut mb = MemoryBudget::new(1000);
        mb.try_allocate(250);
        assert!((mb.utilization() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn memory_budget_zero_total() {
        let mb = MemoryBudget::new(0);
        assert_eq!(mb.utilization(), 0.0);
        assert_eq!(mb.available(), 0);
    }

    #[test]
    fn memory_budget_exact_fit() {
        let mut mb = MemoryBudget::new(100);
        assert!(mb.try_allocate(100));
        assert_eq!(mb.available(), 0);
        assert!(!mb.try_allocate(1));
    }

    #[test]
    fn memory_budget_release_underflow_saturates() {
        let mut mb = MemoryBudget::new(100);
        mb.release(200); // should not panic
        assert_eq!(mb.allocated_bytes, 0);
    }

    #[test]
    fn memory_limits_batch_size() {
        let base = Instant::now();
        // Tiny memory: only 100 KiB; each request needs ~64 KiB (32 tokens * 2 KiB).
        let mem = MemoryBudget::new(100 * 1024);
        let admission = AdmissionController::new(100, u64::MAX);
        let preempt = PreemptionManager::new(2);
        let mut sched = BatchScheduler::new(SchedulingPolicy::Fcfs, 10, mem, admission, preempt);

        for i in 0..5 {
            sched.submit(req_at(i, Priority::NORMAL, 32, base, Duration::from_millis(i))).unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        // 100 KiB / 64 KiB = 1 request fits.
        assert_eq!(slots.len(), 1);
        // Remaining 4 should be back in queue.
        assert_eq!(sched.queue_len(), 4);
    }

    #[test]
    fn memory_releases_after_batch_complete() {
        let base = Instant::now();
        let mem = MemoryBudget::new(200 * 1024);
        let admission = AdmissionController::new(100, u64::MAX);
        let preempt = PreemptionManager::new(2);
        let mut sched = BatchScheduler::new(SchedulingPolicy::Fcfs, 10, mem, admission, preempt);

        sched.submit(req_at(1, Priority::NORMAL, 32, base, Duration::ZERO)).unwrap();
        let (slots, reqs) = sched.schedule_batch();
        assert_eq!(slots.len(), 1);

        let used = sched.memory().allocated_bytes;
        assert!(used > 0);

        sched.complete_batch(&slots, &reqs, Instant::now());
        assert_eq!(sched.memory().allocated_bytes, 0);
    }

    #[test]
    fn memory_oom_rejects_no_crash() {
        let mem = MemoryBudget::new(1); // 1 byte
        let admission = AdmissionController::new(100, u64::MAX);
        let preempt = PreemptionManager::new(2);
        let mut sched = BatchScheduler::new(SchedulingPolicy::Fcfs, 10, mem, admission, preempt);

        sched.submit(BatchRequest::new(1, Priority::NORMAL, 1024)).unwrap();
        let (slots, _) = sched.schedule_batch();
        // Request needs way more than 1 byte — can't be scheduled.
        assert!(slots.is_empty());
        assert_eq!(sched.queue_len(), 1);
    }

    // ── Preemption ─────────────────────────────────────────────────

    #[test]
    fn preemption_high_evicts_low() {
        let pm = PreemptionManager::new(2);
        let incoming = BatchRequest::new(99, Priority::CRITICAL, 64);
        let running_slots = vec![BatchSlot {
            request_id: 1,
            slot_index: 0,
            reserved_memory_bytes: 65536,
            max_tokens: 32,
        }];
        let running_reqs = vec![BatchRequest::new(1, Priority::LOW, 32)];
        let victims = pm.select_victims(&incoming, &running_slots, &running_reqs, 60000);
        assert_eq!(victims, vec![0]);
    }

    #[test]
    fn preemption_insufficient_gap_no_evict() {
        let pm = PreemptionManager::new(2);
        let incoming = BatchRequest::new(99, Priority::NORMAL, 64);
        let running_slots = vec![BatchSlot {
            request_id: 1,
            slot_index: 0,
            reserved_memory_bytes: 65536,
            max_tokens: 32,
        }];
        let running_reqs = vec![BatchRequest::new(1, Priority::LOW, 32)];
        let victims = pm.select_victims(&incoming, &running_slots, &running_reqs, 60000);
        // NORMAL(1) - LOW(0) = 1, gap=2 → not enough.
        assert!(victims.is_empty());
    }

    #[test]
    fn preemption_evicts_only_enough() {
        let pm = PreemptionManager::new(1);
        let incoming = BatchRequest::new(99, Priority::CRITICAL, 64);
        let running_slots = vec![
            BatchSlot {
                request_id: 1,
                slot_index: 0,
                reserved_memory_bytes: 40000,
                max_tokens: 20,
            },
            BatchSlot {
                request_id: 2,
                slot_index: 1,
                reserved_memory_bytes: 40000,
                max_tokens: 20,
            },
            BatchSlot {
                request_id: 3,
                slot_index: 2,
                reserved_memory_bytes: 40000,
                max_tokens: 20,
            },
        ];
        let running_reqs = vec![
            BatchRequest::new(1, Priority::LOW, 20),
            BatchRequest::new(2, Priority::LOW, 20),
            BatchRequest::new(3, Priority::LOW, 20),
        ];
        let victims = pm.select_victims(&incoming, &running_slots, &running_reqs, 50000);
        // Need 50 KB; each slot has 40 KB. Should evict 2.
        assert_eq!(victims.len(), 2);
    }

    #[test]
    fn preemption_no_victims_when_empty() {
        let pm = PreemptionManager::new(1);
        let incoming = BatchRequest::new(99, Priority::CRITICAL, 64);
        let victims = pm.select_victims(&incoming, &[], &[], 10000);
        assert!(victims.is_empty());
    }

    #[test]
    fn preemption_via_scheduler() {
        let mem = MemoryBudget::new(100_000);
        let admission = AdmissionController::new(100, u64::MAX);
        let preempt = PreemptionManager::new(2);
        let mut sched = BatchScheduler::new(SchedulingPolicy::Fcfs, 10, mem, admission, preempt);

        // Simulate running slots using up all memory.
        sched.memory_mut().try_allocate(100_000);
        let running_slots = vec![BatchSlot {
            request_id: 1,
            slot_index: 0,
            reserved_memory_bytes: 100_000,
            max_tokens: 50,
        }];
        let running_reqs = vec![BatchRequest::new(1, Priority::LOW, 50)];

        let incoming = BatchRequest::new(99, Priority::CRITICAL, 32);
        let victims = sched.try_preempt(&incoming, &running_slots, &running_reqs);
        assert_eq!(victims.len(), 1);

        // Release preempted slots.
        let preempted: Vec<BatchSlot> = victims.iter().map(|&i| running_slots[i].clone()).collect();
        sched.release_preempted(&preempted);
        assert_eq!(sched.stats().preemptions, 1);
        assert_eq!(sched.memory().available(), 100_000);
    }

    #[test]
    fn preemption_not_needed_when_memory_free() {
        let mem = MemoryBudget::new(1_000_000);
        let admission = AdmissionController::new(100, u64::MAX);
        let preempt = PreemptionManager::new(2);
        let sched = BatchScheduler::new(SchedulingPolicy::Fcfs, 10, mem, admission, preempt);

        let incoming = BatchRequest::new(99, Priority::CRITICAL, 32);
        let running_slots = vec![BatchSlot {
            request_id: 1,
            slot_index: 0,
            reserved_memory_bytes: 65536,
            max_tokens: 32,
        }];
        let running_reqs = vec![BatchRequest::new(1, Priority::LOW, 32)];
        let victims = sched.try_preempt(&incoming, &running_slots, &running_reqs);
        assert!(victims.is_empty());
    }

    // ── Admission control ──────────────────────────────────────────

    #[test]
    fn admission_rejects_when_queue_full() {
        let mem = MemoryBudget::new(u64::MAX);
        let admission = AdmissionController::new(2, u64::MAX);
        let preempt = PreemptionManager::new(2);
        let mut sched = BatchScheduler::new(SchedulingPolicy::Fcfs, 10, mem, admission, preempt);

        sched.submit(BatchRequest::new(1, Priority::NORMAL, 32)).unwrap();
        sched.submit(BatchRequest::new(2, Priority::NORMAL, 32)).unwrap();
        let result = sched.submit(BatchRequest::new(3, Priority::NORMAL, 32));
        assert!(result.is_err());
        assert_eq!(sched.stats().requests_rejected, 1);
    }

    #[test]
    fn admission_rejects_when_memory_exhausted() {
        let mem = MemoryBudget::new(u64::MAX);
        // pending memory limit: 100 KiB
        let admission = AdmissionController::new(100, 100 * 1024);
        let preempt = PreemptionManager::new(2);
        let mut sched = BatchScheduler::new(SchedulingPolicy::Fcfs, 10, mem, admission, preempt);

        // Each request estimates 32 * 2048 = 64 KiB
        sched.submit(BatchRequest::new(1, Priority::NORMAL, 32)).unwrap();
        // Second would push total to 128 KiB > 100 KiB.
        let result = sched.submit(BatchRequest::new(2, Priority::NORMAL, 32));
        assert!(result.is_err());
    }

    #[test]
    fn admission_accepts_within_limits() {
        let ac = AdmissionController::new(10, 1_000_000);
        let req = BatchRequest::new(1, Priority::NORMAL, 32);
        assert!(ac.should_admit(5, 500_000, &req));
    }

    #[test]
    fn admission_rejects_at_exact_queue_limit() {
        let ac = AdmissionController::new(5, u64::MAX);
        let req = BatchRequest::new(1, Priority::NORMAL, 32);
        assert!(!ac.should_admit(5, 0, &req));
    }

    #[test]
    fn admission_rejected_request_returned() {
        let mem = MemoryBudget::new(u64::MAX);
        let admission = AdmissionController::new(0, u64::MAX); // zero queue depth
        let preempt = PreemptionManager::new(2);
        let mut sched = BatchScheduler::new(SchedulingPolicy::Fcfs, 10, mem, admission, preempt);

        let result = sched.submit(BatchRequest::new(42, Priority::NORMAL, 32));
        assert!(result.is_err());
        let rejected = result.unwrap_err();
        assert_eq!(rejected.id, 42);
    }

    // ── Deadline scheduling ────────────────────────────────────────

    #[test]
    fn deadline_orders_earliest_first() {
        let base = Instant::now();
        let mut reqs = vec![
            BatchRequest::new(1, Priority::NORMAL, 32)
                .with_arrival(base)
                .with_deadline(base + Duration::from_secs(10)),
            BatchRequest::new(2, Priority::NORMAL, 32)
                .with_arrival(base)
                .with_deadline(base + Duration::from_secs(2)),
            BatchRequest::new(3, Priority::NORMAL, 32)
                .with_arrival(base)
                .with_deadline(base + Duration::from_secs(5)),
        ];
        DeadlineScheduler::order(&mut reqs);
        let ids: Vec<u64> = reqs.iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![2, 3, 1]);
    }

    #[test]
    fn deadline_no_deadline_goes_last() {
        let base = Instant::now();
        let mut reqs = vec![
            BatchRequest::new(1, Priority::NORMAL, 32).with_arrival(base),
            BatchRequest::new(2, Priority::NORMAL, 32)
                .with_arrival(base)
                .with_deadline(base + Duration::from_secs(5)),
        ];
        DeadlineScheduler::order(&mut reqs);
        assert_eq!(reqs[0].id, 2);
        assert_eq!(reqs[1].id, 1);
    }

    #[test]
    fn deadline_evaluate_all_met() {
        let base = Instant::now();
        let reqs = vec![
            BatchRequest::new(1, Priority::NORMAL, 32)
                .with_deadline(base + Duration::from_secs(10)),
            BatchRequest::new(2, Priority::NORMAL, 32)
                .with_deadline(base + Duration::from_secs(20)),
        ];
        let (met, missed) = DeadlineScheduler::evaluate(&reqs, base + Duration::from_secs(5));
        assert_eq!(met, 2);
        assert_eq!(missed, 0);
    }

    #[test]
    fn deadline_evaluate_all_missed() {
        let base = Instant::now();
        let reqs = vec![
            BatchRequest::new(1, Priority::NORMAL, 32).with_deadline(base + Duration::from_secs(1)),
            BatchRequest::new(2, Priority::NORMAL, 32).with_deadline(base + Duration::from_secs(2)),
        ];
        let (met, missed) = DeadlineScheduler::evaluate(&reqs, base + Duration::from_secs(10));
        assert_eq!(met, 0);
        assert_eq!(missed, 2);
    }

    #[test]
    fn deadline_evaluate_mixed() {
        let base = Instant::now();
        let reqs = vec![
            BatchRequest::new(1, Priority::NORMAL, 32).with_deadline(base + Duration::from_secs(5)),
            BatchRequest::new(2, Priority::NORMAL, 32)
                .with_deadline(base + Duration::from_secs(15)),
            BatchRequest::new(3, Priority::NORMAL, 32), // no deadline
        ];
        let (met, missed) = DeadlineScheduler::evaluate(&reqs, base + Duration::from_secs(10));
        assert_eq!(met, 1);
        assert_eq!(missed, 1);
    }

    #[test]
    fn deadline_stats_recorded() {
        let base = Instant::now();
        let mem = MemoryBudget::new(u64::MAX);
        let admission = AdmissionController::new(100, u64::MAX);
        let preempt = PreemptionManager::new(2);
        let mut sched = BatchScheduler::new(SchedulingPolicy::Fcfs, 10, mem, admission, preempt);

        sched
            .submit(
                BatchRequest::new(1, Priority::NORMAL, 8)
                    .with_arrival(base)
                    .with_deadline(base + Duration::from_secs(60)),
            )
            .unwrap();
        sched
            .submit(
                BatchRequest::new(2, Priority::NORMAL, 8)
                    .with_arrival(base)
                    .with_deadline(base + Duration::from_secs(1)),
            )
            .unwrap();

        let (slots, reqs) = sched.schedule_batch();
        sched.complete_batch(&slots, &reqs, base + Duration::from_secs(30));
        assert_eq!(sched.stats().deadlines_met, 1);
        assert_eq!(sched.stats().deadlines_missed, 1);
    }

    // ── Batch statistics ───────────────────────────────────────────

    #[test]
    fn stats_initial_zeros() {
        let stats = BatchStats::default();
        assert_eq!(stats.batches_formed, 0);
        assert_eq!(stats.requests_scheduled, 0);
        assert_eq!(stats.avg_wait(), Duration::ZERO);
        assert_eq!(stats.avg_batch_size(), 0.0);
    }

    #[test]
    fn stats_avg_batch_size() {
        let mut sched = default_scheduler(SchedulingPolicy::Fcfs, 3);
        // Submit 6 requests; max batch = 3 → should form 2 batches.
        let base = Instant::now();
        for i in 0..6 {
            sched.submit(req_at(i, Priority::NORMAL, 8, base, Duration::from_millis(i))).unwrap();
        }
        let _ = sched.schedule_batch();
        let _ = sched.schedule_batch();
        assert_eq!(sched.stats().batches_formed, 2);
        assert_eq!(sched.stats().requests_scheduled, 6);
        assert!((sched.stats().avg_batch_size() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn stats_rejected_counted() {
        let mem = MemoryBudget::new(u64::MAX);
        let admission = AdmissionController::new(1, u64::MAX);
        let preempt = PreemptionManager::new(2);
        let mut sched = BatchScheduler::new(SchedulingPolicy::Fcfs, 10, mem, admission, preempt);

        sched.submit(BatchRequest::new(1, Priority::NORMAL, 32)).unwrap();
        let _ = sched.submit(BatchRequest::new(2, Priority::NORMAL, 32));
        let _ = sched.submit(BatchRequest::new(3, Priority::NORMAL, 32));
        assert_eq!(sched.stats().requests_rejected, 2);
    }

    #[test]
    fn stats_preemptions_counted() {
        let mem = MemoryBudget::new(100_000);
        let admission = AdmissionController::new(100, u64::MAX);
        let preempt = PreemptionManager::new(1);
        let mut sched = BatchScheduler::new(SchedulingPolicy::Fcfs, 10, mem, admission, preempt);

        let slots_to_release = vec![
            BatchSlot {
                request_id: 1,
                slot_index: 0,
                reserved_memory_bytes: 50_000,
                max_tokens: 10,
            },
            BatchSlot {
                request_id: 2,
                slot_index: 1,
                reserved_memory_bytes: 50_000,
                max_tokens: 10,
            },
        ];
        sched.release_preempted(&slots_to_release);
        assert_eq!(sched.stats().preemptions, 2);
    }

    // ── Batch capacity ─────────────────────────────────────────────

    #[test]
    fn batch_respects_max_batch_size() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::Fcfs, 3);
        for i in 0..10 {
            sched.submit(req_at(i, Priority::NORMAL, 8, base, Duration::from_millis(i))).unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        assert_eq!(slots.len(), 3);
        assert_eq!(sched.queue_len(), 7);
    }

    #[test]
    fn multiple_batches_drain_queue() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::Fcfs, 4);
        for i in 0..10 {
            sched.submit(req_at(i, Priority::NORMAL, 8, base, Duration::from_millis(i))).unwrap();
        }
        let (b1, r1) = sched.schedule_batch();
        sched.complete_batch(&b1, &r1, Instant::now());
        let (b2, r2) = sched.schedule_batch();
        sched.complete_batch(&b2, &r2, Instant::now());
        let (b3, r3) = sched.schedule_batch();
        sched.complete_batch(&b3, &r3, Instant::now());

        assert_eq!(b1.len(), 4);
        assert_eq!(b2.len(), 4);
        assert_eq!(b3.len(), 2);
        assert_eq!(sched.queue_len(), 0);
    }

    #[test]
    fn slot_indices_are_sequential() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::Fcfs, 5);
        for i in 0..5 {
            sched.submit(req_at(i, Priority::NORMAL, 16, base, Duration::from_millis(i))).unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        let indices: Vec<usize> = slots.iter().map(|s| s.slot_index).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn empty_schedule_no_panic() {
        let mut sched = default_scheduler(SchedulingPolicy::PriorityBased, 10);
        let (s, r) = sched.schedule_batch();
        assert!(s.is_empty() && r.is_empty());
        assert_eq!(sched.stats().batches_formed, 0);
    }

    #[test]
    fn single_request_all_policies() {
        for policy in [
            SchedulingPolicy::Fcfs,
            SchedulingPolicy::ShortestJobFirst,
            SchedulingPolicy::PriorityBased,
            SchedulingPolicy::FairShare,
        ] {
            let mut sched = default_scheduler(policy, 10);
            sched.submit(BatchRequest::new(1, Priority::NORMAL, 16)).unwrap();
            let (slots, _) = sched.schedule_batch();
            assert_eq!(slots.len(), 1, "policy={policy:?} should schedule 1 request");
        }
    }

    #[test]
    fn all_same_priority_all_policies() {
        let base = Instant::now();
        for policy in
            [SchedulingPolicy::Fcfs, SchedulingPolicy::PriorityBased, SchedulingPolicy::FairShare]
        {
            let mut sched = default_scheduler(policy, 10);
            for i in 0..5 {
                sched
                    .submit(req_at(i, Priority::NORMAL, 32, base, Duration::from_millis(i * 10)))
                    .unwrap();
            }
            let (slots, _) = sched.schedule_batch();
            assert_eq!(slots.len(), 5, "policy={policy:?}");
        }
    }

    #[test]
    fn request_builder_chain() {
        let base = Instant::now();
        let r = BatchRequest::new(1, Priority::HIGH, 64)
            .with_arrival(base)
            .with_deadline(base + Duration::from_secs(5))
            .with_memory(1024);
        assert_eq!(r.id, 1);
        assert_eq!(r.priority, Priority::HIGH);
        assert_eq!(r.max_tokens, 64);
        assert_eq!(r.estimated_memory_bytes, 1024);
        assert!(r.deadline.is_some());
    }

    #[test]
    fn priority_ordering() {
        assert!(Priority::LOW < Priority::NORMAL);
        assert!(Priority::NORMAL < Priority::HIGH);
        assert!(Priority::HIGH < Priority::CRITICAL);
    }

    #[test]
    fn batch_slot_fields() {
        let slot =
            BatchSlot { request_id: 7, slot_index: 3, reserved_memory_bytes: 2048, max_tokens: 64 };
        assert_eq!(slot.request_id, 7);
        assert_eq!(slot.slot_index, 3);
        assert_eq!(slot.reserved_memory_bytes, 2048);
        assert_eq!(slot.max_tokens, 64);
    }

    #[test]
    fn scheduling_policy_equality() {
        assert_eq!(SchedulingPolicy::Fcfs, SchedulingPolicy::Fcfs);
        assert_ne!(SchedulingPolicy::Fcfs, SchedulingPolicy::ShortestJobFirst);
    }

    // ── Property-style tests ───────────────────────────────────────

    #[test]
    fn property_fcfs_never_reorders_same_priority() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::Fcfs, 100);
        let n = 30;
        for i in 0..n {
            sched
                .submit(req_at(i, Priority::NORMAL, 16, base, Duration::from_micros(i * 50)))
                .unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        for window in slots.windows(2) {
            assert!(
                window[0].request_id < window[1].request_id,
                "FCFS violated: {} scheduled before {}",
                window[0].request_id,
                window[1].request_id,
            );
        }
    }

    #[test]
    fn property_sjf_non_decreasing_tokens() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::ShortestJobFirst, 100);
        let token_sizes = [256, 32, 128, 8, 64, 512, 16, 1024, 4, 48];
        for (i, &tokens) in token_sizes.iter().enumerate() {
            sched
                .submit(req_at(
                    i as u64,
                    Priority::NORMAL,
                    tokens,
                    base,
                    Duration::from_millis(i as u64),
                ))
                .unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        for window in slots.windows(2) {
            assert!(
                window[0].max_tokens <= window[1].max_tokens,
                "SJF violated: {} tokens before {} tokens",
                window[0].max_tokens,
                window[1].max_tokens,
            );
        }
    }

    #[test]
    fn property_priority_non_increasing() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::PriorityBased, 100);
        let prios = [
            Priority::LOW,
            Priority::CRITICAL,
            Priority::NORMAL,
            Priority::HIGH,
            Priority::LOW,
            Priority::HIGH,
            Priority::CRITICAL,
            Priority::NORMAL,
        ];
        for (i, &prio) in prios.iter().enumerate() {
            sched
                .submit(req_at(i as u64, prio, 32, base, Duration::from_millis(i as u64 * 10)))
                .unwrap();
        }
        let (_, reqs) = sched.schedule_batch();
        for window in reqs.windows(2) {
            assert!(
                window[0].priority >= window[1].priority,
                "Priority violated: {:?} before {:?}",
                window[0].priority,
                window[1].priority,
            );
        }
    }

    #[test]
    fn property_memory_never_exceeds_budget() {
        let total = 256 * 1024u64; // 256 KiB
        let mem = MemoryBudget::new(total);
        let admission = AdmissionController::new(1000, u64::MAX);
        let preempt = PreemptionManager::new(2);
        let mut sched = BatchScheduler::new(SchedulingPolicy::Fcfs, 100, mem, admission, preempt);

        let base = Instant::now();
        for i in 0..50 {
            sched.submit(req_at(i, Priority::NORMAL, 16, base, Duration::from_millis(i))).unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        let total_reserved: u64 = slots.iter().map(|s| s.reserved_memory_bytes).sum();
        assert!(total_reserved <= total, "scheduled {total_reserved} bytes but budget is {total}");
    }

    #[test]
    fn property_batch_size_never_exceeds_max() {
        for max in [1, 2, 5, 10] {
            let base = Instant::now();
            let mut sched = default_scheduler(SchedulingPolicy::Fcfs, max);
            for i in 0..20 {
                sched
                    .submit(req_at(i, Priority::NORMAL, 8, base, Duration::from_millis(i)))
                    .unwrap();
            }
            let (slots, _) = sched.schedule_batch();
            assert!(slots.len() <= max, "max_batch_size={max} but got {} slots", slots.len());
        }
    }

    #[test]
    fn property_all_requests_eventually_scheduled() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::Fcfs, 3);
        let n = 9u64;
        for i in 0..n {
            sched.submit(req_at(i, Priority::NORMAL, 8, base, Duration::from_millis(i))).unwrap();
        }
        let mut all_ids = Vec::new();
        while sched.queue_len() > 0 {
            let (slots, reqs) = sched.schedule_batch();
            all_ids.extend(slots.iter().map(|s| s.request_id));
            sched.complete_batch(&slots, &reqs, Instant::now());
        }
        all_ids.sort();
        let expected: Vec<u64> = (0..n).collect();
        assert_eq!(all_ids, expected);
    }

    #[test]
    fn property_no_request_lost_or_duplicated() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::PriorityBased, 4);
        let n = 12u64;
        let prios = [Priority::LOW, Priority::NORMAL, Priority::HIGH, Priority::CRITICAL];
        for i in 0..n {
            sched
                .submit(req_at(i, prios[i as usize % 4], 16, base, Duration::from_millis(i)))
                .unwrap();
        }
        let mut all_ids = Vec::new();
        while sched.queue_len() > 0 {
            let (slots, reqs) = sched.schedule_batch();
            all_ids.extend(slots.iter().map(|s| s.request_id));
            sched.complete_batch(&slots, &reqs, Instant::now());
        }
        all_ids.sort();
        let expected: Vec<u64> = (0..n).collect();
        assert_eq!(all_ids, expected, "requests lost or duplicated");
    }

    // ── Miscellaneous ──────────────────────────────────────────────

    #[test]
    fn scheduler_accessors() {
        let sched = default_scheduler(SchedulingPolicy::ShortestJobFirst, 8);
        assert_eq!(sched.policy(), SchedulingPolicy::ShortestJobFirst);
        assert_eq!(sched.queue_len(), 0);
    }

    #[test]
    fn memory_budget_clone() {
        let mb = MemoryBudget::new(4096);
        let mb2 = mb.clone();
        assert_eq!(mb.total_bytes, mb2.total_bytes);
        assert_eq!(mb.allocated_bytes, mb2.allocated_bytes);
    }

    #[test]
    fn batch_request_default_memory_heuristic() {
        let r = BatchRequest::new(1, Priority::NORMAL, 100);
        assert_eq!(r.estimated_memory_bytes, 100 * 2048);
    }

    #[test]
    fn deadline_scheduler_all_without_deadlines() {
        let base = Instant::now();
        let mut reqs = vec![
            BatchRequest::new(1, Priority::NORMAL, 32)
                .with_arrival(base + Duration::from_millis(20)),
            BatchRequest::new(2, Priority::NORMAL, 32)
                .with_arrival(base + Duration::from_millis(10)),
            BatchRequest::new(3, Priority::NORMAL, 32).with_arrival(base),
        ];
        DeadlineScheduler::order(&mut reqs);
        // No deadlines → sorted by arrival time.
        let ids: Vec<u64> = reqs.iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![3, 2, 1]);
    }

    #[test]
    fn deadline_evaluate_no_deadlines() {
        let reqs = vec![
            BatchRequest::new(1, Priority::NORMAL, 32),
            BatchRequest::new(2, Priority::NORMAL, 32),
        ];
        let (met, missed) = DeadlineScheduler::evaluate(&reqs, Instant::now());
        assert_eq!(met, 0);
        assert_eq!(missed, 0);
    }

    #[test]
    fn deadline_exact_boundary() {
        let base = Instant::now();
        let reqs = vec![BatchRequest::new(1, Priority::NORMAL, 32).with_deadline(base)];
        // completion_time == deadline → met (<=).
        let (met, missed) = DeadlineScheduler::evaluate(&reqs, base);
        assert_eq!(met, 1);
        assert_eq!(missed, 0);
    }

    #[test]
    fn preemption_manager_priority_gap_zero() {
        let pm = PreemptionManager::new(0);
        let incoming = BatchRequest::new(1, Priority::NORMAL, 32);
        let running_slots = vec![BatchSlot {
            request_id: 2,
            slot_index: 0,
            reserved_memory_bytes: 1000,
            max_tokens: 16,
        }];
        let running_reqs = vec![BatchRequest::new(2, Priority::NORMAL, 16)];
        // gap = 0 → same-priority can preempt.
        let victims = pm.select_victims(&incoming, &running_slots, &running_reqs, 500);
        assert_eq!(victims.len(), 1);
    }

    #[test]
    fn fair_share_empty() {
        let mut sched = default_scheduler(SchedulingPolicy::FairShare, 10);
        let (s, r) = sched.schedule_batch();
        assert!(s.is_empty());
        assert!(r.is_empty());
    }

    #[test]
    fn sjf_large_spread() {
        let base = Instant::now();
        let mut sched = default_scheduler(SchedulingPolicy::ShortestJobFirst, 100);
        let sizes = [10000, 1, 5000, 2, 9999, 3];
        for (i, &s) in sizes.iter().enumerate() {
            sched
                .submit(req_at(
                    i as u64,
                    Priority::NORMAL,
                    s,
                    base,
                    Duration::from_millis(i as u64),
                ))
                .unwrap();
        }
        let (slots, _) = sched.schedule_batch();
        let tokens: Vec<usize> = slots.iter().map(|s| s.max_tokens).collect();
        assert_eq!(tokens, vec![1, 2, 3, 5000, 9999, 10000]);
    }
}
