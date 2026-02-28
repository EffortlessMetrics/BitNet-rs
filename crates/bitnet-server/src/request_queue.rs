//! Priority request queue with deadline-based scheduling for GPU inference.
//!
//! [`PriorityRequestQueue`] orders incoming inference requests by priority
//! level and optional deadline, applies per-priority queue-size limits, and
//! exposes metrics (queue depth, wait time, deadline misses).

use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Priority levels
// ---------------------------------------------------------------------------

/// Priority level for an inference request (lower numeric = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Real-time interactive requests (chat, streaming).
    Interactive = 0,
    /// Standard API requests.
    Normal = 1,
    /// Batch processing requests.
    Batch = 2,
    /// Low-priority background tasks (pre-computation, warm-up).
    Background = 3,
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower numeric value = higher priority → reverse ordering
        (*other as u8).cmp(&(*self as u8))
    }
}

// ---------------------------------------------------------------------------
// Queue entry
// ---------------------------------------------------------------------------

/// A queued inference request.
#[derive(Debug, Clone)]
pub struct QueueEntry {
    /// Unique request identifier.
    pub id: String,
    /// Scheduling priority.
    pub priority: Priority,
    /// Optional hard deadline; the request is considered a miss after this.
    pub deadline: Option<Instant>,
    /// Time the request was enqueued.
    pub enqueued_at: Instant,
    /// Opaque payload size (tokens requested, etc.).
    pub token_budget: u32,
}

/// Wrapper for `BinaryHeap` ordering.
#[derive(Debug)]
struct HeapEntry(QueueEntry);

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.priority == other.0.priority
            && self.0.enqueued_at == other.0.enqueued_at
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // 1. Higher priority (lower numeric) first
        self.0.priority.cmp(&other.0.priority).then_with(|| {
            // 2. Among equal priority: earlier deadline first
            match (&self.0.deadline, &other.0.deadline) {
                (Some(a), Some(b)) => b.cmp(a), // earlier = greater for max-heap
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (None, None) => {
                    // 3. FIFO: earlier enqueue = greater
                    other.0.enqueued_at.cmp(&self.0.enqueued_at)
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the priority request queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    /// Maximum queue depth per priority level.
    pub max_per_priority: HashMap<String, usize>,
    /// Global maximum across all priorities.
    pub max_total: usize,
}

impl Default for QueueConfig {
    fn default() -> Self {
        let mut m = HashMap::new();
        m.insert("Interactive".into(), 64);
        m.insert("Normal".into(), 256);
        m.insert("Batch".into(), 512);
        m.insert("Background".into(), 128);
        Self {
            max_per_priority: m,
            max_total: 960,
        }
    }
}

fn priority_key(p: Priority) -> &'static str {
    match p {
        Priority::Interactive => "Interactive",
        Priority::Normal => "Normal",
        Priority::Batch => "Batch",
        Priority::Background => "Background",
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Error returned when enqueuing fails.
#[derive(Debug, Clone, PartialEq)]
pub enum QueueError {
    /// The queue for this priority is full.
    PriorityQueueFull {
        priority: Priority,
        limit: usize,
    },
    /// Global queue capacity reached.
    GlobalQueueFull {
        limit: usize,
    },
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PriorityQueueFull { priority, limit } => {
                write!(f, "{:?} queue full (limit {})", priority, limit)
            }
            Self::GlobalQueueFull { limit } => {
                write!(f, "global queue full (limit {})", limit)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Snapshot of queue metrics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct QueueMetrics {
    /// Total items across all priorities.
    pub total_depth: usize,
    /// Items per priority level.
    pub depth_per_priority: HashMap<String, usize>,
    /// Total requests dequeued since creation.
    pub total_dequeued: u64,
    /// Total deadline misses observed at dequeue time.
    pub deadline_misses: u64,
    /// Average wait time of dequeued requests (seconds).
    pub avg_wait_secs: f64,
}

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Inner {
    heap: BinaryHeap<HeapEntry>,
    counts: HashMap<String, usize>,
    config: QueueConfig,
    total_dequeued: u64,
    deadline_misses: u64,
    cumulative_wait_secs: f64,
}

impl Inner {
    fn new(config: QueueConfig) -> Self {
        Self {
            heap: BinaryHeap::new(),
            counts: HashMap::new(),
            config,
            total_dequeued: 0,
            deadline_misses: 0,
            cumulative_wait_secs: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// PriorityRequestQueue
// ---------------------------------------------------------------------------

/// Thread-safe priority queue for GPU inference requests.
#[derive(Clone)]
pub struct PriorityRequestQueue {
    inner: Arc<RwLock<Inner>>,
}

impl PriorityRequestQueue {
    pub fn new(config: QueueConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner::new(config))),
        }
    }

    /// Enqueue a request. Returns `Err` on backpressure.
    pub fn enqueue(&self, entry: QueueEntry) -> Result<(), QueueError> {
        let mut inner = self.inner.write().unwrap();
        let key = priority_key(entry.priority);

        // Per-priority limit
        let count = inner.counts.get(key).copied().unwrap_or(0);
        if let Some(&limit) = inner.config.max_per_priority.get(key) {
            if count >= limit {
                return Err(QueueError::PriorityQueueFull {
                    priority: entry.priority,
                    limit,
                });
            }
        }

        // Global limit
        let total: usize = inner.counts.values().sum();
        if total >= inner.config.max_total {
            return Err(QueueError::GlobalQueueFull {
                limit: inner.config.max_total,
            });
        }

        *inner.counts.entry(key.to_string()).or_insert(0) += 1;
        inner.heap.push(HeapEntry(entry));
        Ok(())
    }

    /// Dequeue the highest-priority request.
    ///
    /// Expired entries (past deadline) are still returned but counted as
    /// deadline misses in the metrics.
    pub fn dequeue(&self) -> Option<QueueEntry> {
        self.dequeue_at(Instant::now())
    }

    /// Testable variant with explicit timestamp.
    pub fn dequeue_at(&self, now: Instant) -> Option<QueueEntry> {
        let mut inner = self.inner.write().unwrap();
        let he = inner.heap.pop()?;
        let entry = he.0;
        let key = priority_key(entry.priority);
        if let Some(c) = inner.counts.get_mut(key) {
            *c = c.saturating_sub(1);
        }
        inner.total_dequeued += 1;
        let wait = now.duration_since(entry.enqueued_at).as_secs_f64();
        inner.cumulative_wait_secs += wait;
        if let Some(dl) = entry.deadline {
            if now > dl {
                inner.deadline_misses += 1;
            }
        }
        Some(entry)
    }

    /// Peek at the next entry without removing it.
    pub fn peek_priority(&self) -> Option<Priority> {
        let inner = self.inner.read().unwrap();
        inner.heap.peek().map(|h| h.0.priority)
    }

    /// Current queue depth.
    pub fn len(&self) -> usize {
        let inner = self.inner.read().unwrap();
        inner.heap.len()
    }

    /// Is the queue empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Depth for a specific priority.
    pub fn depth(&self, priority: Priority) -> usize {
        let inner = self.inner.read().unwrap();
        inner
            .counts
            .get(priority_key(priority))
            .copied()
            .unwrap_or(0)
    }

    /// Snapshot of current metrics.
    pub fn metrics(&self) -> QueueMetrics {
        let inner = self.inner.read().unwrap();
        let total_depth: usize = inner.counts.values().sum();
        let avg = if inner.total_dequeued > 0 {
            inner.cumulative_wait_secs / inner.total_dequeued as f64
        } else {
            0.0
        };
        QueueMetrics {
            total_depth,
            depth_per_priority: inner.counts.clone(),
            total_dequeued: inner.total_dequeued,
            deadline_misses: inner.deadline_misses,
            avg_wait_secs: avg,
        }
    }

    /// Drain all entries past their deadline, returning them.
    pub fn drain_expired(&self) -> Vec<QueueEntry> {
        self.drain_expired_at(Instant::now())
    }

    /// Testable variant with explicit timestamp.
    pub fn drain_expired_at(&self, now: Instant) -> Vec<QueueEntry> {
        let mut inner = self.inner.write().unwrap();
        let mut keep = Vec::new();
        let mut expired = Vec::new();
        while let Some(he) = inner.heap.pop() {
            if let Some(dl) = he.0.deadline {
                if now > dl {
                    let key = priority_key(he.0.priority);
                    if let Some(c) = inner.counts.get_mut(key) {
                        *c = c.saturating_sub(1);
                    }
                    inner.deadline_misses += 1;
                    expired.push(he.0);
                    continue;
                }
            }
            keep.push(he);
        }
        for h in keep {
            inner.heap.push(h);
        }
        expired
    }
}

impl Default for PriorityRequestQueue {
    fn default() -> Self {
        Self::new(QueueConfig::default())
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, priority: Priority) -> QueueEntry {
        QueueEntry {
            id: id.to_string(),
            priority,
            deadline: None,
            enqueued_at: Instant::now(),
            token_budget: 32,
        }
    }

    fn entry_at(
        id: &str,
        priority: Priority,
        enqueued_at: Instant,
        deadline: Option<Instant>,
    ) -> QueueEntry {
        QueueEntry {
            id: id.to_string(),
            priority,
            deadline,
            enqueued_at,
            token_budget: 32,
        }
    }

    #[test]
    fn enqueue_and_dequeue_fifo() {
        let q = PriorityRequestQueue::default();
        q.enqueue(entry("a", Priority::Normal)).unwrap();
        q.enqueue(entry("b", Priority::Normal)).unwrap();
        let first = q.dequeue().unwrap();
        assert_eq!(first.id, "a");
        let second = q.dequeue().unwrap();
        assert_eq!(second.id, "b");
    }

    #[test]
    fn priority_ordering() {
        let q = PriorityRequestQueue::default();
        let t = Instant::now();
        // Enqueue lower priority first
        q.enqueue(entry_at("bg", Priority::Background, t, None))
            .unwrap();
        q.enqueue(entry_at("norm", Priority::Normal, t, None))
            .unwrap();
        q.enqueue(entry_at("inter", Priority::Interactive, t, None))
            .unwrap();
        q.enqueue(entry_at("batch", Priority::Batch, t, None))
            .unwrap();

        assert_eq!(q.dequeue().unwrap().id, "inter");
        assert_eq!(q.dequeue().unwrap().id, "norm");
        assert_eq!(q.dequeue().unwrap().id, "batch");
        assert_eq!(q.dequeue().unwrap().id, "bg");
    }

    #[test]
    fn deadline_ordering_within_priority() {
        let q = PriorityRequestQueue::default();
        let t = Instant::now();
        let dl_late = t + Duration::from_secs(10);
        let dl_early = t + Duration::from_secs(2);
        q.enqueue(entry_at("late", Priority::Normal, t, Some(dl_late)))
            .unwrap();
        q.enqueue(entry_at("early", Priority::Normal, t, Some(dl_early)))
            .unwrap();
        // Earlier deadline should come first
        assert_eq!(q.dequeue().unwrap().id, "early");
        assert_eq!(q.dequeue().unwrap().id, "late");
    }

    #[test]
    fn per_priority_backpressure() {
        let mut cfg = QueueConfig::default();
        cfg.max_per_priority
            .insert("Normal".into(), 2);
        cfg.max_total = 100;
        let q = PriorityRequestQueue::new(cfg);
        q.enqueue(entry("a", Priority::Normal)).unwrap();
        q.enqueue(entry("b", Priority::Normal)).unwrap();
        let err = q.enqueue(entry("c", Priority::Normal)).unwrap_err();
        assert_eq!(
            err,
            QueueError::PriorityQueueFull {
                priority: Priority::Normal,
                limit: 2,
            }
        );
        // Different priority should still work
        q.enqueue(entry("d", Priority::Batch)).unwrap();
    }

    #[test]
    fn global_backpressure() {
        let cfg = QueueConfig {
            max_per_priority: {
                let mut m = HashMap::new();
                m.insert("Interactive".into(), 100);
                m
            },
            max_total: 2,
        };
        let q = PriorityRequestQueue::new(cfg);
        q.enqueue(entry("a", Priority::Interactive)).unwrap();
        q.enqueue(entry("b", Priority::Interactive)).unwrap();
        let err = q
            .enqueue(entry("c", Priority::Interactive))
            .unwrap_err();
        assert_eq!(err, QueueError::GlobalQueueFull { limit: 2 });
    }

    #[test]
    fn depth_tracking() {
        let q = PriorityRequestQueue::default();
        q.enqueue(entry("a", Priority::Normal)).unwrap();
        q.enqueue(entry("b", Priority::Interactive)).unwrap();
        q.enqueue(entry("c", Priority::Normal)).unwrap();
        assert_eq!(q.len(), 3);
        assert_eq!(q.depth(Priority::Normal), 2);
        assert_eq!(q.depth(Priority::Interactive), 1);
        q.dequeue(); // removes Interactive
        assert_eq!(q.depth(Priority::Interactive), 0);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn deadline_miss_tracking() {
        let q = PriorityRequestQueue::default();
        let t = Instant::now();
        let past_deadline = t; // deadline is now, dequeue later
        q.enqueue(entry_at(
            "expired",
            Priority::Normal,
            t,
            Some(past_deadline),
        ))
        .unwrap();
        // Dequeue 1 second later → miss
        let later = t + Duration::from_secs(1);
        let e = q.dequeue_at(later).unwrap();
        assert_eq!(e.id, "expired");
        let m = q.metrics();
        assert_eq!(m.deadline_misses, 1);
    }

    #[test]
    fn wait_time_metric() {
        let q = PriorityRequestQueue::default();
        let t0 = Instant::now();
        q.enqueue(entry_at("a", Priority::Normal, t0, None))
            .unwrap();
        // Dequeue 2 seconds later
        let t1 = t0 + Duration::from_secs(2);
        q.dequeue_at(t1);
        let m = q.metrics();
        assert!(
            m.avg_wait_secs >= 1.9 && m.avg_wait_secs <= 2.1,
            "avg_wait_secs should be ~2.0, got {}",
            m.avg_wait_secs
        );
    }

    #[test]
    fn drain_expired_removes_past_deadline() {
        let q = PriorityRequestQueue::default();
        let t = Instant::now();
        let dl_soon = t + Duration::from_millis(100);
        let dl_far = t + Duration::from_secs(60);
        q.enqueue(entry_at("soon", Priority::Normal, t, Some(dl_soon)))
            .unwrap();
        q.enqueue(entry_at("far", Priority::Normal, t, Some(dl_far)))
            .unwrap();
        q.enqueue(entry_at("none", Priority::Normal, t, None))
            .unwrap();
        // 1 second later: "soon" is expired
        let later = t + Duration::from_secs(1);
        let expired = q.drain_expired_at(later);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].id, "soon");
        // Two remain
        assert_eq!(q.len(), 2);
        let m = q.metrics();
        assert_eq!(m.deadline_misses, 1);
    }

    #[test]
    fn peek_priority_returns_highest() {
        let q = PriorityRequestQueue::default();
        assert!(q.peek_priority().is_none());
        let t = Instant::now();
        q.enqueue(entry_at("bg", Priority::Background, t, None))
            .unwrap();
        assert_eq!(q.peek_priority(), Some(Priority::Background));
        q.enqueue(entry_at("inter", Priority::Interactive, t, None))
            .unwrap();
        assert_eq!(q.peek_priority(), Some(Priority::Interactive));
    }

    #[test]
    fn empty_dequeue_returns_none() {
        let q = PriorityRequestQueue::default();
        assert!(q.dequeue().is_none());
        assert!(q.is_empty());
    }

    #[test]
    fn metrics_initial_state() {
        let q = PriorityRequestQueue::default();
        let m = q.metrics();
        assert_eq!(m.total_depth, 0);
        assert_eq!(m.total_dequeued, 0);
        assert_eq!(m.deadline_misses, 0);
        assert_eq!(m.avg_wait_secs, 0.0);
    }
}
