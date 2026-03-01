//! Host-GPU bandwidth optimization for OpenCL transfers.
//!
//! Provides pinned memory allocation, transfer batching, transfer-compute overlap
//! scheduling, and bandwidth measurement/reporting for Intel Arc GPUs.

use log::{debug, info, warn};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Pinned memory allocation
// ---------------------------------------------------------------------------

/// A pinned (page-locked) memory buffer for fast DMA transfers.
///
/// On systems without real OpenCL pinned allocation this falls back to a
/// regular heap `Vec<u8>` but tracks the allocation intent so callers can
/// query [`is_pinned`](PinnedBuffer::is_pinned).
#[derive(Debug)]
pub struct PinnedBuffer {
    data: Vec<u8>,
    pinned: bool,
    capacity: usize,
}

impl PinnedBuffer {
    /// Allocate a new pinned buffer of `size` bytes.
    ///
    /// When real OpenCL host-pinned memory is unavailable the buffer is
    /// heap-allocated and [`is_pinned`](Self::is_pinned) returns `false`.
    pub fn new(size: usize) -> Self {
        debug!("Allocating pinned buffer: {} bytes", size);
        let (data, pinned) = Self::try_alloc_pinned(size);
        Self {
            data,
            pinned,
            capacity: size,
        }
    }

    fn try_alloc_pinned(size: usize) -> (Vec<u8>, bool) {
        // In production this would call clCreateBuffer with
        // CL_MEM_ALLOC_HOST_PTR and map it.  For now we use a regular
        // allocation and mark it as unpinned.
        (vec![0u8; size], false)
    }

    /// Returns `true` if the memory is actually page-locked.
    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Write `src` into the buffer starting at `offset`.
    ///
    /// # Panics
    /// Panics if `offset + src.len()` exceeds capacity.
    pub fn write(&mut self, offset: usize, src: &[u8]) {
        assert!(
            offset + src.len() <= self.capacity,
            "PinnedBuffer::write out of bounds"
        );
        self.data[offset..offset + src.len()].copy_from_slice(src);
    }

    /// Read `len` bytes starting at `offset`.
    pub fn read(&self, offset: usize, len: usize) -> &[u8] {
        &self.data[offset..offset + len]
    }

    /// Mutable access to the underlying byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Immutable access to the underlying byte slice.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// Transfer batching
// ---------------------------------------------------------------------------

/// A pending host→device transfer request.
#[derive(Debug, Clone)]
pub struct TransferRequest {
    /// Logical identifier for the tensor / buffer being transferred.
    pub id: u64,
    /// Payload bytes to send.
    pub data: Vec<u8>,
    /// Priority (lower = higher priority).
    pub priority: u32,
}

/// Collects small transfers and flushes them in a single batched DMA
/// operation when the batch is full or explicitly flushed.
#[derive(Debug)]
pub struct TransferBatcher {
    queue: VecDeque<TransferRequest>,
    /// Maximum number of bytes to accumulate before auto-flush.
    batch_size_limit: usize,
    /// Current accumulated payload size.
    current_size: usize,
    /// Total bytes flushed across all batches (for reporting).
    total_bytes_flushed: u64,
    /// Number of batches flushed.
    batches_flushed: u64,
}

impl TransferBatcher {
    /// Create a new batcher that auto-flushes at `batch_size_limit` bytes.
    pub fn new(batch_size_limit: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            batch_size_limit,
            current_size: 0,
            total_bytes_flushed: 0,
            batches_flushed: 0,
        }
    }

    /// Enqueue a transfer request.  Returns `Some(batch)` if the batch
    /// size limit is reached and an auto-flush occurs.
    pub fn enqueue(&mut self, req: TransferRequest) -> Option<Vec<TransferRequest>> {
        self.current_size += req.data.len();
        self.queue.push_back(req);

        if self.current_size >= self.batch_size_limit {
            Some(self.flush())
        } else {
            None
        }
    }

    /// Flush all pending transfers, returning them sorted by priority.
    pub fn flush(&mut self) -> Vec<TransferRequest> {
        let mut batch: Vec<TransferRequest> = self.queue.drain(..).collect();
        batch.sort_by_key(|r| r.priority);
        let flushed_bytes: usize = batch.iter().map(|r| r.data.len()).sum();
        self.total_bytes_flushed += flushed_bytes as u64;
        self.batches_flushed += 1;
        self.current_size = 0;
        debug!(
            "Flushed batch #{}: {} transfers, {} bytes",
            self.batches_flushed,
            batch.len(),
            flushed_bytes
        );
        batch
    }

    /// Number of pending (unflushed) transfers.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Cumulative bytes flushed.
    pub fn total_bytes_flushed(&self) -> u64 {
        self.total_bytes_flushed
    }

    /// Number of batches flushed so far.
    pub fn batches_flushed(&self) -> u64 {
        self.batches_flushed
    }
}

// ---------------------------------------------------------------------------
// Transfer-compute overlap scheduling
// ---------------------------------------------------------------------------

/// Describes one unit of work that can be either a transfer or a compute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkItem {
    /// Host → Device transfer of `bytes` bytes.
    Transfer { id: u64, bytes: usize },
    /// GPU compute depending on transfer `depends_on`.
    Compute { id: u64, depends_on: Option<u64> },
}

/// Execution plan entry produced by [`OverlapScheduler`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledOp {
    /// The work item.
    pub item: WorkItem,
    /// Pipeline stage (items with the same stage can run concurrently).
    pub stage: u32,
}

/// Schedules transfers and computes so they can overlap on dual-engine GPUs.
///
/// The scheduler assigns pipeline stages: independent transfers and computes
/// that don't share a dependency are placed in the same stage.
#[derive(Debug)]
pub struct OverlapScheduler {
    items: Vec<WorkItem>,
}

impl OverlapScheduler {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Add a work item to the schedule.
    pub fn add(&mut self, item: WorkItem) {
        self.items.push(item);
    }

    /// Build an execution plan.
    ///
    /// Transfers with no compute dependency are placed in stage 0.
    /// Computes whose transfer dependency is in stage N are placed in stage N+1
    /// (but if the compute has no dependency it also goes into stage 0).
    pub fn build_plan(&self) -> Vec<ScheduledOp> {
        use std::collections::HashMap;
        let mut transfer_stage: HashMap<u64, u32> = HashMap::new();
        let mut plan = Vec::new();

        // First pass: assign stages to transfers (all go to stage 0).
        for item in &self.items {
            if let WorkItem::Transfer { id, .. } = item {
                transfer_stage.insert(*id, 0);
                plan.push(ScheduledOp {
                    item: item.clone(),
                    stage: 0,
                });
            }
        }

        // Second pass: assign stages to computes.
        for item in &self.items {
            if let WorkItem::Compute { depends_on, .. } = item {
                let stage = match depends_on {
                    Some(dep_id) => transfer_stage.get(dep_id).copied().unwrap_or(0) + 1,
                    None => 0,
                };
                plan.push(ScheduledOp {
                    item: item.clone(),
                    stage,
                });
            }
        }

        plan.sort_by_key(|op| op.stage);
        plan
    }

    /// Number of queued work items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the scheduler is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Default for OverlapScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Bandwidth measurement and reporting
// ---------------------------------------------------------------------------

/// A single bandwidth measurement sample.
#[derive(Debug, Clone)]
pub struct BandwidthSample {
    /// Number of bytes transferred.
    pub bytes: u64,
    /// Time taken for the transfer.
    pub duration: Duration,
    /// Direction of the transfer.
    pub direction: TransferDirection,
}

/// Transfer direction for bandwidth measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
}

impl std::fmt::Display for TransferDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "H→D"),
            Self::DeviceToHost => write!(f, "D→H"),
        }
    }
}

/// Collects bandwidth samples and produces summary statistics.
#[derive(Debug)]
pub struct BandwidthReporter {
    samples: Vec<BandwidthSample>,
    max_samples: usize,
}

impl BandwidthReporter {
    /// Create a reporter that keeps at most `max_samples` recent entries.
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::new(),
            max_samples,
        }
    }

    /// Record a transfer measurement.
    pub fn record(&mut self, bytes: u64, duration: Duration, direction: TransferDirection) {
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(BandwidthSample {
            bytes,
            duration,
            direction,
        });
    }

    /// Record by measuring a closure's wall-clock time.
    pub fn measure<F: FnOnce()>(
        &mut self,
        bytes: u64,
        direction: TransferDirection,
        f: F,
    ) -> Duration {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed();
        self.record(bytes, elapsed, direction);
        elapsed
    }

    /// Average bandwidth in bytes/second for the given direction, or `None`
    /// if no samples exist for that direction.
    pub fn average_bandwidth(&self, direction: TransferDirection) -> Option<f64> {
        let filtered: Vec<&BandwidthSample> =
            self.samples.iter().filter(|s| s.direction == direction).collect();
        if filtered.is_empty() {
            return None;
        }
        let total_bytes: u64 = filtered.iter().map(|s| s.bytes).sum();
        let total_secs: f64 = filtered.iter().map(|s| s.duration.as_secs_f64()).sum();
        if total_secs == 0.0 {
            return None;
        }
        Some(total_bytes as f64 / total_secs)
    }

    /// Peak bandwidth observed for the given direction (bytes/sec).
    pub fn peak_bandwidth(&self, direction: TransferDirection) -> Option<f64> {
        self.samples
            .iter()
            .filter(|s| s.direction == direction)
            .map(|s| {
                let secs = s.duration.as_secs_f64();
                if secs == 0.0 { f64::INFINITY } else { s.bytes as f64 / secs }
            })
            .reduce(f64::max)
    }

    /// Number of recorded samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Generate a human-readable bandwidth report.
    pub fn report(&self) -> String {
        let h2d_avg = self.average_bandwidth(TransferDirection::HostToDevice);
        let d2h_avg = self.average_bandwidth(TransferDirection::DeviceToHost);
        let h2d_peak = self.peak_bandwidth(TransferDirection::HostToDevice);
        let d2h_peak = self.peak_bandwidth(TransferDirection::DeviceToHost);

        let fmt = |bps: Option<f64>| match bps {
            Some(b) if b >= 1e9 => format!("{:.2} GB/s", b / 1e9),
            Some(b) if b >= 1e6 => format!("{:.2} MB/s", b / 1e6),
            Some(b) => format!("{:.2} KB/s", b / 1e3),
            None => "N/A".into(),
        };

        info!(
            "Bandwidth report: H→D avg={}, D→H avg={}",
            fmt(h2d_avg),
            fmt(d2h_avg)
        );

        format!(
            "Bandwidth Report ({} samples)\n\
             ─────────────────────────────\n\
             H→D  avg: {}  peak: {}\n\
             D→H  avg: {}  peak: {}",
            self.samples.len(),
            fmt(h2d_avg),
            fmt(h2d_peak),
            fmt(d2h_avg),
            fmt(d2h_peak),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- PinnedBuffer tests --

    #[test]
    fn test_pinned_buffer_allocation() {
        let buf = PinnedBuffer::new(4096);
        assert_eq!(buf.capacity(), 4096);
        assert_eq!(buf.as_slice().len(), 4096);
    }

    #[test]
    fn test_pinned_buffer_write_read() {
        let mut buf = PinnedBuffer::new(256);
        buf.write(0, &[1, 2, 3, 4]);
        assert_eq!(buf.read(0, 4), &[1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_pinned_buffer_write_overflow_panics() {
        let mut buf = PinnedBuffer::new(4);
        buf.write(2, &[0, 0, 0]); // 2 + 3 > 4
    }

    // -- TransferBatcher tests --

    #[test]
    fn test_batcher_manual_flush() {
        let mut batcher = TransferBatcher::new(1024);
        batcher.enqueue(TransferRequest {
            id: 1,
            data: vec![0u8; 100],
            priority: 2,
        });
        batcher.enqueue(TransferRequest {
            id: 2,
            data: vec![0u8; 200],
            priority: 1,
        });
        assert_eq!(batcher.pending_count(), 2);

        let batch = batcher.flush();
        assert_eq!(batch.len(), 2);
        // Should be sorted by priority (1 before 2).
        assert_eq!(batch[0].id, 2);
        assert_eq!(batch[1].id, 1);
        assert_eq!(batcher.pending_count(), 0);
        assert_eq!(batcher.total_bytes_flushed(), 300);
    }

    #[test]
    fn test_batcher_auto_flush_on_limit() {
        let mut batcher = TransferBatcher::new(200);
        let r1 = batcher.enqueue(TransferRequest {
            id: 1,
            data: vec![0u8; 100],
            priority: 0,
        });
        assert!(r1.is_none()); // below limit

        let r2 = batcher.enqueue(TransferRequest {
            id: 2,
            data: vec![0u8; 150],
            priority: 0,
        });
        assert!(r2.is_some()); // 100+150 >= 200 → auto-flush
        assert_eq!(batcher.batches_flushed(), 1);
    }

    // -- OverlapScheduler tests --

    #[test]
    fn test_scheduler_empty() {
        let sched = OverlapScheduler::new();
        assert!(sched.is_empty());
        assert_eq!(sched.build_plan().len(), 0);
    }

    #[test]
    fn test_scheduler_transfer_then_compute() {
        let mut sched = OverlapScheduler::new();
        sched.add(WorkItem::Transfer { id: 1, bytes: 1024 });
        sched.add(WorkItem::Compute {
            id: 2,
            depends_on: Some(1),
        });

        let plan = sched.build_plan();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].stage, 0); // transfer
        assert_eq!(plan[1].stage, 1); // compute depends on transfer
    }

    #[test]
    fn test_scheduler_independent_items_same_stage() {
        let mut sched = OverlapScheduler::new();
        sched.add(WorkItem::Transfer { id: 1, bytes: 512 });
        sched.add(WorkItem::Compute {
            id: 2,
            depends_on: None,
        });

        let plan = sched.build_plan();
        assert_eq!(plan.len(), 2);
        // Both should be in stage 0 (no dependency).
        assert!(plan.iter().all(|op| op.stage == 0));
    }

    // -- BandwidthReporter tests --

    #[test]
    fn test_reporter_average_bandwidth() {
        let mut reporter = BandwidthReporter::new(100);
        // 1 GB in 1 second = 1 GB/s
        reporter.record(
            1_000_000_000,
            Duration::from_secs(1),
            TransferDirection::HostToDevice,
        );
        let avg = reporter
            .average_bandwidth(TransferDirection::HostToDevice)
            .unwrap();
        assert!((avg - 1e9).abs() < 1.0);
    }

    #[test]
    fn test_reporter_peak_bandwidth() {
        let mut reporter = BandwidthReporter::new(100);
        reporter.record(
            1_000_000,
            Duration::from_millis(10),
            TransferDirection::DeviceToHost,
        );
        reporter.record(
            1_000_000,
            Duration::from_millis(5),
            TransferDirection::DeviceToHost,
        );
        // Peak = 1MB / 0.005s = 200 MB/s
        let peak = reporter
            .peak_bandwidth(TransferDirection::DeviceToHost)
            .unwrap();
        assert!((peak - 2e8).abs() < 1e3);
    }

    #[test]
    fn test_reporter_report_formatting() {
        let mut reporter = BandwidthReporter::new(100);
        reporter.record(
            500_000_000,
            Duration::from_millis(500),
            TransferDirection::HostToDevice,
        );
        let report = reporter.report();
        assert!(report.contains("Bandwidth Report"));
        assert!(report.contains("H→D"));
        assert!(report.contains("GB/s"));
    }

    #[test]
    fn test_reporter_max_samples_eviction() {
        let mut reporter = BandwidthReporter::new(2);
        for i in 0..5 {
            reporter.record(
                (i + 1) * 1000,
                Duration::from_millis(1),
                TransferDirection::HostToDevice,
            );
        }
        assert_eq!(reporter.sample_count(), 2);
    }
}
