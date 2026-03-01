//! OpenCL-aware thread pool for pipeline parallelism.
//!
//! Orchestrates CPU compute, GPU kernel submission, and host↔device data
//! transfer through a priority-ordered work queue. The current implementation
//! provides single-threaded CPU reference execution; a future version will
//! dispatch to real OpenCL command queues.

use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the thread pool.
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    pub num_compute_threads: usize,
    pub num_transfer_threads: usize,
    pub num_submit_threads: usize,
    pub queue_capacity: usize,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_compute_threads: 4,
            num_transfer_threads: 2,
            num_submit_threads: 1,
            queue_capacity: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Priority of a work item (higher variants are processed first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskPriority {
    Background = 0,
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Direction of a host↔device transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
}

/// CPU-side operation tag.
#[derive(Debug, Clone, PartialEq)]
pub enum CpuOp {
    MatMul { rows: usize, cols: usize, k: usize },
    Softmax,
    RmsNorm { eps: f32 },
    ElementwiseAdd,
}

/// The kind of work a [`WorkItem`] carries.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkType {
    CpuCompute { input: Vec<f32>, operation: CpuOp },
    GpuTransfer { data: Vec<f32>, direction: TransferDirection },
    GpuSubmit { kernel_name: String, data: Vec<f32> },
}

// ---------------------------------------------------------------------------
// Work items & results
// ---------------------------------------------------------------------------

/// A single unit of work in the queue.
#[derive(Debug, Clone)]
pub struct WorkItem {
    pub id: u64,
    pub task_type: WorkType,
    pub priority: TaskPriority,
    pub submitted_ns: u64,
}

/// Result produced after executing a [`WorkItem`].
#[derive(Debug, Clone)]
pub struct WorkResult {
    pub id: u64,
    pub output: Vec<f32>,
    pub completed_ns: u64,
    pub thread_id: usize,
}

// ---------------------------------------------------------------------------
// Stats & errors
// ---------------------------------------------------------------------------

/// Aggregate statistics for the pool.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PoolThreadStats {
    pub total_submitted: u64,
    pub total_completed: u64,
    pub total_compute_time_us: u64,
    pub total_transfer_time_us: u64,
    pub total_submit_time_us: u64,
    pub queue_high_watermark: usize,
}

/// Errors returned by pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    QueueFull,
    InvalidTask,
    ShutdownInProgress,
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull => write!(f, "work queue is full"),
            Self::InvalidTask => write!(f, "invalid task"),
            Self::ShutdownInProgress => write!(f, "pool is shutting down"),
        }
    }
}

impl std::error::Error for PoolError {}

// ---------------------------------------------------------------------------
// Thread pool
// ---------------------------------------------------------------------------

/// Single-threaded reference thread pool.
#[derive(Debug)]
pub struct ThreadPool {
    pub config: ThreadPoolConfig,
    pending: Vec<WorkItem>,
    completed: Vec<WorkResult>,
    next_id: u64,
    stats: PoolThreadStats,
}

// ---------------------------------------------------------------------------
// Public API — construction
// ---------------------------------------------------------------------------

/// Create a new [`ThreadPool`] from the given configuration.
pub fn create_thread_pool(config: ThreadPoolConfig) -> ThreadPool {
    ThreadPool {
        config,
        pending: Vec::new(),
        completed: Vec::new(),
        next_id: 1,
        stats: PoolThreadStats::default(),
    }
}

// ---------------------------------------------------------------------------
// Public API — submission & processing
// ---------------------------------------------------------------------------

/// Enqueue a work item. Returns the assigned work-item ID on success.
pub fn cpu_submit_work(
    pool: &mut ThreadPool,
    work_type: WorkType,
    priority: TaskPriority,
) -> Result<u64, PoolError> {
    if pool.pending.len() >= pool.config.queue_capacity {
        return Err(PoolError::QueueFull);
    }
    let id = pool.next_id;
    pool.next_id += 1;
    pool.pending.push(WorkItem {
        id,
        task_type: work_type,
        priority,
        submitted_ns: now_ns(),
    });
    pool.stats.total_submitted += 1;
    if pool.pending.len() > pool.stats.queue_high_watermark {
        pool.stats.queue_high_watermark = pool.pending.len();
    }
    Ok(id)
}

/// Process the highest-priority pending item, returning the result.
pub fn cpu_process_next(pool: &mut ThreadPool) -> Option<WorkResult> {
    if pool.pending.is_empty() {
        return None;
    }
    // Find the index of the highest-priority item (FIFO within same priority).
    let best_idx = pool
        .pending
        .iter()
        .enumerate()
        .enumerate()
        .max_by(|(seq_a, (_, a)), (seq_b, (_, b))| {
            a.priority.cmp(&b.priority).then(seq_b.cmp(seq_a))
        })
        .map(|(_, (i, _))| i)
        .expect("pending is non-empty");

    let item = pool.pending.remove(best_idx);
    let start = now_ns();
    let output = execute_work_item(&item.task_type);
    let elapsed_us = (now_ns().saturating_sub(start)) / 1000;

    match &item.task_type {
        WorkType::CpuCompute { .. } => pool.stats.total_compute_time_us += elapsed_us,
        WorkType::GpuTransfer { .. } => pool.stats.total_transfer_time_us += elapsed_us,
        WorkType::GpuSubmit { .. } => pool.stats.total_submit_time_us += elapsed_us,
    }

    pool.stats.total_completed += 1;

    let result = WorkResult { id: item.id, output, completed_ns: now_ns(), thread_id: 0 };
    pool.completed.push(result.clone());
    Some(result)
}

/// Drain the pending queue, processing every item and returning all results.
pub fn cpu_process_all(pool: &mut ThreadPool) -> Vec<WorkResult> {
    let mut results = Vec::new();
    while let Some(r) = cpu_process_next(pool) {
        results.push(r);
    }
    results
}

// ---------------------------------------------------------------------------
// Public API — CPU reference compute helpers
// ---------------------------------------------------------------------------

/// Execute a CPU compute operation on `input`.
pub fn cpu_execute_compute(op: &CpuOp, input: &[f32]) -> Vec<f32> {
    match op {
        CpuOp::MatMul { rows, cols, k } => cpu_matmul(input, *rows, *cols, *k),
        CpuOp::Softmax => cpu_softmax(input),
        CpuOp::RmsNorm { eps } => cpu_rms_norm(input, *eps),
        CpuOp::ElementwiseAdd => {
            // Identity — nothing to add to; return a copy.
            input.to_vec()
        }
    }
}

/// Simulate a host↔device transfer (memcpy semantics).
pub fn cpu_simulate_transfer(data: &[f32], _direction: &TransferDirection) -> Vec<f32> {
    data.to_vec()
}

/// Simulate a GPU kernel submission. Returns data scaled by 2.0 as a
/// stand-in for a real kernel.
pub fn cpu_simulate_submit(kernel_name: &str, data: &[f32]) -> Vec<f32> {
    let _ = kernel_name;
    data.iter().map(|&v| v * 2.0).collect()
}

// ---------------------------------------------------------------------------
// Public API — query & management
// ---------------------------------------------------------------------------

/// Number of pending items.
pub fn cpu_get_queue_depth(pool: &ThreadPool) -> usize {
    pool.pending.len()
}

/// Snapshot of pool statistics.
pub fn cpu_get_stats(pool: &ThreadPool) -> PoolThreadStats {
    pool.stats.clone()
}

/// Cancel a pending work item by ID. Returns `true` if found and removed.
pub fn cpu_cancel_work(pool: &mut ThreadPool, id: u64) -> bool {
    let before = pool.pending.len();
    pool.pending.retain(|item| item.id != id);
    pool.pending.len() < before
}

/// Remove and return all pending items without executing them.
pub fn cpu_drain_queue(pool: &mut ThreadPool) -> Vec<WorkItem> {
    std::mem::take(&mut pool.pending)
}

/// Human-readable pool status.
pub fn format_pool_status(pool: &ThreadPool) -> String {
    format!(
        "ThreadPool {{ pending: {}, completed: {}, submitted: {}, \
         compute_threads: {}, transfer_threads: {}, submit_threads: {} }}",
        pool.pending.len(),
        pool.completed.len(),
        pool.stats.total_submitted,
        pool.config.num_compute_threads,
        pool.config.num_transfer_threads,
        pool.config.num_submit_threads,
    )
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn execute_work_item(work_type: &WorkType) -> Vec<f32> {
    match work_type {
        WorkType::CpuCompute { input, operation } => cpu_execute_compute(operation, input),
        WorkType::GpuTransfer { data, direction } => cpu_simulate_transfer(data, direction),
        WorkType::GpuSubmit { kernel_name, data } => cpu_simulate_submit(kernel_name, data),
    }
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Reference scalar matmul: C[m×n] = A[m×k] × B[k×n].
///
/// `input` is interpreted as `A` concatenated with `B`.
fn cpu_matmul(input: &[f32], rows: usize, cols: usize, k: usize) -> Vec<f32> {
    let a_len = rows * k;
    let b_len = k * cols;
    if input.len() < a_len + b_len {
        return vec![0.0; rows * cols];
    }
    let a = &input[..a_len];
    let b = &input[a_len..a_len + b_len];
    let mut c = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * cols + j];
            }
            c[i * cols + j] = sum;
        }
    }
    c
}

/// Numerically-stable softmax.
fn cpu_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; input.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// RMS-Norm: x / sqrt(mean(x²) + eps).
fn cpu_rms_norm(input: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let mean_sq: f32 = input.iter().map(|&x| x * x).sum::<f32>() / input.len() as f32;
    let rms = (mean_sq + eps).sqrt();
    input.iter().map(|&x| x / rms).collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers -----------------------------------------------------------

    fn default_pool() -> ThreadPool {
        create_thread_pool(ThreadPoolConfig::default())
    }

    fn pool_with_capacity(cap: usize) -> ThreadPool {
        create_thread_pool(ThreadPoolConfig { queue_capacity: cap, ..Default::default() })
    }

    fn simple_compute(vals: Vec<f32>) -> WorkType {
        WorkType::CpuCompute { input: vals, operation: CpuOp::Softmax }
    }

    fn simple_transfer(vals: Vec<f32>) -> WorkType {
        WorkType::GpuTransfer { data: vals, direction: TransferDirection::HostToDevice }
    }

    fn simple_submit(vals: Vec<f32>) -> WorkType {
        WorkType::GpuSubmit { kernel_name: "test_kernel".into(), data: vals }
    }

    // -- pool creation -----------------------------------------------------

    #[test]
    fn test_create_pool_empty_queue() {
        let pool = default_pool();
        assert_eq!(cpu_get_queue_depth(&pool), 0);
    }

    #[test]
    fn test_create_pool_default_config() {
        let pool = default_pool();
        assert_eq!(pool.config.num_compute_threads, 4);
        assert_eq!(pool.config.num_transfer_threads, 2);
        assert_eq!(pool.config.num_submit_threads, 1);
        assert_eq!(pool.config.queue_capacity, 256);
    }

    #[test]
    fn test_create_pool_custom_config() {
        let pool = create_thread_pool(ThreadPoolConfig {
            num_compute_threads: 8,
            num_transfer_threads: 4,
            num_submit_threads: 2,
            queue_capacity: 512,
        });
        assert_eq!(pool.config.num_compute_threads, 8);
        assert_eq!(pool.config.queue_capacity, 512);
    }

    // -- submit work -------------------------------------------------------

    #[test]
    fn test_submit_work_returns_id() {
        let mut pool = default_pool();
        let id = cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal);
        assert_eq!(id, Ok(1));
    }

    #[test]
    fn test_submit_work_increments_id() {
        let mut pool = default_pool();
        let id1 = cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal).unwrap();
        let id2 = cpu_submit_work(&mut pool, simple_compute(vec![2.0]), TaskPriority::Normal).unwrap();
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn test_submit_work_increases_depth() {
        let mut pool = default_pool();
        cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal).unwrap();
        assert_eq!(cpu_get_queue_depth(&pool), 1);
    }

    // -- process next ------------------------------------------------------

    #[test]
    fn test_process_next_returns_result() {
        let mut pool = default_pool();
        cpu_submit_work(&mut pool, simple_compute(vec![1.0, 2.0]), TaskPriority::Normal).unwrap();
        let result = cpu_process_next(&mut pool);
        assert!(result.is_some());
    }

    #[test]
    fn test_process_next_highest_priority_first() {
        let mut pool = default_pool();
        let low_id =
            cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Low).unwrap();
        let high_id =
            cpu_submit_work(&mut pool, simple_compute(vec![2.0]), TaskPriority::High).unwrap();
        let _normal_id =
            cpu_submit_work(&mut pool, simple_compute(vec![3.0]), TaskPriority::Normal).unwrap();

        let r1 = cpu_process_next(&mut pool).unwrap();
        assert_eq!(r1.id, high_id);
        let _r2 = cpu_process_next(&mut pool).unwrap();
        let r3 = cpu_process_next(&mut pool).unwrap();
        assert_eq!(r3.id, low_id);
    }

    #[test]
    fn test_process_next_empty_returns_none() {
        let mut pool = default_pool();
        assert!(cpu_process_next(&mut pool).is_none());
    }

    #[test]
    fn test_process_next_decrements_depth() {
        let mut pool = default_pool();
        cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal).unwrap();
        cpu_process_next(&mut pool);
        assert_eq!(cpu_get_queue_depth(&pool), 0);
    }

    // -- process all -------------------------------------------------------

    #[test]
    fn test_process_all_empties_queue() {
        let mut pool = default_pool();
        for i in 0..5 {
            cpu_submit_work(&mut pool, simple_compute(vec![i as f32]), TaskPriority::Normal)
                .unwrap();
        }
        let results = cpu_process_all(&mut pool);
        assert_eq!(results.len(), 5);
        assert_eq!(cpu_get_queue_depth(&pool), 0);
    }

    #[test]
    fn test_process_all_empty_pool() {
        let mut pool = default_pool();
        let results = cpu_process_all(&mut pool);
        assert!(results.is_empty());
    }

    // -- compute correctness -----------------------------------------------

    #[test]
    fn test_execute_compute_matmul_2x2() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let op = CpuOp::MatMul { rows: 2, cols: 2, k: 2 };
        let out = cpu_execute_compute(&op, &input);
        assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_execute_compute_matmul_short_input() {
        let op = CpuOp::MatMul { rows: 2, cols: 2, k: 2 };
        let out = cpu_execute_compute(&op, &[1.0]);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn test_execute_compute_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let out = cpu_execute_compute(&CpuOp::Softmax, &input);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1, got {sum}");
        // Each element should be positive.
        assert!(out.iter().all(|&v| v > 0.0));
        // Monotonically increasing for increasing inputs.
        assert!(out[0] < out[1] && out[1] < out[2]);
    }

    #[test]
    fn test_execute_compute_softmax_empty() {
        let out = cpu_execute_compute(&CpuOp::Softmax, &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_execute_compute_rms_norm() {
        let input = vec![3.0, 4.0];
        let out = cpu_execute_compute(&CpuOp::RmsNorm { eps: 1e-6 }, &input);
        // rms = sqrt((9+16)/2 + 1e-6) ≈ sqrt(12.5) ≈ 3.5355
        let rms = (12.5f32 + 1e-6).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-4);
        assert!((out[1] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_execute_compute_elementwise_add() {
        let input = vec![1.0, 2.0, 3.0];
        let out = cpu_execute_compute(&CpuOp::ElementwiseAdd, &input);
        assert_eq!(out, input);
    }

    // -- transfer & submit -------------------------------------------------

    #[test]
    fn test_simulate_transfer_preserves_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let out_h2d = cpu_simulate_transfer(&data, &TransferDirection::HostToDevice);
        let out_d2h = cpu_simulate_transfer(&data, &TransferDirection::DeviceToHost);
        assert_eq!(out_h2d, data);
        assert_eq!(out_d2h, data);
    }

    #[test]
    fn test_simulate_transfer_empty() {
        let out = cpu_simulate_transfer(&[], &TransferDirection::HostToDevice);
        assert!(out.is_empty());
    }

    #[test]
    fn test_simulate_submit_scales_data() {
        let data = vec![1.0, 2.0, 3.0];
        let out = cpu_simulate_submit("test", &data);
        assert_eq!(out, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_submit_and_process_transfer() {
        let mut pool = default_pool();
        cpu_submit_work(&mut pool, simple_transfer(vec![10.0, 20.0]), TaskPriority::Normal)
            .unwrap();
        let r = cpu_process_next(&mut pool).unwrap();
        assert_eq!(r.output, vec![10.0, 20.0]);
    }

    #[test]
    fn test_submit_and_process_gpu_submit() {
        let mut pool = default_pool();
        cpu_submit_work(&mut pool, simple_submit(vec![5.0]), TaskPriority::Normal).unwrap();
        let r = cpu_process_next(&mut pool).unwrap();
        assert_eq!(r.output, vec![10.0]);
    }

    // -- queue full --------------------------------------------------------

    #[test]
    fn test_queue_full_error() {
        let mut pool = pool_with_capacity(2);
        cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal).unwrap();
        cpu_submit_work(&mut pool, simple_compute(vec![2.0]), TaskPriority::Normal).unwrap();
        let err = cpu_submit_work(&mut pool, simple_compute(vec![3.0]), TaskPriority::Normal);
        assert_eq!(err, Err(PoolError::QueueFull));
    }

    #[test]
    fn test_queue_capacity_one() {
        let mut pool = pool_with_capacity(1);
        cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal).unwrap();
        assert_eq!(
            cpu_submit_work(&mut pool, simple_compute(vec![2.0]), TaskPriority::Normal),
            Err(PoolError::QueueFull)
        );
        // After processing, we can submit again.
        cpu_process_next(&mut pool);
        assert!(
            cpu_submit_work(&mut pool, simple_compute(vec![3.0]), TaskPriority::Normal).is_ok()
        );
    }

    // -- cancel & drain ----------------------------------------------------

    #[test]
    fn test_cancel_work_valid_id() {
        let mut pool = default_pool();
        let id = cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal).unwrap();
        assert!(cpu_cancel_work(&mut pool, id));
        assert_eq!(cpu_get_queue_depth(&pool), 0);
    }

    #[test]
    fn test_cancel_work_invalid_id() {
        let mut pool = default_pool();
        cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal).unwrap();
        assert!(!cpu_cancel_work(&mut pool, 999));
        assert_eq!(cpu_get_queue_depth(&pool), 1);
    }

    #[test]
    fn test_drain_queue_returns_all() {
        let mut pool = default_pool();
        for i in 0..3 {
            cpu_submit_work(&mut pool, simple_compute(vec![i as f32]), TaskPriority::Normal)
                .unwrap();
        }
        let items = cpu_drain_queue(&mut pool);
        assert_eq!(items.len(), 3);
        assert_eq!(cpu_get_queue_depth(&pool), 0);
    }

    #[test]
    fn test_drain_empty_queue() {
        let mut pool = default_pool();
        let items = cpu_drain_queue(&mut pool);
        assert!(items.is_empty());
    }

    // -- stats -------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let pool = default_pool();
        let stats = cpu_get_stats(&pool);
        assert_eq!(stats.total_submitted, 0);
        assert_eq!(stats.total_completed, 0);
    }

    #[test]
    fn test_stats_after_submit_and_process() {
        let mut pool = default_pool();
        cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal).unwrap();
        cpu_submit_work(&mut pool, simple_transfer(vec![2.0]), TaskPriority::Normal).unwrap();
        cpu_process_all(&mut pool);
        let stats = cpu_get_stats(&pool);
        assert_eq!(stats.total_submitted, 2);
        assert_eq!(stats.total_completed, 2);
    }

    #[test]
    fn test_stats_high_watermark() {
        let mut pool = default_pool();
        for i in 0..5 {
            cpu_submit_work(&mut pool, simple_compute(vec![i as f32]), TaskPriority::Normal)
                .unwrap();
        }
        cpu_process_all(&mut pool);
        let stats = cpu_get_stats(&pool);
        assert_eq!(stats.queue_high_watermark, 5);
    }

    // -- priority ordering -------------------------------------------------

    #[test]
    fn test_priority_ordering_all_levels() {
        let mut pool = default_pool();
        let bg = cpu_submit_work(&mut pool, simple_compute(vec![0.0]), TaskPriority::Background).unwrap();
        let low = cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Low).unwrap();
        let norm = cpu_submit_work(&mut pool, simple_compute(vec![2.0]), TaskPriority::Normal).unwrap();
        let high = cpu_submit_work(&mut pool, simple_compute(vec![3.0]), TaskPriority::High).unwrap();
        let crit = cpu_submit_work(&mut pool, simple_compute(vec![4.0]), TaskPriority::Critical).unwrap();

        let results = cpu_process_all(&mut pool);
        let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![crit, high, norm, low, bg]);
    }

    #[test]
    fn test_priority_fifo_within_same_level() {
        let mut pool = default_pool();
        let id1 = cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal).unwrap();
        let id2 = cpu_submit_work(&mut pool, simple_compute(vec![2.0]), TaskPriority::Normal).unwrap();
        let id3 = cpu_submit_work(&mut pool, simple_compute(vec![3.0]), TaskPriority::Normal).unwrap();

        let results = cpu_process_all(&mut pool);
        let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![id1, id2, id3]);
    }

    // -- multiple submits --------------------------------------------------

    #[test]
    fn test_multiple_submits_mixed_types() {
        let mut pool = default_pool();
        cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal).unwrap();
        cpu_submit_work(&mut pool, simple_transfer(vec![2.0]), TaskPriority::High).unwrap();
        cpu_submit_work(&mut pool, simple_submit(vec![3.0]), TaskPriority::Low).unwrap();
        assert_eq!(cpu_get_queue_depth(&pool), 3);

        let results = cpu_process_all(&mut pool);
        assert_eq!(results.len(), 3);
        // Transfer (High) should be processed first.
        assert_eq!(results[0].output, vec![2.0]);
    }

    // -- property tests ----------------------------------------------------

    #[test]
    fn test_completed_le_submitted() {
        let mut pool = default_pool();
        for i in 0..10 {
            cpu_submit_work(&mut pool, simple_compute(vec![i as f32]), TaskPriority::Normal)
                .unwrap();
        }
        // Process only half.
        for _ in 0..5 {
            cpu_process_next(&mut pool);
        }
        let stats = cpu_get_stats(&pool);
        assert!(stats.total_completed <= stats.total_submitted);
    }

    #[test]
    fn test_queue_depth_non_negative() {
        let pool = default_pool();
        // usize is always >= 0, but this validates the API contract.
        assert_eq!(cpu_get_queue_depth(&pool), 0);
    }

    // -- format ------------------------------------------------------------

    #[test]
    fn test_format_pool_status() {
        let pool = default_pool();
        let status = format_pool_status(&pool);
        assert!(status.contains("pending: 0"));
        assert!(status.contains("completed: 0"));
    }

    #[test]
    fn test_format_pool_status_after_work() {
        let mut pool = default_pool();
        cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal).unwrap();
        cpu_process_all(&mut pool);
        let status = format_pool_status(&pool);
        assert!(status.contains("completed: 1"));
        assert!(status.contains("submitted: 1"));
    }

    // -- error display -----------------------------------------------------

    #[test]
    fn test_pool_error_display() {
        assert_eq!(PoolError::QueueFull.to_string(), "work queue is full");
        assert_eq!(PoolError::InvalidTask.to_string(), "invalid task");
        assert_eq!(PoolError::ShutdownInProgress.to_string(), "pool is shutting down");
    }

    // -- work result thread id ---------------------------------------------

    #[test]
    fn test_work_result_thread_id_is_zero() {
        let mut pool = default_pool();
        cpu_submit_work(&mut pool, simple_compute(vec![1.0]), TaskPriority::Normal).unwrap();
        let r = cpu_process_next(&mut pool).unwrap();
        assert_eq!(r.thread_id, 0);
    }

    // -- matmul non-square -------------------------------------------------

    #[test]
    fn test_matmul_1x3_times_3x1() {
        // A=[1,2,3] B=[4;5;6] → C=[1*4+2*5+3*6]=[32]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let op = CpuOp::MatMul { rows: 1, cols: 1, k: 3 };
        let out = cpu_execute_compute(&op, &input);
        assert_eq!(out, vec![32.0]);
    }
}
