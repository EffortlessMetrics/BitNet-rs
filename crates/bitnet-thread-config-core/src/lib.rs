//! Thread pool configuration and work partitioning primitives.

/// Thread pool configuration.
#[derive(Debug, Clone)]
pub struct ThreadConfig {
    pub num_threads: usize,
    pub pin_threads: bool,
    pub priority: ThreadPriority,
    pub stack_size: usize,
}

/// Thread priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadPriority {
    Low,
    Normal,
    High,
    Realtime,
}

impl Default for ThreadConfig {
    fn default() -> Self {
        Self {
            num_threads: available_parallelism(),
            pin_threads: false,
            priority: ThreadPriority::Normal,
            stack_size: 8 * 1024 * 1024,
        }
    }
}

impl ThreadConfig {
    #[must_use]
    pub fn new(num_threads: usize) -> Self {
        Self { num_threads: num_threads.max(1), ..Default::default() }
    }

    #[must_use]
    pub fn single_threaded() -> Self {
        Self::new(1)
    }

    #[must_use]
    pub fn with_pin(mut self) -> Self {
        self.pin_threads = true;
        self
    }

    #[must_use]
    pub fn with_priority(mut self, priority: ThreadPriority) -> Self {
        self.priority = priority;
        self
    }

    #[must_use]
    pub fn with_stack_size(mut self, size: usize) -> Self {
        self.stack_size = size;
        self
    }
}

/// Get available parallelism (number of logical CPUs).
#[must_use]
pub fn available_parallelism() -> usize {
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
}

/// Work partitioning: divide `total` items across `threads`.
#[derive(Debug, Clone)]
pub struct WorkPartition {
    pub thread_id: usize,
    pub start: usize,
    pub end: usize,
}

impl WorkPartition {
    #[must_use]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Partition `total` items across `num_threads` as evenly as possible.
#[must_use]
pub fn partition_work(total: usize, num_threads: usize) -> Vec<WorkPartition> {
    partitioning::build_partitions(total, num_threads)
}

mod partitioning {
    use super::WorkPartition;

    pub(super) fn build_partitions(total: usize, num_threads: usize) -> Vec<WorkPartition> {
        let threads = num_threads.max(1);
        let base_chunk = total / threads;
        let remainder = total % threads;
        let mut partitions = Vec::with_capacity(threads);
        let mut start = 0;

        for thread_id in 0..threads {
            let chunk_size = chunk_size_for(thread_id, base_chunk, remainder);
            let partition = partition_for(thread_id, start, chunk_size);
            start = partition.end;
            partitions.push(partition);
        }

        partitions
    }

    fn chunk_size_for(thread_id: usize, base_chunk: usize, remainder: usize) -> usize {
        let extra = usize::from(thread_id < remainder);
        base_chunk + extra
    }

    fn partition_for(thread_id: usize, start: usize, chunk_size: usize) -> WorkPartition {
        WorkPartition { thread_id, start, end: start + chunk_size }
    }
}

/// Estimate optimal thread count for a given workload size.
#[must_use]
pub fn optimal_threads(work_items: usize, min_items_per_thread: usize) -> usize {
    let max_threads = available_parallelism();
    let min_per = min_items_per_thread.max(1);
    let needed = work_items.div_ceil(min_per);
    needed.clamp(1, max_threads)
}

/// Check if work should be parallelized (heuristic).
#[must_use]
pub fn should_parallelize(work_items: usize, threshold: usize) -> bool {
    work_items >= threshold && available_parallelism() > 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ThreadConfig::default();
        assert!(config.num_threads >= 1);
        assert!(!config.pin_threads);
        assert_eq!(config.priority, ThreadPriority::Normal);
    }

    #[test]
    fn test_single_threaded() {
        let config = ThreadConfig::single_threaded();
        assert_eq!(config.num_threads, 1);
    }

    #[test]
    fn test_builder() {
        let config = ThreadConfig::new(4)
            .with_pin()
            .with_priority(ThreadPriority::High)
            .with_stack_size(16 * 1024 * 1024);
        assert_eq!(config.num_threads, 4);
        assert!(config.pin_threads);
        assert_eq!(config.priority, ThreadPriority::High);
    }

    #[test]
    fn test_available_parallelism() {
        assert!(available_parallelism() >= 1);
    }

    #[test]
    fn test_partition_even() {
        let parts = partition_work(12, 3);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].start, 0);
        assert_eq!(parts[0].end, 4);
        assert_eq!(parts[1].start, 4);
        assert_eq!(parts[1].end, 8);
        assert_eq!(parts[2].start, 8);
        assert_eq!(parts[2].end, 12);
    }

    #[test]
    fn test_partition_uneven() {
        let parts = partition_work(10, 3);
        assert_eq!(parts.len(), 3);
        let total: usize = parts.iter().map(WorkPartition::len).sum();
        assert_eq!(total, 10);
        assert!(parts[0].len() >= parts[2].len());
    }

    #[test]
    fn test_partition_more_threads_than_work() {
        let parts = partition_work(2, 5);
        assert_eq!(parts.len(), 5);
        let non_empty: Vec<_> = parts.iter().filter(|p| !p.is_empty()).collect();
        assert_eq!(non_empty.len(), 2);
    }

    #[test]
    fn test_partition_zero_work() {
        let parts = partition_work(0, 4);
        assert!(parts.iter().all(WorkPartition::is_empty));
    }

    #[test]
    fn test_optimal_threads() {
        let t = optimal_threads(100, 10);
        assert!(t >= 1);
        assert!(t <= available_parallelism());
    }

    #[test]
    fn test_optimal_threads_tiny() {
        let t = optimal_threads(5, 100);
        assert_eq!(t, 1);
    }

    #[test]
    fn test_should_parallelize() {
        assert!(!should_parallelize(1, 100));
        let _ = should_parallelize(10000, 100);
    }

    #[test]
    fn test_min_one_thread() {
        let config = ThreadConfig::new(0);
        assert_eq!(config.num_threads, 1);
    }
}
