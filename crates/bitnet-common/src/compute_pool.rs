//! Compute thread pool configuration and management.
//!
//! Configures parallelism for inference workloads based on
//! available hardware and workload characteristics.

/// Thread pool configuration.
#[derive(Debug, Clone)]
pub struct ComputePoolConfig {
    pub num_threads: usize,
    pub stack_size_bytes: usize,
    pub pin_threads: bool,
    pub name_prefix: String,
}

impl ComputePoolConfig {
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            stack_size_bytes: 8 * 1024 * 1024, // 8 MB default
            pin_threads: false,
            name_prefix: "bitnet-worker".to_string(),
        }
    }

    /// Create config using all available cores.
    pub fn auto() -> Self {
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        Self::new(cpus)
    }

    /// Create config for memory-bound workloads (fewer threads).
    pub fn memory_bound() -> Self {
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        Self::new((cpus / 2).max(1))
    }

    /// Create config for compute-bound workloads (all cores).
    pub fn compute_bound() -> Self {
        Self::auto()
    }

    pub fn with_stack_size(mut self, bytes: usize) -> Self {
        self.stack_size_bytes = bytes;
        self
    }

    pub fn with_pin(mut self, pin: bool) -> Self {
        self.pin_threads = pin;
        self
    }

    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.name_prefix = prefix.into();
        self
    }

    /// Estimate memory overhead for this pool configuration.
    pub fn estimated_overhead_bytes(&self) -> usize {
        self.num_threads * self.stack_size_bytes
    }

    /// Compute optimal chunk size for dividing work across threads.
    pub fn chunk_size(&self, total_items: usize) -> usize {
        if self.num_threads == 0 {
            return total_items;
        }
        total_items.div_ceil(self.num_threads).max(1)
    }
}

impl Default for ComputePoolConfig {
    fn default() -> Self {
        Self::auto()
    }
}

/// Workload type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    /// Compute-bound (matmul, activations).
    ComputeBound,
    /// Memory-bound (embeddings, KV cache).
    MemoryBound,
    /// IO-bound (model loading, tokenization).
    IoBound,
    /// Mixed workload.
    Mixed,
}

impl WorkloadType {
    /// Suggested thread count multiplier for this workload type.
    pub fn thread_multiplier(&self) -> f64 {
        match self {
            WorkloadType::ComputeBound => 1.0,
            WorkloadType::MemoryBound => 0.5,
            WorkloadType::IoBound => 2.0,
            WorkloadType::Mixed => 1.0,
        }
    }

    /// Create a config suited for this workload type.
    pub fn recommended_config(&self) -> ComputePoolConfig {
        let base = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let threads = ((base as f64 * self.thread_multiplier()) as usize).max(1);
        ComputePoolConfig::new(threads)
    }
}

/// Partition a range into chunks for parallel execution.
pub fn partition_work(total: usize, num_workers: usize) -> Vec<(usize, usize)> {
    if num_workers == 0 || total == 0 {
        return vec![];
    }
    let chunk = total.div_ceil(num_workers);
    let mut parts = Vec::new();
    let mut start = 0;
    while start < total {
        let end = (start + chunk).min(total);
        parts.push((start, end));
        start = end;
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let c = ComputePoolConfig::new(4);
        assert_eq!(c.num_threads, 4);
        assert_eq!(c.stack_size_bytes, 8 * 1024 * 1024);
    }

    #[test]
    fn test_config_auto() {
        let c = ComputePoolConfig::auto();
        assert!(c.num_threads >= 1);
    }

    #[test]
    fn test_config_memory_bound() {
        let c = ComputePoolConfig::memory_bound();
        assert!(c.num_threads >= 1);
    }

    #[test]
    fn test_builder() {
        let c = ComputePoolConfig::new(8)
            .with_stack_size(4 * 1024 * 1024)
            .with_pin(true)
            .with_prefix("test");
        assert_eq!(c.stack_size_bytes, 4 * 1024 * 1024);
        assert!(c.pin_threads);
        assert_eq!(c.name_prefix, "test");
    }

    #[test]
    fn test_overhead() {
        let c = ComputePoolConfig::new(4).with_stack_size(1024);
        assert_eq!(c.estimated_overhead_bytes(), 4096);
    }

    #[test]
    fn test_chunk_size() {
        let c = ComputePoolConfig::new(4);
        assert_eq!(c.chunk_size(100), 25);
        assert_eq!(c.chunk_size(7), 2);
    }

    #[test]
    fn test_partition_work() {
        let parts = partition_work(10, 3);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], (0, 4));
        assert_eq!(parts[1], (4, 8));
        assert_eq!(parts[2], (8, 10));
    }

    #[test]
    fn test_partition_exact() {
        let parts = partition_work(8, 4);
        assert_eq!(parts.len(), 4);
        for &(s, e) in &parts {
            assert_eq!(e - s, 2);
        }
    }

    #[test]
    fn test_partition_empty() {
        assert!(partition_work(0, 4).is_empty());
        assert!(partition_work(10, 0).is_empty());
    }

    #[test]
    fn test_workload_type() {
        assert_eq!(WorkloadType::ComputeBound.thread_multiplier(), 1.0);
        assert_eq!(WorkloadType::MemoryBound.thread_multiplier(), 0.5);
        assert_eq!(WorkloadType::IoBound.thread_multiplier(), 2.0);
    }

    #[test]
    fn test_recommended_config() {
        let c = WorkloadType::ComputeBound.recommended_config();
        assert!(c.num_threads >= 1);
    }

    #[test]
    fn test_chunk_zero_threads() {
        let c = ComputePoolConfig::new(0);
        assert_eq!(c.chunk_size(100), 100);
    }
}
