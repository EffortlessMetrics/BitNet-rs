//! Kernel execution profiling.
//!
//! Provides [`KernelProfile`] for individual kernel measurements,
//! [`KernelProfiler`] for collecting profiles, and [`KernelSummary`]
//! for aggregated statistics by kernel name.

use std::collections::HashMap;

/// A single kernel execution profile.
#[derive(Debug, Clone)]
pub struct KernelProfile {
    /// Name identifying the kernel (e.g. "matmul", "attention").
    pub kernel_name: String,
    /// Wall-clock execution time in microseconds.
    pub execution_time_us: f64,
    /// Number of input elements processed.
    pub input_elements: usize,
    /// Number of output elements produced.
    pub output_elements: usize,
    /// Estimated memory bytes touched during execution.
    pub memory_bytes: usize,
}

impl KernelProfile {
    /// Rough FLOPS estimate based on total elements processed.
    pub fn flops_estimate(&self) -> f64 {
        let total_elems = (self.input_elements + self.output_elements) as f64;
        let secs = self.execution_time_us / 1_000_000.0;
        if secs <= 0.0 {
            return 0.0;
        }
        total_elems / secs
    }

    /// Memory bandwidth in GB/s.
    pub fn bandwidth_gbps(&self) -> f64 {
        let secs = self.execution_time_us / 1_000_000.0;
        if secs <= 0.0 {
            return 0.0;
        }
        self.memory_bytes as f64 / secs / 1e9
    }

    /// Elements processed per second (input + output).
    pub fn elements_per_second(&self) -> f64 {
        let total_elems = (self.input_elements + self.output_elements) as f64;
        let secs = self.execution_time_us / 1_000_000.0;
        if secs <= 0.0 {
            return 0.0;
        }
        total_elems / secs
    }
}

/// Aggregated summary for all invocations of a given kernel name.
#[derive(Debug, Clone)]
pub struct KernelSummary {
    pub kernel_name: String,
    pub call_count: usize,
    pub total_time_us: f64,
    pub avg_time_us: f64,
    pub max_time_us: f64,
}

/// Collects [`KernelProfile`] records and provides analysis helpers.
#[derive(Debug)]
pub struct KernelProfiler {
    profiles: Vec<KernelProfile>,
    enabled: bool,
}

impl KernelProfiler {
    /// Create an enabled profiler.
    pub fn new() -> Self {
        Self { profiles: Vec::new(), enabled: true }
    }

    /// Create a disabled (no-op) profiler.
    pub fn disabled() -> Self {
        Self { profiles: Vec::new(), enabled: false }
    }

    /// Record a kernel execution. Ignored when the profiler is disabled.
    pub fn record(
        &mut self,
        name: &str,
        time_us: f64,
        input_elems: usize,
        output_elems: usize,
        mem_bytes: usize,
    ) {
        if !self.enabled {
            return;
        }
        self.profiles.push(KernelProfile {
            kernel_name: name.to_string(),
            execution_time_us: time_us,
            input_elements: input_elems,
            output_elements: output_elems,
            memory_bytes: mem_bytes,
        });
    }

    /// All collected profiles.
    pub fn get_profiles(&self) -> &[KernelProfile] {
        &self.profiles
    }

    /// Sum of execution times across all profiles.
    pub fn total_time_us(&self) -> f64 {
        self.profiles.iter().map(|p| p.execution_time_us).sum()
    }

    /// Profile with the highest execution time.
    pub fn hottest_kernel(&self) -> Option<&KernelProfile> {
        self.profiles.iter().max_by(|a, b| {
            a.execution_time_us
                .partial_cmp(&b.execution_time_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Aggregate profiles by kernel name.
    pub fn summary_by_kernel(&self) -> Vec<KernelSummary> {
        let mut map: HashMap<&str, (usize, f64, f64)> = HashMap::new();
        for p in &self.profiles {
            let entry = map.entry(p.kernel_name.as_str()).or_insert((0, 0.0, 0.0));
            entry.0 += 1;
            entry.1 += p.execution_time_us;
            if p.execution_time_us > entry.2 {
                entry.2 = p.execution_time_us;
            }
        }
        let mut summaries: Vec<KernelSummary> = map
            .into_iter()
            .map(|(name, (count, total, max))| KernelSummary {
                kernel_name: name.to_string(),
                call_count: count,
                total_time_us: total,
                avg_time_us: if count > 0 { total / count as f64 } else { 0.0 },
                max_time_us: max,
            })
            .collect();
        summaries.sort_by(|a, b| {
            b.total_time_us.partial_cmp(&a.total_time_us).unwrap_or(std::cmp::Ordering::Equal)
        });
        summaries
    }

    /// Remove all recorded profiles.
    pub fn clear(&mut self) {
        self.profiles.clear();
    }
}

impl Default for KernelProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_profiler() {
        let p = KernelProfiler::new();
        assert!(p.get_profiles().is_empty());
        assert_eq!(p.total_time_us(), 0.0);
        assert!(p.hottest_kernel().is_none());
        assert!(p.summary_by_kernel().is_empty());
    }

    #[test]
    fn test_single_record() {
        let mut p = KernelProfiler::new();
        p.record("matmul", 100.0, 1024, 512, 8192);
        assert_eq!(p.get_profiles().len(), 1);
        let prof = &p.get_profiles()[0];
        assert_eq!(prof.kernel_name, "matmul");
        assert_eq!(prof.execution_time_us, 100.0);
        assert_eq!(prof.input_elements, 1024);
        assert_eq!(prof.output_elements, 512);
        assert_eq!(prof.memory_bytes, 8192);
    }

    #[test]
    fn test_multiple_records_total_time() {
        let mut p = KernelProfiler::new();
        p.record("matmul", 100.0, 1024, 512, 8192);
        p.record("attention", 50.0, 512, 512, 4096);
        p.record("softmax", 25.0, 256, 256, 2048);
        assert_eq!(p.get_profiles().len(), 3);
        assert!((p.total_time_us() - 175.0).abs() < 1e-9);
    }

    #[test]
    fn test_hottest_kernel() {
        let mut p = KernelProfiler::new();
        p.record("attention", 50.0, 512, 512, 4096);
        p.record("matmul", 200.0, 1024, 512, 8192);
        p.record("softmax", 25.0, 256, 256, 2048);
        let hot = p.hottest_kernel().unwrap();
        assert_eq!(hot.kernel_name, "matmul");
        assert_eq!(hot.execution_time_us, 200.0);
    }

    #[test]
    fn test_hottest_kernel_single() {
        let mut p = KernelProfiler::new();
        p.record("norm", 42.0, 100, 100, 800);
        let hot = p.hottest_kernel().unwrap();
        assert_eq!(hot.kernel_name, "norm");
    }

    #[test]
    fn test_summary_aggregation() {
        let mut p = KernelProfiler::new();
        p.record("matmul", 100.0, 1024, 512, 8192);
        p.record("matmul", 200.0, 1024, 512, 8192);
        p.record("attention", 50.0, 512, 512, 4096);
        let summaries = p.summary_by_kernel();
        assert_eq!(summaries.len(), 2);
        // Sorted by total_time descending: matmul first
        let mm = &summaries[0];
        assert_eq!(mm.kernel_name, "matmul");
        assert_eq!(mm.call_count, 2);
        assert!((mm.total_time_us - 300.0).abs() < 1e-9);
        assert!((mm.avg_time_us - 150.0).abs() < 1e-9);
        assert!((mm.max_time_us - 200.0).abs() < 1e-9);
        let attn = &summaries[1];
        assert_eq!(attn.kernel_name, "attention");
        assert_eq!(attn.call_count, 1);
    }

    #[test]
    fn test_disabled_profiler() {
        let mut p = KernelProfiler::disabled();
        p.record("matmul", 100.0, 1024, 512, 8192);
        p.record("attention", 50.0, 512, 512, 4096);
        assert!(p.get_profiles().is_empty());
        assert_eq!(p.total_time_us(), 0.0);
        assert!(p.hottest_kernel().is_none());
    }

    #[test]
    fn test_flops_estimate() {
        let prof = KernelProfile {
            kernel_name: "matmul".to_string(),
            execution_time_us: 1_000_000.0, // 1 second
            input_elements: 500,
            output_elements: 500,
            memory_bytes: 4000,
        };
        // 1000 elements / 1 second = 1000 FLOPS
        assert!((prof.flops_estimate() - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_bandwidth_gbps() {
        let prof = KernelProfile {
            kernel_name: "matmul".to_string(),
            execution_time_us: 1_000_000.0, // 1 second
            input_elements: 1024,
            output_elements: 1024,
            memory_bytes: 1_000_000_000, // 1 GB
        };
        // 1 GB / 1 s = 1.0 GB/s
        assert!((prof.bandwidth_gbps() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_elements_per_second() {
        let prof = KernelProfile {
            kernel_name: "norm".to_string(),
            execution_time_us: 500_000.0, // 0.5 seconds
            input_elements: 1000,
            output_elements: 1000,
            memory_bytes: 8000,
        };
        // 2000 elements / 0.5 s = 4000 elem/s
        assert!((prof.elements_per_second() - 4000.0).abs() < 1e-9);
    }

    #[test]
    fn test_zero_time_guards() {
        let prof = KernelProfile {
            kernel_name: "empty".to_string(),
            execution_time_us: 0.0,
            input_elements: 100,
            output_elements: 100,
            memory_bytes: 800,
        };
        assert_eq!(prof.flops_estimate(), 0.0);
        assert_eq!(prof.bandwidth_gbps(), 0.0);
        assert_eq!(prof.elements_per_second(), 0.0);
    }

    #[test]
    fn test_clear_resets() {
        let mut p = KernelProfiler::new();
        p.record("matmul", 100.0, 1024, 512, 8192);
        p.record("attention", 50.0, 512, 512, 4096);
        assert_eq!(p.get_profiles().len(), 2);
        p.clear();
        assert!(p.get_profiles().is_empty());
        assert_eq!(p.total_time_us(), 0.0);
        assert!(p.hottest_kernel().is_none());
    }

    #[test]
    fn test_default_is_enabled() {
        let p = KernelProfiler::default();
        // default() delegates to new() which is enabled
        p.get_profiles(); // should not panic
        assert_eq!(p.total_time_us(), 0.0);
    }

    #[test]
    fn test_summary_single_kernel_multiple_calls() {
        let mut p = KernelProfiler::new();
        p.record("softmax", 10.0, 64, 64, 512);
        p.record("softmax", 30.0, 64, 64, 512);
        p.record("softmax", 20.0, 64, 64, 512);
        let summaries = p.summary_by_kernel();
        assert_eq!(summaries.len(), 1);
        let s = &summaries[0];
        assert_eq!(s.call_count, 3);
        assert!((s.total_time_us - 60.0).abs() < 1e-9);
        assert!((s.avg_time_us - 20.0).abs() < 1e-9);
        assert!((s.max_time_us - 30.0).abs() < 1e-9);
    }
}
