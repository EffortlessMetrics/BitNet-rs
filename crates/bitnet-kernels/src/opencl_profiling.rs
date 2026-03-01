//! OpenCL kernel profiling framework for Intel Arc A770 performance analysis.
//!
//! Provides event-based timing infrastructure to measure GPU kernel execution
//! with nanosecond precision, collecting queue, submit, start, and end timestamps
//! from OpenCL profiling events.

use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// KernelProfile — single kernel execution record
// ---------------------------------------------------------------------------

/// A single kernel execution timing record derived from OpenCL event timestamps.
///
/// Timing fields use nanosecond precision matching `CL_PROFILING_COMMAND_*` values.
#[derive(Debug, Clone)]
pub struct KernelProfile {
    /// Kernel function name.
    pub kernel_name: String,
    /// Global work-group dimensions.
    pub global_work_size: Vec<usize>,
    /// Local work-group dimensions.
    pub local_work_size: Vec<usize>,
    /// `CL_PROFILING_COMMAND_QUEUED` — timestamp when command was enqueued (ns).
    pub queued_ns: u64,
    /// `CL_PROFILING_COMMAND_SUBMIT` — timestamp when command was submitted to device (ns).
    pub submit_ns: u64,
    /// `CL_PROFILING_COMMAND_START` — timestamp when command started executing (ns).
    pub start_ns: u64,
    /// `CL_PROFILING_COMMAND_END` — timestamp when command finished executing (ns).
    pub end_ns: u64,
}

impl KernelProfile {
    /// Time spent waiting in the host queue before submission (microseconds).
    pub fn queue_latency_us(&self) -> f64 {
        self.submit_ns.saturating_sub(self.queued_ns) as f64 / 1_000.0
    }

    /// Time spent on device executing the kernel (microseconds).
    pub fn execution_time_us(&self) -> f64 {
        self.end_ns.saturating_sub(self.start_ns) as f64 / 1_000.0
    }

    /// Total wall time from enqueue to completion (microseconds).
    pub fn total_time_us(&self) -> f64 {
        self.end_ns.saturating_sub(self.queued_ns) as f64 / 1_000.0
    }

    /// Effective memory bandwidth in GB/s given the number of bytes transferred.
    pub fn bandwidth_gb_s(&self, bytes_transferred: usize) -> f64 {
        let exec_s = self.end_ns.saturating_sub(self.start_ns) as f64 / 1e9;
        if exec_s == 0.0 {
            return 0.0;
        }
        bytes_transferred as f64 / exec_s / 1e9
    }

    /// Effective compute throughput in GFLOPS.
    pub fn gflops(&self, flop_count: u64) -> f64 {
        let exec_s = self.end_ns.saturating_sub(self.start_ns) as f64 / 1e9;
        if exec_s == 0.0 {
            return 0.0;
        }
        flop_count as f64 / exec_s / 1e9
    }
}

// ---------------------------------------------------------------------------
// KernelStats — per-kernel aggregate statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics for a named kernel across multiple invocations.
#[derive(Debug, Clone)]
pub struct KernelStats {
    pub count: usize,
    pub total_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub avg_us: f64,
    pub std_dev_us: f64,
}

// ---------------------------------------------------------------------------
// SessionSummary
// ---------------------------------------------------------------------------

/// High-level summary of a profiling session.
#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub total_kernels: usize,
    pub total_gpu_time_ms: f64,
    pub avg_kernel_time_us: f64,
    pub kernel_breakdown: HashMap<String, KernelStats>,
}

// ---------------------------------------------------------------------------
// ProfilingSession — collection of profiles
// ---------------------------------------------------------------------------

/// Collects [`KernelProfile`] records during a measurement session.
#[derive(Debug)]
pub struct ProfilingSession {
    profiles: Vec<KernelProfile>,
    start_time: Instant,
}

impl Default for ProfilingSession {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfilingSession {
    /// Create a new, empty profiling session.
    pub fn new() -> Self {
        Self { profiles: Vec::new(), start_time: Instant::now() }
    }

    /// Record a kernel execution profile.
    pub fn record(&mut self, profile: KernelProfile) {
        self.profiles.push(profile);
    }

    /// Wall-clock duration since session creation.
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Number of recorded profiles.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Whether the session has no recorded profiles.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Return all profiles for a given kernel name.
    pub fn by_kernel(&self, name: &str) -> Vec<&KernelProfile> {
        self.profiles.iter().filter(|p| p.kernel_name == name).collect()
    }

    /// Return the `n` slowest kernels by execution time.
    pub fn slowest(&self, n: usize) -> Vec<&KernelProfile> {
        let mut sorted: Vec<&KernelProfile> = self.profiles.iter().collect();
        sorted.sort_by(|a, b| {
            let a_exec = a.end_ns.saturating_sub(a.start_ns);
            let b_exec = b.end_ns.saturating_sub(b.start_ns);
            b_exec.cmp(&a_exec)
        });
        sorted.truncate(n);
        sorted
    }

    /// Total GPU execution time across all recorded kernels (milliseconds).
    pub fn total_gpu_time_ms(&self) -> f64 {
        self.profiles.iter().map(|p| p.execution_time_us()).sum::<f64>() / 1_000.0
    }

    /// Compute a [`SessionSummary`] with per-kernel statistics.
    pub fn summary(&self) -> SessionSummary {
        let total_kernels = self.profiles.len();
        let total_gpu_time_ms = self.total_gpu_time_ms();
        let avg_kernel_time_us = if total_kernels == 0 {
            0.0
        } else {
            total_gpu_time_ms * 1_000.0 / total_kernels as f64
        };

        // Group by kernel name.
        let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
        for p in &self.profiles {
            groups.entry(p.kernel_name.clone()).or_default().push(p.execution_time_us());
        }

        let kernel_breakdown = groups
            .into_iter()
            .map(|(name, times)| {
                let count = times.len();
                let total_us: f64 = times.iter().sum();
                let min_us = times.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_us = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let avg_us = total_us / count as f64;
                let variance = if count > 1 {
                    times.iter().map(|t| (t - avg_us).powi(2)).sum::<f64>() / (count - 1) as f64
                } else {
                    0.0
                };
                let std_dev_us = variance.sqrt();
                (name, KernelStats { count, total_us, min_us, max_us, avg_us, std_dev_us })
            })
            .collect();

        SessionSummary { total_kernels, total_gpu_time_ms, avg_kernel_time_us, kernel_breakdown }
    }
}

// ---------------------------------------------------------------------------
// ProfilingReport — formatted output
// ---------------------------------------------------------------------------

/// Formatted output for a [`SessionSummary`].
pub struct ProfilingReport {
    summary: SessionSummary,
}

impl ProfilingReport {
    pub fn new(summary: SessionSummary) -> Self {
        Self { summary }
    }

    /// Render the summary as an ASCII table.
    pub fn to_table(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!(
            "OpenCL Kernel Profiling Report\n\
             ==============================\n\
             Total kernels : {}\n\
             Total GPU time: {:.3} ms\n\
             Avg kernel    : {:.3} µs\n\n",
            self.summary.total_kernels,
            self.summary.total_gpu_time_ms,
            self.summary.avg_kernel_time_us,
        ));

        // Sort by total time descending for the table.
        let mut entries: Vec<(&String, &KernelStats)> =
            self.summary.kernel_breakdown.iter().collect();
        entries.sort_by(|a, b| b.1.total_us.partial_cmp(&a.1.total_us).unwrap());

        let header = format!(
            "{:<30} {:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "Kernel", "Count", "Total(µs)", "Min(µs)", "Max(µs)", "Avg(µs)", "StdDev(µs)"
        );
        let sep = "-".repeat(header.len());
        out.push_str(&header);
        out.push('\n');
        out.push_str(&sep);
        out.push('\n');

        for (name, stats) in &entries {
            out.push_str(&format!(
                "{:<30} {:>6} {:>12.1} {:>12.1} {:>12.1} {:>12.1} {:>12.1}\n",
                truncate_name(name, 30),
                stats.count,
                stats.total_us,
                stats.min_us,
                stats.max_us,
                stats.avg_us,
                stats.std_dev_us,
            ));
        }
        out.push_str(&sep);
        out.push('\n');
        out
    }

    /// Serialise the summary as a JSON string.
    pub fn to_json(&self) -> String {
        // Hand-rolled to avoid pulling in serde for this module.
        let mut out = String::from("{\n");
        out.push_str(&format!("  \"total_kernels\": {},\n", self.summary.total_kernels));
        out.push_str(&format!("  \"total_gpu_time_ms\": {:.6},\n", self.summary.total_gpu_time_ms));
        out.push_str(&format!(
            "  \"avg_kernel_time_us\": {:.6},\n",
            self.summary.avg_kernel_time_us
        ));
        out.push_str("  \"kernel_breakdown\": {\n");

        let mut entries: Vec<(&String, &KernelStats)> =
            self.summary.kernel_breakdown.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        for (i, (name, stats)) in entries.iter().enumerate() {
            out.push_str(&format!(
                "    \"{}\": {{\n\
                 \x20     \"count\": {},\n\
                 \x20     \"total_us\": {:.6},\n\
                 \x20     \"min_us\": {:.6},\n\
                 \x20     \"max_us\": {:.6},\n\
                 \x20     \"avg_us\": {:.6},\n\
                 \x20     \"std_dev_us\": {:.6}\n\
                 \x20   }}",
                name,
                stats.count,
                stats.total_us,
                stats.min_us,
                stats.max_us,
                stats.avg_us,
                stats.std_dev_us,
            ));
            if i + 1 < entries.len() {
                out.push(',');
            }
            out.push('\n');
        }

        out.push_str("  }\n}");
        out
    }
}

/// Truncate a kernel name to `max_len` characters, appending `..` if needed.
fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len { name.to_string() } else { format!("{}..", &name[..max_len - 2]) }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_profile(name: &str, queued: u64, submit: u64, start: u64, end: u64) -> KernelProfile {
        KernelProfile {
            kernel_name: name.to_string(),
            global_work_size: vec![256],
            local_work_size: vec![64],
            queued_ns: queued,
            submit_ns: submit,
            start_ns: start,
            end_ns: end,
        }
    }

    fn sample_profile() -> KernelProfile {
        // queued=1000, submit=2000, start=3000, end=13000 → 10 µs exec
        make_profile("matmul", 1_000, 2_000, 3_000, 13_000)
    }

    // -----------------------------------------------------------------------
    // KernelProfile derived metrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_queue_latency() {
        let p = sample_profile();
        // submit - queued = 1000 ns = 1.0 µs
        assert!((p.queue_latency_us() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_execution_time() {
        let p = sample_profile();
        // end - start = 10000 ns = 10.0 µs
        assert!((p.execution_time_us() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_total_time() {
        let p = sample_profile();
        // end - queued = 12000 ns = 12.0 µs
        assert!((p.total_time_us() - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bandwidth() {
        let p = make_profile("copy", 0, 0, 0, 1_000_000_000); // 1 second
        // 4 GB in 1 s → 4 GB/s
        let bw = p.bandwidth_gb_s(4_000_000_000);
        assert!((bw - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_gflops() {
        let p = make_profile("gemm", 0, 0, 0, 1_000_000_000); // 1 second
        // 2 GFLOP in 1 s → 2 GFLOPS
        let gf = p.gflops(2_000_000_000);
        assert!((gf - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_duration_bandwidth() {
        let p = make_profile("noop", 0, 0, 100, 100);
        assert_eq!(p.bandwidth_gb_s(1024), 0.0);
    }

    #[test]
    fn test_zero_duration_gflops() {
        let p = make_profile("noop", 0, 0, 100, 100);
        assert_eq!(p.gflops(1_000_000), 0.0);
    }

    #[test]
    fn test_zero_duration_execution_time() {
        let p = make_profile("noop", 5, 5, 5, 5);
        assert_eq!(p.execution_time_us(), 0.0);
        assert_eq!(p.queue_latency_us(), 0.0);
        assert_eq!(p.total_time_us(), 0.0);
    }

    #[test]
    fn test_very_large_timing_values() {
        // Simulate ~1000 seconds on GPU (realistic for extreme kernels)
        let p = make_profile("huge", 0, 0, 0, 1_000_000_000_000);
        let exec_us = p.execution_time_us();
        assert!((exec_us - 1_000_000_000.0).abs() < 0.01);
    }

    #[test]
    fn test_saturating_sub_prevents_underflow() {
        // Nonsensical: queued > end — saturating_sub should yield 0, not panic.
        let p = make_profile("bad", 999, 500, 200, 100);
        assert_eq!(p.execution_time_us(), 0.0);
        assert_eq!(p.total_time_us(), 0.0);
        assert_eq!(p.queue_latency_us(), 0.0);
    }

    // -----------------------------------------------------------------------
    // ProfilingSession basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_session() {
        let session = ProfilingSession::new();
        assert!(session.is_empty());
        assert_eq!(session.len(), 0);
        assert_eq!(session.total_gpu_time_ms(), 0.0);
    }

    #[test]
    fn test_single_kernel_session() {
        let mut session = ProfilingSession::new();
        session.record(sample_profile());
        assert_eq!(session.len(), 1);
        assert!(!session.is_empty());
        // 10 µs = 0.01 ms
        assert!((session.total_gpu_time_ms() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_session_default() {
        let session = ProfilingSession::default();
        assert!(session.is_empty());
    }

    // -----------------------------------------------------------------------
    // ProfilingSession aggregation
    // -----------------------------------------------------------------------

    #[test]
    fn test_total_gpu_time_multiple() {
        let mut session = ProfilingSession::new();
        // Two kernels: 10 µs + 20 µs = 30 µs = 0.03 ms
        session.record(make_profile("a", 0, 0, 0, 10_000));
        session.record(make_profile("b", 0, 0, 0, 20_000));
        assert!((session.total_gpu_time_ms() - 0.03).abs() < 1e-9);
    }

    #[test]
    fn test_by_kernel_filtering() {
        let mut session = ProfilingSession::new();
        session.record(make_profile("matmul", 0, 0, 0, 10_000));
        session.record(make_profile("softmax", 0, 0, 0, 5_000));
        session.record(make_profile("matmul", 0, 0, 0, 15_000));
        let matmuls = session.by_kernel("matmul");
        assert_eq!(matmuls.len(), 2);
        assert!(session.by_kernel("nonexistent").is_empty());
    }

    #[test]
    fn test_slowest_n() {
        let mut session = ProfilingSession::new();
        session.record(make_profile("fast", 0, 0, 0, 1_000)); // 1 µs
        session.record(make_profile("slow", 0, 0, 0, 100_000)); // 100 µs
        session.record(make_profile("mid", 0, 0, 0, 50_000)); // 50 µs
        let top2 = session.slowest(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].kernel_name, "slow");
        assert_eq!(top2[1].kernel_name, "mid");
    }

    #[test]
    fn test_slowest_n_exceeding_count() {
        let mut session = ProfilingSession::new();
        session.record(make_profile("a", 0, 0, 0, 5_000));
        let top5 = session.slowest(5);
        assert_eq!(top5.len(), 1);
    }

    // -----------------------------------------------------------------------
    // SessionSummary correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_summary() {
        let session = ProfilingSession::new();
        let s = session.summary();
        assert_eq!(s.total_kernels, 0);
        assert_eq!(s.total_gpu_time_ms, 0.0);
        assert_eq!(s.avg_kernel_time_us, 0.0);
        assert!(s.kernel_breakdown.is_empty());
    }

    #[test]
    fn test_summary_single_kernel() {
        let mut session = ProfilingSession::new();
        session.record(make_profile("relu", 0, 100, 200, 10_200)); // 10 µs exec
        let s = session.summary();
        assert_eq!(s.total_kernels, 1);
        assert!((s.avg_kernel_time_us - 10.0).abs() < 1e-9);
        let stats = &s.kernel_breakdown["relu"];
        assert_eq!(stats.count, 1);
        assert!((stats.avg_us - 10.0).abs() < 1e-9);
        assert_eq!(stats.std_dev_us, 0.0);
    }

    #[test]
    fn test_summary_kernel_stats_min_max_avg() {
        let mut session = ProfilingSession::new();
        session.record(make_profile("k", 0, 0, 0, 10_000)); // 10 µs
        session.record(make_profile("k", 0, 0, 0, 20_000)); // 20 µs
        session.record(make_profile("k", 0, 0, 0, 30_000)); // 30 µs
        let s = session.summary();
        let stats = &s.kernel_breakdown["k"];
        assert_eq!(stats.count, 3);
        assert!((stats.min_us - 10.0).abs() < 1e-9);
        assert!((stats.max_us - 30.0).abs() < 1e-9);
        assert!((stats.avg_us - 20.0).abs() < 1e-9);
        assert!((stats.total_us - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_summary_std_dev() {
        let mut session = ProfilingSession::new();
        // Two samples: 10 and 20 → mean=15, var=(25+25)/1=50, sd≈7.071
        session.record(make_profile("k", 0, 0, 0, 10_000));
        session.record(make_profile("k", 0, 0, 0, 20_000));
        let stats = &session.summary().kernel_breakdown["k"];
        assert!((stats.std_dev_us - 50.0_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_summary_multiple_kernels() {
        let mut session = ProfilingSession::new();
        session.record(make_profile("matmul", 0, 0, 0, 10_000));
        session.record(make_profile("softmax", 0, 0, 0, 5_000));
        let s = session.summary();
        assert_eq!(s.total_kernels, 2);
        assert_eq!(s.kernel_breakdown.len(), 2);
        assert!(s.kernel_breakdown.contains_key("matmul"));
        assert!(s.kernel_breakdown.contains_key("softmax"));
    }

    #[test]
    fn test_total_gpu_time_property() {
        // Property: total_gpu_time_ms >= each individual kernel's execution time.
        let mut session = ProfilingSession::new();
        session.record(make_profile("a", 0, 0, 0, 10_000));
        session.record(make_profile("b", 0, 0, 0, 20_000));
        session.record(make_profile("c", 0, 0, 0, 30_000));
        let total = session.total_gpu_time_ms();
        for p in
            [&session.by_kernel("a")[0], &session.by_kernel("b")[0], &session.by_kernel("c")[0]]
        {
            assert!(total >= p.execution_time_us() / 1_000.0);
        }
        // total = sum
        let sum = session
            .by_kernel("a")
            .iter()
            .chain(session.by_kernel("b").iter())
            .chain(session.by_kernel("c").iter())
            .map(|p| p.execution_time_us())
            .sum::<f64>()
            / 1_000.0;
        assert!((total - sum).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // ProfilingReport formatting
    // -----------------------------------------------------------------------

    #[test]
    fn test_report_table_contains_header() {
        let session = ProfilingSession::new();
        let report = ProfilingReport::new(session.summary());
        let table = report.to_table();
        assert!(table.contains("OpenCL Kernel Profiling Report"));
        assert!(table.contains("Total kernels"));
    }

    #[test]
    fn test_report_table_contains_kernel_rows() {
        let mut session = ProfilingSession::new();
        session.record(make_profile("matmul_i2s", 0, 0, 0, 50_000));
        let report = ProfilingReport::new(session.summary());
        let table = report.to_table();
        assert!(table.contains("matmul_i2s"));
    }

    #[test]
    fn test_report_json_valid_structure() {
        let mut session = ProfilingSession::new();
        session.record(make_profile("relu", 0, 0, 0, 8_000));
        let report = ProfilingReport::new(session.summary());
        let json = report.to_json();
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        assert!(json.contains("\"total_kernels\": 1"));
        assert!(json.contains("\"relu\""));
        assert!(json.contains("\"count\": 1"));
    }

    #[test]
    fn test_report_json_empty() {
        let session = ProfilingSession::new();
        let report = ProfilingReport::new(session.summary());
        let json = report.to_json();
        assert!(json.contains("\"total_kernels\": 0"));
    }

    #[test]
    fn test_report_table_long_kernel_name_truncated() {
        let long_name = "a".repeat(50);
        let mut session = ProfilingSession::new();
        session.record(make_profile(&long_name, 0, 0, 0, 5_000));
        let report = ProfilingReport::new(session.summary());
        let table = report.to_table();
        // The table column is 30 chars wide; the name should be truncated with ".."
        assert!(table.contains(".."));
    }

    #[test]
    fn test_truncate_name_short() {
        assert_eq!(truncate_name("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_name_exact() {
        assert_eq!(truncate_name("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_name_long() {
        assert_eq!(truncate_name("hello_world", 7), "hello..");
    }

    // -----------------------------------------------------------------------
    // Work-size stored correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_work_size_preserved() {
        let p = KernelProfile {
            kernel_name: "conv2d".into(),
            global_work_size: vec![1024, 1024],
            local_work_size: vec![16, 16],
            queued_ns: 0,
            submit_ns: 0,
            start_ns: 0,
            end_ns: 1_000,
        };
        assert_eq!(p.global_work_size, vec![1024, 1024]);
        assert_eq!(p.local_work_size, vec![16, 16]);
    }
}
