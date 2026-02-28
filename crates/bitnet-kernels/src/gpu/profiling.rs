//! OpenCL kernel profiling infrastructure.
//!
//! Provides event-based timing using `CL_QUEUE_PROFILING_ENABLE` and
//! `cl_event` timestamps (QUEUED → SUBMIT → START → END).
//!
//! Enable at runtime with `BITNET_OPENCL_PROFILE=1`.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Check whether profiling is enabled via environment variable.
static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Initialise the profiling flag from the environment.
///
/// Call once at startup. Reads `BITNET_OPENCL_PROFILE`.
pub fn init_profiling() {
    let enabled = std::env::var("BITNET_OPENCL_PROFILE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    PROFILING_ENABLED.store(enabled, Ordering::Relaxed);
    if enabled {
        log::info!("OpenCL profiling enabled via BITNET_OPENCL_PROFILE");
    }
}

/// Returns `true` if profiling is currently enabled.
pub fn is_profiling_enabled() -> bool {
    PROFILING_ENABLED.load(Ordering::Relaxed)
}

/// Timing breakdown for a single kernel execution.
#[derive(Debug, Clone)]
pub struct KernelTiming {
    /// Human-readable kernel name.
    pub name: String,
    /// Time the command spent queued before submission.
    pub queue_ns: u64,
    /// Time between submission and execution start.
    pub submit_ns: u64,
    /// Actual execution time on the device.
    pub exec_ns: u64,
}

impl KernelTiming {
    /// Create a new timing entry from raw nanosecond timestamps.
    ///
    /// The four timestamps correspond to the OpenCL profiling counters:
    /// `CL_PROFILING_COMMAND_QUEUED`, `CL_PROFILING_COMMAND_SUBMIT`,
    /// `CL_PROFILING_COMMAND_START`, `CL_PROFILING_COMMAND_END`.
    pub fn from_timestamps(
        name: impl Into<String>,
        queued_ns: u64,
        submit_ns: u64,
        start_ns: u64,
        end_ns: u64,
    ) -> Self {
        Self {
            name: name.into(),
            queue_ns: submit_ns.saturating_sub(queued_ns),
            submit_ns: start_ns.saturating_sub(submit_ns),
            exec_ns: end_ns.saturating_sub(start_ns),
        }
    }

    /// Total wall-clock time from queue to completion (nanoseconds).
    pub fn total_ns(&self) -> u64 {
        self.queue_ns + self.submit_ns + self.exec_ns
    }

    /// Execution time as a [`Duration`].
    pub fn exec_duration(&self) -> Duration {
        Duration::from_nanos(self.exec_ns)
    }

    /// Compute GFLOPS given the number of floating-point operations.
    ///
    /// For a matmul of dimensions M×N×K the FLOPs are `2 * M * N * K`.
    pub fn gflops(&self, flop_count: u64) -> f64 {
        if self.exec_ns == 0 {
            return 0.0;
        }
        flop_count as f64 / self.exec_ns as f64 // ns → GFLOPS
    }

    /// Compute effective memory bandwidth in GB/s.
    ///
    /// `bytes_transferred` is the total bytes read + written by the kernel.
    pub fn bandwidth_gbps(&self, bytes_transferred: u64) -> f64 {
        if self.exec_ns == 0 {
            return 0.0;
        }
        bytes_transferred as f64 / self.exec_ns as f64 // ns → GB/s
    }
}

impl fmt::Display for KernelTiming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<24} queue={:>8.3}ms  submit={:>8.3}ms  exec={:>8.3}ms  total={:>8.3}ms",
            self.name,
            self.queue_ns as f64 / 1e6,
            self.submit_ns as f64 / 1e6,
            self.exec_ns as f64 / 1e6,
            self.total_ns() as f64 / 1e6,
        )
    }
}

/// Accumulator for a complete inference run.
#[derive(Debug, Clone, Default)]
pub struct ProfilingReport {
    pub entries: Vec<KernelTiming>,
}

impl ProfilingReport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a kernel timing entry.
    pub fn record(&mut self, timing: KernelTiming) {
        self.entries.push(timing);
    }

    /// Total execution time across all kernels.
    pub fn total_exec_ns(&self) -> u64 {
        self.entries.iter().map(|t| t.exec_ns).sum()
    }

    /// Total wall-clock time across all kernels.
    pub fn total_wall_ns(&self) -> u64 {
        self.entries.iter().map(|t| t.total_ns()).sum()
    }

    /// Number of recorded kernel invocations.
    pub fn kernel_count(&self) -> usize {
        self.entries.len()
    }
}

impl fmt::Display for ProfilingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== OpenCL Profiling Report ===")?;
        writeln!(
            f,
            "{:<24} {:>12}  {:>12}  {:>12}  {:>12}",
            "Kernel", "Queue", "Submit", "Exec", "Total"
        )?;
        writeln!(f, "{}", "-".repeat(80))?;
        for entry in &self.entries {
            writeln!(f, "{}", entry)?;
        }
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(
            f,
            "Kernels launched: {}  Total exec: {:.3}ms  Total wall: {:.3}ms",
            self.kernel_count(),
            self.total_exec_ns() as f64 / 1e6,
            self.total_wall_ns() as f64 / 1e6,
        )?;
        Ok(())
    }
}

/// Print the profiling report to the log at INFO level.
///
/// No-op when profiling is disabled.
pub fn print_report(report: &ProfilingReport) {
    if !is_profiling_enabled() {
        return;
    }
    log::info!("\n{}", report);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_timing_from_timestamps() {
        let t = KernelTiming::from_timestamps("test_kernel", 100, 200, 300, 500);
        assert_eq!(t.queue_ns, 100);
        assert_eq!(t.submit_ns, 100);
        assert_eq!(t.exec_ns, 200);
        assert_eq!(t.total_ns(), 400);
    }

    #[test]
    fn kernel_timing_gflops() {
        let t = KernelTiming::from_timestamps("matmul", 0, 0, 0, 1_000_000);
        // 1M ns = 1ms, 2M FLOPs → 2 GFLOPS
        let gflops = t.gflops(2_000_000);
        assert!((gflops - 2.0).abs() < 1e-6, "got {}", gflops);
    }

    #[test]
    fn kernel_timing_bandwidth() {
        let t = KernelTiming::from_timestamps("memcpy", 0, 0, 0, 1_000_000);
        // 1M ns = 1ms, 1 GB transferred → 1000 GB/s
        let bw = t.bandwidth_gbps(1_000_000_000);
        assert!((bw - 1000.0).abs() < 1e-3, "got {}", bw);
    }

    #[test]
    fn kernel_timing_zero_exec() {
        let t = KernelTiming::from_timestamps("empty", 0, 0, 0, 0);
        assert_eq!(t.gflops(1000), 0.0);
        assert_eq!(t.bandwidth_gbps(1000), 0.0);
    }

    #[test]
    fn profiling_report_display() {
        let mut report = ProfilingReport::new();
        report.record(KernelTiming::from_timestamps("matmul", 0, 100, 200, 500));
        report.record(KernelTiming::from_timestamps("softmax", 0, 50, 100, 300));
        let display = format!("{}", report);
        assert!(display.contains("OpenCL Profiling Report"));
        assert!(display.contains("matmul"));
        assert!(display.contains("softmax"));
        assert!(display.contains("Kernels launched: 2"));
    }

    #[test]
    fn profiling_report_totals() {
        let mut report = ProfilingReport::new();
        report.record(KernelTiming::from_timestamps("a", 0, 0, 0, 100));
        report.record(KernelTiming::from_timestamps("b", 0, 0, 0, 200));
        assert_eq!(report.total_exec_ns(), 300);
        assert_eq!(report.kernel_count(), 2);
    }

    #[test]
    fn profiling_disabled_by_default() {
        // Reset state
        PROFILING_ENABLED.store(false, Ordering::Relaxed);
        assert!(!is_profiling_enabled());
    }

    #[test]
    fn kernel_timing_display_format() {
        let t = KernelTiming::from_timestamps("test_k", 0, 1_000_000, 2_000_000, 5_000_000);
        let s = format!("{}", t);
        assert!(s.contains("test_k"));
        assert!(s.contains("ms"));
    }
}
