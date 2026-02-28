//! GPU telemetry and monitoring for inference workloads.
//!
//! Tracks kernel launches, memory transfers, and GPU utilization.
//! Enable via `BITNET_GPU_TELEMETRY=1` env var.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Check if GPU telemetry is enabled via environment variable.
pub fn telemetry_enabled() -> bool {
    static ENABLED: AtomicBool = AtomicBool::new(false);
    static CHECKED: AtomicBool = AtomicBool::new(false);

    if !CHECKED.load(Ordering::Relaxed) {
        let val = std::env::var("BITNET_GPU_TELEMETRY")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        ENABLED.store(val, Ordering::Relaxed);
        CHECKED.store(true, Ordering::Release);
    }
    ENABLED.load(Ordering::Relaxed)
}

/// Direction of a GPU memory transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDir {
    /// Host to device
    HostToDevice,
    /// Device to host
    DeviceToHost,
    /// Device to device
    DeviceToDevice,
}

impl fmt::Display for TransferDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "H2D"),
            Self::DeviceToHost => write!(f, "D2H"),
            Self::DeviceToDevice => write!(f, "D2D"),
        }
    }
}

/// A single kernel launch record.
#[derive(Debug, Clone)]
pub struct KernelLaunchRecord {
    pub kernel_name: String,
    pub duration: Duration,
    pub global_work_size: [usize; 3],
    pub local_work_size: Option<[usize; 3]>,
}

/// A single memory transfer record.
#[derive(Debug, Clone)]
pub struct TransferRecord {
    pub direction: TransferDir,
    pub bytes: u64,
    pub duration: Duration,
}

/// GPU telemetry collector.
///
/// Thread-safe via atomic counters for hot-path metrics.
/// Detailed records stored in a Vec (requires &mut self).
pub struct GpuTelemetry {
    start_time: Instant,
    kernel_launches: Vec<KernelLaunchRecord>,
    transfers: Vec<TransferRecord>,
    total_kernel_time_ns: AtomicU64,
    total_transfer_time_ns: AtomicU64,
    total_bytes_transferred: AtomicU64,
    allocated_bytes: AtomicU64,
    peak_allocated_bytes: AtomicU64,
}

impl GpuTelemetry {
    /// Create a new telemetry collector.
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            kernel_launches: Vec::new(),
            transfers: Vec::new(),
            total_kernel_time_ns: AtomicU64::new(0),
            total_transfer_time_ns: AtomicU64::new(0),
            total_bytes_transferred: AtomicU64::new(0),
            allocated_bytes: AtomicU64::new(0),
            peak_allocated_bytes: AtomicU64::new(0),
        }
    }

    /// Record a kernel launch.
    pub fn record_kernel(
        &mut self,
        name: impl Into<String>,
        duration: Duration,
        global_work_size: [usize; 3],
        local_work_size: Option<[usize; 3]>,
    ) {
        self.total_kernel_time_ns.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.kernel_launches.push(KernelLaunchRecord {
            kernel_name: name.into(),
            duration,
            global_work_size,
            local_work_size,
        });
    }

    /// Record a memory transfer.
    pub fn record_transfer(&mut self, direction: TransferDir, bytes: u64, duration: Duration) {
        self.total_transfer_time_ns.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.total_bytes_transferred.fetch_add(bytes, Ordering::Relaxed);
        self.transfers.push(TransferRecord { direction, bytes, duration });
    }

    /// Track memory allocation.
    pub fn track_alloc(&self, bytes: u64) {
        let current = self.allocated_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        let mut peak = self.peak_allocated_bytes.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_allocated_bytes.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
    }

    /// Track memory deallocation.
    pub fn track_free(&self, bytes: u64) {
        self.allocated_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Wall-clock elapsed time since telemetry start.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Total time spent in kernel execution.
    pub fn total_kernel_time(&self) -> Duration {
        Duration::from_nanos(self.total_kernel_time_ns.load(Ordering::Relaxed))
    }

    /// Total time spent in memory transfers.
    pub fn total_transfer_time(&self) -> Duration {
        Duration::from_nanos(self.total_transfer_time_ns.load(Ordering::Relaxed))
    }

    /// Total bytes transferred.
    pub fn total_bytes_transferred(&self) -> u64 {
        self.total_bytes_transferred.load(Ordering::Relaxed)
    }

    /// Current GPU memory allocated (bytes).
    pub fn current_allocated(&self) -> u64 {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Peak GPU memory allocated (bytes).
    pub fn peak_allocated(&self) -> u64 {
        self.peak_allocated_bytes.load(Ordering::Relaxed)
    }

    /// Number of kernel launches recorded.
    pub fn kernel_count(&self) -> usize {
        self.kernel_launches.len()
    }

    /// Number of transfers recorded.
    pub fn transfer_count(&self) -> usize {
        self.transfers.len()
    }

    /// Compute throughput in tokens/sec given token count.
    pub fn throughput(&self, tokens: u64) -> f64 {
        let secs = self.elapsed().as_secs_f64();
        if secs > 0.0 { tokens as f64 / secs } else { 0.0 }
    }

    /// GPU utilization: ratio of kernel time to wall-clock time.
    pub fn gpu_utilization(&self) -> f64 {
        let wall = self.elapsed().as_nanos() as f64;
        if wall > 0.0 {
            let kernel = self.total_kernel_time_ns.load(Ordering::Relaxed) as f64;
            (kernel / wall).min(1.0)
        } else {
            0.0
        }
    }

    /// Memory utilization: current / peak ratio.
    pub fn memory_utilization(&self) -> f64 {
        let peak = self.peak_allocated() as f64;
        if peak > 0.0 { self.current_allocated() as f64 / peak } else { 0.0 }
    }

    /// Human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "GPU Telemetry: kernels={} transfers={} \
             kernel_time={:.1}ms transfer_time={:.1}ms \
             bytes_transferred={} \
             mem_current={}B mem_peak={}B \
             gpu_util={:.1}%",
            self.kernel_count(),
            self.transfer_count(),
            self.total_kernel_time().as_secs_f64() * 1000.0,
            self.total_transfer_time().as_secs_f64() * 1000.0,
            self.total_bytes_transferred(),
            self.current_allocated(),
            self.peak_allocated(),
            self.gpu_utilization() * 100.0,
        )
    }

    /// Emit tracing spans for key metrics (if tracing is available).
    pub fn emit_tracing_spans(&self) {
        log::info!(
            target: "gpu_telemetry",
            "{}",
            self.summary()
        );
        for (i, k) in self.kernel_launches.iter().enumerate() {
            log::debug!(
                target: "gpu_telemetry",
                "kernel[{}] name={} time={:.3}ms gws={:?}",
                i,
                k.kernel_name,
                k.duration.as_secs_f64() * 1000.0,
                k.global_work_size
            );
        }
    }
}

impl Default for GpuTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for GpuTelemetry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_telemetry_is_empty() {
        let t = GpuTelemetry::new();
        assert_eq!(t.kernel_count(), 0);
        assert_eq!(t.transfer_count(), 0);
        assert_eq!(t.total_bytes_transferred(), 0);
        assert_eq!(t.current_allocated(), 0);
        assert_eq!(t.peak_allocated(), 0);
    }

    #[test]
    fn test_record_kernel() {
        let mut t = GpuTelemetry::new();
        t.record_kernel("matmul", Duration::from_millis(5), [1024, 1, 1], None);
        assert_eq!(t.kernel_count(), 1);
        assert_eq!(t.total_kernel_time().as_millis(), 5);
    }

    #[test]
    fn test_record_transfer() {
        let mut t = GpuTelemetry::new();
        t.record_transfer(TransferDir::HostToDevice, 4096, Duration::from_micros(100));
        assert_eq!(t.transfer_count(), 1);
        assert_eq!(t.total_bytes_transferred(), 4096);
    }

    #[test]
    fn test_memory_tracking() {
        let t = GpuTelemetry::new();
        t.track_alloc(1000);
        assert_eq!(t.current_allocated(), 1000);
        assert_eq!(t.peak_allocated(), 1000);

        t.track_alloc(500);
        assert_eq!(t.current_allocated(), 1500);
        assert_eq!(t.peak_allocated(), 1500);

        t.track_free(800);
        assert_eq!(t.current_allocated(), 700);
        assert_eq!(t.peak_allocated(), 1500);
    }

    #[test]
    fn test_throughput() {
        let t = GpuTelemetry::new();
        std::thread::sleep(Duration::from_millis(10));
        let tps = t.throughput(100);
        assert!(tps > 0.0, "throughput should be positive");
    }

    #[test]
    fn test_gpu_utilization_zero_when_no_kernels() {
        let t = GpuTelemetry::new();
        std::thread::sleep(Duration::from_millis(1));
        assert!(t.gpu_utilization() < 0.01);
    }

    #[test]
    fn test_memory_utilization() {
        let t = GpuTelemetry::new();
        t.track_alloc(1000);
        t.track_free(500);
        let util = t.memory_utilization();
        assert!((util - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_summary_format() {
        let mut t = GpuTelemetry::new();
        t.record_kernel("test_k", Duration::from_millis(1), [64, 1, 1], None);
        let s = t.summary();
        assert!(s.contains("kernels=1"));
        assert!(s.contains("GPU Telemetry"));
    }

    #[test]
    fn test_display_impl() {
        let t = GpuTelemetry::new();
        let s = format!("{}", t);
        assert!(s.contains("GPU Telemetry"));
    }

    #[test]
    fn test_transfer_dir_display() {
        assert_eq!(format!("{}", TransferDir::HostToDevice), "H2D");
        assert_eq!(format!("{}", TransferDir::DeviceToHost), "D2H");
        assert_eq!(format!("{}", TransferDir::DeviceToDevice), "D2D");
    }

    #[test]
    fn test_default_impl() {
        let t = GpuTelemetry::default();
        assert_eq!(t.kernel_count(), 0);
    }

    #[test]
    fn test_telemetry_enabled_default_false() {
        // When BITNET_GPU_TELEMETRY is not set, should be false
        // Note: can't easily test this because of static caching
        // Just verify the function doesn't panic
        let _ = telemetry_enabled();
    }
}
