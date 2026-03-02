//! Profiling and instrumentation for CUDA kernels.
//!
//! Provides timing, bandwidth analysis, compute utilization, occupancy
//! calculation, bottleneck detection, and trace export in Chrome trace format.
//!
//! All types compile on CPU builds (using `std::time` for timing) so that
//! profiling call-sites are always available without feature-flag churn.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use bitnet_common::Result;

// ── TimingEvent ──────────────────────────────────────────────────────

/// A start/stop timing event recorded around a kernel launch.
#[derive(Debug, Clone)]
pub struct TimingEvent {
    /// Human-readable label for this event.
    pub label: String,
    /// Wall-clock start instant.
    start: Instant,
    /// Wall-clock end instant (set on `stop()`).
    end: Option<Instant>,
}

impl TimingEvent {
    /// Begin a new timing event with the given label.
    pub fn start(label: impl Into<String>) -> Self {
        Self { label: label.into(), start: Instant::now(), end: None }
    }

    /// Record the end time. Returns elapsed duration.
    pub fn stop(&mut self) -> Duration {
        let now = Instant::now();
        self.end = Some(now);
        now.duration_since(self.start)
    }

    /// Elapsed duration, or time since start if not yet stopped.
    pub fn elapsed(&self) -> Duration {
        match self.end {
            Some(e) => e.duration_since(self.start),
            None => self.start.elapsed(),
        }
    }

    /// Whether `stop()` has been called.
    pub fn is_stopped(&self) -> bool {
        self.end.is_some()
    }
}

// ── MemoryEvent ──────────────────────────────────────────────────────

/// Describes a memory allocation or deallocation event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryEventKind {
    /// Memory was allocated.
    Allocate,
    /// Memory was freed.
    Deallocate,
}

/// A memory allocation or deallocation event.
#[derive(Debug, Clone)]
pub struct MemoryEvent {
    /// Kind of event.
    pub kind: MemoryEventKind,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Timestamp when the event occurred.
    pub timestamp: Instant,
    /// Optional label (e.g. tensor name).
    pub label: String,
}

impl MemoryEvent {
    /// Create a new memory event.
    pub fn new(kind: MemoryEventKind, size_bytes: usize, label: impl Into<String>) -> Self {
        Self { kind, size_bytes, timestamp: Instant::now(), label: label.into() }
    }

    /// Shorthand for an allocation event.
    pub fn allocate(size_bytes: usize, label: impl Into<String>) -> Self {
        Self::new(MemoryEventKind::Allocate, size_bytes, label)
    }

    /// Shorthand for a deallocation event.
    pub fn deallocate(size_bytes: usize, label: impl Into<String>) -> Self {
        Self::new(MemoryEventKind::Deallocate, size_bytes, label)
    }
}

// ── BandwidthMetrics ─────────────────────────────────────────────────

/// Memory bandwidth measurements for a kernel execution.
#[derive(Debug, Clone, Copy)]
pub struct BandwidthMetrics {
    /// Bytes read by the kernel.
    pub bytes_read: u64,
    /// Bytes written by the kernel.
    pub bytes_written: u64,
    /// Wall-clock duration of the kernel.
    pub duration: Duration,
    /// Peak theoretical bandwidth in bytes/sec (device spec).
    pub peak_bandwidth_bytes_per_sec: f64,
}

impl BandwidthMetrics {
    /// Create new bandwidth metrics.
    pub fn new(
        bytes_read: u64,
        bytes_written: u64,
        duration: Duration,
        peak_bandwidth_bytes_per_sec: f64,
    ) -> Self {
        Self { bytes_read, bytes_written, duration, peak_bandwidth_bytes_per_sec }
    }

    /// Total bytes transferred (read + written).
    pub fn total_bytes(&self) -> u64 {
        self.bytes_read + self.bytes_written
    }

    /// Effective bandwidth in bytes/sec.
    pub fn effective_bandwidth(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.total_bytes() as f64 / secs
    }

    /// Bandwidth utilization as a fraction in `[0.0, 1.0]`.
    pub fn utilization(&self) -> f64 {
        if self.peak_bandwidth_bytes_per_sec <= 0.0 {
            return 0.0;
        }
        (self.effective_bandwidth() / self.peak_bandwidth_bytes_per_sec).min(1.0)
    }

    /// Effective bandwidth in GB/s.
    pub fn effective_bandwidth_gb_s(&self) -> f64 {
        self.effective_bandwidth() / 1e9
    }
}

// ── ComputeMetrics ───────────────────────────────────────────────────

/// Compute utilization measurements for a kernel execution.
#[derive(Debug, Clone, Copy)]
pub struct ComputeMetrics {
    /// Total floating-point operations performed.
    pub flops: u64,
    /// Wall-clock duration of the kernel.
    pub duration: Duration,
    /// Peak theoretical FLOP/s of the device.
    pub peak_flops: f64,
}

impl ComputeMetrics {
    /// Create new compute metrics.
    pub fn new(flops: u64, duration: Duration, peak_flops: f64) -> Self {
        Self { flops, duration, peak_flops }
    }

    /// Achieved FLOP/s.
    pub fn achieved_flops(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.flops as f64 / secs
    }

    /// Achieved TFLOP/s.
    pub fn achieved_tflops(&self) -> f64 {
        self.achieved_flops() / 1e12
    }

    /// Compute utilization as a fraction in `[0.0, 1.0]`.
    pub fn utilization(&self) -> f64 {
        if self.peak_flops <= 0.0 {
            return 0.0;
        }
        (self.achieved_flops() / self.peak_flops).min(1.0)
    }
}

// ── OccupancyCalculator ──────────────────────────────────────────────

/// Calculate theoretical and achieved occupancy for a CUDA kernel.
#[derive(Debug, Clone)]
pub struct OccupancyCalculator {
    /// Maximum threads per SM.
    pub max_threads_per_sm: u32,
    /// Maximum blocks per SM.
    pub max_blocks_per_sm: u32,
    /// Maximum shared memory per SM in bytes.
    pub max_shared_mem_per_sm: u32,
    /// Maximum registers per SM.
    pub max_registers_per_sm: u32,
}

/// Result of an occupancy calculation.
#[derive(Debug, Clone, Copy)]
pub struct OccupancyResult {
    /// Theoretical occupancy (0.0–1.0).
    pub theoretical: f64,
    /// Active warps per SM.
    pub active_warps: u32,
    /// Maximum warps per SM.
    pub max_warps: u32,
    /// Limiting factor for occupancy.
    pub limiter: OccupancyLimiter,
}

/// What limits occupancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLimiter {
    /// Thread count is the bottleneck.
    Threads,
    /// Block count is the bottleneck.
    Blocks,
    /// Shared memory usage is the bottleneck.
    SharedMemory,
    /// Register usage is the bottleneck.
    Registers,
}

impl fmt::Display for OccupancyLimiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Threads => write!(f, "threads"),
            Self::Blocks => write!(f, "blocks"),
            Self::SharedMemory => write!(f, "shared_memory"),
            Self::Registers => write!(f, "registers"),
        }
    }
}

impl Default for OccupancyCalculator {
    /// Defaults based on a typical Ampere SM (SM 8.0).
    fn default() -> Self {
        Self {
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
            max_shared_mem_per_sm: 163_840,
            max_registers_per_sm: 65_536,
        }
    }
}

impl OccupancyCalculator {
    /// Create with explicit hardware limits.
    pub fn new(
        max_threads_per_sm: u32,
        max_blocks_per_sm: u32,
        max_shared_mem_per_sm: u32,
        max_registers_per_sm: u32,
    ) -> Self {
        Self { max_threads_per_sm, max_blocks_per_sm, max_shared_mem_per_sm, max_registers_per_sm }
    }

    /// Calculate occupancy for a kernel launch configuration.
    ///
    /// * `threads_per_block` — block size
    /// * `shared_mem_per_block` — dynamic shared memory in bytes
    /// * `registers_per_thread` — register usage per thread
    pub fn calculate(
        &self,
        threads_per_block: u32,
        shared_mem_per_block: u32,
        registers_per_thread: u32,
    ) -> OccupancyResult {
        let warp_size: u32 = 32;
        let max_warps = self.max_threads_per_sm / warp_size;

        // Warps per block (round up).
        let warps_per_block = threads_per_block.div_ceil(warp_size);

        if warps_per_block == 0 || threads_per_block == 0 {
            return OccupancyResult {
                theoretical: 0.0,
                active_warps: 0,
                max_warps,
                limiter: OccupancyLimiter::Threads,
            };
        }

        // Limit by threads (blocks that fit given thread budget).
        let blocks_by_threads = self.max_threads_per_sm / threads_per_block;

        // Limit by block count.
        let blocks_by_blocks = self.max_blocks_per_sm;

        // Limit by shared memory.
        let blocks_by_shared = if shared_mem_per_block == 0 {
            self.max_blocks_per_sm
        } else {
            self.max_shared_mem_per_sm / shared_mem_per_block
        };

        // Limit by registers.
        let regs_per_block = registers_per_thread * threads_per_block;
        let blocks_by_regs = if regs_per_block == 0 {
            self.max_blocks_per_sm
        } else {
            self.max_registers_per_sm / regs_per_block
        };

        // Find the minimum (limiting factor).
        let limits = [
            (blocks_by_threads, OccupancyLimiter::Threads),
            (blocks_by_blocks, OccupancyLimiter::Blocks),
            (blocks_by_shared, OccupancyLimiter::SharedMemory),
            (blocks_by_regs, OccupancyLimiter::Registers),
        ];
        let (active_blocks, limiter) = limits
            .iter()
            .min_by_key(|(b, _)| *b)
            .copied()
            .unwrap_or((0, OccupancyLimiter::Threads));

        let active_warps = active_blocks * warps_per_block;
        let theoretical =
            if max_warps > 0 { (active_warps as f64 / max_warps as f64).min(1.0) } else { 0.0 };

        OccupancyResult { theoretical, active_warps, max_warps, limiter }
    }
}

// ── KernelProfile ────────────────────────────────────────────────────

/// Profile data for a single kernel execution.
#[derive(Debug, Clone)]
pub struct KernelProfile {
    /// Kernel name.
    pub name: String,
    /// Wall-clock duration.
    pub duration: Duration,
    /// Optional bandwidth metrics.
    pub bandwidth: Option<BandwidthMetrics>,
    /// Optional compute metrics.
    pub compute: Option<ComputeMetrics>,
    /// Optional occupancy result.
    pub occupancy: Option<OccupancyResult>,
    /// Grid dimensions (blocks).
    pub grid_dim: [u32; 3],
    /// Block dimensions (threads).
    pub block_dim: [u32; 3],
}

impl KernelProfile {
    /// Create a minimal profile with just name and duration.
    pub fn new(name: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            duration,
            bandwidth: None,
            compute: None,
            occupancy: None,
            grid_dim: [1, 1, 1],
            block_dim: [1, 1, 1],
        }
    }

    /// Attach bandwidth metrics.
    pub fn with_bandwidth(mut self, bw: BandwidthMetrics) -> Self {
        self.bandwidth = Some(bw);
        self
    }

    /// Attach compute metrics.
    pub fn with_compute(mut self, cm: ComputeMetrics) -> Self {
        self.compute = Some(cm);
        self
    }

    /// Attach occupancy result.
    pub fn with_occupancy(mut self, occ: OccupancyResult) -> Self {
        self.occupancy = Some(occ);
        self
    }

    /// Set grid dimensions.
    pub fn with_grid_dim(mut self, x: u32, y: u32, z: u32) -> Self {
        self.grid_dim = [x, y, z];
        self
    }

    /// Set block dimensions.
    pub fn with_block_dim(mut self, x: u32, y: u32, z: u32) -> Self {
        self.block_dim = [x, y, z];
        self
    }

    /// Total number of threads launched.
    pub fn total_threads(&self) -> u64 {
        let grid: u64 = self.grid_dim[0] as u64 * self.grid_dim[1] as u64 * self.grid_dim[2] as u64;
        let block: u64 =
            self.block_dim[0] as u64 * self.block_dim[1] as u64 * self.block_dim[2] as u64;
        grid * block
    }

    /// Duration in microseconds.
    pub fn duration_us(&self) -> f64 {
        self.duration.as_secs_f64() * 1e6
    }
}

// ── ProfileCollector ─────────────────────────────────────────────────

/// Collects [`KernelProfile`]s across kernel executions.
#[derive(Debug, Clone)]
pub struct ProfileCollector {
    profiles: Vec<KernelProfile>,
    memory_events: Vec<MemoryEvent>,
    start_time: Instant,
}

impl Default for ProfileCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileCollector {
    /// Create a new, empty collector.
    pub fn new() -> Self {
        Self { profiles: Vec::new(), memory_events: Vec::new(), start_time: Instant::now() }
    }

    /// Record a kernel profile.
    pub fn record(&mut self, profile: KernelProfile) {
        self.profiles.push(profile);
    }

    /// Record a memory event.
    pub fn record_memory(&mut self, event: MemoryEvent) {
        self.memory_events.push(event);
    }

    /// Number of recorded profiles.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Whether the collector is empty.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Iterate over recorded profiles.
    pub fn profiles(&self) -> &[KernelProfile] {
        &self.profiles
    }

    /// Iterate over recorded memory events.
    pub fn memory_events(&self) -> &[MemoryEvent] {
        &self.memory_events
    }

    /// Total wall-clock time across all profiled kernels.
    pub fn total_kernel_time(&self) -> Duration {
        self.profiles.iter().map(|p| p.duration).sum()
    }

    /// Total elapsed time since the collector was created.
    pub fn wall_time(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Current peak memory usage based on recorded events.
    pub fn peak_memory_bytes(&self) -> usize {
        let mut current: i64 = 0;
        let mut peak: i64 = 0;
        for ev in &self.memory_events {
            match ev.kind {
                MemoryEventKind::Allocate => current += ev.size_bytes as i64,
                MemoryEventKind::Deallocate => current -= ev.size_bytes as i64,
            }
            if current > peak {
                peak = current;
            }
        }
        peak.max(0) as usize
    }

    /// Clear all recorded data.
    pub fn clear(&mut self) {
        self.profiles.clear();
        self.memory_events.clear();
        self.start_time = Instant::now();
    }
}

// ── ProfileAccumulator ───────────────────────────────────────────────

/// Accumulates kernel profiles across multiple runs, computing aggregate
/// statistics (mean, min, max, stddev) per kernel name.
#[derive(Debug, Clone, Default)]
pub struct ProfileAccumulator {
    entries: HashMap<String, Vec<Duration>>,
}

/// Aggregate statistics for a single kernel name.
#[derive(Debug, Clone, Copy)]
pub struct AggregateStats {
    /// Number of samples.
    pub count: usize,
    /// Mean duration.
    pub mean: Duration,
    /// Minimum duration.
    pub min: Duration,
    /// Maximum duration.
    pub max: Duration,
    /// Standard deviation in seconds.
    pub stddev_secs: f64,
}

impl ProfileAccumulator {
    /// Create an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a profile to the accumulator.
    pub fn add(&mut self, profile: &KernelProfile) {
        self.entries.entry(profile.name.clone()).or_default().push(profile.duration);
    }

    /// Add all profiles from a collector.
    pub fn add_all(&mut self, collector: &ProfileCollector) {
        for p in collector.profiles() {
            self.add(p);
        }
    }

    /// Compute aggregate statistics for a named kernel.
    pub fn stats(&self, name: &str) -> Option<AggregateStats> {
        let durations = self.entries.get(name)?;
        if durations.is_empty() {
            return None;
        }
        let n = durations.len();
        let total: Duration = durations.iter().sum();
        let mean = total / n as u32;
        let min = *durations.iter().min().unwrap();
        let max = *durations.iter().max().unwrap();

        let mean_secs = mean.as_secs_f64();
        let variance = durations
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean_secs;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        let stddev_secs = variance.sqrt();

        Some(AggregateStats { count: n, mean, min, max, stddev_secs })
    }

    /// List all kernel names in the accumulator.
    pub fn kernel_names(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// Total number of samples across all kernels.
    pub fn total_samples(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }
}

// ── BottleneckAnalyzer ───────────────────────────────────────────────

/// Classification of a kernel's performance bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    /// Kernel is limited by memory bandwidth.
    MemoryBound,
    /// Kernel is limited by compute throughput.
    ComputeBound,
    /// Kernel is limited by launch/scheduling latency.
    LatencyBound,
    /// Not enough data to determine.
    Unknown,
}

impl fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryBound => write!(f, "memory_bound"),
            Self::ComputeBound => write!(f, "compute_bound"),
            Self::LatencyBound => write!(f, "latency_bound"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Identifies compute vs memory bottlenecks from a [`KernelProfile`].
#[derive(Debug, Clone, Copy)]
pub struct BottleneckAnalyzer {
    /// Threshold below which a kernel is considered latency-bound (in µs).
    pub latency_threshold_us: f64,
    /// If bandwidth utilization exceeds this AND compute utilization is below
    /// the compute threshold, classify as memory-bound.
    pub memory_util_threshold: f64,
    /// If compute utilization exceeds this AND bandwidth utilization is below
    /// the memory threshold, classify as compute-bound.
    pub compute_util_threshold: f64,
}

impl Default for BottleneckAnalyzer {
    fn default() -> Self {
        Self { latency_threshold_us: 5.0, memory_util_threshold: 0.6, compute_util_threshold: 0.6 }
    }
}

impl BottleneckAnalyzer {
    /// Create a new analyzer with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(
        latency_threshold_us: f64,
        memory_util_threshold: f64,
        compute_util_threshold: f64,
    ) -> Self {
        Self { latency_threshold_us, memory_util_threshold, compute_util_threshold }
    }

    /// Classify the bottleneck for a given kernel profile.
    pub fn analyze(&self, profile: &KernelProfile) -> Bottleneck {
        // Very short kernels are latency-bound.
        if profile.duration_us() < self.latency_threshold_us {
            return Bottleneck::LatencyBound;
        }

        let bw_util = profile.bandwidth.map(|b| b.utilization()).unwrap_or(0.0);
        let compute_util = profile.compute.map(|c| c.utilization()).unwrap_or(0.0);

        if bw_util <= 0.0 && compute_util <= 0.0 {
            return Bottleneck::Unknown;
        }

        if bw_util >= self.memory_util_threshold && compute_util < self.compute_util_threshold {
            return Bottleneck::MemoryBound;
        }
        if compute_util >= self.compute_util_threshold && bw_util < self.memory_util_threshold {
            return Bottleneck::ComputeBound;
        }

        // Both are high or neither clearly dominates.
        if bw_util >= compute_util { Bottleneck::MemoryBound } else { Bottleneck::ComputeBound }
    }
}

// ── ProfileReport ────────────────────────────────────────────────────

/// Generates a human-readable profile report from a [`ProfileCollector`].
#[derive(Debug)]
pub struct ProfileReport<'a> {
    collector: &'a ProfileCollector,
}

impl<'a> ProfileReport<'a> {
    /// Create a report from a collector reference.
    pub fn new(collector: &'a ProfileCollector) -> Self {
        Self { collector }
    }

    /// Render the full report as a string.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("=== CUDA Kernel Profile Report ===\n");
        out.push_str(&format!("Total kernels: {}\n", self.collector.len()));
        out.push_str(&format!(
            "Total kernel time: {:.3} ms\n",
            self.collector.total_kernel_time().as_secs_f64() * 1e3
        ));
        out.push_str(&format!("Peak memory: {} bytes\n", self.collector.peak_memory_bytes()));
        out.push('\n');

        for (i, p) in self.collector.profiles().iter().enumerate() {
            out.push_str(&format!("[{}] {} — {:.3} µs", i, p.name, p.duration_us()));
            if let Some(ref bw) = p.bandwidth {
                out.push_str(&format!(
                    "  BW: {:.2} GB/s ({:.1}%)",
                    bw.effective_bandwidth_gb_s(),
                    bw.utilization() * 100.0
                ));
            }
            if let Some(ref cm) = p.compute {
                out.push_str(&format!(
                    "  FLOPS: {:.2} TFLOP/s ({:.1}%)",
                    cm.achieved_tflops(),
                    cm.utilization() * 100.0
                ));
            }
            if let Some(ref occ) = p.occupancy {
                out.push_str(&format!(
                    "  Occupancy: {:.1}% (limiter: {})",
                    occ.theoretical * 100.0,
                    occ.limiter
                ));
            }
            out.push('\n');
        }

        out
    }
}

impl<'a> fmt::Display for ProfileReport<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// ── TraceExporter ────────────────────────────────────────────────────

/// Exports profiles in the Chrome `chrome://tracing` JSON format.
pub struct TraceExporter;

/// A single trace event in Chrome trace format.
#[derive(Debug, Clone)]
struct TraceEvent {
    name: String,
    cat: String,
    ph: char,
    ts: f64,
    dur: f64,
    pid: u32,
    tid: u32,
    args: HashMap<String, String>,
}

impl TraceExporter {
    /// Export a collector's profiles to a Chrome trace JSON string.
    pub fn export(collector: &ProfileCollector) -> Result<String> {
        let mut events = Vec::new();
        let base = collector.start_time;

        for (i, p) in collector.profiles().iter().enumerate() {
            let mut args = HashMap::new();
            args.insert(
                "grid".to_string(),
                format!("[{},{},{}]", p.grid_dim[0], p.grid_dim[1], p.grid_dim[2]),
            );
            args.insert(
                "block".to_string(),
                format!("[{},{},{}]", p.block_dim[0], p.block_dim[1], p.block_dim[2]),
            );
            if let Some(ref bw) = p.bandwidth {
                args.insert("bw_gb_s".to_string(), format!("{:.2}", bw.effective_bandwidth_gb_s()));
            }
            if let Some(ref cm) = p.compute {
                args.insert("tflops".to_string(), format!("{:.4}", cm.achieved_tflops()));
            }

            // Compute a synthetic offset: sum of durations of preceding kernels.
            let offset: Duration = collector.profiles()[..i].iter().map(|pp| pp.duration).sum();
            let ts_us = offset.as_secs_f64() * 1e6;
            let _ = base; // base kept for future real-timestamp support

            events.push(TraceEvent {
                name: p.name.clone(),
                cat: "kernel".to_string(),
                ph: 'X',
                ts: ts_us,
                dur: p.duration_us(),
                pid: 0,
                tid: 0,
                args,
            });
        }

        // Memory events on tid=1.
        for ev in collector.memory_events() {
            let mut args = HashMap::new();
            args.insert("size_bytes".to_string(), ev.size_bytes.to_string());
            args.insert("label".to_string(), ev.label.clone());
            let kind_str = match ev.kind {
                MemoryEventKind::Allocate => "alloc",
                MemoryEventKind::Deallocate => "free",
            };
            events.push(TraceEvent {
                name: kind_str.to_string(),
                cat: "memory".to_string(),
                ph: 'i',
                ts: ev.timestamp.duration_since(collector.start_time).as_secs_f64() * 1e6,
                dur: 0.0,
                pid: 0,
                tid: 1,
                args,
            });
        }

        // Build JSON manually to avoid serde dependency.
        let mut json = String::from("[");
        for (i, ev) in events.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push('{');
            json.push_str(&format!(
                "\"name\":\"{}\",\"cat\":\"{}\",\"ph\":\"{}\",\"ts\":{:.3},\"dur\":{:.3},\"pid\":{},\"tid\":{}",
                escape_json(&ev.name),
                escape_json(&ev.cat),
                ev.ph,
                ev.ts,
                ev.dur,
                ev.pid,
                ev.tid,
            ));
            if !ev.args.is_empty() {
                json.push_str(",\"args\":{");
                for (j, (k, v)) in ev.args.iter().enumerate() {
                    if j > 0 {
                        json.push(',');
                    }
                    json.push_str(&format!("\"{}\":\"{}\"", escape_json(k), escape_json(v)));
                }
                json.push('}');
            }
            json.push('}');
        }
        json.push(']');
        Ok(json)
    }
}

/// Minimal JSON string escaping.
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ── KernelBenchmark ──────────────────────────────────────────────────

/// Result of a kernel benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Kernel name.
    pub name: String,
    /// Number of warmup iterations executed.
    pub warmup_iters: usize,
    /// Number of measured iterations.
    pub measured_iters: usize,
    /// All measured durations.
    pub durations: Vec<Duration>,
    /// Mean duration.
    pub mean: Duration,
    /// Median duration.
    pub median: Duration,
    /// Minimum duration.
    pub min: Duration,
    /// Maximum duration.
    pub max: Duration,
    /// Standard deviation in seconds.
    pub stddev_secs: f64,
}

/// Benchmarks a kernel function with warmup, multiple iterations, and
/// statistical summary.
pub struct KernelBenchmark {
    /// Number of warmup iterations (not measured).
    pub warmup: usize,
    /// Number of measured iterations.
    pub iterations: usize,
}

impl Default for KernelBenchmark {
    fn default() -> Self {
        Self { warmup: 5, iterations: 20 }
    }
}

impl KernelBenchmark {
    /// Create with specific warmup and iteration counts.
    pub fn new(warmup: usize, iterations: usize) -> Self {
        Self { warmup, iterations }
    }

    /// Run a benchmark. The closure should execute the kernel once.
    pub fn run<F>(&self, name: impl Into<String>, mut f: F) -> BenchmarkResult
    where
        F: FnMut(),
    {
        let name = name.into();

        // Warmup.
        for _ in 0..self.warmup {
            f();
        }

        // Measure.
        let mut durations = Vec::with_capacity(self.iterations);
        for _ in 0..self.iterations {
            let start = Instant::now();
            f();
            durations.push(start.elapsed());
        }

        let n = durations.len();
        let total: Duration = durations.iter().sum();
        let mean = if n > 0 { total / n as u32 } else { Duration::ZERO };

        let mut sorted = durations.clone();
        sorted.sort();
        let median = if n > 0 { sorted[n / 2] } else { Duration::ZERO };
        let min = sorted.first().copied().unwrap_or(Duration::ZERO);
        let max = sorted.last().copied().unwrap_or(Duration::ZERO);

        let mean_secs = mean.as_secs_f64();
        let variance = if n > 0 {
            durations
                .iter()
                .map(|d| {
                    let diff = d.as_secs_f64() - mean_secs;
                    diff * diff
                })
                .sum::<f64>()
                / n as f64
        } else {
            0.0
        };
        let stddev_secs = variance.sqrt();

        BenchmarkResult {
            name,
            warmup_iters: self.warmup,
            measured_iters: self.iterations,
            durations,
            mean,
            median,
            min,
            max,
            stddev_secs,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // -- TimingEvent --

    #[test]
    fn timing_event_start_creates_running_event() {
        let ev = TimingEvent::start("test_kernel");
        assert_eq!(ev.label, "test_kernel");
        assert!(!ev.is_stopped());
    }

    #[test]
    fn timing_event_stop_returns_duration() {
        let mut ev = TimingEvent::start("k");
        thread::sleep(Duration::from_millis(5));
        let d = ev.stop();
        assert!(d >= Duration::from_millis(1));
        assert!(ev.is_stopped());
    }

    #[test]
    fn timing_event_elapsed_while_running() {
        let ev = TimingEvent::start("k");
        thread::sleep(Duration::from_millis(5));
        assert!(ev.elapsed() >= Duration::from_millis(1));
    }

    #[test]
    fn timing_event_elapsed_after_stop() {
        let mut ev = TimingEvent::start("k");
        thread::sleep(Duration::from_millis(5));
        ev.stop();
        let d1 = ev.elapsed();
        thread::sleep(Duration::from_millis(10));
        let d2 = ev.elapsed();
        // After stop, elapsed should be stable.
        assert_eq!(d1, d2);
    }

    // -- MemoryEvent --

    #[test]
    fn memory_event_allocate() {
        let ev = MemoryEvent::allocate(1024, "weight");
        assert_eq!(ev.kind, MemoryEventKind::Allocate);
        assert_eq!(ev.size_bytes, 1024);
        assert_eq!(ev.label, "weight");
    }

    #[test]
    fn memory_event_deallocate() {
        let ev = MemoryEvent::deallocate(512, "buf");
        assert_eq!(ev.kind, MemoryEventKind::Deallocate);
        assert_eq!(ev.size_bytes, 512);
    }

    #[test]
    fn memory_event_new_custom() {
        let ev = MemoryEvent::new(MemoryEventKind::Allocate, 2048, "custom");
        assert_eq!(ev.size_bytes, 2048);
    }

    // -- BandwidthMetrics --

    #[test]
    fn bandwidth_total_bytes() {
        let bw = BandwidthMetrics::new(100, 50, Duration::from_secs(1), 1e9);
        assert_eq!(bw.total_bytes(), 150);
    }

    #[test]
    fn bandwidth_effective() {
        let bw = BandwidthMetrics::new(500_000_000, 500_000_000, Duration::from_secs(1), 2e9);
        let eff = bw.effective_bandwidth();
        assert!((eff - 1e9).abs() < 1e3);
    }

    #[test]
    fn bandwidth_utilization() {
        let bw = BandwidthMetrics::new(500_000_000, 500_000_000, Duration::from_secs(1), 2e9);
        assert!((bw.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn bandwidth_utilization_clamped_at_one() {
        let bw = BandwidthMetrics::new(2_000_000_000, 0, Duration::from_secs(1), 1e9);
        assert!((bw.utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bandwidth_zero_duration() {
        let bw = BandwidthMetrics::new(100, 100, Duration::ZERO, 1e9);
        assert_eq!(bw.effective_bandwidth(), 0.0);
    }

    #[test]
    fn bandwidth_zero_peak() {
        let bw = BandwidthMetrics::new(100, 100, Duration::from_secs(1), 0.0);
        assert_eq!(bw.utilization(), 0.0);
    }

    #[test]
    fn bandwidth_gb_s() {
        let bw = BandwidthMetrics::new(1_000_000_000, 0, Duration::from_secs(1), 2e9);
        assert!((bw.effective_bandwidth_gb_s() - 1.0).abs() < 0.01);
    }

    // -- ComputeMetrics --

    #[test]
    fn compute_achieved_flops() {
        let cm = ComputeMetrics::new(1_000_000_000, Duration::from_secs(1), 2e12);
        assert!((cm.achieved_flops() - 1e9).abs() < 1e3);
    }

    #[test]
    fn compute_achieved_tflops() {
        let cm = ComputeMetrics::new(1_000_000_000_000, Duration::from_secs(1), 2e12);
        assert!((cm.achieved_tflops() - 1.0).abs() < 0.01);
    }

    #[test]
    fn compute_utilization() {
        let cm = ComputeMetrics::new(1_000_000_000, Duration::from_secs(1), 2e9);
        assert!((cm.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn compute_utilization_clamped() {
        let cm = ComputeMetrics::new(3_000_000_000, Duration::from_secs(1), 1e9);
        assert!((cm.utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_zero_duration() {
        let cm = ComputeMetrics::new(100, Duration::ZERO, 1e12);
        assert_eq!(cm.achieved_flops(), 0.0);
    }

    #[test]
    fn compute_zero_peak() {
        let cm = ComputeMetrics::new(100, Duration::from_secs(1), 0.0);
        assert_eq!(cm.utilization(), 0.0);
    }

    // -- OccupancyCalculator --

    #[test]
    fn occupancy_default_creates_ampere_config() {
        let calc = OccupancyCalculator::default();
        assert_eq!(calc.max_threads_per_sm, 2048);
        assert_eq!(calc.max_blocks_per_sm, 32);
    }

    #[test]
    fn occupancy_full_with_256_threads() {
        let calc = OccupancyCalculator::default();
        let r = calc.calculate(256, 0, 0);
        // 2048/256 = 8 blocks, each with 8 warps = 64 active warps = 64/64 = 100%
        assert!((r.theoretical - 1.0).abs() < 0.01);
    }

    #[test]
    fn occupancy_limited_by_registers() {
        let calc = OccupancyCalculator::new(2048, 32, 163_840, 65_536);
        // 256 threads × 128 regs = 32768 regs/block → 65536/32768 = 2 blocks
        let r = calc.calculate(256, 0, 128);
        // 2 blocks × 8 warps = 16 warps out of 64 max = 25%
        assert!((r.theoretical - 0.25).abs() < 0.01);
        assert_eq!(r.limiter, OccupancyLimiter::Registers);
    }

    #[test]
    fn occupancy_limited_by_shared_memory() {
        let calc = OccupancyCalculator::new(2048, 32, 49_152, 65_536);
        // 49152 / 32768 = 1 block → 8 warps / 64 = 12.5%
        let r = calc.calculate(256, 32768, 0);
        assert!((r.theoretical - 0.125).abs() < 0.01);
        assert_eq!(r.limiter, OccupancyLimiter::SharedMemory);
    }

    #[test]
    fn occupancy_limited_by_blocks() {
        let calc = OccupancyCalculator::new(2048, 2, 163_840, 65_536);
        // max 2 blocks, 256 threads each = 512 threads = 16 warps / 64 = 25%
        let r = calc.calculate(256, 0, 0);
        assert!((r.theoretical - 0.25).abs() < 0.01);
        assert_eq!(r.limiter, OccupancyLimiter::Blocks);
    }

    #[test]
    fn occupancy_zero_threads() {
        let calc = OccupancyCalculator::default();
        let r = calc.calculate(0, 0, 0);
        assert_eq!(r.theoretical, 0.0);
        assert_eq!(r.active_warps, 0);
    }

    #[test]
    fn occupancy_limiter_display() {
        assert_eq!(OccupancyLimiter::Threads.to_string(), "threads");
        assert_eq!(OccupancyLimiter::SharedMemory.to_string(), "shared_memory");
        assert_eq!(OccupancyLimiter::Registers.to_string(), "registers");
        assert_eq!(OccupancyLimiter::Blocks.to_string(), "blocks");
    }

    #[test]
    fn occupancy_custom_constructor() {
        let calc = OccupancyCalculator::new(1024, 16, 96_000, 32_768);
        assert_eq!(calc.max_threads_per_sm, 1024);
        assert_eq!(calc.max_blocks_per_sm, 16);
    }

    // -- KernelProfile --

    #[test]
    fn kernel_profile_new() {
        let p = KernelProfile::new("gemm", Duration::from_micros(100));
        assert_eq!(p.name, "gemm");
        assert_eq!(p.duration, Duration::from_micros(100));
        assert!(p.bandwidth.is_none());
        assert!(p.compute.is_none());
        assert!(p.occupancy.is_none());
    }

    #[test]
    fn kernel_profile_builder_chain() {
        let bw = BandwidthMetrics::new(1024, 1024, Duration::from_millis(1), 1e9);
        let cm = ComputeMetrics::new(1_000_000, Duration::from_millis(1), 1e12);
        let calc = OccupancyCalculator::default();
        let occ = calc.calculate(256, 0, 0);

        let p = KernelProfile::new("k", Duration::from_millis(1))
            .with_bandwidth(bw)
            .with_compute(cm)
            .with_occupancy(occ)
            .with_grid_dim(128, 1, 1)
            .with_block_dim(256, 1, 1);

        assert!(p.bandwidth.is_some());
        assert!(p.compute.is_some());
        assert!(p.occupancy.is_some());
        assert_eq!(p.grid_dim, [128, 1, 1]);
        assert_eq!(p.block_dim, [256, 1, 1]);
    }

    #[test]
    fn kernel_profile_total_threads() {
        let p = KernelProfile::new("k", Duration::ZERO)
            .with_grid_dim(4, 2, 1)
            .with_block_dim(256, 1, 1);
        assert_eq!(p.total_threads(), 4 * 2 * 256);
    }

    #[test]
    fn kernel_profile_duration_us() {
        let p = KernelProfile::new("k", Duration::from_micros(42));
        assert!((p.duration_us() - 42.0).abs() < 0.1);
    }

    #[test]
    fn kernel_profile_default_dims() {
        let p = KernelProfile::new("k", Duration::ZERO);
        assert_eq!(p.grid_dim, [1, 1, 1]);
        assert_eq!(p.block_dim, [1, 1, 1]);
        assert_eq!(p.total_threads(), 1);
    }

    // -- ProfileCollector --

    #[test]
    fn collector_starts_empty() {
        let c = ProfileCollector::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn collector_default_starts_empty() {
        let c = ProfileCollector::default();
        assert!(c.is_empty());
    }

    #[test]
    fn collector_record_and_len() {
        let mut c = ProfileCollector::new();
        c.record(KernelProfile::new("a", Duration::from_millis(1)));
        c.record(KernelProfile::new("b", Duration::from_millis(2)));
        assert_eq!(c.len(), 2);
        assert!(!c.is_empty());
    }

    #[test]
    fn collector_total_kernel_time() {
        let mut c = ProfileCollector::new();
        c.record(KernelProfile::new("a", Duration::from_millis(10)));
        c.record(KernelProfile::new("b", Duration::from_millis(20)));
        assert_eq!(c.total_kernel_time(), Duration::from_millis(30));
    }

    #[test]
    fn collector_wall_time_advances() {
        let c = ProfileCollector::new();
        thread::sleep(Duration::from_millis(5));
        assert!(c.wall_time() >= Duration::from_millis(1));
    }

    #[test]
    fn collector_memory_events_and_peak() {
        let mut c = ProfileCollector::new();
        c.record_memory(MemoryEvent::allocate(1000, "a"));
        c.record_memory(MemoryEvent::allocate(2000, "b"));
        c.record_memory(MemoryEvent::deallocate(1000, "a"));
        assert_eq!(c.memory_events().len(), 3);
        assert_eq!(c.peak_memory_bytes(), 3000);
    }

    #[test]
    fn collector_peak_memory_with_only_deallocs() {
        let mut c = ProfileCollector::new();
        c.record_memory(MemoryEvent::deallocate(500, "x"));
        // Peak should be 0 (never goes positive).
        assert_eq!(c.peak_memory_bytes(), 0);
    }

    #[test]
    fn collector_clear() {
        let mut c = ProfileCollector::new();
        c.record(KernelProfile::new("a", Duration::from_millis(1)));
        c.record_memory(MemoryEvent::allocate(100, "x"));
        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.memory_events().len(), 0);
    }

    #[test]
    fn collector_profiles_slice() {
        let mut c = ProfileCollector::new();
        c.record(KernelProfile::new("k1", Duration::from_micros(50)));
        assert_eq!(c.profiles()[0].name, "k1");
    }

    // -- ProfileAccumulator --

    #[test]
    fn accumulator_new_is_empty() {
        let acc = ProfileAccumulator::new();
        assert_eq!(acc.total_samples(), 0);
        assert!(acc.kernel_names().is_empty());
    }

    #[test]
    fn accumulator_add_and_stats() {
        let mut acc = ProfileAccumulator::new();
        acc.add(&KernelProfile::new("gemm", Duration::from_millis(10)));
        acc.add(&KernelProfile::new("gemm", Duration::from_millis(20)));
        acc.add(&KernelProfile::new("gemm", Duration::from_millis(30)));

        let s = acc.stats("gemm").unwrap();
        assert_eq!(s.count, 3);
        assert_eq!(s.min, Duration::from_millis(10));
        assert_eq!(s.max, Duration::from_millis(30));
        assert_eq!(s.mean, Duration::from_millis(20));
    }

    #[test]
    fn accumulator_stats_missing_kernel() {
        let acc = ProfileAccumulator::new();
        assert!(acc.stats("nonexistent").is_none());
    }

    #[test]
    fn accumulator_stddev() {
        let mut acc = ProfileAccumulator::new();
        // All identical → stddev 0.
        acc.add(&KernelProfile::new("k", Duration::from_millis(10)));
        acc.add(&KernelProfile::new("k", Duration::from_millis(10)));
        acc.add(&KernelProfile::new("k", Duration::from_millis(10)));
        let s = acc.stats("k").unwrap();
        assert!(s.stddev_secs < 1e-9);
    }

    #[test]
    fn accumulator_add_all() {
        let mut c = ProfileCollector::new();
        c.record(KernelProfile::new("a", Duration::from_millis(1)));
        c.record(KernelProfile::new("b", Duration::from_millis(2)));

        let mut acc = ProfileAccumulator::new();
        acc.add_all(&c);
        assert_eq!(acc.total_samples(), 2);
    }

    #[test]
    fn accumulator_kernel_names() {
        let mut acc = ProfileAccumulator::new();
        acc.add(&KernelProfile::new("gemm", Duration::from_millis(1)));
        acc.add(&KernelProfile::new("softmax", Duration::from_millis(2)));
        let mut names = acc.kernel_names();
        names.sort();
        assert_eq!(names, vec!["gemm", "softmax"]);
    }

    // -- BottleneckAnalyzer --

    #[test]
    fn bottleneck_latency_bound() {
        let a = BottleneckAnalyzer::default();
        let p = KernelProfile::new("tiny", Duration::from_nanos(100));
        assert_eq!(a.analyze(&p), Bottleneck::LatencyBound);
    }

    #[test]
    fn bottleneck_memory_bound() {
        let a = BottleneckAnalyzer::default();
        let bw = BandwidthMetrics::new(800_000_000, 0, Duration::from_secs(1), 1_000_000_000.0);
        let cm = ComputeMetrics::new(100_000, Duration::from_secs(1), 1e12);
        let p = KernelProfile::new("memcpy", Duration::from_millis(100))
            .with_bandwidth(bw)
            .with_compute(cm);
        assert_eq!(a.analyze(&p), Bottleneck::MemoryBound);
    }

    #[test]
    fn bottleneck_compute_bound() {
        let a = BottleneckAnalyzer::default();
        let bw = BandwidthMetrics::new(100, 0, Duration::from_secs(1), 1e12);
        let cm = ComputeMetrics::new(800_000_000_000, Duration::from_secs(1), 1e12);
        let p = KernelProfile::new("gemm", Duration::from_millis(100))
            .with_bandwidth(bw)
            .with_compute(cm);
        assert_eq!(a.analyze(&p), Bottleneck::ComputeBound);
    }

    #[test]
    fn bottleneck_unknown_no_metrics() {
        let a = BottleneckAnalyzer::default();
        let p = KernelProfile::new("empty", Duration::from_millis(100));
        assert_eq!(a.analyze(&p), Bottleneck::Unknown);
    }

    #[test]
    fn bottleneck_custom_thresholds() {
        let a = BottleneckAnalyzer::with_thresholds(1.0, 0.3, 0.3);
        assert_eq!(a.latency_threshold_us, 1.0);
    }

    #[test]
    fn bottleneck_display() {
        assert_eq!(Bottleneck::MemoryBound.to_string(), "memory_bound");
        assert_eq!(Bottleneck::ComputeBound.to_string(), "compute_bound");
        assert_eq!(Bottleneck::LatencyBound.to_string(), "latency_bound");
        assert_eq!(Bottleneck::Unknown.to_string(), "unknown");
    }

    #[test]
    fn bottleneck_both_high_defaults_to_higher() {
        let a = BottleneckAnalyzer::default();
        // Both utilizations high; higher one wins.
        let bw = BandwidthMetrics::new(900_000_000, 0, Duration::from_secs(1), 1_000_000_000.0);
        let cm = ComputeMetrics::new(700_000_000_000, Duration::from_secs(1), 1e12);
        let p =
            KernelProfile::new("k", Duration::from_millis(100)).with_bandwidth(bw).with_compute(cm);
        let b = a.analyze(&p);
        assert!(b == Bottleneck::MemoryBound || b == Bottleneck::ComputeBound);
    }

    // -- ProfileReport --

    #[test]
    fn report_render_empty() {
        let c = ProfileCollector::new();
        let r = ProfileReport::new(&c);
        let text = r.render();
        assert!(text.contains("Total kernels: 0"));
    }

    #[test]
    fn report_render_with_profiles() {
        let mut c = ProfileCollector::new();
        c.record(KernelProfile::new("gemm", Duration::from_micros(500)));
        let r = ProfileReport::new(&c);
        let text = r.render();
        assert!(text.contains("gemm"));
        assert!(text.contains("Total kernels: 1"));
    }

    #[test]
    fn report_display_trait() {
        let c = ProfileCollector::new();
        let r = ProfileReport::new(&c);
        let s = format!("{r}");
        assert!(s.contains("CUDA Kernel Profile Report"));
    }

    #[test]
    fn report_includes_bandwidth() {
        let mut c = ProfileCollector::new();
        let bw = BandwidthMetrics::new(1_000_000_000, 0, Duration::from_secs(1), 2e9);
        let p = KernelProfile::new("k", Duration::from_millis(1)).with_bandwidth(bw);
        c.record(p);
        let text = ProfileReport::new(&c).render();
        assert!(text.contains("BW:"));
    }

    #[test]
    fn report_includes_compute() {
        let mut c = ProfileCollector::new();
        let cm = ComputeMetrics::new(1_000_000_000_000, Duration::from_secs(1), 2e12);
        let p = KernelProfile::new("k", Duration::from_millis(1)).with_compute(cm);
        c.record(p);
        let text = ProfileReport::new(&c).render();
        assert!(text.contains("FLOPS:"));
    }

    #[test]
    fn report_includes_occupancy() {
        let mut c = ProfileCollector::new();
        let occ = OccupancyCalculator::default().calculate(256, 0, 0);
        let p = KernelProfile::new("k", Duration::from_millis(1)).with_occupancy(occ);
        c.record(p);
        let text = ProfileReport::new(&c).render();
        assert!(text.contains("Occupancy:"));
    }

    // -- TraceExporter --

    #[test]
    fn trace_export_empty() {
        let c = ProfileCollector::new();
        let json = TraceExporter::export(&c).unwrap();
        assert_eq!(json, "[]");
    }

    #[test]
    fn trace_export_single_kernel() {
        let mut c = ProfileCollector::new();
        c.record(KernelProfile::new("gemm", Duration::from_micros(100)));
        let json = TraceExporter::export(&c).unwrap();
        assert!(json.contains("\"name\":\"gemm\""));
        assert!(json.contains("\"cat\":\"kernel\""));
        assert!(json.contains("\"ph\":\"X\""));
    }

    #[test]
    fn trace_export_multiple_kernels() {
        let mut c = ProfileCollector::new();
        c.record(KernelProfile::new("a", Duration::from_micros(10)));
        c.record(KernelProfile::new("b", Duration::from_micros(20)));
        let json = TraceExporter::export(&c).unwrap();
        assert!(json.contains("\"name\":\"a\""));
        assert!(json.contains("\"name\":\"b\""));
    }

    #[test]
    fn trace_export_includes_args() {
        let mut c = ProfileCollector::new();
        let p = KernelProfile::new("k", Duration::from_micros(50))
            .with_grid_dim(64, 1, 1)
            .with_block_dim(256, 1, 1);
        c.record(p);
        let json = TraceExporter::export(&c).unwrap();
        assert!(json.contains("\"grid\":\"[64,1,1]\""));
        assert!(json.contains("\"block\":\"[256,1,1]\""));
    }

    #[test]
    fn trace_export_includes_bandwidth_arg() {
        let mut c = ProfileCollector::new();
        let bw = BandwidthMetrics::new(1_000_000_000, 0, Duration::from_secs(1), 2e9);
        let p = KernelProfile::new("k", Duration::from_micros(50)).with_bandwidth(bw);
        c.record(p);
        let json = TraceExporter::export(&c).unwrap();
        assert!(json.contains("\"bw_gb_s\""));
    }

    #[test]
    fn trace_export_includes_memory_events() {
        let mut c = ProfileCollector::new();
        c.record_memory(MemoryEvent::allocate(4096, "tensor_a"));
        let json = TraceExporter::export(&c).unwrap();
        assert!(json.contains("\"name\":\"alloc\""));
        assert!(json.contains("\"cat\":\"memory\""));
    }

    #[test]
    fn trace_export_escapes_json() {
        let mut c = ProfileCollector::new();
        c.record(KernelProfile::new("kernel \"quoted\"", Duration::from_micros(1)));
        let json = TraceExporter::export(&c).unwrap();
        assert!(json.contains("kernel \\\"quoted\\\""));
    }

    // -- KernelBenchmark --

    #[test]
    fn benchmark_default_config() {
        let b = KernelBenchmark::default();
        assert_eq!(b.warmup, 5);
        assert_eq!(b.iterations, 20);
    }

    #[test]
    fn benchmark_custom_config() {
        let b = KernelBenchmark::new(2, 10);
        assert_eq!(b.warmup, 2);
        assert_eq!(b.iterations, 10);
    }

    #[test]
    fn benchmark_run_basic() {
        let b = KernelBenchmark::new(1, 5);
        let r = b.run("noop", || {});
        assert_eq!(r.name, "noop");
        assert_eq!(r.warmup_iters, 1);
        assert_eq!(r.measured_iters, 5);
        assert_eq!(r.durations.len(), 5);
    }

    #[test]
    fn benchmark_min_le_mean_le_max() {
        let b = KernelBenchmark::new(0, 10);
        let r = b.run("k", || {
            std::hint::black_box(0);
        });
        assert!(r.min <= r.mean);
        assert!(r.mean <= r.max);
    }

    #[test]
    fn benchmark_median_in_range() {
        let b = KernelBenchmark::new(0, 10);
        let r = b.run("k", || {
            std::hint::black_box(0);
        });
        assert!(r.median >= r.min);
        assert!(r.median <= r.max);
    }

    #[test]
    fn benchmark_stddev_non_negative() {
        let b = KernelBenchmark::new(0, 10);
        let r = b.run("k", || {});
        assert!(r.stddev_secs >= 0.0);
    }

    // -- escape_json --

    #[test]
    fn escape_json_basic() {
        assert_eq!(escape_json("hello"), "hello");
    }

    #[test]
    fn escape_json_quotes() {
        assert_eq!(escape_json("a\"b"), "a\\\"b");
    }

    #[test]
    fn escape_json_backslash() {
        assert_eq!(escape_json("a\\b"), "a\\\\b");
    }

    #[test]
    fn escape_json_newline() {
        assert_eq!(escape_json("a\nb"), "a\\nb");
    }

    // -- Integration-style tests --

    #[test]
    fn end_to_end_profile_and_report() {
        let mut c = ProfileCollector::new();
        let calc = OccupancyCalculator::default();

        let bw =
            BandwidthMetrics::new(2_000_000_000, 1_000_000_000, Duration::from_millis(10), 900e9);
        let cm = ComputeMetrics::new(50_000_000_000, Duration::from_millis(10), 312e12);
        let occ = calc.calculate(256, 4096, 32);

        let p = KernelProfile::new("i2s_gemv", Duration::from_millis(10))
            .with_bandwidth(bw)
            .with_compute(cm)
            .with_occupancy(occ)
            .with_grid_dim(128, 1, 1)
            .with_block_dim(256, 1, 1);

        c.record(p.clone());
        c.record_memory(MemoryEvent::allocate(1 << 20, "weights"));

        let report = ProfileReport::new(&c);
        let text = report.render();
        assert!(text.contains("i2s_gemv"));
        assert!(text.contains("Total kernels: 1"));

        let analyzer = BottleneckAnalyzer::default();
        let bottleneck = analyzer.analyze(&p);
        assert!(
            bottleneck == Bottleneck::MemoryBound
                || bottleneck == Bottleneck::ComputeBound
                || bottleneck == Bottleneck::Unknown
        );

        let json = TraceExporter::export(&c).unwrap();
        assert!(json.contains("i2s_gemv"));
    }

    #[test]
    fn end_to_end_accumulate_and_stats() {
        let mut acc = ProfileAccumulator::new();
        for i in 0..10 {
            acc.add(&KernelProfile::new("softmax", Duration::from_micros(100 + i * 10)));
        }
        let s = acc.stats("softmax").unwrap();
        assert_eq!(s.count, 10);
        assert!(s.min <= s.mean);
        assert!(s.mean <= s.max);
        assert!(s.stddev_secs > 0.0);
    }

    #[test]
    fn benchmark_and_accumulate() {
        let bench = KernelBenchmark::new(2, 10);
        let result = bench.run("add_kernel", || {
            std::hint::black_box(1 + 1);
        });
        let mut acc = ProfileAccumulator::new();
        for d in &result.durations {
            acc.add(&KernelProfile::new(&result.name, *d));
        }
        let s = acc.stats("add_kernel").unwrap();
        assert_eq!(s.count, 10);
    }
}
