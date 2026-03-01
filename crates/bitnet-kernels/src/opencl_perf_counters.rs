//! Hardware performance counter module for OpenCL profiling on Intel Arc A770.
//!
//! Tracks kernel execution timing, memory bandwidth, compute utilization, and
//! occupancy metrics. All functions are CPU reference implementations that
//! operate on in-memory counters — no GPU driver calls required.

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls which performance metrics are collected.
#[derive(Debug, Clone)]
pub struct PerfCounterConfig {
    pub enable_timing: bool,
    pub enable_memory_tracking: bool,
    pub enable_occupancy: bool,
    pub sample_interval_ms: u64,
}

impl Default for PerfCounterConfig {
    fn default() -> Self {
        Self {
            enable_timing: true,
            enable_memory_tracking: true,
            enable_occupancy: false,
            sample_interval_ms: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Metric types
// ---------------------------------------------------------------------------

/// Metrics captured for a single kernel dispatch.
#[derive(Debug, Clone)]
pub struct KernelMetrics {
    pub kernel_name: String,
    pub execution_time_us: u64,
    pub global_work_size: Vec<usize>,
    pub local_work_size: Vec<usize>,
    pub memory_read_bytes: u64,
    pub memory_write_bytes: u64,
    pub estimated_gflops: f64,
}

/// Aggregate device memory statistics.
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub current_allocated: usize,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub fragmentation_pct: f32,
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            peak_allocated: 0,
            current_allocated: 0,
            allocation_count: 0,
            deallocation_count: 0,
            fragmentation_pct: 0.0,
        }
    }
}

/// Workgroup occupancy metrics.
#[derive(Debug, Clone)]
pub struct OccupancyMetrics {
    pub theoretical_occupancy: f32,
    pub achieved_occupancy: f32,
    pub active_warps: u32,
    pub max_warps: u32,
    pub registers_per_thread: u32,
    pub shared_memory_per_workgroup: u32,
}

/// A point-in-time snapshot of all counters.
#[derive(Debug, Clone)]
pub struct PerfSnapshot {
    pub timestamp_ns: u64,
    pub kernel_metrics: Vec<KernelMetrics>,
    pub memory_metrics: MemoryMetrics,
    pub gpu_utilization_pct: f32,
    pub memory_bandwidth_gbps: f64,
}

// ---------------------------------------------------------------------------
// Counter state
// ---------------------------------------------------------------------------

/// Accumulates performance data across kernel dispatches.
#[derive(Debug)]
pub struct PerfCounter {
    pub config: PerfCounterConfig,
    pub snapshots: Vec<PerfSnapshot>,
    pub running: bool,
    pub total_kernels_profiled: u64,
    /// Internal: kernel records since last snapshot.
    pending_kernels: Vec<KernelMetrics>,
    /// Internal: live memory tracking.
    memory: MemoryMetrics,
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

/// Identifies the dominant performance bottleneck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Bottleneck {
    Compute,
    Memory,
    Latency,
    Balanced,
}

impl fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bottleneck::Compute => write!(f, "Compute"),
            Bottleneck::Memory => write!(f, "Memory"),
            Bottleneck::Latency => write!(f, "Latency"),
            Bottleneck::Balanced => write!(f, "Balanced"),
        }
    }
}

/// Summary report generated from collected counters.
#[derive(Debug, Clone)]
pub struct PerfReport {
    pub duration_ms: u64,
    pub total_kernel_time_us: u64,
    pub total_memory_ops: u64,
    pub avg_gpu_utilization: f32,
    pub avg_memory_bandwidth: f64,
    pub hotspot_kernels: Vec<(String, u64)>,
    pub bottleneck: Bottleneck,
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Create a new performance counter with the given configuration.
pub fn create_perf_counter(config: PerfCounterConfig) -> PerfCounter {
    PerfCounter {
        config,
        snapshots: Vec::new(),
        running: true,
        total_kernels_profiled: 0,
        pending_kernels: Vec::new(),
        memory: MemoryMetrics::default(),
    }
}

/// Record a completed kernel execution.
pub fn cpu_record_kernel(
    counter: &mut PerfCounter,
    name: &str,
    exec_time_us: u64,
    global_size: &[usize],
    local_size: &[usize],
    mem_read: u64,
    mem_write: u64,
) {
    let total_ops = global_size.iter().product::<usize>() as u64 * 2; // fma = 2 ops
    let gflops = cpu_estimate_gflops(total_ops, exec_time_us);

    counter.pending_kernels.push(KernelMetrics {
        kernel_name: name.to_string(),
        execution_time_us: exec_time_us,
        global_work_size: global_size.to_vec(),
        local_work_size: local_size.to_vec(),
        memory_read_bytes: mem_read,
        memory_write_bytes: mem_write,
        estimated_gflops: gflops,
    });
    counter.total_kernels_profiled += 1;
}

/// Record a device memory allocation.
pub fn cpu_record_allocation(counter: &mut PerfCounter, bytes: usize) {
    counter.memory.total_allocated += bytes;
    counter.memory.current_allocated += bytes;
    counter.memory.allocation_count += 1;
    if counter.memory.current_allocated > counter.memory.peak_allocated {
        counter.memory.peak_allocated = counter.memory.current_allocated;
    }
    update_fragmentation(&mut counter.memory);
}

/// Record a device memory deallocation.
pub fn cpu_record_deallocation(counter: &mut PerfCounter, bytes: usize) {
    counter.memory.current_allocated = counter.memory.current_allocated.saturating_sub(bytes);
    counter.memory.deallocation_count += 1;
    update_fragmentation(&mut counter.memory);
}

/// Take a snapshot of current counters, draining pending kernels.
pub fn cpu_take_snapshot(counter: &mut PerfCounter) -> PerfSnapshot {
    let kernels: Vec<KernelMetrics> = counter.pending_kernels.drain(..).collect();

    let total_bytes: u64 =
        kernels.iter().map(|k| k.memory_read_bytes + k.memory_write_bytes).sum();
    let total_time_us: u64 = kernels.iter().map(|k| k.execution_time_us).sum();
    let bandwidth = cpu_estimate_bandwidth(total_bytes, total_time_us);

    let utilization = if kernels.is_empty() {
        0.0
    } else {
        // Simple heuristic: ratio of kernel time to wall time (sample interval).
        let wall_us = counter.config.sample_interval_ms * 1000;
        let pct = (total_time_us as f32 / wall_us as f32) * 100.0;
        pct.clamp(0.0, 100.0)
    };

    let snapshot = PerfSnapshot {
        timestamp_ns: now_ns(),
        kernel_metrics: kernels,
        memory_metrics: counter.memory.clone(),
        gpu_utilization_pct: utilization,
        memory_bandwidth_gbps: bandwidth,
    };

    counter.snapshots.push(snapshot.clone());
    snapshot
}

/// Compute theoretical and achieved occupancy for a workgroup configuration.
pub fn cpu_compute_occupancy(
    local_size: &[usize],
    registers: u32,
    shared_mem: u32,
    max_workgroup: usize,
) -> OccupancyMetrics {
    let workgroup_threads: usize = local_size.iter().product();
    let warp_size: usize = 32;

    let active_warps = ((workgroup_threads + warp_size - 1) / warp_size) as u32;
    let max_warps = ((max_workgroup + warp_size - 1) / warp_size) as u32;

    let theoretical = if max_warps == 0 {
        0.0
    } else {
        (active_warps as f32 / max_warps as f32).min(1.0)
    };

    // Register and shared-memory pressure reduce achieved occupancy.
    let reg_factor = if registers > 64 { 0.5 } else { 1.0_f32 };
    let smem_factor = if shared_mem > 32768 { 0.75 } else { 1.0_f32 };
    let achieved = theoretical * reg_factor * smem_factor;

    OccupancyMetrics {
        theoretical_occupancy: theoretical,
        achieved_occupancy: achieved,
        active_warps,
        max_warps,
        registers_per_thread: registers,
        shared_memory_per_workgroup: shared_mem,
    }
}

/// Estimate compute throughput in GFLOPS.
pub fn cpu_estimate_gflops(ops: u64, time_us: u64) -> f64 {
    if time_us == 0 {
        return 0.0;
    }
    ops as f64 / (time_us as f64 * 1e3) // ops / (us * 1e3) = GFLOPS
}

/// Estimate memory bandwidth in GB/s.
pub fn cpu_estimate_bandwidth(bytes: u64, time_us: u64) -> f64 {
    if time_us == 0 {
        return 0.0;
    }
    // bytes / (time_us * 1e-6) = bytes/s,  then / 1e9 = GB/s
    bytes as f64 / (time_us as f64 * 1e-6) / 1e9
}

/// Classify the dominant bottleneck from a set of kernel metrics.
pub fn cpu_identify_bottleneck(kernel_metrics: &[KernelMetrics]) -> Bottleneck {
    if kernel_metrics.is_empty() {
        return Bottleneck::Balanced;
    }

    let avg_gflops: f64 =
        kernel_metrics.iter().map(|k| k.estimated_gflops).sum::<f64>() / kernel_metrics.len() as f64;

    let total_bytes: u64 =
        kernel_metrics.iter().map(|k| k.memory_read_bytes + k.memory_write_bytes).sum();
    let total_time: u64 = kernel_metrics.iter().map(|k| k.execution_time_us).sum();
    let bandwidth = cpu_estimate_bandwidth(total_bytes, total_time);

    // Arithmetic intensity: flops per byte transferred.
    let total_flops: f64 = kernel_metrics.iter().map(|k| k.estimated_gflops).sum::<f64>()
        * (total_time as f64 * 1e3); // back to ops
    let ai = if total_bytes > 0 { total_flops / total_bytes as f64 } else { f64::MAX };

    if ai < 1.0 {
        Bottleneck::Memory
    } else if avg_gflops > bandwidth * 0.5 && ai >= 4.0 {
        Bottleneck::Compute
    } else if total_time > 0 && avg_gflops < 0.1 && bandwidth < 0.1 {
        Bottleneck::Latency
    } else {
        Bottleneck::Balanced
    }
}

/// Generate a summary report from all collected snapshots.
pub fn cpu_generate_report(counter: &PerfCounter) -> PerfReport {
    let total_kernel_time_us: u64 = counter
        .snapshots
        .iter()
        .flat_map(|s| &s.kernel_metrics)
        .map(|k| k.execution_time_us)
        .sum();

    let total_memory_ops: u64 = counter
        .snapshots
        .iter()
        .map(|s| s.memory_metrics.allocation_count + s.memory_metrics.deallocation_count)
        .sum();

    let (avg_util, avg_bw) = if counter.snapshots.is_empty() {
        (0.0, 0.0)
    } else {
        let n = counter.snapshots.len() as f32;
        let util = counter.snapshots.iter().map(|s| s.gpu_utilization_pct).sum::<f32>() / n;
        let bw =
            counter.snapshots.iter().map(|s| s.memory_bandwidth_gbps).sum::<f64>() / n as f64;
        (util, bw)
    };

    let all_kernels: Vec<&KernelMetrics> =
        counter.snapshots.iter().flat_map(|s| &s.kernel_metrics).collect();
    let hotspots = cpu_find_hotspots(counter, 5);
    let bottleneck = cpu_identify_bottleneck(
        &all_kernels.iter().map(|k| (*k).clone()).collect::<Vec<_>>(),
    );

    let duration_ms = if counter.snapshots.len() >= 2 {
        let first = counter.snapshots.first().unwrap().timestamp_ns;
        let last = counter.snapshots.last().unwrap().timestamp_ns;
        (last.saturating_sub(first)) / 1_000_000
    } else {
        0
    };

    PerfReport {
        duration_ms,
        total_kernel_time_us,
        total_memory_ops,
        avg_gpu_utilization: avg_util,
        avg_memory_bandwidth: avg_bw,
        hotspot_kernels: hotspots,
        bottleneck,
    }
}

/// Return the top-N kernels ordered by total execution time (descending).
pub fn cpu_find_hotspots(counter: &PerfCounter, top_n: usize) -> Vec<(String, u64)> {
    let mut by_name: HashMap<String, u64> = HashMap::new();
    for snap in &counter.snapshots {
        for k in &snap.kernel_metrics {
            *by_name.entry(k.kernel_name.clone()).or_default() += k.execution_time_us;
        }
    }
    // Also consider pending (un-snapshotted) kernels.
    for k in &counter.pending_kernels {
        *by_name.entry(k.kernel_name.clone()).or_default() += k.execution_time_us;
    }

    let mut sorted: Vec<(String, u64)> = by_name.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.truncate(top_n);
    sorted
}

/// Reset all counters to their initial state, preserving the config.
pub fn cpu_reset_counters(counter: &mut PerfCounter) {
    counter.snapshots.clear();
    counter.pending_kernels.clear();
    counter.total_kernels_profiled = 0;
    counter.running = true;
    counter.memory = MemoryMetrics::default();
}

/// Format a `PerfReport` into a human-readable string.
pub fn format_perf_report(report: &PerfReport) -> String {
    let mut out = String::new();
    out.push_str("=== Performance Report ===\n");
    out.push_str(&format!("Duration           : {} ms\n", report.duration_ms));
    out.push_str(&format!("Total kernel time  : {} us\n", report.total_kernel_time_us));
    out.push_str(&format!("Total memory ops   : {}\n", report.total_memory_ops));
    out.push_str(&format!("Avg GPU utilization: {:.1}%\n", report.avg_gpu_utilization));
    out.push_str(&format!("Avg mem bandwidth  : {:.2} GB/s\n", report.avg_memory_bandwidth));
    out.push_str(&format!("Bottleneck         : {}\n", report.bottleneck));
    if !report.hotspot_kernels.is_empty() {
        out.push_str("Hotspot kernels:\n");
        for (name, time) in &report.hotspot_kernels {
            out.push_str(&format!("  {name}: {time} us\n"));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64
}

fn update_fragmentation(mem: &mut MemoryMetrics) {
    if mem.peak_allocated == 0 {
        mem.fragmentation_pct = 0.0;
    } else {
        // Heuristic: gap between peak and current as a fraction of peak.
        let gap = mem.peak_allocated.saturating_sub(mem.current_allocated);
        mem.fragmentation_pct = (gap as f32 / mem.peak_allocated as f32) * 100.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PerfCounterConfig {
        PerfCounterConfig::default()
    }

    // 1. Create counter with config
    #[test]
    fn test_create_counter() {
        let cfg = default_config();
        let counter = create_perf_counter(cfg);
        assert!(counter.running);
        assert_eq!(counter.total_kernels_profiled, 0);
        assert!(counter.snapshots.is_empty());
    }

    // 2. Record kernel: metrics stored
    #[test]
    fn test_record_kernel_stores_metrics() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "matmul", 500, &[1024], &[256], 4096, 2048);
        assert_eq!(c.total_kernels_profiled, 1);
        assert_eq!(c.pending_kernels.len(), 1);
        assert_eq!(c.pending_kernels[0].kernel_name, "matmul");
        assert_eq!(c.pending_kernels[0].execution_time_us, 500);
    }

    // 3. Record allocation: memory tracking
    #[test]
    fn test_record_allocation() {
        let mut c = create_perf_counter(default_config());
        cpu_record_allocation(&mut c, 1024);
        assert_eq!(c.memory.current_allocated, 1024);
        assert_eq!(c.memory.peak_allocated, 1024);
        assert_eq!(c.memory.allocation_count, 1);
    }

    // 4. Snapshot captures state
    #[test]
    fn test_snapshot_captures_state() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "relu", 100, &[512], &[64], 512, 512);
        cpu_record_allocation(&mut c, 2048);
        let snap = cpu_take_snapshot(&mut c);
        assert_eq!(snap.kernel_metrics.len(), 1);
        assert_eq!(snap.memory_metrics.current_allocated, 2048);
        assert!(c.pending_kernels.is_empty(), "pending drained after snapshot");
    }

    // 5. Occupancy: theoretical calculation
    #[test]
    fn test_occupancy_theoretical() {
        let occ = cpu_compute_occupancy(&[256], 32, 1024, 1024);
        assert!(occ.theoretical_occupancy > 0.0);
        assert!(occ.theoretical_occupancy <= 1.0);
        assert_eq!(occ.active_warps, 8); // 256/32
        assert_eq!(occ.max_warps, 32); // 1024/32
    }

    // 6. GFLOPS estimation: correct math
    #[test]
    fn test_gflops_estimation() {
        // 1e9 ops in 1e6 us (= 1 second) => 1.0 GFLOPS
        let gflops = cpu_estimate_gflops(1_000_000_000, 1_000_000);
        assert!((gflops - 1.0).abs() < 1e-6);
    }

    // 7. Bandwidth estimation: correct units
    #[test]
    fn test_bandwidth_estimation() {
        // 1 GB in 1 second => 1.0 GB/s
        let bw = cpu_estimate_bandwidth(1_000_000_000, 1_000_000);
        assert!((bw - 1.0).abs() < 1e-6, "expected ~1.0 GB/s, got {bw}");
    }

    // 8. Bottleneck: compute-bound detection
    #[test]
    fn test_bottleneck_compute_bound() {
        let kernels = vec![KernelMetrics {
            kernel_name: "gemm".into(),
            execution_time_us: 1000,
            global_work_size: vec![4096],
            local_work_size: vec![256],
            memory_read_bytes: 64,
            memory_write_bytes: 64,
            estimated_gflops: 500.0,
        }];
        let b = cpu_identify_bottleneck(&kernels);
        assert_eq!(b, Bottleneck::Compute);
    }

    // 9. Bottleneck: memory-bound detection
    #[test]
    fn test_bottleneck_memory_bound() {
        let kernels = vec![KernelMetrics {
            kernel_name: "copy".into(),
            execution_time_us: 1000,
            global_work_size: vec![1024],
            local_work_size: vec![64],
            memory_read_bytes: 1_000_000_000,
            memory_write_bytes: 1_000_000_000,
            estimated_gflops: 0.001,
        }];
        let b = cpu_identify_bottleneck(&kernels);
        assert_eq!(b, Bottleneck::Memory);
    }

    // 10. Report generation: all fields populated
    #[test]
    fn test_report_all_fields() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "k1", 200, &[512], &[64], 1024, 512);
        cpu_take_snapshot(&mut c);
        let report = cpu_generate_report(&c);
        assert!(report.total_kernel_time_us > 0);
        assert!(!report.hotspot_kernels.is_empty());
    }

    // 11. Hotspot detection: top-N ordering
    #[test]
    fn test_hotspot_ordering() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "fast", 10, &[64], &[8], 0, 0);
        cpu_record_kernel(&mut c, "slow", 9999, &[64], &[8], 0, 0);
        cpu_record_kernel(&mut c, "medium", 500, &[64], &[8], 0, 0);
        cpu_take_snapshot(&mut c);
        let hotspots = cpu_find_hotspots(&c, 3);
        assert_eq!(hotspots[0].0, "slow");
        assert_eq!(hotspots[1].0, "medium");
        assert_eq!(hotspots[2].0, "fast");
    }

    // 12. Reset clears all metrics
    #[test]
    fn test_reset_clears_all() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "k", 100, &[64], &[8], 0, 0);
        cpu_record_allocation(&mut c, 4096);
        cpu_take_snapshot(&mut c);
        cpu_reset_counters(&mut c);
        assert_eq!(c.total_kernels_profiled, 0);
        assert!(c.snapshots.is_empty());
        assert_eq!(c.memory.current_allocated, 0);
        assert_eq!(c.memory.peak_allocated, 0);
    }

    // 13. Edge: zero execution time
    #[test]
    fn test_zero_execution_time() {
        assert_eq!(cpu_estimate_gflops(1_000_000, 0), 0.0);
        assert_eq!(cpu_estimate_bandwidth(1_000_000, 0), 0.0);
    }

    // 14. Edge: single kernel profiled
    #[test]
    fn test_single_kernel() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "only", 42, &[128], &[32], 256, 128);
        let snap = cpu_take_snapshot(&mut c);
        assert_eq!(snap.kernel_metrics.len(), 1);
        assert_eq!(snap.kernel_metrics[0].kernel_name, "only");
    }

    // 15. Multiple kernels: sorted by time
    #[test]
    fn test_multiple_kernels_sorted() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "a", 300, &[64], &[8], 0, 0);
        cpu_record_kernel(&mut c, "b", 100, &[64], &[8], 0, 0);
        cpu_record_kernel(&mut c, "c", 200, &[64], &[8], 0, 0);
        cpu_take_snapshot(&mut c);
        let hotspots = cpu_find_hotspots(&c, 10);
        assert_eq!(hotspots[0].0, "a");
        assert_eq!(hotspots[1].0, "c");
        assert_eq!(hotspots[2].0, "b");
    }

    // 16. Memory fragmentation calculation
    #[test]
    fn test_memory_fragmentation() {
        let mut c = create_perf_counter(default_config());
        cpu_record_allocation(&mut c, 1000);
        cpu_record_deallocation(&mut c, 400);
        // peak=1000, current=600, gap=400 => frag = 40%
        assert!((c.memory.fragmentation_pct - 40.0).abs() < 0.1);
    }

    // 17. Property: utilization in [0, 100]
    #[test]
    fn test_utilization_bounds() {
        let mut c = create_perf_counter(PerfCounterConfig {
            sample_interval_ms: 1,
            ..default_config()
        });
        // Kernel time far exceeds sample interval to test clamping.
        cpu_record_kernel(&mut c, "hot", 999_999, &[1024], &[256], 0, 0);
        let snap = cpu_take_snapshot(&mut c);
        assert!(snap.gpu_utilization_pct >= 0.0);
        assert!(snap.gpu_utilization_pct <= 100.0);
    }

    // 18. Property: bandwidth >= 0
    #[test]
    fn test_bandwidth_nonnegative() {
        let bw = cpu_estimate_bandwidth(0, 100);
        assert!(bw >= 0.0);
        let bw2 = cpu_estimate_bandwidth(1024, 1);
        assert!(bw2 >= 0.0);
    }

    // 19. Config defaults
    #[test]
    fn test_config_defaults() {
        let cfg = PerfCounterConfig::default();
        assert!(cfg.enable_timing);
        assert!(cfg.enable_memory_tracking);
        assert!(!cfg.enable_occupancy);
        assert_eq!(cfg.sample_interval_ms, 100);
    }

    // 20. Kernel metrics memory fields
    #[test]
    fn test_kernel_memory_fields() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "memcpy", 50, &[256], &[64], 8192, 4096);
        assert_eq!(c.pending_kernels[0].memory_read_bytes, 8192);
        assert_eq!(c.pending_kernels[0].memory_write_bytes, 4096);
    }

    // 21. Snapshot bandwidth calculation
    #[test]
    fn test_snapshot_bandwidth() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "bw", 1_000_000, &[64], &[8], 1_000_000_000, 0);
        let snap = cpu_take_snapshot(&mut c);
        assert!((snap.memory_bandwidth_gbps - 1.0).abs() < 0.01);
    }

    // 22. Multiple snapshots accumulate
    #[test]
    fn test_multiple_snapshots() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "k1", 100, &[64], &[8], 0, 0);
        cpu_take_snapshot(&mut c);
        cpu_record_kernel(&mut c, "k2", 200, &[64], &[8], 0, 0);
        cpu_take_snapshot(&mut c);
        assert_eq!(c.snapshots.len(), 2);
    }

    // 23. Deallocation saturating sub
    #[test]
    fn test_dealloc_saturating() {
        let mut c = create_perf_counter(default_config());
        cpu_record_allocation(&mut c, 100);
        cpu_record_deallocation(&mut c, 200);
        assert_eq!(c.memory.current_allocated, 0);
    }

    // 24. Peak allocation tracking
    #[test]
    fn test_peak_allocation() {
        let mut c = create_perf_counter(default_config());
        cpu_record_allocation(&mut c, 500);
        cpu_record_allocation(&mut c, 300);
        cpu_record_deallocation(&mut c, 500);
        assert_eq!(c.memory.peak_allocated, 800);
        assert_eq!(c.memory.current_allocated, 300);
    }

    // 25. Occupancy with high register pressure
    #[test]
    fn test_occupancy_high_reg_pressure() {
        let occ = cpu_compute_occupancy(&[256], 128, 1024, 1024);
        assert!(occ.achieved_occupancy < occ.theoretical_occupancy);
    }

    // 26. Occupancy with high shared memory
    #[test]
    fn test_occupancy_high_shared_mem() {
        let occ = cpu_compute_occupancy(&[256], 32, 65536, 1024);
        assert!(occ.achieved_occupancy < occ.theoretical_occupancy);
    }

    // 27. Bottleneck: latency detection
    #[test]
    fn test_bottleneck_latency() {
        let kernels = vec![KernelMetrics {
            kernel_name: "tiny".into(),
            execution_time_us: 1,
            global_work_size: vec![1],
            local_work_size: vec![1],
            memory_read_bytes: 1,
            memory_write_bytes: 1,
            estimated_gflops: 0.00001,
        }];
        let b = cpu_identify_bottleneck(&kernels);
        assert_eq!(b, Bottleneck::Latency);
    }

    // 28. Bottleneck: balanced detection
    #[test]
    fn test_bottleneck_balanced_empty() {
        let b = cpu_identify_bottleneck(&[]);
        assert_eq!(b, Bottleneck::Balanced);
    }

    // 29. Format report contains key fields
    #[test]
    fn test_format_report() {
        let report = PerfReport {
            duration_ms: 1000,
            total_kernel_time_us: 5000,
            total_memory_ops: 20,
            avg_gpu_utilization: 75.5,
            avg_memory_bandwidth: 12.34,
            hotspot_kernels: vec![("gemm".into(), 3000)],
            bottleneck: Bottleneck::Compute,
        };
        let text = format_perf_report(&report);
        assert!(text.contains("1000 ms"));
        assert!(text.contains("5000 us"));
        assert!(text.contains("75.5%"));
        assert!(text.contains("gemm"));
        assert!(text.contains("Compute"));
    }

    // 30. Hotspot top-1
    #[test]
    fn test_hotspot_top1() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "x", 100, &[64], &[8], 0, 0);
        cpu_record_kernel(&mut c, "y", 999, &[64], &[8], 0, 0);
        cpu_take_snapshot(&mut c);
        let h = cpu_find_hotspots(&c, 1);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].0, "y");
    }

    // 31. Report with no snapshots
    #[test]
    fn test_report_empty() {
        let c = create_perf_counter(default_config());
        let report = cpu_generate_report(&c);
        assert_eq!(report.total_kernel_time_us, 0);
        assert_eq!(report.avg_gpu_utilization, 0.0);
    }

    // 32. Counter running flag
    #[test]
    fn test_running_flag() {
        let mut c = create_perf_counter(default_config());
        assert!(c.running);
        c.running = false;
        cpu_reset_counters(&mut c);
        assert!(c.running);
    }

    // 33. Occupancy max_warps zero edge case
    #[test]
    fn test_occupancy_max_zero() {
        let occ = cpu_compute_occupancy(&[64], 32, 0, 0);
        assert_eq!(occ.theoretical_occupancy, 0.0);
    }

    // 34. Duplicate kernel names accumulate in hotspots
    #[test]
    fn test_duplicate_kernel_hotspots() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "gemm", 100, &[64], &[8], 0, 0);
        cpu_record_kernel(&mut c, "gemm", 200, &[64], &[8], 0, 0);
        cpu_take_snapshot(&mut c);
        let h = cpu_find_hotspots(&c, 5);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].1, 300);
    }

    // 35. Allocation / deallocation count tracking
    #[test]
    fn test_alloc_dealloc_counts() {
        let mut c = create_perf_counter(default_config());
        cpu_record_allocation(&mut c, 100);
        cpu_record_allocation(&mut c, 200);
        cpu_record_deallocation(&mut c, 50);
        assert_eq!(c.memory.allocation_count, 2);
        assert_eq!(c.memory.deallocation_count, 1);
    }

    // 36. Snapshot timestamp is non-zero
    #[test]
    fn test_snapshot_timestamp() {
        let mut c = create_perf_counter(default_config());
        let snap = cpu_take_snapshot(&mut c);
        assert!(snap.timestamp_ns > 0);
    }

    // 37. GFLOPS with large values
    #[test]
    fn test_gflops_large_values() {
        // 10^12 ops in 10^6 us = 1000 GFLOPS
        let g = cpu_estimate_gflops(1_000_000_000_000, 1_000_000);
        assert!((g - 1000.0).abs() < 1e-3);
    }

    // 38. Bottleneck display trait
    #[test]
    fn test_bottleneck_display() {
        assert_eq!(format!("{}", Bottleneck::Compute), "Compute");
        assert_eq!(format!("{}", Bottleneck::Memory), "Memory");
        assert_eq!(format!("{}", Bottleneck::Latency), "Latency");
        assert_eq!(format!("{}", Bottleneck::Balanced), "Balanced");
    }

    // 39. Memory metrics default
    #[test]
    fn test_memory_metrics_default() {
        let m = MemoryMetrics::default();
        assert_eq!(m.total_allocated, 0);
        assert_eq!(m.peak_allocated, 0);
        assert_eq!(m.fragmentation_pct, 0.0);
    }

    // 40. Global work size stored correctly
    #[test]
    fn test_global_work_size_multidim() {
        let mut c = create_perf_counter(default_config());
        cpu_record_kernel(&mut c, "conv2d", 100, &[128, 64, 3], &[16, 16, 1], 0, 0);
        assert_eq!(c.pending_kernels[0].global_work_size, vec![128, 64, 3]);
        assert_eq!(c.pending_kernels[0].local_work_size, vec![16, 16, 1]);
    }
}
