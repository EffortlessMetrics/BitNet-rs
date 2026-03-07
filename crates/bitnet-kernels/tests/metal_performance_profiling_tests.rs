#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
#![cfg(feature = "cpu")]

//! Metal GPU performance profiling and tuning tests.
//!
//! CPU-side validation of profiling data structures, bandwidth estimation,
//! occupancy calculations, dispatch tuning, memory access analysis, kernel
//! fusion heuristics, power efficiency modelling, latency hiding strategies,
//! batch sizing, Apple Silicon specifics, and regression detection.
//!
//! No GPU runtime required — all tests exercise pure Rust logic.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

const SIMD_WIDTH: u32 = 32;
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;
const MAX_THREADGROUP_MEMORY: usize = 32 * 1024; // 32 KiB
const BUFFER_ALIGNMENT: usize = 256;

// ═══════════════════════════════════════════════════════════════════════════
// GPU Counter types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
enum GpuCounterKind {
    Timestamp,
    VertexInvocations,
    FragmentInvocations,
    ComputeInvocations,
    ClipperInvocations,
    TotalCycles,
    MemoryReadBytes,
    MemoryWriteBytes,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GpuCounterSet {
    name: String,
    counters: Vec<GpuCounterKind>,
}

impl GpuCounterSet {
    fn common() -> Self {
        Self {
            name: "MTLCommonCounterSet".to_string(),
            counters: vec![
                GpuCounterKind::Timestamp,
                GpuCounterKind::ComputeInvocations,
                GpuCounterKind::TotalCycles,
            ],
        }
    }

    fn memory() -> Self {
        Self {
            name: "MTLMemoryCounterSet".to_string(),
            counters: vec![GpuCounterKind::MemoryReadBytes, GpuCounterKind::MemoryWriteBytes],
        }
    }

    fn extended() -> Self {
        Self {
            name: "MTLExtendedCounterSet".to_string(),
            counters: vec![
                GpuCounterKind::Timestamp,
                GpuCounterKind::ComputeInvocations,
                GpuCounterKind::TotalCycles,
                GpuCounterKind::MemoryReadBytes,
                GpuCounterKind::MemoryWriteBytes,
                GpuCounterKind::VertexInvocations,
                GpuCounterKind::FragmentInvocations,
                GpuCounterKind::ClipperInvocations,
            ],
        }
    }

    fn supports(&self, kind: GpuCounterKind) -> bool {
        self.counters.contains(&kind)
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CounterSample {
    kind: GpuCounterKind,
    value: u64,
    timestamp_ns: u64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CounterSampleBuffer {
    samples: Vec<CounterSample>,
    capacity: usize,
}

impl CounterSampleBuffer {
    fn new(capacity: usize) -> Self {
        Self { samples: Vec::new(), capacity }
    }

    fn record(&mut self, kind: GpuCounterKind, value: u64, ts: u64) -> bool {
        if self.samples.len() >= self.capacity {
            return false; // overflow
        }
        self.samples.push(CounterSample { kind, value, timestamp_ns: ts });
        true
    }

    fn query(&self, kind: GpuCounterKind) -> Vec<&CounterSample> {
        self.samples.iter().filter(|s| s.kind == kind).collect()
    }

    fn is_full(&self) -> bool {
        self.samples.len() >= self.capacity
    }

    fn elapsed_ns(&self) -> Option<u64> {
        let ts: Vec<_> = self.query(GpuCounterKind::Timestamp);
        if ts.len() >= 2 {
            Some(ts.last().unwrap().timestamp_ns - ts.first().unwrap().timestamp_ns)
        } else {
            None
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bandwidth estimation
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BandwidthProfile {
    theoretical_peak_gbps: f64,
    measured_gbps: f64,
    bytes_transferred: u64,
    duration_ns: u64,
}

impl BandwidthProfile {
    fn utilization(&self) -> f64 {
        if self.theoretical_peak_gbps <= 0.0 {
            return 0.0;
        }
        (self.measured_gbps / self.theoretical_peak_gbps).min(1.0)
    }

    fn from_transfer(bytes: u64, duration_ns: u64, peak_gbps: f64) -> Self {
        let measured = if duration_ns > 0 {
            (bytes as f64) / (duration_ns as f64) // bytes/ns = GB/s
        } else {
            0.0
        };
        Self {
            theoretical_peak_gbps: peak_gbps,
            measured_gbps: measured,
            bytes_transferred: bytes,
            duration_ns,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum Bottleneck {
    ComputeBound,
    MemoryBound,
    LatencyBound,
    Balanced,
}

fn classify_bottleneck(compute_util: f64, memory_util: f64) -> Bottleneck {
    let threshold = 0.15;
    // Both utilizations low → latency bound (neither unit is busy)
    if compute_util < 0.3 && memory_util < 0.3 {
        Bottleneck::LatencyBound
    } else if (compute_util - memory_util).abs() < threshold {
        Bottleneck::Balanced
    } else if compute_util > memory_util + threshold {
        Bottleneck::MemoryBound
    } else {
        Bottleneck::ComputeBound
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Occupancy
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct OccupancyInfo {
    threads_per_threadgroup: u32,
    max_concurrent_threadgroups: u32,
    registers_per_thread: u32,
    shared_memory_per_threadgroup: u32,
    achieved_occupancy: f64,
}

fn calculate_occupancy(
    threads_per_tg: u32,
    regs_per_thread: u32,
    shared_mem: u32,
    max_regs: u32,
    max_shared: u32,
    max_tgs_per_core: u32,
) -> f64 {
    if threads_per_tg == 0 || threads_per_tg > MAX_THREADS_PER_THREADGROUP {
        return 0.0;
    }
    let reg_limited = if regs_per_thread > 0 {
        max_regs / (regs_per_thread * threads_per_tg)
    } else {
        max_tgs_per_core
    };
    let shared_limited = if shared_mem > 0 { max_shared / shared_mem } else { max_tgs_per_core };
    let actual_tgs = reg_limited.min(shared_limited).min(max_tgs_per_core);
    let active_threads = actual_tgs * threads_per_tg;
    let max_threads = max_tgs_per_core * MAX_THREADS_PER_THREADGROUP;
    (active_threads as f64 / max_threads as f64).min(1.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline statistics
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PipelineStats {
    alu_instructions: u64,
    memory_instructions: u64,
    control_instructions: u64,
    total_cycles: u64,
    simd_width: u32,
}

impl PipelineStats {
    fn total_instructions(&self) -> u64 {
        self.alu_instructions + self.memory_instructions + self.control_instructions
    }

    fn alu_ratio(&self) -> f64 {
        let total = self.total_instructions();
        if total == 0 {
            return 0.0;
        }
        self.alu_instructions as f64 / total as f64
    }

    fn memory_ratio(&self) -> f64 {
        let total = self.total_instructions();
        if total == 0 {
            return 0.0;
        }
        self.memory_instructions as f64 / total as f64
    }

    fn ipc(&self) -> f64 {
        if self.total_cycles == 0 {
            return 0.0;
        }
        (self.total_instructions() as f64 * self.simd_width as f64) / self.total_cycles as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dispatch tuning
// ═══════════════════════════════════════════════════════════════════════════

fn ceil_div(a: u32, b: u32) -> u32 {
    assert_ne!(b, 0);
    a.div_ceil(b)
}

fn optimal_threadgroup_1d(total: u32) -> u32 {
    if total == 0 {
        return 0;
    }
    let mut best = SIMD_WIDTH;
    let mut tg = SIMD_WIDTH;
    while tg <= MAX_THREADS_PER_THREADGROUP && tg <= total {
        if tg.is_multiple_of(SIMD_WIDTH) {
            best = tg;
        }
        tg += SIMD_WIDTH;
    }
    best
}

fn optimal_threadgroup_2d(width: u32, height: u32) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (0, 0);
    }
    let mut best = (SIMD_WIDTH, 1u32);
    let mut best_waste = u32::MAX;
    for w in (1..=32).filter(|w| w % 4 == 0 || *w == 1) {
        for h in (1..=32).filter(|h| h % 4 == 0 || *h == 1) {
            if w * h > MAX_THREADS_PER_THREADGROUP || w * h < SIMD_WIDTH {
                continue;
            }
            if (w * h) % SIMD_WIDTH != 0 {
                continue;
            }
            let groups_x = ceil_div(width, w);
            let groups_y = ceil_div(height, h);
            let total_threads = groups_x * w * groups_y * h;
            let waste = total_threads - width * height;
            if waste < best_waste {
                best_waste = waste;
                best = (w, h);
            }
        }
    }
    best
}

fn optimal_threadgroup_3d(x: u32, y: u32, z: u32) -> (u32, u32, u32) {
    if x == 0 || y == 0 || z == 0 {
        return (0, 0, 0);
    }
    let tx = x.clamp(1, 8);
    let ty = y.clamp(1, 8);
    let tz = z.clamp(1, 4);
    let product = tx * ty * tz;
    if product > MAX_THREADS_PER_THREADGROUP {
        return (SIMD_WIDTH, 1, 1);
    }
    (tx, ty, tz)
}

// ═══════════════════════════════════════════════════════════════════════════
// Memory access patterns
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum AccessPattern {
    Coalesced,
    Strided,
    Random,
    Broadcast,
}

fn classify_access(stride_bytes: usize, element_size: usize) -> AccessPattern {
    if stride_bytes == element_size {
        AccessPattern::Coalesced
    } else if stride_bytes == 0 {
        AccessPattern::Broadcast
    } else if stride_bytes <= element_size * 4 {
        AccessPattern::Strided
    } else {
        AccessPattern::Random
    }
}

fn estimate_bank_conflicts(threads: u32, stride_words: u32, banks: u32) -> u32 {
    if banks == 0 || stride_words == 0 {
        return 0;
    }
    let mut hit = vec![0u32; banks as usize];
    for t in 0..threads.min(SIMD_WIDTH) {
        let bank = (t * stride_words) % banks;
        hit[bank as usize] += 1;
    }
    let max_hit = hit.iter().max().copied().unwrap_or(1);
    max_hit.saturating_sub(1)
}

fn estimate_cache_hit_ratio(working_set_bytes: usize, cache_size_bytes: usize) -> f64 {
    if cache_size_bytes == 0 {
        return 0.0;
    }
    if working_set_bytes <= cache_size_bytes {
        0.95 // fits in cache
    } else {
        let ratio = cache_size_bytes as f64 / working_set_bytes as f64;
        ratio * 0.9
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel fusion
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum KernelOp {
    MatMul,
    BiasAdd,
    Activation,
    LayerNorm,
    Softmax,
    Quantize,
    Dequantize,
    Reduce,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct FusionCandidate {
    ops: Vec<KernelOp>,
    shared_memory_saved_bytes: usize,
    launches_saved: u32,
    estimated_speedup: f64,
}

fn detect_fusion_opportunities(ops: &[KernelOp]) -> Vec<FusionCandidate> {
    let mut candidates = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        // MatMul + BiasAdd + Activation fusion (check 3-op before 2-op)
        if i + 2 < ops.len()
            && ops[i] == KernelOp::MatMul
            && ops[i + 1] == KernelOp::BiasAdd
            && ops[i + 2] == KernelOp::Activation
        {
            candidates.push(FusionCandidate {
                ops: vec![KernelOp::MatMul, KernelOp::BiasAdd, KernelOp::Activation],
                shared_memory_saved_bytes: 8192,
                launches_saved: 2,
                estimated_speedup: 1.30,
            });
            i += 3;
            continue;
        }
        // MatMul + BiasAdd fusion
        if i + 1 < ops.len() && ops[i] == KernelOp::MatMul && ops[i + 1] == KernelOp::BiasAdd {
            candidates.push(FusionCandidate {
                ops: vec![KernelOp::MatMul, KernelOp::BiasAdd],
                shared_memory_saved_bytes: 4096,
                launches_saved: 1,
                estimated_speedup: 1.15,
            });
            i += 2;
            continue;
        }
        // Quantize + Dequantize elision
        if i + 1 < ops.len() && ops[i] == KernelOp::Quantize && ops[i + 1] == KernelOp::Dequantize {
            candidates.push(FusionCandidate {
                ops: vec![KernelOp::Quantize, KernelOp::Dequantize],
                shared_memory_saved_bytes: 2048,
                launches_saved: 2, // both eliminated
                estimated_speedup: 1.50,
            });
            i += 2;
            continue;
        }
        // LayerNorm + Activation
        if i + 1 < ops.len() && ops[i] == KernelOp::LayerNorm && ops[i + 1] == KernelOp::Activation
        {
            candidates.push(FusionCandidate {
                ops: vec![KernelOp::LayerNorm, KernelOp::Activation],
                shared_memory_saved_bytes: 4096,
                launches_saved: 1,
                estimated_speedup: 1.20,
            });
            i += 2;
            continue;
        }
        i += 1;
    }
    candidates
}

// ═══════════════════════════════════════════════════════════════════════════
// Power efficiency
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PowerProfile {
    tdp_watts: f64,
    measured_watts: f64,
    gpu_temp_celsius: f64,
    throttle_threshold_celsius: f64,
    ops_per_second: f64,
}

impl PowerProfile {
    fn is_throttled(&self) -> bool {
        self.gpu_temp_celsius >= self.throttle_threshold_celsius
    }

    fn energy_per_op_nj(&self) -> f64 {
        if self.ops_per_second <= 0.0 {
            return f64::INFINITY;
        }
        (self.measured_watts * 1e9) / self.ops_per_second
    }

    fn power_headroom(&self) -> f64 {
        if self.tdp_watts <= 0.0 {
            return 0.0;
        }
        ((self.tdp_watts - self.measured_watts) / self.tdp_watts).max(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum PerformanceMode {
    Burst,
    Sustained,
}

fn sustained_throughput(burst_throughput: f64, thermal_margin: f64) -> f64 {
    burst_throughput * thermal_margin.clamp(0.0, 1.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Latency hiding
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PipelineStage {
    name: String,
    gpu_duration_us: f64,
    cpu_duration_us: f64,
}

fn pipeline_throughput(stages: &[PipelineStage], num_buffers: usize) -> f64 {
    if stages.is_empty() || num_buffers == 0 {
        return 0.0;
    }
    let max_stage =
        stages.iter().map(|s| s.gpu_duration_us.max(s.cpu_duration_us)).fold(0.0f64, f64::max);
    if max_stage <= 0.0 {
        return 0.0;
    }
    // With N buffers, we can overlap N-1 stages
    let overlap_factor = (num_buffers as f64).min(stages.len() as f64);
    overlap_factor / max_stage
}

fn gpu_cpu_overlap_ratio(gpu_us: f64, cpu_us: f64) -> f64 {
    if gpu_us <= 0.0 && cpu_us <= 0.0 {
        return 0.0;
    }
    let parallel = gpu_us.min(cpu_us);
    let total = gpu_us.max(cpu_us);
    if total <= 0.0 { 0.0 } else { parallel / total }
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch sizing
// ═══════════════════════════════════════════════════════════════════════════

fn optimal_batch_size(
    element_bytes: usize,
    available_memory: usize,
    gpu_cores: u32,
    overhead_per_launch_us: f64,
    compute_per_element_us: f64,
) -> usize {
    if element_bytes == 0 || compute_per_element_us <= 0.0 {
        return 0;
    }
    let max_by_memory = available_memory / element_bytes;
    // Amortise launch overhead: batch_size * compute >= 10 * overhead
    let min_for_amortisation =
        ((10.0 * overhead_per_launch_us) / compute_per_element_us).ceil() as usize;
    // Saturate GPU cores
    let min_for_saturation = (gpu_cores as usize) * (SIMD_WIDTH as usize);
    let ideal = min_for_amortisation.max(min_for_saturation);
    ideal.clamp(1, max_by_memory)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
enum ComputeRegime {
    MemoryBound,
    ComputeBound,
    LaunchOverheadBound,
}

fn classify_regime(
    batch_size: usize,
    flops_per_element: f64,
    bytes_per_element: f64,
    machine_flops_per_byte: f64,
) -> ComputeRegime {
    if batch_size < 64 {
        return ComputeRegime::LaunchOverheadBound;
    }
    let arithmetic_intensity = flops_per_element / bytes_per_element;
    if arithmetic_intensity > machine_flops_per_byte {
        ComputeRegime::ComputeBound
    } else {
        ComputeRegime::MemoryBound
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Apple Silicon specifics
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AppleSiliconProfile {
    chip_name: String,
    gpu_cores: u32,
    memory_bandwidth_gbps: f64,
    unified_memory_gb: u32,
    gpu_family: u32,
    max_threads_per_threadgroup: u32,
    simd_width: u32,
}

impl AppleSiliconProfile {
    fn m1() -> Self {
        Self {
            chip_name: "Apple M1".into(),
            gpu_cores: 8,
            memory_bandwidth_gbps: 68.25,
            unified_memory_gb: 16,
            gpu_family: 7,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
        }
    }
    fn m2() -> Self {
        Self {
            chip_name: "Apple M2".into(),
            gpu_cores: 10,
            memory_bandwidth_gbps: 100.0,
            unified_memory_gb: 24,
            gpu_family: 8,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
        }
    }
    fn m3() -> Self {
        Self {
            chip_name: "Apple M3".into(),
            gpu_cores: 10,
            memory_bandwidth_gbps: 100.0,
            unified_memory_gb: 36,
            gpu_family: 9,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
        }
    }
    fn m3_max() -> Self {
        Self {
            chip_name: "Apple M3 Max".into(),
            gpu_cores: 40,
            memory_bandwidth_gbps: 400.0,
            unified_memory_gb: 128,
            gpu_family: 9,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
        }
    }

    fn peak_gflops_f32(&self) -> f64 {
        // Each GPU core has 128 ALUs, dual-issue FMA = 2 FLOP each at ~1 GHz
        (self.gpu_cores as f64) * 128.0 * 2.0 * 1.0
    }

    fn total_gpu_threads(&self) -> u32 {
        self.gpu_cores * self.max_threads_per_threadgroup
    }

    fn flops_per_byte(&self) -> f64 {
        if self.memory_bandwidth_gbps <= 0.0 {
            return 0.0;
        }
        self.peak_gflops_f32() / self.memory_bandwidth_gbps
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression detection
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PerfBaseline {
    name: String,
    throughput: f64,
    latency_us: f64,
}

fn check_regression(
    baseline: &PerfBaseline,
    measured_throughput: f64,
    measured_latency_us: f64,
    degradation_threshold: f64,
) -> (bool, String) {
    let tp_ratio =
        if baseline.throughput > 0.0 { measured_throughput / baseline.throughput } else { 1.0 };
    let lat_ratio = if measured_latency_us > 0.0 && baseline.latency_us > 0.0 {
        measured_latency_us / baseline.latency_us
    } else {
        1.0
    };
    if tp_ratio < (1.0 - degradation_threshold) {
        (
            true,
            format!(
                "{}: throughput regression {:.1}% (baseline={:.1}, measured={:.1})",
                baseline.name,
                (1.0 - tp_ratio) * 100.0,
                baseline.throughput,
                measured_throughput,
            ),
        )
    } else if lat_ratio > (1.0 + degradation_threshold) {
        (
            true,
            format!(
                "{}: latency regression {:.1}% (baseline={:.1}µs, measured={:.1}µs)",
                baseline.name,
                (lat_ratio - 1.0) * 100.0,
                baseline.latency_us,
                measured_latency_us,
            ),
        )
    } else {
        (false, format!("{}: OK", baseline.name))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── GPU Counter tests ──────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn counter_set_common_discovery() {
    let cs = GpuCounterSet::common();
    assert!(cs.supports(GpuCounterKind::Timestamp));
    assert!(cs.supports(GpuCounterKind::ComputeInvocations));
    assert!(cs.supports(GpuCounterKind::TotalCycles));
    assert!(!cs.supports(GpuCounterKind::MemoryReadBytes));
}

#[test]
fn counter_set_memory_discovery() {
    let cs = GpuCounterSet::memory();
    assert!(cs.supports(GpuCounterKind::MemoryReadBytes));
    assert!(cs.supports(GpuCounterKind::MemoryWriteBytes));
    assert!(!cs.supports(GpuCounterKind::Timestamp));
}

#[test]
fn counter_set_extended_has_all() {
    let cs = GpuCounterSet::extended();
    assert_eq!(cs.counters.len(), 8);
    assert!(cs.supports(GpuCounterKind::Timestamp));
    assert!(cs.supports(GpuCounterKind::MemoryReadBytes));
    assert!(cs.supports(GpuCounterKind::FragmentInvocations));
}

#[test]
fn counter_sample_timing_basic() {
    let mut buf = CounterSampleBuffer::new(64);
    assert!(buf.record(GpuCounterKind::Timestamp, 0, 1000));
    assert!(buf.record(GpuCounterKind::ComputeInvocations, 256, 1050));
    assert!(buf.record(GpuCounterKind::Timestamp, 0, 2000));
    assert_eq!(buf.elapsed_ns(), Some(1000));
}

#[test]
fn counter_sample_buffer_overflow() {
    let mut buf = CounterSampleBuffer::new(2);
    assert!(buf.record(GpuCounterKind::Timestamp, 0, 100));
    assert!(buf.record(GpuCounterKind::Timestamp, 0, 200));
    assert!(!buf.record(GpuCounterKind::Timestamp, 0, 300));
    assert!(buf.is_full());
}

#[test]
fn counter_query_filters_by_kind() {
    let mut buf = CounterSampleBuffer::new(16);
    buf.record(GpuCounterKind::Timestamp, 0, 100);
    buf.record(GpuCounterKind::ComputeInvocations, 512, 100);
    buf.record(GpuCounterKind::MemoryReadBytes, 4096, 100);
    buf.record(GpuCounterKind::ComputeInvocations, 1024, 200);
    let invocations = buf.query(GpuCounterKind::ComputeInvocations);
    assert_eq!(invocations.len(), 2);
    assert_eq!(invocations[0].value, 512);
    assert_eq!(invocations[1].value, 1024);
}

#[test]
fn counter_multi_pass_accumulation() {
    let mut buf = CounterSampleBuffer::new(128);
    let passes = 4;
    for pass in 0..passes {
        let base_ts = pass as u64 * 1000;
        buf.record(GpuCounterKind::Timestamp, 0, base_ts);
        buf.record(GpuCounterKind::TotalCycles, 500 * (pass as u64 + 1), base_ts + 500);
        buf.record(GpuCounterKind::Timestamp, 0, base_ts + 1000);
    }
    let cycles: u64 = buf.query(GpuCounterKind::TotalCycles).iter().map(|s| s.value).sum();
    assert_eq!(cycles, 500 + 1000 + 1500 + 2000);
}

#[test]
fn counter_elapsed_single_timestamp_returns_none() {
    let mut buf = CounterSampleBuffer::new(8);
    buf.record(GpuCounterKind::Timestamp, 0, 1000);
    assert_eq!(buf.elapsed_ns(), None);
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Bandwidth Estimation tests ─────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bandwidth_theoretical_peak_m1() {
    let m1 = AppleSiliconProfile::m1();
    assert!((m1.memory_bandwidth_gbps - 68.25).abs() < 0.01);
}

#[test]
fn bandwidth_effective_utilization_full() {
    let bp = BandwidthProfile::from_transfer(68_250_000_000, 1_000_000_000, 68.25);
    let util = bp.utilization();
    assert!((util - 1.0).abs() < 0.01, "Full BW should be ~100%: got {util}");
}

#[test]
fn bandwidth_effective_utilization_half() {
    let bp = BandwidthProfile::from_transfer(34_125_000_000, 1_000_000_000, 68.25);
    let util = bp.utilization();
    assert!((util - 0.5).abs() < 0.01, "Half BW: got {util}");
}

#[test]
fn bandwidth_zero_duration_gives_zero() {
    let bp = BandwidthProfile::from_transfer(1024, 0, 68.25);
    assert_eq!(bp.utilization(), 0.0);
}

#[test]
fn bandwidth_bottleneck_memory_bound() {
    let b = classify_bottleneck(0.9, 0.3);
    assert_eq!(b, Bottleneck::MemoryBound);
}

#[test]
fn bandwidth_bottleneck_compute_bound() {
    let b = classify_bottleneck(0.3, 0.9);
    assert_eq!(b, Bottleneck::ComputeBound);
}

#[test]
fn bandwidth_bottleneck_balanced() {
    let b = classify_bottleneck(0.55, 0.50);
    assert_eq!(b, Bottleneck::Balanced);
}

#[test]
fn bandwidth_bottleneck_latency_bound() {
    let b = classify_bottleneck(0.1, 0.1);
    assert_eq!(b, Bottleneck::LatencyBound);
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Occupancy tests ────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn occupancy_full_with_minimal_resources() {
    let occ = calculate_occupancy(1024, 16, 0, 16384, 32768, 4);
    assert!(occ > 0.0, "Should have positive occupancy");
}

#[test]
fn occupancy_zero_threads() {
    let occ = calculate_occupancy(0, 16, 0, 16384, 32768, 4);
    assert_eq!(occ, 0.0);
}

#[test]
fn occupancy_exceeds_max_threads() {
    let occ = calculate_occupancy(2048, 16, 0, 16384, 32768, 4);
    assert_eq!(occ, 0.0);
}

#[test]
fn occupancy_register_pressure_reduces() {
    let occ_low = calculate_occupancy(256, 16, 0, 16384, 32768, 4);
    let occ_high = calculate_occupancy(256, 64, 0, 16384, 32768, 4);
    assert!(
        occ_high <= occ_low,
        "Higher register pressure should not increase occupancy: low={occ_low} high={occ_high}"
    );
}

#[test]
fn occupancy_shared_memory_limits() {
    let occ_small = calculate_occupancy(256, 16, 4096, 16384, 32768, 4);
    let occ_large = calculate_occupancy(256, 16, 32768, 16384, 32768, 4);
    assert!(
        occ_large <= occ_small,
        "More shared mem should not increase occupancy: small={occ_small} large={occ_large}"
    );
}

#[test]
fn occupancy_simd_aligned_threadgroup() {
    let occ = calculate_occupancy(SIMD_WIDTH, 16, 0, 16384, 32768, 4);
    assert!(occ > 0.0);
    assert!(SIMD_WIDTH.is_multiple_of(32), "SIMD width should be 32");
}

#[test]
fn occupancy_multiple_threadgroups_per_core() {
    let occ1 = calculate_occupancy(256, 8, 0, 16384, 32768, 1);
    let occ4 = calculate_occupancy(256, 8, 0, 16384, 32768, 4);
    assert!(occ4 >= occ1, "More TGs per core should not reduce occupancy");
}

#[test]
fn occupancy_combined_pressure() {
    // High register + high shared memory → low occupancy
    let occ = calculate_occupancy(256, 128, 16384, 16384, 32768, 4);
    assert!(occ <= 0.5, "Combined pressure should limit occupancy: {occ}");
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Pipeline Statistics tests ──────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pipeline_instruction_count() {
    let ps = PipelineStats {
        alu_instructions: 1000,
        memory_instructions: 500,
        control_instructions: 100,
        total_cycles: 800,
        simd_width: 32,
    };
    assert_eq!(ps.total_instructions(), 1600);
}

#[test]
fn pipeline_alu_ratio_compute_heavy() {
    let ps = PipelineStats {
        alu_instructions: 900,
        memory_instructions: 100,
        control_instructions: 0,
        total_cycles: 500,
        simd_width: 32,
    };
    assert!((ps.alu_ratio() - 0.9).abs() < 0.001);
}

#[test]
fn pipeline_memory_ratio_memory_heavy() {
    let ps = PipelineStats {
        alu_instructions: 100,
        memory_instructions: 900,
        control_instructions: 0,
        total_cycles: 500,
        simd_width: 32,
    };
    assert!((ps.memory_ratio() - 0.9).abs() < 0.001);
}

#[test]
fn pipeline_ipc_calculation() {
    let ps = PipelineStats {
        alu_instructions: 100,
        memory_instructions: 100,
        control_instructions: 0,
        total_cycles: 100,
        simd_width: 32,
    };
    let ipc = ps.ipc();
    // (200 * 32) / 100 = 64.0
    assert!((ipc - 64.0).abs() < 0.001);
}

#[test]
fn pipeline_zero_instructions() {
    let ps = PipelineStats {
        alu_instructions: 0,
        memory_instructions: 0,
        control_instructions: 0,
        total_cycles: 100,
        simd_width: 32,
    };
    assert_eq!(ps.alu_ratio(), 0.0);
    assert_eq!(ps.memory_ratio(), 0.0);
    assert_eq!(ps.ipc(), 0.0);
}

#[test]
fn pipeline_zero_cycles() {
    let ps = PipelineStats {
        alu_instructions: 100,
        memory_instructions: 100,
        control_instructions: 0,
        total_cycles: 0,
        simd_width: 32,
    };
    assert_eq!(ps.ipc(), 0.0);
}

#[test]
fn pipeline_balanced_workload() {
    let ps = PipelineStats {
        alu_instructions: 500,
        memory_instructions: 500,
        control_instructions: 0,
        total_cycles: 500,
        simd_width: 32,
    };
    assert!((ps.alu_ratio() - 0.5).abs() < 0.001);
    assert!((ps.memory_ratio() - 0.5).abs() < 0.001);
}

#[test]
fn pipeline_with_control_flow() {
    let ps = PipelineStats {
        alu_instructions: 300,
        memory_instructions: 200,
        control_instructions: 500,
        total_cycles: 1000,
        simd_width: 32,
    };
    assert_eq!(ps.total_instructions(), 1000);
    assert!((ps.alu_ratio() - 0.3).abs() < 0.001);
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Dispatch Tuning tests ──────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dispatch_optimal_1d_small() {
    let tg = optimal_threadgroup_1d(16);
    assert!(tg >= SIMD_WIDTH || tg == 16.min(SIMD_WIDTH));
}

#[test]
fn dispatch_optimal_1d_large() {
    let tg = optimal_threadgroup_1d(100_000);
    assert!(tg <= MAX_THREADS_PER_THREADGROUP);
    assert!(tg.is_multiple_of(SIMD_WIDTH));
}

#[test]
fn dispatch_optimal_1d_zero() {
    assert_eq!(optimal_threadgroup_1d(0), 0);
}

#[test]
fn dispatch_optimal_2d_square() {
    let (w, h) = optimal_threadgroup_2d(256, 256);
    assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
    assert!((w * h) % SIMD_WIDTH == 0);
}

#[test]
fn dispatch_optimal_2d_rectangular() {
    let (w, h) = optimal_threadgroup_2d(1024, 64);
    assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
    assert!((w * h) % SIMD_WIDTH == 0);
}

#[test]
fn dispatch_optimal_2d_zero() {
    assert_eq!(optimal_threadgroup_2d(0, 256), (0, 0));
    assert_eq!(optimal_threadgroup_2d(256, 0), (0, 0));
}

#[test]
fn dispatch_optimal_3d_cube() {
    let (tx, ty, tz) = optimal_threadgroup_3d(64, 64, 64);
    assert!(tx * ty * tz <= MAX_THREADS_PER_THREADGROUP);
    assert!(tx >= 1 && ty >= 1 && tz >= 1);
}

#[test]
fn dispatch_ceil_div_exact() {
    assert_eq!(ceil_div(256, 32), 8);
    assert_eq!(ceil_div(1024, 256), 4);
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Memory Access tests ────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn memory_coalesced_access() {
    assert_eq!(classify_access(4, 4), AccessPattern::Coalesced);
}

#[test]
fn memory_strided_access() {
    assert_eq!(classify_access(8, 4), AccessPattern::Strided);
}

#[test]
fn memory_random_access() {
    assert_eq!(classify_access(1024, 4), AccessPattern::Random);
}

#[test]
fn memory_broadcast_access() {
    assert_eq!(classify_access(0, 4), AccessPattern::Broadcast);
}

#[test]
fn memory_bank_conflict_none() {
    // Stride of 1 word, 32 banks → each thread hits a different bank
    let conflicts = estimate_bank_conflicts(32, 1, 32);
    assert_eq!(conflicts, 0);
}

#[test]
fn memory_bank_conflict_stride_two() {
    // Stride of 2 words, 32 banks → 2-way conflict
    let conflicts = estimate_bank_conflicts(32, 2, 32);
    assert!(conflicts > 0, "Stride-2 should cause bank conflicts");
}

#[test]
fn memory_cache_hit_ratio_fits() {
    let ratio = estimate_cache_hit_ratio(16_384, 32_768);
    assert!(ratio > 0.90, "Working set fits in cache: {ratio}");
}

#[test]
fn memory_cache_hit_ratio_spills() {
    let ratio = estimate_cache_hit_ratio(1_000_000, 32_768);
    assert!(ratio < 0.10, "Working set vastly exceeds cache: {ratio}");
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Kernel Fusion tests ────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fusion_matmul_bias() {
    let ops = vec![KernelOp::MatMul, KernelOp::BiasAdd];
    let candidates = detect_fusion_opportunities(&ops);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].launches_saved, 1);
}

#[test]
fn fusion_matmul_bias_activation() {
    let ops = vec![KernelOp::MatMul, KernelOp::BiasAdd, KernelOp::Activation];
    let candidates = detect_fusion_opportunities(&ops);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].ops.len(), 3);
    assert!(candidates[0].estimated_speedup >= 1.20);
}

#[test]
fn fusion_quantize_dequantize_elision() {
    let ops = vec![KernelOp::Quantize, KernelOp::Dequantize];
    let candidates = detect_fusion_opportunities(&ops);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].launches_saved, 2);
}

#[test]
fn fusion_layernorm_activation() {
    let ops = vec![KernelOp::LayerNorm, KernelOp::Activation];
    let candidates = detect_fusion_opportunities(&ops);
    assert_eq!(candidates.len(), 1);
    assert!(candidates[0].estimated_speedup >= 1.10);
}

#[test]
fn fusion_shared_memory_savings() {
    let ops = vec![KernelOp::MatMul, KernelOp::BiasAdd, KernelOp::Activation];
    let candidates = detect_fusion_opportunities(&ops);
    assert!(candidates[0].shared_memory_saved_bytes > 0);
}

#[test]
fn fusion_no_opportunity_standalone() {
    let ops = vec![KernelOp::Softmax];
    let candidates = detect_fusion_opportunities(&ops);
    assert!(candidates.is_empty());
}

#[test]
fn fusion_multiple_in_sequence() {
    let ops = vec![KernelOp::MatMul, KernelOp::BiasAdd, KernelOp::LayerNorm, KernelOp::Activation];
    let candidates = detect_fusion_opportunities(&ops);
    assert_eq!(candidates.len(), 2, "Should detect MatMul+Bias and LayerNorm+Activation");
}

#[test]
fn fusion_launch_overhead_reduction() {
    let ops = vec![KernelOp::Quantize, KernelOp::Dequantize, KernelOp::MatMul, KernelOp::BiasAdd];
    let candidates = detect_fusion_opportunities(&ops);
    let total_launches_saved: u32 = candidates.iter().map(|c| c.launches_saved).sum();
    assert!(total_launches_saved >= 3, "Should save ≥3 launches: got {total_launches_saved}");
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Power Efficiency tests ─────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn power_thermal_throttle_detection() {
    let profile = PowerProfile {
        tdp_watts: 30.0,
        measured_watts: 28.0,
        gpu_temp_celsius: 105.0,
        throttle_threshold_celsius: 100.0,
        ops_per_second: 1e9,
    };
    assert!(profile.is_throttled());
}

#[test]
fn power_no_throttle_below_threshold() {
    let profile = PowerProfile {
        tdp_watts: 30.0,
        measured_watts: 20.0,
        gpu_temp_celsius: 70.0,
        throttle_threshold_celsius: 100.0,
        ops_per_second: 1e9,
    };
    assert!(!profile.is_throttled());
}

#[test]
fn power_energy_per_op() {
    let profile = PowerProfile {
        tdp_watts: 30.0,
        measured_watts: 15.0,
        gpu_temp_celsius: 60.0,
        throttle_threshold_celsius: 100.0,
        ops_per_second: 1e9,
    };
    let nj = profile.energy_per_op_nj();
    assert!((nj - 15.0).abs() < 0.01, "15W / 1 Gops = 15 nJ/op: got {nj}");
}

#[test]
fn power_energy_zero_ops() {
    let profile = PowerProfile {
        tdp_watts: 30.0,
        measured_watts: 15.0,
        gpu_temp_celsius: 60.0,
        throttle_threshold_celsius: 100.0,
        ops_per_second: 0.0,
    };
    assert!(profile.energy_per_op_nj().is_infinite());
}

#[test]
fn power_headroom_calculation() {
    let profile = PowerProfile {
        tdp_watts: 30.0,
        measured_watts: 20.0,
        gpu_temp_celsius: 60.0,
        throttle_threshold_celsius: 100.0,
        ops_per_second: 1e9,
    };
    let headroom = profile.power_headroom();
    assert!((headroom - 1.0 / 3.0).abs() < 0.01, "33% headroom: got {headroom}");
}

#[test]
fn power_burst_vs_sustained_throughput() {
    let burst = 100.0;
    let sustained = sustained_throughput(burst, 0.75);
    assert!((sustained - 75.0).abs() < 0.01);
}

#[test]
fn power_sustained_zero_margin() {
    let sustained = sustained_throughput(100.0, 0.0);
    assert_eq!(sustained, 0.0);
}

#[test]
fn power_sustained_full_margin() {
    let sustained = sustained_throughput(100.0, 1.0);
    assert!((sustained - 100.0).abs() < 0.01);
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Latency Hiding tests ──────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn latency_command_buffer_pipelining_double() {
    let stages = vec![PipelineStage {
        name: "inference".into(),
        gpu_duration_us: 1000.0,
        cpu_duration_us: 500.0,
    }];
    let tp_single = pipeline_throughput(&stages, 1);
    let tp_double = pipeline_throughput(&stages, 2);
    assert!(tp_double >= tp_single, "Double buffering should not reduce throughput");
}

#[test]
fn latency_command_buffer_pipelining_triple() {
    let stages = vec![
        PipelineStage { name: "encode".into(), gpu_duration_us: 500.0, cpu_duration_us: 200.0 },
        PipelineStage { name: "compute".into(), gpu_duration_us: 1000.0, cpu_duration_us: 100.0 },
        PipelineStage { name: "readback".into(), gpu_duration_us: 300.0, cpu_duration_us: 400.0 },
    ];
    let tp = pipeline_throughput(&stages, 3);
    assert!(tp > 0.0);
}

#[test]
fn latency_gpu_cpu_overlap_perfect() {
    let ratio = gpu_cpu_overlap_ratio(1000.0, 1000.0);
    assert!((ratio - 1.0).abs() < 0.001, "Equal GPU/CPU should give 100% overlap");
}

#[test]
fn latency_gpu_cpu_overlap_asymmetric() {
    let ratio = gpu_cpu_overlap_ratio(1000.0, 200.0);
    assert!((ratio - 0.2).abs() < 0.001, "200/1000 = 20%: got {ratio}");
}

#[test]
fn latency_gpu_cpu_overlap_zero() {
    let ratio = gpu_cpu_overlap_ratio(0.0, 0.0);
    assert_eq!(ratio, 0.0);
}

#[test]
fn latency_async_compute_stages() {
    let stages = vec![
        PipelineStage { name: "upload".into(), gpu_duration_us: 0.0, cpu_duration_us: 100.0 },
        PipelineStage { name: "compute".into(), gpu_duration_us: 800.0, cpu_duration_us: 10.0 },
        PipelineStage { name: "download".into(), gpu_duration_us: 0.0, cpu_duration_us: 100.0 },
    ];
    let tp = pipeline_throughput(&stages, 3);
    assert!(tp > 0.0);
}

#[test]
fn latency_empty_pipeline() {
    let tp = pipeline_throughput(&[], 3);
    assert_eq!(tp, 0.0);
}

#[test]
fn latency_zero_buffers() {
    let stages =
        vec![PipelineStage { name: "test".into(), gpu_duration_us: 100.0, cpu_duration_us: 100.0 }];
    let tp = pipeline_throughput(&stages, 0);
    assert_eq!(tp, 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Batch Sizing tests ────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn batch_optimal_size_respects_memory() {
    let bs = optimal_batch_size(4, 1024, 8, 10.0, 0.01);
    assert!(bs * 4 <= 1024, "Should not exceed available memory");
}

#[test]
fn batch_optimal_size_amortises_overhead() {
    let bs = optimal_batch_size(4, 1_000_000, 8, 100.0, 0.001);
    let compute_time = bs as f64 * 0.001;
    assert!(compute_time >= 10.0 * 100.0 * 0.001, "Should amortise launch overhead");
}

#[test]
fn batch_optimal_size_saturates_cores() {
    let bs = optimal_batch_size(4, 10_000_000, 8, 1.0, 1.0);
    assert!(bs >= 8 * 32, "Should saturate GPU cores");
}

#[test]
fn batch_zero_element_size() {
    assert_eq!(optimal_batch_size(0, 1024, 8, 10.0, 0.01), 0);
}

#[test]
fn batch_regime_launch_overhead_bound() {
    let r = classify_regime(16, 100.0, 8.0, 12.5);
    assert_eq!(r, ComputeRegime::LaunchOverheadBound);
}

#[test]
fn batch_regime_compute_bound() {
    // High arithmetic intensity (100 flops / 4 bytes = 25 > machine 10 flops/byte)
    let r = classify_regime(1024, 100.0, 4.0, 10.0);
    assert_eq!(r, ComputeRegime::ComputeBound);
}

#[test]
fn batch_regime_memory_bound() {
    // Low arithmetic intensity (1 flop / 8 bytes = 0.125 < machine 10 flops/byte)
    let r = classify_regime(1024, 1.0, 8.0, 10.0);
    assert_eq!(r, ComputeRegime::MemoryBound);
}

#[test]
fn batch_increasing_size_changes_regime() {
    let small = classify_regime(8, 10.0, 4.0, 5.0);
    let large = classify_regime(4096, 10.0, 4.0, 5.0);
    assert_eq!(small, ComputeRegime::LaunchOverheadBound);
    assert_ne!(large, ComputeRegime::LaunchOverheadBound);
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Apple Silicon tests ────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apple_m1_core_count() {
    let m1 = AppleSiliconProfile::m1();
    assert_eq!(m1.gpu_cores, 8);
    assert_eq!(m1.gpu_family, 7);
}

#[test]
fn apple_m2_core_count() {
    let m2 = AppleSiliconProfile::m2();
    assert_eq!(m2.gpu_cores, 10);
    assert_eq!(m2.gpu_family, 8);
}

#[test]
fn apple_m3_core_count() {
    let m3 = AppleSiliconProfile::m3();
    assert_eq!(m3.gpu_cores, 10);
    assert_eq!(m3.gpu_family, 9);
}

#[test]
fn apple_m3_max_core_count() {
    let m3_max = AppleSiliconProfile::m3_max();
    assert_eq!(m3_max.gpu_cores, 40);
    assert_eq!(m3_max.unified_memory_gb, 128);
}

#[test]
fn apple_unified_memory_bandwidth_scaling() {
    let m1 = AppleSiliconProfile::m1();
    let m3_max = AppleSiliconProfile::m3_max();
    assert!(
        m3_max.memory_bandwidth_gbps > m1.memory_bandwidth_gbps,
        "M3 Max should have higher bandwidth than M1"
    );
}

#[test]
fn apple_gpu_core_mapping_threads() {
    let m1 = AppleSiliconProfile::m1();
    assert_eq!(m1.total_gpu_threads(), 8 * 1024);
    let m3_max = AppleSiliconProfile::m3_max();
    assert_eq!(m3_max.total_gpu_threads(), 40 * 1024);
}

#[test]
fn apple_peak_gflops_scaling() {
    let m1 = AppleSiliconProfile::m1();
    let m3_max = AppleSiliconProfile::m3_max();
    assert!(
        m3_max.peak_gflops_f32() > m1.peak_gflops_f32(),
        "M3 Max should have higher peak GFLOPS"
    );
}

#[test]
fn apple_flops_per_byte_ratio() {
    let m1 = AppleSiliconProfile::m1();
    let fpb = m1.flops_per_byte();
    assert!(fpb > 0.0, "Should compute valid flops/byte ratio");
    // M1: ~2048 GFLOPS / 68.25 GB/s ≈ 30 flops/byte
    assert!(fpb > 10.0 && fpb < 100.0, "Reasonable flops/byte: {fpb}");
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Regression Detection tests ─────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn regression_no_regression() {
    let baseline = PerfBaseline { name: "matmul_256".into(), throughput: 100.0, latency_us: 50.0 };
    let (regressed, msg) = check_regression(&baseline, 105.0, 48.0, 0.10);
    assert!(!regressed, "Should not regress with better perf: {msg}");
}

#[test]
fn regression_throughput_drop() {
    let baseline = PerfBaseline { name: "matmul_256".into(), throughput: 100.0, latency_us: 50.0 };
    let (regressed, msg) = check_regression(&baseline, 80.0, 50.0, 0.10);
    assert!(regressed, "20% throughput drop should regress: {msg}");
    assert!(msg.contains("throughput regression"));
}

#[test]
fn regression_latency_increase() {
    let baseline =
        PerfBaseline { name: "attention_512".into(), throughput: 100.0, latency_us: 50.0 };
    let (regressed, msg) = check_regression(&baseline, 100.0, 70.0, 0.10);
    assert!(regressed, "40% latency increase should regress: {msg}");
    assert!(msg.contains("latency regression"));
}

#[test]
fn regression_within_threshold() {
    let baseline =
        PerfBaseline { name: "softmax_1024".into(), throughput: 100.0, latency_us: 50.0 };
    let (regressed, _) = check_regression(&baseline, 92.0, 54.0, 0.10);
    assert!(!regressed, "8% throughput drop within 10% threshold");
}

#[test]
fn regression_exact_threshold_boundary() {
    let baseline = PerfBaseline { name: "layernorm".into(), throughput: 100.0, latency_us: 100.0 };
    // Exactly at the 10% boundary
    let (regressed, _) = check_regression(&baseline, 90.0, 100.0, 0.10);
    assert!(!regressed, "Exactly at boundary should not regress");
}

#[test]
fn regression_multi_kernel_suite() {
    let baselines = [
        PerfBaseline { name: "matmul".into(), throughput: 100.0, latency_us: 50.0 },
        PerfBaseline { name: "attention".into(), throughput: 80.0, latency_us: 60.0 },
        PerfBaseline { name: "softmax".into(), throughput: 200.0, latency_us: 10.0 },
    ];
    let measured = [(105.0, 48.0), (75.0, 62.0), (195.0, 10.5)];
    let regressions: Vec<_> = baselines
        .iter()
        .zip(measured.iter())
        .map(|(b, (tp, lat))| check_regression(b, *tp, *lat, 0.10))
        .filter(|(regressed, _)| *regressed)
        .collect();
    assert!(regressions.is_empty(), "No kernel should regress: {regressions:?}");
}

#[test]
fn regression_acceptable_degradation_thresholds() {
    let baseline =
        PerfBaseline { name: "qk256_dequant".into(), throughput: 50.0, latency_us: 200.0 };
    // 5% threshold is tighter
    let (reg_tight, _) = check_regression(&baseline, 44.0, 200.0, 0.05);
    // 20% threshold is looser
    let (reg_loose, _) = check_regression(&baseline, 44.0, 200.0, 0.20);
    assert!(reg_tight, "12% drop should fail 5% threshold");
    assert!(!reg_loose, "12% drop should pass 20% threshold");
}
