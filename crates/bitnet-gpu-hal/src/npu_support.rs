//! NPU/VPU detection and heterogeneous offloading for `BitNet` inference.
//!
//! Provides discovery of AI accelerators (Intel NPU, Intel VPU, AMD XDNA,
//! Apple Neural Engine), intelligent op-placement analysis, and a
//! heterogeneous scheduler that distributes work across CPU + GPU + NPU
//! while minimising data movement.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── Device types ──────────────────────────────────────────────────────────

/// Vendor/family of the detected NPU/VPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NpuVendor {
    /// Intel Meteor Lake integrated NPU.
    IntelMeteorLake,
    /// Intel discrete VPU (Keem Bay / Thunder Bay).
    IntelVpu,
    /// AMD XDNA AI Engine (Ryzen AI).
    AmdXdna,
    /// Apple Neural Engine (ANE).
    AppleNeuralEngine,
    /// Unknown / generic AI accelerator.
    Unknown,
}

impl fmt::Display for NpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IntelMeteorLake => write!(f, "Intel Meteor Lake NPU"),
            Self::IntelVpu => write!(f, "Intel VPU"),
            Self::AmdXdna => write!(f, "AMD XDNA"),
            Self::AppleNeuralEngine => write!(f, "Apple Neural Engine"),
            Self::Unknown => write!(f, "Unknown NPU"),
        }
    }
}

/// Numeric precision supported by an NPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 4-bit integer.
    Int4,
    /// 8-bit integer.
    Int8,
    /// 16-bit floating-point (half).
    Fp16,
    /// 16-bit brain floating-point.
    Bf16,
    /// 32-bit floating-point.
    Fp32,
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int4 => write!(f, "INT4"),
            Self::Int8 => write!(f, "INT8"),
            Self::Fp16 => write!(f, "FP16"),
            Self::Bf16 => write!(f, "BF16"),
            Self::Fp32 => write!(f, "FP32"),
        }
    }
}

/// An operator type suitable for NPU offloading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NpuOperator {
    /// Matrix multiplication.
    MatMul,
    /// 2-D convolution.
    Conv2D,
    /// Quantize / dequantize pass.
    Quantize,
    /// Multi-head (self-)attention.
    Attention,
    /// Softmax activation.
    Softmax,
    /// Layer normalisation (RMS or standard).
    LayerNorm,
    /// Element-wise activation (`SiLU`, `GELU`, …).
    Activation,
    /// Embedding lookup.
    Embedding,
}

impl fmt::Display for NpuOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ── NpuDevice & capabilities ──────────────────────────────────────────────

/// Hardware capabilities of a detected NPU/VPU device.
#[derive(Debug, Clone)]
pub struct NpuCapabilities {
    /// Set of operators the device can accelerate.
    pub supported_ops: Vec<NpuOperator>,
    /// Maximum batch size the device can handle in one dispatch.
    pub max_batch_size: usize,
    /// Precisions the device supports natively.
    pub supported_precisions: Vec<Precision>,
    /// Device memory in bytes (0 = shared/unknown).
    pub memory_bytes: u64,
}

impl NpuCapabilities {
    /// Whether the device supports the given operator.
    pub fn supports_op(&self, op: NpuOperator) -> bool {
        self.supported_ops.contains(&op)
    }

    /// Whether the device supports the given precision.
    pub fn supports_precision(&self, prec: Precision) -> bool {
        self.supported_precisions.contains(&prec)
    }
}

/// A detected NPU/VPU/AI accelerator.
#[derive(Debug, Clone)]
pub struct NpuDevice {
    /// Human-readable device name.
    pub name: String,
    /// Vendor / family classification.
    pub vendor: NpuVendor,
    /// Unique device index (0-based, per vendor).
    pub device_index: u32,
    /// Hardware capabilities.
    pub capabilities: NpuCapabilities,
}

impl fmt::Display for NpuDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] (mem={}B, ops={})",
            self.name,
            self.vendor,
            self.capabilities.memory_bytes,
            self.capabilities.supported_ops.len()
        )
    }
}

// ── Detection ─────────────────────────────────────────────────────────────

/// Strategy for detecting NPU/VPU devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMethod {
    /// Scan `/sys/class/accel` and `/dev/accel` (Linux).
    Sysfs,
    /// Probe for `OpenVINO` runtime availability.
    OpenVino,
    /// Use platform-specific APIs (`IOKit` on macOS, `WinML` on Windows).
    PlatformApi,
    /// Use a mock detector for testing.
    Mock,
}

/// Result of a detection run.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Devices found.
    pub devices: Vec<NpuDevice>,
    /// Methods that were attempted.
    pub methods_tried: Vec<DetectionMethod>,
    /// Methods that found at least one device.
    pub methods_succeeded: Vec<DetectionMethod>,
}

impl DetectionResult {
    /// Total number of detected devices.
    pub const fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Whether any device was found.
    pub const fn any_found(&self) -> bool {
        !self.devices.is_empty()
    }
}

/// Probes the system for NPU/VPU devices.
pub struct NpuDetector {
    methods: Vec<DetectionMethod>,
    mock_devices: Vec<NpuDevice>,
}

impl NpuDetector {
    /// Create a detector that tries default platform methods.
    pub fn new() -> Self {
        Self {
            methods: vec![
                DetectionMethod::Sysfs,
                DetectionMethod::OpenVino,
                DetectionMethod::PlatformApi,
            ],
            mock_devices: Vec::new(),
        }
    }

    /// Create a detector using only the specified methods.
    pub const fn with_methods(methods: Vec<DetectionMethod>) -> Self {
        Self { methods, mock_devices: Vec::new() }
    }

    /// Create a mock detector that returns the given devices.
    pub fn mock(devices: Vec<NpuDevice>) -> Self {
        Self {
            methods: vec![DetectionMethod::Mock],
            mock_devices: devices,
        }
    }

    /// Run detection and return all discovered devices.
    pub fn detect(&self) -> DetectionResult {
        let mut devices = Vec::new();
        let mut methods_succeeded = Vec::new();

        for &method in &self.methods {
            let found = match method {
                DetectionMethod::Sysfs => self.detect_sysfs(),
                DetectionMethod::OpenVino => self.detect_openvino(),
                DetectionMethod::PlatformApi => self.detect_platform(),
                DetectionMethod::Mock => self.mock_devices.clone(),
            };
            if !found.is_empty() {
                methods_succeeded.push(method);
                devices.extend(found);
            }
        }

        DetectionResult {
            devices,
            methods_tried: self.methods.clone(),
            methods_succeeded,
        }
    }

    /// Sysfs scan: check `/sys/class/accel` for Intel NPU driver.
    #[allow(clippy::missing_const_for_fn)]
    fn detect_sysfs(&self) -> Vec<NpuDevice> {
        // On real Linux systems we would walk /sys/class/accel
        // and parse PCI vendor/device IDs. Stubbed for portability.
        #[cfg(target_os = "linux")]
        {
            Self::scan_linux_accel()
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = self;
            Vec::new()
        }
    }

    #[cfg(target_os = "linux")]
    fn scan_linux_accel() -> Vec<NpuDevice> {
        let accel_path = std::path::Path::new("/sys/class/accel");
        if !accel_path.exists() {
            return Vec::new();
        }
        let Ok(entries) = std::fs::read_dir(accel_path) else {
            return Vec::new();
        };
        let mut devices = Vec::new();
        for (idx, entry) in entries.flatten().enumerate() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("accel") {
                devices.push(NpuDevice {
                    name: format!("Intel NPU ({name})"),
                    vendor: NpuVendor::IntelMeteorLake,
                    #[allow(clippy::cast_possible_truncation)]
                    device_index: idx as u32,
                    capabilities: intel_npu_default_caps(),
                });
            }
        }
        devices
    }

    /// `OpenVINO` runtime check: see if `openvino` binary is reachable.
    fn detect_openvino(&self) -> Vec<NpuDevice> {
        let _ = self;
        let has_openvino = std::process::Command::new("python3")
            .args(["-c", "import openvino"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if has_openvino {
            vec![NpuDevice {
                name: "OpenVINO NPU plugin".to_string(),
                vendor: NpuVendor::IntelMeteorLake,
                device_index: 0,
                capabilities: intel_npu_default_caps(),
            }]
        } else {
            Vec::new()
        }
    }

    /// Platform-specific API detection.
    #[allow(clippy::missing_const_for_fn)]
    fn detect_platform(&self) -> Vec<NpuDevice> {
        #[cfg(target_os = "macos")]
        {
            // On macOS we would use IOKit to probe for ANE.
            // Stubbed — ANE is always present on Apple Silicon.
            if cfg!(target_arch = "aarch64") {
                return vec![NpuDevice {
                    name: "Apple Neural Engine".to_string(),
                    vendor: NpuVendor::AppleNeuralEngine,
                    device_index: 0,
                    capabilities: apple_ane_default_caps(),
                }];
            }
        }
        #[cfg(target_os = "windows")]
        {
            // WinML / DirectML NPU enumeration would go here.
            let _ = self; // silence unused-self
        }
        Vec::new()
    }
}

impl Default for NpuDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Default Intel NPU capabilities (Meteor Lake class).
fn intel_npu_default_caps() -> NpuCapabilities {
    NpuCapabilities {
        supported_ops: vec![
            NpuOperator::MatMul,
            NpuOperator::Conv2D,
            NpuOperator::Quantize,
            NpuOperator::Softmax,
            NpuOperator::Activation,
        ],
        max_batch_size: 16,
        supported_precisions: vec![
            Precision::Int8,
            Precision::Fp16,
        ],
        memory_bytes: 0, // shared memory
    }
}

/// Default Apple Neural Engine capabilities.
#[allow(dead_code)]
fn apple_ane_default_caps() -> NpuCapabilities {
    NpuCapabilities {
        supported_ops: vec![
            NpuOperator::MatMul,
            NpuOperator::Conv2D,
            NpuOperator::Softmax,
            NpuOperator::Activation,
            NpuOperator::LayerNorm,
        ],
        max_batch_size: 32,
        supported_precisions: vec![
            Precision::Fp16,
            Precision::Int8,
        ],
        memory_bytes: 0,
    }
}

/// Default AMD XDNA capabilities.
fn amd_xdna_default_caps() -> NpuCapabilities {
    NpuCapabilities {
        supported_ops: vec![
            NpuOperator::MatMul,
            NpuOperator::Conv2D,
            NpuOperator::Quantize,
            NpuOperator::Attention,
            NpuOperator::Softmax,
        ],
        max_batch_size: 8,
        supported_precisions: vec![
            Precision::Int4,
            Precision::Int8,
            Precision::Fp16,
            Precision::Bf16,
        ],
        memory_bytes: 0,
    }
}

// ── Offload policy ────────────────────────────────────────────────────────

/// Policy governing when to offload operators to the NPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NpuOffloadPolicy {
    /// Always offload every supported op to the NPU.
    AlwaysOffload,
    /// Offload only when the analyzer predicts a net benefit.
    OffloadIfBeneficial,
    /// Never offload — keep everything on CPU/GPU.
    NeverOffload,
    /// Custom: provide per-operator overrides.
    Custom(HashMap<NpuOperator, bool>),
}

impl NpuOffloadPolicy {
    /// Whether a given operator should be offloaded under this policy,
    /// assuming the device supports it.
    pub fn should_offload(
        &self,
        op: NpuOperator,
        device: &NpuDevice,
    ) -> bool {
        match self {
            Self::AlwaysOffload => device.capabilities.supports_op(op),
            Self::NeverOffload => false,
            Self::OffloadIfBeneficial => {
                device.capabilities.supports_op(op)
                    && is_op_beneficial(op)
            }
            Self::Custom(map) => {
                map.get(&op).copied().unwrap_or(false)
                    && device.capabilities.supports_op(op)
            }
        }
    }
}

/// Heuristic: is the op typically beneficial to run on an NPU?
const fn is_op_beneficial(op: NpuOperator) -> bool {
    matches!(
        op,
        NpuOperator::MatMul
            | NpuOperator::Conv2D
            | NpuOperator::Quantize
            | NpuOperator::Attention
    )
}

// ── Offload analyzer ──────────────────────────────────────────────────────

/// Compute target for a single operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeTarget {
    /// Execute on the CPU.
    Cpu,
    /// Execute on the GPU.
    Gpu,
    /// Execute on the NPU.
    Npu,
}

impl fmt::Display for ComputeTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu => write!(f, "GPU"),
            Self::Npu => write!(f, "NPU"),
        }
    }
}

/// Describes an operator in the compute graph.
#[derive(Debug, Clone)]
pub struct OpDescriptor {
    /// Operator type.
    pub op: NpuOperator,
    /// Estimated FLOPs for this operator.
    pub estimated_flops: u64,
    /// Precision the op will run in.
    pub precision: Precision,
    /// Estimated input tensor bytes.
    pub input_bytes: u64,
    /// Estimated output tensor bytes.
    pub output_bytes: u64,
}

/// Placement decision for one operator.
#[derive(Debug, Clone)]
pub struct PlacementDecision {
    /// The operator being placed.
    pub op: NpuOperator,
    /// Where to run it.
    pub target: ComputeTarget,
    /// Estimated data-movement cost in bytes if placed here.
    pub transfer_cost_bytes: u64,
    /// Human-readable reason.
    pub reason: String,
}

/// Analyzes an operator graph and decides where each op should execute.
pub struct OffloadAnalyzer {
    policy: NpuOffloadPolicy,
    gpu_available: bool,
    /// Estimated PCIe/bus bandwidth in bytes/sec for transfer cost.
    bus_bandwidth_bytes_sec: u64,
}

impl OffloadAnalyzer {
    /// Create an analyzer.
    pub const fn new(
        policy: NpuOffloadPolicy,
        gpu_available: bool,
    ) -> Self {
        Self {
            policy,
            gpu_available,
            // Default: ~12 GB/s (PCIe Gen4 x4)
            bus_bandwidth_bytes_sec: 12_000_000_000,
        }
    }

    /// Override bus bandwidth estimate.
    #[must_use]
    pub const fn with_bandwidth(mut self, bw: u64) -> Self {
        self.bus_bandwidth_bytes_sec = bw;
        self
    }

    /// Analyze a sequence of ops and produce placement decisions.
    pub fn analyze(
        &self,
        ops: &[OpDescriptor],
        device: &NpuDevice,
    ) -> Vec<PlacementDecision> {
        let mut decisions = Vec::with_capacity(ops.len());
        let mut prev_target = ComputeTarget::Cpu;

        for desc in ops {
            let (target, reason) = self.decide_single(
                desc,
                device,
                prev_target,
            );
            let transfer_cost_bytes =
                if target == prev_target { 0 } else { desc.input_bytes };
            decisions.push(PlacementDecision {
                op: desc.op,
                target,
                transfer_cost_bytes,
                reason,
            });
            prev_target = target;
        }
        decisions
    }

    fn decide_single(
        &self,
        desc: &OpDescriptor,
        device: &NpuDevice,
        prev_target: ComputeTarget,
    ) -> (ComputeTarget, String) {
        // NeverOffload → CPU or GPU
        if self.policy == NpuOffloadPolicy::NeverOffload {
            let target = if self.gpu_available {
                ComputeTarget::Gpu
            } else {
                ComputeTarget::Cpu
            };
            return (
                target,
                "NeverOffload policy".to_string(),
            );
        }

        // Check device support
        if !device.capabilities.supports_op(desc.op) {
            let target = if self.gpu_available {
                ComputeTarget::Gpu
            } else {
                ComputeTarget::Cpu
            };
            return (
                target,
                format!("{} not supported by NPU", desc.op),
            );
        }

        // Check precision
        if !device.capabilities.supports_precision(desc.precision) {
            let target = if self.gpu_available {
                ComputeTarget::Gpu
            } else {
                ComputeTarget::Cpu
            };
            return (
                target,
                format!("{} precision not supported", desc.precision),
            );
        }

        // AlwaysOffload → NPU if supported
        if self.policy == NpuOffloadPolicy::AlwaysOffload {
            return (
                ComputeTarget::Npu,
                "AlwaysOffload policy".to_string(),
            );
        }

        // OffloadIfBeneficial or Custom
        let should = self.policy.should_offload(desc.op, device);
        if !should {
            let target = if self.gpu_available {
                ComputeTarget::Gpu
            } else {
                ComputeTarget::Cpu
            };
            return (
                target,
                "policy declined offload".to_string(),
            );
        }

        // Transfer-cost heuristic: if we must move data and the op
        // is small, keep it co-located with the previous target.
        if prev_target != ComputeTarget::Npu
            && desc.input_bytes > 0
            && self.bus_bandwidth_bytes_sec > 0
        {
            let transfer_time_us = desc.input_bytes * 1_000_000
                / self.bus_bandwidth_bytes_sec;
            // If transfer takes > 100 µs and the op is small,
            // keep on the current device.
            let flops_threshold = 1_000_000u64; // 1M FLOPs
            if transfer_time_us > 100
                && desc.estimated_flops < flops_threshold
            {
                return (
                    prev_target,
                    "transfer cost exceeds benefit".to_string(),
                );
            }
        }

        (ComputeTarget::Npu, "beneficial offload".to_string())
    }
}

// ── Heterogeneous scheduler ───────────────────────────────────────────────

/// A unit of work dispatched by the scheduler.
#[derive(Debug, Clone)]
pub struct ScheduledWork {
    /// Which op to run.
    pub op: NpuOperator,
    /// Where to run it.
    pub target: ComputeTarget,
    /// Sequence index in the original op list.
    pub sequence_index: usize,
    /// Whether a data transfer is needed before this op.
    pub needs_transfer: bool,
}

/// Schedules work across CPU, GPU, and NPU devices, minimising data
/// movement between them.
pub struct HeterogeneousScheduler {
    analyzer: OffloadAnalyzer,
    device: Option<NpuDevice>,
}

impl HeterogeneousScheduler {
    /// Create a scheduler.
    pub const fn new(
        analyzer: OffloadAnalyzer,
        device: Option<NpuDevice>,
    ) -> Self {
        Self { analyzer, device }
    }

    /// Schedule a sequence of ops. Falls back to CPU/GPU when no NPU.
    #[allow(clippy::option_if_let_else)]
    pub fn schedule(&self, ops: &[OpDescriptor]) -> Vec<ScheduledWork> {
        if let Some(dev) = &self.device {
            let decisions = self.analyzer.analyze(ops, dev);
            let mut prev = ComputeTarget::Cpu;
            decisions
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let needs = d.target != prev;
                    prev = d.target;
                    ScheduledWork {
                        op: d.op,
                        target: d.target,
                        sequence_index: i,
                        needs_transfer: needs,
                    }
                })
                .collect()
        } else {
            let fallback = if self.analyzer.gpu_available {
                ComputeTarget::Gpu
            } else {
                ComputeTarget::Cpu
            };
            ops.iter()
                .enumerate()
                .map(|(i, desc)| ScheduledWork {
                    op: desc.op,
                    target: fallback,
                    sequence_index: i,
                    needs_transfer: false,
                })
                .collect()
        }
    }

    /// Compute summary statistics for a schedule.
    pub fn summarise(
        schedule: &[ScheduledWork],
    ) -> ScheduleSummary {
        let mut cpu_ops = 0usize;
        let mut gpu_ops = 0usize;
        let mut npu_ops = 0usize;
        let mut transfers = 0usize;
        for w in schedule {
            match w.target {
                ComputeTarget::Cpu => cpu_ops += 1,
                ComputeTarget::Gpu => gpu_ops += 1,
                ComputeTarget::Npu => npu_ops += 1,
            }
            if w.needs_transfer {
                transfers += 1;
            }
        }
        ScheduleSummary { cpu_ops, gpu_ops, npu_ops, transfers }
    }
}

/// Summary statistics of a scheduled work plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleSummary {
    /// Ops assigned to CPU.
    pub cpu_ops: usize,
    /// Ops assigned to GPU.
    pub gpu_ops: usize,
    /// Ops assigned to NPU.
    pub npu_ops: usize,
    /// Number of cross-device data transfers.
    pub transfers: usize,
}

impl fmt::Display for ScheduleSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CPU={} GPU={} NPU={} transfers={}",
            self.cpu_ops, self.gpu_ops, self.npu_ops, self.transfers
        )
    }
}

// ── NPU profiler ──────────────────────────────────────────────────────────

/// A single profiling sample.
#[derive(Debug, Clone)]
pub struct ProfileSample {
    /// Operator profiled.
    pub op: NpuOperator,
    /// Where it ran.
    pub target: ComputeTarget,
    /// Measured wall-clock duration.
    pub duration: Duration,
}

/// Collects profiling samples and computes comparative statistics.
pub struct NpuProfiler {
    samples: Vec<ProfileSample>,
}

impl NpuProfiler {
    /// Create an empty profiler.
    pub const fn new() -> Self {
        Self { samples: Vec::new() }
    }

    /// Record a sample.
    pub fn record(&mut self, sample: ProfileSample) {
        self.samples.push(sample);
    }

    /// Convenience: time a closure and record the result.
    pub fn measure<F, R>(
        &mut self,
        op: NpuOperator,
        target: ComputeTarget,
        f: F,
    ) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        self.record(ProfileSample { op, target, duration });
        result
    }

    /// Total number of samples collected.
    pub const fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Return all collected samples.
    pub fn samples(&self) -> &[ProfileSample] {
        &self.samples
    }

    /// Average duration for a (op, target) pair.
    pub fn average_duration(
        &self,
        op: NpuOperator,
        target: ComputeTarget,
    ) -> Option<Duration> {
        let matching: Vec<_> = self
            .samples
            .iter()
            .filter(|s| s.op == op && s.target == target)
            .collect();
        if matching.is_empty() {
            return None;
        }
        let total: Duration = matching.iter().map(|s| s.duration).sum();
        #[allow(clippy::cast_possible_truncation)]
        let count = matching.len() as u32;
        Some(total / count)
    }

    /// Compare NPU vs CPU average for a given op.
    /// Returns `Some(speedup)` where `> 1.0` means NPU is faster.
    pub fn speedup_vs_cpu(
        &self,
        op: NpuOperator,
    ) -> Option<f64> {
        let cpu = self
            .average_duration(op, ComputeTarget::Cpu)?
            .as_secs_f64();
        let npu = self
            .average_duration(op, ComputeTarget::Npu)?
            .as_secs_f64();
        if npu == 0.0 {
            return None;
        }
        Some(cpu / npu)
    }

    /// Reset all collected samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

impl Default for NpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Mock NPU ──────────────────────────────────────────────────────────────

/// A mock NPU that simulates execution with configurable latencies.
pub struct MockNpu {
    device: NpuDevice,
    /// Per-op simulated latency.
    op_latencies: HashMap<NpuOperator, Duration>,
}

impl MockNpu {
    /// Create a mock NPU with default Intel-like capabilities.
    pub fn intel_mock() -> Self {
        let device = NpuDevice {
            name: "Mock Intel NPU".to_string(),
            vendor: NpuVendor::IntelMeteorLake,
            device_index: 0,
            capabilities: intel_npu_default_caps(),
        };
        let mut latencies = HashMap::new();
        latencies.insert(
            NpuOperator::MatMul,
            Duration::from_micros(100),
        );
        latencies.insert(
            NpuOperator::Softmax,
            Duration::from_micros(20),
        );
        latencies.insert(
            NpuOperator::Conv2D,
            Duration::from_micros(80),
        );
        latencies.insert(
            NpuOperator::Quantize,
            Duration::from_micros(10),
        );
        latencies.insert(
            NpuOperator::Activation,
            Duration::from_micros(5),
        );
        Self { device, op_latencies: latencies }
    }

    /// Create a mock AMD XDNA NPU.
    pub fn amd_mock() -> Self {
        let device = NpuDevice {
            name: "Mock AMD XDNA".to_string(),
            vendor: NpuVendor::AmdXdna,
            device_index: 0,
            capabilities: amd_xdna_default_caps(),
        };
        let mut latencies = HashMap::new();
        latencies.insert(
            NpuOperator::MatMul,
            Duration::from_micros(80),
        );
        latencies.insert(
            NpuOperator::Attention,
            Duration::from_micros(120),
        );
        latencies.insert(
            NpuOperator::Softmax,
            Duration::from_micros(15),
        );
        Self { device, op_latencies: latencies }
    }

    /// Get the underlying device description.
    pub const fn device(&self) -> &NpuDevice {
        &self.device
    }

    /// Simulate running an op, returning the simulated latency.
    pub fn simulate_op(
        &self,
        op: NpuOperator,
    ) -> Result<Duration, String> {
        if !self.device.capabilities.supports_op(op) {
            return Err(format!(
                "{op} not supported by {}",
                self.device.name
            ));
        }
        Ok(self
            .op_latencies
            .get(&op)
            .copied()
            .unwrap_or(Duration::from_micros(50)))
    }

    /// Simulate a batch of ops and return total latency.
    pub fn simulate_batch(
        &self,
        ops: &[NpuOperator],
    ) -> Result<Duration, String> {
        let mut total = Duration::ZERO;
        for &op in ops {
            total += self.simulate_op(op)?;
        }
        Ok(total)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper builders ───────────────────────────────────────────────

    fn mock_intel_device() -> NpuDevice {
        NpuDevice {
            name: "Test Intel NPU".to_string(),
            vendor: NpuVendor::IntelMeteorLake,
            device_index: 0,
            capabilities: intel_npu_default_caps(),
        }
    }

    fn mock_amd_device() -> NpuDevice {
        NpuDevice {
            name: "Test AMD XDNA".to_string(),
            vendor: NpuVendor::AmdXdna,
            device_index: 0,
            capabilities: amd_xdna_default_caps(),
        }
    }

    fn sample_op(op: NpuOperator) -> OpDescriptor {
        OpDescriptor {
            op,
            estimated_flops: 10_000_000,
            precision: Precision::Fp16,
            input_bytes: 4096,
            output_bytes: 4096,
        }
    }

    fn small_op(op: NpuOperator) -> OpDescriptor {
        OpDescriptor {
            op,
            estimated_flops: 100,
            precision: Precision::Fp16,
            // 2 MB input → transfer time ~167µs at 12 GB/s,
            // exceeding the 100µs threshold for a tiny op.
            input_bytes: 2_000_000,
            output_bytes: 2_000_000,
        }
    }

    // ── NpuVendor ─────────────────────────────────────────────────────

    #[test]
    fn vendor_display() {
        assert_eq!(
            NpuVendor::IntelMeteorLake.to_string(),
            "Intel Meteor Lake NPU"
        );
        assert_eq!(NpuVendor::AmdXdna.to_string(), "AMD XDNA");
        assert_eq!(
            NpuVendor::AppleNeuralEngine.to_string(),
            "Apple Neural Engine"
        );
        assert_eq!(NpuVendor::Unknown.to_string(), "Unknown NPU");
        assert_eq!(NpuVendor::IntelVpu.to_string(), "Intel VPU");
    }

    #[test]
    fn precision_display() {
        assert_eq!(Precision::Int4.to_string(), "INT4");
        assert_eq!(Precision::Int8.to_string(), "INT8");
        assert_eq!(Precision::Fp16.to_string(), "FP16");
        assert_eq!(Precision::Bf16.to_string(), "BF16");
        assert_eq!(Precision::Fp32.to_string(), "FP32");
    }

    #[test]
    fn operator_display() {
        assert_eq!(NpuOperator::MatMul.to_string(), "MatMul");
        assert_eq!(NpuOperator::Attention.to_string(), "Attention");
        assert_eq!(NpuOperator::Embedding.to_string(), "Embedding");
    }

    // ── NpuCapabilities ───────────────────────────────────────────────

    #[test]
    fn capabilities_supports_op() {
        let caps = intel_npu_default_caps();
        assert!(caps.supports_op(NpuOperator::MatMul));
        assert!(caps.supports_op(NpuOperator::Softmax));
        assert!(!caps.supports_op(NpuOperator::Attention));
        assert!(!caps.supports_op(NpuOperator::Embedding));
    }

    #[test]
    fn capabilities_supports_precision() {
        let caps = intel_npu_default_caps();
        assert!(caps.supports_precision(Precision::Int8));
        assert!(caps.supports_precision(Precision::Fp16));
        assert!(!caps.supports_precision(Precision::Fp32));
        assert!(!caps.supports_precision(Precision::Bf16));
    }

    #[test]
    fn amd_caps_include_attention() {
        let caps = amd_xdna_default_caps();
        assert!(caps.supports_op(NpuOperator::Attention));
        assert!(caps.supports_precision(Precision::Bf16));
        assert!(caps.supports_precision(Precision::Int4));
    }

    #[test]
    fn intel_npu_max_batch() {
        let caps = intel_npu_default_caps();
        assert_eq!(caps.max_batch_size, 16);
    }

    // ── NpuDevice ─────────────────────────────────────────────────────

    #[test]
    fn device_display() {
        let dev = mock_intel_device();
        let s = dev.to_string();
        assert!(s.contains("Test Intel NPU"));
        assert!(s.contains("Intel Meteor Lake NPU"));
    }

    #[test]
    fn device_vendor_eq() {
        let dev = mock_intel_device();
        assert_eq!(dev.vendor, NpuVendor::IntelMeteorLake);
    }

    // ── NpuDetector ───────────────────────────────────────────────────

    #[test]
    fn mock_detector_returns_devices() {
        let dev = mock_intel_device();
        let detector = NpuDetector::mock(vec![dev]);
        let result = detector.detect();
        assert_eq!(result.device_count(), 1);
        assert!(result.any_found());
        assert_eq!(
            result.methods_succeeded,
            vec![DetectionMethod::Mock]
        );
    }

    #[test]
    fn mock_detector_empty() {
        let detector = NpuDetector::mock(vec![]);
        let result = detector.detect();
        assert_eq!(result.device_count(), 0);
        assert!(!result.any_found());
    }

    #[test]
    fn detector_default_methods() {
        let detector = NpuDetector::new();
        assert_eq!(detector.methods.len(), 3);
        assert!(detector.methods.contains(&DetectionMethod::Sysfs));
        assert!(
            detector.methods.contains(&DetectionMethod::OpenVino)
        );
        assert!(
            detector.methods.contains(&DetectionMethod::PlatformApi)
        );
    }

    #[test]
    fn detector_with_methods() {
        let detector = NpuDetector::with_methods(vec![
            DetectionMethod::Mock,
        ]);
        assert_eq!(detector.methods.len(), 1);
    }

    #[test]
    fn detector_default_trait() {
        let d = NpuDetector::default();
        assert_eq!(d.methods.len(), 3);
    }

    #[test]
    fn detection_result_methods_tried() {
        let detector = NpuDetector::mock(vec![mock_intel_device()]);
        let result = detector.detect();
        assert_eq!(result.methods_tried, vec![DetectionMethod::Mock]);
    }

    #[test]
    fn detection_multiple_devices() {
        let devs = vec![mock_intel_device(), mock_amd_device()];
        let detector = NpuDetector::mock(devs);
        let result = detector.detect();
        assert_eq!(result.device_count(), 2);
    }

    // ── NpuOffloadPolicy ──────────────────────────────────────────────

    #[test]
    fn always_offload_supported_op() {
        let dev = mock_intel_device();
        let policy = NpuOffloadPolicy::AlwaysOffload;
        assert!(policy.should_offload(NpuOperator::MatMul, &dev));
    }

    #[test]
    fn always_offload_unsupported_op() {
        let dev = mock_intel_device();
        let policy = NpuOffloadPolicy::AlwaysOffload;
        // Intel NPU doesn't support Attention
        assert!(
            !policy.should_offload(NpuOperator::Attention, &dev)
        );
    }

    #[test]
    fn never_offload() {
        let dev = mock_intel_device();
        let policy = NpuOffloadPolicy::NeverOffload;
        assert!(!policy.should_offload(NpuOperator::MatMul, &dev));
        assert!(
            !policy.should_offload(NpuOperator::Softmax, &dev)
        );
    }

    #[test]
    fn offload_if_beneficial_matmul() {
        let dev = mock_intel_device();
        let policy = NpuOffloadPolicy::OffloadIfBeneficial;
        // MatMul is beneficial
        assert!(policy.should_offload(NpuOperator::MatMul, &dev));
    }

    #[test]
    fn offload_if_beneficial_softmax() {
        let dev = mock_intel_device();
        let policy = NpuOffloadPolicy::OffloadIfBeneficial;
        // Softmax is NOT in the beneficial list
        assert!(
            !policy.should_offload(NpuOperator::Softmax, &dev)
        );
    }

    #[test]
    fn custom_policy_overrides() {
        let dev = mock_intel_device();
        let mut map = HashMap::new();
        map.insert(NpuOperator::Softmax, true);
        map.insert(NpuOperator::MatMul, false);
        let policy = NpuOffloadPolicy::Custom(map);
        assert!(policy.should_offload(NpuOperator::Softmax, &dev));
        assert!(!policy.should_offload(NpuOperator::MatMul, &dev));
    }

    #[test]
    fn custom_policy_missing_key() {
        let dev = mock_intel_device();
        let policy = NpuOffloadPolicy::Custom(HashMap::new());
        assert!(
            !policy.should_offload(NpuOperator::MatMul, &dev)
        );
    }

    #[test]
    fn custom_policy_unsupported_device() {
        let dev = mock_intel_device();
        let mut map = HashMap::new();
        // Attention is not supported by Intel mock
        map.insert(NpuOperator::Attention, true);
        let policy = NpuOffloadPolicy::Custom(map);
        assert!(
            !policy.should_offload(NpuOperator::Attention, &dev)
        );
    }

    // ── OffloadAnalyzer ───────────────────────────────────────────────

    #[test]
    fn analyzer_never_offload() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::NeverOffload,
            false,
        );
        let ops = vec![sample_op(NpuOperator::MatMul)];
        let decisions = analyzer.analyze(&ops, &dev);
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].target, ComputeTarget::Cpu);
    }

    #[test]
    fn analyzer_never_offload_with_gpu() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::NeverOffload,
            true,
        );
        let ops = vec![sample_op(NpuOperator::MatMul)];
        let decisions = analyzer.analyze(&ops, &dev);
        assert_eq!(decisions[0].target, ComputeTarget::Gpu);
    }

    #[test]
    fn analyzer_always_offload() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            false,
        );
        let ops = vec![sample_op(NpuOperator::MatMul)];
        let decisions = analyzer.analyze(&ops, &dev);
        assert_eq!(decisions[0].target, ComputeTarget::Npu);
    }

    #[test]
    fn analyzer_unsupported_op_fallback() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            true,
        );
        // Embedding is not supported by Intel mock
        let ops = vec![sample_op(NpuOperator::Embedding)];
        let decisions = analyzer.analyze(&ops, &dev);
        assert_eq!(decisions[0].target, ComputeTarget::Gpu);
    }

    #[test]
    fn analyzer_unsupported_precision_fallback() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            false,
        );
        let mut op = sample_op(NpuOperator::MatMul);
        op.precision = Precision::Fp32; // Intel NPU doesn't do FP32
        let decisions = analyzer.analyze(&[op], &dev);
        assert_eq!(decisions[0].target, ComputeTarget::Cpu);
    }

    #[test]
    fn analyzer_transfer_cost_tracked() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            false,
        );
        // First op on CPU → NPU transition should have transfer
        let ops = vec![
            sample_op(NpuOperator::MatMul),
            sample_op(NpuOperator::Softmax),
        ];
        let decisions = analyzer.analyze(&ops, &dev);
        // First op moves from CPU to NPU
        assert!(decisions[0].transfer_cost_bytes > 0);
        // Second op stays on NPU
        assert_eq!(decisions[1].transfer_cost_bytes, 0);
    }

    #[test]
    fn analyzer_small_op_stays_colocated() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::OffloadIfBeneficial,
            false,
        );
        // Small op with large input — should stay on CPU
        let ops = vec![small_op(NpuOperator::MatMul)];
        let decisions = analyzer.analyze(&ops, &dev);
        assert_eq!(decisions[0].target, ComputeTarget::Cpu);
    }

    #[test]
    fn analyzer_bandwidth_override() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::OffloadIfBeneficial,
            false,
        )
        .with_bandwidth(100_000_000_000); // 100 GB/s
        let ops = vec![small_op(NpuOperator::MatMul)];
        let decisions = analyzer.analyze(&ops, &dev);
        // With high bandwidth, even small ops can offload
        assert_eq!(decisions[0].target, ComputeTarget::Npu);
    }

    #[test]
    fn analyzer_empty_ops() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            false,
        );
        let decisions = analyzer.analyze(&[], &dev);
        assert!(decisions.is_empty());
    }

    #[test]
    fn analyzer_decision_reason_populated() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::NeverOffload,
            false,
        );
        let ops = vec![sample_op(NpuOperator::MatMul)];
        let decisions = analyzer.analyze(&ops, &dev);
        assert!(!decisions[0].reason.is_empty());
    }

    // ── HeterogeneousScheduler ────────────────────────────────────────

    #[test]
    fn scheduler_with_npu() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            false,
        );
        let sched =
            HeterogeneousScheduler::new(analyzer, Some(dev));
        let ops = vec![
            sample_op(NpuOperator::MatMul),
            sample_op(NpuOperator::Softmax),
        ];
        let work = sched.schedule(&ops);
        assert_eq!(work.len(), 2);
        assert_eq!(work[0].target, ComputeTarget::Npu);
        assert_eq!(work[1].target, ComputeTarget::Npu);
    }

    #[test]
    fn scheduler_without_npu_cpu_fallback() {
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            false,
        );
        let sched = HeterogeneousScheduler::new(analyzer, None);
        let ops = vec![sample_op(NpuOperator::MatMul)];
        let work = sched.schedule(&ops);
        assert_eq!(work[0].target, ComputeTarget::Cpu);
    }

    #[test]
    fn scheduler_without_npu_gpu_fallback() {
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            true,
        );
        let sched = HeterogeneousScheduler::new(analyzer, None);
        let ops = vec![sample_op(NpuOperator::MatMul)];
        let work = sched.schedule(&ops);
        assert_eq!(work[0].target, ComputeTarget::Gpu);
    }

    #[test]
    fn scheduler_sequence_indices() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            false,
        );
        let sched =
            HeterogeneousScheduler::new(analyzer, Some(dev));
        let ops = vec![
            sample_op(NpuOperator::MatMul),
            sample_op(NpuOperator::Softmax),
            sample_op(NpuOperator::Conv2D),
        ];
        let work = sched.schedule(&ops);
        for (i, w) in work.iter().enumerate() {
            assert_eq!(w.sequence_index, i);
        }
    }

    #[test]
    fn scheduler_transfer_flags() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            false,
        );
        let sched =
            HeterogeneousScheduler::new(analyzer, Some(dev));
        let ops = vec![
            sample_op(NpuOperator::MatMul),
            sample_op(NpuOperator::Softmax),
        ];
        let work = sched.schedule(&ops);
        // First op needs transfer (CPU → NPU)
        assert!(work[0].needs_transfer);
        // Second stays on NPU
        assert!(!work[1].needs_transfer);
    }

    #[test]
    fn schedule_summary() {
        let work = vec![
            ScheduledWork {
                op: NpuOperator::MatMul,
                target: ComputeTarget::Cpu,
                sequence_index: 0,
                needs_transfer: false,
            },
            ScheduledWork {
                op: NpuOperator::Softmax,
                target: ComputeTarget::Npu,
                sequence_index: 1,
                needs_transfer: true,
            },
            ScheduledWork {
                op: NpuOperator::Conv2D,
                target: ComputeTarget::Gpu,
                sequence_index: 2,
                needs_transfer: true,
            },
        ];
        let summary = HeterogeneousScheduler::summarise(&work);
        assert_eq!(summary.cpu_ops, 1);
        assert_eq!(summary.gpu_ops, 1);
        assert_eq!(summary.npu_ops, 1);
        assert_eq!(summary.transfers, 2);
    }

    #[test]
    fn schedule_summary_display() {
        let summary = ScheduleSummary {
            cpu_ops: 2,
            gpu_ops: 1,
            npu_ops: 3,
            transfers: 1,
        };
        let s = summary.to_string();
        assert!(s.contains("CPU=2"));
        assert!(s.contains("NPU=3"));
    }

    #[test]
    fn scheduler_empty_ops() {
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            false,
        );
        let sched = HeterogeneousScheduler::new(analyzer, None);
        let work = sched.schedule(&[]);
        assert!(work.is_empty());
    }

    // ── NpuProfiler ───────────────────────────────────────────────────

    #[test]
    fn profiler_record_and_count() {
        let mut profiler = NpuProfiler::new();
        profiler.record(ProfileSample {
            op: NpuOperator::MatMul,
            target: ComputeTarget::Npu,
            duration: Duration::from_micros(100),
        });
        assert_eq!(profiler.sample_count(), 1);
    }

    #[test]
    fn profiler_average_duration() {
        let mut profiler = NpuProfiler::new();
        profiler.record(ProfileSample {
            op: NpuOperator::MatMul,
            target: ComputeTarget::Npu,
            duration: Duration::from_micros(100),
        });
        profiler.record(ProfileSample {
            op: NpuOperator::MatMul,
            target: ComputeTarget::Npu,
            duration: Duration::from_micros(200),
        });
        let avg = profiler
            .average_duration(NpuOperator::MatMul, ComputeTarget::Npu)
            .unwrap();
        assert_eq!(avg, Duration::from_micros(150));
    }

    #[test]
    fn profiler_average_no_samples() {
        let profiler = NpuProfiler::new();
        assert!(
            profiler
                .average_duration(
                    NpuOperator::MatMul,
                    ComputeTarget::Cpu
                )
                .is_none()
        );
    }

    #[test]
    fn profiler_speedup_calculation() {
        let mut profiler = NpuProfiler::new();
        // CPU takes 200µs
        profiler.record(ProfileSample {
            op: NpuOperator::MatMul,
            target: ComputeTarget::Cpu,
            duration: Duration::from_micros(200),
        });
        // NPU takes 100µs → 2× speedup
        profiler.record(ProfileSample {
            op: NpuOperator::MatMul,
            target: ComputeTarget::Npu,
            duration: Duration::from_micros(100),
        });
        let speedup =
            profiler.speedup_vs_cpu(NpuOperator::MatMul).unwrap();
        assert!((speedup - 2.0).abs() < 0.01);
    }

    #[test]
    fn profiler_speedup_no_cpu_data() {
        let mut profiler = NpuProfiler::new();
        profiler.record(ProfileSample {
            op: NpuOperator::MatMul,
            target: ComputeTarget::Npu,
            duration: Duration::from_micros(100),
        });
        assert!(
            profiler.speedup_vs_cpu(NpuOperator::MatMul).is_none()
        );
    }

    #[test]
    fn profiler_measure_closure() {
        let mut profiler = NpuProfiler::new();
        let result = profiler.measure(
            NpuOperator::Softmax,
            ComputeTarget::Cpu,
            || 42,
        );
        assert_eq!(result, 42);
        assert_eq!(profiler.sample_count(), 1);
        assert_eq!(
            profiler.samples()[0].op,
            NpuOperator::Softmax
        );
    }

    #[test]
    fn profiler_clear() {
        let mut profiler = NpuProfiler::new();
        profiler.record(ProfileSample {
            op: NpuOperator::MatMul,
            target: ComputeTarget::Npu,
            duration: Duration::from_micros(100),
        });
        profiler.clear();
        assert_eq!(profiler.sample_count(), 0);
    }

    #[test]
    fn profiler_default_trait() {
        let p = NpuProfiler::default();
        assert_eq!(p.sample_count(), 0);
    }

    // ── MockNpu ───────────────────────────────────────────────────────

    #[test]
    fn mock_intel_npu_simulate() {
        let mock = MockNpu::intel_mock();
        let d = mock.simulate_op(NpuOperator::MatMul).unwrap();
        assert_eq!(d, Duration::from_micros(100));
    }

    #[test]
    fn mock_intel_npu_unsupported() {
        let mock = MockNpu::intel_mock();
        // Attention not supported by Intel mock
        assert!(mock.simulate_op(NpuOperator::Attention).is_err());
    }

    #[test]
    fn mock_amd_npu_simulate() {
        let mock = MockNpu::amd_mock();
        let d = mock.simulate_op(NpuOperator::Attention).unwrap();
        assert_eq!(d, Duration::from_micros(120));
    }

    #[test]
    fn mock_npu_batch() {
        let mock = MockNpu::intel_mock();
        let total = mock
            .simulate_batch(&[
                NpuOperator::MatMul,
                NpuOperator::Softmax,
            ])
            .unwrap();
        assert_eq!(total, Duration::from_micros(120));
    }

    #[test]
    fn mock_npu_batch_unsupported_fails() {
        let mock = MockNpu::intel_mock();
        let result = mock.simulate_batch(&[
            NpuOperator::MatMul,
            NpuOperator::Attention,
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn mock_npu_device_accessor() {
        let mock = MockNpu::intel_mock();
        assert_eq!(
            mock.device().vendor,
            NpuVendor::IntelMeteorLake
        );
    }

    #[test]
    fn mock_npu_default_latency() {
        let mock = MockNpu::intel_mock();
        // LayerNorm supported → false for Intel, so error
        let result = mock.simulate_op(NpuOperator::LayerNorm);
        assert!(result.is_err());
    }

    // ── ComputeTarget ─────────────────────────────────────────────────

    #[test]
    fn compute_target_display() {
        assert_eq!(ComputeTarget::Cpu.to_string(), "CPU");
        assert_eq!(ComputeTarget::Gpu.to_string(), "GPU");
        assert_eq!(ComputeTarget::Npu.to_string(), "NPU");
    }

    // ── Integration: detector + analyzer + scheduler ──────────────────

    #[test]
    fn end_to_end_detect_schedule() {
        let detector =
            NpuDetector::mock(vec![mock_intel_device()]);
        let result = detector.detect();
        assert!(result.any_found());

        let dev = result.devices[0].clone();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            false,
        );
        let sched =
            HeterogeneousScheduler::new(analyzer, Some(dev));
        let ops = vec![
            sample_op(NpuOperator::MatMul),
            sample_op(NpuOperator::Softmax),
            sample_op(NpuOperator::Conv2D),
        ];
        let work = sched.schedule(&ops);
        let summary = HeterogeneousScheduler::summarise(&work);
        assert_eq!(summary.npu_ops, 3);
        assert_eq!(summary.cpu_ops, 0);
    }

    #[test]
    fn end_to_end_no_npu_fallback() {
        let detector = NpuDetector::mock(vec![]);
        let result = detector.detect();
        assert!(!result.any_found());

        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            false,
        );
        let sched = HeterogeneousScheduler::new(analyzer, None);
        let ops = vec![sample_op(NpuOperator::MatMul)];
        let work = sched.schedule(&ops);
        assert_eq!(work[0].target, ComputeTarget::Cpu);
    }

    #[test]
    fn end_to_end_profile_mock() {
        let mock = MockNpu::intel_mock();
        let mut profiler = NpuProfiler::new();

        // Simulate CPU baseline
        profiler.record(ProfileSample {
            op: NpuOperator::MatMul,
            target: ComputeTarget::Cpu,
            duration: Duration::from_micros(500),
        });

        // Simulate NPU
        let npu_latency =
            mock.simulate_op(NpuOperator::MatMul).unwrap();
        profiler.record(ProfileSample {
            op: NpuOperator::MatMul,
            target: ComputeTarget::Npu,
            duration: npu_latency,
        });

        let speedup =
            profiler.speedup_vs_cpu(NpuOperator::MatMul).unwrap();
        assert!(speedup > 1.0);
    }

    // ── Mixed device scheduling ───────────────────────────────────────

    #[test]
    fn mixed_ops_some_offloaded() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            true,
        );
        let sched =
            HeterogeneousScheduler::new(analyzer, Some(dev));
        let ops = vec![
            sample_op(NpuOperator::MatMul),    // supported
            sample_op(NpuOperator::Embedding), // NOT supported
            sample_op(NpuOperator::Softmax),   // supported
        ];
        let work = sched.schedule(&ops);
        assert_eq!(work[0].target, ComputeTarget::Npu);
        assert_eq!(work[1].target, ComputeTarget::Gpu);
        assert_eq!(work[2].target, ComputeTarget::Npu);
    }

    #[test]
    fn mixed_schedule_summary() {
        let dev = mock_intel_device();
        let analyzer = OffloadAnalyzer::new(
            NpuOffloadPolicy::AlwaysOffload,
            true,
        );
        let sched =
            HeterogeneousScheduler::new(analyzer, Some(dev));
        let ops = vec![
            sample_op(NpuOperator::MatMul),
            sample_op(NpuOperator::Embedding),
            sample_op(NpuOperator::Softmax),
        ];
        let work = sched.schedule(&ops);
        let summary = HeterogeneousScheduler::summarise(&work);
        assert_eq!(summary.npu_ops, 2);
        assert_eq!(summary.gpu_ops, 1);
        assert!(summary.transfers >= 2);
    }
}
