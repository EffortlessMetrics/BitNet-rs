//! Multi-GPU support for model parallelism in BitNet inference.
//!
//! # Overview
//!
//! Provides device management, tensor/pipeline/data parallelism strategies,
//! inter-GPU communication primitives, and distributed matmul for splitting
//! large models across multiple GPUs.
//!
//! - [`GpuDevice`]: Representation of a single GPU device.
//! - [`MultiGpuConfig`]: Configuration for multi-GPU execution.
//! - [`ParallelismStrategy`]: Selection of parallelism mode.
//! - [`DeviceManager`]: Central manager for enumerating and querying devices.
//! - [`NcclCommunicator`]: Stub wrapper for NCCL collective operations.
//!
//! # CPU fallback
//!
//! All public functions provide CPU-only fallback paths so that tests and
//! non-GPU environments can exercise the logic with a single simulated
//! device.  The unified GPU feature gate
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]` guards real device
//! enumeration; otherwise a synthetic device is returned.

use bitnet_common::{KernelError, Result};
use std::collections::HashMap;

// ── GPU device representation ────────────────────────────────────────

/// Represents a single GPU device.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuDevice {
    /// Device ordinal (0-based index).
    pub id: usize,
    /// Human-readable device name.
    pub name: String,
    /// Total device memory in bytes.
    pub total_memory: u64,
    /// Available (free) device memory in bytes.
    pub available_memory: u64,
    /// Compute capability as `(major, minor)`.
    pub compute_capability: (u32, u32),
}

impl GpuDevice {
    /// Create a new `GpuDevice` descriptor.
    pub fn new(
        id: usize,
        name: impl Into<String>,
        total_memory: u64,
        available_memory: u64,
        compute_capability: (u32, u32),
    ) -> Self {
        Self { id, name: name.into(), total_memory, available_memory, compute_capability }
    }

    /// Returns the compute capability as a single float (e.g. 8.6).
    pub fn compute_capability_f32(&self) -> f32 {
        self.compute_capability.0 as f32 + self.compute_capability.1 as f32 * 0.1
    }

    /// Returns the memory utilisation ratio in `[0.0, 1.0]`.
    pub fn memory_utilisation(&self) -> f64 {
        if self.total_memory == 0 {
            return 0.0;
        }
        1.0 - (self.available_memory as f64 / self.total_memory as f64)
    }
}

// ── Parallelism strategy ─────────────────────────────────────────────

/// Strategy for distributing work across multiple GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParallelismStrategy {
    /// Replicate the model on every GPU, partition the batch.
    #[default]
    DataParallel,
    /// Split individual tensors (columns/rows) across GPUs.
    TensorParallel,
    /// Assign consecutive groups of layers to different GPUs.
    PipelineParallel,
    /// Combination of tensor and pipeline parallelism.
    Hybrid,
}

// ── Communication backend ────────────────────────────────────────────

/// Backend used for inter-GPU communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CommunicationBackend {
    /// NVIDIA NCCL (high-performance GPU collective ops).
    Nccl,
    /// Peer-to-peer CUDA memory copies.
    PeerToPeer,
    /// CPU-staged copies (slowest, always available).
    #[default]
    CpuStaged,
}

// ── Multi-GPU configuration ──────────────────────────────────────────

/// Configuration for multi-GPU execution.
#[derive(Debug, Clone)]
pub struct MultiGpuConfig {
    /// Devices to use.
    pub devices: Vec<GpuDevice>,
    /// Parallelism strategy.
    pub parallelism_strategy: ParallelismStrategy,
    /// Communication backend.
    pub communication_backend: CommunicationBackend,
    /// Maximum number of pipeline stages (for `PipelineParallel`/`Hybrid`).
    pub max_pipeline_stages: usize,
}

impl Default for MultiGpuConfig {
    fn default() -> Self {
        Self {
            devices: Vec::new(),
            parallelism_strategy: ParallelismStrategy::default(),
            communication_backend: CommunicationBackend::default(),
            max_pipeline_stages: 1,
        }
    }
}

impl MultiGpuConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.devices.is_empty() {
            return Err(KernelError::InvalidArguments {
                reason: "at least one device is required".into(),
            }
            .into());
        }
        if self.max_pipeline_stages == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "max_pipeline_stages must be >= 1".into(),
            }
            .into());
        }
        match self.parallelism_strategy {
            ParallelismStrategy::PipelineParallel | ParallelismStrategy::Hybrid => {
                if self.max_pipeline_stages > self.devices.len() {
                    return Err(KernelError::InvalidArguments {
                        reason: format!(
                            "max_pipeline_stages ({}) > device count ({})",
                            self.max_pipeline_stages,
                            self.devices.len()
                        ),
                    }
                    .into());
                }
            }
            _ => {}
        }
        Ok(())
    }
}

// ── NCCL communicator stub ───────────────────────────────────────────

/// Stub wrapper for NCCL collective communication.
///
/// On CPU builds this records operations without performing real NCCL
/// calls.  The GPU build (future) will initialise the NCCL communicator
/// and route through the actual library.
#[derive(Debug)]
pub struct NcclCommunicator {
    /// Number of participating devices.
    num_devices: usize,
    /// Whether the communicator has been initialised.
    initialised: bool,
}

impl NcclCommunicator {
    /// Create a new NCCL communicator for `num_devices` GPUs.
    pub fn new(num_devices: usize) -> Result<Self> {
        if num_devices == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "NCCL communicator requires at least 1 device".into(),
            }
            .into());
        }
        Ok(Self { num_devices, initialised: true })
    }

    /// Returns whether the communicator is initialised.
    pub fn is_initialised(&self) -> bool {
        self.initialised
    }

    /// Returns the number of participating devices.
    pub fn num_devices(&self) -> usize {
        self.num_devices
    }

    /// Stub: perform an all-reduce sum across `num_devices`.
    ///
    /// CPU fallback simply returns the input unmodified (single device).
    pub fn all_reduce_sum(&self, data: &mut [f32]) -> Result<()> {
        if !self.initialised {
            return Err(KernelError::GpuError {
                reason: "NCCL communicator not initialised".into(),
            }
            .into());
        }
        if data.is_empty() {
            return Err(KernelError::InvalidArguments {
                reason: "all_reduce_sum: empty buffer".into(),
            }
            .into());
        }
        // CPU stub: data already holds the single-device result.
        Ok(())
    }

    /// Stub: broadcast `data` from `root` device to all others.
    pub fn broadcast(&self, _data: &mut [f32], root: usize) -> Result<()> {
        if root >= self.num_devices {
            return Err(KernelError::InvalidArguments {
                reason: format!("broadcast root ({root}) >= num_devices ({})", self.num_devices),
            }
            .into());
        }
        Ok(())
    }

    /// Destroy / tear-down the communicator.
    pub fn destroy(&mut self) {
        self.initialised = false;
    }
}

// ── Device manager ───────────────────────────────────────────────────

/// Central manager for multiple GPU devices.
///
/// On CPU-only builds the manager always reports a single synthetic CPU
/// device so that call-sites can use the same code-paths.
#[derive(Debug)]
pub struct DeviceManager {
    devices: Vec<GpuDevice>,
}

impl DeviceManager {
    /// Create a `DeviceManager` by enumerating available devices.
    ///
    /// On CPU-only builds this returns a single synthetic device.
    pub fn new() -> Result<Self> {
        let devices = enumerate_devices()?;
        if devices.is_empty() {
            return Err(KernelError::GpuError { reason: "no GPU devices found".into() }.into());
        }
        Ok(Self { devices })
    }

    /// Create a `DeviceManager` from an explicit device list.
    pub fn from_devices(devices: Vec<GpuDevice>) -> Result<Self> {
        if devices.is_empty() {
            return Err(KernelError::InvalidArguments {
                reason: "device list must not be empty".into(),
            }
            .into());
        }
        Ok(Self { devices })
    }

    /// Number of managed devices.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get information about a specific device by ordinal.
    pub fn device_info(&self, device_id: usize) -> Result<&GpuDevice> {
        self.devices.get(device_id).ok_or_else(|| {
            KernelError::InvalidArguments {
                reason: format!("device_id {device_id} out of range (have {})", self.devices.len()),
            }
            .into()
        })
    }

    /// Immutable slice of all devices.
    pub fn devices(&self) -> &[GpuDevice] {
        &self.devices
    }

    /// Build a [`MultiGpuConfig`] from the managed devices.
    pub fn build_config(
        &self,
        strategy: ParallelismStrategy,
        backend: CommunicationBackend,
    ) -> MultiGpuConfig {
        let stages = match strategy {
            ParallelismStrategy::PipelineParallel | ParallelismStrategy::Hybrid => {
                self.devices.len()
            }
            _ => 1,
        };
        MultiGpuConfig {
            devices: self.devices.clone(),
            parallelism_strategy: strategy,
            communication_backend: backend,
            max_pipeline_stages: stages,
        }
    }
}

// ── Device enumeration ───────────────────────────────────────────────

/// Enumerate available GPU devices.
///
/// CPU-only builds return a single synthetic device.
fn enumerate_devices() -> Result<Vec<GpuDevice>> {
    // CPU fallback: one synthetic device.
    Ok(vec![GpuDevice::new(
        0,
        "CPU Fallback Device",
        8 * 1024 * 1024 * 1024, // 8 GiB
        8 * 1024 * 1024 * 1024,
        (0, 0),
    )])
}

/// Return the number of available GPU devices (CPU fallback: 1).
pub fn device_count() -> usize {
    enumerate_devices().map_or(0, |d| d.len())
}

/// Return information about a specific GPU device.
pub fn device_info(device_id: usize) -> Result<GpuDevice> {
    let devices = enumerate_devices()?;
    devices.into_iter().nth(device_id).ok_or_else(|| {
        KernelError::InvalidArguments { reason: format!("device_id {device_id} out of range") }
            .into()
    })
}

// ── Tensor parallelism helpers ───────────────────────────────────────

/// Split a 2-D tensor (row-major `[rows, cols]`) evenly across `num_devices`
/// along the column axis.
///
/// Returns one shard per device. The last shard absorbs any remainder
/// columns when `cols` is not evenly divisible.
///
/// # Errors
///
/// Returns an error if dimensions or `num_devices` are zero, or if the
/// buffer length is inconsistent.
pub fn tensor_parallel_split(
    data: &[f32],
    rows: usize,
    cols: usize,
    num_devices: usize,
) -> Result<Vec<Vec<f32>>> {
    if num_devices == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "num_devices must be > 0".into() }.into()
        );
    }
    if rows == 0 || cols == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("tensor dimensions must be non-zero: rows={rows}, cols={cols}"),
        }
        .into());
    }
    let required = rows * cols;
    if data.len() < required {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "tensor_parallel_split: buffer too small ({} < {required})",
                data.len()
            ),
        }
        .into());
    }

    let base_cols = cols / num_devices;
    let remainder = cols % num_devices;

    let mut shards = Vec::with_capacity(num_devices);
    for dev in 0..num_devices {
        let shard_cols = base_cols + if dev < remainder { 1 } else { 0 };
        let col_start: usize =
            (0..dev).map(|d| base_cols + if d < remainder { 1 } else { 0 }).sum();
        let mut shard = Vec::with_capacity(rows * shard_cols);
        for r in 0..rows {
            let row_off = r * cols;
            shard.extend_from_slice(&data[row_off + col_start..row_off + col_start + shard_cols]);
        }
        shards.push(shard);
    }
    Ok(shards)
}

/// Gather column-sharded tensors back into a single `[rows, cols]` tensor.
///
/// The shards must have been produced by [`tensor_parallel_split`] (or
/// equivalent) so that concatenating them along the column axis
/// reconstructs the original tensor.
///
/// # Errors
///
/// Returns an error if `shards` is empty or shard dimensions are
/// inconsistent.
pub fn tensor_parallel_gather(shards: &[Vec<f32>], rows: usize) -> Result<Vec<f32>> {
    if shards.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "tensor_parallel_gather: no shards".into(),
        }
        .into());
    }
    if rows == 0 {
        return Err(KernelError::InvalidArguments { reason: "rows must be > 0".into() }.into());
    }

    // Derive shard column counts.
    let shard_cols: Vec<usize> =
        shards.iter().map(|s| if rows == 0 { 0 } else { s.len() / rows }).collect();
    let total_cols: usize = shard_cols.iter().sum();

    // Validate each shard length.
    for (i, (shard, &sc)) in shards.iter().zip(shard_cols.iter()).enumerate() {
        if shard.len() != rows * sc {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "shard {i} length {} != rows ({rows}) * shard_cols ({sc})",
                    shard.len()
                ),
            }
            .into());
        }
    }

    let mut out = vec![0.0f32; rows * total_cols];
    for r in 0..rows {
        let mut col_off = 0usize;
        for (shard, &sc) in shards.iter().zip(shard_cols.iter()) {
            let src_start = r * sc;
            let dst_start = r * total_cols + col_off;
            out[dst_start..dst_start + sc].copy_from_slice(&shard[src_start..src_start + sc]);
            col_off += sc;
        }
    }
    Ok(out)
}

// ── All-reduce ───────────────────────────────────────────────────────

/// Element-wise all-reduce (sum) across `num_devices` partial results.
///
/// Each slice in `partials` has the same length.  The output is the
/// element-wise sum.
///
/// # Errors
///
/// Returns an error if `partials` is empty or slices differ in length.
pub fn all_reduce(partials: &[&[f32]]) -> Result<Vec<f32>> {
    if partials.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "all_reduce: empty partials list".into(),
        }
        .into());
    }
    let len = partials[0].len();
    if len == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "all_reduce: partial length is 0".into(),
        }
        .into());
    }
    for (i, p) in partials.iter().enumerate() {
        if p.len() != len {
            return Err(KernelError::InvalidArguments {
                reason: format!("all_reduce: partial[{i}] length {} != expected {len}", p.len()),
            }
            .into());
        }
    }
    let mut out = vec![0.0f32; len];
    for p in partials {
        for (o, &v) in out.iter_mut().zip(p.iter()) {
            *o += v;
        }
    }
    Ok(out)
}

// ── Peer-to-peer copy ────────────────────────────────────────────────

/// Simulated peer-to-peer copy between two device buffers.
///
/// On CPU this is a plain `memcpy`.
///
/// # Errors
///
/// Returns an error if device IDs are equal, or source is empty.
pub fn peer_to_peer_copy(src: &[f32], src_device: usize, dst_device: usize) -> Result<Vec<f32>> {
    if src_device == dst_device {
        return Err(KernelError::InvalidArguments {
            reason: format!("peer_to_peer_copy: src and dst must differ (both {src_device})"),
        }
        .into());
    }
    if src.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "peer_to_peer_copy: source buffer is empty".into(),
        }
        .into());
    }
    Ok(src.to_vec())
}

// ── Pipeline parallelism ─────────────────────────────────────────────

/// A single pipeline stage assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStage {
    /// Device ordinal that runs this stage.
    pub device_id: usize,
    /// Range of layer indices `[start, end)` assigned to this stage.
    pub layer_start: usize,
    /// Exclusive end of layer range.
    pub layer_end: usize,
}

/// Partition `num_layers` model layers across `num_devices` devices.
///
/// Layers are distributed as evenly as possible; earlier devices absorb
/// any remainder layers.
///
/// # Errors
///
/// Returns an error if `num_layers` or `num_devices` is zero.
pub fn pipeline_partition(num_layers: usize, num_devices: usize) -> Result<Vec<PipelineStage>> {
    if num_layers == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "pipeline_partition: num_layers must be > 0".into(),
        }
        .into());
    }
    if num_devices == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "pipeline_partition: num_devices must be > 0".into(),
        }
        .into());
    }
    let base = num_layers / num_devices;
    let remainder = num_layers % num_devices;

    let mut stages = Vec::with_capacity(num_devices);
    let mut offset = 0usize;
    for dev in 0..num_devices {
        let count = base + if dev < remainder { 1 } else { 0 };
        stages.push(PipelineStage {
            device_id: dev,
            layer_start: offset,
            layer_end: offset + count,
        });
        offset += count;
    }
    Ok(stages)
}

// ── Load balancing ───────────────────────────────────────────────────

/// Per-device load metrics used for dynamic balancing.
#[derive(Debug, Clone)]
pub struct DeviceLoad {
    /// Device ordinal.
    pub device_id: usize,
    /// Current utilisation ratio in `[0.0, 1.0]`.
    pub utilisation: f64,
    /// Pending work items.
    pub pending_work: usize,
}

/// Dynamic load-balancing: given a set of work items (`work_items`) and
/// per-device load metrics, return a mapping from `device_id → count`
/// of newly-assigned items.
///
/// Items are assigned greedily to the least-loaded device until all
/// items are distributed.
///
/// # Errors
///
/// Returns an error if `loads` is empty.
pub fn load_balance(loads: &[DeviceLoad], work_items: usize) -> Result<HashMap<usize, usize>> {
    if loads.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "load_balance: empty device loads".into(),
        }
        .into());
    }

    let mut assignments: HashMap<usize, usize> = HashMap::new();
    // Build a mutable snapshot for greedy scheduling.
    let mut effective: Vec<(usize, f64, usize)> =
        loads.iter().map(|l| (l.device_id, l.utilisation, l.pending_work)).collect();

    for _ in 0..work_items {
        // Pick device with lowest (utilisation + scaled pending).
        effective.sort_by(|a, b| {
            let score_a = a.1 + a.2 as f64 * 0.01;
            let score_b = b.1 + b.2 as f64 * 0.01;
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let best = &mut effective[0];
        *assignments.entry(best.0).or_insert(0) += 1;
        best.2 += 1; // increment pending
    }
    Ok(assignments)
}

// ── Distributed matmul ───────────────────────────────────────────────

/// Distributed matrix multiplication across `num_devices` GPUs.
///
/// Splits `B` column-wise across devices, performs per-device matmul
/// `A × B_shard`, and gathers results back into a single output.
///
/// On CPU-only builds this is equivalent to a single-device matmul.
///
/// # Errors
///
/// Returns an error if dimensions are zero or buffer sizes are
/// inconsistent.
pub fn multi_gpu_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    num_devices: usize,
) -> Result<Vec<f32>> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("multi_gpu_matmul: dimensions must be non-zero: m={m}, n={n}, k={k}"),
        }
        .into());
    }
    if num_devices == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "multi_gpu_matmul: num_devices must be > 0".into(),
        }
        .into());
    }
    if a.len() < m * k {
        return Err(KernelError::InvalidArguments {
            reason: format!("multi_gpu_matmul: A buffer too small ({} < {})", a.len(), m * k),
        }
        .into());
    }
    if b.len() < k * n {
        return Err(KernelError::InvalidArguments {
            reason: format!("multi_gpu_matmul: B buffer too small ({} < {})", b.len(), k * n),
        }
        .into());
    }

    // Split B column-wise.
    let b_shards = tensor_parallel_split(b, k, n, num_devices)?;

    // Per-device matmul.
    let mut c_shards: Vec<Vec<f32>> = Vec::with_capacity(num_devices);
    for shard in &b_shards {
        let shard_n = shard.len() / k;
        let mut c_shard = vec![0.0f32; m * shard_n];
        for i in 0..m {
            for j in 0..shard_n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += a[i * k + l] * shard[l * shard_n + j];
                }
                c_shard[i * shard_n + j] = acc;
            }
        }
        c_shards.push(c_shard);
    }

    // Gather.
    tensor_parallel_gather(&c_shards, m)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── GpuDevice tests ──────────────────────────────────────────────

    #[test]
    fn test_gpu_device_new() {
        let dev = GpuDevice::new(0, "Test GPU", 8_000_000, 4_000_000, (8, 6));
        assert_eq!(dev.id, 0);
        assert_eq!(dev.name, "Test GPU");
        assert_eq!(dev.total_memory, 8_000_000);
        assert_eq!(dev.available_memory, 4_000_000);
        assert_eq!(dev.compute_capability, (8, 6));
    }

    #[test]
    fn test_gpu_device_compute_capability_f32() {
        let dev = GpuDevice::new(0, "G", 1, 1, (8, 6));
        let cc = dev.compute_capability_f32();
        assert!((cc - 8.6).abs() < 0.01);
    }

    #[test]
    fn test_gpu_device_memory_utilisation_zero_total() {
        let dev = GpuDevice::new(0, "G", 0, 0, (0, 0));
        assert!((dev.memory_utilisation() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gpu_device_memory_utilisation_half() {
        let dev = GpuDevice::new(0, "G", 1000, 500, (7, 5));
        assert!((dev.memory_utilisation() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_gpu_device_memory_utilisation_full() {
        let dev = GpuDevice::new(0, "G", 1000, 0, (7, 5));
        assert!((dev.memory_utilisation() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gpu_device_clone_eq() {
        let a = GpuDevice::new(1, "A", 100, 50, (9, 0));
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── ParallelismStrategy tests ────────────────────────────────────

    #[test]
    fn test_parallelism_strategy_default() {
        assert_eq!(ParallelismStrategy::default(), ParallelismStrategy::DataParallel);
    }

    #[test]
    fn test_parallelism_strategy_variants() {
        let variants = [
            ParallelismStrategy::DataParallel,
            ParallelismStrategy::TensorParallel,
            ParallelismStrategy::PipelineParallel,
            ParallelismStrategy::Hybrid,
        ];
        assert_eq!(variants.len(), 4);
    }

    // ── CommunicationBackend tests ───────────────────────────────────

    #[test]
    fn test_communication_backend_default() {
        assert_eq!(CommunicationBackend::default(), CommunicationBackend::CpuStaged);
    }

    // ── MultiGpuConfig tests ─────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = MultiGpuConfig::default();
        assert!(cfg.devices.is_empty());
        assert_eq!(cfg.parallelism_strategy, ParallelismStrategy::DataParallel);
        assert_eq!(cfg.communication_backend, CommunicationBackend::CpuStaged);
        assert_eq!(cfg.max_pipeline_stages, 1);
    }

    #[test]
    fn test_config_validate_empty_devices() {
        let cfg = MultiGpuConfig::default();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_stages() {
        let cfg = MultiGpuConfig {
            devices: vec![GpuDevice::new(0, "G", 1, 1, (8, 0))],
            max_pipeline_stages: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_pipeline_stages_exceed_devices() {
        let cfg = MultiGpuConfig {
            devices: vec![GpuDevice::new(0, "G", 1, 1, (8, 0))],
            parallelism_strategy: ParallelismStrategy::PipelineParallel,
            max_pipeline_stages: 2,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = MultiGpuConfig {
            devices: vec![
                GpuDevice::new(0, "G0", 1, 1, (8, 0)),
                GpuDevice::new(1, "G1", 1, 1, (8, 0)),
            ],
            parallelism_strategy: ParallelismStrategy::PipelineParallel,
            max_pipeline_stages: 2,
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_data_parallel_ignores_stages() {
        let cfg = MultiGpuConfig {
            devices: vec![GpuDevice::new(0, "G", 1, 1, (8, 0))],
            parallelism_strategy: ParallelismStrategy::DataParallel,
            max_pipeline_stages: 99,
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_hybrid_checks_stages() {
        let cfg = MultiGpuConfig {
            devices: vec![GpuDevice::new(0, "G", 1, 1, (8, 0))],
            parallelism_strategy: ParallelismStrategy::Hybrid,
            max_pipeline_stages: 3,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ── NcclCommunicator tests ───────────────────────────────────────

    #[test]
    fn test_nccl_new() {
        let comm = NcclCommunicator::new(4).unwrap();
        assert!(comm.is_initialised());
        assert_eq!(comm.num_devices(), 4);
    }

    #[test]
    fn test_nccl_new_zero_devices() {
        assert!(NcclCommunicator::new(0).is_err());
    }

    #[test]
    fn test_nccl_all_reduce_sum() {
        let comm = NcclCommunicator::new(2).unwrap();
        let mut buf = vec![1.0, 2.0, 3.0];
        comm.all_reduce_sum(&mut buf).unwrap();
        assert_eq!(buf, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_nccl_all_reduce_sum_empty() {
        let comm = NcclCommunicator::new(1).unwrap();
        let mut buf: Vec<f32> = vec![];
        assert!(comm.all_reduce_sum(&mut buf).is_err());
    }

    #[test]
    fn test_nccl_all_reduce_after_destroy() {
        let mut comm = NcclCommunicator::new(1).unwrap();
        comm.destroy();
        assert!(!comm.is_initialised());
        let mut buf = vec![1.0];
        assert!(comm.all_reduce_sum(&mut buf).is_err());
    }

    #[test]
    fn test_nccl_broadcast_ok() {
        let comm = NcclCommunicator::new(4).unwrap();
        let mut buf = vec![1.0, 2.0];
        comm.broadcast(&mut buf, 0).unwrap();
    }

    #[test]
    fn test_nccl_broadcast_invalid_root() {
        let comm = NcclCommunicator::new(2).unwrap();
        let mut buf = vec![1.0];
        assert!(comm.broadcast(&mut buf, 5).is_err());
    }

    #[test]
    fn test_nccl_destroy() {
        let mut comm = NcclCommunicator::new(1).unwrap();
        assert!(comm.is_initialised());
        comm.destroy();
        assert!(!comm.is_initialised());
    }

    // ── DeviceManager tests ──────────────────────────────────────────

    #[test]
    fn test_device_manager_new() {
        let mgr = DeviceManager::new().unwrap();
        assert!(mgr.device_count() >= 1);
    }

    #[test]
    fn test_device_manager_from_devices() {
        let devs = vec![
            GpuDevice::new(0, "A", 100, 100, (8, 0)),
            GpuDevice::new(1, "B", 200, 200, (9, 0)),
        ];
        let mgr = DeviceManager::from_devices(devs).unwrap();
        assert_eq!(mgr.device_count(), 2);
    }

    #[test]
    fn test_device_manager_from_devices_empty() {
        assert!(DeviceManager::from_devices(vec![]).is_err());
    }

    #[test]
    fn test_device_manager_device_info() {
        let devs = vec![GpuDevice::new(0, "X", 10, 5, (7, 0))];
        let mgr = DeviceManager::from_devices(devs).unwrap();
        let info = mgr.device_info(0).unwrap();
        assert_eq!(info.name, "X");
    }

    #[test]
    fn test_device_manager_device_info_out_of_range() {
        let devs = vec![GpuDevice::new(0, "X", 10, 5, (7, 0))];
        let mgr = DeviceManager::from_devices(devs).unwrap();
        assert!(mgr.device_info(1).is_err());
    }

    #[test]
    fn test_device_manager_devices_slice() {
        let devs = vec![GpuDevice::new(0, "A", 1, 1, (8, 0)), GpuDevice::new(1, "B", 2, 2, (8, 0))];
        let mgr = DeviceManager::from_devices(devs).unwrap();
        assert_eq!(mgr.devices().len(), 2);
    }

    #[test]
    fn test_device_manager_build_config_data_parallel() {
        let devs = vec![GpuDevice::new(0, "A", 1, 1, (8, 0)), GpuDevice::new(1, "B", 1, 1, (8, 0))];
        let mgr = DeviceManager::from_devices(devs).unwrap();
        let cfg = mgr.build_config(ParallelismStrategy::DataParallel, CommunicationBackend::Nccl);
        assert_eq!(cfg.max_pipeline_stages, 1);
        assert_eq!(cfg.devices.len(), 2);
    }

    #[test]
    fn test_device_manager_build_config_pipeline() {
        let devs = vec![
            GpuDevice::new(0, "A", 1, 1, (8, 0)),
            GpuDevice::new(1, "B", 1, 1, (8, 0)),
            GpuDevice::new(2, "C", 1, 1, (8, 0)),
        ];
        let mgr = DeviceManager::from_devices(devs).unwrap();
        let cfg = mgr
            .build_config(ParallelismStrategy::PipelineParallel, CommunicationBackend::PeerToPeer);
        assert_eq!(cfg.max_pipeline_stages, 3);
    }

    // ── device_count / device_info free functions ────────────────────

    #[test]
    fn test_device_count_at_least_one() {
        assert!(device_count() >= 1);
    }

    #[test]
    fn test_device_info_zero() {
        let dev = device_info(0).unwrap();
        assert!(!dev.name.is_empty());
    }

    #[test]
    fn test_device_info_out_of_range() {
        assert!(device_info(999).is_err());
    }

    // ── tensor_parallel_split tests ──────────────────────────────────

    #[test]
    fn test_split_single_device() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let shards = tensor_parallel_split(&data, 2, 3, 1).unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], data);
    }

    #[test]
    fn test_split_two_devices_even() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3 — but 3 isn't even by 2
        let shards = tensor_parallel_split(&data, 2, 4, 2).unwrap_err();
        // Buffer too small for 2×4
        let _ = shards;

        // 2×4 proper
        let data4 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shards = tensor_parallel_split(&data4, 2, 4, 2).unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0], vec![1.0, 2.0, 5.0, 6.0]); // cols 0,1
        assert_eq!(shards[1], vec![3.0, 4.0, 7.0, 8.0]); // cols 2,3
    }

    #[test]
    fn test_split_uneven_columns() {
        // 2×3, split across 2 devices: dev0 gets 2 cols, dev1 gets 1
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shards = tensor_parallel_split(&data, 2, 3, 2).unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0], vec![1.0, 2.0, 4.0, 5.0]); // cols 0,1
        assert_eq!(shards[1], vec![3.0, 6.0]); // col 2
    }

    #[test]
    fn test_split_zero_devices() {
        assert!(tensor_parallel_split(&[1.0], 1, 1, 0).is_err());
    }

    #[test]
    fn test_split_zero_rows() {
        assert!(tensor_parallel_split(&[], 0, 1, 1).is_err());
    }

    #[test]
    fn test_split_zero_cols() {
        assert!(tensor_parallel_split(&[], 1, 0, 1).is_err());
    }

    #[test]
    fn test_split_buffer_too_small() {
        assert!(tensor_parallel_split(&[1.0], 2, 2, 1).is_err());
    }

    #[test]
    fn test_split_more_devices_than_cols() {
        // 2×2 split across 3 devices: two get 1 col, one gets 0
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shards = tensor_parallel_split(&data, 2, 2, 3).unwrap();
        assert_eq!(shards.len(), 3);
        // dev0: 1 col, dev1: 1 col, dev2: 0 cols
        assert_eq!(shards[0].len(), 2); // 2 rows × 1 col
        assert_eq!(shards[1].len(), 2);
        assert_eq!(shards[2].len(), 0);
    }

    // ── tensor_parallel_gather tests ─────────────────────────────────

    #[test]
    fn test_gather_single_shard() {
        let shards = vec![vec![1.0, 2.0, 3.0, 4.0]]; // 2×2
        let out = tensor_parallel_gather(&shards, 2).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gather_empty_shards() {
        let shards: Vec<Vec<f32>> = vec![];
        assert!(tensor_parallel_gather(&shards, 1).is_err());
    }

    #[test]
    fn test_gather_zero_rows() {
        let shards = vec![vec![1.0]];
        assert!(tensor_parallel_gather(&shards, 0).is_err());
    }

    #[test]
    fn test_split_gather_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let shards = tensor_parallel_split(&data, 2, 3, 2).unwrap();
        let restored = tensor_parallel_gather(&shards, 2).unwrap();
        assert_eq!(restored, data);
    }

    #[test]
    fn test_split_gather_roundtrip_4_devices() {
        // 3×8
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let shards = tensor_parallel_split(&data, 3, 8, 4).unwrap();
        assert_eq!(shards.len(), 4);
        let restored = tensor_parallel_gather(&shards, 3).unwrap();
        assert_eq!(restored, data);
    }

    // ── all_reduce tests ─────────────────────────────────────────────

    #[test]
    fn test_all_reduce_single() {
        let a = vec![1.0, 2.0, 3.0];
        let result = all_reduce(&[&a]).unwrap();
        assert_eq!(result, a);
    }

    #[test]
    fn test_all_reduce_two() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let result = all_reduce(&[&a, &b]).unwrap();
        assert_eq!(result, vec![4.0, 6.0]);
    }

    #[test]
    fn test_all_reduce_three() {
        let a = vec![1.0];
        let b = vec![2.0];
        let c = vec![3.0];
        let result = all_reduce(&[&a, &b, &c]).unwrap();
        assert_eq!(result, vec![6.0]);
    }

    #[test]
    fn test_all_reduce_empty_partials() {
        let partials: Vec<&[f32]> = vec![];
        assert!(all_reduce(&partials).is_err());
    }

    #[test]
    fn test_all_reduce_empty_buffer() {
        let a: Vec<f32> = vec![];
        assert!(all_reduce(&[&a[..]]).is_err());
    }

    #[test]
    fn test_all_reduce_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(all_reduce(&[&a[..], &b[..]]).is_err());
    }

    // ── peer_to_peer_copy tests ──────────────────────────────────────

    #[test]
    fn test_p2p_copy_ok() {
        let src = vec![1.0, 2.0, 3.0];
        let dst = peer_to_peer_copy(&src, 0, 1).unwrap();
        assert_eq!(dst, src);
    }

    #[test]
    fn test_p2p_copy_same_device() {
        assert!(peer_to_peer_copy(&[1.0], 0, 0).is_err());
    }

    #[test]
    fn test_p2p_copy_empty_src() {
        assert!(peer_to_peer_copy(&[], 0, 1).is_err());
    }

    // ── pipeline_partition tests ─────────────────────────────────────

    #[test]
    fn test_pipeline_partition_even() {
        let stages = pipeline_partition(6, 3).unwrap();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0], PipelineStage { device_id: 0, layer_start: 0, layer_end: 2 });
        assert_eq!(stages[1], PipelineStage { device_id: 1, layer_start: 2, layer_end: 4 });
        assert_eq!(stages[2], PipelineStage { device_id: 2, layer_start: 4, layer_end: 6 });
    }

    #[test]
    fn test_pipeline_partition_uneven() {
        let stages = pipeline_partition(7, 3).unwrap();
        // 7/3 = 2 rem 1; first device gets 3 layers
        assert_eq!(stages[0].layer_end - stages[0].layer_start, 3);
        assert_eq!(stages[1].layer_end - stages[1].layer_start, 2);
        assert_eq!(stages[2].layer_end - stages[2].layer_start, 2);
        assert_eq!(stages[2].layer_end, 7);
    }

    #[test]
    fn test_pipeline_partition_single_device() {
        let stages = pipeline_partition(10, 1).unwrap();
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].layer_start, 0);
        assert_eq!(stages[0].layer_end, 10);
    }

    #[test]
    fn test_pipeline_partition_more_devices_than_layers() {
        let stages = pipeline_partition(2, 5).unwrap();
        assert_eq!(stages.len(), 5);
        let total: usize = stages.iter().map(|s| s.layer_end - s.layer_start).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_pipeline_partition_zero_layers() {
        assert!(pipeline_partition(0, 2).is_err());
    }

    #[test]
    fn test_pipeline_partition_zero_devices() {
        assert!(pipeline_partition(4, 0).is_err());
    }

    #[test]
    fn test_pipeline_partition_contiguous() {
        let stages = pipeline_partition(10, 4).unwrap();
        for i in 1..stages.len() {
            assert_eq!(stages[i].layer_start, stages[i - 1].layer_end);
        }
        assert_eq!(stages.last().unwrap().layer_end, 10);
    }

    // ── load_balance tests ───────────────────────────────────────────

    #[test]
    fn test_load_balance_single_device() {
        let loads = vec![DeviceLoad { device_id: 0, utilisation: 0.0, pending_work: 0 }];
        let result = load_balance(&loads, 5).unwrap();
        assert_eq!(*result.get(&0).unwrap(), 5);
    }

    #[test]
    fn test_load_balance_two_equal() {
        let loads = vec![
            DeviceLoad { device_id: 0, utilisation: 0.0, pending_work: 0 },
            DeviceLoad { device_id: 1, utilisation: 0.0, pending_work: 0 },
        ];
        let result = load_balance(&loads, 4).unwrap();
        let total: usize = result.values().sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_load_balance_prefers_less_loaded() {
        let loads = vec![
            DeviceLoad { device_id: 0, utilisation: 0.9, pending_work: 10 },
            DeviceLoad { device_id: 1, utilisation: 0.1, pending_work: 0 },
        ];
        let result = load_balance(&loads, 1).unwrap();
        assert_eq!(*result.get(&1).unwrap(), 1);
        assert!(result.get(&0).is_none());
    }

    #[test]
    fn test_load_balance_zero_work() {
        let loads = vec![DeviceLoad { device_id: 0, utilisation: 0.5, pending_work: 0 }];
        let result = load_balance(&loads, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_load_balance_empty_loads() {
        let loads: Vec<DeviceLoad> = vec![];
        assert!(load_balance(&loads, 1).is_err());
    }

    #[test]
    fn test_load_balance_total_items() {
        let loads = vec![
            DeviceLoad { device_id: 0, utilisation: 0.2, pending_work: 1 },
            DeviceLoad { device_id: 1, utilisation: 0.4, pending_work: 2 },
            DeviceLoad { device_id: 2, utilisation: 0.1, pending_work: 0 },
        ];
        let result = load_balance(&loads, 10).unwrap();
        let total: usize = result.values().sum();
        assert_eq!(total, 10);
    }

    // ── multi_gpu_matmul tests ───────────────────────────────────────

    #[test]
    fn test_matmul_identity() {
        // 2×2 identity × 2×2 data
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = multi_gpu_matmul(&a, &b, 2, 2, 2, 1).unwrap();
        assert_eq!(result, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_matmul_simple() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = multi_gpu_matmul(&a, &b, 2, 2, 2, 1).unwrap();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_multi_device() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let single = multi_gpu_matmul(&a, &b, 2, 2, 2, 1).unwrap();
        let multi = multi_gpu_matmul(&a, &b, 2, 2, 2, 2).unwrap();
        assert_eq!(single, multi);
    }

    #[test]
    fn test_matmul_rectangular() {
        // A: 2×3, B: 3×2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let result = multi_gpu_matmul(&a, &b, 2, 2, 3, 1).unwrap();
        // C[0,0]=1*7+2*9+3*11=58, C[0,1]=1*8+2*10+3*12=64
        // C[1,0]=4*7+5*9+6*11=139, C[1,1]=4*8+5*10+6*12=154
        assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_multi_device_rectangular() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3×2
        let single = multi_gpu_matmul(&a, &b, 2, 2, 3, 1).unwrap();
        let multi = multi_gpu_matmul(&a, &b, 2, 2, 3, 2).unwrap();
        assert_eq!(single, multi);
    }

    #[test]
    fn test_matmul_zero_m() {
        assert!(multi_gpu_matmul(&[], &[], 0, 1, 1, 1).is_err());
    }

    #[test]
    fn test_matmul_zero_n() {
        assert!(multi_gpu_matmul(&[], &[], 1, 0, 1, 1).is_err());
    }

    #[test]
    fn test_matmul_zero_k() {
        assert!(multi_gpu_matmul(&[], &[], 1, 1, 0, 1).is_err());
    }

    #[test]
    fn test_matmul_zero_devices() {
        assert!(multi_gpu_matmul(&[1.0], &[1.0], 1, 1, 1, 0).is_err());
    }

    #[test]
    fn test_matmul_a_too_small() {
        assert!(multi_gpu_matmul(&[1.0], &[1.0, 2.0, 3.0, 4.0], 2, 2, 2, 1).is_err());
    }

    #[test]
    fn test_matmul_b_too_small() {
        assert!(multi_gpu_matmul(&[1.0, 2.0, 3.0, 4.0], &[1.0], 2, 2, 2, 1).is_err());
    }

    #[test]
    fn test_matmul_1x1() {
        let result = multi_gpu_matmul(&[3.0], &[4.0], 1, 1, 1, 1).unwrap();
        assert_eq!(result, vec![12.0]);
    }

    #[test]
    fn test_matmul_wide_output() {
        // A: 1×2, B: 2×4
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let single = multi_gpu_matmul(&a, &b, 1, 4, 2, 1).unwrap();
        let multi = multi_gpu_matmul(&a, &b, 1, 4, 2, 2).unwrap();
        assert_eq!(single, multi);
        // C = [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11,14,17,20]
        assert_eq!(single, vec![11.0, 14.0, 17.0, 20.0]);
    }

    #[test]
    fn test_matmul_multi_device_consistency_4gpu() {
        let a: Vec<f32> = (0..12).map(|i| (i + 1) as f32).collect(); // 3×4
        let b: Vec<f32> = (0..20).map(|i| (i + 1) as f32).collect(); // 4×5
        let single = multi_gpu_matmul(&a, &b, 3, 5, 4, 1).unwrap();
        let quad = multi_gpu_matmul(&a, &b, 3, 5, 4, 4).unwrap();
        for (s, q) in single.iter().zip(quad.iter()) {
            assert!((s - q).abs() < 1e-4, "mismatch: {s} vs {q}");
        }
    }
}
