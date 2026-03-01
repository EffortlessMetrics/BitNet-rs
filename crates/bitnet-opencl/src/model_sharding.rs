//! Model sharding infrastructure for distributing inference across multiple
//! GPU devices.
//!
//! Provides [`ShardingStrategy`] selection, [`ShardPlanner`] for memory-aware
//! placement, and [`CrossDeviceTransfer`] for activation movement between
//! shards.

use std::fmt;

// ── Strategy ────────────────────────────────────────────────────────────

/// Strategy for distributing a model across multiple devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShardingStrategy {
    /// Each device owns a contiguous range of transformer layers.
    LayerWise,
    /// Individual tensors (e.g., attention heads) are split across devices.
    TensorParallel,
    /// Stages of the pipeline are mapped to devices; micro-batches flow
    /// through.
    PipelineParallel,
}

impl fmt::Display for ShardingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayerWise => write!(f, "LayerWise"),
            Self::TensorParallel => write!(f, "TensorParallel"),
            Self::PipelineParallel => write!(f, "PipelineParallel"),
        }
    }
}

// ── Device descriptor ───────────────────────────────────────────────────

/// Describes a single compute device available for sharding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDescriptor {
    /// Unique device identifier (index or UUID).
    pub id: usize,
    /// Human-readable label (e.g. "CUDA:0", "OpenCL:1").
    pub label: String,
    /// Available memory in bytes.
    pub memory_bytes: u64,
    /// Backend kind for heterogeneous configurations.
    pub backend: DeviceBackend,
}

/// Backend kind tag used for heterogeneous device support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceBackend {
    Cuda,
    OpenCL,
    Cpu,
}

impl fmt::Display for DeviceBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda => write!(f, "CUDA"),
            Self::OpenCL => write!(f, "OpenCL"),
            Self::Cpu => write!(f, "CPU"),
        }
    }
}

// ── Model metadata ──────────────────────────────────────────────────────

/// Lightweight description of a model's architecture used by the planner.
#[derive(Debug, Clone)]
pub struct ModelArchitecture {
    /// Total number of transformer layers.
    pub num_layers: usize,
    /// Estimated memory per layer in bytes.
    pub memory_per_layer_bytes: u64,
    /// Hidden dimension size (used for tensor-parallel splits).
    pub hidden_dim: usize,
    /// Number of attention heads (for tensor-parallel head splits).
    pub num_attention_heads: usize,
    /// Fixed overhead (embeddings, final norm, LM head) in bytes.
    pub fixed_overhead_bytes: u64,
}

impl ModelArchitecture {
    /// Total estimated model memory (fixed overhead + layers).
    #[must_use]
    pub const fn total_memory_bytes(&self) -> u64 {
        self.fixed_overhead_bytes + self.num_layers as u64 * self.memory_per_layer_bytes
    }
}

// ── Shard assignment ────────────────────────────────────────────────────

/// Maps a contiguous range of layers to a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardAssignment {
    pub device_id: usize,
    /// Inclusive start layer index.
    pub start_layer: usize,
    /// Exclusive end layer index.
    pub end_layer: usize,
}

impl ShardAssignment {
    #[must_use]
    pub const fn num_layers(&self) -> usize {
        self.end_layer.saturating_sub(self.start_layer)
    }
}

// ── Model shard ─────────────────────────────────────────────────────────

/// A partition of the model assigned to a specific device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelShard {
    pub assignment: ShardAssignment,
    pub device: DeviceDescriptor,
    /// Estimated memory consumption for this shard in bytes.
    pub estimated_memory_bytes: u64,
}

// ── Cross-device transfer ───────────────────────────────────────────────

/// Describes an activation transfer between two devices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrossDeviceTransfer {
    pub source_device_id: usize,
    pub target_device_id: usize,
    /// Activation tensor size in bytes.
    pub tensor_bytes: u64,
    /// Whether source and target share the same backend.
    pub same_backend: bool,
}

impl CrossDeviceTransfer {
    /// Heuristic transfer cost in microseconds.
    ///
    /// Same-backend transfers are assumed ~2× faster than cross-backend
    /// ones because they can use peer-to-peer DMA.
    #[must_use]
    pub const fn estimated_cost_us(&self) -> u64 {
        // 12 GB/s ≈ 12 bytes/µs for same-backend, 6 bytes/µs cross.
        let bandwidth = if self.same_backend { 12 } else { 6 };
        self.tensor_bytes / bandwidth
    }
}

// ── Sharding plan ───────────────────────────────────────────────────────

/// Complete sharding plan produced by [`ShardPlanner`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardingPlan {
    pub strategy: ShardingStrategy,
    pub shards: Vec<ModelShard>,
    pub transfers: Vec<CrossDeviceTransfer>,
}

impl ShardingPlan {
    /// Total estimated memory across all shards.
    #[must_use]
    pub fn total_memory_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.estimated_memory_bytes).sum()
    }

    /// Number of cross-device transfers in the plan.
    #[must_use]
    pub const fn num_transfers(&self) -> usize {
        self.transfers.len()
    }

    /// Diagnostic summary printed to the log.
    #[allow(clippy::cast_precision_loss)]
    pub fn print_summary(&self) {
        log::info!("Sharding plan: strategy={}", self.strategy);
        for shard in &self.shards {
            log::info!(
                "  Device {} ({}): layers [{}, {}) — {:.1} MB",
                shard.device.id,
                shard.device.label,
                shard.assignment.start_layer,
                shard.assignment.end_layer,
                shard.estimated_memory_bytes as f64 / 1_048_576.0,
            );
        }
        for xfer in &self.transfers {
            log::info!(
                "  Transfer: dev {} → dev {} — {} bytes (cost ~{} µs)",
                xfer.source_device_id,
                xfer.target_device_id,
                xfer.tensor_bytes,
                xfer.estimated_cost_us(),
            );
        }
    }
}

// ── Shard planner ───────────────────────────────────────────────────────

/// Analyzes a [`ModelArchitecture`] and a set of devices to produce an
/// optimal [`ShardingPlan`].
pub struct ShardPlanner {
    devices: Vec<DeviceDescriptor>,
    strategy: ShardingStrategy,
    /// Activation size for cross-device transfer estimation (bytes).
    activation_bytes: u64,
}

impl ShardPlanner {
    #[must_use]
    pub const fn new(devices: Vec<DeviceDescriptor>, strategy: ShardingStrategy) -> Self {
        Self { devices, strategy, activation_bytes: 0 }
    }

    /// Set the activation tensor size used for transfer cost estimation.
    #[must_use]
    pub const fn with_activation_bytes(mut self, bytes: u64) -> Self {
        self.activation_bytes = bytes;
        self
    }

    /// Build a sharding plan for the given model.
    ///
    /// # Errors
    ///
    /// Returns `Err` when no devices are provided or the model cannot fit
    /// into the available memory.
    pub fn plan(&self, model: &ModelArchitecture) -> Result<ShardingPlan, ShardingError> {
        if self.devices.is_empty() {
            return Err(ShardingError::NoDevices);
        }

        match self.strategy {
            ShardingStrategy::LayerWise | ShardingStrategy::PipelineParallel => {
                self.plan_layer_wise(model)
            }
            ShardingStrategy::TensorParallel => self.plan_tensor_parallel(model),
        }
    }

    // ── Layer-wise / pipeline parallel ──────────────────────────────

    fn plan_layer_wise(&self, model: &ModelArchitecture) -> Result<ShardingPlan, ShardingError> {
        let n_devices = self.devices.len();
        let n_layers = model.num_layers;

        // Compute per-device layer count (even distribution, ceil).
        let assignments = even_layer_split(n_layers, n_devices);

        let mut shards = Vec::with_capacity(n_devices);
        let mut cursor = 0usize;

        for (idx, &count) in assignments.iter().enumerate() {
            let device = &self.devices[idx];
            let mem = model.memory_per_layer_bytes * count as u64
                + if idx == 0 { model.fixed_overhead_bytes } else { 0 };

            if mem > device.memory_bytes {
                return Err(ShardingError::InsufficientMemory {
                    device_id: device.id,
                    required: mem,
                    available: device.memory_bytes,
                });
            }

            shards.push(ModelShard {
                assignment: ShardAssignment {
                    device_id: device.id,
                    start_layer: cursor,
                    end_layer: cursor + count,
                },
                device: device.clone(),
                estimated_memory_bytes: mem,
            });
            cursor += count;
        }

        let transfers = self.build_transfers(&shards);
        Ok(ShardingPlan { strategy: self.strategy, shards, transfers })
    }

    // ── Tensor parallel ─────────────────────────────────────────────

    fn plan_tensor_parallel(
        &self,
        model: &ModelArchitecture,
    ) -> Result<ShardingPlan, ShardingError> {
        let n_devices = self.devices.len();

        // Every device holds *all* layers, but each layer's tensors are
        // split across devices.
        let per_device_mem = model.memory_per_layer_bytes * model.num_layers as u64
            / n_devices as u64
            + model.fixed_overhead_bytes;

        let mut shards = Vec::with_capacity(n_devices);
        for device in &self.devices {
            if per_device_mem > device.memory_bytes {
                return Err(ShardingError::InsufficientMemory {
                    device_id: device.id,
                    required: per_device_mem,
                    available: device.memory_bytes,
                });
            }
            shards.push(ModelShard {
                assignment: ShardAssignment {
                    device_id: device.id,
                    start_layer: 0,
                    end_layer: model.num_layers,
                },
                device: device.clone(),
                estimated_memory_bytes: per_device_mem,
            });
        }

        // Tensor-parallel requires an all-reduce after every layer.
        let mut transfers = Vec::new();
        for i in 0..n_devices {
            for j in (i + 1)..n_devices {
                let same_backend = self.devices[i].backend == self.devices[j].backend;
                transfers.push(CrossDeviceTransfer {
                    source_device_id: self.devices[i].id,
                    target_device_id: self.devices[j].id,
                    tensor_bytes: self.activation_bytes,
                    same_backend,
                });
            }
        }

        Ok(ShardingPlan { strategy: self.strategy, shards, transfers })
    }

    // ── Helpers ─────────────────────────────────────────────────────

    fn build_transfers(&self, shards: &[ModelShard]) -> Vec<CrossDeviceTransfer> {
        let mut transfers = Vec::new();
        for pair in shards.windows(2) {
            let src = &pair[0];
            let dst = &pair[1];
            transfers.push(CrossDeviceTransfer {
                source_device_id: src.device.id,
                target_device_id: dst.device.id,
                tensor_bytes: self.activation_bytes,
                same_backend: src.device.backend == dst.device.backend,
            });
        }
        transfers
    }
}

// ── Even split helper ───────────────────────────────────────────────────

/// Split `total` items across `parts` bins as evenly as possible.
/// First bins get one extra item when `total` is not evenly divisible.
fn even_layer_split(total: usize, parts: usize) -> Vec<usize> {
    if parts == 0 {
        return vec![];
    }
    let base = total / parts;
    let remainder = total % parts;
    (0..parts).map(|i| if i < remainder { base + 1 } else { base }).collect()
}

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors produced during shard planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardingError {
    /// No devices were provided to the planner.
    NoDevices,
    /// A device does not have enough memory for its assigned shard.
    InsufficientMemory { device_id: usize, required: u64, available: u64 },
}

impl fmt::Display for ShardingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevices => write!(f, "no devices available for sharding"),
            Self::InsufficientMemory { device_id, required, available } => write!(
                f,
                "device {device_id}: needs {required} bytes \
                 but only {available} available",
            ),
        }
    }
}

impl std::error::Error for ShardingError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_device(id: usize, mem_gb: u64) -> DeviceDescriptor {
        DeviceDescriptor {
            id,
            label: format!("CUDA:{id}"),
            memory_bytes: mem_gb * 1_073_741_824,
            backend: DeviceBackend::Cuda,
        }
    }

    fn ocl_device(id: usize, mem_gb: u64) -> DeviceDescriptor {
        DeviceDescriptor {
            id,
            label: format!("OpenCL:{id}"),
            memory_bytes: mem_gb * 1_073_741_824,
            backend: DeviceBackend::OpenCL,
        }
    }

    fn sample_model(num_layers: usize) -> ModelArchitecture {
        ModelArchitecture {
            num_layers,
            memory_per_layer_bytes: 100 * 1_048_576, // 100 MB
            hidden_dim: 4096,
            num_attention_heads: 32,
            fixed_overhead_bytes: 50 * 1_048_576, // 50 MB
        }
    }

    #[test]
    fn single_device_gets_all_layers() {
        let devices = vec![cuda_device(0, 8)];
        let planner = ShardPlanner::new(devices, ShardingStrategy::LayerWise);
        let plan = planner.plan(&sample_model(10)).unwrap();

        assert_eq!(plan.shards.len(), 1);
        assert_eq!(plan.shards[0].assignment.start_layer, 0);
        assert_eq!(plan.shards[0].assignment.end_layer, 10);
        assert!(plan.transfers.is_empty());
    }

    #[test]
    fn two_devices_even_split() {
        let devices = vec![cuda_device(0, 8), cuda_device(1, 8)];
        let planner = ShardPlanner::new(devices, ShardingStrategy::LayerWise);
        let plan = planner.plan(&sample_model(10)).unwrap();

        assert_eq!(plan.shards.len(), 2);
        assert_eq!(plan.shards[0].assignment.num_layers(), 5);
        assert_eq!(plan.shards[1].assignment.num_layers(), 5);
        assert_eq!(plan.shards[0].assignment.end_layer, 5);
        assert_eq!(plan.shards[1].assignment.start_layer, 5);
    }

    #[test]
    fn three_devices_uneven_layers() {
        let devices = vec![cuda_device(0, 8), cuda_device(1, 8), cuda_device(2, 8)];
        let planner = ShardPlanner::new(devices, ShardingStrategy::LayerWise);
        let plan = planner.plan(&sample_model(10)).unwrap();

        // 10 / 3 = 3 remainder 1 → first device gets 4, rest get 3.
        assert_eq!(plan.shards[0].assignment.num_layers(), 4);
        assert_eq!(plan.shards[1].assignment.num_layers(), 3);
        assert_eq!(plan.shards[2].assignment.num_layers(), 3);
    }

    #[test]
    fn no_devices_returns_error() {
        let planner = ShardPlanner::new(vec![], ShardingStrategy::LayerWise);
        assert_eq!(planner.plan(&sample_model(10)), Err(ShardingError::NoDevices),);
    }

    #[test]
    fn insufficient_memory_error() {
        // 1 GB device cannot hold 10 × 100 MB layers + 50 MB overhead.
        let devices = vec![cuda_device(0, 1)];
        let planner = ShardPlanner::new(devices, ShardingStrategy::LayerWise);
        let result = planner.plan(&sample_model(10));
        assert!(matches!(result, Err(ShardingError::InsufficientMemory { .. })));
    }

    #[test]
    fn cross_device_transfer_same_backend() {
        let xfer = CrossDeviceTransfer {
            source_device_id: 0,
            target_device_id: 1,
            tensor_bytes: 12_000,
            same_backend: true,
        };
        assert_eq!(xfer.estimated_cost_us(), 1_000);
    }

    #[test]
    fn cross_device_transfer_different_backend() {
        let xfer = CrossDeviceTransfer {
            source_device_id: 0,
            target_device_id: 1,
            tensor_bytes: 12_000,
            same_backend: false,
        };
        assert_eq!(xfer.estimated_cost_us(), 2_000);
    }

    #[test]
    fn heterogeneous_devices() {
        let devices = vec![cuda_device(0, 8), ocl_device(1, 8)];
        let planner = ShardPlanner::new(devices, ShardingStrategy::LayerWise)
            .with_activation_bytes(1_048_576);
        let plan = planner.plan(&sample_model(10)).unwrap();

        assert_eq!(plan.shards.len(), 2);
        assert!(!plan.transfers[0].same_backend);
    }

    #[test]
    fn tensor_parallel_all_layers_on_all_devices() {
        let devices = vec![cuda_device(0, 8), cuda_device(1, 8)];
        let planner = ShardPlanner::new(devices, ShardingStrategy::TensorParallel)
            .with_activation_bytes(4096);
        let plan = planner.plan(&sample_model(4)).unwrap();

        // Each device sees all 4 layers.
        for shard in &plan.shards {
            assert_eq!(shard.assignment.start_layer, 0);
            assert_eq!(shard.assignment.end_layer, 4);
        }
        // One pair → one transfer entry.
        assert_eq!(plan.num_transfers(), 1);
    }

    #[test]
    fn pipeline_parallel_uses_layer_wise_split() {
        let devices = vec![cuda_device(0, 8), cuda_device(1, 8)];
        let planner = ShardPlanner::new(devices, ShardingStrategy::PipelineParallel);
        let plan = planner.plan(&sample_model(8)).unwrap();

        assert_eq!(plan.strategy, ShardingStrategy::PipelineParallel);
        assert_eq!(plan.shards[0].assignment.num_layers(), 4);
        assert_eq!(plan.shards[1].assignment.num_layers(), 4);
    }

    #[test]
    fn empty_model_zero_layers() {
        let devices = vec![cuda_device(0, 8)];
        let planner = ShardPlanner::new(devices, ShardingStrategy::LayerWise);
        let plan = planner.plan(&sample_model(0)).unwrap();

        assert_eq!(plan.shards[0].assignment.num_layers(), 0);
    }

    #[test]
    fn total_memory_is_sum_of_shards() {
        let devices = vec![cuda_device(0, 8), cuda_device(1, 8)];
        let planner = ShardPlanner::new(devices, ShardingStrategy::LayerWise);
        let plan = planner.plan(&sample_model(10)).unwrap();

        let sum: u64 = plan.shards.iter().map(|s| s.estimated_memory_bytes).sum();
        assert_eq!(plan.total_memory_bytes(), sum);
    }

    #[test]
    fn even_layer_split_exact_division() {
        assert_eq!(even_layer_split(12, 3), vec![4, 4, 4]);
    }

    #[test]
    fn even_layer_split_with_remainder() {
        assert_eq!(even_layer_split(10, 3), vec![4, 3, 3]);
    }

    #[test]
    fn even_layer_split_single_part() {
        assert_eq!(even_layer_split(7, 1), vec![7]);
    }

    #[test]
    fn even_layer_split_zero_parts() {
        assert!(even_layer_split(10, 0).is_empty());
    }

    #[test]
    fn model_architecture_total_memory() {
        let m = sample_model(10);
        let expected = 50 * 1_048_576 + 10u64 * 100 * 1_048_576;
        assert_eq!(m.total_memory_bytes(), expected);
    }

    #[test]
    fn sharding_strategy_display() {
        assert_eq!(ShardingStrategy::LayerWise.to_string(), "LayerWise");
        assert_eq!(ShardingStrategy::TensorParallel.to_string(), "TensorParallel",);
        assert_eq!(ShardingStrategy::PipelineParallel.to_string(), "PipelineParallel",);
    }

    #[test]
    fn device_backend_display() {
        assert_eq!(DeviceBackend::Cuda.to_string(), "CUDA");
        assert_eq!(DeviceBackend::OpenCL.to_string(), "OpenCL");
        assert_eq!(DeviceBackend::Cpu.to_string(), "CPU");
    }

    #[test]
    fn sharding_error_display() {
        let e = ShardingError::NoDevices;
        assert_eq!(e.to_string(), "no devices available for sharding");

        let e = ShardingError::InsufficientMemory { device_id: 2, required: 1000, available: 500 };
        assert!(e.to_string().contains("device 2"));
    }
}
