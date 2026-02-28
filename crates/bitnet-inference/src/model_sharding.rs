//! Model weight sharding across multiple GPU devices.
//!
//! When inference runs on multiple GPUs (e.g. 2× Intel Arc), the model
//! weights must be distributed across devices. This module provides:
//!
//! - [`ShardingStrategy`] — how a single tensor is split across devices.
//! - [`LayerAssignment`] — which device owns which layers.
//! - [`ShardingPlan`] — a full plan describing the distribution.
//! - Cross-device gather / scatter descriptors for sharded matmul.
//!
//! # Strategies
//!
//! | Strategy        | Split axis | Communication         |
//! |-----------------|------------|-----------------------|
//! | ColumnParallel  | columns    | AllGather after matmul|
//! | RowParallel     | rows       | ReduceScatter         |
//! | Pipeline        | layers     | Point-to-point        |

use std::fmt;

// ---------------------------------------------------------------------------
// Sharding strategy
// ---------------------------------------------------------------------------

/// How a tensor is partitioned across devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShardingStrategy {
    /// Split along the column (output) dimension.
    /// Each device computes a slice of the output; results are
    /// gathered (concatenated) after the matmul.
    ColumnParallel,
    /// Split along the row (input/reduction) dimension.
    /// Each device computes a partial sum; results are reduced
    /// (summed) across devices after the matmul.
    RowParallel,
    /// Full layers are assigned to different devices (pipeline
    /// parallelism). No tensor splitting within a layer.
    Pipeline,
}

impl fmt::Display for ShardingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColumnParallel => write!(f, "ColumnParallel"),
            Self::RowParallel => write!(f, "RowParallel"),
            Self::Pipeline => write!(f, "Pipeline"),
        }
    }
}

// ---------------------------------------------------------------------------
// Sharding configuration
// ---------------------------------------------------------------------------

/// Configuration for sharding a model across `n` devices.
#[derive(Debug, Clone)]
pub struct ShardingConfig {
    /// Number of devices to shard across (2, 4, or 8).
    pub num_devices: usize,
    /// Strategy to apply to attention projection weights.
    pub attention_strategy: ShardingStrategy,
    /// Strategy to apply to FFN / MLP weights.
    pub ffn_strategy: ShardingStrategy,
    /// Per-device memory budget in bytes (used for layer balancing).
    pub per_device_memory: Vec<u64>,
}

impl ShardingConfig {
    /// Create a uniform config where all devices have the same memory budget.
    pub fn uniform(num_devices: usize, memory_per_device: u64) -> Self {
        assert!(
            matches!(num_devices, 2 | 4 | 8),
            "num_devices must be 2, 4, or 8 (got {num_devices})"
        );
        Self {
            num_devices,
            attention_strategy: ShardingStrategy::ColumnParallel,
            ffn_strategy: ShardingStrategy::ColumnParallel,
            per_device_memory: vec![memory_per_device; num_devices],
        }
    }

    /// Create a 2-way config with column-parallel strategy.
    pub fn two_way(memory_per_device: u64) -> Self {
        Self::uniform(2, memory_per_device)
    }

    /// Create a 4-way config.
    pub fn four_way(memory_per_device: u64) -> Self {
        Self::uniform(4, memory_per_device)
    }

    /// Create an 8-way config.
    pub fn eight_way(memory_per_device: u64) -> Self {
        Self::uniform(8, memory_per_device)
    }
}

// ---------------------------------------------------------------------------
// Layer assignment
// ---------------------------------------------------------------------------

/// Describes the assignment of a single model layer to a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerAssignment {
    /// Layer index (0-based).
    pub layer_index: usize,
    /// Device index this layer is assigned to.
    pub device_index: usize,
    /// Estimated memory consumption of this layer (bytes).
    pub memory_bytes: u64,
}

/// Information about a model used for planning.
#[derive(Debug, Clone)]
pub struct ModelTopology {
    /// Total number of transformer layers.
    pub num_layers: usize,
    /// Estimated memory per layer (bytes). Length must equal `num_layers`.
    pub layer_memory: Vec<u64>,
    /// Hidden dimension (used for shard-size calculations).
    pub hidden_dim: usize,
    /// Number of attention heads (must be divisible by num_devices for
    /// column-parallel attention).
    pub num_heads: usize,
}

// ---------------------------------------------------------------------------
// Cross-device communication descriptors
// ---------------------------------------------------------------------------

/// Describes a cross-device data transfer for a sharded matmul.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommOp {
    /// All-gather: each device broadcasts its shard, all devices
    /// end up with the full output.
    AllGather {
        /// Size of each shard in elements.
        shard_elements: usize,
        /// Number of participating devices.
        num_devices: usize,
    },
    /// Reduce-scatter: each device holds a partial sum; after the op
    /// each device holds one reduced shard.
    ReduceScatter {
        shard_elements: usize,
        num_devices: usize,
    },
    /// Point-to-point transfer between two devices (pipeline).
    SendRecv {
        src_device: usize,
        dst_device: usize,
        num_elements: usize,
    },
    /// No communication needed (layer is fully local).
    None,
}

impl fmt::Display for CommOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllGather { shard_elements, num_devices } => {
                write!(f, "AllGather({shard_elements}×{num_devices})")
            }
            Self::ReduceScatter { shard_elements, num_devices } => {
                write!(f, "ReduceScatter({shard_elements}×{num_devices})")
            }
            Self::SendRecv { src_device, dst_device, num_elements } => {
                write!(f, "SendRecv({src_device}→{dst_device}, {num_elements} elems)")
            }
            Self::None => write!(f, "None"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shard descriptor (per-tensor)
// ---------------------------------------------------------------------------

/// Describes how one weight tensor is sharded and what communication
/// is needed after the matmul.
#[derive(Debug, Clone)]
pub struct ShardDescriptor {
    /// Name of the tensor (e.g. `"layers.0.attn.q_proj"`).
    pub tensor_name: String,
    /// Strategy applied to this tensor.
    pub strategy: ShardingStrategy,
    /// Original shape `[rows, cols]`.
    pub original_shape: [usize; 2],
    /// Per-device shard shape `[rows, cols]`.
    pub shard_shape: [usize; 2],
    /// Communication operation after the matmul.
    pub comm_op: CommOp,
}

// ---------------------------------------------------------------------------
// Sharding plan
// ---------------------------------------------------------------------------

/// A complete sharding plan for a model.
#[derive(Debug, Clone)]
pub struct ShardingPlan {
    /// Per-layer device assignments.
    pub layer_assignments: Vec<LayerAssignment>,
    /// Per-device memory usage summary (bytes).
    pub device_memory_usage: Vec<u64>,
    /// How many devices participate.
    pub num_devices: usize,
    /// Strategy used for attention.
    pub attention_strategy: ShardingStrategy,
    /// Strategy used for FFN.
    pub ffn_strategy: ShardingStrategy,
}

impl ShardingPlan {
    /// Maximum memory usage across devices.
    pub fn peak_device_memory(&self) -> u64 {
        self.device_memory_usage.iter().copied().max().unwrap_or(0)
    }

    /// Memory balance ratio: min / max usage (1.0 = perfectly balanced).
    pub fn balance_ratio(&self) -> f64 {
        let min = self.device_memory_usage.iter().copied().min().unwrap_or(0) as f64;
        let max = self.device_memory_usage.iter().copied().max().unwrap_or(1) as f64;
        if max == 0.0 {
            return 1.0;
        }
        min / max
    }
}

impl fmt::Display for ShardingPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ShardingPlan ({} devices):", self.num_devices)?;
        writeln!(f, "  attention: {}", self.attention_strategy)?;
        writeln!(f, "  ffn:       {}", self.ffn_strategy)?;
        writeln!(f, "  balance:   {:.2}", self.balance_ratio())?;
        for (i, mem) in self.device_memory_usage.iter().enumerate() {
            writeln!(f, "  device {i}: {:.1} MB", *mem as f64 / 1_048_576.0)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Plan builder
// ---------------------------------------------------------------------------

/// Assign layers to devices, attempting to balance memory.
///
/// Uses a greedy algorithm: iterate layers in order and assign each
/// to the device with the least accumulated memory so far.
pub fn assign_layers(topology: &ModelTopology, config: &ShardingConfig) -> Vec<LayerAssignment> {
    let n = config.num_devices;
    let mut device_mem = vec![0u64; n];
    let mut assignments = Vec::with_capacity(topology.num_layers);

    for (i, &mem) in topology.layer_memory.iter().enumerate() {
        // Pick device with lowest current usage.
        let target = device_mem
            .iter()
            .enumerate()
            .min_by_key(|(_, m)| *m)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        device_mem[target] += mem;
        assignments.push(LayerAssignment {
            layer_index: i,
            device_index: target,
            memory_bytes: mem,
        });
    }

    assignments
}

/// Build a complete sharding plan.
pub fn build_plan(topology: &ModelTopology, config: &ShardingConfig) -> ShardingPlan {
    let assignments = assign_layers(topology, config);
    let mut device_mem = vec![0u64; config.num_devices];
    for a in &assignments {
        device_mem[a.device_index] += a.memory_bytes;
    }

    ShardingPlan {
        layer_assignments: assignments,
        device_memory_usage: device_mem,
        num_devices: config.num_devices,
        attention_strategy: config.attention_strategy,
        ffn_strategy: config.ffn_strategy,
    }
}

/// Compute the shard descriptor for a weight tensor under a given strategy.
pub fn shard_tensor(
    name: impl Into<String>,
    rows: usize,
    cols: usize,
    strategy: ShardingStrategy,
    num_devices: usize,
) -> ShardDescriptor {
    let name = name.into();
    match strategy {
        ShardingStrategy::ColumnParallel => {
            assert!(
                cols % num_devices == 0,
                "cols ({cols}) must be divisible by num_devices ({num_devices})"
            );
            let shard_cols = cols / num_devices;
            ShardDescriptor {
                tensor_name: name,
                strategy,
                original_shape: [rows, cols],
                shard_shape: [rows, shard_cols],
                comm_op: CommOp::AllGather {
                    shard_elements: rows * shard_cols,
                    num_devices,
                },
            }
        }
        ShardingStrategy::RowParallel => {
            assert!(
                rows % num_devices == 0,
                "rows ({rows}) must be divisible by num_devices ({num_devices})"
            );
            let shard_rows = rows / num_devices;
            ShardDescriptor {
                tensor_name: name,
                strategy,
                original_shape: [rows, cols],
                shard_shape: [shard_rows, cols],
                comm_op: CommOp::ReduceScatter {
                    shard_elements: shard_rows * cols,
                    num_devices,
                },
            }
        }
        ShardingStrategy::Pipeline => ShardDescriptor {
            tensor_name: name,
            strategy,
            original_shape: [rows, cols],
            shard_shape: [rows, cols],
            comm_op: CommOp::None,
        },
    }
}

/// Build a pipeline send/recv descriptor between consecutive stages.
pub fn pipeline_comm(
    src_device: usize,
    dst_device: usize,
    hidden_dim: usize,
    seq_len: usize,
) -> CommOp {
    CommOp::SendRecv {
        src_device,
        dst_device,
        num_elements: hidden_dim * seq_len,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_topology(num_layers: usize) -> ModelTopology {
        ModelTopology {
            num_layers,
            layer_memory: vec![100_000_000; num_layers], // 100 MB each
            hidden_dim: 2048,
            num_heads: 16,
        }
    }

    #[test]
    fn test_sharding_strategy_display() {
        assert_eq!(ShardingStrategy::ColumnParallel.to_string(), "ColumnParallel");
        assert_eq!(ShardingStrategy::RowParallel.to_string(), "RowParallel");
        assert_eq!(ShardingStrategy::Pipeline.to_string(), "Pipeline");
    }

    #[test]
    fn test_uniform_config_2way() {
        let cfg = ShardingConfig::two_way(4_000_000_000);
        assert_eq!(cfg.num_devices, 2);
        assert_eq!(cfg.per_device_memory.len(), 2);
        assert_eq!(cfg.per_device_memory[0], 4_000_000_000);
    }

    #[test]
    fn test_uniform_config_4way() {
        let cfg = ShardingConfig::four_way(8_000_000_000);
        assert_eq!(cfg.num_devices, 4);
    }

    #[test]
    fn test_uniform_config_8way() {
        let cfg = ShardingConfig::eight_way(16_000_000_000);
        assert_eq!(cfg.num_devices, 8);
    }

    #[test]
    #[should_panic(expected = "num_devices must be 2, 4, or 8")]
    fn test_invalid_device_count() {
        ShardingConfig::uniform(3, 1_000_000_000);
    }

    #[test]
    fn test_layer_assignment_balanced() {
        let topo = sample_topology(8);
        let cfg = ShardingConfig::two_way(4_000_000_000);
        let assignments = assign_layers(&topo, &cfg);

        assert_eq!(assignments.len(), 8);

        // Each device should get 4 layers (balanced).
        let dev0 = assignments.iter().filter(|a| a.device_index == 0).count();
        let dev1 = assignments.iter().filter(|a| a.device_index == 1).count();
        assert_eq!(dev0, 4);
        assert_eq!(dev1, 4);
    }

    #[test]
    fn test_layer_assignment_uneven() {
        // 7 layers across 4 devices: expect 2+2+2+1 or similar balanced distribution.
        let topo = sample_topology(7);
        let cfg = ShardingConfig::four_way(4_000_000_000);
        let assignments = assign_layers(&topo, &cfg);

        let mut counts = vec![0u32; 4];
        for a in &assignments {
            counts[a.device_index] += 1;
        }
        let max = *counts.iter().max().unwrap();
        let min = *counts.iter().min().unwrap();
        // Should be at most 1 apart.
        assert!(max - min <= 1, "imbalanced: {counts:?}");
    }

    #[test]
    fn test_build_plan_memory_balance() {
        let topo = sample_topology(8);
        let cfg = ShardingConfig::two_way(4_000_000_000);
        let plan = build_plan(&topo, &cfg);

        assert_eq!(plan.num_devices, 2);
        assert!((plan.balance_ratio() - 1.0).abs() < 0.01);
        assert_eq!(plan.peak_device_memory(), 400_000_000); // 4 × 100 MB
    }

    #[test]
    fn test_shard_tensor_column_parallel() {
        let sd = shard_tensor("q_proj", 4096, 4096, ShardingStrategy::ColumnParallel, 2);
        assert_eq!(sd.shard_shape, [4096, 2048]);
        assert!(matches!(
            sd.comm_op,
            CommOp::AllGather { shard_elements: 8_388_608, num_devices: 2 }
        ));
    }

    #[test]
    fn test_shard_tensor_row_parallel() {
        let sd = shard_tensor("v_proj", 4096, 4096, ShardingStrategy::RowParallel, 4);
        assert_eq!(sd.shard_shape, [1024, 4096]);
        assert!(matches!(
            sd.comm_op,
            CommOp::ReduceScatter { shard_elements: 4_194_304, num_devices: 4 }
        ));
    }

    #[test]
    fn test_shard_tensor_pipeline() {
        let sd = shard_tensor("w", 512, 512, ShardingStrategy::Pipeline, 2);
        assert_eq!(sd.shard_shape, [512, 512]); // not split
        assert_eq!(sd.comm_op, CommOp::None);
    }

    #[test]
    fn test_pipeline_comm_descriptor() {
        let op = pipeline_comm(0, 1, 2048, 128);
        assert_eq!(
            op,
            CommOp::SendRecv {
                src_device: 0,
                dst_device: 1,
                num_elements: 2048 * 128,
            }
        );
    }

    #[test]
    fn test_comm_op_display() {
        let ag = CommOp::AllGather { shard_elements: 1024, num_devices: 2 };
        assert!(ag.to_string().contains("AllGather"));

        let rs = CommOp::ReduceScatter { shard_elements: 512, num_devices: 4 };
        assert!(rs.to_string().contains("ReduceScatter"));

        let sr = CommOp::SendRecv { src_device: 0, dst_device: 1, num_elements: 256 };
        assert!(sr.to_string().contains("SendRecv"));

        assert_eq!(CommOp::None.to_string(), "None");
    }

    #[test]
    fn test_plan_display() {
        let topo = sample_topology(4);
        let cfg = ShardingConfig::two_way(4_000_000_000);
        let plan = build_plan(&topo, &cfg);
        let s = plan.to_string();
        assert!(s.contains("2 devices"));
        assert!(s.contains("ColumnParallel"));
    }
}
