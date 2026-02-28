//! Tensor parallelism across multiple GPU devices.
//!
//! Provides [`TensorParallelConfig`] for specifying device assignment, weight
//! sharding across GPUs, and an AllReduce implementation for aggregating
//! partial results.  Supports 2-way and 4-way parallelism.

use std::fmt;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Device assignment strategy
// ---------------------------------------------------------------------------

/// Strategy for assigning tensor slices to GPU devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceAssignment {
    /// Round-robin across all available devices.
    RoundRobin,
    /// Assign by layer index modulo device count.
    LayerModulo,
    /// Manually specified (handled externally via the mapping table).
    Manual,
}

impl Default for DeviceAssignment {
    fn default() -> Self {
        Self::RoundRobin
    }
}

// ---------------------------------------------------------------------------
// Parallelism degree
// ---------------------------------------------------------------------------

/// Supported parallelism degrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismDegree {
    /// Split across 2 devices.
    TwoWay,
    /// Split across 4 devices.
    FourWay,
}

impl ParallelismDegree {
    /// Number of devices required.
    pub fn device_count(self) -> usize {
        match self {
            Self::TwoWay => 2,
            Self::FourWay => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for tensor-parallel inference.
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// How many devices to split across.
    pub degree: ParallelismDegree,
    /// Strategy used to map weight shards to devices.
    pub assignment: DeviceAssignment,
    /// Logical device IDs participating in the parallel group.
    pub device_ids: Vec<usize>,
    /// Per-device manual layer mapping (only used when `assignment == Manual`).
    /// Key = layer index, Value = device index within `device_ids`.
    pub layer_device_map: Vec<(usize, usize)>,
}

impl TensorParallelConfig {
    /// Create a config with the given degree using round-robin assignment.
    pub fn new(degree: ParallelismDegree, device_ids: Vec<usize>) -> Result<Self, String> {
        let required = degree.device_count();
        if device_ids.len() != required {
            return Err(format!(
                "expected {required} devices for {degree:?}, got {}",
                device_ids.len()
            ));
        }
        if device_ids.is_empty() {
            return Err("device_ids must not be empty".into());
        }
        Ok(Self {
            degree,
            assignment: DeviceAssignment::RoundRobin,
            device_ids,
            layer_device_map: Vec::new(),
        })
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        let required = self.degree.device_count();
        if self.device_ids.len() != required {
            return Err(format!(
                "device_ids length {} != required {required}",
                self.device_ids.len()
            ));
        }
        // Check for duplicate device IDs.
        let mut seen = self.device_ids.clone();
        seen.sort();
        seen.dedup();
        if seen.len() != self.device_ids.len() {
            return Err("duplicate device IDs".into());
        }
        if self.assignment == DeviceAssignment::Manual && self.layer_device_map.is_empty() {
            return Err("manual assignment requires a non-empty layer_device_map".into());
        }
        Ok(())
    }

    /// Return the device id for a given layer index.
    pub fn device_for_layer(&self, layer_idx: usize) -> usize {
        match self.assignment {
            DeviceAssignment::RoundRobin | DeviceAssignment::LayerModulo => {
                self.device_ids[layer_idx % self.device_ids.len()]
            }
            DeviceAssignment::Manual => self
                .layer_device_map
                .iter()
                .find(|(l, _)| *l == layer_idx)
                .map(|(_, d)| *d)
                .unwrap_or(self.device_ids[0]),
        }
    }
}

// ---------------------------------------------------------------------------
// Weight sharding
// ---------------------------------------------------------------------------

/// A shard of a weight matrix destined for a specific device.
#[derive(Debug, Clone)]
pub struct WeightShard {
    /// Index of this shard (0-based).
    pub shard_index: usize,
    /// Total number of shards.
    pub total_shards: usize,
    /// Device id this shard is assigned to.
    pub device_id: usize,
    /// Row range [start, end) in the original weight matrix.
    pub row_range: (usize, usize),
    /// Flattened shard data (row-major f32).
    pub data: Vec<f32>,
}

/// Split a weight matrix (given as flat row-major `data` with `rows × cols`)
/// into `num_shards` roughly equal row-wise shards.
pub fn shard_weights(
    data: &[f32],
    rows: usize,
    cols: usize,
    device_ids: &[usize],
) -> Result<Vec<WeightShard>, String> {
    if data.len() != rows * cols {
        return Err(format!(
            "data length {} != rows({rows}) × cols({cols})",
            data.len()
        ));
    }
    if device_ids.is_empty() {
        return Err("device_ids must not be empty".into());
    }

    let num_shards = device_ids.len();
    let base_rows = rows / num_shards;
    let remainder = rows % num_shards;

    let mut shards = Vec::with_capacity(num_shards);
    let mut offset = 0usize;

    for (i, &dev) in device_ids.iter().enumerate() {
        let shard_rows = base_rows + if i < remainder { 1 } else { 0 };
        let start = offset;
        let end = offset + shard_rows;
        let shard_data = data[start * cols..end * cols].to_vec();
        shards.push(WeightShard {
            shard_index: i,
            total_shards: num_shards,
            device_id: dev,
            row_range: (start, end),
            data: shard_data,
        });
        offset = end;
    }

    Ok(shards)
}

// ---------------------------------------------------------------------------
// AllReduce
// ---------------------------------------------------------------------------

/// Reduction operation used during AllReduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sum => write!(f, "sum"),
            Self::Mean => write!(f, "mean"),
            Self::Max => write!(f, "max"),
        }
    }
}

/// Simulated AllReduce that aggregates partial output vectors from each device.
///
/// In production this would use NCCL / oneAPI collectives; here we provide a
/// CPU reference implementation for correctness testing.
pub struct AllReduce {
    op: ReduceOp,
}

impl AllReduce {
    pub fn new(op: ReduceOp) -> Self {
        Self { op }
    }

    /// Reduce `partial_outputs` (one per device, all same length) into a
    /// single output vector.
    pub fn reduce(&self, partial_outputs: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        if partial_outputs.is_empty() {
            return Err("no partial outputs to reduce".into());
        }
        let len = partial_outputs[0].len();
        if partial_outputs.iter().any(|v| v.len() != len) {
            return Err("partial outputs have mismatched lengths".into());
        }
        let n = partial_outputs.len() as f32;

        let mut result = vec![0.0f32; len];
        match self.op {
            ReduceOp::Sum => {
                for partial in partial_outputs {
                    for (r, &v) in result.iter_mut().zip(partial.iter()) {
                        *r += v;
                    }
                }
            }
            ReduceOp::Mean => {
                for partial in partial_outputs {
                    for (r, &v) in result.iter_mut().zip(partial.iter()) {
                        *r += v;
                    }
                }
                for r in &mut result {
                    *r /= n;
                }
            }
            ReduceOp::Max => {
                result.copy_from_slice(&partial_outputs[0]);
                for partial in &partial_outputs[1..] {
                    for (r, &v) in result.iter_mut().zip(partial.iter()) {
                        if v > *r {
                            *r = v;
                        }
                    }
                }
            }
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Parallel execution context
// ---------------------------------------------------------------------------

/// Holds shared state for a tensor-parallel forward pass.
pub struct TensorParallelContext {
    pub config: Arc<TensorParallelConfig>,
    pub all_reduce: AllReduce,
}

impl TensorParallelContext {
    pub fn new(config: TensorParallelConfig, reduce_op: ReduceOp) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config: Arc::new(config),
            all_reduce: AllReduce::new(reduce_op),
        })
    }

    /// Convenience: number of participating devices.
    pub fn num_devices(&self) -> usize {
        self.config.device_ids.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config ---------------------------------------------------------------

    #[test]
    fn test_config_two_way_valid() {
        let cfg = TensorParallelConfig::new(ParallelismDegree::TwoWay, vec![0, 1]);
        assert!(cfg.is_ok());
        assert!(cfg.unwrap().validate().is_ok());
    }

    #[test]
    fn test_config_four_way_valid() {
        let cfg = TensorParallelConfig::new(ParallelismDegree::FourWay, vec![0, 1, 2, 3]);
        assert!(cfg.is_ok());
        assert!(cfg.unwrap().validate().is_ok());
    }

    #[test]
    fn test_config_wrong_device_count() {
        let cfg = TensorParallelConfig::new(ParallelismDegree::TwoWay, vec![0]);
        assert!(cfg.is_err());
    }

    #[test]
    fn test_config_duplicate_devices_rejected() {
        let mut cfg = TensorParallelConfig::new(ParallelismDegree::TwoWay, vec![0, 1]).unwrap();
        cfg.device_ids = vec![0, 0];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_device_for_layer_round_robin() {
        let cfg = TensorParallelConfig::new(ParallelismDegree::TwoWay, vec![10, 20]).unwrap();
        assert_eq!(cfg.device_for_layer(0), 10);
        assert_eq!(cfg.device_for_layer(1), 20);
        assert_eq!(cfg.device_for_layer(2), 10);
        assert_eq!(cfg.device_for_layer(3), 20);
    }

    // -- Weight sharding ------------------------------------------------------

    #[test]
    fn test_shard_weights_even_split() {
        // 4 rows × 2 cols, 2 devices → 2 rows each
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let shards = shard_weights(&data, 4, 2, &[0, 1]).unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].row_range, (0, 2));
        assert_eq!(shards[0].data, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(shards[1].row_range, (2, 4));
        assert_eq!(shards[1].data, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_shard_weights_uneven_split() {
        // 5 rows × 1 col, 2 devices → 3 + 2
        let data: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let shards = shard_weights(&data, 5, 1, &[0, 1]).unwrap();
        assert_eq!(shards[0].row_range, (0, 3));
        assert_eq!(shards[1].row_range, (3, 5));
    }

    #[test]
    fn test_shard_weights_data_mismatch() {
        let data = vec![1.0; 10];
        assert!(shard_weights(&data, 3, 4, &[0]).is_err());
    }

    // -- AllReduce ------------------------------------------------------------

    #[test]
    fn test_allreduce_sum() {
        let ar = AllReduce::new(ReduceOp::Sum);
        let partials = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = ar.reduce(&partials).unwrap();
        assert_eq!(result, vec![4.0, 6.0]);
    }

    #[test]
    fn test_allreduce_mean() {
        let ar = AllReduce::new(ReduceOp::Mean);
        let partials = vec![vec![2.0, 4.0], vec![4.0, 8.0]];
        let result = ar.reduce(&partials).unwrap();
        assert_eq!(result, vec![3.0, 6.0]);
    }

    #[test]
    fn test_allreduce_max() {
        let ar = AllReduce::new(ReduceOp::Max);
        let partials = vec![vec![1.0, 5.0], vec![3.0, 2.0]];
        let result = ar.reduce(&partials).unwrap();
        assert_eq!(result, vec![3.0, 5.0]);
    }

    #[test]
    fn test_allreduce_mismatched_lengths() {
        let ar = AllReduce::new(ReduceOp::Sum);
        let partials = vec![vec![1.0], vec![1.0, 2.0]];
        assert!(ar.reduce(&partials).is_err());
    }

    #[test]
    fn test_allreduce_empty_input() {
        let ar = AllReduce::new(ReduceOp::Sum);
        assert!(ar.reduce(&[]).is_err());
    }

    // -- Context --------------------------------------------------------------

    #[test]
    fn test_parallel_context_creation() {
        let cfg = TensorParallelConfig::new(ParallelismDegree::TwoWay, vec![0, 1]).unwrap();
        let ctx = TensorParallelContext::new(cfg, ReduceOp::Sum).unwrap();
        assert_eq!(ctx.num_devices(), 2);
    }
}
