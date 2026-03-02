//! Model parallelism for multi-GPU and heterogeneous GPU/CPU inference.
//!
//! Supports three parallelism strategies and a hybrid combination:
//!
//! - **Tensor parallel**: split weight matrices across devices (row or column)
//!   so each device computes a slice of the result.
//! - **Pipeline parallel**: assign consecutive transformer layers to different
//!   devices, overlapping forward passes via GPipe / 1F1B scheduling.
//! - **Data parallel**: replicate the model on every device and split the input
//!   batch, gathering outputs after each forward pass.
//! - **Hybrid**: combine tensor + pipeline parallelism.
//!
//! All operations have pure-Rust CPU reference implementations.

use std::fmt;

// ---------------------------------------------------------------------------
// Parallelism mode
// ---------------------------------------------------------------------------

/// Strategy used to distribute work across multiple devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParallelismMode {
    /// Split individual weight tensors across devices.
    TensorParallel,
    /// Assign layers to pipeline stages on different devices.
    PipelineParallel,
    /// Replicate model, split input batch.
    DataParallel,
    /// Combine tensor and pipeline parallelism.
    Hybrid,
}

impl fmt::Display for ParallelismMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TensorParallel => write!(f, "TensorParallel"),
            Self::PipelineParallel => write!(f, "PipelineParallel"),
            Self::DataParallel => write!(f, "DataParallel"),
            Self::Hybrid => write!(f, "Hybrid"),
        }
    }
}

// ---------------------------------------------------------------------------
// Device abstraction
// ---------------------------------------------------------------------------

/// Kind of compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Gpu,
    Cpu,
}

/// A single compute device with estimated capabilities.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Unique identifier within the group.
    pub id: usize,
    /// Human-readable label (e.g. "Arc A770 #0").
    pub name: String,
    /// Device kind.
    pub kind: DeviceKind,
    /// Available memory in bytes.
    pub memory_bytes: u64,
    /// Estimated bandwidth in GB/s.
    pub bandwidth_gbps: f64,
}

impl DeviceInfo {
    pub fn new(
        id: usize,
        name: impl Into<String>,
        kind: DeviceKind,
        memory_bytes: u64,
        bandwidth_gbps: f64,
    ) -> Self {
        Self { id, name: name.into(), kind, memory_bytes, bandwidth_gbps }
    }
}

/// Ordered collection of devices participating in parallel execution.
#[derive(Debug, Clone)]
pub struct DeviceGroup {
    devices: Vec<DeviceInfo>,
}

impl DeviceGroup {
    /// Create a group from an ordered list of devices.
    ///
    /// # Errors
    ///
    /// Returns an error string if the list is empty.
    pub fn new(devices: Vec<DeviceInfo>) -> Result<Self, String> {
        if devices.is_empty() {
            return Err("DeviceGroup requires at least one device".into());
        }
        Ok(Self { devices })
    }

    pub fn len(&self) -> usize {
        self.devices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Total memory across all devices.
    pub fn total_memory(&self) -> u64 {
        self.devices.iter().map(|d| d.memory_bytes).sum()
    }

    /// Minimum bandwidth across all devices.
    pub fn min_bandwidth(&self) -> f64 {
        self.devices.iter().map(|d| d.bandwidth_gbps).fold(f64::INFINITY, f64::min)
    }

    /// True when the group mixes GPU and CPU devices.
    pub fn is_heterogeneous(&self) -> bool {
        let has_gpu = self.devices.iter().any(|d| d.kind == DeviceKind::Gpu);
        let has_cpu = self.devices.iter().any(|d| d.kind == DeviceKind::Cpu);
        has_gpu && has_cpu
    }
}

// ---------------------------------------------------------------------------
// Tensor splitting
// ---------------------------------------------------------------------------

/// How a weight tensor is split across devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitStrategy {
    /// Split along the row dimension (axis 0).
    Row,
    /// Split along the column dimension (axis 1).
    Column,
}

/// Describes one shard of a split tensor.
#[derive(Debug, Clone)]
pub struct TensorShard {
    pub device_id: usize,
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

/// Splits 2-D weight tensors across devices.
pub struct TensorSplitter;

impl TensorSplitter {
    /// Split `data` (row-major, shape `[rows, cols]`) across `n_devices`.
    ///
    /// Returns one [`TensorShard`] per device. When `rows` (or `cols`) is
    /// not evenly divisible, the last shard absorbs the remainder.
    pub fn split(
        data: &[f32],
        rows: usize,
        cols: usize,
        n_devices: usize,
        strategy: SplitStrategy,
    ) -> Result<Vec<TensorShard>, String> {
        if data.len() != rows * cols {
            return Err(format!(
                "data length {} != rows*cols ({}*{}={})",
                data.len(),
                rows,
                cols,
                rows * cols
            ));
        }
        if n_devices == 0 {
            return Err("n_devices must be > 0".into());
        }

        match strategy {
            SplitStrategy::Row => Self::split_rows(data, rows, cols, n_devices),
            SplitStrategy::Column => Self::split_cols(data, rows, cols, n_devices),
        }
    }

    fn split_rows(
        data: &[f32],
        rows: usize,
        cols: usize,
        n: usize,
    ) -> Result<Vec<TensorShard>, String> {
        let base = rows / n;
        let remainder = rows % n;
        let mut shards = Vec::with_capacity(n);
        let mut offset = 0usize;

        for i in 0..n {
            let r = base + if i < remainder { 1 } else { 0 };
            let count = r * cols;
            let shard_data = data[offset..offset + count].to_vec();
            shards.push(TensorShard { device_id: i, data: shard_data, rows: r, cols });
            offset += count;
        }
        Ok(shards)
    }

    fn split_cols(
        data: &[f32],
        rows: usize,
        cols: usize,
        n: usize,
    ) -> Result<Vec<TensorShard>, String> {
        let base = cols / n;
        let remainder = cols % n;
        let mut shards: Vec<TensorShard> = (0..n)
            .map(|i| {
                let c = base + if i < remainder { 1 } else { 0 };
                TensorShard { device_id: i, data: Vec::with_capacity(rows * c), rows, cols: c }
            })
            .collect();

        for row in 0..rows {
            let row_start = row * cols;
            let mut col_offset = 0usize;
            for shard in &mut shards {
                let end = col_offset + shard.cols;
                shard.data.extend_from_slice(&data[row_start + col_offset..row_start + end]);
                col_offset = end;
            }
        }

        Ok(shards)
    }

    /// Reconstruct the original tensor from shards.
    pub fn reconstruct(
        shards: &[TensorShard],
        strategy: SplitStrategy,
    ) -> Result<(Vec<f32>, usize, usize), String> {
        if shards.is_empty() {
            return Err("no shards to reconstruct".into());
        }
        match strategy {
            SplitStrategy::Row => Self::reconstruct_rows(shards),
            SplitStrategy::Column => Self::reconstruct_cols(shards),
        }
    }

    fn reconstruct_rows(shards: &[TensorShard]) -> Result<(Vec<f32>, usize, usize), String> {
        let cols = shards[0].cols;
        let total_rows: usize = shards.iter().map(|s| s.rows).sum();
        let mut out = Vec::with_capacity(total_rows * cols);
        for shard in shards {
            if shard.cols != cols {
                return Err("mismatched cols in row shards".into());
            }
            out.extend_from_slice(&shard.data);
        }
        Ok((out, total_rows, cols))
    }

    fn reconstruct_cols(shards: &[TensorShard]) -> Result<(Vec<f32>, usize, usize), String> {
        let rows = shards[0].rows;
        let total_cols: usize = shards.iter().map(|s| s.cols).sum();
        let mut out = vec![0.0f32; rows * total_cols];

        for row in 0..rows {
            let mut col_offset = 0usize;
            for shard in shards {
                if shard.rows != rows {
                    return Err("mismatched rows in column shards".into());
                }
                let src_start = row * shard.cols;
                let dst_start = row * total_cols + col_offset;
                out[dst_start..dst_start + shard.cols]
                    .copy_from_slice(&shard.data[src_start..src_start + shard.cols]);
                col_offset += shard.cols;
            }
        }
        Ok((out, rows, total_cols))
    }
}

// ---------------------------------------------------------------------------
// All-reduce
// ---------------------------------------------------------------------------

/// Reduction operation for cross-device synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

/// Simulated ring all-reduce across device buffers.
pub struct AllReduceOp;

impl AllReduceOp {
    /// Perform an all-reduce over `buffers` (one per device) in-place.
    ///
    /// After completion every buffer contains the same reduced result.
    pub fn all_reduce(buffers: &mut [Vec<f32>], op: ReduceOp) -> Result<(), String> {
        if buffers.is_empty() {
            return Err("no buffers for all-reduce".into());
        }
        let len = buffers[0].len();
        for (i, buf) in buffers.iter().enumerate() {
            if buf.len() != len {
                return Err(format!("buffer {} length {} != expected {}", i, buf.len(), len));
            }
        }
        if buffers.len() == 1 {
            if op == ReduceOp::Mean {
                // mean of a single buffer is itself — no-op
            }
            return Ok(());
        }

        let n = buffers.len() as f32;
        let mut reduced = vec![0.0f32; len];

        match op {
            ReduceOp::Sum | ReduceOp::Mean => {
                for buf in buffers.iter() {
                    for (r, &v) in reduced.iter_mut().zip(buf.iter()) {
                        *r += v;
                    }
                }
                if op == ReduceOp::Mean {
                    for r in &mut reduced {
                        *r /= n;
                    }
                }
            }
            ReduceOp::Max => {
                reduced.copy_from_slice(&buffers[0]);
                for buf in buffers.iter().skip(1) {
                    for (r, &v) in reduced.iter_mut().zip(buf.iter()) {
                        *r = r.max(v);
                    }
                }
            }
            ReduceOp::Min => {
                reduced.copy_from_slice(&buffers[0]);
                for buf in buffers.iter().skip(1) {
                    for (r, &v) in reduced.iter_mut().zip(buf.iter()) {
                        *r = r.min(v);
                    }
                }
            }
        }

        for buf in buffers.iter_mut() {
            buf.copy_from_slice(&reduced);
        }
        Ok(())
    }

    /// Estimate the bytes transferred during a ring all-reduce of `n_elements`
    /// across `n_devices` using the standard 2*(N-1)/N algorithm.
    pub fn estimated_transfer_bytes(n_elements: usize, n_devices: usize) -> u64 {
        if n_devices <= 1 {
            return 0;
        }
        let element_bytes = std::mem::size_of::<f32>() as u64;
        let total = n_elements as u64 * element_bytes;
        // Ring all-reduce: 2 * (N-1) / N * total_bytes
        2 * (n_devices as u64 - 1) * total / n_devices as u64
    }
}

// ---------------------------------------------------------------------------
// Pipeline parallelism
// ---------------------------------------------------------------------------

/// One stage in a pipeline, mapped to a single device.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Device that owns this stage.
    pub device_id: usize,
    /// Layer indices assigned to this stage (inclusive range).
    pub layer_start: usize,
    pub layer_end: usize,
}

impl PipelineStage {
    pub fn num_layers(&self) -> usize {
        if self.layer_end >= self.layer_start { self.layer_end - self.layer_start + 1 } else { 0 }
    }
}

/// Assigns transformer layers to pipeline stages.
pub struct PipelineScheduler;

/// Micro-batch action emitted by the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineAction {
    /// Compute forward pass for micro-batch `micro_batch` on `stage`.
    Forward { stage: usize, micro_batch: usize },
    /// Idle slot (pipeline bubble).
    Bubble { stage: usize },
}

impl PipelineScheduler {
    /// Evenly assign `n_layers` across `n_devices`.
    ///
    /// When layers don't divide evenly the first stages receive one extra
    /// layer. If there are more devices than layers the extra devices get
    /// empty stages.
    pub fn assign_stages(n_layers: usize, n_devices: usize) -> Result<Vec<PipelineStage>, String> {
        if n_devices == 0 {
            return Err("n_devices must be > 0".into());
        }
        if n_layers == 0 {
            return Ok((0..n_devices)
                .map(|i| PipelineStage { device_id: i, layer_start: 0, layer_end: 0 })
                .collect());
        }
        let effective = n_devices.min(n_layers);
        let base = n_layers / effective;
        let remainder = n_layers % effective;
        let mut stages = Vec::with_capacity(n_devices);
        let mut offset = 0usize;

        for i in 0..effective {
            let count = base + if i < remainder { 1 } else { 0 };
            stages.push(PipelineStage {
                device_id: i,
                layer_start: offset,
                layer_end: offset + count - 1,
            });
            offset += count;
        }

        // Extra devices with no layers
        for i in effective..n_devices {
            stages.push(PipelineStage { device_id: i, layer_start: 0, layer_end: 0 });
        }

        Ok(stages)
    }

    /// Produce GPipe-style forward schedule for `n_micro_batches` across
    /// `n_stages`.
    ///
    /// Returns a 2-D grid `[time_step][stage]` of [`PipelineAction`]s.
    pub fn gpipe_schedule(n_stages: usize, n_micro_batches: usize) -> Vec<Vec<PipelineAction>> {
        if n_stages == 0 || n_micro_batches == 0 {
            return vec![];
        }
        let total_steps = n_stages + n_micro_batches - 1;
        let mut schedule = Vec::with_capacity(total_steps);

        for t in 0..total_steps {
            let mut step = Vec::with_capacity(n_stages);
            for s in 0..n_stages {
                if t >= s && (t - s) < n_micro_batches {
                    step.push(PipelineAction::Forward { stage: s, micro_batch: t - s });
                } else {
                    step.push(PipelineAction::Bubble { stage: s });
                }
            }
            schedule.push(step);
        }
        schedule
    }

    /// Compute the bubble ratio (fraction of idle slots) for a GPipe
    /// schedule.
    pub fn bubble_ratio(n_stages: usize, n_micro_batches: usize) -> f64 {
        if n_stages == 0 || n_micro_batches == 0 {
            return 1.0;
        }
        let total_steps = n_stages + n_micro_batches - 1;
        let total_slots = total_steps * n_stages;
        let useful_slots = n_stages * n_micro_batches;
        (total_slots - useful_slots) as f64 / total_slots as f64
    }
}

// ---------------------------------------------------------------------------
// Data parallelism
// ---------------------------------------------------------------------------

/// Splits an input batch across devices and gathers the result.
pub struct DataParallelBatch;

impl DataParallelBatch {
    /// Split `batch` (shape `[batch_size, feature_dim]`) across `n_devices`.
    pub fn split(
        batch: &[f32],
        batch_size: usize,
        feature_dim: usize,
        n_devices: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        if batch.len() != batch_size * feature_dim {
            return Err(format!(
                "batch length {} != batch_size*feature_dim ({}*{}={})",
                batch.len(),
                batch_size,
                feature_dim,
                batch_size * feature_dim
            ));
        }
        if n_devices == 0 {
            return Err("n_devices must be > 0".into());
        }
        let base = batch_size / n_devices;
        let remainder = batch_size % n_devices;
        let mut splits = Vec::with_capacity(n_devices);
        let mut offset = 0usize;

        for i in 0..n_devices {
            let count = base + if i < remainder { 1 } else { 0 };
            let elems = count * feature_dim;
            splits.push(batch[offset..offset + elems].to_vec());
            offset += elems;
        }
        Ok(splits)
    }

    /// Gather device outputs back into a single batch.
    pub fn gather(parts: &[Vec<f32>]) -> Vec<f32> {
        parts.iter().flat_map(|p| p.iter().copied()).collect()
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Per-device utilization snapshot.
#[derive(Debug, Clone)]
pub struct DeviceUtilization {
    pub device_id: usize,
    /// Fraction in `[0.0, 1.0]` of time the device is computing.
    pub compute_fraction: f64,
    /// Bytes transferred during communication phases.
    pub comm_bytes: u64,
}

/// Aggregate statistics for a parallel execution.
#[derive(Debug, Clone)]
pub struct ParallelStats {
    pub mode: ParallelismMode,
    pub device_utils: Vec<DeviceUtilization>,
    /// Fraction of total slots that are pipeline bubbles (0 when not
    /// pipeline).
    pub bubble_ratio: f64,
    /// Estimated communication overhead in bytes.
    pub total_comm_bytes: u64,
}

impl ParallelStats {
    /// Mean compute utilization across all devices.
    pub fn mean_utilization(&self) -> f64 {
        if self.device_utils.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.device_utils.iter().map(|d| d.compute_fraction).sum();
        sum / self.device_utils.len() as f64
    }

    /// Build stats for a tensor-parallel workload.
    pub fn for_tensor_parallel(n_devices: usize, n_elements: usize) -> Self {
        let comm = AllReduceOp::estimated_transfer_bytes(n_elements, n_devices);
        let per_device = comm / n_devices.max(1) as u64;
        let utils = (0..n_devices)
            .map(|id| DeviceUtilization {
                device_id: id,
                compute_fraction: 1.0,
                comm_bytes: per_device,
            })
            .collect();
        Self {
            mode: ParallelismMode::TensorParallel,
            device_utils: utils,
            bubble_ratio: 0.0,
            total_comm_bytes: comm,
        }
    }

    /// Build stats for a pipeline-parallel workload.
    pub fn for_pipeline_parallel(n_stages: usize, n_micro_batches: usize) -> Self {
        let br = PipelineScheduler::bubble_ratio(n_stages, n_micro_batches);
        let utils = (0..n_stages)
            .map(|id| DeviceUtilization {
                device_id: id,
                compute_fraction: 1.0 - br,
                comm_bytes: 0,
            })
            .collect();
        Self {
            mode: ParallelismMode::PipelineParallel,
            device_utils: utils,
            bubble_ratio: br,
            total_comm_bytes: 0,
        }
    }

    /// Build stats for a data-parallel workload.
    pub fn for_data_parallel(n_devices: usize, n_elements: usize) -> Self {
        let comm = AllReduceOp::estimated_transfer_bytes(n_elements, n_devices);
        let per_device = comm / n_devices.max(1) as u64;
        let utils = (0..n_devices)
            .map(|id| DeviceUtilization {
                device_id: id,
                compute_fraction: 1.0,
                comm_bytes: per_device,
            })
            .collect();
        Self {
            mode: ParallelismMode::DataParallel,
            device_utils: utils,
            bubble_ratio: 0.0,
            total_comm_bytes: comm,
        }
    }
}

// ---------------------------------------------------------------------------
// Weighted stage assignment for heterogeneous devices
// ---------------------------------------------------------------------------

impl PipelineScheduler {
    /// Assign layers proportionally to each device's memory budget.
    ///
    /// Devices with more memory receive more layers. Devices whose share
    /// rounds to zero receive no stage.
    pub fn assign_stages_weighted(
        n_layers: usize,
        memory_budgets: &[u64],
    ) -> Result<Vec<PipelineStage>, String> {
        if memory_budgets.is_empty() {
            return Err("memory_budgets must not be empty".into());
        }
        let total_mem: u64 = memory_budgets.iter().sum();
        if total_mem == 0 {
            return Err("total memory budget is zero".into());
        }

        // Compute fractional share, then round.
        let n = memory_budgets.len();
        let mut counts: Vec<usize> = memory_budgets
            .iter()
            .map(|&m| ((m as f64 / total_mem as f64) * n_layers as f64).round() as usize)
            .collect();

        // Fix up rounding errors.
        let assigned: usize = counts.iter().sum();
        if assigned < n_layers {
            // Give remaining to the device with the most memory.
            let max_idx =
                memory_budgets.iter().enumerate().max_by_key(|(_, m)| *m).map(|(i, _)| i).unwrap();
            counts[max_idx] += n_layers - assigned;
        } else if assigned > n_layers {
            let max_idx =
                counts.iter().enumerate().max_by_key(|(_, c)| *c).map(|(i, _)| i).unwrap();
            counts[max_idx] = counts[max_idx].saturating_sub(assigned - n_layers);
        }

        let mut stages = Vec::with_capacity(n);
        let mut offset = 0usize;
        for (i, &count) in counts.iter().enumerate() {
            if count == 0 {
                stages.push(PipelineStage { device_id: i, layer_start: 0, layer_end: 0 });
            } else {
                stages.push(PipelineStage {
                    device_id: i,
                    layer_start: offset,
                    layer_end: offset + count - 1,
                });
                offset += count;
            }
        }
        Ok(stages)
    }
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Helper factories
    // ---------------------------------------------------------------

    fn gpu(id: usize, mem_gb: u64) -> DeviceInfo {
        DeviceInfo::new(id, format!("GPU-{id}"), DeviceKind::Gpu, mem_gb * 1_073_741_824, 560.0)
    }

    fn cpu_dev(id: usize, mem_gb: u64) -> DeviceInfo {
        DeviceInfo::new(id, format!("CPU-{id}"), DeviceKind::Cpu, mem_gb * 1_073_741_824, 50.0)
    }

    fn make_matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|i| i as f32).collect()
    }

    // ---------------------------------------------------------------
    // ParallelismMode
    // ---------------------------------------------------------------

    #[test]
    fn mode_display() {
        assert_eq!(ParallelismMode::TensorParallel.to_string(), "TensorParallel");
        assert_eq!(ParallelismMode::PipelineParallel.to_string(), "PipelineParallel");
        assert_eq!(ParallelismMode::DataParallel.to_string(), "DataParallel");
        assert_eq!(ParallelismMode::Hybrid.to_string(), "Hybrid");
    }

    #[test]
    fn mode_equality() {
        assert_eq!(ParallelismMode::Hybrid, ParallelismMode::Hybrid);
        assert_ne!(ParallelismMode::Hybrid, ParallelismMode::DataParallel);
    }

    // ---------------------------------------------------------------
    // DeviceGroup
    // ---------------------------------------------------------------

    #[test]
    fn device_group_empty_rejected() {
        assert!(DeviceGroup::new(vec![]).is_err());
    }

    #[test]
    fn device_group_single() {
        let g = DeviceGroup::new(vec![gpu(0, 16)]).unwrap();
        assert_eq!(g.len(), 1);
        assert!(!g.is_empty());
        assert!(!g.is_heterogeneous());
    }

    #[test]
    fn device_group_total_memory() {
        let g = DeviceGroup::new(vec![gpu(0, 16), gpu(1, 16)]).unwrap();
        assert_eq!(g.total_memory(), 32 * 1_073_741_824);
    }

    #[test]
    fn device_group_min_bandwidth() {
        let g = DeviceGroup::new(vec![gpu(0, 16), cpu_dev(1, 64)]).unwrap();
        assert!((g.min_bandwidth() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn device_group_heterogeneous() {
        let g = DeviceGroup::new(vec![gpu(0, 16), cpu_dev(1, 64)]).unwrap();
        assert!(g.is_heterogeneous());
    }

    #[test]
    fn device_group_homogeneous_gpu() {
        let g = DeviceGroup::new(vec![gpu(0, 16), gpu(1, 16)]).unwrap();
        assert!(!g.is_heterogeneous());
    }

    #[test]
    fn device_group_homogeneous_cpu() {
        let g = DeviceGroup::new(vec![cpu_dev(0, 64), cpu_dev(1, 64)]).unwrap();
        assert!(!g.is_heterogeneous());
    }

    #[test]
    fn device_group_devices_accessor() {
        let g = DeviceGroup::new(vec![gpu(0, 8), gpu(1, 16)]).unwrap();
        assert_eq!(g.devices().len(), 2);
        assert_eq!(g.devices()[0].id, 0);
        assert_eq!(g.devices()[1].memory_bytes, 16 * 1_073_741_824);
    }

    // ---------------------------------------------------------------
    // TensorSplitter — row split
    // ---------------------------------------------------------------

    #[test]
    fn row_split_even() {
        let m = make_matrix(4, 3);
        let shards = TensorSplitter::split(&m, 4, 3, 2, SplitStrategy::Row).unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].rows, 2);
        assert_eq!(shards[1].rows, 2);
        assert_eq!(shards[0].data, &m[..6]);
        assert_eq!(shards[1].data, &m[6..]);
    }

    #[test]
    fn row_split_uneven() {
        let m = make_matrix(5, 2);
        let shards = TensorSplitter::split(&m, 5, 2, 3, SplitStrategy::Row).unwrap();
        assert_eq!(shards[0].rows, 2); // 5/3 = 1 + 1 remainder for first 2
        assert_eq!(shards[1].rows, 2);
        assert_eq!(shards[2].rows, 1);
    }

    #[test]
    fn row_split_single_device() {
        let m = make_matrix(3, 4);
        let shards = TensorSplitter::split(&m, 3, 4, 1, SplitStrategy::Row).unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].data, m);
    }

    #[test]
    fn row_split_reconstruct_identity() {
        let m = make_matrix(6, 4);
        let shards = TensorSplitter::split(&m, 6, 4, 3, SplitStrategy::Row).unwrap();
        let (recon, rows, cols) = TensorSplitter::reconstruct(&shards, SplitStrategy::Row).unwrap();
        assert_eq!(rows, 6);
        assert_eq!(cols, 4);
        assert_eq!(recon, m);
    }

    #[test]
    fn row_split_reconstruct_uneven() {
        let m = make_matrix(7, 3);
        let shards = TensorSplitter::split(&m, 7, 3, 4, SplitStrategy::Row).unwrap();
        let (recon, rows, cols) = TensorSplitter::reconstruct(&shards, SplitStrategy::Row).unwrap();
        assert_eq!(rows, 7);
        assert_eq!(cols, 3);
        assert_eq!(recon, m);
    }

    // ---------------------------------------------------------------
    // TensorSplitter — column split
    // ---------------------------------------------------------------

    #[test]
    fn col_split_even() {
        let m = make_matrix(2, 4);
        let shards = TensorSplitter::split(&m, 2, 4, 2, SplitStrategy::Column).unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].cols, 2);
        assert_eq!(shards[1].cols, 2);
        // First shard: columns 0,1 from each row
        assert_eq!(shards[0].data, vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(shards[1].data, vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn col_split_uneven() {
        let m = make_matrix(2, 5);
        let shards = TensorSplitter::split(&m, 2, 5, 3, SplitStrategy::Column).unwrap();
        assert_eq!(shards[0].cols, 2);
        assert_eq!(shards[1].cols, 2);
        assert_eq!(shards[2].cols, 1);
    }

    #[test]
    fn col_split_reconstruct_identity() {
        let m = make_matrix(4, 6);
        let shards = TensorSplitter::split(&m, 4, 6, 3, SplitStrategy::Column).unwrap();
        let (recon, rows, cols) =
            TensorSplitter::reconstruct(&shards, SplitStrategy::Column).unwrap();
        assert_eq!(rows, 4);
        assert_eq!(cols, 6);
        assert_eq!(recon, m);
    }

    #[test]
    fn col_split_reconstruct_uneven() {
        let m = make_matrix(3, 7);
        let shards = TensorSplitter::split(&m, 3, 7, 4, SplitStrategy::Column).unwrap();
        let (recon, rows, cols) =
            TensorSplitter::reconstruct(&shards, SplitStrategy::Column).unwrap();
        assert_eq!(rows, 3);
        assert_eq!(cols, 7);
        assert_eq!(recon, m);
    }

    // ---------------------------------------------------------------
    // TensorSplitter — error cases
    // ---------------------------------------------------------------

    #[test]
    fn split_bad_length() {
        let data = vec![1.0; 10];
        assert!(TensorSplitter::split(&data, 3, 4, 2, SplitStrategy::Row).is_err());
    }

    #[test]
    fn split_zero_devices() {
        let data = vec![1.0; 6];
        assert!(TensorSplitter::split(&data, 2, 3, 0, SplitStrategy::Row).is_err());
    }

    #[test]
    fn reconstruct_empty_shards() {
        assert!(TensorSplitter::reconstruct(&[], SplitStrategy::Row).is_err());
    }

    // ---------------------------------------------------------------
    // AllReduceOp — sum
    // ---------------------------------------------------------------

    #[test]
    fn all_reduce_sum_two_devices() {
        let mut bufs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        AllReduceOp::all_reduce(&mut bufs, ReduceOp::Sum).unwrap();
        assert_eq!(bufs[0], vec![5.0, 7.0, 9.0]);
        assert_eq!(bufs[1], vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn all_reduce_sum_three_devices() {
        let mut bufs = vec![vec![1.0, 0.0], vec![2.0, 3.0], vec![4.0, 5.0]];
        AllReduceOp::all_reduce(&mut bufs, ReduceOp::Sum).unwrap();
        assert_eq!(bufs[0], vec![7.0, 8.0]);
        assert_eq!(bufs[2], vec![7.0, 8.0]);
    }

    // ---------------------------------------------------------------
    // AllReduceOp — mean
    // ---------------------------------------------------------------

    #[test]
    fn all_reduce_mean() {
        let mut bufs = vec![vec![2.0, 4.0], vec![6.0, 8.0]];
        AllReduceOp::all_reduce(&mut bufs, ReduceOp::Mean).unwrap();
        assert_eq!(bufs[0], vec![4.0, 6.0]);
        assert_eq!(bufs[1], vec![4.0, 6.0]);
    }

    #[test]
    fn all_reduce_mean_three() {
        let mut bufs = vec![vec![3.0, 6.0], vec![6.0, 9.0], vec![9.0, 12.0]];
        AllReduceOp::all_reduce(&mut bufs, ReduceOp::Mean).unwrap();
        assert_eq!(bufs[0], vec![6.0, 9.0]);
    }

    // ---------------------------------------------------------------
    // AllReduceOp — max / min
    // ---------------------------------------------------------------

    #[test]
    fn all_reduce_max() {
        let mut bufs = vec![vec![1.0, 5.0], vec![3.0, 2.0]];
        AllReduceOp::all_reduce(&mut bufs, ReduceOp::Max).unwrap();
        assert_eq!(bufs[0], vec![3.0, 5.0]);
        assert_eq!(bufs[1], vec![3.0, 5.0]);
    }

    #[test]
    fn all_reduce_min() {
        let mut bufs = vec![vec![1.0, 5.0], vec![3.0, 2.0]];
        AllReduceOp::all_reduce(&mut bufs, ReduceOp::Min).unwrap();
        assert_eq!(bufs[0], vec![1.0, 2.0]);
        assert_eq!(bufs[1], vec![1.0, 2.0]);
    }

    // ---------------------------------------------------------------
    // AllReduceOp — edge cases
    // ---------------------------------------------------------------

    #[test]
    fn all_reduce_single_device() {
        let mut bufs = vec![vec![1.0, 2.0]];
        AllReduceOp::all_reduce(&mut bufs, ReduceOp::Sum).unwrap();
        assert_eq!(bufs[0], vec![1.0, 2.0]);
    }

    #[test]
    fn all_reduce_single_device_mean() {
        let mut bufs = vec![vec![5.0]];
        AllReduceOp::all_reduce(&mut bufs, ReduceOp::Mean).unwrap();
        assert_eq!(bufs[0], vec![5.0]);
    }

    #[test]
    fn all_reduce_empty_buffers() {
        let mut bufs: Vec<Vec<f32>> = vec![];
        assert!(AllReduceOp::all_reduce(&mut bufs, ReduceOp::Sum).is_err());
    }

    #[test]
    fn all_reduce_mismatched_lengths() {
        let mut bufs = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(AllReduceOp::all_reduce(&mut bufs, ReduceOp::Sum).is_err());
    }

    // ---------------------------------------------------------------
    // AllReduceOp — transfer estimation
    // ---------------------------------------------------------------

    #[test]
    fn transfer_bytes_single_device() {
        assert_eq!(AllReduceOp::estimated_transfer_bytes(100, 1), 0);
    }

    #[test]
    fn transfer_bytes_two_devices() {
        // 2*(2-1)/2 * 100*4 = 1*400 = 400
        assert_eq!(AllReduceOp::estimated_transfer_bytes(100, 2), 400);
    }

    #[test]
    fn transfer_bytes_four_devices() {
        // 2*(4-1)/4 * 256*4 = 6/4 * 1024 = 1536
        assert_eq!(AllReduceOp::estimated_transfer_bytes(256, 4), 1536);
    }

    // ---------------------------------------------------------------
    // PipelineScheduler — stage assignment
    // ---------------------------------------------------------------

    #[test]
    fn assign_stages_even() {
        let stages = PipelineScheduler::assign_stages(12, 4).unwrap();
        assert_eq!(stages.len(), 4);
        assert_eq!(stages[0].layer_start, 0);
        assert_eq!(stages[0].layer_end, 2);
        assert_eq!(stages[1].layer_start, 3);
        assert_eq!(stages[1].layer_end, 5);
        assert_eq!(stages[3].layer_end, 11);
    }

    #[test]
    fn assign_stages_uneven() {
        let stages = PipelineScheduler::assign_stages(10, 3).unwrap();
        // 10/3 = 3 base, remainder 1
        assert_eq!(stages[0].num_layers(), 4); // 3+1
        assert_eq!(stages[1].num_layers(), 3);
        assert_eq!(stages[2].num_layers(), 3);
        let total: usize = stages.iter().map(|s| s.num_layers()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn assign_stages_single_device() {
        let stages = PipelineScheduler::assign_stages(8, 1).unwrap();
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].num_layers(), 8);
    }

    #[test]
    fn assign_stages_more_devices_than_layers() {
        let stages = PipelineScheduler::assign_stages(2, 5).unwrap();
        assert_eq!(stages.len(), 5);
        let _active: Vec<_> = stages.iter().filter(|s| s.num_layers() > 0).collect();
        // Only 2 stages are active (or at least the total layers assigned = 2).
        // Extra stages have layer_start == layer_end == 0 (placeholder).
        let total_from_active: usize = stages.iter().take(2).map(|s| s.num_layers()).sum();
        assert_eq!(total_from_active, 2);
    }

    #[test]
    fn assign_stages_zero_devices_error() {
        assert!(PipelineScheduler::assign_stages(5, 0).is_err());
    }

    #[test]
    fn assign_stages_zero_layers() {
        let stages = PipelineScheduler::assign_stages(0, 3).unwrap();
        assert_eq!(stages.len(), 3);
    }

    // ---------------------------------------------------------------
    // PipelineScheduler — GPipe schedule
    // ---------------------------------------------------------------

    #[test]
    fn gpipe_basic() {
        let sched = PipelineScheduler::gpipe_schedule(3, 2);
        // total_steps = 3 + 2 - 1 = 4
        assert_eq!(sched.len(), 4);
        assert_eq!(sched[0].len(), 3);

        // t=0: stage 0 gets mb0, others bubble
        assert_eq!(sched[0][0], PipelineAction::Forward { stage: 0, micro_batch: 0 });
        assert_eq!(sched[0][1], PipelineAction::Bubble { stage: 1 });
        assert_eq!(sched[0][2], PipelineAction::Bubble { stage: 2 });

        // t=1: stage 0 gets mb1, stage 1 gets mb0
        assert_eq!(sched[1][0], PipelineAction::Forward { stage: 0, micro_batch: 1 });
        assert_eq!(sched[1][1], PipelineAction::Forward { stage: 1, micro_batch: 0 });
    }

    #[test]
    fn gpipe_single_stage() {
        let sched = PipelineScheduler::gpipe_schedule(1, 4);
        assert_eq!(sched.len(), 4);
        for (t, step) in sched.iter().enumerate() {
            assert_eq!(step[0], PipelineAction::Forward { stage: 0, micro_batch: t });
        }
    }

    #[test]
    fn gpipe_single_microbatch() {
        let sched = PipelineScheduler::gpipe_schedule(3, 1);
        assert_eq!(sched.len(), 3);
        for (t, step) in sched.iter().enumerate() {
            assert_eq!(step[t], PipelineAction::Forward { stage: t, micro_batch: 0 });
        }
    }

    #[test]
    fn gpipe_empty() {
        assert!(PipelineScheduler::gpipe_schedule(0, 5).is_empty());
        assert!(PipelineScheduler::gpipe_schedule(3, 0).is_empty());
    }

    // ---------------------------------------------------------------
    // PipelineScheduler — bubble ratio
    // ---------------------------------------------------------------

    #[test]
    fn bubble_ratio_single_stage() {
        assert!((PipelineScheduler::bubble_ratio(1, 10) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn bubble_ratio_known_value() {
        // 4 stages, 4 micro-batches: total 7*4 = 28, useful 16, bubble 12/28
        let br = PipelineScheduler::bubble_ratio(4, 4);
        let expected = 12.0 / 28.0;
        assert!((br - expected).abs() < 1e-9);
    }

    #[test]
    fn bubble_ratio_more_microbatches_lower() {
        let br4 = PipelineScheduler::bubble_ratio(4, 4);
        let br16 = PipelineScheduler::bubble_ratio(4, 16);
        assert!(br16 < br4);
    }

    #[test]
    fn bubble_ratio_zero() {
        assert!((PipelineScheduler::bubble_ratio(0, 5) - 1.0).abs() < 1e-9);
    }

    // ---------------------------------------------------------------
    // DataParallelBatch
    // ---------------------------------------------------------------

    #[test]
    fn data_parallel_split_even() {
        let batch = make_matrix(4, 3);
        let parts = DataParallelBatch::split(&batch, 4, 3, 2).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].len(), 6);
        assert_eq!(parts[1].len(), 6);
    }

    #[test]
    fn data_parallel_split_uneven() {
        let batch = make_matrix(5, 2);
        let parts = DataParallelBatch::split(&batch, 5, 2, 3).unwrap();
        assert_eq!(parts[0].len(), 4); // 2 rows
        assert_eq!(parts[1].len(), 4); // 2 rows
        assert_eq!(parts[2].len(), 2); // 1 row
    }

    #[test]
    fn data_parallel_split_gather_identity() {
        let batch = make_matrix(6, 4);
        let parts = DataParallelBatch::split(&batch, 6, 4, 3).unwrap();
        let gathered = DataParallelBatch::gather(&parts);
        assert_eq!(gathered, batch);
    }

    #[test]
    fn data_parallel_split_single_device() {
        let batch = make_matrix(3, 5);
        let parts = DataParallelBatch::split(&batch, 3, 5, 1).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], batch);
    }

    #[test]
    fn data_parallel_bad_length() {
        assert!(DataParallelBatch::split(&[1.0; 7], 3, 3, 2).is_err());
    }

    #[test]
    fn data_parallel_zero_devices() {
        assert!(DataParallelBatch::split(&[1.0; 6], 2, 3, 0).is_err());
    }

    #[test]
    fn data_parallel_gather_empty() {
        let result = DataParallelBatch::gather(&[]);
        assert!(result.is_empty());
    }

    // ---------------------------------------------------------------
    // ParallelStats
    // ---------------------------------------------------------------

    #[test]
    fn stats_tensor_parallel() {
        let stats = ParallelStats::for_tensor_parallel(4, 1024);
        assert_eq!(stats.mode, ParallelismMode::TensorParallel);
        assert_eq!(stats.device_utils.len(), 4);
        assert!(stats.total_comm_bytes > 0);
        assert!((stats.bubble_ratio - 0.0).abs() < 1e-9);
    }

    #[test]
    fn stats_pipeline_parallel() {
        let stats = ParallelStats::for_pipeline_parallel(4, 8);
        assert_eq!(stats.mode, ParallelismMode::PipelineParallel);
        assert!(stats.bubble_ratio > 0.0);
        assert!(stats.bubble_ratio < 1.0);
    }

    #[test]
    fn stats_data_parallel() {
        let stats = ParallelStats::for_data_parallel(2, 512);
        assert_eq!(stats.mode, ParallelismMode::DataParallel);
        assert!(stats.total_comm_bytes > 0);
    }

    #[test]
    fn stats_mean_utilization() {
        let stats = ParallelStats {
            mode: ParallelismMode::TensorParallel,
            device_utils: vec![
                DeviceUtilization { device_id: 0, compute_fraction: 0.8, comm_bytes: 0 },
                DeviceUtilization { device_id: 1, compute_fraction: 0.6, comm_bytes: 0 },
            ],
            bubble_ratio: 0.0,
            total_comm_bytes: 0,
        };
        assert!((stats.mean_utilization() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn stats_mean_utilization_empty() {
        let stats = ParallelStats {
            mode: ParallelismMode::DataParallel,
            device_utils: vec![],
            bubble_ratio: 0.0,
            total_comm_bytes: 0,
        };
        assert!((stats.mean_utilization() - 0.0).abs() < 1e-9);
    }

    // ---------------------------------------------------------------
    // Weighted stage assignment (heterogeneous)
    // ---------------------------------------------------------------

    #[test]
    fn weighted_assign_proportional() {
        // 16 GB and 48 GB → 1:3 ratio → 4 and 12 layers
        let stages =
            PipelineScheduler::assign_stages_weighted(16, &[16_000_000_000, 48_000_000_000])
                .unwrap();
        let total: usize = stages.iter().map(|s| s.num_layers()).sum();
        assert_eq!(total, 16);
        // GPU with 3× memory should get ~3× layers
        assert!(stages[1].num_layers() > stages[0].num_layers());
    }

    #[test]
    fn weighted_assign_equal() {
        let stages =
            PipelineScheduler::assign_stages_weighted(10, &[100, 100, 100, 100, 100]).unwrap();
        for s in &stages {
            assert_eq!(s.num_layers(), 2);
        }
    }

    #[test]
    fn weighted_assign_empty_budgets() {
        assert!(PipelineScheduler::assign_stages_weighted(10, &[]).is_err());
    }

    #[test]
    fn weighted_assign_zero_total() {
        assert!(PipelineScheduler::assign_stages_weighted(10, &[0, 0]).is_err());
    }

    #[test]
    fn weighted_assign_single_device() {
        let stages = PipelineScheduler::assign_stages_weighted(8, &[1000]).unwrap();
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].num_layers(), 8);
    }

    // ---------------------------------------------------------------
    // Property-style tests
    // ---------------------------------------------------------------

    #[test]
    fn property_row_split_reconstruct_sizes() {
        for rows in 1..=10 {
            for cols in 1..=6 {
                for n in 1..=5 {
                    let m = make_matrix(rows, cols);
                    let shards =
                        TensorSplitter::split(&m, rows, cols, n, SplitStrategy::Row).unwrap();
                    let (recon, r, c) =
                        TensorSplitter::reconstruct(&shards, SplitStrategy::Row).unwrap();
                    assert_eq!(r, rows);
                    assert_eq!(c, cols);
                    assert_eq!(recon, m);
                }
            }
        }
    }

    #[test]
    fn property_col_split_reconstruct_sizes() {
        for rows in 1..=8 {
            for cols in 1..=8 {
                for n in 1..=5 {
                    let m = make_matrix(rows, cols);
                    let shards =
                        TensorSplitter::split(&m, rows, cols, n, SplitStrategy::Column).unwrap();
                    let (recon, r, c) =
                        TensorSplitter::reconstruct(&shards, SplitStrategy::Column).unwrap();
                    assert_eq!(r, rows);
                    assert_eq!(c, cols);
                    assert_eq!(recon, m);
                }
            }
        }
    }

    #[test]
    fn property_data_parallel_split_gather() {
        for batch_size in 1..=10 {
            for feat in 1..=5 {
                for n in 1..=4 {
                    let batch = make_matrix(batch_size, feat);
                    let parts = DataParallelBatch::split(&batch, batch_size, feat, n).unwrap();
                    let gathered = DataParallelBatch::gather(&parts);
                    assert_eq!(gathered, batch);
                }
            }
        }
    }

    #[test]
    fn property_all_reduce_sum_is_commutative() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut bufs1 = vec![a.clone(), b.clone()];
        let mut bufs2 = vec![b, a];
        AllReduceOp::all_reduce(&mut bufs1, ReduceOp::Sum).unwrap();
        AllReduceOp::all_reduce(&mut bufs2, ReduceOp::Sum).unwrap();
        assert_eq!(bufs1[0], bufs2[0]);
    }

    #[test]
    fn property_gpipe_forward_count() {
        for stages in 1..=6 {
            for mbs in 1..=8 {
                let sched = PipelineScheduler::gpipe_schedule(stages, mbs);
                let forwards: usize = sched
                    .iter()
                    .flat_map(|step| step.iter())
                    .filter(|a| matches!(a, PipelineAction::Forward { .. }))
                    .count();
                assert_eq!(forwards, stages * mbs);
            }
        }
    }

    // ---------------------------------------------------------------
    // Communication overhead estimation
    // ---------------------------------------------------------------

    #[test]
    fn comm_overhead_grows_with_devices() {
        let c2 = AllReduceOp::estimated_transfer_bytes(1000, 2);
        let c4 = AllReduceOp::estimated_transfer_bytes(1000, 4);
        let c8 = AllReduceOp::estimated_transfer_bytes(1000, 8);
        assert!(c4 > c2);
        assert!(c8 > c4);
    }

    #[test]
    fn comm_overhead_zero_elements() {
        assert_eq!(AllReduceOp::estimated_transfer_bytes(0, 4), 0);
    }

    // ---------------------------------------------------------------
    // PipelineStage helpers
    // ---------------------------------------------------------------

    #[test]
    fn pipeline_stage_num_layers() {
        let s = PipelineStage { device_id: 0, layer_start: 3, layer_end: 7 };
        assert_eq!(s.num_layers(), 5);
    }

    #[test]
    fn pipeline_stage_single_layer() {
        let s = PipelineStage { device_id: 1, layer_start: 5, layer_end: 5 };
        assert_eq!(s.num_layers(), 1);
    }
}
