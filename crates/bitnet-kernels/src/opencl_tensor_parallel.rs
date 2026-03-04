//! Tensor-parallel communication primitives for multi-GPU inference.
//!
//! Provides collective operations (all-reduce, all-gather, reduce-scatter,
//! broadcast), tensor sharding strategies, communication topologies, and
//! simulated multi-device execution — all with CPU reference implementations
//! so the module compiles and tests without an OpenCL runtime.

use std::fmt;

// ---------------------------------------------------------------------------
// CommunicationTopology
// ---------------------------------------------------------------------------

/// Logical interconnect topology between devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommunicationTopology {
    /// Each device communicates with its two neighbours in a ring.
    Ring,
    /// Binary-tree reduction / broadcast.
    Tree,
    /// 2-D mesh (row × column).
    Mesh,
    /// Every device can talk to every other device directly.
    AllToAll,
}

impl fmt::Display for CommunicationTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ring => write!(f, "ring"),
            Self::Tree => write!(f, "tree"),
            Self::Mesh => write!(f, "mesh"),
            Self::AllToAll => write!(f, "all_to_all"),
        }
    }
}

// ---------------------------------------------------------------------------
// ReduceOp
// ---------------------------------------------------------------------------

/// Element-wise reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sum => write!(f, "sum"),
            Self::Mean => write!(f, "mean"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
        }
    }
}

// ---------------------------------------------------------------------------
// ShardSpec
// ---------------------------------------------------------------------------

/// Describes how a tensor is partitioned across devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShardSpec {
    /// Split columns (dim-1 of a 2-D weight matrix) across devices.
    ColumnParallel,
    /// Split rows (dim-0 of a 2-D weight matrix) across devices.
    RowParallel,
    /// Replicate the full tensor on every device (no sharding).
    Replicated,
}

impl fmt::Display for ShardSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColumnParallel => write!(f, "column_parallel"),
            Self::RowParallel => write!(f, "row_parallel"),
            Self::Replicated => write!(f, "replicated"),
        }
    }
}

impl ShardSpec {
    /// The tensor dimension along which this spec splits.
    /// Returns `None` for `Replicated`.
    pub fn split_dim(&self) -> Option<usize> {
        match self {
            Self::ColumnParallel => Some(1),
            Self::RowParallel => Some(0),
            Self::Replicated => None,
        }
    }
}

// ---------------------------------------------------------------------------
// CommStats
// ---------------------------------------------------------------------------

/// Accumulated statistics for communication operations.
#[derive(Debug, Clone, Default)]
pub struct CommStats {
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Total number of messages sent.
    pub message_count: u64,
    /// Estimated total latency in microseconds.
    pub latency_us: f64,
    /// Per-operation breakdown: `(op_name, bytes, latency_us)`.
    pub operations: Vec<(String, u64, f64)>,
}

impl CommStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one communication operation.
    pub fn record(&mut self, op_name: &str, bytes: u64, latency_us: f64) {
        self.bytes_transferred += bytes;
        self.message_count += 1;
        self.latency_us += latency_us;
        self.operations.push((op_name.to_string(), bytes, latency_us));
    }

    /// Effective bandwidth in GB/s (returns 0 if latency is zero).
    pub fn bandwidth_gbps(&self) -> f64 {
        if self.latency_us <= 0.0 {
            return 0.0;
        }
        // bytes / µs == MB/s ; ÷ 1000 → GB/s
        (self.bytes_transferred as f64) / self.latency_us / 1000.0
    }

    /// Merge another stats object into this one.
    pub fn merge(&mut self, other: &CommStats) {
        self.bytes_transferred += other.bytes_transferred;
        self.message_count += other.message_count;
        self.latency_us += other.latency_us;
        self.operations.extend(other.operations.iter().cloned());
    }
}

impl fmt::Display for CommStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CommStats {{ msgs={}, bytes={}, latency={:.1}µs, bw={:.2} GB/s }}",
            self.message_count,
            self.bytes_transferred,
            self.latency_us,
            self.bandwidth_gbps(),
        )
    }
}

// ---------------------------------------------------------------------------
// SimulatedDevice
// ---------------------------------------------------------------------------

/// A mock compute device that holds a local data buffer.
///
/// Used to test multi-device communication patterns without real GPUs.
#[derive(Debug, Clone)]
pub struct SimulatedDevice {
    /// Zero-based device identifier.
    pub id: usize,
    /// Local data buffer (f32 elements).
    pub data: Vec<f32>,
    /// Accumulated communication statistics.
    pub stats: CommStats,
}

impl SimulatedDevice {
    pub fn new(id: usize, data: Vec<f32>) -> Self {
        Self { id, data, stats: CommStats::new() }
    }

    /// Number of elements in the local buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the local buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl fmt::Display for SimulatedDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device[{}](len={})", self.id, self.data.len())
    }
}

// ---------------------------------------------------------------------------
// AllReduce
// ---------------------------------------------------------------------------

/// All-reduce: every device ends up with the element-wise reduction of all
/// device buffers.
pub struct AllReduce;

impl AllReduce {
    /// Perform all-reduce **sum** across `devices` using `topology`.
    pub fn sum(devices: &mut [SimulatedDevice], topology: CommunicationTopology) {
        Self::reduce(devices, topology, ReduceOp::Sum);
    }

    /// Perform all-reduce **mean** across `devices` using `topology`.
    pub fn mean(devices: &mut [SimulatedDevice], topology: CommunicationTopology) {
        Self::reduce(devices, topology, ReduceOp::Mean);
    }

    /// Generic all-reduce with an arbitrary `ReduceOp`.
    pub fn reduce(devices: &mut [SimulatedDevice], topology: CommunicationTopology, op: ReduceOp) {
        let n = devices.len();
        if n <= 1 {
            return;
        }
        let len = devices[0].data.len();
        assert!(devices.iter().all(|d| d.data.len() == len), "all buffers must have equal length");

        // Compute the reduced result regardless of topology — the topology
        // only affects the simulated message count / cost.
        let mut result = vec![0.0f32; len];
        match op {
            ReduceOp::Sum | ReduceOp::Mean => {
                for d in devices.iter() {
                    for (i, &v) in d.data.iter().enumerate() {
                        result[i] += v;
                    }
                }
                if op == ReduceOp::Mean {
                    let inv = 1.0 / n as f32;
                    for v in &mut result {
                        *v *= inv;
                    }
                }
            }
            ReduceOp::Max => {
                result.copy_from_slice(&devices[0].data);
                for d in &devices[1..] {
                    for (i, &v) in d.data.iter().enumerate() {
                        if v > result[i] {
                            result[i] = v;
                        }
                    }
                }
            }
            ReduceOp::Min => {
                result.copy_from_slice(&devices[0].data);
                for d in &devices[1..] {
                    for (i, &v) in d.data.iter().enumerate() {
                        if v < result[i] {
                            result[i] = v;
                        }
                    }
                }
            }
        }

        // Record stats based on topology pattern.
        let bytes_per_elem = std::mem::size_of::<f32>() as u64;
        let total_bytes = len as u64 * bytes_per_elem;
        let (msgs, lat) = topology_cost(n, total_bytes, topology);

        for d in devices.iter_mut() {
            d.data.copy_from_slice(&result);
            d.stats.record("allreduce", total_bytes * (n as u64 - 1), lat);
            d.stats.message_count += msgs.saturating_sub(1); // first was counted by record
        }
    }
}

// ---------------------------------------------------------------------------
// AllGather
// ---------------------------------------------------------------------------

/// All-gather: every device contributes its local shard; after the operation
/// every device holds the full concatenated tensor.
pub struct AllGather;

impl AllGather {
    /// Run all-gather across `devices`. Each device's data is treated as one
    /// shard; the concatenation (in device-id order) becomes the result on
    /// every device.
    pub fn run(devices: &mut [SimulatedDevice], topology: CommunicationTopology) {
        let n = devices.len();
        if n <= 1 {
            return;
        }
        let full: Vec<f32> = devices.iter().flat_map(|d| d.data.iter().copied()).collect();
        let bytes_per_elem = std::mem::size_of::<f32>() as u64;
        let total_bytes = full.len() as u64 * bytes_per_elem;
        let (msgs, lat) = topology_cost(n, total_bytes, topology);

        for d in devices.iter_mut() {
            d.data = full.clone();
            d.stats.record("allgather", total_bytes, lat);
            d.stats.message_count += msgs.saturating_sub(1);
        }
    }
}

// ---------------------------------------------------------------------------
// ReduceScatter
// ---------------------------------------------------------------------------

/// Reduce-scatter: element-wise reduce, then scatter non-overlapping chunks
/// so each device receives a unique shard of the result.
pub struct ReduceScatter;

impl ReduceScatter {
    /// Run reduce-scatter (sum) across `devices`.
    pub fn sum(devices: &mut [SimulatedDevice], topology: CommunicationTopology) {
        Self::run(devices, topology, ReduceOp::Sum);
    }

    pub fn run(devices: &mut [SimulatedDevice], topology: CommunicationTopology, op: ReduceOp) {
        let n = devices.len();
        if n <= 1 {
            return;
        }
        let len = devices[0].data.len();
        assert!(devices.iter().all(|d| d.data.len() == len));

        // Reduce.
        let mut total = vec![0.0f32; len];
        match op {
            ReduceOp::Sum | ReduceOp::Mean => {
                for d in devices.iter() {
                    for (i, &v) in d.data.iter().enumerate() {
                        total[i] += v;
                    }
                }
                if op == ReduceOp::Mean {
                    let inv = 1.0 / n as f32;
                    for v in &mut total {
                        *v *= inv;
                    }
                }
            }
            ReduceOp::Max => {
                total.copy_from_slice(&devices[0].data);
                for d in &devices[1..] {
                    for (i, &v) in d.data.iter().enumerate() {
                        if v > total[i] {
                            total[i] = v;
                        }
                    }
                }
            }
            ReduceOp::Min => {
                total.copy_from_slice(&devices[0].data);
                for d in &devices[1..] {
                    for (i, &v) in d.data.iter().enumerate() {
                        if v < total[i] {
                            total[i] = v;
                        }
                    }
                }
            }
        }

        // Scatter equal-sized chunks.
        let chunk_size = len.div_ceil(n);
        let bytes_per_elem = std::mem::size_of::<f32>() as u64;
        let total_bytes = len as u64 * bytes_per_elem;
        let (msgs, lat) = topology_cost(n, total_bytes, topology);

        for (i, d) in devices.iter_mut().enumerate() {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(len);
            d.data = total[start..end].to_vec();
            d.stats.record("reduce_scatter", total_bytes, lat);
            d.stats.message_count += msgs.saturating_sub(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Broadcast
// ---------------------------------------------------------------------------

/// Broadcast: one device sends its buffer to all others.
pub struct Broadcast;

impl Broadcast {
    /// Broadcast the data on device `root` to every other device.
    pub fn run(devices: &mut [SimulatedDevice], root: usize, topology: CommunicationTopology) {
        let n = devices.len();
        assert!(root < n, "root device index out of range");
        if n <= 1 {
            return;
        }
        let source = devices[root].data.clone();
        let bytes_per_elem = std::mem::size_of::<f32>() as u64;
        let total_bytes = source.len() as u64 * bytes_per_elem;
        let (msgs, lat) = topology_cost(n, total_bytes, topology);

        for d in devices.iter_mut() {
            d.data = source.clone();
            d.stats.record("broadcast", total_bytes, lat);
            d.stats.message_count += msgs.saturating_sub(1);
        }
    }
}

// ---------------------------------------------------------------------------
// TensorShardManager
// ---------------------------------------------------------------------------

/// Manages sharding a tensor across simulated devices according to a
/// [`ShardSpec`].
#[derive(Debug)]
pub struct TensorShardManager {
    pub num_devices: usize,
    pub spec: ShardSpec,
}

impl TensorShardManager {
    pub fn new(num_devices: usize, spec: ShardSpec) -> Self {
        assert!(num_devices > 0);
        Self { num_devices, spec }
    }

    /// Shard a 2-D row-major tensor (`rows × cols`) across devices.
    ///
    /// Returns one `Vec<f32>` per device. For `Replicated` every device gets
    /// a full copy.
    pub fn shard(&self, data: &[f32], rows: usize, cols: usize) -> Vec<Vec<f32>> {
        assert_eq!(data.len(), rows * cols, "data length must equal rows*cols");
        let n = self.num_devices;

        match self.spec {
            ShardSpec::Replicated => (0..n).map(|_| data.to_vec()).collect(),
            ShardSpec::ColumnParallel => {
                let chunk = cols.div_ceil(n);
                (0..n)
                    .map(|dev| {
                        let c_start = dev * chunk;
                        let c_end = (c_start + chunk).min(cols);
                        let mut shard = Vec::with_capacity(rows * (c_end - c_start));
                        for r in 0..rows {
                            let row_start = r * cols;
                            shard.extend_from_slice(&data[row_start + c_start..row_start + c_end]);
                        }
                        shard
                    })
                    .collect()
            }
            ShardSpec::RowParallel => {
                let chunk = rows.div_ceil(n);
                (0..n)
                    .map(|dev| {
                        let r_start = dev * chunk;
                        let r_end = (r_start + chunk).min(rows);
                        data[r_start * cols..r_end * cols].to_vec()
                    })
                    .collect()
            }
        }
    }

    /// Gather shards back into the full tensor.
    pub fn gather(&self, shards: &[Vec<f32>], rows: usize, cols: usize) -> Vec<f32> {
        let n = self.num_devices;
        assert_eq!(shards.len(), n);

        match self.spec {
            ShardSpec::Replicated => shards[0].clone(),
            ShardSpec::ColumnParallel => {
                let chunk = cols.div_ceil(n);
                let mut out = vec![0.0f32; rows * cols];
                for (dev, shard) in shards.iter().enumerate() {
                    let c_start = dev * chunk;
                    let c_end = (c_start + chunk).min(cols);
                    let w = c_end - c_start;
                    for r in 0..rows {
                        let dst_start = r * cols + c_start;
                        let src_start = r * w;
                        out[dst_start..dst_start + w]
                            .copy_from_slice(&shard[src_start..src_start + w]);
                    }
                }
                out
            }
            ShardSpec::RowParallel => {
                let mut out = Vec::with_capacity(rows * cols);
                for shard in shards {
                    out.extend_from_slice(shard);
                }
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Topology cost model
// ---------------------------------------------------------------------------

/// Assumed per-message overhead (µs) and bandwidth (bytes/µs ≈ 12.8 GB/s).
const MSG_OVERHEAD_US: f64 = 5.0;
const BW_BYTES_PER_US: f64 = 12800.0;

/// Returns `(message_count, estimated_latency_us)` for a collective of
/// `total_bytes` across `n` devices under `topology`.
fn topology_cost(n: usize, total_bytes: u64, topology: CommunicationTopology) -> (u64, f64) {
    let transfer = total_bytes as f64 / BW_BYTES_PER_US;
    match topology {
        CommunicationTopology::Ring => {
            let steps = 2 * (n as u64 - 1);
            (steps, steps as f64 * MSG_OVERHEAD_US + 2.0 * transfer)
        }
        CommunicationTopology::Tree => {
            let depth = (n as f64).log2().ceil() as u64;
            let steps = 2 * depth; // reduce + broadcast
            (steps, steps as f64 * MSG_OVERHEAD_US + 2.0 * transfer)
        }
        CommunicationTopology::Mesh => {
            // Approximate: sqrt(n) steps in each mesh dimension.
            let side = (n as f64).sqrt().ceil() as u64;
            let steps = 2 * side;
            (steps, steps as f64 * MSG_OVERHEAD_US + 2.0 * transfer)
        }
        CommunicationTopology::AllToAll => {
            let steps = n as u64 - 1;
            (steps, steps as f64 * MSG_OVERHEAD_US + (n as f64 - 1.0) * transfer)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    /// Create `n` devices each holding `len` elements: device i gets
    /// `[i*len, i*len+1, …, i*len+len-1]` cast to f32.
    fn make_devices(n: usize, len: usize) -> Vec<SimulatedDevice> {
        (0..n)
            .map(|i| {
                let data: Vec<f32> = (0..len).map(|j| (i * len + j) as f32).collect();
                SimulatedDevice::new(i, data)
            })
            .collect()
    }

    fn expected_sum(devices: &[SimulatedDevice]) -> Vec<f32> {
        let len = devices[0].data.len();
        let mut sum = vec![0.0f32; len];
        for d in devices {
            for (i, &v) in d.data.iter().enumerate() {
                sum[i] += v;
            }
        }
        sum
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    // ── CommunicationTopology ───────────────────────────────────────────

    #[test]
    fn topology_display() {
        assert_eq!(CommunicationTopology::Ring.to_string(), "ring");
        assert_eq!(CommunicationTopology::Tree.to_string(), "tree");
        assert_eq!(CommunicationTopology::Mesh.to_string(), "mesh");
        assert_eq!(CommunicationTopology::AllToAll.to_string(), "all_to_all");
    }

    #[test]
    fn topology_equality() {
        assert_eq!(CommunicationTopology::Ring, CommunicationTopology::Ring);
        assert_ne!(CommunicationTopology::Ring, CommunicationTopology::Tree);
    }

    // ── ReduceOp ────────────────────────────────────────────────────────

    #[test]
    fn reduce_op_display() {
        assert_eq!(ReduceOp::Sum.to_string(), "sum");
        assert_eq!(ReduceOp::Mean.to_string(), "mean");
        assert_eq!(ReduceOp::Max.to_string(), "max");
        assert_eq!(ReduceOp::Min.to_string(), "min");
    }

    // ── ShardSpec ───────────────────────────────────────────────────────

    #[test]
    fn shard_spec_display() {
        assert_eq!(ShardSpec::ColumnParallel.to_string(), "column_parallel");
        assert_eq!(ShardSpec::RowParallel.to_string(), "row_parallel");
        assert_eq!(ShardSpec::Replicated.to_string(), "replicated");
    }

    #[test]
    fn shard_spec_split_dim() {
        assert_eq!(ShardSpec::ColumnParallel.split_dim(), Some(1));
        assert_eq!(ShardSpec::RowParallel.split_dim(), Some(0));
        assert_eq!(ShardSpec::Replicated.split_dim(), None);
    }

    // ── CommStats ───────────────────────────────────────────────────────

    #[test]
    fn comm_stats_default_empty() {
        let s = CommStats::new();
        assert_eq!(s.bytes_transferred, 0);
        assert_eq!(s.message_count, 0);
        assert_eq!(s.latency_us, 0.0);
        assert!(s.operations.is_empty());
    }

    #[test]
    fn comm_stats_record_accumulates() {
        let mut s = CommStats::new();
        s.record("allreduce", 1024, 10.0);
        s.record("broadcast", 512, 5.0);
        assert_eq!(s.bytes_transferred, 1536);
        assert_eq!(s.message_count, 2);
        assert!((s.latency_us - 15.0).abs() < 1e-6);
        assert_eq!(s.operations.len(), 2);
    }

    #[test]
    fn comm_stats_bandwidth_zero_latency() {
        let s = CommStats::new();
        assert_eq!(s.bandwidth_gbps(), 0.0);
    }

    #[test]
    fn comm_stats_bandwidth_positive() {
        let mut s = CommStats::new();
        // 1 MB in 100 µs → 10 GB/s
        s.record("test", 1_000_000, 100.0);
        let bw = s.bandwidth_gbps();
        assert!((bw - 10.0).abs() < 0.01, "bw={bw}");
    }

    #[test]
    fn comm_stats_merge() {
        let mut a = CommStats::new();
        a.record("op1", 100, 1.0);
        let mut b = CommStats::new();
        b.record("op2", 200, 2.0);
        a.merge(&b);
        assert_eq!(a.bytes_transferred, 300);
        assert_eq!(a.message_count, 2);
        assert!((a.latency_us - 3.0).abs() < 1e-6);
        assert_eq!(a.operations.len(), 2);
    }

    #[test]
    fn comm_stats_display() {
        let mut s = CommStats::new();
        s.record("test", 1000, 50.0);
        let text = s.to_string();
        assert!(text.contains("msgs=1"));
        assert!(text.contains("bytes=1000"));
    }

    // ── SimulatedDevice ─────────────────────────────────────────────────

    #[test]
    fn simulated_device_basic() {
        let d = SimulatedDevice::new(3, vec![1.0, 2.0, 3.0]);
        assert_eq!(d.id, 3);
        assert_eq!(d.len(), 3);
        assert!(!d.is_empty());
    }

    #[test]
    fn simulated_device_empty() {
        let d = SimulatedDevice::new(0, vec![]);
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn simulated_device_display() {
        let d = SimulatedDevice::new(2, vec![0.0; 10]);
        let text = d.to_string();
        assert!(text.contains("Device[2]"));
        assert!(text.contains("len=10"));
    }

    // ── AllReduce sum — 2 devices ───────────────────────────────────────

    #[test]
    fn allreduce_sum_2_devices_ring() {
        let mut devs = make_devices(2, 4);
        let exp = expected_sum(&devs);
        AllReduce::sum(&mut devs, CommunicationTopology::Ring);
        for d in &devs {
            assert_eq!(d.data, exp);
        }
    }

    #[test]
    fn allreduce_sum_2_devices_tree() {
        let mut devs = make_devices(2, 4);
        let exp = expected_sum(&devs);
        AllReduce::sum(&mut devs, CommunicationTopology::Tree);
        for d in &devs {
            assert_eq!(d.data, exp);
        }
    }

    #[test]
    fn allreduce_sum_2_devices_all_to_all() {
        let mut devs = make_devices(2, 4);
        let exp = expected_sum(&devs);
        AllReduce::sum(&mut devs, CommunicationTopology::AllToAll);
        for d in &devs {
            assert_eq!(d.data, exp);
        }
    }

    // ── AllReduce sum — 4 devices ───────────────────────────────────────

    #[test]
    fn allreduce_sum_4_devices_ring() {
        let mut devs = make_devices(4, 8);
        let exp = expected_sum(&devs);
        AllReduce::sum(&mut devs, CommunicationTopology::Ring);
        for d in &devs {
            assert_eq!(d.data, exp);
        }
    }

    #[test]
    fn allreduce_sum_4_devices_tree() {
        let mut devs = make_devices(4, 8);
        let exp = expected_sum(&devs);
        AllReduce::sum(&mut devs, CommunicationTopology::Tree);
        for d in &devs {
            assert_eq!(d.data, exp);
        }
    }

    #[test]
    fn allreduce_sum_4_devices_mesh() {
        let mut devs = make_devices(4, 8);
        let exp = expected_sum(&devs);
        AllReduce::sum(&mut devs, CommunicationTopology::Mesh);
        for d in &devs {
            assert_eq!(d.data, exp);
        }
    }

    #[test]
    fn allreduce_sum_4_devices_all_to_all() {
        let mut devs = make_devices(4, 8);
        let exp = expected_sum(&devs);
        AllReduce::sum(&mut devs, CommunicationTopology::AllToAll);
        for d in &devs {
            assert_eq!(d.data, exp);
        }
    }

    // ── AllReduce sum — 8 devices ───────────────────────────────────────

    #[test]
    fn allreduce_sum_8_devices_ring() {
        let mut devs = make_devices(8, 16);
        let exp = expected_sum(&devs);
        AllReduce::sum(&mut devs, CommunicationTopology::Ring);
        for d in &devs {
            assert_eq!(d.data, exp);
        }
    }

    #[test]
    fn allreduce_sum_8_devices_tree() {
        let mut devs = make_devices(8, 16);
        let exp = expected_sum(&devs);
        AllReduce::sum(&mut devs, CommunicationTopology::Tree);
        for d in &devs {
            assert_eq!(d.data, exp);
        }
    }

    #[test]
    fn allreduce_sum_8_devices_all_to_all() {
        let mut devs = make_devices(8, 16);
        let exp = expected_sum(&devs);
        AllReduce::sum(&mut devs, CommunicationTopology::AllToAll);
        for d in &devs {
            assert_eq!(d.data, exp);
        }
    }

    // ── AllReduce mean ──────────────────────────────────────────────────

    #[test]
    fn allreduce_mean_2_devices() {
        let mut devs = make_devices(2, 4);
        let sum = expected_sum(&devs);
        AllReduce::mean(&mut devs, CommunicationTopology::Ring);
        let expected: Vec<f32> = sum.iter().map(|v| v / 2.0).collect();
        for d in &devs {
            assert!(approx_eq(&d.data, &expected, 1e-5));
        }
    }

    #[test]
    fn allreduce_mean_4_devices() {
        let mut devs = make_devices(4, 8);
        let sum = expected_sum(&devs);
        AllReduce::mean(&mut devs, CommunicationTopology::Tree);
        let expected: Vec<f32> = sum.iter().map(|v| v / 4.0).collect();
        for d in &devs {
            assert!(approx_eq(&d.data, &expected, 1e-5));
        }
    }

    // ── AllReduce max / min ─────────────────────────────────────────────

    #[test]
    fn allreduce_max_correctness() {
        let mut devs = vec![
            SimulatedDevice::new(0, vec![1.0, 5.0, 3.0]),
            SimulatedDevice::new(1, vec![4.0, 2.0, 6.0]),
        ];
        AllReduce::reduce(&mut devs, CommunicationTopology::Ring, ReduceOp::Max);
        for d in &devs {
            assert_eq!(d.data, vec![4.0, 5.0, 6.0]);
        }
    }

    #[test]
    fn allreduce_min_correctness() {
        let mut devs = vec![
            SimulatedDevice::new(0, vec![1.0, 5.0, 3.0]),
            SimulatedDevice::new(1, vec![4.0, 2.0, 6.0]),
        ];
        AllReduce::reduce(&mut devs, CommunicationTopology::Ring, ReduceOp::Min);
        for d in &devs {
            assert_eq!(d.data, vec![1.0, 2.0, 3.0]);
        }
    }

    // ── AllReduce single device (no-op) ─────────────────────────────────

    #[test]
    fn allreduce_single_device_noop() {
        let mut devs = vec![SimulatedDevice::new(0, vec![1.0, 2.0, 3.0])];
        AllReduce::sum(&mut devs, CommunicationTopology::Ring);
        assert_eq!(devs[0].data, vec![1.0, 2.0, 3.0]);
        assert_eq!(devs[0].stats.message_count, 0);
    }

    // ── AllReduce records stats ─────────────────────────────────────────

    #[test]
    fn allreduce_records_stats() {
        let mut devs = make_devices(4, 8);
        AllReduce::sum(&mut devs, CommunicationTopology::Ring);
        for d in &devs {
            assert!(d.stats.bytes_transferred > 0);
            assert!(d.stats.message_count > 0);
            assert!(d.stats.latency_us > 0.0);
        }
    }

    // ── Property: allreduce(sum) / n == mean ────────────────────────────

    #[test]
    fn property_sum_div_n_equals_mean() {
        for n in [2, 3, 4, 5, 8] {
            let mut sum_devs = make_devices(n, 16);
            let mut mean_devs = make_devices(n, 16);
            AllReduce::sum(&mut sum_devs, CommunicationTopology::Ring);
            AllReduce::mean(&mut mean_devs, CommunicationTopology::Ring);

            let divided: Vec<f32> = sum_devs[0].data.iter().map(|v| v / n as f32).collect();
            assert!(approx_eq(&divided, &mean_devs[0].data, 1e-4), "failed for n={n}");
        }
    }

    // ── AllGather ────────────────────────────────────────────────────────

    #[test]
    fn allgather_2_devices() {
        let mut devs =
            vec![SimulatedDevice::new(0, vec![1.0, 2.0]), SimulatedDevice::new(1, vec![3.0, 4.0])];
        AllGather::run(&mut devs, CommunicationTopology::Ring);
        for d in &devs {
            assert_eq!(d.data, vec![1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn allgather_4_devices() {
        let mut devs = vec![
            SimulatedDevice::new(0, vec![10.0]),
            SimulatedDevice::new(1, vec![20.0]),
            SimulatedDevice::new(2, vec![30.0]),
            SimulatedDevice::new(3, vec![40.0]),
        ];
        AllGather::run(&mut devs, CommunicationTopology::Tree);
        for d in &devs {
            assert_eq!(d.data, vec![10.0, 20.0, 30.0, 40.0]);
        }
    }

    #[test]
    fn allgather_uneven_shards() {
        let mut devs = vec![
            SimulatedDevice::new(0, vec![1.0, 2.0, 3.0]),
            SimulatedDevice::new(1, vec![4.0, 5.0]),
        ];
        AllGather::run(&mut devs, CommunicationTopology::AllToAll);
        for d in &devs {
            assert_eq!(d.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        }
    }

    #[test]
    fn allgather_single_device_noop() {
        let mut devs = vec![SimulatedDevice::new(0, vec![42.0])];
        AllGather::run(&mut devs, CommunicationTopology::Ring);
        assert_eq!(devs[0].data, vec![42.0]);
    }

    #[test]
    fn allgather_reconstructs_full_tensor() {
        // Shard a tensor across 3 devices then all-gather to reconstruct.
        let full: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let chunk = 4;
        let mut devs: Vec<SimulatedDevice> = (0..3)
            .map(|i| {
                let start = i * chunk;
                let end = start + chunk;
                SimulatedDevice::new(i, full[start..end].to_vec())
            })
            .collect();
        AllGather::run(&mut devs, CommunicationTopology::Ring);
        for d in &devs {
            assert_eq!(d.data, full);
        }
    }

    // ── ReduceScatter ───────────────────────────────────────────────────

    #[test]
    fn reduce_scatter_2_devices() {
        let mut devs = vec![
            SimulatedDevice::new(0, vec![1.0, 2.0, 3.0, 4.0]),
            SimulatedDevice::new(1, vec![5.0, 6.0, 7.0, 8.0]),
        ];
        ReduceScatter::sum(&mut devs, CommunicationTopology::Ring);
        assert_eq!(devs[0].data, vec![6.0, 8.0]);
        assert_eq!(devs[1].data, vec![10.0, 12.0]);
    }

    #[test]
    fn reduce_scatter_3_devices() {
        let mut devs = vec![
            SimulatedDevice::new(0, vec![1.0, 2.0, 3.0]),
            SimulatedDevice::new(1, vec![4.0, 5.0, 6.0]),
            SimulatedDevice::new(2, vec![7.0, 8.0, 9.0]),
        ];
        ReduceScatter::sum(&mut devs, CommunicationTopology::Tree);
        // Sum = [12, 15, 18], chunk_size=1
        assert_eq!(devs[0].data, vec![12.0]);
        assert_eq!(devs[1].data, vec![15.0]);
        assert_eq!(devs[2].data, vec![18.0]);
    }

    #[test]
    fn reduce_scatter_single_device_noop() {
        let mut devs = vec![SimulatedDevice::new(0, vec![1.0, 2.0, 3.0])];
        ReduceScatter::sum(&mut devs, CommunicationTopology::Ring);
        assert_eq!(devs[0].data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn reduce_scatter_each_device_gets_correct_shard() {
        let mut devs = make_devices(4, 8);
        let sum = expected_sum(&devs);
        ReduceScatter::sum(&mut devs, CommunicationTopology::AllToAll);
        let chunk_size = 8usize.div_ceil(4);
        for (i, d) in devs.iter().enumerate() {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(8);
            assert_eq!(d.data, sum[start..end]);
        }
    }

    // ── Broadcast ───────────────────────────────────────────────────────

    #[test]
    fn broadcast_from_root_0() {
        let mut devs = vec![
            SimulatedDevice::new(0, vec![10.0, 20.0]),
            SimulatedDevice::new(1, vec![0.0, 0.0]),
            SimulatedDevice::new(2, vec![0.0, 0.0]),
        ];
        Broadcast::run(&mut devs, 0, CommunicationTopology::Tree);
        for d in &devs {
            assert_eq!(d.data, vec![10.0, 20.0]);
        }
    }

    #[test]
    fn broadcast_from_non_zero_root() {
        let mut devs = vec![
            SimulatedDevice::new(0, vec![0.0]),
            SimulatedDevice::new(1, vec![0.0]),
            SimulatedDevice::new(2, vec![99.0]),
        ];
        Broadcast::run(&mut devs, 2, CommunicationTopology::Ring);
        for d in &devs {
            assert_eq!(d.data, vec![99.0]);
        }
    }

    #[test]
    fn broadcast_single_device_noop() {
        let mut devs = vec![SimulatedDevice::new(0, vec![5.0, 6.0])];
        Broadcast::run(&mut devs, 0, CommunicationTopology::Ring);
        assert_eq!(devs[0].data, vec![5.0, 6.0]);
    }

    #[test]
    fn broadcast_records_stats() {
        let mut devs = make_devices(3, 4);
        Broadcast::run(&mut devs, 0, CommunicationTopology::Ring);
        for d in &devs {
            assert!(d.stats.bytes_transferred > 0);
        }
    }

    // ── TensorShardManager — column-parallel ────────────────────────────

    #[test]
    fn column_parallel_shard_even() {
        let mgr = TensorShardManager::new(2, ShardSpec::ColumnParallel);
        // 2×4 matrix → each device gets 2×2
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let shards = mgr.shard(&data, 2, 4);
        assert_eq!(shards.len(), 2);
        // Row 0 cols 0-1, row 1 cols 0-1
        assert_eq!(shards[0], vec![0.0, 1.0, 4.0, 5.0]);
        // Row 0 cols 2-3, row 1 cols 2-3
        assert_eq!(shards[1], vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn column_parallel_shard_uneven() {
        let mgr = TensorShardManager::new(3, ShardSpec::ColumnParallel);
        // 2×5 matrix across 3 devices → chunk=2, sizes: 2, 2, 1
        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let shards = mgr.shard(&data, 2, 5);
        assert_eq!(shards.len(), 3);
        assert_eq!(shards[0], vec![0.0, 1.0, 5.0, 6.0]);
        assert_eq!(shards[1], vec![2.0, 3.0, 7.0, 8.0]);
        assert_eq!(shards[2], vec![4.0, 9.0]);
    }

    #[test]
    fn column_parallel_roundtrip() {
        let mgr = TensorShardManager::new(4, ShardSpec::ColumnParallel);
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let shards = mgr.shard(&data, 3, 8);
        let restored = mgr.gather(&shards, 3, 8);
        assert_eq!(restored, data);
    }

    // ── TensorShardManager — row-parallel ───────────────────────────────

    #[test]
    fn row_parallel_shard_even() {
        let mgr = TensorShardManager::new(2, ShardSpec::RowParallel);
        // 4×3 matrix → each device gets 2 rows
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shards = mgr.shard(&data, 4, 3);
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(shards[1], vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn row_parallel_shard_uneven() {
        let mgr = TensorShardManager::new(3, ShardSpec::RowParallel);
        // 5×2 matrix → chunk=2, sizes: 2, 2, 1
        let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let shards = mgr.shard(&data, 5, 2);
        assert_eq!(shards[0], vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(shards[1], vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(shards[2], vec![8.0, 9.0]);
    }

    #[test]
    fn row_parallel_roundtrip() {
        let mgr = TensorShardManager::new(4, ShardSpec::RowParallel);
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let shards = mgr.shard(&data, 8, 3);
        let restored = mgr.gather(&shards, 8, 3);
        assert_eq!(restored, data);
    }

    // ── TensorShardManager — replicated ─────────────────────────────────

    #[test]
    fn replicated_shard_all_identical() {
        let mgr = TensorShardManager::new(3, ShardSpec::Replicated);
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let shards = mgr.shard(&data, 2, 3);
        for s in &shards {
            assert_eq!(s, &data);
        }
    }

    #[test]
    fn replicated_gather_returns_first() {
        let mgr = TensorShardManager::new(2, ShardSpec::Replicated);
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let shards = mgr.shard(&data, 2, 3);
        let restored = mgr.gather(&shards, 2, 3);
        assert_eq!(restored, data);
    }

    // ── Ring topology communication pattern ─────────────────────────────

    #[test]
    fn ring_topology_message_count() {
        // For n=4 ring all-reduce: 2*(n-1) = 6 steps
        let (msgs, _lat) = topology_cost(4, 1024, CommunicationTopology::Ring);
        assert_eq!(msgs, 6);
    }

    #[test]
    fn ring_topology_latency_positive() {
        let (_msgs, lat) = topology_cost(4, 1024, CommunicationTopology::Ring);
        assert!(lat > 0.0);
    }

    // ── Tree topology communication pattern ─────────────────────────────

    #[test]
    fn tree_topology_message_count() {
        // For n=8 tree: 2*ceil(log2(8)) = 2*3 = 6 steps
        let (msgs, _lat) = topology_cost(8, 1024, CommunicationTopology::Tree);
        assert_eq!(msgs, 6);
    }

    #[test]
    fn tree_topology_fewer_steps_than_ring_for_large_n() {
        let (ring_msgs, _) = topology_cost(16, 1024, CommunicationTopology::Ring);
        let (tree_msgs, _) = topology_cost(16, 1024, CommunicationTopology::Tree);
        assert!(tree_msgs < ring_msgs);
    }

    // ── Mesh topology ───────────────────────────────────────────────────

    #[test]
    fn mesh_topology_steps() {
        // n=4 → sqrt(4)=2 → 2*2 = 4 steps
        let (msgs, _lat) = topology_cost(4, 1024, CommunicationTopology::Mesh);
        assert_eq!(msgs, 4);
    }

    // ── AllToAll topology ───────────────────────────────────────────────

    #[test]
    fn alltoall_topology_steps() {
        // n=4 → n-1 = 3 steps
        let (msgs, _lat) = topology_cost(4, 1024, CommunicationTopology::AllToAll);
        assert_eq!(msgs, 3);
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn allreduce_sum_with_zeros() {
        let mut devs =
            vec![SimulatedDevice::new(0, vec![0.0; 4]), SimulatedDevice::new(1, vec![0.0; 4])];
        AllReduce::sum(&mut devs, CommunicationTopology::Ring);
        for d in &devs {
            assert_eq!(d.data, vec![0.0; 4]);
        }
    }

    #[test]
    fn allreduce_sum_with_negative_values() {
        let mut devs = vec![
            SimulatedDevice::new(0, vec![-1.0, 2.0]),
            SimulatedDevice::new(1, vec![3.0, -4.0]),
        ];
        AllReduce::sum(&mut devs, CommunicationTopology::Ring);
        for d in &devs {
            assert_eq!(d.data, vec![2.0, -2.0]);
        }
    }

    #[test]
    fn allreduce_large_buffer() {
        let len = 1024;
        let mut devs = make_devices(4, len);
        let exp = expected_sum(&devs);
        AllReduce::sum(&mut devs, CommunicationTopology::Ring);
        for d in &devs {
            assert_eq!(d.data, exp);
        }
    }

    // ── Shard + collective round-trip ───────────────────────────────────

    #[test]
    fn column_shard_allgather_roundtrip() {
        let mgr = TensorShardManager::new(2, ShardSpec::ColumnParallel);
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let shards = mgr.shard(&data, 3, 4);
        // Load shards into devices and all-gather
        let mut devs: Vec<SimulatedDevice> =
            shards.into_iter().enumerate().map(|(i, s)| SimulatedDevice::new(i, s)).collect();
        AllGather::run(&mut devs, CommunicationTopology::Ring);
        // After all-gather every device has all column shards concatenated.
        // Verify total element count is correct.
        assert_eq!(devs[0].data.len(), devs[1].data.len());
    }

    #[test]
    fn row_shard_allgather_roundtrip() {
        let mgr = TensorShardManager::new(2, ShardSpec::RowParallel);
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let shards = mgr.shard(&data, 4, 2);
        let mut devs: Vec<SimulatedDevice> =
            shards.into_iter().enumerate().map(|(i, s)| SimulatedDevice::new(i, s)).collect();
        AllGather::run(&mut devs, CommunicationTopology::Ring);
        // Row-parallel shards concatenated = original tensor.
        for d in &devs {
            assert_eq!(d.data, data);
        }
    }

    // ── ReduceScatter + AllGather == AllReduce ───────────────────────────

    #[test]
    fn reduce_scatter_then_allgather_equals_allreduce() {
        let n = 4;
        let len = 8;
        let mut rs_devs = make_devices(n, len);
        let mut ar_devs = make_devices(n, len);

        ReduceScatter::sum(&mut rs_devs, CommunicationTopology::Ring);
        AllGather::run(&mut rs_devs, CommunicationTopology::Ring);

        AllReduce::sum(&mut ar_devs, CommunicationTopology::Ring);

        // After RS+AG every device should hold the full sum.
        for d in &rs_devs {
            assert_eq!(d.data, ar_devs[0].data);
        }
    }

    // ── Topology cost monotonicity ──────────────────────────────────────

    #[test]
    fn latency_increases_with_devices() {
        let (_, l2) = topology_cost(2, 4096, CommunicationTopology::Ring);
        let (_, l8) = topology_cost(8, 4096, CommunicationTopology::Ring);
        assert!(l8 > l2);
    }

    #[test]
    fn latency_increases_with_bytes() {
        let (_, small) = topology_cost(4, 1024, CommunicationTopology::AllToAll);
        let (_, large) = topology_cost(4, 1_048_576, CommunicationTopology::AllToAll);
        assert!(large > small);
    }

    // ── ReduceScatter with max/min ops ──────────────────────────────────

    #[test]
    fn reduce_scatter_max() {
        let mut devs = vec![
            SimulatedDevice::new(0, vec![1.0, 5.0, 3.0, 7.0]),
            SimulatedDevice::new(1, vec![4.0, 2.0, 6.0, 8.0]),
        ];
        ReduceScatter::run(&mut devs, CommunicationTopology::Ring, ReduceOp::Max);
        assert_eq!(devs[0].data, vec![4.0, 5.0]);
        assert_eq!(devs[1].data, vec![6.0, 8.0]);
    }

    // ── Multiple sequential collectives ─────────────────────────────────

    #[test]
    fn sequential_allreduce_accumulates_stats() {
        let mut devs = make_devices(2, 4);
        AllReduce::sum(&mut devs, CommunicationTopology::Ring);
        let bytes_after_1 = devs[0].stats.bytes_transferred;
        // Reset data for a second round.
        for (i, d) in devs.iter_mut().enumerate() {
            d.data = vec![(i + 1) as f32; 4];
        }
        AllReduce::sum(&mut devs, CommunicationTopology::Ring);
        assert!(devs[0].stats.bytes_transferred > bytes_after_1);
    }

    // ── ShardManager edge: single device ────────────────────────────────

    #[test]
    fn shard_manager_single_device_column() {
        let mgr = TensorShardManager::new(1, ShardSpec::ColumnParallel);
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let shards = mgr.shard(&data, 2, 3);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], data);
    }

    #[test]
    fn shard_manager_single_device_row() {
        let mgr = TensorShardManager::new(1, ShardSpec::RowParallel);
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let shards = mgr.shard(&data, 2, 3);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], data);
    }

    // ── Topology equality / hashing ─────────────────────────────────────

    #[test]
    fn topology_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CommunicationTopology::Ring);
        set.insert(CommunicationTopology::Ring);
        set.insert(CommunicationTopology::Tree);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn reduce_op_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ReduceOp::Sum);
        set.insert(ReduceOp::Sum);
        set.insert(ReduceOp::Max);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn shard_spec_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ShardSpec::ColumnParallel);
        set.insert(ShardSpec::ColumnParallel);
        set.insert(ShardSpec::RowParallel);
        set.insert(ShardSpec::Replicated);
        assert_eq!(set.len(), 3);
    }
}
