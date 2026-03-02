//! NUMA-aware operations for CPU inference optimization.
//!
//! Provides topology detection, memory placement, thread pinning, and
//! tensor partitioning that respect Non-Uniform Memory Access (NUMA)
//! boundaries. On systems without NUMA (or where `/sys/devices/system/node`
//! is unavailable) every operation degrades gracefully to a single logical
//! node containing all CPUs.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the NUMA-aware operations subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumaError {
    /// A configuration parameter is invalid.
    InvalidConfig(String),
    /// The requested NUMA node does not exist.
    NodeNotFound { node_id: usize, total_nodes: usize },
    /// The requested CPU does not belong to the specified node.
    CpuNotOnNode { cpu_id: usize, node_id: usize },
    /// Memory allocation on the target node failed.
    AllocationFailed { node_id: usize, bytes: usize },
    /// Tensor cannot be evenly partitioned across nodes.
    UnevenPartition { elements: usize, num_nodes: usize },
    /// A thread-pinning operation failed.
    PinningFailed(String),
    /// Topology detection is unavailable.
    TopologyUnavailable(String),
}

impl fmt::Display for NumaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid NUMA config: {msg}"),
            Self::NodeNotFound { node_id, total_nodes } => {
                write!(f, "NUMA node {node_id} not found (total {total_nodes})")
            }
            Self::CpuNotOnNode { cpu_id, node_id } => {
                write!(f, "CPU {cpu_id} is not on NUMA node {node_id}")
            }
            Self::AllocationFailed { node_id, bytes } => {
                write!(f, "allocation of {bytes} bytes failed on node {node_id}")
            }
            Self::UnevenPartition { elements, num_nodes } => {
                write!(f, "{elements} elements cannot be evenly split across {num_nodes} nodes")
            }
            Self::PinningFailed(msg) => write!(f, "thread pinning failed: {msg}"),
            Self::TopologyUnavailable(msg) => write!(f, "NUMA topology unavailable: {msg}"),
        }
    }
}

impl std::error::Error for NumaError {}

// ---------------------------------------------------------------------------
// Memory interleaving policy
// ---------------------------------------------------------------------------

/// Policy controlling how memory is distributed across NUMA nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum InterleavingPolicy {
    /// Bind all memory to a single node.
    #[default]
    Local,
    /// Round-robin pages across all nodes for balanced bandwidth.
    RoundRobin,
    /// Weight-proportional interleaving based on per-node bandwidth.
    Weighted,
}

impl fmt::Display for InterleavingPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local => write!(f, "Local"),
            Self::RoundRobin => write!(f, "RoundRobin"),
            Self::Weighted => write!(f, "Weighted"),
        }
    }
}

// ---------------------------------------------------------------------------
// Topology types
// ---------------------------------------------------------------------------

/// Information about a single NUMA node.
#[derive(Debug, Clone, PartialEq)]
pub struct NumaNodeInfo {
    /// Node identifier (0-based).
    pub node_id: usize,
    /// Logical CPU ids assigned to this node.
    pub cpus: Vec<usize>,
    /// Total memory in bytes available on this node.
    pub memory_bytes: u64,
    /// Free memory in bytes on this node (snapshot at detection time).
    pub free_memory_bytes: u64,
}

/// Complete NUMA topology of the system.
#[derive(Debug, Clone, PartialEq)]
pub struct NumaTopology {
    /// Per-node information, indexed by node position (not necessarily node id).
    pub nodes: Vec<NumaNodeInfo>,
    /// Distance matrix: `distances[i][j]` gives the relative cost of
    /// accessing memory on node `j` from a CPU on node `i`.
    /// A value of 10 is the conventional "local" distance.
    pub distances: Vec<Vec<u32>>,
}

impl NumaTopology {
    /// Total number of NUMA nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of CPUs across all nodes.
    pub fn total_cpus(&self) -> usize {
        self.nodes.iter().map(|n| n.cpus.len()).sum()
    }

    /// Total system memory across all nodes.
    pub fn total_memory_bytes(&self) -> u64 {
        self.nodes.iter().map(|n| n.memory_bytes).sum()
    }

    /// Look up which node owns a given CPU, or `None`.
    pub fn node_for_cpu(&self, cpu_id: usize) -> Option<usize> {
        self.nodes.iter().find(|n| n.cpus.contains(&cpu_id)).map(|n| n.node_id)
    }

    /// Return the distance between two nodes.
    pub fn distance(&self, from: usize, to: usize) -> Result<u32, NumaError> {
        if from >= self.nodes.len() {
            return Err(NumaError::NodeNotFound { node_id: from, total_nodes: self.nodes.len() });
        }
        if to >= self.nodes.len() {
            return Err(NumaError::NodeNotFound { node_id: to, total_nodes: self.nodes.len() });
        }
        Ok(self.distances[from][to])
    }

    /// Validate internal consistency.
    pub fn validate(&self) -> Result<(), NumaError> {
        if self.nodes.is_empty() {
            return Err(NumaError::InvalidConfig("topology has no nodes".into()));
        }
        let n = self.nodes.len();
        if self.distances.len() != n {
            return Err(NumaError::InvalidConfig(format!(
                "distance matrix rows ({}) != node count ({n})",
                self.distances.len(),
            )));
        }
        for (i, row) in self.distances.iter().enumerate() {
            if row.len() != n {
                return Err(NumaError::InvalidConfig(format!(
                    "distance matrix row {i} length ({}) != node count ({n})",
                    row.len(),
                )));
            }
        }
        // Local distance must be the smallest per row.
        for (i, row) in self.distances.iter().enumerate() {
            if row[i] != *row.iter().min().unwrap_or(&0) {
                return Err(NumaError::InvalidConfig(format!(
                    "local distance for node {i} is not the minimum in its row",
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// NUMA-aware allocation
// ---------------------------------------------------------------------------

/// A buffer logically "allocated" on a specific NUMA node.
///
/// In a real deployment this would use `mbind(2)` / `set_mempolicy(2)`.
/// Here we track placement metadata for scheduling and partitioning.
#[derive(Debug, Clone, PartialEq)]
pub struct NumaBuffer {
    /// The payload.
    pub data: Vec<f32>,
    /// The NUMA node this buffer is associated with.
    pub node_id: usize,
    /// The interleaving policy that was active at allocation time.
    pub policy: InterleavingPolicy,
}

// ---------------------------------------------------------------------------
// Thread pin plan
// ---------------------------------------------------------------------------

/// A plan mapping worker threads to specific CPUs on specific NUMA nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadPinPlan {
    /// `(worker_index, cpu_id, node_id)` triples.
    pub assignments: Vec<(usize, usize, usize)>,
}

// ---------------------------------------------------------------------------
// Tensor partition
// ---------------------------------------------------------------------------

/// One slice of a tensor, placed on a specific NUMA node.
#[derive(Debug, Clone, PartialEq)]
pub struct NumaTensorPartition {
    /// The data for this partition.
    pub data: Vec<f32>,
    /// Which NUMA node this partition lives on.
    pub node_id: usize,
    /// Partition index (0-based).
    pub partition_index: usize,
    /// Total number of partitions.
    pub total_partitions: usize,
}

// ---------------------------------------------------------------------------
// Transfer plan
// ---------------------------------------------------------------------------

/// Describes an optimal data movement between two NUMA nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct NumaTransferPlan {
    /// Source node.
    pub src_node: usize,
    /// Destination node.
    pub dst_node: usize,
    /// Distance (from topology matrix).
    pub distance: u32,
    /// Number of elements to move.
    pub num_elements: usize,
    /// Whether a local copy suffices (distance == local distance).
    pub is_local: bool,
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Performance counters for a NUMA-aware operation.
#[derive(Debug, Clone, Default)]
pub struct NumaMetrics {
    /// Wall-clock time for topology detection in milliseconds.
    pub detection_time_ms: f64,
    /// Wall-clock time for data placement / allocation in milliseconds.
    pub placement_time_ms: f64,
    /// Number of cross-node transfers required.
    pub cross_node_transfers: usize,
    /// Total elements moved across node boundaries.
    pub cross_node_elements: usize,
}

// ---------------------------------------------------------------------------
// Topology detection
// ---------------------------------------------------------------------------

/// Detect the NUMA topology of the current system.
///
/// On Linux this inspects `/sys/devices/system/node/`. On other platforms
/// (or when sysfs is unavailable) a single-node fallback topology is
/// returned containing all logical CPUs reported by `sysinfo`.
pub fn detect_topology() -> Result<(NumaTopology, NumaMetrics), NumaError> {
    let start = Instant::now();
    let topo = detect_topology_inner()?;
    topo.validate()?;
    let elapsed = start.elapsed();
    let metrics =
        NumaMetrics { detection_time_ms: elapsed.as_secs_f64() * 1000.0, ..Default::default() };
    Ok((topo, metrics))
}

/// Build a single-node fallback topology with `num_cpus` logical processors
/// and `total_mem` bytes of RAM.
pub fn fallback_single_node(num_cpus: usize, total_mem: u64) -> NumaTopology {
    let cpus: Vec<usize> = (0..num_cpus).collect();
    NumaTopology {
        nodes: vec![NumaNodeInfo {
            node_id: 0,
            cpus,
            memory_bytes: total_mem,
            free_memory_bytes: total_mem,
        }],
        distances: vec![vec![10]],
    }
}

fn detect_topology_inner() -> Result<NumaTopology, NumaError> {
    // Try Linux sysfs first.
    #[cfg(target_os = "linux")]
    {
        if let Ok(topo) = detect_linux_sysfs() {
            return Ok(topo);
        }
    }

    // Fallback: single-node with all CPUs.
    let sys = sysinfo::System::new_all();
    let cpus = sys.cpus().len().max(1);
    let mem = sys.total_memory();
    Ok(fallback_single_node(cpus, mem))
}

/// Linux-specific: parse `/sys/devices/system/node/node*`.
#[cfg(target_os = "linux")]
fn detect_linux_sysfs() -> Result<NumaTopology, NumaError> {
    use std::fs;
    use std::path::Path;

    let base = Path::new("/sys/devices/system/node");
    if !base.exists() {
        return Err(NumaError::TopologyUnavailable("sysfs node dir missing".into()));
    }

    let mut nodes = Vec::new();
    let mut node_ids: Vec<usize> = Vec::new();

    for entry in fs::read_dir(base).map_err(|e| NumaError::TopologyUnavailable(e.to_string()))? {
        let entry = entry.map_err(|e| NumaError::TopologyUnavailable(e.to_string()))?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if let Some(id_str) = name_str.strip_prefix("node")
            && let Ok(id) = id_str.parse::<usize>()
        {
            node_ids.push(id);
        }
    }
    node_ids.sort_unstable();

    if node_ids.is_empty() {
        return Err(NumaError::TopologyUnavailable("no NUMA nodes found in sysfs".into()));
    }

    for &nid in &node_ids {
        let cpulist_path = base.join(format!("node{nid}/cpulist"));
        let cpus = if cpulist_path.exists() {
            let raw = fs::read_to_string(&cpulist_path)
                .map_err(|e| NumaError::TopologyUnavailable(e.to_string()))?;
            parse_cpu_list(raw.trim())
        } else {
            Vec::new()
        };

        let meminfo_path = base.join(format!("node{nid}/meminfo"));
        let (total, free) = if meminfo_path.exists() {
            let raw = fs::read_to_string(&meminfo_path)
                .map_err(|e| NumaError::TopologyUnavailable(e.to_string()))?;
            parse_node_meminfo(&raw)
        } else {
            (0, 0)
        };

        nodes.push(NumaNodeInfo {
            node_id: nid,
            cpus,
            memory_bytes: total,
            free_memory_bytes: free,
        });
    }

    // Build distance matrix.
    let n = nodes.len();
    let mut distances = vec![vec![10u32; n]; n];
    let distance_path = base.join("node0/distance");
    if distance_path.exists() {
        for (i, nid) in node_ids.iter().enumerate() {
            let p = base.join(format!("node{nid}/distance"));
            if let Ok(raw) = fs::read_to_string(&p) {
                let vals: Vec<u32> =
                    raw.split_whitespace().filter_map(|s| s.parse().ok()).collect();
                if vals.len() == n {
                    distances[i] = vals;
                }
            }
        }
    }

    Ok(NumaTopology { nodes, distances })
}

/// Parse a Linux CPU-list string like `"0-3,8-11"` into a sorted `Vec<usize>`.
fn parse_cpu_list(s: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    if s.is_empty() {
        return cpus;
    }
    for part in s.split(',') {
        let part = part.trim();
        if let Some((lo, hi)) = part.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (lo.trim().parse::<usize>(), hi.trim().parse::<usize>()) {
                cpus.extend(lo..=hi);
            }
        } else if let Ok(v) = part.parse::<usize>() {
            cpus.push(v);
        }
    }
    cpus.sort_unstable();
    cpus
}

/// Parse a Linux node `meminfo` for MemTotal / MemFree (in kB → bytes).
fn parse_node_meminfo(raw: &str) -> (u64, u64) {
    let mut total: u64 = 0;
    let mut free: u64 = 0;
    for line in raw.lines() {
        if line.contains("MemTotal")
            && let Some(kb) = extract_kb(line)
        {
            total = kb * 1024;
        } else if line.contains("MemFree")
            && let Some(kb) = extract_kb(line)
        {
            free = kb * 1024;
        }
    }
    (total, free)
}

fn extract_kb(line: &str) -> Option<u64> {
    // Value follows the colon, e.g. "Node 0 MemTotal:    16384000 kB"
    let after_colon = line.split_once(':')?.1;
    after_colon.split_whitespace().find_map(|w| w.parse::<u64>().ok())
}

// ---------------------------------------------------------------------------
// NUMA-aware allocation
// ---------------------------------------------------------------------------

/// Allocate a zeroed buffer logically placed on `node_id`.
///
/// Validates that `node_id` exists in the supplied topology.  In a real
/// system the kernel page policy would be set via `mbind`; here we
/// simulate placement metadata.
pub fn allocate_on_node(
    num_elements: usize,
    node_id: usize,
    topology: &NumaTopology,
    policy: InterleavingPolicy,
) -> Result<(NumaBuffer, NumaMetrics), NumaError> {
    if node_id >= topology.num_nodes() {
        return Err(NumaError::NodeNotFound { node_id, total_nodes: topology.num_nodes() });
    }
    if num_elements == 0 {
        return Err(NumaError::InvalidConfig("num_elements must be > 0".into()));
    }
    let start = Instant::now();
    let data = vec![0.0f32; num_elements];
    let elapsed = start.elapsed();

    Ok((
        NumaBuffer { data, node_id, policy },
        NumaMetrics { placement_time_ms: elapsed.as_secs_f64() * 1000.0, ..Default::default() },
    ))
}

/// Allocate with round-robin interleaving across all nodes.
///
/// Splits `num_elements` into `topology.num_nodes()` chunks, one per node.
pub fn allocate_interleaved(
    num_elements: usize,
    topology: &NumaTopology,
) -> Result<(Vec<NumaBuffer>, NumaMetrics), NumaError> {
    if num_elements == 0 {
        return Err(NumaError::InvalidConfig("num_elements must be > 0".into()));
    }
    let start = Instant::now();
    let n = topology.num_nodes();
    let base = num_elements / n;
    let remainder = num_elements % n;
    let mut buffers = Vec::with_capacity(n);
    for i in 0..n {
        let size = base + if i < remainder { 1 } else { 0 };
        buffers.push(NumaBuffer {
            data: vec![0.0f32; size],
            node_id: topology.nodes[i].node_id,
            policy: InterleavingPolicy::RoundRobin,
        });
    }
    let elapsed = start.elapsed();
    Ok((
        buffers,
        NumaMetrics { placement_time_ms: elapsed.as_secs_f64() * 1000.0, ..Default::default() },
    ))
}

/// Allocate with weighted interleaving proportional to per-node memory.
pub fn allocate_weighted(
    num_elements: usize,
    topology: &NumaTopology,
) -> Result<(Vec<NumaBuffer>, NumaMetrics), NumaError> {
    if num_elements == 0 {
        return Err(NumaError::InvalidConfig("num_elements must be > 0".into()));
    }
    let start = Instant::now();
    let total_mem: u64 = topology.total_memory_bytes();
    if total_mem == 0 {
        return Err(NumaError::InvalidConfig("total memory is zero".into()));
    }
    let n = topology.num_nodes();
    let mut sizes: Vec<usize> = Vec::with_capacity(n);
    let mut assigned = 0usize;
    for (i, node) in topology.nodes.iter().enumerate() {
        let share = if i == n - 1 {
            num_elements - assigned
        } else {
            let fraction = node.memory_bytes as f64 / total_mem as f64;
            (num_elements as f64 * fraction).round() as usize
        };
        sizes.push(share);
        assigned += share;
    }
    let mut buffers = Vec::with_capacity(n);
    for (i, &sz) in sizes.iter().enumerate() {
        buffers.push(NumaBuffer {
            data: vec![0.0f32; sz],
            node_id: topology.nodes[i].node_id,
            policy: InterleavingPolicy::Weighted,
        });
    }
    let elapsed = start.elapsed();
    Ok((
        buffers,
        NumaMetrics { placement_time_ms: elapsed.as_secs_f64() * 1000.0, ..Default::default() },
    ))
}

// ---------------------------------------------------------------------------
// Thread pinning
// ---------------------------------------------------------------------------

/// Build a thread-pin plan that distributes `num_workers` across NUMA nodes
/// proportional to the number of CPUs each node provides.
///
/// Workers are assigned round-robin within each node's CPU list.
pub fn plan_thread_pinning(
    num_workers: usize,
    topology: &NumaTopology,
) -> Result<ThreadPinPlan, NumaError> {
    if num_workers == 0 {
        return Err(NumaError::InvalidConfig("num_workers must be > 0".into()));
    }
    let total_cpus = topology.total_cpus();
    if total_cpus == 0 {
        return Err(NumaError::InvalidConfig("topology has no CPUs".into()));
    }

    let mut assignments = Vec::with_capacity(num_workers);
    let n = topology.num_nodes();
    // Proportional distribution.
    let mut per_node: Vec<usize> = Vec::with_capacity(n);
    let mut assigned = 0;
    for (i, node) in topology.nodes.iter().enumerate() {
        let share = if i == n - 1 {
            num_workers - assigned
        } else {
            let fraction = node.cpus.len() as f64 / total_cpus as f64;
            (num_workers as f64 * fraction).round() as usize
        };
        per_node.push(share);
        assigned += share;
    }

    let mut worker_idx = 0;
    for (node_pos, count) in per_node.iter().enumerate() {
        let node = &topology.nodes[node_pos];
        if node.cpus.is_empty() {
            continue;
        }
        for j in 0..*count {
            let cpu = node.cpus[j % node.cpus.len()];
            assignments.push((worker_idx, cpu, node.node_id));
            worker_idx += 1;
        }
    }
    Ok(ThreadPinPlan { assignments })
}

/// Pin the current thread to a specific CPU.
///
/// On Linux this calls `sched_setaffinity`. On other platforms it is a
/// no-op that returns `Ok`.
pub fn pin_thread_to_cpu(cpu_id: usize) -> Result<(), NumaError> {
    #[cfg(target_os = "linux")]
    {
        pin_thread_linux(cpu_id)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu_id;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn pin_thread_linux(cpu_id: usize) -> Result<(), NumaError> {
    use std::mem;

    // libc cpu_set_t operations.
    unsafe {
        let mut set: libc::cpu_set_t = mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu_id, &mut set);
        let ret = libc::sched_setaffinity(0, mem::size_of::<libc::cpu_set_t>(), &set);
        if ret != 0 {
            return Err(NumaError::PinningFailed(format!(
                "sched_setaffinity failed for cpu {cpu_id}: errno {}",
                *libc::__errno_location()
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Cross-NUMA transfer optimisation
// ---------------------------------------------------------------------------

/// Build a transfer plan from `src_node` to `dst_node` for `num_elements`.
///
/// The plan annotates the distance and whether the transfer is local
/// (same-node) so the caller can decide whether to copy or share.
pub fn plan_transfer(
    src_node: usize,
    dst_node: usize,
    num_elements: usize,
    topology: &NumaTopology,
) -> Result<NumaTransferPlan, NumaError> {
    let dist = topology.distance(src_node, dst_node)?;
    let local_dist = topology.distance(src_node, src_node)?;
    Ok(NumaTransferPlan {
        src_node,
        dst_node,
        distance: dist,
        num_elements,
        is_local: dist == local_dist,
    })
}

/// Execute a transfer: copy data into a new buffer on `dst_node`.
///
/// Returns `(destination_buffer, metrics)`.
pub fn execute_transfer(
    source: &NumaBuffer,
    dst_node: usize,
    topology: &NumaTopology,
) -> Result<(NumaBuffer, NumaMetrics), NumaError> {
    if dst_node >= topology.num_nodes() {
        return Err(NumaError::NodeNotFound {
            node_id: dst_node,
            total_nodes: topology.num_nodes(),
        });
    }
    let plan = plan_transfer(source.node_id, dst_node, source.data.len(), topology)?;
    let start = Instant::now();
    let dst = NumaBuffer {
        data: source.data.clone(),
        node_id: dst_node,
        policy: InterleavingPolicy::Local,
    };
    let elapsed = start.elapsed();
    Ok((
        dst,
        NumaMetrics {
            placement_time_ms: elapsed.as_secs_f64() * 1000.0,
            cross_node_transfers: if plan.is_local { 0 } else { 1 },
            cross_node_elements: if plan.is_local { 0 } else { source.data.len() },
            ..Default::default()
        },
    ))
}

/// Compute optimal placement: given a list of consumer nodes, pick the
/// source node that minimises total distance to all consumers.
pub fn optimal_placement(
    consumer_nodes: &[usize],
    topology: &NumaTopology,
) -> Result<usize, NumaError> {
    if consumer_nodes.is_empty() {
        return Err(NumaError::InvalidConfig("consumer_nodes must not be empty".into()));
    }
    let mut best_node = 0;
    let mut best_cost = u64::MAX;
    for (idx, node) in topology.nodes.iter().enumerate() {
        let cost: u64 = consumer_nodes
            .iter()
            .map(|&c| topology.distance(idx, c).unwrap_or(u32::MAX) as u64)
            .sum();
        if cost < best_cost {
            best_cost = cost;
            best_node = node.node_id;
        }
    }
    Ok(best_node)
}

// ---------------------------------------------------------------------------
// Tensor partitioning
// ---------------------------------------------------------------------------

/// Partition a tensor across NUMA nodes, evenly or proportionally.
///
/// With `proportional = false` every node gets `ceil(len / num_nodes)`
/// elements (the last partition may be smaller).
/// With `proportional = true` elements are distributed proportional to
/// each node's memory capacity.
pub fn partition_tensor(
    tensor: &[f32],
    topology: &NumaTopology,
    proportional: bool,
) -> Result<(Vec<NumaTensorPartition>, NumaMetrics), NumaError> {
    if tensor.is_empty() {
        return Err(NumaError::InvalidConfig("tensor must not be empty".into()));
    }
    let n = topology.num_nodes();
    let start = Instant::now();
    let sizes = if proportional {
        proportional_sizes(tensor.len(), topology)?
    } else {
        even_sizes(tensor.len(), n)
    };

    let mut partitions = Vec::with_capacity(n);
    let mut offset = 0;
    for (i, &sz) in sizes.iter().enumerate() {
        let end = (offset + sz).min(tensor.len());
        partitions.push(NumaTensorPartition {
            data: tensor[offset..end].to_vec(),
            node_id: topology.nodes[i].node_id,
            partition_index: i,
            total_partitions: n,
        });
        offset = end;
    }
    let elapsed = start.elapsed();
    Ok((
        partitions,
        NumaMetrics { placement_time_ms: elapsed.as_secs_f64() * 1000.0, ..Default::default() },
    ))
}

/// Gather partitions back into a single contiguous tensor.
pub fn gather_partitions(partitions: &[NumaTensorPartition]) -> Result<Vec<f32>, NumaError> {
    if partitions.is_empty() {
        return Err(NumaError::InvalidConfig("no partitions to gather".into()));
    }
    let mut sorted: Vec<_> = partitions.to_vec();
    sorted.sort_by_key(|p| p.partition_index);
    Ok(sorted.iter().flat_map(|p| p.data.iter().copied()).collect())
}

fn even_sizes(len: usize, n: usize) -> Vec<usize> {
    let base = len / n;
    let rem = len % n;
    (0..n).map(|i| base + if i < rem { 1 } else { 0 }).collect()
}

fn proportional_sizes(len: usize, topology: &NumaTopology) -> Result<Vec<usize>, NumaError> {
    let total_mem = topology.total_memory_bytes();
    if total_mem == 0 {
        return Err(NumaError::InvalidConfig("total memory is zero".into()));
    }
    let n = topology.num_nodes();
    let mut sizes = Vec::with_capacity(n);
    let mut assigned = 0;
    for (i, node) in topology.nodes.iter().enumerate() {
        let sz = if i == n - 1 {
            len - assigned
        } else {
            let frac = node.memory_bytes as f64 / total_mem as f64;
            (len as f64 * frac).round() as usize
        };
        sizes.push(sz);
        assigned += sz;
    }
    Ok(sizes)
}

// ---------------------------------------------------------------------------
// First-touch optimisation
// ---------------------------------------------------------------------------

/// Perform first-touch initialisation: write `value` to every element so
/// that pages are faulted on the "current" NUMA node (the caller is
/// responsible for running this from a thread pinned to the target node).
///
/// Returns the number of elements touched.
pub fn first_touch_init(buf: &mut NumaBuffer, value: f32) -> usize {
    let len = buf.data.len();
    for v in buf.data.iter_mut() {
        *v = value;
    }
    len
}

/// First-touch with a stride pattern (useful for huge buffers where
/// touching every cache-line suffices).
pub fn first_touch_strided(buf: &mut NumaBuffer, value: f32, stride: usize) -> usize {
    if stride == 0 {
        return 0;
    }
    let mut count = 0;
    let mut i = 0;
    while i < buf.data.len() {
        buf.data[i] = value;
        count += 1;
        i += stride;
    }
    count
}

// ---------------------------------------------------------------------------
// NUMA distance matrix helpers
// ---------------------------------------------------------------------------

/// Build a synthetic distance matrix for `num_nodes` nodes.
///
/// Local distance = 10, remote distance = `remote_distance` (typically 20–21).
pub fn build_uniform_distance_matrix(num_nodes: usize, remote_distance: u32) -> Vec<Vec<u32>> {
    (0..num_nodes)
        .map(|i| (0..num_nodes).map(|j| if i == j { 10 } else { remote_distance }).collect())
        .collect()
}

/// Build a hierarchical distance matrix where nodes in the same socket
/// group have a lower distance than nodes in different socket groups.
///
/// `nodes_per_socket` defines how many NUMA nodes share a socket.
pub fn build_hierarchical_distance_matrix(
    num_nodes: usize,
    nodes_per_socket: usize,
    intra_socket: u32,
    inter_socket: u32,
) -> Vec<Vec<u32>> {
    if nodes_per_socket == 0 {
        return build_uniform_distance_matrix(num_nodes, inter_socket);
    }
    (0..num_nodes)
        .map(|i| {
            (0..num_nodes)
                .map(|j| {
                    if i == j {
                        10
                    } else if i / nodes_per_socket == j / nodes_per_socket {
                        intra_socket
                    } else {
                        inter_socket
                    }
                })
                .collect()
        })
        .collect()
}

/// Find the nearest neighbour node for `node_id` (minimum non-local distance).
pub fn nearest_neighbor(node_id: usize, topology: &NumaTopology) -> Result<usize, NumaError> {
    if node_id >= topology.num_nodes() {
        return Err(NumaError::NodeNotFound { node_id, total_nodes: topology.num_nodes() });
    }
    if topology.num_nodes() == 1 {
        return Ok(node_id);
    }
    let row = &topology.distances[node_id];
    let local = row[node_id];
    let mut best = node_id;
    let mut best_dist = u32::MAX;
    for (j, &d) in row.iter().enumerate() {
        if j != node_id && d < best_dist && d >= local {
            best_dist = d;
            best = j;
        }
    }
    Ok(best)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --------------------------------------------------------

    fn two_node_topology() -> NumaTopology {
        NumaTopology {
            nodes: vec![
                NumaNodeInfo {
                    node_id: 0,
                    cpus: vec![0, 1, 2, 3],
                    memory_bytes: 16 * 1024 * 1024 * 1024,
                    free_memory_bytes: 8 * 1024 * 1024 * 1024,
                },
                NumaNodeInfo {
                    node_id: 1,
                    cpus: vec![4, 5, 6, 7],
                    memory_bytes: 16 * 1024 * 1024 * 1024,
                    free_memory_bytes: 8 * 1024 * 1024 * 1024,
                },
            ],
            distances: vec![vec![10, 20], vec![20, 10]],
        }
    }

    fn four_node_topology() -> NumaTopology {
        NumaTopology {
            nodes: vec![
                NumaNodeInfo {
                    node_id: 0,
                    cpus: vec![0, 1],
                    memory_bytes: 8_000_000_000,
                    free_memory_bytes: 4_000_000_000,
                },
                NumaNodeInfo {
                    node_id: 1,
                    cpus: vec![2, 3],
                    memory_bytes: 8_000_000_000,
                    free_memory_bytes: 4_000_000_000,
                },
                NumaNodeInfo {
                    node_id: 2,
                    cpus: vec![4, 5],
                    memory_bytes: 16_000_000_000,
                    free_memory_bytes: 8_000_000_000,
                },
                NumaNodeInfo {
                    node_id: 3,
                    cpus: vec![6, 7],
                    memory_bytes: 16_000_000_000,
                    free_memory_bytes: 8_000_000_000,
                },
            ],
            distances: build_hierarchical_distance_matrix(4, 2, 15, 25),
        }
    }

    fn single_node_topology() -> NumaTopology {
        fallback_single_node(4, 32 * 1024 * 1024 * 1024)
    }

    fn asymmetric_topology() -> NumaTopology {
        NumaTopology {
            nodes: vec![
                NumaNodeInfo {
                    node_id: 0,
                    cpus: vec![0, 1, 2, 3, 4, 5],
                    memory_bytes: 32_000_000_000,
                    free_memory_bytes: 16_000_000_000,
                },
                NumaNodeInfo {
                    node_id: 1,
                    cpus: vec![6, 7],
                    memory_bytes: 8_000_000_000,
                    free_memory_bytes: 4_000_000_000,
                },
            ],
            distances: vec![vec![10, 20], vec![20, 10]],
        }
    }

    // == Topology detection =============================================

    #[test]
    fn detect_topology_returns_at_least_one_node() {
        let (topo, metrics) = detect_topology().unwrap();
        assert!(topo.num_nodes() >= 1);
        assert!(metrics.detection_time_ms >= 0.0);
    }

    #[test]
    fn detect_topology_has_cpus() {
        let (topo, _) = detect_topology().unwrap();
        assert!(topo.total_cpus() >= 1);
    }

    #[test]
    fn detect_topology_has_memory() {
        let (topo, _) = detect_topology().unwrap();
        assert!(topo.total_memory_bytes() > 0);
    }

    #[test]
    fn detect_topology_distance_matrix_is_square() {
        let (topo, _) = detect_topology().unwrap();
        let n = topo.num_nodes();
        assert_eq!(topo.distances.len(), n);
        for row in &topo.distances {
            assert_eq!(row.len(), n);
        }
    }

    #[test]
    fn detect_topology_local_distance_is_minimum() {
        let (topo, _) = detect_topology().unwrap();
        for (i, row) in topo.distances.iter().enumerate() {
            assert_eq!(row[i], *row.iter().min().unwrap());
        }
    }

    // == Fallback topology ==============================================

    #[test]
    fn fallback_single_node_has_one_node() {
        let t = fallback_single_node(8, 1024);
        assert_eq!(t.num_nodes(), 1);
        assert_eq!(t.total_cpus(), 8);
        assert_eq!(t.total_memory_bytes(), 1024);
    }

    #[test]
    fn fallback_single_node_validates() {
        let t = fallback_single_node(4, 2048);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn fallback_local_distance_is_10() {
        let t = fallback_single_node(2, 100);
        assert_eq!(t.distances[0][0], 10);
    }

    // == Topology validation ============================================

    #[test]
    fn validate_empty_topology_fails() {
        let t = NumaTopology { nodes: vec![], distances: vec![] };
        assert!(t.validate().is_err());
    }

    #[test]
    fn validate_mismatched_distance_rows() {
        let mut t = two_node_topology();
        t.distances = vec![vec![10, 20]]; // only 1 row
        assert!(t.validate().is_err());
    }

    #[test]
    fn validate_mismatched_distance_cols() {
        let mut t = two_node_topology();
        t.distances[0] = vec![10]; // wrong length
        assert!(t.validate().is_err());
    }

    #[test]
    fn validate_local_not_min_fails() {
        let mut t = two_node_topology();
        t.distances[0] = vec![25, 20]; // local > remote
        assert!(t.validate().is_err());
    }

    #[test]
    fn validate_good_topology_ok() {
        assert!(two_node_topology().validate().is_ok());
        assert!(four_node_topology().validate().is_ok());
    }

    // == Topology queries ===============================================

    #[test]
    fn node_for_cpu_found() {
        let t = two_node_topology();
        assert_eq!(t.node_for_cpu(0), Some(0));
        assert_eq!(t.node_for_cpu(5), Some(1));
    }

    #[test]
    fn node_for_cpu_not_found() {
        let t = two_node_topology();
        assert_eq!(t.node_for_cpu(99), None);
    }

    #[test]
    fn distance_local_is_10() {
        let t = two_node_topology();
        assert_eq!(t.distance(0, 0).unwrap(), 10);
        assert_eq!(t.distance(1, 1).unwrap(), 10);
    }

    #[test]
    fn distance_remote() {
        let t = two_node_topology();
        assert_eq!(t.distance(0, 1).unwrap(), 20);
    }

    #[test]
    fn distance_out_of_bounds() {
        let t = two_node_topology();
        assert!(t.distance(5, 0).is_err());
        assert!(t.distance(0, 5).is_err());
    }

    #[test]
    fn total_cpus_two_node() {
        let t = two_node_topology();
        assert_eq!(t.total_cpus(), 8);
    }

    // == NUMA allocation ================================================

    #[test]
    fn allocate_on_node_basic() {
        let t = two_node_topology();
        let (buf, m) = allocate_on_node(256, 0, &t, InterleavingPolicy::Local).unwrap();
        assert_eq!(buf.data.len(), 256);
        assert_eq!(buf.node_id, 0);
        assert_eq!(buf.policy, InterleavingPolicy::Local);
        assert!(m.placement_time_ms >= 0.0);
    }

    #[test]
    fn allocate_on_node_second_node() {
        let t = two_node_topology();
        let (buf, _) = allocate_on_node(128, 1, &t, InterleavingPolicy::Local).unwrap();
        assert_eq!(buf.node_id, 1);
    }

    #[test]
    fn allocate_on_node_invalid_node() {
        let t = two_node_topology();
        assert!(matches!(
            allocate_on_node(10, 5, &t, InterleavingPolicy::Local),
            Err(NumaError::NodeNotFound { .. })
        ));
    }

    #[test]
    fn allocate_on_node_zero_elements() {
        let t = two_node_topology();
        assert!(allocate_on_node(0, 0, &t, InterleavingPolicy::Local).is_err());
    }

    #[test]
    fn allocate_on_node_data_is_zeroed() {
        let t = single_node_topology();
        let (buf, _) = allocate_on_node(64, 0, &t, InterleavingPolicy::Local).unwrap();
        assert!(buf.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn allocate_on_node_with_weighted_policy() {
        let t = two_node_topology();
        let (buf, _) = allocate_on_node(32, 0, &t, InterleavingPolicy::Weighted).unwrap();
        assert_eq!(buf.policy, InterleavingPolicy::Weighted);
    }

    // == Interleaved allocation =========================================

    #[test]
    fn allocate_interleaved_even() {
        let t = two_node_topology();
        let (bufs, _) = allocate_interleaved(100, &t).unwrap();
        assert_eq!(bufs.len(), 2);
        assert_eq!(bufs[0].data.len(), 50);
        assert_eq!(bufs[1].data.len(), 50);
    }

    #[test]
    fn allocate_interleaved_odd() {
        let t = two_node_topology();
        let (bufs, _) = allocate_interleaved(101, &t).unwrap();
        let total: usize = bufs.iter().map(|b| b.data.len()).sum();
        assert_eq!(total, 101);
        // first node gets the extra element
        assert_eq!(bufs[0].data.len(), 51);
        assert_eq!(bufs[1].data.len(), 50);
    }

    #[test]
    fn allocate_interleaved_single_node() {
        let t = single_node_topology();
        let (bufs, _) = allocate_interleaved(50, &t).unwrap();
        assert_eq!(bufs.len(), 1);
        assert_eq!(bufs[0].data.len(), 50);
    }

    #[test]
    fn allocate_interleaved_zero_elements_err() {
        let t = two_node_topology();
        assert!(allocate_interleaved(0, &t).is_err());
    }

    #[test]
    fn allocate_interleaved_policy_is_round_robin() {
        let t = two_node_topology();
        let (bufs, _) = allocate_interleaved(10, &t).unwrap();
        for b in &bufs {
            assert_eq!(b.policy, InterleavingPolicy::RoundRobin);
        }
    }

    #[test]
    fn allocate_interleaved_four_nodes() {
        let t = four_node_topology();
        let (bufs, _) = allocate_interleaved(200, &t).unwrap();
        assert_eq!(bufs.len(), 4);
        let total: usize = bufs.iter().map(|b| b.data.len()).sum();
        assert_eq!(total, 200);
    }

    // == Weighted allocation =============================================

    #[test]
    fn allocate_weighted_equal_memory() {
        let t = two_node_topology();
        let (bufs, _) = allocate_weighted(100, &t).unwrap();
        assert_eq!(bufs.len(), 2);
        let total: usize = bufs.iter().map(|b| b.data.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn allocate_weighted_asymmetric() {
        let t = asymmetric_topology();
        let (bufs, _) = allocate_weighted(100, &t).unwrap();
        // Node 0 has 32 GB, node 1 has 8 GB → 80% / 20%.
        assert!(bufs[0].data.len() > bufs[1].data.len());
        assert_eq!(bufs.iter().map(|b| b.data.len()).sum::<usize>(), 100);
    }

    #[test]
    fn allocate_weighted_zero_err() {
        let t = two_node_topology();
        assert!(allocate_weighted(0, &t).is_err());
    }

    #[test]
    fn allocate_weighted_policy_tag() {
        let t = two_node_topology();
        let (bufs, _) = allocate_weighted(20, &t).unwrap();
        for b in &bufs {
            assert_eq!(b.policy, InterleavingPolicy::Weighted);
        }
    }

    // == Thread pinning =================================================

    #[test]
    fn plan_pinning_basic() {
        let t = two_node_topology();
        let plan = plan_thread_pinning(4, &t).unwrap();
        assert_eq!(plan.assignments.len(), 4);
    }

    #[test]
    fn plan_pinning_workers_eq_cpus() {
        let t = two_node_topology();
        let plan = plan_thread_pinning(8, &t).unwrap();
        assert_eq!(plan.assignments.len(), 8);
        // Every CPU should appear at least once.
        let cpus: Vec<usize> = plan.assignments.iter().map(|a| a.1).collect();
        for c in 0..8 {
            assert!(cpus.contains(&c), "cpu {c} missing");
        }
    }

    #[test]
    fn plan_pinning_more_workers_than_cpus() {
        let t = two_node_topology();
        let plan = plan_thread_pinning(16, &t).unwrap();
        assert_eq!(plan.assignments.len(), 16);
    }

    #[test]
    fn plan_pinning_single_worker() {
        let t = two_node_topology();
        let plan = plan_thread_pinning(1, &t).unwrap();
        assert_eq!(plan.assignments.len(), 1);
    }

    #[test]
    fn plan_pinning_zero_workers_err() {
        let t = two_node_topology();
        assert!(plan_thread_pinning(0, &t).is_err());
    }

    #[test]
    fn plan_pinning_single_node() {
        let t = single_node_topology();
        let plan = plan_thread_pinning(4, &t).unwrap();
        // All workers should be on node 0.
        for &(_, _, nid) in &plan.assignments {
            assert_eq!(nid, 0);
        }
    }

    #[test]
    fn plan_pinning_proportional_distribution() {
        let t = asymmetric_topology();
        let plan = plan_thread_pinning(8, &t).unwrap();
        let on_0 = plan.assignments.iter().filter(|a| a.2 == 0).count();
        let on_1 = plan.assignments.iter().filter(|a| a.2 == 1).count();
        // Node 0 has 6 cpus, node 1 has 2 → ~75%/25%.
        assert!(on_0 > on_1);
        assert_eq!(on_0 + on_1, 8);
    }

    #[test]
    fn plan_pinning_worker_indices_are_sequential() {
        let t = four_node_topology();
        let plan = plan_thread_pinning(8, &t).unwrap();
        for (i, &(wid, _, _)) in plan.assignments.iter().enumerate() {
            assert_eq!(wid, i);
        }
    }

    // == Transfer planning ==============================================

    #[test]
    fn plan_transfer_local() {
        let t = two_node_topology();
        let plan = plan_transfer(0, 0, 100, &t).unwrap();
        assert!(plan.is_local);
        assert_eq!(plan.distance, 10);
    }

    #[test]
    fn plan_transfer_remote() {
        let t = two_node_topology();
        let plan = plan_transfer(0, 1, 200, &t).unwrap();
        assert!(!plan.is_local);
        assert_eq!(plan.distance, 20);
        assert_eq!(plan.num_elements, 200);
    }

    #[test]
    fn plan_transfer_bad_node() {
        let t = two_node_topology();
        assert!(plan_transfer(0, 5, 10, &t).is_err());
    }

    #[test]
    fn execute_transfer_local() {
        let t = two_node_topology();
        let src =
            NumaBuffer { data: vec![1.0, 2.0, 3.0], node_id: 0, policy: InterleavingPolicy::Local };
        let (dst, m) = execute_transfer(&src, 0, &t).unwrap();
        assert_eq!(dst.data, src.data);
        assert_eq!(dst.node_id, 0);
        assert_eq!(m.cross_node_transfers, 0);
    }

    #[test]
    fn execute_transfer_remote() {
        let t = two_node_topology();
        let src =
            NumaBuffer { data: vec![4.0, 5.0], node_id: 0, policy: InterleavingPolicy::Local };
        let (dst, m) = execute_transfer(&src, 1, &t).unwrap();
        assert_eq!(dst.data, src.data);
        assert_eq!(dst.node_id, 1);
        assert_eq!(m.cross_node_transfers, 1);
        assert_eq!(m.cross_node_elements, 2);
    }

    #[test]
    fn execute_transfer_bad_dst() {
        let t = two_node_topology();
        let src = NumaBuffer { data: vec![1.0], node_id: 0, policy: InterleavingPolicy::Local };
        assert!(execute_transfer(&src, 9, &t).is_err());
    }

    // == Optimal placement ==============================================

    #[test]
    fn optimal_placement_single_consumer() {
        let t = two_node_topology();
        let node = optimal_placement(&[1], &t).unwrap();
        assert_eq!(node, 1);
    }

    #[test]
    fn optimal_placement_both_nodes() {
        let t = two_node_topology();
        // Consumers on both nodes → either node is acceptable (equal cost).
        let node = optimal_placement(&[0, 1], &t).unwrap();
        assert!(node == 0 || node == 1);
    }

    #[test]
    fn optimal_placement_empty_err() {
        let t = two_node_topology();
        assert!(optimal_placement(&[], &t).is_err());
    }

    #[test]
    fn optimal_placement_majority_consumers() {
        let t = four_node_topology();
        // Three consumers on nodes 0,1 (same socket) → node 0 or 1 preferred.
        let node = optimal_placement(&[0, 0, 1], &t).unwrap();
        assert!(node == 0 || node == 1);
    }

    // == Tensor partitioning ============================================

    #[test]
    fn partition_even() {
        let t = two_node_topology();
        let data: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let (parts, _) = partition_tensor(&data, &t, false).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].data.len(), 50);
        assert_eq!(parts[1].data.len(), 50);
    }

    #[test]
    fn partition_uneven() {
        let t = two_node_topology();
        let data: Vec<f32> = (0..101).map(|x| x as f32).collect();
        let (parts, _) = partition_tensor(&data, &t, false).unwrap();
        let total: usize = parts.iter().map(|p| p.data.len()).sum();
        assert_eq!(total, 101);
    }

    #[test]
    fn partition_proportional_asymmetric() {
        let t = asymmetric_topology();
        let data: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let (parts, _) = partition_tensor(&data, &t, true).unwrap();
        assert!(parts[0].data.len() > parts[1].data.len());
        let total: usize = parts.iter().map(|p| p.data.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn partition_node_ids_correct() {
        let t = two_node_topology();
        let data = vec![1.0; 20];
        let (parts, _) = partition_tensor(&data, &t, false).unwrap();
        assert_eq!(parts[0].node_id, 0);
        assert_eq!(parts[1].node_id, 1);
    }

    #[test]
    fn partition_indices_sequential() {
        let t = four_node_topology();
        let data = vec![0.0; 40];
        let (parts, _) = partition_tensor(&data, &t, false).unwrap();
        for (i, p) in parts.iter().enumerate() {
            assert_eq!(p.partition_index, i);
            assert_eq!(p.total_partitions, 4);
        }
    }

    #[test]
    fn partition_empty_tensor_err() {
        let t = two_node_topology();
        assert!(partition_tensor(&[], &t, false).is_err());
    }

    #[test]
    fn partition_single_node() {
        let t = single_node_topology();
        let data = vec![1.0; 50];
        let (parts, _) = partition_tensor(&data, &t, false).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].data.len(), 50);
    }

    // == Gather partitions ==============================================

    #[test]
    fn gather_round_trip() {
        let t = two_node_topology();
        let data: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let (parts, _) = partition_tensor(&data, &t, false).unwrap();
        let gathered = gather_partitions(&parts).unwrap();
        assert_eq!(gathered, data);
    }

    #[test]
    fn gather_out_of_order() {
        let t = two_node_topology();
        let data: Vec<f32> = (0..20).map(|x| x as f32).collect();
        let (mut parts, _) = partition_tensor(&data, &t, false).unwrap();
        parts.reverse();
        let gathered = gather_partitions(&parts).unwrap();
        assert_eq!(gathered, data);
    }

    #[test]
    fn gather_empty_err() {
        assert!(gather_partitions(&[]).is_err());
    }

    #[test]
    fn gather_four_node_round_trip() {
        let t = four_node_topology();
        let data: Vec<f32> = (0..200).map(|x| x as f32).collect();
        let (parts, _) = partition_tensor(&data, &t, false).unwrap();
        let gathered = gather_partitions(&parts).unwrap();
        assert_eq!(gathered, data);
    }

    // == First-touch ====================================================

    #[test]
    fn first_touch_init_all() {
        let t = single_node_topology();
        let (mut buf, _) = allocate_on_node(64, 0, &t, InterleavingPolicy::Local).unwrap();
        let n = first_touch_init(&mut buf, 1.0);
        assert_eq!(n, 64);
        assert!(buf.data.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn first_touch_init_overwrites_existing() {
        let mut buf =
            NumaBuffer { data: vec![5.0; 10], node_id: 0, policy: InterleavingPolicy::Local };
        first_touch_init(&mut buf, 0.0);
        assert!(buf.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn first_touch_strided_basic() {
        let mut buf =
            NumaBuffer { data: vec![0.0; 16], node_id: 0, policy: InterleavingPolicy::Local };
        let n = first_touch_strided(&mut buf, 7.0, 4);
        assert_eq!(n, 4); // indices 0, 4, 8, 12
        assert_eq!(buf.data[0], 7.0);
        assert_eq!(buf.data[4], 7.0);
        assert_eq!(buf.data[8], 7.0);
        assert_eq!(buf.data[12], 7.0);
        assert_eq!(buf.data[1], 0.0); // untouched
    }

    #[test]
    fn first_touch_strided_stride_1() {
        let mut buf =
            NumaBuffer { data: vec![0.0; 8], node_id: 0, policy: InterleavingPolicy::Local };
        let n = first_touch_strided(&mut buf, 1.0, 1);
        assert_eq!(n, 8);
        assert!(buf.data.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn first_touch_strided_zero_stride() {
        let mut buf =
            NumaBuffer { data: vec![0.0; 4], node_id: 0, policy: InterleavingPolicy::Local };
        let n = first_touch_strided(&mut buf, 1.0, 0);
        assert_eq!(n, 0);
    }

    // == Distance matrix builders =======================================

    #[test]
    fn uniform_distance_matrix_diagonal() {
        let d = build_uniform_distance_matrix(3, 20);
        for i in 0..3 {
            assert_eq!(d[i][i], 10);
        }
    }

    #[test]
    fn uniform_distance_matrix_off_diagonal() {
        let d = build_uniform_distance_matrix(3, 21);
        assert_eq!(d[0][1], 21);
        assert_eq!(d[1][2], 21);
    }

    #[test]
    fn uniform_distance_matrix_single_node() {
        let d = build_uniform_distance_matrix(1, 20);
        assert_eq!(d, vec![vec![10]]);
    }

    #[test]
    fn hierarchical_distance_matrix_intra_socket() {
        let d = build_hierarchical_distance_matrix(4, 2, 15, 25);
        // Nodes 0,1 share a socket.
        assert_eq!(d[0][1], 15);
        assert_eq!(d[1][0], 15);
    }

    #[test]
    fn hierarchical_distance_matrix_inter_socket() {
        let d = build_hierarchical_distance_matrix(4, 2, 15, 25);
        assert_eq!(d[0][2], 25);
        assert_eq!(d[0][3], 25);
    }

    #[test]
    fn hierarchical_distance_matrix_diagonal() {
        let d = build_hierarchical_distance_matrix(4, 2, 15, 25);
        for i in 0..4 {
            assert_eq!(d[i][i], 10);
        }
    }

    #[test]
    fn hierarchical_zero_nodes_per_socket_fallback() {
        let d = build_hierarchical_distance_matrix(3, 0, 15, 25);
        // Should behave like uniform with remote_distance=25.
        assert_eq!(d[0][1], 25);
    }

    // == Nearest neighbour ==============================================

    #[test]
    fn nearest_neighbor_two_nodes() {
        let t = two_node_topology();
        assert_eq!(nearest_neighbor(0, &t).unwrap(), 1);
        assert_eq!(nearest_neighbor(1, &t).unwrap(), 0);
    }

    #[test]
    fn nearest_neighbor_four_nodes_hierarchical() {
        let t = four_node_topology();
        // Nodes 0,1 share a socket (distance 15), 2,3 share another.
        assert_eq!(nearest_neighbor(0, &t).unwrap(), 1);
        assert_eq!(nearest_neighbor(2, &t).unwrap(), 3);
    }

    #[test]
    fn nearest_neighbor_single_node_self() {
        let t = single_node_topology();
        assert_eq!(nearest_neighbor(0, &t).unwrap(), 0);
    }

    #[test]
    fn nearest_neighbor_bad_node() {
        let t = two_node_topology();
        assert!(nearest_neighbor(10, &t).is_err());
    }

    // == CPU-list parser ================================================

    #[test]
    fn parse_cpu_list_range() {
        assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn parse_cpu_list_mixed() {
        assert_eq!(parse_cpu_list("0-1,4,6-7"), vec![0, 1, 4, 6, 7]);
    }

    #[test]
    fn parse_cpu_list_single() {
        assert_eq!(parse_cpu_list("5"), vec![5]);
    }

    #[test]
    fn parse_cpu_list_empty() {
        assert!(parse_cpu_list("").is_empty());
    }

    // == Meminfo parser =================================================

    #[test]
    fn parse_node_meminfo_basic() {
        let raw = "Node 0 MemTotal:    16384000 kB\nNode 0 MemFree:     8192000 kB\n";
        let (total, free) = parse_node_meminfo(raw);
        assert_eq!(total, 16_384_000 * 1024);
        assert_eq!(free, 8_192_000 * 1024);
    }

    #[test]
    fn parse_node_meminfo_missing_free() {
        let raw = "Node 0 MemTotal:    1024 kB\n";
        let (total, free) = parse_node_meminfo(raw);
        assert_eq!(total, 1024 * 1024);
        assert_eq!(free, 0);
    }

    #[test]
    fn parse_node_meminfo_empty() {
        let (total, free) = parse_node_meminfo("");
        assert_eq!(total, 0);
        assert_eq!(free, 0);
    }

    // == Error display ==================================================

    #[test]
    fn error_display_invalid_config() {
        let e = NumaError::InvalidConfig("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn error_display_node_not_found() {
        let e = NumaError::NodeNotFound { node_id: 5, total_nodes: 2 };
        let s = e.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("2"));
    }

    #[test]
    fn error_display_allocation_failed() {
        let e = NumaError::AllocationFailed { node_id: 1, bytes: 1024 };
        assert!(e.to_string().contains("1024"));
    }

    #[test]
    fn error_display_uneven_partition() {
        let e = NumaError::UnevenPartition { elements: 7, num_nodes: 3 };
        assert!(e.to_string().contains("7"));
    }

    #[test]
    fn error_display_pinning_failed() {
        let e = NumaError::PinningFailed("oops".into());
        assert!(e.to_string().contains("oops"));
    }

    #[test]
    fn error_display_topology_unavailable() {
        let e = NumaError::TopologyUnavailable("nope".into());
        assert!(e.to_string().contains("nope"));
    }

    #[test]
    fn error_display_cpu_not_on_node() {
        let e = NumaError::CpuNotOnNode { cpu_id: 3, node_id: 1 };
        let s = e.to_string();
        assert!(s.contains("3"));
        assert!(s.contains("1"));
    }

    // == InterleavingPolicy display =====================================

    #[test]
    fn interleaving_policy_display() {
        assert_eq!(InterleavingPolicy::Local.to_string(), "Local");
        assert_eq!(InterleavingPolicy::RoundRobin.to_string(), "RoundRobin");
        assert_eq!(InterleavingPolicy::Weighted.to_string(), "Weighted");
    }

    #[test]
    fn interleaving_policy_default_is_local() {
        assert_eq!(InterleavingPolicy::default(), InterleavingPolicy::Local);
    }

    // == Metrics ========================================================

    #[test]
    fn metrics_default_zeroed() {
        let m = NumaMetrics::default();
        assert_eq!(m.detection_time_ms, 0.0);
        assert_eq!(m.placement_time_ms, 0.0);
        assert_eq!(m.cross_node_transfers, 0);
        assert_eq!(m.cross_node_elements, 0);
    }

    // == Integration: partition → transfer → gather =====================

    #[test]
    fn integration_partition_transfer_gather() {
        let t = two_node_topology();
        let data: Vec<f32> = (0..50).map(|x| x as f32).collect();
        let (parts, _) = partition_tensor(&data, &t, false).unwrap();
        // Transfer partition 0 to node 1.
        let buf0 = NumaBuffer {
            data: parts[0].data.clone(),
            node_id: parts[0].node_id,
            policy: InterleavingPolicy::Local,
        };
        let (moved, m) = execute_transfer(&buf0, 1, &t).unwrap();
        assert_eq!(m.cross_node_transfers, 1);
        assert_eq!(moved.data, parts[0].data);
        // Gather originals (no transfer needed for correctness).
        let gathered = gather_partitions(&parts).unwrap();
        assert_eq!(gathered, data);
    }

    #[test]
    fn integration_allocate_first_touch_verify() {
        let t = two_node_topology();
        let (mut buf, _) = allocate_on_node(128, 1, &t, InterleavingPolicy::Local).unwrap();
        assert!(buf.data.iter().all(|&v| v == 0.0));
        first_touch_init(&mut buf, 42.0);
        assert!(buf.data.iter().all(|&v| v == 42.0));
        assert_eq!(buf.node_id, 1);
    }

    #[test]
    fn integration_four_node_partition_gather() {
        let t = four_node_topology();
        let data: Vec<f32> = (0..1000).map(|x| x as f32).collect();
        let (parts, _) = partition_tensor(&data, &t, false).unwrap();
        assert_eq!(parts.len(), 4);
        let gathered = gather_partitions(&parts).unwrap();
        assert_eq!(gathered, data);
    }

    #[test]
    fn integration_weighted_partition_preserves_data() {
        let t = asymmetric_topology();
        let data: Vec<f32> = (0..500).map(|x| x as f32).collect();
        let (parts, _) = partition_tensor(&data, &t, true).unwrap();
        let gathered = gather_partitions(&parts).unwrap();
        assert_eq!(gathered, data);
    }

    #[test]
    fn integration_pin_plan_covers_all_nodes() {
        let t = four_node_topology();
        let plan = plan_thread_pinning(8, &t).unwrap();
        let nodes: std::collections::HashSet<usize> =
            plan.assignments.iter().map(|a| a.2).collect();
        for i in 0..4 {
            assert!(nodes.contains(&i), "node {i} not covered");
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_topology() -> impl Strategy<Value = NumaTopology> {
        (1..=4usize).prop_flat_map(|num_nodes| {
            let cpus_per = prop::collection::vec(1..=8usize, num_nodes..=num_nodes);
            let mem_per =
                prop::collection::vec(1_000_000_000u64..=64_000_000_000u64, num_nodes..=num_nodes);
            (Just(num_nodes), cpus_per, mem_per).prop_map(|(n, cpus_per, mem_per)| {
                let mut cpu_id = 0;
                let nodes: Vec<NumaNodeInfo> = (0..n)
                    .map(|i| {
                        let cpus: Vec<usize> = (cpu_id..cpu_id + cpus_per[i]).collect();
                        cpu_id += cpus_per[i];
                        NumaNodeInfo {
                            node_id: i,
                            cpus,
                            memory_bytes: mem_per[i],
                            free_memory_bytes: mem_per[i] / 2,
                        }
                    })
                    .collect();
                let distances = build_uniform_distance_matrix(n, 20);
                NumaTopology { nodes, distances }
            })
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_partition_preserves_total(
            topo in arb_topology(),
            len in 1..2000usize,
        ) {
            let data: Vec<f32> = (0..len).map(|x| x as f32).collect();
            let (parts, _) = partition_tensor(&data, &topo, false).unwrap();
            let total: usize = parts.iter().map(|p| p.data.len()).sum();
            prop_assert_eq!(total, len);
        }

        #[test]
        fn prop_partition_gather_round_trip(
            topo in arb_topology(),
            len in 1..2000usize,
        ) {
            let data: Vec<f32> = (0..len).map(|x| x as f32).collect();
            let (parts, _) = partition_tensor(&data, &topo, false).unwrap();
            let gathered = gather_partitions(&parts).unwrap();
            prop_assert_eq!(gathered, data);
        }

        #[test]
        fn prop_interleaved_total_preserved(
            topo in arb_topology(),
            len in 1..5000usize,
        ) {
            let (bufs, _) = allocate_interleaved(len, &topo).unwrap();
            let total: usize = bufs.iter().map(|b| b.data.len()).sum();
            prop_assert_eq!(total, len);
        }

        #[test]
        fn prop_weighted_total_preserved(
            topo in arb_topology(),
            len in 1..5000usize,
        ) {
            let (bufs, _) = allocate_weighted(len, &topo).unwrap();
            let total: usize = bufs.iter().map(|b| b.data.len()).sum();
            prop_assert_eq!(total, len);
        }

        #[test]
        fn prop_topology_validates(topo in arb_topology()) {
            prop_assert!(topo.validate().is_ok());
        }

        #[test]
        fn prop_pin_plan_has_correct_count(
            topo in arb_topology(),
            workers in 1..32usize,
        ) {
            let plan = plan_thread_pinning(workers, &topo).unwrap();
            prop_assert_eq!(plan.assignments.len(), workers);
        }

        #[test]
        fn prop_first_touch_fills_all(
            len in 1..4096usize,
            val in -100.0f32..100.0,
        ) {
            let mut buf = NumaBuffer {
                data: vec![0.0; len],
                node_id: 0,
                policy: InterleavingPolicy::Local,
            };
            let n = first_touch_init(&mut buf, val);
            prop_assert_eq!(n, len);
            prop_assert!(buf.data.iter().all(|&v| v == val));
        }

        #[test]
        fn prop_nearest_neighbor_valid(topo in arb_topology()) {
            for i in 0..topo.num_nodes() {
                let nn = nearest_neighbor(i, &topo).unwrap();
                prop_assert!(nn < topo.num_nodes());
            }
        }
    }
}
