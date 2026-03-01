//! Module stub - implementation pending merge from feature branch
//! DAG-based execution planner for GPU inference workloads.
//!
//! Plans, schedules, and optimizes execution graphs for neural network inference.
//! Includes memory planning, stream scheduling, kernel launch configuration,
//! pipeline parallelism, cost modeling, and plan optimization (fusion, reordering,
//! memory reuse).
//!
//! All types provide CPU reference implementations suitable for testing and
//! fallback when GPU hardware is unavailable.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Operation kind ──────────────────────────────────────────────────────────

/// Kind of compute operation in the execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    MatMul,
    Conv,
    Attention,
    LayerNorm,
    Activation,
    Elementwise,
    Reduce,
    Transpose,
    Gather,
    Scatter,
    Softmax,
    Embedding,
    Custom,
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::MatMul => "MatMul",
            Self::Conv => "Conv",
            Self::Attention => "Attention",
            Self::LayerNorm => "LayerNorm",
            Self::Activation => "Activation",
            Self::Elementwise => "Elementwise",
            Self::Reduce => "Reduce",
            Self::Transpose => "Transpose",
            Self::Gather => "Gather",
            Self::Scatter => "Scatter",
            Self::Softmax => "Softmax",
            Self::Embedding => "Embedding",
            Self::Custom => "Custom",
        };
        write!(f, "{s}")
    }
}

// ── Optimization level ──────────────────────────────────────────────────────

/// Optimization aggressiveness for the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OptimizationLevel {
    /// No optimizations; baseline schedule.
    None,
    /// Basic dead-node elimination and topological ordering.
    Basic,
    /// Memory reuse and stream overlap.
    #[default]
    Standard,
    /// Aggressive fusion, reordering, and pipeline parallelism.
    Aggressive,
}

// ── PlanConfig ──────────────────────────────────────────────────────────────

/// Configuration for the execution planner.
#[derive(Debug, Clone)]
pub struct PlanConfig {
    /// Optimization level.
    pub optimization_level: OptimizationLevel,
    /// Maximum device memory budget in bytes.
    pub max_memory_bytes: u64,
    /// Maximum number of concurrent compute streams/queues.
    pub max_parallelism: u32,
    /// Whether to enable operator fusion.
    pub enable_fusion: bool,
    /// Whether to enable memory reuse across non-overlapping lifetimes.
    pub enable_memory_reuse: bool,
    /// Alignment requirement for memory allocations (bytes).
    pub memory_alignment: u64,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Standard,
            max_memory_bytes: 4 * 1024 * 1024 * 1024, // 4 GiB
            max_parallelism: 4,
            enable_fusion: true,
            enable_memory_reuse: true,
            memory_alignment: 256,
        }
    }
}

impl PlanConfig {
    /// Create a config with the given memory budget and parallelism.
    pub const fn new(max_memory_bytes: u64, max_parallelism: u32) -> Self {
        Self {
            optimization_level: OptimizationLevel::Standard,
            max_memory_bytes,
            max_parallelism,
            enable_fusion: true,
            enable_memory_reuse: true,
            memory_alignment: 256,
        }
    }

    /// Return aligned size (rounds up to `memory_alignment`).
    pub const fn align(&self, size: u64) -> u64 {
        if self.memory_alignment == 0 {
            return size;
        }
        size.div_ceil(self.memory_alignment) * self.memory_alignment
    }
}

// ── ExecutionNode ───────────────────────────────────────────────────────────

/// Unique identifier for a node in the execution graph.
pub type NodeId = usize;

/// A single node in the execution graph.
#[derive(Debug, Clone)]
pub struct ExecutionNode {
    /// Unique identifier.
    pub id: NodeId,
    /// Human-readable label.
    pub label: String,
    /// Kind of operation.
    pub op: OpKind,
    /// IDs of nodes this node depends on.
    pub dependencies: Vec<NodeId>,
    /// Output memory requirement in bytes.
    pub output_bytes: u64,
    /// Workspace (scratch) memory requirement in bytes.
    pub workspace_bytes: u64,
    /// Estimated FLOP count for cost modeling.
    pub flops: u64,
    /// Shape of the primary output tensor (for launch planning).
    pub output_shape: Vec<usize>,
}

impl ExecutionNode {
    /// Create a new execution node.
    pub fn new(id: NodeId, label: impl Into<String>, op: OpKind) -> Self {
        Self {
            id,
            label: label.into(),
            op,
            dependencies: Vec::new(),
            output_bytes: 0,
            workspace_bytes: 0,
            flops: 0,
            output_shape: Vec::new(),
        }
    }

    /// Add a dependency.
    #[must_use]
    pub fn with_dep(mut self, dep: NodeId) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Set output bytes.
    #[must_use]
    pub const fn with_output_bytes(mut self, bytes: u64) -> Self {
        self.output_bytes = bytes;
        self
    }

    /// Set workspace bytes.
    #[must_use]
    pub const fn with_workspace_bytes(mut self, bytes: u64) -> Self {
        self.workspace_bytes = bytes;
        self
    }

    /// Set estimated FLOPs.
    #[must_use]
    pub const fn with_flops(mut self, flops: u64) -> Self {
        self.flops = flops;
        self
    }

    /// Set output shape.
    #[must_use]
    pub fn with_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shape = shape;
        self
    }

    /// Total memory this node needs (output + workspace).
    pub const fn total_memory(&self) -> u64 {
        self.output_bytes + self.workspace_bytes
    }
}

// ── ExecutionGraph ──────────────────────────────────────────────────────────

/// Directed acyclic graph of execution nodes.
#[derive(Debug, Clone)]
pub struct ExecutionGraph {
    nodes: Vec<ExecutionNode>,
    adjacency: HashMap<NodeId, Vec<NodeId>>,
}

impl ExecutionGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new(), adjacency: HashMap::new() }
    }

    /// Add a node. Returns its id.
    pub fn add_node(&mut self, node: ExecutionNode) -> NodeId {
        let id = node.id;
        for &dep in &node.dependencies {
            self.adjacency.entry(dep).or_default().push(id);
        }
        self.nodes.push(node);
        id
    }

    /// Number of nodes.
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty.
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get a node by id.
    pub fn node(&self, id: NodeId) -> Option<&ExecutionNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Return all nodes.
    pub fn nodes(&self) -> &[ExecutionNode] {
        &self.nodes
    }

    /// Successors (consumers) of a given node.
    pub fn successors(&self, id: NodeId) -> &[NodeId] {
        self.adjacency.get(&id).map(Vec::as_slice).unwrap_or_default()
    }

    /// Topological sort (Kahn's algorithm). Returns `None` if cyclic.
    pub fn topological_sort(&self) -> Option<Vec<NodeId>> {
        let mut in_deg: HashMap<NodeId, usize> = HashMap::new();
        for n in &self.nodes {
            in_deg.entry(n.id).or_insert(0);
        }
        for n in &self.nodes {
            for succ in self.successors(n.id) {
                *in_deg.entry(*succ).or_insert(0) += 1;
            }
        }

        let mut q_vec: Vec<NodeId> =
            in_deg.iter().filter(|&(_, &d)| d == 0).map(|(&id, _)| id).collect();
        q_vec.sort_unstable();
        let mut queue: VecDeque<NodeId> = q_vec.into();

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = queue.pop_front() {
            order.push(id);
            let mut next: Vec<NodeId> = Vec::new();
            for &succ in self.successors(id) {
                if let Some(d) = in_deg.get_mut(&succ) {
                    *d -= 1;
                    if *d == 0 {
                        next.push(succ);
                    }
                }
            }
            next.sort_unstable();
            queue.extend(next);
        }

        if order.len() == self.nodes.len() {
            Some(order)
        } else {
            None // cycle detected
        }
    }

    /// Return root nodes (no dependencies).
    pub fn roots(&self) -> Vec<NodeId> {
        self.nodes.iter().filter(|n| n.dependencies.is_empty()).map(|n| n.id).collect()
    }

    /// Return leaf nodes (no successors).
    pub fn leaves(&self) -> Vec<NodeId> {
        self.nodes.iter().filter(|n| self.successors(n.id).is_empty()).map(|n| n.id).collect()
    }

    /// Total output bytes across all nodes.
    pub fn total_output_bytes(&self) -> u64 {
        self.nodes.iter().map(|n| n.output_bytes).sum()
    }

    /// Total FLOPs across all nodes.
    pub fn total_flops(&self) -> u64 {
        self.nodes.iter().map(|n| n.flops).sum()
    }
}

impl Default for ExecutionGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── MemoryPlanner ───────────────────────────────────────────────────────────

/// A planned memory allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryAllocation {
    /// Node whose output this allocation serves.
    pub node_id: NodeId,
    /// Byte offset within the memory pool.
    pub offset: u64,
    /// Allocation size in bytes (aligned).
    pub size: u64,
    /// Whether this allocation reuses a previously freed slot.
    pub reused: bool,
}

/// Plans memory allocation and reuse across an execution graph.
#[derive(Debug, Clone)]
pub struct MemoryPlanner {
    alignment: u64,
    enable_reuse: bool,
    allocations: Vec<MemoryAllocation>,
    peak_usage: u64,
}

impl MemoryPlanner {
    /// Create a planner from config.
    pub const fn new(config: &PlanConfig) -> Self {
        Self {
            alignment: config.memory_alignment,
            enable_reuse: config.enable_memory_reuse,
            allocations: Vec::new(),
            peak_usage: 0,
        }
    }

    /// Align a size up.
    const fn align(&self, size: u64) -> u64 {
        if self.alignment == 0 {
            return size;
        }
        size.div_ceil(self.alignment) * self.alignment
    }

    /// Plan allocations for the graph in the given execution order.
    ///
    /// Uses a greedy first-fit reuse strategy: when a node's output is no
    /// longer needed (all consumers have executed), its slot becomes available
    /// for reuse.
    pub fn plan(&mut self, graph: &ExecutionGraph, exec_order: &[NodeId]) -> &[MemoryAllocation] {
        self.allocations.clear();
        self.peak_usage = 0;

        // last_use[node] = position in exec_order of the last consumer
        let mut last_use: HashMap<NodeId, usize> = HashMap::new();
        for (pos, &nid) in exec_order.iter().enumerate() {
            if let Some(node) = graph.node(nid) {
                for &dep in &node.dependencies {
                    let entry = last_use.entry(dep).or_insert(0);
                    if pos > *entry {
                        *entry = pos;
                    }
                }
            }
        }
        // Nodes with no consumers: last_use = their own position
        for &nid in exec_order {
            last_use
                .entry(nid)
                .or_insert_with(|| exec_order.iter().position(|&x| x == nid).unwrap_or(0));
        }

        // free_slots: (offset, size) sorted by offset
        let mut free_slots: Vec<(u64, u64)> = Vec::new();
        // active: node_id -> (offset, size, last_use_pos)
        let mut active: HashMap<NodeId, (u64, u64, usize)> = HashMap::new();
        let mut next_offset: u64 = 0;

        for (pos, &nid) in exec_order.iter().enumerate() {
            // Free allocations whose lifetime has ended
            let expired: Vec<NodeId> =
                active.iter().filter(|&(_, &(_, _, lu))| lu < pos).map(|(&id, _)| id).collect();
            for eid in expired {
                if let Some((off, sz, _)) = active.remove(&eid)
                    && self.enable_reuse
                {
                    free_slots.push((off, sz));
                    free_slots.sort_by_key(|&(o, _)| o);
                }
            }

            let Some(node) = graph.node(nid) else {
                continue;
            };
            let needed = self.align(node.output_bytes);
            if needed == 0 {
                self.allocations.push(MemoryAllocation {
                    node_id: nid,
                    offset: 0,
                    size: 0,
                    reused: false,
                });
                continue;
            }

            // Try to reuse a free slot (first-fit)
            let mut reused = false;
            let mut alloc_offset = 0u64;
            if self.enable_reuse
                && let Some(idx) = free_slots.iter().position(|&(_, sz)| sz >= needed)
            {
                let (off, _) = free_slots.remove(idx);
                alloc_offset = off;
                reused = true;
            }
            if !reused {
                alloc_offset = next_offset;
                next_offset += needed;
            }

            let lu = last_use.get(&nid).copied().unwrap_or(pos);
            active.insert(nid, (alloc_offset, needed, lu));
            self.allocations.push(MemoryAllocation {
                node_id: nid,
                offset: alloc_offset,
                size: needed,
                reused,
            });

            let current_peak = active.values().map(|&(o, s, _)| o + s).max().unwrap_or(0);
            if current_peak > self.peak_usage {
                self.peak_usage = current_peak;
            }
        }

        &self.allocations
    }

    /// Peak memory usage after planning.
    pub const fn peak_usage(&self) -> u64 {
        self.peak_usage
    }

    /// All computed allocations.
    pub fn allocations(&self) -> &[MemoryAllocation] {
        &self.allocations
    }
}

// ── StreamScheduler ─────────────────────────────────────────────────────────

/// Assignment of a node to a specific compute stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamAssignment {
    pub node_id: NodeId,
    pub stream_id: u32,
    /// Position within the stream (execution step).
    pub step: usize,
}

/// Schedules nodes across multiple compute streams/queues.
#[derive(Debug, Clone)]
pub struct StreamScheduler {
    max_streams: u32,
    assignments: Vec<StreamAssignment>,
}

impl StreamScheduler {
    /// Create a scheduler with the given maximum number of streams.
    pub const fn new(max_streams: u32) -> Self {
        let max_streams = if max_streams == 0 { 1 } else { max_streams };
        Self { max_streams, assignments: Vec::new() }
    }

    /// Schedule nodes given the graph and a topological order.
    ///
    /// Uses a level-based assignment: nodes at the same depth (longest path
    /// from root) may execute in parallel on different streams.
    pub fn schedule(
        &mut self,
        graph: &ExecutionGraph,
        exec_order: &[NodeId],
    ) -> &[StreamAssignment] {
        self.assignments.clear();

        // Compute depth (longest path from any root) for each node
        let mut depth: HashMap<NodeId, usize> = HashMap::new();
        for &nid in exec_order {
            let Some(node) = graph.node(nid) else {
                continue;
            };
            let d = node
                .dependencies
                .iter()
                .filter_map(|dep| depth.get(dep))
                .max()
                .map_or(0, |m| m + 1);
            depth.insert(nid, d);
        }

        // Group by depth
        let max_depth = depth.values().copied().max().unwrap_or(0);
        let mut levels: Vec<Vec<NodeId>> = vec![Vec::new(); max_depth + 1];
        for &nid in exec_order {
            if let Some(&d) = depth.get(&nid) {
                levels[d].push(nid);
            }
        }

        // Round-robin within each level
        let mut stream_step: Vec<usize> = vec![0; self.max_streams as usize];
        for level in &levels {
            for (i, &nid) in level.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let sid = (i as u32) % self.max_streams;
                let step = stream_step[sid as usize];
                stream_step[sid as usize] += 1;
                self.assignments.push(StreamAssignment { node_id: nid, stream_id: sid, step });
            }
        }

        &self.assignments
    }

    /// Return assignments.
    pub fn assignments(&self) -> &[StreamAssignment] {
        &self.assignments
    }

    /// Number of streams actually used.
    #[allow(clippy::cast_possible_truncation)]
    pub fn streams_used(&self) -> u32 {
        self.assignments.iter().map(|a| a.stream_id).collect::<HashSet<_>>().len() as u32
    }
}

// ── LaunchPlanner ───────────────────────────────────────────────────────────

/// Kernel launch configuration (grid/block dimensions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchConfig {
    pub node_id: NodeId,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Shared memory in bytes.
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Total number of threads in one block.
    pub const fn threads_per_block(&self) -> u32 {
        self.block[0] * self.block[1] * self.block[2]
    }

    /// Total number of blocks.
    pub const fn total_blocks(&self) -> u64 {
        self.grid[0] as u64 * self.grid[1] as u64 * self.grid[2] as u64
    }

    /// Total number of threads launched.
    pub const fn total_threads(&self) -> u64 {
        self.total_blocks() * self.threads_per_block() as u64
    }
}

/// Plans kernel launch configurations for execution nodes.
#[derive(Debug, Clone)]
pub struct LaunchPlanner {
    max_threads_per_block: u32,
    max_shared_mem: u32,
    warp_size: u32,
    configs: Vec<LaunchConfig>,
}

impl LaunchPlanner {
    /// Create a launch planner with device limits.
    pub const fn new(max_threads_per_block: u32, max_shared_mem: u32, warp_size: u32) -> Self {
        let max_threads_per_block =
            if max_threads_per_block < 32 { 32 } else { max_threads_per_block };
        let warp_size = if warp_size == 0 { 1 } else { warp_size };
        Self { max_threads_per_block, max_shared_mem, warp_size, configs: Vec::new() }
    }

    /// CPU-reference defaults (256 threads/block, 48 KiB shared, warp 32).
    pub const fn cpu_default() -> Self {
        Self::new(256, 48 * 1024, 32)
    }

    /// Plan launch configs for all nodes in the graph.
    pub fn plan(&mut self, graph: &ExecutionGraph) -> &[LaunchConfig] {
        self.configs.clear();
        for node in graph.nodes() {
            self.configs.push(self.plan_node(node));
        }
        &self.configs
    }

    /// Plan a single node's launch config.
    #[allow(clippy::cast_possible_truncation)]
    fn plan_node(&self, node: &ExecutionNode) -> LaunchConfig {
        let total_elements: u64 = if node.output_shape.is_empty() {
            1
        } else {
            node.output_shape.iter().product::<usize>() as u64
        };

        // Choose block size: round up to warp multiple, cap at max
        let raw = total_elements.min(u64::from(self.max_threads_per_block)) as u32;
        let block_x = raw
            .div_ceil(self.warp_size)
            .saturating_mul(self.warp_size)
            .min(self.max_threads_per_block)
            .max(self.warp_size);

        // Grid covers all elements
        let grid_x = total_elements.div_ceil(u64::from(block_x)).min(u64::from(u32::MAX)) as u32;
        let grid_x = grid_x.max(1);

        // Shared memory heuristic per operation kind
        let shared = match node.op {
            OpKind::MatMul | OpKind::Attention | OpKind::Reduce | OpKind::Softmax => {
                (block_x * 4).min(self.max_shared_mem)
            }
            _ => 0,
        };

        LaunchConfig {
            node_id: node.id,
            grid: [grid_x, 1, 1],
            block: [block_x, 1, 1],
            shared_mem_bytes: shared,
        }
    }

    /// All computed launch configs.
    pub fn configs(&self) -> &[LaunchConfig] {
        &self.configs
    }
}

// ── PipelineStage ───────────────────────────────────────────────────────────

/// A stage in a pipeline-parallel execution scheme.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Nodes assigned to this stage.
    pub node_ids: Vec<NodeId>,
    /// Estimated time for this stage in microseconds.
    pub estimated_time_us: u64,
    /// Memory footprint in bytes.
    pub memory_bytes: u64,
}

impl PipelineStage {
    /// Create an empty stage.
    pub const fn new(stage_id: u32) -> Self {
        Self { stage_id, node_ids: Vec::new(), estimated_time_us: 0, memory_bytes: 0 }
    }

    /// Add a node to this stage, updating estimates.
    pub fn add_node(&mut self, node: &ExecutionNode, cost: &CostEstimate) {
        self.node_ids.push(node.id);
        self.estimated_time_us += cost.time_us;
        self.memory_bytes += node.total_memory();
    }

    /// Number of nodes in this stage.
    pub const fn node_count(&self) -> usize {
        self.node_ids.len()
    }

    /// Whether the stage is empty.
    pub const fn is_empty(&self) -> bool {
        self.node_ids.is_empty()
    }
}

/// Partitions an execution graph into pipeline stages.
#[derive(Debug, Clone)]
pub struct PipelinePartitioner {
    num_stages: u32,
    stages: Vec<PipelineStage>,
}

impl PipelinePartitioner {
    /// Create a partitioner targeting `num_stages` stages.
    pub const fn new(num_stages: u32) -> Self {
        let num_stages = if num_stages == 0 { 1 } else { num_stages };
        Self { num_stages, stages: Vec::new() }
    }

    /// Partition the graph into roughly equal-cost stages.
    pub fn partition(
        &mut self,
        graph: &ExecutionGraph,
        exec_order: &[NodeId],
        cost_model: &CostModel,
    ) -> &[PipelineStage] {
        self.stages.clear();

        let costs: Vec<(NodeId, CostEstimate)> = exec_order
            .iter()
            .filter_map(|&nid| graph.node(nid).map(|n| (nid, cost_model.estimate(n))))
            .collect();

        let total_cost: u64 = costs.iter().map(|(_, c)| c.time_us).sum();
        let target_per_stage =
            if self.num_stages == 0 { total_cost } else { total_cost / u64::from(self.num_stages) };
        let target_per_stage = target_per_stage.max(1);

        let mut current = PipelineStage::new(0);
        let mut stage_idx = 0u32;

        for (nid, cost) in &costs {
            if current.estimated_time_us >= target_per_stage && stage_idx + 1 < self.num_stages {
                self.stages.push(current);
                stage_idx += 1;
                current = PipelineStage::new(stage_idx);
            }
            if let Some(node) = graph.node(*nid) {
                current.add_node(node, cost);
            }
        }
        if !current.is_empty() {
            self.stages.push(current);
        }

        &self.stages
    }

    /// Computed stages.
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// The stage with the longest estimated time (bottleneck).
    pub fn bottleneck(&self) -> Option<&PipelineStage> {
        self.stages.iter().max_by_key(|s| s.estimated_time_us)
    }
}

// ── CostModel ───────────────────────────────────────────────────────────────

/// Estimated cost for one operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostEstimate {
    /// Estimated wall-clock time in microseconds.
    pub time_us: u64,
    /// Memory bandwidth consumed in bytes.
    pub bandwidth_bytes: u64,
    /// FLOPs performed.
    pub flops: u64,
}

/// Estimates execution cost per operation (time, memory bandwidth, FLOPs).
///
/// Uses a simple roofline-style model: cost is the maximum of compute-bound
/// and memory-bound estimates.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Peak compute throughput in GFLOP/s.
    pub peak_gflops: f64,
    /// Peak memory bandwidth in GB/s.
    pub peak_bandwidth_gbs: f64,
    /// Per-kernel launch overhead in microseconds.
    pub launch_overhead_us: u64,
}

impl Default for CostModel {
    /// CPU reference: modest throughput.
    fn default() -> Self {
        Self { peak_gflops: 100.0, peak_bandwidth_gbs: 50.0, launch_overhead_us: 5 }
    }
}

impl CostModel {
    /// Create a cost model with given hardware parameters.
    #[allow(clippy::missing_const_for_fn)] // f64::max not const-stable
    pub fn new(peak_gflops: f64, peak_bandwidth_gbs: f64, launch_overhead_us: u64) -> Self {
        Self {
            peak_gflops: peak_gflops.max(0.001),
            peak_bandwidth_gbs: peak_bandwidth_gbs.max(0.001),
            launch_overhead_us,
        }
    }

    /// Estimate cost for an execution node.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
    pub fn estimate(&self, node: &ExecutionNode) -> CostEstimate {
        let flops = node.flops;
        let bytes = node.output_bytes + node.workspace_bytes;

        // Compute-bound time (µs): GFLOP/s → MFLOP/µs
        let compute_us = if self.peak_gflops > 0.0 {
            (flops as f64 / (self.peak_gflops * 1e3)) as u64
        } else {
            0
        };

        // Memory-bound time (µs): GB/s → MB/µs
        let mem_us = if self.peak_bandwidth_gbs > 0.0 {
            (bytes as f64 / (self.peak_bandwidth_gbs * 1e3)) as u64
        } else {
            0
        };

        let time_us = compute_us.max(mem_us) + self.launch_overhead_us;

        CostEstimate { time_us, bandwidth_bytes: bytes, flops }
    }

    /// Total estimated cost over a sequence of nodes.
    pub fn total_cost<'a>(&self, nodes: impl Iterator<Item = &'a ExecutionNode>) -> CostEstimate {
        let mut total = CostEstimate { time_us: 0, bandwidth_bytes: 0, flops: 0 };
        for n in nodes {
            let c = self.estimate(n);
            total.time_us += c.time_us;
            total.bandwidth_bytes += c.bandwidth_bytes;
            total.flops += c.flops;
        }
        total
    }
}

// ── PlanOptimizer ───────────────────────────────────────────────────────────

/// Describes a fusion of multiple nodes into one.
#[derive(Debug, Clone)]
pub struct FusionGroup {
    /// Nodes that were fused.
    pub node_ids: Vec<NodeId>,
    /// Label for the fused kernel.
    pub fused_label: String,
}

/// Result of optimization.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Reordered execution order.
    pub exec_order: Vec<NodeId>,
    /// Fusion groups found.
    pub fusions: Vec<FusionGroup>,
    /// Nodes eliminated (dead code).
    pub eliminated: Vec<NodeId>,
    /// Estimated speedup factor.
    pub estimated_speedup: f64,
}

/// Optimizes an execution plan via fusion, reordering, and dead-code
/// elimination.
#[derive(Debug, Clone)]
pub struct PlanOptimizer {
    config: PlanConfig,
}

impl PlanOptimizer {
    /// Create an optimizer with the given config.
    pub const fn new(config: PlanConfig) -> Self {
        Self { config }
    }

    /// Optimize the graph. Returns the optimization result.
    pub fn optimize(&self, graph: &ExecutionGraph) -> OptimizationResult {
        let Some(mut exec_order) = graph.topological_sort() else {
            return OptimizationResult {
                exec_order: graph.nodes().iter().map(|n| n.id).collect(),
                fusions: Vec::new(),
                eliminated: Vec::new(),
                estimated_speedup: 1.0,
            };
        };

        let mut fusions = Vec::new();
        let mut eliminated = Vec::new();

        // Dead-code elimination: remove zero-output, zero-flop leaf nodes.
        if self.config.optimization_level != OptimizationLevel::None {
            for &nid in &exec_order {
                if let Some(node) = graph.node(nid)
                    && node.output_bytes == 0
                    && graph.successors(nid).is_empty()
                    && node.flops == 0
                {
                    eliminated.push(nid);
                }
            }
            let elim_set: HashSet<NodeId> = eliminated.iter().copied().collect();
            exec_order.retain(|id| !elim_set.contains(id));
        }

        // Fusion: merge consecutive fusable pairs.
        if self.config.enable_fusion
            && matches!(
                self.config.optimization_level,
                OptimizationLevel::Standard | OptimizationLevel::Aggressive
            )
        {
            let mut i = 0;
            while i + 1 < exec_order.len() {
                let a_id = exec_order[i];
                let b_id = exec_order[i + 1];
                let fusable = graph.node(a_id).zip(graph.node(b_id)).is_some_and(|(a, b)| {
                    Self::can_fuse(a, b)
                        && b.dependencies.contains(&a_id)
                        && graph.successors(a_id).len() == 1
                });
                if fusable {
                    let a_label = graph.node(a_id).map_or("?", |n| n.label.as_str());
                    let b_label = graph.node(b_id).map_or("?", |n| n.label.as_str());
                    fusions.push(FusionGroup {
                        node_ids: vec![a_id, b_id],
                        fused_label: format!("fused({a_label}+{b_label})"),
                    });
                }
                i += 1;
            }
        }

        let num_fusions = fusions.len();
        let baseline_nodes = graph.len();
        let effective_nodes = baseline_nodes.saturating_sub(num_fusions);
        #[allow(clippy::cast_precision_loss)]
        let estimated_speedup =
            if effective_nodes > 0 { baseline_nodes as f64 / effective_nodes as f64 } else { 1.0 };

        OptimizationResult { exec_order, fusions, eliminated, estimated_speedup }
    }

    /// Whether two adjacent nodes can be fused.
    const fn can_fuse(a: &ExecutionNode, b: &ExecutionNode) -> bool {
        matches!(
            (a.op, b.op),
            (OpKind::Elementwise | OpKind::LayerNorm | OpKind::MatMul, OpKind::Activation)
                | (OpKind::Activation | OpKind::Elementwise, OpKind::Elementwise)
        )
    }
}

// ── ExecutionPlannerEngine ──────────────────────────────────────────────────

/// A fully assembled execution plan.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Ordered node IDs.
    pub exec_order: Vec<NodeId>,
    /// Memory allocations.
    pub allocations: Vec<MemoryAllocation>,
    /// Stream assignments.
    pub stream_assignments: Vec<StreamAssignment>,
    /// Launch configurations.
    pub launch_configs: Vec<LaunchConfig>,
    /// Pipeline stages.
    pub pipeline_stages: Vec<PipelineStage>,
    /// Optimization summary.
    pub optimization: OptimizationResult,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: u64,
    /// Estimated total time in microseconds.
    pub estimated_time_us: u64,
}

/// Top-level planner that assembles an optimized execution plan from a graph.
///
/// Orchestrates the optimizer, memory planner, stream scheduler, launch
/// planner, pipeline partitioner, and cost model.
#[derive(Debug, Clone)]
pub struct ExecutionPlannerEngine {
    config: PlanConfig,
    cost_model: CostModel,
}

impl ExecutionPlannerEngine {
    /// Create an engine with the given config and cost model.
    pub const fn new(config: PlanConfig, cost_model: CostModel) -> Self {
        Self { config, cost_model }
    }

    /// Create an engine with default CPU-reference settings.
    pub fn cpu_default() -> Self {
        Self::new(PlanConfig::default(), CostModel::default())
    }

    /// Generate a fully optimized execution plan for the given graph.
    pub fn plan(&self, graph: &ExecutionGraph) -> Result<ExecutionPlan, String> {
        if graph.is_empty() {
            return Ok(ExecutionPlan {
                exec_order: Vec::new(),
                allocations: Vec::new(),
                stream_assignments: Vec::new(),
                launch_configs: Vec::new(),
                pipeline_stages: Vec::new(),
                optimization: OptimizationResult {
                    exec_order: Vec::new(),
                    fusions: Vec::new(),
                    eliminated: Vec::new(),
                    estimated_speedup: 1.0,
                },
                peak_memory_bytes: 0,
                estimated_time_us: 0,
            });
        }

        // 1. Optimize
        let optimizer = PlanOptimizer::new(self.config.clone());
        let opt_result = optimizer.optimize(graph);
        let exec_order = &opt_result.exec_order;

        // 2. Memory planning
        let mut mem_planner = MemoryPlanner::new(&self.config);
        mem_planner.plan(graph, exec_order);
        let peak = mem_planner.peak_usage();

        if peak > self.config.max_memory_bytes {
            return Err(format!(
                "peak memory {peak} exceeds budget {}",
                self.config.max_memory_bytes
            ));
        }

        // 3. Stream scheduling
        let mut scheduler = StreamScheduler::new(self.config.max_parallelism);
        scheduler.schedule(graph, exec_order);

        // 4. Launch planning
        let mut launch = LaunchPlanner::cpu_default();
        launch.plan(graph);

        // 5. Pipeline partitioning
        let num_stages = self.config.max_parallelism.min(4);
        let mut partitioner = PipelinePartitioner::new(num_stages);
        partitioner.partition(graph, exec_order, &self.cost_model);

        // 6. Total cost estimate
        let total_cost =
            self.cost_model.total_cost(exec_order.iter().filter_map(|nid| graph.node(*nid)));

        Ok(ExecutionPlan {
            exec_order: exec_order.clone(),
            allocations: mem_planner.allocations().to_vec(),
            stream_assignments: scheduler.assignments().to_vec(),
            launch_configs: launch.configs().to_vec(),
            pipeline_stages: partitioner.stages().to_vec(),
            optimization: opt_result,
            peak_memory_bytes: peak,
            estimated_time_us: total_cost.time_us,
        })
    }

    /// Access the cost model.
    pub const fn cost_model(&self) -> &CostModel {
        &self.cost_model
    }

    /// Access the config.
    pub const fn config(&self) -> &PlanConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_node(id: NodeId, op: OpKind) -> ExecutionNode {
        ExecutionNode::new(id, format!("node_{id}"), op)
            .with_output_bytes(1024)
            .with_flops(1_000_000)
    }

    fn linear_graph(n: usize) -> ExecutionGraph {
        let mut g = ExecutionGraph::new();
        for i in 0..n {
            let mut node = simple_node(i, OpKind::MatMul);
            if i > 0 {
                node = node.with_dep(i - 1);
            }
            g.add_node(node);
        }
        g
    }

    fn diamond_graph() -> ExecutionGraph {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::Embedding));
        g.add_node(simple_node(1, OpKind::MatMul).with_dep(0));
        g.add_node(simple_node(2, OpKind::MatMul).with_dep(0));
        g.add_node(simple_node(3, OpKind::Reduce).with_dep(1).with_dep(2));
        g
    }

    // ── OpKind ──────────────────────────────────────────────────────────

    #[test]
    fn op_kind_display() {
        assert_eq!(format!("{}", OpKind::MatMul), "MatMul");
        assert_eq!(format!("{}", OpKind::Attention), "Attention");
        assert_eq!(format!("{}", OpKind::Custom), "Custom");
    }

    #[test]
    fn optimization_level_default() {
        assert_eq!(OptimizationLevel::default(), OptimizationLevel::Standard);
    }

    // ── PlanConfig ──────────────────────────────────────────────────────

    #[test]
    fn plan_config_default() {
        let cfg = PlanConfig::default();
        assert_eq!(cfg.max_memory_bytes, 4 * 1024 * 1024 * 1024);
        assert_eq!(cfg.max_parallelism, 4);
        assert!(cfg.enable_fusion);
        assert!(cfg.enable_memory_reuse);
        assert_eq!(cfg.memory_alignment, 256);
    }

    #[test]
    fn plan_config_new() {
        let cfg = PlanConfig::new(1024, 8);
        assert_eq!(cfg.max_memory_bytes, 1024);
        assert_eq!(cfg.max_parallelism, 8);
    }

    #[test]
    fn plan_config_align() {
        let cfg = PlanConfig::default();
        assert_eq!(cfg.align(0), 0);
        assert_eq!(cfg.align(1), 256);
        assert_eq!(cfg.align(256), 256);
        assert_eq!(cfg.align(257), 512);
        assert_eq!(cfg.align(512), 512);
    }

    #[test]
    fn plan_config_align_zero_alignment() {
        let mut cfg = PlanConfig::default();
        cfg.memory_alignment = 0;
        assert_eq!(cfg.align(100), 100);
    }

    // ── ExecutionNode ───────────────────────────────────────────────────

    #[test]
    fn execution_node_new() {
        let n = ExecutionNode::new(0, "test", OpKind::MatMul);
        assert_eq!(n.id, 0);
        assert_eq!(n.label, "test");
        assert_eq!(n.op, OpKind::MatMul);
        assert!(n.dependencies.is_empty());
        assert_eq!(n.output_bytes, 0);
    }

    #[test]
    fn execution_node_builder() {
        let n = ExecutionNode::new(1, "mm", OpKind::MatMul)
            .with_dep(0)
            .with_output_bytes(4096)
            .with_workspace_bytes(512)
            .with_flops(1_000_000)
            .with_output_shape(vec![32, 128]);
        assert_eq!(n.dependencies, vec![0]);
        assert_eq!(n.output_bytes, 4096);
        assert_eq!(n.workspace_bytes, 512);
        assert_eq!(n.flops, 1_000_000);
        assert_eq!(n.output_shape, vec![32, 128]);
    }

    #[test]
    fn execution_node_total_memory() {
        let n = ExecutionNode::new(0, "x", OpKind::Reduce)
            .with_output_bytes(1000)
            .with_workspace_bytes(500);
        assert_eq!(n.total_memory(), 1500);
    }

    #[test]
    fn execution_node_clone_and_debug() {
        let n = simple_node(0, OpKind::Conv);
        let n2 = n.clone();
        assert_eq!(n.id, n2.id);
        let dbg = format!("{n:?}");
        assert!(dbg.contains("Conv"));
    }

    // ── ExecutionGraph ──────────────────────────────────────────────────

    #[test]
    fn graph_empty() {
        let g = ExecutionGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn graph_default() {
        let g = ExecutionGraph::default();
        assert!(g.is_empty());
    }

    #[test]
    fn graph_add_and_query() {
        let g = linear_graph(3);
        assert_eq!(g.len(), 3);
        assert!(!g.is_empty());
        assert!(g.node(0).is_some());
        assert!(g.node(5).is_none());
    }

    #[test]
    fn graph_successors() {
        let g = diamond_graph();
        let s = g.successors(0);
        assert!(s.contains(&1));
        assert!(s.contains(&2));
        assert!(g.successors(3).is_empty());
    }

    #[test]
    fn graph_roots_and_leaves() {
        let g = diamond_graph();
        assert_eq!(g.roots(), vec![0]);
        assert_eq!(g.leaves(), vec![3]);
    }

    #[test]
    fn graph_linear_roots_leaves() {
        let g = linear_graph(4);
        assert_eq!(g.roots(), vec![0]);
        assert_eq!(g.leaves(), vec![3]);
    }

    #[test]
    fn graph_topo_sort_linear() {
        let g = linear_graph(5);
        let order = g.topological_sort().unwrap();
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn graph_topo_sort_diamond() {
        let g = diamond_graph();
        let order = g.topological_sort().unwrap();
        assert_eq!(order[0], 0);
        assert_eq!(*order.last().unwrap(), 3);
        let pos = |id: NodeId| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(3));
    }

    #[test]
    fn graph_total_output_bytes() {
        let g = linear_graph(3);
        assert_eq!(g.total_output_bytes(), 3 * 1024);
    }

    #[test]
    fn graph_total_flops() {
        let g = linear_graph(3);
        assert_eq!(g.total_flops(), 3 * 1_000_000);
    }

    #[test]
    fn graph_single_node() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::Activation));
        assert_eq!(g.roots(), vec![0]);
        assert_eq!(g.leaves(), vec![0]);
        assert_eq!(g.topological_sort().unwrap(), vec![0]);
    }

    #[test]
    fn graph_parallel_independent() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::MatMul));
        g.add_node(simple_node(1, OpKind::MatMul));
        g.add_node(simple_node(2, OpKind::MatMul));
        assert_eq!(g.roots().len(), 3);
        assert_eq!(g.leaves().len(), 3);
        assert_eq!(g.topological_sort().unwrap().len(), 3);
    }

    // ── MemoryPlanner ───────────────────────────────────────────────────

    #[test]
    fn memory_planner_linear() {
        let cfg = PlanConfig::default();
        let g = linear_graph(3);
        let order = g.topological_sort().unwrap();
        let mut mp = MemoryPlanner::new(&cfg);
        mp.plan(&g, &order);
        assert_eq!(mp.allocations().len(), 3);
        assert!(mp.peak_usage() > 0);
    }

    #[test]
    fn memory_planner_reuse() {
        let mut cfg = PlanConfig::default();
        cfg.enable_memory_reuse = true;
        cfg.memory_alignment = 256;
        let g = linear_graph(3);
        let order = g.topological_sort().unwrap();
        let mut mp = MemoryPlanner::new(&cfg);
        mp.plan(&g, &order);
        let peak = mp.peak_usage();
        let aligned = cfg.align(1024);
        assert!(peak <= aligned * 2, "peak {peak} should be <= {}", aligned * 2);
    }

    #[test]
    fn memory_planner_no_reuse() {
        let mut cfg = PlanConfig::default();
        cfg.enable_memory_reuse = false;
        let g = linear_graph(3);
        let order = g.topological_sort().unwrap();
        let mut mp = MemoryPlanner::new(&cfg);
        mp.plan(&g, &order);
        assert!(mp.peak_usage() >= cfg.align(1024) * 2);
    }

    #[test]
    fn memory_planner_diamond() {
        let cfg = PlanConfig::default();
        let g = diamond_graph();
        let order = g.topological_sort().unwrap();
        let mut mp = MemoryPlanner::new(&cfg);
        mp.plan(&g, &order);
        assert_eq!(mp.allocations().len(), 4);
    }

    #[test]
    fn memory_planner_empty_graph() {
        let cfg = PlanConfig::default();
        let g = ExecutionGraph::new();
        let mut mp = MemoryPlanner::new(&cfg);
        mp.plan(&g, &[]);
        assert_eq!(mp.peak_usage(), 0);
        assert!(mp.allocations().is_empty());
    }

    #[test]
    fn memory_planner_zero_byte_node() {
        let cfg = PlanConfig::default();
        let mut g = ExecutionGraph::new();
        g.add_node(ExecutionNode::new(0, "noop", OpKind::Custom));
        let order = g.topological_sort().unwrap();
        let mut mp = MemoryPlanner::new(&cfg);
        mp.plan(&g, &order);
        assert_eq!(mp.peak_usage(), 0);
        assert_eq!(mp.allocations().len(), 1);
        assert_eq!(mp.allocations()[0].size, 0);
    }

    #[test]
    fn memory_allocation_reused_flag() {
        let mut cfg = PlanConfig::default();
        cfg.enable_memory_reuse = true;
        let g = linear_graph(3);
        let order = g.topological_sort().unwrap();
        let mut mp = MemoryPlanner::new(&cfg);
        mp.plan(&g, &order);
        assert!(!mp.allocations()[0].reused);
    }

    #[test]
    fn memory_planner_alignment() {
        let mut cfg = PlanConfig::default();
        cfg.memory_alignment = 512;
        let g = linear_graph(1);
        let order = g.topological_sort().unwrap();
        let mut mp = MemoryPlanner::new(&cfg);
        mp.plan(&g, &order);
        assert_eq!(mp.allocations()[0].size, 1024);
    }

    // ── StreamScheduler ─────────────────────────────────────────────────

    #[test]
    fn stream_scheduler_linear() {
        let g = linear_graph(4);
        let order = g.topological_sort().unwrap();
        let mut ss = StreamScheduler::new(2);
        ss.schedule(&g, &order);
        assert_eq!(ss.assignments().len(), 4);
        assert_eq!(ss.streams_used(), 1);
    }

    #[test]
    fn stream_scheduler_parallel() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::Embedding));
        g.add_node(simple_node(1, OpKind::MatMul).with_dep(0));
        g.add_node(simple_node(2, OpKind::MatMul).with_dep(0));
        let order = g.topological_sort().unwrap();
        let mut ss = StreamScheduler::new(4);
        ss.schedule(&g, &order);
        assert_eq!(ss.assignments().len(), 3);
        assert!(ss.streams_used() >= 1);
    }

    #[test]
    fn stream_scheduler_single_stream() {
        let g = diamond_graph();
        let order = g.topological_sort().unwrap();
        let mut ss = StreamScheduler::new(1);
        ss.schedule(&g, &order);
        assert_eq!(ss.streams_used(), 1);
    }

    #[test]
    fn stream_scheduler_empty() {
        let g = ExecutionGraph::new();
        let mut ss = StreamScheduler::new(4);
        ss.schedule(&g, &[]);
        assert!(ss.assignments().is_empty());
        assert_eq!(ss.streams_used(), 0);
    }

    #[test]
    fn stream_scheduler_min_one_stream() {
        let ss = StreamScheduler::new(0);
        assert_eq!(ss.max_streams, 1);
    }

    #[test]
    fn stream_assignment_fields() {
        let a = StreamAssignment { node_id: 5, stream_id: 2, step: 3 };
        assert_eq!(a.node_id, 5);
        assert_eq!(a.stream_id, 2);
        assert_eq!(a.step, 3);
    }

    // ── LaunchPlanner ───────────────────────────────────────────────────

    #[test]
    fn launch_planner_cpu_default() {
        let lp = LaunchPlanner::cpu_default();
        assert_eq!(lp.max_threads_per_block, 256);
        assert_eq!(lp.warp_size, 32);
    }

    #[test]
    fn launch_planner_basic() {
        let g = linear_graph(2);
        let mut lp = LaunchPlanner::cpu_default();
        lp.plan(&g);
        assert_eq!(lp.configs().len(), 2);
        for cfg in lp.configs() {
            assert!(cfg.threads_per_block() > 0);
            assert!(cfg.total_blocks() > 0);
        }
    }

    #[test]
    fn launch_config_threads_per_block() {
        let cfg =
            LaunchConfig { node_id: 0, grid: [4, 1, 1], block: [128, 1, 1], shared_mem_bytes: 0 };
        assert_eq!(cfg.threads_per_block(), 128);
        assert_eq!(cfg.total_blocks(), 4);
        assert_eq!(cfg.total_threads(), 512);
    }

    #[test]
    fn launch_config_3d() {
        let cfg =
            LaunchConfig { node_id: 0, grid: [2, 3, 4], block: [8, 8, 4], shared_mem_bytes: 1024 };
        assert_eq!(cfg.total_blocks(), 24);
        assert_eq!(cfg.threads_per_block(), 256);
        assert_eq!(cfg.total_threads(), 24 * 256);
    }

    #[test]
    fn launch_planner_matmul_has_shared_mem() {
        let mut g = ExecutionGraph::new();
        g.add_node(
            ExecutionNode::new(0, "mm", OpKind::MatMul)
                .with_output_bytes(4096)
                .with_output_shape(vec![64, 64]),
        );
        let mut lp = LaunchPlanner::cpu_default();
        lp.plan(&g);
        assert!(lp.configs()[0].shared_mem_bytes > 0);
    }

    #[test]
    fn launch_planner_elementwise_no_shared_mem() {
        let mut g = ExecutionGraph::new();
        g.add_node(
            ExecutionNode::new(0, "ew", OpKind::Elementwise)
                .with_output_bytes(4096)
                .with_output_shape(vec![1024]),
        );
        let mut lp = LaunchPlanner::cpu_default();
        lp.plan(&g);
        assert_eq!(lp.configs()[0].shared_mem_bytes, 0);
    }

    #[test]
    fn launch_planner_empty_shape() {
        let mut g = ExecutionGraph::new();
        g.add_node(ExecutionNode::new(0, "x", OpKind::Custom));
        let mut lp = LaunchPlanner::cpu_default();
        lp.plan(&g);
        assert!(lp.configs()[0].total_threads() >= 1);
    }

    #[test]
    fn launch_planner_large_shape() {
        let mut g = ExecutionGraph::new();
        g.add_node(
            ExecutionNode::new(0, "big", OpKind::Elementwise).with_output_shape(vec![1_000_000]),
        );
        let mut lp = LaunchPlanner::cpu_default();
        lp.plan(&g);
        assert!(lp.configs()[0].total_threads() >= 1_000_000);
    }

    #[test]
    fn launch_planner_min_warp() {
        let lp = LaunchPlanner::new(256, 0, 0);
        assert_eq!(lp.warp_size, 1);
    }

    // ── PipelineStage ───────────────────────────────────────────────────

    #[test]
    fn pipeline_stage_new() {
        let s = PipelineStage::new(0);
        assert_eq!(s.stage_id, 0);
        assert!(s.is_empty());
        assert_eq!(s.node_count(), 0);
    }

    #[test]
    fn pipeline_stage_add_node() {
        let mut s = PipelineStage::new(1);
        let n = simple_node(0, OpKind::MatMul);
        let cost = CostEstimate { time_us: 100, bandwidth_bytes: 0, flops: 0 };
        s.add_node(&n, &cost);
        assert_eq!(s.node_count(), 1);
        assert!(!s.is_empty());
        assert_eq!(s.estimated_time_us, 100);
    }

    #[test]
    fn pipeline_partitioner_basic() {
        let g = linear_graph(8);
        let order = g.topological_sort().unwrap();
        let cost = CostModel::default();
        let mut pp = PipelinePartitioner::new(4);
        pp.partition(&g, &order, &cost);
        assert!(!pp.stages().is_empty());
        assert!(pp.stages().len() <= 4);
        let total: usize = pp.stages().iter().map(|s| s.node_count()).sum();
        assert_eq!(total, 8);
    }

    #[test]
    fn pipeline_partitioner_single_stage() {
        let g = linear_graph(3);
        let order = g.topological_sort().unwrap();
        let cost = CostModel::default();
        let mut pp = PipelinePartitioner::new(1);
        pp.partition(&g, &order, &cost);
        assert_eq!(pp.stages().len(), 1);
        assert_eq!(pp.stages()[0].node_count(), 3);
    }

    #[test]
    fn pipeline_partitioner_bottleneck() {
        let g = linear_graph(4);
        let order = g.topological_sort().unwrap();
        let cost = CostModel::default();
        let mut pp = PipelinePartitioner::new(2);
        pp.partition(&g, &order, &cost);
        assert!(pp.bottleneck().unwrap().estimated_time_us > 0);
    }

    #[test]
    fn pipeline_partitioner_empty() {
        let g = ExecutionGraph::new();
        let cost = CostModel::default();
        let mut pp = PipelinePartitioner::new(4);
        pp.partition(&g, &[], &cost);
        assert!(pp.stages().is_empty());
    }

    // ── CostModel ───────────────────────────────────────────────────────

    #[test]
    fn cost_model_default() {
        let cm = CostModel::default();
        assert!((cm.peak_gflops - 100.0).abs() < f64::EPSILON);
        assert!((cm.peak_bandwidth_gbs - 50.0).abs() < f64::EPSILON);
        assert_eq!(cm.launch_overhead_us, 5);
    }

    #[test]
    fn cost_model_estimate_basic() {
        let cm = CostModel::default();
        let n = simple_node(0, OpKind::MatMul);
        let est = cm.estimate(&n);
        assert!(est.time_us >= cm.launch_overhead_us);
        assert_eq!(est.flops, 1_000_000);
        assert_eq!(est.bandwidth_bytes, 1024);
    }

    #[test]
    fn cost_model_estimate_zero_flops() {
        let cm = CostModel::default();
        let n = ExecutionNode::new(0, "noop", OpKind::Custom);
        let est = cm.estimate(&n);
        assert_eq!(est.time_us, cm.launch_overhead_us);
    }

    #[test]
    fn cost_model_total_cost() {
        let cm = CostModel::default();
        let nodes = vec![simple_node(0, OpKind::MatMul), simple_node(1, OpKind::Conv)];
        let total = cm.total_cost(nodes.iter());
        assert_eq!(total.flops, 2_000_000);
    }

    #[test]
    fn cost_model_high_throughput() {
        let cm = CostModel::new(10000.0, 1000.0, 1);
        let n = simple_node(0, OpKind::MatMul);
        let est = cm.estimate(&n);
        assert!(est.time_us <= 10);
    }

    #[test]
    fn cost_model_clamped() {
        let cm = CostModel::new(-1.0, -1.0, 0);
        assert!(cm.peak_gflops > 0.0);
        assert!(cm.peak_bandwidth_gbs > 0.0);
    }

    #[test]
    fn cost_estimate_eq() {
        let a = CostEstimate { time_us: 10, bandwidth_bytes: 100, flops: 1000 };
        let b = CostEstimate { time_us: 10, bandwidth_bytes: 100, flops: 1000 };
        assert_eq!(a, b);
    }

    // ── PlanOptimizer ───────────────────────────────────────────────────

    #[test]
    fn optimizer_basic() {
        let g = linear_graph(4);
        let opt = PlanOptimizer::new(PlanConfig::default());
        let result = opt.optimize(&g);
        assert_eq!(result.exec_order.len(), 4);
        assert!(result.estimated_speedup >= 1.0);
    }

    #[test]
    fn optimizer_fusion_elementwise_activation() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::Elementwise).with_output_bytes(1024));
        g.add_node(simple_node(1, OpKind::Activation).with_dep(0).with_output_bytes(1024));
        let opt = PlanOptimizer::new(PlanConfig { enable_fusion: true, ..PlanConfig::default() });
        let result = opt.optimize(&g);
        assert_eq!(result.fusions.len(), 1);
        assert!(result.fusions[0].fused_label.contains("fused"));
    }

    #[test]
    fn optimizer_no_fusion_when_disabled() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::Elementwise));
        g.add_node(simple_node(1, OpKind::Activation).with_dep(0));
        let opt = PlanOptimizer::new(PlanConfig { enable_fusion: false, ..PlanConfig::default() });
        let result = opt.optimize(&g);
        assert!(result.fusions.is_empty());
    }

    #[test]
    fn optimizer_no_fusion_opt_none() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::Elementwise));
        g.add_node(simple_node(1, OpKind::Activation).with_dep(0));
        let opt = PlanOptimizer::new(PlanConfig {
            optimization_level: OptimizationLevel::None,
            ..PlanConfig::default()
        });
        let result = opt.optimize(&g);
        assert!(result.fusions.is_empty());
    }

    #[test]
    fn optimizer_dead_code_elimination() {
        let mut g = ExecutionGraph::new();
        g.add_node(ExecutionNode::new(0, "dead", OpKind::Custom));
        g.add_node(simple_node(1, OpKind::MatMul));
        let opt = PlanOptimizer::new(PlanConfig::default());
        let result = opt.optimize(&g);
        assert!(result.eliminated.contains(&0));
        assert!(!result.exec_order.contains(&0));
    }

    #[test]
    fn optimizer_speedup_with_fusion() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::MatMul));
        g.add_node(simple_node(1, OpKind::Activation).with_dep(0));
        let opt = PlanOptimizer::new(PlanConfig::default());
        let result = opt.optimize(&g);
        assert!(result.estimated_speedup >= 1.0);
    }

    #[test]
    fn optimizer_preserves_order() {
        let g = linear_graph(5);
        let opt = PlanOptimizer::new(PlanConfig::default());
        let result = opt.optimize(&g);
        for w in result.exec_order.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn optimizer_fusion_matmul_activation() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::MatMul));
        g.add_node(simple_node(1, OpKind::Activation).with_dep(0));
        let opt = PlanOptimizer::new(PlanConfig::default());
        assert_eq!(opt.optimize(&g).fusions.len(), 1);
    }

    #[test]
    fn optimizer_no_fusion_non_adjacent_ops() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::MatMul));
        g.add_node(simple_node(1, OpKind::Reduce).with_dep(0));
        let opt = PlanOptimizer::new(PlanConfig::default());
        assert!(opt.optimize(&g).fusions.is_empty());
    }

    #[test]
    fn optimizer_layernorm_activation_fuses() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::LayerNorm));
        g.add_node(simple_node(1, OpKind::Activation).with_dep(0));
        let opt = PlanOptimizer::new(PlanConfig::default());
        assert_eq!(opt.optimize(&g).fusions.len(), 1);
    }

    // ── ExecutionPlannerEngine ──────────────────────────────────────────

    #[test]
    fn engine_cpu_default() {
        let engine = ExecutionPlannerEngine::cpu_default();
        assert_eq!(engine.config().optimization_level, OptimizationLevel::Standard);
    }

    #[test]
    fn engine_plan_empty_graph() {
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&ExecutionGraph::new()).unwrap();
        assert!(plan.exec_order.is_empty());
        assert_eq!(plan.peak_memory_bytes, 0);
        assert_eq!(plan.estimated_time_us, 0);
    }

    #[test]
    fn engine_plan_linear() {
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&linear_graph(5)).unwrap();
        assert_eq!(plan.exec_order.len(), 5);
        assert!(!plan.allocations.is_empty());
        assert!(!plan.stream_assignments.is_empty());
        assert!(!plan.launch_configs.is_empty());
        assert!(!plan.pipeline_stages.is_empty());
        assert!(plan.peak_memory_bytes > 0);
        assert!(plan.estimated_time_us > 0);
    }

    #[test]
    fn engine_plan_diamond() {
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&diamond_graph()).unwrap();
        assert_eq!(plan.exec_order.len(), 4);
    }

    #[test]
    fn engine_memory_budget_exceeded() {
        let engine = ExecutionPlannerEngine::new(PlanConfig::new(100, 1), CostModel::default());
        let result = engine.plan(&linear_graph(3));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("peak memory"));
    }

    #[test]
    fn engine_plan_has_optimization() {
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&linear_graph(3)).unwrap();
        assert!(plan.optimization.estimated_speedup >= 1.0);
    }

    #[test]
    fn engine_cost_model_accessor() {
        let engine = ExecutionPlannerEngine::cpu_default();
        assert!((engine.cost_model().peak_gflops - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn engine_config_accessor() {
        let engine = ExecutionPlannerEngine::cpu_default();
        assert!(engine.config().enable_fusion);
    }

    #[test]
    fn engine_plan_with_fusion() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::Elementwise));
        g.add_node(simple_node(1, OpKind::Activation).with_dep(0));
        g.add_node(simple_node(2, OpKind::Reduce).with_dep(1));
        let engine = ExecutionPlannerEngine::cpu_default();
        assert!(!engine.plan(&g).unwrap().optimization.fusions.is_empty());
    }

    #[test]
    fn engine_plan_many_nodes() {
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&linear_graph(50)).unwrap();
        assert_eq!(plan.exec_order.len(), 50);
        assert!(!plan.pipeline_stages.is_empty());
    }

    #[test]
    fn engine_plan_wide_graph() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::Embedding));
        for i in 1..=10 {
            g.add_node(simple_node(i, OpKind::MatMul).with_dep(0));
        }
        g.add_node({
            let mut n = simple_node(11, OpKind::Reduce);
            for i in 1..=10 {
                n = n.with_dep(i);
            }
            n
        });
        let engine = ExecutionPlannerEngine::cpu_default();
        assert_eq!(engine.plan(&g).unwrap().exec_order.len(), 12);
    }

    // ── Cross-component integration ─────────────────────────────────────

    #[test]
    fn integration_full_pipeline() {
        let mut g = ExecutionGraph::new();
        g.add_node(
            ExecutionNode::new(0, "embed", OpKind::Embedding)
                .with_output_bytes(4096)
                .with_flops(100_000)
                .with_output_shape(vec![32, 128]),
        );
        g.add_node(
            ExecutionNode::new(1, "attn", OpKind::Attention)
                .with_dep(0)
                .with_output_bytes(8192)
                .with_flops(10_000_000)
                .with_workspace_bytes(2048)
                .with_output_shape(vec![32, 128]),
        );
        g.add_node(
            ExecutionNode::new(2, "ln", OpKind::LayerNorm)
                .with_dep(1)
                .with_output_bytes(4096)
                .with_flops(500_000)
                .with_output_shape(vec![32, 128]),
        );
        g.add_node(
            ExecutionNode::new(3, "act", OpKind::Activation)
                .with_dep(2)
                .with_output_bytes(4096)
                .with_flops(200_000)
                .with_output_shape(vec![32, 128]),
        );
        g.add_node(
            ExecutionNode::new(4, "proj", OpKind::MatMul)
                .with_dep(3)
                .with_output_bytes(8192)
                .with_flops(20_000_000)
                .with_output_shape(vec![32, 256]),
        );

        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&g).unwrap();

        assert_eq!(plan.exec_order.len(), 5);
        assert_eq!(plan.allocations.len(), 5);
        assert_eq!(plan.stream_assignments.len(), 5);
        assert_eq!(plan.launch_configs.len(), 5);
        assert!(!plan.pipeline_stages.is_empty());
        assert!(plan.peak_memory_bytes > 0);
        assert!(plan.estimated_time_us > 0);
        assert!(!plan.optimization.fusions.is_empty());
    }

    #[test]
    fn integration_memory_fits_budget() {
        let engine =
            ExecutionPlannerEngine::new(PlanConfig::new(1024 * 1024, 2), CostModel::default());
        let plan = engine.plan(&linear_graph(10)).unwrap();
        assert!(plan.peak_memory_bytes <= 1024 * 1024);
    }

    #[test]
    fn integration_stream_assignments_cover_all_nodes() {
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&diamond_graph()).unwrap();
        let assigned: HashSet<NodeId> = plan.stream_assignments.iter().map(|a| a.node_id).collect();
        for &nid in &plan.exec_order {
            assert!(assigned.contains(&nid));
        }
    }

    #[test]
    fn integration_launch_configs_cover_all_nodes() {
        let g = linear_graph(5);
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&g).unwrap();
        let launched: HashSet<NodeId> = plan.launch_configs.iter().map(|c| c.node_id).collect();
        for n in g.nodes() {
            assert!(launched.contains(&n.id));
        }
    }

    #[test]
    fn integration_pipeline_covers_all_nodes() {
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&linear_graph(8)).unwrap();
        let piped: usize = plan.pipeline_stages.iter().map(|s| s.node_count()).sum();
        assert_eq!(piped, plan.exec_order.len());
    }

    #[test]
    fn integration_optimizer_no_duplicate_nodes() {
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&diamond_graph()).unwrap();
        let mut seen = HashSet::new();
        for &nid in &plan.exec_order {
            assert!(seen.insert(nid));
        }
    }

    #[test]
    fn integration_topo_order_respected() {
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&diamond_graph()).unwrap();
        let pos = |id: NodeId| plan.exec_order.iter().position(|&x| x == id).unwrap();
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(3));
        assert_eq!(pos(0), 0);
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn single_node_plan() {
        let mut g = ExecutionGraph::new();
        g.add_node(simple_node(0, OpKind::MatMul).with_output_shape(vec![1024]));
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&g).unwrap();
        assert_eq!(plan.exec_order.len(), 1);
        assert_eq!(plan.allocations.len(), 1);
    }

    #[test]
    fn many_independent_nodes() {
        let mut g = ExecutionGraph::new();
        for i in 0..20 {
            g.add_node(simple_node(i, OpKind::Elementwise));
        }
        let engine = ExecutionPlannerEngine::cpu_default();
        assert_eq!(engine.plan(&g).unwrap().exec_order.len(), 20);
    }

    #[test]
    fn deep_chain() {
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&linear_graph(100)).unwrap();
        assert_eq!(plan.exec_order.len(), 100);
        let streams: HashSet<u32> = plan.stream_assignments.iter().map(|a| a.stream_id).collect();
        assert_eq!(streams.len(), 1);
    }

    #[test]
    fn fusion_group_debug() {
        let fg = FusionGroup { node_ids: vec![0, 1], fused_label: "fused(a+b)".to_string() };
        assert!(format!("{fg:?}").contains("fused(a+b)"));
    }

    #[test]
    fn optimization_result_debug() {
        let r = OptimizationResult {
            exec_order: vec![0],
            fusions: Vec::new(),
            eliminated: Vec::new(),
            estimated_speedup: 1.5,
        };
        assert!(format!("{r:?}").contains("1.5"));
    }

    #[test]
    fn execution_plan_debug() {
        let engine = ExecutionPlannerEngine::cpu_default();
        let plan = engine.plan(&linear_graph(2)).unwrap();
        assert!(format!("{plan:?}").contains("exec_order"));
    }

    #[test]
    fn plan_config_clone() {
        let cfg = PlanConfig::default();
        assert_eq!(cfg.max_memory_bytes, cfg.clone().max_memory_bytes);
    }

    #[test]
    fn cost_model_clone() {
        let cm = CostModel::default();
        assert!((cm.peak_gflops - cm.clone().peak_gflops).abs() < f64::EPSILON);
    }

    #[test]
    fn stream_scheduler_clone() {
        let mut ss = StreamScheduler::new(4);
        let g = linear_graph(2);
        ss.schedule(&g, &g.topological_sort().unwrap());
        assert_eq!(ss.assignments().len(), ss.clone().assignments().len());
    }

    #[test]
    fn memory_planner_clone() {
        let cfg = PlanConfig::default();
        let mp = MemoryPlanner::new(&cfg);
        assert_eq!(mp.peak_usage(), mp.clone().peak_usage());
    }

    #[test]
    fn engine_clone() {
        let e = ExecutionPlannerEngine::cpu_default();
        assert_eq!(e.config().max_parallelism, e.clone().config().max_parallelism);
    }

    #[test]
    fn pipeline_partitioner_clone() {
        let pp = PipelinePartitioner::new(4);
        assert_eq!(pp.stages().len(), pp.clone().stages().len());
    }

    #[test]
    fn op_kind_all_variants_display() {
        let ops = [
            OpKind::MatMul,
            OpKind::Conv,
            OpKind::Attention,
            OpKind::LayerNorm,
            OpKind::Activation,
            OpKind::Elementwise,
            OpKind::Reduce,
            OpKind::Transpose,
            OpKind::Gather,
            OpKind::Scatter,
            OpKind::Softmax,
            OpKind::Embedding,
            OpKind::Custom,
        ];
        for op in ops {
            assert!(!format!("{op}").is_empty());
        }
    }

    #[test]
    fn optimization_level_variants() {
        for l in [
            OptimizationLevel::None,
            OptimizationLevel::Basic,
            OptimizationLevel::Standard,
            OptimizationLevel::Aggressive,
        ] {
            let _ = format!("{l:?}");
        }
    }

    #[test]
    fn memory_allocation_eq() {
        let a = MemoryAllocation { node_id: 0, offset: 0, size: 256, reused: false };
        assert_eq!(a, a.clone());
    }

    #[test]
    fn launch_config_eq() {
        let a =
            LaunchConfig { node_id: 0, grid: [1, 1, 1], block: [32, 1, 1], shared_mem_bytes: 0 };
        assert_eq!(a, a.clone());
    }

    #[test]
    fn stream_assignment_eq() {
        let a = StreamAssignment { node_id: 0, stream_id: 0, step: 0 };
        assert_eq!(a, a.clone());
    }

    #[test]
    fn graph_nodes_accessor() {
        let g = linear_graph(3);
        assert_eq!(g.nodes().len(), 3);
        assert_eq!(g.nodes()[0].id, 0);
    }

    #[test]
    fn launch_planner_softmax_shared_mem() {
        let mut g = ExecutionGraph::new();
        g.add_node(ExecutionNode::new(0, "sm", OpKind::Softmax).with_output_shape(vec![256]));
        let mut lp = LaunchPlanner::cpu_default();
        lp.plan(&g);
        assert!(lp.configs()[0].shared_mem_bytes > 0);
    }

    #[test]
    fn cost_model_bandwidth_dominated() {
        let cm = CostModel::new(100_000.0, 1.0, 0);
        let n = ExecutionNode::new(0, "big", OpKind::MatMul)
            .with_output_bytes(100_000_000)
            .with_flops(100);
        assert!(cm.estimate(&n).bandwidth_bytes > 0);
    }

    #[test]
    fn pipeline_stage_accumulates() {
        let mut s = PipelineStage::new(0);
        let c1 = CostEstimate { time_us: 100, bandwidth_bytes: 0, flops: 0 };
        let c2 = CostEstimate { time_us: 200, bandwidth_bytes: 0, flops: 0 };
        s.add_node(&simple_node(0, OpKind::MatMul), &c1);
        s.add_node(&simple_node(1, OpKind::Conv), &c2);
        assert_eq!(s.estimated_time_us, 300);
        assert_eq!(s.node_count(), 2);
    }
}
