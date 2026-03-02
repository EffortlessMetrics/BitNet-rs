//! CUDA graph capture and replay for reduced kernel launch overhead.
//!
//! # Overview
//!
//! CUDA graphs allow recording a sequence of GPU operations once and replaying
//! them with minimal CPU-side overhead. This module provides:
//!
//! - [`CudaGraph`] — captured graph of GPU operations
//! - [`GraphBuilder`] — builder for constructing graphs node-by-node
//! - [`GraphNode`] / [`GraphEdge`] — topology primitives
//! - [`capture`] / [`execute`] — capture and replay entry points
//! - [`update_params`] — update kernel parameters without recapture
//! - [`graph_from_model_layer`] — capture a full transformer layer as a graph
//! - [`GraphPool`] — pool of pre-captured graphs keyed by sequence length
//! - [`conditional_graph`] — graph with conditional execution paths
//! - [`multi_stream_graph`] — graph spanning multiple streams
//! - [`GraphOptimizer`] — topology optimizations (merge, redundant-sync removal)
//! - [`GraphProfiler`] — execution profiling and statistics
//!
//! All GPU dispatch is feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations simulate graph execution for testing on
//! non-GPU hosts.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bitnet_common::{KernelError, Result};

// ── Identifiers ──────────────────────────────────────────────────────

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);

impl NodeId {
    fn next() -> Self {
        Self(NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node-{}", self.0)
    }
}

/// Unique identifier for a captured graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphId(u64);

impl GraphId {
    fn next() -> Self {
        Self(NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for GraphId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "graph-{}", self.0)
    }
}

// ── Graph node ───────────────────────────────────────────────────────

/// The kind of operation represented by a graph node.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    /// GPU kernel launch with name, grid dims, block dims.
    Kernel { name: String, grid: [u32; 3], block: [u32; 3] },
    /// Device-to-device memory copy (bytes).
    MemCopy { bytes: usize },
    /// Memory set / fill (bytes).
    MemSet { bytes: usize },
    /// Host-side callback.
    HostCallback { label: String },
    /// Synchronization barrier.
    Barrier,
    /// Empty / no-op node used as graph entry or exit sentinel.
    Empty,
}

/// A node in the execution graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique node identifier.
    pub id: NodeId,
    /// Operation kind.
    pub kind: NodeKind,
    /// Arbitrary parameters stored as key-value pairs.
    pub params: HashMap<String, f64>,
    /// Stream index this node is assigned to (0 = default stream).
    pub stream: u32,
    /// Whether this node is enabled for conditional execution.
    pub enabled: bool,
}

impl GraphNode {
    /// Create a new kernel node.
    pub fn kernel(name: &str, grid: [u32; 3], block: [u32; 3]) -> Self {
        Self {
            id: NodeId::next(),
            kind: NodeKind::Kernel { name: name.to_string(), grid, block },
            params: HashMap::new(),
            stream: 0,
            enabled: true,
        }
    }

    /// Create a memcopy node.
    pub fn memcopy(bytes: usize) -> Self {
        Self {
            id: NodeId::next(),
            kind: NodeKind::MemCopy { bytes },
            params: HashMap::new(),
            stream: 0,
            enabled: true,
        }
    }

    /// Create a memset node.
    pub fn memset(bytes: usize) -> Self {
        Self {
            id: NodeId::next(),
            kind: NodeKind::MemSet { bytes },
            params: HashMap::new(),
            stream: 0,
            enabled: true,
        }
    }

    /// Create a barrier node.
    pub fn barrier() -> Self {
        Self {
            id: NodeId::next(),
            kind: NodeKind::Barrier,
            params: HashMap::new(),
            stream: 0,
            enabled: true,
        }
    }

    /// Create an empty sentinel node.
    pub fn empty() -> Self {
        Self {
            id: NodeId::next(),
            kind: NodeKind::Empty,
            params: HashMap::new(),
            stream: 0,
            enabled: true,
        }
    }

    /// Create a host callback node.
    pub fn host_callback(label: &str) -> Self {
        Self {
            id: NodeId::next(),
            kind: NodeKind::HostCallback { label: label.to_string() },
            params: HashMap::new(),
            stream: 0,
            enabled: true,
        }
    }

    /// Set a parameter value.
    pub fn with_param(mut self, key: &str, value: f64) -> Self {
        self.params.insert(key.to_string(), value);
        self
    }

    /// Assign this node to a specific stream.
    pub fn on_stream(mut self, stream: u32) -> Self {
        self.stream = stream;
        self
    }

    /// Estimated execution cost (heuristic, microseconds).
    pub fn estimated_cost_us(&self) -> f64 {
        match &self.kind {
            NodeKind::Kernel { grid, block, .. } => {
                let threads = grid[0] as f64
                    * grid[1] as f64
                    * grid[2] as f64
                    * block[0] as f64
                    * block[1] as f64
                    * block[2] as f64;
                // ~1 µs base + proportional to total threads
                1.0 + threads * 0.001
            }
            NodeKind::MemCopy { bytes } | NodeKind::MemSet { bytes } => {
                0.5 + *bytes as f64 / (400.0 * 1e9 / 1e6) // ~400 GB/s bandwidth
            }
            NodeKind::HostCallback { .. } => 5.0,
            NodeKind::Barrier => 0.1,
            NodeKind::Empty => 0.0,
        }
    }
}

// ── Graph edge ───────────────────────────────────────────────────────

/// A dependency edge between two graph nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphEdge {
    /// Source node (must complete before `to` begins).
    pub from: NodeId,
    /// Target node.
    pub to: NodeId,
}

impl GraphEdge {
    /// Create a new edge from `from` to `to`.
    pub fn new(from: NodeId, to: NodeId) -> Self {
        Self { from, to }
    }
}

impl fmt::Display for GraphEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.from, self.to)
    }
}

// ── Capture state ────────────────────────────────────────────────────

/// State of graph capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureState {
    /// Not capturing.
    Idle,
    /// Actively recording operations.
    Capturing,
    /// Capture completed, graph is ready.
    Complete,
}

// ── CudaGraph ────────────────────────────────────────────────────────

/// A captured CUDA graph of GPU operations.
///
/// On CPU-only builds the graph tracks topology and simulates execution
/// in topological order.
#[derive(Debug, Clone)]
pub struct CudaGraph {
    /// Unique graph identifier.
    pub id: GraphId,
    /// All nodes in the graph.
    nodes: Vec<GraphNode>,
    /// Dependency edges.
    edges: Vec<GraphEdge>,
    /// Number of times this graph has been executed.
    exec_count: u64,
    /// Whether the graph has been instantiated (ready for replay).
    instantiated: bool,
    /// Label for debugging.
    label: String,
}

impl CudaGraph {
    /// Create an empty graph.
    pub fn new(label: &str) -> Self {
        Self {
            id: GraphId::next(),
            nodes: Vec::new(),
            edges: Vec::new(),
            exec_count: 0,
            instantiated: false,
            label: label.to_string(),
        }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Slice of all nodes.
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Slice of all edges.
    pub fn edges(&self) -> &[GraphEdge] {
        &self.edges
    }

    /// How many times this graph has been executed.
    pub fn exec_count(&self) -> u64 {
        self.exec_count
    }

    /// Whether the graph has been instantiated.
    pub fn is_instantiated(&self) -> bool {
        self.instantiated
    }

    /// Debug label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Add a node, returning its id.
    pub fn add_node(&mut self, node: GraphNode) -> NodeId {
        let id = node.id;
        self.nodes.push(node);
        self.instantiated = false; // topology changed
        id
    }

    /// Add a dependency edge.
    pub fn add_edge(&mut self, edge: GraphEdge) -> Result<()> {
        let from_exists = self.nodes.iter().any(|n| n.id == edge.from);
        let to_exists = self.nodes.iter().any(|n| n.id == edge.to);
        if !from_exists || !to_exists {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "edge references missing node(s): from={} to={}",
                    edge.from, edge.to
                ),
            }
            .into());
        }
        self.edges.push(edge);
        self.instantiated = false;
        Ok(())
    }

    /// Instantiate the graph (prepare for replay).
    pub fn instantiate(&mut self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(KernelError::InvalidArguments {
                reason: "cannot instantiate empty graph".into(),
            }
            .into());
        }
        self.validate_acyclic()?;
        self.instantiated = true;
        Ok(())
    }

    /// Execute the captured graph (CPU fallback: simulate in topological order).
    pub fn execute(&mut self) -> Result<GraphExecResult> {
        if !self.instantiated {
            self.instantiate()?;
        }
        let start = Instant::now();
        let order = self.topological_order()?;
        let mut executed = Vec::new();
        for &idx in &order {
            let node = &self.nodes[idx];
            if node.enabled {
                executed.push(node.id);
            }
        }
        self.exec_count += 1;
        Ok(GraphExecResult {
            graph_id: self.id,
            nodes_executed: executed.len(),
            wall_time: start.elapsed(),
            estimated_gpu_time_us: order
                .iter()
                .filter(|&&i| self.nodes[i].enabled)
                .map(|&i| self.nodes[i].estimated_cost_us())
                .sum(),
        })
    }

    /// Topological sort returning indices into `self.nodes`.
    fn topological_order(&self) -> Result<Vec<usize>> {
        let n = self.nodes.len();
        let id_to_idx: HashMap<NodeId, usize> =
            self.nodes.iter().enumerate().map(|(i, nd)| (nd.id, i)).collect();

        let mut in_deg = vec![0u32; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in &self.edges {
            if let (Some(&fi), Some(&ti)) = (id_to_idx.get(&e.from), id_to_idx.get(&e.to)) {
                adj[fi].push(ti);
                in_deg[ti] += 1;
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop() {
            order.push(u);
            for &v in &adj[u] {
                in_deg[v] -= 1;
                if in_deg[v] == 0 {
                    queue.push(v);
                }
            }
        }
        if order.len() != n {
            return Err(
                KernelError::InvalidArguments { reason: "graph contains a cycle".into() }.into()
            );
        }
        Ok(order)
    }

    /// Check that the graph is acyclic.
    fn validate_acyclic(&self) -> Result<()> {
        self.topological_order().map(|_| ())
    }

    /// Find a node by id.
    pub fn find_node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Find a mutable node by id.
    pub fn find_node_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// Return all root nodes (no incoming edges).
    pub fn roots(&self) -> Vec<NodeId> {
        let targets: std::collections::HashSet<NodeId> = self.edges.iter().map(|e| e.to).collect();
        self.nodes.iter().filter(|n| !targets.contains(&n.id)).map(|n| n.id).collect()
    }

    /// Return all leaf nodes (no outgoing edges).
    pub fn leaves(&self) -> Vec<NodeId> {
        let sources: std::collections::HashSet<NodeId> =
            self.edges.iter().map(|e| e.from).collect();
        self.nodes.iter().filter(|n| !sources.contains(&n.id)).map(|n| n.id).collect()
    }

    /// Count distinct streams referenced by nodes.
    pub fn stream_count(&self) -> usize {
        let streams: std::collections::HashSet<u32> = self.nodes.iter().map(|n| n.stream).collect();
        streams.len()
    }
}

/// Result of executing a graph.
#[derive(Debug, Clone)]
pub struct GraphExecResult {
    /// Which graph was executed.
    pub graph_id: GraphId,
    /// Number of nodes that were actually executed.
    pub nodes_executed: usize,
    /// Wall-clock time for the CPU simulation.
    pub wall_time: Duration,
    /// Estimated GPU time in microseconds (heuristic sum).
    pub estimated_gpu_time_us: f64,
}

// ── GraphBuilder ─────────────────────────────────────────────────────

/// Builder for constructing [`CudaGraph`] instances.
pub struct GraphBuilder {
    label: String,
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
}

impl GraphBuilder {
    /// Create a new builder.
    pub fn new(label: &str) -> Self {
        Self { label: label.to_string(), nodes: Vec::new(), edges: Vec::new() }
    }

    /// Add a node, returning its id.
    pub fn add_node(&mut self, node: GraphNode) -> NodeId {
        let id = node.id;
        self.nodes.push(node);
        id
    }

    /// Add an edge.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) -> &mut Self {
        self.edges.push(GraphEdge::new(from, to));
        self
    }

    /// Add a linear chain of nodes (each depends on the previous).
    pub fn add_chain(&mut self, nodes: Vec<GraphNode>) -> Vec<NodeId> {
        let mut ids = Vec::with_capacity(nodes.len());
        for node in nodes {
            let id = self.add_node(node);
            if let Some(&prev) = ids.last() {
                self.add_edge(prev, id);
            }
            ids.push(id);
        }
        ids
    }

    /// Build the graph.
    pub fn build(self) -> Result<CudaGraph> {
        if self.nodes.is_empty() {
            return Err(KernelError::InvalidArguments {
                reason: "cannot build empty graph".into(),
            }
            .into());
        }
        let mut g = CudaGraph::new(&self.label);
        for node in self.nodes {
            g.nodes.push(node);
        }
        for edge in self.edges {
            let from_exists = g.nodes.iter().any(|n| n.id == edge.from);
            let to_exists = g.nodes.iter().any(|n| n.id == edge.to);
            if !from_exists || !to_exists {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "edge references missing node: from={}, to={}",
                        edge.from, edge.to
                    ),
                }
                .into());
            }
            g.edges.push(edge);
        }
        g.validate_acyclic()?;
        Ok(g)
    }
}

// ── Capture / Execute free functions ─────────────────────────────────

/// Start graph capture mode and record operations via the closure.
///
/// The closure receives a [`GraphBuilder`] and should populate it with
/// the operations to capture.  On GPU, this would wrap
/// `cudaStreamBeginCapture` / `cudaStreamEndCapture`.
pub fn capture<F>(label: &str, f: F) -> Result<CudaGraph>
where
    F: FnOnce(&mut GraphBuilder),
{
    let mut builder = GraphBuilder::new(label);
    f(&mut builder);
    builder.build()
}

/// Execute a captured graph (convenience wrapper around [`CudaGraph::execute`]).
pub fn execute(graph: &mut CudaGraph) -> Result<GraphExecResult> {
    graph.execute()
}

/// Update parameters on nodes matching `name` without recapturing.
pub fn update_params(
    graph: &mut CudaGraph,
    kernel_name: &str,
    params: &HashMap<String, f64>,
) -> Result<usize> {
    let mut updated = 0usize;
    for node in &mut graph.nodes {
        if let NodeKind::Kernel { name, .. } = &node.kind
            && name == kernel_name
        {
            for (k, v) in params {
                node.params.insert(k.clone(), *v);
            }
            updated += 1;
        }
    }
    if updated == 0 {
        return Err(KernelError::InvalidArguments {
            reason: format!("no kernel node named '{kernel_name}' found"),
        }
        .into());
    }
    graph.instantiated = false; // mark for re-instantiation
    Ok(updated)
}

// ── graph_from_model_layer ───────────────────────────────────────────

/// Configuration for capturing a transformer layer graph.
#[derive(Debug, Clone)]
pub struct LayerGraphConfig {
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Whether to include the MLP block.
    pub include_mlp: bool,
    /// Whether to include the attention block.
    pub include_attention: bool,
    /// Stream id for attention sub-graph.
    pub attention_stream: u32,
    /// Stream id for MLP sub-graph.
    pub mlp_stream: u32,
}

impl Default for LayerGraphConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 2048,
            num_heads: 16,
            seq_len: 128,
            include_mlp: true,
            include_attention: true,
            attention_stream: 0,
            mlp_stream: 0,
        }
    }
}

/// Capture a full transformer layer as a graph.
///
/// Creates the standard pre-norm transformer block:
/// `residual → RMSNorm → Attention → Add → RMSNorm → MLP → Add`
pub fn graph_from_model_layer(cfg: &LayerGraphConfig) -> Result<CudaGraph> {
    if cfg.hidden_dim == 0 || cfg.num_heads == 0 || cfg.seq_len == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "hidden_dim, num_heads, and seq_len must be non-zero".into(),
        }
        .into());
    }

    let threads_per_block = 256u32;
    let blocks = |n: usize| (n as u32).div_ceil(threads_per_block);

    let mut builder = GraphBuilder::new("transformer_layer");

    // Entry sentinel
    let entry = builder.add_node(GraphNode::empty());

    // Pre-attention RMSNorm
    let norm1 = builder.add_node(
        GraphNode::kernel("rmsnorm", [blocks(cfg.hidden_dim), 1, 1], [threads_per_block, 1, 1])
            .on_stream(cfg.attention_stream)
            .with_param("hidden_dim", cfg.hidden_dim as f64),
    );
    builder.add_edge(entry, norm1);

    let mut last_attn = norm1;

    if cfg.include_attention {
        // QKV projection
        let qkv = builder.add_node(
            GraphNode::kernel(
                "qkv_proj",
                [blocks(cfg.hidden_dim * 3), 1, 1],
                [threads_per_block, 1, 1],
            )
            .on_stream(cfg.attention_stream),
        );
        builder.add_edge(norm1, qkv);

        // Attention
        let head_dim = cfg.hidden_dim / cfg.num_heads;
        let attn = builder.add_node(
            GraphNode::kernel(
                "attention",
                [cfg.num_heads as u32, blocks(cfg.seq_len), 1],
                [threads_per_block, 1, 1],
            )
            .on_stream(cfg.attention_stream)
            .with_param("head_dim", head_dim as f64)
            .with_param("seq_len", cfg.seq_len as f64),
        );
        builder.add_edge(qkv, attn);

        // Output projection
        let out_proj = builder.add_node(
            GraphNode::kernel(
                "out_proj",
                [blocks(cfg.hidden_dim), 1, 1],
                [threads_per_block, 1, 1],
            )
            .on_stream(cfg.attention_stream),
        );
        builder.add_edge(attn, out_proj);
        last_attn = out_proj;
    }

    // Residual add
    let add1 = builder.add_node(
        GraphNode::kernel(
            "residual_add",
            [blocks(cfg.hidden_dim), 1, 1],
            [threads_per_block, 1, 1],
        )
        .on_stream(0),
    );
    builder.add_edge(last_attn, add1);

    let mut last_node = add1;

    if cfg.include_mlp {
        // Pre-MLP RMSNorm
        let norm2 = builder.add_node(
            GraphNode::kernel("rmsnorm", [blocks(cfg.hidden_dim), 1, 1], [threads_per_block, 1, 1])
                .on_stream(cfg.mlp_stream)
                .with_param("hidden_dim", cfg.hidden_dim as f64),
        );
        builder.add_edge(add1, norm2);

        // Gate + up projection
        let gate_up = builder.add_node(
            GraphNode::kernel(
                "gate_up_proj",
                [blocks(cfg.hidden_dim * 2), 1, 1],
                [threads_per_block, 1, 1],
            )
            .on_stream(cfg.mlp_stream),
        );
        builder.add_edge(norm2, gate_up);

        // SiLU activation
        let silu = builder.add_node(
            GraphNode::kernel(
                "silu_gate",
                [blocks(cfg.hidden_dim), 1, 1],
                [threads_per_block, 1, 1],
            )
            .on_stream(cfg.mlp_stream),
        );
        builder.add_edge(gate_up, silu);

        // Down projection
        let down = builder.add_node(
            GraphNode::kernel(
                "down_proj",
                [blocks(cfg.hidden_dim), 1, 1],
                [threads_per_block, 1, 1],
            )
            .on_stream(cfg.mlp_stream),
        );
        builder.add_edge(silu, down);

        // Residual add
        let add2 = builder.add_node(
            GraphNode::kernel(
                "residual_add",
                [blocks(cfg.hidden_dim), 1, 1],
                [threads_per_block, 1, 1],
            )
            .on_stream(0),
        );
        builder.add_edge(down, add2);
        last_node = add2;
    }

    // Exit sentinel
    let exit = builder.add_node(GraphNode::empty());
    builder.add_edge(last_node, exit);

    builder.build()
}

// ── GraphPool ────────────────────────────────────────────────────────

/// Pool of pre-captured graphs keyed by sequence length.
///
/// During autoregressive inference the sequence length changes each step;
/// this pool caches graphs for common lengths to avoid recapture.
pub struct GraphPool {
    graphs: HashMap<usize, CudaGraph>,
    max_entries: usize,
    base_config: LayerGraphConfig,
    hits: u64,
    misses: u64,
}

impl GraphPool {
    /// Create a new pool with a maximum number of cached entries.
    pub fn new(base_config: LayerGraphConfig, max_entries: usize) -> Result<Self> {
        if max_entries == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "max_entries must be > 0".into() }.into()
            );
        }
        Ok(Self { graphs: HashMap::new(), max_entries, base_config, hits: 0, misses: 0 })
    }

    /// Get or capture a graph for the given sequence length.
    pub fn get_or_capture(&mut self, seq_len: usize) -> Result<&mut CudaGraph> {
        if self.graphs.contains_key(&seq_len) {
            self.hits += 1;
            return Ok(self.graphs.get_mut(&seq_len).unwrap());
        }

        self.misses += 1;

        // Evict oldest if at capacity (simple strategy: remove smallest seq_len).
        if self.graphs.len() >= self.max_entries
            && let Some(&key) = self.graphs.keys().min()
        {
            self.graphs.remove(&key);
        }

        let mut cfg = self.base_config.clone();
        cfg.seq_len = seq_len;
        let graph = graph_from_model_layer(&cfg)?;
        self.graphs.insert(seq_len, graph);
        Ok(self.graphs.get_mut(&seq_len).unwrap())
    }

    /// Number of cached graphs.
    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Cache hit count.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache miss count.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hit rate as a fraction (0.0 – 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Remove all cached graphs.
    pub fn clear(&mut self) {
        self.graphs.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Whether a graph for the given sequence length is cached.
    pub fn contains(&self, seq_len: usize) -> bool {
        self.graphs.contains_key(&seq_len)
    }

    /// Execute the graph for `seq_len`, capturing first if needed.
    pub fn execute(&mut self, seq_len: usize) -> Result<GraphExecResult> {
        self.get_or_capture(seq_len)?;
        self.graphs.get_mut(&seq_len).unwrap().execute()
    }
}

// ── conditional_graph ────────────────────────────────────────────────

/// Condition function type for conditional graph execution.
pub type ConditionFn = Box<dyn Fn(&HashMap<String, f64>) -> bool + Send + Sync>;

/// Build a graph with conditional execution paths.
///
/// `condition_params` are evaluated at execution time; nodes whose `enabled`
/// flag is `false` are skipped.
pub fn conditional_graph(
    label: &str,
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    conditions: &HashMap<NodeId, bool>,
) -> Result<CudaGraph> {
    let mut builder = GraphBuilder::new(label);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for mut node in nodes {
        let old_id = node.id;
        if let Some(&enabled) = conditions.get(&old_id) {
            node.enabled = enabled;
        }
        let new_id = builder.add_node(node);
        id_map.insert(old_id, new_id);
    }

    for edge in edges {
        let from = id_map.get(&edge.from).copied().unwrap_or(edge.from);
        let to = id_map.get(&edge.to).copied().unwrap_or(edge.to);
        builder.add_edge(from, to);
    }

    builder.build()
}

// ── multi_stream_graph ───────────────────────────────────────────────

/// Configuration for a multi-stream graph.
#[derive(Debug, Clone)]
pub struct MultiStreamConfig {
    /// Number of streams to use.
    pub num_streams: u32,
    /// Whether to add inter-stream sync barriers.
    pub sync_barriers: bool,
}

impl Default for MultiStreamConfig {
    fn default() -> Self {
        Self { num_streams: 2, sync_barriers: true }
    }
}

/// Build a graph that distributes work across multiple CUDA streams.
///
/// `per_stream_nodes` maps stream index → ordered list of nodes.
/// If `config.sync_barriers` is true, a barrier is inserted between streams
/// at the end.
pub fn multi_stream_graph(
    label: &str,
    per_stream_nodes: HashMap<u32, Vec<GraphNode>>,
    config: &MultiStreamConfig,
) -> Result<CudaGraph> {
    if per_stream_nodes.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "per_stream_nodes must not be empty".into(),
        }
        .into());
    }

    let mut builder = GraphBuilder::new(label);
    let mut stream_tails: HashMap<u32, NodeId> = HashMap::new();

    for (&stream_idx, nodes) in &per_stream_nodes {
        let mut prev: Option<NodeId> = None;
        for node in nodes {
            let mut n = node.clone();
            n.stream = stream_idx;
            // Assign a fresh id to the cloned node.
            n.id = NodeId::next();
            let id = builder.add_node(n);
            if let Some(p) = prev {
                builder.add_edge(p, id);
            }
            prev = Some(id);
        }
        if let Some(tail) = prev {
            stream_tails.insert(stream_idx, tail);
        }
    }

    // Add a final sync barrier if requested.
    if config.sync_barriers && stream_tails.len() > 1 {
        let barrier = builder.add_node(GraphNode::barrier());
        for &tail in stream_tails.values() {
            builder.add_edge(tail, barrier);
        }
    }

    builder.build()
}

// ── GraphOptimizer ───────────────────────────────────────────────────

/// Statistics from a graph optimization pass.
#[derive(Debug, Clone, Default)]
pub struct OptimizeStats {
    /// Number of nodes removed.
    pub nodes_removed: usize,
    /// Number of edges removed.
    pub edges_removed: usize,
    /// Number of nodes merged.
    pub nodes_merged: usize,
}

/// Optimize graph topology.
pub struct GraphOptimizer;

impl GraphOptimizer {
    /// Remove empty / no-op nodes that have at most one predecessor and one
    /// successor, re-wiring the edge.
    pub fn remove_empty_nodes(graph: &mut CudaGraph) -> OptimizeStats {
        let mut stats = OptimizeStats::default();
        let empty_ids: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Empty))
            .map(|n| n.id)
            .collect();

        for id in empty_ids {
            let incoming: Vec<NodeId> =
                graph.edges.iter().filter(|e| e.to == id).map(|e| e.from).collect();
            let outgoing: Vec<NodeId> =
                graph.edges.iter().filter(|e| e.from == id).map(|e| e.to).collect();

            // Only remove if it's a simple pass-through (≤1 in, ≤1 out).
            if incoming.len() <= 1 && outgoing.len() <= 1 {
                graph.edges.retain(|e| e.from != id && e.to != id);
                graph.nodes.retain(|n| n.id != id);

                // Re-wire predecessor → successor.
                if let (Some(&pred), Some(&succ)) = (incoming.first(), outgoing.first()) {
                    graph.edges.push(GraphEdge::new(pred, succ));
                }

                stats.nodes_removed += 1;
                graph.instantiated = false;
            }
        }

        stats
    }

    /// Remove redundant barriers: a barrier with a single predecessor is useless.
    pub fn remove_redundant_barriers(graph: &mut CudaGraph) -> OptimizeStats {
        let mut stats = OptimizeStats::default();
        let barrier_ids: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Barrier))
            .map(|n| n.id)
            .collect();

        for id in barrier_ids {
            let incoming_count = graph.edges.iter().filter(|e| e.to == id).count();
            if incoming_count <= 1 {
                let incoming: Vec<NodeId> =
                    graph.edges.iter().filter(|e| e.to == id).map(|e| e.from).collect();
                let outgoing: Vec<NodeId> =
                    graph.edges.iter().filter(|e| e.from == id).map(|e| e.to).collect();

                graph.edges.retain(|e| e.from != id && e.to != id);
                graph.nodes.retain(|n| n.id != id);

                for &pred in &incoming {
                    for &succ in &outgoing {
                        graph.edges.push(GraphEdge::new(pred, succ));
                    }
                }

                stats.nodes_removed += 1;
                graph.instantiated = false;
            }
        }

        stats
    }

    /// Merge consecutive kernel nodes on the same stream that share the
    /// same kernel name into a single node with combined grid dims.
    pub fn merge_consecutive_kernels(graph: &mut CudaGraph) -> OptimizeStats {
        let mut stats = OptimizeStats::default();
        // Find pairs where A→B, same stream, same kernel name, A has 1 out, B has 1 in.
        loop {
            let mut merge_pair = None;
            for edge in &graph.edges {
                let a = graph.nodes.iter().find(|n| n.id == edge.from);
                let b = graph.nodes.iter().find(|n| n.id == edge.to);
                if let (Some(a), Some(b)) = (a, b)
                    && a.stream == b.stream
                    && let (
                        NodeKind::Kernel { name: na, grid: ga, .. },
                        NodeKind::Kernel { name: nb, grid: _gb, .. },
                    ) = (&a.kind, &b.kind)
                    && na == nb
                {
                    let a_out = graph.edges.iter().filter(|e| e.from == a.id).count();
                    let b_in = graph.edges.iter().filter(|e| e.to == b.id).count();
                    if a_out == 1 && b_in == 1 {
                        merge_pair = Some((a.id, b.id, ga));
                        break;
                    }
                }
            }

            if let Some((a_id, b_id, _grid_a)) = merge_pair {
                // Merge B into A: double A's grid.x, remove B, re-wire B's successors to A.
                if let Some(a_node) = graph.nodes.iter_mut().find(|n| n.id == a_id)
                    && let NodeKind::Kernel { ref mut grid, .. } = a_node.kind
                {
                    grid[0] = grid[0].saturating_mul(2);
                }

                let b_successors: Vec<NodeId> =
                    graph.edges.iter().filter(|e| e.from == b_id).map(|e| e.to).collect();

                graph.edges.retain(|e| e.from != b_id && e.to != b_id);
                graph.nodes.retain(|n| n.id != b_id);

                for succ in b_successors {
                    graph.edges.push(GraphEdge::new(a_id, succ));
                }

                stats.nodes_merged += 1;
                graph.instantiated = false;
            } else {
                break;
            }
        }

        stats
    }

    /// Run all optimization passes and return aggregate statistics.
    pub fn optimize_all(graph: &mut CudaGraph) -> OptimizeStats {
        let mut total = OptimizeStats::default();

        let s1 = Self::remove_empty_nodes(graph);
        total.nodes_removed += s1.nodes_removed;
        total.edges_removed += s1.edges_removed;

        let s2 = Self::remove_redundant_barriers(graph);
        total.nodes_removed += s2.nodes_removed;
        total.edges_removed += s2.edges_removed;

        let s3 = Self::merge_consecutive_kernels(graph);
        total.nodes_merged += s3.nodes_merged;

        total
    }
}

// ── GraphProfiler ────────────────────────────────────────────────────

/// Per-execution timing sample.
#[derive(Debug, Clone)]
pub struct ProfileSample {
    /// Graph id.
    pub graph_id: GraphId,
    /// Execution index (1-based).
    pub exec_index: u64,
    /// Wall-clock time.
    pub wall_time: Duration,
    /// Nodes executed.
    pub nodes_executed: usize,
    /// Estimated GPU time in µs.
    pub estimated_gpu_us: f64,
}

/// Profiles graph execution over multiple runs.
pub struct GraphProfiler {
    samples: Vec<ProfileSample>,
    max_samples: usize,
}

impl GraphProfiler {
    /// Create a profiler that retains at most `max_samples` entries.
    pub fn new(max_samples: usize) -> Self {
        Self { samples: Vec::new(), max_samples }
    }

    /// Execute and record a profiling sample.
    pub fn profile_execution(&mut self, graph: &mut CudaGraph) -> Result<GraphExecResult> {
        let result = graph.execute()?;
        let sample = ProfileSample {
            graph_id: result.graph_id,
            exec_index: graph.exec_count(),
            wall_time: result.wall_time,
            nodes_executed: result.nodes_executed,
            estimated_gpu_us: result.estimated_gpu_time_us,
        };
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(sample);
        Ok(result)
    }

    /// All collected samples.
    pub fn samples(&self) -> &[ProfileSample] {
        &self.samples
    }

    /// Number of samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Average wall-clock time across samples.
    pub fn avg_wall_time(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.samples.iter().map(|s| s.wall_time).sum();
        total / self.samples.len() as u32
    }

    /// Average estimated GPU time in µs.
    pub fn avg_estimated_gpu_us(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let total: f64 = self.samples.iter().map(|s| s.estimated_gpu_us).sum();
        total / self.samples.len() as f64
    }

    /// Minimum wall-clock time observed.
    pub fn min_wall_time(&self) -> Duration {
        self.samples.iter().map(|s| s.wall_time).min().unwrap_or(Duration::ZERO)
    }

    /// Maximum wall-clock time observed.
    pub fn max_wall_time(&self) -> Duration {
        self.samples.iter().map(|s| s.wall_time).max().unwrap_or(Duration::ZERO)
    }

    /// Clear all profiling data.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── NodeId / GraphId ─────────────────────────────────────────

    #[test]
    fn node_id_uniqueness() {
        let a = NodeId::next();
        let b = NodeId::next();
        assert_ne!(a, b);
    }

    #[test]
    fn graph_id_uniqueness() {
        let a = GraphId::next();
        let b = GraphId::next();
        assert_ne!(a, b);
    }

    #[test]
    fn node_id_display() {
        let id = NodeId(42);
        assert_eq!(format!("{id}"), "node-42");
    }

    #[test]
    fn graph_id_display() {
        let id = GraphId(7);
        assert_eq!(format!("{id}"), "graph-7");
    }

    // ── GraphNode construction ───────────────────────────────────

    #[test]
    fn kernel_node_basic() {
        let n = GraphNode::kernel("matmul", [8, 1, 1], [256, 1, 1]);
        assert!(matches!(n.kind, NodeKind::Kernel { .. }));
        assert!(n.enabled);
        assert_eq!(n.stream, 0);
    }

    #[test]
    fn memcopy_node() {
        let n = GraphNode::memcopy(4096);
        assert!(matches!(n.kind, NodeKind::MemCopy { bytes: 4096 }));
    }

    #[test]
    fn memset_node() {
        let n = GraphNode::memset(1024);
        assert!(matches!(n.kind, NodeKind::MemSet { bytes: 1024 }));
    }

    #[test]
    fn barrier_node() {
        let n = GraphNode::barrier();
        assert!(matches!(n.kind, NodeKind::Barrier));
    }

    #[test]
    fn empty_node() {
        let n = GraphNode::empty();
        assert!(matches!(n.kind, NodeKind::Empty));
    }

    #[test]
    fn host_callback_node() {
        let n = GraphNode::host_callback("my_cb");
        assert!(matches!(n.kind, NodeKind::HostCallback { .. }));
    }

    #[test]
    fn node_with_param() {
        let n = GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]).with_param("eps", 1e-5);
        assert!((n.params["eps"] - 1e-5).abs() < f64::EPSILON);
    }

    #[test]
    fn node_on_stream() {
        let n = GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]).on_stream(3);
        assert_eq!(n.stream, 3);
    }

    #[test]
    fn estimated_cost_kernel() {
        let n = GraphNode::kernel("k", [4, 1, 1], [256, 1, 1]);
        assert!(n.estimated_cost_us() > 0.0);
    }

    #[test]
    fn estimated_cost_memcopy() {
        let n = GraphNode::memcopy(1_000_000);
        assert!(n.estimated_cost_us() > 0.0);
    }

    #[test]
    fn estimated_cost_empty_is_zero() {
        assert_eq!(GraphNode::empty().estimated_cost_us(), 0.0);
    }

    // ── GraphEdge ────────────────────────────────────────────────

    #[test]
    fn edge_new() {
        let a = NodeId(1);
        let b = NodeId(2);
        let e = GraphEdge::new(a, b);
        assert_eq!(e.from, a);
        assert_eq!(e.to, b);
    }

    #[test]
    fn edge_display() {
        let e = GraphEdge::new(NodeId(1), NodeId(2));
        assert_eq!(format!("{e}"), "node-1 -> node-2");
    }

    #[test]
    fn edge_equality() {
        let e1 = GraphEdge::new(NodeId(1), NodeId(2));
        let e2 = GraphEdge::new(NodeId(1), NodeId(2));
        assert_eq!(e1, e2);
    }

    // ── CudaGraph basics ─────────────────────────────────────────

    #[test]
    fn new_graph_is_empty() {
        let g = CudaGraph::new("test");
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.exec_count(), 0);
        assert!(!g.is_instantiated());
        assert_eq!(g.label(), "test");
    }

    #[test]
    fn add_node_increments_count() {
        let mut g = CudaGraph::new("t");
        g.add_node(GraphNode::empty());
        g.add_node(GraphNode::barrier());
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn add_edge_valid() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::empty());
        let b = g.add_node(GraphNode::empty());
        g.add_edge(GraphEdge::new(a, b)).unwrap();
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn add_edge_missing_node_rejected() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::empty());
        let bogus = NodeId(999_999);
        assert!(g.add_edge(GraphEdge::new(a, bogus)).is_err());
    }

    #[test]
    fn instantiate_empty_graph_fails() {
        let mut g = CudaGraph::new("t");
        assert!(g.instantiate().is_err());
    }

    #[test]
    fn instantiate_single_node() {
        let mut g = CudaGraph::new("t");
        g.add_node(GraphNode::empty());
        g.instantiate().unwrap();
        assert!(g.is_instantiated());
    }

    #[test]
    fn instantiate_clears_on_topology_change() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::empty());
        g.instantiate().unwrap();
        assert!(g.is_instantiated());
        g.add_node(GraphNode::empty());
        assert!(!g.is_instantiated());
        // Adding edge also clears:
        let b = g.nodes.last().unwrap().id;
        g.add_edge(GraphEdge::new(a, b)).unwrap();
        assert!(!g.is_instantiated());
    }

    #[test]
    fn execute_single_node() {
        let mut g = CudaGraph::new("t");
        g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        let res = g.execute().unwrap();
        assert_eq!(res.nodes_executed, 1);
        assert_eq!(g.exec_count(), 1);
    }

    #[test]
    fn execute_chain() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::kernel("a", [1, 1, 1], [1, 1, 1]));
        let b = g.add_node(GraphNode::kernel("b", [1, 1, 1], [1, 1, 1]));
        let c = g.add_node(GraphNode::kernel("c", [1, 1, 1], [1, 1, 1]));
        g.add_edge(GraphEdge::new(a, b)).unwrap();
        g.add_edge(GraphEdge::new(b, c)).unwrap();
        let res = g.execute().unwrap();
        assert_eq!(res.nodes_executed, 3);
    }

    #[test]
    fn execute_skips_disabled_nodes() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::kernel("a", [1, 1, 1], [1, 1, 1]));
        let mut disabled = GraphNode::kernel("b", [1, 1, 1], [1, 1, 1]);
        disabled.enabled = false;
        let b = g.add_node(disabled);
        let c = g.add_node(GraphNode::kernel("c", [1, 1, 1], [1, 1, 1]));
        g.add_edge(GraphEdge::new(a, b)).unwrap();
        g.add_edge(GraphEdge::new(b, c)).unwrap();
        let res = g.execute().unwrap();
        assert_eq!(res.nodes_executed, 2);
    }

    #[test]
    fn execute_increments_count() {
        let mut g = CudaGraph::new("t");
        g.add_node(GraphNode::empty());
        g.execute().unwrap();
        g.execute().unwrap();
        g.execute().unwrap();
        assert_eq!(g.exec_count(), 3);
    }

    #[test]
    fn cycle_detected() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::empty());
        let b = g.add_node(GraphNode::empty());
        g.add_edge(GraphEdge::new(a, b)).unwrap();
        g.add_edge(GraphEdge::new(b, a)).unwrap();
        assert!(g.execute().is_err());
    }

    #[test]
    fn find_node() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::kernel("matmul", [1, 1, 1], [1, 1, 1]));
        assert!(g.find_node(a).is_some());
        assert!(g.find_node(NodeId(999_999)).is_none());
    }

    #[test]
    fn find_node_mut() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        g.find_node_mut(a).unwrap().enabled = false;
        assert!(!g.find_node(a).unwrap().enabled);
    }

    #[test]
    fn roots_and_leaves() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::empty());
        let b = g.add_node(GraphNode::empty());
        let c = g.add_node(GraphNode::empty());
        g.add_edge(GraphEdge::new(a, b)).unwrap();
        g.add_edge(GraphEdge::new(b, c)).unwrap();
        assert_eq!(g.roots(), vec![a]);
        assert_eq!(g.leaves(), vec![c]);
    }

    #[test]
    fn stream_count() {
        let mut g = CudaGraph::new("t");
        g.add_node(GraphNode::kernel("a", [1, 1, 1], [1, 1, 1]).on_stream(0));
        g.add_node(GraphNode::kernel("b", [1, 1, 1], [1, 1, 1]).on_stream(1));
        g.add_node(GraphNode::kernel("c", [1, 1, 1], [1, 1, 1]).on_stream(2));
        assert_eq!(g.stream_count(), 3);
    }

    // ── GraphBuilder ─────────────────────────────────────────────

    #[test]
    fn builder_basic() {
        let mut b = GraphBuilder::new("test");
        let n1 = b.add_node(GraphNode::kernel("a", [1, 1, 1], [1, 1, 1]));
        let n2 = b.add_node(GraphNode::kernel("b", [1, 1, 1], [1, 1, 1]));
        b.add_edge(n1, n2);
        let g = b.build().unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn builder_empty_fails() {
        let b = GraphBuilder::new("empty");
        assert!(b.build().is_err());
    }

    #[test]
    fn builder_invalid_edge_fails() {
        let mut b = GraphBuilder::new("bad");
        b.add_node(GraphNode::empty());
        b.add_edge(NodeId(0), NodeId(999_999));
        assert!(b.build().is_err());
    }

    #[test]
    fn builder_add_chain() {
        let mut b = GraphBuilder::new("chain");
        let ids = b.add_chain(vec![
            GraphNode::kernel("a", [1, 1, 1], [1, 1, 1]),
            GraphNode::kernel("b", [1, 1, 1], [1, 1, 1]),
            GraphNode::kernel("c", [1, 1, 1], [1, 1, 1]),
        ]);
        assert_eq!(ids.len(), 3);
        let g = b.build().unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn builder_chain_single_node() {
        let mut b = GraphBuilder::new("single");
        let ids = b.add_chain(vec![GraphNode::empty()]);
        assert_eq!(ids.len(), 1);
        let g = b.build().unwrap();
        assert_eq!(g.edge_count(), 0);
    }

    // ── capture / execute free functions ─────────────────────────

    #[test]
    fn capture_basic() {
        let mut g = capture("test", |b| {
            let a = b.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
            let c = b.add_node(GraphNode::kernel("k2", [1, 1, 1], [1, 1, 1]));
            b.add_edge(a, c);
        })
        .unwrap();
        let res = execute(&mut g).unwrap();
        assert_eq!(res.nodes_executed, 2);
    }

    #[test]
    fn capture_empty_fails() {
        assert!(capture("empty", |_b| {}).is_err());
    }

    // ── update_params ────────────────────────────────────────────

    #[test]
    fn update_params_basic() {
        let mut g = capture("test", |b| {
            b.add_node(GraphNode::kernel("matmul", [1, 1, 1], [1, 1, 1]));
        })
        .unwrap();
        let mut params = HashMap::new();
        params.insert("alpha".to_string(), 2.0);
        let count = update_params(&mut g, "matmul", &params).unwrap();
        assert_eq!(count, 1);
        let node = &g.nodes()[0];
        assert!((node.params["alpha"] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn update_params_multiple_matches() {
        let mut g = capture("test", |b| {
            let a = b.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
            let c = b.add_node(GraphNode::kernel("k", [2, 1, 1], [1, 1, 1]));
            b.add_edge(a, c);
        })
        .unwrap();
        let mut params = HashMap::new();
        params.insert("x".to_string(), 42.0);
        let count = update_params(&mut g, "k", &params).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn update_params_no_match_fails() {
        let mut g = capture("test", |b| {
            b.add_node(GraphNode::kernel("matmul", [1, 1, 1], [1, 1, 1]));
        })
        .unwrap();
        let params = HashMap::new();
        assert!(update_params(&mut g, "nonexistent", &params).is_err());
    }

    #[test]
    fn update_params_invalidates_instantiation() {
        let mut g = capture("test", |b| {
            b.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        })
        .unwrap();
        g.instantiate().unwrap();
        assert!(g.is_instantiated());
        let mut params = HashMap::new();
        params.insert("val".to_string(), 1.0);
        update_params(&mut g, "k", &params).unwrap();
        assert!(!g.is_instantiated());
    }

    // ── graph_from_model_layer ───────────────────────────────────

    #[test]
    fn model_layer_default() {
        let g = graph_from_model_layer(&LayerGraphConfig::default()).unwrap();
        // entry + norm1 + qkv + attn + out_proj + add1 + norm2 + gate_up + silu + down + add2 + exit = 12
        assert_eq!(g.node_count(), 12);
        assert!(g.edge_count() >= 11);
    }

    #[test]
    fn model_layer_attention_only() {
        let cfg = LayerGraphConfig { include_mlp: false, ..Default::default() };
        let g = graph_from_model_layer(&cfg).unwrap();
        // entry + norm1 + qkv + attn + out_proj + add1 + exit = 7
        assert_eq!(g.node_count(), 7);
    }

    #[test]
    fn model_layer_mlp_only() {
        let cfg = LayerGraphConfig { include_attention: false, ..Default::default() };
        let g = graph_from_model_layer(&cfg).unwrap();
        // entry + norm1 + add1 + norm2 + gate_up + silu + down + add2 + exit = 9
        assert_eq!(g.node_count(), 9);
    }

    #[test]
    fn model_layer_neither() {
        let cfg =
            LayerGraphConfig { include_attention: false, include_mlp: false, ..Default::default() };
        let g = graph_from_model_layer(&cfg).unwrap();
        // entry + norm1 + add1 + exit = 4
        assert_eq!(g.node_count(), 4);
    }

    #[test]
    fn model_layer_zero_hidden_fails() {
        let cfg = LayerGraphConfig { hidden_dim: 0, ..Default::default() };
        assert!(graph_from_model_layer(&cfg).is_err());
    }

    #[test]
    fn model_layer_zero_heads_fails() {
        let cfg = LayerGraphConfig { num_heads: 0, ..Default::default() };
        assert!(graph_from_model_layer(&cfg).is_err());
    }

    #[test]
    fn model_layer_zero_seq_len_fails() {
        let cfg = LayerGraphConfig { seq_len: 0, ..Default::default() };
        assert!(graph_from_model_layer(&cfg).is_err());
    }

    #[test]
    fn model_layer_is_acyclic() {
        let mut g = graph_from_model_layer(&LayerGraphConfig::default()).unwrap();
        g.instantiate().unwrap();
    }

    #[test]
    fn model_layer_executable() {
        let mut g = graph_from_model_layer(&LayerGraphConfig::default()).unwrap();
        let res = g.execute().unwrap();
        assert!(res.nodes_executed > 0);
    }

    #[test]
    fn model_layer_multi_stream() {
        let cfg = LayerGraphConfig { attention_stream: 0, mlp_stream: 1, ..Default::default() };
        let g = graph_from_model_layer(&cfg).unwrap();
        assert!(g.stream_count() >= 2);
    }

    // ── GraphPool ────────────────────────────────────────────────

    #[test]
    fn pool_creation() {
        let pool = GraphPool::new(LayerGraphConfig::default(), 4).unwrap();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn pool_zero_entries_fails() {
        assert!(GraphPool::new(LayerGraphConfig::default(), 0).is_err());
    }

    #[test]
    fn pool_get_or_capture() {
        let mut pool = GraphPool::new(LayerGraphConfig::default(), 4).unwrap();
        pool.get_or_capture(128).unwrap();
        assert_eq!(pool.len(), 1);
        assert!(pool.contains(128));
    }

    #[test]
    fn pool_hit_miss_counting() {
        let mut pool = GraphPool::new(LayerGraphConfig::default(), 4).unwrap();
        pool.get_or_capture(64).unwrap(); // miss
        pool.get_or_capture(64).unwrap(); // hit
        pool.get_or_capture(128).unwrap(); // miss
        assert_eq!(pool.hits(), 1);
        assert_eq!(pool.misses(), 2);
    }

    #[test]
    fn pool_hit_rate() {
        let mut pool = GraphPool::new(LayerGraphConfig::default(), 4).unwrap();
        pool.get_or_capture(64).unwrap(); // miss
        pool.get_or_capture(64).unwrap(); // hit
        assert!((pool.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn pool_hit_rate_empty() {
        let pool = GraphPool::new(LayerGraphConfig::default(), 4).unwrap();
        assert_eq!(pool.hit_rate(), 0.0);
    }

    #[test]
    fn pool_eviction() {
        let mut pool = GraphPool::new(LayerGraphConfig::default(), 2).unwrap();
        pool.get_or_capture(64).unwrap();
        pool.get_or_capture(128).unwrap();
        pool.get_or_capture(256).unwrap(); // evicts smallest (64)
        assert_eq!(pool.len(), 2);
        assert!(!pool.contains(64));
        assert!(pool.contains(128));
        assert!(pool.contains(256));
    }

    #[test]
    fn pool_clear() {
        let mut pool = GraphPool::new(LayerGraphConfig::default(), 4).unwrap();
        pool.get_or_capture(64).unwrap();
        pool.get_or_capture(128).unwrap();
        pool.clear();
        assert!(pool.is_empty());
        assert_eq!(pool.hits(), 0);
        assert_eq!(pool.misses(), 0);
    }

    #[test]
    fn pool_execute() {
        let mut pool = GraphPool::new(LayerGraphConfig::default(), 4).unwrap();
        let res = pool.execute(128).unwrap();
        assert!(res.nodes_executed > 0);
    }

    // ── conditional_graph ────────────────────────────────────────

    #[test]
    fn conditional_graph_enables() {
        let n1 = GraphNode::kernel("a", [1, 1, 1], [1, 1, 1]);
        let n2 = GraphNode::kernel("b", [1, 1, 1], [1, 1, 1]);
        let id1 = n1.id;
        let id2 = n2.id;

        let edges = vec![GraphEdge::new(id1, id2)];
        let mut conds = HashMap::new();
        conds.insert(id2, false);

        let mut g = conditional_graph("cond", vec![n1, n2], edges, &conds).unwrap();
        let res = g.execute().unwrap();
        assert_eq!(res.nodes_executed, 1); // only n1 enabled
    }

    #[test]
    fn conditional_graph_all_enabled() {
        let n1 = GraphNode::kernel("a", [1, 1, 1], [1, 1, 1]);
        let n2 = GraphNode::kernel("b", [1, 1, 1], [1, 1, 1]);
        let id1 = n1.id;
        let id2 = n2.id;

        let edges = vec![GraphEdge::new(id1, id2)];
        let conds = HashMap::new(); // no overrides → all enabled

        let mut g = conditional_graph("cond", vec![n1, n2], edges, &conds).unwrap();
        let res = g.execute().unwrap();
        assert_eq!(res.nodes_executed, 2);
    }

    // ── multi_stream_graph ───────────────────────────────────────

    #[test]
    fn multi_stream_basic() {
        let mut streams = HashMap::new();
        streams.insert(
            0,
            vec![
                GraphNode::kernel("s0a", [1, 1, 1], [1, 1, 1]),
                GraphNode::kernel("s0b", [1, 1, 1], [1, 1, 1]),
            ],
        );
        streams.insert(1, vec![GraphNode::kernel("s1a", [1, 1, 1], [1, 1, 1])]);
        let cfg = MultiStreamConfig::default();
        let mut g = multi_stream_graph("ms", streams, &cfg).unwrap();
        // 2 + 1 nodes + 1 barrier = 4 nodes
        assert_eq!(g.node_count(), 4);
        let res = g.execute().unwrap();
        assert_eq!(res.nodes_executed, 4);
    }

    #[test]
    fn multi_stream_no_barrier() {
        let mut streams = HashMap::new();
        streams.insert(0, vec![GraphNode::kernel("a", [1, 1, 1], [1, 1, 1])]);
        streams.insert(1, vec![GraphNode::kernel("b", [1, 1, 1], [1, 1, 1])]);
        let cfg = MultiStreamConfig { sync_barriers: false, ..Default::default() };
        let g = multi_stream_graph("ms", streams, &cfg).unwrap();
        assert_eq!(g.node_count(), 2); // no barrier
    }

    #[test]
    fn multi_stream_empty_fails() {
        let streams = HashMap::new();
        assert!(multi_stream_graph("ms", streams, &MultiStreamConfig::default()).is_err());
    }

    #[test]
    fn multi_stream_single_stream_no_barrier() {
        let mut streams = HashMap::new();
        streams.insert(
            0,
            vec![
                GraphNode::kernel("a", [1, 1, 1], [1, 1, 1]),
                GraphNode::kernel("b", [1, 1, 1], [1, 1, 1]),
            ],
        );
        let cfg = MultiStreamConfig { sync_barriers: true, ..Default::default() };
        let g = multi_stream_graph("ms", streams, &cfg).unwrap();
        // Only 1 stream → no barrier even if requested.
        assert_eq!(g.node_count(), 2);
    }

    // ── GraphOptimizer ───────────────────────────────────────────

    #[test]
    fn optimizer_remove_empty_passthrough() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        let empty = g.add_node(GraphNode::empty());
        let b = g.add_node(GraphNode::kernel("k2", [1, 1, 1], [1, 1, 1]));
        g.add_edge(GraphEdge::new(a, empty)).unwrap();
        g.add_edge(GraphEdge::new(empty, b)).unwrap();

        let stats = GraphOptimizer::remove_empty_nodes(&mut g);
        assert_eq!(stats.nodes_removed, 1);
        assert_eq!(g.node_count(), 2);
        // Check re-wiring: a → b.
        assert!(g.edges().iter().any(|e| e.from == a && e.to == b));
    }

    #[test]
    fn optimizer_does_not_remove_fanout_empty() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        let empty = g.add_node(GraphNode::empty());
        let b = g.add_node(GraphNode::kernel("k2", [1, 1, 1], [1, 1, 1]));
        let c = g.add_node(GraphNode::kernel("k3", [1, 1, 1], [1, 1, 1]));
        g.add_edge(GraphEdge::new(a, empty)).unwrap();
        g.add_edge(GraphEdge::new(empty, b)).unwrap();
        g.add_edge(GraphEdge::new(empty, c)).unwrap();

        let stats = GraphOptimizer::remove_empty_nodes(&mut g);
        assert_eq!(stats.nodes_removed, 0); // fan-out > 1, not removed
    }

    #[test]
    fn optimizer_remove_redundant_single_pred_barrier() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        let bar = g.add_node(GraphNode::barrier());
        let b = g.add_node(GraphNode::kernel("k2", [1, 1, 1], [1, 1, 1]));
        g.add_edge(GraphEdge::new(a, bar)).unwrap();
        g.add_edge(GraphEdge::new(bar, b)).unwrap();

        let stats = GraphOptimizer::remove_redundant_barriers(&mut g);
        assert_eq!(stats.nodes_removed, 1);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn optimizer_keeps_multi_pred_barrier() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::kernel("a", [1, 1, 1], [1, 1, 1]));
        let b = g.add_node(GraphNode::kernel("b", [1, 1, 1], [1, 1, 1]));
        let bar = g.add_node(GraphNode::barrier());
        let c = g.add_node(GraphNode::kernel("c", [1, 1, 1], [1, 1, 1]));
        g.add_edge(GraphEdge::new(a, bar)).unwrap();
        g.add_edge(GraphEdge::new(b, bar)).unwrap();
        g.add_edge(GraphEdge::new(bar, c)).unwrap();

        let stats = GraphOptimizer::remove_redundant_barriers(&mut g);
        assert_eq!(stats.nodes_removed, 0);
    }

    #[test]
    fn optimizer_merge_consecutive_same_kernel() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::kernel("k", [4, 1, 1], [256, 1, 1]));
        let b = g.add_node(GraphNode::kernel("k", [4, 1, 1], [256, 1, 1]));
        g.add_edge(GraphEdge::new(a, b)).unwrap();

        let stats = GraphOptimizer::merge_consecutive_kernels(&mut g);
        assert_eq!(stats.nodes_merged, 1);
        assert_eq!(g.node_count(), 1);
        // Merged node has doubled grid.x.
        if let NodeKind::Kernel { grid, .. } = &g.nodes()[0].kind {
            assert_eq!(grid[0], 8);
        } else {
            panic!("expected Kernel node");
        }
    }

    #[test]
    fn optimizer_no_merge_different_kernels() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::kernel("k1", [1, 1, 1], [1, 1, 1]));
        let b = g.add_node(GraphNode::kernel("k2", [1, 1, 1], [1, 1, 1]));
        g.add_edge(GraphEdge::new(a, b)).unwrap();

        let stats = GraphOptimizer::merge_consecutive_kernels(&mut g);
        assert_eq!(stats.nodes_merged, 0);
    }

    #[test]
    fn optimizer_optimize_all() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        let empty = g.add_node(GraphNode::empty());
        let b = g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        g.add_edge(GraphEdge::new(a, empty)).unwrap();
        g.add_edge(GraphEdge::new(empty, b)).unwrap();

        let stats = GraphOptimizer::optimize_all(&mut g);
        assert!(stats.nodes_removed > 0 || stats.nodes_merged > 0);
    }

    #[test]
    fn optimizer_graph_still_executable_after() {
        let mut g = graph_from_model_layer(&LayerGraphConfig::default()).unwrap();
        GraphOptimizer::optimize_all(&mut g);
        let res = g.execute().unwrap();
        assert!(res.nodes_executed > 0);
    }

    // ── GraphProfiler ────────────────────────────────────────────

    #[test]
    fn profiler_basic() {
        let profiler = GraphProfiler::new(100);
        assert_eq!(profiler.sample_count(), 0);
    }

    #[test]
    fn profiler_records_samples() {
        let mut profiler = GraphProfiler::new(100);
        let mut g = CudaGraph::new("t");
        g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        profiler.profile_execution(&mut g).unwrap();
        profiler.profile_execution(&mut g).unwrap();
        assert_eq!(profiler.sample_count(), 2);
    }

    #[test]
    fn profiler_evicts_oldest() {
        let mut profiler = GraphProfiler::new(2);
        let mut g = CudaGraph::new("t");
        g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        profiler.profile_execution(&mut g).unwrap();
        profiler.profile_execution(&mut g).unwrap();
        profiler.profile_execution(&mut g).unwrap();
        assert_eq!(profiler.sample_count(), 2);
    }

    #[test]
    fn profiler_avg_wall_time_empty() {
        let profiler = GraphProfiler::new(10);
        assert_eq!(profiler.avg_wall_time(), Duration::ZERO);
    }

    #[test]
    fn profiler_avg_wall_time() {
        let mut profiler = GraphProfiler::new(100);
        let mut g = CudaGraph::new("t");
        g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        profiler.profile_execution(&mut g).unwrap();
        assert!(profiler.avg_wall_time() >= Duration::ZERO);
    }

    #[test]
    fn profiler_avg_gpu_time_empty() {
        let profiler = GraphProfiler::new(10);
        assert_eq!(profiler.avg_estimated_gpu_us(), 0.0);
    }

    #[test]
    fn profiler_min_max_wall_time() {
        let mut profiler = GraphProfiler::new(100);
        let mut g = CudaGraph::new("t");
        g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        profiler.profile_execution(&mut g).unwrap();
        profiler.profile_execution(&mut g).unwrap();
        assert!(profiler.min_wall_time() <= profiler.max_wall_time());
    }

    #[test]
    fn profiler_clear() {
        let mut profiler = GraphProfiler::new(100);
        let mut g = CudaGraph::new("t");
        g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        profiler.profile_execution(&mut g).unwrap();
        profiler.clear();
        assert_eq!(profiler.sample_count(), 0);
    }

    // ── GraphExecResult ──────────────────────────────────────────

    #[test]
    fn exec_result_has_graph_id() {
        let mut g = CudaGraph::new("t");
        g.add_node(GraphNode::kernel("k", [1, 1, 1], [1, 1, 1]));
        let res = g.execute().unwrap();
        assert_eq!(res.graph_id, g.id);
    }

    #[test]
    fn exec_result_estimated_gpu_time() {
        let mut g = CudaGraph::new("t");
        g.add_node(GraphNode::kernel("k", [4, 1, 1], [256, 1, 1]));
        let res = g.execute().unwrap();
        assert!(res.estimated_gpu_time_us > 0.0);
    }

    // ── Edge cases ───────────────────────────────────────────────

    #[test]
    fn diamond_dag() {
        let mut g = CudaGraph::new("diamond");
        let a = g.add_node(GraphNode::kernel("a", [1, 1, 1], [1, 1, 1]));
        let b = g.add_node(GraphNode::kernel("b", [1, 1, 1], [1, 1, 1]));
        let c = g.add_node(GraphNode::kernel("c", [1, 1, 1], [1, 1, 1]));
        let d = g.add_node(GraphNode::kernel("d", [1, 1, 1], [1, 1, 1]));
        g.add_edge(GraphEdge::new(a, b)).unwrap();
        g.add_edge(GraphEdge::new(a, c)).unwrap();
        g.add_edge(GraphEdge::new(b, d)).unwrap();
        g.add_edge(GraphEdge::new(c, d)).unwrap();
        let res = g.execute().unwrap();
        assert_eq!(res.nodes_executed, 4);
    }

    #[test]
    fn disconnected_nodes_execute() {
        let mut g = CudaGraph::new("disconnected");
        g.add_node(GraphNode::kernel("a", [1, 1, 1], [1, 1, 1]));
        g.add_node(GraphNode::kernel("b", [1, 1, 1], [1, 1, 1]));
        // No edges
        let res = g.execute().unwrap();
        assert_eq!(res.nodes_executed, 2);
    }

    #[test]
    fn large_chain_executes() {
        let mut b = GraphBuilder::new("big");
        let nodes: Vec<_> =
            (0..100).map(|i| GraphNode::kernel(&format!("k{i}"), [1, 1, 1], [1, 1, 1])).collect();
        b.add_chain(nodes);
        let mut g = b.build().unwrap();
        let res = g.execute().unwrap();
        assert_eq!(res.nodes_executed, 100);
    }

    #[test]
    fn self_loop_detected() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::empty());
        g.add_edge(GraphEdge::new(a, a)).unwrap();
        assert!(g.execute().is_err());
    }

    #[test]
    fn multiple_roots_multiple_leaves() {
        let mut g = CudaGraph::new("t");
        let a = g.add_node(GraphNode::kernel("a", [1, 1, 1], [1, 1, 1]));
        let b = g.add_node(GraphNode::kernel("b", [1, 1, 1], [1, 1, 1]));
        let c = g.add_node(GraphNode::kernel("c", [1, 1, 1], [1, 1, 1]));
        let d = g.add_node(GraphNode::kernel("d", [1, 1, 1], [1, 1, 1]));
        g.add_edge(GraphEdge::new(a, c)).unwrap();
        g.add_edge(GraphEdge::new(b, d)).unwrap();
        assert_eq!(g.roots().len(), 2);
        assert_eq!(g.leaves().len(), 2);
    }
}
