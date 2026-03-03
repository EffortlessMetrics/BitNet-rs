//! NEON-optimized computation graph executor for Apple Silicon.
//!
//! Provides a typed computation graph with topological execution ordering,
//! memory planning, operator fusion detection, and execution statistics.
//! NEON intrinsics accelerate elementwise, matmul, and normalization kernels
//! on AArch64; a scalar fallback is used on other architectures.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::let_and_return,
    clippy::derivable_impls,
    clippy::excessive_precision,
    clippy::manual_is_multiple_of
)]
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// ── Lane constant ───────────────────────────────────────────────────────

/// NEON f32 lane width.
const LANES: usize = 4;

// ── Node operation types ────────────────────────────────────────────────

/// Operations supported by the computation graph.
#[derive(Debug, Clone, PartialEq)]
pub enum OpType {
    /// Elementwise addition of two tensors.
    Add,
    /// Elementwise multiplication of two tensors.
    Mul,
    /// Matrix multiplication (M×K) × (K×N) → (M×N).
    MatMul { m: usize, k: usize, n: usize },
    /// Rectified linear unit: max(0, x).
    ReLU,
    /// Softmax over a 1-D vector.
    Softmax,
    /// Layer normalization with affine parameters.
    LayerNorm { eps: f32 },
    /// Constant / input tensor (no computation, just provides data).
    Input,
}

// ── Graph node ──────────────────────────────────────────────────────────

/// Unique identifier for a node in the computation graph.
pub type NodeId = usize;

/// A single node in the computation graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier.
    pub id: NodeId,
    /// Human-readable name.
    pub name: String,
    /// Operation to perform.
    pub op: OpType,
    /// Input node ids (edges).
    pub inputs: Vec<NodeId>,
    /// Output size in number of f32 elements.
    pub output_size: usize,
}

// ── Fusion opportunities ────────────────────────────────────────────────

/// A detected operator-fusion opportunity.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionOpportunity {
    /// The nodes that could be fused (in execution order).
    pub nodes: Vec<NodeId>,
    /// Human-readable description of the fusion.
    pub description: String,
}

// ── Execution statistics ────────────────────────────────────────────────

/// Per-node execution timing.
#[derive(Debug, Clone)]
pub struct NodeStats {
    pub node_id: NodeId,
    pub node_name: String,
    pub op: OpType,
    pub elapsed: Duration,
}

/// Aggregate statistics for a full graph execution.
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub total_elapsed: Duration,
    pub node_stats: Vec<NodeStats>,
    pub peak_memory_bytes: usize,
    pub nodes_executed: usize,
}

// ── Memory plan ─────────────────────────────────────────────────────────

/// A memory allocation plan computed ahead of execution.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Byte offset for each node's output buffer.
    pub offsets: HashMap<NodeId, usize>,
    /// Total bytes required for the arena.
    pub total_bytes: usize,
}

// ── Computation graph ───────────────────────────────────────────────────

/// A computation graph that can be built incrementally and executed.
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    nodes: Vec<GraphNode>,
}

impl ComputationGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a node and return its id.
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        op: OpType,
        inputs: Vec<NodeId>,
        output_size: usize,
    ) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(GraphNode { id, name: name.into(), op, inputs, output_size });
        id
    }

    /// Return a reference to all nodes.
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Return the number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Compute a topological ordering of the nodes via Kahn's algorithm.
    ///
    /// Returns `None` if the graph contains a cycle.
    pub fn topological_sort(&self) -> Option<Vec<NodeId>> {
        let n = self.nodes.len();
        let mut in_degree = vec![0u32; n];
        let mut dependents: Vec<Vec<NodeId>> = vec![vec![]; n];

        for node in &self.nodes {
            in_degree[node.id] = node.inputs.len() as u32;
            for &inp in &node.inputs {
                dependents[inp].push(node.id);
            }
        }

        let mut queue: VecDeque<NodeId> = VecDeque::new();
        for (id, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(id);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(id) = queue.pop_front() {
            order.push(id);
            for &dep in &dependents[id] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    queue.push_back(dep);
                }
            }
        }

        if order.len() == n { Some(order) } else { None }
    }

    /// Compute a memory plan that assigns non-overlapping buffer regions.
    pub fn memory_plan(&self) -> MemoryPlan {
        let mut offsets = HashMap::new();
        let mut current = 0usize;
        // Allocate in node-id order (simple bump allocator).
        for node in &self.nodes {
            let bytes = node.output_size * std::mem::size_of::<f32>();
            offsets.insert(node.id, current);
            current += bytes;
        }
        MemoryPlan { offsets, total_bytes: current }
    }

    /// Detect operator-fusion opportunities.
    ///
    /// Currently recognises:
    /// - MatMul followed by ReLU  (fused GEMM + activation)
    /// - MatMul followed by Add   (bias-add fusion)
    /// - Add followed by ReLU     (residual + activation)
    /// - LayerNorm followed by Add (norm + residual)
    pub fn detect_fusions(&self) -> Vec<FusionOpportunity> {
        let order = match self.topological_sort() {
            Some(o) => o,
            None => return vec![],
        };

        // Build a map from producer → list of consumers.
        let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for node in &self.nodes {
            for &inp in &node.inputs {
                consumers.entry(inp).or_default().push(node.id);
            }
        }

        let mut fusions = Vec::new();

        for &id in &order {
            let node = &self.nodes[id];
            // Only consider single-consumer edges for fusion.
            if let Some(succs) = consumers.get(&id) {
                if succs.len() == 1 {
                    let succ = &self.nodes[succs[0]];
                    let pair = (&node.op, &succ.op);
                    let desc = match pair {
                        (OpType::MatMul { .. }, OpType::ReLU) => {
                            Some("MatMul+ReLU: fused GEMM activation")
                        }
                        (OpType::MatMul { .. }, OpType::Add) => Some("MatMul+Add: bias-add fusion"),
                        (OpType::Add, OpType::ReLU) => Some("Add+ReLU: residual activation fusion"),
                        (OpType::LayerNorm { .. }, OpType::Add) => {
                            Some("LayerNorm+Add: norm-residual fusion")
                        }
                        _ => None,
                    };
                    if let Some(d) = desc {
                        fusions.push(FusionOpportunity {
                            nodes: vec![id, succs[0]],
                            description: d.to_string(),
                        });
                    }
                }
            }
        }

        fusions
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Graph executor ──────────────────────────────────────────────────────

/// Executes a [`ComputationGraph`] with NEON acceleration (aarch64) or
/// scalar fallback.
pub struct GraphExecutor {
    graph: ComputationGraph,
}

impl GraphExecutor {
    /// Wrap an existing graph for execution.
    pub fn new(graph: ComputationGraph) -> Self {
        Self { graph }
    }

    /// Access the underlying graph.
    pub fn graph(&self) -> &ComputationGraph {
        &self.graph
    }

    /// Execute the graph, given pre-filled input buffers keyed by node id.
    ///
    /// Returns a map from every node id to its output buffer, plus execution
    /// statistics.
    pub fn execute(
        &self,
        inputs: &HashMap<NodeId, Vec<f32>>,
    ) -> Result<(HashMap<NodeId, Vec<f32>>, ExecutionStats), GraphExecutorError> {
        let order = self.graph.topological_sort().ok_or(GraphExecutorError::CycleDetected)?;

        let plan = self.graph.memory_plan();
        let mut buffers: HashMap<NodeId, Vec<f32>> = HashMap::new();
        let mut node_stats: Vec<NodeStats> = Vec::new();
        let total_start = Instant::now();

        for &id in &order {
            let node = &self.graph.nodes[id];
            let start = Instant::now();

            let result = match &node.op {
                OpType::Input => {
                    let buf = inputs.get(&id).ok_or(GraphExecutorError::MissingInput(id))?;
                    buf.clone()
                }
                OpType::Add => {
                    let a = Self::get_buffer(&buffers, node.inputs[0])?;
                    let b = Self::get_buffer(&buffers, node.inputs[1])?;
                    Self::exec_add(a, b)
                }
                OpType::Mul => {
                    let a = Self::get_buffer(&buffers, node.inputs[0])?;
                    let b = Self::get_buffer(&buffers, node.inputs[1])?;
                    Self::exec_mul(a, b)
                }
                OpType::MatMul { m, k, n } => {
                    let a = Self::get_buffer(&buffers, node.inputs[0])?;
                    let b = Self::get_buffer(&buffers, node.inputs[1])?;
                    Self::exec_matmul(a, b, *m, *k, *n)
                }
                OpType::ReLU => {
                    let a = Self::get_buffer(&buffers, node.inputs[0])?;
                    Self::exec_relu(a)
                }
                OpType::Softmax => {
                    let a = Self::get_buffer(&buffers, node.inputs[0])?;
                    Self::exec_softmax(a)
                }
                OpType::LayerNorm { eps } => {
                    let input = Self::get_buffer(&buffers, node.inputs[0])?;
                    let gamma = Self::get_buffer(&buffers, node.inputs[1])?;
                    let beta = Self::get_buffer(&buffers, node.inputs[2])?;
                    Self::exec_layer_norm(input, gamma, beta, *eps)
                }
            };

            let elapsed = start.elapsed();
            node_stats.push(NodeStats {
                node_id: id,
                node_name: node.name.clone(),
                op: node.op.clone(),
                elapsed,
            });

            buffers.insert(id, result);
        }

        let stats = ExecutionStats {
            total_elapsed: total_start.elapsed(),
            node_stats,
            peak_memory_bytes: plan.total_bytes,
            nodes_executed: order.len(),
        };

        Ok((buffers, stats))
    }

    // ── helpers ─────────────────────────────────────────────────────────

    fn get_buffer(
        buffers: &HashMap<NodeId, Vec<f32>>,
        id: NodeId,
    ) -> Result<&Vec<f32>, GraphExecutorError> {
        buffers.get(&id).ok_or(GraphExecutorError::MissingInput(id))
    }

    // ── kernel dispatch (NEON or scalar) ────────────────────────────────

    fn exec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "add: length mismatch");
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is always available on aarch64.
            unsafe { neon_add(a, b) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            a.iter().zip(b).map(|(x, y)| x + y).collect()
        }
    }

    fn exec_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "mul: length mismatch");
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { neon_mul(a, b) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            a.iter().zip(b).map(|(x, y)| x * y).collect()
        }
    }

    fn exec_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        assert_eq!(a.len(), m * k, "matmul: A size mismatch");
        assert_eq!(b.len(), k * n, "matmul: B size mismatch");
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { neon_matmul(a, b, m, k, n) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            scalar_matmul(a, b, m, k, n)
        }
    }

    fn exec_relu(a: &[f32]) -> Vec<f32> {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { neon_relu(a) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            a.iter().map(|&x| x.max(0.0)).collect()
        }
    }

    fn exec_softmax(a: &[f32]) -> Vec<f32> {
        // Numerically stable softmax (scalar – suitable for short vectors).
        scalar_softmax(a)
    }

    fn exec_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        assert_eq!(input.len(), gamma.len(), "layernorm: gamma length mismatch");
        assert_eq!(input.len(), beta.len(), "layernorm: beta length mismatch");
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { neon_layer_norm(input, gamma, beta, eps) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            scalar_layer_norm(input, gamma, beta, eps)
        }
    }
}

// ── Error type ──────────────────────────────────────────────────────────

/// Errors that can occur during graph execution.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphExecutorError {
    /// A cycle was detected in the graph — topological sort impossible.
    CycleDetected,
    /// An expected input buffer was not provided.
    MissingInput(NodeId),
}

impl std::fmt::Display for GraphExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycleDetected => write!(f, "cycle detected in computation graph"),
            Self::MissingInput(id) => write!(f, "missing input buffer for node {id}"),
        }
    }
}

impl std::error::Error for GraphExecutorError {}

// ── NEON kernels ────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut out = vec![0.0f32; n];
    let chunks = n / LANES;
    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_f32(a.as_ptr().add(off));
        let vb = vld1q_f32(b.as_ptr().add(off));
        vst1q_f32(out.as_mut_ptr().add(off), vaddq_f32(va, vb));
    }
    for i in (chunks * LANES)..n {
        out[i] = a[i] + b[i];
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut out = vec![0.0f32; n];
    let chunks = n / LANES;
    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_f32(a.as_ptr().add(off));
        let vb = vld1q_f32(b.as_ptr().add(off));
        vst1q_f32(out.as_mut_ptr().add(off), vmulq_f32(va, vb));
    }
    for i in (chunks * LANES)..n {
        out[i] = a[i] * b[i];
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_relu(a: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut out = vec![0.0f32; n];
    let chunks = n / LANES;
    let zero = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_f32(a.as_ptr().add(off));
        vst1q_f32(out.as_mut_ptr().add(off), vmaxq_f32(va, zero));
    }
    for i in (chunks * LANES)..n {
        out[i] = a[i].max(0.0);
    }
    out
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    let k_chunks = k / LANES;
    for row in 0..m {
        for col in 0..n {
            let mut acc = vdupq_n_f32(0.0);
            for kk in 0..k_chunks {
                let off = kk * LANES;
                let va = vld1q_f32(a.as_ptr().add(row * k + off));
                let vb = load_col_f32(b, off, col, n, k);
                acc = vfmaq_f32(acc, va, vb);
            }
            let mut sum: f32 = vaddvq_f32(acc);
            for kk in (k_chunks * LANES)..k {
                sum += a[row * k + kk] * b[kk * n + col];
            }
            out[row * n + col] = sum;
        }
    }
    out
}

/// Load 4 elements from column `col` of a row-major matrix (stride = `n`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_col_f32(b: &[f32], row_start: usize, col: usize, n: usize, k: usize) -> float32x4_t {
    let _ = k;
    let i0 = row_start * n + col;
    let i1 = (row_start + 1) * n + col;
    let i2 = (row_start + 2) * n + col;
    let i3 = (row_start + 3) * n + col;
    let mut v = vdupq_n_f32(0.0);
    v = vsetq_lane_f32::<0>(b[i0], v);
    v = vsetq_lane_f32::<1>(b[i1], v);
    v = vsetq_lane_f32::<2>(b[i2], v);
    v = vsetq_lane_f32::<3>(b[i3], v);
    v
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mut out = vec![0.0f32; n];

    // Mean.
    let mut sum_acc = vdupq_n_f32(0.0);
    let chunks = n / LANES;
    for i in 0..chunks {
        let off = i * LANES;
        let v = vld1q_f32(input.as_ptr().add(off));
        sum_acc = vaddq_f32(sum_acc, v);
    }
    let mut mean: f32 = vaddvq_f32(sum_acc);
    for i in (chunks * LANES)..n {
        mean += input[i];
    }
    mean /= n as f32;

    // Variance.
    let mean_v = vdupq_n_f32(mean);
    let mut var_acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let off = i * LANES;
        let v = vld1q_f32(input.as_ptr().add(off));
        let d = vsubq_f32(v, mean_v);
        var_acc = vfmaq_f32(var_acc, d, d);
    }
    let mut variance: f32 = vaddvq_f32(var_acc);
    for i in (chunks * LANES)..n {
        let d = input[i] - mean;
        variance += d * d;
    }
    variance /= n as f32;
    let inv_std = 1.0 / (variance + eps).sqrt();

    // Normalize with affine.
    let inv_v = vdupq_n_f32(inv_std);
    for i in 0..chunks {
        let off = i * LANES;
        let v = vld1q_f32(input.as_ptr().add(off));
        let g = vld1q_f32(gamma.as_ptr().add(off));
        let b = vld1q_f32(beta.as_ptr().add(off));
        let d = vsubq_f32(v, mean_v);
        let normed = vmulq_f32(d, inv_v);
        let res = vfmaq_f32(b, normed, g);
        vst1q_f32(out.as_mut_ptr().add(off), res);
    }
    for i in (chunks * LANES)..n {
        out[i] = gamma[i] * ((input[i] - mean) * inv_std) + beta[i];
    }
    out
}

// ── Scalar fallbacks ────────────────────────────────────────────────────

fn scalar_softmax(a: &[f32]) -> Vec<f32> {
    if a.is_empty() {
        return vec![];
    }
    let max = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = a.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(not(target_arch = "aarch64"))]
fn scalar_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a[row * k + kk] * b[kk * n + col];
            }
            out[row * n + col] = s;
        }
    }
    out
}

#[cfg(not(target_arch = "aarch64"))]
fn scalar_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    input.iter().zip(gamma).zip(beta).map(|((&x, &g), &b)| g * ((x - mean) * inv_std) + b).collect()
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for floating-point comparisons.
    const EPS: f32 = 1e-5;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() < tol, "index {i}: {x} vs {y} (diff {})", (x - y).abs());
        }
    }

    // ── Graph construction ──────────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = ComputationGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn test_add_single_node() {
        let mut g = ComputationGraph::new();
        let id = g.add_node("input0", OpType::Input, vec![], 4);
        assert_eq!(id, 0);
        assert_eq!(g.len(), 1);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_add_multiple_nodes() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let c = g.add_node("add", OpType::Add, vec![a, b], 4);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn test_node_fields() {
        let mut g = ComputationGraph::new();
        g.add_node("my_input", OpType::Input, vec![], 8);
        let node = &g.nodes()[0];
        assert_eq!(node.name, "my_input");
        assert_eq!(node.op, OpType::Input);
        assert!(node.inputs.is_empty());
        assert_eq!(node.output_size, 8);
    }

    #[test]
    fn test_default_graph() {
        let g = ComputationGraph::default();
        assert!(g.is_empty());
    }

    // ── Topological sort ────────────────────────────────────────────────

    #[test]
    fn test_topo_sort_empty() {
        let g = ComputationGraph::new();
        assert_eq!(g.topological_sort(), Some(vec![]));
    }

    #[test]
    fn test_topo_sort_single_input() {
        let mut g = ComputationGraph::new();
        g.add_node("x", OpType::Input, vec![], 4);
        assert_eq!(g.topological_sort(), Some(vec![0]));
    }

    #[test]
    fn test_topo_sort_linear_chain() {
        let mut g = ComputationGraph::new();
        let x = g.add_node("x", OpType::Input, vec![], 4);
        let r = g.add_node("relu", OpType::ReLU, vec![x], 4);
        let s = g.add_node("softmax", OpType::Softmax, vec![r], 4);
        let order = g.topological_sort().unwrap();
        // x must come before relu, relu before softmax.
        let pos = |id: NodeId| order.iter().position(|&n| n == id).unwrap();
        assert!(pos(x) < pos(r));
        assert!(pos(r) < pos(s));
    }

    #[test]
    fn test_topo_sort_diamond() {
        let mut g = ComputationGraph::new();
        let x = g.add_node("x", OpType::Input, vec![], 4);
        let a = g.add_node("relu", OpType::ReLU, vec![x], 4);
        let b = g.add_node("softmax", OpType::Softmax, vec![x], 4);
        let c = g.add_node("add", OpType::Add, vec![a, b], 4);
        let order = g.topological_sort().unwrap();
        let pos = |id: NodeId| order.iter().position(|&n| n == id).unwrap();
        assert!(pos(x) < pos(a));
        assert!(pos(x) < pos(b));
        assert!(pos(a) < pos(c));
        assert!(pos(b) < pos(c));
    }

    #[test]
    fn test_topo_sort_two_independent_inputs() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let c = g.add_node("add", OpType::Add, vec![a, b], 4);
        let order = g.topological_sort().unwrap();
        let pos = |id: NodeId| order.iter().position(|&n| n == id).unwrap();
        assert!(pos(a) < pos(c));
        assert!(pos(b) < pos(c));
    }

    // ── Memory planning ─────────────────────────────────────────────────

    #[test]
    fn test_memory_plan_empty() {
        let g = ComputationGraph::new();
        let plan = g.memory_plan();
        assert_eq!(plan.total_bytes, 0);
        assert!(plan.offsets.is_empty());
    }

    #[test]
    fn test_memory_plan_single_node() {
        let mut g = ComputationGraph::new();
        g.add_node("x", OpType::Input, vec![], 8);
        let plan = g.memory_plan();
        assert_eq!(plan.total_bytes, 8 * 4); // 8 f32 × 4 bytes
        assert_eq!(*plan.offsets.get(&0).unwrap(), 0);
    }

    #[test]
    fn test_memory_plan_multiple_nodes() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 8);
        let _c = g.add_node("add", OpType::Add, vec![a, b], 4);
        let plan = g.memory_plan();
        assert_eq!(plan.total_bytes, (4 + 8 + 4) * 4);
        assert_eq!(*plan.offsets.get(&0).unwrap(), 0);
        assert_eq!(*plan.offsets.get(&1).unwrap(), 4 * 4);
        assert_eq!(*plan.offsets.get(&2).unwrap(), (4 + 8) * 4);
    }

    #[test]
    fn test_memory_plan_offsets_non_overlapping() {
        let mut g = ComputationGraph::new();
        for i in 0..5 {
            let size = (i + 1) * 4;
            g.add_node(format!("n{i}"), OpType::Input, vec![], size);
        }
        let plan = g.memory_plan();
        let mut sorted: Vec<_> = plan.offsets.iter().collect();
        sorted.sort_by_key(|&(_, &off)| off);
        for w in sorted.windows(2) {
            let (&id0, &off0) = w[0];
            let (_, &off1) = w[1];
            let sz = g.nodes()[id0].output_size * 4;
            assert!(off0 + sz <= off1, "buffers overlap");
        }
    }

    // ── Fusion detection ────────────────────────────────────────────────

    #[test]
    fn test_fusion_matmul_relu() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let mm = g.add_node("mm", OpType::MatMul { m: 2, k: 2, n: 2 }, vec![a, b], 4);
        let _r = g.add_node("relu", OpType::ReLU, vec![mm], 4);
        let fusions = g.detect_fusions();
        assert_eq!(fusions.len(), 1);
        assert!(fusions[0].description.contains("MatMul+ReLU"));
    }

    #[test]
    fn test_fusion_matmul_add() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let bias = g.add_node("bias", OpType::Input, vec![], 4);
        let mm = g.add_node("mm", OpType::MatMul { m: 2, k: 2, n: 2 }, vec![a, b], 4);
        let _add = g.add_node("add", OpType::Add, vec![mm, bias], 4);
        let fusions = g.detect_fusions();
        assert_eq!(fusions.len(), 1);
        assert!(fusions[0].description.contains("MatMul+Add"));
    }

    #[test]
    fn test_fusion_add_relu() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let add = g.add_node("add", OpType::Add, vec![a, b], 4);
        let _r = g.add_node("relu", OpType::ReLU, vec![add], 4);
        let fusions = g.detect_fusions();
        assert_eq!(fusions.len(), 1);
        assert!(fusions[0].description.contains("Add+ReLU"));
    }

    #[test]
    fn test_fusion_layernorm_add() {
        let mut g = ComputationGraph::new();
        let x = g.add_node("x", OpType::Input, vec![], 4);
        let gamma = g.add_node("gamma", OpType::Input, vec![], 4);
        let beta = g.add_node("beta", OpType::Input, vec![], 4);
        let res = g.add_node("residual", OpType::Input, vec![], 4);
        let ln = g.add_node("ln", OpType::LayerNorm { eps: 1e-5 }, vec![x, gamma, beta], 4);
        let _add = g.add_node("add", OpType::Add, vec![ln, res], 4);
        let fusions = g.detect_fusions();
        assert_eq!(fusions.len(), 1);
        assert!(fusions[0].description.contains("LayerNorm+Add"));
    }

    #[test]
    fn test_no_fusion_when_multi_consumer() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let add = g.add_node("add", OpType::Add, vec![a, b], 4);
        // Two consumers of add → no fusion.
        let _r1 = g.add_node("relu1", OpType::ReLU, vec![add], 4);
        let _r2 = g.add_node("relu2", OpType::ReLU, vec![add], 4);
        let fusions = g.detect_fusions();
        assert!(fusions.is_empty());
    }

    #[test]
    fn test_no_fusion_unrelated_ops() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let r = g.add_node("relu", OpType::ReLU, vec![a], 4);
        let _s = g.add_node("softmax", OpType::Softmax, vec![r], 4);
        let fusions = g.detect_fusions();
        assert!(fusions.is_empty());
    }

    #[test]
    fn test_fusion_nodes_field() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let add = g.add_node("add", OpType::Add, vec![a, b], 4);
        let relu = g.add_node("relu", OpType::ReLU, vec![add], 4);
        let fusions = g.detect_fusions();
        assert_eq!(fusions[0].nodes, vec![add, relu]);
    }

    // ── Executor: Add ───────────────────────────────────────────────────

    #[test]
    fn test_exec_add_basic() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let c = g.add_node("add", OpType::Add, vec![a, b], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0, 3.0, 4.0]);
        inputs.insert(b, vec![10.0, 20.0, 30.0, 40.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        approx_eq(&bufs[&c], &[11.0, 22.0, 33.0, 44.0], EPS);
    }

    #[test]
    fn test_exec_add_non_aligned() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 5);
        let b = g.add_node("b", OpType::Input, vec![], 5);
        let c = g.add_node("add", OpType::Add, vec![a, b], 5);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        inputs.insert(b, vec![0.5; 5]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        approx_eq(&bufs[&c], &[1.5, 2.5, 3.5, 4.5, 5.5], EPS);
    }

    // ── Executor: Mul ───────────────────────────────────────────────────

    #[test]
    fn test_exec_mul_basic() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let c = g.add_node("mul", OpType::Mul, vec![a, b], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![2.0, 3.0, 4.0, 5.0]);
        inputs.insert(b, vec![0.5, 0.5, 0.5, 0.5]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        approx_eq(&bufs[&c], &[1.0, 1.5, 2.0, 2.5], EPS);
    }

    #[test]
    fn test_exec_mul_zeros() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let c = g.add_node("mul", OpType::Mul, vec![a, b], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0, 3.0, 4.0]);
        inputs.insert(b, vec![0.0; 4]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        approx_eq(&bufs[&c], &[0.0; 4], EPS);
    }

    // ── Executor: MatMul ────────────────────────────────────────────────

    #[test]
    fn test_exec_matmul_identity() {
        // 2×2 identity × arbitrary = arbitrary
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let c = g.add_node("mm", OpType::MatMul { m: 2, k: 2, n: 2 }, vec![a, b], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 0.0, 0.0, 1.0]); // I₂
        inputs.insert(b, vec![5.0, 6.0, 7.0, 8.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        approx_eq(&bufs[&c], &[5.0, 6.0, 7.0, 8.0], EPS);
    }

    #[test]
    fn test_exec_matmul_2x3_3x2() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 6);
        let b = g.add_node("b", OpType::Input, vec![], 6);
        let c = g.add_node("mm", OpType::MatMul { m: 2, k: 3, n: 2 }, vec![a, b], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        // A = [[1,2,3],[4,5,6]]  B = [[7,8],[9,10],[11,12]]
        inputs.insert(a, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        inputs.insert(b, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        // C = [[58,64],[139,154]]
        approx_eq(&bufs[&c], &[58.0, 64.0, 139.0, 154.0], EPS);
    }

    #[test]
    fn test_exec_matmul_1x4_4x1() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let c = g.add_node("mm", OpType::MatMul { m: 1, k: 4, n: 1 }, vec![a, b], 1);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0, 3.0, 4.0]);
        inputs.insert(b, vec![1.0, 1.0, 1.0, 1.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        approx_eq(&bufs[&c], &[10.0], EPS);
    }

    // ── Executor: ReLU ──────────────────────────────────────────────────

    #[test]
    fn test_exec_relu_basic() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let r = g.add_node("relu", OpType::ReLU, vec![a], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![-1.0, 0.0, 1.0, 2.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        approx_eq(&bufs[&r], &[0.0, 0.0, 1.0, 2.0], EPS);
    }

    #[test]
    fn test_exec_relu_all_negative() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 8);
        let r = g.add_node("relu", OpType::ReLU, vec![a], 8);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, -0.1, -100.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        approx_eq(&bufs[&r], &[0.0; 8], EPS);
    }

    #[test]
    fn test_exec_relu_non_aligned() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 7);
        let r = g.add_node("relu", OpType::ReLU, vec![a], 7);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        approx_eq(&bufs[&r], &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0], EPS);
    }

    // ── Executor: Softmax ───────────────────────────────────────────────

    #[test]
    fn test_exec_softmax_uniform() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let s = g.add_node("sm", OpType::Softmax, vec![a], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 1.0, 1.0, 1.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        approx_eq(&bufs[&s], &[0.25, 0.25, 0.25, 0.25], EPS);
    }

    #[test]
    fn test_exec_softmax_sums_to_one() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 5);
        let s = g.add_node("sm", OpType::Softmax, vec![a], 5);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        let sum: f32 = bufs[&s].iter().sum();
        assert!((sum - 1.0).abs() < EPS, "softmax sum = {sum}");
    }

    #[test]
    fn test_exec_softmax_all_positive() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 3);
        let s = g.add_node("sm", OpType::Softmax, vec![a], 3);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![-10.0, 0.0, 10.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        for &v in &bufs[&s] {
            assert!(v >= 0.0, "softmax output must be non-negative");
        }
    }

    #[test]
    fn test_exec_softmax_monotonic() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let s = g.add_node("sm", OpType::Softmax, vec![a], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 2.0, 3.0, 4.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        let out = &bufs[&s];
        for w in out.windows(2) {
            assert!(w[0] <= w[1], "softmax should be monotonic for sorted input");
        }
    }

    // ── Executor: LayerNorm ─────────────────────────────────────────────

    #[test]
    fn test_exec_layer_norm_basic() {
        let mut g = ComputationGraph::new();
        let x = g.add_node("x", OpType::Input, vec![], 4);
        let gamma = g.add_node("gamma", OpType::Input, vec![], 4);
        let beta = g.add_node("beta", OpType::Input, vec![], 4);
        let ln = g.add_node("ln", OpType::LayerNorm { eps: 1e-5 }, vec![x, gamma, beta], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(x, vec![1.0, 2.0, 3.0, 4.0]);
        inputs.insert(gamma, vec![1.0; 4]);
        inputs.insert(beta, vec![0.0; 4]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        let out = &bufs[&ln];
        // With gamma=1, beta=0 the output should have ~zero mean.
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-4, "layernorm mean = {mean}");
    }

    #[test]
    fn test_exec_layer_norm_unit_variance() {
        let mut g = ComputationGraph::new();
        let x = g.add_node("x", OpType::Input, vec![], 8);
        let gamma = g.add_node("gamma", OpType::Input, vec![], 8);
        let beta = g.add_node("beta", OpType::Input, vec![], 8);
        let ln = g.add_node("ln", OpType::LayerNorm { eps: 1e-5 }, vec![x, gamma, beta], 8);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(x, vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
        inputs.insert(gamma, vec![1.0; 8]);
        inputs.insert(beta, vec![0.0; 8]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        let out = &bufs[&ln];
        let n = out.len() as f32;
        let mean: f32 = out.iter().sum::<f32>() / n;
        let var: f32 = out.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
        assert!((var - 1.0).abs() < 0.01, "layernorm variance = {var}");
    }

    #[test]
    fn test_exec_layer_norm_affine() {
        let mut g = ComputationGraph::new();
        let x = g.add_node("x", OpType::Input, vec![], 4);
        let gamma = g.add_node("gamma", OpType::Input, vec![], 4);
        let beta = g.add_node("beta", OpType::Input, vec![], 4);
        let ln = g.add_node("ln", OpType::LayerNorm { eps: 1e-5 }, vec![x, gamma, beta], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(x, vec![1.0, 2.0, 3.0, 4.0]);
        inputs.insert(gamma, vec![2.0; 4]);
        inputs.insert(beta, vec![1.0; 4]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        let out = &bufs[&ln];
        // Mean of output should be close to beta (1.0) since gamma scales around zero-mean.
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!((mean - 1.0).abs() < 0.01, "affine layernorm mean = {mean}");
    }

    // ── Executor: chain / pipeline ──────────────────────────────────────

    #[test]
    fn test_exec_add_then_relu() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let add = g.add_node("add", OpType::Add, vec![a, b], 4);
        let r = g.add_node("relu", OpType::ReLU, vec![add], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![-5.0, -1.0, 1.0, 5.0]);
        inputs.insert(b, vec![2.0, 2.0, 2.0, 2.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        approx_eq(&bufs[&r], &[0.0, 1.0, 3.0, 7.0], EPS);
    }

    #[test]
    fn test_exec_matmul_then_softmax() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let mm = g.add_node("mm", OpType::MatMul { m: 2, k: 2, n: 2 }, vec![a, b], 4);
        let s = g.add_node("softmax", OpType::Softmax, vec![mm], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, 0.0, 0.0, 1.0]);
        inputs.insert(b, vec![1.0, 2.0, 3.0, 4.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        let sum: f32 = bufs[&s].iter().sum();
        assert!((sum - 1.0).abs() < EPS, "softmax sum = {sum}");
    }

    // ── Executor: statistics ────────────────────────────────────────────

    #[test]
    fn test_exec_stats_nodes_executed() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let b = g.add_node("b", OpType::Input, vec![], 4);
        let _c = g.add_node("add", OpType::Add, vec![a, b], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0; 4]);
        inputs.insert(b, vec![1.0; 4]);
        let (_, stats) = exec.execute(&inputs).unwrap();
        assert_eq!(stats.nodes_executed, 3);
    }

    #[test]
    fn test_exec_stats_per_node() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let r = g.add_node("relu", OpType::ReLU, vec![a], 4);
        let _s = g.add_node("softmax", OpType::Softmax, vec![r], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0, -1.0, 2.0, -2.0]);
        let (_, stats) = exec.execute(&inputs).unwrap();
        assert_eq!(stats.node_stats.len(), 3);
        assert_eq!(stats.node_stats[0].node_name, "a");
        assert_eq!(stats.node_stats[1].node_name, "relu");
        assert_eq!(stats.node_stats[2].node_name, "softmax");
    }

    #[test]
    fn test_exec_stats_total_time() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 4);
        let _r = g.add_node("relu", OpType::ReLU, vec![a], 4);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0; 4]);
        let (_, stats) = exec.execute(&inputs).unwrap();
        assert!(stats.total_elapsed >= Duration::ZERO);
    }

    #[test]
    fn test_exec_stats_peak_memory() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 16);
        let _r = g.add_node("relu", OpType::ReLU, vec![a], 16);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0; 16]);
        let (_, stats) = exec.execute(&inputs).unwrap();
        assert!(stats.peak_memory_bytes > 0);
    }

    // ── Error handling ──────────────────────────────────────────────────

    #[test]
    fn test_exec_missing_input() {
        let mut g = ComputationGraph::new();
        g.add_node("a", OpType::Input, vec![], 4);
        let exec = GraphExecutor::new(g);
        let inputs = HashMap::new(); // no inputs provided
        let err = exec.execute(&inputs).unwrap_err();
        assert_eq!(err, GraphExecutorError::MissingInput(0));
    }

    #[test]
    fn test_error_display_cycle() {
        let err = GraphExecutorError::CycleDetected;
        assert_eq!(format!("{err}"), "cycle detected in computation graph");
    }

    #[test]
    fn test_error_display_missing_input() {
        let err = GraphExecutorError::MissingInput(42);
        assert_eq!(format!("{err}"), "missing input buffer for node 42");
    }

    // ── Op type equality ────────────────────────────────────────────────

    #[test]
    fn test_op_type_eq() {
        assert_eq!(OpType::Add, OpType::Add);
        assert_ne!(OpType::Add, OpType::Mul);
        assert_eq!(OpType::MatMul { m: 2, k: 3, n: 4 }, OpType::MatMul { m: 2, k: 3, n: 4 });
        assert_ne!(OpType::MatMul { m: 2, k: 3, n: 4 }, OpType::MatMul { m: 2, k: 3, n: 5 });
    }

    #[test]
    fn test_layer_norm_eps_eq() {
        assert_eq!(OpType::LayerNorm { eps: 1e-5 }, OpType::LayerNorm { eps: 1e-5 });
    }

    // ── Large vector (exercises NEON + remainder) ───────────────────────

    #[test]
    fn test_exec_add_large_vector() {
        let n = 1025; // not a multiple of 4
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], n);
        let b = g.add_node("b", OpType::Input, vec![], n);
        let c = g.add_node("add", OpType::Add, vec![a, b], n);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1.0; n]);
        inputs.insert(b, vec![2.0; n]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        assert_eq!(bufs[&c].len(), n);
        approx_eq(&bufs[&c], &vec![3.0; n], EPS);
    }

    #[test]
    fn test_exec_relu_large_vector() {
        let n = 1023;
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], n);
        let r = g.add_node("relu", OpType::ReLU, vec![a], n);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        let data: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        inputs.insert(a, data);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        for (i, &v) in bufs[&r].iter().enumerate() {
            if i % 2 == 0 {
                assert!((v - 1.0).abs() < EPS);
            } else {
                assert!((v).abs() < EPS);
            }
        }
    }

    // ── Graph accessor ──────────────────────────────────────────────────

    #[test]
    fn test_executor_graph_accessor() {
        let mut g = ComputationGraph::new();
        g.add_node("a", OpType::Input, vec![], 4);
        let exec = GraphExecutor::new(g);
        assert_eq!(exec.graph().len(), 1);
    }

    // ── Softmax numerical stability ─────────────────────────────────────

    #[test]
    fn test_softmax_large_values() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 3);
        let s = g.add_node("sm", OpType::Softmax, vec![a], 3);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![1000.0, 1001.0, 1002.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        let sum: f32 = bufs[&s].iter().sum();
        assert!((sum - 1.0).abs() < EPS, "softmax sum with large values = {sum}");
        for &v in &bufs[&s] {
            assert!(v.is_finite(), "softmax must be finite");
        }
    }

    #[test]
    fn test_softmax_negative_values() {
        let mut g = ComputationGraph::new();
        let a = g.add_node("a", OpType::Input, vec![], 3);
        let s = g.add_node("sm", OpType::Softmax, vec![a], 3);
        let exec = GraphExecutor::new(g);
        let mut inputs = HashMap::new();
        inputs.insert(a, vec![-1000.0, -999.0, -998.0]);
        let (bufs, _) = exec.execute(&inputs).unwrap();
        let sum: f32 = bufs[&s].iter().sum();
        assert!((sum - 1.0).abs() < EPS, "softmax sum with negative values = {sum}");
    }

    // ── Clone / Debug ───────────────────────────────────────────────────

    #[test]
    fn test_graph_clone() {
        let mut g = ComputationGraph::new();
        g.add_node("a", OpType::Input, vec![], 4);
        let g2 = g.clone();
        assert_eq!(g2.len(), 1);
    }

    #[test]
    fn test_graph_debug() {
        let g = ComputationGraph::new();
        let s = format!("{g:?}");
        assert!(s.contains("ComputationGraph"));
    }

    #[test]
    fn test_fusion_opportunity_debug() {
        let f = FusionOpportunity { nodes: vec![0, 1], description: "test".to_string() };
        let s = format!("{f:?}");
        assert!(s.contains("FusionOpportunity"));
    }
}
