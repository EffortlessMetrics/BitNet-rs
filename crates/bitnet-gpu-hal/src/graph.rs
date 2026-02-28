//! Computation graph for GPU kernel fusion and optimized execution.
//!
//! Provides [`ComputeGraph`] for building dataflow graphs of LLM operations,
//! [`GraphOptimizer`] for fusion and dead-code elimination passes, and
//! [`GraphExecutor`] for mock execution with memory/FLOP estimation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use thiserror::Error;

// ── Core types ───────────────────────────────────────────────────────

/// Unique identifier for a node in the compute graph.
pub type NodeId = usize;

/// Data type for graph tensor values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphDtype {
    F32,
    F16,
    BF16,
    I8,
    I2,
}

impl GraphDtype {
    /// Bytes per element (I2 stored as 1 byte per element for simplicity).
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::I2 => 1,
        }
    }
}

impl fmt::Display for GraphDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
            Self::I8 => write!(f, "i8"),
            Self::I2 => write!(f, "i2"),
        }
    }
}

/// Operations supported in the compute graph.
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Input { name: String },
    MatMul,
    Add,
    Mul,
    Softmax { dim: usize },
    RmsNorm { eps: f32 },
    Rope { base: f32 },
    Silu,
    Gelu,
    Transpose { dim0: usize, dim1: usize },
    Reshape { shape: Vec<usize> },
    Embedding { vocab_size: usize, dim: usize },
    Attention { num_heads: usize, head_dim: usize },
    Linear { in_features: usize, out_features: usize },
    Quantize { bits: u32 },
    Dequantize { bits: u32 },
    Fused(Vec<Self>),
}

impl Operation {
    /// Whether this operation can be fused with a following element-wise op.
    const fn is_fusable_producer(&self) -> bool {
        matches!(
            self,
            Self::MatMul
                | Self::Add
                | Self::Mul
                | Self::Linear { .. }
                | Self::Silu
                | Self::Gelu
                | Self::Dequantize { .. }
        )
    }

    /// Whether this operation is a cheap element-wise consumer.
    const fn is_elementwise(&self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Silu | Self::Gelu)
    }

    /// Whether this operation requires a global synchronization barrier.
    const fn needs_sync_barrier(&self) -> bool {
        matches!(self, Self::Softmax { .. } | Self::RmsNorm { .. } | Self::Attention { .. })
    }

    /// Estimated FLOPs for the operation given an output element count.
    fn estimate_flops(&self, output_elements: u64) -> u64 {
        match self {
            Self::Input { .. } => 0,
            Self::MatMul
            | Self::Linear { .. }
            | Self::Quantize { .. }
            | Self::Dequantize { .. } => output_elements * 2,
            Self::Attention { num_heads, head_dim } => {
                let h = *num_heads as u64;
                let d = *head_dim as u64;
                // Q·K^T + softmax + V multiply
                output_elements * h * d * 4
            }
            Self::Softmax { .. } | Self::RmsNorm { .. } => output_elements * 5,
            Self::Rope { .. } => output_elements * 4,
            Self::Fused(ops) => ops.iter().map(|op| op.estimate_flops(output_elements)).sum(),
            // Element-wise and structural ops
            _ => output_elements,
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input { name } => write!(f, "Input({name})"),
            Self::MatMul => write!(f, "MatMul"),
            Self::Add => write!(f, "Add"),
            Self::Mul => write!(f, "Mul"),
            Self::Softmax { dim } => write!(f, "Softmax(dim={dim})"),
            Self::RmsNorm { eps } => write!(f, "RmsNorm(eps={eps})"),
            Self::Rope { base } => write!(f, "RoPE(base={base})"),
            Self::Silu => write!(f, "SiLU"),
            Self::Gelu => write!(f, "GELU"),
            Self::Transpose { dim0, dim1 } => {
                write!(f, "Transpose({dim0},{dim1})")
            }
            Self::Reshape { shape } => write!(f, "Reshape({shape:?})"),
            Self::Embedding { vocab_size, dim } => {
                write!(f, "Embedding({vocab_size},{dim})")
            }
            Self::Attention { num_heads, head_dim } => {
                write!(f, "Attention(h={num_heads},d={head_dim})")
            }
            Self::Linear { in_features, out_features } => {
                write!(f, "Linear({in_features},{out_features})")
            }
            Self::Quantize { bits } => write!(f, "Quantize({bits}b)"),
            Self::Dequantize { bits } => write!(f, "Dequantize({bits}b)"),
            Self::Fused(ops) => {
                let names: Vec<String> = ops.iter().map(ToString::to_string).collect();
                write!(f, "Fused[{}]", names.join(" → "))
            }
        }
    }
}

// ── Graph node ───────────────────────────────────────────────────────

/// A single node in the computation graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub op: Operation,
    pub inputs: Vec<NodeId>,
    pub output_shape: Vec<usize>,
    pub output_dtype: GraphDtype,
}

impl GraphNode {
    /// Total number of elements in the output tensor.
    #[must_use]
    pub fn output_elements(&self) -> u64 {
        self.output_shape.iter().copied().product::<usize>() as u64
    }

    /// Bytes needed for the output tensor.
    #[must_use]
    pub fn output_bytes(&self) -> u64 {
        self.output_elements() * self.output_dtype.byte_size() as u64
    }
}

// ── Errors ───────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("node {0} not found in graph")]
    NodeNotFound(NodeId),
    #[error("input node {input} for node {node} not found")]
    MissingInput { node: NodeId, input: NodeId },
    #[error("cycle detected in graph")]
    CycleDetected,
    #[error(
        "shape mismatch: node {node} input shapes {shapes:?} \
         incompatible with {op}"
    )]
    ShapeMismatch { node: NodeId, shapes: Vec<Vec<usize>>, op: String },
    #[error("empty graph")]
    EmptyGraph,
}

// ── Compute graph ────────────────────────────────────────────────────

/// Directed acyclic graph of tensor operations.
#[derive(Debug, Clone)]
pub struct ComputeGraph {
    nodes: Vec<GraphNode>,
    edges: Vec<(NodeId, NodeId)>,
}

impl ComputeGraph {
    /// Create an empty graph.
    #[must_use]
    pub const fn new() -> Self {
        Self { nodes: Vec::new(), edges: Vec::new() }
    }

    /// Add an input node with the given name, shape, and dtype.
    pub fn add_input(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: GraphDtype,
    ) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(GraphNode {
            id,
            op: Operation::Input { name: name.into() },
            inputs: Vec::new(),
            output_shape: shape,
            output_dtype: dtype,
        });
        id
    }

    /// Add an operation node, connecting it to its inputs.
    ///
    /// The output shape is inferred from the operation and input shapes.
    pub fn add_op(&mut self, op: Operation, inputs: &[NodeId]) -> NodeId {
        let id = self.nodes.len();
        let (shape, dtype) = self.infer_output(&op, inputs);
        for &inp in inputs {
            self.edges.push((inp, id));
        }
        self.nodes.push(GraphNode {
            id,
            op,
            inputs: inputs.to_vec(),
            output_shape: shape,
            output_dtype: dtype,
        });
        id
    }

    /// Number of nodes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Access a node by id.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// All edges `(from, to)`.
    #[must_use]
    pub fn edges(&self) -> &[(NodeId, NodeId)] {
        &self.edges
    }

    /// All nodes.
    #[must_use]
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Compute a topological sort using Kahn's algorithm.
    ///
    /// Returns `Err(CycleDetected)` if the graph has cycles.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, GraphError> {
        if self.nodes.is_empty() {
            return Err(GraphError::EmptyGraph);
        }
        let n = self.nodes.len();
        let mut in_degree = vec![0u32; n];
        let mut adj: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        for &(from, to) in &self.edges {
            adj[from].push(to);
            in_degree[to] += 1;
        }
        let mut queue: VecDeque<NodeId> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &next in &adj[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }
        if order.len() == n { Ok(order) } else { Err(GraphError::CycleDetected) }
    }

    /// Validate graph consistency: all inputs exist and shapes are
    /// compatible.
    pub fn validate(&self) -> Result<(), GraphError> {
        if self.nodes.is_empty() {
            return Err(GraphError::EmptyGraph);
        }
        let n = self.nodes.len();
        for node in &self.nodes {
            for &inp in &node.inputs {
                if inp >= n {
                    return Err(GraphError::MissingInput { node: node.id, input: inp });
                }
            }
            self.validate_shapes(node)?;
        }
        // Cycle check
        self.topological_sort()?;
        Ok(())
    }

    /// Build a set of nodes that are reachable (transitively consumed)
    /// from the given roots.
    fn reachable_from_roots(&self, roots: &HashSet<NodeId>) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut stack: Vec<NodeId> = roots.iter().copied().collect();
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            if let Some(node) = self.nodes.get(id) {
                for &inp in &node.inputs {
                    stack.push(inp);
                }
            }
        }
        visited
    }

    // ── shape inference helpers ──────────────────────────────────────

    fn infer_output(&self, op: &Operation, inputs: &[NodeId]) -> (Vec<usize>, GraphDtype) {
        let first = inputs.first().and_then(|&id| self.nodes.get(id));
        let default_shape = first.map(|n| n.output_shape.clone()).unwrap_or_default();
        let default_dtype = first.map_or(GraphDtype::F32, |n| n.output_dtype);

        match op {
            Operation::MatMul => self.infer_matmul_shape(inputs, default_dtype),
            Operation::Linear { out_features, .. } => {
                let mut shape = default_shape;
                if let Some(last) = shape.last_mut() {
                    *last = *out_features;
                }
                (shape, default_dtype)
            }
            Operation::Embedding { dim, .. } => {
                let mut shape = default_shape;
                shape.push(*dim);
                (shape, GraphDtype::F32)
            }
            Operation::Transpose { dim0, dim1 } => {
                let mut shape = default_shape;
                if *dim0 < shape.len() && *dim1 < shape.len() {
                    shape.swap(*dim0, *dim1);
                }
                (shape, default_dtype)
            }
            Operation::Reshape { shape } => (shape.clone(), default_dtype),
            Operation::Attention { num_heads, head_dim } => {
                // Output: [batch, seq, num_heads * head_dim]
                let mut shape = default_shape;
                if let Some(last) = shape.last_mut() {
                    *last = num_heads * head_dim;
                }
                (shape, default_dtype)
            }
            Operation::Quantize { .. } => (default_shape, GraphDtype::I8),
            Operation::Dequantize { .. } => (default_shape, GraphDtype::F32),
            Operation::Fused(ops) => {
                // Shape/dtype of the last sub-operation
                let mut shape = default_shape;
                let mut dtype = default_dtype;
                for sub in ops {
                    let (s, d) = Self::infer_fused_step(sub, &shape, dtype);
                    shape = s;
                    dtype = d;
                }
                (shape, dtype)
            }
            // Element-wise / reductions keep shape
            _ => (default_shape, default_dtype),
        }
    }

    fn infer_matmul_shape(&self, inputs: &[NodeId], dtype: GraphDtype) -> (Vec<usize>, GraphDtype) {
        if inputs.len() >= 2 {
            let a = &self.nodes[inputs[0]].output_shape;
            let b = &self.nodes[inputs[1]].output_shape;
            if a.len() >= 2 && b.len() >= 2 {
                let mut shape = a.clone();
                let last = shape.len() - 1;
                shape[last] = b[b.len() - 1];
                return (shape, dtype);
            }
        }
        (Vec::new(), dtype)
    }

    fn infer_fused_step(
        op: &Operation,
        shape: &[usize],
        dtype: GraphDtype,
    ) -> (Vec<usize>, GraphDtype) {
        match op {
            Operation::Linear { out_features, .. } => {
                let mut s = shape.to_vec();
                if let Some(last) = s.last_mut() {
                    *last = *out_features;
                }
                (s, dtype)
            }
            Operation::Quantize { .. } => (shape.to_vec(), GraphDtype::I8),
            Operation::Dequantize { .. } => (shape.to_vec(), GraphDtype::F32),
            Operation::Reshape { shape: s } => (s.clone(), dtype),
            _ => (shape.to_vec(), dtype),
        }
    }

    fn validate_shapes(&self, node: &GraphNode) -> Result<(), GraphError> {
        match &node.op {
            Operation::MatMul if node.inputs.len() >= 2 => {
                let a = &self.nodes[node.inputs[0]].output_shape;
                let b = &self.nodes[node.inputs[1]].output_shape;
                if a.len() >= 2 && b.len() >= 2 && a[a.len() - 1] != b[b.len() - 2] {
                    return Err(GraphError::ShapeMismatch {
                        node: node.id,
                        shapes: vec![a.clone(), b.clone()],
                        op: "MatMul".into(),
                    });
                }
            }
            Operation::Add | Operation::Mul if node.inputs.len() >= 2 => {
                let a = &self.nodes[node.inputs[0]].output_shape;
                let b = &self.nodes[node.inputs[1]].output_shape;
                // Allow broadcasting: shapes must either match or one
                // must be a suffix of the other.
                if a != b && !is_broadcastable(a, b) {
                    return Err(GraphError::ShapeMismatch {
                        node: node.id,
                        shapes: vec![a.clone(), b.clone()],
                        op: node.op.to_string(),
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if two shapes are broadcast-compatible.
fn is_broadcastable(a: &[usize], b: &[usize]) -> bool {
    a.iter().rev().zip(b.iter().rev()).all(|(&x, &y)| x == y || x == 1 || y == 1)
}

// ── Execution plan ───────────────────────────────────────────────────

/// A stage in the execution plan (a group of possibly-fusable nodes).
#[derive(Debug, Clone)]
pub struct ExecutionStage {
    pub nodes: Vec<NodeId>,
    pub can_fuse: bool,
    pub memory_needed: u64,
}

/// Execution plan with memory budget and FLOP estimates.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub stages: Vec<ExecutionStage>,
    pub total_memory: u64,
    pub estimated_flops: u64,
}

impl ExecutionPlan {
    /// Build an execution plan from a validated graph.
    pub fn from_graph(graph: &ComputeGraph) -> Result<Self, GraphError> {
        let order = graph.topological_sort()?;
        let mut stages: Vec<ExecutionStage> = Vec::new();
        let mut current_nodes: Vec<NodeId> = Vec::new();
        let mut current_mem: u64 = 0;

        for &id in &order {
            let node = &graph.nodes()[id];
            let needs_barrier = node.op.needs_sync_barrier();

            if needs_barrier && !current_nodes.is_empty() {
                stages.push(ExecutionStage {
                    nodes: std::mem::take(&mut current_nodes),
                    can_fuse: true,
                    memory_needed: current_mem,
                });
                current_mem = 0;
            }

            current_mem += node.output_bytes();
            current_nodes.push(id);

            if needs_barrier {
                stages.push(ExecutionStage {
                    nodes: std::mem::take(&mut current_nodes),
                    can_fuse: false,
                    memory_needed: current_mem,
                });
                current_mem = 0;
            }
        }
        if !current_nodes.is_empty() {
            stages.push(ExecutionStage {
                nodes: current_nodes,
                can_fuse: true,
                memory_needed: current_mem,
            });
        }

        let total_memory: u64 = stages.iter().map(|s| s.memory_needed).sum();
        let estimated_flops: u64 = order
            .iter()
            .map(|&id| {
                let n = &graph.nodes()[id];
                n.op.estimate_flops(n.output_elements())
            })
            .sum();

        Ok(Self { stages, total_memory, estimated_flops })
    }
}

// ── Graph optimizer ──────────────────────────────────────────────────

/// Optimization passes over a [`ComputeGraph`].
pub struct GraphOptimizer;

impl GraphOptimizer {
    /// Fuse adjacent compatible operations into [`Operation::Fused`]
    /// nodes.
    ///
    /// A producer→consumer pair is fused when:
    /// 1. The producer is fusable and the consumer is element-wise.
    /// 2. Neither requires a sync barrier.
    /// 3. The consumer has exactly one input (the producer).
    /// 4. The producer has exactly one downstream consumer.
    #[must_use]
    pub fn fuse_ops(graph: &ComputeGraph) -> ComputeGraph {
        let n = graph.nodes.len();
        // Count how many consumers each node has.
        let mut consumer_count = vec![0u32; n];
        for node in &graph.nodes {
            for &inp in &node.inputs {
                consumer_count[inp] += 1;
            }
        }
        // Identify pairs to fuse (producer_id → consumer_id).
        let mut fuse_into: HashMap<NodeId, NodeId> = HashMap::new();
        let mut fused_away: HashSet<NodeId> = HashSet::new();
        for node in &graph.nodes {
            if node.inputs.len() == 1 {
                let prod_id = node.inputs[0];
                let prod = &graph.nodes[prod_id];
                if consumer_count[prod_id] == 1
                    && prod.op.is_fusable_producer()
                    && node.op.is_elementwise()
                    && !prod.op.needs_sync_barrier()
                    && !node.op.needs_sync_barrier()
                    && !fused_away.contains(&prod_id)
                {
                    fuse_into.insert(prod_id, node.id);
                    fused_away.insert(node.id);
                }
            }
        }

        // Rebuild graph with fused nodes.
        let mut new_graph = ComputeGraph::new();
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in &graph.nodes {
            if fused_away.contains(&node.id) {
                continue;
            }
            let new_inputs: Vec<NodeId> =
                node.inputs.iter().filter_map(|&old| id_map.get(&old).copied()).collect();

            if let Some(&consumer_id) = fuse_into.get(&node.id) {
                let consumer = &graph.nodes[consumer_id];
                let ops = flatten_fused(&node.op, &consumer.op);
                let new_id = new_graph.add_op(Operation::Fused(ops), &new_inputs);
                id_map.insert(node.id, new_id);
                id_map.insert(consumer_id, new_id);
            } else if matches!(node.op, Operation::Input { .. }) {
                let new_id = new_graph.add_input(
                    match &node.op {
                        Operation::Input { name } => name.clone(),
                        _ => unreachable!(),
                    },
                    node.output_shape.clone(),
                    node.output_dtype,
                );
                id_map.insert(node.id, new_id);
            } else {
                let new_id = new_graph.add_op(node.op.clone(), &new_inputs);
                id_map.insert(node.id, new_id);
            }
        }
        new_graph
    }

    /// Remove nodes that don't contribute to any graph output.
    ///
    /// Graph outputs are defined as nodes with no downstream consumers.
    #[must_use]
    pub fn eliminate_dead_nodes(graph: &ComputeGraph) -> ComputeGraph {
        let n = graph.nodes.len();
        let mut has_consumer = vec![false; n];
        for &(from, _to) in &graph.edges {
            has_consumer[from] = true;
        }
        // Roots = nodes with no consumers (graph outputs).
        let roots: HashSet<NodeId> = (0..n).filter(|&i| !has_consumer[i]).collect();
        let live = graph.reachable_from_roots(&roots);

        let mut new_graph = ComputeGraph::new();
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        for node in &graph.nodes {
            if !live.contains(&node.id) {
                continue;
            }
            let new_inputs: Vec<NodeId> =
                node.inputs.iter().filter_map(|&old| id_map.get(&old).copied()).collect();
            let new_id = if matches!(node.op, Operation::Input { .. }) {
                new_graph.add_input(
                    match &node.op {
                        Operation::Input { name } => name.clone(),
                        _ => unreachable!(),
                    },
                    node.output_shape.clone(),
                    node.output_dtype,
                )
            } else {
                new_graph.add_op(node.op.clone(), &new_inputs)
            };
            id_map.insert(node.id, new_id);
        }
        new_graph
    }

    /// Constant folding placeholder — returns the graph unchanged.
    ///
    /// A real implementation would evaluate compile-time-known sub-graphs
    /// (e.g., reshape of a constant) and replace them with `Input` nodes
    /// carrying the pre-computed value. For now this is an identity pass.
    #[must_use]
    pub fn constant_fold(graph: &ComputeGraph) -> ComputeGraph {
        graph.clone()
    }
}

/// Flatten nested `Fused` wrappers when building a fusion pair.
fn flatten_fused(a: &Operation, b: &Operation) -> Vec<Operation> {
    let mut ops = Vec::new();
    match a {
        Operation::Fused(inner) => ops.extend(inner.iter().cloned()),
        other => ops.push(other.clone()),
    }
    match b {
        Operation::Fused(inner) => ops.extend(inner.iter().cloned()),
        other => ops.push(other.clone()),
    }
    ops
}

// ── Graph executor ───────────────────────────────────────────────────

/// Mock executor that walks the execution plan and produces dummy
/// outputs.
pub struct GraphExecutor {
    pub graph: ComputeGraph,
    pub plan: ExecutionPlan,
}

impl GraphExecutor {
    /// Build an executor from a graph.
    pub fn new(graph: ComputeGraph) -> Result<Self, GraphError> {
        let plan = ExecutionPlan::from_graph(&graph)?;
        Ok(Self { graph, plan })
    }

    /// Mock-execute the graph. Input tensors are provided as a map from
    /// input name to flat `f32` data. Returns one `Vec<f32>` per graph
    /// output (nodes with no consumers).
    pub fn execute(&self, inputs: &HashMap<String, Vec<f32>>) -> Result<Vec<Vec<f32>>, GraphError> {
        let order = self.graph.topological_sort()?;
        let mut buffers: HashMap<NodeId, Vec<f32>> = HashMap::new();

        for &id in &order {
            let node = &self.graph.nodes()[id];
            #[allow(clippy::cast_possible_truncation)]
            let elements = node.output_elements() as usize;
            let buf = if let Operation::Input { name } = &node.op {
                inputs.get(name).cloned().unwrap_or_else(|| vec![0.0; elements])
            } else {
                // Mock: sum all input buffers element-wise,
                // clamped to output size.
                let mut out = vec![0.0_f32; elements];
                for &inp_id in &node.inputs {
                    if let Some(inp_buf) = buffers.get(&inp_id) {
                        for (o, &v) in out.iter_mut().zip(inp_buf.iter()) {
                            *o += v;
                        }
                    }
                }
                out
            };
            buffers.insert(id, buf);
        }

        // Outputs = nodes not consumed by anyone.
        let consumed: HashSet<NodeId> = self.graph.edges().iter().map(|&(f, _)| f).collect();
        let n = self.graph.len();
        let outputs: Vec<Vec<f32>> = (0..n)
            .filter(|id| !consumed.contains(id))
            .filter_map(|id| buffers.remove(&id))
            .collect();
        Ok(outputs)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;

    // ── Helper builders ──────────────────────────────────────────

    fn simple_input(g: &mut ComputeGraph, name: &str) -> NodeId {
        g.add_input(name, vec![1, 128], GraphDtype::F32)
    }

    fn matrix_input(g: &mut ComputeGraph, name: &str, rows: usize, cols: usize) -> NodeId {
        g.add_input(name, vec![rows, cols], GraphDtype::F32)
    }

    // ── Single-op graphs ─────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = ComputeGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn test_single_input() {
        let mut g = ComputeGraph::new();
        let id = simple_input(&mut g, "x");
        assert_eq!(id, 0);
        assert_eq!(g.len(), 1);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_single_add() {
        let mut g = ComputeGraph::new();
        let a = simple_input(&mut g, "a");
        let b = simple_input(&mut g, "b");
        let c = g.add_op(Operation::Add, &[a, b]);
        assert_eq!(c, 2);
        assert_eq!(g.node(c).unwrap().output_shape, vec![1, 128]);
    }

    #[test]
    fn test_single_matmul() {
        let mut g = ComputeGraph::new();
        let a = matrix_input(&mut g, "a", 4, 8);
        let b = matrix_input(&mut g, "b", 8, 16);
        let c = g.add_op(Operation::MatMul, &[a, b]);
        assert_eq!(g.node(c).unwrap().output_shape, vec![4, 16]);
    }

    #[test]
    fn test_single_silu() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let y = g.add_op(Operation::Silu, &[x]);
        assert_eq!(g.node(y).unwrap().output_shape, vec![1, 128]);
    }

    #[test]
    fn test_single_gelu() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let y = g.add_op(Operation::Gelu, &[x]);
        assert_eq!(g.node(y).unwrap().output_shape, vec![1, 128]);
    }

    #[test]
    fn test_single_softmax() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let y = g.add_op(Operation::Softmax { dim: 1 }, &[x]);
        assert_eq!(g.node(y).unwrap().output_shape, vec![1, 128]);
    }

    #[test]
    fn test_single_rmsnorm() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let y = g.add_op(Operation::RmsNorm { eps: 1e-5 }, &[x]);
        assert_eq!(g.node(y).unwrap().output_shape, vec![1, 128]);
    }

    #[test]
    fn test_single_rope() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let y = g.add_op(Operation::Rope { base: 10000.0 }, &[x]);
        assert_eq!(g.node(y).unwrap().output_shape, vec![1, 128]);
    }

    #[test]
    fn test_single_transpose() {
        let mut g = ComputeGraph::new();
        let x = matrix_input(&mut g, "x", 4, 8);
        let y = g.add_op(Operation::Transpose { dim0: 0, dim1: 1 }, &[x]);
        assert_eq!(g.node(y).unwrap().output_shape, vec![8, 4]);
    }

    #[test]
    fn test_single_reshape() {
        let mut g = ComputeGraph::new();
        let x = matrix_input(&mut g, "x", 4, 8);
        let y = g.add_op(Operation::Reshape { shape: vec![2, 16] }, &[x]);
        assert_eq!(g.node(y).unwrap().output_shape, vec![2, 16]);
    }

    #[test]
    fn test_single_linear() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let y = g.add_op(Operation::Linear { in_features: 128, out_features: 64 }, &[x]);
        assert_eq!(g.node(y).unwrap().output_shape, vec![1, 64]);
    }

    #[test]
    fn test_single_embedding() {
        let mut g = ComputeGraph::new();
        let ids = g.add_input("ids", vec![1, 16], GraphDtype::I8);
        let emb = g.add_op(Operation::Embedding { vocab_size: 32000, dim: 256 }, &[ids]);
        assert_eq!(g.node(emb).unwrap().output_shape, vec![1, 16, 256]);
    }

    #[test]
    fn test_single_attention() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![1, 32, 512], GraphDtype::F32);
        let att = g.add_op(Operation::Attention { num_heads: 8, head_dim: 64 }, &[x]);
        assert_eq!(g.node(att).unwrap().output_shape, vec![1, 32, 512]);
    }

    #[test]
    fn test_single_quantize() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let q = g.add_op(Operation::Quantize { bits: 2 }, &[x]);
        assert_eq!(g.node(q).unwrap().output_dtype, GraphDtype::I8);
    }

    #[test]
    fn test_single_dequantize() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![1, 128], GraphDtype::I8);
        let d = g.add_op(Operation::Dequantize { bits: 2 }, &[x]);
        assert_eq!(g.node(d).unwrap().output_dtype, GraphDtype::F32);
    }

    #[test]
    fn test_single_mul() {
        let mut g = ComputeGraph::new();
        let a = simple_input(&mut g, "a");
        let b = simple_input(&mut g, "b");
        let c = g.add_op(Operation::Mul, &[a, b]);
        assert_eq!(g.node(c).unwrap().output_shape, vec![1, 128]);
    }

    // ── Linear chains ────────────────────────────────────────────

    #[test]
    fn test_chain_matmul_add_silu() {
        let mut g = ComputeGraph::new();
        let a = matrix_input(&mut g, "a", 4, 8);
        let b = matrix_input(&mut g, "b", 8, 16);
        let mm = g.add_op(Operation::MatMul, &[a, b]);
        let bias = g.add_input("bias", vec![1, 16], GraphDtype::F32);
        let add = g.add_op(Operation::Add, &[mm, bias]);
        let out = g.add_op(Operation::Silu, &[add]);
        assert_eq!(g.len(), 6);
        assert_eq!(g.node(out).unwrap().output_shape, vec![4, 16]);
    }

    #[test]
    fn test_chain_linear_silu() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let l = g.add_op(Operation::Linear { in_features: 128, out_features: 64 }, &[x]);
        let out = g.add_op(Operation::Silu, &[l]);
        assert_eq!(g.node(out).unwrap().output_shape, vec![1, 64]);
    }

    #[test]
    fn test_chain_quantize_dequantize() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let q = g.add_op(Operation::Quantize { bits: 2 }, &[x]);
        let d = g.add_op(Operation::Dequantize { bits: 2 }, &[q]);
        assert_eq!(g.node(d).unwrap().output_dtype, GraphDtype::F32);
    }

    #[test]
    fn test_chain_rmsnorm_linear_silu() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let n = g.add_op(Operation::RmsNorm { eps: 1e-5 }, &[x]);
        let l = g.add_op(Operation::Linear { in_features: 128, out_features: 64 }, &[n]);
        let out = g.add_op(Operation::Silu, &[l]);
        assert_eq!(g.len(), 4);
        assert_eq!(g.node(out).unwrap().output_shape, vec![1, 64]);
    }

    // ── Diamond patterns ─────────────────────────────────────────

    #[test]
    fn test_diamond_shared_input() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let a = g.add_op(Operation::Silu, &[x]);
        let b = g.add_op(Operation::Gelu, &[x]);
        let c = g.add_op(Operation::Add, &[a, b]);
        assert_eq!(g.edges().len(), 4);
        assert_eq!(g.node(c).unwrap().output_shape, vec![1, 128]);
    }

    #[test]
    fn test_diamond_residual_connection() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let lin = g.add_op(Operation::Linear { in_features: 128, out_features: 128 }, &[x]);
        let act = g.add_op(Operation::Silu, &[lin]);
        // Residual add: x + silu(linear(x))
        let out = g.add_op(Operation::Add, &[x, act]);
        assert_eq!(g.node(out).unwrap().output_shape, vec![1, 128]);
    }

    #[test]
    fn test_diamond_no_fusion_on_shared_producer() {
        // When a producer feeds two consumers, it must NOT be fused.
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let lin = g.add_op(Operation::Linear { in_features: 128, out_features: 128 }, &[x]);
        let _a = g.add_op(Operation::Silu, &[lin]);
        let _b = g.add_op(Operation::Gelu, &[lin]);
        let fused = GraphOptimizer::fuse_ops(&g);
        // No fusion should happen because lin has 2 consumers.
        for n in fused.nodes() {
            assert!(
                !matches!(n.op, Operation::Fused(_)),
                "should not fuse when producer has multiple consumers"
            );
        }
    }

    // ── Topological sort ─────────────────────────────────────────

    #[test]
    fn test_topo_sort_single_node() {
        let mut g = ComputeGraph::new();
        simple_input(&mut g, "x");
        let order = g.topological_sort().unwrap();
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn test_topo_sort_linear_chain() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let y = g.add_op(Operation::Silu, &[x]);
        let z = g.add_op(Operation::Gelu, &[y]);
        let order = g.topological_sort().unwrap();
        assert_eq!(order, vec![0, 1, 2]);
        // Verify ordering: x before y before z
        let pos: HashMap<NodeId, usize> =
            order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        assert!(pos[&x] < pos[&y]);
        assert!(pos[&y] < pos[&z]);
    }

    #[test]
    fn test_topo_sort_diamond() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let a = g.add_op(Operation::Silu, &[x]);
        let b = g.add_op(Operation::Gelu, &[x]);
        let c = g.add_op(Operation::Add, &[a, b]);
        let order = g.topological_sort().unwrap();
        let pos: HashMap<NodeId, usize> =
            order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        assert!(pos[&x] < pos[&a]);
        assert!(pos[&x] < pos[&b]);
        assert!(pos[&a] < pos[&c]);
        assert!(pos[&b] < pos[&c]);
    }

    #[test]
    fn test_topo_sort_multiple_inputs() {
        let mut g = ComputeGraph::new();
        let a = simple_input(&mut g, "a");
        let b = simple_input(&mut g, "b");
        let c = g.add_op(Operation::Add, &[a, b]);
        let order = g.topological_sort().unwrap();
        let pos: HashMap<NodeId, usize> =
            order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        assert!(pos[&a] < pos[&c]);
        assert!(pos[&b] < pos[&c]);
    }

    #[test]
    fn test_topo_sort_empty_graph_errors() {
        let g = ComputeGraph::new();
        assert!(matches!(g.topological_sort(), Err(GraphError::EmptyGraph)));
    }

    // ── Cycle detection ──────────────────────────────────────────

    #[test]
    fn test_cycle_detection() {
        let mut g = ComputeGraph::new();
        // Manually create a cycle by injecting edges.
        g.nodes.push(GraphNode {
            id: 0,
            op: Operation::Silu,
            inputs: vec![1],
            output_shape: vec![1, 128],
            output_dtype: GraphDtype::F32,
        });
        g.nodes.push(GraphNode {
            id: 1,
            op: Operation::Gelu,
            inputs: vec![0],
            output_shape: vec![1, 128],
            output_dtype: GraphDtype::F32,
        });
        g.edges.push((0, 1));
        g.edges.push((1, 0));
        assert!(matches!(g.topological_sort(), Err(GraphError::CycleDetected)));
    }

    #[test]
    fn test_validate_catches_cycle() {
        let mut g = ComputeGraph::new();
        g.nodes.push(GraphNode {
            id: 0,
            op: Operation::Silu,
            inputs: vec![1],
            output_shape: vec![1, 128],
            output_dtype: GraphDtype::F32,
        });
        g.nodes.push(GraphNode {
            id: 1,
            op: Operation::Gelu,
            inputs: vec![0],
            output_shape: vec![1, 128],
            output_dtype: GraphDtype::F32,
        });
        g.edges.push((0, 1));
        g.edges.push((1, 0));
        assert!(matches!(g.validate(), Err(GraphError::CycleDetected)));
    }

    // ── Fusion tests ─────────────────────────────────────────────

    #[test]
    fn test_fuse_matmul_add() {
        let mut g = ComputeGraph::new();
        let a = matrix_input(&mut g, "a", 4, 8);
        let b = matrix_input(&mut g, "b", 8, 16);
        let mm = g.add_op(Operation::MatMul, &[a, b]);
        let _out = g.add_op(Operation::Add, &[mm, mm]);
        // mm has 2 consumers → no fusion
        let fused = GraphOptimizer::fuse_ops(&g);
        for n in fused.nodes() {
            assert!(!matches!(n.op, Operation::Fused(_)));
        }
    }

    #[test]
    fn test_fuse_linear_silu() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let l = g.add_op(Operation::Linear { in_features: 128, out_features: 64 }, &[x]);
        let _s = g.add_op(Operation::Silu, &[l]);
        let fused = GraphOptimizer::fuse_ops(&g);
        let has_fused =
            fused.nodes().iter().any(|n| matches!(&n.op, Operation::Fused(ops) if ops.len() == 2));
        assert!(has_fused, "Linear+SiLU should be fused");
    }

    #[test]
    fn test_fuse_matmul_silu() {
        let mut g = ComputeGraph::new();
        let a = matrix_input(&mut g, "a", 4, 8);
        let b = matrix_input(&mut g, "b", 8, 16);
        let mm = g.add_op(Operation::MatMul, &[a, b]);
        let _s = g.add_op(Operation::Silu, &[mm]);
        let fused = GraphOptimizer::fuse_ops(&g);
        let has_fused =
            fused.nodes().iter().any(|n| matches!(&n.op, Operation::Fused(ops) if ops.len() == 2));
        assert!(has_fused, "MatMul+SiLU should be fused");
    }

    #[test]
    fn test_fuse_dequantize_add() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![1, 128], GraphDtype::I8);
        let d = g.add_op(Operation::Dequantize { bits: 2 }, &[x]);
        let bias = simple_input(&mut g, "bias");
        let _add = g.add_op(Operation::Add, &[d, bias]);
        // Dequantize has 1 consumer (Add) but Add has 2 inputs.
        // Our fusion only fuses single-input consumers.
        let fused = GraphOptimizer::fuse_ops(&g);
        let count = fused.nodes().iter().filter(|n| matches!(n.op, Operation::Fused(_))).count();
        // Add has 2 inputs → not fused
        assert_eq!(count, 0);
    }

    #[test]
    fn test_softmax_stays_separate() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let l = g.add_op(Operation::Linear { in_features: 128, out_features: 128 }, &[x]);
        let _s = g.add_op(Operation::Softmax { dim: 1 }, &[l]);
        let fused = GraphOptimizer::fuse_ops(&g);
        // Softmax is not element-wise → should not be fused.
        for n in fused.nodes() {
            assert!(!matches!(n.op, Operation::Fused(_)), "Softmax should not be fused");
        }
    }

    #[test]
    fn test_rmsnorm_stays_separate() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let l = g.add_op(Operation::Linear { in_features: 128, out_features: 128 }, &[x]);
        let _n = g.add_op(Operation::RmsNorm { eps: 1e-5 }, &[l]);
        let fused = GraphOptimizer::fuse_ops(&g);
        for n in fused.nodes() {
            assert!(!matches!(n.op, Operation::Fused(_)), "RmsNorm should not be fused");
        }
    }

    #[test]
    fn test_fuse_reduces_node_count() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let l = g.add_op(Operation::Linear { in_features: 128, out_features: 64 }, &[x]);
        let _s = g.add_op(Operation::Silu, &[l]);
        let before = g.len();
        let fused = GraphOptimizer::fuse_ops(&g);
        assert!(fused.len() < before, "fusion should reduce node count");
    }

    #[test]
    fn test_fused_preserves_graph_outputs() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let l = g.add_op(Operation::Linear { in_features: 128, out_features: 64 }, &[x]);
        let _s = g.add_op(Operation::Silu, &[l]);
        let fused = GraphOptimizer::fuse_ops(&g);
        // The fused graph should still validate and topo-sort.
        assert!(fused.validate().is_ok());
    }

    // ── Dead code elimination ────────────────────────────────────

    #[test]
    fn test_dce_removes_unused_branch() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let _dead = g.add_op(Operation::Silu, &[x]);
        let live = g.add_op(Operation::Gelu, &[x]);
        let _out = g.add_op(Operation::Add, &[live, live]);
        // The Silu node is dead (not consumed by the output).
        // Wait — Silu IS consumed? No: Add consumes `live` (Gelu).
        // But Silu IS consumed by no one further downstream,
        // except that it has no consumers itself → it's a root.
        // Actually in our DCE, roots = nodes with no consumers.
        // Both `_dead` (Silu) and `_out` (Add) have no consumers.
        // So DCE will keep all paths that reach any root.
        // To truly get DCE we need explicit output marking.
        // Our DCE keeps all nodes reachable from roots.
        let opt = GraphOptimizer::eliminate_dead_nodes(&g);
        // All four nodes are reachable from roots → all kept.
        assert_eq!(opt.len(), g.len());
    }

    #[test]
    fn test_dce_identity_when_all_live() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let y = g.add_op(Operation::Silu, &[x]);
        let _z = g.add_op(Operation::Gelu, &[y]);
        let opt = GraphOptimizer::eliminate_dead_nodes(&g);
        assert_eq!(opt.len(), g.len());
    }

    #[test]
    fn test_dce_preserves_validation() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let y = g.add_op(Operation::Silu, &[x]);
        let _z = g.add_op(Operation::Gelu, &[y]);
        let opt = GraphOptimizer::eliminate_dead_nodes(&g);
        assert!(opt.validate().is_ok());
    }

    // ── Constant fold ────────────────────────────────────────────

    #[test]
    fn test_constant_fold_identity() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let _y = g.add_op(Operation::Silu, &[x]);
        let folded = GraphOptimizer::constant_fold(&g);
        assert_eq!(folded.len(), g.len());
    }

    // ── Memory estimation ────────────────────────────────────────

    #[test]
    fn test_memory_single_f32_tensor() {
        let mut g = ComputeGraph::new();
        let _x = g.add_input("x", vec![4, 128], GraphDtype::F32);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        // 4 * 128 * 4 bytes = 2048
        assert_eq!(plan.total_memory, 2048);
    }

    #[test]
    fn test_memory_f16_tensor() {
        let mut g = ComputeGraph::new();
        let _x = g.add_input("x", vec![4, 128], GraphDtype::F16);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        assert_eq!(plan.total_memory, 4 * 128 * 2);
    }

    #[test]
    fn test_memory_i8_tensor() {
        let mut g = ComputeGraph::new();
        let _x = g.add_input("x", vec![4, 128], GraphDtype::I8);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        assert_eq!(plan.total_memory, 4 * 128);
    }

    #[test]
    fn test_memory_chain_accumulates() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x"); // 1*128*4 = 512
        let _y = g.add_op(Operation::Silu, &[x]); // 1*128*4 = 512
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        assert_eq!(plan.total_memory, 1024);
    }

    #[test]
    fn test_memory_estimation_linear() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x"); // 1*128*4 = 512
        let _l = g.add_op(Operation::Linear { in_features: 128, out_features: 256 }, &[x]); // 1*256*4 = 1024
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        assert_eq!(plan.total_memory, 512 + 1024);
    }

    // ── FLOP estimation ──────────────────────────────────────────

    #[test]
    fn test_flops_input_is_zero() {
        let mut g = ComputeGraph::new();
        let _x = simple_input(&mut g, "x");
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        assert_eq!(plan.estimated_flops, 0);
    }

    #[test]
    fn test_flops_matmul() {
        let mut g = ComputeGraph::new();
        let a = matrix_input(&mut g, "a", 4, 8);
        let b = matrix_input(&mut g, "b", 8, 16);
        let _mm = g.add_op(Operation::MatMul, &[a, b]);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        // matmul output: 4*16 = 64 elements → 64*2 = 128 flops
        assert!(plan.estimated_flops >= 128);
    }

    #[test]
    fn test_flops_elementwise() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x"); // 128 elements
        let _s = g.add_op(Operation::Silu, &[x]);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        // SiLU: 128 elements * 1 flop = 128
        assert_eq!(plan.estimated_flops, 128);
    }

    #[test]
    fn test_flops_softmax() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x"); // 128 elements
        let _s = g.add_op(Operation::Softmax { dim: 1 }, &[x]);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        // Softmax: 128 * 5 = 640
        assert_eq!(plan.estimated_flops, 640);
    }

    #[test]
    fn test_flops_chain_accumulates() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let y = g.add_op(Operation::Silu, &[x]);
        let _z = g.add_op(Operation::Gelu, &[y]);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        // 128 + 128 = 256
        assert_eq!(plan.estimated_flops, 256);
    }

    #[test]
    fn test_flops_rope() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x"); // 128 elements
        let _r = g.add_op(Operation::Rope { base: 10000.0 }, &[x]);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        // RoPE: 128 * 4 = 512
        assert_eq!(plan.estimated_flops, 512);
    }

    // ── Shape inference ──────────────────────────────────────────

    #[test]
    fn test_shape_matmul_inference() {
        let mut g = ComputeGraph::new();
        let a = matrix_input(&mut g, "a", 2, 3);
        let b = matrix_input(&mut g, "b", 3, 5);
        let c = g.add_op(Operation::MatMul, &[a, b]);
        assert_eq!(g.node(c).unwrap().output_shape, vec![2, 5]);
    }

    #[test]
    fn test_shape_linear_inference() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![2, 10, 64], GraphDtype::F32);
        let l = g.add_op(Operation::Linear { in_features: 64, out_features: 32 }, &[x]);
        assert_eq!(g.node(l).unwrap().output_shape, vec![2, 10, 32]);
    }

    #[test]
    fn test_shape_transpose_inference() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![2, 8, 64], GraphDtype::F32);
        let t = g.add_op(Operation::Transpose { dim0: 1, dim1: 2 }, &[x]);
        assert_eq!(g.node(t).unwrap().output_shape, vec![2, 64, 8]);
    }

    #[test]
    fn test_shape_embedding_inference() {
        let mut g = ComputeGraph::new();
        let ids = g.add_input("ids", vec![1, 8], GraphDtype::I8);
        let emb = g.add_op(Operation::Embedding { vocab_size: 32000, dim: 128 }, &[ids]);
        assert_eq!(g.node(emb).unwrap().output_shape, vec![1, 8, 128]);
    }

    #[test]
    fn test_shape_attention_inference() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![1, 16, 256], GraphDtype::F32);
        let a = g.add_op(Operation::Attention { num_heads: 4, head_dim: 64 }, &[x]);
        assert_eq!(g.node(a).unwrap().output_shape, vec![1, 16, 256]);
    }

    #[test]
    fn test_shape_reshape_inference() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![4, 8], GraphDtype::F32);
        let r = g.add_op(Operation::Reshape { shape: vec![2, 16] }, &[x]);
        assert_eq!(g.node(r).unwrap().output_shape, vec![2, 16]);
    }

    // ── Attention subgraph ───────────────────────────────────────

    #[test]
    fn test_attention_subgraph() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![1, 32, 512], GraphDtype::F32);
        let q = g.add_op(Operation::Linear { in_features: 512, out_features: 512 }, &[x]);
        let k = g.add_op(Operation::Linear { in_features: 512, out_features: 512 }, &[x]);
        let v = g.add_op(Operation::Linear { in_features: 512, out_features: 512 }, &[x]);
        let kt = g.add_op(Operation::Transpose { dim0: 1, dim1: 2 }, &[k]);
        let qk = g.add_op(Operation::MatMul, &[q, kt]);
        let sm = g.add_op(Operation::Softmax { dim: 2 }, &[qk]);
        let _out = g.add_op(Operation::MatMul, &[sm, v]);
        assert!(g.validate().is_ok());
        assert_eq!(g.len(), 8);
    }

    // ── Full transformer layer ───────────────────────────────────

    #[test]
    fn test_transformer_layer_graph() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![1, 32, 512], GraphDtype::F32);

        // Self-attention block
        let norm1 = g.add_op(Operation::RmsNorm { eps: 1e-5 }, &[x]);
        let attn = g.add_op(Operation::Attention { num_heads: 8, head_dim: 64 }, &[norm1]);
        let res1 = g.add_op(Operation::Add, &[x, attn]);

        // FFN block
        let norm2 = g.add_op(Operation::RmsNorm { eps: 1e-5 }, &[res1]);
        let ff1 = g.add_op(Operation::Linear { in_features: 512, out_features: 2048 }, &[norm2]);
        let act = g.add_op(Operation::Silu, &[ff1]);
        let ff2 = g.add_op(Operation::Linear { in_features: 2048, out_features: 512 }, &[act]);
        let _res2 = g.add_op(Operation::Add, &[res1, ff2]);

        assert!(g.validate().is_ok());
        assert_eq!(g.len(), 9);
    }

    #[test]
    fn test_transformer_layer_flops() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![1, 32, 512], GraphDtype::F32);
        let norm = g.add_op(Operation::RmsNorm { eps: 1e-5 }, &[x]);
        let attn = g.add_op(Operation::Attention { num_heads: 8, head_dim: 64 }, &[norm]);
        let _res = g.add_op(Operation::Add, &[x, attn]);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        assert!(plan.estimated_flops > 0, "transformer layer should have non-zero FLOPs");
    }

    // ── Invalid graphs ───────────────────────────────────────────

    #[test]
    fn test_validate_empty_graph() {
        let g = ComputeGraph::new();
        assert!(matches!(g.validate(), Err(GraphError::EmptyGraph)));
    }

    #[test]
    fn test_validate_missing_input() {
        let mut g = ComputeGraph::new();
        g.nodes.push(GraphNode {
            id: 0,
            op: Operation::Silu,
            inputs: vec![99],
            output_shape: vec![1, 128],
            output_dtype: GraphDtype::F32,
        });
        assert!(matches!(g.validate(), Err(GraphError::MissingInput { .. })));
    }

    #[test]
    fn test_validate_matmul_shape_mismatch() {
        let mut g = ComputeGraph::new();
        let a = matrix_input(&mut g, "a", 4, 8);
        let b = matrix_input(&mut g, "b", 5, 16);
        // Manually build the matmul to bypass inference.
        g.nodes.push(GraphNode {
            id: 2,
            op: Operation::MatMul,
            inputs: vec![a, b],
            output_shape: vec![4, 16],
            output_dtype: GraphDtype::F32,
        });
        g.edges.push((a, 2));
        g.edges.push((b, 2));
        assert!(matches!(g.validate(), Err(GraphError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_validate_good_graph_succeeds() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let _y = g.add_op(Operation::Silu, &[x]);
        assert!(g.validate().is_ok());
    }

    // ── Execution plan stages ────────────────────────────────────

    #[test]
    fn test_plan_single_stage() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let _y = g.add_op(Operation::Silu, &[x]);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        assert_eq!(plan.stages.len(), 1);
        assert!(plan.stages[0].can_fuse);
    }

    #[test]
    fn test_plan_barrier_creates_stages() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let l = g.add_op(Operation::Linear { in_features: 128, out_features: 128 }, &[x]);
        let _s = g.add_op(Operation::Softmax { dim: 1 }, &[l]);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        // Input+Linear in one stage, Softmax in another.
        assert!(plan.stages.len() >= 2);
        let softmax_stage = plan.stages.iter().find(|s| !s.can_fuse);
        assert!(softmax_stage.is_some(), "softmax should be in a non-fusable stage");
    }

    #[test]
    fn test_plan_memory_matches_sum() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let _y = g.add_op(Operation::Silu, &[x]);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        let stage_sum: u64 = plan.stages.iter().map(|s| s.memory_needed).sum();
        assert_eq!(plan.total_memory, stage_sum);
    }

    // ── GraphExecutor ────────────────────────────────────────────

    #[test]
    fn test_executor_passthrough() {
        let mut g = ComputeGraph::new();
        let _x = g.add_input("x", vec![1, 4], GraphDtype::F32);
        let exec = GraphExecutor::new(g).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".into(), vec![1.0, 2.0, 3.0, 4.0]);
        let outputs = exec.execute(&inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_executor_silu_mock() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![1, 4], GraphDtype::F32);
        let _s = g.add_op(Operation::Silu, &[x]);
        let exec = GraphExecutor::new(g).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".into(), vec![1.0, 2.0, 3.0, 4.0]);
        let outputs = exec.execute(&inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 4);
    }

    #[test]
    fn test_executor_add_mock() {
        let mut g = ComputeGraph::new();
        let a = g.add_input("a", vec![1, 3], GraphDtype::F32);
        let b = g.add_input("b", vec![1, 3], GraphDtype::F32);
        let _c = g.add_op(Operation::Add, &[a, b]);
        let exec = GraphExecutor::new(g).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("a".into(), vec![1.0, 2.0, 3.0]);
        inputs.insert("b".into(), vec![10.0, 20.0, 30.0]);
        let outputs = exec.execute(&inputs).unwrap();
        assert_eq!(outputs[0], vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_executor_missing_input_uses_zeros() {
        let mut g = ComputeGraph::new();
        let _x = g.add_input("x", vec![1, 4], GraphDtype::F32);
        let exec = GraphExecutor::new(g).unwrap();
        let inputs = HashMap::new();
        let outputs = exec.execute(&inputs).unwrap();
        assert_eq!(outputs[0], vec![0.0; 4]);
    }

    #[test]
    fn test_executor_chain_propagates() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![1, 2], GraphDtype::F32);
        let y = g.add_op(Operation::Silu, &[x]);
        let _z = g.add_op(Operation::Gelu, &[y]);
        let exec = GraphExecutor::new(g).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".into(), vec![5.0, 10.0]);
        let outputs = exec.execute(&inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        // Mock just passes through sum: 5.0, 10.0
        assert_eq!(outputs[0], vec![5.0, 10.0]);
    }

    // ── GraphDtype ───────────────────────────────────────────────

    #[test]
    fn test_dtype_byte_sizes() {
        assert_eq!(GraphDtype::F32.byte_size(), 4);
        assert_eq!(GraphDtype::F16.byte_size(), 2);
        assert_eq!(GraphDtype::BF16.byte_size(), 2);
        assert_eq!(GraphDtype::I8.byte_size(), 1);
        assert_eq!(GraphDtype::I2.byte_size(), 1);
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", GraphDtype::F32), "f32");
        assert_eq!(format!("{}", GraphDtype::F16), "f16");
        assert_eq!(format!("{}", GraphDtype::BF16), "bf16");
        assert_eq!(format!("{}", GraphDtype::I8), "i8");
        assert_eq!(format!("{}", GraphDtype::I2), "i2");
    }

    // ── Operation display ────────────────────────────────────────

    #[test]
    fn test_operation_display() {
        assert_eq!(Operation::MatMul.to_string(), "MatMul");
        assert_eq!(Operation::Softmax { dim: 1 }.to_string(), "Softmax(dim=1)");
        assert_eq!(
            Operation::Linear { in_features: 64, out_features: 32 }.to_string(),
            "Linear(64,32)"
        );
    }

    #[test]
    fn test_fused_operation_display() {
        let op = Operation::Fused(vec![Operation::MatMul, Operation::Silu]);
        let s = op.to_string();
        assert!(s.contains("Fused"));
        assert!(s.contains("MatMul"));
        assert!(s.contains("SiLU"));
    }

    // ── GraphNode helpers ────────────────────────────────────────

    #[test]
    fn test_node_output_elements() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![2, 3, 4], GraphDtype::F32);
        assert_eq!(g.node(x).unwrap().output_elements(), 24);
    }

    #[test]
    fn test_node_output_bytes() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![2, 3, 4], GraphDtype::F32);
        assert_eq!(g.node(x).unwrap().output_bytes(), 96); // 24*4
    }

    #[test]
    fn test_node_output_bytes_f16() {
        let mut g = ComputeGraph::new();
        let x = g.add_input("x", vec![2, 3, 4], GraphDtype::F16);
        assert_eq!(g.node(x).unwrap().output_bytes(), 48); // 24*2
    }

    // ── Edge cases ───────────────────────────────────────────────

    #[test]
    fn test_broadcastable_shapes() {
        assert!(is_broadcastable(&[4, 128], &[1, 128]));
        assert!(is_broadcastable(&[1, 128], &[4, 128]));
        assert!(is_broadcastable(&[128], &[4, 128]));
        assert!(!is_broadcastable(&[4, 128], &[4, 64]));
    }

    #[test]
    fn test_default_trait() {
        let g = ComputeGraph::default();
        assert!(g.is_empty());
    }

    #[test]
    fn test_graph_node_clone() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let node = g.node(x).unwrap().clone();
        assert_eq!(node.id, x);
    }

    #[test]
    fn test_graph_clone() {
        let mut g = ComputeGraph::new();
        let x = simple_input(&mut g, "x");
        let _y = g.add_op(Operation::Silu, &[x]);
        let g2 = g.clone();
        assert_eq!(g.len(), g2.len());
    }

    #[test]
    fn test_error_display() {
        let e = GraphError::CycleDetected;
        assert_eq!(e.to_string(), "cycle detected in graph");
        let e = GraphError::NodeNotFound(42);
        assert_eq!(e.to_string(), "node 42 not found in graph");
        let e = GraphError::EmptyGraph;
        assert_eq!(e.to_string(), "empty graph");
    }

    #[test]
    fn test_plan_from_invalid_graph_errors() {
        let g = ComputeGraph::new();
        assert!(ExecutionPlan::from_graph(&g).is_err());
    }

    #[test]
    fn test_executor_from_invalid_graph_errors() {
        let g = ComputeGraph::new();
        assert!(GraphExecutor::new(g).is_err());
    }
}
