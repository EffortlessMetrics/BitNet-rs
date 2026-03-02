//! Operator graph optimizer for Intel Arc A770 OpenCL inference.
//!
//! Represents inference computation as a DAG of operators and applies
//! graph-level transformations: constant folding, dead code elimination,
//! operator fusion, and layout optimization. Generates an ordered
//! [`ExecutionPlan`] from the optimized graph.
//!
//! # CPU reference
//!
//! All graph operations and optimization passes are implemented as pure-CPU
//! reference code — no OpenCL runtime required.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Node / Edge identifiers ────────────────────────────────────────

/// Unique identifier for a node in the operator graph.
pub type NodeId = u64;

/// Unique identifier for an edge in the operator graph.
pub type EdgeId = u64;

// ── Data types ─────────────────────────────────────────────────────

/// Element data type for tensor operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I2,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I8 => "i8",
            Self::I2 => "i2",
        };
        write!(f, "{s}")
    }
}

// ── Operator enum ──────────────────────────────────────────────────

/// Elementary operators that can appear in the inference graph.
#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    MatMul,
    Add,
    RMSNorm { eps: f32 },
    RoPE { base_freq: f32 },
    Softmax,
    SiLU,
    Mul,
    Gather { axis: usize },
    Concat { axis: usize },
    Reshape { target_shape: Vec<usize> },
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatMul => write!(f, "MatMul"),
            Self::Add => write!(f, "Add"),
            Self::RMSNorm { eps } => write!(f, "RMSNorm(eps={eps})"),
            Self::RoPE { base_freq } => write!(f, "RoPE(θ={base_freq})"),
            Self::Softmax => write!(f, "Softmax"),
            Self::SiLU => write!(f, "SiLU"),
            Self::Mul => write!(f, "Mul"),
            Self::Gather { axis } => write!(f, "Gather(axis={axis})"),
            Self::Concat { axis } => write!(f, "Concat(axis={axis})"),
            Self::Reshape { target_shape } => {
                write!(f, "Reshape({target_shape:?})")
            }
        }
    }
}

// ── OperatorNode ───────────────────────────────────────────────────

/// Metadata attached to an operator node.
#[derive(Debug, Clone, Default)]
pub struct NodeMetadata {
    /// Human-readable name.
    pub name: Option<String>,
    /// Whether the node holds a compile-time constant value.
    pub is_constant: bool,
    /// Optional constant scalar value (for constant-folding).
    pub constant_value: Option<f64>,
    /// Preferred memory layout for A770 EU cache hierarchy.
    pub layout_hint: Option<LayoutHint>,
}

/// Preferred tensor memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutHint {
    /// Row-major (C-order), default.
    RowMajor,
    /// Column-major (Fortran-order), faster for A770 Xe-core reads.
    ColMajor,
    /// Tiled 16×16 blocks aligned to A770 sub-slice cache.
    Tiled16x16,
}

/// A single node in the operator graph.
#[derive(Debug, Clone)]
pub struct OperatorNode {
    pub id: NodeId,
    pub op: Operator,
    pub inputs: Vec<NodeId>,
    pub output_shape: Vec<usize>,
    pub dtype: DType,
    pub metadata: NodeMetadata,
}

// ── Edge ───────────────────────────────────────────────────────────

/// Directed edge from one node's output to another node's input.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Edge {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
}

// ── OperatorGraph ──────────────────────────────────────────────────

/// DAG of [`OperatorNode`]s connected by [`Edge`]s.
#[derive(Debug, Clone)]
pub struct OperatorGraph {
    nodes: HashMap<NodeId, OperatorNode>,
    edges: Vec<Edge>,
    input_nodes: Vec<NodeId>,
    output_nodes: Vec<NodeId>,
    next_node_id: NodeId,
    next_edge_id: EdgeId,
}

impl Default for OperatorGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            input_nodes: Vec::new(),
            output_nodes: Vec::new(),
            next_node_id: 0,
            next_edge_id: 0,
        }
    }

    /// Add a node and return its id.
    pub fn add_node(
        &mut self,
        op: Operator,
        inputs: Vec<NodeId>,
        output_shape: Vec<usize>,
        dtype: DType,
    ) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;

        // Create edges from each input to this node.
        for &input_id in &inputs {
            let edge = Edge { id: self.next_edge_id, from: input_id, to: id };
            self.next_edge_id += 1;
            self.edges.push(edge);
        }

        let node =
            OperatorNode { id, op, inputs, output_shape, dtype, metadata: NodeMetadata::default() };
        self.nodes.insert(id, node);
        id
    }

    /// Add a constant scalar node.
    pub fn add_constant(&mut self, value: f64, shape: Vec<usize>, dtype: DType) -> NodeId {
        let id = self.add_node(Operator::Add, vec![], shape, dtype);
        if let Some(node) = self.nodes.get_mut(&id) {
            node.metadata.is_constant = true;
            node.metadata.constant_value = Some(value);
        }
        id
    }

    /// Mark nodes as graph inputs.
    pub fn set_inputs(&mut self, ids: Vec<NodeId>) {
        self.input_nodes = ids;
    }

    /// Mark nodes as graph outputs.
    pub fn set_outputs(&mut self, ids: Vec<NodeId>) {
        self.output_nodes = ids;
    }

    pub fn input_nodes(&self) -> &[NodeId] {
        &self.input_nodes
    }

    pub fn output_nodes(&self) -> &[NodeId] {
        &self.output_nodes
    }

    pub fn node(&self, id: NodeId) -> Option<&OperatorNode> {
        self.nodes.get(&id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut OperatorNode> {
        self.nodes.get_mut(&id)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }

    /// Consumers of a given node.
    pub fn consumers(&self, id: NodeId) -> Vec<NodeId> {
        self.edges.iter().filter(|e| e.from == id).map(|e| e.to).collect()
    }

    /// Producers (inputs) of a given node.
    pub fn producers(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes.get(&id).map(|n| n.inputs.clone()).unwrap_or_default()
    }

    /// Remove a node and all its associated edges.
    pub fn remove_node(&mut self, id: NodeId) {
        self.nodes.remove(&id);
        self.edges.retain(|e| e.from != id && e.to != id);
        // Clean up references in other nodes' input lists.
        for node in self.nodes.values_mut() {
            node.inputs.retain(|inp| *inp != id);
        }
        self.input_nodes.retain(|n| *n != id);
        self.output_nodes.retain(|n| *n != id);
    }

    /// Replace one node with another in all consumer input lists.
    pub fn replace_node_references(&mut self, old_id: NodeId, new_id: NodeId) {
        for node in self.nodes.values_mut() {
            for inp in &mut node.inputs {
                if *inp == old_id {
                    *inp = new_id;
                }
            }
        }
        for edge in &mut self.edges {
            if edge.from == old_id {
                edge.from = new_id;
            }
            if edge.to == old_id {
                edge.to = new_id;
            }
        }
        // Remove self-loops created by the replacement.
        self.edges.retain(|e| e.from != e.to);
        for node in self.nodes.values_mut() {
            let id = node.id;
            node.inputs.retain(|&inp| inp != id);
        }
    }

    /// Topological sort via Kahn's algorithm.
    /// Returns `Err` if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, GraphError> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for &id in self.nodes.keys() {
            in_degree.entry(id).or_insert(0);
        }
        for edge in &self.edges {
            if self.nodes.contains_key(&edge.from) && self.nodes.contains_key(&edge.to) {
                *in_degree.entry(edge.to).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<NodeId> =
            in_degree.iter().filter(|&(_, &deg)| deg == 0).map(|(&id, _)| id).collect();

        // Sort queue for deterministic output.
        let mut sorted_start: Vec<NodeId> = queue.drain(..).collect();
        sorted_start.sort();
        queue.extend(sorted_start);

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = queue.pop_front() {
            order.push(id);
            let mut next_ready = Vec::new();
            for edge in &self.edges {
                if edge.from == id
                    && self.nodes.contains_key(&edge.to)
                    && let Some(deg) = in_degree.get_mut(&edge.to)
                {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        next_ready.push(edge.to);
                    }
                }
            }
            next_ready.sort();
            next_ready.dedup();
            queue.extend(next_ready);
        }

        if order.len() != self.nodes.len() {
            return Err(GraphError::CycleDetected);
        }
        Ok(order)
    }

    /// Detect whether the graph contains a cycle.
    pub fn has_cycle(&self) -> bool {
        self.topological_sort().is_err()
    }
}

// ── GraphError ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    CycleDetected,
    NodeNotFound(NodeId),
    InvalidPass(String),
    EmptyGraph,
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CycleDetected => write!(f, "cycle detected in operator graph"),
            Self::NodeNotFound(id) => write!(f, "node {id} not found"),
            Self::InvalidPass(msg) => write!(f, "invalid pass: {msg}"),
            Self::EmptyGraph => write!(f, "empty operator graph"),
        }
    }
}

impl std::error::Error for GraphError {}

// ── PassResult ─────────────────────────────────────────────────────

/// Result of applying a graph optimization pass.
#[derive(Debug, Clone)]
pub struct PassResult {
    /// Number of nodes removed or transformed.
    pub nodes_changed: usize,
    /// Human-readable description of what changed.
    pub description: String,
}

// ── GraphPass trait ────────────────────────────────────────────────

/// Extensible optimization pass over an [`OperatorGraph`].
pub trait GraphPass {
    /// Name of the pass for logging.
    fn name(&self) -> &str;

    /// Mutate the graph in place; return a summary of changes.
    fn apply(&self, graph: &mut OperatorGraph) -> Result<PassResult, GraphError>;
}

// ── ConstantFoldPass ───────────────────────────────────────────────

/// Folds operations whose inputs are all compile-time constants.
pub struct ConstantFoldPass;

impl ConstantFoldPass {
    /// Evaluate a binary op on two constant scalars (CPU reference).
    fn eval_binary(op: &Operator, a: f64, b: f64) -> Option<f64> {
        match op {
            Operator::Add => Some(a + b),
            Operator::Mul => Some(a * b),
            _ => None,
        }
    }
}

impl GraphPass for ConstantFoldPass {
    fn name(&self) -> &str {
        "ConstantFold"
    }

    fn apply(&self, graph: &mut OperatorGraph) -> Result<PassResult, GraphError> {
        let mut folded = 0usize;
        // Iterate in topo order so predecessors are folded first.
        let order = graph.topological_sort()?;

        for id in order {
            let node = match graph.node(id) {
                Some(n) => n.clone(),
                None => continue,
            };
            if node.metadata.is_constant || node.inputs.is_empty() {
                continue;
            }
            // Check if ALL inputs are constant scalars.
            let input_vals: Vec<f64> = node
                .inputs
                .iter()
                .filter_map(|&inp| {
                    graph.node(inp).and_then(|n| {
                        if n.metadata.is_constant { n.metadata.constant_value } else { None }
                    })
                })
                .collect();

            if input_vals.len() != node.inputs.len() || node.inputs.is_empty() {
                continue;
            }

            // Attempt to fold.
            let result = if input_vals.len() == 2 {
                Self::eval_binary(&node.op, input_vals[0], input_vals[1])
            } else {
                None
            };

            if let Some(val) = result
                && let Some(n) = graph.node_mut(id)
            {
                n.metadata.is_constant = true;
                n.metadata.constant_value = Some(val);
                n.inputs.clear();
                folded += 1;
            }
        }

        Ok(PassResult {
            nodes_changed: folded,
            description: format!("folded {folded} constant expressions"),
        })
    }
}

// ── DeadCodeEliminationPass ────────────────────────────────────────

/// Removes nodes that do not contribute to any graph output.
pub struct DeadCodeEliminationPass;

impl GraphPass for DeadCodeEliminationPass {
    fn name(&self) -> &str {
        "DeadCodeElimination"
    }

    fn apply(&self, graph: &mut OperatorGraph) -> Result<PassResult, GraphError> {
        // Walk backwards from outputs to find all live nodes.
        let mut live: HashSet<NodeId> = HashSet::new();
        let mut worklist: VecDeque<NodeId> = graph.output_nodes().iter().copied().collect();

        while let Some(id) = worklist.pop_front() {
            if !live.insert(id) {
                continue;
            }
            for &inp in &graph.producers(id) {
                worklist.push_back(inp);
            }
        }

        // Also keep input nodes that are live.
        for &id in graph.input_nodes() {
            live.insert(id);
        }

        let all_ids: Vec<NodeId> = graph.node_ids();
        let mut removed = 0usize;
        for id in all_ids {
            if !live.contains(&id) {
                graph.remove_node(id);
                removed += 1;
            }
        }

        Ok(PassResult {
            nodes_changed: removed,
            description: format!("eliminated {removed} dead nodes"),
        })
    }
}

// ── OperatorFusionPass ─────────────────────────────────────────────

/// Fused operator pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum FusedOp {
    MatMulAdd,
    RMSNormMul,
    SiLUMul,
}

impl fmt::Display for FusedOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatMulAdd => write!(f, "FusedMatMulAdd"),
            Self::RMSNormMul => write!(f, "FusedRMSNormMul"),
            Self::SiLUMul => write!(f, "FusedSiLUMul"),
        }
    }
}

/// Fuses compatible adjacent operators (MatMul+Add, RMSNorm+Mul, SiLU+Mul).
pub struct OperatorFusionPass;

impl OperatorFusionPass {
    /// Check if `consumer` is the sole consumer of `producer`.
    fn is_sole_consumer(graph: &OperatorGraph, producer: NodeId, consumer: NodeId) -> bool {
        let consumers = graph.consumers(producer);
        consumers.len() == 1 && consumers[0] == consumer
    }

    /// Attempt to detect a fusion pattern at `node_id`.
    fn detect_fusion(graph: &OperatorGraph, node_id: NodeId) -> Option<(FusedOp, NodeId, NodeId)> {
        let node = graph.node(node_id)?;
        match &node.op {
            Operator::Add => {
                // Look for MatMul → Add pattern.
                for &inp in &node.inputs {
                    if let Some(pred) = graph.node(inp)
                        && pred.op == Operator::MatMul
                        && Self::is_sole_consumer(graph, inp, node_id)
                    {
                        return Some((FusedOp::MatMulAdd, inp, node_id));
                    }
                }
                None
            }
            Operator::Mul => {
                // Look for RMSNorm → Mul or SiLU → Mul patterns.
                for &inp in &node.inputs {
                    if let Some(pred) = graph.node(inp) {
                        if matches!(pred.op, Operator::RMSNorm { .. })
                            && Self::is_sole_consumer(graph, inp, node_id)
                        {
                            return Some((FusedOp::RMSNormMul, inp, node_id));
                        }
                        if pred.op == Operator::SiLU && Self::is_sole_consumer(graph, inp, node_id)
                        {
                            return Some((FusedOp::SiLUMul, inp, node_id));
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }
}

impl GraphPass for OperatorFusionPass {
    fn name(&self) -> &str {
        "OperatorFusion"
    }

    fn apply(&self, graph: &mut OperatorGraph) -> Result<PassResult, GraphError> {
        let mut fused = 0usize;
        // Multiple passes until no more fusions are found.
        loop {
            let order = graph.topological_sort()?;
            let mut fusion_found = None;
            for &id in &order {
                if let Some(pattern) = Self::detect_fusion(graph, id) {
                    fusion_found = Some(pattern);
                    break;
                }
            }

            let Some((fused_op, first_id, second_id)) = fusion_found else {
                break;
            };

            // Merge: keep the second node, update its metadata to record
            // the fusion, and re-wire inputs from the first node.
            let first_inputs = graph.node(first_id).map(|n| n.inputs.clone()).unwrap_or_default();
            let first_shape =
                graph.node(first_id).map(|n| n.output_shape.clone()).unwrap_or_default();

            if let Some(second) = graph.node_mut(second_id) {
                // Collect remaining inputs that aren't the fused producer.
                let other_inputs: Vec<NodeId> =
                    second.inputs.iter().filter(|&&i| i != first_id).copied().collect();
                let mut new_inputs = first_inputs;
                new_inputs.extend(other_inputs);
                second.inputs = new_inputs;
                second.metadata.name = Some(format!("fused:{fused_op}"));
                if second.output_shape.is_empty() {
                    second.output_shape = first_shape;
                }
            }

            // Redirect anyone who consumed `first_id` (other than second)
            // to `second_id`, then remove `first_id`.
            graph.replace_node_references(first_id, second_id);
            graph.remove_node(first_id);
            fused += 1;
        }

        Ok(PassResult {
            nodes_changed: fused,
            description: format!("fused {fused} operator pairs"),
        })
    }
}

// ── LayoutOptimizationPass ─────────────────────────────────────────

/// Optimizes tensor layouts for the A770 memory hierarchy.
///
/// Rules (CPU reference heuristics):
/// - MatMul RHS → `ColMajor` for better vectorized column access.
/// - Large (≥256 in both dims) MatMul operands → `Tiled16x16`.
/// - Everything else defaults to `RowMajor`.
pub struct LayoutOptimizationPass;

impl GraphPass for LayoutOptimizationPass {
    fn name(&self) -> &str {
        "LayoutOptimization"
    }

    fn apply(&self, graph: &mut OperatorGraph) -> Result<PassResult, GraphError> {
        let mut changed = 0usize;
        let ids: Vec<NodeId> = graph.node_ids();

        for id in ids {
            let hint = {
                let node = match graph.node(id) {
                    Some(n) => n,
                    None => continue,
                };
                if node.metadata.layout_hint.is_some() {
                    continue;
                }
                match &node.op {
                    Operator::MatMul => {
                        let large = node.output_shape.len() >= 2
                            && node.output_shape.iter().all(|&d| d >= 256);
                        if large { LayoutHint::Tiled16x16 } else { LayoutHint::ColMajor }
                    }
                    _ => LayoutHint::RowMajor,
                }
            };

            if let Some(node) = graph.node_mut(id) {
                node.metadata.layout_hint = Some(hint);
                changed += 1;
            }
        }

        Ok(PassResult {
            nodes_changed: changed,
            description: format!("assigned layout hints to {changed} nodes"),
        })
    }
}

// ── ExecutionPlan ──────────────────────────────────────────────────

/// Memory plan entry for one node.
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub node_id: NodeId,
    pub size_bytes: usize,
    pub offset: usize,
}

/// Ordered execution plan derived from an optimized graph.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Topologically sorted operator ids.
    pub ordered_ops: Vec<NodeId>,
    /// Per-node memory allocations.
    pub memory_plan: Vec<MemoryAllocation>,
    /// Estimated total execution time in microseconds (heuristic).
    pub estimated_time_us: u64,
}

impl ExecutionPlan {
    /// Build an execution plan from a (presumably optimized) graph.
    pub fn from_graph(graph: &OperatorGraph) -> Result<Self, GraphError> {
        if graph.node_count() == 0 {
            return Err(GraphError::EmptyGraph);
        }
        let ordered_ops = graph.topological_sort()?;

        // Simple bump-allocator memory plan.
        let mut memory_plan = Vec::new();
        let mut offset = 0usize;
        for &id in &ordered_ops {
            if let Some(node) = graph.node(id) {
                let elem_size = match node.dtype {
                    DType::F32 => 4,
                    DType::F16 | DType::BF16 => 2,
                    DType::I8 => 1,
                    DType::I2 => 1, // minimum addressable
                };
                let numel: usize = node.output_shape.iter().product::<usize>().max(1);
                let size_bytes = numel * elem_size;
                memory_plan.push(MemoryAllocation { node_id: id, size_bytes, offset });
                offset += size_bytes;
            }
        }

        // Heuristic: ~1 µs per node on A770 at ~2 GHz EU clock.
        let estimated_time_us = ordered_ops.len() as u64;

        Ok(Self { ordered_ops, memory_plan, estimated_time_us })
    }

    /// Total memory required in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.memory_plan.iter().map(|a| a.offset + a.size_bytes).max().unwrap_or(0)
    }
}

// ── Pipeline helper ────────────────────────────────────────────────

/// Apply a sequence of passes and produce an execution plan.
pub fn optimize_and_plan(
    graph: &mut OperatorGraph,
    passes: &[&dyn GraphPass],
) -> Result<ExecutionPlan, GraphError> {
    for pass in passes {
        pass.apply(graph)?;
    }
    ExecutionPlan::from_graph(graph)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────

    fn shape(dims: &[usize]) -> Vec<usize> {
        dims.to_vec()
    }

    /// Build a simple linear graph: input → matmul → add → output.
    fn simple_matmul_add_graph() -> OperatorGraph {
        let mut g = OperatorGraph::new();
        let inp = g.add_node(Operator::Gather { axis: 0 }, vec![], shape(&[1, 2048]), DType::F32);
        let mm = g.add_node(Operator::MatMul, vec![inp], shape(&[1, 2048]), DType::F32);
        let bias = g.add_constant(0.0, shape(&[2048]), DType::F32);
        let add = g.add_node(Operator::Add, vec![mm, bias], shape(&[1, 2048]), DType::F32);
        g.set_inputs(vec![inp]);
        g.set_outputs(vec![add]);
        g
    }

    // ── graph construction ─────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = OperatorGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_single_node() {
        let mut g = OperatorGraph::new();
        let id = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.node(id).unwrap().op, Operator::SiLU);
    }

    #[test]
    fn test_add_multiple_nodes() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::Mul, vec![a], shape(&[4]), DType::F32);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.node(b).unwrap().inputs, vec![a]);
    }

    #[test]
    fn test_add_constant_node() {
        let mut g = OperatorGraph::new();
        let c = g.add_constant(3.14, shape(&[1]), DType::F32);
        let n = g.node(c).unwrap();
        assert!(n.metadata.is_constant);
        assert_eq!(n.metadata.constant_value, Some(3.14));
    }

    #[test]
    fn test_set_inputs_outputs() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::Mul, vec![a], shape(&[4]), DType::F32);
        g.set_inputs(vec![a]);
        g.set_outputs(vec![b]);
        assert_eq!(g.input_nodes(), &[a]);
        assert_eq!(g.output_nodes(), &[b]);
    }

    #[test]
    fn test_consumers_and_producers() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::Mul, vec![a], shape(&[4]), DType::F32);
        assert_eq!(g.consumers(a), vec![b]);
        assert_eq!(g.producers(b), vec![a]);
    }

    #[test]
    fn test_remove_node() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::Mul, vec![a], shape(&[4]), DType::F32);
        g.remove_node(a);
        assert_eq!(g.node_count(), 1);
        assert!(g.node(a).is_none());
        assert!(g.node(b).unwrap().inputs.is_empty());
    }

    #[test]
    fn test_replace_node_references() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let c = g.add_node(Operator::Mul, vec![a], shape(&[4]), DType::F32);
        g.replace_node_references(a, b);
        assert_eq!(g.node(c).unwrap().inputs, vec![b]);
    }

    // ── topological sort ───────────────────────────────────────────

    #[test]
    fn test_topo_sort_linear() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::Mul, vec![a], shape(&[4]), DType::F32);
        let c = g.add_node(Operator::Softmax, vec![b], shape(&[4]), DType::F32);
        let order = g.topological_sort().unwrap();
        assert_eq!(order, vec![a, b, c]);
    }

    #[test]
    fn test_topo_sort_diamond() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::Mul, vec![a], shape(&[4]), DType::F32);
        let c = g.add_node(Operator::Add, vec![a], shape(&[4]), DType::F32);
        let d = g.add_node(Operator::Add, vec![b, c], shape(&[4]), DType::F32);
        let order = g.topological_sort().unwrap();
        // a must come first, d last; b and c in between.
        assert_eq!(order[0], a);
        assert_eq!(*order.last().unwrap(), d);
    }

    #[test]
    fn test_topo_sort_empty_graph() {
        let g = OperatorGraph::new();
        let order = g.topological_sort().unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn test_topo_sort_single_node() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let order = g.topological_sort().unwrap();
        assert_eq!(order, vec![a]);
    }

    // ── cycle detection ────────────────────────────────────────────

    #[test]
    fn test_no_cycle_in_dag() {
        let g = simple_matmul_add_graph();
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::Mul, vec![a], shape(&[4]), DType::F32);
        // Manually inject a back-edge to create a cycle.
        g.edges.push(Edge { id: g.next_edge_id, from: b, to: a });
        g.next_edge_id += 1;
        assert!(g.has_cycle());
    }

    // ── constant folding ───────────────────────────────────────────

    #[test]
    fn test_constant_fold_add() {
        let mut g = OperatorGraph::new();
        let a = g.add_constant(2.0, shape(&[1]), DType::F32);
        let b = g.add_constant(3.0, shape(&[1]), DType::F32);
        let c = g.add_node(Operator::Add, vec![a, b], shape(&[1]), DType::F32);
        g.set_outputs(vec![c]);

        let pass = ConstantFoldPass;
        let result = pass.apply(&mut g).unwrap();
        assert_eq!(result.nodes_changed, 1);

        let node = g.node(c).unwrap();
        assert!(node.metadata.is_constant);
        assert_eq!(node.metadata.constant_value, Some(5.0));
    }

    #[test]
    fn test_constant_fold_mul() {
        let mut g = OperatorGraph::new();
        let a = g.add_constant(4.0, shape(&[1]), DType::F32);
        let b = g.add_constant(5.0, shape(&[1]), DType::F32);
        let c = g.add_node(Operator::Mul, vec![a, b], shape(&[1]), DType::F32);
        g.set_outputs(vec![c]);

        let pass = ConstantFoldPass;
        let result = pass.apply(&mut g).unwrap();
        assert_eq!(result.nodes_changed, 1);
        assert_eq!(g.node(c).unwrap().metadata.constant_value, Some(20.0));
    }

    #[test]
    fn test_constant_fold_chain() {
        let mut g = OperatorGraph::new();
        let a = g.add_constant(1.0, shape(&[1]), DType::F32);
        let b = g.add_constant(2.0, shape(&[1]), DType::F32);
        let c = g.add_node(Operator::Add, vec![a, b], shape(&[1]), DType::F32);
        let d = g.add_constant(3.0, shape(&[1]), DType::F32);
        let e = g.add_node(Operator::Mul, vec![c, d], shape(&[1]), DType::F32);
        g.set_outputs(vec![e]);

        // First pass folds c=1+2=3, second pass folds e=3*3=9.
        let pass = ConstantFoldPass;
        pass.apply(&mut g).unwrap();
        let result = pass.apply(&mut g).unwrap();
        // Second pass should fold the chain result.
        assert!(
            result.nodes_changed >= 1 || {
                let n = g.node(e).unwrap();
                n.metadata.is_constant && n.metadata.constant_value == Some(9.0)
            }
        );
    }

    #[test]
    fn test_constant_fold_noop_no_constants() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::Mul, vec![a], shape(&[4]), DType::F32);
        g.set_outputs(vec![b]);

        let pass = ConstantFoldPass;
        let result = pass.apply(&mut g).unwrap();
        assert_eq!(result.nodes_changed, 0);
    }

    #[test]
    fn test_constant_fold_partial_constant_inputs() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_constant(2.0, shape(&[4]), DType::F32);
        let c = g.add_node(Operator::Mul, vec![a, b], shape(&[4]), DType::F32);
        g.set_outputs(vec![c]);

        let pass = ConstantFoldPass;
        let result = pass.apply(&mut g).unwrap();
        // Should NOT fold because a is not constant.
        assert_eq!(result.nodes_changed, 0);
    }

    // ── dead code elimination ──────────────────────────────────────

    #[test]
    fn test_dce_removes_unused() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let _dead = g.add_node(Operator::Softmax, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::Mul, vec![a], shape(&[4]), DType::F32);
        g.set_inputs(vec![a]);
        g.set_outputs(vec![b]);

        let pass = DeadCodeEliminationPass;
        let result = pass.apply(&mut g).unwrap();
        assert_eq!(result.nodes_changed, 1);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_dce_keeps_all_when_reachable() {
        let g = simple_matmul_add_graph();
        let before = g.node_count();
        let mut g = g;
        let pass = DeadCodeEliminationPass;
        let result = pass.apply(&mut g).unwrap();
        assert_eq!(result.nodes_changed, 0);
        assert_eq!(g.node_count(), before);
    }

    #[test]
    fn test_dce_empty_graph() {
        let mut g = OperatorGraph::new();
        let pass = DeadCodeEliminationPass;
        let result = pass.apply(&mut g).unwrap();
        assert_eq!(result.nodes_changed, 0);
    }

    #[test]
    fn test_dce_all_dead() {
        let mut g = OperatorGraph::new();
        let _a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let _b = g.add_node(Operator::Softmax, vec![], shape(&[4]), DType::F32);
        // No outputs set — everything is dead.
        let pass = DeadCodeEliminationPass;
        let result = pass.apply(&mut g).unwrap();
        assert_eq!(result.nodes_changed, 2);
        assert_eq!(g.node_count(), 0);
    }

    // ── operator fusion ────────────────────────────────────────────

    #[test]
    fn test_fusion_matmul_add() {
        let mut g = simple_matmul_add_graph();
        let before = g.node_count();
        let pass = OperatorFusionPass;
        let result = pass.apply(&mut g).unwrap();
        assert!(result.nodes_changed >= 1);
        assert!(g.node_count() < before);
        // The fused node should exist.
        let order = g.topological_sort().unwrap();
        let fused = order.iter().any(|&id| {
            g.node(id)
                .and_then(|n| n.metadata.name.as_deref())
                .is_some_and(|name| name.contains("FusedMatMulAdd"))
        });
        assert!(fused, "expected a FusedMatMulAdd node");
    }

    #[test]
    fn test_fusion_rmsnorm_mul() {
        let mut g = OperatorGraph::new();
        let inp = g.add_node(Operator::Gather { axis: 0 }, vec![], shape(&[1, 2048]), DType::F32);
        let norm =
            g.add_node(Operator::RMSNorm { eps: 1e-6 }, vec![inp], shape(&[1, 2048]), DType::F32);
        let scale = g.add_constant(1.0, shape(&[2048]), DType::F32);
        let mul = g.add_node(Operator::Mul, vec![norm, scale], shape(&[1, 2048]), DType::F32);
        g.set_inputs(vec![inp]);
        g.set_outputs(vec![mul]);

        let pass = OperatorFusionPass;
        let result = pass.apply(&mut g).unwrap();
        assert!(result.nodes_changed >= 1);
        let has_fused = g.node_ids().iter().any(|&id| {
            g.node(id)
                .and_then(|n| n.metadata.name.as_deref())
                .is_some_and(|name| name.contains("RMSNormMul"))
        });
        assert!(has_fused, "expected FusedRMSNormMul");
    }

    #[test]
    fn test_fusion_silu_mul() {
        let mut g = OperatorGraph::new();
        let inp = g.add_node(Operator::Gather { axis: 0 }, vec![], shape(&[1, 2048]), DType::F32);
        let silu = g.add_node(Operator::SiLU, vec![inp], shape(&[1, 2048]), DType::F32);
        let other = g.add_node(Operator::Gather { axis: 0 }, vec![], shape(&[1, 2048]), DType::F32);
        let mul = g.add_node(Operator::Mul, vec![silu, other], shape(&[1, 2048]), DType::F32);
        g.set_inputs(vec![inp, other]);
        g.set_outputs(vec![mul]);

        let pass = OperatorFusionPass;
        let result = pass.apply(&mut g).unwrap();
        assert!(result.nodes_changed >= 1);
    }

    #[test]
    fn test_fusion_no_fusion_when_multiple_consumers() {
        let mut g = OperatorGraph::new();
        let inp = g.add_node(Operator::Gather { axis: 0 }, vec![], shape(&[1, 2048]), DType::F32);
        let mm = g.add_node(Operator::MatMul, vec![inp], shape(&[1, 2048]), DType::F32);
        let bias = g.add_constant(0.0, shape(&[2048]), DType::F32);
        let add = g.add_node(Operator::Add, vec![mm, bias], shape(&[1, 2048]), DType::F32);
        // Second consumer of mm prevents fusion.
        let add2 = g.add_node(Operator::Add, vec![mm], shape(&[1, 2048]), DType::F32);
        g.set_inputs(vec![inp]);
        g.set_outputs(vec![add, add2]);

        let pass = OperatorFusionPass;
        let result = pass.apply(&mut g).unwrap();
        assert_eq!(result.nodes_changed, 0);
    }

    #[test]
    fn test_fusion_noop_on_unfusable() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::Softmax, vec![a], shape(&[4]), DType::F32);
        g.set_inputs(vec![a]);
        g.set_outputs(vec![b]);

        let pass = OperatorFusionPass;
        let result = pass.apply(&mut g).unwrap();
        assert_eq!(result.nodes_changed, 0);
    }

    // ── layout optimization ────────────────────────────────────────

    #[test]
    fn test_layout_matmul_gets_hint() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::MatMul, vec![], shape(&[64, 64]), DType::F32);
        g.set_outputs(vec![a]);

        let pass = LayoutOptimizationPass;
        pass.apply(&mut g).unwrap();
        assert_eq!(g.node(a).unwrap().metadata.layout_hint, Some(LayoutHint::ColMajor));
    }

    #[test]
    fn test_layout_large_matmul_gets_tiled() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::MatMul, vec![], shape(&[512, 512]), DType::F32);
        g.set_outputs(vec![a]);

        let pass = LayoutOptimizationPass;
        pass.apply(&mut g).unwrap();
        assert_eq!(g.node(a).unwrap().metadata.layout_hint, Some(LayoutHint::Tiled16x16));
    }

    #[test]
    fn test_layout_non_matmul_row_major() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        g.set_outputs(vec![a]);

        let pass = LayoutOptimizationPass;
        pass.apply(&mut g).unwrap();
        assert_eq!(g.node(a).unwrap().metadata.layout_hint, Some(LayoutHint::RowMajor));
    }

    #[test]
    fn test_layout_preserves_existing_hint() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::MatMul, vec![], shape(&[64, 64]), DType::F32);
        g.node_mut(a).unwrap().metadata.layout_hint = Some(LayoutHint::RowMajor);
        g.set_outputs(vec![a]);

        let pass = LayoutOptimizationPass;
        let result = pass.apply(&mut g).unwrap();
        assert_eq!(result.nodes_changed, 0);
        assert_eq!(g.node(a).unwrap().metadata.layout_hint, Some(LayoutHint::RowMajor));
    }

    // ── execution plan ─────────────────────────────────────────────

    #[test]
    fn test_plan_from_simple_graph() {
        let g = simple_matmul_add_graph();
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        assert_eq!(plan.ordered_ops.len(), g.node_count());
        assert!(!plan.memory_plan.is_empty());
        assert!(plan.total_memory_bytes() > 0);
    }

    #[test]
    fn test_plan_empty_graph_error() {
        let g = OperatorGraph::new();
        let err = ExecutionPlan::from_graph(&g).unwrap_err();
        assert_eq!(err, GraphError::EmptyGraph);
    }

    #[test]
    fn test_plan_memory_offsets_non_overlapping() {
        let g = simple_matmul_add_graph();
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        for (i, alloc) in plan.memory_plan.iter().enumerate() {
            for other in &plan.memory_plan[i + 1..] {
                let end_a = alloc.offset + alloc.size_bytes;
                let end_b = other.offset + other.size_bytes;
                assert!(
                    end_a <= other.offset || end_b <= alloc.offset,
                    "overlapping allocations: {:?} vs {:?}",
                    alloc,
                    other
                );
            }
        }
    }

    #[test]
    fn test_plan_estimated_time_proportional() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::Mul, vec![a], shape(&[4]), DType::F32);
        let c = g.add_node(Operator::Softmax, vec![b], shape(&[4]), DType::F32);
        g.set_outputs(vec![c]);
        let plan = ExecutionPlan::from_graph(&g).unwrap();
        assert_eq!(plan.estimated_time_us, 3);
    }

    // ── optimize_and_plan pipeline ─────────────────────────────────

    #[test]
    fn test_full_pipeline() {
        let mut g = simple_matmul_add_graph();
        let passes: Vec<&dyn GraphPass> = vec![
            &ConstantFoldPass,
            &DeadCodeEliminationPass,
            &OperatorFusionPass,
            &LayoutOptimizationPass,
        ];
        let plan = optimize_and_plan(&mut g, &passes).unwrap();
        assert!(!plan.ordered_ops.is_empty());
    }

    #[test]
    fn test_pipeline_with_dead_code() {
        let mut g = simple_matmul_add_graph();
        // Add a dead branch.
        let _dead = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);

        let passes: Vec<&dyn GraphPass> = vec![&DeadCodeEliminationPass, &OperatorFusionPass];
        let plan = optimize_and_plan(&mut g, &passes).unwrap();
        // Dead node should have been removed.
        assert!(!plan.ordered_ops.contains(&_dead));
    }

    // ── disconnected subgraphs ─────────────────────────────────────

    #[test]
    fn test_disconnected_subgraphs() {
        let mut g = OperatorGraph::new();
        let a = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let b = g.add_node(Operator::Mul, vec![a], shape(&[4]), DType::F32);
        // Disconnected subgraph.
        let c = g.add_node(Operator::Softmax, vec![], shape(&[8]), DType::F16);
        let d = g.add_node(Operator::Add, vec![c], shape(&[8]), DType::F16);
        g.set_outputs(vec![b, d]);
        let order = g.topological_sort().unwrap();
        assert_eq!(order.len(), 4);
        // a before b, c before d.
        assert!(
            order.iter().position(|&x| x == a).unwrap()
                < order.iter().position(|&x| x == b).unwrap()
        );
        assert!(
            order.iter().position(|&x| x == c).unwrap()
                < order.iter().position(|&x| x == d).unwrap()
        );
    }

    // ── operator display ───────────────────────────────────────────

    #[test]
    fn test_operator_display() {
        assert_eq!(format!("{}", Operator::MatMul), "MatMul");
        assert_eq!(format!("{}", Operator::RMSNorm { eps: 1e-5 }), "RMSNorm(eps=0.00001)");
        assert_eq!(
            format!("{}", Operator::Reshape { target_shape: vec![2, 3] }),
            "Reshape([2, 3])"
        );
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::I2), "i2");
    }

    // ── graph error display ────────────────────────────────────────

    #[test]
    fn test_graph_error_display() {
        assert_eq!(format!("{}", GraphError::CycleDetected), "cycle detected in operator graph");
        assert_eq!(format!("{}", GraphError::NodeNotFound(42)), "node 42 not found");
    }

    // ── property: passes preserve graph validity ───────────────────

    #[test]
    fn test_property_topo_sort_preserved_after_constant_fold() {
        let mut g = OperatorGraph::new();
        let a = g.add_constant(1.0, shape(&[1]), DType::F32);
        let b = g.add_constant(2.0, shape(&[1]), DType::F32);
        let c = g.add_node(Operator::Add, vec![a, b], shape(&[1]), DType::F32);
        let d = g.add_node(Operator::Softmax, vec![c], shape(&[1]), DType::F32);
        g.set_outputs(vec![d]);

        ConstantFoldPass.apply(&mut g).unwrap();
        // Graph must still be acyclic and sortable.
        assert!(!g.has_cycle());
        let order = g.topological_sort().unwrap();
        assert_eq!(order.len(), g.node_count());
    }

    #[test]
    fn test_property_topo_sort_preserved_after_dce() {
        let mut g = simple_matmul_add_graph();
        let _dead = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        DeadCodeEliminationPass.apply(&mut g).unwrap();
        assert!(!g.has_cycle());
        let order = g.topological_sort().unwrap();
        assert_eq!(order.len(), g.node_count());
    }

    #[test]
    fn test_property_topo_sort_preserved_after_fusion() {
        let mut g = simple_matmul_add_graph();
        OperatorFusionPass.apply(&mut g).unwrap();
        assert!(!g.has_cycle());
        let order = g.topological_sort().unwrap();
        assert_eq!(order.len(), g.node_count());
    }

    #[test]
    fn test_property_node_count_nonincreasing_after_dce() {
        let mut g = simple_matmul_add_graph();
        let _dead = g.add_node(Operator::SiLU, vec![], shape(&[4]), DType::F32);
        let before = g.node_count();
        DeadCodeEliminationPass.apply(&mut g).unwrap();
        assert!(g.node_count() <= before);
    }

    #[test]
    fn test_property_node_count_nonincreasing_after_fusion() {
        let mut g = simple_matmul_add_graph();
        let before = g.node_count();
        OperatorFusionPass.apply(&mut g).unwrap();
        assert!(g.node_count() <= before);
    }

    // ── edge cases ─────────────────────────────────────────────────

    #[test]
    fn test_all_operator_variants() {
        let mut g = OperatorGraph::new();
        let ops = vec![
            Operator::MatMul,
            Operator::Add,
            Operator::RMSNorm { eps: 1e-6 },
            Operator::RoPE { base_freq: 10000.0 },
            Operator::Softmax,
            Operator::SiLU,
            Operator::Mul,
            Operator::Gather { axis: 0 },
            Operator::Concat { axis: 1 },
            Operator::Reshape { target_shape: vec![4, 8] },
        ];
        for op in ops {
            g.add_node(op, vec![], shape(&[4]), DType::F32);
        }
        assert_eq!(g.node_count(), 10);
    }

    #[test]
    fn test_all_dtypes() {
        let mut g = OperatorGraph::new();
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8, DType::I2] {
            g.add_node(Operator::SiLU, vec![], shape(&[4]), dtype);
        }
        assert_eq!(g.node_count(), 5);
    }

    #[test]
    fn test_memory_plan_dtype_sizes() {
        let mut g = OperatorGraph::new();
        let f32_node = g.add_node(Operator::SiLU, vec![], shape(&[100]), DType::F32);
        let f16_node = g.add_node(Operator::SiLU, vec![], shape(&[100]), DType::F16);
        g.set_outputs(vec![f32_node, f16_node]);

        let plan = ExecutionPlan::from_graph(&g).unwrap();
        let f32_alloc = plan.memory_plan.iter().find(|a| a.node_id == f32_node).unwrap();
        let f16_alloc = plan.memory_plan.iter().find(|a| a.node_id == f16_node).unwrap();
        assert_eq!(f32_alloc.size_bytes, 400); // 100 * 4
        assert_eq!(f16_alloc.size_bytes, 200); // 100 * 2
    }

    #[test]
    fn test_fused_op_display() {
        assert_eq!(format!("{}", FusedOp::MatMulAdd), "FusedMatMulAdd");
        assert_eq!(format!("{}", FusedOp::RMSNormMul), "FusedRMSNormMul");
        assert_eq!(format!("{}", FusedOp::SiLUMul), "FusedSiLUMul");
    }

    #[test]
    fn test_pass_names() {
        assert_eq!(ConstantFoldPass.name(), "ConstantFold");
        assert_eq!(DeadCodeEliminationPass.name(), "DeadCodeElimination");
        assert_eq!(OperatorFusionPass.name(), "OperatorFusion");
        assert_eq!(LayoutOptimizationPass.name(), "LayoutOptimization");
    }

    #[test]
    fn test_default_graph() {
        let g = OperatorGraph::default();
        assert_eq!(g.node_count(), 0);
    }
}
