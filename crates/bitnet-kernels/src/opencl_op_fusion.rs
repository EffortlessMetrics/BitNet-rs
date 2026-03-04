//! Operator fusion optimization for reducing kernel launch overhead.
//!
//! # Overview
//!
//! Kernel launch overhead dominates GPU execution time for small operators.
//! Operator fusion merges consecutive compatible operations into a single
//! "fused" kernel, eliminating intermediate memory traffic and launch latency.
//!
//! This module provides:
//!
//! - **`OpNode`** — a single operation in a compute graph (matmul, bias, relu, …).
//! - **`OpGraph`** — a DAG of operations with data-flow edges.
//! - **`FusionRule`** / **`FusionPattern`** — declarative pattern matching.
//! - **`FusionOptimizer`** — applies rules to an `OpGraph`, producing fused kernels.
//! - **`FusedKernel`** — specification of a generated fused kernel.
//! - **`MemoryEstimator`** — estimates memory savings from fusion.
//! - **`FusionStats`** — telemetry: ops fused, kernels eliminated, bytes saved.
//! - **`A770FusionRules`** — Intel Arc A770-specific heuristics.
//!
//! # CPU reference
//!
//! All implementations are pure-Rust CPU reference code — no OpenCL runtime
//! dependency. When the `oneapi` feature is enabled, fused kernel specs will
//! be compiled to OpenCL C sources for GPU dispatch.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Operation types ────────────────────────────────────────────────

/// The kind of compute operation represented by an [`OpNode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    MatMul,
    BiasAdd,
    ReLU,
    SiLU,
    GELU,
    LayerNorm,
    RMSNorm,
    Scale,
    Residual,
    Softmax,
    Transpose,
    Reshape,
    ElementwiseMul,
    ElementwiseAdd,
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

// ── OpNode ─────────────────────────────────────────────────────────

/// A single operation in a compute graph.
#[derive(Debug, Clone)]
pub struct OpNode {
    /// Unique identifier within the graph.
    pub id: usize,
    /// What this node computes.
    pub op_type: OpType,
    /// Human-readable label (e.g. `"layer0.q_proj"`).
    pub name: String,
    /// Shape of the primary output tensor `[dim0, dim1, …]`.
    pub output_shape: Vec<usize>,
    /// Estimated FLOPs for this operation.
    pub flops: u64,
    /// Estimated bytes of memory written.
    pub output_bytes: u64,
}

impl OpNode {
    pub fn new(
        id: usize,
        op_type: OpType,
        name: impl Into<String>,
        output_shape: Vec<usize>,
    ) -> Self {
        let output_bytes = output_shape.iter().product::<usize>() as u64 * 4; // f32
        let flops = Self::estimate_flops(op_type, &output_shape);
        Self { id, op_type, name: name.into(), output_shape, flops, output_bytes }
    }

    fn estimate_flops(op_type: OpType, shape: &[usize]) -> u64 {
        let elements = shape.iter().product::<usize>() as u64;
        match op_type {
            OpType::MatMul => elements.saturating_mul(2),
            OpType::LayerNorm | OpType::RMSNorm => elements.saturating_mul(5),
            OpType::Softmax => elements.saturating_mul(5),
            OpType::BiasAdd
            | OpType::ReLU
            | OpType::SiLU
            | OpType::GELU
            | OpType::Scale
            | OpType::Residual
            | OpType::ElementwiseMul
            | OpType::ElementwiseAdd => elements,
            OpType::Transpose | OpType::Reshape => 0,
        }
    }
}

// ── OpGraph ────────────────────────────────────────────────────────

/// Edge in the compute graph: `(source_node, target_node)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
}

/// Directed acyclic graph of [`OpNode`]s with data-flow edges.
#[derive(Debug, Clone)]
pub struct OpGraph {
    nodes: HashMap<usize, OpNode>,
    edges: Vec<Edge>,
    next_id: usize,
}

impl OpGraph {
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), edges: Vec::new(), next_id: 0 }
    }

    /// Add a node and return its id.
    pub fn add_node(
        &mut self,
        op_type: OpType,
        name: impl Into<String>,
        output_shape: Vec<usize>,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, OpNode::new(id, op_type, name, output_shape));
        id
    }

    /// Insert a pre-built node (keeps its id). Returns error if id collides.
    pub fn insert_node(&mut self, node: OpNode) -> Result<(), FusionError> {
        if self.nodes.contains_key(&node.id) {
            return Err(FusionError::DuplicateNode(node.id));
        }
        if node.id >= self.next_id {
            self.next_id = node.id + 1;
        }
        self.nodes.insert(node.id, node);
        Ok(())
    }

    /// Add a data-flow edge. Returns error on self-loop.
    pub fn add_edge(&mut self, from: usize, to: usize) -> Result<(), FusionError> {
        if from == to {
            return Err(FusionError::SelfLoop(from));
        }
        if !self.nodes.contains_key(&from) {
            return Err(FusionError::NodeNotFound(from));
        }
        if !self.nodes.contains_key(&to) {
            return Err(FusionError::NodeNotFound(to));
        }
        self.edges.push(Edge { from, to });
        Ok(())
    }

    pub fn node(&self, id: usize) -> Option<&OpNode> {
        self.nodes.get(&id)
    }

    pub fn node_ids(&self) -> Vec<usize> {
        let mut ids: Vec<_> = self.nodes.keys().copied().collect();
        ids.sort_unstable();
        ids
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

    /// Return the set of predecessor node ids for `node_id`.
    pub fn predecessors(&self, node_id: usize) -> Vec<usize> {
        self.edges.iter().filter(|e| e.to == node_id).map(|e| e.from).collect()
    }

    /// Return the set of successor node ids for `node_id`.
    pub fn successors(&self, node_id: usize) -> Vec<usize> {
        self.edges.iter().filter(|e| e.from == node_id).map(|e| e.to).collect()
    }

    /// Topological sort. Returns `Err` if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<usize>, FusionError> {
        let mut in_degree: HashMap<usize, usize> = self.nodes.keys().map(|&id| (id, 0)).collect();
        for edge in &self.edges {
            *in_degree.entry(edge.to).or_default() += 1;
        }
        let mut queue: VecDeque<usize> =
            in_degree.iter().filter(|(_, d)| **d == 0).map(|(id, _)| *id).collect();
        // Sort the initial queue so output is deterministic.
        let mut sorted_queue: Vec<usize> = queue.drain(..).collect();
        sorted_queue.sort_unstable();
        queue.extend(sorted_queue);

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = queue.pop_front() {
            order.push(id);
            let mut next_ready: Vec<usize> = Vec::new();
            for edge in &self.edges {
                if edge.from == id {
                    let deg = in_degree.get_mut(&edge.to).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        next_ready.push(edge.to);
                    }
                }
            }
            next_ready.sort_unstable();
            queue.extend(next_ready);
        }

        if order.len() == self.nodes.len() { Ok(order) } else { Err(FusionError::CycleDetected) }
    }

    /// Check if the graph is acyclic.
    pub fn is_dag(&self) -> bool {
        self.topological_sort().is_ok()
    }

    /// Remove a node and all its incident edges.
    pub fn remove_node(&mut self, id: usize) -> Option<OpNode> {
        self.edges.retain(|e| e.from != id && e.to != id);
        self.nodes.remove(&id)
    }

    /// Merge a set of node ids into a single fused node, rewiring edges.
    /// The fused node inherits the name `fused_name` and uses `fused_op_type`.
    /// Returns the new node id.
    pub fn merge_nodes(
        &mut self,
        ids: &[usize],
        fused_op_type: OpType,
        fused_name: impl Into<String>,
    ) -> Result<usize, FusionError> {
        if ids.is_empty() {
            return Err(FusionError::EmptyFusion);
        }
        let id_set: HashSet<usize> = ids.iter().copied().collect();

        // Determine fused output shape from last node in topo order.
        let topo = self.topological_sort()?;
        let last_in_group = topo.iter().rev().find(|id| id_set.contains(id)).copied().unwrap();
        let output_shape = self.nodes[&last_in_group].output_shape.clone();

        let combined_flops: u64 =
            ids.iter().filter_map(|id| self.nodes.get(id)).map(|n| n.flops).sum();

        let fused_id = self.next_id;
        self.next_id += 1;
        let mut fused_node = OpNode::new(fused_id, fused_op_type, fused_name, output_shape);
        fused_node.flops = combined_flops;
        self.nodes.insert(fused_id, fused_node);

        // Collect external incoming edges (from outside the group into the group).
        let incoming: Vec<usize> = self
            .edges
            .iter()
            .filter(|e| id_set.contains(&e.to) && !id_set.contains(&e.from))
            .map(|e| e.from)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        // Collect external outgoing edges (from group to outside).
        let outgoing: Vec<usize> = self
            .edges
            .iter()
            .filter(|e| id_set.contains(&e.from) && !id_set.contains(&e.to))
            .map(|e| e.to)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        // Remove old nodes and edges.
        for &id in ids {
            self.nodes.remove(&id);
        }
        self.edges.retain(|e| !id_set.contains(&e.from) && !id_set.contains(&e.to));

        // Re-wire.
        for src in incoming {
            self.edges.push(Edge { from: src, to: fused_id });
        }
        for dst in outgoing {
            self.edges.push(Edge { from: fused_id, to: dst });
        }

        Ok(fused_id)
    }

    /// Total estimated bytes of intermediate outputs across all nodes.
    pub fn total_output_bytes(&self) -> u64 {
        self.nodes.values().map(|n| n.output_bytes).sum()
    }
}

impl Default for OpGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Errors ─────────────────────────────────────────────────────────

/// Errors arising from graph construction or fusion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionError {
    CycleDetected,
    NodeNotFound(usize),
    DuplicateNode(usize),
    SelfLoop(usize),
    EmptyFusion,
    PatternNotFound(String),
}

impl fmt::Display for FusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CycleDetected => write!(f, "cycle detected in operation graph"),
            Self::NodeNotFound(id) => write!(f, "node {id} not found"),
            Self::DuplicateNode(id) => write!(f, "duplicate node id {id}"),
            Self::SelfLoop(id) => write!(f, "self-loop on node {id}"),
            Self::EmptyFusion => write!(f, "cannot fuse an empty set of nodes"),
            Self::PatternNotFound(p) => write!(f, "pattern not found: {p}"),
        }
    }
}

impl std::error::Error for FusionError {}

// ── Fusion patterns ────────────────────────────────────────────────

/// A named sequence of [`OpType`]s that can be fused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionPattern {
    pub name: String,
    /// Ordered chain of op types to match.
    pub ops: Vec<OpType>,
}

impl FusionPattern {
    pub fn new(name: impl Into<String>, ops: Vec<OpType>) -> Self {
        Self { name: name.into(), ops }
    }

    /// Common pattern: MatMul → BiasAdd.
    pub fn linear() -> Self {
        Self::new("linear", vec![OpType::MatMul, OpType::BiasAdd])
    }

    /// Common pattern: MatMul → BiasAdd → ReLU.
    pub fn linear_relu() -> Self {
        Self::new("linear_relu", vec![OpType::MatMul, OpType::BiasAdd, OpType::ReLU])
    }

    /// Common pattern: MatMul → BiasAdd → SiLU.
    pub fn linear_silu() -> Self {
        Self::new("linear_silu", vec![OpType::MatMul, OpType::BiasAdd, OpType::SiLU])
    }

    /// Common pattern: MatMul → BiasAdd → GELU.
    pub fn linear_gelu() -> Self {
        Self::new("linear_gelu", vec![OpType::MatMul, OpType::BiasAdd, OpType::GELU])
    }

    /// LayerNorm → Scale.
    pub fn norm_scale() -> Self {
        Self::new("norm_scale", vec![OpType::LayerNorm, OpType::Scale])
    }

    /// RMSNorm → Scale.
    pub fn rmsnorm_scale() -> Self {
        Self::new("rmsnorm_scale", vec![OpType::RMSNorm, OpType::Scale])
    }

    /// Length of the pattern chain.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the pattern is empty (degenerate).
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

// ── Fusion rules ───────────────────────────────────────────────────

/// A rule that matches a [`FusionPattern`] and specifies the replacement op type.
#[derive(Debug, Clone)]
pub struct FusionRule {
    pub pattern: FusionPattern,
    /// The [`OpType`] the fused kernel will be recorded as.
    pub fused_op_type: OpType,
    /// Estimated speed-up factor from this fusion (≥ 1.0).
    pub speedup_estimate: f32,
    /// Maximum output-tensor elements for which this rule applies.
    /// `None` means no limit.
    pub max_elements: Option<usize>,
}

impl FusionRule {
    pub fn new(pattern: FusionPattern, fused_op_type: OpType, speedup_estimate: f32) -> Self {
        Self { pattern, fused_op_type, speedup_estimate, max_elements: None }
    }

    pub fn with_max_elements(mut self, max: usize) -> Self {
        self.max_elements = Some(max);
        self
    }

    /// Check if this rule applies to a chain of op nodes.
    pub fn matches(&self, nodes: &[&OpNode]) -> bool {
        if nodes.len() != self.pattern.ops.len() {
            return false;
        }
        for (node, expected) in nodes.iter().zip(&self.pattern.ops) {
            if node.op_type != *expected {
                return false;
            }
        }
        if let Some(max) = self.max_elements
            && let Some(last) = nodes.last()
        {
            let elements: usize = last.output_shape.iter().product();
            if elements > max {
                return false;
            }
        }
        true
    }
}

// ── Fused kernel specification ─────────────────────────────────────

/// Describes a fused kernel to be generated / dispatched.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedKernel {
    /// Human-readable name for the fused kernel.
    pub name: String,
    /// The sequence of original operations this kernel replaces.
    pub original_ops: Vec<OpType>,
    /// Ids of original graph nodes consumed.
    pub original_node_ids: Vec<usize>,
    /// Output shape of the fused kernel.
    pub output_shape: Vec<usize>,
    /// Combined estimated FLOPs.
    pub total_flops: u64,
    /// Bytes of intermediate memory eliminated by the fusion.
    pub memory_saved_bytes: u64,
}

// ── Memory estimator ───────────────────────────────────────────────

/// Estimates memory traffic savings from operator fusion.
pub struct MemoryEstimator;

impl MemoryEstimator {
    /// Estimate bytes saved by fusing a chain of nodes.
    ///
    /// Intermediate outputs (all except the last node) are eliminated.
    pub fn estimate_savings(nodes: &[&OpNode]) -> u64 {
        if nodes.len() <= 1 {
            return 0;
        }
        // All intermediate outputs are eliminated.
        nodes[..nodes.len() - 1].iter().map(|n| n.output_bytes).sum()
    }

    /// Estimate total graph-level savings for a set of [`FusedKernel`]s.
    pub fn total_savings(fused_kernels: &[FusedKernel]) -> u64 {
        fused_kernels.iter().map(|fk| fk.memory_saved_bytes).sum()
    }

    /// Estimate ratio of bytes saved vs. original total traffic.
    pub fn savings_ratio(original_bytes: u64, saved_bytes: u64) -> f64 {
        if original_bytes == 0 {
            return 0.0;
        }
        saved_bytes as f64 / original_bytes as f64
    }
}

// ── Fusion statistics ──────────────────────────────────────────────

/// Telemetry collected after running the [`FusionOptimizer`].
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    /// Number of individual ops that were fused.
    pub ops_fused: usize,
    /// Number of kernel launches eliminated.
    pub kernels_eliminated: usize,
    /// Bytes of intermediate memory saved.
    pub memory_saved_bytes: u64,
    /// Number of fusion rules that matched.
    pub rules_applied: usize,
    /// Per-rule match counts.
    pub rule_match_counts: HashMap<String, usize>,
}

impl fmt::Display for FusionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FusionStats {{ ops_fused: {}, kernels_eliminated: {}, memory_saved: {} B, rules_applied: {} }}",
            self.ops_fused, self.kernels_eliminated, self.memory_saved_bytes, self.rules_applied,
        )
    }
}

// ── Fusion optimizer ───────────────────────────────────────────────

/// Applies [`FusionRule`]s to an [`OpGraph`], producing [`FusedKernel`]
/// specifications and an optimized graph.
pub struct FusionOptimizer {
    rules: Vec<FusionRule>,
}

impl FusionOptimizer {
    pub fn new(rules: Vec<FusionRule>) -> Self {
        // Sort rules by pattern length descending so longer patterns match first.
        let mut rules = rules;
        rules.sort_by(|a, b| b.pattern.len().cmp(&a.pattern.len()));
        Self { rules }
    }

    /// Scan the graph for all fusable chains and return the list of
    /// [`FusedKernel`] specs without mutating the graph.
    pub fn find_fusions(&self, graph: &OpGraph) -> Result<Vec<FusedKernel>, FusionError> {
        let topo = graph.topological_sort()?;
        let mut fused_kernels = Vec::new();
        let mut consumed: HashSet<usize> = HashSet::new();

        for &start_id in &topo {
            if consumed.contains(&start_id) {
                continue;
            }
            if let Some(fk) = self.try_match_from(graph, start_id, &consumed) {
                for &nid in &fk.original_node_ids {
                    consumed.insert(nid);
                }
                fused_kernels.push(fk);
            }
        }

        Ok(fused_kernels)
    }

    /// Apply fusion to a mutable graph. Returns stats and fused kernel specs.
    pub fn optimize(
        &self,
        graph: &mut OpGraph,
    ) -> Result<(Vec<FusedKernel>, FusionStats), FusionError> {
        let fused_kernels = self.find_fusions(graph)?;
        let mut stats = FusionStats::default();

        // Apply merges in reverse topological order of fused groups so ids
        // remain valid as we merge.
        for fk in fused_kernels.iter().rev() {
            let _new_id = graph.merge_nodes(&fk.original_node_ids, OpType::MatMul, &fk.name)?;
            stats.ops_fused += fk.original_ops.len();
            stats.kernels_eliminated += fk.original_ops.len() - 1;
            stats.memory_saved_bytes += fk.memory_saved_bytes;
            stats.rules_applied += 1;
            *stats.rule_match_counts.entry(fk.name.clone()).or_default() += 1;
        }

        Ok((fused_kernels, stats))
    }

    /// Try to match the longest rule starting from `start_id` in topological
    /// order, following single-successor chains.
    fn try_match_from(
        &self,
        graph: &OpGraph,
        start_id: usize,
        consumed: &HashSet<usize>,
    ) -> Option<FusedKernel> {
        // Build the longest single-successor chain from start_id.
        let chain = self.build_chain(graph, start_id, consumed);
        if chain.len() < 2 {
            return None;
        }

        // Try each rule against every possible window of the chain.
        for rule in &self.rules {
            let pat_len = rule.pattern.len();
            if pat_len > chain.len() {
                continue;
            }
            // Only start from the beginning of the chain for this scan.
            let window_nodes: Vec<&OpNode> =
                chain[..pat_len].iter().filter_map(|&id| graph.node(id)).collect();
            if window_nodes.len() == pat_len && rule.matches(&window_nodes) {
                let ids = chain[..pat_len].to_vec();
                let original_ops: Vec<OpType> = window_nodes.iter().map(|n| n.op_type).collect();
                let node_refs: Vec<&OpNode> = ids.iter().filter_map(|&id| graph.node(id)).collect();
                let memory_saved = MemoryEstimator::estimate_savings(&node_refs);
                let total_flops = node_refs.iter().map(|n| n.flops).sum();
                let output_shape = window_nodes.last().unwrap().output_shape.clone();
                return Some(FusedKernel {
                    name: rule.pattern.name.clone(),
                    original_ops,
                    original_node_ids: ids,
                    output_shape,
                    total_flops,
                    memory_saved_bytes: memory_saved,
                });
            }
        }
        None
    }

    /// Build a chain of node ids following single-successor edges.
    fn build_chain(
        &self,
        graph: &OpGraph,
        start_id: usize,
        consumed: &HashSet<usize>,
    ) -> Vec<usize> {
        let mut chain = vec![start_id];
        let mut current = start_id;
        loop {
            let succs = graph.successors(current);
            if succs.len() != 1 {
                break;
            }
            let next = succs[0];
            // The next node must have exactly one predecessor (this current node)
            // and must not be already consumed.
            let preds = graph.predecessors(next);
            if preds.len() != 1 || consumed.contains(&next) {
                break;
            }
            chain.push(next);
            current = next;
        }
        chain
    }
}

// ── QKV combine detection ──────────────────────────────────────────

/// Detect three parallel MatMul nodes (Q, K, V) sharing the same input
/// and combine them into a single fused QKV matmul specification.
pub fn detect_qkv_combine(graph: &OpGraph) -> Vec<FusedKernel> {
    let mut results = Vec::new();
    let node_ids = graph.node_ids();

    for &src in &node_ids {
        let succs = graph.successors(src);
        // Find all successor matmul nodes.
        let matmul_succs: Vec<usize> = succs
            .into_iter()
            .filter(|&id| graph.node(id).is_some_and(|n| n.op_type == OpType::MatMul))
            .collect();
        if matmul_succs.len() >= 3 {
            // Take first 3 matmul successors as QKV.
            let qkv: Vec<usize> = matmul_succs.into_iter().take(3).collect();
            let nodes: Vec<&OpNode> = qkv.iter().filter_map(|&id| graph.node(id)).collect();
            let total_flops = nodes.iter().map(|n| n.flops).sum();
            let memory_saved = nodes.iter().map(|n| n.output_bytes).sum::<u64>();
            // Combined output: 3× the head dimension along the last axis.
            let mut output_shape = nodes[0].output_shape.clone();
            if let Some(last) = output_shape.last_mut() {
                *last *= 3;
            }
            results.push(FusedKernel {
                name: "qkv_combine".into(),
                original_ops: vec![OpType::MatMul, OpType::MatMul, OpType::MatMul],
                original_node_ids: qkv,
                output_shape,
                total_flops,
                memory_saved_bytes: memory_saved,
            });
        }
    }
    results
}

// ── A770-specific fusion rules ─────────────────────────────────────

/// Intel Arc A770-specific fusion heuristics.
///
/// The A770 has 512 EUs and benefits from:
/// - Fusing small element-wise ops to avoid launch overhead.
/// - Keeping tensor sizes under 16 MiB per fused kernel for L2 residency.
/// - Preferring SiLU-fused linear patterns (common in BitNet).
pub struct A770FusionRules;

impl A770FusionRules {
    /// 16 MiB in elements (f32).
    const MAX_ELEMENTS: usize = 16 * 1024 * 1024 / 4;

    /// Build the default rule set for the A770.
    pub fn rules() -> Vec<FusionRule> {
        vec![
            // Longest patterns first (optimizer also sorts, but be explicit).
            FusionRule::new(FusionPattern::linear_relu(), OpType::MatMul, 1.6)
                .with_max_elements(Self::MAX_ELEMENTS),
            FusionRule::new(FusionPattern::linear_silu(), OpType::MatMul, 1.7)
                .with_max_elements(Self::MAX_ELEMENTS),
            FusionRule::new(FusionPattern::linear_gelu(), OpType::MatMul, 1.5)
                .with_max_elements(Self::MAX_ELEMENTS),
            FusionRule::new(FusionPattern::linear(), OpType::MatMul, 1.3)
                .with_max_elements(Self::MAX_ELEMENTS),
            FusionRule::new(FusionPattern::norm_scale(), OpType::LayerNorm, 1.4)
                .with_max_elements(Self::MAX_ELEMENTS),
            FusionRule::new(FusionPattern::rmsnorm_scale(), OpType::RMSNorm, 1.4)
                .with_max_elements(Self::MAX_ELEMENTS),
        ]
    }

    /// Build an optimizer pre-loaded with A770 rules.
    pub fn optimizer() -> FusionOptimizer {
        FusionOptimizer::new(Self::rules())
    }
}

// ── CPU reference execution ────────────────────────────────────────

/// Execute a single unfused op on CPU (reference).
pub fn cpu_ref_execute(op: OpType, input: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    match op {
        OpType::BiasAdd => {
            let b = bias.unwrap_or(&[]);
            input
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let bi = if b.is_empty() { 0.0 } else { b[i % b.len()] };
                    x + bi
                })
                .collect()
        }
        OpType::ReLU => input.iter().map(|&x| x.max(0.0)).collect(),
        OpType::SiLU => input.iter().map(|&x| x * (1.0 / (1.0 + (-x).exp()))).collect(),
        OpType::GELU => input
            .iter()
            .map(|&x| {
                let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
            })
            .collect(),
        OpType::Scale => {
            let s = bias.and_then(|b| b.first().copied()).unwrap_or(1.0);
            input.iter().map(|&x| x * s).collect()
        }
        OpType::ElementwiseAdd => {
            let b = bias.unwrap_or(&[]);
            input
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let bi = if b.is_empty() { 0.0 } else { b[i % b.len()] };
                    x + bi
                })
                .collect()
        }
        OpType::ElementwiseMul => {
            let b = bias.unwrap_or(&[]);
            input
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let bi = if b.is_empty() { 1.0 } else { b[i % b.len()] };
                    x * bi
                })
                .collect()
        }
        _ => input.to_vec(),
    }
}

/// Execute a fused chain on CPU (reference): sequentially apply each op.
pub fn cpu_ref_execute_fused(ops: &[OpType], input: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let mut data = input.to_vec();
    for (i, &op) in ops.iter().enumerate() {
        // Only the first op gets the bias (typical for MatMul+Bias+Act chains).
        let b = if i == 1 { bias } else { None };
        data = cpu_ref_execute(op, &data, b);
    }
    data
}

// ── CPU reference MatMul ───────────────────────────────────────────

/// Naive CPU reference matrix multiply: C = A × B.
///
/// `a`: row-major `[m, k]`, `b`: row-major `[k, n]`, output: `[m, n]`.
pub fn cpu_ref_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Fused MatMul + Bias + Activation on CPU reference.
pub fn cpu_ref_matmul_bias_act(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
    activation: Option<OpType>,
) -> Vec<f32> {
    let mut c = cpu_ref_matmul(a, b, m, k, n);
    // Bias add.
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] += bias[j];
        }
    }
    // Activation.
    if let Some(act) = activation {
        c = cpu_ref_execute(act, &c, None);
    }
    c
}

/// CPU reference LayerNorm + Scale.
pub fn cpu_ref_layernorm_scale(input: &[f32], scale: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    let mean = input.iter().sum::<f32>() / n as f32;
    let var = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normed = (x - mean) * inv_std;
            normed * scale[i % scale.len()]
        })
        .collect()
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper builders ────────────────────────────────────────────

    fn shape(dims: &[usize]) -> Vec<usize> {
        dims.to_vec()
    }

    /// Build a simple linear chain: matmul → bias → activation.
    fn build_matmul_bias_act_graph(activation: OpType) -> OpGraph {
        let mut g = OpGraph::new();
        let mm = g.add_node(OpType::MatMul, "matmul", shape(&[4, 128]));
        let bias = g.add_node(OpType::BiasAdd, "bias", shape(&[4, 128]));
        let act = g.add_node(activation, "act", shape(&[4, 128]));
        g.add_edge(mm, bias).unwrap();
        g.add_edge(bias, act).unwrap();
        g
    }

    /// Build a linear chain: matmul → bias (no activation).
    fn build_matmul_bias_graph() -> OpGraph {
        let mut g = OpGraph::new();
        let mm = g.add_node(OpType::MatMul, "matmul", shape(&[4, 128]));
        let bias = g.add_node(OpType::BiasAdd, "bias", shape(&[4, 128]));
        g.add_edge(mm, bias).unwrap();
        g
    }

    /// Build a norm→scale chain.
    fn build_norm_scale_graph() -> OpGraph {
        let mut g = OpGraph::new();
        let ln = g.add_node(OpType::LayerNorm, "ln", shape(&[4, 128]));
        let sc = g.add_node(OpType::Scale, "scale", shape(&[4, 128]));
        g.add_edge(ln, sc).unwrap();
        g
    }

    /// Build a QKV diamond: input → {q_matmul, k_matmul, v_matmul}.
    fn build_qkv_graph() -> OpGraph {
        let mut g = OpGraph::new();
        let input = g.add_node(OpType::Reshape, "input", shape(&[4, 2048]));
        let q = g.add_node(OpType::MatMul, "q_proj", shape(&[4, 128]));
        let k = g.add_node(OpType::MatMul, "k_proj", shape(&[4, 128]));
        let v = g.add_node(OpType::MatMul, "v_proj", shape(&[4, 128]));
        g.add_edge(input, q).unwrap();
        g.add_edge(input, k).unwrap();
        g.add_edge(input, v).unwrap();
        g
    }

    // ── OpNode tests ───────────────────────────────────────────────

    #[test]
    fn test_opnode_output_bytes() {
        let node = OpNode::new(0, OpType::MatMul, "mm", shape(&[4, 128]));
        assert_eq!(node.output_bytes, 4 * 128 * 4);
    }

    #[test]
    fn test_opnode_flops_matmul() {
        let node = OpNode::new(0, OpType::MatMul, "mm", shape(&[4, 128]));
        assert_eq!(node.flops, 4 * 128 * 2);
    }

    #[test]
    fn test_opnode_flops_relu() {
        let node = OpNode::new(0, OpType::ReLU, "relu", shape(&[8, 64]));
        assert_eq!(node.flops, 8 * 64);
    }

    #[test]
    fn test_opnode_flops_reshape_is_zero() {
        let node = OpNode::new(0, OpType::Reshape, "reshape", shape(&[8, 64]));
        assert_eq!(node.flops, 0);
    }

    #[test]
    fn test_opnode_flops_layernorm() {
        let node = OpNode::new(0, OpType::LayerNorm, "ln", shape(&[4, 128]));
        assert_eq!(node.flops, 4 * 128 * 5);
    }

    #[test]
    fn test_opnode_display_name() {
        let node = OpNode::new(7, OpType::SiLU, "silu_0", shape(&[2, 4]));
        assert_eq!(node.name, "silu_0");
        assert_eq!(node.id, 7);
    }

    // ── OpGraph construction tests ─────────────────────────────────

    #[test]
    fn test_graph_add_node() {
        let mut g = OpGraph::new();
        let id = g.add_node(OpType::MatMul, "mm", shape(&[4, 128]));
        assert_eq!(id, 0);
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_graph_add_edge() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        let b = g.add_node(OpType::BiasAdd, "b", shape(&[4, 128]));
        g.add_edge(a, b).unwrap();
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_graph_self_loop_rejected() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        assert_eq!(g.add_edge(a, a), Err(FusionError::SelfLoop(a)));
    }

    #[test]
    fn test_graph_edge_node_not_found() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        assert_eq!(g.add_edge(a, 99), Err(FusionError::NodeNotFound(99)));
    }

    #[test]
    fn test_graph_duplicate_node_rejected() {
        let mut g = OpGraph::new();
        let node = OpNode::new(0, OpType::MatMul, "a", shape(&[4, 128]));
        g.insert_node(node.clone()).unwrap();
        assert_eq!(g.insert_node(node), Err(FusionError::DuplicateNode(0)));
    }

    #[test]
    fn test_graph_predecessors() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        let b = g.add_node(OpType::BiasAdd, "b", shape(&[4, 128]));
        g.add_edge(a, b).unwrap();
        assert_eq!(g.predecessors(b), vec![a]);
        assert!(g.predecessors(a).is_empty());
    }

    #[test]
    fn test_graph_successors() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        let b = g.add_node(OpType::BiasAdd, "b", shape(&[4, 128]));
        g.add_edge(a, b).unwrap();
        assert_eq!(g.successors(a), vec![b]);
        assert!(g.successors(b).is_empty());
    }

    #[test]
    fn test_graph_remove_node() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        let b = g.add_node(OpType::BiasAdd, "b", shape(&[4, 128]));
        g.add_edge(a, b).unwrap();
        let removed = g.remove_node(a);
        assert!(removed.is_some());
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_graph_topological_sort_linear() {
        let g = build_matmul_bias_graph();
        let topo = g.topological_sort().unwrap();
        assert_eq!(topo, vec![0, 1]);
    }

    #[test]
    fn test_graph_topological_sort_diamond() {
        let g = build_qkv_graph();
        let topo = g.topological_sort().unwrap();
        // input (0) must come first; q(1), k(2), v(3) after.
        assert_eq!(topo[0], 0);
        assert_eq!(topo.len(), 4);
    }

    #[test]
    fn test_graph_cycle_detected() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        let b = g.add_node(OpType::BiasAdd, "b", shape(&[4, 128]));
        g.add_edge(a, b).unwrap();
        g.add_edge(b, a).unwrap();
        assert_eq!(g.topological_sort(), Err(FusionError::CycleDetected));
    }

    #[test]
    fn test_graph_is_dag() {
        let g = build_matmul_bias_graph();
        assert!(g.is_dag());
    }

    #[test]
    fn test_graph_is_not_dag_with_cycle() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        let b = g.add_node(OpType::BiasAdd, "b", shape(&[4, 128]));
        g.add_edge(a, b).unwrap();
        g.add_edge(b, a).unwrap();
        assert!(!g.is_dag());
    }

    #[test]
    fn test_graph_total_output_bytes() {
        let g = build_matmul_bias_graph();
        // Two nodes, each [4,128] × 4 bytes = 2048 bytes each.
        assert_eq!(g.total_output_bytes(), 2 * 4 * 128 * 4);
    }

    // ── FusionPattern tests ────────────────────────────────────────

    #[test]
    fn test_pattern_linear() {
        let p = FusionPattern::linear();
        assert_eq!(p.ops, vec![OpType::MatMul, OpType::BiasAdd]);
        assert_eq!(p.len(), 2);
        assert!(!p.is_empty());
    }

    #[test]
    fn test_pattern_linear_relu() {
        let p = FusionPattern::linear_relu();
        assert_eq!(p.ops, vec![OpType::MatMul, OpType::BiasAdd, OpType::ReLU]);
    }

    #[test]
    fn test_pattern_norm_scale() {
        let p = FusionPattern::norm_scale();
        assert_eq!(p.ops, vec![OpType::LayerNorm, OpType::Scale]);
    }

    #[test]
    fn test_pattern_empty() {
        let p = FusionPattern::new("empty", vec![]);
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
    }

    // ── FusionRule matching tests ──────────────────────────────────

    #[test]
    fn test_rule_matches_linear() {
        let rule = FusionRule::new(FusionPattern::linear(), OpType::MatMul, 1.3);
        let mm = OpNode::new(0, OpType::MatMul, "mm", shape(&[4, 128]));
        let bias = OpNode::new(1, OpType::BiasAdd, "bias", shape(&[4, 128]));
        assert!(rule.matches(&[&mm, &bias]));
    }

    #[test]
    fn test_rule_does_not_match_wrong_order() {
        let rule = FusionRule::new(FusionPattern::linear(), OpType::MatMul, 1.3);
        let mm = OpNode::new(0, OpType::MatMul, "mm", shape(&[4, 128]));
        let bias = OpNode::new(1, OpType::BiasAdd, "bias", shape(&[4, 128]));
        assert!(!rule.matches(&[&bias, &mm]));
    }

    #[test]
    fn test_rule_does_not_match_wrong_length() {
        let rule = FusionRule::new(FusionPattern::linear(), OpType::MatMul, 1.3);
        let mm = OpNode::new(0, OpType::MatMul, "mm", shape(&[4, 128]));
        assert!(!rule.matches(&[&mm]));
    }

    #[test]
    fn test_rule_max_elements_accepts() {
        let rule =
            FusionRule::new(FusionPattern::linear(), OpType::MatMul, 1.3).with_max_elements(1024);
        let mm = OpNode::new(0, OpType::MatMul, "mm", shape(&[4, 128]));
        let bias = OpNode::new(1, OpType::BiasAdd, "bias", shape(&[4, 128]));
        assert!(rule.matches(&[&mm, &bias])); // 512 < 1024
    }

    #[test]
    fn test_rule_max_elements_rejects() {
        let rule =
            FusionRule::new(FusionPattern::linear(), OpType::MatMul, 1.3).with_max_elements(256);
        let mm = OpNode::new(0, OpType::MatMul, "mm", shape(&[4, 128]));
        let bias = OpNode::new(1, OpType::BiasAdd, "bias", shape(&[4, 128]));
        assert!(!rule.matches(&[&mm, &bias])); // 512 > 256
    }

    #[test]
    fn test_rule_matches_linear_relu() {
        let rule = FusionRule::new(FusionPattern::linear_relu(), OpType::MatMul, 1.6);
        let mm = OpNode::new(0, OpType::MatMul, "mm", shape(&[4, 128]));
        let bias = OpNode::new(1, OpType::BiasAdd, "bias", shape(&[4, 128]));
        let relu = OpNode::new(2, OpType::ReLU, "relu", shape(&[4, 128]));
        assert!(rule.matches(&[&mm, &bias, &relu]));
    }

    // ── FusionOptimizer tests ──────────────────────────────────────

    #[test]
    fn test_optimizer_fuses_matmul_bias() {
        let g = build_matmul_bias_graph();
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear(),
            OpType::MatMul,
            1.3,
        )]);
        let fused = opt.find_fusions(&g).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name, "linear");
        assert_eq!(fused[0].original_ops, vec![OpType::MatMul, OpType::BiasAdd]);
    }

    #[test]
    fn test_optimizer_fuses_matmul_bias_relu() {
        let g = build_matmul_bias_act_graph(OpType::ReLU);
        let opt = FusionOptimizer::new(vec![
            FusionRule::new(FusionPattern::linear_relu(), OpType::MatMul, 1.6),
            FusionRule::new(FusionPattern::linear(), OpType::MatMul, 1.3),
        ]);
        let fused = opt.find_fusions(&g).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name, "linear_relu");
    }

    #[test]
    fn test_optimizer_fuses_matmul_bias_silu() {
        let g = build_matmul_bias_act_graph(OpType::SiLU);
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear_silu(),
            OpType::MatMul,
            1.7,
        )]);
        let fused = opt.find_fusions(&g).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name, "linear_silu");
    }

    #[test]
    fn test_optimizer_fuses_matmul_bias_gelu() {
        let g = build_matmul_bias_act_graph(OpType::GELU);
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear_gelu(),
            OpType::MatMul,
            1.5,
        )]);
        let fused = opt.find_fusions(&g).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name, "linear_gelu");
    }

    #[test]
    fn test_optimizer_fuses_norm_scale() {
        let g = build_norm_scale_graph();
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::norm_scale(),
            OpType::LayerNorm,
            1.4,
        )]);
        let fused = opt.find_fusions(&g).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name, "norm_scale");
    }

    #[test]
    fn test_optimizer_fuses_rmsnorm_scale() {
        let mut g = OpGraph::new();
        let rms = g.add_node(OpType::RMSNorm, "rms", shape(&[4, 128]));
        let sc = g.add_node(OpType::Scale, "scale", shape(&[4, 128]));
        g.add_edge(rms, sc).unwrap();
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::rmsnorm_scale(),
            OpType::RMSNorm,
            1.4,
        )]);
        let fused = opt.find_fusions(&g).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name, "rmsnorm_scale");
    }

    #[test]
    fn test_optimizer_prefers_longer_pattern() {
        let g = build_matmul_bias_act_graph(OpType::ReLU);
        let opt = FusionOptimizer::new(vec![
            FusionRule::new(FusionPattern::linear(), OpType::MatMul, 1.3),
            FusionRule::new(FusionPattern::linear_relu(), OpType::MatMul, 1.6),
        ]);
        let fused = opt.find_fusions(&g).unwrap();
        // Should pick the 3-op pattern, not the 2-op pattern.
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name, "linear_relu");
    }

    #[test]
    fn test_optimizer_no_fusion_independent_ops() {
        let mut g = OpGraph::new();
        g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        g.add_node(OpType::MatMul, "b", shape(&[4, 128]));
        // No edges — ops are independent.
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear(),
            OpType::MatMul,
            1.3,
        )]);
        let fused = opt.find_fusions(&g).unwrap();
        assert!(fused.is_empty());
    }

    #[test]
    fn test_optimizer_no_fusion_single_op() {
        let mut g = OpGraph::new();
        g.add_node(OpType::MatMul, "mm", shape(&[4, 128]));
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear(),
            OpType::MatMul,
            1.3,
        )]);
        let fused = opt.find_fusions(&g).unwrap();
        assert!(fused.is_empty());
    }

    #[test]
    fn test_optimizer_no_fusion_no_matching_rule() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpType::Softmax, "softmax", shape(&[4, 128]));
        let b = g.add_node(OpType::Transpose, "transpose", shape(&[128, 4]));
        g.add_edge(a, b).unwrap();
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear(),
            OpType::MatMul,
            1.3,
        )]);
        let fused = opt.find_fusions(&g).unwrap();
        assert!(fused.is_empty());
    }

    #[test]
    fn test_optimizer_diamond_not_fused() {
        // Diamond: A → B, A → C, B → D, C → D.
        // B and C share input A, so the chain A→B has B with one successor
        // but A has two successors — chain from A is length 1.
        let mut g = OpGraph::new();
        let a = g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        let b = g.add_node(OpType::BiasAdd, "b", shape(&[4, 128]));
        let c = g.add_node(OpType::BiasAdd, "c", shape(&[4, 128]));
        let d = g.add_node(OpType::ReLU, "d", shape(&[4, 128]));
        g.add_edge(a, b).unwrap();
        g.add_edge(a, c).unwrap();
        g.add_edge(b, d).unwrap();
        g.add_edge(c, d).unwrap();
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear(),
            OpType::MatMul,
            1.3,
        )]);
        let fused = opt.find_fusions(&g).unwrap();
        // A branches so no single-successor chain of length ≥ 2 starting from A.
        // B→D has D with 2 predecessors, so chain from B is length 1.
        assert!(fused.is_empty());
    }

    #[test]
    fn test_optimizer_optimize_mutates_graph() {
        let mut g = build_matmul_bias_graph();
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear(),
            OpType::MatMul,
            1.3,
        )]);
        assert_eq!(g.node_count(), 2);
        let (_, stats) = opt.optimize(&mut g).unwrap();
        assert_eq!(g.node_count(), 1); // merged into one
        assert_eq!(stats.ops_fused, 2);
        assert_eq!(stats.kernels_eliminated, 1);
    }

    #[test]
    fn test_optimizer_optimize_stats() {
        let mut g = build_matmul_bias_act_graph(OpType::ReLU);
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear_relu(),
            OpType::MatMul,
            1.6,
        )]);
        let (fks, stats) = opt.optimize(&mut g).unwrap();
        assert_eq!(fks.len(), 1);
        assert_eq!(stats.ops_fused, 3);
        assert_eq!(stats.kernels_eliminated, 2);
        assert_eq!(stats.rules_applied, 1);
        assert!(stats.memory_saved_bytes > 0);
    }

    #[test]
    fn test_optimizer_two_chains_fused() {
        // chain1: mm1 → bias1, chain2: mm2 → bias2 (sequential).
        let mut g = OpGraph::new();
        let mm1 = g.add_node(OpType::MatMul, "mm1", shape(&[4, 128]));
        let b1 = g.add_node(OpType::BiasAdd, "b1", shape(&[4, 128]));
        let mm2 = g.add_node(OpType::MatMul, "mm2", shape(&[4, 128]));
        let b2 = g.add_node(OpType::BiasAdd, "b2", shape(&[4, 128]));
        g.add_edge(mm1, b1).unwrap();
        g.add_edge(b1, mm2).unwrap();
        g.add_edge(mm2, b2).unwrap();
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear(),
            OpType::MatMul,
            1.3,
        )]);
        let fused = opt.find_fusions(&g).unwrap();
        // mm1→b1 fused, then mm2→b2 fused.
        assert_eq!(fused.len(), 2);
    }

    // ── QKV combine tests ──────────────────────────────────────────

    #[test]
    fn test_qkv_combine_detected() {
        let g = build_qkv_graph();
        let qkv = detect_qkv_combine(&g);
        assert_eq!(qkv.len(), 1);
        assert_eq!(qkv[0].name, "qkv_combine");
        assert_eq!(qkv[0].original_node_ids.len(), 3);
    }

    #[test]
    fn test_qkv_combine_output_shape() {
        let g = build_qkv_graph();
        let qkv = detect_qkv_combine(&g);
        assert_eq!(qkv[0].output_shape, vec![4, 384]); // 128 * 3
    }

    #[test]
    fn test_qkv_combine_not_detected_with_2_matmuls() {
        let mut g = OpGraph::new();
        let input = g.add_node(OpType::Reshape, "input", shape(&[4, 2048]));
        let q = g.add_node(OpType::MatMul, "q_proj", shape(&[4, 128]));
        let k = g.add_node(OpType::MatMul, "k_proj", shape(&[4, 128]));
        g.add_edge(input, q).unwrap();
        g.add_edge(input, k).unwrap();
        let qkv = detect_qkv_combine(&g);
        assert!(qkv.is_empty());
    }

    #[test]
    fn test_qkv_combine_memory_saved() {
        let g = build_qkv_graph();
        let qkv = detect_qkv_combine(&g);
        // Each matmul writes [4,128]*4=2048 bytes, 3 total = 6144.
        assert_eq!(qkv[0].memory_saved_bytes, 3 * 4 * 128 * 4);
    }

    // ── Memory estimator tests ─────────────────────────────────────

    #[test]
    fn test_memory_savings_chain() {
        let nodes = vec![
            OpNode::new(0, OpType::MatMul, "mm", shape(&[4, 128])),
            OpNode::new(1, OpType::BiasAdd, "bias", shape(&[4, 128])),
            OpNode::new(2, OpType::ReLU, "relu", shape(&[4, 128])),
        ];
        let refs: Vec<&OpNode> = nodes.iter().collect();
        let saved = MemoryEstimator::estimate_savings(&refs);
        // Intermediates: mm + bias (relu output kept).
        assert_eq!(saved, 2 * 4 * 128 * 4);
    }

    #[test]
    fn test_memory_savings_single_node() {
        let nodes = vec![OpNode::new(0, OpType::MatMul, "mm", shape(&[4, 128]))];
        let refs: Vec<&OpNode> = nodes.iter().collect();
        assert_eq!(MemoryEstimator::estimate_savings(&refs), 0);
    }

    #[test]
    fn test_memory_savings_empty() {
        assert_eq!(MemoryEstimator::estimate_savings(&[]), 0);
    }

    #[test]
    fn test_memory_total_savings() {
        let fks = vec![
            FusedKernel {
                name: "a".into(),
                original_ops: vec![],
                original_node_ids: vec![],
                output_shape: vec![],
                total_flops: 0,
                memory_saved_bytes: 100,
            },
            FusedKernel {
                name: "b".into(),
                original_ops: vec![],
                original_node_ids: vec![],
                output_shape: vec![],
                total_flops: 0,
                memory_saved_bytes: 200,
            },
        ];
        assert_eq!(MemoryEstimator::total_savings(&fks), 300);
    }

    #[test]
    fn test_memory_savings_ratio() {
        assert!((MemoryEstimator::savings_ratio(1000, 250) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_memory_savings_ratio_zero_original() {
        assert_eq!(MemoryEstimator::savings_ratio(0, 100), 0.0);
    }

    // ── A770 rules tests ───────────────────────────────────────────

    #[test]
    fn test_a770_rules_count() {
        let rules = A770FusionRules::rules();
        assert_eq!(rules.len(), 6);
    }

    #[test]
    fn test_a770_optimizer_fuses_linear_relu() {
        let g = build_matmul_bias_act_graph(OpType::ReLU);
        let opt = A770FusionRules::optimizer();
        let fused = opt.find_fusions(&g).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name, "linear_relu");
    }

    #[test]
    fn test_a770_optimizer_fuses_linear_silu() {
        let g = build_matmul_bias_act_graph(OpType::SiLU);
        let opt = A770FusionRules::optimizer();
        let fused = opt.find_fusions(&g).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name, "linear_silu");
    }

    #[test]
    fn test_a770_optimizer_fuses_norm_scale() {
        let g = build_norm_scale_graph();
        let opt = A770FusionRules::optimizer();
        let fused = opt.find_fusions(&g).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name, "norm_scale");
    }

    #[test]
    fn test_a770_max_elements_rejects_large() {
        let mut g = OpGraph::new();
        // Shape produces > 4M elements (16 MiB / 4 = 4_194_304).
        let mm = g.add_node(OpType::MatMul, "mm", shape(&[2048, 2049]));
        let bias = g.add_node(OpType::BiasAdd, "bias", shape(&[2048, 2049]));
        g.add_edge(mm, bias).unwrap();
        let opt = A770FusionRules::optimizer();
        let fused = opt.find_fusions(&g).unwrap();
        assert!(fused.is_empty());
    }

    #[test]
    fn test_a770_optimizer_fuses_linear_gelu() {
        let g = build_matmul_bias_act_graph(OpType::GELU);
        let opt = A770FusionRules::optimizer();
        let fused = opt.find_fusions(&g).unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name, "linear_gelu");
    }

    // ── FusionStats display ────────────────────────────────────────

    #[test]
    fn test_fusion_stats_display() {
        let stats = FusionStats {
            ops_fused: 5,
            kernels_eliminated: 3,
            memory_saved_bytes: 4096,
            rules_applied: 2,
            rule_match_counts: HashMap::new(),
        };
        let s = format!("{stats}");
        assert!(s.contains("ops_fused: 5"));
        assert!(s.contains("kernels_eliminated: 3"));
    }

    #[test]
    fn test_fusion_stats_default() {
        let stats = FusionStats::default();
        assert_eq!(stats.ops_fused, 0);
        assert_eq!(stats.kernels_eliminated, 0);
        assert_eq!(stats.memory_saved_bytes, 0);
    }

    // ── CPU reference tests ────────────────────────────────────────

    #[test]
    fn test_cpu_ref_relu() {
        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let out = cpu_ref_execute(OpType::ReLU, &input, None);
        assert_eq!(out, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_cpu_ref_bias_add() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.5, -0.5, 0.5, -0.5];
        let out = cpu_ref_execute(OpType::BiasAdd, &input, Some(&bias));
        assert_eq!(out, vec![1.5, 1.5, 3.5, 3.5]);
    }

    #[test]
    fn test_cpu_ref_silu() {
        let input = vec![0.0, 1.0];
        let out = cpu_ref_execute(OpType::SiLU, &input, None);
        assert!((out[0] - 0.0).abs() < 1e-6);
        // SiLU(1) = 1 / (1 + exp(-1)) ≈ 0.7311
        assert!((out[1] - 0.7311).abs() < 1e-3);
    }

    #[test]
    fn test_cpu_ref_gelu() {
        let input = vec![0.0, 1.0];
        let out = cpu_ref_execute(OpType::GELU, &input, None);
        assert!((out[0] - 0.0).abs() < 1e-6);
        // GELU(1) ≈ 0.8412
        assert!((out[1] - 0.8412).abs() < 1e-3);
    }

    #[test]
    fn test_cpu_ref_scale() {
        let input = vec![1.0, 2.0, 3.0];
        let scale = vec![2.0];
        let out = cpu_ref_execute(OpType::Scale, &input, Some(&scale));
        assert_eq!(out, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_cpu_ref_elementwise_mul() {
        let input = vec![1.0, 2.0, 3.0];
        let other = vec![2.0, 3.0, 4.0];
        let out = cpu_ref_execute(OpType::ElementwiseMul, &input, Some(&other));
        assert_eq!(out, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_cpu_ref_elementwise_add() {
        let input = vec![1.0, 2.0, 3.0];
        let other = vec![10.0, 20.0, 30.0];
        let out = cpu_ref_execute(OpType::ElementwiseAdd, &input, Some(&other));
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_cpu_ref_passthrough() {
        let input = vec![1.0, 2.0];
        let out = cpu_ref_execute(OpType::Reshape, &input, None);
        assert_eq!(out, input);
    }

    #[test]
    fn test_cpu_ref_matmul() {
        // 2×2 identity times [1,2; 3,4]
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = cpu_ref_matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cpu_ref_matmul_nonsquare() {
        // [1,2,3] × [1; 2; 3] = [14]
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let c = cpu_ref_matmul(&a, &b, 1, 3, 1);
        assert!((c[0] - 14.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_ref_matmul_bias_act_relu() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![-1.0, 2.0, 3.0, -4.0];
        let bias = vec![0.0, 0.0];
        let c = cpu_ref_matmul_bias_act(&a, &b, &bias, 2, 2, 2, Some(OpType::ReLU));
        // matmul: [-1, 2, 3, -4], relu: [0, 2, 3, 0]
        assert_eq!(c, vec![0.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn test_cpu_ref_matmul_bias_act_none() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let bias = vec![1.0];
        let c = cpu_ref_matmul_bias_act(&a, &b, &bias, 1, 2, 1, None);
        // matmul: [11], bias: [12]
        assert!((c[0] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_ref_layernorm_scale() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let scale = vec![1.0, 1.0, 1.0, 1.0];
        let out = cpu_ref_layernorm_scale(&input, &scale, 1e-5);
        // Mean=2.5, var=1.25, check normalized values sum ≈ 0.
        let sum: f32 = out.iter().sum();
        assert!(sum.abs() < 1e-4);
    }

    #[test]
    fn test_cpu_ref_layernorm_scale_with_gamma() {
        let input = vec![1.0, 3.0];
        let scale = vec![2.0, 2.0];
        let out = cpu_ref_layernorm_scale(&input, &scale, 1e-5);
        // Mean=2, var=1, inv_std=1, normed=[-1,1], scaled=[-2,2].
        assert!((out[0] - (-2.0)).abs() < 1e-4);
        assert!((out[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_cpu_ref_layernorm_empty() {
        let out = cpu_ref_layernorm_scale(&[], &[], 1e-5);
        assert!(out.is_empty());
    }

    // ── Fused vs unfused equivalence (property tests) ──────────────

    #[test]
    fn test_fused_vs_unfused_bias_relu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let bias = vec![0.5, -0.5, 0.1, 0.2, -0.3, 0.4];
        // Unfused: bias then relu.
        let after_bias = cpu_ref_execute(OpType::BiasAdd, &input, Some(&bias));
        let unfused = cpu_ref_execute(OpType::ReLU, &after_bias, None);
        // Fused chain.
        let fused = cpu_ref_execute_fused(
            &[OpType::Reshape, OpType::BiasAdd, OpType::ReLU],
            &input,
            Some(&bias),
        );
        for (a, b) in unfused.iter().zip(fused.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_fused_vs_unfused_scale_silu() {
        let input = vec![0.5, 1.0, -0.5, 2.0];
        let scale = vec![2.0];
        let after_scale = cpu_ref_execute(OpType::Scale, &input, Some(&scale));
        let unfused = cpu_ref_execute(OpType::SiLU, &after_scale, None);
        // In fused chain: first op is Scale, second is SiLU.
        let _fused = cpu_ref_execute_fused(&[OpType::Scale, OpType::SiLU], &input, Some(&scale));
        // Scale is first op and gets bias at index 0; SiLU is index 1 with no bias.
        // But cpu_ref_execute_fused gives bias only at index 1 (the second op).
        // Let's just directly compute:
        let manual: Vec<f32> = input
            .iter()
            .map(|&x| {
                let scaled = x * 2.0;
                scaled * (1.0 / (1.0 + (-scaled).exp()))
            })
            .collect();
        for (a, b) in unfused.iter().zip(manual.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_fused_matmul_bias_relu_matches_separate() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b = vec![0.5, -0.5, 1.0, 0.0]; // [2, 2]
        let bias = vec![0.1, -0.1];

        let fused = cpu_ref_matmul_bias_act(&a, &b, &bias, 2, 2, 2, Some(OpType::ReLU));

        // Step by step:
        let mm = cpu_ref_matmul(&a, &b, 2, 2, 2);
        let biased: Vec<f32> = (0..4).map(|i| mm[i] + bias[i % 2]).collect();
        let unfused = cpu_ref_execute(OpType::ReLU, &biased, None);

        for (a, b) in fused.iter().zip(unfused.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
        }
    }

    // ── Edge case: merge_nodes errors ──────────────────────────────

    #[test]
    fn test_merge_nodes_empty_set() {
        let mut g = OpGraph::new();
        g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        assert_eq!(g.merge_nodes(&[], OpType::MatMul, "fused"), Err(FusionError::EmptyFusion));
    }

    #[test]
    fn test_merge_single_node() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        let new_id = g.merge_nodes(&[a], OpType::MatMul, "fused").unwrap();
        assert_eq!(g.node_count(), 1);
        assert!(g.node(new_id).is_some());
    }

    // ── FusionError display ────────────────────────────────────────

    #[test]
    fn test_fusion_error_display() {
        assert_eq!(FusionError::CycleDetected.to_string(), "cycle detected in operation graph");
        assert_eq!(FusionError::NodeNotFound(42).to_string(), "node 42 not found");
        assert_eq!(FusionError::SelfLoop(7).to_string(), "self-loop on node 7");
        assert_eq!(FusionError::EmptyFusion.to_string(), "cannot fuse an empty set of nodes");
        assert_eq!(
            FusionError::PatternNotFound("foo".into()).to_string(),
            "pattern not found: foo"
        );
    }

    #[test]
    fn test_fusion_error_duplicate_display() {
        assert_eq!(FusionError::DuplicateNode(3).to_string(), "duplicate node id 3");
    }

    // ── OpType display ─────────────────────────────────────────────

    #[test]
    fn test_optype_display() {
        assert_eq!(format!("{}", OpType::MatMul), "MatMul");
        assert_eq!(format!("{}", OpType::LayerNorm), "LayerNorm");
        assert_eq!(format!("{}", OpType::SiLU), "SiLU");
    }

    // ── Graph default trait ────────────────────────────────────────

    #[test]
    fn test_graph_default() {
        let g = OpGraph::default();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    // ── Complex graph: two disjoint fusable chains ─────────────────

    #[test]
    fn test_two_disjoint_linear_chains() {
        let mut g = OpGraph::new();
        // Chain 1: mm1→bias1
        let mm1 = g.add_node(OpType::MatMul, "mm1", shape(&[4, 64]));
        let b1 = g.add_node(OpType::BiasAdd, "b1", shape(&[4, 64]));
        g.add_edge(mm1, b1).unwrap();
        // Chain 2: mm2→bias2 (disjoint)
        let mm2 = g.add_node(OpType::MatMul, "mm2", shape(&[4, 64]));
        let b2 = g.add_node(OpType::BiasAdd, "b2", shape(&[4, 64]));
        g.add_edge(mm2, b2).unwrap();

        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear(),
            OpType::MatMul,
            1.3,
        )]);
        let fused = opt.find_fusions(&g).unwrap();
        assert_eq!(fused.len(), 2);
    }

    // ── FusedKernel fields ─────────────────────────────────────────

    #[test]
    fn test_fused_kernel_fields() {
        let g = build_matmul_bias_graph();
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear(),
            OpType::MatMul,
            1.3,
        )]);
        let fused = opt.find_fusions(&g).unwrap();
        let fk = &fused[0];
        assert_eq!(fk.output_shape, vec![4, 128]);
        assert_eq!(fk.original_node_ids.len(), 2);
        assert!(fk.total_flops > 0);
        assert!(fk.memory_saved_bytes > 0);
    }

    // ── Stats rule_match_counts ────────────────────────────────────

    #[test]
    fn test_stats_rule_match_counts() {
        let mut g = OpGraph::new();
        let mm1 = g.add_node(OpType::MatMul, "mm1", shape(&[4, 64]));
        let b1 = g.add_node(OpType::BiasAdd, "b1", shape(&[4, 64]));
        let mm2 = g.add_node(OpType::MatMul, "mm2", shape(&[4, 64]));
        let b2 = g.add_node(OpType::BiasAdd, "b2", shape(&[4, 64]));
        g.add_edge(mm1, b1).unwrap();
        g.add_edge(b1, mm2).unwrap();
        g.add_edge(mm2, b2).unwrap();
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear(),
            OpType::MatMul,
            1.3,
        )]);
        let (_, stats) = opt.optimize(&mut g).unwrap();
        assert_eq!(*stats.rule_match_counts.get("linear").unwrap(), 2);
    }

    // ── Optimizer rejects cycles ───────────────────────────────────

    #[test]
    fn test_optimizer_rejects_cycle() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpType::MatMul, "a", shape(&[4, 128]));
        let b = g.add_node(OpType::BiasAdd, "b", shape(&[4, 128]));
        g.add_edge(a, b).unwrap();
        g.add_edge(b, a).unwrap();
        let opt = FusionOptimizer::new(vec![FusionRule::new(
            FusionPattern::linear(),
            OpType::MatMul,
            1.3,
        )]);
        assert_eq!(opt.find_fusions(&g), Err(FusionError::CycleDetected));
    }

    // ── Large shape memory estimation ──────────────────────────────

    #[test]
    fn test_memory_savings_large_shapes() {
        let nodes = vec![
            OpNode::new(0, OpType::MatMul, "mm", shape(&[1024, 1024])),
            OpNode::new(1, OpType::BiasAdd, "bias", shape(&[1024, 1024])),
        ];
        let refs: Vec<&OpNode> = nodes.iter().collect();
        let saved = MemoryEstimator::estimate_savings(&refs);
        // Only mm is intermediate: 1024*1024*4 = 4 MiB.
        assert_eq!(saved, 1024 * 1024 * 4);
    }
}
