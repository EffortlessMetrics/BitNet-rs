//! OpenCL layer fusion optimizer for Intel Arc A770.
//!
//! Fuses consecutive operations into single kernel launches to reduce
//! memory bandwidth. The optimizer scans a directed operation graph
//! ([`OpGraph`]) for fusible patterns and rewrites the graph in-place.
//!
//! # Architecture
//!
//! ```text
//! OpGraph → FusionOptimizer (pattern matching) → fused OpGraph + FusionStats
//! ```
//!
//! [`A770FusionHeuristics`] encodes hardware-specific constraints (SLM budget,
//! register pressure) that gate which fusions are profitable on the A770.
//!
//! # CPU reference
//!
//! Every fused pattern has a scalar CPU reference so correctness can be
//! validated without a GPU. [`FusionValidator`] compares fused vs unfused
//! outputs within a configurable tolerance.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Fusion patterns
// ---------------------------------------------------------------------------

/// Recognised layer-fusion patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// Matmul followed by bias addition.
    MatmulBias,
    /// Matmul followed by an activation (GELU / SiLU).
    MatmulActivation,
    /// RMSNorm followed by a linear projection.
    NormLinear,
    /// Three linear projections (Q, K, V) fused into one.
    QKVProjection,
    /// Gate + up projections for SwiGLU-style FFN.
    GateUp,
    /// Residual add followed by RMSNorm.
    ResidualNorm,
}

impl fmt::Display for FusionPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::MatmulBias => "MatmulBias",
            Self::MatmulActivation => "MatmulActivation",
            Self::NormLinear => "NormLinear",
            Self::QKVProjection => "QKVProjection",
            Self::GateUp => "GateUp",
            Self::ResidualNorm => "ResidualNorm",
        };
        f.write_str(name)
    }
}

impl FusionPattern {
    /// The sequence of [`OpKind`]s that this pattern matches.
    pub fn op_sequence(self) -> Vec<OpKind> {
        match self {
            Self::MatmulBias => vec![OpKind::MatMul, OpKind::BiasAdd],
            Self::MatmulActivation => vec![OpKind::MatMul, OpKind::Activation],
            Self::NormLinear => vec![OpKind::RmsNorm, OpKind::MatMul],
            Self::QKVProjection => vec![OpKind::MatMul, OpKind::MatMul, OpKind::MatMul],
            Self::GateUp => vec![OpKind::MatMul, OpKind::MatMul, OpKind::Activation, OpKind::Mul],
            Self::ResidualNorm => vec![OpKind::Add, OpKind::RmsNorm],
        }
    }
}

// ---------------------------------------------------------------------------
// Operation kinds
// ---------------------------------------------------------------------------

/// Primitive operation kind used in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    MatMul,
    BiasAdd,
    Add,
    Mul,
    Activation,
    RmsNorm,
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ---------------------------------------------------------------------------
// Activation type
// ---------------------------------------------------------------------------

/// Activation function used within fused kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusedActivation {
    SiLU,
    GELU,
    ReLU,
}

impl FusedActivation {
    /// Scalar application of the activation.
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::SiLU => x * sigmoid(x),
            Self::GELU => {
                let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
            }
            Self::ReLU => x.max(0.0),
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Fusion rule
// ---------------------------------------------------------------------------

/// Predicate for when a fusion is valid.
#[derive(Debug, Clone)]
pub struct FusionRule {
    pub pattern: FusionPattern,
    /// Maximum combined output elements before the fusion is rejected.
    pub max_elements: usize,
    /// Required: all fused ops must share this dtype.
    pub require_same_dtype: bool,
}

impl FusionRule {
    pub fn new(pattern: FusionPattern) -> Self {
        Self { pattern, max_elements: usize::MAX, require_same_dtype: true }
    }

    pub fn with_max_elements(mut self, max: usize) -> Self {
        self.max_elements = max;
        self
    }

    /// Check whether a set of nodes satisfies this rule.
    pub fn is_valid(&self, nodes: &[&OpNode]) -> bool {
        if nodes.is_empty() {
            return false;
        }
        let seq = self.pattern.op_sequence();
        if nodes.len() != seq.len() {
            return false;
        }
        for (node, expected) in nodes.iter().zip(seq.iter()) {
            if node.kind != *expected {
                return false;
            }
        }
        if self.require_same_dtype {
            let dt = nodes[0].dtype;
            if !nodes.iter().all(|n| n.dtype == dt) {
                return false;
            }
        }
        let total_elements: usize = nodes.iter().map(|n| n.output_elements()).sum();
        total_elements <= self.max_elements
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Element data type flowing through the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElemDType {
    F32,
    F16,
    I8,
}

impl ElemDType {
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Operation graph
// ---------------------------------------------------------------------------

/// A single operation node in the graph.
#[derive(Debug, Clone)]
pub struct OpNode {
    pub id: u64,
    pub kind: OpKind,
    pub output_shape: Vec<usize>,
    pub dtype: ElemDType,
    /// Activation type for [`OpKind::Activation`] nodes.
    pub activation: Option<FusedActivation>,
}

impl OpNode {
    pub fn output_elements(&self) -> usize {
        self.output_shape.iter().product()
    }
}

/// Directed graph of operations with data dependency edges.
#[derive(Debug, Clone)]
pub struct OpGraph {
    nodes: Vec<OpNode>,
    /// Edges: (from_id, to_id).
    edges: Vec<(u64, u64)>,
    next_id: u64,
}

impl Default for OpGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl OpGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), edges: Vec::new(), next_id: 0 }
    }

    /// Add a node and return its id.
    pub fn add_node(&mut self, kind: OpKind, output_shape: Vec<usize>, dtype: ElemDType) -> u64 {
        self.add_node_full(kind, output_shape, dtype, None)
    }

    /// Add a node with optional activation metadata.
    pub fn add_node_full(
        &mut self,
        kind: OpKind,
        output_shape: Vec<usize>,
        dtype: ElemDType,
        activation: Option<FusedActivation>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(OpNode { id, kind, output_shape, dtype, activation });
        id
    }

    /// Add a directed dependency edge.
    pub fn add_edge(&mut self, from: u64, to: u64) {
        if !self.edges.contains(&(from, to)) {
            self.edges.push((from, to));
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn nodes(&self) -> &[OpNode] {
        &self.nodes
    }

    pub fn edges(&self) -> &[(u64, u64)] {
        &self.edges
    }

    /// Get a node by id.
    pub fn get_node(&self, id: u64) -> Option<&OpNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Return the ids of direct predecessors of `node_id`.
    pub fn predecessors(&self, node_id: u64) -> Vec<u64> {
        self.edges.iter().filter(|(_, to)| *to == node_id).map(|(from, _)| *from).collect()
    }

    /// Return the ids of direct successors of `node_id`.
    pub fn successors(&self, node_id: u64) -> Vec<u64> {
        self.edges.iter().filter(|(from, _)| *from == node_id).map(|(_, to)| *to).collect()
    }

    /// Topological sort (Kahn). Returns `None` if the graph has a cycle.
    pub fn topological_sort(&self) -> Option<Vec<u64>> {
        let ids: HashSet<u64> = self.nodes.iter().map(|n| n.id).collect();
        let mut in_deg: HashMap<u64, usize> = ids.iter().map(|&id| (id, 0)).collect();
        let mut succs: HashMap<u64, Vec<u64>> = HashMap::new();
        for &(from, to) in &self.edges {
            if ids.contains(&from) && ids.contains(&to) {
                *in_deg.entry(to).or_default() += 1;
                succs.entry(from).or_default().push(to);
            }
        }
        let mut queue: VecDeque<u64> = {
            let mut roots: Vec<u64> =
                in_deg.iter().filter(|(_, d)| **d == 0).map(|(&id, _)| id).collect();
            roots.sort_unstable();
            roots.into_iter().collect()
        };
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(n) = queue.pop_front() {
            order.push(n);
            if let Some(s) = succs.get(&n) {
                let mut sorted = s.clone();
                sorted.sort_unstable();
                for &t in &sorted {
                    let d = in_deg.get_mut(&t).unwrap();
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(t);
                    }
                }
            }
        }
        if order.len() == self.nodes.len() { Some(order) } else { None }
    }

    /// Remove a node (and its edges) by id.
    pub fn remove_node(&mut self, id: u64) {
        self.nodes.retain(|n| n.id != id);
        self.edges.retain(|&(f, t)| f != id && t != id);
    }
}

// ---------------------------------------------------------------------------
// Fused kernel
// ---------------------------------------------------------------------------

/// Represents a fused operation with combined OpenCL source.
#[derive(Debug, Clone)]
pub struct FusedKernel {
    pub pattern: FusionPattern,
    /// Ids of the original nodes that were fused.
    pub original_node_ids: Vec<u64>,
    pub opencl_source: String,
    pub kernel_name: String,
    pub output_shape: Vec<usize>,
    pub dtype: ElemDType,
}

impl FusedKernel {
    pub fn estimated_bandwidth_bytes(&self) -> usize {
        let elements: usize = self.output_shape.iter().product();
        // One read + one write of the output tensor.
        elements * self.dtype.size_bytes() * 2
    }
}

// ---------------------------------------------------------------------------
// Fusion stats
// ---------------------------------------------------------------------------

/// Statistics produced by a fusion pass.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionStats {
    pub kernels_before: usize,
    pub kernels_after: usize,
    pub fusions_applied: usize,
    pub estimated_bandwidth_savings: f64,
    pub patterns_matched: Vec<FusionPattern>,
}

impl FusionStats {
    pub fn reduction_ratio(&self) -> f64 {
        if self.kernels_before == 0 {
            return 0.0;
        }
        1.0 - (self.kernels_after as f64 / self.kernels_before as f64)
    }
}

// ---------------------------------------------------------------------------
// A770 fusion heuristics
// ---------------------------------------------------------------------------

/// A770-specific fusion rules encoding SLM budget and register pressure.
#[derive(Debug, Clone)]
pub struct A770FusionHeuristics {
    /// Shared local memory budget per work-group (bytes).
    pub slm_budget_bytes: usize,
    /// Maximum registers per work-item before spilling.
    pub max_registers: usize,
    /// Maximum intermediate elements that may reside in SLM.
    pub max_intermediate_elements: usize,
}

impl Default for A770FusionHeuristics {
    fn default() -> Self {
        Self {
            slm_budget_bytes: 65_536, // 64 KiB SLM per sub-slice
            max_registers: 128,
            max_intermediate_elements: 16_384,
        }
    }
}

impl A770FusionHeuristics {
    /// Check whether a fusion respects hardware limits.
    pub fn is_profitable(&self, pattern: FusionPattern, nodes: &[&OpNode]) -> bool {
        let total_elements: usize = nodes.iter().map(|n| n.output_elements()).sum();
        if total_elements > self.max_intermediate_elements {
            return false;
        }
        let bytes_needed: usize =
            nodes.iter().map(|n| n.output_elements() * n.dtype.size_bytes()).sum();
        if bytes_needed > self.slm_budget_bytes {
            return false;
        }
        // QKV fusion requires 3 matmuls sharing the same input shape.
        if pattern == FusionPattern::QKVProjection {
            if nodes.len() != 3 {
                return false;
            }
            let shape0 = &nodes[0].output_shape;
            if !nodes.iter().all(|n| n.output_shape == *shape0) {
                return false;
            }
        }
        true
    }

    /// Estimated register pressure for a pattern (0.0–1.0).
    pub fn register_pressure(&self, pattern: FusionPattern) -> f64 {
        let regs_used = match pattern {
            FusionPattern::MatmulBias => 16,
            FusionPattern::MatmulActivation => 20,
            FusionPattern::NormLinear => 24,
            FusionPattern::QKVProjection => 48,
            FusionPattern::GateUp => 40,
            FusionPattern::ResidualNorm => 12,
        };
        regs_used as f64 / self.max_registers as f64
    }
}

// ---------------------------------------------------------------------------
// Fusion optimizer
// ---------------------------------------------------------------------------

/// Scans an [`OpGraph`] for fusible patterns and applies fusions.
#[derive(Debug)]
pub struct FusionOptimizer {
    rules: Vec<FusionRule>,
    heuristics: A770FusionHeuristics,
}

impl FusionOptimizer {
    pub fn new(rules: Vec<FusionRule>, heuristics: A770FusionHeuristics) -> Self {
        Self { rules, heuristics }
    }

    /// Create an optimizer with all default A770 rules.
    pub fn a770_default() -> Self {
        let rules = vec![
            FusionRule::new(FusionPattern::MatmulBias),
            FusionRule::new(FusionPattern::MatmulActivation),
            FusionRule::new(FusionPattern::NormLinear),
            FusionRule::new(FusionPattern::QKVProjection),
            FusionRule::new(FusionPattern::GateUp),
            FusionRule::new(FusionPattern::ResidualNorm),
        ];
        Self::new(rules, A770FusionHeuristics::default())
    }

    /// Scan the graph and return all applicable fusions without modifying it.
    pub fn find_fusions(&self, graph: &OpGraph) -> Vec<FusedKernel> {
        let topo = match graph.topological_sort() {
            Some(order) => order,
            None => return Vec::new(),
        };
        let mut fused: Vec<FusedKernel> = Vec::new();
        let mut consumed: HashSet<u64> = HashSet::new();

        for &node_id in &topo {
            if consumed.contains(&node_id) {
                continue;
            }
            for rule in &self.rules {
                if let Some(kernel) = self.try_match_pattern(graph, node_id, rule, &consumed) {
                    for &id in &kernel.original_node_ids {
                        consumed.insert(id);
                    }
                    fused.push(kernel);
                    break;
                }
            }
        }
        fused
    }

    /// Apply fusions and return stats.
    pub fn optimize(&self, graph: &OpGraph) -> (Vec<FusedKernel>, FusionStats) {
        let kernels_before = graph.node_count();
        let fused_kernels = self.find_fusions(graph);
        let fusions_applied = fused_kernels.len();
        let fused_node_count: usize = fused_kernels.iter().map(|k| k.original_node_ids.len()).sum();
        let remaining = kernels_before.saturating_sub(fused_node_count);
        let kernels_after = remaining + fusions_applied;
        let bw_before: usize =
            graph.nodes().iter().map(|n| n.output_elements() * n.dtype.size_bytes() * 2).sum();
        let bw_after: usize =
            fused_kernels.iter().map(|k| k.estimated_bandwidth_bytes()).sum::<usize>()
                + graph
                    .nodes()
                    .iter()
                    .filter(|n| !fused_kernels.iter().any(|k| k.original_node_ids.contains(&n.id)))
                    .map(|n| n.output_elements() * n.dtype.size_bytes() * 2)
                    .sum::<usize>();
        let savings = if bw_before > 0 { 1.0 - (bw_after as f64 / bw_before as f64) } else { 0.0 };
        let patterns: Vec<FusionPattern> = fused_kernels.iter().map(|k| k.pattern).collect();
        let stats = FusionStats {
            kernels_before,
            kernels_after,
            fusions_applied,
            estimated_bandwidth_savings: savings,
            patterns_matched: patterns,
        };
        (fused_kernels, stats)
    }

    // --- internal helpers ---------------------------------------------------

    fn try_match_pattern(
        &self,
        graph: &OpGraph,
        start_id: u64,
        rule: &FusionRule,
        consumed: &HashSet<u64>,
    ) -> Option<FusedKernel> {
        let seq = rule.pattern.op_sequence();
        let chain = self.collect_chain(graph, start_id, &seq, consumed)?;
        let node_refs: Vec<&OpNode> = chain.iter().filter_map(|id| graph.get_node(*id)).collect();
        if node_refs.len() != seq.len() {
            return None;
        }
        if !rule.is_valid(&node_refs) {
            return None;
        }
        if !self.heuristics.is_profitable(rule.pattern, &node_refs) {
            return None;
        }
        let last = node_refs.last().unwrap();
        Some(FusedKernel {
            pattern: rule.pattern,
            original_node_ids: chain,
            opencl_source: fused_kernel_source(rule.pattern),
            kernel_name: format!("fused_{}", rule.pattern.to_string().to_lowercase()),
            output_shape: last.output_shape.clone(),
            dtype: last.dtype,
        })
    }

    /// Walk the graph from `start` collecting a chain of nodes matching `seq`.
    fn collect_chain(
        &self,
        graph: &OpGraph,
        start: u64,
        seq: &[OpKind],
        consumed: &HashSet<u64>,
    ) -> Option<Vec<u64>> {
        if seq.is_empty() {
            return None;
        }
        let node = graph.get_node(start)?;
        if node.kind != seq[0] || consumed.contains(&start) {
            return None;
        }
        // Special case: QKVProjection needs 3 sibling matmuls sharing an input.
        if seq.len() == 3 && seq.iter().all(|s| *s == OpKind::MatMul) {
            return self.collect_qkv_siblings(graph, start, consumed);
        }
        // Special case: GateUp needs two parallel matmuls then activation+mul.
        if seq.len() == 4 {
            return self.collect_gate_up(graph, start, consumed);
        }
        let mut chain = vec![start];
        let mut current = start;
        for expected in &seq[1..] {
            let succs = graph.successors(current);
            let next = succs.iter().find(|&&s| {
                !consumed.contains(&s) && graph.get_node(s).is_some_and(|n| n.kind == *expected)
            });
            match next {
                Some(&nid) => {
                    chain.push(nid);
                    current = nid;
                }
                None => return None,
            }
        }
        Some(chain)
    }

    /// Collect 3 sibling matmuls that share the same predecessor (QKV pattern).
    fn collect_qkv_siblings(
        &self,
        graph: &OpGraph,
        start: u64,
        consumed: &HashSet<u64>,
    ) -> Option<Vec<u64>> {
        let preds = graph.predecessors(start);
        if preds.is_empty() {
            // Accept standalone matmuls if they are siblings through edges.
            return self.collect_sibling_matmuls_from(graph, start, consumed);
        }
        for &pred in &preds {
            let siblings: Vec<u64> = graph
                .successors(pred)
                .into_iter()
                .filter(|s| {
                    !consumed.contains(s)
                        && graph.get_node(*s).is_some_and(|n| n.kind == OpKind::MatMul)
                })
                .collect();
            if siblings.len() >= 3 {
                return Some(siblings[..3].to_vec());
            }
        }
        None
    }

    fn collect_sibling_matmuls_from(
        &self,
        graph: &OpGraph,
        start: u64,
        consumed: &HashSet<u64>,
    ) -> Option<Vec<u64>> {
        let mut matmuls: Vec<u64> = graph
            .nodes()
            .iter()
            .filter(|n| n.kind == OpKind::MatMul && !consumed.contains(&n.id))
            .map(|n| n.id)
            .collect();
        matmuls.sort_unstable();
        if matmuls.len() >= 3 && matmuls.contains(&start) {
            let _idx = matmuls.iter().position(|&id| id == start).unwrap();
            // Take start and the next two.
            let mut group = vec![start];
            for &id in &matmuls {
                if id != start && group.len() < 3 {
                    group.push(id);
                }
            }
            if group.len() == 3 {
                return Some(group);
            }
        }
        None
    }

    /// Collect GateUp pattern: two parallel matmuls → activation → mul.
    fn collect_gate_up(
        &self,
        graph: &OpGraph,
        start: u64,
        consumed: &HashSet<u64>,
    ) -> Option<Vec<u64>> {
        let node = graph.get_node(start)?;
        if node.kind != OpKind::MatMul {
            return None;
        }
        // Find a sibling matmul sharing a predecessor.
        let preds = graph.predecessors(start);
        let sibling = preds.iter().find_map(|&pred| {
            graph.successors(pred).into_iter().find(|&s| {
                s != start
                    && !consumed.contains(&s)
                    && graph.get_node(s).is_some_and(|n| n.kind == OpKind::MatMul)
            })
        })?;
        // One of the matmuls must feed an activation.
        let act_id = graph.successors(start).into_iter().find(|&s| {
            !consumed.contains(&s)
                && graph.get_node(s).is_some_and(|n| n.kind == OpKind::Activation)
        })?;
        // Activation feeds a mul.
        let mul_id = graph.successors(act_id).into_iter().find(|&s| {
            !consumed.contains(&s) && graph.get_node(s).is_some_and(|n| n.kind == OpKind::Mul)
        })?;
        Some(vec![start, sibling, act_id, mul_id])
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel sources for fused patterns
// ---------------------------------------------------------------------------

/// Return the OpenCL C source for a fused pattern.
pub fn fused_kernel_source(pattern: FusionPattern) -> String {
    match pattern {
        FusionPattern::MatmulBias => MATMUL_BIAS_CL.to_string(),
        FusionPattern::MatmulActivation => MATMUL_ACTIVATION_CL.to_string(),
        FusionPattern::NormLinear => NORM_LINEAR_CL.to_string(),
        FusionPattern::QKVProjection => QKV_PROJECTION_CL.to_string(),
        FusionPattern::GateUp => GATE_UP_CL.to_string(),
        FusionPattern::ResidualNorm => RESIDUAL_NORM_CL.to_string(),
    }
}

pub const MATMUL_BIAS_CL: &str = r#"
__kernel void fused_matmulbias(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M, const int K, const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc + bias[col];
}
"#;

pub const MATMUL_ACTIVATION_CL: &str = r#"
float silu_act(float x) { return x / (1.0f + exp(-x)); }
float gelu_act(float x) {
    float c = sqrt(2.0f / 3.14159265f);
    return 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}
__kernel void fused_matmulactivation(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int K, const int N,
    const int activation_type)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    if (activation_type == 0) acc = silu_act(acc);
    else if (activation_type == 1) acc = gelu_act(acc);
    else if (activation_type == 2) acc = max(acc, 0.0f);
    C[row * N + col] = acc;
}
"#;

pub const NORM_LINEAR_CL: &str = r#"
__kernel void fused_normlinear(
    __global const float* x,
    __global const float* norm_weight,
    __global const float* W,
    __global float* out,
    const int seq_len, const int hidden,
    const int out_dim, const float eps)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row >= seq_len || col >= out_dim) return;
    // RMSNorm inline
    float ss = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float v = x[row * hidden + i];
        ss += v * v;
    }
    float rms = sqrt(ss / (float)hidden + eps);
    float acc = 0.0f;
    for (int k = 0; k < hidden; k++) {
        float normed = (x[row * hidden + k] / rms) * norm_weight[k];
        acc += normed * W[k * out_dim + col];
    }
    out[row * out_dim + col] = acc;
}
"#;

pub const QKV_PROJECTION_CL: &str = r#"
__kernel void fused_qkvprojection(
    __global const float* x,
    __global const float* Wq,
    __global const float* Wk,
    __global const float* Wv,
    __global float* Q,
    __global float* K,
    __global float* V,
    const int seq_len, const int hidden, const int head_dim)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row >= seq_len || col >= head_dim) return;
    float q_acc = 0.0f, k_acc = 0.0f, v_acc = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float xi = x[row * hidden + i];
        q_acc += xi * Wq[i * head_dim + col];
        k_acc += xi * Wk[i * head_dim + col];
        v_acc += xi * Wv[i * head_dim + col];
    }
    Q[row * head_dim + col] = q_acc;
    K[row * head_dim + col] = k_acc;
    V[row * head_dim + col] = v_acc;
}
"#;

pub const GATE_UP_CL: &str = r#"
float silu_gate(float x) { return x / (1.0f + exp(-x)); }
__kernel void fused_gateup(
    __global const float* x,
    __global const float* W_gate,
    __global const float* W_up,
    __global float* out,
    const int seq_len, const int hidden, const int intermediate)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row >= seq_len || col >= intermediate) return;
    float gate_val = 0.0f, up_val = 0.0f;
    for (int k = 0; k < hidden; k++) {
        float xk = x[row * hidden + k];
        gate_val += xk * W_gate[k * intermediate + col];
        up_val   += xk * W_up[k * intermediate + col];
    }
    out[row * intermediate + col] = silu_gate(gate_val) * up_val;
}
"#;

pub const RESIDUAL_NORM_CL: &str = r#"
__kernel void fused_residualnorm(
    __global const float* residual,
    __global const float* skip,
    __global const float* norm_weight,
    __global float* out,
    const int seq_len, const int hidden, const float eps)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row >= seq_len || col >= hidden) return;
    float val = residual[row * hidden + col] + skip[row * hidden + col];
    // Compute RMS across the row (needs reduction — simplified here).
    float ss = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float ri = residual[row * hidden + i] + skip[row * hidden + i];
        ss += ri * ri;
    }
    float rms = sqrt(ss / (float)hidden + eps);
    out[row * hidden + col] = (val / rms) * norm_weight[col];
}
"#;

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Matmul + bias: `C[m,n] = A[m,k] @ B[k,n] + bias[n]`.
pub fn matmul_bias_ref(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc + bias[j];
        }
    }
}

/// Matmul + activation: `C[m,n] = act(A[m,k] @ B[k,n])`.
pub fn matmul_activation_ref(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    act: FusedActivation,
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = act.apply(acc);
        }
    }
}

/// RMSNorm + linear: `out[s,o] = (norm(x)[s,:] @ W[:,o])`.
#[allow(clippy::too_many_arguments)]
pub fn norm_linear_ref(
    x: &[f32],
    norm_weight: &[f32],
    w: &[f32],
    out: &mut [f32],
    seq_len: usize,
    hidden: usize,
    out_dim: usize,
    eps: f32,
) {
    let mut normed = vec![0.0_f32; seq_len * hidden];
    for s in 0..seq_len {
        let row = &x[s * hidden..(s + 1) * hidden];
        let ss: f32 = row.iter().map(|v| v * v).sum();
        let rms = (ss / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            normed[s * hidden + i] = (row[i] / rms) * norm_weight[i];
        }
    }
    matmul_ref_inner(&normed, w, out, seq_len, hidden, out_dim);
}

/// QKV projection: 3 matmuls from shared input.
#[allow(clippy::too_many_arguments)]
pub fn qkv_projection_ref(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    seq_len: usize,
    hidden: usize,
    head_dim: usize,
) {
    matmul_ref_inner(x, wq, q, seq_len, hidden, head_dim);
    matmul_ref_inner(x, wk, k, seq_len, hidden, head_dim);
    matmul_ref_inner(x, wv, v, seq_len, hidden, head_dim);
}

/// Gate + up SwiGLU fusion: `out = silu(x @ W_gate) * (x @ W_up)`.
pub fn gate_up_ref(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    out: &mut [f32],
    seq_len: usize,
    hidden: usize,
    intermediate: usize,
) {
    let len = seq_len * intermediate;
    let mut gate = vec![0.0_f32; len];
    let mut up = vec![0.0_f32; len];
    matmul_ref_inner(x, w_gate, &mut gate, seq_len, hidden, intermediate);
    matmul_ref_inner(x, w_up, &mut up, seq_len, hidden, intermediate);
    for i in 0..len {
        out[i] = FusedActivation::SiLU.apply(gate[i]) * up[i];
    }
}

/// Residual + RMSNorm: `out = rms_norm(residual + skip)`.
pub fn residual_norm_ref(
    residual: &[f32],
    skip: &[f32],
    norm_weight: &[f32],
    out: &mut [f32],
    seq_len: usize,
    hidden: usize,
    eps: f32,
) {
    for s in 0..seq_len {
        let base = s * hidden;
        let ss: f32 = (0..hidden)
            .map(|i| {
                let v = residual[base + i] + skip[base + i];
                v * v
            })
            .sum();
        let rms = (ss / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            let v = residual[base + i] + skip[base + i];
            out[base + i] = (v / rms) * norm_weight[i];
        }
    }
}

/// Simple row-major matmul helper.
fn matmul_ref_inner(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion validator
// ---------------------------------------------------------------------------

/// Verifies fused output matches unfused reference within tolerance.
#[derive(Debug, Clone)]
pub struct FusionValidator {
    pub atol: f32,
    pub rtol: f32,
}

impl Default for FusionValidator {
    fn default() -> Self {
        Self { atol: 1e-5, rtol: 1e-4 }
    }
}

impl FusionValidator {
    pub fn new(atol: f32, rtol: f32) -> Self {
        Self { atol, rtol }
    }

    /// Compare two output buffers element-wise.
    /// Returns `Ok(max_diff)` or `Err(FusionValidationError)`.
    pub fn validate(&self, reference: &[f32], fused: &[f32]) -> Result<f32, FusionValidationError> {
        if reference.len() != fused.len() {
            return Err(FusionValidationError::LengthMismatch {
                expected: reference.len(),
                got: fused.len(),
            });
        }
        let mut max_diff: f32 = 0.0;
        for (i, (&r, &f)) in reference.iter().zip(fused.iter()).enumerate() {
            let diff = (r - f).abs();
            let tol = self.atol + self.rtol * r.abs();
            if diff > tol {
                return Err(FusionValidationError::ValueMismatch {
                    index: i,
                    expected: r,
                    got: f,
                    diff,
                    tolerance: tol,
                });
            }
            max_diff = max_diff.max(diff);
        }
        Ok(max_diff)
    }
}

/// Error from fusion validation.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionValidationError {
    LengthMismatch { expected: usize, got: usize },
    ValueMismatch { index: usize, expected: f32, got: f32, diff: f32, tolerance: f32 },
}

impl fmt::Display for FusionValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::ValueMismatch { index, expected, got, diff, tolerance } => {
                write!(
                    f,
                    "value mismatch at index {index}: expected {expected}, got {got} \
                     (diff={diff}, tolerance={tolerance})"
                )
            }
        }
    }
}

impl std::error::Error for FusionValidationError {}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn identity_weight(rows: usize, cols: usize) -> Vec<f32> {
        let mut w = vec![0.0_f32; rows * cols];
        for i in 0..rows.min(cols) {
            w[i * cols + i] = 1.0;
        }
        w
    }

    fn ones(len: usize) -> Vec<f32> {
        vec![1.0_f32; len]
    }

    fn sequential(len: usize) -> Vec<f32> {
        (0..len).map(|i| (i + 1) as f32 * 0.1).collect()
    }

    fn zeros(len: usize) -> Vec<f32> {
        vec![0.0_f32; len]
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    // -----------------------------------------------------------------------
    // FusionPattern tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pattern_display() {
        assert_eq!(FusionPattern::MatmulBias.to_string(), "MatmulBias");
        assert_eq!(FusionPattern::GateUp.to_string(), "GateUp");
        assert_eq!(FusionPattern::ResidualNorm.to_string(), "ResidualNorm");
    }

    #[test]
    fn test_pattern_op_sequences() {
        assert_eq!(FusionPattern::MatmulBias.op_sequence(), vec![OpKind::MatMul, OpKind::BiasAdd]);
        assert_eq!(
            FusionPattern::MatmulActivation.op_sequence(),
            vec![OpKind::MatMul, OpKind::Activation]
        );
        assert_eq!(FusionPattern::NormLinear.op_sequence(), vec![OpKind::RmsNorm, OpKind::MatMul]);
        assert_eq!(
            FusionPattern::QKVProjection.op_sequence(),
            vec![OpKind::MatMul, OpKind::MatMul, OpKind::MatMul]
        );
        assert_eq!(FusionPattern::ResidualNorm.op_sequence(), vec![OpKind::Add, OpKind::RmsNorm]);
    }

    #[test]
    fn test_pattern_eq_and_hash() {
        let mut set = HashSet::new();
        set.insert(FusionPattern::MatmulBias);
        set.insert(FusionPattern::MatmulBias);
        assert_eq!(set.len(), 1);
        set.insert(FusionPattern::GateUp);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_all_patterns_enumerated() {
        let all = [
            FusionPattern::MatmulBias,
            FusionPattern::MatmulActivation,
            FusionPattern::NormLinear,
            FusionPattern::QKVProjection,
            FusionPattern::GateUp,
            FusionPattern::ResidualNorm,
        ];
        assert_eq!(all.len(), 6);
        for p in &all {
            assert!(!p.op_sequence().is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // OpKind tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_opkind_display() {
        assert_eq!(OpKind::MatMul.to_string(), "MatMul");
        assert_eq!(OpKind::RmsNorm.to_string(), "RmsNorm");
    }

    // -----------------------------------------------------------------------
    // FusedActivation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_silu_activation() {
        let v = FusedActivation::SiLU.apply(0.0);
        assert!((v - 0.0).abs() < 1e-6);
        let v = FusedActivation::SiLU.apply(1.0);
        assert!(v > 0.7 && v < 0.8);
    }

    #[test]
    fn test_gelu_activation() {
        let v = FusedActivation::GELU.apply(0.0);
        assert!((v - 0.0).abs() < 1e-6);
        let v = FusedActivation::GELU.apply(1.0);
        assert!(v > 0.8 && v < 0.9);
    }

    #[test]
    fn test_relu_activation() {
        assert_eq!(FusedActivation::ReLU.apply(-1.0), 0.0);
        assert_eq!(FusedActivation::ReLU.apply(0.0), 0.0);
        assert_eq!(FusedActivation::ReLU.apply(1.0), 1.0);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
        let a = sigmoid(2.0);
        let b = sigmoid(-2.0);
        assert!((a + b - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // ElemDType tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(ElemDType::F32.size_bytes(), 4);
        assert_eq!(ElemDType::F16.size_bytes(), 2);
        assert_eq!(ElemDType::I8.size_bytes(), 1);
    }

    // -----------------------------------------------------------------------
    // FusionRule tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rule_valid_matmul_bias() {
        let rule = FusionRule::new(FusionPattern::MatmulBias);
        let n0 = OpNode {
            id: 0,
            kind: OpKind::MatMul,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        let n1 = OpNode {
            id: 1,
            kind: OpKind::BiasAdd,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        assert!(rule.is_valid(&[&n0, &n1]));
    }

    #[test]
    fn test_rule_invalid_wrong_ops() {
        let rule = FusionRule::new(FusionPattern::MatmulBias);
        let n0 = OpNode {
            id: 0,
            kind: OpKind::Add,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        let n1 = OpNode {
            id: 1,
            kind: OpKind::BiasAdd,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        assert!(!rule.is_valid(&[&n0, &n1]));
    }

    #[test]
    fn test_rule_invalid_dtype_mismatch() {
        let rule = FusionRule::new(FusionPattern::MatmulBias);
        let n0 = OpNode {
            id: 0,
            kind: OpKind::MatMul,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        let n1 = OpNode {
            id: 1,
            kind: OpKind::BiasAdd,
            output_shape: vec![4, 8],
            dtype: ElemDType::F16,
            activation: None,
        };
        assert!(!rule.is_valid(&[&n0, &n1]));
    }

    #[test]
    fn test_rule_max_elements_exceeded() {
        let rule = FusionRule::new(FusionPattern::MatmulBias).with_max_elements(10);
        let n0 = OpNode {
            id: 0,
            kind: OpKind::MatMul,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        let n1 = OpNode {
            id: 1,
            kind: OpKind::BiasAdd,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        assert!(!rule.is_valid(&[&n0, &n1])); // 32 + 32 = 64 > 10
    }

    #[test]
    fn test_rule_empty_nodes() {
        let rule = FusionRule::new(FusionPattern::MatmulBias);
        assert!(!rule.is_valid(&[]));
    }

    #[test]
    fn test_rule_wrong_length() {
        let rule = FusionRule::new(FusionPattern::MatmulBias);
        let n0 = OpNode {
            id: 0,
            kind: OpKind::MatMul,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        assert!(!rule.is_valid(&[&n0])); // expects 2 nodes
    }

    // -----------------------------------------------------------------------
    // OpGraph tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_graph_empty() {
        let g = OpGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.topological_sort(), Some(vec![]));
    }

    #[test]
    fn test_graph_default() {
        let g = OpGraph::default();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_graph_add_nodes_and_edges() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.successors(a), vec![b]);
        assert_eq!(g.predecessors(b), vec![a]);
    }

    #[test]
    fn test_graph_duplicate_edge() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4], ElemDType::F32);
        let b = g.add_node(OpKind::Add, vec![4], ElemDType::F32);
        g.add_edge(a, b);
        g.add_edge(a, b); // duplicate
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_graph_topological_sort_linear() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![4, 8], ElemDType::F32);
        let c = g.add_node(OpKind::Activation, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        g.add_edge(b, c);
        assert_eq!(g.topological_sort(), Some(vec![a, b, c]));
    }

    #[test]
    fn test_graph_topological_sort_diamond() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4], ElemDType::F32);
        let b = g.add_node(OpKind::Activation, vec![4], ElemDType::F32);
        let c = g.add_node(OpKind::Mul, vec![4], ElemDType::F32);
        let d = g.add_node(OpKind::Add, vec![4], ElemDType::F32);
        g.add_edge(a, b);
        g.add_edge(a, c);
        g.add_edge(b, d);
        g.add_edge(c, d);
        let order = g.topological_sort().unwrap();
        assert_eq!(order[0], a);
        assert_eq!(order[3], d);
    }

    #[test]
    fn test_graph_cycle_detected() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4], ElemDType::F32);
        let b = g.add_node(OpKind::Add, vec![4], ElemDType::F32);
        g.add_edge(a, b);
        g.add_edge(b, a);
        assert_eq!(g.topological_sort(), None);
    }

    #[test]
    fn test_graph_get_node() {
        let mut g = OpGraph::new();
        let id = g.add_node(OpKind::RmsNorm, vec![2, 4], ElemDType::F32);
        assert!(g.get_node(id).is_some());
        assert!(g.get_node(999).is_none());
    }

    #[test]
    fn test_graph_remove_node() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4], ElemDType::F32);
        let b = g.add_node(OpKind::Add, vec![4], ElemDType::F32);
        g.add_edge(a, b);
        g.remove_node(a);
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_graph_add_node_full() {
        let mut g = OpGraph::new();
        let id = g.add_node_full(
            OpKind::Activation,
            vec![4, 8],
            ElemDType::F32,
            Some(FusedActivation::SiLU),
        );
        let n = g.get_node(id).unwrap();
        assert_eq!(n.activation, Some(FusedActivation::SiLU));
    }

    #[test]
    fn test_node_output_elements() {
        let n = OpNode {
            id: 0,
            kind: OpKind::MatMul,
            output_shape: vec![2, 3, 4],
            dtype: ElemDType::F32,
            activation: None,
        };
        assert_eq!(n.output_elements(), 24);
    }

    // -----------------------------------------------------------------------
    // FusedKernel tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fused_kernel_bandwidth() {
        let k = FusedKernel {
            pattern: FusionPattern::MatmulBias,
            original_node_ids: vec![0, 1],
            opencl_source: String::new(),
            kernel_name: "test".into(),
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
        };
        // 32 elements * 4 bytes * 2 (read+write) = 256
        assert_eq!(k.estimated_bandwidth_bytes(), 256);
    }

    #[test]
    fn test_fused_kernel_bandwidth_f16() {
        let k = FusedKernel {
            pattern: FusionPattern::MatmulBias,
            original_node_ids: vec![0, 1],
            opencl_source: String::new(),
            kernel_name: "test".into(),
            output_shape: vec![4, 8],
            dtype: ElemDType::F16,
        };
        assert_eq!(k.estimated_bandwidth_bytes(), 128);
    }

    // -----------------------------------------------------------------------
    // FusionStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_reduction_ratio() {
        let s = FusionStats {
            kernels_before: 10,
            kernels_after: 6,
            fusions_applied: 2,
            estimated_bandwidth_savings: 0.3,
            patterns_matched: vec![FusionPattern::MatmulBias, FusionPattern::NormLinear],
        };
        assert!((s.reduction_ratio() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_stats_reduction_ratio_zero_before() {
        let s = FusionStats {
            kernels_before: 0,
            kernels_after: 0,
            fusions_applied: 0,
            estimated_bandwidth_savings: 0.0,
            patterns_matched: vec![],
        };
        assert_eq!(s.reduction_ratio(), 0.0);
    }

    #[test]
    fn test_stats_no_fusions() {
        let s = FusionStats {
            kernels_before: 5,
            kernels_after: 5,
            fusions_applied: 0,
            estimated_bandwidth_savings: 0.0,
            patterns_matched: vec![],
        };
        assert!((s.reduction_ratio() - 0.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // A770FusionHeuristics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_heuristics_defaults() {
        let h = A770FusionHeuristics::default();
        assert_eq!(h.slm_budget_bytes, 65_536);
        assert_eq!(h.max_registers, 128);
    }

    #[test]
    fn test_heuristics_profitable_small() {
        let h = A770FusionHeuristics::default();
        let n = OpNode {
            id: 0,
            kind: OpKind::MatMul,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        assert!(h.is_profitable(FusionPattern::MatmulBias, &[&n]));
    }

    #[test]
    fn test_heuristics_rejects_too_large() {
        let h = A770FusionHeuristics { max_intermediate_elements: 10, ..Default::default() };
        let n = OpNode {
            id: 0,
            kind: OpKind::MatMul,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        assert!(!h.is_profitable(FusionPattern::MatmulBias, &[&n]));
    }

    #[test]
    fn test_heuristics_slm_budget_exceeded() {
        let h = A770FusionHeuristics { slm_budget_bytes: 16, ..Default::default() };
        let n = OpNode {
            id: 0,
            kind: OpKind::MatMul,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32, // 32 * 4 = 128 > 16
            activation: None,
        };
        assert!(!h.is_profitable(FusionPattern::MatmulBias, &[&n]));
    }

    #[test]
    fn test_heuristics_qkv_requires_same_shape() {
        let h = A770FusionHeuristics::default();
        let n0 = OpNode {
            id: 0,
            kind: OpKind::MatMul,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        let n1 = OpNode {
            id: 1,
            kind: OpKind::MatMul,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        let n2 = OpNode {
            id: 2,
            kind: OpKind::MatMul,
            output_shape: vec![4, 16], // different
            dtype: ElemDType::F32,
            activation: None,
        };
        assert!(!h.is_profitable(FusionPattern::QKVProjection, &[&n0, &n1, &n2]));
    }

    #[test]
    fn test_heuristics_qkv_same_shape_ok() {
        let h = A770FusionHeuristics::default();
        let make = |id| OpNode {
            id,
            kind: OpKind::MatMul,
            output_shape: vec![4, 8],
            dtype: ElemDType::F32,
            activation: None,
        };
        let nodes: Vec<OpNode> = (0..3).map(make).collect();
        let refs: Vec<&OpNode> = nodes.iter().collect();
        assert!(h.is_profitable(FusionPattern::QKVProjection, &refs));
    }

    #[test]
    fn test_heuristics_register_pressure() {
        let h = A770FusionHeuristics::default();
        let p = h.register_pressure(FusionPattern::MatmulBias);
        assert!(p > 0.0 && p < 1.0);
        let p_qkv = h.register_pressure(FusionPattern::QKVProjection);
        assert!(p_qkv > p);
    }

    // -----------------------------------------------------------------------
    // FusionOptimizer tests — graph pattern identification
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimizer_matmul_bias_fusion() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        let opt = FusionOptimizer::a770_default();
        let fused = opt.find_fusions(&g);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].pattern, FusionPattern::MatmulBias);
        assert_eq!(fused[0].original_node_ids, vec![a, b]);
    }

    #[test]
    fn test_optimizer_matmul_activation_fusion() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let b = g.add_node_full(
            OpKind::Activation,
            vec![4, 8],
            ElemDType::F32,
            Some(FusedActivation::GELU),
        );
        g.add_edge(a, b);
        let opt = FusionOptimizer::a770_default();
        let fused = opt.find_fusions(&g);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].pattern, FusionPattern::MatmulActivation);
    }

    #[test]
    fn test_optimizer_norm_linear_fusion() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::RmsNorm, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::MatMul, vec![4, 16], ElemDType::F32);
        g.add_edge(a, b);
        let opt = FusionOptimizer::a770_default();
        let fused = opt.find_fusions(&g);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].pattern, FusionPattern::NormLinear);
    }

    #[test]
    fn test_optimizer_residual_norm_fusion() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::Add, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::RmsNorm, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        let opt = FusionOptimizer::a770_default();
        let fused = opt.find_fusions(&g);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].pattern, FusionPattern::ResidualNorm);
    }

    #[test]
    fn test_optimizer_no_fusion_incompatible() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::Add, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::Add, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        let opt = FusionOptimizer::a770_default();
        let fused = opt.find_fusions(&g);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_optimizer_single_node_no_fusion() {
        let mut g = OpGraph::new();
        g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let opt = FusionOptimizer::a770_default();
        let fused = opt.find_fusions(&g);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_optimizer_empty_graph() {
        let g = OpGraph::new();
        let opt = FusionOptimizer::a770_default();
        let (fused, stats) = opt.optimize(&g);
        assert!(fused.is_empty());
        assert_eq!(stats.kernels_before, 0);
        assert_eq!(stats.kernels_after, 0);
    }

    #[test]
    fn test_optimizer_stats_accuracy() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![4, 8], ElemDType::F32);
        let c = g.add_node(OpKind::Add, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        g.add_edge(b, c);
        let opt = FusionOptimizer::a770_default();
        let (_, stats) = opt.optimize(&g);
        assert_eq!(stats.kernels_before, 3);
        assert_eq!(stats.fusions_applied, 1);
        // 3 - 2 (fused) + 1 (fused kernel) = 2
        assert_eq!(stats.kernels_after, 2);
    }

    #[test]
    fn test_optimizer_all_fusible() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        let opt = FusionOptimizer::a770_default();
        let (_, stats) = opt.optimize(&g);
        assert_eq!(stats.kernels_before, 2);
        assert_eq!(stats.kernels_after, 1);
        assert_eq!(stats.fusions_applied, 1);
    }

    #[test]
    fn test_optimizer_none_fusible() {
        let mut g = OpGraph::new();
        g.add_node(OpKind::Add, vec![4], ElemDType::F32);
        g.add_node(OpKind::Mul, vec![4], ElemDType::F32);
        // No edges, no patterns.
        let opt = FusionOptimizer::a770_default();
        let (_, stats) = opt.optimize(&g);
        assert_eq!(stats.fusions_applied, 0);
        assert_eq!(stats.kernels_before, stats.kernels_after);
    }

    #[test]
    fn test_optimizer_cyclic_graph_no_crash() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![4], ElemDType::F32);
        g.add_edge(a, b);
        g.add_edge(b, a);
        let opt = FusionOptimizer::a770_default();
        let fused = opt.find_fusions(&g);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_optimizer_bandwidth_savings_positive() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        let opt = FusionOptimizer::a770_default();
        let (_, stats) = opt.optimize(&g);
        assert!(stats.estimated_bandwidth_savings > 0.0);
    }

    #[test]
    fn test_optimizer_multiple_independent_fusions() {
        let mut g = OpGraph::new();
        // Chain 1: MatMul → BiasAdd
        let a = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        // Chain 2: Add → RmsNorm
        let c = g.add_node(OpKind::Add, vec![4, 8], ElemDType::F32);
        let d = g.add_node(OpKind::RmsNorm, vec![4, 8], ElemDType::F32);
        g.add_edge(c, d);
        let opt = FusionOptimizer::a770_default();
        let (fused, stats) = opt.optimize(&g);
        assert_eq!(fused.len(), 2);
        assert_eq!(stats.fusions_applied, 2);
        assert_eq!(stats.kernels_after, 2);
    }

    // -----------------------------------------------------------------------
    // OpenCL kernel source tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_kernel_sources_not_empty() {
        assert!(!MATMUL_BIAS_CL.is_empty());
        assert!(!MATMUL_ACTIVATION_CL.is_empty());
        assert!(!NORM_LINEAR_CL.is_empty());
        assert!(!QKV_PROJECTION_CL.is_empty());
        assert!(!GATE_UP_CL.is_empty());
        assert!(!RESIDUAL_NORM_CL.is_empty());
    }

    #[test]
    fn test_kernel_source_contains_kernel_keyword() {
        for pattern in [
            FusionPattern::MatmulBias,
            FusionPattern::MatmulActivation,
            FusionPattern::NormLinear,
            FusionPattern::QKVProjection,
            FusionPattern::GateUp,
            FusionPattern::ResidualNorm,
        ] {
            let src = fused_kernel_source(pattern);
            assert!(src.contains("__kernel"), "pattern {pattern} missing __kernel");
        }
    }

    #[test]
    fn test_fused_kernel_name_format() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        let opt = FusionOptimizer::a770_default();
        let fused = opt.find_fusions(&g);
        assert_eq!(fused[0].kernel_name, "fused_matmulbias");
    }

    // -----------------------------------------------------------------------
    // CPU reference: matmul_bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_bias_identity() {
        let m = 2;
        let k = 4;
        let n = 4;
        let a = sequential(m * k);
        let b = identity_weight(k, n);
        let bias = zeros(n);
        let mut out = zeros(m * n);
        matmul_bias_ref(&a, &b, &bias, &mut out, m, k, n);
        assert!(approx_eq(&out, &a, 1e-5));
    }

    #[test]
    fn test_matmul_bias_with_bias() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 1×4
        let b = identity_weight(4, 4);
        let bias = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = zeros(4);
        matmul_bias_ref(&a, &b, &bias, &mut out, 1, 4, 4);
        assert!(approx_eq(&out, &[11.0, 22.0, 33.0, 44.0], 1e-5));
    }

    // -----------------------------------------------------------------------
    // CPU reference: matmul_activation
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_activation_relu() {
        let a = vec![-1.0, 2.0, -3.0, 4.0]; // 1×4
        let b = identity_weight(4, 4);
        let mut out = zeros(4);
        matmul_activation_ref(&a, &b, &mut out, 1, 4, 4, FusedActivation::ReLU);
        assert!(approx_eq(&out, &[0.0, 2.0, 0.0, 4.0], 1e-5));
    }

    #[test]
    fn test_matmul_activation_silu() {
        let a = vec![1.0]; // 1×1
        let b = vec![1.0]; // 1×1
        let mut out = zeros(1);
        matmul_activation_ref(&a, &b, &mut out, 1, 1, 1, FusedActivation::SiLU);
        let expected = FusedActivation::SiLU.apply(1.0);
        assert!((out[0] - expected).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // CPU reference: norm_linear
    // -----------------------------------------------------------------------

    #[test]
    fn test_norm_linear_identity() {
        let seq = 2;
        let hidden = 4;
        let x = ones(seq * hidden);
        let nw = ones(hidden);
        let w = identity_weight(hidden, hidden);
        let mut out = zeros(seq * hidden);
        norm_linear_ref(&x, &nw, &w, &mut out, seq, hidden, hidden, 1e-5);
        // After norm of all-ones vector: each element = 1/rms * 1 * 1
        // rms = sqrt(1.0 + eps) ≈ 1.0
        for &v in &out {
            assert!((v - 1.0).abs() < 0.01, "got {v}");
        }
    }

    #[test]
    fn test_norm_linear_scales() {
        let x = vec![2.0, 2.0, 2.0, 2.0]; // 1×4
        let nw = vec![0.5, 0.5, 0.5, 0.5];
        let w = identity_weight(4, 4);
        let mut out = zeros(4);
        norm_linear_ref(&x, &nw, &w, &mut out, 1, 4, 4, 1e-5);
        // rms = 2.0, normed = 1.0, scaled = 0.5
        for &v in &out {
            assert!((v - 0.5).abs() < 0.01, "got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // CPU reference: qkv_projection
    // -----------------------------------------------------------------------

    #[test]
    fn test_qkv_projection_identity() {
        let seq = 2;
        let hidden = 4;
        let head_dim = 4;
        let x = sequential(seq * hidden);
        let w = identity_weight(hidden, head_dim);
        let mut q = zeros(seq * head_dim);
        let mut k = zeros(seq * head_dim);
        let mut v = zeros(seq * head_dim);
        qkv_projection_ref(&x, &w, &w, &w, &mut q, &mut k, &mut v, seq, hidden, head_dim);
        assert!(approx_eq(&q, &k, 1e-5));
        assert!(approx_eq(&k, &v, 1e-5));
        assert!(approx_eq(&q, &x, 1e-5));
    }

    #[test]
    fn test_qkv_projection_different_weights() {
        let x = vec![1.0, 0.0, 0.0, 1.0]; // 1×4
        let wq = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 4×2 — first 2 cols
        let wk = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]; // 4×2 — last 2 cols
        let wv = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]; // 4×2 — uniform
        let mut q = zeros(2);
        let mut k = zeros(2);
        let mut v = zeros(2);
        qkv_projection_ref(&x, &wq, &wk, &wv, &mut q, &mut k, &mut v, 1, 4, 2);
        assert!(approx_eq(&q, &[1.0, 0.0], 1e-5));
        assert!(approx_eq(&k, &[0.0, 1.0], 1e-5));
        assert!(approx_eq(&v, &[1.0, 1.0], 1e-5));
    }

    // -----------------------------------------------------------------------
    // CPU reference: gate_up (SwiGLU)
    // -----------------------------------------------------------------------

    #[test]
    fn test_gate_up_identity() {
        let x = vec![1.0, 0.0]; // 1×2
        let w_gate = identity_weight(2, 2);
        let w_up = identity_weight(2, 2);
        let mut out = zeros(2);
        gate_up_ref(&x, &w_gate, &w_up, &mut out, 1, 2, 2);
        // gate = [1,0], silu(1) ≈ 0.7311, up = [1,0]
        // out = [silu(1)*1, silu(0)*0] = [0.7311, 0]
        assert!((out[0] - FusedActivation::SiLU.apply(1.0)).abs() < 1e-4);
        assert!((out[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_gate_up_all_ones() {
        let seq = 1;
        let hidden = 2;
        let inter = 2;
        let x = ones(seq * hidden);
        let w_gate = ones(hidden * inter);
        let w_up = ones(hidden * inter);
        let mut out = zeros(seq * inter);
        gate_up_ref(&x, &w_gate, &w_up, &mut out, seq, hidden, inter);
        // gate_val = 2.0, up_val = 2.0, silu(2) ≈ 1.762
        let expected = FusedActivation::SiLU.apply(2.0) * 2.0;
        for &v in &out {
            assert!((v - expected).abs() < 0.01, "got {v}, expected {expected}");
        }
    }

    // -----------------------------------------------------------------------
    // CPU reference: residual_norm
    // -----------------------------------------------------------------------

    #[test]
    fn test_residual_norm_zeros() {
        let seq = 1;
        let hidden = 4;
        let residual = zeros(seq * hidden);
        let skip = zeros(seq * hidden);
        let nw = ones(hidden);
        let mut out = zeros(seq * hidden);
        residual_norm_ref(&residual, &skip, &nw, &mut out, seq, hidden, 1e-5);
        for &v in &out {
            assert!((v - 0.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_residual_norm_add_then_norm() {
        let seq = 1;
        let hidden = 4;
        let residual = vec![1.0, 1.0, 1.0, 1.0];
        let skip = vec![1.0, 1.0, 1.0, 1.0];
        let nw = ones(hidden);
        let mut out = zeros(seq * hidden);
        residual_norm_ref(&residual, &skip, &nw, &mut out, seq, hidden, 1e-5);
        // sum = 2.0 each, rms = 2.0, normed = 1.0
        for &v in &out {
            assert!((v - 1.0).abs() < 0.01, "got {v}");
        }
    }

    #[test]
    fn test_residual_norm_multi_seq() {
        let seq = 2;
        let hidden = 2;
        let residual = vec![1.0, 0.0, 0.0, 1.0];
        let skip = vec![1.0, 0.0, 0.0, 1.0];
        let nw = ones(hidden);
        let mut out = zeros(seq * hidden);
        residual_norm_ref(&residual, &skip, &nw, &mut out, seq, hidden, 1e-5);
        // Row 0: [2,0], rms≈sqrt(2), out=[2/sqrt(2), 0] ≈ [1.414, 0]
        assert!((out[0] - 2.0_f32 / 2.0_f32.sqrt()).abs() < 0.01);
        assert!((out[1] - 0.0).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // FusionValidator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validator_exact_match() {
        let v = FusionValidator::default();
        let a = vec![1.0, 2.0, 3.0];
        assert!(v.validate(&a, &a).is_ok());
    }

    #[test]
    fn test_validator_within_tolerance() {
        let v = FusionValidator::new(1e-3, 1e-3);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0001, 2.0001, 3.0001];
        assert!(v.validate(&a, &b).is_ok());
    }

    #[test]
    fn test_validator_exceeds_tolerance() {
        let v = FusionValidator::new(1e-6, 1e-6);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 4.0];
        let err = v.validate(&a, &b).unwrap_err();
        match err {
            FusionValidationError::ValueMismatch { index, .. } => assert_eq!(index, 2),
            _ => panic!("expected ValueMismatch"),
        }
    }

    #[test]
    fn test_validator_length_mismatch() {
        let v = FusionValidator::default();
        let err = v.validate(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(err, FusionValidationError::LengthMismatch { .. }));
    }

    #[test]
    fn test_validator_empty_buffers() {
        let v = FusionValidator::default();
        let max = v.validate(&[], &[]).unwrap();
        assert_eq!(max, 0.0);
    }

    #[test]
    fn test_validator_error_display() {
        let e = FusionValidationError::LengthMismatch { expected: 10, got: 5 };
        let s = e.to_string();
        assert!(s.contains("10"));
        assert!(s.contains("5"));
    }

    // -----------------------------------------------------------------------
    // Property-like: fused output ≈ unfused
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_bias_fused_matches_unfused() {
        let m = 4;
        let k = 8;
        let n = 6;
        let a = sequential(m * k);
        let b = sequential(k * n);
        let bias = sequential(n);
        // Unfused: matmul then add bias.
        let mut mm = zeros(m * n);
        matmul_ref_inner(&a, &b, &mut mm, m, k, n);
        for i in 0..m {
            for j in 0..n {
                mm[i * n + j] += bias[j];
            }
        }
        // Fused.
        let mut fused = zeros(m * n);
        matmul_bias_ref(&a, &b, &bias, &mut fused, m, k, n);
        let v = FusionValidator::default();
        assert!(v.validate(&mm, &fused).is_ok());
    }

    #[test]
    fn test_matmul_gelu_fused_matches_unfused() {
        let m = 2;
        let k = 4;
        let n = 3;
        let a = sequential(m * k);
        let b = sequential(k * n);
        // Unfused.
        let mut mm = zeros(m * n);
        matmul_ref_inner(&a, &b, &mut mm, m, k, n);
        for v in &mut mm {
            *v = FusedActivation::GELU.apply(*v);
        }
        // Fused.
        let mut fused = zeros(m * n);
        matmul_activation_ref(&a, &b, &mut fused, m, k, n, FusedActivation::GELU);
        let v = FusionValidator::default();
        assert!(v.validate(&mm, &fused).is_ok());
    }

    #[test]
    fn test_matmul_silu_fused_matches_unfused() {
        let m = 3;
        let k = 3;
        let n = 3;
        let a = sequential(m * k);
        let b = sequential(k * n);
        let mut mm = zeros(m * n);
        matmul_ref_inner(&a, &b, &mut mm, m, k, n);
        for v in &mut mm {
            *v = FusedActivation::SiLU.apply(*v);
        }
        let mut fused = zeros(m * n);
        matmul_activation_ref(&a, &b, &mut fused, m, k, n, FusedActivation::SiLU);
        let v = FusionValidator::default();
        assert!(v.validate(&mm, &fused).is_ok());
    }

    #[test]
    fn test_norm_linear_fused_matches_unfused() {
        let seq = 2;
        let hidden = 4;
        let out_dim = 4;
        let eps = 1e-5;
        let x = sequential(seq * hidden);
        let nw = ones(hidden);
        let w = sequential(hidden * out_dim);
        // Unfused: norm then matmul.
        let mut normed = zeros(seq * hidden);
        for s in 0..seq {
            let row = &x[s * hidden..(s + 1) * hidden];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let rms = (ss / hidden as f32 + eps).sqrt();
            for i in 0..hidden {
                normed[s * hidden + i] = (row[i] / rms) * nw[i];
            }
        }
        let mut unfused = zeros(seq * out_dim);
        matmul_ref_inner(&normed, &w, &mut unfused, seq, hidden, out_dim);
        // Fused.
        let mut fused = zeros(seq * out_dim);
        norm_linear_ref(&x, &nw, &w, &mut fused, seq, hidden, out_dim, eps);
        let v = FusionValidator::default();
        assert!(v.validate(&unfused, &fused).is_ok());
    }

    #[test]
    fn test_residual_norm_fused_matches_unfused() {
        let seq = 2;
        let hidden = 4;
        let eps = 1e-5;
        let residual = sequential(seq * hidden);
        let skip = sequential(seq * hidden);
        let nw = ones(hidden);
        // Unfused: add then norm.
        let mut added = zeros(seq * hidden);
        for i in 0..seq * hidden {
            added[i] = residual[i] + skip[i];
        }
        let mut unfused = zeros(seq * hidden);
        for s in 0..seq {
            let row = &added[s * hidden..(s + 1) * hidden];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let rms = (ss / hidden as f32 + eps).sqrt();
            for i in 0..hidden {
                unfused[s * hidden + i] = (row[i] / rms) * nw[i];
            }
        }
        // Fused.
        let mut fused = zeros(seq * hidden);
        residual_norm_ref(&residual, &skip, &nw, &mut fused, seq, hidden, eps);
        let v = FusionValidator::default();
        assert!(v.validate(&unfused, &fused).is_ok());
    }

    // -----------------------------------------------------------------------
    // Edge-case & misc tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimizer_with_custom_rules() {
        let rules = vec![FusionRule::new(FusionPattern::MatmulBias)];
        let opt = FusionOptimizer::new(rules, A770FusionHeuristics::default());
        let mut g = OpGraph::new();
        // NormLinear shouldn't match since we only have MatmulBias rule.
        let a = g.add_node(OpKind::RmsNorm, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        let fused = opt.find_fusions(&g);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_large_shape_heuristic_rejection() {
        let h = A770FusionHeuristics { max_intermediate_elements: 100, ..Default::default() };
        let rules = vec![FusionRule::new(FusionPattern::MatmulBias)];
        let opt = FusionOptimizer::new(rules, h);
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![100, 100], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![100, 100], ElemDType::F32);
        g.add_edge(a, b);
        let fused = opt.find_fusions(&g);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_pattern_matched_in_stats() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        let opt = FusionOptimizer::a770_default();
        let (_, stats) = opt.optimize(&g);
        assert!(stats.patterns_matched.contains(&FusionPattern::MatmulBias));
    }

    #[test]
    fn test_fused_kernel_output_shape_propagated() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        let opt = FusionOptimizer::a770_default();
        let fused = opt.find_fusions(&g);
        assert_eq!(fused[0].output_shape, vec![4, 8]);
        assert_eq!(fused[0].dtype, ElemDType::F32);
    }

    #[test]
    fn test_fused_kernel_opencl_source_nonempty() {
        let mut g = OpGraph::new();
        let a = g.add_node(OpKind::MatMul, vec![4, 8], ElemDType::F32);
        let b = g.add_node(OpKind::BiasAdd, vec![4, 8], ElemDType::F32);
        g.add_edge(a, b);
        let opt = FusionOptimizer::a770_default();
        let fused = opt.find_fusions(&g);
        assert!(!fused[0].opencl_source.is_empty());
    }
}
