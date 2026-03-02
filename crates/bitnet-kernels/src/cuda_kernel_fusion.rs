//! CUDA kernel fusion planner with pattern matching and bandwidth estimation.
//!
//! Analyses operator dependency graphs to detect fusible patterns
//! (elementwise chains, reduction+elementwise, matmul+bias, norm+activation)
//! and decides whether fusing them is beneficial based on estimated memory
//! bandwidth savings.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Tensor descriptor
// ---------------------------------------------------------------------------

/// Lightweight tensor shape/dtype descriptor used by the fusion planner.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorDescriptor {
    /// Symbolic name (e.g. `"hidden_states"`).
    pub name: String,
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Element size in bytes (e.g. 4 for f32, 2 for f16).
    pub element_bytes: usize,
}

impl TensorDescriptor {
    pub fn new(name: impl Into<String>, shape: Vec<usize>, element_bytes: usize) -> Self {
        Self { name: name.into(), shape, element_bytes }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.element_bytes
    }
}

// ---------------------------------------------------------------------------
// Operator kinds and graph nodes
// ---------------------------------------------------------------------------

/// High-level operator category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    Elementwise,
    Reduction,
    Matmul,
    Normalization,
    Activation,
    BiasAdd,
    Softmax,
    Other,
}

/// One node in the operator dependency graph.
#[derive(Debug, Clone)]
pub struct OpNode {
    pub id: usize,
    pub name: String,
    pub kind: OpKind,
    pub inputs: Vec<TensorDescriptor>,
    pub outputs: Vec<TensorDescriptor>,
}

impl OpNode {
    pub fn new(
        id: usize,
        name: impl Into<String>,
        kind: OpKind,
        inputs: Vec<TensorDescriptor>,
        outputs: Vec<TensorDescriptor>,
    ) -> Self {
        Self { id, name: name.into(), kind, inputs, outputs }
    }

    /// Estimated bytes read by this operator.
    fn input_bytes(&self) -> usize {
        self.inputs.iter().map(TensorDescriptor::size_bytes).sum()
    }

    /// Estimated bytes written by this operator.
    fn output_bytes(&self) -> usize {
        self.outputs.iter().map(TensorDescriptor::size_bytes).sum()
    }

    /// Total memory traffic (read + write).
    pub fn memory_traffic(&self) -> usize {
        self.input_bytes() + self.output_bytes()
    }
}

// ---------------------------------------------------------------------------
// Fusion types and patterns
// ---------------------------------------------------------------------------

/// The kind of fused kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionType {
    /// Chain of element-wise ops (e.g. add → relu → mul).
    Elementwise,
    /// Reduction followed by element-wise (e.g. layernorm → gelu).
    ReduceElementwise,
    /// Matrix multiply fused with bias addition.
    MatmulBias,
    /// Normalization fused with activation (e.g. layernorm + gelu).
    NormActivation,
}

impl fmt::Display for FusionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Elementwise => write!(f, "Elementwise"),
            Self::ReduceElementwise => write!(f, "ReduceElementwise"),
            Self::MatmulBias => write!(f, "MatmulBias"),
            Self::NormActivation => write!(f, "NormActivation"),
        }
    }
}

/// Named fusion patterns that the planner recognises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// Two or more consecutive element-wise operators.
    ElementwiseChain,
    /// Bias addition after matmul / linear projection.
    BiasAdd,
    /// LayerNorm immediately followed by GELU activation.
    LayerNormGelu,
    /// Matmul → bias → activation in sequence.
    MatmulBiasActivation,
}

impl fmt::Display for FusionPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ElementwiseChain => write!(f, "ElementwiseChain"),
            Self::BiasAdd => write!(f, "BiasAdd"),
            Self::LayerNormGelu => write!(f, "LayerNormGelu"),
            Self::MatmulBiasActivation => write!(f, "MatmulBiasActivation"),
        }
    }
}

// ---------------------------------------------------------------------------
// FusedKernel
// ---------------------------------------------------------------------------

/// Describes a fused kernel produced by the planner.
#[derive(Debug, Clone)]
pub struct FusedKernel {
    /// Human-readable description (e.g. `"fused_layernorm_gelu"`).
    pub description: String,
    /// IDs of the original operators that were fused.
    pub fused_op_ids: Vec<usize>,
    /// Combined inputs (external, not intermediate).
    pub inputs: Vec<TensorDescriptor>,
    /// Combined outputs.
    pub outputs: Vec<TensorDescriptor>,
    /// Estimated speed-up factor relative to running the ops separately.
    pub estimated_speedup: f64,
    /// Fusion type category.
    pub fusion_type: FusionType,
    /// The pattern that triggered this fusion.
    pub pattern: FusionPattern,
}

// ---------------------------------------------------------------------------
// FusionPlanner
// ---------------------------------------------------------------------------

/// Minimum estimated speed-up to accept a fusion candidate.
const MIN_SPEEDUP_THRESHOLD: f64 = 1.05;

/// Analyses an operator graph and proposes kernel fusions.
pub struct FusionPlanner {
    /// All operator nodes, keyed by id.
    nodes: HashMap<usize, OpNode>,
    /// Directed edges: `edges[a]` contains all nodes that **depend on** `a`.
    edges: HashMap<usize, Vec<usize>>,
    /// Reverse edges: `rev_edges[b]` contains all nodes that `b` depends on.
    rev_edges: HashMap<usize, Vec<usize>>,
    /// Assumed device memory bandwidth in bytes/s (used for estimation).
    memory_bandwidth_bytes_per_sec: f64,
}

impl FusionPlanner {
    /// Create a new planner with the given bandwidth assumption (bytes/s).
    pub fn new(memory_bandwidth_bytes_per_sec: f64) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            rev_edges: HashMap::new(),
            memory_bandwidth_bytes_per_sec,
        }
    }

    /// Default planner assuming ~900 GB/s (A100 HBM2e).
    pub fn default_a100() -> Self {
        Self::new(900.0 * 1024.0 * 1024.0 * 1024.0)
    }

    // -- graph building -----------------------------------------------------

    /// Add an operator node.
    pub fn add_op(&mut self, node: OpNode) {
        let id = node.id;
        self.nodes.insert(id, node);
        self.edges.entry(id).or_default();
        self.rev_edges.entry(id).or_default();
    }

    /// Declare that `from` must execute before `to`.
    pub fn add_dependency(&mut self, from: usize, to: usize) {
        self.edges.entry(from).or_default().push(to);
        self.rev_edges.entry(to).or_default().push(from);
    }

    // -- graph queries ------------------------------------------------------

    /// Return a topological ordering of the graph, or `None` if cyclic.
    pub fn topological_order(&self) -> Option<Vec<usize>> {
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        for &id in self.nodes.keys() {
            in_degree.entry(id).or_insert(0);
        }
        for successors in self.edges.values() {
            for &s in successors {
                *in_degree.entry(s).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<usize> =
            in_degree.iter().filter(|entry| *entry.1 == 0).map(|entry| *entry.0).collect();
        // Deterministic ordering for tests.
        let mut start: Vec<usize> = queue.drain(..).collect();
        start.sort_unstable();
        queue.extend(start);

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = queue.pop_front() {
            order.push(id);
            if let Some(succs) = self.edges.get(&id) {
                let mut sorted_succs: Vec<usize> = succs.clone();
                sorted_succs.sort_unstable();
                for s in sorted_succs {
                    let deg = in_degree.get_mut(&s).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(s);
                    }
                }
            }
        }

        if order.len() == self.nodes.len() { Some(order) } else { None }
    }

    /// Predecessors (direct dependencies) of `id`.
    pub fn predecessors(&self, id: usize) -> &[usize] {
        self.rev_edges.get(&id).map_or(&[], Vec::as_slice)
    }

    /// Successors (direct dependents) of `id`.
    pub fn successors(&self, id: usize) -> &[usize] {
        self.edges.get(&id).map_or(&[], Vec::as_slice)
    }

    /// True when `id` has exactly one consumer and that consumer has exactly
    /// one producer — the simplest case for safe fusion.
    fn is_single_use_edge(&self, id: usize) -> bool {
        let succs = self.successors(id);
        if succs.len() != 1 {
            return false;
        }
        let consumer = succs[0];
        self.predecessors(consumer).len() == 1
    }

    // -- pattern detection --------------------------------------------------

    /// Walk forward from `start` collecting an unbroken chain of element-wise
    /// single-use edges.
    fn collect_elementwise_chain(&self, start: usize) -> Vec<usize> {
        let mut chain = vec![start];
        let mut cur = start;
        loop {
            let succs = self.successors(cur);
            if succs.len() != 1 {
                break;
            }
            let next = succs[0];
            if self.predecessors(next).len() != 1 {
                break;
            }
            match self.nodes.get(&next) {
                Some(n) if n.kind == OpKind::Elementwise => {
                    chain.push(next);
                    cur = next;
                }
                _ => break,
            }
        }
        chain
    }

    /// Detect all fusion candidates in the graph.
    pub fn detect_candidates(&self) -> Vec<FusedKernel> {
        let mut candidates = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();

        let topo = match self.topological_order() {
            Some(t) => t,
            None => return candidates, // cyclic graph — bail
        };

        for &id in &topo {
            if visited.contains(&id) {
                continue;
            }
            let node = match self.nodes.get(&id) {
                Some(n) => n,
                None => continue,
            };

            // -- MatmulBiasActivation / MatmulBias / BiasAdd ----------------
            if node.kind == OpKind::Matmul && self.is_single_use_edge(id) {
                let bias_id = self.successors(id)[0];
                if let Some(bias_node) = self.nodes.get(&bias_id) {
                    if bias_node.kind == OpKind::BiasAdd {
                        // Check for trailing activation.
                        if self.is_single_use_edge(bias_id) {
                            let act_id = self.successors(bias_id)[0];
                            if let Some(act_node) = self.nodes.get(&act_id) {
                                if act_node.kind == OpKind::Activation {
                                    if let Some(fk) = self.try_fuse(
                                        &[id, bias_id, act_id],
                                        FusionPattern::MatmulBiasActivation,
                                        FusionType::MatmulBias,
                                    ) {
                                        visited.extend([id, bias_id, act_id]);
                                        candidates.push(fk);
                                        continue;
                                    }
                                }
                            }
                        }
                        // Matmul + bias only.
                        if let Some(fk) = self.try_fuse(
                            &[id, bias_id],
                            FusionPattern::BiasAdd,
                            FusionType::MatmulBias,
                        ) {
                            visited.extend([id, bias_id]);
                            candidates.push(fk);
                            continue;
                        }
                    }
                }
            }

            // -- LayerNormGelu -----------------------------------------------
            if node.kind == OpKind::Normalization && self.is_single_use_edge(id) {
                let act_id = self.successors(id)[0];
                if let Some(act_node) = self.nodes.get(&act_id) {
                    if act_node.kind == OpKind::Activation {
                        if let Some(fk) = self.try_fuse(
                            &[id, act_id],
                            FusionPattern::LayerNormGelu,
                            FusionType::NormActivation,
                        ) {
                            visited.extend([id, act_id]);
                            candidates.push(fk);
                            continue;
                        }
                    }
                }
            }

            // -- Reduction + elementwise -------------------------------------
            if node.kind == OpKind::Reduction && self.is_single_use_edge(id) {
                let next_id = self.successors(id)[0];
                if let Some(next_node) = self.nodes.get(&next_id) {
                    if next_node.kind == OpKind::Elementwise {
                        if let Some(fk) = self.try_fuse(
                            &[id, next_id],
                            FusionPattern::ElementwiseChain,
                            FusionType::ReduceElementwise,
                        ) {
                            visited.extend([id, next_id]);
                            candidates.push(fk);
                            continue;
                        }
                    }
                }
            }

            // -- Elementwise chains ------------------------------------------
            if node.kind == OpKind::Elementwise {
                let chain = self.collect_elementwise_chain(id);
                if chain.len() >= 2 {
                    if let Some(fk) = self.try_fuse(
                        &chain,
                        FusionPattern::ElementwiseChain,
                        FusionType::Elementwise,
                    ) {
                        visited.extend(&chain);
                        candidates.push(fk);
                        continue;
                    }
                }
            }
        }

        candidates
    }

    // -- bandwidth estimation -----------------------------------------------

    /// Estimate memory traffic (bytes) when running `ops` *without* fusion.
    pub fn estimate_unfused_traffic(&self, op_ids: &[usize]) -> usize {
        op_ids.iter().filter_map(|id| self.nodes.get(id)).map(OpNode::memory_traffic).sum()
    }

    /// Estimate memory traffic (bytes) when running `ops` as a single fused
    /// kernel.  We assume intermediates are kept in registers / shared memory
    /// and only the external inputs of the first op and external outputs of the
    /// last op hit global memory.
    pub fn estimate_fused_traffic(&self, op_ids: &[usize]) -> usize {
        if op_ids.is_empty() {
            return 0;
        }
        let first = op_ids[0];
        let last = *op_ids.last().unwrap();

        let input_bytes: usize = self
            .nodes
            .get(&first)
            .map(|n| n.inputs.iter().map(TensorDescriptor::size_bytes).sum())
            .unwrap_or(0);

        let output_bytes: usize = self
            .nodes
            .get(&last)
            .map(|n| n.outputs.iter().map(TensorDescriptor::size_bytes).sum())
            .unwrap_or(0);

        input_bytes + output_bytes
    }

    /// Estimated speed-up from fusing `op_ids`.
    pub fn estimated_speedup(&self, op_ids: &[usize]) -> f64 {
        let unfused = self.estimate_unfused_traffic(op_ids) as f64;
        let fused = self.estimate_fused_traffic(op_ids) as f64;
        if fused == 0.0 {
            return 1.0;
        }
        unfused / fused
    }

    /// Estimated time saving in seconds from fusing `op_ids`.
    pub fn estimated_time_saving_secs(&self, op_ids: &[usize]) -> f64 {
        let unfused = self.estimate_unfused_traffic(op_ids) as f64;
        let fused = self.estimate_fused_traffic(op_ids) as f64;
        (unfused - fused) / self.memory_bandwidth_bytes_per_sec
    }

    // -- fusion decision ----------------------------------------------------

    /// Try to build a `FusedKernel` for the given op IDs if beneficial.
    fn try_fuse(
        &self,
        op_ids: &[usize],
        pattern: FusionPattern,
        fusion_type: FusionType,
    ) -> Option<FusedKernel> {
        let speedup = self.estimated_speedup(op_ids);
        if speedup < MIN_SPEEDUP_THRESHOLD {
            return None;
        }

        let first = self.nodes.get(&op_ids[0])?;
        let last = self.nodes.get(op_ids.last()?)?;

        let description = format!(
            "fused_{}_{}",
            pattern,
            op_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join("_"),
        );

        Some(FusedKernel {
            description,
            fused_op_ids: op_ids.to_vec(),
            inputs: first.inputs.clone(),
            outputs: last.outputs.clone(),
            estimated_speedup: speedup,
            fusion_type,
            pattern,
        })
    }

    /// Run full planning: detect candidates that pass the threshold.
    pub fn plan(&self) -> Vec<FusedKernel> {
        self.detect_candidates()
    }

    /// Number of operator nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of directed edges.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(Vec::len).sum()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ------------------------------------------------------------

    fn td(name: &str, numel: usize) -> TensorDescriptor {
        TensorDescriptor::new(name, vec![numel], 4) // f32
    }

    fn td_2d(name: &str, rows: usize, cols: usize) -> TensorDescriptor {
        TensorDescriptor::new(name, vec![rows, cols], 4)
    }

    fn ew_op(id: usize, name: &str, numel: usize) -> OpNode {
        OpNode::new(id, name, OpKind::Elementwise, vec![td("in", numel)], vec![td("out", numel)])
    }

    fn matmul_op(id: usize, m: usize, n: usize, k: usize) -> OpNode {
        OpNode::new(
            id,
            "matmul",
            OpKind::Matmul,
            vec![td_2d("a", m, k), td_2d("b", k, n)],
            vec![td_2d("c", m, n)],
        )
    }

    fn bias_op(id: usize, numel: usize) -> OpNode {
        OpNode::new(id, "bias_add", OpKind::BiasAdd, vec![td("x", numel)], vec![td("y", numel)])
    }

    fn act_op(id: usize, numel: usize) -> OpNode {
        OpNode::new(
            id,
            "activation",
            OpKind::Activation,
            vec![td("in", numel)],
            vec![td("out", numel)],
        )
    }

    fn norm_op(id: usize, numel: usize) -> OpNode {
        OpNode::new(
            id,
            "layernorm",
            OpKind::Normalization,
            vec![td("in", numel)],
            vec![td("out", numel)],
        )
    }

    fn reduction_op(id: usize, in_numel: usize, out_numel: usize) -> OpNode {
        OpNode::new(
            id,
            "reduce",
            OpKind::Reduction,
            vec![td("in", in_numel)],
            vec![td("out", out_numel)],
        )
    }

    // -- TensorDescriptor tests ---------------------------------------------

    #[test]
    fn tensor_descriptor_numel() {
        let t = TensorDescriptor::new("x", vec![2, 3, 4], 4);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn tensor_descriptor_size_bytes() {
        let t = TensorDescriptor::new("x", vec![1024], 2);
        assert_eq!(t.size_bytes(), 2048);
    }

    #[test]
    fn tensor_descriptor_scalar() {
        let t = TensorDescriptor::new("s", vec![1], 4);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.size_bytes(), 4);
    }

    #[test]
    fn tensor_descriptor_equality() {
        let a = TensorDescriptor::new("x", vec![10], 4);
        let b = TensorDescriptor::new("x", vec![10], 4);
        assert_eq!(a, b);
    }

    // -- OpNode tests -------------------------------------------------------

    #[test]
    fn op_node_memory_traffic() {
        let op = ew_op(0, "relu", 1024);
        // input 1024*4 + output 1024*4
        assert_eq!(op.memory_traffic(), 1024 * 4 * 2);
    }

    // -- graph building / topo order ----------------------------------------

    #[test]
    fn empty_graph_topological_order() {
        let planner = FusionPlanner::default_a100();
        assert_eq!(planner.topological_order(), Some(vec![]));
    }

    #[test]
    fn single_node_topological_order() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "relu", 1024));
        assert_eq!(planner.topological_order(), Some(vec![0]));
    }

    #[test]
    fn linear_chain_topological_order() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "a", 1024));
        planner.add_op(ew_op(1, "b", 1024));
        planner.add_op(ew_op(2, "c", 1024));
        planner.add_dependency(0, 1);
        planner.add_dependency(1, 2);
        assert_eq!(planner.topological_order(), Some(vec![0, 1, 2]));
    }

    #[test]
    fn diamond_topological_order() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "src", 1024));
        planner.add_op(ew_op(1, "left", 1024));
        planner.add_op(ew_op(2, "right", 1024));
        planner.add_op(ew_op(3, "sink", 1024));
        planner.add_dependency(0, 1);
        planner.add_dependency(0, 2);
        planner.add_dependency(1, 3);
        planner.add_dependency(2, 3);
        let order = planner.topological_order().unwrap();
        assert_eq!(order[0], 0);
        assert_eq!(order[3], 3);
    }

    #[test]
    fn cycle_detected() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "a", 1024));
        planner.add_op(ew_op(1, "b", 1024));
        planner.add_dependency(0, 1);
        planner.add_dependency(1, 0);
        assert!(planner.topological_order().is_none());
    }

    #[test]
    fn predecessors_and_successors() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "a", 1024));
        planner.add_op(ew_op(1, "b", 1024));
        planner.add_dependency(0, 1);
        assert_eq!(planner.successors(0), &[1]);
        assert_eq!(planner.predecessors(1), &[0]);
        assert!(planner.predecessors(0).is_empty());
    }

    #[test]
    fn node_and_edge_counts() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "a", 1024));
        planner.add_op(ew_op(1, "b", 1024));
        planner.add_dependency(0, 1);
        assert_eq!(planner.node_count(), 2);
        assert_eq!(planner.edge_count(), 1);
    }

    // -- elementwise chain detection ----------------------------------------

    #[test]
    fn detect_elementwise_chain_two() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "add", 4096));
        planner.add_op(ew_op(1, "relu", 4096));
        planner.add_dependency(0, 1);
        let candidates = planner.detect_candidates();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].pattern, FusionPattern::ElementwiseChain);
        assert_eq!(candidates[0].fusion_type, FusionType::Elementwise);
    }

    #[test]
    fn detect_elementwise_chain_three() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "add", 4096));
        planner.add_op(ew_op(1, "mul", 4096));
        planner.add_op(ew_op(2, "relu", 4096));
        planner.add_dependency(0, 1);
        planner.add_dependency(1, 2);
        let candidates = planner.detect_candidates();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].fused_op_ids, vec![0, 1, 2]);
    }

    #[test]
    fn elementwise_chain_speedup_increasing_with_length() {
        let mut p2 = FusionPlanner::default_a100();
        p2.add_op(ew_op(0, "a", 4096));
        p2.add_op(ew_op(1, "b", 4096));
        p2.add_dependency(0, 1);
        let s2 = p2.estimated_speedup(&[0, 1]);

        let mut p3 = FusionPlanner::default_a100();
        p3.add_op(ew_op(0, "a", 4096));
        p3.add_op(ew_op(1, "b", 4096));
        p3.add_op(ew_op(2, "c", 4096));
        p3.add_dependency(0, 1);
        p3.add_dependency(1, 2);
        let s3 = p3.estimated_speedup(&[0, 1, 2]);

        assert!(s3 > s2, "longer chain should yield higher speedup");
    }

    #[test]
    fn no_fusion_for_multi_consumer() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "a", 4096));
        planner.add_op(ew_op(1, "b", 4096));
        planner.add_op(ew_op(2, "c", 4096));
        planner.add_dependency(0, 1);
        planner.add_dependency(0, 2); // 0 has two consumers
        let candidates = planner.detect_candidates();
        assert!(candidates.is_empty());
    }

    // -- matmul + bias patterns ---------------------------------------------

    #[test]
    fn detect_matmul_bias() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(matmul_op(0, 128, 256, 512));
        planner.add_op(bias_op(1, 128 * 256));
        planner.add_dependency(0, 1);
        let candidates = planner.detect_candidates();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].pattern, FusionPattern::BiasAdd);
        assert_eq!(candidates[0].fusion_type, FusionType::MatmulBias);
    }

    #[test]
    fn detect_matmul_bias_activation() {
        let mut planner = FusionPlanner::default_a100();
        let n = 128 * 256;
        planner.add_op(matmul_op(0, 128, 256, 512));
        planner.add_op(bias_op(1, n));
        planner.add_op(act_op(2, n));
        planner.add_dependency(0, 1);
        planner.add_dependency(1, 2);
        let candidates = planner.detect_candidates();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].pattern, FusionPattern::MatmulBiasActivation);
        assert_eq!(candidates[0].fused_op_ids, vec![0, 1, 2]);
    }

    // -- norm + activation --------------------------------------------------

    #[test]
    fn detect_layernorm_gelu() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(norm_op(0, 4096));
        planner.add_op(act_op(1, 4096));
        planner.add_dependency(0, 1);
        let candidates = planner.detect_candidates();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].pattern, FusionPattern::LayerNormGelu);
        assert_eq!(candidates[0].fusion_type, FusionType::NormActivation);
    }

    // -- reduction + elementwise --------------------------------------------

    #[test]
    fn detect_reduce_elementwise() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(reduction_op(0, 4096, 128));
        planner.add_op(ew_op(1, "scale", 128));
        planner.add_dependency(0, 1);
        let candidates = planner.detect_candidates();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].fusion_type, FusionType::ReduceElementwise);
    }

    // -- bandwidth estimation -----------------------------------------------

    #[test]
    fn unfused_traffic_sum() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "a", 1024));
        planner.add_op(ew_op(1, "b", 1024));
        planner.add_dependency(0, 1);
        // Each op: 1024*4 in + 1024*4 out = 8192
        assert_eq!(planner.estimate_unfused_traffic(&[0, 1]), 8192 * 2);
    }

    #[test]
    fn fused_traffic_less_than_unfused() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "a", 1024));
        planner.add_op(ew_op(1, "b", 1024));
        planner.add_dependency(0, 1);
        let unfused = planner.estimate_unfused_traffic(&[0, 1]);
        let fused = planner.estimate_fused_traffic(&[0, 1]);
        assert!(fused < unfused);
    }

    #[test]
    fn speedup_for_two_elementwise() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "a", 1024));
        planner.add_op(ew_op(1, "b", 1024));
        planner.add_dependency(0, 1);
        let speedup = planner.estimated_speedup(&[0, 1]);
        // 2 * (4096+4096) / (4096+4096) = 2.0
        assert!((speedup - 2.0).abs() < 1e-6);
    }

    #[test]
    fn time_saving_positive() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "a", 1024));
        planner.add_op(ew_op(1, "b", 1024));
        planner.add_dependency(0, 1);
        assert!(planner.estimated_time_saving_secs(&[0, 1]) > 0.0);
    }

    #[test]
    fn empty_op_ids_speedup_is_one() {
        let planner = FusionPlanner::default_a100();
        assert!((planner.estimated_speedup(&[]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fused_traffic_empty_is_zero() {
        let planner = FusionPlanner::default_a100();
        assert_eq!(planner.estimate_fused_traffic(&[]), 0);
    }

    // -- plan (end-to-end) --------------------------------------------------

    #[test]
    fn plan_returns_beneficial_fusions_only() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "a", 4096));
        planner.add_op(ew_op(1, "b", 4096));
        planner.add_dependency(0, 1);
        let plan = planner.plan();
        assert!(!plan.is_empty());
        for fk in &plan {
            assert!(fk.estimated_speedup >= MIN_SPEEDUP_THRESHOLD);
        }
    }

    #[test]
    fn plan_on_cyclic_graph_returns_empty() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "a", 4096));
        planner.add_op(ew_op(1, "b", 4096));
        planner.add_dependency(0, 1);
        planner.add_dependency(1, 0);
        assert!(planner.plan().is_empty());
    }

    // -- FusionType / FusionPattern Display ---------------------------------

    #[test]
    fn fusion_type_display() {
        assert_eq!(FusionType::Elementwise.to_string(), "Elementwise");
        assert_eq!(FusionType::ReduceElementwise.to_string(), "ReduceElementwise");
        assert_eq!(FusionType::MatmulBias.to_string(), "MatmulBias");
        assert_eq!(FusionType::NormActivation.to_string(), "NormActivation");
    }

    #[test]
    fn fusion_pattern_display() {
        assert_eq!(FusionPattern::ElementwiseChain.to_string(), "ElementwiseChain");
        assert_eq!(FusionPattern::BiasAdd.to_string(), "BiasAdd");
        assert_eq!(FusionPattern::LayerNormGelu.to_string(), "LayerNormGelu");
        assert_eq!(FusionPattern::MatmulBiasActivation.to_string(), "MatmulBiasActivation");
    }

    // -- mixed graph --------------------------------------------------------

    #[test]
    fn mixed_graph_multiple_fusions() {
        let mut planner = FusionPlanner::default_a100();
        // Branch 1: matmul → bias → act
        let n = 128 * 256;
        planner.add_op(matmul_op(0, 128, 256, 512));
        planner.add_op(bias_op(1, n));
        planner.add_op(act_op(2, n));
        planner.add_dependency(0, 1);
        planner.add_dependency(1, 2);
        // Branch 2: norm → act (independent)
        planner.add_op(norm_op(10, 4096));
        planner.add_op(act_op(11, 4096));
        planner.add_dependency(10, 11);
        let candidates = planner.plan();
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn isolated_node_no_fusion() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "lonely", 1024));
        assert!(planner.plan().is_empty());
    }

    #[test]
    fn fused_kernel_description_contains_pattern() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(norm_op(0, 4096));
        planner.add_op(act_op(1, 4096));
        planner.add_dependency(0, 1);
        let fk = &planner.plan()[0];
        assert!(fk.description.contains("LayerNormGelu"));
    }

    #[test]
    fn fused_kernel_inputs_from_first_op() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "first", 4096));
        planner.add_op(ew_op(1, "second", 4096));
        planner.add_dependency(0, 1);
        let fk = &planner.plan()[0];
        assert_eq!(fk.inputs.len(), 1);
        assert_eq!(fk.inputs[0].name, "in");
    }

    #[test]
    fn fused_kernel_outputs_from_last_op() {
        let mut planner = FusionPlanner::default_a100();
        planner.add_op(ew_op(0, "first", 4096));
        planner.add_op(ew_op(1, "second", 4096));
        planner.add_dependency(0, 1);
        let fk = &planner.plan()[0];
        assert_eq!(fk.outputs.len(), 1);
        assert_eq!(fk.outputs[0].name, "out");
    }

    // -- proptest -----------------------------------------------------------

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        prop_compose! {
            fn arb_tensor_desc()(
                numel in 1usize..=8192,
                elem_bytes in prop_oneof![Just(1usize), Just(2usize), Just(4usize)],
            ) -> TensorDescriptor {
                TensorDescriptor::new("t", vec![numel], elem_bytes)
            }
        }

        proptest! {
            #[test]
            fn numel_matches_shape(shape in prop::collection::vec(1usize..=64, 1..=4)) {
                let expected: usize = shape.iter().product();
                let td = TensorDescriptor::new("x", shape, 4);
                prop_assert_eq!(td.numel(), expected);
            }

            #[test]
            fn size_bytes_eq_numel_times_elem(td in arb_tensor_desc()) {
                prop_assert_eq!(td.size_bytes(), td.numel() * td.element_bytes);
            }

            #[test]
            fn fused_traffic_leq_unfused(numel in 64usize..=4096) {
                let mut planner = FusionPlanner::default_a100();
                planner.add_op(ew_op(0, "a", numel));
                planner.add_op(ew_op(1, "b", numel));
                planner.add_dependency(0, 1);
                let unfused = planner.estimate_unfused_traffic(&[0, 1]);
                let fused = planner.estimate_fused_traffic(&[0, 1]);
                prop_assert!(fused <= unfused);
            }

            #[test]
            fn speedup_geq_one_for_chain(numel in 64usize..=4096) {
                let mut planner = FusionPlanner::default_a100();
                planner.add_op(ew_op(0, "a", numel));
                planner.add_op(ew_op(1, "b", numel));
                planner.add_dependency(0, 1);
                prop_assert!(planner.estimated_speedup(&[0, 1]) >= 1.0);
            }

            #[test]
            fn chain_length_monotonic_speedup(numel in 256usize..=2048) {
                // 2-chain
                let mut p2 = FusionPlanner::default_a100();
                p2.add_op(ew_op(0, "a", numel));
                p2.add_op(ew_op(1, "b", numel));
                p2.add_dependency(0, 1);
                let s2 = p2.estimated_speedup(&[0, 1]);
                // 3-chain
                let mut p3 = FusionPlanner::default_a100();
                p3.add_op(ew_op(0, "a", numel));
                p3.add_op(ew_op(1, "b", numel));
                p3.add_op(ew_op(2, "c", numel));
                p3.add_dependency(0, 1);
                p3.add_dependency(1, 2);
                let s3 = p3.estimated_speedup(&[0, 1, 2]);
                prop_assert!(s3 >= s2);
            }
        }
    }
}
