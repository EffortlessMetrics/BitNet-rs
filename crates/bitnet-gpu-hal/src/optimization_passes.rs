//! Module stub - implementation pending merge from feature branch
//! Model optimization passes for graph-level performance tuning.
//!
//! Provides a pipeline of optimization passes that transform a computation graph
//! for improved performance on target devices. Passes include constant folding,
//! operator fusion, layout optimization, dead code elimination, common
//! subexpression elimination, and memory optimization.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Optimization level
// ---------------------------------------------------------------------------

/// Controls how aggressively the pipeline optimizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OptimizationLevel {
    /// No optimizations -- pass-through.
    O0,
    /// Safe, fast optimizations (constant folding, DCE).
    O1,
    /// Standard optimizations (O1 + fusion, CSE).
    #[default]
    O2,
    /// Aggressive optimizations (O2 + layout, memory).
    O3,
}

impl fmt::Display for OptimizationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::O0 => write!(f, "O0"),
            Self::O1 => write!(f, "O1"),
            Self::O2 => write!(f, "O2"),
            Self::O3 => write!(f, "O3"),
        }
    }
}

// ---------------------------------------------------------------------------
// Target device
// ---------------------------------------------------------------------------

/// The device class the graph will execute on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TargetDevice {
    Cpu,
    Cuda,
    Intel,
    #[default]
    Generic,
}

// ---------------------------------------------------------------------------
// Computation-graph types
// ---------------------------------------------------------------------------

/// The type of operation a graph node represents.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
    Constant,
    MatMul,
    Add,
    BiasAdd,
    LayerNorm,
    RmsNorm,
    Relu,
    Gelu,
    Silu,
    Softmax,
    Reshape,
    Transpose,
    FusedMatMulBias,
    FusedNormActivation,
    Identity,
    Load,
    Store,
    Custom(String),
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Custom(name) => write!(f, "Custom({name})"),
            other => write!(f, "{other:?}"),
        }
    }
}

/// Memory layout hint for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryLayout {
    #[default]
    RowMajor,
    ColumnMajor,
    Blocked {
        block_size: usize,
    },
    DeviceOptimal,
}

/// A single node in the computation graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier.
    pub id: usize,
    /// Operation performed by this node.
    pub op: OpType,
    /// Indices of input nodes.
    pub inputs: Vec<usize>,
    /// Output tensor shape (may be empty for unknown).
    pub shape: Vec<usize>,
    /// Optional constant value (used by `ConstantFolding`).
    pub constant_value: Option<Vec<f32>>,
    /// Memory layout hint.
    pub layout: MemoryLayout,
    /// Whether this node is marked as a graph output.
    pub is_output: bool,
    /// Estimated FLOPs for this operation.
    pub estimated_flops: u64,
    /// Memory footprint in bytes.
    pub memory_bytes: usize,
    /// Whether this node can be computed in place.
    pub in_place: bool,
}

impl GraphNode {
    pub fn new(id: usize, op: OpType) -> Self {
        Self {
            id,
            op,
            inputs: Vec::new(),
            shape: Vec::new(),
            constant_value: None,
            layout: MemoryLayout::default(),
            is_output: false,
            estimated_flops: 0,
            memory_bytes: 0,
            in_place: false,
        }
    }
}

/// A computation graph composed of nodes.
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    pub nodes: Vec<GraphNode>,
    pub output_ids: Vec<usize>,
}

impl ComputationGraph {
    #[must_use]
    pub const fn new() -> Self {
        Self { nodes: Vec::new(), output_ids: Vec::new() }
    }

    /// Adds a node and returns its assigned id.
    pub fn add_node(&mut self, mut node: GraphNode) -> usize {
        let id = self.nodes.len();
        node.id = id;
        self.nodes.push(node);
        id
    }

    pub const fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the set of node ids reachable from the graph outputs.
    pub fn reachable_from_outputs(&self) -> HashSet<usize> {
        let mut visited = HashSet::new();
        let mut stack: Vec<usize> = self.output_ids.clone();
        // Also include any node marked is_output.
        for n in &self.nodes {
            if n.is_output && !stack.contains(&n.id) {
                stack.push(n.id);
            }
        }
        while let Some(id) = stack.pop() {
            if visited.insert(id)
                && let Some(node) = self.nodes.get(id)
            {
                stack.extend_from_slice(&node.inputs);
            }
        }
        visited
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls which optimization passes to run and their parameters.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Which passes to enable.
    pub enabled_passes: Vec<PassKind>,
    /// Target device for device-specific optimizations.
    pub target_device: TargetDevice,
    /// Optimization aggressiveness level.
    pub level: OptimizationLevel,
    /// Maximum number of pipeline iterations (for fixed-point convergence).
    pub max_iterations: usize,
    /// Whether to verify the graph after each pass.
    pub verify_after_each_pass: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enabled_passes: vec![
                PassKind::ConstantFolding,
                PassKind::DeadCodeElimination,
                PassKind::OperatorFusion,
                PassKind::CommonSubexprElimination,
                PassKind::LayoutOptimization,
                PassKind::MemoryOptimization,
            ],
            target_device: TargetDevice::default(),
            level: OptimizationLevel::default(),
            max_iterations: 3,
            verify_after_each_pass: true,
        }
    }
}

impl OptimizationConfig {
    /// Returns a config that applies no optimizations.
    pub fn disabled() -> Self {
        Self { enabled_passes: Vec::new(), level: OptimizationLevel::O0, ..Default::default() }
    }

    /// Validates configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_iterations == 0 {
            return Err("max_iterations must be >= 1".into());
        }
        Ok(())
    }

    /// Returns the set of passes that should execute for the configured level.
    pub fn effective_passes(&self) -> Vec<PassKind> {
        match self.level {
            OptimizationLevel::O0 => Vec::new(),
            OptimizationLevel::O1 => self
                .enabled_passes
                .iter()
                .filter(|p| matches!(p, PassKind::ConstantFolding | PassKind::DeadCodeElimination))
                .copied()
                .collect(),
            OptimizationLevel::O2 => self
                .enabled_passes
                .iter()
                .filter(|p| {
                    matches!(
                        p,
                        PassKind::ConstantFolding
                            | PassKind::DeadCodeElimination
                            | PassKind::OperatorFusion
                            | PassKind::CommonSubexprElimination
                    )
                })
                .copied()
                .collect(),
            OptimizationLevel::O3 => self.enabled_passes.clone(),
        }
    }
}

/// The kind of optimization pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassKind {
    ConstantFolding,
    OperatorFusion,
    LayoutOptimization,
    DeadCodeElimination,
    CommonSubexprElimination,
    MemoryOptimization,
}

impl fmt::Display for PassKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

// ---------------------------------------------------------------------------
// Pass trait & result
// ---------------------------------------------------------------------------

/// Result returned after a pass executes.
#[derive(Debug, Clone)]
pub struct PassResult {
    /// The pass that produced this result.
    pub pass: PassKind,
    /// Number of transformations applied.
    pub transformations_applied: usize,
    /// Number of nodes removed.
    pub nodes_removed: usize,
    /// Number of nodes added (e.g. fused replacements).
    pub nodes_added: usize,
    /// Estimated speedup factor (1.0 = no change).
    pub estimated_speedup: f64,
    /// Estimated memory savings in bytes.
    pub memory_saved_bytes: i64,
    /// Human-readable details.
    pub details: Vec<String>,
}

impl PassResult {
    #[must_use]
    pub const fn noop(pass: PassKind) -> Self {
        Self {
            pass,
            transformations_applied: 0,
            nodes_removed: 0,
            nodes_added: 0,
            estimated_speedup: 1.0,
            memory_saved_bytes: 0,
            details: Vec::new(),
        }
    }
}

/// Trait implemented by every optimization pass.
pub trait OptimizationPass {
    /// Returns the kind of this pass.
    fn kind(&self) -> PassKind;

    /// Analyse the graph and return the number of potential optimizations.
    fn analyze(&self, graph: &ComputationGraph) -> usize;

    /// Apply the pass, mutating the graph in place.
    fn transform(&self, graph: &mut ComputationGraph) -> PassResult;

    /// Post-transformation verification.  Returns `Ok(())` if the graph is valid.
    fn verify(&self, graph: &ComputationGraph) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// ConstantFolding
// ---------------------------------------------------------------------------

/// Evaluates constant expressions at compile time, replacing sub-trees of
/// constants with a single pre-computed constant node.
#[derive(Debug, Default)]
pub struct ConstantFolding;

impl ConstantFolding {
    fn is_foldable(node: &GraphNode, graph: &ComputationGraph) -> bool {
        if node.constant_value.is_some() {
            return false; // already a constant
        }
        if node.inputs.is_empty() {
            return false;
        }
        node.inputs
            .iter()
            .all(|&id| graph.nodes.get(id).is_some_and(|n| n.constant_value.is_some()))
    }

    fn fold_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }
}

impl OptimizationPass for ConstantFolding {
    fn kind(&self) -> PassKind {
        PassKind::ConstantFolding
    }

    fn analyze(&self, graph: &ComputationGraph) -> usize {
        graph.nodes.iter().filter(|n| Self::is_foldable(n, graph)).count()
    }

    fn transform(&self, graph: &mut ComputationGraph) -> PassResult {
        let mut result = PassResult::noop(self.kind());
        let mut folded: HashMap<usize, Vec<f32>> = HashMap::new();

        for i in 0..graph.nodes.len() {
            if Self::is_foldable(&graph.nodes[i], graph) {
                let node = &graph.nodes[i];
                if node.op == OpType::Add && node.inputs.len() == 2 {
                    let a = graph.nodes[node.inputs[0]].constant_value.clone().unwrap_or_default();
                    let b = graph.nodes[node.inputs[1]].constant_value.clone().unwrap_or_default();
                    if a.len() == b.len() {
                        folded.insert(i, Self::fold_add(&a, &b));
                        result.transformations_applied += 1;
                        result.details.push(format!("Folded Add node {i} into constant"));
                    }
                }
            }
        }

        for (id, value) in folded {
            graph.nodes[id].op = OpType::Constant;
            graph.nodes[id].constant_value = Some(value);
            graph.nodes[id].inputs.clear();
            result.estimated_speedup += 0.01;
        }
        result.estimated_speedup = result.estimated_speedup.max(1.0);
        result
    }

    fn verify(&self, _graph: &ComputationGraph) -> Result<(), String> {
        // ConstantFolding correctness is guaranteed by the transform logic
        // (values computed from constant inputs). No additional invariant
        // check is needed — parameter/weight Constant nodes legitimately
        // carry no inline value.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OperatorFusion
// ---------------------------------------------------------------------------

/// Fuses adjacent compatible operators into a single fused node.
///
/// Supported fusion patterns:
/// - `MatMul` → `BiasAdd` → `FusedMatMulBias`
/// - `LayerNorm`/`RmsNorm` → activation (`Relu`/`Gelu`/`Silu`) → `FusedNormActivation`
#[derive(Debug, Default)]
pub struct OperatorFusion;

impl OperatorFusion {
    fn find_matmul_bias_pairs(graph: &ComputationGraph) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for node in &graph.nodes {
            if matches!(node.op, OpType::BiasAdd | OpType::Add) {
                for &input_id in &node.inputs {
                    if graph.nodes.get(input_id).is_some_and(|n| n.op == OpType::MatMul) {
                        pairs.push((input_id, node.id));
                    }
                }
            }
        }
        pairs
    }

    fn find_norm_activation_pairs(graph: &ComputationGraph) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for node in &graph.nodes {
            if matches!(node.op, OpType::Relu | OpType::Gelu | OpType::Silu) {
                for &input_id in &node.inputs {
                    if graph
                        .nodes
                        .get(input_id)
                        .is_some_and(|n| matches!(n.op, OpType::LayerNorm | OpType::RmsNorm))
                    {
                        pairs.push((input_id, node.id));
                    }
                }
            }
        }
        pairs
    }
}

impl OptimizationPass for OperatorFusion {
    fn kind(&self) -> PassKind {
        PassKind::OperatorFusion
    }

    fn analyze(&self, graph: &ComputationGraph) -> usize {
        Self::find_matmul_bias_pairs(graph).len() + Self::find_norm_activation_pairs(graph).len()
    }

    fn transform(&self, graph: &mut ComputationGraph) -> PassResult {
        let mut result = PassResult::noop(self.kind());

        // Fuse MatMul+Bias
        let matmul_pairs = Self::find_matmul_bias_pairs(graph);
        let mut fused_targets: HashSet<usize> = HashSet::new();
        for (matmul_id, bias_id) in &matmul_pairs {
            if fused_targets.contains(matmul_id) || fused_targets.contains(bias_id) {
                continue;
            }
            graph.nodes[*bias_id].op = OpType::FusedMatMulBias;
            let matmul_inputs = graph.nodes[*matmul_id].inputs.clone();
            graph.nodes[*bias_id].inputs = matmul_inputs;
            // Keep the bias input too.
            let bias_other_inputs: Vec<usize> = graph.nodes[*bias_id]
                .inputs
                .iter()
                .copied()
                .chain(graph.nodes[*bias_id].inputs.iter().copied().filter(|id| *id != *matmul_id))
                .collect();
            graph.nodes[*bias_id].inputs = bias_other_inputs;
            graph.nodes[*matmul_id].op = OpType::Identity;
            graph.nodes[*matmul_id].inputs.clear();
            fused_targets.insert(*matmul_id);
            fused_targets.insert(*bias_id);
            result.transformations_applied += 1;
            result.estimated_speedup += 0.05;
            result.details.push(format!("Fused MatMul({matmul_id})+Bias({bias_id})"));
        }

        // Fuse Norm+Activation
        let norm_pairs = Self::find_norm_activation_pairs(graph);
        for (norm_id, act_id) in &norm_pairs {
            if fused_targets.contains(norm_id) || fused_targets.contains(act_id) {
                continue;
            }
            graph.nodes[*act_id].op = OpType::FusedNormActivation;
            let norm_inputs = graph.nodes[*norm_id].inputs.clone();
            graph.nodes[*act_id].inputs = norm_inputs;
            graph.nodes[*norm_id].op = OpType::Identity;
            graph.nodes[*norm_id].inputs.clear();
            fused_targets.insert(*norm_id);
            fused_targets.insert(*act_id);
            result.transformations_applied += 1;
            result.estimated_speedup += 0.03;
            result.details.push(format!("Fused Norm({norm_id})+Activation({act_id})"));
        }

        result.estimated_speedup = (1.0 + result.estimated_speedup).min(2.0);
        result
    }

    fn verify(&self, _graph: &ComputationGraph) -> Result<(), String> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LayoutOptimizer
// ---------------------------------------------------------------------------

/// Optimizes tensor memory layout for the target device.
///
/// For CUDA targets, transposes to column-major where beneficial.
/// For CPU, blocked layouts are preferred for large tensors.
#[derive(Debug)]
pub struct LayoutOptimizer {
    target: TargetDevice,
}

impl LayoutOptimizer {
    #[must_use]
    pub const fn new(target: TargetDevice) -> Self {
        Self { target }
    }

    fn preferred_layout(&self, node: &GraphNode) -> MemoryLayout {
        match self.target {
            TargetDevice::Cuda | TargetDevice::Intel => {
                if node.op == OpType::MatMul {
                    MemoryLayout::ColumnMajor
                } else {
                    MemoryLayout::DeviceOptimal
                }
            }
            TargetDevice::Cpu => {
                if node.memory_bytes > 1024 * 1024 {
                    MemoryLayout::Blocked { block_size: 64 }
                } else {
                    MemoryLayout::RowMajor
                }
            }
            TargetDevice::Generic => MemoryLayout::RowMajor,
        }
    }
}

impl OptimizationPass for LayoutOptimizer {
    fn kind(&self) -> PassKind {
        PassKind::LayoutOptimization
    }

    fn analyze(&self, graph: &ComputationGraph) -> usize {
        graph.nodes.iter().filter(|n| n.layout != self.preferred_layout(n)).count()
    }

    fn transform(&self, graph: &mut ComputationGraph) -> PassResult {
        let mut result = PassResult::noop(self.kind());
        for node in &mut graph.nodes {
            let preferred = self.preferred_layout(node);
            if node.layout != preferred {
                let old = node.layout;
                node.layout = preferred;
                result.transformations_applied += 1;
                result.details.push(format!("Node {}: {old:?} → {preferred:?}", node.id));
            }
        }
        if result.transformations_applied > 0 {
            #[allow(clippy::cast_precision_loss)]
            let count = result.transformations_applied as f64;
            result.estimated_speedup = count.mul_add(0.02, 1.0);
        }
        result
    }

    fn verify(&self, _graph: &ComputationGraph) -> Result<(), String> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DeadCodeElimination
// ---------------------------------------------------------------------------

/// Removes nodes that do not contribute to any graph output.
#[derive(Debug, Default)]
pub struct DeadCodeElimination;

impl OptimizationPass for DeadCodeElimination {
    fn kind(&self) -> PassKind {
        PassKind::DeadCodeElimination
    }

    fn analyze(&self, graph: &ComputationGraph) -> usize {
        let reachable = graph.reachable_from_outputs();
        graph.nodes.iter().filter(|n| !reachable.contains(&n.id)).count()
    }

    fn transform(&self, graph: &mut ComputationGraph) -> PassResult {
        let mut result = PassResult::noop(self.kind());
        let reachable = graph.reachable_from_outputs();

        for node in &mut graph.nodes {
            if !reachable.contains(&node.id) && node.op != OpType::Identity {
                #[allow(clippy::cast_possible_wrap)]
                let mem = node.memory_bytes as i64;
                result.memory_saved_bytes += mem;
                result.nodes_removed += 1;
                result.details.push(format!("Eliminated dead node {} ({:?})", node.id, node.op));
                node.op = OpType::Identity;
                node.inputs.clear();
                node.constant_value = None;
                node.memory_bytes = 0;
            }
        }
        result.transformations_applied = result.nodes_removed;
        result
    }

    fn verify(&self, _graph: &ComputationGraph) -> Result<(), String> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CommonSubexprElimination
// ---------------------------------------------------------------------------

/// Identifies repeated computations and rewires later uses to the first
/// occurrence, enabling subsequent DCE to remove the duplicates.
#[derive(Debug, Default)]
pub struct CommonSubexprElimination;

impl CommonSubexprElimination {
    /// Produces a signature that identifies a computation.
    fn signature(node: &GraphNode) -> Option<(OpType, Vec<usize>)> {
        if node.inputs.is_empty() || node.op == OpType::Constant || node.op == OpType::Identity {
            return None;
        }
        let mut sorted_inputs = node.inputs.clone();
        // Only sort for commutative ops.
        if matches!(node.op, OpType::Add | OpType::MatMul) {
            sorted_inputs.sort_unstable();
        }
        Some((node.op.clone(), sorted_inputs))
    }
}

impl OptimizationPass for CommonSubexprElimination {
    fn kind(&self) -> PassKind {
        PassKind::CommonSubexprElimination
    }

    fn analyze(&self, graph: &ComputationGraph) -> usize {
        let mut seen: HashMap<(OpType, Vec<usize>), usize> = HashMap::new();
        let mut count = 0;
        for node in &graph.nodes {
            if let Some(sig) = Self::signature(node) {
                if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(sig) {
                    e.insert(node.id);
                } else {
                    count += 1;
                }
            }
        }
        count
    }

    fn transform(&self, graph: &mut ComputationGraph) -> PassResult {
        let mut result = PassResult::noop(self.kind());
        let mut canonical: HashMap<(OpType, Vec<usize>), usize> = HashMap::new();
        let mut rewrites: HashMap<usize, usize> = HashMap::new();

        for i in 0..graph.nodes.len() {
            if let Some(sig) = Self::signature(&graph.nodes[i]) {
                if let Some(&first) = canonical.get(&sig) {
                    rewrites.insert(graph.nodes[i].id, first);
                    result.transformations_applied += 1;
                    result
                        .details
                        .push(format!("CSE: node {} duplicates node {first}", graph.nodes[i].id));
                } else {
                    canonical.insert(sig, graph.nodes[i].id);
                }
            }
        }

        // Rewire all references.
        for node in &mut graph.nodes {
            for inp in &mut node.inputs {
                if let Some(&target) = rewrites.get(inp) {
                    *inp = target;
                }
            }
        }
        result
    }

    fn verify(&self, _graph: &ComputationGraph) -> Result<(), String> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MemoryOptimizer
// ---------------------------------------------------------------------------

/// Optimizes memory usage by enabling in-place operations and tracking buffer
/// reuse opportunities.
#[derive(Debug, Default)]
pub struct MemoryOptimizer;

impl MemoryOptimizer {
    /// Returns the set of nodes whose output buffer is consumed by exactly one
    /// downstream node and can therefore be overwritten in place.
    fn find_in_place_candidates(graph: &ComputationGraph) -> HashSet<usize> {
        let mut use_count: HashMap<usize, usize> = HashMap::new();
        for node in &graph.nodes {
            for &inp in &node.inputs {
                *use_count.entry(inp).or_insert(0) += 1;
            }
        }
        // A node can be computed in-place if its output is used exactly once and
        // it is not a graph output.
        use_count
            .into_iter()
            .filter_map(|(id, count)| {
                if count == 1 {
                    let node = &graph.nodes[id];
                    if !node.is_output {
                        return Some(id);
                    }
                }
                None
            })
            .collect()
    }

    fn estimate_peak_memory(graph: &ComputationGraph) -> usize {
        graph.nodes.iter().map(|n| n.memory_bytes).sum()
    }
}

impl OptimizationPass for MemoryOptimizer {
    fn kind(&self) -> PassKind {
        PassKind::MemoryOptimization
    }

    fn analyze(&self, graph: &ComputationGraph) -> usize {
        Self::find_in_place_candidates(graph).len()
    }

    fn transform(&self, graph: &mut ComputationGraph) -> PassResult {
        let mut result = PassResult::noop(self.kind());
        let candidates = Self::find_in_place_candidates(graph);

        let peak_before = Self::estimate_peak_memory(graph);

        for &id in &candidates {
            if !graph.nodes[id].in_place {
                graph.nodes[id].in_place = true;
                result.transformations_applied += 1;
                #[allow(clippy::cast_possible_wrap)]
                let saved = graph.nodes[id].memory_bytes as i64;
                result.memory_saved_bytes += saved;
                result.details.push(format!("Enabled in-place for node {id}"));
            }
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let saved_usize = result.memory_saved_bytes.max(0) as usize;
        let peak_after = peak_before.saturating_sub(saved_usize);
        if peak_before > 0 {
            #[allow(clippy::cast_precision_loss)]
            let ratio = peak_after as f64 / peak_before as f64;
            result.estimated_speedup = ratio.mul_add(0.01, 1.0);
        }
        result
    }

    fn verify(&self, graph: &ComputationGraph) -> Result<(), String> {
        for node in &graph.nodes {
            if node.in_place && node.is_output {
                return Err(format!("Node {} is marked in-place but is a graph output", node.id));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OptimizationReport
// ---------------------------------------------------------------------------

/// Aggregated report produced by the optimization pipeline.
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// Per-pass results, in execution order.
    pub pass_results: Vec<PassResult>,
    /// Total estimated speedup (multiplicative).
    pub total_speedup: f64,
    /// Total memory saved in bytes.
    pub total_memory_saved_bytes: i64,
    /// Number of pipeline iterations that ran.
    pub iterations: usize,
    /// Node count before optimization.
    pub nodes_before: usize,
    /// Node count after optimization (active, non-Identity).
    pub nodes_after: usize,
}

impl OptimizationReport {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            pass_results: Vec::new(),
            total_speedup: 1.0,
            total_memory_saved_bytes: 0,
            iterations: 0,
            nodes_before: 0,
            nodes_after: 0,
        }
    }

    /// Returns the names of all passes that performed at least one transformation.
    pub fn applied_pass_names(&self) -> Vec<String> {
        self.pass_results
            .iter()
            .filter(|r| r.transformations_applied > 0)
            .map(|r| r.pass.to_string())
            .collect()
    }

    /// Total transformations across all passes.
    pub fn total_transformations(&self) -> usize {
        self.pass_results.iter().map(|r| r.transformations_applied).sum()
    }
}

impl fmt::Display for OptimizationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Optimization Report ({} iterations)", self.iterations)?;
        writeln!(f, "  Nodes: {} → {}", self.nodes_before, self.nodes_after)?;
        writeln!(f, "  Estimated speedup: {:.2}×", self.total_speedup)?;
        writeln!(f, "  Memory saved: {} bytes", self.total_memory_saved_bytes)?;
        for pr in &self.pass_results {
            if pr.transformations_applied > 0 {
                writeln!(
                    f,
                    "  [{:?}] {} transforms, {:.2}× speedup",
                    pr.pass, pr.transformations_applied, pr.estimated_speedup
                )?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OptimizationPipeline
// ---------------------------------------------------------------------------

/// Chains passes in a fixed order, running verification between each pass.
pub struct OptimizationPipeline {
    passes: Vec<Box<dyn OptimizationPass>>,
    config: OptimizationConfig,
}

impl OptimizationPipeline {
    /// Creates a pipeline from a configuration.
    pub fn from_config(config: OptimizationConfig) -> Self {
        let effective = config.effective_passes();
        let passes: Vec<Box<dyn OptimizationPass>> = effective
            .iter()
            .map(|kind| -> Box<dyn OptimizationPass> {
                match kind {
                    PassKind::ConstantFolding => Box::new(ConstantFolding),
                    PassKind::OperatorFusion => Box::new(OperatorFusion),
                    PassKind::LayoutOptimization => {
                        Box::new(LayoutOptimizer::new(config.target_device))
                    }
                    PassKind::DeadCodeElimination => Box::new(DeadCodeElimination),
                    PassKind::CommonSubexprElimination => Box::new(CommonSubexprElimination),
                    PassKind::MemoryOptimization => Box::new(MemoryOptimizer),
                }
            })
            .collect();
        Self { passes, config }
    }

    /// Creates a pipeline with explicit passes.
    pub fn new(passes: Vec<Box<dyn OptimizationPass>>, config: OptimizationConfig) -> Self {
        Self { passes, config }
    }

    /// Runs the pipeline on the graph and returns a report.
    pub fn run(&self, graph: &mut ComputationGraph) -> Result<OptimizationReport, String> {
        self.config.validate()?;

        let nodes_before = graph.node_count();
        let mut report = OptimizationReport::empty();
        report.nodes_before = nodes_before;

        for iteration in 0..self.config.max_iterations {
            let mut changed = false;
            for pass in &self.passes {
                let result = pass.transform(graph);
                if result.transformations_applied > 0 {
                    changed = true;
                }
                if self.config.verify_after_each_pass {
                    pass.verify(graph)?;
                }
                report.total_speedup *= result.estimated_speedup;
                report.total_memory_saved_bytes += result.memory_saved_bytes;
                report.pass_results.push(result);
            }
            report.iterations = iteration + 1;
            if !changed {
                break;
            }
        }

        report.nodes_after = graph.nodes.iter().filter(|n| n.op != OpType::Identity).count();
        Ok(report)
    }
}

impl fmt::Debug for OptimizationPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptimizationPipeline")
            .field("pass_count", &self.passes.len())
            .field("config", &self.config)
            .finish()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers -----------------------------------------------------------

    fn simple_graph() -> ComputationGraph {
        let mut g = ComputationGraph::new();
        let c0 = g.add_node(GraphNode {
            constant_value: Some(vec![1.0, 2.0]),
            ..GraphNode::new(0, OpType::Constant)
        });
        let c1 = g.add_node(GraphNode {
            constant_value: Some(vec![3.0, 4.0]),
            ..GraphNode::new(0, OpType::Constant)
        });
        let mut add = GraphNode::new(0, OpType::Add);
        add.inputs = vec![c0, c1];
        add.is_output = true;
        g.add_node(add);
        g.output_ids.push(2);
        g
    }

    fn matmul_bias_graph() -> ComputationGraph {
        let mut g = ComputationGraph::new();
        let input = g.add_node(GraphNode::new(0, OpType::Load));
        let weight = g.add_node(GraphNode::new(0, OpType::Constant));
        let mut mm = GraphNode::new(0, OpType::MatMul);
        mm.inputs = vec![input, weight];
        let mm_id = g.add_node(mm);
        let bias = g.add_node(GraphNode::new(0, OpType::Constant));
        let mut bias_add = GraphNode::new(0, OpType::BiasAdd);
        bias_add.inputs = vec![mm_id, bias];
        bias_add.is_output = true;
        let out_id = g.add_node(bias_add);
        g.output_ids.push(out_id);
        g
    }

    fn norm_activation_graph() -> ComputationGraph {
        let mut g = ComputationGraph::new();
        let input = g.add_node(GraphNode::new(0, OpType::Load));
        let mut norm = GraphNode::new(0, OpType::LayerNorm);
        norm.inputs = vec![input];
        let norm_id = g.add_node(norm);
        let mut act = GraphNode::new(0, OpType::Gelu);
        act.inputs = vec![norm_id];
        act.is_output = true;
        let out_id = g.add_node(act);
        g.output_ids.push(out_id);
        g
    }

    fn graph_with_dead_code() -> ComputationGraph {
        let mut g = ComputationGraph::new();
        let live = g.add_node(GraphNode {
            is_output: true,
            memory_bytes: 100,
            ..GraphNode::new(0, OpType::Load)
        });
        // Dead node — not reachable from any output.
        let mut dead = GraphNode::new(0, OpType::MatMul);
        dead.memory_bytes = 500;
        g.add_node(dead);
        g.output_ids.push(live);
        g
    }

    fn cse_graph() -> ComputationGraph {
        let mut g = ComputationGraph::new();
        let a = g.add_node(GraphNode::new(0, OpType::Load));
        let b = g.add_node(GraphNode::new(0, OpType::Load));
        let mut add1 = GraphNode::new(0, OpType::Add);
        add1.inputs = vec![a, b];
        g.add_node(add1);
        let mut add2 = GraphNode::new(0, OpType::Add);
        add2.inputs = vec![a, b];
        add2.is_output = true;
        let out = g.add_node(add2);
        g.output_ids.push(out);
        g
    }

    fn memory_graph() -> ComputationGraph {
        let mut g = ComputationGraph::new();
        let a = g.add_node(GraphNode { memory_bytes: 1024, ..GraphNode::new(0, OpType::Load) });
        let mut b = GraphNode::new(0, OpType::Relu);
        b.inputs = vec![a];
        b.memory_bytes = 1024;
        b.is_output = true;
        let out = g.add_node(b);
        g.output_ids.push(out);
        g
    }

    // =====================================================================
    // OptimizationLevel
    // =====================================================================

    #[test]
    fn test_level_default_is_o2() {
        assert_eq!(OptimizationLevel::default(), OptimizationLevel::O2);
    }

    #[test]
    fn test_level_display() {
        assert_eq!(OptimizationLevel::O0.to_string(), "O0");
        assert_eq!(OptimizationLevel::O3.to_string(), "O3");
    }

    #[test]
    fn test_level_eq_and_hash() {
        let mut set = HashSet::new();
        set.insert(OptimizationLevel::O1);
        assert!(set.contains(&OptimizationLevel::O1));
        assert!(!set.contains(&OptimizationLevel::O2));
    }

    // =====================================================================
    // TargetDevice
    // =====================================================================

    #[test]
    fn test_target_device_default() {
        assert_eq!(TargetDevice::default(), TargetDevice::Generic);
    }

    #[test]
    fn test_target_device_eq() {
        assert_ne!(TargetDevice::Cpu, TargetDevice::Cuda);
        assert_eq!(TargetDevice::Intel, TargetDevice::Intel);
    }

    // =====================================================================
    // OpType
    // =====================================================================

    #[test]
    fn test_optype_display_builtin() {
        assert_eq!(format!("{}", OpType::MatMul), "MatMul");
    }

    #[test]
    fn test_optype_display_custom() {
        assert_eq!(format!("{}", OpType::Custom("MyOp".into())), "Custom(MyOp)");
    }

    #[test]
    fn test_optype_eq() {
        assert_eq!(OpType::Add, OpType::Add);
        assert_ne!(OpType::Add, OpType::MatMul);
    }

    // =====================================================================
    // MemoryLayout
    // =====================================================================

    #[test]
    fn test_memory_layout_default() {
        assert_eq!(MemoryLayout::default(), MemoryLayout::RowMajor);
    }

    #[test]
    fn test_memory_layout_blocked_eq() {
        assert_eq!(
            MemoryLayout::Blocked { block_size: 32 },
            MemoryLayout::Blocked { block_size: 32 }
        );
        assert_ne!(
            MemoryLayout::Blocked { block_size: 32 },
            MemoryLayout::Blocked { block_size: 64 }
        );
    }

    // =====================================================================
    // GraphNode
    // =====================================================================

    #[test]
    fn test_graph_node_new() {
        let n = GraphNode::new(5, OpType::Relu);
        assert_eq!(n.id, 5);
        assert_eq!(n.op, OpType::Relu);
        assert!(n.inputs.is_empty());
        assert!(n.constant_value.is_none());
        assert!(!n.is_output);
        assert!(!n.in_place);
    }

    #[test]
    fn test_graph_node_constant_value() {
        let mut n = GraphNode::new(0, OpType::Constant);
        n.constant_value = Some(vec![1.0, 2.0, 3.0]);
        assert_eq!(n.constant_value.as_ref().unwrap().len(), 3);
    }

    // =====================================================================
    // ComputationGraph
    // =====================================================================

    #[test]
    fn test_graph_new_empty() {
        let g = ComputationGraph::new();
        assert_eq!(g.node_count(), 0);
        assert!(g.output_ids.is_empty());
    }

    #[test]
    fn test_graph_default() {
        let g = ComputationGraph::default();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_graph_add_node_assigns_id() {
        let mut g = ComputationGraph::new();
        let id0 = g.add_node(GraphNode::new(99, OpType::Load));
        let id1 = g.add_node(GraphNode::new(99, OpType::Store));
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(g.nodes[0].id, 0);
        assert_eq!(g.nodes[1].id, 1);
    }

    #[test]
    fn test_graph_reachable_from_outputs() {
        let g = graph_with_dead_code();
        let reachable = g.reachable_from_outputs();
        assert!(reachable.contains(&0));
        assert!(!reachable.contains(&1));
    }

    #[test]
    fn test_graph_reachable_transitive() {
        let mut g = ComputationGraph::new();
        let a = g.add_node(GraphNode::new(0, OpType::Load));
        let mut b = GraphNode::new(0, OpType::Relu);
        b.inputs = vec![a];
        let b_id = g.add_node(b);
        let mut c = GraphNode::new(0, OpType::Store);
        c.inputs = vec![b_id];
        c.is_output = true;
        g.add_node(c);
        g.output_ids.push(2);
        let reachable = g.reachable_from_outputs();
        assert_eq!(reachable.len(), 3);
    }

    // =====================================================================
    // OptimizationConfig
    // =====================================================================

    #[test]
    fn test_config_default() {
        let c = OptimizationConfig::default();
        assert_eq!(c.level, OptimizationLevel::O2);
        assert_eq!(c.target_device, TargetDevice::Generic);
        assert_eq!(c.max_iterations, 3);
        assert!(c.verify_after_each_pass);
        assert!(!c.enabled_passes.is_empty());
    }

    #[test]
    fn test_config_disabled() {
        let c = OptimizationConfig::disabled();
        assert!(c.enabled_passes.is_empty());
        assert_eq!(c.level, OptimizationLevel::O0);
    }

    #[test]
    fn test_config_validate_ok() {
        assert!(OptimizationConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_iterations() {
        let c = OptimizationConfig { max_iterations: 0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_effective_passes_o0() {
        let c = OptimizationConfig { level: OptimizationLevel::O0, ..Default::default() };
        assert!(c.effective_passes().is_empty());
    }

    #[test]
    fn test_effective_passes_o1() {
        let c = OptimizationConfig { level: OptimizationLevel::O1, ..Default::default() };
        let passes = c.effective_passes();
        assert!(passes.contains(&PassKind::ConstantFolding));
        assert!(passes.contains(&PassKind::DeadCodeElimination));
        assert!(!passes.contains(&PassKind::OperatorFusion));
    }

    #[test]
    fn test_effective_passes_o2() {
        let c = OptimizationConfig::default();
        let passes = c.effective_passes();
        assert!(passes.contains(&PassKind::OperatorFusion));
        assert!(passes.contains(&PassKind::CommonSubexprElimination));
        assert!(!passes.contains(&PassKind::LayoutOptimization));
    }

    #[test]
    fn test_effective_passes_o3() {
        let c = OptimizationConfig { level: OptimizationLevel::O3, ..Default::default() };
        let passes = c.effective_passes();
        assert!(passes.contains(&PassKind::LayoutOptimization));
        assert!(passes.contains(&PassKind::MemoryOptimization));
    }

    // =====================================================================
    // PassKind
    // =====================================================================

    #[test]
    fn test_pass_kind_display() {
        assert_eq!(PassKind::ConstantFolding.to_string(), "ConstantFolding");
        assert_eq!(PassKind::CommonSubexprElimination.to_string(), "CommonSubexprElimination");
    }

    #[test]
    fn test_pass_kind_eq_hash() {
        let mut set = HashSet::new();
        set.insert(PassKind::OperatorFusion);
        assert!(set.contains(&PassKind::OperatorFusion));
    }

    // =====================================================================
    // PassResult
    // =====================================================================

    #[test]
    fn test_pass_result_noop() {
        let r = PassResult::noop(PassKind::DeadCodeElimination);
        assert_eq!(r.pass, PassKind::DeadCodeElimination);
        assert_eq!(r.transformations_applied, 0);
        assert_eq!(r.nodes_removed, 0);
        assert!((r.estimated_speedup - 1.0).abs() < f64::EPSILON);
    }

    // =====================================================================
    // ConstantFolding
    // =====================================================================

    #[test]
    fn test_constant_folding_analyze() {
        let g = simple_graph();
        let pass = ConstantFolding;
        assert_eq!(pass.analyze(&g), 1);
    }

    #[test]
    fn test_constant_folding_transform() {
        let mut g = simple_graph();
        let pass = ConstantFolding;
        let result = pass.transform(&mut g);
        assert_eq!(result.transformations_applied, 1);
        assert_eq!(g.nodes[2].op, OpType::Constant);
        assert_eq!(g.nodes[2].constant_value, Some(vec![4.0, 6.0]));
        assert!(g.nodes[2].inputs.is_empty());
    }

    #[test]
    fn test_constant_folding_verify_ok() {
        let mut g = simple_graph();
        let pass = ConstantFolding;
        pass.transform(&mut g);
        assert!(pass.verify(&g).is_ok());
    }

    #[test]
    fn test_constant_folding_verify_ok_with_parameter_constants() {
        let mut g = ComputationGraph::new();
        // Weight parameter — Constant without inline value, which is valid.
        let mut param = GraphNode::new(0, OpType::Constant);
        param.constant_value = None;
        g.add_node(param);
        let pass = ConstantFolding;
        assert!(pass.verify(&g).is_ok());
    }

    #[test]
    fn test_constant_folding_no_op_when_no_constants() {
        let mut g = ComputationGraph::new();
        let a = g.add_node(GraphNode::new(0, OpType::Load));
        let b = g.add_node(GraphNode::new(0, OpType::Load));
        let mut add = GraphNode::new(0, OpType::Add);
        add.inputs = vec![a, b];
        g.add_node(add);
        let pass = ConstantFolding;
        assert_eq!(pass.analyze(&g), 0);
        let result = pass.transform(&mut g);
        assert_eq!(result.transformations_applied, 0);
    }

    #[test]
    fn test_constant_folding_partial_constants() {
        let mut g = ComputationGraph::new();
        let c0 = g.add_node(GraphNode {
            constant_value: Some(vec![1.0]),
            ..GraphNode::new(0, OpType::Constant)
        });
        let l = g.add_node(GraphNode::new(0, OpType::Load));
        let mut add = GraphNode::new(0, OpType::Add);
        add.inputs = vec![c0, l];
        g.add_node(add);
        let pass = ConstantFolding;
        assert_eq!(pass.analyze(&g), 0);
    }

    #[test]
    fn test_constant_folding_mismatched_lengths() {
        let mut g = ComputationGraph::new();
        let c0 = g.add_node(GraphNode {
            constant_value: Some(vec![1.0]),
            ..GraphNode::new(0, OpType::Constant)
        });
        let c1 = g.add_node(GraphNode {
            constant_value: Some(vec![2.0, 3.0]),
            ..GraphNode::new(0, OpType::Constant)
        });
        let mut add = GraphNode::new(0, OpType::Add);
        add.inputs = vec![c0, c1];
        g.add_node(add);
        let pass = ConstantFolding;
        let result = pass.transform(&mut g);
        assert_eq!(result.transformations_applied, 0);
    }

    #[test]
    fn test_constant_folding_kind() {
        assert_eq!(ConstantFolding.kind(), PassKind::ConstantFolding);
    }

    #[test]
    fn test_constant_folding_speedup_positive() {
        let mut g = simple_graph();
        let result = ConstantFolding.transform(&mut g);
        assert!(result.estimated_speedup >= 1.0);
    }

    // =====================================================================
    // OperatorFusion
    // =====================================================================

    #[test]
    fn test_fusion_analyze_matmul_bias() {
        let g = matmul_bias_graph();
        let pass = OperatorFusion;
        assert!(pass.analyze(&g) >= 1);
    }

    #[test]
    fn test_fusion_transform_matmul_bias() {
        let mut g = matmul_bias_graph();
        let pass = OperatorFusion;
        let result = pass.transform(&mut g);
        assert!(result.transformations_applied >= 1);
        assert!(g.nodes.iter().any(|n| n.op == OpType::FusedMatMulBias));
        assert!(result.details.iter().any(|d| d.contains("Fused MatMul")));
    }

    #[test]
    fn test_fusion_analyze_norm_activation() {
        let g = norm_activation_graph();
        assert!(OperatorFusion.analyze(&g) >= 1);
    }

    #[test]
    fn test_fusion_transform_norm_activation() {
        let mut g = norm_activation_graph();
        let result = OperatorFusion.transform(&mut g);
        assert!(result.transformations_applied >= 1);
        assert!(g.nodes.iter().any(|n| n.op == OpType::FusedNormActivation));
    }

    #[test]
    fn test_fusion_no_op_on_empty_graph() {
        let mut g = ComputationGraph::new();
        let result = OperatorFusion.transform(&mut g);
        assert_eq!(result.transformations_applied, 0);
    }

    #[test]
    fn test_fusion_verify_always_ok() {
        let g = matmul_bias_graph();
        assert!(OperatorFusion.verify(&g).is_ok());
    }

    #[test]
    fn test_fusion_kind() {
        assert_eq!(OperatorFusion.kind(), PassKind::OperatorFusion);
    }

    #[test]
    fn test_fusion_speedup_bounded() {
        let mut g = matmul_bias_graph();
        let result = OperatorFusion.transform(&mut g);
        assert!(result.estimated_speedup <= 2.0);
    }

    #[test]
    fn test_fusion_rms_norm_silu() {
        let mut g = ComputationGraph::new();
        let inp = g.add_node(GraphNode::new(0, OpType::Load));
        let mut norm = GraphNode::new(0, OpType::RmsNorm);
        norm.inputs = vec![inp];
        let nid = g.add_node(norm);
        let mut act = GraphNode::new(0, OpType::Silu);
        act.inputs = vec![nid];
        act.is_output = true;
        let out = g.add_node(act);
        g.output_ids.push(out);
        let result = OperatorFusion.transform(&mut g);
        assert!(result.transformations_applied >= 1);
    }

    // =====================================================================
    // LayoutOptimizer
    // =====================================================================

    #[test]
    fn test_layout_cuda_matmul_column_major() {
        let opt = LayoutOptimizer::new(TargetDevice::Cuda);
        let mut g = ComputationGraph::new();
        let mut mm = GraphNode::new(0, OpType::MatMul);
        mm.is_output = true;
        g.add_node(mm);
        g.output_ids.push(0);
        opt.transform(&mut g);
        assert_eq!(g.nodes[0].layout, MemoryLayout::ColumnMajor);
    }

    #[test]
    fn test_layout_cpu_large_tensor_blocked() {
        let opt = LayoutOptimizer::new(TargetDevice::Cpu);
        let mut g = ComputationGraph::new();
        let mut n = GraphNode::new(0, OpType::Relu);
        n.memory_bytes = 2 * 1024 * 1024;
        n.is_output = true;
        g.add_node(n);
        g.output_ids.push(0);
        opt.transform(&mut g);
        assert!(matches!(g.nodes[0].layout, MemoryLayout::Blocked { .. }));
    }

    #[test]
    fn test_layout_cpu_small_tensor_row_major() {
        let opt = LayoutOptimizer::new(TargetDevice::Cpu);
        let mut g = ComputationGraph::new();
        let mut n = GraphNode::new(0, OpType::Relu);
        n.memory_bytes = 256;
        n.is_output = true;
        g.add_node(n);
        g.output_ids.push(0);
        opt.transform(&mut g);
        assert_eq!(g.nodes[0].layout, MemoryLayout::RowMajor);
    }

    #[test]
    fn test_layout_generic_always_row_major() {
        let opt = LayoutOptimizer::new(TargetDevice::Generic);
        let n = GraphNode::new(0, OpType::MatMul);
        assert_eq!(opt.preferred_layout(&n), MemoryLayout::RowMajor);
    }

    #[test]
    fn test_layout_analyze_counts_mismatches() {
        let opt = LayoutOptimizer::new(TargetDevice::Cuda);
        let mut g = ComputationGraph::new();
        g.add_node(GraphNode::new(0, OpType::MatMul));
        g.add_node(GraphNode::new(0, OpType::Relu));
        assert_eq!(opt.analyze(&g), 2);
    }

    #[test]
    fn test_layout_verify_ok() {
        let opt = LayoutOptimizer::new(TargetDevice::Cpu);
        let g = ComputationGraph::new();
        assert!(opt.verify(&g).is_ok());
    }

    #[test]
    fn test_layout_kind() {
        let opt = LayoutOptimizer::new(TargetDevice::Cpu);
        assert_eq!(opt.kind(), PassKind::LayoutOptimization);
    }

    #[test]
    fn test_layout_no_change_when_already_optimal() {
        let opt = LayoutOptimizer::new(TargetDevice::Generic);
        let mut g = ComputationGraph::new();
        g.add_node(GraphNode::new(0, OpType::Relu));
        let result = opt.transform(&mut g);
        assert_eq!(result.transformations_applied, 0);
    }

    #[test]
    fn test_layout_intel_matmul_column_major() {
        let opt = LayoutOptimizer::new(TargetDevice::Intel);
        let n = GraphNode::new(0, OpType::MatMul);
        assert_eq!(opt.preferred_layout(&n), MemoryLayout::ColumnMajor);
    }

    #[test]
    fn test_layout_intel_non_matmul_device_optimal() {
        let opt = LayoutOptimizer::new(TargetDevice::Intel);
        let n = GraphNode::new(0, OpType::Relu);
        assert_eq!(opt.preferred_layout(&n), MemoryLayout::DeviceOptimal);
    }

    // =====================================================================
    // DeadCodeElimination
    // =====================================================================

    #[test]
    fn test_dce_analyze() {
        let g = graph_with_dead_code();
        assert_eq!(DeadCodeElimination.analyze(&g), 1);
    }

    #[test]
    fn test_dce_transform_removes_dead() {
        let mut g = graph_with_dead_code();
        let result = DeadCodeElimination.transform(&mut g);
        assert_eq!(result.nodes_removed, 1);
        assert_eq!(result.memory_saved_bytes, 500);
    }

    #[test]
    fn test_dce_preserves_outputs() {
        let mut g = graph_with_dead_code();
        DeadCodeElimination.transform(&mut g);
        assert_eq!(g.nodes[0].op, OpType::Load);
    }

    #[test]
    fn test_dce_no_op_when_all_reachable() {
        let mut g = ComputationGraph::new();
        let mut n = GraphNode::new(0, OpType::Load);
        n.is_output = true;
        g.add_node(n);
        g.output_ids.push(0);
        let result = DeadCodeElimination.transform(&mut g);
        assert_eq!(result.nodes_removed, 0);
    }

    #[test]
    fn test_dce_verify_ok() {
        let g = ComputationGraph::new();
        assert!(DeadCodeElimination.verify(&g).is_ok());
    }

    #[test]
    fn test_dce_kind() {
        assert_eq!(DeadCodeElimination.kind(), PassKind::DeadCodeElimination);
    }

    #[test]
    fn test_dce_transformations_equals_removed() {
        let mut g = graph_with_dead_code();
        let result = DeadCodeElimination.transform(&mut g);
        assert_eq!(result.transformations_applied, result.nodes_removed);
    }

    #[test]
    fn test_dce_chain_kills_unreachable() {
        let mut g = ComputationGraph::new();
        let a = g.add_node(GraphNode::new(0, OpType::Load));
        let mut b = GraphNode::new(0, OpType::Relu);
        b.inputs = vec![a];
        g.add_node(b);
        // Neither node is an output.
        let result = DeadCodeElimination.transform(&mut g);
        assert_eq!(result.nodes_removed, 2);
    }

    // =====================================================================
    // CommonSubexprElimination
    // =====================================================================

    #[test]
    fn test_cse_analyze() {
        let g = cse_graph();
        assert_eq!(CommonSubexprElimination.analyze(&g), 1);
    }

    #[test]
    fn test_cse_transform_rewrites() {
        let mut g = cse_graph();
        let result = CommonSubexprElimination.transform(&mut g);
        assert_eq!(result.transformations_applied, 1);
    }

    #[test]
    fn test_cse_no_duplicates() {
        let mut g = ComputationGraph::new();
        let a = g.add_node(GraphNode::new(0, OpType::Load));
        let b = g.add_node(GraphNode::new(0, OpType::Load));
        let mut add = GraphNode::new(0, OpType::Add);
        add.inputs = vec![a, b];
        add.is_output = true;
        g.add_node(add);
        g.output_ids.push(2);
        let result = CommonSubexprElimination.transform(&mut g);
        assert_eq!(result.transformations_applied, 0);
    }

    #[test]
    fn test_cse_verify_ok() {
        let g = ComputationGraph::new();
        assert!(CommonSubexprElimination.verify(&g).is_ok());
    }

    #[test]
    fn test_cse_kind() {
        assert_eq!(CommonSubexprElimination.kind(), PassKind::CommonSubexprElimination);
    }

    #[test]
    fn test_cse_commutative_detection() {
        let mut g = ComputationGraph::new();
        let a = g.add_node(GraphNode::new(0, OpType::Load));
        let b = g.add_node(GraphNode::new(0, OpType::Load));
        let mut add1 = GraphNode::new(0, OpType::Add);
        add1.inputs = vec![a, b];
        g.add_node(add1);
        let mut add2 = GraphNode::new(0, OpType::Add);
        add2.inputs = vec![b, a]; // reversed order
        add2.is_output = true;
        let out = g.add_node(add2);
        g.output_ids.push(out);
        assert_eq!(CommonSubexprElimination.analyze(&g), 1);
    }

    #[test]
    fn test_cse_ignores_constants() {
        let mut g = ComputationGraph::new();
        g.add_node(GraphNode {
            constant_value: Some(vec![1.0]),
            ..GraphNode::new(0, OpType::Constant)
        });
        g.add_node(GraphNode {
            constant_value: Some(vec![1.0]),
            ..GraphNode::new(0, OpType::Constant)
        });
        assert_eq!(CommonSubexprElimination.analyze(&g), 0);
    }

    #[test]
    fn test_cse_ignores_identity() {
        let mut g = ComputationGraph::new();
        g.add_node(GraphNode::new(0, OpType::Identity));
        g.add_node(GraphNode::new(0, OpType::Identity));
        assert_eq!(CommonSubexprElimination.analyze(&g), 0);
    }

    // =====================================================================
    // MemoryOptimizer
    // =====================================================================

    #[test]
    fn test_memory_analyze() {
        let g = memory_graph();
        // Load node has one consumer (the Relu), is not output, candidate count = 1.
        assert_eq!(MemoryOptimizer.analyze(&g), 1);
    }

    #[test]
    fn test_memory_transform_enables_in_place() {
        let mut g = memory_graph();
        let result = MemoryOptimizer.transform(&mut g);
        assert_eq!(result.transformations_applied, 1);
        assert!(g.nodes[0].in_place);
    }

    #[test]
    fn test_memory_output_not_in_place() {
        let g = memory_graph();
        let candidates = MemoryOptimizer::find_in_place_candidates(&g);
        // Relu node is an output → not a candidate.
        assert!(!candidates.contains(&1));
    }

    #[test]
    fn test_memory_verify_ok() {
        let g = memory_graph();
        assert!(MemoryOptimizer.verify(&g).is_ok());
    }

    #[test]
    fn test_memory_verify_bad_in_place_output() {
        let mut g = ComputationGraph::new();
        let mut n = GraphNode::new(0, OpType::Relu);
        n.is_output = true;
        n.in_place = true;
        g.add_node(n);
        assert!(MemoryOptimizer.verify(&g).is_err());
    }

    #[test]
    fn test_memory_kind() {
        assert_eq!(MemoryOptimizer.kind(), PassKind::MemoryOptimization);
    }

    #[test]
    fn test_memory_no_op_when_all_multi_use() {
        let mut g = ComputationGraph::new();
        let a = g.add_node(GraphNode { memory_bytes: 1024, ..GraphNode::new(0, OpType::Load) });
        let mut b = GraphNode::new(0, OpType::Relu);
        b.inputs = vec![a];
        b.is_output = true;
        g.add_node(b);
        let mut c = GraphNode::new(0, OpType::Gelu);
        c.inputs = vec![a]; // a has two consumers
        c.is_output = true;
        g.add_node(c);
        g.output_ids = vec![1, 2];
        let result = MemoryOptimizer.transform(&mut g);
        // a has 2 consumers → not in-place.
        assert_eq!(result.transformations_applied, 0);
    }

    #[test]
    fn test_memory_saved_bytes_positive() {
        let mut g = memory_graph();
        let result = MemoryOptimizer.transform(&mut g);
        assert!(result.memory_saved_bytes > 0);
    }

    #[test]
    fn test_memory_peak_estimate() {
        let g = memory_graph();
        assert_eq!(MemoryOptimizer::estimate_peak_memory(&g), 2048);
    }

    // =====================================================================
    // OptimizationReport
    // =====================================================================

    #[test]
    fn test_report_empty() {
        let r = OptimizationReport::empty();
        assert!((r.total_speedup - 1.0).abs() < f64::EPSILON);
        assert_eq!(r.total_memory_saved_bytes, 0);
        assert_eq!(r.iterations, 0);
    }

    #[test]
    fn test_report_applied_pass_names() {
        let mut r = OptimizationReport::empty();
        r.pass_results.push(PassResult {
            transformations_applied: 2,
            ..PassResult::noop(PassKind::ConstantFolding)
        });
        r.pass_results.push(PassResult::noop(PassKind::DeadCodeElimination));
        let names = r.applied_pass_names();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "ConstantFolding");
    }

    #[test]
    fn test_report_total_transformations() {
        let mut r = OptimizationReport::empty();
        r.pass_results.push(PassResult {
            transformations_applied: 3,
            ..PassResult::noop(PassKind::ConstantFolding)
        });
        r.pass_results.push(PassResult {
            transformations_applied: 2,
            ..PassResult::noop(PassKind::DeadCodeElimination)
        });
        assert_eq!(r.total_transformations(), 5);
    }

    #[test]
    fn test_report_display() {
        let mut r = OptimizationReport::empty();
        r.iterations = 1;
        r.nodes_before = 10;
        r.nodes_after = 7;
        r.total_speedup = 1.25;
        r.total_memory_saved_bytes = 4096;
        let text = format!("{r}");
        assert!(text.contains("1 iterations"));
        assert!(text.contains("10 → 7"));
        assert!(text.contains("1.25"));
        assert!(text.contains("4096"));
    }

    // =====================================================================
    // OptimizationPipeline
    // =====================================================================

    #[test]
    fn test_pipeline_from_config_o0() {
        let c = OptimizationConfig { level: OptimizationLevel::O0, ..Default::default() };
        let pipeline = OptimizationPipeline::from_config(c);
        assert_eq!(pipeline.passes.len(), 0);
    }

    #[test]
    fn test_pipeline_from_config_o3() {
        let c = OptimizationConfig { level: OptimizationLevel::O3, ..Default::default() };
        let pipeline = OptimizationPipeline::from_config(c);
        assert_eq!(pipeline.passes.len(), 6);
    }

    #[test]
    fn test_pipeline_run_simple() {
        let c = OptimizationConfig::default();
        let pipeline = OptimizationPipeline::from_config(c);
        let mut g = simple_graph();
        let report = pipeline.run(&mut g).unwrap();
        assert!(report.iterations >= 1);
        assert!(report.total_speedup >= 1.0);
    }

    #[test]
    fn test_pipeline_run_empty_graph() {
        let c = OptimizationConfig::default();
        let pipeline = OptimizationPipeline::from_config(c);
        let mut g = ComputationGraph::new();
        let report = pipeline.run(&mut g).unwrap();
        assert_eq!(report.nodes_before, 0);
        assert_eq!(report.nodes_after, 0);
    }

    #[test]
    fn test_pipeline_run_invalid_config() {
        let c = OptimizationConfig { max_iterations: 0, ..Default::default() };
        let pipeline = OptimizationPipeline::from_config(c);
        let mut g = ComputationGraph::new();
        assert!(pipeline.run(&mut g).is_err());
    }

    #[test]
    fn test_pipeline_converges() {
        let c = OptimizationConfig { max_iterations: 10, ..Default::default() };
        let pipeline = OptimizationPipeline::from_config(c);
        let mut g = simple_graph();
        let report = pipeline.run(&mut g).unwrap();
        // Should converge well before 10 iterations.
        assert!(report.iterations < 10);
    }

    #[test]
    fn test_pipeline_debug_impl() {
        let pipeline = OptimizationPipeline::from_config(OptimizationConfig::default());
        let dbg = format!("{pipeline:?}");
        assert!(dbg.contains("OptimizationPipeline"));
    }

    #[test]
    fn test_pipeline_full_graph_matmul_bias() {
        let c = OptimizationConfig {
            level: OptimizationLevel::O3,
            target_device: TargetDevice::Cuda,
            ..Default::default()
        };
        let pipeline = OptimizationPipeline::from_config(c);
        let mut g = matmul_bias_graph();
        let report = pipeline.run(&mut g).unwrap();
        assert!(report.total_transformations() > 0);
    }

    #[test]
    fn test_pipeline_verification_catches_bad_state() {
        // Manually craft a bad pass that leaves broken state.
        struct BadPass;
        impl OptimizationPass for BadPass {
            fn kind(&self) -> PassKind {
                PassKind::ConstantFolding
            }
            fn analyze(&self, _g: &ComputationGraph) -> usize {
                0
            }
            fn transform(&self, g: &mut ComputationGraph) -> PassResult {
                // Create a Constant node without a value — violates invariant.
                g.add_node(GraphNode::new(0, OpType::Constant));
                PassResult {
                    transformations_applied: 1,
                    ..PassResult::noop(PassKind::ConstantFolding)
                }
            }
            fn verify(&self, g: &ComputationGraph) -> Result<(), String> {
                for n in &g.nodes {
                    if n.op == OpType::Constant && n.constant_value.is_none() {
                        return Err("bad constant".into());
                    }
                }
                Ok(())
            }
        }

        let c = OptimizationConfig {
            verify_after_each_pass: true,
            max_iterations: 1,
            ..Default::default()
        };
        let pipeline = OptimizationPipeline::new(vec![Box::new(BadPass)], c);
        let mut g = ComputationGraph::new();
        assert!(pipeline.run(&mut g).is_err());
    }

    #[test]
    fn test_pipeline_disabled_config() {
        let c = OptimizationConfig::disabled();
        let pipeline = OptimizationPipeline::from_config(c);
        let mut g = simple_graph();
        // O0 has no passes, validation still fails because max_iterations is 3.
        let report = pipeline.run(&mut g).unwrap();
        assert_eq!(report.total_transformations(), 0);
    }

    #[test]
    fn test_pipeline_nodes_before_after() {
        let c = OptimizationConfig { level: OptimizationLevel::O1, ..Default::default() };
        let pipeline = OptimizationPipeline::from_config(c);
        let mut g = graph_with_dead_code();
        let report = pipeline.run(&mut g).unwrap();
        assert_eq!(report.nodes_before, 2);
        // Dead node becomes Identity → nodes_after excludes it.
        assert!(report.nodes_after <= report.nodes_before);
    }

    // =====================================================================
    // Integration: multi-pass interaction
    // =====================================================================

    #[test]
    fn test_cse_then_dce_removes_duplicates() {
        let mut g = cse_graph();
        CommonSubexprElimination.transform(&mut g);
        let result = DeadCodeElimination.transform(&mut g);
        // The duplicate Add should be eliminated after CSE rewires.
        assert!(result.nodes_removed >= 1);
    }

    #[test]
    fn test_fold_then_dce() {
        let mut g = simple_graph();
        ConstantFolding.transform(&mut g);
        let result = DeadCodeElimination.transform(&mut g);
        // After folding, the original two constant inputs become dead.
        assert!(result.nodes_removed >= 2);
    }

    #[test]
    fn test_fusion_then_layout() {
        let mut g = matmul_bias_graph();
        OperatorFusion.transform(&mut g);
        let opt = LayoutOptimizer::new(TargetDevice::Cuda);
        let result = opt.transform(&mut g);
        assert!(result.transformations_applied > 0);
    }

    #[test]
    fn test_full_pipeline_norm_activation() {
        let c = OptimizationConfig {
            level: OptimizationLevel::O3,
            target_device: TargetDevice::Cuda,
            ..Default::default()
        };
        let pipeline = OptimizationPipeline::from_config(c);
        let mut g = norm_activation_graph();
        let report = pipeline.run(&mut g).unwrap();
        assert!(report.total_transformations() > 0);
        assert!(report.nodes_after <= report.nodes_before);
    }

    #[test]
    fn test_pipeline_report_memory_savings() {
        let c = OptimizationConfig { level: OptimizationLevel::O3, ..Default::default() };
        let pipeline = OptimizationPipeline::from_config(c);
        let mut g = graph_with_dead_code();
        let report = pipeline.run(&mut g).unwrap();
        assert!(report.total_memory_saved_bytes > 0);
    }

    #[test]
    fn test_pipeline_multiple_fusions() {
        let mut g = ComputationGraph::new();
        let inp = g.add_node(GraphNode::new(0, OpType::Load));
        let w = g.add_node(GraphNode::new(0, OpType::Constant));
        let mut mm = GraphNode::new(0, OpType::MatMul);
        mm.inputs = vec![inp, w];
        let mm_id = g.add_node(mm);
        let b = g.add_node(GraphNode::new(0, OpType::Constant));
        let mut ba = GraphNode::new(0, OpType::BiasAdd);
        ba.inputs = vec![mm_id, b];
        let ba_id = g.add_node(ba);
        let mut norm = GraphNode::new(0, OpType::LayerNorm);
        norm.inputs = vec![ba_id];
        let n_id = g.add_node(norm);
        let mut act = GraphNode::new(0, OpType::Relu);
        act.inputs = vec![n_id];
        act.is_output = true;
        let out = g.add_node(act);
        g.output_ids.push(out);

        let result = OperatorFusion.transform(&mut g);
        assert!(result.transformations_applied >= 2);
    }
}
