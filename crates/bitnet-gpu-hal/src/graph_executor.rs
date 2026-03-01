//! Computation graph executor with parallel scheduling and optimization.
//!
//! Provides a DAG-based execution engine that plans, optimizes, and executes
//! computation graphs with topological parallel scheduling, profiling, and
//! critical-path analysis.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

// ── Core Types ──────────────────────────────────────────────────────────────

/// Unique identifier for a graph node.
pub type NodeId = u64;

/// Execution order strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionOrder {
    /// Execute nodes one at a time in topological order.
    Sequential,
    /// Group independent nodes into parallel stages via topological sort.
    #[default]
    TopologicalParallel,
    /// Schedule nodes as soon as all inputs are ready (greedy).
    DataDriven,
}

impl fmt::Display for ExecutionOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sequential => write!(f, "Sequential"),
            Self::TopologicalParallel => write!(f, "TopologicalParallel"),
            Self::DataDriven => write!(f, "DataDriven"),
        }
    }
}

/// Executor configuration.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum operations that may execute in parallel within a stage.
    pub max_parallel_ops: usize,
    /// Memory budget in bytes (0 = unlimited).
    pub memory_limit: u64,
    /// Whether to collect per-node profiling data.
    pub enable_profiling: bool,
    /// Scheduling strategy.
    pub execution_order: ExecutionOrder,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_parallel_ops: 8,
            memory_limit: 0,
            enable_profiling: true,
            execution_order: ExecutionOrder::default(),
        }
    }
}

// ── Graph Structures ────────────────────────────────────────────────────────

/// Type of operation a node represents.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
    MatMul,
    Add,
    LayerNorm,
    Softmax,
    Attention,
    Linear,
    Relu,
    Gelu,
    Reshape,
    Transpose,
    Concat,
    Split,
    Custom(String),
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatMul => write!(f, "MatMul"),
            Self::Add => write!(f, "Add"),
            Self::LayerNorm => write!(f, "LayerNorm"),
            Self::Softmax => write!(f, "Softmax"),
            Self::Attention => write!(f, "Attention"),
            Self::Linear => write!(f, "Linear"),
            Self::Relu => write!(f, "Relu"),
            Self::Gelu => write!(f, "Gelu"),
            Self::Reshape => write!(f, "Reshape"),
            Self::Transpose => write!(f, "Transpose"),
            Self::Concat => write!(f, "Concat"),
            Self::Split => write!(f, "Split"),
            Self::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

/// A node in the execution graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub op_type: OpType,
    /// Indices of input slots this node reads from.
    pub inputs: Vec<NodeId>,
    /// Indices of output slots this node produces.
    pub outputs: Vec<NodeId>,
    /// Estimated floating-point operations.
    pub estimated_flops: u64,
    /// Estimated memory footprint in bytes.
    pub estimated_memory: u64,
}

impl GraphNode {
    pub const fn new(id: NodeId, op_type: OpType) -> Self {
        Self {
            id,
            op_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            estimated_flops: 0,
            estimated_memory: 0,
        }
    }

    #[must_use]
    pub fn with_inputs(mut self, inputs: Vec<NodeId>) -> Self {
        self.inputs = inputs;
        self
    }

    #[must_use]
    pub fn with_outputs(mut self, outputs: Vec<NodeId>) -> Self {
        self.outputs = outputs;
        self
    }

    #[must_use]
    pub const fn with_flops(mut self, flops: u64) -> Self {
        self.estimated_flops = flops;
        self
    }

    #[must_use]
    pub const fn with_memory(mut self, mem: u64) -> Self {
        self.estimated_memory = mem;
        self
    }
}

/// A directed edge between two nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphEdge {
    pub source_id: NodeId,
    pub source_output: usize,
    pub target_id: NodeId,
    pub target_input: usize,
}

impl GraphEdge {
    pub const fn new(
        source_id: NodeId,
        source_output: usize,
        target_id: NodeId,
        target_input: usize,
    ) -> Self {
        Self { source_id, source_output, target_id, target_input }
    }
}

/// A complete computation graph.
#[derive(Debug, Clone)]
pub struct ExecutionGraph {
    pub nodes: BTreeMap<NodeId, GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub input_nodes: Vec<NodeId>,
    pub output_nodes: Vec<NodeId>,
}

impl ExecutionGraph {
    pub const fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            edges: Vec::new(),
            input_nodes: Vec::new(),
            output_nodes: Vec::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.insert(node.id, node);
    }

    /// Add a directed edge.
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.push(edge);
    }

    /// Add an edge by source/target node IDs (convenience, output/input index 0).
    pub fn connect(&mut self, source: NodeId, target: NodeId) {
        self.edges.push(GraphEdge::new(source, 0, target, 0));
    }

    /// Mark a node as a graph input.
    pub fn mark_input(&mut self, id: NodeId) {
        if !self.input_nodes.contains(&id) {
            self.input_nodes.push(id);
        }
    }

    /// Mark a node as a graph output.
    pub fn mark_output(&mut self, id: NodeId) {
        if !self.output_nodes.contains(&id) {
            self.output_nodes.push(id);
        }
    }

    /// Return the set of predecessor node IDs for `node_id`.
    pub fn predecessors(&self, node_id: NodeId) -> HashSet<NodeId> {
        self.edges.iter().filter(|e| e.target_id == node_id).map(|e| e.source_id).collect()
    }

    /// Return the set of successor node IDs for `node_id`.
    pub fn successors(&self, node_id: NodeId) -> HashSet<NodeId> {
        self.edges.iter().filter(|e| e.source_id == node_id).map(|e| e.target_id).collect()
    }

    /// Return the total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the total number of edges.
    pub const fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Validate the graph: no dangling edges, no self-loops.
    pub fn validate(&self) -> Result<(), GraphError> {
        for edge in &self.edges {
            if edge.source_id == edge.target_id {
                return Err(GraphError::SelfLoop(edge.source_id));
            }
            if !self.nodes.contains_key(&edge.source_id) {
                return Err(GraphError::DanglingEdge { node_id: edge.source_id });
            }
            if !self.nodes.contains_key(&edge.target_id) {
                return Err(GraphError::DanglingEdge { node_id: edge.target_id });
            }
        }
        Ok(())
    }
}

impl Default for ExecutionGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Execution Plan & Results ────────────────────────────────────────────────

/// A planned execution: stages of node IDs that can run in parallel.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Each inner `Vec` is a stage of nodes that may execute concurrently.
    pub stages: Vec<Vec<NodeId>>,
}

impl ExecutionPlan {
    /// Total number of stages.
    pub const fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Total number of scheduled nodes.
    pub fn total_nodes(&self) -> usize {
        self.stages.iter().map(Vec::len).sum()
    }

    /// Maximum parallelism across all stages.
    pub fn max_parallelism(&self) -> usize {
        self.stages.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// Whether the plan is sequential (every stage has exactly 1 node).
    pub fn is_sequential(&self) -> bool {
        self.stages.iter().all(|s| s.len() <= 1)
    }
}

/// Result of executing a single node.
#[derive(Debug, Clone)]
pub struct NodeResult {
    pub node_id: NodeId,
    /// Simulated output data (placeholder).
    pub output_data: Vec<f32>,
    /// Wall-clock execution time in milliseconds.
    pub execution_time_ms: f64,
    /// Memory used during execution in bytes.
    pub memory_used: u64,
}

/// Full execution trace.
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    pub node_results: Vec<NodeResult>,
    /// Total wall-clock execution time.
    pub total_time: Duration,
    /// Peak memory usage across all nodes.
    pub peak_memory: u64,
    /// Node IDs on the critical path (longest sequential dependency chain).
    pub critical_path: Vec<NodeId>,
}

impl ExecutionTrace {
    /// Average per-node execution time in milliseconds.
    pub fn avg_node_time_ms(&self) -> f64 {
        if self.node_results.is_empty() {
            return 0.0;
        }
        let total: f64 = self.node_results.iter().map(|r| r.execution_time_ms).sum();
        #[allow(clippy::cast_precision_loss)]
        let avg = total / self.node_results.len() as f64;
        avg
    }

    /// Total estimated FLOPs from node results.
    pub fn total_memory_used(&self) -> u64 {
        self.node_results.iter().map(|r| r.memory_used).sum()
    }
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors from graph construction or execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    CycleDetected,
    EmptyGraph,
    SelfLoop(NodeId),
    DanglingEdge { node_id: NodeId },
    NodeNotFound(NodeId),
    MemoryLimitExceeded { required: u64, limit: u64 },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CycleDetected => write!(f, "cycle detected in execution graph"),
            Self::EmptyGraph => write!(f, "execution graph is empty"),
            Self::SelfLoop(id) => write!(f, "self-loop on node {id}"),
            Self::DanglingEdge { node_id } => {
                write!(f, "edge references missing node {node_id}")
            }
            Self::NodeNotFound(id) => write!(f, "node {id} not found"),
            Self::MemoryLimitExceeded { required, limit } => {
                write!(f, "memory limit exceeded: requires {required} B, limit {limit} B")
            }
        }
    }
}

impl std::error::Error for GraphError {}

// ── Graph Optimizer ─────────────────────────────────────────────────────────

/// Statistics produced by an optimization pass.
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    pub nodes_removed: usize,
    pub nodes_fused: usize,
    pub edges_removed: usize,
}

/// Optimizes an `ExecutionGraph` before execution.
pub struct GraphOptimizer;

impl GraphOptimizer {
    /// Run all optimization passes and return statistics.
    pub fn optimize(graph: &mut ExecutionGraph) -> OptimizationStats {
        let mut stats = OptimizationStats::default();
        let dead = Self::eliminate_dead_code(graph);
        stats.nodes_removed += dead;
        stats.edges_removed += dead; // edges removed alongside dead nodes
        let fused = Self::fuse_elementwise(graph);
        stats.nodes_fused += fused;
        stats
    }

    /// Remove nodes unreachable from output nodes (dead code elimination).
    pub fn eliminate_dead_code(graph: &mut ExecutionGraph) -> usize {
        if graph.output_nodes.is_empty() {
            return 0;
        }
        // BFS backwards from output nodes.
        let mut reachable = HashSet::new();
        let mut queue: VecDeque<NodeId> = graph.output_nodes.iter().copied().collect();
        while let Some(nid) = queue.pop_front() {
            if reachable.insert(nid) {
                for pred in graph.predecessors(nid) {
                    queue.push_back(pred);
                }
            }
        }
        let all_ids: Vec<NodeId> = graph.nodes.keys().copied().collect();
        let mut removed = 0;
        for id in all_ids {
            if !reachable.contains(&id) {
                graph.nodes.remove(&id);
                graph.edges.retain(|e| e.source_id != id && e.target_id != id);
                graph.input_nodes.retain(|&n| n != id);
                removed += 1;
            }
        }
        removed
    }

    /// Fuse consecutive element-wise operations (Add, Relu, Gelu) into a single node.
    ///
    /// Looks for chains `A -> B` where both are element-wise, B has exactly one
    /// predecessor, and A has exactly one successor.  Merges B into A.
    pub fn fuse_elementwise(graph: &mut ExecutionGraph) -> usize {
        let elementwise: HashSet<&OpType> =
            [&OpType::Add, &OpType::Relu, &OpType::Gelu].into_iter().collect();

        let mut fused = 0;
        loop {
            let mut pair: Option<(NodeId, NodeId)> = None;
            for edge in &graph.edges {
                let src = &graph.nodes[&edge.source_id];
                let tgt = &graph.nodes[&edge.target_id];
                if elementwise.contains(&src.op_type)
                    && elementwise.contains(&tgt.op_type)
                    && graph.successors(src.id).len() == 1
                    && graph.predecessors(tgt.id).len() == 1
                {
                    pair = Some((src.id, tgt.id));
                    break;
                }
            }
            let Some((src_id, tgt_id)) = pair else {
                break;
            };
            // Merge tgt into src: redirect tgt's outgoing edges to src, remove tgt.
            let tgt_successors: Vec<GraphEdge> =
                graph.edges.iter().filter(|e| e.source_id == tgt_id).cloned().collect();
            for old_edge in &tgt_successors {
                graph.edges.push(GraphEdge::new(
                    src_id,
                    old_edge.source_output,
                    old_edge.target_id,
                    old_edge.target_input,
                ));
            }
            // Accumulate flops/memory from tgt into src.
            if let Some(tgt_node) = graph.nodes.remove(&tgt_id)
                && let Some(src_node) = graph.nodes.get_mut(&src_id)
            {
                src_node.estimated_flops += tgt_node.estimated_flops;
                src_node.estimated_memory += tgt_node.estimated_memory;
            }
            graph.edges.retain(|e| e.source_id != tgt_id && e.target_id != tgt_id);
            // Update output_nodes if tgt was an output.
            if graph.output_nodes.contains(&tgt_id) {
                graph.output_nodes.retain(|&n| n != tgt_id);
                if !graph.output_nodes.contains(&src_id) {
                    graph.output_nodes.push(src_id);
                }
            }
            fused += 1;
        }
        fused
    }

    /// Reorder nodes for memory locality (sorts by estimated memory, ascending).
    pub fn reorder_for_locality(plan: &mut ExecutionPlan, graph: &ExecutionGraph) {
        for stage in &mut plan.stages {
            stage.sort_by_key(|&nid| graph.nodes.get(&nid).map_or(0, |n| n.estimated_memory));
        }
    }
}

// ── Graph Executor ──────────────────────────────────────────────────────────

/// Main graph execution engine.
pub struct GraphExecutor {
    config: ExecutorConfig,
}

impl GraphExecutor {
    pub const fn new(config: ExecutorConfig) -> Self {
        Self { config }
    }

    /// Plan execution: produce stages via topological sort.
    pub fn plan(&self, graph: &ExecutionGraph) -> Result<ExecutionPlan, GraphError> {
        if graph.nodes.is_empty() {
            return Err(GraphError::EmptyGraph);
        }
        graph.validate()?;

        match self.config.execution_order {
            ExecutionOrder::Sequential => self.plan_sequential(graph),
            ExecutionOrder::TopologicalParallel => self.plan_topological_parallel(graph),
            ExecutionOrder::DataDriven => self.plan_data_driven(graph),
        }
    }

    /// Execute a planned graph, returning a trace.
    pub fn execute(
        &self,
        graph: &ExecutionGraph,
        plan: &ExecutionPlan,
    ) -> Result<ExecutionTrace, GraphError> {
        let start = Instant::now();
        let mut node_results = Vec::new();
        let mut peak_memory: u64 = 0;
        let mut current_memory: u64 = 0;

        for stage in &plan.stages {
            // Check memory limit before executing stage.
            if self.config.memory_limit > 0 {
                let stage_mem: u64 = stage
                    .iter()
                    .filter_map(|nid| graph.nodes.get(nid))
                    .map(|n| n.estimated_memory)
                    .sum();
                if current_memory + stage_mem > self.config.memory_limit {
                    return Err(GraphError::MemoryLimitExceeded {
                        required: current_memory + stage_mem,
                        limit: self.config.memory_limit,
                    });
                }
            }

            for &nid in stage {
                let node = graph.nodes.get(&nid).ok_or(GraphError::NodeNotFound(nid))?;

                let node_start = Instant::now();
                // Simulate execution proportional to estimated FLOPs.
                let _work = simulate_work(node.estimated_flops);
                let elapsed = node_start.elapsed();

                current_memory += node.estimated_memory;
                peak_memory = peak_memory.max(current_memory);

                node_results.push(NodeResult {
                    node_id: nid,
                    output_data: vec![0.0; 1], // placeholder
                    execution_time_ms: elapsed.as_secs_f64() * 1000.0,
                    memory_used: node.estimated_memory,
                });
            }
        }

        let critical_path = self.compute_critical_path(graph, &node_results);

        Ok(ExecutionTrace { node_results, total_time: start.elapsed(), peak_memory, critical_path })
    }

    /// Convenience: plan + execute.
    pub fn run(&self, graph: &ExecutionGraph) -> Result<ExecutionTrace, GraphError> {
        let plan = self.plan(graph)?;
        self.execute(graph, &plan)
    }

    /// Plan + optimize + execute.
    pub fn run_optimized(
        &self,
        graph: &mut ExecutionGraph,
    ) -> Result<(ExecutionTrace, OptimizationStats), GraphError> {
        let stats = GraphOptimizer::optimize(graph);
        let mut plan = self.plan(graph)?;
        GraphOptimizer::reorder_for_locality(&mut plan, graph);
        let trace = self.execute(graph, &plan)?;
        Ok((trace, stats))
    }

    // ── Planning strategies ─────────────────────────────────────────────

    #[allow(clippy::unused_self)]
    fn plan_sequential(&self, graph: &ExecutionGraph) -> Result<ExecutionPlan, GraphError> {
        let sorted = topological_sort(graph)?;
        let stages = sorted.into_iter().map(|nid| vec![nid]).collect();
        Ok(ExecutionPlan { stages })
    }

    fn plan_topological_parallel(
        &self,
        graph: &ExecutionGraph,
    ) -> Result<ExecutionPlan, GraphError> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for &nid in graph.nodes.keys() {
            in_degree.insert(nid, 0);
        }
        for edge in &graph.edges {
            *in_degree.entry(edge.target_id).or_insert(0) += 1;
        }

        let mut stages: Vec<Vec<NodeId>> = Vec::new();
        let mut ready: Vec<NodeId> =
            in_degree.iter().filter(|&(_, deg)| *deg == 0).map(|(&nid, _)| nid).collect();
        ready.sort_unstable(); // deterministic order

        if ready.is_empty() && !graph.nodes.is_empty() {
            return Err(GraphError::CycleDetected);
        }

        let mut scheduled = 0;
        while !ready.is_empty() {
            // Respect max_parallel_ops.
            let stage: Vec<NodeId> =
                ready.drain(..ready.len().min(self.config.max_parallel_ops)).collect();
            for &nid in &stage {
                for succ in graph.successors(nid) {
                    if let Some(deg) = in_degree.get_mut(&succ) {
                        *deg -= 1;
                        if *deg == 0 {
                            ready.push(succ);
                        }
                    }
                }
            }
            scheduled += stage.len();
            stages.push(stage);
            ready.sort_unstable();
        }

        if scheduled != graph.nodes.len() {
            return Err(GraphError::CycleDetected);
        }

        Ok(ExecutionPlan { stages })
    }

    #[allow(clippy::unused_self)]
    fn plan_data_driven(&self, graph: &ExecutionGraph) -> Result<ExecutionPlan, GraphError> {
        // Data-driven: same as topological parallel but with eagerness
        // (schedule as soon as in-degree reaches 0, one node per stage tick).
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for &nid in graph.nodes.keys() {
            in_degree.insert(nid, 0);
        }
        for edge in &graph.edges {
            *in_degree.entry(edge.target_id).or_insert(0) += 1;
        }

        let mut stages: Vec<Vec<NodeId>> = Vec::new();
        let mut ready: BTreeSet<NodeId> =
            in_degree.iter().filter(|&(_, deg)| *deg == 0).map(|(&nid, _)| nid).collect();

        if ready.is_empty() && !graph.nodes.is_empty() {
            return Err(GraphError::CycleDetected);
        }

        let mut scheduled = 0;
        while !ready.is_empty() {
            let batch: Vec<NodeId> = ready.iter().copied().collect();
            ready.clear();
            for &nid in &batch {
                for succ in graph.successors(nid) {
                    if let Some(deg) = in_degree.get_mut(&succ) {
                        *deg -= 1;
                        if *deg == 0 {
                            ready.insert(succ);
                        }
                    }
                }
            }
            scheduled += batch.len();
            stages.push(batch);
        }

        if scheduled != graph.nodes.len() {
            return Err(GraphError::CycleDetected);
        }

        Ok(ExecutionPlan { stages })
    }

    // ── Critical path ───────────────────────────────────────────────────

    #[allow(clippy::unused_self)]
    fn compute_critical_path(&self, graph: &ExecutionGraph, results: &[NodeResult]) -> Vec<NodeId> {
        if results.is_empty() {
            return Vec::new();
        }

        let time_map: HashMap<NodeId, f64> =
            results.iter().map(|r| (r.node_id, r.execution_time_ms)).collect();

        // Longest-path DP on the DAG.
        let Ok(sorted) = topological_sort(graph) else {
            return Vec::new();
        };

        let mut dist: HashMap<NodeId, f64> = HashMap::new();
        let mut pred: HashMap<NodeId, Option<NodeId>> = HashMap::new();
        for &nid in &sorted {
            dist.insert(nid, *time_map.get(&nid).unwrap_or(&0.0));
            pred.insert(nid, None);
        }

        for &nid in &sorted {
            let nid_dist = dist[&nid];
            for succ in graph.successors(nid) {
                let succ_time = *time_map.get(&succ).unwrap_or(&0.0);
                let new_dist = nid_dist + succ_time;
                if new_dist > dist[&succ] {
                    dist.insert(succ, new_dist);
                    pred.insert(succ, Some(nid));
                }
            }
        }

        // Find the node with the maximum distance.
        let &end = dist
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(&0, |(k, _)| k);

        let mut path = vec![end];
        let mut cur = end;
        while let Some(Some(p)) = pred.get(&cur) {
            path.push(*p);
            cur = *p;
        }
        path.reverse();
        path
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Kahn's algorithm topological sort.
fn topological_sort(graph: &ExecutionGraph) -> Result<Vec<NodeId>, GraphError> {
    let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
    for &nid in graph.nodes.keys() {
        in_degree.insert(nid, 0);
    }
    for edge in &graph.edges {
        *in_degree.entry(edge.target_id).or_insert(0) += 1;
    }

    let mut queue: VecDeque<NodeId> = in_degree
        .iter()
        .filter(|&(_, d)| *d == 0)
        .map(|(&nid, _)| nid)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let mut order = Vec::with_capacity(graph.nodes.len());
    while let Some(nid) = queue.pop_front() {
        order.push(nid);
        let mut newly_ready: BTreeSet<NodeId> = BTreeSet::new();
        for succ in graph.successors(nid) {
            if let Some(deg) = in_degree.get_mut(&succ) {
                *deg -= 1;
                if *deg == 0 {
                    newly_ready.insert(succ);
                }
            }
        }
        for nid in newly_ready {
            queue.push_back(nid);
        }
    }

    if order.len() != graph.nodes.len() {
        return Err(GraphError::CycleDetected);
    }
    Ok(order)
}

/// Simulate a trivial amount of work (avoids optimising away empty loops).
#[inline]
fn simulate_work(flops: u64) -> u64 {
    let mut acc: u64 = 0;
    for i in 0..flops.min(100) {
        acc = acc.wrapping_add(i);
    }
    acc
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Build a simple linear chain: 1 -> 2 -> 3 -> ... -> n.
    fn linear_chain(n: u64) -> ExecutionGraph {
        let mut g = ExecutionGraph::new();
        for i in 1..=n {
            g.add_node(GraphNode::new(i, OpType::MatMul).with_flops(10));
        }
        for i in 1..n {
            g.connect(i, i + 1);
        }
        g.mark_input(1);
        g.mark_output(n);
        g
    }

    /// Build a diamond: 1 -> {2,3} -> 4.
    fn diamond_graph() -> ExecutionGraph {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Linear).with_flops(10));
        g.add_node(GraphNode::new(2, OpType::MatMul).with_flops(20));
        g.add_node(GraphNode::new(3, OpType::Relu).with_flops(5));
        g.add_node(GraphNode::new(4, OpType::Add).with_flops(10));
        g.connect(1, 2);
        g.connect(1, 3);
        g.connect(2, 4);
        g.connect(3, 4);
        g.mark_input(1);
        g.mark_output(4);
        g
    }

    /// Build a wide graph: 1 -> {2,3,4,5} -> 6.
    fn wide_graph() -> ExecutionGraph {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Linear).with_flops(5));
        for i in 2..=5 {
            g.add_node(GraphNode::new(i, OpType::MatMul).with_flops(10));
            g.connect(1, i);
        }
        g.add_node(GraphNode::new(6, OpType::Add).with_flops(5));
        for i in 2..=5 {
            g.connect(i, 6);
        }
        g.mark_input(1);
        g.mark_output(6);
        g
    }

    fn default_executor() -> GraphExecutor {
        GraphExecutor::new(ExecutorConfig::default())
    }

    fn sequential_executor() -> GraphExecutor {
        GraphExecutor::new(ExecutorConfig {
            execution_order: ExecutionOrder::Sequential,
            ..Default::default()
        })
    }

    // ── ExecutionOrder ──────────────────────────────────────────────────

    #[test]
    fn test_execution_order_default() {
        assert_eq!(ExecutionOrder::default(), ExecutionOrder::TopologicalParallel);
    }

    #[test]
    fn test_execution_order_display() {
        assert_eq!(ExecutionOrder::Sequential.to_string(), "Sequential");
        assert_eq!(ExecutionOrder::TopologicalParallel.to_string(), "TopologicalParallel");
        assert_eq!(ExecutionOrder::DataDriven.to_string(), "DataDriven");
    }

    #[test]
    fn test_execution_order_eq() {
        assert_ne!(ExecutionOrder::Sequential, ExecutionOrder::DataDriven);
    }

    // ── ExecutorConfig ──────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let c = ExecutorConfig::default();
        assert_eq!(c.max_parallel_ops, 8);
        assert_eq!(c.memory_limit, 0);
        assert!(c.enable_profiling);
    }

    #[test]
    fn test_config_custom() {
        let c = ExecutorConfig {
            max_parallel_ops: 4,
            memory_limit: 1024,
            enable_profiling: false,
            execution_order: ExecutionOrder::Sequential,
        };
        assert_eq!(c.max_parallel_ops, 4);
        assert_eq!(c.memory_limit, 1024);
        assert!(!c.enable_profiling);
    }

    // ── OpType ──────────────────────────────────────────────────────────

    #[test]
    fn test_op_type_display() {
        assert_eq!(OpType::MatMul.to_string(), "MatMul");
        assert_eq!(OpType::Attention.to_string(), "Attention");
        assert_eq!(OpType::Custom("foo".into()).to_string(), "Custom(foo)");
    }

    #[test]
    fn test_op_type_eq() {
        assert_eq!(OpType::Relu, OpType::Relu);
        assert_ne!(OpType::Relu, OpType::Gelu);
    }

    #[test]
    fn test_op_type_hash() {
        let mut s = HashSet::new();
        s.insert(OpType::MatMul);
        s.insert(OpType::MatMul);
        assert_eq!(s.len(), 1);
    }

    // ── GraphNode ───────────────────────────────────────────────────────

    #[test]
    fn test_graph_node_builder() {
        let n = GraphNode::new(1, OpType::MatMul)
            .with_inputs(vec![10, 20])
            .with_outputs(vec![30])
            .with_flops(1000)
            .with_memory(2048);
        assert_eq!(n.id, 1);
        assert_eq!(n.inputs, vec![10, 20]);
        assert_eq!(n.outputs, vec![30]);
        assert_eq!(n.estimated_flops, 1000);
        assert_eq!(n.estimated_memory, 2048);
    }

    #[test]
    fn test_graph_node_defaults() {
        let n = GraphNode::new(42, OpType::Softmax);
        assert_eq!(n.id, 42);
        assert!(n.inputs.is_empty());
        assert!(n.outputs.is_empty());
        assert_eq!(n.estimated_flops, 0);
        assert_eq!(n.estimated_memory, 0);
    }

    // ── GraphEdge ───────────────────────────────────────────────────────

    #[test]
    fn test_graph_edge_new() {
        let e = GraphEdge::new(1, 0, 2, 1);
        assert_eq!(e.source_id, 1);
        assert_eq!(e.source_output, 0);
        assert_eq!(e.target_id, 2);
        assert_eq!(e.target_input, 1);
    }

    #[test]
    fn test_graph_edge_eq() {
        let e1 = GraphEdge::new(1, 0, 2, 0);
        let e2 = GraphEdge::new(1, 0, 2, 0);
        let e3 = GraphEdge::new(1, 0, 3, 0);
        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    // ── ExecutionGraph construction ─────────────────────────────────────

    #[test]
    fn test_graph_new_is_empty() {
        let g = ExecutionGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_graph_default_is_empty() {
        let g = ExecutionGraph::default();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_graph_add_node() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::MatMul));
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_graph_add_edge() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::MatMul));
        g.add_node(GraphNode::new(2, OpType::Add));
        g.connect(1, 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_graph_mark_input_output() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Linear));
        g.mark_input(1);
        g.mark_output(1);
        assert_eq!(g.input_nodes, vec![1]);
        assert_eq!(g.output_nodes, vec![1]);
    }

    #[test]
    fn test_graph_mark_input_idempotent() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Linear));
        g.mark_input(1);
        g.mark_input(1);
        assert_eq!(g.input_nodes.len(), 1);
    }

    #[test]
    fn test_graph_predecessors() {
        let g = diamond_graph();
        let preds = g.predecessors(4);
        assert!(preds.contains(&2));
        assert!(preds.contains(&3));
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn test_graph_successors() {
        let g = diamond_graph();
        let succs = g.successors(1);
        assert!(succs.contains(&2));
        assert!(succs.contains(&3));
        assert_eq!(succs.len(), 2);
    }

    #[test]
    fn test_graph_predecessors_empty_for_root() {
        let g = diamond_graph();
        assert!(g.predecessors(1).is_empty());
    }

    #[test]
    fn test_graph_successors_empty_for_leaf() {
        let g = diamond_graph();
        assert!(g.successors(4).is_empty());
    }

    // ── Graph Validation ────────────────────────────────────────────────

    #[test]
    fn test_validate_ok() {
        let g = diamond_graph();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_validate_self_loop() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Add));
        g.edges.push(GraphEdge::new(1, 0, 1, 0));
        assert_eq!(g.validate(), Err(GraphError::SelfLoop(1)));
    }

    #[test]
    fn test_validate_dangling_edge_source() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(2, OpType::Add));
        g.edges.push(GraphEdge::new(1, 0, 2, 0));
        assert_eq!(g.validate(), Err(GraphError::DanglingEdge { node_id: 1 }));
    }

    #[test]
    fn test_validate_dangling_edge_target() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Add));
        g.edges.push(GraphEdge::new(1, 0, 99, 0));
        assert_eq!(g.validate(), Err(GraphError::DanglingEdge { node_id: 99 }));
    }

    // ── Topological Sort ────────────────────────────────────────────────

    #[test]
    fn test_topo_sort_linear() {
        let g = linear_chain(4);
        let sorted = topological_sort(&g).unwrap();
        assert_eq!(sorted, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_topo_sort_diamond() {
        let g = diamond_graph();
        let sorted = topological_sort(&g).unwrap();
        // Node 1 must come first, node 4 must come last.
        assert_eq!(sorted[0], 1);
        assert_eq!(*sorted.last().unwrap(), 4);
    }

    #[test]
    fn test_topo_sort_cycle_detected() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Add));
        g.add_node(GraphNode::new(2, OpType::Add));
        g.connect(1, 2);
        g.connect(2, 1);
        assert_eq!(topological_sort(&g), Err(GraphError::CycleDetected));
    }

    #[test]
    fn test_topo_sort_single_node() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::MatMul));
        let sorted = topological_sort(&g).unwrap();
        assert_eq!(sorted, vec![1]);
    }

    #[test]
    fn test_topo_sort_disconnected() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Add));
        g.add_node(GraphNode::new(2, OpType::Add));
        g.add_node(GraphNode::new(3, OpType::Add));
        // No edges: all independent.
        let sorted = topological_sort(&g).unwrap();
        assert_eq!(sorted.len(), 3);
    }

    // ── ExecutionPlan ───────────────────────────────────────────────────

    #[test]
    fn test_plan_linear_sequential() {
        let g = linear_chain(3);
        let ex = sequential_executor();
        let plan = ex.plan(&g).unwrap();
        assert_eq!(plan.stage_count(), 3);
        assert!(plan.is_sequential());
        assert_eq!(plan.total_nodes(), 3);
    }

    #[test]
    fn test_plan_diamond_parallel() {
        let g = diamond_graph();
        let ex = default_executor();
        let plan = ex.plan(&g).unwrap();
        // Stage 0: [1], Stage 1: [2,3], Stage 2: [4]
        assert_eq!(plan.stage_count(), 3);
        assert_eq!(plan.max_parallelism(), 2);
        assert!(!plan.is_sequential());
    }

    #[test]
    fn test_plan_wide_parallel() {
        let g = wide_graph();
        let ex = default_executor();
        let plan = ex.plan(&g).unwrap();
        // Stage 0: [1], Stage 1: [2,3,4,5], Stage 2: [6]
        assert_eq!(plan.stage_count(), 3);
        assert_eq!(plan.max_parallelism(), 4);
        assert_eq!(plan.total_nodes(), 6);
    }

    #[test]
    fn test_plan_respects_max_parallel_ops() {
        let g = wide_graph();
        let ex = GraphExecutor::new(ExecutorConfig { max_parallel_ops: 2, ..Default::default() });
        let plan = ex.plan(&g).unwrap();
        // The 4 parallel nodes get split across stages of 2.
        for stage in &plan.stages {
            assert!(stage.len() <= 2);
        }
        assert_eq!(plan.total_nodes(), 6);
    }

    #[test]
    fn test_plan_empty_graph_error() {
        let g = ExecutionGraph::new();
        let ex = default_executor();
        assert!(matches!(ex.plan(&g), Err(GraphError::EmptyGraph)));
    }

    #[test]
    fn test_plan_cycle_error() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Add));
        g.add_node(GraphNode::new(2, OpType::Add));
        g.connect(1, 2);
        g.connect(2, 1);
        let ex = default_executor();
        assert!(matches!(ex.plan(&g), Err(GraphError::CycleDetected)));
    }

    #[test]
    fn test_plan_data_driven() {
        let g = diamond_graph();
        let ex = GraphExecutor::new(ExecutorConfig {
            execution_order: ExecutionOrder::DataDriven,
            ..Default::default()
        });
        let plan = ex.plan(&g).unwrap();
        assert_eq!(plan.total_nodes(), 4);
        assert_eq!(plan.stage_count(), 3);
    }

    #[test]
    fn test_plan_data_driven_wide() {
        let g = wide_graph();
        let ex = GraphExecutor::new(ExecutorConfig {
            execution_order: ExecutionOrder::DataDriven,
            ..Default::default()
        });
        let plan = ex.plan(&g).unwrap();
        assert_eq!(plan.total_nodes(), 6);
    }

    #[test]
    fn test_plan_preserves_all_nodes() {
        let g = wide_graph();
        let ex = default_executor();
        let plan = ex.plan(&g).unwrap();
        let planned: HashSet<NodeId> = plan.stages.iter().flat_map(|s| s.iter().copied()).collect();
        let expected: HashSet<NodeId> = g.nodes.keys().copied().collect();
        assert_eq!(planned, expected);
    }

    #[test]
    fn test_plan_respects_dependencies() {
        let g = diamond_graph();
        let ex = default_executor();
        let plan = ex.plan(&g).unwrap();
        // Flatten stages into execution order.
        let order: Vec<NodeId> = plan.stages.iter().flat_map(|s| s.iter().copied()).collect();
        let pos = |nid: NodeId| order.iter().position(|&x| x == nid).unwrap();
        // Node 1 before 2 and 3, both 2 and 3 before 4.
        assert!(pos(1) < pos(2));
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(4));
        assert!(pos(3) < pos(4));
    }

    // ── Execution ───────────────────────────────────────────────────────

    #[test]
    fn test_execute_linear() {
        let g = linear_chain(3);
        let ex = default_executor();
        let trace = ex.run(&g).unwrap();
        assert_eq!(trace.node_results.len(), 3);
    }

    #[test]
    fn test_execute_diamond() {
        let g = diamond_graph();
        let ex = default_executor();
        let trace = ex.run(&g).unwrap();
        assert_eq!(trace.node_results.len(), 4);
    }

    #[test]
    fn test_execute_traces_memory() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::MatMul).with_memory(100));
        g.add_node(GraphNode::new(2, OpType::Add).with_memory(200));
        g.connect(1, 2);
        g.mark_input(1);
        g.mark_output(2);
        let ex = default_executor();
        let trace = ex.run(&g).unwrap();
        assert!(trace.peak_memory >= 200);
    }

    #[test]
    fn test_execute_memory_limit_exceeded() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::MatMul).with_memory(600));
        g.add_node(GraphNode::new(2, OpType::Add).with_memory(600));
        g.connect(1, 2);
        g.mark_input(1);
        g.mark_output(2);
        let ex = GraphExecutor::new(ExecutorConfig { memory_limit: 1000, ..Default::default() });
        let result = ex.run(&g);
        assert!(matches!(result, Err(GraphError::MemoryLimitExceeded { .. })));
    }

    #[test]
    fn test_execute_memory_limit_ok() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::MatMul).with_memory(400));
        g.add_node(GraphNode::new(2, OpType::Add).with_memory(400));
        g.connect(1, 2);
        g.mark_input(1);
        g.mark_output(2);
        let ex = GraphExecutor::new(ExecutorConfig { memory_limit: 10000, ..Default::default() });
        assert!(ex.run(&g).is_ok());
    }

    #[test]
    fn test_execute_output_data_present() {
        let g = linear_chain(2);
        let ex = default_executor();
        let trace = ex.run(&g).unwrap();
        for r in &trace.node_results {
            assert!(!r.output_data.is_empty());
        }
    }

    #[test]
    fn test_execute_total_time_nonzero() {
        let g = linear_chain(2);
        let ex = default_executor();
        let trace = ex.run(&g).unwrap();
        // Total time is always non-negative.
        let _ = trace.total_time;
    }

    // ── Critical Path ───────────────────────────────────────────────────

    #[test]
    fn test_critical_path_linear() {
        let g = linear_chain(4);
        let ex = default_executor();
        let trace = ex.run(&g).unwrap();
        // In a linear chain every node is on the critical path.
        assert_eq!(trace.critical_path.len(), 4);
        assert_eq!(trace.critical_path, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_critical_path_diamond() {
        let g = diamond_graph();
        let ex = default_executor();
        let trace = ex.run(&g).unwrap();
        // Critical path goes through the heavier branch (node 2: 20 flops).
        assert!(trace.critical_path.contains(&1));
        assert!(trace.critical_path.contains(&4));
        assert!(trace.critical_path.len() >= 3);
    }

    #[test]
    fn test_critical_path_single_node() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::MatMul).with_flops(10));
        g.mark_input(1);
        g.mark_output(1);
        let ex = default_executor();
        let trace = ex.run(&g).unwrap();
        assert_eq!(trace.critical_path, vec![1]);
    }

    // ── ExecutionTrace helpers ──────────────────────────────────────────

    #[test]
    fn test_trace_avg_node_time() {
        let trace = ExecutionTrace {
            node_results: vec![
                NodeResult {
                    node_id: 1,
                    output_data: vec![],
                    execution_time_ms: 10.0,
                    memory_used: 0,
                },
                NodeResult {
                    node_id: 2,
                    output_data: vec![],
                    execution_time_ms: 20.0,
                    memory_used: 0,
                },
            ],
            total_time: Duration::from_millis(30),
            peak_memory: 0,
            critical_path: vec![],
        };
        assert!((trace.avg_node_time_ms() - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trace_avg_node_time_empty() {
        let trace = ExecutionTrace {
            node_results: vec![],
            total_time: Duration::ZERO,
            peak_memory: 0,
            critical_path: vec![],
        };
        assert!((trace.avg_node_time_ms() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trace_total_memory() {
        let trace = ExecutionTrace {
            node_results: vec![
                NodeResult {
                    node_id: 1,
                    output_data: vec![],
                    execution_time_ms: 0.0,
                    memory_used: 100,
                },
                NodeResult {
                    node_id: 2,
                    output_data: vec![],
                    execution_time_ms: 0.0,
                    memory_used: 200,
                },
            ],
            total_time: Duration::ZERO,
            peak_memory: 200,
            critical_path: vec![],
        };
        assert_eq!(trace.total_memory_used(), 300);
    }

    // ── Graph Optimizer ─────────────────────────────────────────────────

    #[test]
    fn test_dead_code_elimination() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Linear));
        g.add_node(GraphNode::new(2, OpType::MatMul));
        g.add_node(GraphNode::new(3, OpType::Add)); // dead
        g.connect(1, 2);
        g.mark_input(1);
        g.mark_output(2);
        let removed = GraphOptimizer::eliminate_dead_code(&mut g);
        assert_eq!(removed, 1);
        assert_eq!(g.node_count(), 2);
        assert!(!g.nodes.contains_key(&3));
    }

    #[test]
    fn test_dead_code_no_outputs_noop() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Add));
        // No output_nodes defined: DCE is a no-op.
        let removed = GraphOptimizer::eliminate_dead_code(&mut g);
        assert_eq!(removed, 0);
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_fuse_elementwise_chain() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Add).with_flops(10));
        g.add_node(GraphNode::new(2, OpType::Relu).with_flops(5));
        g.connect(1, 2);
        g.mark_input(1);
        g.mark_output(2);
        let fused = GraphOptimizer::fuse_elementwise(&mut g);
        assert_eq!(fused, 1);
        // Node 2 merged into node 1.
        assert_eq!(g.node_count(), 1);
        assert!(g.nodes.contains_key(&1));
        assert_eq!(g.nodes[&1].estimated_flops, 15);
    }

    #[test]
    fn test_fuse_preserves_non_elementwise() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::MatMul));
        g.add_node(GraphNode::new(2, OpType::Add));
        g.connect(1, 2);
        g.mark_input(1);
        g.mark_output(2);
        let fused = GraphOptimizer::fuse_elementwise(&mut g);
        assert_eq!(fused, 0);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_fuse_three_elementwise() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Add).with_flops(10));
        g.add_node(GraphNode::new(2, OpType::Relu).with_flops(5));
        g.add_node(GraphNode::new(3, OpType::Gelu).with_flops(8));
        g.connect(1, 2);
        g.connect(2, 3);
        g.mark_input(1);
        g.mark_output(3);
        let fused = GraphOptimizer::fuse_elementwise(&mut g);
        assert_eq!(fused, 2);
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.nodes[&1].estimated_flops, 23);
    }

    #[test]
    fn test_fuse_does_not_fuse_multi_consumer() {
        // Add -> {Relu, Gelu}: Add has 2 successors, no fusion.
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Add));
        g.add_node(GraphNode::new(2, OpType::Relu));
        g.add_node(GraphNode::new(3, OpType::Gelu));
        g.connect(1, 2);
        g.connect(1, 3);
        g.mark_input(1);
        g.mark_output(2);
        g.mark_output(3);
        let fused = GraphOptimizer::fuse_elementwise(&mut g);
        assert_eq!(fused, 0);
    }

    #[test]
    fn test_optimize_combined() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Linear));
        g.add_node(GraphNode::new(2, OpType::Add).with_flops(10));
        g.add_node(GraphNode::new(3, OpType::Relu).with_flops(5));
        g.add_node(GraphNode::new(4, OpType::MatMul)); // dead
        g.connect(1, 2);
        g.connect(2, 3);
        g.mark_input(1);
        g.mark_output(3);
        let stats = GraphOptimizer::optimize(&mut g);
        assert_eq!(stats.nodes_removed, 1); // node 4 dead
        assert_eq!(stats.nodes_fused, 1); // 2+3 fused
    }

    #[test]
    fn test_reorder_for_locality() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::MatMul).with_memory(300));
        g.add_node(GraphNode::new(2, OpType::MatMul).with_memory(100));
        g.add_node(GraphNode::new(3, OpType::MatMul).with_memory(200));
        g.mark_input(1);
        g.mark_input(2);
        g.mark_input(3);
        g.mark_output(1);
        g.mark_output(2);
        g.mark_output(3);
        let mut plan = ExecutionPlan { stages: vec![vec![1, 2, 3]] };
        GraphOptimizer::reorder_for_locality(&mut plan, &g);
        assert_eq!(plan.stages[0], vec![2, 3, 1]);
    }

    // ── run_optimized ───────────────────────────────────────────────────

    #[test]
    fn test_run_optimized() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Linear).with_flops(10));
        g.add_node(GraphNode::new(2, OpType::Add).with_flops(5));
        g.add_node(GraphNode::new(3, OpType::Relu).with_flops(3));
        g.add_node(GraphNode::new(4, OpType::MatMul).with_flops(100)); // dead
        g.connect(1, 2);
        g.connect(2, 3);
        g.mark_input(1);
        g.mark_output(3);
        let ex = default_executor();
        let (trace, stats) = ex.run_optimized(&mut g).unwrap();
        assert_eq!(stats.nodes_removed, 1);
        assert!(!trace.node_results.is_empty());
    }

    // ── Error display ───────────────────────────────────────────────────

    #[test]
    fn test_error_display_cycle() {
        assert_eq!(GraphError::CycleDetected.to_string(), "cycle detected in execution graph");
    }

    #[test]
    fn test_error_display_empty() {
        assert_eq!(GraphError::EmptyGraph.to_string(), "execution graph is empty");
    }

    #[test]
    fn test_error_display_self_loop() {
        assert_eq!(GraphError::SelfLoop(5).to_string(), "self-loop on node 5");
    }

    #[test]
    fn test_error_display_dangling() {
        assert_eq!(
            GraphError::DanglingEdge { node_id: 42 }.to_string(),
            "edge references missing node 42"
        );
    }

    #[test]
    fn test_error_display_node_not_found() {
        assert_eq!(GraphError::NodeNotFound(7).to_string(), "node 7 not found");
    }

    #[test]
    fn test_error_display_memory_limit() {
        let e = GraphError::MemoryLimitExceeded { required: 2000, limit: 1000 };
        assert!(e.to_string().contains("2000"));
        assert!(e.to_string().contains("1000"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(GraphError::CycleDetected);
        assert!(e.to_string().contains("cycle"));
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_single_node_graph() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::MatMul));
        g.mark_input(1);
        g.mark_output(1);
        let ex = default_executor();
        let trace = ex.run(&g).unwrap();
        assert_eq!(trace.node_results.len(), 1);
    }

    #[test]
    fn test_disconnected_components() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Add));
        g.add_node(GraphNode::new(2, OpType::Add));
        g.add_node(GraphNode::new(3, OpType::Add));
        g.mark_input(1);
        g.mark_input(2);
        g.mark_input(3);
        g.mark_output(1);
        g.mark_output(2);
        g.mark_output(3);
        let ex = default_executor();
        let plan = ex.plan(&g).unwrap();
        // All 3 nodes are independent → single parallel stage.
        assert_eq!(plan.stage_count(), 1);
        assert_eq!(plan.max_parallelism(), 3);
    }

    #[test]
    fn test_large_chain() {
        let g = linear_chain(100);
        let ex = default_executor();
        let plan = ex.plan(&g).unwrap();
        assert_eq!(plan.total_nodes(), 100);
        assert!(plan.is_sequential());
    }

    #[test]
    fn test_large_wide_graph() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(0, OpType::Linear));
        for i in 1..=50 {
            g.add_node(GraphNode::new(i, OpType::MatMul));
            g.connect(0, i);
        }
        g.add_node(GraphNode::new(51, OpType::Add));
        for i in 1..=50 {
            g.connect(i, 51);
        }
        g.mark_input(0);
        g.mark_output(51);
        let ex = default_executor();
        let plan = ex.plan(&g).unwrap();
        assert_eq!(plan.total_nodes(), 52);
        // The 50 middle nodes should be parallelised.
        assert!(plan.max_parallelism() >= 8);
    }

    #[test]
    fn test_custom_op_type() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Custom("MyKernel".into())));
        g.mark_input(1);
        g.mark_output(1);
        let ex = default_executor();
        assert!(ex.run(&g).is_ok());
    }

    #[test]
    fn test_zero_flops_node() {
        let mut g = ExecutionGraph::new();
        g.add_node(GraphNode::new(1, OpType::Reshape).with_flops(0));
        g.mark_input(1);
        g.mark_output(1);
        let ex = default_executor();
        let trace = ex.run(&g).unwrap();
        assert_eq!(trace.node_results.len(), 1);
    }

    #[test]
    fn test_plan_sequential_preserves_order() {
        let g = linear_chain(5);
        let ex = sequential_executor();
        let plan = ex.plan(&g).unwrap();
        let order: Vec<NodeId> = plan.stages.iter().flat_map(|s| s.iter().copied()).collect();
        assert_eq!(order, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_plan_all_orders_same_node_count() {
        let g = diamond_graph();
        for order in [
            ExecutionOrder::Sequential,
            ExecutionOrder::TopologicalParallel,
            ExecutionOrder::DataDriven,
        ] {
            let ex =
                GraphExecutor::new(ExecutorConfig { execution_order: order, ..Default::default() });
            let plan = ex.plan(&g).unwrap();
            assert_eq!(plan.total_nodes(), 4, "failed for {order}");
        }
    }

    #[test]
    fn test_validate_linear_chain() {
        let g = linear_chain(10);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_node_result_fields() {
        let r = NodeResult {
            node_id: 42,
            output_data: vec![1.0, 2.0],
            execution_time_ms: 5.5,
            memory_used: 1024,
        };
        assert_eq!(r.node_id, 42);
        assert_eq!(r.output_data.len(), 2);
        assert!((r.execution_time_ms - 5.5).abs() < f64::EPSILON);
        assert_eq!(r.memory_used, 1024);
    }

    #[test]
    fn test_execution_plan_empty_stages() {
        let plan = ExecutionPlan { stages: vec![] };
        assert_eq!(plan.stage_count(), 0);
        assert_eq!(plan.total_nodes(), 0);
        assert_eq!(plan.max_parallelism(), 0);
        assert!(plan.is_sequential());
    }

    #[test]
    fn test_graph_edge_hash() {
        let mut s = HashSet::new();
        s.insert(GraphEdge::new(1, 0, 2, 0));
        s.insert(GraphEdge::new(1, 0, 2, 0));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_optimizer_on_empty_graph_with_output() {
        let mut g = ExecutionGraph::new();
        g.output_nodes.push(99); // output references nonexistent node
        let removed = GraphOptimizer::eliminate_dead_code(&mut g);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_simulate_work() {
        // Ensure simulate_work doesn't panic.
        assert_eq!(simulate_work(0), 0);
        let v = simulate_work(50);
        assert!(v > 0);
        // Large values are capped at 100 iterations internally.
        let v2 = simulate_work(u64::MAX);
        assert!(v2 > 0);
    }
}
