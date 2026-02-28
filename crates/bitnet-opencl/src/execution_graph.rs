//! DAG-based execution scheduler for GPU kernel launches.
//!
//! Builds a directed acyclic graph from the pipeline's stages, computes
//! topological order, identifies the critical path, and groups
//! independent nodes into parallelisable batches.

use crate::pipeline::{GpuInferencePipeline, PipelineStage};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single node in the execution graph.
#[derive(Debug, Clone)]
pub struct ExecutionNode {
    /// Unique node identifier (equal to its index in the graph).
    pub id: usize,
    /// The pipeline stage this node represents.
    pub stage: PipelineStage,
    /// IDs of nodes that must complete before this one starts.
    pub dependencies: Vec<usize>,
    /// Estimated execution time in microseconds (heuristic).
    pub estimated_time_us: u64,
}

/// Directed acyclic graph of GPU kernel executions.
#[derive(Debug)]
pub struct ExecutionGraph {
    nodes: Vec<ExecutionNode>,
    /// Explicit dependency edges `(from, to)`.
    edges: Vec<(usize, usize)>,
}

impl ExecutionGraph {
    /// Build an execution graph from a pipeline.
    ///
    /// Every stage depends on its predecessor, forming a linear chain.
    /// Future work may detect independent sub-graphs (e.g. parallel
    /// attention heads).
    #[must_use]
    pub fn from_pipeline(pipeline: &GpuInferencePipeline) -> Self {
        let stages = pipeline.stages();
        let mut nodes = Vec::with_capacity(stages.len());
        let mut edges = Vec::new();

        for (i, stage) in stages.iter().enumerate() {
            let deps = if i == 0 { vec![] } else { vec![i - 1] };
            if i > 0 {
                edges.push((i - 1, i));
            }
            nodes.push(ExecutionNode {
                id: i,
                stage: stage.clone(),
                dependencies: deps,
                estimated_time_us: Self::estimate_time(stage),
            });
        }

        Self { nodes, edges }
    }

    /// Build from explicit nodes and edges (for testing).
    #[must_use]
    pub const fn from_parts(nodes: Vec<ExecutionNode>, edges: Vec<(usize, usize)>) -> Self {
        Self { nodes, edges }
    }

    /// All nodes in the graph.
    #[must_use]
    pub fn nodes(&self) -> &[ExecutionNode] {
        &self.nodes
    }

    /// All dependency edges.
    #[must_use]
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Topological ordering of node IDs (Kahn's algorithm).
    #[must_use]
    pub fn topological_order(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut in_degree = vec![0u32; n];
        for &(_, to) in &self.edges {
            if to < n {
                in_degree[to] += 1;
            }
        }

        let mut queue: Vec<usize> =
            in_degree.iter().enumerate().filter(|(_, d)| **d == 0).map(|(i, _)| i).collect();

        let mut order = Vec::with_capacity(n);
        while let Some(node) = queue.pop() {
            order.push(node);
            for &(from, to) in &self.edges {
                if from == node && to < n {
                    in_degree[to] -= 1;
                    if in_degree[to] == 0 {
                        queue.push(to);
                    }
                }
            }
        }

        order
    }

    /// IDs of nodes on the critical (longest) path through the DAG.
    ///
    /// Uses dynamic programming on the topological order.
    #[must_use]
    pub fn critical_path(&self) -> Vec<usize> {
        let n = self.nodes.len();
        if n == 0 {
            return vec![];
        }

        let order = self.topological_order();
        let mut dist = vec![0u64; n];
        let mut pred: Vec<Option<usize>> = vec![None; n];

        for &node_id in &order {
            let node_time = self.nodes[node_id].estimated_time_us;
            for &dep in &self.nodes[node_id].dependencies {
                let candidate = dist[dep] + node_time;
                if candidate > dist[node_id] {
                    dist[node_id] = candidate;
                    pred[node_id] = Some(dep);
                }
            }
            if self.nodes[node_id].dependencies.is_empty() {
                dist[node_id] = node_time;
            }
        }

        let end = dist.iter().enumerate().max_by_key(|(_, d)| *d).map_or(0, |(i, _)| i);

        let mut path = vec![end];
        let mut cur = end;
        while let Some(p) = pred[cur] {
            path.push(p);
            cur = p;
        }
        path.reverse();
        path
    }

    /// Group nodes into batches that can execute in parallel.
    ///
    /// Each batch contains nodes whose dependencies are all satisfied
    /// by nodes in earlier batches.
    #[must_use]
    pub fn parallelizable_groups(&self) -> Vec<Vec<usize>> {
        let n = self.nodes.len();
        if n == 0 {
            return vec![];
        }

        let order = self.topological_order();
        let mut depth = vec![0usize; n];
        for &node_id in &order {
            for &dep in &self.nodes[node_id].dependencies {
                let candidate = depth[dep] + 1;
                if candidate > depth[node_id] {
                    depth[node_id] = candidate;
                }
            }
        }

        let max_depth = depth.iter().copied().max().unwrap_or(0);
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_depth + 1];
        for (id, &d) in depth.iter().enumerate() {
            groups[d].push(id);
        }
        groups
    }

    /// Heuristic kernel-time estimate based on stage type.
    const fn estimate_time(stage: &PipelineStage) -> u64 {
        match *stage {
            PipelineStage::Embedding { dim, .. } => dim as u64 / 10,
            PipelineStage::RmsNorm { dim, .. } | PipelineStage::Softmax { dim } => dim as u64 / 100,
            PipelineStage::Attention { num_heads, head_dim } => {
                (num_heads as u64 * head_dim as u64) / 5
            }
            PipelineStage::FeedForward { dim, hidden_dim } => {
                (dim as u64 * hidden_dim as u64) / 1000
            }
            PipelineStage::Linear { in_features, out_features } => {
                (in_features as u64 * out_features as u64) / 1000
            }
            PipelineStage::Dequantize { block_size, .. } => block_size as u64 / 50,
        }
    }
}
