//! GPU computation graph for OpenCL execution.
//!
//! Models inference operations as a directed acyclic graph (DAG) that
//! can be compiled into an optimized execution plan. Supports kernel
//! dispatch, host↔device transfers, and barrier nodes with automatic
//! redundant-barrier elimination during compilation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors arising during graph construction or compilation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum GraphError {
    /// Referenced a node id that does not exist in the graph.
    #[error("invalid node id: {0}")]
    InvalidNodeId(usize),

    /// Attempted to add a self-edge (node → itself).
    #[error("self-edge on node {0}")]
    SelfEdge(usize),

    /// The graph contains a cycle and cannot be topologically sorted.
    #[error("cycle detected in graph")]
    CycleDetected,

    /// The graph is empty (has no nodes).
    #[error("graph is empty")]
    EmptyGraph,
}

// ---------------------------------------------------------------------------
// Node types
// ---------------------------------------------------------------------------

/// The kind of work a graph node represents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuNodeKind {
    /// A compute kernel identified by name.
    Kernel { name: String },
    /// A memory transfer between host and device.
    Transfer { direction: TransferDirection },
    /// An execution barrier / synchronisation point.
    Barrier,
}

/// Direction of a host↔device memory transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

impl fmt::Display for TransferDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "H→D"),
            Self::DeviceToHost => write!(f, "D→H"),
            Self::DeviceToDevice => write!(f, "D→D"),
        }
    }
}

/// A single node in the computation graph.
#[derive(Debug, Clone)]
pub struct GpuNode {
    /// Unique identifier within the graph.
    pub id: usize,
    /// What this node does.
    pub kind: GpuNodeKind,
    /// Human-readable label (for debugging / visualisation).
    pub label: String,
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

/// A mutable directed acyclic graph of GPU operations.
#[derive(Debug, Clone)]
pub struct GpuGraph {
    nodes: Vec<GpuNode>,
    /// Adjacency list: edges[src] = set of dst node ids.
    edges: HashMap<usize, HashSet<usize>>,
}

impl GpuGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: HashMap::new(),
        }
    }

    /// Add a node and return its id.
    pub fn add_node(&mut self, kind: GpuNodeKind, label: impl Into<String>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(GpuNode {
            id,
            kind,
            label: label.into(),
        });
        id
    }

    /// Add a directed edge `from → to`.
    pub fn add_edge(&mut self, from: usize, to: usize) -> Result<(), GraphError> {
        if from == to {
            return Err(GraphError::SelfEdge(from));
        }
        let max_id = self.nodes.len();
        if from >= max_id {
            return Err(GraphError::InvalidNodeId(from));
        }
        if to >= max_id {
            return Err(GraphError::InvalidNodeId(to));
        }
        self.edges.entry(from).or_default().insert(to);
        Ok(())
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|s| s.len()).sum()
    }

    /// Return the node for a given id, if it exists.
    pub fn get_node(&self, id: usize) -> Option<&GpuNode> {
        self.nodes.get(id)
    }

    /// Compile the graph into an optimised execution plan.
    ///
    /// Performs topological sort (Kahn's algorithm) and eliminates
    /// redundant barrier nodes that have no downstream dependants
    /// or are immediately adjacent to another barrier.
    pub fn compile(&self) -> Result<CompiledGraph, GraphError> {
        if self.nodes.is_empty() {
            return Err(GraphError::EmptyGraph);
        }

        let order = self.topological_sort()?;
        let optimised = self.eliminate_redundant_barriers(&order);

        Ok(CompiledGraph {
            execution_order: optimised,
            original_node_count: self.nodes.len(),
            original_edge_count: self.edge_count(),
        })
    }

    // -- internal helpers ---------------------------------------------------

    /// Kahn's algorithm for topological sorting.
    fn topological_sort(&self) -> Result<Vec<GpuNode>, GraphError> {
        let n = self.nodes.len();
        let mut in_degree = vec![0u32; n];

        for dsts in self.edges.values() {
            for &d in dsts {
                in_degree[d] += 1;
            }
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut sorted: Vec<GpuNode> = Vec::with_capacity(n);
        while let Some(node_id) = queue.pop_front() {
            sorted.push(self.nodes[node_id].clone());
            if let Some(neighbours) = self.edges.get(&node_id) {
                for &nb in neighbours {
                    in_degree[nb] -= 1;
                    if in_degree[nb] == 0 {
                        queue.push_back(nb);
                    }
                }
            }
        }

        if sorted.len() != n {
            return Err(GraphError::CycleDetected);
        }
        Ok(sorted)
    }

    /// Remove barriers that are immediately followed by another barrier
    /// or that sit at the very end of the execution order with no
    /// successors.
    fn eliminate_redundant_barriers(&self, order: &[GpuNode]) -> Vec<GpuNode> {
        let mut result: Vec<GpuNode> = Vec::with_capacity(order.len());
        for (i, node) in order.iter().enumerate() {
            if node.kind == GpuNodeKind::Barrier {
                // Skip trailing barriers.
                let is_last = i + 1 == order.len();
                // Skip consecutive barriers.
                let next_is_barrier = order
                    .get(i + 1)
                    .map_or(false, |n| n.kind == GpuNodeKind::Barrier);
                if is_last || next_is_barrier {
                    continue;
                }
            }
            result.push(node.clone());
        }
        result
    }
}

impl Default for GpuGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Compiled graph
// ---------------------------------------------------------------------------

/// An optimised, immutable execution plan produced by [`GpuGraph::compile`].
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    /// Topologically sorted nodes with redundant barriers removed.
    pub execution_order: Vec<GpuNode>,
    /// Number of nodes in the original graph (before optimisation).
    pub original_node_count: usize,
    /// Number of edges in the original graph.
    pub original_edge_count: usize,
}

impl CompiledGraph {
    /// Number of steps in the optimised plan.
    pub fn step_count(&self) -> usize {
        self.execution_order.len()
    }

    /// Returns how many nodes were eliminated during optimisation.
    pub fn eliminated_count(&self) -> usize {
        self.original_node_count - self.execution_order.len()
    }

    /// Iterate over the execution steps.
    pub fn steps(&self) -> impl Iterator<Item = &GpuNode> {
        self.execution_order.iter()
    }

    /// Replay the compiled graph, calling `f` for each step in order.
    pub fn replay<F>(&self, mut f: F)
    where
        F: FnMut(usize, &GpuNode),
    {
        for (i, node) in self.execution_order.iter().enumerate() {
            f(i, node);
        }
    }
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Convenience builder for constructing common graph patterns.
pub struct GraphBuilder {
    graph: GpuGraph,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: GpuGraph::new(),
        }
    }

    /// Add a kernel node and return its id.
    pub fn kernel(&mut self, name: impl Into<String>) -> usize {
        let name_str: String = name.into();
        let label = format!("kernel:{name_str}");
        self.graph.add_node(GpuNodeKind::Kernel { name: name_str }, label)
    }

    /// Add a transfer node and return its id.
    pub fn transfer(&mut self, direction: TransferDirection) -> usize {
        let label = format!("transfer:{direction}");
        self.graph
            .add_node(GpuNodeKind::Transfer { direction }, label)
    }

    /// Add a barrier node and return its id.
    pub fn barrier(&mut self) -> usize {
        self.graph
            .add_node(GpuNodeKind::Barrier, "barrier".to_string())
    }

    /// Add an edge.
    pub fn edge(&mut self, from: usize, to: usize) -> Result<&mut Self, GraphError> {
        self.graph.add_edge(from, to)?;
        Ok(self)
    }

    /// Add a linear chain of edges: a → b → c → ...
    pub fn chain(&mut self, ids: &[usize]) -> Result<&mut Self, GraphError> {
        for pair in ids.windows(2) {
            self.graph.add_edge(pair[0], pair[1])?;
        }
        Ok(self)
    }

    /// Consume the builder and return the graph.
    pub fn build(self) -> GpuGraph {
        self.graph
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- basic graph construction ------------------------------------------

    #[test]
    fn empty_graph_compile_returns_error() {
        let g = GpuGraph::new();
        assert_eq!(g.compile().unwrap_err(), GraphError::EmptyGraph);
    }

    #[test]
    fn single_kernel_node_compiles() {
        let mut g = GpuGraph::new();
        g.add_node(GpuNodeKind::Kernel { name: "matmul".into() }, "m");
        let compiled = g.compile().unwrap();
        assert_eq!(compiled.step_count(), 1);
        assert_eq!(compiled.eliminated_count(), 0);
    }

    #[test]
    fn self_edge_rejected() {
        let mut g = GpuGraph::new();
        let a = g.add_node(GpuNodeKind::Barrier, "b");
        assert_eq!(g.add_edge(a, a).unwrap_err(), GraphError::SelfEdge(a));
    }

    #[test]
    fn invalid_node_id_rejected() {
        let mut g = GpuGraph::new();
        g.add_node(GpuNodeKind::Barrier, "b");
        assert_eq!(
            g.add_edge(0, 99).unwrap_err(),
            GraphError::InvalidNodeId(99)
        );
    }

    // -- topological ordering ----------------------------------------------

    #[test]
    fn linear_chain_preserves_order() {
        let mut g = GpuGraph::new();
        let a = g.add_node(GpuNodeKind::Kernel { name: "a".into() }, "a");
        let b = g.add_node(GpuNodeKind::Kernel { name: "b".into() }, "b");
        let c = g.add_node(GpuNodeKind::Kernel { name: "c".into() }, "c");
        g.add_edge(a, b).unwrap();
        g.add_edge(b, c).unwrap();

        let compiled = g.compile().unwrap();
        let names: Vec<&str> = compiled
            .steps()
            .filter_map(|n| match &n.kind {
                GpuNodeKind::Kernel { name } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn diamond_dag_compiles_all_nodes() {
        // A → B, A → C, B → D, C → D
        let mut g = GpuGraph::new();
        let a = g.add_node(GpuNodeKind::Kernel { name: "a".into() }, "a");
        let b = g.add_node(GpuNodeKind::Kernel { name: "b".into() }, "b");
        let c = g.add_node(GpuNodeKind::Kernel { name: "c".into() }, "c");
        let d = g.add_node(GpuNodeKind::Kernel { name: "d".into() }, "d");
        g.add_edge(a, b).unwrap();
        g.add_edge(a, c).unwrap();
        g.add_edge(b, d).unwrap();
        g.add_edge(c, d).unwrap();

        let compiled = g.compile().unwrap();
        assert_eq!(compiled.step_count(), 4);

        // A must come before B and C; D must come last.
        let ids: Vec<usize> = compiled.steps().map(|n| n.id).collect();
        assert_eq!(ids[0], a);
        assert_eq!(*ids.last().unwrap(), d);
    }

    #[test]
    fn cycle_detected() {
        let mut g = GpuGraph::new();
        let a = g.add_node(GpuNodeKind::Barrier, "a");
        let b = g.add_node(GpuNodeKind::Barrier, "b");
        g.add_edge(a, b).unwrap();
        g.add_edge(b, a).unwrap();
        assert_eq!(g.compile().unwrap_err(), GraphError::CycleDetected);
    }

    // -- barrier elimination -----------------------------------------------

    #[test]
    fn trailing_barrier_eliminated() {
        let mut g = GpuGraph::new();
        let k = g.add_node(GpuNodeKind::Kernel { name: "k".into() }, "k");
        let b = g.add_node(GpuNodeKind::Barrier, "barrier");
        g.add_edge(k, b).unwrap();

        let compiled = g.compile().unwrap();
        assert_eq!(compiled.step_count(), 1);
        assert_eq!(compiled.eliminated_count(), 1);
    }

    #[test]
    fn consecutive_barriers_collapsed() {
        let mut g = GpuGraph::new();
        let k1 = g.add_node(GpuNodeKind::Kernel { name: "k1".into() }, "k1");
        let b1 = g.add_node(GpuNodeKind::Barrier, "b1");
        let b2 = g.add_node(GpuNodeKind::Barrier, "b2");
        let k2 = g.add_node(GpuNodeKind::Kernel { name: "k2".into() }, "k2");
        g.add_edge(k1, b1).unwrap();
        g.add_edge(b1, b2).unwrap();
        g.add_edge(b2, k2).unwrap();

        let compiled = g.compile().unwrap();
        // b1 is consecutive with b2 → b1 removed; b2 remains
        // so we get k1, b2, k2 → 3 steps, 1 eliminated
        assert_eq!(compiled.step_count(), 3);
        assert_eq!(compiled.eliminated_count(), 1);
    }

    #[test]
    fn non_redundant_barrier_kept() {
        let mut g = GpuGraph::new();
        let k1 = g.add_node(GpuNodeKind::Kernel { name: "k1".into() }, "k1");
        let b = g.add_node(GpuNodeKind::Barrier, "sync");
        let k2 = g.add_node(GpuNodeKind::Kernel { name: "k2".into() }, "k2");
        g.add_edge(k1, b).unwrap();
        g.add_edge(b, k2).unwrap();

        let compiled = g.compile().unwrap();
        assert_eq!(compiled.step_count(), 3);
        assert_eq!(compiled.eliminated_count(), 0);
    }

    // -- transfer nodes ----------------------------------------------------

    #[test]
    fn transfer_direction_display() {
        assert_eq!(TransferDirection::HostToDevice.to_string(), "H→D");
        assert_eq!(TransferDirection::DeviceToHost.to_string(), "D→H");
        assert_eq!(TransferDirection::DeviceToDevice.to_string(), "D→D");
    }

    #[test]
    fn transfer_nodes_in_graph() {
        let mut g = GpuGraph::new();
        let h2d = g.add_node(
            GpuNodeKind::Transfer {
                direction: TransferDirection::HostToDevice,
            },
            "upload",
        );
        let k = g.add_node(GpuNodeKind::Kernel { name: "gemm".into() }, "gemm");
        let d2h = g.add_node(
            GpuNodeKind::Transfer {
                direction: TransferDirection::DeviceToHost,
            },
            "download",
        );
        g.add_edge(h2d, k).unwrap();
        g.add_edge(k, d2h).unwrap();

        let compiled = g.compile().unwrap();
        assert_eq!(compiled.step_count(), 3);
    }

    // -- builder -----------------------------------------------------------

    #[test]
    fn builder_chain_produces_valid_graph() {
        let mut b = GraphBuilder::new();
        let upload = b.transfer(TransferDirection::HostToDevice);
        let matmul = b.kernel("matmul");
        let sync = b.barrier();
        let download = b.transfer(TransferDirection::DeviceToHost);
        b.chain(&[upload, matmul, sync, download]).unwrap();

        let g = b.build();
        assert_eq!(g.node_count(), 4);
        assert_eq!(g.edge_count(), 3);

        let compiled = g.compile().unwrap();
        // trailing download is not a barrier, sync before download is useful
        assert_eq!(compiled.step_count(), 4);
    }

    // -- replay ------------------------------------------------------------

    #[test]
    fn replay_visits_all_steps_in_order() {
        let mut b = GraphBuilder::new();
        let a = b.kernel("a");
        let bb = b.kernel("b");
        b.edge(a, bb).unwrap();
        let compiled = b.build().compile().unwrap();

        let mut visited = Vec::new();
        compiled.replay(|i, node| visited.push((i, node.label.clone())));
        assert_eq!(visited.len(), 2);
        assert_eq!(visited[0].0, 0);
        assert_eq!(visited[1].0, 1);
    }

    // -- edge / node counts ------------------------------------------------

    #[test]
    fn node_and_edge_counts_accurate() {
        let mut g = GpuGraph::new();
        let a = g.add_node(GpuNodeKind::Barrier, "a");
        let b = g.add_node(GpuNodeKind::Barrier, "b");
        let c = g.add_node(GpuNodeKind::Barrier, "c");
        g.add_edge(a, b).unwrap();
        g.add_edge(a, c).unwrap();
        g.add_edge(b, c).unwrap();

        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);
    }
}
