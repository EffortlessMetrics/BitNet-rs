//! Compute graph builder.
//!
//! DAG-based execution planning for inference operations.

use std::collections::HashMap;

/// Node identifier.
pub type NodeId = usize;

/// Operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    MatMul,
    Add,
    Norm,
    Activation,
    Attention,
    Embedding,
    Projection,
    Concat,
    Reshape,
    Softmax,
}

impl OpType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MatMul => "matmul",
            Self::Add => "add",
            Self::Norm => "norm",
            Self::Activation => "activation",
            Self::Attention => "attention",
            Self::Embedding => "embedding",
            Self::Projection => "projection",
            Self::Concat => "concat",
            Self::Reshape => "reshape",
            Self::Softmax => "softmax",
        }
    }
}

/// A node in the compute graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: OpType,
    pub name: String,
    pub inputs: Vec<NodeId>,
    pub output_shape: Vec<usize>,
}

/// Compute graph.
#[derive(Debug, Clone)]
pub struct ComputeGraph {
    nodes: Vec<Node>,
    outputs: Vec<NodeId>,
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), outputs: Vec::new() }
    }

    pub fn add_node(
        &mut self,
        op: OpType,
        name: &str,
        inputs: Vec<NodeId>,
        output_shape: Vec<usize>,
    ) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node { id, op, name: name.to_string(), inputs, output_shape });
        id
    }

    pub fn mark_output(&mut self, id: NodeId) {
        if !self.outputs.contains(&id) {
            self.outputs.push(id);
        }
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn output_nodes(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Get nodes that depend on a given node.
    pub fn consumers(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes.iter().filter(|n| n.inputs.contains(&id)).map(|n| n.id).collect()
    }

    /// Topological sort (for execution order).
    pub fn topo_sort(&self) -> Option<Vec<NodeId>> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        for node in &self.nodes {
            for &input in &node.inputs {
                if input < n {
                    in_degree[node.id] += 1;
                }
            }
        }

        let mut queue: Vec<NodeId> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut result = Vec::new();

        while let Some(id) = queue.pop() {
            result.push(id);
            for consumer in self.consumers(id) {
                in_degree[consumer] -= 1;
                if in_degree[consumer] == 0 {
                    queue.push(consumer);
                }
            }
        }

        if result.len() == n { Some(result) } else { None }
    }

    /// Count operations by type.
    pub fn op_counts(&self) -> HashMap<OpType, usize> {
        let mut map = HashMap::new();
        for node in &self.nodes {
            *map.entry(node.op).or_insert(0) += 1;
        }
        map
    }

    /// Estimated FLOPS for the graph.
    pub fn estimated_flops(&self) -> u64 {
        let mut total = 0u64;
        for node in &self.nodes {
            let numel: u64 = node.output_shape.iter().map(|&d| d as u64).product();
            total += match node.op {
                OpType::MatMul => numel * 2,
                OpType::Attention => numel * 4,
                OpType::Norm | OpType::Softmax => numel * 3,
                OpType::Activation => numel,
                _ => numel,
            };
        }
        total
    }
}

/// Builder for creating standard transformer graphs.
pub struct GraphBuilder {
    graph: ComputeGraph,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self { graph: ComputeGraph::new() }
    }

    pub fn embedding(&mut self, name: &str, shape: Vec<usize>) -> NodeId {
        self.graph.add_node(OpType::Embedding, name, vec![], shape)
    }

    pub fn matmul(&mut self, name: &str, inputs: Vec<NodeId>, shape: Vec<usize>) -> NodeId {
        self.graph.add_node(OpType::MatMul, name, inputs, shape)
    }

    pub fn norm(&mut self, name: &str, input: NodeId, shape: Vec<usize>) -> NodeId {
        self.graph.add_node(OpType::Norm, name, vec![input], shape)
    }

    pub fn activation(&mut self, name: &str, input: NodeId, shape: Vec<usize>) -> NodeId {
        self.graph.add_node(OpType::Activation, name, vec![input], shape)
    }

    pub fn output(&mut self, id: NodeId) {
        self.graph.mark_output(id);
    }

    pub fn build(self) -> ComputeGraph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_graph() {
        let g = ComputeGraph::new();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut g = ComputeGraph::new();
        let id = g.add_node(OpType::Embedding, "embed", vec![], vec![1, 512, 768]);
        assert_eq!(id, 0);
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_connections() {
        let mut g = ComputeGraph::new();
        let e = g.add_node(OpType::Embedding, "embed", vec![], vec![1, 512, 768]);
        let n = g.add_node(OpType::Norm, "norm", vec![e], vec![1, 512, 768]);
        let consumers = g.consumers(e);
        assert_eq!(consumers, vec![n]);
    }

    #[test]
    fn test_topo_sort() {
        let mut g = ComputeGraph::new();
        let a = g.add_node(OpType::Embedding, "a", vec![], vec![1]);
        let b = g.add_node(OpType::Norm, "b", vec![a], vec![1]);
        let c = g.add_node(OpType::Activation, "c", vec![b], vec![1]);
        let order = g.topo_sort().unwrap();
        assert_eq!(order.len(), 3);
        // a must come before b, b before c
        let pos_a = order.iter().position(|&x| x == a).unwrap();
        let pos_b = order.iter().position(|&x| x == b).unwrap();
        let pos_c = order.iter().position(|&x| x == c).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_op_counts() {
        let mut g = ComputeGraph::new();
        g.add_node(OpType::MatMul, "m1", vec![], vec![1]);
        g.add_node(OpType::MatMul, "m2", vec![], vec![1]);
        g.add_node(OpType::Norm, "n1", vec![], vec![1]);
        let counts = g.op_counts();
        assert_eq!(counts[&OpType::MatMul], 2);
        assert_eq!(counts[&OpType::Norm], 1);
    }

    #[test]
    fn test_estimated_flops() {
        let mut g = ComputeGraph::new();
        g.add_node(OpType::MatMul, "m", vec![], vec![10, 10]);
        let flops = g.estimated_flops();
        assert_eq!(flops, 200); // 100 * 2
    }

    #[test]
    fn test_mark_output() {
        let mut g = ComputeGraph::new();
        let id = g.add_node(OpType::Projection, "proj", vec![], vec![1]);
        g.mark_output(id);
        assert_eq!(g.output_nodes(), &[id]);
    }

    #[test]
    fn test_builder() {
        let mut b = GraphBuilder::new();
        let e = b.embedding("embed", vec![1, 512, 768]);
        let n = b.norm("norm", e, vec![1, 512, 768]);
        let a = b.activation("silu", n, vec![1, 512, 768]);
        b.output(a);
        let g = b.build();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.output_nodes().len(), 1);
    }

    #[test]
    fn test_op_type_str() {
        assert_eq!(OpType::MatMul.as_str(), "matmul");
        assert_eq!(OpType::Softmax.as_str(), "softmax");
    }

    #[test]
    fn test_default_graph() {
        let g = ComputeGraph::default();
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_node_lookup() {
        let mut g = ComputeGraph::new();
        let id = g.add_node(OpType::Add, "add", vec![], vec![1, 2]);
        let node = g.node(id).unwrap();
        assert_eq!(node.op, OpType::Add);
        assert!(g.node(999).is_none());
    }

    #[test]
    fn test_builder_default() {
        let b = GraphBuilder::default();
        let g = b.build();
        assert_eq!(g.node_count(), 0);
    }
}
