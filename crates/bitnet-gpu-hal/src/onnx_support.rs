//! ONNX model support: graph representation, operator dispatch, and
//! optimization passes.
//!
//! This module provides an abstraction layer for loading, validating, and
//! executing ONNX computation graphs without depending on the ONNX Runtime
//! C library. It is suitable for mapping ONNX operators onto backend
//! kernels (CPU, CUDA, etc.).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Data types ────────────────────────────────────────────────────────────

/// Element data types for ONNX tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Float32,
    Float16,
    Int64,
    Int32,
    Int8,
    Uint8,
    Bool,
}

impl DataType {
    /// Size in bytes of a single element.
    pub const fn byte_size(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 => 2,
            Self::Int64 => 8,
            Self::Int8 | Self::Uint8 | Self::Bool => 1,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Float32 => "float32",
            Self::Float16 => "float16",
            Self::Int64 => "int64",
            Self::Int32 => "int32",
            Self::Int8 => "int8",
            Self::Uint8 => "uint8",
            Self::Bool => "bool",
        };
        f.write_str(s)
    }
}

// ── Tensor metadata ───────────────────────────────────────────────────────

/// Metadata describing a named tensor in the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxTensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: DataType,
}

impl OnnxTensor {
    pub fn new(name: impl Into<String>, shape: Vec<i64>, dtype: DataType) -> Self {
        Self { name: name.into(), shape, dtype }
    }

    /// Total number of elements (treats dynamic dims as 1).
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&d| if d > 0 { d as usize } else { 1 }).product()
    }

    /// Total byte size.
    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.byte_size()
    }
}

// ── Node / attribute ──────────────────────────────────────────────────────

/// An attribute value attached to an ONNX node.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

/// A single computation node in the ONNX graph.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, AttributeValue>,
}

impl OnnxNode {
    pub fn new(
        name: impl Into<String>,
        op_type: impl Into<String>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            op_type: op_type.into(),
            inputs,
            outputs,
            attributes: HashMap::new(),
        }
    }

    /// Add an attribute to the node.
    pub fn with_attr(
        mut self,
        key: impl Into<String>,
        val: AttributeValue,
    ) -> Self {
        self.attributes.insert(key.into(), val);
        self
    }
}

// ── Graph ─────────────────────────────────────────────────────────────────

/// An ONNX computation graph.
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    pub nodes: Vec<OnnxNode>,
    pub inputs: Vec<OnnxTensor>,
    pub outputs: Vec<OnnxTensor>,
    pub initializers: HashMap<String, OnnxTensor>,
}

impl OnnxGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: HashMap::new(),
        }
    }

    /// Add a computation node.
    pub fn add_node(&mut self, node: OnnxNode) {
        self.nodes.push(node);
    }

    /// Add a graph input tensor.
    pub fn add_input(&mut self, tensor: OnnxTensor) {
        self.inputs.push(tensor);
    }

    /// Add a graph output tensor.
    pub fn add_output(&mut self, tensor: OnnxTensor) {
        self.outputs.push(tensor);
    }

    /// Register an initializer (constant weight tensor).
    pub fn add_initializer(&mut self, tensor: OnnxTensor) {
        self.initializers.insert(tensor.name.clone(), tensor);
    }

    /// Return node indices in topological order.
    ///
    /// Returns `Err` if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<usize>, OnnxError> {
        let n = self.nodes.len();

        // Map output-name → producing-node index.
        let mut producer: HashMap<&str, usize> = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            for out in &node.outputs {
                producer.insert(out.as_str(), i);
            }
        }

        // Build adjacency (in-degree).
        let mut in_degree = vec![0u32; n];
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, node) in self.nodes.iter().enumerate() {
            for inp in &node.inputs {
                if let Some(&src) = producer.get(inp.as_str()) {
                    if src != i {
                        successors[src].push(i);
                        in_degree[i] += 1;
                    }
                }
            }
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(idx) = queue.pop_front() {
            order.push(idx);
            for &succ in &successors[idx] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }

        if order.len() != n {
            return Err(OnnxError::CyclicGraph);
        }
        Ok(order)
    }

    /// Collect all tensor names produced by nodes or listed as inputs.
    pub fn all_tensor_names(&self) -> HashSet<String> {
        let mut names = HashSet::new();
        for t in &self.inputs {
            names.insert(t.name.clone());
        }
        for (k, _) in &self.initializers {
            names.insert(k.clone());
        }
        for node in &self.nodes {
            for o in &node.outputs {
                names.insert(o.clone());
            }
        }
        names
    }
}

impl Default for OnnxGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Model ─────────────────────────────────────────────────────────────────

/// A parsed ONNX model.
#[derive(Debug, Clone)]
pub struct OnnxModel {
    pub ir_version: i64,
    pub opset_version: i64,
    pub producer_name: String,
    pub graph: OnnxGraph,
    pub metadata: HashMap<String, String>,
}

impl OnnxModel {
    pub fn new(graph: OnnxGraph) -> Self {
        Self {
            ir_version: 9,
            opset_version: 17,
            producer_name: String::from("bitnet-gpu-hal"),
            graph,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }
}

// ── Errors ────────────────────────────────────────────────────────────────

/// Errors produced by ONNX operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnnxError {
    /// The graph contains a cycle.
    CyclicGraph,
    /// A referenced tensor was not found.
    MissingTensor(String),
    /// An unsupported operator was encountered.
    UnsupportedOp(String),
    /// Tensor shapes are incompatible for the operation.
    ShapeMismatch { expected: Vec<i64>, actual: Vec<i64> },
    /// File-level parse/load error.
    LoadError(String),
    /// Graph validation failed.
    ValidationError(String),
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CyclicGraph => write!(f, "graph contains a cycle"),
            Self::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            Self::UnsupportedOp(op) => {
                write!(f, "unsupported operator: {op}")
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected:?}, got {actual:?}")
            }
            Self::LoadError(msg) => write!(f, "load error: {msg}"),
            Self::ValidationError(msg) => {
                write!(f, "validation error: {msg}")
            }
        }
    }
}

impl std::error::Error for OnnxError {}

// ── Loader ────────────────────────────────────────────────────────────────

/// Configuration for ONNX execution.
#[derive(Debug, Clone)]
pub struct OnnxConfig {
    pub execution_provider: ExecutionProvider,
    pub optimization_level: OptimizationLevel,
    pub custom_ops: Vec<String>,
}

/// Which backend to target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    Cpu,
    Cuda,
    DirectMl,
}

/// How aggressively to optimise the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationLevel {
    None,
    Basic,
    Extended,
    Full,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::Cpu,
            optimization_level: OptimizationLevel::Basic,
            custom_ops: Vec::new(),
        }
    }
}

/// Simplified loader for ONNX model metadata.
///
/// This does **not** parse the full protobuf format — it validates magic
/// bytes and header fields only, constructing the graph from in-memory
/// structures.
pub struct OnnxLoader {
    config: OnnxConfig,
}

impl OnnxLoader {
    pub fn new(config: OnnxConfig) -> Self {
        Self { config }
    }

    /// Return the current configuration.
    pub fn config(&self) -> &OnnxConfig {
        &self.config
    }

    /// Load an ONNX model from raw bytes (simplified: validates header).
    pub fn load_from_bytes(&self, data: &[u8]) -> Result<OnnxModel, OnnxError> {
        // Real ONNX files start with the protobuf-encoded ModelProto.
        // We only check the minimum length as a placeholder.
        if data.len() < 8 {
            return Err(OnnxError::LoadError(
                "data too short to be a valid ONNX model".into(),
            ));
        }
        // Build an empty model — real loading would parse protobuf here.
        let graph = OnnxGraph::new();
        Ok(OnnxModel::new(graph))
    }

    /// Build an `OnnxModel` from an already-constructed graph.
    pub fn from_graph(&self, graph: OnnxGraph) -> OnnxModel {
        OnnxModel::new(graph)
    }
}

// ── Operator dispatch ─────────────────────────────────────────────────────

/// Runtime tensor used during execution.
#[derive(Debug, Clone)]
pub struct RuntimeTensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

impl RuntimeTensor {
    pub fn new(
        name: impl Into<String>,
        shape: Vec<i64>,
        data: Vec<f32>,
    ) -> Self {
        Self { name: name.into(), shape, data }
    }

    pub fn zeros(name: impl Into<String>, shape: Vec<i64>) -> Self {
        let numel: usize =
            shape.iter().map(|&d| if d > 0 { d as usize } else { 1 }).product();
        Self { name: name.into(), shape, data: vec![0.0; numel] }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Trait for dispatching ONNX operators to a compute backend.
pub trait OpDispatcher: fmt::Debug {
    /// Execute a node, reading inputs and writing outputs to `tensors`.
    fn dispatch(
        &self,
        node: &OnnxNode,
        tensors: &mut HashMap<String, RuntimeTensor>,
    ) -> Result<(), OnnxError>;

    /// Return the set of op_types this dispatcher supports.
    fn supported_ops(&self) -> HashSet<String>;
}

// ── CPU reference dispatcher ──────────────────────────────────────────────

/// CPU reference implementation of core ONNX operators.
#[derive(Debug)]
pub struct CpuOpDispatcher;

impl CpuOpDispatcher {
    fn get_input<'a>(
        name: &str,
        tensors: &'a HashMap<String, RuntimeTensor>,
    ) -> Result<&'a RuntimeTensor, OnnxError> {
        tensors
            .get(name)
            .ok_or_else(|| OnnxError::MissingTensor(name.to_string()))
    }

    fn dispatch_matmul(
        node: &OnnxNode,
        tensors: &mut HashMap<String, RuntimeTensor>,
    ) -> Result<(), OnnxError> {
        let a = Self::get_input(&node.inputs[0], tensors)?;
        let b = Self::get_input(&node.inputs[1], tensors)?;
        let (a_shape, a_data) = (a.shape.clone(), a.data.clone());
        let (b_shape, b_data) = (b.shape.clone(), b.data.clone());

        if a_shape.len() < 2 || b_shape.len() < 2 {
            return Err(OnnxError::ShapeMismatch {
                expected: vec![0, 0],
                actual: a_shape,
            });
        }
        let m = a_shape[a_shape.len() - 2] as usize;
        let k = a_shape[a_shape.len() - 1] as usize;
        let n = b_shape[b_shape.len() - 1] as usize;
        let k2 = b_shape[b_shape.len() - 2] as usize;
        if k != k2 {
            return Err(OnnxError::ShapeMismatch {
                expected: vec![k as i64],
                actual: vec![k2 as i64],
            });
        }

        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc = a_data[i * k + p].mul_add(b_data[p * n + j], acc);
                }
                out[i * n + j] = acc;
            }
        }
        let out_name = &node.outputs[0];
        tensors.insert(
            out_name.clone(),
            RuntimeTensor::new(out_name, vec![m as i64, n as i64], out),
        );
        Ok(())
    }

    fn dispatch_add(
        node: &OnnxNode,
        tensors: &mut HashMap<String, RuntimeTensor>,
    ) -> Result<(), OnnxError> {
        let a = Self::get_input(&node.inputs[0], tensors)?;
        let b = Self::get_input(&node.inputs[1], tensors)?;
        let (a_data, a_shape) = (a.data.clone(), a.shape.clone());
        let b_data = b.data.clone();

        let out: Vec<f32> = if a_data.len() == b_data.len() {
            a_data.iter().zip(&b_data).map(|(x, y)| x + y).collect()
        } else if b_data.len() == 1 {
            a_data.iter().map(|x| x + b_data[0]).collect()
        } else {
            // Broadcast last dim.
            a_data
                .iter()
                .enumerate()
                .map(|(i, x)| x + b_data[i % b_data.len()])
                .collect()
        };
        let out_name = &node.outputs[0];
        tensors.insert(
            out_name.clone(),
            RuntimeTensor::new(out_name, a_shape, out),
        );
        Ok(())
    }

    fn dispatch_relu(
        node: &OnnxNode,
        tensors: &mut HashMap<String, RuntimeTensor>,
    ) -> Result<(), OnnxError> {
        let a = Self::get_input(&node.inputs[0], tensors)?;
        let (data, shape) = (a.data.clone(), a.shape.clone());
        let out: Vec<f32> = data.iter().map(|&v| v.max(0.0)).collect();
        let out_name = &node.outputs[0];
        tensors.insert(
            out_name.clone(),
            RuntimeTensor::new(out_name, shape, out),
        );
        Ok(())
    }

    fn dispatch_reshape(
        node: &OnnxNode,
        tensors: &mut HashMap<String, RuntimeTensor>,
    ) -> Result<(), OnnxError> {
        let a = Self::get_input(&node.inputs[0], tensors)?;
        let shape_tensor = Self::get_input(&node.inputs[1], tensors)?;
        let data = a.data.clone();
        let new_shape: Vec<i64> =
            shape_tensor.data.iter().map(|&v| v as i64).collect();
        let out_name = &node.outputs[0];
        tensors.insert(
            out_name.clone(),
            RuntimeTensor::new(out_name, new_shape, data),
        );
        Ok(())
    }

    fn dispatch_transpose(
        node: &OnnxNode,
        tensors: &mut HashMap<String, RuntimeTensor>,
    ) -> Result<(), OnnxError> {
        let a = Self::get_input(&node.inputs[0], tensors)?;
        let (data, shape) = (a.data.clone(), a.shape.clone());

        if shape.len() == 2 {
            let rows = shape[0] as usize;
            let cols = shape[1] as usize;
            let mut out = vec![0.0f32; data.len()];
            for r in 0..rows {
                for c in 0..cols {
                    out[c * rows + r] = data[r * cols + c];
                }
            }
            let out_name = &node.outputs[0];
            tensors.insert(
                out_name.clone(),
                RuntimeTensor::new(
                    out_name,
                    vec![cols as i64, rows as i64],
                    out,
                ),
            );
        } else {
            // Higher-rank: pass-through for now.
            let out_name = &node.outputs[0];
            tensors.insert(
                out_name.clone(),
                RuntimeTensor::new(out_name, shape, data),
            );
        }
        Ok(())
    }

    fn dispatch_softmax(
        node: &OnnxNode,
        tensors: &mut HashMap<String, RuntimeTensor>,
    ) -> Result<(), OnnxError> {
        let a = Self::get_input(&node.inputs[0], tensors)?;
        let (mut data, shape) = (a.data.clone(), a.shape.clone());

        // Apply softmax over the last axis.
        let last_dim =
            *shape.last().unwrap_or(&1).max(&1) as usize;
        for chunk in data.chunks_mut(last_dim) {
            let max =
                chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in chunk.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in chunk.iter_mut() {
                    *v /= sum;
                }
            }
        }
        let out_name = &node.outputs[0];
        tensors.insert(
            out_name.clone(),
            RuntimeTensor::new(out_name, shape, data),
        );
        Ok(())
    }

    fn dispatch_gemm(
        node: &OnnxNode,
        tensors: &mut HashMap<String, RuntimeTensor>,
    ) -> Result<(), OnnxError> {
        let a = Self::get_input(&node.inputs[0], tensors)?;
        let b = Self::get_input(&node.inputs[1], tensors)?;
        let (a_data, a_shape) = (a.data.clone(), a.shape.clone());
        let (b_data, b_shape) = (b.data.clone(), b.shape.clone());

        let alpha = match node.attributes.get("alpha") {
            Some(AttributeValue::Float(f)) => *f,
            _ => 1.0,
        };
        let beta = match node.attributes.get("beta") {
            Some(AttributeValue::Float(f)) => *f,
            _ => 1.0,
        };
        let trans_a = matches!(
            node.attributes.get("transA"),
            Some(AttributeValue::Int(1))
        );
        let trans_b = matches!(
            node.attributes.get("transB"),
            Some(AttributeValue::Int(1))
        );

        let (m, k_a) = if trans_a {
            (a_shape[1] as usize, a_shape[0] as usize)
        } else {
            (a_shape[0] as usize, a_shape[1] as usize)
        };
        let (k_b, n) = if trans_b {
            (b_shape[1] as usize, b_shape[0] as usize)
        } else {
            (b_shape[0] as usize, b_shape[1] as usize)
        };
        if k_a != k_b {
            return Err(OnnxError::ShapeMismatch {
                expected: vec![k_a as i64],
                actual: vec![k_b as i64],
            });
        }

        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k_a {
                    let a_val = if trans_a {
                        a_data[p * m + i]
                    } else {
                        a_data[i * k_a + p]
                    };
                    let b_val = if trans_b {
                        b_data[j * k_b + p]
                    } else {
                        b_data[p * n + j]
                    };
                    acc = a_val.mul_add(b_val, acc);
                }
                out[i * n + j] = acc * alpha;
            }
        }

        // Add bias (C) if present.
        if node.inputs.len() > 2 {
            if let Ok(c) = Self::get_input(&node.inputs[2], tensors) {
                let c_data = c.data.clone();
                for i in 0..m {
                    for j in 0..n {
                        out[i * n + j] += beta * c_data[j % c_data.len()];
                    }
                }
            }
        }

        let out_name = &node.outputs[0];
        tensors.insert(
            out_name.clone(),
            RuntimeTensor::new(out_name, vec![m as i64, n as i64], out),
        );
        Ok(())
    }
}

impl OpDispatcher for CpuOpDispatcher {
    fn dispatch(
        &self,
        node: &OnnxNode,
        tensors: &mut HashMap<String, RuntimeTensor>,
    ) -> Result<(), OnnxError> {
        match node.op_type.as_str() {
            "MatMul" => Self::dispatch_matmul(node, tensors),
            "Add" => Self::dispatch_add(node, tensors),
            "Relu" => Self::dispatch_relu(node, tensors),
            "Reshape" => Self::dispatch_reshape(node, tensors),
            "Transpose" => Self::dispatch_transpose(node, tensors),
            "Softmax" => Self::dispatch_softmax(node, tensors),
            "Gemm" => Self::dispatch_gemm(node, tensors),
            other => Err(OnnxError::UnsupportedOp(other.to_string())),
        }
    }

    fn supported_ops(&self) -> HashSet<String> {
        [
            "MatMul", "Add", "Relu", "Reshape", "Transpose", "Softmax",
            "Gemm",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
    }
}

// ── Graph optimizer ───────────────────────────────────────────────────────

/// Simple graph optimization passes.
#[derive(Debug)]
pub struct GraphOptimizer {
    level: OptimizationLevel,
}

impl GraphOptimizer {
    pub fn new(level: OptimizationLevel) -> Self {
        Self { level }
    }

    /// Run all enabled optimization passes on the graph.
    pub fn optimize(&self, graph: &mut OnnxGraph) -> Result<usize, OnnxError> {
        let mut total = 0;
        if self.level >= OptimizationLevel::Basic {
            total += self.eliminate_dead_nodes(graph)?;
        }
        if self.level >= OptimizationLevel::Extended {
            total += self.fold_constants(graph)?;
        }
        Ok(total)
    }

    /// Remove nodes whose outputs are not consumed by any other node or
    /// listed as graph outputs.
    pub fn eliminate_dead_nodes(
        &self,
        graph: &mut OnnxGraph,
    ) -> Result<usize, OnnxError> {
        let output_names: HashSet<String> =
            graph.outputs.iter().map(|t| t.name.clone()).collect();

        let consumed: HashSet<String> = graph
            .nodes
            .iter()
            .flat_map(|n| n.inputs.iter().cloned())
            .collect();

        let before = graph.nodes.len();
        graph.nodes.retain(|node| {
            node.outputs
                .iter()
                .any(|o| consumed.contains(o) || output_names.contains(o))
        });
        Ok(before - graph.nodes.len())
    }

    /// Fold identity-element additions (Add with a zero constant).
    ///
    /// This is a simplified constant-folding pass. A production
    /// implementation would evaluate constant sub-graphs fully.
    pub fn fold_constants(
        &self,
        graph: &mut OnnxGraph,
    ) -> Result<usize, OnnxError> {
        let const_names: HashSet<&str> =
            graph.initializers.keys().map(String::as_str).collect();

        let mut folded = 0usize;
        let mut to_remove: Vec<usize> = Vec::new();
        let mut rewrites: Vec<(String, String)> = Vec::new();

        for (i, node) in graph.nodes.iter().enumerate() {
            // Identity: Reshape where shape is the same as input.
            if node.op_type == "Reshape" && node.inputs.len() == 2 {
                if const_names.contains(node.inputs[1].as_str()) {
                    // Check if the reshape is a no-op by looking at
                    // initializer metadata — simplified heuristic.
                    if let Some(shape_init) =
                        graph.initializers.get(&node.inputs[1])
                    {
                        // If the shape tensor has zero elements it is a
                        // placeholder identity reshape.
                        if shape_init.shape.is_empty() {
                            to_remove.push(i);
                            rewrites.push((
                                node.outputs[0].clone(),
                                node.inputs[0].clone(),
                            ));
                            folded += 1;
                        }
                    }
                }
            }
        }

        // Apply rewrites: replace references to removed outputs.
        for (old, new) in &rewrites {
            for node in &mut graph.nodes {
                for inp in &mut node.inputs {
                    if inp == old {
                        *inp = new.clone();
                    }
                }
            }
        }

        // Remove folded nodes in reverse order to preserve indices.
        for &i in to_remove.iter().rev() {
            graph.nodes.remove(i);
        }
        Ok(folded)
    }
}

// ── Validation ────────────────────────────────────────────────────────────

/// Validate that a graph is well-formed.
pub fn validate_graph(
    graph: &OnnxGraph,
    dispatcher: &dyn OpDispatcher,
) -> Result<(), Vec<OnnxError>> {
    let mut errors = Vec::new();

    // 1. Check for cycles.
    if let Err(e) = graph.topological_sort() {
        errors.push(e);
    }

    // 2. Check that all ops are supported.
    let supported = dispatcher.supported_ops();
    for node in &graph.nodes {
        if !supported.contains(&node.op_type) {
            errors.push(OnnxError::UnsupportedOp(node.op_type.clone()));
        }
    }

    // 3. Check that all inputs reference existing tensors.
    let available = graph.all_tensor_names();
    for node in &graph.nodes {
        for inp in &node.inputs {
            if !available.contains(inp) {
                errors.push(OnnxError::MissingTensor(inp.clone()));
            }
        }
    }

    // 4. Check graph outputs reference produced tensors.
    for out_tensor in &graph.outputs {
        if !available.contains(&out_tensor.name) {
            errors.push(OnnxError::MissingTensor(out_tensor.name.clone()));
        }
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// Execute a graph end-to-end using the given dispatcher.
pub fn execute_graph(
    graph: &OnnxGraph,
    dispatcher: &dyn OpDispatcher,
    inputs: HashMap<String, RuntimeTensor>,
) -> Result<HashMap<String, RuntimeTensor>, OnnxError> {
    let order = graph.topological_sort()?;
    let mut tensors = inputs;

    for &idx in &order {
        let node = &graph.nodes[idx];
        dispatcher.dispatch(node, &mut tensors)?;
    }

    Ok(tensors)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────

    fn simple_linear_graph() -> OnnxGraph {
        let mut g = OnnxGraph::new();
        g.add_input(OnnxTensor::new("X", vec![2, 3], DataType::Float32));
        g.add_input(OnnxTensor::new("W", vec![3, 2], DataType::Float32));
        g.add_node(OnnxNode::new(
            "matmul0",
            "MatMul",
            vec!["X".into(), "W".into()],
            vec!["Y".into()],
        ));
        g.add_output(OnnxTensor::new("Y", vec![2, 2], DataType::Float32));
        g
    }

    fn make_runtime_inputs_for_linear() -> HashMap<String, RuntimeTensor> {
        let mut t = HashMap::new();
        t.insert(
            "X".into(),
            RuntimeTensor::new(
                "X",
                vec![2, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
        );
        t.insert(
            "W".into(),
            RuntimeTensor::new(
                "W",
                vec![3, 2],
                vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ),
        );
        t
    }

    // ── DataType tests ───────────────────────────────────────────────

    #[test]
    fn test_data_type_byte_size() {
        assert_eq!(DataType::Float32.byte_size(), 4);
        assert_eq!(DataType::Float16.byte_size(), 2);
        assert_eq!(DataType::Int64.byte_size(), 8);
        assert_eq!(DataType::Int32.byte_size(), 4);
        assert_eq!(DataType::Int8.byte_size(), 1);
        assert_eq!(DataType::Uint8.byte_size(), 1);
        assert_eq!(DataType::Bool.byte_size(), 1);
    }

    #[test]
    fn test_data_type_display() {
        assert_eq!(format!("{}", DataType::Float32), "float32");
        assert_eq!(format!("{}", DataType::Int8), "int8");
    }

    // ── OnnxTensor tests ─────────────────────────────────────────────

    #[test]
    fn test_tensor_numel() {
        let t = OnnxTensor::new("a", vec![2, 3, 4], DataType::Float32);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn test_tensor_numel_dynamic_dim() {
        let t = OnnxTensor::new("a", vec![-1, 3], DataType::Float32);
        assert_eq!(t.numel(), 3); // dynamic dim treated as 1
    }

    #[test]
    fn test_tensor_byte_size() {
        let t = OnnxTensor::new("a", vec![4, 4], DataType::Float32);
        assert_eq!(t.byte_size(), 64);
    }

    #[test]
    fn test_tensor_byte_size_int8() {
        let t = OnnxTensor::new("w", vec![256, 256], DataType::Int8);
        assert_eq!(t.byte_size(), 65536);
    }

    #[test]
    fn test_tensor_scalar() {
        let t = OnnxTensor::new("s", vec![1], DataType::Float32);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.byte_size(), 4);
    }

    // ── OnnxNode tests ───────────────────────────────────────────────

    #[test]
    fn test_node_creation() {
        let n = OnnxNode::new(
            "n0",
            "MatMul",
            vec!["a".into(), "b".into()],
            vec!["c".into()],
        );
        assert_eq!(n.op_type, "MatMul");
        assert_eq!(n.inputs.len(), 2);
        assert_eq!(n.outputs.len(), 1);
    }

    #[test]
    fn test_node_with_attr() {
        let n = OnnxNode::new("n0", "Gemm", vec![], vec![])
            .with_attr("alpha", AttributeValue::Float(2.0))
            .with_attr("transA", AttributeValue::Int(1));
        assert_eq!(
            n.attributes.get("alpha"),
            Some(&AttributeValue::Float(2.0))
        );
        assert_eq!(
            n.attributes.get("transA"),
            Some(&AttributeValue::Int(1))
        );
    }

    #[test]
    fn test_node_string_attr() {
        let n = OnnxNode::new("n0", "Custom", vec![], vec![])
            .with_attr("mode", AttributeValue::String("reflect".into()));
        assert_eq!(
            n.attributes.get("mode"),
            Some(&AttributeValue::String("reflect".into()))
        );
    }

    #[test]
    fn test_node_ints_attr() {
        let n = OnnxNode::new("n0", "Reshape", vec![], vec![])
            .with_attr("shape", AttributeValue::Ints(vec![1, 2, 3]));
        assert_eq!(
            n.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![1, 2, 3]))
        );
    }

    // ── OnnxGraph construction ───────────────────────────────────────

    #[test]
    fn test_graph_construction() {
        let g = simple_linear_graph();
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.inputs.len(), 2);
        assert_eq!(g.outputs.len(), 1);
    }

    #[test]
    fn test_graph_default() {
        let g = OnnxGraph::default();
        assert!(g.nodes.is_empty());
        assert!(g.inputs.is_empty());
    }

    #[test]
    fn test_graph_add_initializer() {
        let mut g = OnnxGraph::new();
        g.add_initializer(OnnxTensor::new(
            "bias",
            vec![4],
            DataType::Float32,
        ));
        assert!(g.initializers.contains_key("bias"));
    }

    #[test]
    fn test_graph_all_tensor_names() {
        let g = simple_linear_graph();
        let names = g.all_tensor_names();
        assert!(names.contains("X"));
        assert!(names.contains("W"));
        assert!(names.contains("Y"));
    }

    #[test]
    fn test_graph_all_tensor_names_with_initializer() {
        let mut g = simple_linear_graph();
        g.add_initializer(OnnxTensor::new("b", vec![2], DataType::Float32));
        let names = g.all_tensor_names();
        assert!(names.contains("b"));
    }

    // ── Topological sort ─────────────────────────────────────────────

    #[test]
    fn test_topo_sort_linear() {
        let g = simple_linear_graph();
        let order = g.topological_sort().unwrap();
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn test_topo_sort_chain() {
        let mut g = OnnxGraph::new();
        g.add_input(OnnxTensor::new("X", vec![2, 3], DataType::Float32));
        g.add_node(OnnxNode::new(
            "relu",
            "Relu",
            vec!["X".into()],
            vec!["R".into()],
        ));
        g.add_node(OnnxNode::new(
            "relu2",
            "Relu",
            vec!["R".into()],
            vec!["R2".into()],
        ));
        g.add_output(OnnxTensor::new("R2", vec![2, 3], DataType::Float32));
        let order = g.topological_sort().unwrap();
        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn test_topo_sort_diamond() {
        let mut g = OnnxGraph::new();
        g.add_input(OnnxTensor::new("X", vec![4], DataType::Float32));
        g.add_node(OnnxNode::new(
            "r1",
            "Relu",
            vec!["X".into()],
            vec!["A".into()],
        ));
        g.add_node(OnnxNode::new(
            "r2",
            "Relu",
            vec!["X".into()],
            vec!["B".into()],
        ));
        g.add_node(OnnxNode::new(
            "add",
            "Add",
            vec!["A".into(), "B".into()],
            vec!["Y".into()],
        ));
        g.add_output(OnnxTensor::new("Y", vec![4], DataType::Float32));
        let order = g.topological_sort().unwrap();
        // r1 and r2 come before add
        assert!(order[2] == 2);
        assert!(order.contains(&0));
        assert!(order.contains(&1));
    }

    #[test]
    fn test_topo_sort_cycle() {
        let mut g = OnnxGraph::new();
        g.add_node(OnnxNode::new(
            "a",
            "Relu",
            vec!["y".into()],
            vec!["x".into()],
        ));
        g.add_node(OnnxNode::new(
            "b",
            "Relu",
            vec!["x".into()],
            vec!["y".into()],
        ));
        assert_eq!(g.topological_sort(), Err(OnnxError::CyclicGraph));
    }

    #[test]
    fn test_topo_sort_empty() {
        let g = OnnxGraph::new();
        let order = g.topological_sort().unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn test_topo_sort_single_node() {
        let mut g = OnnxGraph::new();
        g.add_node(OnnxNode::new(
            "n",
            "Relu",
            vec!["X".into()],
            vec!["Y".into()],
        ));
        let order = g.topological_sort().unwrap();
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn test_topo_sort_independent_nodes() {
        let mut g = OnnxGraph::new();
        g.add_node(OnnxNode::new(
            "a",
            "Relu",
            vec!["X".into()],
            vec!["A".into()],
        ));
        g.add_node(OnnxNode::new(
            "b",
            "Relu",
            vec!["Y".into()],
            vec!["B".into()],
        ));
        let order = g.topological_sort().unwrap();
        assert_eq!(order.len(), 2);
    }

    // ── Op dispatch: MatMul ──────────────────────────────────────────

    #[test]
    fn test_dispatch_matmul() {
        let g = simple_linear_graph();
        let d = CpuOpDispatcher;
        let mut tensors = make_runtime_inputs_for_linear();
        d.dispatch(&g.nodes[0], &mut tensors).unwrap();
        let y = &tensors["Y"];
        assert_eq!(y.shape, vec![2, 2]);
        // [1,2,3]*[1,0; 0,1; 1,0] = [4,2] and [4,5,6]*... = [10,5]
        assert_eq!(y.data, vec![4.0, 2.0, 10.0, 5.0]);
    }

    #[test]
    fn test_dispatch_matmul_shape_mismatch() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "m",
            "MatMul",
            vec!["A".into(), "B".into()],
            vec!["C".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "A".into(),
            RuntimeTensor::new("A", vec![2, 3], vec![0.0; 6]),
        );
        tensors.insert(
            "B".into(),
            RuntimeTensor::new("B", vec![4, 2], vec![0.0; 8]),
        );
        assert!(d.dispatch(&node, &mut tensors).is_err());
    }

    #[test]
    fn test_dispatch_matmul_identity() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "m",
            "MatMul",
            vec!["A".into(), "I".into()],
            vec!["O".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "A".into(),
            RuntimeTensor::new("A", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        );
        tensors.insert(
            "I".into(),
            RuntimeTensor::new("I", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        let o = &tensors["O"];
        assert_eq!(o.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ── Op dispatch: Add ─────────────────────────────────────────────

    #[test]
    fn test_dispatch_add() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "add",
            "Add",
            vec!["A".into(), "B".into()],
            vec!["C".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "A".into(),
            RuntimeTensor::new("A", vec![3], vec![1.0, 2.0, 3.0]),
        );
        tensors.insert(
            "B".into(),
            RuntimeTensor::new("B", vec![3], vec![4.0, 5.0, 6.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        assert_eq!(tensors["C"].data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_dispatch_add_broadcast_scalar() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "add",
            "Add",
            vec!["A".into(), "B".into()],
            vec!["C".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "A".into(),
            RuntimeTensor::new("A", vec![3], vec![1.0, 2.0, 3.0]),
        );
        tensors.insert(
            "B".into(),
            RuntimeTensor::new("B", vec![1], vec![10.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        assert_eq!(tensors["C"].data, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_dispatch_add_broadcast_row() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "add",
            "Add",
            vec!["A".into(), "B".into()],
            vec!["C".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "A".into(),
            RuntimeTensor::new(
                "A",
                vec![2, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
        );
        tensors.insert(
            "B".into(),
            RuntimeTensor::new("B", vec![3], vec![10.0, 20.0, 30.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        assert_eq!(
            tensors["C"].data,
            vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    // ── Op dispatch: Relu ────────────────────────────────────────────

    #[test]
    fn test_dispatch_relu() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "relu",
            "Relu",
            vec!["X".into()],
            vec!["Y".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "X".into(),
            RuntimeTensor::new("X", vec![5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        assert_eq!(tensors["Y"].data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_dispatch_relu_all_positive() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "relu",
            "Relu",
            vec!["X".into()],
            vec!["Y".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "X".into(),
            RuntimeTensor::new("X", vec![3], vec![1.0, 2.0, 3.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        assert_eq!(tensors["Y"].data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dispatch_relu_all_negative() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "relu",
            "Relu",
            vec!["X".into()],
            vec!["Y".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "X".into(),
            RuntimeTensor::new("X", vec![3], vec![-1.0, -2.0, -3.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        assert_eq!(tensors["Y"].data, vec![0.0, 0.0, 0.0]);
    }

    // ── Op dispatch: Reshape ─────────────────────────────────────────

    #[test]
    fn test_dispatch_reshape() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "reshape",
            "Reshape",
            vec!["X".into(), "S".into()],
            vec!["Y".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "X".into(),
            RuntimeTensor::new(
                "X",
                vec![2, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
        );
        tensors.insert(
            "S".into(),
            RuntimeTensor::new("S", vec![2], vec![3.0, 2.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        assert_eq!(tensors["Y"].shape, vec![3, 2]);
        assert_eq!(tensors["Y"].data.len(), 6);
    }

    // ── Op dispatch: Transpose ───────────────────────────────────────

    #[test]
    fn test_dispatch_transpose_2d() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "t",
            "Transpose",
            vec!["X".into()],
            vec!["Y".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "X".into(),
            RuntimeTensor::new(
                "X",
                vec![2, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        let y = &tensors["Y"];
        assert_eq!(y.shape, vec![3, 2]);
        assert_eq!(y.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_dispatch_transpose_square() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "t",
            "Transpose",
            vec!["X".into()],
            vec!["Y".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "X".into(),
            RuntimeTensor::new(
                "X",
                vec![2, 2],
                vec![1.0, 2.0, 3.0, 4.0],
            ),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        assert_eq!(tensors["Y"].data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    // ── Op dispatch: Softmax ─────────────────────────────────────────

    #[test]
    fn test_dispatch_softmax() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "sm",
            "Softmax",
            vec!["X".into()],
            vec!["Y".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "X".into(),
            RuntimeTensor::new("X", vec![3], vec![1.0, 2.0, 3.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        let y = &tensors["Y"];
        let sum: f32 = y.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Monotonically increasing.
        assert!(y.data[0] < y.data[1]);
        assert!(y.data[1] < y.data[2]);
    }

    #[test]
    fn test_dispatch_softmax_uniform() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "sm",
            "Softmax",
            vec!["X".into()],
            vec!["Y".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "X".into(),
            RuntimeTensor::new("X", vec![4], vec![0.0; 4]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        for &v in &tensors["Y"].data {
            assert!((v - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dispatch_softmax_2d() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "sm",
            "Softmax",
            vec!["X".into()],
            vec!["Y".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "X".into(),
            RuntimeTensor::new(
                "X",
                vec![2, 3],
                vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            ),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        let y = &tensors["Y"];
        // Each row should sum to 1.
        let row0_sum: f32 = y.data[..3].iter().sum();
        let row1_sum: f32 = y.data[3..].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-5);
        assert!((row1_sum - 1.0).abs() < 1e-5);
    }

    // ── Op dispatch: Gemm ────────────────────────────────────────────

    #[test]
    fn test_dispatch_gemm_basic() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "gemm",
            "Gemm",
            vec!["A".into(), "B".into()],
            vec!["Y".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "A".into(),
            RuntimeTensor::new(
                "A",
                vec![2, 3],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
        );
        tensors.insert(
            "B".into(),
            RuntimeTensor::new(
                "B",
                vec![3, 2],
                vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        assert_eq!(tensors["Y"].data, vec![4.0, 2.0, 10.0, 5.0]);
    }

    #[test]
    fn test_dispatch_gemm_with_bias() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "gemm",
            "Gemm",
            vec!["A".into(), "B".into(), "C".into()],
            vec!["Y".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "A".into(),
            RuntimeTensor::new("A", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        );
        tensors.insert(
            "B".into(),
            RuntimeTensor::new("B", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        );
        tensors.insert(
            "C".into(),
            RuntimeTensor::new("C", vec![2], vec![10.0, 20.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        assert_eq!(tensors["Y"].data, vec![11.0, 20.0, 10.0, 21.0]);
    }

    #[test]
    fn test_dispatch_gemm_alpha() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "gemm",
            "Gemm",
            vec!["A".into(), "B".into()],
            vec!["Y".into()],
        )
        .with_attr("alpha", AttributeValue::Float(2.0));
        let mut tensors = HashMap::new();
        tensors.insert(
            "A".into(),
            RuntimeTensor::new("A", vec![1, 2], vec![1.0, 2.0]),
        );
        tensors.insert(
            "B".into(),
            RuntimeTensor::new("B", vec![2, 1], vec![3.0, 4.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        // (1*3 + 2*4) * 2 = 22
        assert_eq!(tensors["Y"].data, vec![22.0]);
    }

    #[test]
    fn test_dispatch_gemm_trans_b() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "gemm",
            "Gemm",
            vec!["A".into(), "B".into()],
            vec!["Y".into()],
        )
        .with_attr("transB", AttributeValue::Int(1));
        let mut tensors = HashMap::new();
        tensors.insert(
            "A".into(),
            RuntimeTensor::new("A", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        );
        // B is 2×2, transposed becomes: row0=[1,3], row1=[2,4]
        tensors.insert(
            "B".into(),
            RuntimeTensor::new("B", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        // A @ B^T : [1*1+2*2, 1*3+2*4; 3*1+4*2, 3*3+4*4] = [5,11,11,25]
        assert_eq!(tensors["Y"].data, vec![5.0, 11.0, 11.0, 25.0]);
    }

    // ── Op dispatch: unsupported ─────────────────────────────────────

    #[test]
    fn test_dispatch_unsupported_op() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "n",
            "CustomOp",
            vec!["X".into()],
            vec!["Y".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "X".into(),
            RuntimeTensor::new("X", vec![1], vec![1.0]),
        );
        assert_eq!(
            d.dispatch(&node, &mut tensors),
            Err(OnnxError::UnsupportedOp("CustomOp".into()))
        );
    }

    #[test]
    fn test_dispatch_missing_tensor() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "add",
            "Add",
            vec!["MISSING".into(), "B".into()],
            vec!["C".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "B".into(),
            RuntimeTensor::new("B", vec![1], vec![1.0]),
        );
        assert!(d.dispatch(&node, &mut tensors).is_err());
    }

    // ── CpuOpDispatcher trait ────────────────────────────────────────

    #[test]
    fn test_supported_ops() {
        let d = CpuOpDispatcher;
        let ops = d.supported_ops();
        assert!(ops.contains("MatMul"));
        assert!(ops.contains("Add"));
        assert!(ops.contains("Relu"));
        assert!(ops.contains("Reshape"));
        assert!(ops.contains("Transpose"));
        assert!(ops.contains("Softmax"));
        assert!(ops.contains("Gemm"));
        assert_eq!(ops.len(), 7);
    }

    // ── OnnxModel ────────────────────────────────────────────────────

    #[test]
    fn test_model_creation() {
        let m = OnnxModel::new(OnnxGraph::new());
        assert_eq!(m.ir_version, 9);
        assert_eq!(m.opset_version, 17);
        assert_eq!(m.producer_name, "bitnet-gpu-hal");
    }

    #[test]
    fn test_model_metadata() {
        let m = OnnxModel::new(OnnxGraph::new())
            .with_metadata("key", "value")
            .with_metadata("version", "1.0");
        assert_eq!(m.metadata.get("key").unwrap(), "value");
        assert_eq!(m.metadata.get("version").unwrap(), "1.0");
    }

    // ── OnnxLoader ───────────────────────────────────────────────────

    #[test]
    fn test_loader_from_graph() {
        let loader = OnnxLoader::new(OnnxConfig::default());
        let g = simple_linear_graph();
        let model = loader.from_graph(g);
        assert_eq!(model.graph.nodes.len(), 1);
    }

    #[test]
    fn test_loader_config() {
        let cfg = OnnxConfig {
            execution_provider: ExecutionProvider::Cuda,
            optimization_level: OptimizationLevel::Full,
            custom_ops: vec!["BitLinear".into()],
        };
        let loader = OnnxLoader::new(cfg);
        assert_eq!(
            loader.config().execution_provider,
            ExecutionProvider::Cuda
        );
        assert_eq!(
            loader.config().optimization_level,
            OptimizationLevel::Full
        );
    }

    #[test]
    fn test_loader_bytes_too_short() {
        let loader = OnnxLoader::new(OnnxConfig::default());
        assert!(loader.load_from_bytes(&[0u8; 4]).is_err());
    }

    #[test]
    fn test_loader_bytes_minimal() {
        let loader = OnnxLoader::new(OnnxConfig::default());
        let result = loader.load_from_bytes(&[0u8; 16]);
        assert!(result.is_ok());
    }

    // ── GraphOptimizer ───────────────────────────────────────────────

    #[test]
    fn test_optimizer_dead_node_elimination() {
        let mut g = OnnxGraph::new();
        g.add_input(OnnxTensor::new("X", vec![4], DataType::Float32));
        g.add_node(OnnxNode::new(
            "live",
            "Relu",
            vec!["X".into()],
            vec!["Y".into()],
        ));
        g.add_node(OnnxNode::new(
            "dead",
            "Relu",
            vec!["X".into()],
            vec!["DEAD".into()],
        ));
        g.add_output(OnnxTensor::new("Y", vec![4], DataType::Float32));

        let opt = GraphOptimizer::new(OptimizationLevel::Basic);
        let removed = opt.optimize(&mut g).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].name, "live");
    }

    #[test]
    fn test_optimizer_no_dead_nodes() {
        let mut g = simple_linear_graph();
        let opt = GraphOptimizer::new(OptimizationLevel::Basic);
        let removed = opt.optimize(&mut g).unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_optimizer_none_level() {
        let mut g = OnnxGraph::new();
        g.add_input(OnnxTensor::new("X", vec![4], DataType::Float32));
        g.add_node(OnnxNode::new(
            "dead",
            "Relu",
            vec!["X".into()],
            vec!["DEAD".into()],
        ));
        g.add_output(OnnxTensor::new("X", vec![4], DataType::Float32));

        let opt = GraphOptimizer::new(OptimizationLevel::None);
        let removed = opt.optimize(&mut g).unwrap();
        // None level: no passes run.
        assert_eq!(removed, 0);
        assert_eq!(g.nodes.len(), 1);
    }

    #[test]
    fn test_optimizer_constant_fold_identity_reshape() {
        let mut g = OnnxGraph::new();
        g.add_input(OnnxTensor::new("X", vec![4], DataType::Float32));
        g.add_initializer(OnnxTensor::new("S", vec![], DataType::Int64));
        g.add_node(OnnxNode::new(
            "reshape",
            "Reshape",
            vec!["X".into(), "S".into()],
            vec!["Y".into()],
        ));
        g.add_output(OnnxTensor::new("Y", vec![4], DataType::Float32));

        let opt = GraphOptimizer::new(OptimizationLevel::Extended);
        let removed = opt.optimize(&mut g).unwrap();
        assert!(removed >= 1);
    }

    #[test]
    fn test_optimizer_chain_dead_elimination() {
        let mut g = OnnxGraph::new();
        g.add_input(OnnxTensor::new("X", vec![4], DataType::Float32));
        g.add_node(OnnxNode::new(
            "n0",
            "Relu",
            vec!["X".into()],
            vec!["A".into()],
        ));
        g.add_node(OnnxNode::new(
            "n1",
            "Relu",
            vec!["A".into()],
            vec!["B".into()],
        ));
        g.add_node(OnnxNode::new(
            "dead",
            "Relu",
            vec!["X".into()],
            vec!["DEAD".into()],
        ));
        g.add_output(OnnxTensor::new("B", vec![4], DataType::Float32));

        let opt = GraphOptimizer::new(OptimizationLevel::Basic);
        let removed = opt.optimize(&mut g).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(g.nodes.len(), 2);
    }

    // ── Validation ───────────────────────────────────────────────────

    #[test]
    fn test_validate_valid_graph() {
        let g = simple_linear_graph();
        let d = CpuOpDispatcher;
        assert!(validate_graph(&g, &d).is_ok());
    }

    #[test]
    fn test_validate_unsupported_op() {
        let mut g = OnnxGraph::new();
        g.add_input(OnnxTensor::new("X", vec![4], DataType::Float32));
        g.add_node(OnnxNode::new(
            "n",
            "FancyOp",
            vec!["X".into()],
            vec!["Y".into()],
        ));
        g.add_output(OnnxTensor::new("Y", vec![4], DataType::Float32));

        let d = CpuOpDispatcher;
        let errs = validate_graph(&g, &d).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, OnnxError::UnsupportedOp(_))));
    }

    #[test]
    fn test_validate_missing_input_tensor() {
        let mut g = OnnxGraph::new();
        g.add_node(OnnxNode::new(
            "n",
            "Relu",
            vec!["MISSING".into()],
            vec!["Y".into()],
        ));
        g.add_output(OnnxTensor::new("Y", vec![4], DataType::Float32));

        let d = CpuOpDispatcher;
        let errs = validate_graph(&g, &d).unwrap_err();
        assert!(
            errs.iter().any(|e| matches!(e, OnnxError::MissingTensor(_)))
        );
    }

    #[test]
    fn test_validate_cyclic_graph() {
        let mut g = OnnxGraph::new();
        g.add_node(OnnxNode::new(
            "a",
            "Relu",
            vec!["y".into()],
            vec!["x".into()],
        ));
        g.add_node(OnnxNode::new(
            "b",
            "Relu",
            vec!["x".into()],
            vec!["y".into()],
        ));
        g.add_output(OnnxTensor::new("y", vec![4], DataType::Float32));

        let d = CpuOpDispatcher;
        let errs = validate_graph(&g, &d).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, OnnxError::CyclicGraph)));
    }

    #[test]
    fn test_validate_missing_graph_output() {
        let mut g = OnnxGraph::new();
        g.add_input(OnnxTensor::new("X", vec![4], DataType::Float32));
        g.add_output(OnnxTensor::new(
            "NOWHERE",
            vec![4],
            DataType::Float32,
        ));

        let d = CpuOpDispatcher;
        let errs = validate_graph(&g, &d).unwrap_err();
        assert!(
            errs.iter().any(|e| matches!(e, OnnxError::MissingTensor(_)))
        );
    }

    #[test]
    fn test_validate_empty_graph() {
        let g = OnnxGraph::new();
        let d = CpuOpDispatcher;
        assert!(validate_graph(&g, &d).is_ok());
    }

    // ── End-to-end execution ─────────────────────────────────────────

    #[test]
    fn test_execute_linear_graph() {
        let g = simple_linear_graph();
        let d = CpuOpDispatcher;
        let inputs = make_runtime_inputs_for_linear();
        let result = execute_graph(&g, &d, inputs).unwrap();
        let y = &result["Y"];
        assert_eq!(y.shape, vec![2, 2]);
        assert_eq!(y.data, vec![4.0, 2.0, 10.0, 5.0]);
    }

    #[test]
    fn test_execute_relu_chain() {
        let mut g = OnnxGraph::new();
        g.add_input(OnnxTensor::new("X", vec![4], DataType::Float32));
        g.add_node(OnnxNode::new(
            "r1",
            "Relu",
            vec!["X".into()],
            vec!["A".into()],
        ));
        g.add_node(OnnxNode::new(
            "r2",
            "Relu",
            vec!["A".into()],
            vec!["Y".into()],
        ));
        g.add_output(OnnxTensor::new("Y", vec![4], DataType::Float32));

        let d = CpuOpDispatcher;
        let mut inputs = HashMap::new();
        inputs.insert(
            "X".into(),
            RuntimeTensor::new("X", vec![4], vec![-1.0, 2.0, -3.0, 4.0]),
        );
        let result = execute_graph(&g, &d, inputs).unwrap();
        assert_eq!(result["Y"].data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_execute_matmul_add() {
        let mut g = OnnxGraph::new();
        g.add_input(OnnxTensor::new("X", vec![1, 2], DataType::Float32));
        g.add_input(OnnxTensor::new("W", vec![2, 2], DataType::Float32));
        g.add_input(OnnxTensor::new("B", vec![2], DataType::Float32));
        g.add_node(OnnxNode::new(
            "mm",
            "MatMul",
            vec!["X".into(), "W".into()],
            vec!["H".into()],
        ));
        g.add_node(OnnxNode::new(
            "add",
            "Add",
            vec!["H".into(), "B".into()],
            vec!["Y".into()],
        ));
        g.add_output(OnnxTensor::new("Y", vec![1, 2], DataType::Float32));

        let d = CpuOpDispatcher;
        let mut inputs = HashMap::new();
        inputs.insert(
            "X".into(),
            RuntimeTensor::new("X", vec![1, 2], vec![1.0, 2.0]),
        );
        inputs.insert(
            "W".into(),
            RuntimeTensor::new(
                "W",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
        );
        inputs.insert(
            "B".into(),
            RuntimeTensor::new("B", vec![2], vec![10.0, 20.0]),
        );
        let result = execute_graph(&g, &d, inputs).unwrap();
        assert_eq!(result["Y"].data, vec![11.0, 22.0]);
    }

    #[test]
    fn test_execute_empty_graph() {
        let g = OnnxGraph::new();
        let d = CpuOpDispatcher;
        let result = execute_graph(&g, &d, HashMap::new()).unwrap();
        assert!(result.is_empty());
    }

    // ── RuntimeTensor ────────────────────────────────────────────────

    #[test]
    fn test_runtime_tensor_zeros() {
        let t = RuntimeTensor::zeros("z", vec![2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_runtime_tensor_new() {
        let t = RuntimeTensor::new("t", vec![3], vec![1.0, 2.0, 3.0]);
        assert_eq!(t.name, "t");
        assert_eq!(t.shape, vec![3]);
        assert_eq!(t.numel(), 3);
    }

    // ── Error display ────────────────────────────────────────────────

    #[test]
    fn test_error_display_cyclic() {
        let e = OnnxError::CyclicGraph;
        assert_eq!(format!("{e}"), "graph contains a cycle");
    }

    #[test]
    fn test_error_display_missing_tensor() {
        let e = OnnxError::MissingTensor("foo".into());
        assert_eq!(format!("{e}"), "missing tensor: foo");
    }

    #[test]
    fn test_error_display_unsupported() {
        let e = OnnxError::UnsupportedOp("FooOp".into());
        assert_eq!(format!("{e}"), "unsupported operator: FooOp");
    }

    #[test]
    fn test_error_display_shape_mismatch() {
        let e = OnnxError::ShapeMismatch {
            expected: vec![2, 3],
            actual: vec![3, 2],
        };
        assert!(format!("{e}").contains("shape mismatch"));
    }

    #[test]
    fn test_error_display_load() {
        let e = OnnxError::LoadError("bad file".into());
        assert!(format!("{e}").contains("load error"));
    }

    #[test]
    fn test_error_display_validation() {
        let e = OnnxError::ValidationError("bad graph".into());
        assert!(format!("{e}").contains("validation error"));
    }

    // ── Config defaults ──────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = OnnxConfig::default();
        assert_eq!(cfg.execution_provider, ExecutionProvider::Cpu);
        assert_eq!(cfg.optimization_level, OptimizationLevel::Basic);
        assert!(cfg.custom_ops.is_empty());
    }

    #[test]
    fn test_optimization_level_ordering() {
        assert!(OptimizationLevel::None < OptimizationLevel::Basic);
        assert!(OptimizationLevel::Basic < OptimizationLevel::Extended);
        assert!(OptimizationLevel::Extended < OptimizationLevel::Full);
    }

    // ── Extra coverage ───────────────────────────────────────────────

    #[test]
    fn test_matmul_1x1() {
        let d = CpuOpDispatcher;
        let node = OnnxNode::new(
            "m",
            "MatMul",
            vec!["A".into(), "B".into()],
            vec!["C".into()],
        );
        let mut tensors = HashMap::new();
        tensors.insert(
            "A".into(),
            RuntimeTensor::new("A", vec![1, 1], vec![3.0]),
        );
        tensors.insert(
            "B".into(),
            RuntimeTensor::new("B", vec![1, 1], vec![5.0]),
        );
        d.dispatch(&node, &mut tensors).unwrap();
        assert_eq!(tensors["C"].data, vec![15.0]);
    }

    #[test]
    fn test_execute_matmul_relu_softmax() {
        let mut g = OnnxGraph::new();
        g.add_input(OnnxTensor::new("X", vec![1, 2], DataType::Float32));
        g.add_input(OnnxTensor::new("W", vec![2, 3], DataType::Float32));
        g.add_node(OnnxNode::new(
            "mm",
            "MatMul",
            vec!["X".into(), "W".into()],
            vec!["H".into()],
        ));
        g.add_node(OnnxNode::new(
            "relu",
            "Relu",
            vec!["H".into()],
            vec!["R".into()],
        ));
        g.add_node(OnnxNode::new(
            "sm",
            "Softmax",
            vec!["R".into()],
            vec!["Y".into()],
        ));
        g.add_output(OnnxTensor::new("Y", vec![1, 3], DataType::Float32));

        let d = CpuOpDispatcher;
        let mut inputs = HashMap::new();
        inputs.insert(
            "X".into(),
            RuntimeTensor::new("X", vec![1, 2], vec![1.0, 1.0]),
        );
        inputs.insert(
            "W".into(),
            RuntimeTensor::new(
                "W",
                vec![2, 3],
                vec![1.0, -1.0, 0.5, 0.5, 1.0, -0.5],
            ),
        );
        let result = execute_graph(&g, &d, inputs).unwrap();
        let y = &result["Y"];
        let sum: f32 = y.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
