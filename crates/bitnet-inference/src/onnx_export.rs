//! ONNX export with GPU acceleration metadata.
//!
//! Provides facilities to export a BitNet model graph annotated with GPU
//! kernel selection hints, device capabilities, and custom operator
//! registrations for BitNet-specific quantized operations.
//!
//! The output is a self-describing JSON structure (not a binary protobuf) so
//! that downstream tooling can inspect or convert it without an ONNX runtime
//! dependency.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Device capabilities
// ---------------------------------------------------------------------------

/// Snapshot of GPU device capabilities embedded in ONNX metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceCapabilities {
    pub device_name: String,
    pub compute_units: u32,
    pub max_work_group_size: u32,
    pub global_mem_bytes: u64,
    pub supports_fp16: bool,
    pub supports_subgroups: bool,
    pub driver_version: String,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            device_name: "cpu".to_string(),
            compute_units: 1,
            max_work_group_size: 1,
            global_mem_bytes: 0,
            supports_fp16: false,
            supports_subgroups: false,
            driver_version: "n/a".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel annotation
// ---------------------------------------------------------------------------

/// Annotation attached to a graph node indicating which kernel to prefer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KernelAnnotation {
    /// Human-readable kernel name (e.g. `"i2s_gemv_avx2"`).
    pub kernel_name: String,
    /// Target device type: `"cpu"`, `"cuda"`, `"opencl"`.
    pub target_device: String,
    /// Optional SIMD level hint (e.g. `"avx2"`, `"avx512"`, `"neon"`).
    pub simd_hint: Option<String>,
    /// Arbitrary key-value hints consumed by the runtime.
    pub extra: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Custom operator registration
// ---------------------------------------------------------------------------

/// Registration entry for a custom BitNet quantized operator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomOpRegistration {
    /// Operator domain (e.g. `"com.bitnet"`).
    pub domain: String,
    /// Operator name within the domain.
    pub op_type: String,
    /// Operator version.
    pub version: u32,
    /// Supported input quantization types.
    pub input_types: Vec<String>,
    /// Supported output types.
    pub output_types: Vec<String>,
    /// Human description.
    pub description: String,
}

// ---------------------------------------------------------------------------
// Graph node
// ---------------------------------------------------------------------------

/// Simplified representation of a single node in the exported model graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// Optional GPU kernel annotation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_annotation: Option<KernelAnnotation>,
    /// Arbitrary attributes (weights shape, etc.).
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub attributes: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// ONNX model graph
// ---------------------------------------------------------------------------

/// Top-level exported model graph with GPU metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OnnxModelGraph {
    /// Model name / identifier.
    pub model_name: String,
    /// IR version (ONNX-style).
    pub ir_version: u32,
    /// Device capabilities snapshot.
    pub device_capabilities: DeviceCapabilities,
    /// Custom operator registrations.
    pub custom_ops: Vec<CustomOpRegistration>,
    /// Graph nodes (topological order).
    pub nodes: Vec<GraphNode>,
    /// Graph-level metadata (e.g. quantization scheme).
    pub metadata: HashMap<String, String>,
}

impl OnnxModelGraph {
    /// Create a new empty graph.
    pub fn new(model_name: impl Into<String>, device: DeviceCapabilities) -> Self {
        Self {
            model_name: model_name.into(),
            ir_version: 9,
            device_capabilities: device,
            custom_ops: Vec::new(),
            nodes: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Register a custom BitNet operator.
    pub fn register_custom_op(&mut self, op: CustomOpRegistration) {
        self.custom_ops.push(op);
    }

    /// Add a graph node.
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.push(node);
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Serialize the graph to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec_pretty(self)
    }

    /// Deserialize from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }

    /// Return all kernel annotations present in the graph.
    pub fn kernel_hints(&self) -> Vec<&KernelAnnotation> {
        self.nodes.iter().filter_map(|n| n.kernel_annotation.as_ref()).collect()
    }
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Convenience builder for a BitNet I2_S GEMV custom op registration.
pub fn bitnet_i2s_gemv_op() -> CustomOpRegistration {
    CustomOpRegistration {
        domain: "com.bitnet".to_string(),
        op_type: "I2S_GEMV".to_string(),
        version: 1,
        input_types: vec!["i2_s".to_string(), "f32".to_string()],
        output_types: vec!["f32".to_string()],
        description: "Ternary GEMV with I2_S packed weights".to_string(),
    }
}

/// Convenience builder for a QK256 dequantize custom op registration.
pub fn bitnet_qk256_dequant_op() -> CustomOpRegistration {
    CustomOpRegistration {
        domain: "com.bitnet".to_string(),
        op_type: "QK256_Dequantize".to_string(),
        version: 1,
        input_types: vec!["qk256".to_string()],
        output_types: vec!["f32".to_string()],
        description: "QK256 block dequantization (256-element groups)".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_device() -> DeviceCapabilities {
        DeviceCapabilities {
            device_name: "Intel Arc A770".to_string(),
            compute_units: 512,
            max_work_group_size: 1024,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            supports_fp16: true,
            supports_subgroups: true,
            driver_version: "24.17.31.20".to_string(),
        }
    }

    #[test]
    fn test_empty_graph_creation() {
        let graph = OnnxModelGraph::new("test-model", sample_device());
        assert_eq!(graph.model_name, "test-model");
        assert_eq!(graph.ir_version, 9);
        assert!(graph.nodes.is_empty());
        assert!(graph.custom_ops.is_empty());
        assert_eq!(graph.device_capabilities.device_name, "Intel Arc A770");
    }

    #[test]
    fn test_custom_op_registration() {
        let mut graph = OnnxModelGraph::new("model", DeviceCapabilities::default());
        graph.register_custom_op(bitnet_i2s_gemv_op());
        graph.register_custom_op(bitnet_qk256_dequant_op());
        assert_eq!(graph.custom_ops.len(), 2);
        assert_eq!(graph.custom_ops[0].op_type, "I2S_GEMV");
        assert_eq!(graph.custom_ops[1].op_type, "QK256_Dequantize");
    }

    #[test]
    fn test_add_node_with_kernel_annotation() {
        let mut graph = OnnxModelGraph::new("model", sample_device());
        let node = GraphNode {
            name: "linear_0".to_string(),
            op_type: "I2S_GEMV".to_string(),
            inputs: vec!["input".to_string(), "weights_0".to_string()],
            outputs: vec!["hidden_0".to_string()],
            kernel_annotation: Some(KernelAnnotation {
                kernel_name: "i2s_gemv_avx2".to_string(),
                target_device: "cpu".to_string(),
                simd_hint: Some("avx2".to_string()),
                extra: HashMap::new(),
            }),
            attributes: HashMap::new(),
        };
        graph.add_node(node);
        assert_eq!(graph.nodes.len(), 1);
        let hints = graph.kernel_hints();
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].kernel_name, "i2s_gemv_avx2");
    }

    #[test]
    fn test_json_round_trip() {
        let mut graph = OnnxModelGraph::new("round-trip", sample_device());
        graph.register_custom_op(bitnet_i2s_gemv_op());
        graph.set_metadata("quantization", "i2_s");
        graph.add_node(GraphNode {
            name: "matmul_0".to_string(),
            op_type: "MatMul".to_string(),
            inputs: vec!["a".to_string()],
            outputs: vec!["b".to_string()],
            kernel_annotation: None,
            attributes: HashMap::new(),
        });

        let json = graph.to_json().expect("serialize");
        let restored = OnnxModelGraph::from_json(&json).expect("deserialize");
        assert_eq!(graph, restored);
    }

    #[test]
    fn test_metadata_set_and_read() {
        let mut graph = OnnxModelGraph::new("m", DeviceCapabilities::default());
        graph.set_metadata("quant", "qk256");
        graph.set_metadata("version", "0.2.1-dev");
        assert_eq!(graph.metadata.get("quant").unwrap(), "qk256");
        assert_eq!(graph.metadata.get("version").unwrap(), "0.2.1-dev");
    }

    #[test]
    fn test_kernel_hints_filters_none() {
        let mut graph = OnnxModelGraph::new("m", DeviceCapabilities::default());
        graph.add_node(GraphNode {
            name: "a".to_string(),
            op_type: "Add".to_string(),
            inputs: vec![],
            outputs: vec![],
            kernel_annotation: None,
            attributes: HashMap::new(),
        });
        graph.add_node(GraphNode {
            name: "b".to_string(),
            op_type: "I2S_GEMV".to_string(),
            inputs: vec![],
            outputs: vec![],
            kernel_annotation: Some(KernelAnnotation {
                kernel_name: "gemv_opencl".to_string(),
                target_device: "opencl".to_string(),
                simd_hint: None,
                extra: HashMap::new(),
            }),
            attributes: HashMap::new(),
        });
        let hints = graph.kernel_hints();
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].target_device, "opencl");
    }
}
