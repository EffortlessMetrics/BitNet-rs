//! Model layer inspection and analysis.
//!
//! Enumerate, classify, and inspect transformer layers and their
//! constituent tensors for debugging and validation.

/// Type of a transformer layer component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentType {
    Embedding,
    Attention,
    FeedForward,
    Normalization,
    OutputHead,
    Other,
}

impl ComponentType {
    pub fn name(&self) -> &'static str {
        match self {
            ComponentType::Embedding => "embedding",
            ComponentType::Attention => "attention",
            ComponentType::FeedForward => "feed_forward",
            ComponentType::Normalization => "normalization",
            ComponentType::OutputHead => "output_head",
            ComponentType::Other => "other",
        }
    }
}

/// A tensor within a layer.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub component: ComponentType,
}

impl TensorInfo {
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: impl Into<String>) -> Self {
        let name_str: String = name.into();
        let component = classify_component(&name_str);
        Self { name: name_str, shape, dtype: dtype.into(), component }
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        let elem_size = match self.dtype.as_str() {
            "f32" | "i32" => 4,
            "f16" | "bf16" => 2,
            "i8" | "u8" => 1,
            _ => 4, // default to f32
        };
        self.num_elements() * elem_size
    }
}

/// Classify a tensor name into a component type.
fn classify_component(name: &str) -> ComponentType {
    let lower = name.to_lowercase();
    if lower.contains("embed") || lower.contains("wte") || lower.contains("wpe") {
        ComponentType::Embedding
    } else if lower.contains("attn")
        || lower.contains("attention")
        || lower.contains("q_proj")
        || lower.contains("k_proj")
        || lower.contains("v_proj")
        || lower.contains("o_proj")
    {
        ComponentType::Attention
    } else if lower.contains("mlp")
        || lower.contains("ffn")
        || lower.contains("gate_proj")
        || lower.contains("up_proj")
        || lower.contains("down_proj")
    {
        ComponentType::FeedForward
    } else if lower.contains("norm") || lower.contains("ln_") || lower.contains("layer_norm") {
        ComponentType::Normalization
    } else if lower.contains("lm_head") || lower.contains("output") {
        ComponentType::OutputHead
    } else {
        ComponentType::Other
    }
}

/// A model layer (group of tensors at the same depth).
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub index: usize,
    pub tensors: Vec<TensorInfo>,
}

impl LayerInfo {
    pub fn new(index: usize) -> Self {
        Self { index, tensors: Vec::new() }
    }

    pub fn add_tensor(&mut self, tensor: TensorInfo) {
        self.tensors.push(tensor);
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn total_params(&self) -> usize {
        self.tensors.iter().map(|t| t.num_elements()).sum()
    }

    pub fn total_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.size_bytes()).sum()
    }

    pub fn has_attention(&self) -> bool {
        self.tensors.iter().any(|t| t.component == ComponentType::Attention)
    }

    pub fn has_ffn(&self) -> bool {
        self.tensors.iter().any(|t| t.component == ComponentType::FeedForward)
    }

    pub fn has_norm(&self) -> bool {
        self.tensors.iter().any(|t| t.component == ComponentType::Normalization)
    }
}

/// Complete layer inspection report.
#[derive(Debug)]
pub struct InspectionReport {
    pub layers: Vec<LayerInfo>,
    pub non_layer_tensors: Vec<TensorInfo>,
}

impl InspectionReport {
    pub fn new() -> Self {
        Self { layers: Vec::new(), non_layer_tensors: Vec::new() }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn total_tensors(&self) -> usize {
        self.layers.iter().map(|l| l.tensor_count()).sum::<usize>() + self.non_layer_tensors.len()
    }

    pub fn total_params(&self) -> usize {
        self.layers.iter().map(|l| l.total_params()).sum::<usize>()
            + self.non_layer_tensors.iter().map(|t| t.num_elements()).sum::<usize>()
    }

    pub fn total_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.total_bytes()).sum::<usize>()
            + self.non_layer_tensors.iter().map(|t| t.size_bytes()).sum::<usize>()
    }

    pub fn params_millions(&self) -> f64 {
        self.total_params() as f64 / 1e6
    }

    pub fn summary(&self) -> String {
        format!(
            "{} layers, {} tensors, {:.1}M params, {:.1} MB",
            self.num_layers(),
            self.total_tensors(),
            self.params_millions(),
            self.total_bytes() as f64 / (1024.0 * 1024.0),
        )
    }
}

impl Default for InspectionReport {
    fn default() -> Self {
        Self::new()
    }
}

pub use bitnet_layer_index_core::extract_layer_index;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_attention() {
        assert_eq!(
            classify_component("model.layers.0.self_attn.q_proj.weight"),
            ComponentType::Attention
        );
    }

    #[test]
    fn test_classify_ffn() {
        assert_eq!(
            classify_component("model.layers.0.mlp.gate_proj.weight"),
            ComponentType::FeedForward
        );
    }

    #[test]
    fn test_classify_norm() {
        assert_eq!(
            classify_component("model.layers.0.input_layernorm.weight"),
            ComponentType::Normalization
        );
    }

    #[test]
    fn test_classify_embedding() {
        assert_eq!(classify_component("model.embed_tokens.weight"), ComponentType::Embedding);
    }

    #[test]
    fn test_classify_output() {
        assert_eq!(classify_component("lm_head.weight"), ComponentType::OutputHead);
    }

    #[test]
    fn test_tensor_info() {
        let t = TensorInfo::new("test.weight", vec![512, 512], "f32");
        assert_eq!(t.num_elements(), 262144);
        assert_eq!(t.size_bytes(), 1048576);
    }

    #[test]
    fn test_tensor_f16_size() {
        let t = TensorInfo::new("test.weight", vec![100], "f16");
        assert_eq!(t.size_bytes(), 200);
    }

    #[test]
    fn test_layer_info() {
        let mut layer = LayerInfo::new(0);
        layer.add_tensor(TensorInfo::new("attn.q_proj", vec![512, 512], "f32"));
        layer.add_tensor(TensorInfo::new("mlp.gate_proj", vec![512, 1024], "f32"));
        assert!(layer.has_attention());
        assert!(layer.has_ffn());
        assert_eq!(layer.tensor_count(), 2);
    }

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(extract_layer_index("model.layers.5.attn.q_proj"), Some(5));
        assert_eq!(extract_layer_index("model.layers.39.mlp"), Some(39));
        assert_eq!(extract_layer_index("embed_tokens.weight"), None);
    }

    #[test]
    fn test_inspection_report() {
        let mut report = InspectionReport::new();
        let mut layer = LayerInfo::new(0);
        layer.add_tensor(TensorInfo::new("attn", vec![100], "f32"));
        report.layers.push(layer);
        report.non_layer_tensors.push(TensorInfo::new("embed", vec![1000], "f16"));
        assert_eq!(report.num_layers(), 1);
        assert_eq!(report.total_tensors(), 2);
    }

    #[test]
    fn test_report_summary() {
        let report = InspectionReport::new();
        let s = report.summary();
        assert!(s.contains("layers"));
    }

    #[test]
    fn test_component_name() {
        assert_eq!(ComponentType::Attention.name(), "attention");
        assert_eq!(ComponentType::FeedForward.name(), "feed_forward");
    }
}
