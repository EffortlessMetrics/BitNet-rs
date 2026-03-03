//! Unified weight loading pipeline.
//!
//! Loads weights from any supported format through a multi-stage pipeline:
//! detect format → read tensors → validate shapes → convert dtypes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Supported model file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
    Onnx,
    PyTorch,
    Unknown,
}

impl ModelFormat {
    pub fn detect(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("gguf") => Self::Gguf,
            Some("safetensors") => Self::SafeTensors,
            Some("onnx") => Self::Onnx,
            Some("pt") | Some("pth") | Some("bin") => Self::PyTorch,
            _ => Self::Unknown,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Gguf => "GGUF",
            Self::SafeTensors => "SafeTensors",
            Self::Onnx => "ONNX",
            Self::PyTorch => "PyTorch",
            Self::Unknown => "Unknown",
        }
    }
}

/// Data type for weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    F32,
    F16,
    BF16,
    I8,
    I4,
    I2,
}

impl WeightDtype {
    pub fn bytes_per_element(&self) -> f64 {
        match self {
            Self::F32 => 4.0,
            Self::F16 | Self::BF16 => 2.0,
            Self::I8 => 1.0,
            Self::I4 => 0.5,
            Self::I2 => 0.25,
        }
    }
}

/// A weight tensor descriptor (metadata, not actual data).
#[derive(Debug, Clone)]
pub struct WeightDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
    pub size_bytes: usize,
}

impl WeightDescriptor {
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: WeightDtype) -> Self {
        let elements: usize = shape.iter().product();
        let size_bytes = (elements as f64 * dtype.bytes_per_element()) as usize;
        Self { name: name.into(), shape, dtype, size_bytes }
    }

    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    Detection,
    Reading,
    Validation,
    Conversion,
    Complete,
}

/// Validation result for a weight.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub tensor_name: String,
    pub passed: bool,
    pub message: String,
}

/// dtype conversion rule.
#[derive(Debug, Clone, Copy)]
pub struct ConversionRule {
    pub from: WeightDtype,
    pub to: WeightDtype,
}

impl ConversionRule {
    pub fn new(from: WeightDtype, to: WeightDtype) -> Self {
        Self { from, to }
    }

    pub fn is_identity(&self) -> bool {
        self.from == self.to
    }
}

/// Weight loading pipeline configuration.
#[derive(Debug, Clone)]
pub struct LoaderPipeline {
    pub source_path: PathBuf,
    pub format: ModelFormat,
    pub conversion_rules: Vec<ConversionRule>,
    pub expected_shapes: HashMap<String, Vec<usize>>,
    pub stage: PipelineStage,
    pub descriptors: Vec<WeightDescriptor>,
    pub validations: Vec<ValidationResult>,
}

impl LoaderPipeline {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let format = ModelFormat::detect(&path);
        Self {
            source_path: path,
            format,
            conversion_rules: Vec::new(),
            expected_shapes: HashMap::new(),
            stage: PipelineStage::Detection,
            descriptors: Vec::new(),
            validations: Vec::new(),
        }
    }

    pub fn with_format(mut self, format: ModelFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_conversion(mut self, rule: ConversionRule) -> Self {
        self.conversion_rules.push(rule);
        self
    }

    pub fn expect_shape(mut self, name: impl Into<String>, shape: Vec<usize>) -> Self {
        self.expected_shapes.insert(name.into(), shape);
        self
    }

    /// Register discovered tensors.
    pub fn register_tensors(&mut self, descriptors: Vec<WeightDescriptor>) {
        self.descriptors = descriptors;
        self.stage = PipelineStage::Reading;
    }

    /// Validate registered tensors against expected shapes.
    pub fn validate(&mut self) -> bool {
        self.validations.clear();
        let mut all_ok = true;

        for desc in &self.descriptors {
            if let Some(expected) = self.expected_shapes.get(&desc.name) {
                let passed = &desc.shape == expected;
                if !passed {
                    all_ok = false;
                }
                self.validations.push(ValidationResult {
                    tensor_name: desc.name.clone(),
                    passed,
                    message: if passed {
                        "shape matches".to_string()
                    } else {
                        format!("expected {:?}, got {:?}", expected, desc.shape)
                    },
                });
            }
        }

        self.stage = PipelineStage::Validation;
        all_ok
    }

    /// Apply conversion rules and return the resulting dtype for a tensor.
    pub fn resolve_dtype(&self, original: WeightDtype) -> WeightDtype {
        for rule in &self.conversion_rules {
            if rule.from == original {
                return rule.to;
            }
        }
        original
    }

    /// Mark pipeline complete.
    pub fn complete(&mut self) {
        self.stage = PipelineStage::Complete;
    }

    /// Total bytes across all descriptors.
    pub fn total_bytes(&self) -> usize {
        self.descriptors.iter().map(|d| d.size_bytes).sum()
    }

    /// Count of tensors.
    pub fn tensor_count(&self) -> usize {
        self.descriptors.len()
    }

    pub fn failed_validations(&self) -> Vec<&ValidationResult> {
        self.validations.iter().filter(|v| !v.passed).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection_gguf() {
        assert_eq!(ModelFormat::detect(Path::new("model.gguf")), ModelFormat::Gguf);
    }

    #[test]
    fn test_format_detection_safetensors() {
        assert_eq!(ModelFormat::detect(Path::new("model.safetensors")), ModelFormat::SafeTensors);
    }

    #[test]
    fn test_format_detection_unknown() {
        assert_eq!(ModelFormat::detect(Path::new("model.xyz")), ModelFormat::Unknown);
    }

    #[test]
    fn test_weight_descriptor() {
        let desc = WeightDescriptor::new("layer.weight", vec![768, 768], WeightDtype::F16);
        assert_eq!(desc.elements(), 768 * 768);
        assert_eq!(desc.size_bytes, 768 * 768 * 2);
    }

    #[test]
    fn test_bytes_per_element() {
        assert_eq!(WeightDtype::F32.bytes_per_element(), 4.0);
        assert_eq!(WeightDtype::I4.bytes_per_element(), 0.5);
        assert_eq!(WeightDtype::I2.bytes_per_element(), 0.25);
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = LoaderPipeline::new("model.gguf");
        assert_eq!(pipeline.format, ModelFormat::Gguf);
        assert_eq!(pipeline.stage, PipelineStage::Detection);
    }

    #[test]
    fn test_pipeline_validation_pass() {
        let mut pipeline =
            LoaderPipeline::new("model.safetensors").expect_shape("embed", vec![100, 512]);
        pipeline.register_tensors(vec![WeightDescriptor::new(
            "embed",
            vec![100, 512],
            WeightDtype::F16,
        )]);
        assert!(pipeline.validate());
        assert!(pipeline.failed_validations().is_empty());
    }

    #[test]
    fn test_pipeline_validation_fail() {
        let mut pipeline =
            LoaderPipeline::new("model.safetensors").expect_shape("embed", vec![100, 512]);
        pipeline.register_tensors(vec![WeightDescriptor::new(
            "embed",
            vec![200, 512],
            WeightDtype::F16,
        )]);
        assert!(!pipeline.validate());
        assert_eq!(pipeline.failed_validations().len(), 1);
    }

    #[test]
    fn test_conversion_rule() {
        let pipeline = LoaderPipeline::new("m.gguf")
            .with_conversion(ConversionRule::new(WeightDtype::BF16, WeightDtype::F16));
        assert_eq!(pipeline.resolve_dtype(WeightDtype::BF16), WeightDtype::F16);
        assert_eq!(pipeline.resolve_dtype(WeightDtype::F32), WeightDtype::F32);
    }

    #[test]
    fn test_identity_conversion() {
        let rule = ConversionRule::new(WeightDtype::F32, WeightDtype::F32);
        assert!(rule.is_identity());
    }

    #[test]
    fn test_total_bytes() {
        let mut pipeline = LoaderPipeline::new("m.gguf");
        pipeline.register_tensors(vec![
            WeightDescriptor::new("a", vec![100], WeightDtype::F32),
            WeightDescriptor::new("b", vec![200], WeightDtype::F16),
        ]);
        assert_eq!(pipeline.total_bytes(), 100 * 4 + 200 * 2);
    }

    #[test]
    fn test_pipeline_stages() {
        let mut pipeline = LoaderPipeline::new("m.safetensors");
        assert_eq!(pipeline.stage, PipelineStage::Detection);
        pipeline.register_tensors(vec![]);
        assert_eq!(pipeline.stage, PipelineStage::Reading);
        pipeline.validate();
        assert_eq!(pipeline.stage, PipelineStage::Validation);
        pipeline.complete();
        assert_eq!(pipeline.stage, PipelineStage::Complete);
    }
}
