//! Inference pipeline builder.
//!
//! Declarative configuration of model loading → inference pipeline steps.

use std::collections::HashMap;

/// A step in the inference pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStep {
    LoadModel { path: String },
    LoadTokenizer { path: String },
    SetContext { max_tokens: usize },
    SetBatchSize { size: usize },
    EnableQuantization { format: String },
    SetDevice { device: String },
    Warmup { tokens: usize },
    Custom { name: String, config: HashMap<String, String> },
}

/// Validation errors for pipeline configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineError {
    MissingModel,
    MissingTokenizer,
    InvalidContext(usize),
    InvalidBatchSize(usize),
    DuplicateStep(String),
    InvalidDevice(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingModel => write!(f, "no model path specified"),
            Self::MissingTokenizer => write!(f, "no tokenizer path specified"),
            Self::InvalidContext(n) => write!(f, "invalid context length: {n}"),
            Self::InvalidBatchSize(n) => write!(f, "invalid batch size: {n}"),
            Self::DuplicateStep(s) => write!(f, "duplicate step: {s}"),
            Self::InvalidDevice(d) => write!(f, "invalid device: {d}"),
        }
    }
}

/// Builder for inference pipelines.
#[derive(Debug, Clone)]
pub struct PipelineBuilder {
    steps: Vec<PipelineStep>,
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    context_size: usize,
    batch_size: usize,
    device: String,
    warmup_tokens: Option<usize>,
    metadata: HashMap<String, String>,
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            model_path: None,
            tokenizer_path: None,
            context_size: 2048,
            batch_size: 1,
            device: "cpu".to_string(),
            warmup_tokens: None,
            metadata: HashMap::new(),
        }
    }

    pub fn model(mut self, path: &str) -> Self {
        self.model_path = Some(path.to_string());
        self.steps.push(PipelineStep::LoadModel { path: path.to_string() });
        self
    }

    pub fn tokenizer(mut self, path: &str) -> Self {
        self.tokenizer_path = Some(path.to_string());
        self.steps.push(PipelineStep::LoadTokenizer { path: path.to_string() });
        self
    }

    pub fn context_size(mut self, size: usize) -> Self {
        self.context_size = size;
        self.steps.push(PipelineStep::SetContext { max_tokens: size });
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self.steps.push(PipelineStep::SetBatchSize { size });
        self
    }

    pub fn quantization(mut self, format: &str) -> Self {
        self.steps.push(PipelineStep::EnableQuantization { format: format.to_string() });
        self
    }

    pub fn device(mut self, device: &str) -> Self {
        self.device = device.to_string();
        self.steps.push(PipelineStep::SetDevice { device: device.to_string() });
        self
    }

    pub fn warmup(mut self, tokens: usize) -> Self {
        self.warmup_tokens = Some(tokens);
        self.steps.push(PipelineStep::Warmup { tokens });
        self
    }

    pub fn custom_step(mut self, name: &str, config: HashMap<String, String>) -> Self {
        self.steps.push(PipelineStep::Custom { name: name.to_string(), config });
        self
    }

    pub fn metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Validate the pipeline configuration.
    pub fn validate(&self) -> Result<(), Vec<PipelineError>> {
        let mut errors = Vec::new();

        if self.model_path.is_none() {
            errors.push(PipelineError::MissingModel);
        }
        if self.tokenizer_path.is_none() {
            errors.push(PipelineError::MissingTokenizer);
        }
        if self.context_size == 0 || self.context_size > 1_048_576 {
            errors.push(PipelineError::InvalidContext(self.context_size));
        }
        if self.batch_size == 0 || self.batch_size > 1024 {
            errors.push(PipelineError::InvalidBatchSize(self.batch_size));
        }
        let valid_devices = ["cpu", "cuda", "metal", "vulkan", "opencl"];
        if !valid_devices.contains(&self.device.as_str()) {
            errors.push(PipelineError::InvalidDevice(self.device.clone()));
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    /// Build the pipeline configuration (returns steps if valid).
    pub fn build(self) -> Result<PipelineConfig, Vec<PipelineError>> {
        self.validate()?;
        Ok(PipelineConfig {
            steps: self.steps,
            model_path: self.model_path.unwrap_or_default(),
            tokenizer_path: self.tokenizer_path.unwrap_or_default(),
            context_size: self.context_size,
            batch_size: self.batch_size,
            device: self.device,
            warmup_tokens: self.warmup_tokens,
            metadata: self.metadata,
        })
    }

    pub fn steps(&self) -> &[PipelineStep] {
        &self.steps
    }
}

/// A validated pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub steps: Vec<PipelineStep>,
    pub model_path: String,
    pub tokenizer_path: String,
    pub context_size: usize,
    pub batch_size: usize,
    pub device: String,
    pub warmup_tokens: Option<usize>,
    pub metadata: HashMap<String, String>,
}

impl PipelineConfig {
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
    pub fn has_warmup(&self) -> bool {
        self.warmup_tokens.is_some()
    }
    pub fn has_quantization(&self) -> bool {
        self.steps.iter().any(|s| matches!(s, PipelineStep::EnableQuantization { .. }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_builder() -> PipelineBuilder {
        PipelineBuilder::new().model("model.gguf").tokenizer("tokenizer.json")
    }

    #[test]
    fn test_build_valid() {
        let cfg = valid_builder().build().unwrap();
        assert_eq!(cfg.model_path, "model.gguf");
        assert_eq!(cfg.tokenizer_path, "tokenizer.json");
        assert_eq!(cfg.context_size, 2048);
    }

    #[test]
    fn test_missing_model() {
        let r = PipelineBuilder::new().tokenizer("t.json").build();
        assert!(r.is_err());
        assert!(r.unwrap_err().contains(&PipelineError::MissingModel));
    }

    #[test]
    fn test_missing_tokenizer() {
        let r = PipelineBuilder::new().model("m.gguf").build();
        assert!(r.is_err());
        assert!(r.unwrap_err().contains(&PipelineError::MissingTokenizer));
    }

    #[test]
    fn test_context_size() {
        let cfg = valid_builder().context_size(16384).build().unwrap();
        assert_eq!(cfg.context_size, 16384);
    }

    #[test]
    fn test_invalid_context() {
        let r = valid_builder().context_size(0).build();
        assert!(r.is_err());
    }

    #[test]
    fn test_batch_size() {
        let cfg = valid_builder().batch_size(8).build().unwrap();
        assert_eq!(cfg.batch_size, 8);
    }

    #[test]
    fn test_invalid_batch() {
        let r = valid_builder().batch_size(0).build();
        assert!(r.is_err());
    }

    #[test]
    fn test_device_cpu() {
        let cfg = valid_builder().device("cpu").build().unwrap();
        assert_eq!(cfg.device, "cpu");
    }

    #[test]
    fn test_device_cuda() {
        let cfg = valid_builder().device("cuda").build().unwrap();
        assert_eq!(cfg.device, "cuda");
    }

    #[test]
    fn test_invalid_device() {
        let r = valid_builder().device("tpu").build();
        assert!(r.is_err());
    }

    #[test]
    fn test_warmup() {
        let cfg = valid_builder().warmup(32).build().unwrap();
        assert!(cfg.has_warmup());
        assert_eq!(cfg.warmup_tokens, Some(32));
    }

    #[test]
    fn test_no_warmup() {
        let cfg = valid_builder().build().unwrap();
        assert!(!cfg.has_warmup());
    }

    #[test]
    fn test_quantization() {
        let cfg = valid_builder().quantization("int4").build().unwrap();
        assert!(cfg.has_quantization());
    }

    #[test]
    fn test_step_count() {
        let cfg = valid_builder().warmup(10).device("cpu").build().unwrap();
        assert!(cfg.step_count() >= 4); // model + tokenizer + warmup + device
    }

    #[test]
    fn test_metadata() {
        let cfg = valid_builder().metadata("version", "1.0").build().unwrap();
        assert_eq!(cfg.metadata.get("version").unwrap(), "1.0");
    }

    #[test]
    fn test_default() {
        let b = PipelineBuilder::default();
        assert_eq!(b.steps().len(), 0);
    }
}
