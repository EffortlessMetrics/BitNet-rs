//! Fluent API for assembling inference pipelines.
//!
//! Compose preprocessing, inference stages, and postprocessing
//! into a configurable execution pipeline.

use std::collections::HashMap;

/// A stage in the inference pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StageKind {
    /// Input tokenization.
    Tokenize,
    /// Input embedding lookup.
    Embed,
    /// Transformer layer processing.
    TransformerBlock,
    /// Final layer norm.
    FinalNorm,
    /// Logits projection.
    LogitsProjection,
    /// Sampling (greedy, top-k, etc.).
    Sample,
    /// Token decoding.
    Decode,
    /// Custom named stage.
    Custom(String),
}

impl StageKind {
    pub fn name(&self) -> &str {
        match self {
            Self::Tokenize => "tokenize",
            Self::Embed => "embed",
            Self::TransformerBlock => "transformer_block",
            Self::FinalNorm => "final_norm",
            Self::LogitsProjection => "logits_projection",
            Self::Sample => "sample",
            Self::Decode => "decode",
            Self::Custom(name) => name,
        }
    }
}

/// Configuration for a pipeline stage.
#[derive(Debug, Clone)]
pub struct StageConfig {
    pub kind: StageKind,
    pub enabled: bool,
    pub params: HashMap<String, String>,
}

impl StageConfig {
    pub fn new(kind: StageKind) -> Self {
        Self { kind, enabled: true, params: HashMap::new() }
    }

    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

/// Builder for constructing inference pipelines.
#[derive(Debug, Default)]
pub struct PipelineBuilder {
    stages: Vec<StageConfig>,
    name: Option<String>,
    num_layers: Option<usize>,
    batch_size: usize,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self { stages: Vec::new(), name: None, num_layers: None, batch_size: 1 }
    }

    /// Set pipeline name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set number of transformer layers.
    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = Some(n);
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    /// Add a stage.
    pub fn add_stage(mut self, config: StageConfig) -> Self {
        self.stages.push(config);
        self
    }

    /// Add tokenization stage.
    pub fn tokenize(self) -> Self {
        self.add_stage(StageConfig::new(StageKind::Tokenize))
    }

    /// Add embedding stage.
    pub fn embed(self) -> Self {
        self.add_stage(StageConfig::new(StageKind::Embed))
    }

    /// Add N transformer block stages.
    pub fn transformer_blocks(mut self) -> Self {
        let n = self.num_layers.unwrap_or(1);
        for _ in 0..n {
            self.stages.push(StageConfig::new(StageKind::TransformerBlock));
        }
        self
    }

    /// Add final normalization stage.
    pub fn final_norm(self) -> Self {
        self.add_stage(StageConfig::new(StageKind::FinalNorm))
    }

    /// Add logits projection stage.
    pub fn logits_projection(self) -> Self {
        self.add_stage(StageConfig::new(StageKind::LogitsProjection))
    }

    /// Add sampling stage.
    pub fn sample(self) -> Self {
        self.add_stage(StageConfig::new(StageKind::Sample))
    }

    /// Add decoding stage.
    pub fn decode(self) -> Self {
        self.add_stage(StageConfig::new(StageKind::Decode))
    }

    /// Build the pipeline.
    pub fn build(self) -> Pipeline {
        Pipeline {
            name: self.name.unwrap_or_else(|| "default".into()),
            stages: self.stages,
            batch_size: self.batch_size,
        }
    }

    /// Build a standard text generation pipeline.
    pub fn text_generation(num_layers: usize) -> Pipeline {
        PipelineBuilder::new()
            .name("text_generation")
            .num_layers(num_layers)
            .tokenize()
            .embed()
            .transformer_blocks()
            .final_norm()
            .logits_projection()
            .sample()
            .decode()
            .build()
    }
}

/// A configured inference pipeline.
#[derive(Debug)]
pub struct Pipeline {
    pub name: String,
    pub stages: Vec<StageConfig>,
    pub batch_size: usize,
}

impl Pipeline {
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    pub fn enabled_stages(&self) -> Vec<&StageConfig> {
        self.stages.iter().filter(|s| s.enabled).collect()
    }

    pub fn has_stage(&self, kind: &StageKind) -> bool {
        self.stages.iter().any(|s| s.kind == *kind)
    }

    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.kind.name()).collect()
    }

    /// Summary of the pipeline.
    pub fn summary(&self) -> String {
        format!(
            "Pipeline '{}': {} stages, batch_size={}",
            self.name,
            self.stage_count(),
            self.batch_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_kind_name() {
        assert_eq!(StageKind::Tokenize.name(), "tokenize");
        assert_eq!(StageKind::TransformerBlock.name(), "transformer_block");
        assert_eq!(StageKind::Custom("my_stage".into()).name(), "my_stage");
    }

    #[test]
    fn test_stage_config_params() {
        let cfg = StageConfig::new(StageKind::Sample)
            .with_param("temperature", "0.7")
            .with_param("top_k", "50");
        assert_eq!(cfg.params.len(), 2);
        assert_eq!(cfg.params["temperature"], "0.7");
    }

    #[test]
    fn test_stage_disabled() {
        let cfg = StageConfig::new(StageKind::Embed).disabled();
        assert!(!cfg.enabled);
    }

    #[test]
    fn test_builder_basic() {
        let pipeline = PipelineBuilder::new().name("test").tokenize().embed().build();
        assert_eq!(pipeline.name, "test");
        assert_eq!(pipeline.stage_count(), 2);
    }

    #[test]
    fn test_builder_batch_size() {
        let pipeline = PipelineBuilder::new().batch_size(8).tokenize().build();
        assert_eq!(pipeline.batch_size, 8);
    }

    #[test]
    fn test_transformer_blocks() {
        let pipeline = PipelineBuilder::new().num_layers(4).transformer_blocks().build();
        let tb_count =
            pipeline.stages.iter().filter(|s| s.kind == StageKind::TransformerBlock).count();
        assert_eq!(tb_count, 4);
    }

    #[test]
    fn test_text_generation_pipeline() {
        let pipeline = PipelineBuilder::text_generation(2);
        assert_eq!(pipeline.name, "text_generation");
        assert!(pipeline.has_stage(&StageKind::Tokenize));
        assert!(pipeline.has_stage(&StageKind::Sample));
        assert!(pipeline.has_stage(&StageKind::Decode));
    }

    #[test]
    fn test_enabled_stages() {
        let pipeline = PipelineBuilder::new()
            .add_stage(StageConfig::new(StageKind::Embed))
            .add_stage(StageConfig::new(StageKind::Sample).disabled())
            .build();
        assert_eq!(pipeline.enabled_stages().len(), 1);
    }

    #[test]
    fn test_stage_names() {
        let pipeline = PipelineBuilder::new().tokenize().embed().build();
        assert_eq!(pipeline.stage_names(), vec!["tokenize", "embed"]);
    }

    #[test]
    fn test_summary() {
        let pipeline = PipelineBuilder::new().name("my_pipeline").tokenize().build();
        let s = pipeline.summary();
        assert!(s.contains("my_pipeline"));
        assert!(s.contains("1 stages"));
    }

    #[test]
    fn test_has_stage() {
        let pipeline = PipelineBuilder::new().tokenize().build();
        assert!(pipeline.has_stage(&StageKind::Tokenize));
        assert!(!pipeline.has_stage(&StageKind::Sample));
    }

    #[test]
    fn test_default_pipeline_name() {
        let pipeline = PipelineBuilder::new().build();
        assert_eq!(pipeline.name, "default");
    }
}
