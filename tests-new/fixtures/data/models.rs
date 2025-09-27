use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::common::{ModelFormat, ModelType, TestError, TestResult};

/// Definition of a test model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestModel {
    /// Unique identifier for the model
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Model size category
    pub size: ModelSize,
    /// Model format
    pub format: ModelFormat,
    /// Model type
    pub model_type: ModelType,
    /// Expected file size in bytes
    pub file_size: u64,
    /// SHA256 checksum
    pub checksum: String,
    /// Download URL (if available)
    pub download_url: Option<String>,
    /// Local file path (if available)
    pub local_path: Option<PathBuf>,
    /// Model parameters
    pub parameters: ModelParameters,
    /// Description
    pub description: String,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl TestModel {
    /// Create a new test model definition
    pub fn new<S: Into<String>>(
        id: S,
        name: S,
        size: ModelSize,
        format: ModelFormat,
        model_type: ModelType,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            size,
            format,
            model_type,
            file_size: 0,
            checksum: String::new(),
            download_url: None,
            local_path: None,
            parameters: ModelParameters::default(),
            description: String::new(),
            tags: Vec::new(),
        }
    }

    /// Set file size
    pub fn with_file_size(mut self, size: u64) -> Self {
        self.file_size = size;
        self
    }

    /// Set checksum
    pub fn with_checksum<S: Into<String>>(mut self, checksum: S) -> Self {
        self.checksum = checksum.into();
        self
    }

    /// Set download URL
    pub fn with_download_url<S: Into<String>>(mut self, url: S) -> Self {
        self.download_url = Some(url.into());
        self
    }

    /// Set local path
    pub fn with_local_path(mut self, path: PathBuf) -> Self {
        self.local_path = Some(path);
        self
    }

    /// Set parameters
    pub fn with_parameters(mut self, parameters: ModelParameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Set description
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = description.into();
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Check if model is available locally
    pub fn is_local(&self) -> bool {
        self.local_path.as_ref().map_or(false, |p| p.exists())
    }

    /// Check if model can be downloaded
    pub fn is_downloadable(&self) -> bool {
        self.download_url.is_some()
    }

    /// Get expected memory usage for this model
    pub fn expected_memory_usage(&self) -> u64 {
        // Rough estimate: model size + 50% overhead for inference
        (self.file_size as f64 * 1.5) as u64
    }
}

/// Model size categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSize {
    /// Tiny models (< 100MB) for basic testing
    Tiny,
    /// Small models (100MB - 1GB) for integration testing
    Small,
    /// Medium models (1GB - 10GB) for performance testing
    Medium,
    /// Large models (> 10GB) for stress testing
    Large,
}

impl ModelSize {
    /// Get the typical file size range for this model size
    pub fn size_range(&self) -> (u64, u64) {
        match self {
            Self::Tiny => (0, 100 * BYTES_PER_MB), // 0 - 100MB
            Self::Small => (100 * BYTES_PER_MB, BYTES_PER_MB * 1024), // 100MB - 1GB
            Self::Medium => (BYTES_PER_MB * 1024, 10 * BYTES_PER_MB * 1024), // 1GB - 10GB
            Self::Large => (10 * BYTES_PER_MB * 1024, u64::MAX), // > 10GB
        }
    }

    /// Check if a file size fits this category
    pub fn fits_size(&self, size: u64) -> bool {
        let (min, max) = self.size_range();
        (min..=max).contains(&size)
    }
}

/// Model parameters and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Number of parameters in the model
    pub parameter_count: Option<u64>,
    /// Context length
    pub context_length: Option<u32>,
    /// Vocabulary size
    pub vocab_size: Option<u32>,
    /// Number of layers
    pub num_layers: Option<u32>,
    /// Hidden dimension size
    pub hidden_size: Option<u32>,
    /// Number of attention heads
    pub num_heads: Option<u32>,
    /// Quantization bits
    pub quantization_bits: Option<u8>,
    /// Additional custom parameters
    pub custom: HashMap<String, String>,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            parameter_count: None,
            context_length: None,
            vocab_size: None,
            num_layers: None,
            hidden_size: None,
            num_heads: None,
            quantization_bits: None,
            custom: HashMap::new(),
        }
    }
}

impl ModelParameters {
    /// Create parameters for a BitNet model
    pub fn bitnet(
        parameter_count: u64,
        context_length: u32,
        vocab_size: u32,
        num_layers: u32,
        hidden_size: u32,
    ) -> Self {
        Self {
            parameter_count: Some(parameter_count),
            context_length: Some(context_length),
            vocab_size: Some(vocab_size),
            num_layers: Some(num_layers),
            hidden_size: Some(hidden_size),
            num_heads: Some(hidden_size / 64), // Typical head dimension is 64
            quantization_bits: Some(1),        // BitNet uses 1-bit quantization
            custom: HashMap::new(),
        }
    }
}

/// Registry for managing test models
pub struct TestModelRegistry {
    models: HashMap<String, TestModel>,
}

impl TestModelRegistry {
    /// Create a new model registry with built-in models
    pub async fn new() -> TestResult<Self> {
        let mut registry = Self {
            models: HashMap::new(),
        };

        registry.load_builtin_models().await?;
        Ok(registry)
    }

    /// Register a test model
    pub fn register(&mut self, model: TestModel) {
        self.models.insert(model.id.clone(), model);
    }

    /// Get a model by ID
    pub fn get(&self, id: &str) -> Option<&TestModel> {
        self.models.get(id)
    }

    /// Get all models
    pub fn all(&self) -> Vec<&TestModel> {
        self.models.values().collect()
    }

    /// Get models by size
    pub fn by_size(&self, size: ModelSize) -> Vec<&TestModel> {
        self.models.values().filter(|m| m.size == size).collect()
    }

    /// Get models by format
    pub fn by_format(&self, format: ModelFormat) -> Vec<&TestModel> {
        self.models
            .values()
            .filter(|m| m.format == format)
            .collect()
    }

    /// Get models by type
    pub fn by_type(&self, model_type: ModelType) -> Vec<&TestModel> {
        self.models
            .values()
            .filter(|m| m.model_type == model_type)
            .collect()
    }

    /// Get models with specific tags
    pub fn by_tags(&self, tags: &[String]) -> Vec<&TestModel> {
        self.models
            .values()
            .filter(|m| tags.iter().any(|tag| m.tags.contains(tag)))
            .collect()
    }

    /// Get locally available models
    pub fn local_models(&self) -> Vec<&TestModel> {
        self.models.values().filter(|m| m.is_local()).collect()
    }

    /// Get downloadable models
    pub fn downloadable_models(&self) -> Vec<&TestModel> {
        self.models
            .values()
            .filter(|m| m.is_downloadable())
            .collect()
    }

    /// Load built-in test models
    async fn load_builtin_models(&mut self) -> TestResult<()> {
        // Tiny models for basic testing
        let tiny_bitnet = TestModel::new(
            "tiny-bitnet",
            "Tiny BitNet Model",
            ModelSize::Tiny,
            ModelFormat::Gguf,
            ModelType::BitNet,
        )
        .with_file_size(50 * BYTES_PER_MB) // 50MB
        .with_checksum("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        .with_parameters(ModelParameters::bitnet(
            125_000_000,  // 125M parameters
            2048,         // 2K context
            32000,        // 32K vocab
            12,           // 12 layers
            768,          // 768 hidden size
        ))
        .with_description("Tiny BitNet model for basic functionality testing")
        .with_tags(vec!["tiny".to_string(), "basic".to_string(), "fast".to_string()]);

        let tiny_transformer = TestModel::new(
            "tiny-transformer",
            "Tiny Transformer Model",
            ModelSize::Tiny,
            ModelFormat::SafeTensors,
            ModelType::Transformer,
        )
        .with_file_size(75 * BYTES_PER_MB) // 75MB
        .with_checksum("d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35")
        .with_parameters(ModelParameters {
            parameter_count: Some(125_000_000),
            context_length: Some(2048),
            vocab_size: Some(32000),
            num_layers: Some(12),
            hidden_size: Some(768),
            num_heads: Some(12),
            quantization_bits: Some(16), // FP16
            custom: HashMap::new(),
        })
        .with_description("Tiny transformer model for comparison testing")
        .with_tags(vec!["tiny".to_string(), "comparison".to_string(), "transformer".to_string()]);

        // Small models for integration testing
        let small_bitnet = TestModel::new(
            "small-bitnet",
            "Small BitNet Model",
            ModelSize::Small,
            ModelFormat::Gguf,
            ModelType::BitNet,
        )
        .with_file_size(500 * BYTES_PER_MB) // 500MB
        .with_checksum("aec070645fe53ee3b3763059376134f058cc337247c978add178b6ccdfb0019f")
        .with_parameters(ModelParameters::bitnet(
            1_300_000_000, // 1.3B parameters
            4096,          // 4K context
            50000,         // 50K vocab
            24,            // 24 layers
            2048,          // 2048 hidden size
        ))
        .with_description("Small BitNet model for integration and performance testing")
        .with_tags(vec!["small".to_string(), "integration".to_string(), "performance".to_string()]);

        // Medium model for performance testing
        let medium_bitnet = TestModel::new(
            "medium-bitnet",
            "Medium BitNet Model",
            ModelSize::Medium,
            ModelFormat::Gguf,
            ModelType::BitNet,
        )
        .with_file_size(3 * BYTES_PER_MB * 1024) // 3GB
        .with_checksum("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9")
        .with_parameters(ModelParameters::bitnet(
            7_000_000_000, // 7B parameters
            8192,          // 8K context
            100000,        // 100K vocab
            32,            // 32 layers
            4096,          // 4096 hidden size
        ))
        .with_description("Medium BitNet model for comprehensive performance testing")
        .with_tags(vec!["medium".to_string(), "performance".to_string(), "comprehensive".to_string()]);

        // Register all models
        self.register(tiny_bitnet);
        self.register(tiny_transformer);
        self.register(small_bitnet);
        self.register(medium_bitnet);

        tracing::debug!("Loaded {} built-in test models", self.models.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_size_ranges() {
        assert_eq!(ModelSize::Tiny.size_range(), (0, 100 * BYTES_PER_MB));
        assert_eq!(
            ModelSize::Small.size_range(),
            (100 * BYTES_PER_MB, BYTES_PER_MB * 1024)
        );

        assert!(ModelSize::Tiny.fits_size(50 * BYTES_PER_MB));
        assert!(!ModelSize::Tiny.fits_size(200 * BYTES_PER_MB));
        assert!(ModelSize::Small.fits_size(500 * BYTES_PER_MB));
    }

    #[test]
    fn test_test_model_creation() {
        let model = TestModel::new(
            "test-model",
            "Test Model",
            ModelSize::Tiny,
            ModelFormat::Gguf,
            ModelType::BitNet,
        )
        .with_file_size(1024)
        .with_checksum("test-checksum")
        .with_description("Test description")
        .with_tags(vec!["test".to_string()]);

        assert_eq!(model.id, "test-model");
        assert_eq!(model.name, "Test Model");
        assert_eq!(model.size, ModelSize::Tiny);
        assert_eq!(model.file_size, 1024);
        assert_eq!(model.checksum, "test-checksum");
        assert_eq!(model.description, "Test description");
        assert_eq!(model.tags, vec!["test".to_string()]);

        assert!(!model.is_local());
        assert!(!model.is_downloadable());
    }

    #[test]
    fn test_model_parameters() {
        let params = ModelParameters::bitnet(1_000_000, 2048, 32000, 12, 768);

        assert_eq!(params.parameter_count, Some(1_000_000));
        assert_eq!(params.context_length, Some(2048));
        assert_eq!(params.vocab_size, Some(32000));
        assert_eq!(params.num_layers, Some(12));
        assert_eq!(params.hidden_size, Some(768));
        assert_eq!(params.num_heads, Some(12)); // 768 / 64
        assert_eq!(params.quantization_bits, Some(1));
    }

    #[tokio::test]
    async fn test_model_registry() {
        let registry = TestModelRegistry::new().await.unwrap();

        // Should have built-in models
        assert!(!registry.all().is_empty());

        // Test getting models by different criteria
        let tiny_models = registry.by_size(ModelSize::Tiny);
        assert!(!tiny_models.is_empty());

        let bitnet_models = registry.by_type(ModelType::BitNet);
        assert!(!bitnet_models.is_empty());

        let gguf_models = registry.by_format(ModelFormat::Gguf);
        assert!(!gguf_models.is_empty());

        // Test getting specific model
        let tiny_bitnet = registry.get("tiny-bitnet");
        assert!(tiny_bitnet.is_some());
        assert_eq!(tiny_bitnet.unwrap().name, "Tiny BitNet Model");
    }

    #[test]
    fn test_expected_memory_usage() {
        let model = TestModel::new(
            "test",
            "Test",
            ModelSize::Tiny,
            ModelFormat::Gguf,
            ModelType::BitNet,
        )
        .with_file_size(1000);

        // Should be file size + 50% overhead
        assert_eq!(model.expected_memory_usage(), 1500);
    }
}
