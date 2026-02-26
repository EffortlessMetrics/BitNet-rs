# [DATA] GGUF Metadata Hardcoded Defaults Elimination for Robust Model Loading

## Problem Description

The `read_gguf_metadata` function uses hardcoded default values for critical model parameters (`vocab_size`, `hidden_size`, `num_layers`, `num_heads`) when these fields are missing from GGUF metadata. This approach masks model compatibility issues and can lead to silent failures, incorrect inference results, and difficult-to-debug runtime errors.

## Environment

- **Component**: `bitnet-models` crate
- **File**: `crates/bitnet-models/src/gguf_parity.rs`
- **Rust Version**: 1.90.0+ (2024 edition)
- **GGUF Versions**: All supported GGUF format versions
- **Model Types**: LLaMA, Mistral, CodeLlama, custom BitNet models

## Current Implementation Analysis

### Problematic Default Value Injection
```rust
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata> {
    // PROBLEM: Hardcoded defaults mask missing metadata
    let vocab_size = metadata
        .get("llama.vocab_size")
        .or_else(|| metadata.get("tokenizer.ggml.tokens"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(50257); // GPT-2 default - wrong for most models!

    let hidden_size = metadata
        .get("llama.embedding_length")
        .or_else(|| metadata.get("llama.hidden_size"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(768); // Arbitrary default

    let num_layers = metadata
        .get("llama.block_count")
        .or_else(|| metadata.get("llama.layer_count"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12); // GPT-2 default

    let num_heads = metadata
        .get("llama.attention.head_count")
        .or_else(|| metadata.get("llama.head_count"))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12); // Arbitrary default
}
```

### Silent Failure Scenarios
1. **Wrong Vocabulary Size**: Using 50257 (GPT-2) for LLaMA models (32000 vocab)
2. **Incorrect Dimensions**: Using 768 hidden size for 4096-dimension models
3. **Layer Count Mismatch**: Using 12 layers for 32-layer models
4. **Attention Head Errors**: Using 12 heads for models with 32+ heads

## Root Cause Analysis

1. **Development Convenience**: Defaults added to avoid handling missing metadata
2. **Insufficient Validation**: No verification of metadata completeness
3. **Silent Failures**: Errors not surfaced until runtime inference
4. **Model Incompatibility**: Defaults prevent proper error detection
5. **Debugging Complexity**: Hidden issues difficult to diagnose

## Impact Assessment

**Severity**: High - Can cause silent model loading failures and incorrect inference

**Affected Operations**:
- Model initialization with wrong parameters
- Tensor dimension mismatches during inference
- Vocabulary lookup errors
- Attention mechanism failures
- Memory allocation issues

**User Impact**:
- Subtle inference quality degradation
- Hard-to-debug runtime failures
- Incorrect model behavior
- Wasted debugging time

## Proposed Solution

### Comprehensive GGUF Metadata Validation Architecture

```rust
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// Enhanced GGUF metadata with comprehensive validation
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub max_seq_len: Option<usize>,
    pub rope_base: Option<f32>,
    pub rope_scaling: Option<RopeScaling>,
    pub model_type: ModelType,
    pub quantization_type: Option<QuantizationType>,
    pub metadata_version: Option<String>,
    pub raw_metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum ModelType {
    Llama,
    Mistral,
    CodeLlama,
    BitNet,
    Unknown(String),
}

#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub scale_type: String,
    pub factor: f32,
}

/// Metadata extraction with validation and fallback strategies
#[derive(Debug)]
pub struct GgufMetadataExtractor {
    /// Known metadata key mappings for different model types
    key_mappings: HashMap<ModelType, MetadataKeySet>,
    /// Validation rules for different model architectures
    validation_rules: HashMap<ModelType, ValidationRules>,
}

#[derive(Debug, Clone)]
pub struct MetadataKeySet {
    pub vocab_size_keys: Vec<&'static str>,
    pub hidden_size_keys: Vec<&'static str>,
    pub num_layers_keys: Vec<&'static str>,
    pub num_heads_keys: Vec<&'static str>,
    pub model_type_keys: Vec<&'static str>,
}

#[derive(Debug, Clone)]
pub struct ValidationRules {
    pub vocab_size_range: (usize, usize),
    pub hidden_size_values: Vec<usize>,
    pub num_layers_range: (usize, usize),
    pub num_heads_range: (usize, usize),
    pub required_fields: Vec<&'static str>,
}

impl GgufMetadataExtractor {
    pub fn new() -> Self {
        let mut key_mappings = HashMap::new();
        let mut validation_rules = HashMap::new();

        // LLaMA model metadata mapping
        key_mappings.insert(ModelType::Llama, MetadataKeySet {
            vocab_size_keys: vec!["llama.vocab_size", "tokenizer.ggml.tokens"],
            hidden_size_keys: vec!["llama.embedding_length", "llama.hidden_size"],
            num_layers_keys: vec!["llama.block_count", "llama.layer_count"],
            num_heads_keys: vec!["llama.attention.head_count", "llama.head_count"],
            model_type_keys: vec!["general.architecture"],
        });

        // LLaMA validation rules
        validation_rules.insert(ModelType::Llama, ValidationRules {
            vocab_size_range: (1000, 200000),
            hidden_size_values: vec![2048, 4096, 5120, 8192, 11008],
            num_layers_range: (6, 80),
            num_heads_range: (8, 64),
            required_fields: vec!["llama.vocab_size", "llama.embedding_length", "llama.block_count"],
        });

        // Mistral model metadata mapping
        key_mappings.insert(ModelType::Mistral, MetadataKeySet {
            vocab_size_keys: vec!["mistral.vocab_size", "tokenizer.ggml.tokens"],
            hidden_size_keys: vec!["mistral.embedding_length", "mistral.hidden_size"],
            num_layers_keys: vec!["mistral.block_count", "mistral.layer_count"],
            num_heads_keys: vec!["mistral.attention.head_count"],
            model_type_keys: vec!["general.architecture"],
        });

        validation_rules.insert(ModelType::Mistral, ValidationRules {
            vocab_size_range: (1000, 100000),
            hidden_size_values: vec![4096, 8192],
            num_layers_range: (8, 40),
            num_heads_range: (8, 32),
            required_fields: vec!["mistral.vocab_size", "mistral.embedding_length"],
        });

        Self {
            key_mappings,
            validation_rules,
        }
    }

    /// Extract and validate GGUF metadata with comprehensive error handling
    pub fn extract_metadata(&self, path: &Path) -> Result<GgufMetadata, GgufMetadataError> {
        // Read raw GGUF metadata
        let raw_metadata = self.read_raw_gguf_metadata(path)?;

        // Detect model type
        let model_type = self.detect_model_type(&raw_metadata)?;

        // Extract core parameters with validation
        let vocab_size = self.extract_vocab_size(&raw_metadata, &model_type)?;
        let hidden_size = self.extract_hidden_size(&raw_metadata, &model_type)?;
        let num_layers = self.extract_num_layers(&raw_metadata, &model_type)?;
        let num_heads = self.extract_num_heads(&raw_metadata, &model_type)?;

        // Extract optional parameters
        let num_kv_heads = self.extract_num_kv_heads(&raw_metadata, &model_type);
        let intermediate_size = self.extract_intermediate_size(&raw_metadata, &model_type);
        let max_seq_len = self.extract_max_seq_len(&raw_metadata, &model_type);
        let rope_base = self.extract_rope_base(&raw_metadata);
        let rope_scaling = self.extract_rope_scaling(&raw_metadata);
        let quantization_type = self.extract_quantization_type(&raw_metadata);

        // Validate extracted metadata
        let metadata = GgufMetadata {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            intermediate_size,
            max_seq_len,
            rope_base,
            rope_scaling,
            model_type: model_type.clone(),
            quantization_type,
            metadata_version: raw_metadata.get("general.version").cloned(),
            raw_metadata: raw_metadata.clone(),
        };

        self.validate_metadata(&metadata)?;

        Ok(metadata)
    }

    /// Detect model type from GGUF metadata
    fn detect_model_type(&self, metadata: &HashMap<String, String>) -> Result<ModelType, GgufMetadataError> {
        if let Some(arch) = metadata.get("general.architecture") {
            match arch.as_str() {
                "llama" => Ok(ModelType::Llama),
                "mistral" => Ok(ModelType::Mistral),
                "code_llama" => Ok(ModelType::CodeLlama),
                "bitnet" => Ok(ModelType::BitNet),
                unknown => Ok(ModelType::Unknown(unknown.to_string())),
            }
        } else {
            // Fallback detection based on available keys
            if metadata.keys().any(|k| k.starts_with("llama.")) {
                Ok(ModelType::Llama)
            } else if metadata.keys().any(|k| k.starts_with("mistral.")) {
                Ok(ModelType::Mistral)
            } else {
                Err(GgufMetadataError::UnknownModelType {
                    available_keys: metadata.keys().cloned().collect(),
                })
            }
        }
    }

    /// Extract vocab size with comprehensive key searching
    fn extract_vocab_size(&self, metadata: &HashMap<String, String>, model_type: &ModelType) -> Result<usize, GgufMetadataError> {
        let key_set = self.key_mappings.get(model_type)
            .ok_or_else(|| GgufMetadataError::UnsupportedModelType(model_type.clone()))?;

        for key in &key_set.vocab_size_keys {
            if let Some(value_str) = metadata.get(*key) {
                if let Ok(value) = value_str.parse::<usize>() {
                    return Ok(value);
                }
            }
        }

        // Try alternative extraction methods
        if let Some(tokens_value) = metadata.get("tokenizer.ggml.tokens") {
            // Sometimes vocab size is in array format
            if tokens_value.starts_with('[') {
                // Parse array length as vocab size
                // This would need more sophisticated parsing
            }
        }

        Err(GgufMetadataError::MissingRequiredField {
            field_name: "vocab_size".to_string(),
            searched_keys: key_set.vocab_size_keys.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Extract hidden size with validation
    fn extract_hidden_size(&self, metadata: &HashMap<String, String>, model_type: &ModelType) -> Result<usize, GgufMetadataError> {
        let key_set = self.key_mappings.get(model_type)
            .ok_or_else(|| GgufMetadataError::UnsupportedModelType(model_type.clone()))?;

        for key in &key_set.hidden_size_keys {
            if let Some(value_str) = metadata.get(*key) {
                if let Ok(value) = value_str.parse::<usize>() {
                    return Ok(value);
                }
            }
        }

        Err(GgufMetadataError::MissingRequiredField {
            field_name: "hidden_size".to_string(),
            searched_keys: key_set.hidden_size_keys.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Extract number of layers
    fn extract_num_layers(&self, metadata: &HashMap<String, String>, model_type: &ModelType) -> Result<usize, GgufMetadataError> {
        let key_set = self.key_mappings.get(model_type)
            .ok_or_else(|| GgufMetadataError::UnsupportedModelType(model_type.clone()))?;

        for key in &key_set.num_layers_keys {
            if let Some(value_str) = metadata.get(*key) {
                if let Ok(value) = value_str.parse::<usize>() {
                    return Ok(value);
                }
            }
        }

        Err(GgufMetadataError::MissingRequiredField {
            field_name: "num_layers".to_string(),
            searched_keys: key_set.num_layers_keys.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Extract number of attention heads
    fn extract_num_heads(&self, metadata: &HashMap<String, String>, model_type: &ModelType) -> Result<usize, GgufMetadataError> {
        let key_set = self.key_mappings.get(model_type)
            .ok_or_else(|| GgufMetadataError::UnsupportedModelType(model_type.clone()))?;

        for key in &key_set.num_heads_keys {
            if let Some(value_str) = metadata.get(*key) {
                if let Ok(value) = value_str.parse::<usize>() {
                    return Ok(value);
                }
            }
        }

        Err(GgufMetadataError::MissingRequiredField {
            field_name: "num_heads".to_string(),
            searched_keys: key_set.num_heads_keys.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Validate extracted metadata against model-specific rules
    fn validate_metadata(&self, metadata: &GgufMetadata) -> Result<(), GgufMetadataError> {
        if let Some(rules) = self.validation_rules.get(&metadata.model_type) {
            // Validate vocab size range
            if metadata.vocab_size < rules.vocab_size_range.0 || metadata.vocab_size > rules.vocab_size_range.1 {
                return Err(GgufMetadataError::InvalidValue {
                    field: "vocab_size".to_string(),
                    value: metadata.vocab_size.to_string(),
                    expected: format!("{}-{}", rules.vocab_size_range.0, rules.vocab_size_range.1),
                });
            }

            // Validate hidden size against known values
            if !rules.hidden_size_values.is_empty() && !rules.hidden_size_values.contains(&metadata.hidden_size) {
                return Err(GgufMetadataError::InvalidValue {
                    field: "hidden_size".to_string(),
                    value: metadata.hidden_size.to_string(),
                    expected: format!("one of {:?}", rules.hidden_size_values),
                });
            }

            // Validate layer count
            if metadata.num_layers < rules.num_layers_range.0 || metadata.num_layers > rules.num_layers_range.1 {
                return Err(GgufMetadataError::InvalidValue {
                    field: "num_layers".to_string(),
                    value: metadata.num_layers.to_string(),
                    expected: format!("{}-{}", rules.num_layers_range.0, rules.num_layers_range.1),
                });
            }

            // Validate head count
            if metadata.num_heads < rules.num_heads_range.0 || metadata.num_heads > rules.num_heads_range.1 {
                return Err(GgufMetadataError::InvalidValue {
                    field: "num_heads".to_string(),
                    value: metadata.num_heads.to_string(),
                    expected: format!("{}-{}", rules.num_heads_range.0, rules.num_heads_range.1),
                });
            }

            // Validate architectural consistency
            if metadata.hidden_size % metadata.num_heads != 0 {
                return Err(GgufMetadataError::ArchitecturalInconsistency {
                    issue: format!(
                        "hidden_size ({}) must be divisible by num_heads ({})",
                        metadata.hidden_size, metadata.num_heads
                    ),
                });
            }
        }

        Ok(())
    }

    /// Read raw GGUF metadata from file
    fn read_raw_gguf_metadata(&self, path: &Path) -> Result<HashMap<String, String>, GgufMetadataError> {
        // Implementation would read GGUF file format
        // This is a placeholder for the actual GGUF parsing logic
        todo!("Implement actual GGUF file parsing")
    }

    // Additional extraction methods for optional fields...
    fn extract_num_kv_heads(&self, metadata: &HashMap<String, String>, model_type: &ModelType) -> Option<usize> {
        metadata.get("llama.attention.key_value_head_count")
            .or_else(|| metadata.get("mistral.attention.key_value_head_count"))
            .and_then(|v| v.parse().ok())
    }

    fn extract_intermediate_size(&self, metadata: &HashMap<String, String>, model_type: &ModelType) -> Option<usize> {
        metadata.get("llama.feed_forward_length")
            .or_else(|| metadata.get("mistral.feed_forward_length"))
            .and_then(|v| v.parse().ok())
    }

    fn extract_max_seq_len(&self, metadata: &HashMap<String, String>, model_type: &ModelType) -> Option<usize> {
        metadata.get("llama.context_length")
            .or_else(|| metadata.get("mistral.context_length"))
            .and_then(|v| v.parse().ok())
    }

    fn extract_rope_base(&self, metadata: &HashMap<String, String>) -> Option<f32> {
        metadata.get("llama.rope.freq_base")
            .and_then(|v| v.parse().ok())
    }

    fn extract_rope_scaling(&self, metadata: &HashMap<String, String>) -> Option<RopeScaling> {
        let scale_type = metadata.get("llama.rope.scaling.type")?;
        let factor = metadata.get("llama.rope.scaling.factor")?.parse().ok()?;

        Some(RopeScaling {
            scale_type: scale_type.clone(),
            factor,
        })
    }

    fn extract_quantization_type(&self, metadata: &HashMap<String, String>) -> Option<QuantizationType> {
        metadata.get("general.quantization_version")
            .and_then(|v| match v.as_str() {
                "I2_S" => Some(QuantizationType::I2S),
                "TL1" => Some(QuantizationType::TL1),
                "TL2" => Some(QuantizationType::TL2),
                _ => None,
            })
    }
}

#[derive(Debug, Error)]
pub enum GgufMetadataError {
    #[error("Missing required field '{field_name}', searched keys: {searched_keys:?}")]
    MissingRequiredField {
        field_name: String,
        searched_keys: Vec<String>,
    },

    #[error("Invalid value for field '{field}': got '{value}', expected {expected}")]
    InvalidValue {
        field: String,
        value: String,
        expected: String,
    },

    #[error("Unknown model type - available metadata keys: {available_keys:?}")]
    UnknownModelType {
        available_keys: Vec<String>,
    },

    #[error("Unsupported model type: {0:?}")]
    UnsupportedModelType(ModelType),

    #[error("Architectural inconsistency: {issue}")]
    ArchitecturalInconsistency {
        issue: String,
    },

    #[error("GGUF file parsing error: {0}")]
    FileParseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Enhanced public interface
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata, GgufMetadataError> {
    let extractor = GgufMetadataExtractor::new();
    extractor.extract_metadata(path)
}
```

## Implementation Plan

### Phase 1: Validation Infrastructure (Week 1)
- [ ] Implement comprehensive metadata validation framework
- [ ] Add model-specific key mappings and validation rules
- [ ] Create detailed error handling with actionable messages
- [ ] Establish testing framework for metadata parsing

### Phase 2: Model Type Support (Week 2)
- [ ] Add support for LLaMA, Mistral, CodeLlama metadata formats
- [ ] Implement architectural consistency validation
- [ ] Add fallback detection mechanisms
- [ ] Create comprehensive test suite for each model type

### Phase 3: Integration & Testing (Week 3)
- [ ] Replace hardcoded defaults with validation-based extraction
- [ ] Add comprehensive error handling throughout BitNet-rs
- [ ] Test with diverse GGUF model files
- [ ] Validate error messages and debugging information

### Phase 4: Production Hardening (Week 4)
- [ ] Add performance optimization for metadata parsing
- [ ] Implement caching for frequently accessed models
- [ ] Add monitoring and diagnostics
- [ ] Documentation and troubleshooting guide

## Success Criteria

- [ ] **Zero Hardcoded Defaults**: No fallback to arbitrary default values
- [ ] **Comprehensive Validation**: All critical model parameters verified
- [ ] **Clear Error Messages**: Actionable feedback for missing/invalid metadata
- [ ] **Model Support**: Support for major GGUF model formats
- [ ] **Architectural Consistency**: Validation of model parameter relationships
- [ ] **Debugging Support**: Clear error reporting for troubleshooting

## Related Issues

- #XXX: Model loading validation comprehensive framework
- #XXX: GGUF format support expansion
- #XXX: Error handling standardization across BitNet-rs
- #XXX: Model compatibility testing automation

## Implementation Notes

This implementation eliminates silent failures from hardcoded defaults while providing comprehensive validation and clear error reporting. The solution enables robust model loading with proper error detection and debugging support.
